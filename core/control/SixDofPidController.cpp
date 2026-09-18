/**
 * @file SixDofPidController.cpp
 * @brief 六自由度级联 PID 控制器实现
 * @author GhostFace
 * @date 2026/9/17
 */

#include "SixDofPidController.h"
#include "SixDofDynamics.h"

#include <algorithm>
#include <cmath>

namespace oi3 {

namespace {

constexpr double kPi = 3.14159265358979323846;

double clampd(double v, double lo, double hi) {
    return std::max(lo, std::min(hi, v));
}

struct V3d {
    double x, y, z;
};

V3d readV3(const Tensor &t) {
    const float *p = t.data<float>();
    return {p[0], p[1], p[2]};
}

V3d cross(const V3d &a, const V3d &b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

double dot(const V3d &a, const V3d &b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

double norm(const V3d &a) { return std::sqrt(dot(a, a)); }

} // namespace

SixDofPidController::SixDofPidController(SixDofConfig cfg, SixDofPidGains gains)
    : _cfg(std::move(cfg)), _gains(gains) {
    // 三轴增益在构造时一次性确定：按期望带宽/阻尼比 + 惯量推导，或用一组手动值。
    // 放在构造期而不是每步计算，是因为惯量在一次飞行里不变，而 compute 是 1 kHz 热路径。
    if (_gains.derive_attitude_from_inertia) {
        const double wn = std::max(1e-3, _gains.att_bandwidth);
        const double zeta = std::max(1e-3, _gains.att_damping);
        for (int i = 0; i < 3; ++i) {
            const double I = std::max(1e-9, _cfg.inertia[i]);
            _att_kp[i] = I * wn * wn;
            _att_kd[i] = 2.0 * zeta * I * wn;
        }
    } else {
        for (int i = 0; i < 3; ++i) {
            _att_kp[i] = _gains.att_kp;
            _att_kd[i] = _gains.att_kd;
        }
    }
}

void SixDofPidController::reset() { _last_tilt_deg = 0.0; }

namespace {

/**
 * @brief 由期望加速度解算推力与力矩（姿态环 + 推力分配）
 *
 * 抽出来给定点与跟踪两条入口共用。姿态解算与位置环无关，重复实现只会让两条
 * 路径慢慢漂开。
 *
 * @param att_kp/att_kd 三轴姿态增益（构造期已确定）
 * @param tilt_out      输出：实际使用的倾角（度）
 */
SixDofCommand solveCommand(const SixDofConfig &cfg, const SixDofPidGains &gains,
                           const double att_kp[3], const double att_kd[3],
                           const SixDofState &state, const V3d &a_in, double &tilt_out) {
    const double m = cfg.base.mass;
    const double g = cfg.base.gravity;

    const double a_des[3] = {a_in.x, a_in.y, a_in.z};

    // ---- 合力与期望推力方向 ----
    // F = m·(a_des − g_vec)，g_vec = [0, 0, g]
    const V3d f_des{m * a_des[0], m * a_des[1], m * (a_des[2] - g)};
    const double f_norm = norm(f_des);

    // 期望机体 z 轴（机体 z 朝上，推力沿 −z，故 z_des 与合力反向）
    V3d z_des{0.0, 0.0, 1.0};
    if (f_norm > 1e-9) {
        z_des = {-f_des.x / f_norm, -f_des.y / f_norm, -f_des.z / f_norm};
    }

    // 当前机体 z 轴在 NED 系中的方向
    const Tensor z_body_frame = makeVec3(0.0f, 0.0f, 1.0f);
    const V3d z_cur = readV3(rotateBodyToNed(state.quat, z_body_frame));

    // ---- 姿态误差：把 z_cur 旋到 z_des 的轴角 ----
    const V3d axis_raw = cross(z_cur, z_des);
    const double sin_theta = norm(axis_raw);
    const double cos_theta = clampd(dot(z_cur, z_des), -1.0, 1.0);
    double theta = std::atan2(sin_theta, cos_theta);

    // 倾角限幅：过大的倾角会让竖直可用推力不足以维持高度
    const double max_tilt = gains.max_tilt_deg / 180.0 * kPi;
    theta = clampd(theta, -max_tilt, max_tilt);
    tilt_out = theta / kPi * 180.0;

    V3d rotvec_ned{0.0, 0.0, 0.0};
    if (sin_theta > 1e-9) {
        rotvec_ned = {axis_raw.x / sin_theta * theta, axis_raw.y / sin_theta * theta,
                      axis_raw.z / sin_theta * theta};
    }

    // 误差旋转向量在 NED 系，力矩须在机体系施加，故旋转回机体系
    const Tensor rotvec_ned_t =
        makeVec3(static_cast<float>(rotvec_ned.x), static_cast<float>(rotvec_ned.y),
                 static_cast<float>(rotvec_ned.z));
    const Tensor rotvec_body_t = rotateNedToBody(state.quat, rotvec_ned_t);
    const V3d rotvec_body = readV3(rotvec_body_t);
    const V3d omega = readV3(state.omega);

    // ---- 姿态环：PD 力矩 ----
    SixDofCommand cmd;
    cmd.thrust_body = f_norm; // 推力大小取合力模长；倾斜时自动增大以维持竖直分量

    cmd.torque = makeVec3(
        static_cast<float>(att_kp[0] * rotvec_body.x - att_kd[0] * omega.x),
        static_cast<float>(att_kp[1] * rotvec_body.y - att_kd[1] * omega.y),
        static_cast<float>(att_kp[2] * rotvec_body.z - att_kd[2] * omega.z));

    return cmd;
}

} // namespace

SixDofCommand SixDofPidController::compute(const SixDofState &state, const Tensor &target,
                                           double /*time*/) {
    const V3d pos = readV3(state.pos);
    const V3d vel = readV3(state.vel);
    const V3d tgt = readV3(target);

    // ---- 位置环：期望加速度 ----
    // 定点版本把期望速度与期望加速度都视为零。下面的表达式刻意保持原样
    // （写成 kp*e + kd*(-vel) 而不是 kd*(0-vel)），使定点悬停的全部既有实测
    // 结果逐位不变 —— 改控制律时最忌讳把回归数据悄悄改掉。
    const V3d e_pos{tgt.x - pos.x, tgt.y - pos.y, tgt.z - pos.z};
    const double raw[3] = {
        _gains.pos_kp * e_pos.x + _gains.pos_kd * (-vel.x),
        _gains.pos_kp * e_pos.y + _gains.pos_kd * (-vel.y),
        _gains.pos_kp * e_pos.z + _gains.pos_kd * (-vel.z),
    };
    const V3d a_des{clampd(raw[0], -_gains.max_accel, _gains.max_accel),
                    clampd(raw[1], -_gains.max_accel, _gains.max_accel),
                    clampd(raw[2], -_gains.max_accel, _gains.max_accel)};

    return solveCommand(_cfg, _gains, _att_kp, _att_kd, state, a_des, _last_tilt_deg);
}

SixDofCommand SixDofPidController::computeWithWind(const SixDofState &state,
                                                    const Tensor &target,
                                                    const std::array<double, 3> &v_wind,
                                                    double /*time*/) {
    const double m = _cfg.base.mass;

    const V3d pos = readV3(state.pos);
    const V3d vel = readV3(state.vel);
    const V3d tgt = readV3(target);

    // 风阻前馈：稳态下相对气流 v_rel = −v_wind，故阻力
    //   F = −k·|v_rel|·v_rel = k·|v_wind|·v_wind   （方向与风同，把飞行器吹走）
    // 要抵消它，需要大小相等方向相反的加速度。
    const double k = _cfg.base.drag_coeff;
    V3d a_ff{0.0, 0.0, 0.0};
    if (k > 0.0) {
        const double sp = std::sqrt(v_wind[0] * v_wind[0] + v_wind[1] * v_wind[1] +
                                    v_wind[2] * v_wind[2]);
        a_ff = {-k * sp * v_wind[0] / m, -k * sp * v_wind[1] / m, -k * sp * v_wind[2] / m};
    }

    const V3d e_pos{tgt.x - pos.x, tgt.y - pos.y, tgt.z - pos.z};
    const double raw[3] = {
        a_ff.x + _gains.pos_kp * e_pos.x + _gains.pos_kd * (-vel.x),
        a_ff.y + _gains.pos_kp * e_pos.y + _gains.pos_kd * (-vel.y),
        a_ff.z + _gains.pos_kp * e_pos.z + _gains.pos_kd * (-vel.z),
    };
    const V3d a_des{clampd(raw[0], -_gains.max_accel, _gains.max_accel),
                    clampd(raw[1], -_gains.max_accel, _gains.max_accel),
                    clampd(raw[2], -_gains.max_accel, _gains.max_accel)};

    return solveCommand(_cfg, _gains, _att_kp, _att_kd, state, a_des, _last_tilt_deg);
}

SixDofCommand SixDofPidController::computeTracking(const SixDofState &state,
                                                   const SixDofSetpoint &ref,
                                                   double /*time*/) {
    const V3d pos = readV3(state.pos);
    const V3d vel = readV3(state.vel);

    // ---- 位置环：带参考速度与加速度前馈 ----
    //   a_des = a_ref + kp·(p_ref − p) + kd·(v_ref − v)
    // 与定点版本的两点差别：向心/切向加速度由 a_ref 提供（不必再用位置误差换），
    // 阻尼作用在速度误差上（不再把轨迹本身的运动速度当成要消除的量）。
    const V3d e_pos{ref.pos[0] - pos.x, ref.pos[1] - pos.y, ref.pos[2] - pos.z};
    const V3d e_vel{ref.vel[0] - vel.x, ref.vel[1] - vel.y, ref.vel[2] - vel.z};
    const double raw[3] = {
        ref.acc[0] + _gains.pos_kp * e_pos.x + _gains.pos_kd * e_vel.x,
        ref.acc[1] + _gains.pos_kp * e_pos.y + _gains.pos_kd * e_vel.y,
        ref.acc[2] + _gains.pos_kp * e_pos.z + _gains.pos_kd * e_vel.z,
    };
    const V3d a_des{clampd(raw[0], -_gains.max_accel, _gains.max_accel),
                    clampd(raw[1], -_gains.max_accel, _gains.max_accel),
                    clampd(raw[2], -_gains.max_accel, _gains.max_accel)};

    return solveCommand(_cfg, _gains, _att_kp, _att_kd, state, a_des, _last_tilt_deg);
}

} // namespace oi3
