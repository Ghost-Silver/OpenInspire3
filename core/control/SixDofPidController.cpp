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
    : _cfg(std::move(cfg)), _gains(gains) {}

void SixDofPidController::reset() { _last_tilt_deg = 0.0; }

SixDofCommand SixDofPidController::compute(const SixDofState &state, const Tensor &target,
                                           double /*time*/) {
    const double m = _cfg.base.mass;
    const double g = _cfg.base.gravity;

    const V3d pos = readV3(state.pos);
    const V3d vel = readV3(state.vel);
    const V3d tgt = readV3(target);

    // ---- 位置环：期望加速度 ----
    const V3d e_pos{tgt.x - pos.x, tgt.y - pos.y, tgt.z - pos.z};
    double a_des[3];
    const double raw[3] = {
        _gains.pos_kp * e_pos.x + _gains.pos_kd * (-vel.x),
        _gains.pos_kp * e_pos.y + _gains.pos_kd * (-vel.y),
        _gains.pos_kp * e_pos.z + _gains.pos_kd * (-vel.z),
    };
    for (int i = 0; i < 3; ++i) {
        a_des[i] = clampd(raw[i], -_gains.max_accel, _gains.max_accel);
    }

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
    const double max_tilt = _gains.max_tilt_deg / 180.0 * kPi;
    theta = clampd(theta, -max_tilt, max_tilt);
    _last_tilt_deg = theta / kPi * 180.0;

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
        static_cast<float>(_gains.att_kp * rotvec_body.x - _gains.att_kd * omega.x),
        static_cast<float>(_gains.att_kp * rotvec_body.y - _gains.att_kd * omega.y),
        static_cast<float>(_gains.att_kp * rotvec_body.z - _gains.att_kd * omega.z));

    return cmd;
}

} // namespace oi3
