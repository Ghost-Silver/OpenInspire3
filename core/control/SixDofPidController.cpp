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

    // 风阻前馈。气动力取决于**相对气流**，因此必须用飞行器速度与风速之差：
    //
    //   v_rel = v_aircraft − v_wind
    //   F_aero = −k·|v_rel|·v_rel        （作用在机身上，把它推走）
    //   a_ff   = −F_aero / m = +k·|v_rel|·v_rel / m
    //
    // 第一版写的是 a_ff = −k·|v_wind|·v_wind / m，即把 v_rel 简化成了 −v_wind。
    // 那个式子在**稳态静止**时与上式等价，但飞行器一旦有速度（湍流下始终如此）
    // 就会失配 —— 等于把一个本身带建模误差的基线当成对照，让后续比较失去意义。
    // 阻力系数：优先使用各轴异性（若启用），否则回退到标量。
    // 未支持异性之前，前馈恒按各向同性补偿 —— 于是「把真实气动告诉控制器」
    // 这一操作形同虚设（两种配置下误差逐位相同），第 4 段判据因而失效。
    const bool use_axis = _cfg.drag_coeff_axis[0] > 0.0 && _cfg.drag_coeff_axis[1] > 0.0 &&
                          _cfg.drag_coeff_axis[2] > 0.0;
    const double kx = use_axis ? _cfg.drag_coeff_axis[0] : _cfg.base.drag_coeff;
    const double ky = use_axis ? _cfg.drag_coeff_axis[1] : _cfg.base.drag_coeff;
    const double kz = use_axis ? _cfg.drag_coeff_axis[2] : _cfg.base.drag_coeff;

    V3d a_ff{0.0, 0.0, 0.0};
    if (_cfg.base.drag_coeff > 0.0 || use_axis) {
        const double rx = vel.x - v_wind[0];
        const double ry = vel.y - v_wind[1];
        const double rz = vel.z - v_wind[2];
        const double sp = std::sqrt(rx * rx + ry * ry + rz * rz);
        a_ff = {kx * sp * rx / m, ky * sp * ry / m, kz * sp * rz / m};
    }

    // 桨盘入流补偿：实际推力 T_eff = T·(1 − mu·v_axial)，指令推力被入流打了
    // 折扣，故需补足。
    //
    // 补偿量 = 损失的推力，而损失**沿机体 −z 方向**（推力方向），不是沿 NED 的
    // z 轴。机身倾斜时这个损失在 NED 里有水平分量 —— 只补 z 轴会漏掉它们，
    // 留下与倾角相关的系统性残差。
    //
    // 幅值也不能用 T ≈ m·g 近似：机动时推力可达 mg 的 1.2~1.5 倍，近似会低估
    // 损失。这里用**上一拍的指令推力**作为实际推力的估计（一阶近似，误差远小于
    // 用 mg 替代）。
    //
    // 两处修正前，解析补偿只吃掉了总差距的 95%，残余 0.028 m 与倾角相关
    // （相关系数 0.41）；修正后残余应进一步下降。
    // 入流系数来源：优先用在线估计值（若已挂载估计器且其报告样本充足），
    // 否则回退到配置里的固定值。回退是必要的 —— 估计未收敛时用它会引入
    // 比「不知道」更差的补偿。
    double mu_lin = _cfg.inflow_linear;
    double mu_quad = _cfg.inflow_quad;
    if (_gains.use_online_inflow_estimate && _inflow_src != nullptr &&
        _inflow_src->inflowEstimateReady()) {
        mu_lin = _inflow_src->inflowMu();
        mu_quad = 0.0; // 在线估计当前只覆盖线性项
    }
    _active_mu = mu_lin;

    if (mu_lin != 0.0 || mu_quad != 0.0) {
        const Tensor wind_ned =
            makeVec3(static_cast<float>(v_wind[0]), static_cast<float>(v_wind[1]),
                     static_cast<float>(v_wind[2]));
        const Tensor rel_body = rotateNedToBody(state.quat, state.vel - wind_ned);
        const float *rb = rel_body.data<float>();
        const double v_axial = rb[2];
        const double corr = mu_lin * v_axial + mu_quad * v_axial * std::fabs(v_axial);

        // 损失的推力大小（牛顿）：T_loss = T·corr
        const double t_est = (_last_thrust > 1e-9) ? _last_thrust : _cfg.base.mass * _cfg.base.gravity;
        const double f_loss = t_est * corr;

        // 方向沿机体 −z（推力方向）：在机体中为 [0,0,−1]，转到 NED 后取反
        const Tensor loss_body = makeVec3(0.0f, 0.0f, static_cast<float>(-f_loss));
        const Tensor loss_ned = rotateBodyToNed(state.quat, loss_body);

        // 补偿 = **加上**损失矢量（而非减去）。
        //
        // 用水平姿态这个特例验符号：推力损失等效于一个**向下的额外力** T·corr，
        // 要抵消它须让期望推力增大，即 a_ff.z 要**减小**（NED 中 z 向下为正）。
        // 水平时 loss_body = [0,0,−f_loss] 转到 NED 仍是 [0,0,−f_loss]，故
        // `a_ff += loss_ned/m` 恰好给出 a_ff.z -= f_loss/m ✓
        //
        // 第一版写成 `a_ff -= loss_ned/m`，符号反了：补偿变成「再施加一份同等
        // 损失」，误差从 0.137 涨到 1.284（比完全不补偿还差一倍）。
        const float *ln = loss_ned.data<float>();
        a_ff.x += ln[0] / m;
        a_ff.y += ln[1] / m;
        a_ff.z += ln[2] / m;
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

    SixDofCommand cmd = solveCommand(_cfg, _gains, _att_kp, _att_kd, state, a_des,
                                     _last_tilt_deg);
    _last_thrust = cmd.thrust_body; // 供下一拍入流补偿估计实际推力
    return cmd;
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

    SixDofCommand cmd =
        solveCommand(_cfg, _gains, _att_kp, _att_kd, state, a_des, _last_tilt_deg);

    // ---- 平坦前馈：角速度参考 ----
    //
    // 姿态环原本的阻尼项是 `−kd·ω`，语义是「把角速度阻尼到零」—— 这对定点
    // 悬停成立，对机动飞行则是在**持续对抗**机身本该有的转动。目标律是
    //
    //     τ = kp·e − kd·(ω − ω_des)
    //
    // 而 solveCommand 已给出 `τ_old = kp·e − kd·ω`，故**增量恰为 `+kd·ω_des`**。
    // （写成 `+kd·(ω_des − ω)` 会多减一次 kd·ω、把阻尼翻倍，实测使跟踪误差
    // 反而放大 5 倍，且误差与振幅成正比 —— 常值滞后的典型指纹。）
    //
    // 关闭开关或参考未提供 jerk 时 ω_des 为零，增量为零，行为与改造前逐位相同。
    if (_gains.use_flat_omega_feedforward) {
        FlatReference fr;
        fr.pos = ref.pos;
        fr.vel = ref.vel;
        fr.acc = ref.acc;
        fr.jerk = ref.jerk;
        fr.yaw = ref.yaw;
        fr.yaw_rate = ref.yaw_rate;

        const FlatOutput fo = computeFlatFeedforward(fr, _cfg.base.mass, _cfg.base.gravity);
        if (fo.valid) {
            const float *tq = cmd.torque.data<float>();
            const std::array<float, 3> tnew{
                static_cast<float>(tq[0] + _att_kd[0] * fo.omega[0]),
                static_cast<float>(tq[1] + _att_kd[1] * fo.omega[1]),
                static_cast<float>(tq[2] + _att_kd[2] * fo.omega[2])};
            cmd.torque = makeVec3(tnew[0], tnew[1], tnew[2]);
        }
    }

    return cmd;
}

} // namespace oi3
