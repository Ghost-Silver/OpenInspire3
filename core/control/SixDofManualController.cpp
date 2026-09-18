/**
 * @file SixDofManualController.cpp
 * @brief 手动模式控制器实现
 * @author GhostFace
 * @date 2026/9/18
 *
 * 全部用标量计算：手动模式在 1 kHz 热路径上运行，且四元数误差的推导需要显式的
 * 符号处理（取短弧、w<0 时翻转轴），用张量算子表达反而更容易出错。
 */

#include "SixDofManualController.h"

#include <algorithm>
#include <cmath>

namespace oi3 {

namespace {

constexpr double kPi = 3.14159265358979323846;

double clampd(double v, double lo, double hi) { return std::max(lo, std::min(hi, v)); }

struct V3d {
    double x, y, z;
};

struct Quat {
    double w, x, y, z;
};

Quat readQuat(const Tensor &t) {
    const float *p = t.data<float>();
    return {p[0], p[1], p[2], p[3]};
}

V3d readV3(const Tensor &t) {
    const float *p = t.data<float>();
    return {p[0], p[1], p[2]};
}

/// 四元数乘法 a ⊗ b
Quat mul(const Quat &a, const Quat &b) {
    return {a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
            a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
            a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
            a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w};
}

/// 共轭（= 逆，对单位四元数）
Quat conj(const Quat &q) { return {q.w, -q.x, -q.y, -q.z}; }

/// ZYX 欧拉角（弧度）→ 四元数
Quat fromEuler(double roll, double pitch, double yaw) {
    const double cr = std::cos(roll * 0.5), sr = std::sin(roll * 0.5);
    const double cp = std::cos(pitch * 0.5), sp = std::sin(pitch * 0.5);
    const double cy = std::cos(yaw * 0.5), sy = std::sin(yaw * 0.5);
    return {cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy};
}

/// 从四元数取偏航角（ZYX）
double yawOf(const Quat &q) {
    return std::atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z));
}

/**
 * @brief 四元数误差 → 旋转向量（轴 × 角）
 *
 * 取**短弧**：四元数 q 与 −q 表示同一旋转，若 w<0 说明走的是长弧，需要翻转。
 * 不做这一步的话，姿态误差超过 180° 时力矩方向会反掉，表现为「越纠越偏」。
 */
V3d rotationVector(const Quat &q) {
    const double w = clampd(q.w, -1.0, 1.0);
    const double s = std::sqrt(std::max(0.0, 1.0 - w * w));
    if (s < 1e-12) {
        return {0.0, 0.0, 0.0};
    }
    const double angle = 2.0 * std::atan2(s, std::fabs(w));
    const double sign = (w < 0.0) ? -1.0 : 1.0;
    const double k = sign * angle / s;
    return {q.x * k, q.y * k, q.z * k};
}

} // namespace

SixDofManualController::SixDofManualController(SixDofConfig cfg, ManualConfig mc)
    : _cfg(std::move(cfg)), _mc(mc) {
    // 与定点控制器同样的做法：增益由期望带宽、阻尼比与惯量推导，三轴闭环特性一致
    const double wn = std::max(1e-3, _mc.att_bandwidth);
    const double zeta = std::max(1e-3, _mc.att_damping);
    for (int i = 0; i < 3; ++i) {
        const double I = std::max(1e-9, _cfg.inertia[i]);
        _att_kp[i] = I * wn * wn;
        _att_kd[i] = 2.0 * zeta * I * wn;
    }
}

void SixDofManualController::reset() { _target_tilt_deg = 0.0; }

SixDofCommand SixDofManualController::compute(const SixDofState &state,
                                              const RcStick &stick) {
    const double m = _cfg.base.mass;
    const double g = _cfg.base.gravity;

    // ---- 1. 杆量 → 目标滚转/俯仰角 ----
    const double max_tilt = _mc.max_tilt_deg / 180.0 * kPi;
    const double roll_des = clampd(stick.roll, -1.0, 1.0) * max_tilt;
    const double pitch_des = clampd(stick.pitch, -1.0, 1.0) * max_tilt;

    // 目标倾角（机体 z 轴与竖直方向的夹角），用于推力补偿与倾角保护
    const double cos_tilt = std::cos(roll_des) * std::cos(pitch_des);
    const double tilt = std::acos(clampd(cos_tilt, -1.0, 1.0));
    _target_tilt_deg = tilt * 180.0 / kPi;

    // ---- 2. 目标姿态 ----
    // 偏航维度的目标是「保持当前偏航」：偏航由速率环单独控制（见第 5 步），
    // 若这里也塞入一个偏航目标，两条通路会互相打架。
    const Quat q_cur = readQuat(state.quat);
    const Quat q_des = fromEuler(roll_des, pitch_des, yawOf(q_cur));

    // ---- 3. 姿态误差（机体系）----
    // q_err = q_cur⁻¹ ⊗ q_des 给出「从当前姿态转到目标姿态」在机体坐标系下的旋转，
    // 而力矩正是在机体系施加的，因此不需要再做坐标系变换。
    const V3d e = rotationVector(mul(conj(q_cur), q_des));

    // ---- 4. 力矩 ----
    const V3d omega = readV3(state.omega);
    const double yaw_rate_des = clampd(stick.yaw, -1.0, 1.0) * _mc.yaw_rate_max;

    SixDofCommand cmd;
    cmd.torque = makeVec3(
        static_cast<float>(_att_kp[0] * e.x - _att_kd[0] * omega.x),
        static_cast<float>(_att_kp[1] * e.y - _att_kd[1] * omega.y),
        // 偏航轴：阻尼作用在**速率误差**上而非绝对角速度 —— 否则偏航杆给出的
        // 指令会被阻尼项直接抵消，表现为「打偏航杆没反应」。
        static_cast<float>(_att_kp[2] * e.z +
                           _att_kd[2] * (yaw_rate_des - omega.z)));

    // ---- 5. 推力 ----
    // 悬停推力为基础，倾斜时按 1/cosθ 补偿使竖直分量恒为 mg。
    // 不补偿的话，倾角 35° 时竖直分量只剩 82%，会明显掉高。
    //
    // 补偿必须用**实际**倾角而不是目标倾角。过渡期间实际倾角小于指令（姿态环
    // 需要时间跟上），若按目标倾角补偿就会在过渡期持续过推 —— 实测这样会爬升
    // 2.9 m，而且爬升速度建立后不会自行消失。用实际倾角补偿则竖直分量在任意
    // 时刻都等于 mg（T·cosθ_act = mg/cosθ_act · cosθ_act），高度自然守住。
    double thrust = m * g;
    if (_mc.thrust_compensation) {
        // R33 即机体 z 轴在 NED 竖直方向的投影，等于 cos(实际倾角)
        const double cos_actual = 1.0 - 2.0 * (q_cur.x * q_cur.x + q_cur.y * q_cur.y);
        // 下限 0.2 对应约 78° 倾角，防止极端姿态下推力爆掉
        thrust /= std::max(0.2, cos_actual);
    }
    thrust += clampd(stick.throttle, -1.0, 1.0) * _mc.throttle_range;
    cmd.thrust_body = std::max(0.0, thrust);

    return cmd;
}

} // namespace oi3
