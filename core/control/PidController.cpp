/**
 * @file PidController.cpp
 * @brief 位置 PID 控制器实现
 * @author GhostFace
 * @date 2026/9/16
 */

#include "PidController.h"
#include "TensorUtils.h"

#include <algorithm>

namespace oi3 {

namespace {

double clampDouble(double value, double lo, double hi) {
    return std::max(lo, std::min(hi, value));
}

} // namespace

PidController::PidController(Config cfg, PidGains gains)
    : _cfg(cfg), _gains(gains) {}

void PidController::reset() {
    _integral[0] = 0.0;
    _integral[1] = 0.0;
    _integral[2] = 0.0;
}

Tensor PidController::computeThrust(const DroneState &state, const Tensor &target,
                                    double /*time*/) {
    const std::vector<float> pos = toVector(state.pos);
    const std::vector<float> vel = toVector(state.vel);
    const std::vector<float> tgt = toVector(target);

    const double dt = _cfg.dt;
    double a_des[3];

    for (int i = 0; i < 3; ++i) {
        const double e_pos = static_cast<double>(tgt[i]) - static_cast<double>(pos[i]);
        const double e_vel = -static_cast<double>(vel[i]);

        _integral[i] = clampDouble(_integral[i] + e_pos * dt, -_gains.integral_limit,
                                   _gains.integral_limit);

        const double raw = _gains.kp * e_pos + _gains.kd * e_vel + _gains.ki * _integral[i];
        a_des[i] = clampDouble(raw, -_gains.max_accel, _gains.max_accel);
    }

    // 重力补偿：F = m*a_des - m*g_vec，其中 g_vec = [0, 0, g]（NED 系向下为正）。
    // 悬停时 a_des = 0，输出 [0, 0, -m*g]，即抵消重力的向上推力。
    const float m = static_cast<float>(_cfg.mass);
    const float g = static_cast<float>(_cfg.gravity);
    return makeVec3(static_cast<float>(a_des[0]) * m, static_cast<float>(a_des[1]) * m,
                    static_cast<float>(a_des[2]) * m - m * g);
}

} // namespace oi3
