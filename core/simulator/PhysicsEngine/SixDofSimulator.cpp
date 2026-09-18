/**
 * @file SixDofSimulator.cpp
 * @brief 六自由度飞行仿真器实现
 * @author GhostFace
 * @date 2026/9/17
 */

#include "SixDofSimulator.h"
#include "TensorUtils.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <utility>

namespace oi3 {

namespace {

/// 推力限幅：[0, max]，负推力在真实螺旋桨上不可实现（只能减速不能反推）
double clampThrust(double thrust, const SixDofConfig &cfg) {
    if (cfg.max_body_thrust <= 0.0) {
        return std::max(0.0, thrust);
    }
    return std::max(0.0, std::min(cfg.max_body_thrust, thrust));
}

/// 力矩逐轴限幅
Tensor clampTorque(const Tensor &torque, const SixDofConfig &cfg) {
    if (cfg.torque_limit <= 0.0) {
        return Tensor(torque);
    }
    const float limit = static_cast<float>(cfg.torque_limit);
    return torque.clamp(-limit, limit);
}

} // namespace

SixDofSimulator::SixDofSimulator(SixDofConfig config, SixDofState initial_state)
    : _config(std::move(config)), _state(std::move(initial_state)) {}

void SixDofSimulator::step(double thrust_body, const Tensor &torque) {
    const double limited_thrust = clampThrust(thrust_body, _config);
    const Tensor limited_torque = clampTorque(torque, _config);

    // 风速按步首值取一次。无风路径完全不构造张量 —— 既避免 1 kHz 热路径上的
    // 无谓分配，也保证无风时的数值结果与加风场之前逐位一致。
    Tensor wind_vec;
    const Tensor *wind_ptr = nullptr;
    if (_wind != nullptr) {
        const WindVec w = _wind->at(_time);
        if (w[0] != 0.0 || w[1] != 0.0 || w[2] != 0.0) {
            wind_vec = makeVec3(static_cast<float>(w[0]), static_cast<float>(w[1]),
                                static_cast<float>(w[2]));
            wind_ptr = &wind_vec;
        }
    }

    _state =
        rk4StepSixDof(_state, limited_thrust, limited_torque, _config, _config.base.dt, wind_ptr);
    _time += _config.base.dt;
    ++_step_count;
}

WindVec SixDofSimulator::currentWind() const {
    if (_wind == nullptr) {
        return {0.0, 0.0, 0.0};
    }
    return _wind->at(_time);
}

void SixDofSimulator::step(double thrust_body, const Tensor &torque, int n) {
    for (int i = 0; i < n; ++i) {
        step(thrust_body, torque);
    }
}

void SixDofSimulator::reset(SixDofState initial_state) {
    _state = std::move(initial_state);
    _time = 0.0;
    _step_count = 0;
}

void SixDofSimulator::print() const {
    const std::vector<float> pos = toVector(_state.pos);
    const std::vector<float> vel = toVector(_state.vel);
    const std::vector<float> euler = toVector(quatToEuler(_state.quat));

    const double rad2deg = 180.0 / M_PI;
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "t = " << _time << " s"
              << "  pos[n,e,d] = [" << pos[0] << ", " << pos[1] << ", " << pos[2] << "]"
              << "  vel = [" << vel[0] << ", " << vel[1] << ", " << vel[2] << "]"
              << "  rpy(deg) = [" << euler[0] * rad2deg << ", " << euler[1] * rad2deg
              << ", " << euler[2] * rad2deg << "]" << std::endl;
}

} // namespace oi3
