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

void SixDofSimulator::advanceActuator(double thrust_cmd, const Tensor &torque_cmd) {
    const double limited_thrust = clampThrust(thrust_cmd, _config);
    const Tensor limited_torque = clampTorque(torque_cmd, _config);

    const bool has_dynamics =
        _config.actuator_tau > 0.0 || _config.actuator_rate_limit > 0.0;
    if (!has_dynamics) {
        // 理想执行器：直接透传。这条路径不引入任何额外运算，
        // 因此关闭执行机构动态时既有结果逐位不变。
        _act_thrust = limited_thrust;
        _act_torque = limited_torque;
        _act_initialized = true;
        return;
    }

    const double dt = _config.base.dt;
    double target = limited_thrust;
    Tensor target_torque = limited_torque;

    if (!_act_initialized) {
        // 首个步：无历史可比，直接置为指令值（避免从 0 起爬的假瞬态）
        _act_thrust = target;
        _act_torque = target_torque;
        _act_initialized = true;
        return;
    }

    // ---- 速率限幅（非线性）----
    if (_config.actuator_rate_limit > 0.0) {
        const double max_delta = _config.actuator_rate_limit * dt;
        const double d = target - _act_thrust;
        if (d > max_delta) {
            target = _act_thrust + max_delta;
        } else if (d < -max_delta) {
            target = _act_thrust - max_delta;
        }
        // 力矩三轴同样限速率
        const float *tp = target_torque.data<float>();
        const float *ap = _act_torque.data<float>();
        std::array<float, 3> lim{};
        for (int i = 0; i < 3; ++i) {
            const double d2 = static_cast<double>(tp[i]) - static_cast<double>(ap[i]);
            double v = tp[i];
            if (d2 > max_delta) {
                v = static_cast<float>(ap[i] + max_delta);
            } else if (d2 < -max_delta) {
                v = static_cast<float>(ap[i] - max_delta);
            }
            lim[static_cast<std::size_t>(i)] = v;
        }
        target_torque = makeVec3(lim[0], lim[1], lim[2]);
    }

    // ---- 一阶滞后：x += (u − x)·dt/τ，等效 1/(1 + τ·s) ----
    if (_config.actuator_tau > 0.0) {
        const double a = std::min(1.0, dt / _config.actuator_tau);
        _act_thrust += (target - _act_thrust) * a;

        const float *tp = target_torque.data<float>();
        const float *ap = _act_torque.data<float>();
        std::array<float, 3> next{};
        for (int i = 0; i < 3; ++i) {
            const double cur = ap[i];
            next[static_cast<std::size_t>(i)] =
                static_cast<float>(cur + (static_cast<double>(tp[i]) - cur) * a);
        }
        _act_torque = makeVec3(next[0], next[1], next[2]);
    } else {
        _act_thrust = target;
        _act_torque = target_torque;
    }
}

void SixDofSimulator::pushHistory() {
    const bool need_history =
        _config.sensor_delay > 0.0 || _config.sensor_rate_ratio > 0.0;
    if (!need_history) {
        return; // 无延迟需求时完全不维护历史，避免额外开销与内存占用
    }
    if (_history_count < kHistoryCap) {
        _history[static_cast<std::size_t>(_history_count)] = _state;
        ++_history_count;
    } else {
        for (int i = 1; i < kHistoryCap; ++i) {
            _history[static_cast<std::size_t>(i - 1)] =
                _history[static_cast<std::size_t>(i)];
        }
        _history[kHistoryCap - 1] = _state;
    }
}

const SixDofState &SixDofSimulator::observedState() const {
    const bool need_history =
        _config.sensor_delay > 0.0 || _config.sensor_rate_ratio > 0.0;
    if (!need_history || _history_count == 0) {
        return _state; // 无延迟：真值即观测量
    }

    // 延迟步数 = 传输延迟 + 采样量化（同频为 0，否则平均半帧）
    double delay_s = _config.sensor_delay;
    if (_config.sensor_rate_ratio > 1.0) {
        // 控制周期 / 采样周期 = ratio，量化误差取平均半帧
        delay_s += 0.5 * _config.base.dt / _config.sensor_rate_ratio;
    }
    int back = static_cast<int>(delay_s / _config.base.dt + 0.5);
    back = std::max(0, std::min(back, _history_count - 1));
    return _history[static_cast<std::size_t>(_history_count - 1 - back)];
}

void SixDofSimulator::step(double thrust_body, const Tensor &torque) {
    // 先把当前真值存入历史（供本步之后的传感器延迟查询）
    pushHistory();

    // 推进执行机构动态，得到**实际**作用于机体的推力与力矩
    advanceActuator(thrust_body, torque);
    const double limited_thrust = _act_thrust;
    const Tensor limited_torque = _act_torque;

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
    _act_thrust = 0.0;
    _act_torque = Tensor{};
    _act_initialized = false;
    _history_count = 0;
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
