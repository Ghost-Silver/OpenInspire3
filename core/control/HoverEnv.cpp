/**
 * @file HoverEnv.cpp
 * @brief 定点悬停环境实现
 * @author GhostFace
 * @date 2026/9/16
 */

#include "HoverEnv.h"
#include "TensorUtils.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace oi3 {

namespace {

double clampUnit(double v) {
    return std::max(-1.0, std::min(1.0, v));
}

} // namespace

HoverEnv::HoverEnv(HoverEnvConfig config, std::uint32_t seed)
    : _cfg(std::move(config)), _rng(seed),
      _sim(_cfg.plant, DroneState{makeVec3(0.0f, 0.0f, 0.0f),
                                  makeVec3(0.0f, 0.0f, 0.0f)}) {
    _steps_per_episode = std::max(
        1, static_cast<int>(_cfg.episode_seconds /
                            (_cfg.plant.dt * std::max(1, _cfg.control_decimation))));
}

std::vector<float> HoverEnv::makeObservation(double pos_err_n, double pos_err_e,
                                             double pos_err_d) const {
    std::vector<float> obs(static_cast<std::size_t>(kObsDim), 0.0f);

    // 位置误差按目标采样范围归一化，速度按 vel_scale 归一化，
    // 使各维量纲接近，便于网络学习
    const double pos_norm = std::max(1e-6, _cfg.target_range);
    const double vel_norm = std::max(1e-6, _cfg.vel_scale);

    obs[0] = static_cast<float>(pos_err_n / pos_norm);
    obs[1] = static_cast<float>(pos_err_e / pos_norm);
    obs[2] = static_cast<float>(pos_err_d / pos_norm);

    const std::vector<float> vel = toVector(_sim.state().vel);
    obs[3] = static_cast<float>(vel[0] / vel_norm);
    obs[4] = static_cast<float>(vel[1] / vel_norm);
    obs[5] = static_cast<float>(vel[2] / vel_norm);

    // 限幅：发散时位置误差可达数十米，归一化后数值远超训练分布，
    // 会导致策略输入爆炸且无法恢复
    const float clip = static_cast<float>(_cfg.obs_clip);
    for (float &v : obs) {
        v = std::max(-clip, std::min(clip, v));
    }
    return obs;
}

std::vector<float> HoverEnv::reset() {
    // 随机采样目标点与初始位置：策略必须对任意初值都能收敛，
    // 而不是过拟合到某一个固定场景
    for (int i = 0; i < 3; ++i) {
        _target[i] = _uniform(_rng) * _cfg.target_range;
    }

    const double p0[3] = {_uniform(_rng) * _cfg.start_range,
                          _uniform(_rng) * _cfg.start_range,
                          _uniform(_rng) * _cfg.start_range};

    _sim.reset(DroneState{makeVec3(static_cast<float>(p0[0]), static_cast<float>(p0[1]),
                                   static_cast<float>(p0[2])),
                          makeVec3(0.0f, 0.0f, 0.0f)});
    _step_index = 0;
    _in_band_steps = 0;
    _ever_held = false;
    _last_pos_error = 0.0;

    return makeObservation(_target[0] - p0[0], _target[1] - p0[1], _target[2] - p0[2]);
}

void HoverEnv::setCurriculum(double target_range, double abort_radius) {
    _cfg.target_range = std::max(1e-3, target_range);
    _cfg.abort_radius = std::max(_cfg.target_range * 2.0, abort_radius);
}

std::array<float, 3> HoverEnv::thrustToAction(const std::array<double, 3> &thrust) const {
    const double m = _cfg.plant.mass;
    const double g = _cfg.plant.gravity;
    const double hover[3] = {0.0, 0.0, -m * g};
    const double scale = std::max(1e-9, _cfg.thrust_scale * m * g);

    std::array<float, 3> action{};
    for (int i = 0; i < 3; ++i) {
        const double a = (thrust[static_cast<std::size_t>(i)] - hover[i]) / scale;
        action[static_cast<std::size_t>(i)] =
            static_cast<float>(std::max(-1.0, std::min(1.0, a)));
    }
    return action;
}

HoverEnv::StepResult HoverEnv::step(const std::array<float, 3> &action) {
    const double m = _cfg.plant.mass;
    const double g = _cfg.plant.gravity;

    // 动作以悬停推力为中心：网络只需学习相对悬停量的修正
    const double hover[3] = {0.0, 0.0, -m * g};
    const double scale = _cfg.thrust_scale * m * g;

    const float thrust_n = static_cast<float>(hover[0] + clampUnit(action[0]) * scale);
    const float thrust_e = static_cast<float>(hover[1] + clampUnit(action[1]) * scale);
    const float thrust_d = static_cast<float>(hover[2] + clampUnit(action[2]) * scale);

    // 该控制量在整个控制周期内保持不变，仿真以 1kHz 细步推进
    const Tensor thrust = makeVec3(thrust_n, thrust_e, thrust_d);
    for (int i = 0; i < std::max(1, _cfg.control_decimation); ++i) {
        _sim.step(thrust);
    }

    const std::vector<float> pos = toVector(_sim.state().pos);
    const std::vector<float> vel = toVector(_sim.state().vel);

    const double err[3] = {_target[0] - static_cast<double>(pos[0]),
                           _target[1] - static_cast<double>(pos[1]),
                           _target[2] - static_cast<double>(pos[2])};
    const double pos_err2 = err[0] * err[0] + err[1] * err[1] + err[2] * err[2];
    const double vel2 = static_cast<double>(vel[0]) * vel[0] +
                        static_cast<double>(vel[1]) * vel[1] +
                        static_cast<double>(vel[2]) * vel[2];

    // 推力偏离悬停量的归一化平方
    const double dn = (static_cast<double>(thrust_n) - hover[0]) / std::max(1e-9, scale);
    const double de = (static_cast<double>(thrust_e) - hover[1]) / std::max(1e-9, scale);
    const double dd = (static_cast<double>(thrust_d) - hover[2]) / std::max(1e-9, scale);
    const double thrust_dev2 = dn * dn + de * de + dd * dd;

    StepResult result;
    result.reward = static_cast<float>(-(_cfg.w_pos * pos_err2 + _cfg.w_vel * vel2 +
                                         _cfg.w_thrust * thrust_dev2) *
                                       _cfg.reward_scale);

    const double pos_error = std::sqrt(pos_err2);
    const double speed = std::sqrt(vel2);
    _last_pos_error = pos_error;

    // 连续在带计数：成功 = 回合内曾经连续稳定悬停，而非结束时刻恰好合格
    if (pos_error < _cfg.success_tolerance && speed < _cfg.success_speed) {
        ++_in_band_steps;
    } else {
        _in_band_steps = 0;
    }
    if (_in_band_steps >= _cfg.success_hold_steps) {
        _ever_held = true;
    }

    ++_step_index;
    const bool out_of_bounds = pos_error > _cfg.abort_radius;
    result.done = out_of_bounds || (_step_index >= _steps_per_episode);
    result.success = result.done && !out_of_bounds && _ever_held;
    if (out_of_bounds) {
        result.reward -= static_cast<float>(_cfg.abort_penalty);
    }

    result.obs = makeObservation(err[0], err[1], err[2]);
    return result;
}

} // namespace oi3
