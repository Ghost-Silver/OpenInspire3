/**
 * @file ShootingMpc.cpp
 * @brief 滚动时域控制器实现
 * @author GhostFace
 * @date 2026/9/18
 *
 * 每个控制周期只重规划一次、执行前 control_horizon 段动作；重规划时把上一次的解
 * 向前平移作为热启动。整段实现复用 ShootingExpert 的求解器与 HoverEnv 的动作契约，
 * 因此 MPC 的输出可以直接喂给同一个环境，与 PID 做同条件对比。
 */

#include "ShootingMpc.h"

#include "TensorUtils.h"

#include <algorithm>
#include <chrono>
#include <cmath>

namespace oi3 {

ShootingMpc::ShootingMpc(HoverEnvConfig env, Config cfg)
    : _env(std::move(env)), _cfg(cfg) {
    _cfg.horizon = std::max(1, _cfg.horizon);
    _cfg.control_horizon = std::max(1, std::min(_cfg.control_horizon, _cfg.horizon));
}

bool ShootingMpc::replan(const DroneState &st, const std::array<double, 3> &target) {
    const std::vector<float> pos = toVector(st.pos);
    const std::vector<float> vel = toVector(st.vel);
    const std::array<double, 3> p0 = {pos[0], pos[1], pos[2]};
    const std::array<double, 3> v0 = {vel[0], vel[1], vel[2]};

    // 热启动：把上一次的解向前平移 control_horizon 段（已执行的部分丢弃），
    // 末尾用最后一段补齐。相邻两次规划的问题只差一小段时间平移，因此初值已经很近。
    std::vector<std::array<double, 3>> warm;
    if (!_warm.empty()) {
        warm.reserve(static_cast<std::size_t>(_cfg.horizon));
        for (int s = 0; s < _cfg.horizon; ++s) {
            const std::size_t src = static_cast<std::size_t>(s + _cfg.control_horizon);
            if (src < _warm.size()) {
                warm.push_back(_warm[src]);
            } else if (!_warm.empty()) {
                warm.push_back(_warm.back());
            }
        }
    }

    ShootingConfig sc;
    sc.segments = _cfg.horizon;
    sc.iters = _cfg.iters;
    sc.step = _cfg.step;
    sc.w_traj = _cfg.w_traj;

    const auto t0 = std::chrono::steady_clock::now();
    ShootingResult r = shootHover(p0, v0, target, _env, sc, warm);
    const auto t1 = std::chrono::steady_clock::now();
    _plan_seconds += std::chrono::duration<double>(t1 - t0).count();
    ++_plan_count;

    if (!r.finite || r.action_seq.empty()) {
        _actions.clear();
        _cursor = 0;
        return false;
    }

    _warm = r.thrust_seq;
    _last_cost = r.cost;

    // 只保留控制时域内的动作：其余留作下一次的热启动信息
    _actions.clear();
    _actions.reserve(static_cast<std::size_t>(_cfg.control_horizon));
    for (int s = 0; s < _cfg.control_horizon; ++s) {
        const std::size_t idx = static_cast<std::size_t>(s);
        if (idx < r.action_seq.size()) {
            _actions.push_back(r.action_seq[idx]);
        }
    }
    _cursor = 0;
    return !_actions.empty();
}

bool ShootingMpc::nextAction(std::array<float, 3> &action) {
    if (_cursor >= _actions.size()) {
        return false;
    }
    action = _actions[_cursor];
    ++_cursor;
    return true;
}

} // namespace oi3
