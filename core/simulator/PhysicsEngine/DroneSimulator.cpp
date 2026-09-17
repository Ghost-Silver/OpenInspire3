/**
 * @file DroneSimulator.cpp
 * @brief OpenInspire3 飞行仿真器实现
 * @author GhostFace
 * @date 2026/9/16
 */

#include "DroneSimulator.h"
#include "TensorUtils.h"

#include <iostream>
#include <utility>

namespace oi3 {

DroneSimulator::DroneSimulator(Config config, DroneState initial_state)
    : _config(config), _state(std::move(initial_state)) {}

void DroneSimulator::step(const Tensor &thrust) {
    // 执行器饱和发生在推力作用于机体之前，故在积分前限幅。
    //
    // 限幅按分支写成两条互斥路径，而不是先构造一个 `Tensor limited = thrust;`
    // 再判断——那是一次拷贝构造，会把 limited 的 autograd 节点替换成新建的
    // GradAccumulator，与上游就此断开：前向数值完全正确，但梯度恒为零。
    // 因此「不需要限幅」这条路径必须原样传递引用，不做任何张量拷贝。
    if (_config.max_thrust > 0.0) {
        const float limit = static_cast<float>(_config.max_thrust);
        _state = rk4StepDrone(_state, thrust.clamp(-limit, limit), _config, _config.dt);
    } else {
        _state = rk4StepDrone(_state, thrust, _config, _config.dt);
    }
    _time += _config.dt;
    ++_step_count;
}

void DroneSimulator::step(const Tensor &thrust, int n) {
    for (int i = 0; i < n; ++i) {
        step(thrust);
    }
}

void DroneSimulator::reset(DroneState initial_state) {
    _state = std::move(initial_state);
    _time = 0.0;
    _step_count = 0;
}

void DroneSimulator::print() const {
    std::cout << "t = " << _time << " s, ";
    printVector(_state.pos, "pos[n,e,d]", std::cout);
    std::cout << ", ";
    printVector(_state.vel, "vel[n,e,d]", std::cout);
    std::cout << std::endl;
}

} // namespace oi3
