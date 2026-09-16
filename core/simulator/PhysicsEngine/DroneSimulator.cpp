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
    _state = rk4StepDrone(_state, thrust, _config, _config.dt);
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
