/**
 * @file FlightControlLoop.cpp
 * @brief 飞控主循环实现
 */

#include "FlightControlLoop.h"

#include <algorithm>
#include <cmath>

namespace oi3 {

FlightControlLoop::FlightControlLoop(const FlightControlConfig &cfg,
                                     HalSensorReader *sensors,
                                     HalActuatorWriter *actuators,
                                     HalSetpointSource *setpoint,
                                     SixDofController *controller)
    : _cfg(cfg), _sensors(sensors), _actuators(actuators), _setpoint(setpoint),
      _estimator(cfg.estimator), _policy(cfg.policy),
      _executor(*controller, cfg.executor), _controller(controller) {

    // 位置更新分频：IMU 频率 / 位置频率。
    // 默认 IMU 1 kHz，位置 100 Hz → decim = 10。
    if (cfg.pos_update_hz > 1e-6) {
        _pos_decim = static_cast<int>(std::round(1000.0 / cfg.pos_update_hz));
        _pos_decim = std::max(1, _pos_decim);
    } else {
        _pos_decim = 10;
    }
}

void FlightControlLoop::init(const std::array<double, 4> &initial_quat) {
    _estimator.reset(initial_quat);
    _controller->reset();
    _executor.reset();
    _step_count = 0;
    _emergency = false;
    _decision = DegradeDecision{};
    _health = SensorHealthReport{};
}

bool FlightControlLoop::runOneCycle() {
    if (_emergency) {
        return false;
    }

    // ---- 1. 读传感器 ----
    const ImuSample imu = _sensors->readImu();
    const double dt = 0.001; // 默认 1 kHz；将来可从时间戳推导

    // ---- 2. 更新估计器（IMU，高频） ----
    _estimator.updateImu(imu, dt);

    // ---- 3. 位置测量（低频，由 HAL 决定时机） ----
    if (_sensors->hasPositionUpdate()) {
        const auto pos = _sensors->readPosition();
        _estimator.updatePosition(pos, dt * _pos_decim);
    }

    // ---- 4. 降级决策 ----
    _health = _estimator.sensorHealth();
    _decision = _policy.decide(_health);

    // ---- 5. 应用降级（协调：估计器 + 执行器） ----
    applyDegradation(_decision);

    // ---- 5.5 通知制导源决策变化（让返航/紧急降落有轨迹可执行） ----
    const double t = _sensors->time();
    const auto est_pos_vec = toVector(_estimator.state().pos);
    const std::array<double, 3> est_pos = {static_cast<double>(est_pos_vec[0]),
                                           static_cast<double>(est_pos_vec[1]),
                                           static_cast<double>(est_pos_vec[2])};
    _setpoint->onDecisionChanged(_decision, est_pos, t);

    // ---- 6. 控制计算 ----
    const Tensor target = _setpoint->currentTarget(t);
    const SixDofCommand cmd = _executor.compute(_estimator.state(), target, t);

    // ---- 7. 输出执行器 ----
    _actuators->writeCommand(cmd);

    ++_step_count;

    // ---- 8. 紧急降落判定 ----
    // 紧急降落时 DegradeExecutor 不再调用内层控制器，直接输出固定推力。
    // 当飞机触地（高度接近零）或超时后，标记完成。
    // 此处只做高层状态标记，具体触地检测由 HAL 层提供或在外部处理。
    if (_decision.action == DegradeAction::EmergencyLand) {
        _emergency = true;
    }

    return true;
}

double FlightControlLoop::flightTime() const {
    return _sensors ? _sensors->time() : 0.0;
}

void FlightControlLoop::applyDegradation(const DegradeDecision &d) {
    // 关键：决策必须**同时**作用于估计器与执行器。
    // 只作用于执行器是不够的 —— 传感器失效后继续用错误数据做估计，
    // 会把污染持续注入控制状态（AccelFaultClosedLoopTest 已验证）。
    _estimator.setAccelCorrectionEnabled(d.use_accel_correction);
    _estimator.setTrustPosition(d.trust_position);
    _executor.setDecision(d);
}

} // namespace oi3
