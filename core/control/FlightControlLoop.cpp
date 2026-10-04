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
    _emergency_complete = false;
    _emergency_cycles = 0;
    _decision = DegradeDecision{};
    _health = SensorHealthReport{};
}

bool FlightControlLoop::runOneCycle() {
    // 入口守卫：仅在**降落已完成**（触地或超时）时停止。
    //
    // 曾经的错误做法是在 `_emergency` 为真时立即返回 false，结果是紧急降落
    // 只被输出一拍就被切断 —— 实测陀螺失效后主循环于第 2150 周期终止，
    // 而飞机仍停在 5.000 m 原高度，始终没有下降。制导层生成的降落轨迹、
    // 执行器准备好的下降推力都无法执行。三个模块各自正确，组合起来降落无人执行。
    if (_emergency_complete) {
        return false;
    }

    // ---- 1. 读传感器 ----
    const ImuSample imu = _sensors->readImu();

    // 步长由 HAL 提供的 IMU 时间戳差分得到，而非硬编码。
    //
    // 原先硬编码 0.001（1 kHz），后果是系统只能在 1 kHz 附近工作：实测以
    // 仿真步长模拟真机频率时，500 Hz 即失控（末态高度 22.91 m、姿态误差
    // 峰值 76.15°），250 Hz 及以下坠地。500 Hz 在低成本飞控上很常见，
    // 该缺口会直接导致真机不可用。
    //
    // HAL 不提供时间戳（返回值 <= 0）时回退到 default_dt —— 其默认值等于
    // 原硬编码值，故既有行为逐位不变。
    double dt = _cfg.default_dt;
    {
        const double ts = _sensors->imuTimestamp();
        if (ts > 0.0) {
            if (_last_imu_timestamp > 0.0) {
                const double measured = ts - _last_imu_timestamp;
                // 合理性检查：非正、过小（抖动）或过大（跳变/丢帧）一律忽略，
                // 避免异常时间戳污染积分与滤波器。
                if (measured >= _cfg.min_dt && measured <= _cfg.max_dt) {
                    dt = measured;
                }
            }
            _last_imu_timestamp = ts;
        }
    }

    // ---- 2. 更新估计器（IMU，高频） ----
    // 一并传入 HAL 的读取有效标志：HAL 明确报告失败时，该帧数据不可信，
    // 健康检测走外部失败路径（而非从数据特征去猜）。
    _estimator.updateImu(imu, dt, _sensors->lastImuValid());

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

    // ---- 8. 紧急降落：持续执行下降，直到触地或超时 ----
    // 进入紧急降落后主循环**继续运行**：DegradeExecutor 不再调用内层控制器，
    // 而是持续输出低于悬停的固定推力，飞机因此在重力下下降。此处负责判定
    // 降落何时结束。
    //
    // 先前的实现只标记状态、下一周期即返回 false，导致降落从未真正执行
    // （实测末态高度仍是 5.000 m）。那属于「停止接受指令」与「停止整个循环」
    // 被混为一谈。现在两者分开：`_emergency` 表示降级状态，`_emergency_complete`
    // 才表示过程结束。
    if (_decision.action == DegradeAction::EmergencyLand) {
        if (!_emergency) {
            _emergency = true;
        }
        ++_emergency_cycles;

        // 触地判定：直接读量测高度（而非估计值）—— 降落是否结束属于物理事实，
        // 不应依赖此时已不可信的传感器估计。
        const auto pos = _sensors->readPosition();
        const double altitude = -static_cast<double>(pos[2]);
        if (altitude <= _cfg.touchdown_altitude) {
            _emergency_complete = true;
        } else if (_emergency_cycles >= _cfg.max_emergency_cycles) {
            // 超时保护：若因传感器失效导致触地始终无法判定，强制结束，
            // 避免主循环无限运行。
            _emergency_complete = true;
        }
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
