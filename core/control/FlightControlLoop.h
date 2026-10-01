/**
 * @file FlightControlLoop.h
 * @brief 飞控主循环：把估计、决策、控制、执行串成独立生产代码
 *
 * @par 与测试代码的区别
 *
 * `AccelFaultClosedLoopTest` 里的循环是参考实现，但它埋在测试文件里、
 * 与具体的 IMU 模型和仿真器耦合。本文件把它提取成：
 * - 通过 HAL 接口与底层解耦；
 * - 降级协调逻辑内聚在主循环中（决策同时作用于估计器与执行器）；
 * - 可被真机主程序直接包含使用。
 *
 * @par 降级协调是核心职责
 *
 * 主循环不做「降级决策」（那是 ImuDegradePolicy 的事），
 * 也不做「降级执行」（那是 DegradeExecutor 的事）。
 * 它的专属职责是**协调**：把决策同时应用到估计器（校正开关/位置信任）
 * 与执行器（限幅/接管）。这个协调不能藏在执行器里（它不该持有估计器），
 * 也不能拆到两个独立线程（决策与执行之间不允许有 race）。
 *
 * @par 单周期调用顺序
 *
 * @verbatim
 *   1. readImu() → updateImu()                 // 高频，每步
 *   2. [若有] readPosition() → updatePosition() // 低频，例如每 10 步
 *   3. sensorHealth() → decide()                // 做决策
 *   4. applyDecision() → setAccelCorrectionEnabled / setTrustPosition / setDecision
 *   5. compute() → writeCommand()               // 控制输出
 * @endverbatim
 */

#ifndef OI3_FLIGHT_CONTROL_LOOP_H
#define OI3_FLIGHT_CONTROL_LOOP_H

#include "DegradeExecutor.h"
#include "HalAbstraction.h"
#include "ImuDegradePolicy.h"
#include "SixDofPidController.h"
#include "StateEstimator.h"

namespace oi3 {

/**
 * @struct FlightControlConfig
 * @brief 主循环配置：把各子模块的配置打包，避免构造时传十几个参数
 */
struct FlightControlConfig {
    EstimatorConfig estimator{};
    DegradePolicyConfig policy{};
    DegradeExecutorConfig executor{};
    SixDofPidGains pid_gains{};

    /// 位置测量更新频率（Hz），用于计算 decimation
    double pos_update_hz = 100.0;

    /// 悬停推力（N），由机体质量决定，用于紧急降落计算
    double hover_thrust = 9.81;

    /// 到达判定容差（米）
    double arrival_tolerance = 0.05;
};

/**
 * @class FlightControlLoop
 * @brief 飞控主循环
 */
class FlightControlLoop {
  public:
    /**
     * @param cfg        主循环配置
     * @param sensors    传感器读取（不拥有生命周期）
     * @param actuators  执行器写入（不拥有生命周期）
     * @param setpoint   目标点来源（不拥有生命周期）
     * @param controller 内层控制器（不拥有生命周期）。
     *                   通常为 SixDofPidController；将来可为 MPC 或学习型控制器。
     */
    FlightControlLoop(const FlightControlConfig &cfg,
                      HalSensorReader *sensors,
                      HalActuatorWriter *actuators,
                      HalSetpointSource *setpoint,
                      SixDofController *controller);

    /**
     * @brief 初始化估计器姿态
     *
     * 应在首次 runOneCycle() 前调用。若未调用，估计器默认水平姿态。
     */
    void init(const std::array<double, 4> &initial_quat = {1.0, 0.0, 0.0, 0.0});

    /**
     * @brief 运行一个控制周期
     *
     * @return true  本周期正常执行
     * @return false 已触发紧急降落完成或失控，不应继续调用
     */
    bool runOneCycle();

    /// 当前步数
    [[nodiscard]] int stepCount() const { return _step_count; }

    /// 当前是否处于紧急降落状态
    [[nodiscard]] bool isEmergency() const { return _emergency; }

    /// 当前降级决策（供日志与地面站）
    [[nodiscard]] const DegradeDecision &currentDecision() const { return _decision; }

    /// 当前传感器健康报告（供日志与地面站）
    [[nodiscard]] const SensorHealthReport &currentHealth() const { return _health; }

    /// 估计器状态（供诊断与外部可视化）
    [[nodiscard]] const StateEstimator &estimator() const { return _estimator; }

    /// 飞行时间（秒）
    [[nodiscard]] double flightTime() const;

    /// 主循环配置（只读）
    [[nodiscard]] const FlightControlConfig &config() const { return _cfg; }

  private:
    /// 把降级决策同时应用到估计器与执行器
    void applyDegradation(const DegradeDecision &d);

    FlightControlConfig _cfg;
    HalSensorReader *_sensors;
    HalActuatorWriter *_actuators;
    HalSetpointSource *_setpoint;

    StateEstimator _estimator;
    ImuDegradePolicy _policy;
    DegradeExecutor _executor;
    SixDofController *_controller;

    int _step_count = 0;
    bool _emergency = false;
    DegradeDecision _decision{};
    SensorHealthReport _health{};

    /// 每多少步更新一次位置（由构造时根据 pos_update_hz 计算）
    int _pos_decim = 10;
};

} // namespace oi3

#endif // OI3_FLIGHT_CONTROL_LOOP_H
