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

    /**
     * @brief 触地判定高度（米）
     *
     * 紧急降落时，高度降到该值以下即视为着陆完成、主循环停止。
     * 取 0.10 m 而非 0：仿真与真机的垂直接近速度都会使高度短暂越过零点，
     * 用严格的 0 可能永不满足。
     */
    double touchdown_altitude = 0.10;

    /**
     * @brief 默认/回退步长（秒）
     *
     * 当 HAL 不提供 IMU 时间戳（`imuTimestamp() <= 0`）时使用。默认 0.001
     * 等于改动前硬编码的值，故不实现时间戳的 HAL（如 SimSensorReader）
     * 行为逐位不变。
     */
    /**
     * @brief 连续多少次「执行器写入未被接受」即判定执行器失联
     *
     * 取 10 次（1 kHz 下 10 ms）：单次写入失败可能只是总线抖动，连续失败则
     * 基本可确认执行器链路异常。
     */
    int actuator_fail_steps = 10;

    double default_dt = 0.001;

    /**
     * @brief 时间戳差分得到的 dt 的合理性上界（秒）
     *
     * 超出此值的间隔视为异常（时间戳跳变、长时间丢帧），忽略并回退到
     * `default_dt`。另设下界 `min_dt` 防止时间戳抖动导致 dt 趋零。
     */
    double max_dt = 0.05;
    double min_dt = 1e-6;

    /**
     * @brief 紧急降落的超时保护周期数
     *
     * 若因传感器失效导致触地始终无法判定，则强制结束，避免主循环无限运行。
     * 默认 30000 周期（1 kHz 下 30 s），足以从数米高度降落到地面。
     */
    int max_emergency_cycles = 30000;
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
     * @return false 已触发紧急降落，不应继续调用；控制权应交由外部处理
     *
     * @warning 返回 false **不表示降落已完成**。本循环只负责「检测到需要紧急
     *          降落并停止接受指令」，**不执行降落过程本身** —— 既不生成下降
     *          轨迹，也不做触地检测，飞机此时仍停在原高度。
     *
     *          因此集成方必须提供外部处理：或由上层状态机接管下降与触地判定，
     *          或由 HAL 层提供降落执行接口。若直接停止调用本循环而不作处理，
     *          飞机会保持当前高度悬停至电量耗尽。
     *
     *          「自主紧急降落」（持续下降至触地）是本模块的已知空缺。
     */
    bool runOneCycle();

    /// 当前步数
    [[nodiscard]] int stepCount() const { return _step_count; }

    /// 当前是否处于紧急降落状态
    [[nodiscard]] bool isEmergency() const { return _emergency; }

    /**
     * @brief 紧急降落是否已完成（已触地或超时）
     *
     * 与 `isEmergency()` 的区别：`isEmergency()` 表示「已进入紧急降落」，
     * 此时主循环仍在运行并持续输出下降指令；本函数为真表示降落过程已结束。
     */
    [[nodiscard]] bool isEmergencyComplete() const { return _emergency_complete; }

    /**
     * @brief 执行器是否已判定失联
     *
     * 连续 `actuator_fail_steps` 次写入未被接受后置位。**注意：这是告知性
     * 状态，不构成可恢复故障** —— 执行器失联后没有补救手段，该标志用于
     * 触发外部告警与被动安全措施。
     */
    [[nodiscard]] bool isActuatorLost() const { return _actuator_lost; }

    /// 累计写入失败次数（供黑匣子 / 地面站）
    [[nodiscard]] long long actuatorFailures() const { return _actuator_fail_total; }

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

    /// 紧急降落是否已完成（触地或超时）
    bool _emergency_complete = false;

    /// 进入紧急降落后的周期计数，用于超时保护
    int _emergency_cycles = 0;

    /// 上一次 IMU 时间戳（<=0 表示尚无有效值）
    double _last_imu_timestamp = -1.0;

    /// 连续执行器写入失败计数
    int _actuator_fail_count = 0;

    /// 累计执行器写入失败次数（供外部诊断）
    long long _actuator_fail_total = 0;

    /// 执行器是否已判定失联
    bool _actuator_lost = false;
    DegradeDecision _decision{};
    SensorHealthReport _health{};

    /// 每多少步更新一次位置（由构造时根据 pos_update_hz 计算）
    int _pos_decim = 10;
};

} // namespace oi3

#endif // OI3_FLIGHT_CONTROL_LOOP_H
