/**
 * @file SixDofSimulator.h
 * @brief 六自由度飞行仿真器
 * @author GhostFace
 * @date 2026/9/17
 */

#ifndef OI3_SIX_DOF_SIMULATOR_H
#define OI3_SIX_DOF_SIMULATOR_H

#include "SixDofDynamics.h"
#include "SixDofTypes.h"

#include <array>
#include "Tensor.h"
#include "WindModel.h"

namespace oi3 {

/**
 * @class SixDofSimulator
 * @brief 六自由度仿真器：持有状态与配置，把动力学串成仿真循环
 *
 * 执行器饱和在此处生效：推力受 `max_body_thrust` 约束、力矩受 `torque_limit`
 * 约束。约束施加在积分之前 —— 这是执行器的物理位置，也避免控制器给出物理上
 * 无法实现的指令却不被察觉。
 */
class SixDofSimulator {
  public:
    SixDofSimulator(SixDofConfig config, SixDofState initial_state);

    /**
     * @brief 推进一个步长
     * @param thrust_body 机体 z 轴推力（牛顿，向上为正，会被限幅到 [0, max]）
     * @param torque      机体三轴力矩 {3}（N·m，会被限幅）
     */
    void step(double thrust_body, const Tensor &torque);

    /// 连续推进 n 步（指令保持不变）
    void step(double thrust_body, const Tensor &torque, int n);

    void reset(SixDofState initial_state);

    /**
     * @brief 设置风场（不获取所有权，传 nullptr 表示无风）
     *
     * 风在每步推进前查询一次并按步内常值处理。无风或风速为零时不构造任何张量，
     * 因此对既有（无风）仿真的开销为零、结果逐位不变。
     *
     * @note 风场对象由调用方持有并管理生命周期；simulator 不会重置它的内部状态。
     */
    void setWind(WindModel *wind) { _wind = wind; }

    /// 当前风场（可能为空）
    [[nodiscard]] WindModel *wind() const { return _wind; }

    /// 当前时刻的风速（无风时返回零向量）
    [[nodiscard]] WindVec currentWind() const;

    [[nodiscard]] const SixDofState &state() const { return _state; }
    [[nodiscard]] const SixDofConfig &config() const { return _config; }
    [[nodiscard]] double time() const { return _time; }
    [[nodiscard]] long long stepCount() const { return _step_count; }

    /**
     * @brief 控制器实际看到的状态（含传感器延迟）
     *
     * 无延迟配置（`sensor_delay <= 0` 且 `sensor_rate_ratio <= 0`）时返回真值，
     * 与 `state()` 完全一致 —— 保证既有调用方行为不变。
     *
     * 有延迟时返回历史缓冲中对应时刻的状态。这是**控制器侧**该用的接口：
     * 真机上控制器拿不到当前真值，只能拿到若干毫秒前的测量。
     */
    [[nodiscard]] const SixDofState &observedState() const;

    /// 当前执行机构实际输出的推力（含一阶滞后与速率限幅）
    [[nodiscard]] double actuatorThrust() const { return _act_thrust; }

    /**
     * @brief 设置推力效率（运行时故障注入）
     *
     * 用于在仿真中途模拟桨叶损伤、电机退化等造成的推力损失。
     * 与电机失效（MotorMixer 的 failed 数组）不同 —— 那是某电机完全失效、
     * 由混控重新分配；这里是**推力整体打折**，混控无从察觉，只能由 FDI
     * 从飞行数据中发现。
     *
     * @param eff 效率系数（0~1），1 表示正常
     */
    void setThrustEfficiency(double eff) { _config.thrust_efficiency = eff; }

    /// 当前执行机构实际输出的力矩
    [[nodiscard]] const Tensor &actuatorTorque() const { return _act_torque; }

    /// 打印当前位置、速度与姿态（欧拉角，便于阅读）
    void print() const;

  private:
    /// 推进执行机构动态（一阶滞后 + 速率限幅）；关闭时直接透传
    void advanceActuator(double thrust_cmd, const Tensor &torque_cmd);
    /// 把当前真值推入历史缓冲（供传感器延迟读取）
    void pushHistory();

    SixDofConfig _config;
    SixDofState _state;
    double _time = 0.0;
    long long _step_count = 0;
    WindModel *_wind = nullptr; ///< 非拥有指针

    // ---- 执行机构状态 ----
    double _act_thrust = 0.0;  ///< 实际推力（一阶滞后后的值）
    Tensor _act_torque;        ///< 实际力矩
    bool _act_initialized = false;

    // ---- 传感器延迟历史缓冲 ----
    //
    // 环形缓冲存最近若干个状态快照。容量按最大延迟需求分配（默认 64 步，
    // 在 1 kHz 下覆盖 64 ms，足以容纳 IMU 滤波 + 总线 + 解算的典型延迟）。
    static constexpr int kHistoryCap = 64;
    std::array<SixDofState, kHistoryCap> _history{};
    int _history_count = 0;
};

/// 构造水平姿态的单位四元数 (w=1, x=y=z=0)
[[nodiscard]] inline Tensor identityQuat() {
    return Tensor{1.0f, 0.0f, 0.0f, 0.0f};
}

} // namespace oi3

#endif // OI3_SIX_DOF_SIMULATOR_H
