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

    /// 打印当前位置、速度与姿态（欧拉角，便于阅读）
    void print() const;

  private:
    SixDofConfig _config;
    SixDofState _state;
    double _time = 0.0;
    long long _step_count = 0;
    WindModel *_wind = nullptr; ///< 非拥有指针
};

/// 构造水平姿态的单位四元数 (w=1, x=y=z=0)
[[nodiscard]] inline Tensor identityQuat() {
    return Tensor{1.0f, 0.0f, 0.0f, 0.0f};
}

} // namespace oi3

#endif // OI3_SIX_DOF_SIMULATOR_H
