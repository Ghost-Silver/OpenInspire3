/**
 * @file DegradeExecutor.h
 * @brief 降级执行器：把 ImuDegradePolicy 的决策落到实际控制指令上
 *
 * @par 为什么做成包装器而不是改 PID
 *
 * 降级逻辑有三种可能的落点：塞进 PID 控制器、写在上层主循环、或做成独立的
 * 包装器。这里选第三种，理由有三条：
 *
 * 1. **PID 控制器是经过充分验证的核心**，把它与故障处理耦合会让每次改故障
 *    策略都可能影响正常飞行的控制品质。包装器让 PID 一行不改。
 * 2. **任何控制器都能被包装**。项目的 `SixDofController` 是纯虚接口，定点 PID、
 *    手动模式、将来的学习型控制器都实现它。写死在 PID 里就等于其他控制器
 *    无法复用这套降级能力。
 * 3. **可单独测试**。降级行为与 PID 的调参互不干扰，测试可以分别验证
 *    「PID 正常工作时对不对」与「降级时是否正确接管」。
 *
 * @par 三种模式如何落实
 *
 * - **Normal**：完全透传给内层控制器，**逐位不变**。这是默认状态，
 *   也是本文件不改变既有行为的保证。
 * - **EmergencyLand**：**不再调用内层控制器**，直接输出「低于悬停的推力 +
 *   零力矩」。闭环实测证实这样做能让飞机以接近水平的姿态落地，而继续做姿态
 *   控制反而会因反馈不可信而加剧发散（见 docs/imu-fault-tolerance.md §6.1）。
 * - **Cautious / ReturnHome**：仍调用内层控制器，但对输出力矩**限幅**，
 *   按决策给出的倾角上限相对标称值的比例缩放。力矩上限降低即等效限制了
 *   机动能力，避免大机动把传感器偏置误差放大。
 *
 * @par 一个必须说明的接口约束
 *
 * 内层控制器以裸指针持有，**不接管其生命周期**。调用方须保证内层对象的生存期
 * 长于本执行器。这是刻意的：执行器不应凭空拥有一个由外部创建的控制器。
 */

#ifndef OI3_DEGRADE_EXECUTOR_H
#define OI3_DEGRADE_EXECUTOR_H

#include "DroneTypes.h"    // makeVec3
#include "ImuDegradePolicy.h"
#include "SixDofPidController.h"
#include "SixDofTypes.h"
#include "TensorUtils.h"   // toVector

#include <algorithm>
#include <cmath>
#include <vector>

namespace oi3 {

/// 降级执行器配置
struct DegradeExecutorConfig {
    /// 标称倾角上限（度），用于把 max_tilt_deg 归一化成力矩缩放系数
    double nominal_tilt_deg = 35.0;

    /// 紧急降落时的推力系数（相对悬停推力的比例）。
    /// 取 0.6：低于悬停使飞机持续下降，但保留足够推力避免自由落体式的冲击。
    double emergency_thrust_ratio = 0.6;

    /// 悬停推力（牛顿）。由调用方按机体质量填入（m · g）。
    double hover_thrust = 0.0;
};

/**
 * @class DegradeExecutor
 * @brief 按降级决策切换控制行为的包装器
 */
class DegradeExecutor : public SixDofController {
  public:
    /**
     * @param inner 内层控制器（不接管生命周期，须存活至本对象销毁）
     * @param cfg   执行器配置
     */
    explicit DegradeExecutor(SixDofController &inner, const DegradeExecutorConfig &cfg = {})
        : _inner(&inner), _cfg(cfg) {}

    /**
     * @brief 更新当前降级决策
     *
     * 由飞控主循环每周期调用，传入由 ImuDegradePolicy 算出的决策。
     * 未调用时默认为 Normal（完全透传）。
     */
    void setDecision(const DegradeDecision &d) { _decision = d; }

    /// 当前决策
    [[nodiscard]] const DegradeDecision &decision() const { return _decision; }

    /// 当前是否处于紧急降落
    [[nodiscard]] bool emergency() const {
        return _decision.action == DegradeAction::EmergencyLand;
    }

    /**
     * @brief 计算控制指令
     *
     * @note 紧急降落时**不调用**内层控制器。这是关键设计：陀螺失效后姿态反馈
     *       不可信，继续做姿态修正会用错误反馈加剧发散；切断力矩让飞机在重力下
     *       自然保持姿态。闭环实测支持这一做法（触地倾角 0.38° vs 不降级的 164.61°）。
     */
    [[nodiscard]] SixDofCommand compute(const SixDofState &state, const Tensor &target,
                                        double time) override {
        if (!emergency()) {
            // 正常/谨慎/返航：调用内层控制器，必要时限幅
            SixDofCommand cmd = _inner->compute(state, target, time);
            if (_decision.action != DegradeAction::Normal) {
                limitTorque(cmd);
            }
            return cmd;
        }

        // ---- 紧急降落：完全接管 ----
        SixDofCommand cmd;
        cmd.thrust_body = _cfg.hover_thrust * _cfg.emergency_thrust_ratio;
        cmd.torque = makeVec3(0.0f, 0.0f, 0.0f);
        return cmd;
    }

    [[nodiscard]] const char *name() const override { return "DegradeExecutor"; }

    void reset() override { _inner->reset(); }

  private:
    /// 按决策的倾角上限缩放输出力矩（等效限制机动能力）
    void limitTorque(SixDofCommand &cmd) const {
        if (!cmd.valid() || _cfg.nominal_tilt_deg <= 1e-9) {
            return;
        }
        // 缩放系数 = 当前允许倾角 / 标称倾角，截断到 [0,1]。
        // 用倾角比例而非直接给定力矩上限：倾角上限是有物理含义的量
        // （g·tan(θ) 即水平加速度能力），而力矩上限依机型而异、难以直观设定。
        const double ratio =
            std::max(0.0, std::min(1.0, _decision.max_tilt_deg / _cfg.nominal_tilt_deg));
        const std::vector<float> tq = toVector(cmd.torque);
        Tensor limited =
            makeVec3(static_cast<float>(static_cast<double>(tq[0]) * ratio),
                     static_cast<float>(static_cast<double>(tq[1]) * ratio),
                     static_cast<float>(static_cast<double>(tq[2]) * ratio));
        cmd.torque = limited;
    }

    SixDofController *_inner; ///< 内层控制器（不拥有）
    DegradeExecutorConfig _cfg;
    DegradeDecision _decision{}; ///< 默认 Normal
};

} // namespace oi3

#endif // OI3_DEGRADE_EXECUTOR_H
