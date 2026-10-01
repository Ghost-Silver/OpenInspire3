/**
 * @file ImuDegradePolicy.h
 * @brief IMU 失效后的降级决策：由健康状态推出「现在该怎么办」
 *
 * @par 为什么策略要与检测分离
 *
 * SensorHealth 只回答「传感器还能信吗」，不回答「飞机该怎么办」。
 * 后者依机型、任务与运行环境而异：同一套传感器故障，在开阔场地可以就地降落，
 * 在水面上方就必须返航。把策略写死在检测里，会让检测模块无法复用。
 * 因此本文件只做「状态 → 动作」的映射，且所有阈值可配置。
 *
 * @par 分级的核心依据：故障发生在哪个控制环
 *
 * 四旋翼是级联控制：**姿态环在内、位置环在外**。这个层次决定了各故障的紧迫程度
 * 完全不同，不能一刀切「IMU 坏了就降落」：
 *
 * - **陀螺失效 → 立刻紧急降落**。姿态环直接依赖角速度反馈。四旋翼是欠驱动系统，
 *   姿态一旦发散没有第二次机会——不存在「先观察一会儿」的余地。这是本模块中
 *   唯一没有缓冲时间的决策，判据也最激进（Failed 即触发，不等确认步数之外
 *   的额外延迟）。
 * - **加速度计失效 → 降级飞行**。姿态仍可由陀螺积分维持（短期可用，长期漂移），
 *   失去的是：姿态的重力方向校正、以及位置/速度的预积分。若还有位置测量
 *   （GPS / 光流 / 视觉），位置环仍可工作，只是精度下降。这是**分钟级**的退化，
 *   不是立刻坠机，因此给的是「谨慎返航」而非「立刻降落」。
 * - **偏置类 Degraded → 限幅谨慎飞行**。数据仍可用但系统性偏斜，大机动会让
 *   偏差放大，故压低倾角与速度上限，避免把误差激励出来。
 *
 * @par 一个容易犯的错误：把 Unknown 当作 Healthy
 *
 * 估计器刚启动时样本不足，健康状态是 Unknown。此时若按「健康」放行，
 * 等于在没有任何证据的情况下假设传感器正常。本模块对 Unknown 的处理是
 * 「降级到 Cautious 但不触发降落」——既不盲目信任，也不过度反应。
 */

#ifndef OI3_IMU_DEGRADE_POLICY_H
#define OI3_IMU_DEGRADE_POLICY_H

#include "SensorHealth.h"

#include <string>

namespace oi3 {

/// 降级动作
enum class DegradeAction {
    Normal = 0,     ///< 正常飞行
    Cautious,       ///< 谨慎：限幅飞行，继续任务但降低机动性
    ReturnHome,     ///< 返航：飞回起飞点，不再继续执行任务
    EmergencyLand,  ///< 紧急降落：就地下降，最高优先级
};

/// 降级决策的结果
struct DegradeDecision {
    DegradeAction action = DegradeAction::Normal;

    /// 降级后的倾角上限（度）。正常时取飞控默认 35 度。
    double max_tilt_deg = 35.0;

    /// 降级后的水平速度上限（m/s）
    double max_speed = 5.0;

    /// 是否仍使用加速度计做姿态方向校正。
    /// 加速度计失效时置 false：此时它只会把错误方向「烧」进姿态估计。
    bool use_accel_correction = true;

    /// 是否仍信任位置/速度估计（加速度计失效后预积分不可靠）
    bool trust_position = true;

    /// 人类可读的原因，用于日志与地面站
    std::string reason;
};

/// 降级策略配置
struct DegradePolicyConfig {
    /// 正常飞行倾角上限（度）。与飞控 max_tilt_deg 一致。
    double nominal_tilt_deg = 35.0;

    /// 正常飞行水平速度上限（m/s）
    double nominal_speed = 5.0;

    /// 谨慎模式下的倾角上限（度）。压低以限制机动，避免放大偏置误差。
    double cautious_tilt_deg = 15.0;

    /// 谨慎模式下的速度上限（m/s）
    double cautious_speed = 2.0;

    /// 返航模式下的倾角上限（度）
    double return_tilt_deg = 20.0;

    /// 返航模式下的速度上限（m/s）
    double return_speed = 3.0;

    /// 紧急降落时的下降率（m/s），供上层执行
    double emergency_descent_rate = 1.5;
};

/**
 * @class ImuDegradePolicy
 * @brief 由 IMU 健康报告推出降级动作
 */
class ImuDegradePolicy {
  public:
    explicit ImuDegradePolicy(const DegradePolicyConfig &cfg = {}) : _cfg(cfg) {}

    /**
     * @brief 做出降级决策
     *
     * @param h 当前传感器健康报告
     * @return 决策结果（含动作、限幅与原因）
     */
    [[nodiscard]] DegradeDecision decide(const SensorHealthReport &h) const {
        DegradeDecision d;

        // ---- 0. 样本不足：不盲目信任，但也不过度反应 ----
        // Unknown 意味着「还没有证据」，既不是健康也不是故障。
        // 按「谨慎」处理：限幅飞行，等证据积累后再升级或解除。
        if (h.accel == SensorStatus::Unknown || h.gyro == SensorStatus::Unknown) {
            d.action = DegradeAction::Cautious;
            d.max_tilt_deg = _cfg.cautious_tilt_deg;
            d.max_speed = _cfg.cautious_speed;
            d.reason = "传感器健康未知（样本不足），限幅飞行待观测";
            return d;
        }

        // ---- 1. 陀螺失效：最高优先级，无缓冲 ----
        // 姿态环直接依赖角速度，四旋翼欠驱动、姿态发散不可逆。
        // 这是唯一「必须立即执行」的分支。
        if (h.gyro == SensorStatus::Failed) {
            d.action = DegradeAction::EmergencyLand;
            d.max_tilt_deg = 0.0; // 不再做姿态机动
            d.max_speed = 0.0;
            d.use_accel_correction = false;
            // 紧急降落不再做位置控制（max_tilt=0、不再机动），故位置预积分
            // 的取舍不影响结果；保持 false 以表明「不再依赖任何 IMU 派生的
            // 位置信息」。与加速度计失效场景的处置不同，理由见该分支注释。
            d.trust_position = false;
            d.reason = "陀螺失效：姿态环失去反馈，立即就地降落";
            return d;
        }

        // ---- 2. 加速度计失效：降级返航 ----
        // 姿态仍可由陀螺积分维持，但失去重力方向校正与位置预积分。
        // 这是分钟级退化而非立刻坠机，故给返航而非紧急降落。
        if (h.accel == SensorStatus::Failed) {
            d.action = DegradeAction::ReturnHome;
            d.max_tilt_deg = _cfg.return_tilt_deg;
            d.max_speed = _cfg.return_speed;
            // 失效后继续用加速度计做方向校正是有害的 ——
            // 它会把错误方向当成姿态误差持续注入姿态估计。
            // （注：对「输出恒零」型失效此项无实际影响，因为归一化后模长为零、
            //   校正分支本就不执行；但对「偏置漂移」型失效它是必要的。）
            d.use_accel_correction = false;
            // **位置预积分必须保留**。这一点由闭环实测纠正：
            // 初版把它设为 false（理由是「依赖加速度计故不可信」），结果飞机从
            // 5.35 m 飘到 16.14 m、倾角 35°、水平偏 2.6 m —— 降级反而把飞机
            // 搞坏了。根因是：预积分虽含偏差，却为位置环提供必要的高频信息，
            // 而外部位置测量负责低频纠偏，这正是互补滤波的分工。关掉预积分
            // 等于切断了高频通路，位置环失去阻尼而漂移。
            d.trust_position = true;
            d.reason = "加速度计失效：姿态改由陀螺积分维持并停止方向校正，位置预积分保留（靠外部量测纠偏），返航";
            return d;
        }

        // ---- 3. 降级（偏置类）：限幅谨慎飞行 ----
        // 数据仍可用但系统性偏斜。大机动会把偏置误差放大，故压低限幅。
        if (h.accel == SensorStatus::Degraded || h.gyro == SensorStatus::Degraded) {
            d.action = DegradeAction::Cautious;
            d.max_tilt_deg = _cfg.cautious_tilt_deg;
            d.max_speed = _cfg.cautious_speed;
            d.reason = "传感器存在系统性偏差，限幅飞行以抑制误差放大";
            return d;
        }

        // ---- 4. 正常 ----
        d.action = DegradeAction::Normal;
        d.max_tilt_deg = _cfg.nominal_tilt_deg;
        d.max_speed = _cfg.nominal_speed;
        d.reason = "传感器健康";
        return d;
    }

    /// 紧急降落时的下降率（m/s）
    [[nodiscard]] double emergencyDescentRate() const { return _cfg.emergency_descent_rate; }

  private:
    DegradePolicyConfig _cfg;
};

} // namespace oi3

#endif // OI3_IMU_DEGRADE_POLICY_H
