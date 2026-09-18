/**
 * @file SpinningController.h
 * @brief 旋转容错模式控制器：偏航自由旋转，俯仰/滚转与高度受控
 * @author GhostFace
 * @date 2026/9/18
 *
 * @par 它针对的是什么工况
 *
 * 四旋翼失去一个电机后，混控矩阵从 4×4 降为 4×3，秩为 3。精确的约束是
 *
 * @verbatim
 *   τz = c·T − (c/a)·τx + (c/a)·τy
 * @endverbatim
 *
 * 即 **T、τx、τy 三个量仍可独立控制，失去自由的只有 τz**。（这一点容易说错：
 * 并非「τz 被 τy 锁定」，偏航力矩同时受总推力与两个水平力矩影响。）
 *
 * 由此得到两条结论，也正是本控制器的设计依据：
 *
 *  1. **俯仰与滚转仍然完全可控** —— 可以主动把机身倾住；
 *  2. **偏航不可控** —— 飞行器会绕偏航轴自由旋转。
 *
 * 所以「旋转」不是失控翻滚，而是主动选择：放弃偏航这个自由度，换取让推力
 * 方向在惯性系中随自旋扫圈。倾角在**机体系**中固定，于是推力方向在 NED 系中
 * 以自旋角速度旋转 —— 这正是后续用周期调制产生净水平力的前提。
 *
 * @par 本控制器只做第一步
 *
 * 它维持**高度**与**倾角**，不控制偏航、也不做周期调制。目的是先回答
 * 「三电机能否稳住在旋转状态」这个问题 —— 如果连倾角都稳不住，调制就无从谈起。
 *
 * @note 倾角的目标方向定义在机体系中（`[sinθ, 0, cosθ]`），因此随偏航一起旋转。
 *       若定义在 NED 系中，飞行器需持续跟踪一个转动的目标，是另一个问题。
 */

#ifndef OI3_SPINNING_CONTROLLER_H
#define OI3_SPINNING_CONTROLLER_H

#include "SixDofTypes.h"
#include "Tensor.h"

namespace oi3 {

/// 旋转模式配置
struct SpinConfig {
    /// 机体倾角（度）：机体 z 轴相对竖直方向的夹角，方向定义在机体系
    /// （绕机体 x 轴倾斜），因此随偏航一起旋转
    double tilt_deg = 25.0;

    /// 高度保持的目标高度（NED，米，向下为正）
    double target_z = -5.0;

    /// 高度环增益（输出期望竖直加速度）
    double z_kp = 3.0;
    double z_kd = 4.0;
    double max_accel = 8.0;

    /// 姿态环带宽与阻尼比（增益由惯量推导）
    double att_bandwidth = 9.0;
    double att_damping = 1.0;

    /// 是否控制偏航。容错模式下应为 false —— 偏航力矩本就不可独立指定，
    /// 强行下发会让混控把它转嫁给 τx/τy，反而破坏倾角控制。
    bool control_yaw = false;

    /**
     * @brief 是否补偿陀螺耦合项 ω×(Iω)
     *
     * 姿态动力学是 `I·ω̇ = τ − ω×(Iω) − kω`，而朴素的 PD 控制律只给
     * `τ = kp·e − kd·ω` —— 陀螺项未被补偿，它作为扰动直接作用在系统上。
     *
     * 低速时这一项可以忽略，所以定点悬停与手动模式的控制器都没有它。
     * 但旋转容错模式的核心恰恰是**让偏航持续自旋**，ω_z 不再是可忽略的小量：
     *
     * @verbatim
     *   τ_gyro,x = (I_z − I_y)·ω_y·ω_z
     *   τ_gyro,y = (I_x − I_z)·ω_x·ω_z
     * @endverbatim
     *
     * 注意这是**交叉耦合**：驱动 x 轴的是 ω_y 而不是 ω_x。于是两轴互相激励，
     * 而各自的阻尼项只作用于自身角速度，构不成交叉阻尼 —— 这正是实测中
     * 「从精确目标倾角出发也会漂到 93°、ωx 从 0 涨到 4.4」的候选原因。
     *
     * 补偿方式：要得到期望闭环 `I·ω̇ = kp·e − kd·ω`，控制力矩应为
     * `τ = kp·e − kd·ω + ω×(Iω)`。
     */
    bool compensate_gyroscopic = true;
};

/**
 * @class SpinningController
 * @brief 三电机容错：维持倾角与高度，任由偏航旋转
 */
class SpinningController {
  public:
    explicit SpinningController(SixDofConfig cfg, SpinConfig sc = {});

    /// 由状态算出推力与力矩
    [[nodiscard]] SixDofCommand compute(const SixDofState &state) const;

    [[nodiscard]] const char *name() const { return "Spin-Recovery"; }

    [[nodiscard]] const SpinConfig &config() const { return _sc; }

    /// 当前的目标倾角（度）
    [[nodiscard]] double targetTiltDeg() const { return _sc.tilt_deg; }

    [[nodiscard]] double attKp(int axis) const { return _att_kp[axis]; }
    [[nodiscard]] double attKd(int axis) const { return _att_kd[axis]; }

  private:
    SixDofConfig _cfg;
    SpinConfig _sc;
    double _att_kp[3] = {0.0, 0.0, 0.0};
    double _att_kd[3] = {0.0, 0.0, 0.0};
};

} // namespace oi3

#endif // OI3_SPINNING_CONTROLLER_H
