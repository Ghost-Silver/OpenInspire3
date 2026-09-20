/**
 * @file SixDofDynamics.h
 * @brief 六自由度飞行器动力学与积分
 * @author GhostFace
 * @date 2026/9/17
 *
 * 相对三自由度模型的本质差别：推力不再可以任意指定方向，而只能沿机体轴产生。
 * 水平运动因此必须通过倾斜机身获得分量 —— 这正是真实四旋翼的操纵方式，也是
 * 「飞控」二字的前提。
 *
 * @par 关于可微性（重要）
 *
 * 三自由度版本的 `droneAcceleration` 保持完整可微：推力与速度都经由 Tensor 算子
 * 参与运算。本文件**不再保证姿态环节可微**，原因有二：
 *
 *  1. 四元数旋转需要构造旋转矩阵与叉积，而 CTorch 缺少可微的索引/切片能力
 *     （`sqrt` 也不存在），用 Tensor 算子实现需要绕行且代价高；
 *  2. 策略梯度的计算不需要穿过环境（PPO 只用奖励标量），可微性在当前阶段
 *     没有实际用途。
 *
 * 因此姿态相关的量用标量计算后再构造常量张量。这是一个明确的取舍，不是遗漏：
 * 若将来需要端到端可微仿真，应先在 CTorch 侧补齐索引/切片算子，再回头改造此处。
 */

#ifndef OI3_SIX_DOF_DYNAMICS_H
#define OI3_SIX_DOF_DYNAMICS_H

#include "SixDofTypes.h"
#include "Tensor.h"

namespace oi3 {

/// 把四元数归一化到单位模长（抑制积分过程中的数值漂移）
[[nodiscard]] Tensor normalizeQuat(const Tensor &quat);

/// 机体坐标系向量旋转到 NED 系：v_ned = R(q) · v_body
[[nodiscard]] Tensor rotateBodyToNed(const Tensor &quat, const Tensor &v_body);

/// NED 系向量旋转到机体坐标系：v_body = R(q)ᵀ · v_ned
[[nodiscard]] Tensor rotateNedToBody(const Tensor &quat, const Tensor &v_ned);

/// 由四元数取欧拉角（Z-Y-X 顺序），仅用于日志与调试，不参与控制
[[nodiscard]] Tensor quatToEuler(const Tensor &quat);

/**
 * @brief 平动加速度
 *
 * @param vel           NED 速度 {3}（用于计算相对气流）
 * @param quat          姿态四元数 {4}
 * @param thrust_body   机体 z 轴推力（牛顿，向上为正）
 * @param cfg           配置
 * @param v_wind        NED 风速 {3}；nullptr 表示无风（模型退化回原有阻力项）
 * @return NED 加速度 {3}，米/秒²
 *
 * 合力 = 重力 + R(q)·[0,0,-T] + 气动阻力。
 * 推力沿机体轴 —— 姿态水平时它只抵消重力，机身倾斜时才产生水平分量。
 *
 * 气动阻力取决于**相对速度** v_rel = v − v_wind 而非地速：顺风时相对气流小、
 * 阻力小，逆风时阻力大。这是风对飞行器作用的核心机制，也是抗风能力分析的起点。
 */
[[nodiscard]] Tensor sixDofAcceleration(const Tensor &vel, const Tensor &quat,
                                        double thrust_body, const SixDofConfig &cfg,
                                        const Tensor *v_wind = nullptr);

/**
 * @brief 转动加速度（欧拉方程）
 *
 * ω̇ = I⁻¹ (τ − ω × (Iω))，惯量取对角阵。
 *
 * @param omega  机体角速度 {3}（弧度/秒）
 * @param torque 机体三轴力矩 {3}（N·m）
 * @param cfg    六自由度配置
 * @return 角加速度 {3}，弧度/秒²
 */
[[nodiscard]] Tensor angularAcceleration(const Tensor &omega, const Tensor &torque,
                                         const SixDofConfig &cfg);

/// 四元数导数：q̇ = ½ · q ⊗ [0, ω]
[[nodiscard]] Tensor quatDerivative(const Tensor &quat, const Tensor &omega);

/**
 * @brief 六自由度单步 RK4 积分
 *
 * @param y           当前状态
 * @param thrust_body 步内机体推力（牛顿）
 * @param torque      步内机体力矩 {3}（N·m）
 * @param cfg         配置
 * @param dt          步长（秒）
 * @param v_wind      NED 风速 {3}；nullptr 表示无风
 *
 * @note 风速在整个 RK4 步内取常数（步首值）。步长 1 ms 而湍流的时间尺度在秒级，
 *       步内变化可以忽略；若将来把步长放大到与湍流尺度可比，需要改成按级取值。
 */
[[nodiscard]] SixDofState rk4StepSixDof(const SixDofState &y, double thrust_body,
                                        const Tensor &torque, const SixDofConfig &cfg,
                                        double dt, const Tensor *v_wind = nullptr);

} // namespace oi3

#endif // OI3_SIX_DOF_DYNAMICS_H
