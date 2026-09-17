/**
 * @file SixDofDynamicsDiff.h
 * @brief 六自由度动力学的可微张量实现
 * @author GhostFace
 * @date 2026/9/17
 *
 * 与 SixDofDynamics 的关系：那个版本用标量算姿态、再把结果包成常量张量，因此
 * 姿态环节不可微（其头文件记录了当时的取舍：等 CTorch 补齐索引/切片算子后再回头
 * 改造）。本文档就是那次改造 —— 现在 CTorch 有了 slice / reshape / concat / sqrt
 * 四个可微算子，四元数旋转可以全部用张量运算表达，整条动力学生成完整计算图。
 *
 * @par 可微性有什么实际用途
 *
 * PPO 这类策略梯度只用奖励标量，穿过环境求导没有意义；但下面两件事必须要有梯度：
 *
 *  1. **打靶法（shooting）控制**：把一段控制序列当作参数，直接对轨迹末端的代价
 *     求导，一步回传到控制量。相比采样估计梯度，同样的物理仿真次数能得到高得多
 *     的信噪比 —— 对算力有限的开源飞控项目尤其关键。
 *  2. **参数辨识**：转动惯量、力臂、阻力系数这些量常是估的。有梯度就能用少量
 *     实测轨迹做最小二乘拟合，而不是网格搜索。
 *
 * @par 与标量版的一致性
 *
 * 两者的方程逐项对应，同一初值同一步长下单步结果应在 float 精度内一致；测试
 * （SixDofGradTest）对两个实现做逐步数值对照。差异只允许出现在四元数归一化的
 * 零模长保护上：标量版在 n <= 1e-12 时退化为不归一化，张量版加 eps 后仍是除法
 * （差值与 float 精度同量级）。
 *
 * @par 调用方的强制约束：状态必须从持久参数派生
 *
 * CTorch 的 `GradAccumulator` 用 **weak_ptr** 绑定它所属的张量对象，梯度只能写回
 * 那个对象仍然活着时。而状态演化写成 `s = rk4Step(s, ...)` 时，`s.pos` 等成员会被
 * 移动赋值**覆盖**，原先的对象随即析构 —— 于是「状态张量本身当叶子」这种直觉写法
 * 会静默丢梯度（`grad_ptr()` 返回 nullptr，不报错）。
 *
 * 正确的模式有两类：
 *
 *  1. **持久的待学参数 + 一次派生**。参数张量（推力序列、控制序列、物理参数）在
 *     循环外持有、全程不被覆盖；状态初值由它经一次逐元素运算派生：
 *     @code
 *     Tensor thrust0 = ...;                       // 叶子，持久
 *     SixDofState s{pos0 * 1.0f, ..., quat0 * 1.0f, ...};  // 派生，锚在参数上
 *     for (...) s = rk4StepSixDofDiff(s, thrust0, ...);     // 覆盖的是 s，不是参数
 *     @endcode
 *     梯度沿 `s` 的演化链回流到参数，因为图的上游节点属于参数对象。
 *
 *  2. **短窗口打靶**。每个优化步只对一段短轨迹求导，窗口内状态从参数派生，
 *     窗口结束即反传。长时域靠滚动窗口拼接，避免单张图过大 —— 图规模随步数
 *     线性增长，一次几十步的图节点数已在万级。
 *
 * 参考实现见 `SixDofGradTest` 的 Part B/C。
 */

#ifndef OI3_SIX_DOF_DYNAMICS_DIFF_H
#define OI3_SIX_DOF_DYNAMICS_DIFF_H

#include "SixDofDynamics.h"
#include "SixDofTypes.h"
#include "Tensor.h"

#include <array>

namespace oi3 {

/**
 * @brief 由三个标量张量拼装 {3} 向量
 *
 * 输入需为 {1} 形状（典型来源是 slice(0, i, 1) 的结果）。拼接经 CTorch 的
 * concat 算子完成，因此梯度可以分别回到三个分量。
 */
[[nodiscard]] Tensor stackVec3(const Tensor &x, const Tensor &y, const Tensor &z);

/// 由四个标量张量拼装 {4} 向量（四元数分量）
[[nodiscard]] Tensor stackVec4(const Tensor &w, const Tensor &x, const Tensor &y,
                               const Tensor &z);

/**
 * @brief 旋转矩阵的九个元素（行主序），每个为 {1} 张量
 *
 * 供「展开式」矩阵-向量乘法使用：四旋翼需要的是 R·v，把九个元素与三个分量逐项
 * 组合比先拼成 {3,3} 再 matmul 更直接，也便于与标量版逐项对照。
 */
[[nodiscard]] std::array<Tensor, 9> rotationMatrixEntries(const Tensor &quat);

/// 由单位四元数构造机体系 -> NED 的旋转矩阵 {3,3}
[[nodiscard]] Tensor rotationMatrixFromQuat(const Tensor &quat);

/// 四元数归一化（可微版本，带零模长保护）
[[nodiscard]] Tensor normalizeQuatDiff(const Tensor &quat);

/// 机体坐标系向量旋转到 NED 系：v_ned = R(q) · v_body
[[nodiscard]] Tensor rotateBodyToNedDiff(const Tensor &quat, const Tensor &v_body);

/**
 * @brief 平动加速度（可微）
 *
 * @param vel     NED 速度 {3}
 * @param quat    姿态四元数 {4}
 * @param thrust  机体 z 轴推力 {1}（牛顿）—— 用张量而非标量，推力才能成为可学参数
 * @param cfg     配置
 */
[[nodiscard]] Tensor sixDofAccelerationDiff(const Tensor &vel, const Tensor &quat,
                                            const Tensor &thrust, const Config &cfg);

/// 转动加速度（可微）：ω̇ = I⁻¹ (τ − ω × (Iω))
[[nodiscard]] Tensor angularAccelerationDiff(const Tensor &omega, const Tensor &torque,
                                             const SixDofConfig &cfg);

/// 四元数导数（可微）：q̇ = ½ · q ⊗ [0, ω]
[[nodiscard]] Tensor quatDerivativeDiff(const Tensor &quat, const Tensor &omega);

/**
 * @brief 六自由度单步 RK4 积分（可微）
 *
 * 与标量版 rk4StepSixDof 同方程、同步长、同零阶保持（步内推力与力矩恒定）。
 *
 * @param y       当前状态
 * @param thrust  步内机体推力 {1}（牛顿）
 * @param torque  步内机体力矩 {3}（N·m）
 * @param cfg     配置
 * @param dt      步长（秒）
 * @return 下一步状态；梯度可回传至 y / thrust / torque
 */
[[nodiscard]] SixDofState rk4StepSixDofDiff(const SixDofState &y, const Tensor &thrust,
                                            const Tensor &torque, const SixDofConfig &cfg,
                                            float dt);

} // namespace oi3

#endif // OI3_SIX_DOF_DYNAMICS_DIFF_H
