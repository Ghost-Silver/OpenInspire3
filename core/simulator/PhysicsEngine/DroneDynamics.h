/**
 * @file DroneDynamics.h
 * @brief OpenInspire3 飞行器动力学与积分步进
 * @author GhostFace
 * @date 2026/9/16
 */

#ifndef OI3_DRONE_DYNAMICS_H
#define OI3_DRONE_DYNAMICS_H

#include "DroneTypes.h"
#include "Tensor.h"

namespace oi3 {

/**
 * @brief 加速度 a = (F_gravity + thrust) / mass
 *
 * @param vel   机体系速度 {3}（NED）。当前质点模型不含与速度相关的力
 *              （气动阻力、风扰等），该参数为后续扩展保留。
 * @param thrust 合推力 {3}（NED 系，牛顿）
 * @param cfg   仿真配置
 * @return 加速度 {3}，米/秒²
 *
 * **符号约定**：NED 系向下为正，重力矢量恒为 `[0, 0, +m*g]`。
 * 悬停时 `thrust[2] = -m*g`。
 *
 * **可微性**：只使用 Tensor 四则运算符，不触碰底层数据指针，
 * 因此 autograd 计算图完整，梯度可回传到 `thrust` 与 `vel`。
 */
[[nodiscard]] Tensor droneAcceleration(const Tensor &vel, const Tensor &thrust,
                                       const Config &cfg);

/**
 * @brief 对飞行器状态执行单步 RK4 积分
 *
 * @param y      当前状态
 * @param thrust 步内合推力（NED 系，牛顿）
 * @param cfg    仿真配置
 * @param dt     步长（秒）
 * @return 步进后的状态
 *
 * @par 为什么不用通用的 RK4Integrator 模板
 *
 * 通用积分器要求右端函数返回完整的状态导数 `(d(pos)/dt, d(vel)/dt)`，而
 * `d(pos)/dt ≡ vel` 是**运动学结构性事实**，与物理模型无关。若让动力学函数
 * 把速度原样搬进返回值，必然发生一次 Tensor 拷贝；而 CTorch 的拷贝构造会把
 * 副本的 autograd 节点替换成新建的 GradAccumulator（见 `Tensor(const Tensor&)`
 * 中的 `createGradAccumulator`），副本与上游就此断开——它看起来仍是
 * `requires_grad=true` 且带 node，但反向传播到它为止，梯度静默丢失。
 *
 * 因此本函数直接使用 `y.vel` 参与积分，所有中间张量都由算术运算产生
 * （右值 → 移动构造，保留节点），全程不发生 Tensor 拷贝。
 *
 * @par 与标准 RK4 的等价性
 *
 * 记 a_i 为第 i 阶段的加速度，v_i 为第 i 阶段的速度：
 * @verbatim
 *   v1 = y.vel            a1 = accel(v1)
 *   v2 = y.vel + a1*h/2   a2 = accel(v2)
 *   v3 = y.vel + a2*h/2   a3 = accel(v3)
 *   v4 = y.vel + a3*h     a4 = accel(v4)
 *   pos' = y.pos + h/6 * (v1 + 2v2 + 2v3 + v4)
 *   vel' = y.vel + h/6 * (a1 + 2a2 + 2a3 + a4)
 * @endverbatim
 */
[[nodiscard]] DroneState rk4StepDrone(const DroneState &y, const Tensor &thrust,
                                      const Config &cfg, double dt);

} // namespace oi3

#endif // OI3_DRONE_DYNAMICS_H
