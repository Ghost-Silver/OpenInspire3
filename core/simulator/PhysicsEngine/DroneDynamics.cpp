/**
 * @file DroneDynamics.cpp
 * @brief OpenInspire3 飞行器动力学与积分步进实现
 * @author GhostFace
 * @date 2026/9/16
 */

#include "DroneDynamics.h"

namespace oi3 {

Tensor droneAcceleration(const Tensor &vel, const Tensor &thrust, const Config &cfg) {
    // 当前质点模型不含速度相关力（气动阻力、风扰等），保留参数以便扩展。
    (void)vel;

    // 重力：NED 系向下为正，故 down 分量为 +m*g
    const Tensor f_gravity =
        makeVec3(0.0f, 0.0f, static_cast<float>(cfg.mass * cfg.gravity));

    // 合外力（后续可在此叠加气动阻力、风扰等）
    const Tensor f_total = f_gravity + thrust;

    // 牛顿第二定律：a = F / m
    return f_total / static_cast<float>(cfg.mass);
}

DroneState rk4StepDrone(const DroneState &y, const Tensor &thrust, const Config &cfg,
                        double dt) {
    const float h = static_cast<float>(dt);
    const float half = h * 0.5f;
    const float sixth = h / 6.0f;

    // 阶段 1：在 (pos, vel) 处求加速度
    const Tensor a1 = droneAcceleration(y.vel, thrust, cfg);

    // 阶段 2：在 (pos + v1*h/2, vel + a1*h/2) 处
    const Tensor v2 = y.vel + a1 * half;
    const Tensor a2 = droneAcceleration(v2, thrust, cfg);

    // 阶段 3：在 (pos + v2*h/2, vel + a2*h/2) 处
    const Tensor v3 = y.vel + a2 * half;
    const Tensor a3 = droneAcceleration(v3, thrust, cfg);

    // 阶段 4：在 (pos + v3*h, vel + a3*h) 处
    const Tensor v4 = y.vel + a3 * h;
    const Tensor a4 = droneAcceleration(v4, thrust, cfg);

    // 注：当前质点模型的加速度只依赖推力，四个数值相同。此处仍完整走四个
    // 阶段，是为了在引入速度/位置相关力（阻力等）时无需改动结构；
    // 状态维度为 3，四次求值开销可忽略。

    // 组合：v1 = y.vel，因此 pos 的线性组合直接用 y.vel。
    //
    // 返回处直接写表达式而不先存入局部变量：右值触发 DroneState 的移动构造，
    // Tensor 的移动构造保留 autograd 节点（仅 rebind 弱引用）。若先存成
    // `const Tensor pos_new = ...` 再构造，会走拷贝构造，节点被替换为新建的
    // GradAccumulator，梯度链就在这一行断开。
    return DroneState{y.pos + (y.vel + v2 * 2.0f + v3 * 2.0f + v4) * sixth,
                      y.vel + (a1 + a2 * 2.0f + a3 * 2.0f + a4) * sixth};
}

} // namespace oi3
