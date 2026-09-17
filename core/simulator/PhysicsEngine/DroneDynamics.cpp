/**
 * @file DroneDynamics.cpp
 * @brief OpenInspire3 飞行器动力学与积分步进实现
 * @author GhostFace
 * @date 2026/9/16
 */

#include "DroneDynamics.h"

namespace oi3 {

Tensor dragForce(const Tensor &vel, const Config &cfg) {
    // F_i = -k * |v_i| * v_i，逐轴二次形式，不含 sqrt
    const Tensor abs_vel = vel.abs();
    return (abs_vel * vel) * static_cast<float>(-cfg.drag_coeff);
}

Tensor droneAcceleration(const Tensor &vel, const Tensor &thrust, const Config &cfg) {
    // 重力：NED 系向下为正，故 down 分量为 +m*g
    const Tensor f_gravity =
        makeVec3(0.0f, 0.0f, static_cast<float>(cfg.mass * cfg.gravity));

    // 合外力：先算重力与推力，再按需叠加阻力。
    //
    // 这里刻意不在无阻力时也做一次 `+ dragForce(...)`：虽然加上零向量在数值上
    // 等价，但会多引入一个算子节点；更重要的是，任何「把张量先拷一份再拼装」的
    // 写法都会因 Tensor 拷贝构造替换 autograd 节点而静默切断计算图
    // （Test 4 / Test 5 的梯度回归正是为此设置）。
    Tensor f_total = f_gravity + thrust;
    if (cfg.drag_coeff > 0.0) {
        f_total = f_total + dragForce(vel, cfg);
    }

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
