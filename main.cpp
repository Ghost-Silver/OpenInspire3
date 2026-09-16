/**
 * @file main.cpp
 * @brief OpenInspire3 飞行仿真演示：自由落体与悬停
 * @author GhostFace
 * @date 2026/9/16
 *
 * 演示两个基础场景，并对自由落体做解析解对照：
 *   A. 自由落体 —— 推力为零，验证 pd = ½gt², vd = gt
 *   B. 悬停     —— 推力抵消重力，验证位置与速度保持不动
 */

#include "core/simulator/PhysicsEngine/DroneSimulator.h"
#include "core/simulator/PhysicsEngine/TensorUtils.h"

#include <cmath>
#include <iostream>

using namespace oi3;

namespace {

/// 场景 A：自由落体，并与解析解对照
void runFreeFall(const Config &cfg, int steps, double duration) {
    std::cout << "\n[A] Free fall (thrust = 0)\n";
    std::cout << "----------------------------------------\n";

    DroneSimulator sim(cfg, DroneState{makeVec3(0.0f, 0.0f, 0.0f),
                                       makeVec3(0.0f, 0.0f, 0.0f)});
    const Tensor thrust = makeVec3(0.0f, 0.0f, 0.0f);

    for (int i = 0; i < steps; ++i) {
        sim.step(thrust);
        if (i % 100 == 0) {
            sim.print();
        }
    }
    sim.print();

    const std::vector<float> pos = toVector(sim.state().pos);
    const std::vector<float> vel = toVector(sim.state().vel);

    const double analytic_pd = 0.5 * cfg.gravity * duration * duration;
    const double analytic_vd = cfg.gravity * duration;

    std::cout << "\n  analytic:  pd = " << analytic_pd << " m, vd = "
              << analytic_vd << " m/s\n";
    std::cout << "  simulated: pd = " << pos[2] << " m, vd = " << vel[2]
              << " m/s\n";
    std::cout << "  abs error: " << std::fabs(pos[2] - analytic_pd) << " m, "
              << std::fabs(vel[2] - analytic_vd) << " m/s\n";
}

/// 场景 B：悬停，推力取 -m*g（NED 系向上），状态应保持不动
void runHover(const Config &cfg, int steps, double duration) {
    std::cout << "\n[B] Hover (thrust = -m*g on down axis)\n";
    std::cout << "----------------------------------------\n";

    DroneSimulator sim(cfg, DroneState{makeVec3(0.0f, 0.0f, 0.0f),
                                       makeVec3(0.0f, 0.0f, 0.0f)});
    const Tensor thrust =
        makeVec3(0.0f, 0.0f, static_cast<float>(-cfg.mass * cfg.gravity));

    sim.step(thrust, steps);
    sim.print();

    const std::vector<float> pos = toVector(sim.state().pos);
    const std::vector<float> vel = toVector(sim.state().vel);

    std::cout << "\n  drift after " << duration << " s: |pos| = "
              << std::fabs(pos[2]) << " m, |vel| = " << std::fabs(vel[2])
              << " m/s\n";
}

} // namespace

int main() {
    CtorchError::setPrintLevel(PrintLevel::MINIUM);

    Config cfg;
    cfg.dt = 0.001;     // 1kHz 步长
    cfg.mass = 1.0;     // 1 kg
    cfg.gravity = 9.81; // m/s^2

    const double duration = 1.0;
    const int steps = static_cast<int>(duration / cfg.dt);

    std::cout << "OpenInspire3 — Drone free fall / hover simulation\n";
    std::cout << "========================================\n";
    std::cout << "mass      = " << cfg.mass << " kg\n";
    std::cout << "gravity   = " << cfg.gravity << " m/s^2\n";
    std::cout << "dt        = " << cfg.dt << " s\n";
    std::cout << "duration  = " << duration << " s (" << steps << " steps)\n";
    std::cout << "========================================\n";

    runFreeFall(cfg, steps, duration);
    runHover(cfg, steps, duration);

    std::cout << "\nDone.\n";
    return 0;
}
