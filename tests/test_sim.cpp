/**
 * @file test_sim.cpp
 * @brief OpenInspire3 仿真回归测试
 * @author GhostFace
 * @date 2026/9/16
 *
 * 覆盖四类断言：
 *   Test 1  自由落体 vs 解析解
 *   Test 2  悬停（推力抵消重力）
 *   Test 3  RK4 收敛阶（步长减半，误差应降约 2^4 = 16 倍）
 *   Test 4  可微性：loss 对 thrust 的梯度可回传且数值符合解析式
 *   Test 5  自动微分 vs 中心差分交叉验证
 *
 * Test 4/5 是本次重构的核心验收项：旧实现用裸指针拼装导数向量，
 * 计算图被切断，梯度恒为零，这两项必然失败。
 */

#include "AutoGrad.h"
#include "core/simulator/PhysicsEngine/DroneSimulator.h"
#include "core/simulator/PhysicsEngine/RK4Solver.h"
#include "core/simulator/PhysicsEngine/TensorUtils.h"

#include <cmath>
#include <iostream>
#include <string>

using namespace oi3;

namespace {

int g_checks = 0;
int g_failures = 0;

void expectNear(double got, double want, double tol, const std::string &what) {
    ++g_checks;
    const double err = std::fabs(got - want);
    if (err > tol) {
        ++g_failures;
        std::cout << "  [FAIL] " << what << ": got " << got << ", want " << want
                  << ", |err| = " << err << " > tol " << tol << "\n";
    } else {
        std::cout << "  [ ok ] " << what << "  (|err| = " << err << ")\n";
    }
}

void expectTrue(bool cond, const std::string &what) {
    ++g_checks;
    if (!cond) {
        ++g_failures;
        std::cout << "  [FAIL] " << what << "\n";
    } else {
        std::cout << "  [ ok ] " << what << "\n";
    }
}

Config defaultConfig() {
    Config cfg;
    cfg.dt = 0.001;
    cfg.mass = 1.0;
    cfg.gravity = 9.81;
    return cfg;
}

/// 跑 N 步定推力仿真，返回 sum(vel) 作标量 loss；
/// grad_out 非空时额外回传 d(loss)/d(thrust_down)
double simulateLoss(const Config &cfg, int steps, float thrust_down,
                    double *grad_out) {
    Tensor thrust = makeVec3(0.0f, 0.0f, thrust_down);
    const bool want_grad = (grad_out != nullptr);
    if (want_grad) {
        thrust.requires_grad(true);
    }

    DroneSimulator sim(cfg, DroneState{makeVec3(0.0f, 0.0f, 0.0f),
                                       makeVec3(0.0f, 0.0f, 0.0f)});
    for (int i = 0; i < steps; ++i) {
        sim.step(thrust);
    }

    Tensor loss = sim.state().vel.sum();
    if (want_grad) {
        AutoGrad::backward(loss.getRelatedNode(), false);
        *grad_out = toVector(thrust.grad())[2];
    }
    return static_cast<double>(toVector(loss)[0]);
}

// ============================ Test 1 ============================
void testFreeFall() {
    std::cout << "\n[Test 1] Free fall vs analytic solution\n";
    const Config cfg = defaultConfig();
    const double T = 1.0;
    const int steps = static_cast<int>(T / cfg.dt);

    DroneSimulator sim(cfg, DroneState{makeVec3(0, 0, 0), makeVec3(0, 0, 0)});
    const Tensor zero = makeVec3(0.0f, 0.0f, 0.0f);
    sim.step(zero, steps);

    const std::vector<float> pos = toVector(sim.state().pos);
    const std::vector<float> vel = toVector(sim.state().vel);

    expectNear(pos[2], 0.5 * cfg.gravity * T * T, 1e-3, "pd == 0.5*g*t^2");
    expectNear(vel[2], cfg.gravity * T, 1e-3, "vd == g*t");
    expectNear(pos[0], 0.0, 1e-6, "pn stays 0");
    expectNear(pos[1], 0.0, 1e-6, "pe stays 0");
    expectNear(vel[0], 0.0, 1e-6, "vn stays 0");
}

// ============================ Test 2 ============================
void testHover() {
    std::cout << "\n[Test 2] Hover holds position\n";
    const Config cfg = defaultConfig();
    const double T = 1.0;
    const int steps = static_cast<int>(T / cfg.dt);

    DroneSimulator sim(cfg, DroneState{makeVec3(0, 0, 0), makeVec3(0, 0, 0)});
    const Tensor thrust =
        makeVec3(0.0f, 0.0f, static_cast<float>(-cfg.mass * cfg.gravity));
    sim.step(thrust, steps);

    const std::vector<float> pos = toVector(sim.state().pos);
    const std::vector<float> vel = toVector(sim.state().vel);

    expectNear(pos[2], 0.0, 1e-3, "pd stays 0 under hover thrust");
    expectNear(vel[2], 0.0, 1e-3, "vd stays 0 under hover thrust");
}

// ============================ Test 3 ============================
// 用标量 ODE dy/dt = y 测 RK4 收敛阶。
// 不用自由落体测阶：其解是二次多项式，RK4 对多项式精确，残差会被 float
// 舍入地板（~1e-6）掩盖，测不出阶数。
void testRK4ConvergenceOrder() {
    std::cout << "\n[Test 3] RK4 convergence order on dy/dt = y\n";
    const double exact = std::exp(1.0);

    auto errorAt = [&](double dt) {
        RK4Integrator integ(dt);
        const auto f = [](double /*t*/, float y) -> float { return y; };
        const float y_end = integ.integrate(f, 0.0, 1.0, 1.0f);
        return std::fabs(static_cast<double>(y_end) - exact);
    };

    const double e1 = errorAt(0.2);
    const double e2 = errorAt(0.1);
    const double ratio = e1 / e2;

    std::cout << "    e(dt=0.2) = " << e1 << ", e(dt=0.1) = " << e2
              << ", ratio = " << ratio << " (ideal 16)\n";
    expectTrue(ratio > 10.0 && ratio < 25.0,
               "error ratio in [10, 25], consistent with O(dt^4)");
}

// ============================ Test 4 ============================
void testDifferentiability() {
    std::cout << "\n[Test 4] Gradient flows back to thrust\n";
    const Config cfg = defaultConfig();
    const int N = 10;

    Tensor thrust = makeVec3(0.0f, 0.0f, 1.0f);
    thrust.requires_grad(true);

    DroneSimulator sim(cfg, DroneState{makeVec3(0, 0, 0), makeVec3(0, 0, 0)});
    for (int i = 0; i < N; ++i) {
        sim.step(thrust);
    }

    Tensor loss = sim.state().vel.sum();
    AutoGrad::backward(loss.getRelatedNode(), false);

    const std::vector<float> g = toVector(thrust.grad());

    // vel_N = N * dt * thrust / m  =>  d(sum vel)/d(thrust_i) = N*dt/m
    const double expected =
        static_cast<double>(N) * cfg.dt / cfg.mass;

    std::cout << "    grad = [" << g[0] << ", " << g[1] << ", " << g[2]
              << "], analytic = " << expected << "\n";
    expectNear(g[2], expected, 1e-5, "d(sum vel)/d(thrust_down) == N*dt/m");
    expectNear(g[0], expected, 1e-5, "d(sum vel)/d(thrust_north) == N*dt/m");
    expectTrue(std::fabs(g[2]) > 1e-8,
               "gradient is non-zero (autograd graph is intact)");
}

// ============================ Test 5 ============================
void testGradientVsFiniteDifference() {
    std::cout << "\n[Test 5] Autograd vs central finite difference\n";
    const Config cfg = defaultConfig();
    const int N = 10;

    double auto_grad = 0.0;
    simulateLoss(cfg, N, 0.0f, &auto_grad);

    const double eps = 1e-2;
    const double loss_plus = simulateLoss(cfg, N, static_cast<float>(eps), nullptr);
    const double loss_minus = simulateLoss(cfg, N, static_cast<float>(-eps), nullptr);
    const double fd_grad = (loss_plus - loss_minus) / (2.0 * eps);

    std::cout << "    autograd = " << auto_grad << ", central-diff = " << fd_grad
              << "\n";
    expectNear(auto_grad, fd_grad, 1e-4,
               "autograd gradient matches central difference");
}

} // namespace

int main() {
    CtorchError::setPrintLevel(PrintLevel::MINIUM);

    std::cout << "========================================\n";
    std::cout << "OpenInspire3 simulation regression tests\n";
    std::cout << "========================================\n";

    testFreeFall();
    testHover();
    testRK4ConvergenceOrder();
    testDifferentiability();
    testGradientVsFiniteDifference();

    std::cout << "\n========================================\n";
    std::cout << (g_checks - g_failures) << " / " << g_checks
              << " checks passed\n";
    if (g_failures != 0) {
        std::cout << g_failures << " FAILED\n";
    }
    std::cout << "========================================\n";

    return g_failures == 0 ? 0 : 1;
}
