/**
 * @file SixDofHoverTest.cpp
 * @brief 六自由度动力学与闭环控制测试
 * @author GhostFace
 * @date 2026/9/17
 *
 * 分两部分：
 *   Part A 物理正确性 —— 倾斜机身必须产生水平分量，且升力不足会掉高度。
 *          这是六自由度相对三质点的本质差别，必须先验证动力学本身是对的。
 *   Part B 闭环控制   —— 级联 PID 能否把飞行器带到目标点并稳住。
 *
 * 退出码反映全部断言是否通过。
 */

#include "SixDofDynamics.h"
#include "SixDofPidController.h"
#include "SixDofSimulator.h"
#include "TensorUtils.h"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace oi3;

namespace {

int g_checks = 0;
int g_failed = 0;

void check(const char *name, double got, double want, double tol) {
    ++g_checks;
    const double err = std::fabs(got - want);
    const bool ok = err <= tol;
    if (!ok) {
        ++g_failed;
    }
    std::cout << "  [" << (ok ? " ok " : "FAIL") << "] " << name << "  (got " << got
              << ", want " << want << ", |err| = " << err << ")\n";
}

void checkTrue(const char *name, bool cond) {
    ++g_checks;
    if (!cond) {
        ++g_failed;
    }
    std::cout << "  [" << (cond ? " ok " : "FAIL") << "] " << name << "\n";
}

SixDofConfig makeConfig() {
    SixDofConfig cfg;
    cfg.base.dt = 0.001;
    cfg.base.mass = 1.0;
    cfg.base.gravity = 9.81;
    cfg.base.drag_coeff = 0.0; // 物理验证阶段先关阻力，便于与解析解对照
    cfg.base.max_thrust = 0.0;
    cfg.inertia[0] = 0.01;
    cfg.inertia[1] = 0.01;
    cfg.inertia[2] = 0.02;
    cfg.max_body_thrust = 20.0;
    cfg.torque_limit = 1.0;
    return cfg;
}

/// 绕 NED 的 y 轴旋转 angle（弧度）对应的四元数
Tensor quatFromPitch(double angle) {
    const double half = angle * 0.5;
    return Tensor{static_cast<float>(std::cos(half)), 0.0f,
                  static_cast<float>(std::sin(half)), 0.0f};
}

void testTiltedThrust(const SixDofConfig &cfg) {
    std::cout << "\n[Part A] 倾斜姿态下的推力分解\n";

    const double m = cfg.base.mass;
    const double g = cfg.base.gravity;
    const double angle = 30.0 / 180.0 * M_PI;

    SixDofState s;
    s.pos = makeVec3(0.0f, 0.0f, 0.0f);
    s.vel = makeVec3(0.0f, 0.0f, 0.0f);
    s.quat = quatFromPitch(angle);
    s.omega = makeVec3(0.0f, 0.0f, 0.0f);

    // 推力取 m*g：倾斜后竖直分量只剩 cos(30°)，不足以完全抵消重力
    const Tensor a = sixDofAcceleration(s.vel, s.quat, m * g, cfg.base);
    const std::vector<float> av = toVector(a);

    // 解析解：a = [ -g·sinθ, 0, g·(1 − cosθ) ]
    const double want_n = -g * std::sin(angle);
    const double want_d = g * (1.0 - std::cos(angle));

    check("水平分量 a_n = -g·sinθ", av[0], want_n, 1e-4);
    check("水平分量 a_e = 0", av[1], 0.0, 1e-6);
    check("竖直分量 a_d = g·(1-cosθ)", av[2], want_d, 1e-4);

    std::cout << "  -> 倾斜 30° 且推力仅等于 mg 时，水平加速度 "
              << av[0] << " m/s²，同时以 " << av[2] << " m/s² 掉高度\n";

    // 水平姿态时推力应恰好抵消重力，加速度为零
    SixDofState level = s;
    level.quat = identityQuat();
    const Tensor a_level = sixDofAcceleration(level.vel, level.quat, m * g, cfg.base);
    const std::vector<float> al = toVector(a_level);
    check("水平悬停 a_n = 0", al[0], 0.0, 1e-5);
    check("水平悬停 a_e = 0", al[1], 0.0, 1e-5);
    check("水平悬停 a_d = 0", al[2], 0.0, 1e-5);
}

void testQuatIntegrity(const SixDofConfig &cfg) {
    std::cout << "\n[Part A2] 四元数积分保持单位模长\n";

    SixDofState s;
    s.pos = makeVec3(0.0f, 0.0f, 0.0f);
    s.vel = makeVec3(0.0f, 0.0f, 0.0f);
    s.quat = identityQuat();
    s.omega = makeVec3(0.0f, 0.0f, 0.0f);

    // 恒定角速度下积分 2 秒，检查四元数模长与积分角度的解析解
    const double rate = 1.0; // rad/s 绕机体 x 轴
    const Tensor torque = makeVec3(static_cast<float>(rate * cfg.inertia[0]), 0.0f, 0.0f);

    SixDofState cur = s;
    const int steps = 2000;
    for (int i = 0; i < steps; ++i) {
        cur = rk4StepSixDof(cur, 0.0, torque, cfg, cfg.base.dt);
    }

    const std::vector<float> q = toVector(cur.quat);
    const double qn = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    check("四元数模长保持为 1", qn, 1.0, 1e-6);

    // 绕单轴恒角速度积分 2 秒 => 转过 rate*2 弧度
    const double half = rate * 2.0 * 0.5;
    check("旋转角符合解析解 (q.w = cos(θ/2))", q[0], std::cos(half), 1e-3);
}

void testClosedLoopHover(const SixDofConfig &cfg) {
    std::cout << "\n[Part B] 级联 PID 定点悬停\n";

    SixDofPidController pid(cfg);

    SixDofState s0;
    s0.pos = makeVec3(0.0f, 0.0f, 0.0f);
    s0.vel = makeVec3(0.0f, 0.0f, 0.0f);
    s0.quat = identityQuat();
    s0.omega = makeVec3(0.0f, 0.0f, 0.0f);

    SixDofSimulator sim(cfg, s0);
    const Tensor target = makeVec3(1.0f, 1.0f, -1.0f); // NED：向北 1m、向东 1m、向上 1m

    const int steps = 6000; // 6 秒
    double settle_time = -1.0;
    double max_tilt = 0.0;

    for (int i = 0; i < steps; ++i) {
        const SixDofCommand cmd = pid.compute(sim.state(), target, sim.time());
        sim.step(cmd.thrust_body, cmd.torque);

        max_tilt = std::max(max_tilt, std::fabs(pid.lastTiltDeg()));

        const std::vector<float> p = toVector(sim.state().pos);
        const double err = std::sqrt((p[0] - 1.0) * (p[0] - 1.0) +
                                     (p[1] - 1.0) * (p[1] - 1.0) +
                                     (p[2] + 1.0) * (p[2] + 1.0));
        if (err < 0.05 && settle_time < 0.0) {
            settle_time = sim.time();
        }
    }

    const std::vector<float> p = toVector(sim.state().pos);
    const std::vector<float> v = toVector(sim.state().vel);
    const std::vector<float> euler = toVector(quatToEuler(sim.state().quat));
    const double rad2deg = 180.0 / M_PI;

    const double final_err = std::sqrt((p[0] - 1.0) * (p[0] - 1.0) +
                                       (p[1] - 1.0) * (p[1] - 1.0) +
                                       (p[2] + 1.0) * (p[2] + 1.0));

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  终止位置 [" << p[0] << ", " << p[1] << ", " << p[2] << "]"
              << "  速度模 " << std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
              << "  姿态 rpy(deg) [" << euler[0] * rad2deg << ", " << euler[1] * rad2deg
              << ", " << euler[2] * rad2deg << "]" << std::endl;
    std::cout << "  收敛时间 " << settle_time << " s，最大倾角 " << max_tilt << " deg\n";

    check("终止位置误差 < 0.05 m", final_err, 0.0, 0.05);
    checkTrue("在时限内收敛", settle_time > 0.0);
    checkTrue("倾角未超过上限", max_tilt <= pid.gains().max_tilt_deg + 1e-6);
}

} // namespace

int main() {
    CtorchError::setPrintLevel(PrintLevel::MINIUM);

    const SixDofConfig cfg = makeConfig();

    std::cout << "========================================\n";
    std::cout << "OpenInspire3 六自由度测试\n";
    std::cout << "========================================\n";
    std::cout << "质量 " << cfg.base.mass << " kg，重力 " << cfg.base.gravity
              << " m/s²，惯量 [" << cfg.inertia[0] << ", " << cfg.inertia[1] << ", "
              << cfg.inertia[2] << "] kg·m²\n";

    testTiltedThrust(cfg);
    testQuatIntegrity(cfg);
    testClosedLoopHover(cfg);

    std::cout << "\n========================================\n";
    std::cout << g_checks - g_failed << " / " << g_checks << " checks passed\n";
    if (g_failed > 0) {
        std::cout << g_failed << " FAILED\n";
    }
    std::cout << "========================================\n";
    return g_failed == 0 ? 0 : 1;
}
