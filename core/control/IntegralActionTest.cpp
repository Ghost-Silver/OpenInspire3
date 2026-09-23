/**
 * @file IntegralActionTest.cpp
 * @brief 积分兜底：代价（相位裕度）与收益（消除模型外稳态误差）的量化
 * @author GhostFace
 * @date 2026/9/18
 *
 * @par 为什么此前没有积分
 *
 * 控制器一直是纯 PD 结构，这是**有意为之**：常值扰动（风）由前馈补偿，
 * 不需要积分慢慢消除。而积分会引入相位滞后 —— DelayMarginTest 已经证明
 * **延迟是相位裕度的主要杀手**，此时再无条件加积分是危险的。
 *
 * @par 但纯 PD 有一个真实缺口
 *
 * 前馈依赖模型准确性。当出现**完全未建模**的效应时（机身不对称、电机安装
 * 偏斜、重心偏移、传感器零偏），PD 会留下稳态误差，而没有任何机制去消除它。
 * 自适应补偿补的是「已知效应 + 未知系数」，对这类结构性偏差无能为力。
 *
 * 积分正是为这一类误差存在的。所以问题不是「要不要积分」，而是
 * **「加多少积分、代价多大、换来什么」** —— 本测试量化这三件事。
 *
 * @par 相位代价的解析式
 *
 * 位置环开环（被控对象为双积分器 `1/s²`）：
 *
 * @verbatim
 *   L(s) = (kp + kd·s + ki/s) / s²
 *
 *   在 s = jω：分子 = kp + j(kd·ω − ki/ω)，分母 = −ω²
 *   故  ∠L(jω) = atan2(kd·ω − ki/ω, kp) − 180°
 *       PM     = atan2(kd·ωc − ki/ωc, kp) − ωc·T      （T 为总延迟）
 * @endverbatim
 *
 * 关键在于 **`ki/ω` 是从 `kd·ω` 里减掉的** —— 这就是积分吃裕度的机制。
 * 穿越频率 ωc 由 `|L| = 1` 定出，因此加 ki 后 ωc 也会变，须数值求解。
 *
 * @par 收益的量化方式
 *
 * 注入一个控制器**不知道**的常值外力（模拟重心偏移、电机安装偏斜），
 * 测量稳态位置误差。纯 PD 下该误差为 `F/kp`（解析可预测），加积分后应趋零。
 */

#include "SixDofPidController.h"
#include "SixDofSimulator.h"
#include "SixDofTypes.h"
#include "TensorUtils.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace oi3;

namespace {

int g_checks = 0;
int g_failed = 0;

void checkTrue(const char *name, bool cond) {
    ++g_checks;
    if (!cond) {
        ++g_failed;
    }
    std::cout << "  [" << (cond ? " ok " : "FAIL") << "] " << name << "\n";
}

std::array<double, 3> readVec(const Tensor &t) {
    const std::vector<float> v = toVector(t);
    return {v[0], v[1], v[2]};
}

/// 位置环开环频响：L(s) = (kp + kd·s + ki/s)/s²
std::complex<double> posOpenLoop(double kp, double kd, double ki, double omega) {
    const std::complex<double> s(0.0, omega);
    return (kp + kd * s + ki / s) / (s * s);
}

/// 数值求穿越频率与相位裕度（含纯延迟 T）
struct Margin {
    double wc = 0.0;
    double pm_deg = 0.0;
};

Margin posMargin(double kp, double kd, double ki, double delay) {
    // 延迟不改变幅值，故穿越频率由无延迟的开环定出
    double lo = 1e-3, hi = 1e3;
    for (int it = 0; it < 100; ++it) {
        const double mid = std::sqrt(lo * hi);
        if (std::abs(posOpenLoop(kp, kd, ki, mid)) > 1.0) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Margin m;
    m.wc = std::sqrt(lo * hi);
    const double ideal = std::arg(posOpenLoop(kp, kd, ki, m.wc)) * 180.0 / M_PI;
    const double loss = m.wc * delay * 180.0 / M_PI;
    m.pm_deg = 180.0 + ideal - loss;
    return m;
}

struct SettleResult {
    double ss_err = 0.0;   ///< 稳态误差（末段均值）
    double peak_err = 0.0; ///< 峰值误差
    double integral_max = 0.0; ///< 积分项的最大幅值（检验限幅是否生效）
    bool diverged = false;
};

/**
 * @brief 注入控制器不知道的常值外力，测稳态误差
 *
 * @param force 外力（NED，牛顿）—— 控制器**不知道**它存在
 * @param ki    积分增益（0 = 纯 PD）
 */
SettleResult runWithUnmodeledForce(double force_x, double ki, double seconds = 20.0) {
    SixDofConfig cfg;
    cfg.base.dt = 0.001;
    cfg.base.mass = 1.0;
    cfg.base.gravity = 9.81;
    cfg.base.drag_coeff = 0.0; // 隔离变量：只看外力，不加风阻
    cfg.base.max_thrust = 0.0;
    cfg.inertia[0] = 0.015;
    cfg.inertia[1] = 0.018;
    cfg.inertia[2] = 0.028;
    cfg.torque_limit = 5.0;
    cfg.max_body_thrust = 40.0;

    const double dt = cfg.base.dt;
    const int steps = static_cast<int>(seconds / dt);
    const std::array<double, 3> target = {0.0, 0.0, -5.0};
    const SixDofState init{makeVec3(0.0f, 0.0f, -5.0f), makeVec3(0.0f, 0.0f, 0.0f),
                           Tensor{1.0f, 0.0f, 0.0f, 0.0f}, makeVec3(0.0f, 0.0f, 0.0f)};
    SixDofSimulator sim(cfg, init);

    SixDofPidGains gains;
    gains.pos_ki = ki;
    SixDofPidController ctrl(cfg, gains);
    const Tensor tgt = makeVec3(0.0f, 0.0f, -5.0f);

    SettleResult out;
    double sq_ss = 0.0, mx = 0.0;
    int n = 0;
    const int from = steps * 3 / 4; // 末 1/4 段视为稳态

    for (int k = 0; k < steps; ++k) {
        const double t = static_cast<double>(k) * dt;
        const SixDofCommand cmd = ctrl.compute(sim.state(), tgt, t);
        sim.step(cmd.thrust_body, cmd.torque);

        // 外力注入：在**位置积分之后**叠加一个速度增量，等效于常值外力。
        //
        // 注意必须在 step() 之后改速度（step 内部已推进位置）。第一版在
        // step 之后立刻 setState 覆盖整个状态 —— 而 setState 是「直接设置
        // 状态」，读改写时会用 step 刚算出的新状态，看似没问题；真正的错在
        // 于当时 compute() 那条路径根本没有积分项（独立实现漏改），
        // 于是积分恒为 0、稳态误差恒等于初始偏差 5.02 m。
        if (force_x != 0.0) {
            const SixDofState st = sim.state();
            const std::vector<float> v = toVector(st.vel);
            SixDofState ns = st;
            ns.vel = makeVec3(static_cast<float>(v[0] + force_x / cfg.base.mass * dt),
                              static_cast<float>(v[1]), static_cast<float>(v[2]));
            sim.setState(ns);
        }

        const std::array<double, 3> p = readVec(sim.state().pos);
        if (!std::isfinite(p[0]) || std::fabs(p[0]) > 1e4) {
            out.diverged = true;
            return out;
        }
        // 误差必须是**相对目标**的偏差，而不是相对原点的距离。
        //
        // 第一版写成 sqrt(p[0]² + p[2]²)，而目标在 z = −5 —— 于是 p[2] ≈ −5
        // 主导了整个量，算出的「稳态误差」恒为 5.0（实际是高度差，不是误差）。
        // 正确写法减去目标位置。
        const double ex = p[0] - target[0];
        const double ey = p[1] - target[1];
        const double ez = p[2] - target[2];
        const double e = std::sqrt(ex * ex + ey * ey + ez * ez);
        mx = std::max(mx, e);
        if (k >= from) {
            sq_ss += e * e;
            ++n;
        }
    }

    if (n > 0) {
        out.ss_err = std::sqrt(sq_ss / n);
        out.peak_err = mx;
    }
    out.integral_max = ctrl.integralMagnitude();
    return out;
}

} // namespace

int main() {
    CtorchError::setPrintLevel(PrintLevel::MINIUM);

    const double kp = 4.0, kd = 3.0; // 默认位置环增益

    std::cout << "========================================\n";
    std::cout << "积分兜底：相位裕度代价 vs 稳态误差收益\n";
    std::cout << "========================================\n";

    // ---- 1. 解析：积分对相位裕度的影响 ----
    std::cout << "\n[1] 积分的相位裕度代价（解析，无延迟）\n";
    std::cout << "  PM = atan2(kd·ωc − ki/ωc, kp) —— ki/ωc 从 kd·ωc 中减去。\n\n";
    std::cout << "  " << std::setw(14) << "ki" << std::setw(18) << "穿越频率ωc"
              << std::setw(18) << "相位裕度(deg)" << std::setw(20) << "相对无积分损失"
              << "\n";

    std::array<double, 5> kis = {0.0, 0.5, 1.0, 2.0, 4.0};
    std::array<double, 5> pm0{};
    const Margin base = posMargin(kp, kd, 0.0, 0.0);
    for (int i = 0; i < 5; ++i) {
        const double ki = kis[static_cast<std::size_t>(i)];
        const Margin m = posMargin(kp, kd, ki, 0.0);
        pm0[static_cast<std::size_t>(i)] = m.pm_deg;
        std::cout << "  " << std::setw(14) << std::fixed << std::setprecision(2) << ki
                  << std::setw(18) << std::setprecision(4) << m.wc << std::setw(18)
                  << std::setprecision(2) << m.pm_deg << std::setw(20) << std::setprecision(2)
                  << (base.pm_deg - m.pm_deg) << "\n";
    }
    checkTrue("无积分时相位裕度充足（> 60°）", base.pm_deg > 60.0);
    checkTrue("积分使相位裕度单调下降（代价随 ki 增长）", pm0[0] > pm0[4]);
    checkTrue("小积分（ki ≤ 1）的代价可接受（损失 < 5°）", (pm0[0] - pm0[2]) < 5.0);

    // ---- 2. 有延迟时的联合代价 ----
    //
    // 这才是真机上的情形：延迟已经在吃裕度，积分再吃一口。
    std::cout << "\n[2] 与延迟的联合代价（这才是真机情形）\n";
    std::cout << "  延迟已在吃裕度，积分再吃一口 —— 必须一起算。\n\n";
    std::cout << "  " << std::setw(14) << "总延迟(ms)" << std::setw(16) << "ki=0 PM"
              << std::setw(16) << "ki=1 PM" << std::setw(16) << "ki=2 PM"
              << std::setw(18) << "ki=2 是否安全" << "\n";
    for (double td : {0.0, 0.002, 0.005, 0.010, 0.020}) {
        const Margin a = posMargin(kp, kd, 0.0, td);
        const Margin b = posMargin(kp, kd, 1.0, td);
        const Margin c = posMargin(kp, kd, 2.0, td);
        std::cout << "  " << std::setw(14) << std::fixed << std::setprecision(1) << (td * 1000.0)
                  << std::setw(16) << std::setprecision(2) << a.pm_deg << std::setw(16) << b.pm_deg
                  << std::setw(16) << c.pm_deg << std::setw(18)
                  << (c.pm_deg > 45.0 ? "是" : "否") << "\n";
    }
    checkTrue("延迟与积分的代价可叠加（两者都吃裕度）",
              posMargin(kp, kd, 2.0, 0.005).pm_deg < posMargin(kp, kd, 0.0, 0.0).pm_deg);

    // ---- 3. 收益：消除模型外稳态误差 ----
    std::cout << "\n[3] 收益：消除**模型外**常值外力造成的稳态误差\n";
    std::cout << "  注入 2 N 常值外力（控制器不知道），模拟重心偏移/电机偏斜。\n";
    std::cout << "  纯 PD 下的解析稳态误差 = F/kp = " << std::setprecision(4) << (2.0 / kp)
              << " m。\n\n";
    std::cout << "  " << std::setw(14) << "ki" << std::setw(20) << "稳态误差(m)"
              << std::setw(18) << "峰值误差(m)" << std::setw(22) << "积分项峰值" << "\n";

    std::array<double, 4> kis2 = {0.0, 0.5, 1.0, 2.0};
    std::array<double, 4> ss{};
    for (int i = 0; i < 4; ++i) {
        const double ki = kis2[static_cast<std::size_t>(i)];
        const SettleResult r = runWithUnmodeledForce(2.0, ki);
        ss[static_cast<std::size_t>(i)] = r.ss_err;
        std::cout << "  " << std::setw(14) << std::fixed << std::setprecision(2) << ki
                  << std::setw(20) << std::setprecision(6) << r.ss_err << std::setw(18)
                  << std::setprecision(4) << r.peak_err << std::setw(22) << std::setprecision(4)
                  << r.integral_max << "\n";
    }
    checkTrue("纯 PD 存在稳态误差（与解析 F/kp 吻合）",
              std::fabs(ss[0] - 2.0 / kp) < 0.05);
    checkTrue("加积分后稳态误差显著减小（至少降一个量级）", ss[3] < ss[0] * 0.1);

    // ---- 4. 积分限幅的必要性 ----
    std::cout << "\n[4] 积分限幅（anti-windup）\n";
    std::cout << "  无外力时积分项应保持很小；有外力时不应无限增长。\n\n";
    {
        const SettleResult no_force = runWithUnmodeledForce(0.0, 2.0);
        const SettleResult with_force = runWithUnmodeledForce(2.0, 2.0);
        std::cout << "  无外力时积分项峰值 " << std::setprecision(6) << no_force.integral_max
                  << "，稳态误差 " << no_force.ss_err << "\n";
        std::cout << "  有外力时积分项峰值 " << with_force.integral_max << "，稳态误差 "
                  << with_force.ss_err << "\n";
        checkTrue("无外力时积分项保持小（不产生虚假补偿）", no_force.integral_max < 0.1);
        checkTrue("有外力时积分项有界（限幅生效，未发散）",
                  with_force.integral_max < 1.0 && !with_force.diverged);
    }

    std::cout << "\n[结论]\n";
    std::cout << "  1. 积分的代价有解析式：`PM = atan2(kd·ωc − ki/ωc, kp) − ωc·T`，\n";
    std::cout << "     其中 `ki/ωc` 从 `kd·ωc` 中减去 —— 这就是吃裕度的机制。\n";
    std::cout << "  2. 小积分（ki ≤ 1）的相位代价 < 5°，可接受；且它与延迟的代价\n";
    std::cout << "     可叠加，真机上必须一起算（本测试给出联合表）。\n";
    std::cout << "  3. 收益明确：纯 PD 对**模型外**常值外力留下 `F/kp` 的稳态误差，\n";
    std::cout << "     加积分后降低一个量级以上。前馈补不了这一类偏差（它补的是\n";
    std::cout << "     已知效应），这正是积分的不可替代之处。\n";
    std::cout << "  4. 因此默认仍关闭积分（既有结果逐位不变），但在模型不确定的\n";
    std::cout << "     真机部署中应开启小积分（推荐 ki ≈ 0.5~1.0）。\n";

    std::cout << "\n========================================\n";
    std::cout << g_checks - g_failed << " / " << g_checks << " checks passed\n";
    if (g_failed > 0) {
        std::cout << g_failed << " FAILED\n";
    }
    std::cout << "========================================\n";
    return g_failed == 0 ? 0 : 1;
}
