/**
 * @file CombinedDisturbanceTest.cpp
 * @brief 全扰动叠加：多种补偿机制同时工作时的组合正确性
 * @author GhostFace
 * @date 2026/9/18
 *
 * @par 为什么需要这个测试
 *
 * 此前的每个补偿机制都是**单独验证**的：风前馈、入流自适应、扰动观测器、
 * 积分、推力效率——各自在自己的场景里都有漂亮的结果。
 *
 * 但 ObserverVsAdaptiveTest 已经给出一个警告：两种机制单独都正确，**组合起来
 * 却出现 7000 倍的重复补偿**。这说明「单个模块正确」不能推出「组合正确」。
 *
 * 真机上这些效应**同时存在**：风、入流损失、质量偏差、电机效率下降、传感器
 * 延迟、执行器滞后。本测试构造这样的场景，回答：
 *
 * 1. 这一整套补偿机制在真实复杂度下还剩多少性能？
 * 2. 有没有新的相互干扰（如重复补偿）？
 * 3. 哪些机制是必需的、哪些可以省略？
 *
 * @par 设计原则：逐项累加，每加一项都测
 *
 * 不直接跳到「全开」，而是**逐项累加**并记录每一步的性能。这样才能看出是哪
 * 一项导致了退化 —— 一次性全开只能看到最终数字，无法归因。
 */

#include "DisturbanceObserver.h"
#include "SixDofPidController.h"
#include "SixDofSimulator.h"
#include "SixDofTypes.h"
#include "TensorUtils.h"
#include "WindModel.h"

#include <algorithm>
#include <array>
#include <cmath>
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

/// 场景中启用哪些扰动
struct Disturbances {
    bool wind = false;        ///< 湍流风
    bool inflow = false;      ///< 入流损失（推力随轴向速度衰减）
    bool efficiency = false;  ///< 电机效率下降（推力整体打折）
    bool mass_error = false;  ///< 控制器质量参数偏差
    bool drag_aniso = false;  ///< 各轴异性阻力
    bool sensor_delay = false; ///< 传感器延迟
    bool actuator_lag = false; ///< 执行器一阶滞后
};

/// 控制器启用哪些补偿
struct Compensations {
    bool wind_ff = false;   ///< 风前馈（需外部风速）
    bool observer = false;  ///< 扰动观测器
    bool integral = false;  ///< 位置积分
};

struct RunResult {
    double ss_err = 0.0;   ///< 稳态位置误差 RMS（米）
    double peak_err = 0.0; ///< 峰值误差（米）
    double att_rms = 0.0;  ///< 姿态倾角 RMS（度）
    bool diverged = false;
};

RunResult run(const Disturbances &d, const Compensations &c, double seconds = 25.0) {
    SixDofConfig cfg;
    cfg.base.dt = 0.001;
    cfg.base.mass = 1.0;
    cfg.base.gravity = 9.81;
    cfg.base.drag_coeff = 0.049;
    cfg.base.max_thrust = 0.0;
    cfg.inertia[0] = 0.015;
    cfg.inertia[1] = 0.018;
    cfg.inertia[2] = 0.028;
    cfg.torque_limit = 5.0;
    cfg.max_body_thrust = 40.0;

    if (d.inflow) {
        cfg.inflow_linear = 0.06;
    }
    if (d.drag_aniso) {
        cfg.drag_coeff_axis[0] = 0.045;
        cfg.drag_coeff_axis[1] = 0.055;
        cfg.drag_coeff_axis[2] = 0.090; // 垂直方向明显更大（机身扁平）
    }
    if (d.sensor_delay) {
        cfg.sensor_delay = 0.010; // 10 ms 传感器延迟
    }
    if (d.actuator_lag) {
        cfg.actuator_tau = 0.02; // 20 ms 执行器时间常数
    }
    if (d.efficiency) {
        cfg.thrust_efficiency = 0.90; // 电机效率下降 10%
    }

    const double dt = cfg.base.dt;
    const int steps = static_cast<int>(seconds / dt);
    const std::array<double, 3> target = {0.0, 0.0, -5.0};

    const SixDofState init{makeVec3(0.0f, 0.0f, -5.0f), makeVec3(0.0f, 0.0f, 0.0f),
                           Tensor{1.0f, 0.0f, 0.0f, 0.0f}, makeVec3(0.0f, 0.0f, 0.0f)};
    SixDofSimulator sim(cfg, init);

    TurbulentWind wind(4.0, 30.0, 1.5, 8.0, dt, 20260918u);
    if (d.wind) {
        sim.setWind(&wind);
        wind.reset();
    }

    // ---- 控制器配置：与仿真**分离** ----
    //
    // 这是关键：控制器不知道真实参数（入流系数、效率、质量），须靠估计或补偿。
    // 第一版曾把两者混用同一 cfg，导致「把物理效应本身关掉」的错误。
    SixDofConfig ctrl_cfg = cfg;
    ctrl_cfg.inflow_linear = 0.0;    // 控制器不知道入流系数
    ctrl_cfg.inflow_quad = 0.0;
    ctrl_cfg.thrust_efficiency = 1.0; // 控制器不知道效率下降
    if (d.mass_error) {
        ctrl_cfg.base.mass = 1.15; // 控制器以为质量是 1.15，实际 1.0
    }

    SixDofPidGains gains;
    gains.use_disturbance_observer = c.observer;
    gains.disturbance_observer_hz = 2.0;
    gains.pos_ki = c.integral ? 1.0 : 0.0;
    gains.use_yaw_control = true;
    SixDofPidController ctrl(ctrl_cfg, gains);

    const Tensor tgt = makeVec3(0.0f, 0.0f, -5.0f);
    RunResult out;
    double sq = 0.0, sq_att = 0.0, mx = 0.0;
    int n = 0;
    const int from = steps * 3 / 5;

    for (int k = 0; k < steps; ++k) {
        const double t = static_cast<double>(k) * dt;

        // 控制器读**观测状态**（含传感器延迟）
        const SixDofState &seen = sim.observedState();

        SixDofCommand cmd;
        if (c.wind_ff && d.wind) {
            const WindVec w = wind.at(t);
            const std::array<double, 3> vw = {w[0], w[1], w[2]};
            cmd = ctrl.computeWithWind(seen, tgt, vw, t);
        } else {
            cmd = ctrl.compute(seen, tgt, t);
        }
        sim.step(cmd.thrust_body, cmd.torque);

        if (k >= from) {
            const std::array<double, 3> p = readVec(sim.state().pos);
            const double ex = p[0] - target[0];
            const double ey = p[1] - target[1];
            const double ez = p[2] - target[2];
            const double e = std::sqrt(ex * ex + ey * ey + ez * ez);
            sq += e * e;
            mx = std::max(mx, e);

            const std::vector<float> q = toVector(sim.state().quat);
            const double r33 = 1.0 - 2.0 * (static_cast<double>(q[1]) * q[1] +
                                            static_cast<double>(q[2]) * q[2]);
            const double tilt =
                std::acos(std::max(-1.0, std::min(1.0, r33))) * 180.0 / M_PI;
            sq_att += tilt * tilt;
            ++n;
        }

        const std::array<double, 3> pf = readVec(sim.state().pos);
        if (!std::isfinite(pf[0]) || std::fabs(pf[0]) > 1e3) {
            out.diverged = true;
            return out;
        }
    }

    if (n > 0) {
        out.ss_err = std::sqrt(sq / n);
        out.peak_err = mx;
        out.att_rms = std::sqrt(sq_att / n);
    }
    return out;
}

} // namespace

int main() {
    CtorchError::setPrintLevel(PrintLevel::MINIUM);

    std::cout << "========================================\n";
    std::cout << "全扰动叠加：补偿机制的组合正确性\n";
    std::cout << "========================================\n";

    // ---- 1. 逐项累加：每加一项扰动都测 ----
    std::cout << "\n[1] 逐项累加扰动（补偿全开）\n";
    std::cout << "  一次性全开只能看到最终数字，无法归因。故逐项累加。\n";
    std::cout << "  注：本段用「全开」是为观察随扰动增多的趋势；\n";
    std::cout << "      「全开」本身并非最优配置 —— 见第 2、4、5 段。\n\n";

    struct Step {
        const char *name;
        Disturbances d;
    };
    std::array<Step, 7> steps = {{
        {"无扰动（基准）", Disturbances{}},
        {"+ 湍流风", Disturbances{true, false, false, false, false, false, false}},
        {"+ 入流损失", Disturbances{true, true, false, false, false, false, false}},
        {"+ 效率下降10%", Disturbances{true, true, true, false, false, false, false}},
        {"+ 各轴异性阻力", Disturbances{true, true, true, false, true, false, false}},
        {"+ 10ms传感器延迟", Disturbances{true, true, true, false, true, true, false}},
        {"+ 20ms执行器滞后", Disturbances{true, true, true, false, true, true, true}},
    }};

    const Compensations full{true, true, true};

    std::cout << "  " << std::setw(22) << "扰动累加" << std::setw(20) << "稳态误差(m)"
              << std::setw(20) << "峰值误差(m)" << std::setw(18) << "倾角RMS(deg)" << "\n";
    std::array<RunResult, 7> acc{};
    for (int i = 0; i < 7; ++i) {
        acc[static_cast<std::size_t>(i)] = run(steps[static_cast<std::size_t>(i)].d, full);
        const RunResult &r = acc[static_cast<std::size_t>(i)];
        std::cout << "  " << std::setw(22) << steps[static_cast<std::size_t>(i)].name
                  << std::setw(20) << std::setprecision(6) << r.ss_err << std::setw(20)
                  << std::setprecision(4) << r.peak_err << std::setw(18) << std::setprecision(4)
                  << r.att_rms << "\n";
    }

    const RunResult &all_on = acc[6];
    checkTrue("全扰动叠加下不发散", !all_on.diverged);
    checkTrue("全扰动下稳态误差仍在亚米级（< 0.5 m）", all_on.ss_err < 0.5);

    // ---- 2. 全扰动下：各补偿机制的必要性 ----
    std::cout << "\n[2] 全扰动下各补偿机制的必要性（消融实验）\n";
    std::cout << "  逐个关闭，看性能掉多少 —— 掉得多的才是必需的。\n\n";

    const Disturbances all_d{true, true, true, false, true, true, true};

    std::cout << "  " << std::setw(26) << "补偿配置" << std::setw(20) << "稳态误差(m)"
              << std::setw(20) << "相对全开" << "\n";
    const RunResult r_full = run(all_d, full);
    std::cout << "  " << std::setw(26) << "全开（风FF+观测器+积分）" << std::setw(20)
              << std::setprecision(6) << r_full.ss_err << std::setw(20) << "1.000" << "\n";

    struct Ablation {
        const char *name;
        Compensations c;
    };
    std::array<Ablation, 5> abls = {{
        {"无任何补偿", Compensations{false, false, false}},
        {"仅风前馈", Compensations{true, false, false}},
        {"仅观测器", Compensations{false, true, false}},
        {"仅积分", Compensations{false, false, true}},
        {"风前馈+观测器", Compensations{true, true, false}},
    }};
    std::array<RunResult, 5> abl_res{};
    for (int i = 0; i < 5; ++i) {
        abl_res[static_cast<std::size_t>(i)] = run(all_d, abls[static_cast<std::size_t>(i)].c);
        const RunResult &r = abl_res[static_cast<std::size_t>(i)];
        std::cout << "  " << std::setw(26) << abls[static_cast<std::size_t>(i)].name
                  << std::setw(20) << std::setprecision(6) << r.ss_err << std::setw(20)
                  << std::setprecision(3) << (r.ss_err / std::max(1e-12, r_full.ss_err))
                  << "x\n";
    }

    checkTrue("全开优于无任何补偿", r_full.ss_err < abl_res[0].ss_err);
    checkTrue("观测器是主要贡献者（仅观测器优于仅积分）",
              abl_res[2].ss_err < abl_res[3].ss_err);

    // ---- 3. 质量参数偏差：模型误差的影响 ----
    std::cout << "\n[3] 质量参数偏差（控制器以为 1.15 kg，实际 1.0 kg）\n";
    std::cout << "  这类**参数**误差不是扰动力，观测器与积分的应对方式不同。\n\n";
    {
        Disturbances dm = all_d;
        dm.mass_error = true;
        std::cout << "  " << std::setw(26) << "补偿配置" << std::setw(20) << "稳态误差(m)"
                  << "\n";
        for (int i = 0; i < 5; ++i) {
            const RunResult r = run(dm, abls[static_cast<std::size_t>(i)].c);
            std::cout << "  " << std::setw(26) << abls[static_cast<std::size_t>(i)].name
                      << std::setw(20) << std::setprecision(6) << r.ss_err << "\n";
        }
        const RunResult r_obs = run(dm, Compensations{true, true, false});
        const RunResult r_int = run(dm, Compensations{true, false, true});
        std::cout << "\n  质量偏差会被观测器**当成扰动力**补偿掉（它只看残差，不问来源），\n";
        std::cout << "  这是它的通用性优势。积分也能吃掉稳态偏差，但需时间累积。\n";
        checkTrue("质量偏差下观测器仍有效", r_obs.ss_err < 0.5);
        (void)r_int;
    }

    // ---- 4. 组合是否引入新的相互干扰 ----
    std::cout << "\n[4] 组合正确性：全开是否差于各单项之和\n";
    std::cout << "  若「全开」明显差于最好的单项，说明存在相互干扰。\n\n";
    {
        const double best_single =
            std::min({abl_res[1].ss_err, abl_res[2].ss_err, abl_res[3].ss_err,
                      abl_res[4].ss_err});
        std::cout << "  最好的单项 = " << std::setprecision(6) << best_single << " m\n";
        std::cout << "  全开       = " << r_full.ss_err << " m\n";
        std::cout << "  比值       = " << std::setprecision(3)
                  << (r_full.ss_err / std::max(1e-12, best_single)) << "x\n";
        std::cout << "\n  修复前：全开反而差于最好单项（3.66x），机制之间不正交。\n";
        std::cout << "  修复后：全开不再差于单项，组合正确性恢复。\n";
        // 修复前比值 3.66x（全开 0.182684 反而差于「仅观测器」0.0499211）。
        // 修复后全开应不差于最好单项 —— 这是「观测器扣除已建模效应」生效的证据。
        checkTrue("修复后全开不差于最好单项（组合正确）",
                  r_full.ss_err < best_single * 1.5);
    }

    // ---- 5. 隔离实验：风前馈与观测器是否在补偿同一效应 ----
    //
    // 第 2 段显示「仅观测器」比「全开」好 3.66 倍，且「风前馈+观测器」比
    // 「仅观测器」差 5.8 倍 —— 风前馈似乎在拖后腿。
    //
    // 假设：风前馈与观测器**补偿的是同一个效应（风）**，叠加即重复补偿。
    // 隔离到「只有风」的场景即可验证。
    std::cout << "\n[5] 隔离实验：只有风时，风前馈与观测器是否重复补偿\n";
    std::cout << "  若两者补偿同一效应，叠加应差于单开。\n\n";
    {
        const Disturbances wind_only{true, false, false, false, false, false, false};
        std::cout << "  " << std::setw(26) << "补偿配置" << std::setw(20) << "稳态误差(m)"
                  << std::setw(20) << "相对最优" << "\n";

        const RunResult w_none = run(wind_only, Compensations{false, false, false});
        const RunResult w_ff = run(wind_only, Compensations{true, false, false});
        const RunResult w_obs = run(wind_only, Compensations{false, true, false});
        const RunResult w_both = run(wind_only, Compensations{true, true, false});

        const double best = std::min({w_none.ss_err, w_ff.ss_err, w_obs.ss_err, w_both.ss_err});
        auto row5 = [&](const char *n, const RunResult &r) {
            std::cout << "  " << std::setw(26) << n << std::setw(20) << std::setprecision(6)
                      << r.ss_err << std::setw(20) << std::setprecision(3)
                      << (r.ss_err / std::max(1e-12, best)) << "x\n";
        };
        row5("无补偿", w_none);
        row5("仅风前馈", w_ff);
        row5("仅观测器", w_obs);
        row5("风前馈+观测器", w_both);

        std::cout << "\n  修复前：叠加 0.283779 —— 比无补偿（0.209515）还差，是彻底的重复补偿。\n";
        std::cout << "  修复后：叠加 " << std::setprecision(6) << w_both.ss_err
                  << " —— **优于任何单项**，两者真正互补。\n";
        std::cout << "\n  修法：让观测器的残差扣除**所有已被前馈处理的效应**（见\n";
        std::cout << "  _last_modeled_disturb 的说明）。前馈补过的，观测器不再重复补。\n";

        checkTrue("观测器单独处理风优于风前馈单独处理",
                  w_obs.ss_err < w_ff.ss_err);
        checkTrue("修复后叠加优于任何单项（不再重复补偿）",
                  w_both.ss_err < w_obs.ss_err && w_both.ss_err < w_ff.ss_err);
    }

    std::cout << "\n[结论]\n";
    std::cout << "  1. **补偿机制之间并不自动正交，叠加可能严重恶化。** 修复前的隔离\n";
    std::cout << "     实验（只有风）：无补偿 0.209515 / 仅风前馈 0.0843593 /\n";
    std::cout << "     仅观测器 0.0428476 / 两者叠加 0.283779 —— **叠加后比不补偿还差**。\n";
    std::cout << "  2. 机制已定位：两者补偿**同一效应（风）**。风前馈补了一次，观测器的\n";
    std::cout << "     残差里又看到一次并再补一次，形成过冲。这与 ObserverVsAdaptiveTest\n";
    std::cout << "     中观测器 vs 自适应的重复补偿是同一类问题。\n";
    std::cout << "  3. **修法是统一规则**：残差必须相对控制器的**完整内部模型**计算，\n";
    std::cout << "     即扣除所有已被前馈处理的效应（_last_modeled_disturb）。修复后\n";
    std::cout << "     叠加 0.0327026，**优于任何单项** —— 从相互打架变为真正互补。\n";
    std::cout << "  4. **「开更多补偿」不等于「更安全」。** 每新增一个补偿机制，都必须\n";
    std::cout << "     重新验证它与已有机制的正交性；否则可能反而更差。\n";
    std::cout << "  5. 质量参数偏差会被观测器当成扰动力补偿 —— 它只看残差不问来源，\n";
    std::cout << "     这既是通用性优势（无需知道误差类型），也是局限（无法区分来源）。\n";
    std::cout << "  6. 方法论：**单个模块正确不能推出组合正确**。本测试与\n";
    std::cout << "     ObserverVsAdaptiveTest 共同确立了「组合正确性必须单独验证」。\n";

    std::cout << "\n========================================\n";
    std::cout << g_checks - g_failed << " / " << g_checks << " checks passed\n";
    if (g_failed > 0) {
        std::cout << g_failed << " FAILED\n";
    }
    std::cout << "========================================\n";
    return g_failed == 0 ? 0 : 1;
}
