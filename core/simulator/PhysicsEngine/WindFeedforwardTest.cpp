/**
 * @file WindFeedforwardTest.cpp
 * @brief 被动抗风 vs 主动抗风：解析前馈能补多少，剩下的是什么
 * @author GhostFace
 * @date 2026/9/18
 *
 * @par 这个测试要回答什么
 *
 * 定点 PID 对风是**被动响应**：只能靠位置误差感知风，因此必然产生稳态偏移
 * `e = k·|v_wind|²/(m·kp)`。但风速是可测的 —— 已知扰动的大小和方向，就没有
 * 理由等它把飞行器推偏再去纠正。
 *
 * 本测试量化「被动」与「主动」的差距，并把误差拆成两个分量：
 *
 * @verbatim
 *   |e|² = （均值分量）² + （波动分量）²
 * @endverbatim
 *
 * 解析前馈补偿的是**稳态力**，也就是均值分量。湍流的时变部分它管不了 ——
 * 那部分靠位置误差反馈衰减，衰减多少取决于回路带宽。
 *
 * @par 为什么这个分解重要
 *
 * 它是判断「风补偿任务上学习有没有空间」的直接依据：
 *
 *  - 若解析前馈把误差压到接近零，说明这个任务没有学习价值（简单方法够用）；
 *  - 若前馈消掉均值后仍留下可观的波动，那部分就是学习的目标 —— 而且有
 *    天然的对照基线（同一个风场下，有/无补偿的误差分解）。
 *
 * @par 第三段测的是另一件事：信息延迟
 *
 * 真实飞控的风速估计有延迟。用**滞后**的风速做前馈，补偿本身会失配，
 * 甚至可能比不补偿更差。这一延迟造成的误差同样无法用解析式消除，
 * 但它是**可学的**（网络可以从历史观测中预测当前扰动）。
 */

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

void checkNear(const char *name, double got, double want, double rel_tol) {
    ++g_checks;
    const double tol = std::fabs(want) * rel_tol + 1e-9;
    const bool ok = std::isfinite(got) && std::fabs(got - want) <= tol;
    if (!ok) {
        ++g_failed;
    }
    std::cout << "  [" << (ok ? " ok " : "FAIL") << "] " << name << "（实测 " << got
              << "，预测 " << want << "）\n";
}

std::array<double, 3> readVec(const Tensor &t) {
    const std::vector<float> v = toVector(t);
    return {v[0], v[1], v[2]};
}

/// 误差分解结果
struct ErrorStat {
    double mean_mag = 0.0; ///< 误差均值的模（系统性偏移）
    double rms = 0.0;      ///< 误差的均方根
    double deviation = 0.0; ///< 围绕均值的波动分量

    [[nodiscard]] double meanShare() const {
        return rms > 1e-12 ? mean_mag / rms : 0.0;
    }
};

/**
 * @brief 在给定风场下悬停，统计稳态误差
 *
 * @param use_feedforward true = 用 computeWithWind（主动前馈）
 * @param wind_delay      前馈所用的风速相对当前时刻的滞后（秒）；0 = 完美信息
 */
ErrorStat hoverStats(const SixDofConfig &cfg, const std::array<double, 3> &target,
                     WindModel &wind, double seconds, bool use_feedforward,
                     double wind_delay = 0.0) {
    const SixDofState init{makeVec3(static_cast<float>(target[0]),
                                    static_cast<float>(target[1]),
                                    static_cast<float>(target[2])),
                           makeVec3(0.0f, 0.0f, 0.0f), Tensor{1.0f, 0.0f, 0.0f, 0.0f},
                           makeVec3(0.0f, 0.0f, 0.0f)};
    SixDofSimulator sim(cfg, init);
    sim.setWind(&wind);
    wind.reset();
    SixDofPidController ctrl(cfg, {});

    const double dt = cfg.base.dt;
    const int steps = static_cast<int>(seconds / dt);
    const int from = steps / 2; // 后一半作为稳态段

    double sx = 0.0, sy = 0.0, sz = 0.0;
    double sq = 0.0;
    int n = 0;

    for (int k = 0; k < steps; ++k) {
        const double t = static_cast<double>(k) * dt;
        SixDofCommand cmd;
        if (use_feedforward) {
            const WindVec w = (wind_delay > 0.0) ? wind.at(std::max(0.0, t - wind_delay))
                                                 : wind.at(t);
            cmd = ctrl.computeWithWind(sim.state(),
                                       makeVec3(static_cast<float>(target[0]),
                                                static_cast<float>(target[1]),
                                                static_cast<float>(target[2])),
                                       w, t);
        } else {
            cmd = ctrl.compute(sim.state(),
                               makeVec3(static_cast<float>(target[0]),
                                        static_cast<float>(target[1]),
                                        static_cast<float>(target[2])),
                               t);
        }
        sim.step(cmd.thrust_body, cmd.torque);

        if (k >= from) {
            const std::array<double, 3> p = readVec(sim.state().pos);
            const double ex = p[0] - target[0];
            const double ey = p[1] - target[1];
            const double ez = p[2] - target[2];
            sx += ex;
            sy += ey;
            sz += ez;
            sq += ex * ex + ey * ey + ez * ez;
            ++n;
        }
    }

    ErrorStat out;
    if (n > 0) {
        const double mx = sx / n, my = sy / n, mz = sz / n;
        out.mean_mag = std::sqrt(mx * mx + my * my + mz * mz);
        out.rms = std::sqrt(sq / n);
        const double var = std::max(0.0, out.rms * out.rms - out.mean_mag * out.mean_mag);
        out.deviation = std::sqrt(var);
    }
    return out;
}

} // namespace

int main() {
    CtorchError::setPrintLevel(PrintLevel::MINIUM);

    SixDofConfig cfg;
    cfg.base.dt = 0.001;
    cfg.base.mass = 1.0;
    cfg.base.gravity = 9.81;
    cfg.base.drag_coeff = 0.049; // 典型小四轴 Cd·A ≈ 0.08 m²
    cfg.base.max_thrust = 0.0;
    cfg.inertia[0] = 0.015;
    cfg.inertia[1] = 0.018;
    cfg.inertia[2] = 0.028;
    cfg.torque_limit = 1.0;
    cfg.max_body_thrust = 20.0;

    const double k = cfg.base.drag_coeff;
    const double m = cfg.base.mass;
    SixDofPidGains gains;
    const std::array<double, 3> target = {0.0, 0.0, -5.0};

    std::cout << "========================================\n";
    std::cout << "被动抗风 vs 主动抗风\n";
    std::cout << "解析前馈：F_wind = k·|v|·v，a_ff = −F_wind/m（稳态 v_rel = −v_wind）\n";
    std::cout << "========================================\n";

    // ---- 1. 常值风：前馈能否消除稳态偏移 ----
    std::cout << "\n[1] 常值风下的悬停（风速沿 +x 吹）\n";
    std::cout << "  被动预测偏移 e = k·v²/(m·kp)，主动前馈理论上把它降到零\n\n";
    std::cout << "  " << std::setw(10) << "风速(m/s)" << std::setw(16) << "被动误差(m)"
              << std::setw(18) << "被动预测(m)" << std::setw(16) << "主动误差(m)"
              << std::setw(16) << "改善倍数" << "\n";

    std::array<double, 5> winds = {2.0, 4.0, 6.0, 8.0, 10.0};
    std::array<double, 5> passive{}, active{}, predicted{};
    for (int i = 0; i < 5; ++i) {
        const double v = winds[static_cast<std::size_t>(i)];
        SteadyWind w(v, 0.0);
        const ErrorStat pe = hoverStats(cfg, target, w, 12.0, false);
        const ErrorStat ae = hoverStats(cfg, target, w, 12.0, true);

        predicted[static_cast<std::size_t>(i)] = k * v * v / (m * gains.pos_kp);
        passive[static_cast<std::size_t>(i)] = pe.mean_mag;
        active[static_cast<std::size_t>(i)] = ae.mean_mag;

        std::cout << "  " << std::setw(10) << std::fixed << std::setprecision(1) << v
                  << std::setw(16) << std::setprecision(4) << pe.mean_mag << std::setw(18)
                  << predicted[static_cast<std::size_t>(i)] << std::setw(16) << ae.mean_mag
                  << std::setw(16) << std::setprecision(1)
                  << (ae.mean_mag > 1e-9 ? pe.mean_mag / ae.mean_mag : 0.0) << "\n";
    }

    for (int i = 0; i < 3; ++i) {
        checkNear(("被动抗风 " + std::to_string(static_cast<int>(winds[static_cast<std::size_t>(i)])) +
                   " m/s 的偏移与解析式吻合")
                      .c_str(),
                  passive[static_cast<std::size_t>(i)],
                  predicted[static_cast<std::size_t>(i)], 0.15);
    }
    // 核心结论：主动前馈把系统性偏移消掉
    bool all_better = true;
    for (int i = 0; i < 5; ++i) {
        if (active[static_cast<std::size_t>(i)] > 0.1 * passive[static_cast<std::size_t>(i)]) {
            all_better = false;
        }
    }
    checkTrue("主动前馈把稳态偏移降到被动情形的 1/10 以下（全部风速）", all_better);

    // ---- 2. 湍流：均值 vs 波动的分解 ----
    std::cout << "\n[2] 湍流下的误差分解（背景风 5 m/s + Dryden 湍流 σ=1.5 m/s）\n";
    std::cout << "  解析前馈补偿的是稳态力（均值分量）；时变部分要靠反馈衰减。\n";
    std::cout << "  第二列起把误差拆为「均值模」与「围绕均值的波动」两部分。\n\n";
    std::cout << "  " << std::setw(14) << "配置" << std::setw(14) << "误差RMS(m)"
              << std::setw(16) << "均值分量(m)" << std::setw(18) << "波动分量(m)"
              << std::setw(16) << "均值占比" << "\n";

    ErrorStat turb_passive, turb_active;
    {
        TurbulentWind w1(5.0, 0.0, 1.5, 10.0, cfg.base.dt, 2026u);
        turb_passive = hoverStats(cfg, target, w1, 20.0, false);
        TurbulentWind w2(5.0, 0.0, 1.5, 10.0, cfg.base.dt, 2026u);
        turb_active = hoverStats(cfg, target, w2, 20.0, true);

        auto row = [](const char *label, const ErrorStat &e) {
            std::cout << "  " << std::setw(14) << label << std::setw(14)
                      << std::setprecision(4) << e.rms << std::setw(16) << e.mean_mag
                      << std::setw(18) << e.deviation << std::setw(16)
                      << std::setprecision(1) << (e.meanShare() * 100.0) << "%\n";
        };
        row("被动", turb_passive);
        row("主动前馈", turb_active);
    }
    std::cout << "\n  前馈把均值分量降到 "
              << std::setprecision(1)
              << (turb_passive.mean_mag > 1e-9
                      ? 100.0 * turb_active.mean_mag / turb_passive.mean_mag
                      : 0.0)
              << "%，但波动分量只降到 "
              << (turb_passive.deviation > 1e-9
                      ? 100.0 * turb_active.deviation / turb_passive.deviation
                      : 0.0)
              << "% —— 后者由位置环带宽决定，前馈无从改善。\n";

    checkTrue("主动前馈显著削减均值分量（降到被动的 30% 以下）",
              turb_active.mean_mag < 0.3 * turb_passive.mean_mag);
    checkTrue("湍流下波动分量仍显著存在（这即是学习可作用的余量）",
              turb_active.deviation > 0.01);
    // 波动分量占 RMS 的比例，是「反馈能管多少」的直接度量
    checkTrue("主动前馈下波动分量成为误差主体（占比超过 60%）",
              turb_active.deviation > 0.6 * turb_active.rms);

    // ---- 3. 风速估计延迟的影响 ----
    std::cout << "\n[3] 风速估计延迟对前馈的影响（常值风 6 m/s + 阵风）\n";
    std::cout << "  真实飞控的风速估计有延迟；用滞后的风做前馈会失配。\n\n";
    std::cout << "  " << std::setw(16) << "延迟(s)" << std::setw(16) << "误差RMS(m)"
              << std::setw(16) << "峰值误差(m)" << "\n";
    {
        std::array<double, 4> delays = {0.0, 0.05, 0.1, 0.2};
        std::array<double, 4> rms{};
        for (int i = 0; i < 4; ++i) {
            // 用 1-cos 阵风测延迟的影响：常值风下延迟几乎无害（风不变）
            GustWind w(6.0, 0.0, 6.0, 0.0, 3.0, 2.0);
            const ErrorStat e =
                hoverStats(cfg, target, w, 10.0, true, delays[static_cast<std::size_t>(i)]);
            rms[static_cast<std::size_t>(i)] = e.rms;
            std::cout << "  " << std::setw(16) << std::fixed << std::setprecision(2)
                      << delays[static_cast<std::size_t>(i)] << std::setw(16)
                      << std::setprecision(4) << e.rms << std::setw(16)
                      << (e.mean_mag + e.deviation) << "\n";
        }
        checkTrue("零延迟的前馈优于有延迟的前馈（信息时效性有影响）",
                  rms[0] <= rms[3] * 1.02);
    }

    std::cout << "\n[结论]\n";
    std::cout << "  1. 主动前馈把常值风下的系统性偏移消掉一个数量级以上 —— 这是\n";
    std::cout << "     「已知扰动就该前馈掉」的直接验证。\n";
    std::cout << "  2. 湍流下误差可分为均值与波动两部分：前馈消除均值，波动由位置\n";
    std::cout << "     环带宽决定。**波动部分是学习可作用的余量**。\n";
    std::cout << "  3. 风速估计延迟使前馈失配，这部分误差解析式同样无法消除。\n";
    std::cout << "  4. 因此本任务对学习型控制的价值不在「补偿常值风」（解析式已够），\n";
    std::cout << "     而在：(a) 湍流下超越固定带宽的波动抑制；\n";
    std::cout << "             (b) 从历史观测预测扰动以抵消估计延迟。\n";

    std::cout << "\n========================================\n";
    std::cout << g_checks - g_failed << " / " << g_checks << " checks passed\n";
    if (g_failed > 0) {
        std::cout << g_failed << " FAILED\n";
    }
    std::cout << "========================================\n";
    return g_failed == 0 ? 0 : 1;
}
