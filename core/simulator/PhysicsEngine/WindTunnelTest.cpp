/**
 * @file WindTunnelTest.cpp
 * @brief 风洞：抗风能力实测
 * @author GhostFace
 * @date 2026/9/18
 *
 * @par 这个测试要回答什么
 *
 * 此前所有仿真都在静止大气里进行。风是真实飞行中最主要的外部扰动，也是把
 * 「仿真里飞得好」与「真机上飞得住」区分开的东西。本测试给控制器加上风，
 * 并验证两条可以事先算出的解析预测。
 *
 * @par 两条解析预测
 *
 * 四旋翼靠倾斜产生水平力。在常值风下达到稳态时，飞行器静止悬停在某个偏移
 * 位置上，此时需要平衡风阻 `k·v²`：
 *
 * @verbatim
 *   倾角：  tanθ = k·v² / (m·g)        θ = atan(k·v²/(m·g))
 *   偏移：  kp·e = k·v² / m            e = k·v² / (m·kp)
 * @endverbatim
 *
 * 偏移的来源是定点控制律的固有特性：它只能靠位置误差产生加速度指令，所以要
 * 抵抗一个恒定的外力，就必须先偏离目标 —— 这和外力为零时的「零稳态误差」并
 * 不矛盾，后者只是前者的特例。
 *
 * @par 抗风上限
 *
 * 倾角限幅给出水平力的天花板 `tan(θmax)·m·g`，于是可悬停的最大风速是
 *
 * @verbatim
 *   v_max = sqrt( tan(θmax)·m·g / k )
 * @endverbatim
 *
 * 默认参数（m=1、θmax=35°、k=0.049）给出约 11.8 m/s。超过它无法悬停 ——
 * 不是控制器不够好，而是推力方向被倾角限幅卡死了。
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
              << "，预测 " << want << "，容差 " << rel_tol * 100 << "%）\n";
}

std::array<double, 3> readVec(const Tensor &t) {
    const std::vector<float> v = toVector(t);
    return {v[0], v[1], v[2]};
}

/// 机体 z 轴相对竖直方向的夹角（度）—— 即真实的物理倾角
double tiltDeg(const Tensor &quat) {
    const std::vector<float> q = toVector(quat);
    const double x = q[1], y = q[2];
    // 旋转矩阵的 R33 分量，即机体 z 轴在 NED 系中的 z 分量
    const double r33 = 1.0 - 2.0 * (x * x + y * y);
    const double c = std::max(-1.0, std::min(1.0, r33));
    return std::acos(c) * 180.0 / M_PI;
}

struct WindResult {
    double mean_err = 0.0;       ///< 稳态平均位置误差（米）
    double mean_tilt_deg = 0.0;  ///< 稳态平均倾角（度）
    double max_tilt_deg = 0.0;   ///< 全程最大倾角（度）
    std::array<double, 3> final_pos{};
    bool stable = true; ///< 稳态段位置是否仍有界（失稳判据见下）
};

/**
 * @brief 在给定风场下做定点悬停
 *
 * @param steady_from 稳态段起点（占总时长的比例）
 */
WindResult hoverInWind(const SixDofConfig &cfg, SixDofPidGains gains, WindModel &wind,
                       const std::array<double, 3> &target, double seconds,
                       double steady_from = 0.5) {
    const SixDofState init{makeVec3(static_cast<float>(target[0]),
                                    static_cast<float>(target[1]),
                                    static_cast<float>(target[2])),
                           makeVec3(0.0f, 0.0f, 0.0f), Tensor{1.0f, 0.0f, 0.0f, 0.0f},
                           makeVec3(0.0f, 0.0f, 0.0f)};
    SixDofSimulator sim(cfg, init);
    sim.setWind(&wind);
    wind.reset();
    SixDofPidController ctrl(cfg, gains);

    const double dt = cfg.base.dt;
    const int steps = static_cast<int>(seconds / dt);
    const int from = static_cast<int>(steady_from * steps);

    WindResult out;
    double err_sum = 0.0, tilt_sum = 0.0;
    int n = 0;

    for (int k = 0; k < steps; ++k) {
        const SixDofCommand cmd = ctrl.compute(
            sim.state(),
            makeVec3(static_cast<float>(target[0]), static_cast<float>(target[1]),
                     static_cast<float>(target[2])),
            static_cast<double>(k) * dt);
        sim.step(cmd.thrust_body, cmd.torque);

        const double tilt = tiltDeg(sim.state().quat);
        out.max_tilt_deg = std::max(out.max_tilt_deg, tilt);

        if (k >= from) {
            const std::array<double, 3> p = readVec(sim.state().pos);
            const double e = std::sqrt(std::pow(p[0] - target[0], 2) +
                                       std::pow(p[1] - target[1], 2) +
                                       std::pow(p[2] - target[2], 2));
            err_sum += e;
            tilt_sum += tilt;
            ++n;
            out.final_pos = p;
            // 被吹走超过 20 m 视为失稳（远大于任何稳态偏移）
            if (e > 20.0) {
                out.stable = false;
            }
        }
    }

    out.mean_err = err_sum / std::max(1, n);
    out.mean_tilt_deg = tilt_sum / std::max(1, n);
    return out;
}

} // namespace

int main() {
    CtorchError::setPrintLevel(PrintLevel::MINIUM);

    SixDofConfig cfg;
    cfg.base.dt = 0.001;
    cfg.base.mass = 1.0;
    cfg.base.gravity = 9.81;
    // 典型小型四旋翼的等效阻力：F = ½·ρ·Cd·A·v²，取 Cd·A ≈ 0.08 m²
    //   k = ½ · 1.225 · 0.08 ≈ 0.049 N·s²/m²
    // 这个值必须显式设置：drag_coeff 默认为 0（保证早期回归不受影响），
    // 而风正是通过相对气流产生阻力起作用的 —— 阻力为零时风吹不动飞行器。
    cfg.base.drag_coeff = 0.049;
    cfg.base.max_thrust = 0.0;
    cfg.inertia[0] = 0.015;
    cfg.inertia[1] = 0.018;
    cfg.inertia[2] = 0.028;
    cfg.torque_limit = 1.0;
    cfg.max_body_thrust = 20.0;

    SixDofPidGains gains;
    const double k_drag = cfg.base.drag_coeff;
    const double m = cfg.base.mass;
    const double g = cfg.base.gravity;
    const std::array<double, 3> target = {0.0, 0.0, -5.0};

    std::cout << "========================================\n";
    std::cout << "风洞：抗风能力实测\n";
    std::cout << "质量 " << m << " kg，阻力系数 " << k_drag << " N·s²/m²，倾角上限 "
              << gains.max_tilt_deg << " deg\n";
    std::cout << "理论抗风上限 sqrt(tan(θmax)·mg/k) = "
              << std::sqrt(std::tan(gains.max_tilt_deg * M_PI / 180.0) * m * g / k_drag)
              << " m/s\n";
    std::cout << "========================================\n";

    // ---- 1. 湍流模型自检：σ 参数是否真的等于输出标准差 ----
    std::cout << "\n[1] 湍流模型自检（σ 标定）\n";
    {
        const double sigma_set = 1.5;
        TurbulentWind tw(5.0, 0.0, sigma_set, 10.0, 0.001, 12345u);
        const int N = 60000;
        double sum[3] = {0.0, 0.0, 0.0}, sum2[3] = {0.0, 0.0, 0.0};
        for (int i = 0; i < N; ++i) {
            const WindVec w = tw.step();
            for (int a = 0; a < 3; ++a) {
                // 去掉 5 m/s 的背景风（只施加在 x 轴）
                const double dev = w[static_cast<std::size_t>(a)] -
                                   (a == 0 ? 5.0 : 0.0);
                sum[a] += dev;
                sum2[a] += dev * dev;
            }
        }
        double worst = 0.0;
        for (int a = 0; a < 3; ++a) {
            const double mean = sum[a] / N;
            const double var = sum2[a] / N - mean * mean;
            const double sd = std::sqrt(std::max(0.0, var));
            worst = std::max(worst, std::fabs(sd - sigma_set));
            std::cout << "    轴 " << a << "：实测标准差 " << std::fixed << std::setprecision(4)
                      << sd << " m/s（设定 " << sigma_set << "）\n";
        }
        checkTrue("湍流输出标准差与设定强度 σ 一致（0.15 m/s 以内）", worst < 0.15);
    }

    // ---- 2. 悬停抗风：风速扫描 + 两条解析预测 ----
    std::cout << "\n[2] 常值风下的定点悬停（风沿 +x 吹向，即吹向北）\n";
    std::cout << "  预测：倾角 θ = atan(k·v²/(mg))，位置偏移 e = k·v²/(m·kp)\n\n";
    std::cout << "  抗风上限由两个约束中更紧的那个决定：\n";
    std::cout << "    (a) 期望加速度限幅 max_accel ≥ k·v²/m    => v ≤ "
              << std::sqrt(gains.max_accel * m / k_drag) << " m/s\n";
    const double cos_t = m * g / cfg.max_body_thrust;
    std::cout << "    (b) 竖直推力 tan θ · mg ≤ ...            => v ≤ "
              << std::sqrt(std::sqrt(1.0 - cos_t * cos_t) / cos_t * m * g / k_drag)
              << " m/s（推力上限 " << cfg.max_body_thrust << " N）\n\n";
    std::cout << "  " << std::setw(10) << "风速(m/s)" << std::setw(14) << "稳态误差"
              << std::setw(14) << "预测偏移" << std::setw(14) << "稳态倾角"
              << std::setw(14) << "预测倾角" << std::setw(10) << "状态" << "\n";

    // 上限由期望加速度限幅决定（15.65 m/s），比推力上限（18.9 m/s）更紧
    const double v_max = std::sqrt(gains.max_accel * m / k_drag);
    std::array<double, 8> winds = {2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0};
    std::array<double, 8> got_err{}, got_tilt{}, pred_err{}, pred_tilt{};
    for (int i = 0; i < 8; ++i) {
        const double v = winds[static_cast<std::size_t>(i)];
        SteadyWind wind(v, 0.0);
        const WindResult r = hoverInWind(cfg, gains, wind, target, 12.0);

        const double f_wind = k_drag * v * v;
        pred_err[static_cast<std::size_t>(i)] = f_wind / (m * gains.pos_kp);
        pred_tilt[static_cast<std::size_t>(i)] =
            std::atan(f_wind / (m * g)) * 180.0 / M_PI;
        got_err[static_cast<std::size_t>(i)] = r.mean_err;
        got_tilt[static_cast<std::size_t>(i)] = r.mean_tilt_deg;

        std::cout << "  " << std::setw(10) << std::fixed << std::setprecision(1) << v
                  << std::setw(14) << std::setprecision(4) << r.mean_err << std::setw(14)
                  << pred_err[static_cast<std::size_t>(i)] << std::setw(14)
                  << r.mean_tilt_deg << std::setw(14) << pred_tilt[static_cast<std::size_t>(i)]
                  << std::setw(10) << (r.stable ? "稳定" : "失稳") << "\n";
    }

    std::cout << "\n  由期望加速度限幅决定的临界风速 = " << std::setprecision(2) << v_max
              << " m/s\n";
    // 极限以内的六个点，两条解析式都应成立
    for (int i = 0; i < 6; ++i) {
        checkNear(std::string("风速 " + std::to_string(static_cast<int>(
                                         winds[static_cast<std::size_t>(i)])) +
                                  " m/s 的稳态偏移与预测 k·v²/(m·kp) 相符")
                      .c_str(),
                  got_err[static_cast<std::size_t>(i)],
                  pred_err[static_cast<std::size_t>(i)], 0.15);
    }
    for (int i = 0; i < 6; ++i) {
        checkNear(std::string("风速 " + std::to_string(static_cast<int>(
                                         winds[static_cast<std::size_t>(i)])) +
                                  " m/s 的稳态倾角与预测 atan(k·v²/mg) 相符")
                      .c_str(),
                  got_tilt[static_cast<std::size_t>(i)],
                  pred_tilt[static_cast<std::size_t>(i)], 0.20);
    }
    // 超过加速度限幅后，控制律给出的水平指令被截断，无法平衡风阻 -> 被吹走
    checkTrue("风速 16 m/s（超过加速度限幅决定的 15.65 m/s）时无法维持定点",
              !(std::fabs(got_err[7] - pred_err[7]) < 0.15 * pred_err[7]));

    // ---- 3. 阵风响应 ----
    std::cout << "\n[3] 阵风响应（背景风 5 m/s + 阵风 A=8 m/s，3s 起持续 2s）\n";
    {
        GustWind wind(5.0, 0.0, 8.0, 0.0, 3.0, 2.0);

        // 阵风需要峰值时刻的瞬时量，因此直接跑时间序列而不复用稳态统计算法
        const SixDofState init{makeVec3(0.0f, 0.0f, -5.0f), makeVec3(0.0f, 0.0f, 0.0f),
                               Tensor{1.0f, 0.0f, 0.0f, 0.0f}, makeVec3(0.0f, 0.0f, 0.0f)};
        SixDofSimulator sim(cfg, init);
        sim.setWind(&wind);
        wind.reset();
        SixDofPidController ctrl(cfg, gains);
        const double dt = cfg.base.dt;
        double peak_err = 0.0, peak_tilt = 0.0;
        double at_peak = 0.0;
        for (int k = 0; k < static_cast<int>(10.0 / dt); ++k) {
            const SixDofCommand cmd = ctrl.compute(
                sim.state(), makeVec3(0.0f, 0.0f, -5.0f), static_cast<double>(k) * dt);
            sim.step(cmd.thrust_body, cmd.torque);
            const std::array<double, 3> p = readVec(sim.state().pos);
            const double e = std::sqrt(p[0] * p[0] + p[1] * p[1] + (p[2] + 5.0) * (p[2] + 5.0));
            if (e > peak_err) {
                peak_err = e;
                at_peak = static_cast<double>(k) * dt;
            }
            peak_tilt = std::max(peak_tilt, tiltDeg(sim.state().quat));
        }
        const std::array<double, 3> p_end = readVec(sim.state().pos);
        const double final_err =
            std::sqrt(p_end[0] * p_end[0] + p_end[1] * p_end[1] + (p_end[2] + 5.0) * (p_end[2] + 5.0));

        std::cout << "    峰值偏移 " << std::setprecision(4) << peak_err << " m（t = "
                  << at_peak << " s），最大倾角 " << peak_tilt << " deg\n";
        std::cout << "    阵风结束后末端偏移 " << final_err << " m\n";
        checkTrue("阵风引起的峰值偏移被限制在 2 m 以内（阵风幅度 8 m/s）", peak_err < 2.0);
        checkTrue("阵风过后位置恢复（末端偏移小于峰值的一半）", final_err < 0.5 * peak_err);
    }

    // ---- 4. 湍流下的悬停性能 ----
    std::cout << "\n[4] 湍流下的悬停（背景风 5 m/s，对比不同湍流强度）\n";
    std::cout << "  " << std::setw(12) << "湍流σ(m/s)" << std::setw(16) << "稳态误差RMS"
              << std::setw(16) << "稳态误差峰值" << std::setw(14) << "平均倾角" << "\n";
    {
        const double sigmas[3] = {0.0, 1.0, 2.0};
        double rms[3] = {};
        for (int i = 0; i < 3; ++i) {
            TurbulentWind wind(5.0, 0.0, sigmas[i], 10.0, cfg.base.dt, 999u);
            // 用平均值与 RMS 分开统计，才能看出湍流带来的是「平均偏移」还是「抖动」
            const SixDofState init{makeVec3(0.0f, 0.0f, -5.0f), makeVec3(0.0f, 0.0f, 0.0f),
                                   Tensor{1.0f, 0.0f, 0.0f, 0.0f}, makeVec3(0.0f, 0.0f, 0.0f)};
            SixDofSimulator sim(cfg, init);
            sim.setWind(&wind);
            wind.reset();
            SixDofPidController ctrl(cfg, gains);
            const double dt = cfg.base.dt;
            const int steps = static_cast<int>(20.0 / dt);
            double e2 = 0.0, peak = 0.0, tilt_sum = 0.0;
            int n = 0;
            for (int k = 0; k < steps; ++k) {
                const SixDofCommand cmd = ctrl.compute(
                    sim.state(), makeVec3(0.0f, 0.0f, -5.0f), static_cast<double>(k) * dt);
                sim.step(cmd.thrust_body, cmd.torque);
                if (k >= steps / 2) {
                    const std::array<double, 3> p = readVec(sim.state().pos);
                    const double e =
                        std::sqrt(p[0] * p[0] + p[1] * p[1] + (p[2] + 5.0) * (p[2] + 5.0));
                    e2 += e * e;
                    peak = std::max(peak, e);
                    tilt_sum += tiltDeg(sim.state().quat);
                    ++n;
                }
            }
            rms[i] = std::sqrt(e2 / std::max(1, n));
            std::cout << "  " << std::setw(12) << std::fixed << std::setprecision(1) << sigmas[i]
                      << std::setw(16) << std::setprecision(4) << rms[i] << std::setw(16)
                      << peak << std::setw(14) << (tilt_sum / std::max(1, n)) << "\n";
        }
        checkTrue("湍流强度增大使跟踪误差 RMS 单调增大", rms[2] > rms[1] && rms[1] >= rms[0]);
        checkTrue("强湍流（σ=2 m/s）下仍保持稳定（RMS 小于 3 m）", rms[2] < 3.0);
    }

    std::cout << "\n[结论]\n";
    std::cout << "  1. 常值风下定点控制律的稳态偏移与倾角均可由解析式预测并实测验证，\n";
    std::cout << "     说明风模型与气动力接入是物理自洽的。\n";
    std::cout << "  2. 抗风上限由期望加速度限幅决定（本配置约 " << std::setprecision(1) << v_max
              << " m/s），而非 max_tilt_deg：\n";
    std::cout << "     后者限的是姿态指令的激进程度，稳态下不生效。机体倾角可以远超 35°\n";
    std::cout << "     （实测 12 m/s 风下稳定倾斜 35.7°），真正的天花板来自加速度与推力限幅。\n";
    std::cout << "  3. 风的存在把「零稳态误差」变成了「与外力成正比的稳态偏移」。\n";
    std::cout << "     定点控制律只能靠位置误差产生抵抗外力的加速度，这是它的结构性特征。\n";

    std::cout << "\n========================================\n";
    std::cout << g_checks - g_failed << " / " << g_checks << " checks passed\n";
    if (g_failed > 0) {
        std::cout << g_failed << " FAILED\n";
    }
    std::cout << "========================================\n";
    return g_failed == 0 ? 0 : 1;
}
