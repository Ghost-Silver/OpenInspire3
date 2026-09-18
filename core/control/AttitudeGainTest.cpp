/**
 * @file AttitudeGainTest.cpp
 * @brief 姿态增益该由惯量推导 —— 以及为什么惯量标定得准很重要
 * @author GhostFace
 * @date 2026/9/18
 *
 * @par 三层判据
 *
 * 1. **三轴一致性（数学层）**：姿态环的量纲是「力矩 → 角加速度」，而角加速度 = τ/I，
 *    所以增益必须随惯量缩放，三轴才能有相同的闭环带宽与阻尼比。用一组标量增益套
 *    三轴时，只要三轴惯量不等，闭环特性就不同 —— 这一层可以纯解析地验证。
 *
 * 2. **惯量误差会传导到增益上（物理层）**：推导增益需要惯量，而惯量本身是辨识出来的、
 *    带残差的。用不准确的惯量去推导，得到的就是不准确的增益 —— 这正是参数辨识的
 *    价值所在。
 *
 * 3. **闭环行为（系统层）**：上面两层最终要落到飞得好不好。用悬停任务测收敛时间、
 *    稳态误差与三轴姿态残差，对比「手填惯量推导」「校准惯量推导」两种配置。
 *
 * 判据不设「谁更快」的胜负断言（那取决于任务），只断言：
 *   - 推导模式下三轴闭环特性一致；
 *   - 用校准惯量推导的增益，其闭环表现优于用手填惯量推导的。
 */

#include "SixDofDynamics.h"
#include "SixDofPidController.h"
#include "SixDofSimulator.h"
#include "TensorUtils.h"

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

void checkNear(const char *name, double got, double want, double tol) {
    ++g_checks;
    const bool ok = std::isfinite(got) && std::fabs(got - want) <= tol;
    if (!ok) {
        ++g_failed;
    }
    std::cout << "  [" << (ok ? " ok " : "FAIL") << "] " << name << "（got " << got
              << "，want " << want << "）\n";
}

/// 由增益与惯量反推闭环自然频率与阻尼比
struct ClosedLoop {
    double wn = 0.0;
    double zeta = 0.0;
};

ClosedLoop analyze(double kp, double kd, double inertia) {
    ClosedLoop c;
    const double I = std::max(1e-9, inertia);
    c.wn = std::sqrt(std::max(0.0, kp / I));
    c.zeta = (c.wn > 0.0) ? kd / (2.0 * std::sqrt(std::max(1e-12, kp * I))) : 0.0;
    return c;
}

/// 悬停闭环的一次运行统计
struct HoverRun {
    double settle_time = -1.0;
    double final_pos_err = 0.0;
    double max_att_err_deg = 0.0;
    double att_err_rms_deg = 0.0;
};

HoverRun runHover(const SixDofConfig &sim_cfg, const SixDofPidGains &gains,
                  double seconds = 4.0) {
    SixDofConfig cfg = sim_cfg;
    cfg.base.dt = 0.001;

    const std::array<double, 3> target = {0.8, -0.4, -5.0};
    SixDofSimulator sim(cfg, SixDofState{
                                 makeVec3(0.0f, 0.0f, -5.0f),
                                 makeVec3(0.0f, 0.0f, 0.0f),
                                 Tensor{1.0f, 0.0f, 0.0f, 0.0f},
                                 makeVec3(0.0f, 0.0f, 0.0f)});
    SixDofPidController ctrl(cfg, gains);

    const double dt = cfg.base.dt;
    const int steps = static_cast<int>(seconds / dt);
    const double tol = 0.05;

    HoverRun out;
    int hold = 0;
    double att_err2_sum = 0.0;
    int att_err_n = 0;

    for (int k = 0; k < steps; ++k) {
        const SixDofState st = sim.state();
        const SixDofCommand cmd = ctrl.compute(st, makeVec3(static_cast<float>(target[0]),
                                                            static_cast<float>(target[1]),
                                                            static_cast<float>(target[2])),
                                              static_cast<double>(k) * dt);
        sim.step(cmd.thrust_body, cmd.torque);

        const std::vector<float> pos = toVector(sim.state().pos);
        const double err = std::sqrt(
            std::pow(static_cast<double>(pos[0]) - target[0], 2) +
            std::pow(static_cast<double>(pos[1]) - target[1], 2) +
            std::pow(static_cast<double>(pos[2]) - target[2], 2));

        // 姿态误差：四元数 w 分量偏离 1 的角度
        const std::vector<float> q = toVector(sim.state().quat);
        const double w = std::max(-1.0, std::min(1.0, static_cast<double>(q[0])));
        const double att_deg = 2.0 * std::acos(std::fabs(w)) * 180.0 / M_PI;
        out.max_att_err_deg = std::max(out.max_att_err_deg, att_deg);
        att_err2_sum += att_deg * att_deg;
        ++att_err_n;

        if (err < tol) {
            if (++hold >= 500 && out.settle_time < 0.0) {
                out.settle_time = static_cast<double>(k) * dt;
            }
        } else {
            hold = 0;
        }
        out.final_pos_err = err;
    }
    out.att_err_rms_deg =
        std::sqrt(att_err2_sum / std::max(1, att_err_n));
    return out;
}

} // namespace

int main() {
    CtorchError::setPrintLevel(PrintLevel::MINIUM);

    // 真值惯量（模拟真机；即 ParameterIdTest 的辨识目标）
    const double truth[3] = {0.015, 0.018, 0.028};
    // 手填典型值（OI3 原默认）
    const double handfilled[3] = {0.01, 0.01, 0.02};

    std::cout << "========================================\n";
    std::cout << "姿态增益：由惯量推导，以及惯量标定的价值\n";
    std::cout << "========================================\n";

    // ---- 1. 三轴一致性（数学层）----
    std::cout << "\n[1] 三轴闭环特性（ωn rad/s / ζ）\n";

    SixDofPidGains manual;
    manual.derive_attitude_from_inertia = false;
    manual.att_kp = 0.9;
    manual.att_kd = 0.25;
    {
        std::cout << "  标量增益（旧默认 kp=0.9, kd=0.25）：\n";
        double wn_min = 1e9, wn_max = -1e9, zeta_min = 1e9, zeta_max = -1e9;
        for (int i = 0; i < 3; ++i) {
            const ClosedLoop c = analyze(manual.att_kp, manual.att_kd, truth[i]);
            wn_min = std::min(wn_min, c.wn);
            wn_max = std::max(wn_max, c.wn);
            zeta_min = std::min(zeta_min, c.zeta);
            zeta_max = std::max(zeta_max, c.zeta);
            std::cout << "    轴 " << i << "（I=" << truth[i] << "）：ωn=" << c.wn
                      << "  ζ=" << c.zeta << "\n";
        }
        std::cout << "    ωn 极差 " << (wn_max - wn_min) << "，ζ 极差 "
                  << (zeta_max - zeta_min) << "\n";
        checkTrue("标量增益下三轴闭环特性明显不一致（ωn 极差 > 2 rad/s）",
                  (wn_max - wn_min) > 2.0);
    }

    SixDofPidGains derived;
    derived.derive_attitude_from_inertia = true;
    derived.att_bandwidth = 9.0;
    derived.att_damping = 1.0;
    {
        std::cout << "  按惯量推导（ωn=9.0, ζ=1.0）：\n";
        double wn_min = 1e9, wn_max = -1e9, zeta_min = 1e9, zeta_max = -1e9;
        for (int i = 0; i < 3; ++i) {
            const double I = truth[i];
            const ClosedLoop c = analyze(I * 9.0 * 9.0, 2.0 * 1.0 * I * 9.0, I);
            wn_min = std::min(wn_min, c.wn);
            wn_max = std::max(wn_max, c.wn);
            zeta_min = std::min(zeta_min, c.zeta);
            zeta_max = std::max(zeta_max, c.zeta);
            std::cout << "    轴 " << i << "（I=" << I << "，kp=" << (I * 81.0)
                      << "，kd=" << (2.0 * I * 9.0) << "）：ωn=" << c.wn << "  ζ=" << c.zeta
                      << "\n";
        }
        checkNear("推导模式三轴带宽完全一致", wn_max - wn_min, 0.0, 1e-9);
        checkNear("推导模式三轴阻尼比完全一致", zeta_max - zeta_min, 0.0, 1e-9);
    }

    // ---- 2. 惯量误差 → 增益误差（物理层）----
    std::cout << "\n[2] 惯量不准会直接变成增益不准\n";
    {
        double worst_rel = 0.0;
        for (int i = 0; i < 3; ++i) {
            const double kp_hand = handfilled[i] * 81.0;
            const double kp_true = truth[i] * 81.0;
            const double rel = std::fabs(kp_hand - kp_true) / kp_true;
            worst_rel = std::max(worst_rel, rel);
            std::cout << "    轴 " << i << " 的 kp：手填惯量给出 " << kp_hand
                      << "，校准惯量给出 " << kp_true << "（相对差 " << rel << "）\n";
        }
        checkTrue("惯量误差直接放大为同量级的增益误差（> 20%）", worst_rel > 0.20);
    }

    // ---- 3. 闭环行为（系统层）----
    std::cout << "\n[3] 悬停任务闭环（仿真用真值惯量，控制器分别用两种惯量推导增益）\n";

    SixDofConfig sim_cfg;
    sim_cfg.base.dt = 0.001;
    sim_cfg.base.mass = 1.0;
    sim_cfg.base.gravity = 9.81;
    sim_cfg.base.drag_coeff = 0.0;
    sim_cfg.torque_limit = 1.0;
    sim_cfg.max_body_thrust = 20.0;

    // 控制器 A：用手填惯量推导（惯量不准）
    SixDofConfig cfg_hand = sim_cfg;
    cfg_hand.inertia[0] = handfilled[0];
    cfg_hand.inertia[1] = handfilled[1];
    cfg_hand.inertia[2] = handfilled[2];

    // 控制器 B：用真值惯量推导（= 校准后的结果）
    SixDofConfig cfg_calib = sim_cfg;
    cfg_calib.inertia[0] = truth[0];
    cfg_calib.inertia[1] = truth[1];
    cfg_calib.inertia[2] = truth[2];

    // 仿真本身始终用真值惯量（模拟真机）
    SixDofConfig sim_true = sim_cfg;
    sim_true.inertia[0] = truth[0];
    sim_true.inertia[1] = truth[1];
    sim_true.inertia[2] = truth[2];

    // 注意：runHover 内部把 cfg 同时用于仿真与控制器，这里需要分别传入 ——
    // 因此拆成两次调用：仿真用 sim_true，控制器用各自的 cfg。
    // 为保持接口简单，改为在 runHover 之外构造：见下方 lambda。
    auto runWithControllerConfig = [&](const SixDofConfig &ctrl_cfg,
                                       const SixDofPidGains &gains) {
        const std::array<double, 3> target = {0.8, -0.4, -5.0};
        SixDofSimulator sim(sim_true, SixDofState{
                                          makeVec3(0.0f, 0.0f, -5.0f),
                                          makeVec3(0.0f, 0.0f, 0.0f),
                                          Tensor{1.0f, 0.0f, 0.0f, 0.0f},
                                          makeVec3(0.0f, 0.0f, 0.0f)});
        SixDofPidController ctrl(ctrl_cfg, gains);

        const double dt = sim_true.base.dt;
        const int steps = static_cast<int>(4.0 / dt);
        HoverRun out;
        int hold = 0;
        double att_err2 = 0.0;
        int n_att = 0;
        for (int k = 0; k < steps; ++k) {
            const SixDofCommand cmd =
                ctrl.compute(sim.state(),
                             makeVec3(static_cast<float>(target[0]),
                                      static_cast<float>(target[1]),
                                      static_cast<float>(target[2])),
                             static_cast<double>(k) * dt);
            sim.step(cmd.thrust_body, cmd.torque);

            const std::vector<float> pos = toVector(sim.state().pos);
            const double err = std::sqrt(
                std::pow(static_cast<double>(pos[0]) - target[0], 2) +
                std::pow(static_cast<double>(pos[1]) - target[1], 2) +
                std::pow(static_cast<double>(pos[2]) - target[2], 2));
            const std::vector<float> q = toVector(sim.state().quat);
            const double w = std::max(-1.0, std::min(1.0, static_cast<double>(q[0])));
            const double att_deg = 2.0 * std::acos(std::fabs(w)) * 180.0 / M_PI;
            out.max_att_err_deg = std::max(out.max_att_err_deg, att_deg);
            att_err2 += att_deg * att_deg;
            ++n_att;
            if (err < 0.05) {
                if (++hold >= 500 && out.settle_time < 0.0) {
                    out.settle_time = static_cast<double>(k) * dt;
                }
            } else {
                hold = 0;
            }
            out.final_pos_err = err;
        }
        out.att_err_rms_deg = std::sqrt(att_err2 / std::max(1, n_att));
        return out;
    };

    const HoverRun a = runWithControllerConfig(cfg_hand, derived);
    const HoverRun b = runWithControllerConfig(cfg_calib, derived);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  " << std::left << std::setw(26) << "配置" << std::setw(14) << "收敛时间"
              << std::setw(14) << "末端误差" << std::setw(16) << "姿态 RMS(deg)"
              << std::setw(16) << "最大姿态(deg)" << "\n";
    std::cout << "  " << std::setw(26) << "手填惯量推导增益" << std::setw(14) << a.settle_time
              << std::setw(14) << a.final_pos_err << std::setw(16) << a.att_err_rms_deg
              << std::setw(16) << a.max_att_err_deg << "\n";
    std::cout << "  " << std::setw(26) << "校准惯量推导增益" << std::setw(14) << b.settle_time
              << std::setw(14) << b.final_pos_err << std::setw(16) << b.att_err_rms_deg
              << std::setw(16) << b.max_att_err_deg << "\n";

    checkTrue("两种配置都能稳定收敛到目标", a.settle_time > 0.0 && b.settle_time > 0.0);

    // 判据要针对**被评对象**：本次改的是姿态环增益，因此用姿态残差衡量它。
    // 位置收敛主要由位置环增益决定（本次未改动），两者只应处于同一量级 ——
    // 拿位置指标去判姿态增益会引入归因错误，这与前面「用闭环性能评模型精度」
    // 是同一类错误（见 ModelCalibrationTest 的说明）。
    checkTrue("校准惯量使姿态 RMS 残差更小", b.att_err_rms_deg < a.att_err_rms_deg);
    checkTrue("校准惯量使姿态峰值更小", b.max_att_err_deg < a.max_att_err_deg);
    checkTrue("位置收敛处于同一量级（差异 < 20%）",
              std::fabs(b.settle_time - a.settle_time) < 0.2 * a.settle_time);

    std::cout << "\n[结论]\n";
    std::cout << "  姿态增益必须随惯量缩放 —— 标量增益套三轴会让闭环带宽与阻尼比\n";
    std::cout << "  各不相同（本例 ωn 相差 "
              << (analyze(manual.att_kp, manual.att_kd, truth[0]).wn -
                  analyze(manual.att_kp, manual.att_kd, truth[2]).wn)
              << " rad/s），三轴响应不同步。\n";
    std::cout << "  而推导增益依赖惯量的准确性 —— 惯量差 33%~50% 会原样放大成增益误差，\n";
    std::cout << "  这正是 ParameterIdentification 的价值所在：惯量准了，增益才是算出来的。\n";

    std::cout << "\n========================================\n";
    std::cout << g_checks - g_failed << " / " << g_checks << " checks passed\n";
    if (g_failed > 0) {
        std::cout << g_failed << " FAILED\n";
    }
    std::cout << "========================================\n";
    return g_failed == 0 ? 0 : 1;
}
