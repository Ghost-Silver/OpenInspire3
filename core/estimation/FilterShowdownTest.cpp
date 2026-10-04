/**
 * @file FilterShowdownTest.cpp
 * @brief 位置滤波器对决：α-β vs 死区补丁 vs 卡尔曼（切换默认值的证据基础）
 *
 * @par 目的
 *
 * 路线图项「位置滤波器升级」的决策依据。本测试把判决固化为可复现的断言，
 * 避免将来凭印象讨论「要不要切」。
 *
 * @par 判决（悬停稳态 RMS 误差，对真值）
 *
 * | 场景 | α-β（原默认） | α-β+死区 0.15 | 卡尔曼 |
 * |---|---|---|---|
 * | 理想量测 | 0.0229 m | 0.2884 m | 0.0234 m |
 * | 光流级 σ=0.05 | 0.0725 m | 1.2140 m | **0.0509 m** |
 * | 中档 σ=0.15 | 0.2904 m | 0.7080 m | **0.1259 m** |
 * | GPS 级 σ=0.30 | 0.4677 m | 0.6141 m | **0.3072 m** |
 * | 廉价 GPS σ=0.60 | 0.8818 m | 0.9657 m | **0.8072 m** |
 * | 延迟 100 ms（σ=0.05） | 0.1016 m | 1.6848 m | **0.0613 m** |
 * | 机动 + GPS 级 | 1.8098 m | 2.0425 m | **1.0374 m** |
 *
 * 结论：**卡尔曼在噪声、延迟、机动三个维度上全面优于 α-β，且随工况变差而
 * 优势扩大**（机动 + GPS 级好 74%）。唯一持平的是理想量测场景（此时误差由
 * 控制滞后主导，滤波器差异被淹没）。
 *
 * @par 死区补丁的判死刑
 *
 * 它在**小噪声**场景把误差从 0.0725 m 放大到 **1.2140 m（17 倍）** —— 固定阈值
 * 对小噪声属过度抑制。该方案在所有场景中都是三者最差，不应再作为候选。
 *
 * @par 关于 kalman_accel_noise（Q）
 *
 * 追踪中发现 Q 的最优值**同时依赖噪声水平与延迟**（σ=0.30 时：延迟 0 → Q≈200，
 * 延迟 150 → Q≈400），即 Q 在为「模型不确定度」与「未建模延迟」两个效应同时
 * 服务。实测 Q 在 100~400 区间内差异不显著（远小于滤波器选型带来的差异），
 * 故当前保持默认 Q=100 并不追求「最优 Q」—— 那属于独立的调参议题。
 *
 * @par 一处方法学记录
 *
 * 本测试的初版曾得出「卡尔曼在极端工况下输给 α-β」的结论，后查明是探针缺陷：
 * 两行「不同配置」实际执行同一份代码（Q 参数未真正区分），导致对照失效。
 * 修正后结论反转。提醒：对照实验中「数值逐位相同」是可疑信号，必须追查。
 */#include "FlightControlLoop.h"
#include "MinimumSnapTrajectory.h"
#include "HalSimulator.h"
#include "ImuModel.h"
#include "SixDofPidController.h"
#include "SixDofSimulator.h"
#include "TensorUtils.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace oi3;

namespace {
int g_pass = 0;
int g_fail = 0;

void check(bool cond, const std::string &name) {
    if (cond) {
        std::printf("[ ok ] %s\n", name.c_str());
        ++g_pass;
    } else {
        std::printf("[FAIL] %s\n", name.c_str());
        ++g_fail;
    }
}

struct ShowdownCfg {
    const char *name;
    PosFilterKind kind;
    double deadzone;
    double q = 100.0; ///< <0 表示按噪声自适应
};

const ShowdownCfg kConfigs[3] = {
    {"α-β  (默认)", PosFilterKind::AlphaBeta, 0.0, 100.0},
    {"α-β+死区0.15", PosFilterKind::AlphaBeta, 0.15, 100.0},
    {"卡尔曼 Q=100", PosFilterKind::Kalman, 0.0, 100.0},
};

/// 自适应 Q：按实测规律 Q ≈ 1000·σ（σ 小时退回默认下限 100）
double adaptiveQ(double sigma) {
    return sigma > 0.0 ? std::max(100.0, 1000.0 * sigma) : 100.0;
}

struct Result {
    double final_err = 0.0;   ///< 末态三维误差
    double rms_err = 0.0;     ///< 全程位置误差 RMS（对真值）
    double max_err = 0.0;     ///< 最大位置误差
    bool crashed = false;
    int n = 0;
};

std::array<double, 3> rv(const Tensor &t) {
    const auto v = toVector(t);
    return {v[0], v[1], v[2]};
}

/**
 * @param sigma       位置量测噪声
 * @param delay       位置量测延迟（IMU 步）
 * @param maneuver    true 则做机动（目标跳变），false 则纯悬停
 * @param steps       步数
 */
Result run(const ShowdownCfg &cfg, double sigma, int delay, bool maneuver, int steps = 8000) {
    SixDofConfig sc;
    SixDofState init{makeVec3(0.0f, 0.0f, -5.0f), makeVec3(0.0f, 0.0f, 0.0f),
                     Tensor{1.0f, 0.0f, 0.0f, 0.0f}, makeVec3(0.0f, 0.0f, 0.0f)};
    SixDofSimulator sim(sc, init);
    ImuConfig ic;
    ic.explicit_bias = true;
    ImuModel imu(ic, 20260918u);

    PositionImperfection imp;
    imp.sigma = sigma;
    imp.delay_steps = delay;
    SimSensorReader sensors(&sim, &imu, {0.0, 0.0, sc.base.gravity}, 10, imp);
    SimActuatorWriter actuators(&sim);

    struct Staged : public HalSetpointSource {
        bool man;
        explicit Staged(bool m) : man(m) {}
        Tensor currentTarget(double t) override {
            if (!man || t < 3.0) {
                return makeVec3(0.0f, 0.0f, -5.0f);
            }
            return makeVec3(8.0f, 0.0f, -8.0f);
        }
        [[nodiscard]] bool hasArrived(const std::array<double, 3> &, double) const override {
            return false;
        }
    } src(maneuver);

    SixDofPidController ctrl(sc, {});
    FlightControlConfig fc;
    fc.estimator.sensor_health.enabled = true;
    fc.estimator.pos_filter = cfg.kind;
    fc.estimator.pos_residual_deadzone = cfg.deadzone;
    if (cfg.kind == PosFilterKind::Kalman && sigma > 0.0) {
        fc.estimator.kalman_pos_noise = sigma;
        fc.estimator.kalman_accel_noise = cfg.q < 0.0 ? adaptiveQ(sigma) : cfg.q;
    }
    fc.hover_thrust = static_cast<double>(sc.base.mass) * static_cast<double>(sc.base.gravity);

    FlightControlLoop loop(fc, &sensors, &actuators, &src, &ctrl);
    loop.init();

    Result r;
    double sum_sq = 0.0;
    for (int k = 0; k < steps; ++k) {
        if (!loop.runOneCycle()) {
            break;
        }
        const auto p = rv(sim.state().pos);
        // 真值目标（用仿真器时间判断阶段）
        const double tt = sim.time();
        const double tgt_z = (maneuver && tt >= 3.0) ? -8.0 : -5.0;
        const double tgt_x = (maneuver && tt >= 3.0) ? 8.0 : 0.0;
        const double ex = p[0] - tgt_x;
        const double ey = p[1];
        const double ez = p[2] - tgt_z;
        // 只统计稳态（跳变后 1.5 s 起），避免把机动过程算作误差
        if (tt > (maneuver ? 4.5 : 2.0)) {
            const double e2 = ex * ex + ey * ey + ez * ez;
            sum_sq += e2;
            r.max_err = std::max(r.max_err, std::sqrt(e2));
            ++r.n;
        }
        r.final_err = std::sqrt(ex * ex + ey * ey + ez * ez);
        const double alt = -p[2];
        if (alt <= 0.0) {
            r.crashed = true;
            break;
        }
    }
    r.rms_err = r.n > 0 ? std::sqrt(sum_sq / static_cast<double>(r.n)) : 0.0;
    return r;
}

/// 用与 showdown 完全相同的 run() 扫 Q —— 保证场景、统计窗口一致
void qSweep(double sigma, int delay) {
    const double qs[7] = {50.0, 100.0, 200.0, 400.0, 800.0, 1600.0, 3200.0};
    double best_rms = 1e18, best_q = 0.0;
    std::printf("  σ=%.2f 延迟=%dms 下扫描 Q（同一场景与统计窗口）：\n", sigma, delay);
    for (double q : qs) {
        ShowdownCfg c{"kf", PosFilterKind::Kalman, 0.0, q};
        const Result r = run(c, sigma, delay, false);
        std::printf("    Q=%-8.0f RMS %.4f m   峰值 %.4f m\n", q, r.rms_err, r.max_err);
        if (r.rms_err < best_rms) {
            best_rms = r.rms_err;
            best_q = q;
        }
    }
    std::printf("    => 最优 Q=%.0f（RMS %.4f），Q/σ=%.0f\n\n", best_q, best_rms,
                sigma > 0 ? best_q / sigma : 0.0);
}

void showdown(const char *title, double sigma, int delay, bool maneuver) {
    std::printf("%s\n", title);
    std::printf("  %-16s %12s %12s %12s %8s\n", "配置", "稳态 RMS", "最大误差", "末态误差", "坠地");
    for (const auto &c : kConfigs) {
        const Result r = run(c, sigma, delay, maneuver);
        std::printf("  %-16s %11.4f m %11.4f m %11.4f m %8s\n", c.name, r.rms_err, r.max_err,
                    r.final_err, r.crashed ? "是" : "否");
    }
    {
        ShowdownCfg ac{"卡尔曼 自适应Q", PosFilterKind::Kalman, 0.0, -1.0};
        const Result r = run(ac, sigma, delay, maneuver);
        std::printf("  %-16s %11.4f m %11.4f m %11.4f m %8s\n", ac.name, r.rms_err, r.max_err,
                    r.final_err, r.crashed ? "是" : "否");
    }
    std::printf("\n");
}

} // namespace

int main() {
    std::printf("=== FilterShowdownTest: 位置滤波器对决 ===\n");
    std::printf("（度量：对真值的稳态位置误差 RMS）\n\n");

    const ShowdownCfg ab{"ab", PosFilterKind::AlphaBeta, 0.0, 100.0};
    const ShowdownCfg dz{"dz", PosFilterKind::AlphaBeta, 0.15, 100.0};
    const ShowdownCfg kf{"kf", PosFilterKind::Kalman, 0.0, 100.0};

    struct Case {
        const char *label;
        double sigma;
        int delay;
        bool maneuver;
    };
    const Case cases[7] = {
        {"理想量测", 0.0, 0, false},
        {"光流级 σ=0.05", 0.05, 0, false},
        {"中档 σ=0.15", 0.15, 0, false},
        {"GPS级 σ=0.30", 0.30, 0, false},
        {"廉价GPS σ=0.60", 0.60, 0, false},
        {"延迟100ms σ=0.05", 0.05, 100, false},
        {"机动+GPS级 σ=0.30", 0.30, 20, true},
    };

    double ab_rms[7], dz_rms[7], kf_rms[7];
    std::printf("%-20s %12s %12s %12s %10s\n", "场景", "α-β", "α-β+死区", "卡尔曼", "卡尔曼优势");
    for (int i = 0; i < 7; ++i) {
        const Case &c = cases[i];
        const Result ra = run(ab, c.sigma, c.delay, c.maneuver);
        const Result rd = run(dz, c.sigma, c.delay, c.maneuver);
        const Result rk = run(kf, c.sigma, c.delay, c.maneuver);
        ab_rms[i] = ra.rms_err;
        dz_rms[i] = rd.rms_err;
        kf_rms[i] = rk.rms_err;
        std::printf("%-20s %11.4f m %11.4f m %11.4f m %9.2fx\n", c.label, ra.rms_err,
                    rd.rms_err, rk.rms_err, rk.rms_err > 0 ? ra.rms_err / rk.rms_err : 0.0);
    }
    std::printf("\n");

    // ---- 1. 激励确认：这些场景确实存在量测噪声/延迟 ----
    check(ab_rms[1] > ab_rms[0] * 1.5,
          "激励确认：噪声确实恶化了 α-β 的表现（σ=0.05 vs 理想）");
    check(ab_rms[5] > ab_rms[1],
          "激励确认：延迟确实进一步恶化表现（延迟100ms vs 无延迟）");

    // ---- 2. 核心：卡尔曼在噪声场景优于 α-β ----
    check(kf_rms[1] < ab_rms[1], "核心：光流级噪声下卡尔曼优于 α-β");
    check(kf_rms[2] < ab_rms[2] * 0.6, "核心：中档噪声下卡尔曼显著优于 α-β（>1.6 倍）");
    check(kf_rms[3] < ab_rms[3], "核心：GPS 级噪声下卡尔曼优于 α-β");
    check(kf_rms[4] < ab_rms[4], "核心：廉价 GPS 下卡尔曼优于 α-β");

    // ---- 3. 核心：延迟与机动场景同样占优 ----
    check(kf_rms[5] < ab_rms[5] * 0.8, "核心：延迟场景下卡尔曼显著优于 α-β（>1.25 倍）");
    check(kf_rms[6] < ab_rms[6] * 0.8, "核心：机动+大噪声下卡尔曼显著优于 α-β");

    // ---- 4. 趋势：工况越差，卡尔曼优势越大（自适应能力的体现）----
    const double adv_small = ab_rms[1] / kf_rms[1];
    const double adv_large = ab_rms[6] / kf_rms[6];
    std::printf("  优势随工况恶化扩大：光流级 %.2fx → 机动+GPS %.2fx\n", adv_small, adv_large);
    check(adv_large > adv_small,
          "核心趋势：卡尔曼的优势随工况恶化而扩大（增益自适应的价值）");

    // ---- 5. 死区方案被判为最差：小噪声下严重劣化 ----
    std::printf("  死区方案：光流级 %.4f m（α-β 为 %.4f m）\n", dz_rms[1], ab_rms[1]);
    check(dz_rms[1] > ab_rms[1] * 5.0,
          "核心：死区补丁在小噪声下严重劣化（>5 倍，固定阈值过度抑制）");
    check(dz_rms[4] > kf_rms[4] && dz_rms[3] > kf_rms[3],
          "核心：死区补丁在噪声场景亦不如卡尔曼");

    // ---- 6. 边界：理想量测下两者相当（说明切换不伤害理想场景）----
    std::printf("  理想量测：α-β %.4f m vs 卡尔曼 %.4f m\n", ab_rms[0], kf_rms[0]);
    check(kf_rms[0] < ab_rms[0] * 1.3,
          "边界：理想量测下卡尔曼未劣化（差距 <30%，切换不伤害该场景）");

    std::printf("\n========================================\n");
    std::printf("%d / %d checks passed\n", g_pass, g_pass + g_fail);
    std::printf("========================================\n");
    return g_fail == 0 ? 0 : 1;
}
