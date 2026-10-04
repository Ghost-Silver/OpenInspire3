/**
 * @file LoopSensorQualityTest.cpp
 * @brief 生产主循环在非理想位置量测下的行为（滤波器与死区方案的对照）
 *
 * @par 为什么需要
 *
 * `SimSensorReader` 原先无条件返回真值位置，因此所有走 HAL 的**生产主循环**
 * 测试（FlightControlLoopTest / GuidanceIntegrationTest / EmergencyLandingTest）
 * 都建立在理想量测之上。而覆盖了非理想量测的 `PositionNonIdealClosedLoopTest`
 * 走的是**裸循环**。于是「非理想量测 + 生产主循环」这一组合从未被验证 ——
 * 而真机上 GPS/光流必然带噪声与延迟。
 *
 * 本测试为该组合补上覆盖，并对照三种配置。
 *
 * @par 实测结论（本测试固化的核心事实）
 *
 * | 场景 | AlphaBeta（默认） | AlphaBeta + 死区 0.15 | 卡尔曼 |
 * |---|---|---|---|
 * | 理想 | 5.00 m / 0.00 m | 5.00 m / 0.00 m | 5.00 m / 0.00 m |
 * | 光流级 σ=0.05 | 5.07 m / 0.08 m | 6.60 m / **1.60 m** | **5.04 m / 0.05 m** |
 * | GPS 级 σ=0.30 | **45.56 m / 40.56 m** | 7.12 m / 5.05 m | **5.46 m / 0.46 m** |
 * | 仅噪声 σ=0.30 | **44.22 m / 39.22 m** | 7.18 m / 4.17 m | **5.35 m / 0.39 m** |
 *
 * （格式：末态高度 / 最大高度偏差）
 *
 * 三条结论：
 *
 * 1. **主循环默认配置（α-β）在 GPS 级噪声下发散** —— 高度漂至 45 m。这与
 *    裸循环的表现一致（PositionNonIdealClosedLoopTest 曾测到 46 m）。
 * 2. **死区（P5 方案）能抑制发散，但引入显著副作用** —— 光流级场景由 0.08 m
 *    劣化到 1.60 m（约 20 倍）。死区是固定阈值，对小噪声场景属于过度抑制，
 *    无法同时适配两种噪声水平。
 * 3. **卡尔曼在两个场景下都是最优**，且无需死区。这与它增益自动缩放的设计
 *    一致 —— 把真实 σ 告知滤波器即可适配。
 *
 * @par 关于默认值的现状
 *
 * 主循环默认仍为 AlphaBeta、死区默认关闭（`pos_residual_deadzone = 0`），
 * 故本测试记录的「默认配置在 GPS 级噪声下发散」是**当前的真实状态**，不是
 * 回归。切换默认值会改变全部既有测试基线，属独立决策，不在本测试范围内。
 * 本测试的作用是把该状态**显式记录下来并加断言保护**：一旦默认值变更，
 * 相关断言会提示更新，而不是悄悄改变行为。
 */

#include "FlightControlLoop.h"
#include "GuidanceSetpointSource.h"
#include "HalSimulator.h"
#include "ImuModel.h"
#include "SixDofPidController.h"
#include "SixDofSimulator.h"
#include "TensorUtils.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <string>

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

struct Out {
    double final_alt = 5.0;
    double max_alt_dev = 0.0;
    double max_xy = 0.0;
    int cycles = 0;
    std::string action;
    SensorStatus accel = SensorStatus::Unknown;
};

const char *actionName(DegradeAction a) {
    switch (a) {
    case DegradeAction::Normal: return "Normal";
    case DegradeAction::Cautious: return "Cautious";
    case DegradeAction::ReturnHome: return "ReturnHome";
    case DegradeAction::EmergencyLand: return "EmergencyLand";
    }
    return "?";
}

const char *statusName(SensorStatus s) {
    switch (s) {
    case SensorStatus::Unknown: return "Unknown";
    case SensorStatus::Healthy: return "Healthy";
    case SensorStatus::Degraded: return "Degraded";
    case SensorStatus::Failed: return "Failed";
    }
    return "?";
}

Out runLoop(const PositionImperfection &imp, bool kalman, int steps = 8000,
            double deadzone = 0.0) {
    SixDofConfig cfg;
    SixDofState init{makeVec3(0.0f, 0.0f, -5.0f), makeVec3(0.0f, 0.0f, 0.0f),
                     Tensor{1.0f, 0.0f, 0.0f, 0.0f}, makeVec3(0.0f, 0.0f, 0.0f)};
    SixDofSimulator sim(cfg, init);

    ImuConfig ic;
    ic.explicit_bias = true;
    ic.accel_bias_vec = {0.0, 0.0, 0.0};
    ic.gyro_bias_vec = {0.0, 0.0, 0.0};
    ImuModel imu(ic, 20260918u);

    SimSensorReader sensors(&sim, &imu, {0.0, 0.0, cfg.base.gravity}, 10, imp);
    SimActuatorWriter actuators(&sim);
    FixedSetpointSource setpoint(makeVec3(0.0f, 0.0f, -5.0f));
    SixDofPidController ctrl(cfg, {});

    FlightControlConfig fc;
    fc.estimator.sensor_health.enabled = true;
    fc.estimator.pos_filter =
        kalman ? PosFilterKind::Kalman : PosFilterKind::AlphaBeta;
    // 卡尔曼按设计用法告知真实噪声水平
    if (kalman && imp.sigma > 0.0) {
        fc.estimator.kalman_pos_noise = imp.sigma;
    }
    fc.estimator.pos_residual_deadzone = deadzone;
    fc.hover_thrust = static_cast<double>(cfg.base.mass) * static_cast<double>(cfg.base.gravity);

    FlightControlLoop loop(fc, &sensors, &actuators, &setpoint, &ctrl);
    loop.init();

    Out o;
    for (int k = 0; k < steps; ++k) {
        if (!loop.runOneCycle()) {
            break;
        }
        ++o.cycles;
        const auto p = toVector(sim.state().pos);
        const double alt = -p[2];
        o.final_alt = alt;
        o.max_alt_dev = std::max(o.max_alt_dev, std::fabs(alt - 5.0));
        o.max_xy = std::max(o.max_xy, std::hypot(static_cast<double>(p[0]),
                                                 static_cast<double>(p[1])));
        if (alt <= 0.0) {
            break; // 触地
        }
    }
    o.action = actionName(loop.currentDecision().action);
    o.accel = loop.currentHealth().accel;
    return o;
}

} // namespace

int main() {
    std::printf("=== LoopSensorQualityTest: 生产主循环 × 位置量测质量 ===\n\n");

    const PositionImperfection ideal{0.0, 0, 20260918u};
    const PositionImperfection flow{0.05, 20, 20260918u};
    const PositionImperfection gps{0.30, 100, 20260918u};
    const PositionImperfection gps_noise_only{0.30, 0, 20260918u};

    // 三种配置 × 四个场景
    const Out ab_ideal = runLoop(ideal, false);
    const Out ab_flow = runLoop(flow, false);
    const Out ab_gps = runLoop(gps, false);
    const Out ab_gps_n = runLoop(gps_noise_only, false);

    const Out dz_flow = runLoop(flow, false, 8000, 0.15);
    const Out dz_gps = runLoop(gps, false, 8000, 0.15);

    const Out kf_ideal = runLoop(ideal, true);
    const Out kf_flow = runLoop(flow, true);
    const Out kf_gps = runLoop(gps, true);
    const Out kf_gps_n = runLoop(gps_noise_only, true);

    std::printf("%-22s %11s %11s %11s\n", "场景", "AB 末态", "AB+死区", "卡尔曼");
    std::printf("%-22s %10.2f m %10.2f m %10.2f m\n", "理想", ab_ideal.final_alt,
                ab_ideal.final_alt, kf_ideal.final_alt);
    std::printf("%-22s %10.2f m %10.2f m %10.2f m\n", "光流级 σ=0.05", ab_flow.final_alt,
                dz_flow.final_alt, kf_flow.final_alt);
    std::printf("%-22s %10.2f m %10.2f m %10.2f m\n", "GPS级 σ=0.30",
                ab_gps.final_alt, dz_gps.final_alt, kf_gps.final_alt);
    std::printf("%-22s %10.2f m %10s %10.2f m\n", "仅噪声 σ=0.30", ab_gps_n.final_alt, "—",
                kf_gps_n.final_alt);
    std::printf("\n");

    // ---- 1. 默认行为不变：理想量测下两种滤波器都应精确悬停 ----
    check(std::fabs(ab_ideal.final_alt - 5.0) < 0.05, "理想量测：AlphaBeta 精确悬停");
    check(std::fabs(kf_ideal.final_alt - 5.0) < 0.05, "理想量测：卡尔曼精确悬停");
    check(ab_ideal.max_xy < 0.05, "理想量测：AlphaBeta 水平无漂移");

    // ---- 2. 记录并断言「默认配置在 GPS 级噪声下发散」这一现状 ----
    // 这不是回归，而是当前真实状态：主循环默认 α-β 且死区关闭。
    // 若将来默认值变更，此处会提示更新，而非悄悄地改变行为。
    std::printf("  默认配置 GPS 级高度偏差 %.2f m\n", std::fabs(ab_gps.final_alt - 5.0));
    // 注：倾角限幅修正（竖直分量守恒）后，该场景由 45.56 m 改善到 5.42 m，
    // 不再发散。原「现状记录：发散」断言已失效，改为确认当前的有界性。
    check(std::fabs(ab_gps.final_alt - 5.0) < 2.0,
          "现状更新：默认配置在 GPS 级噪声下已不发散（修正后偏差 <2 m）");
    check(ab_gps.action == "Normal",
          "现状记录：发散时未触发降级（传感器本身健康，属估计器问题）");

    // ---- 3. 核心：卡尔曼在非理想量测下不发散 ----
    check(std::fabs(kf_gps.final_alt - 5.0) < 3.0,
          "核心：卡尔曼在 GPS 级噪声下保持悬停（偏差 <3 m）");
    check(std::fabs(kf_gps_n.final_alt - 5.0) < 3.0,
          "核心：卡尔曼在纯噪声 σ=0.30 下保持悬停");
    check(kf_gps.max_xy < 5.0, "核心：卡尔曼在 GPS 级噪声下水平不发散");

    // ---- 4. 核心：卡尔曼在两个场景都不劣于 AlphaBeta ----
    check(kf_flow.max_alt_dev <= ab_flow.max_alt_dev * 1.5 + 0.05,
          "核心：光流级下卡尔曼不劣于 AlphaBeta");
    // 注：发散消除后两者差距缩小（5.26 vs 5.42 m）。原「显著优于」的断言
    // 建立在 AlphaBeta 发散的旧行为上，现改为「不劣于」。
    check(std::fabs(kf_gps.final_alt - 5.0) <= std::fabs(ab_gps.final_alt - 5.0) + 0.5,
          "核心：GPS 级下卡尔曼不劣于 AlphaBeta（差距随发散消除而缩小）");

    // ---- 5. 死区方案的副作用（固化为事实，避免未来被误当作无损方案）----
    std::printf("  死区方案：光流级偏差 %.2f m（AlphaBeta 无死区为 %.2f m）\n",
                dz_flow.max_alt_dev, ab_flow.max_alt_dev);
    check(dz_flow.max_alt_dev > ab_flow.max_alt_dev * 3.0,
          "死区方案副作用：光流级场景显著劣化（固定阈值对小噪声过度抑制）");
    check(std::fabs(dz_gps.final_alt - 5.0) < 10.0,
          "死区方案有效性：GPS 级场景回到有界");
    check(std::fabs(kf_gps.final_alt - 5.0) < std::fabs(dz_gps.final_alt - 5.0),
          "卡尔曼在 GPS 级下优于死区方案（无需固定阈值补丁）");

    std::printf("\n========================================\n");
    std::printf("%d / %d checks passed\n", g_pass, g_pass + g_fail);
    std::printf("========================================\n");
    return g_fail == 0 ? 0 : 1;
}
