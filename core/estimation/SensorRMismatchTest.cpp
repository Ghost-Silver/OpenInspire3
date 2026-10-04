/**
 * @file SensorRMismatchTest.cpp
 * @brief 卡尔曼 R 与实际量测噪声不匹配的影响，及多传感器下的配置策略
 *
 * @par 背景：架构上无法为多个传感器分别设 R
 *
 * `HalSensorReader::readPosition()` 返回单一位置量测，没有传感器标识；估计器
 * 也只有单一的 `kalman_pos_noise`，每帧量测共用同一个 R。因此同时存在特性不同
 * 的多个传感器（如 GPS + 光流）时，只能取一个全局值。
 *
 * 本测试量化「全局 R 取值」的代价，并给出可操作的配置规则。
 *
 * @par 重要更新（倾角限幅修正后）
 *
 * 本测试原先记录「R 偏小会发散」。倾角限幅修正（限幅后竖直分量守恒，见
 * SixDofPidController 的说明）消除了该发散 —— 实测 R=0.02 时由 20.01 m
 * 改善到 0.79 m，30 倍噪声差异下由 20.01 m 改善到 1.10 m。相关断言已由
 * 「记录发散」改为「确认有界」。**因此下述「危险方向」的结论在当前实现下
 * 不再表现为发散，但 R 偏小仍使性能劣化（0.79 m vs 0.25 m），保守取值的
 * 建议依然成立。**
 *
 * @par 核心发现：R 的偏差方向严重不对称
 *
 * 真实噪声 σ=0.30 时，改变配置给滤波器的 R：
 *
 * | 配置 R | 比值 | 最大高度偏差 |
 * |---|---|---|
 * | 0.02 | 0.07× | **20.01 m（发散）** |
 * | 0.05 | 0.17× | 4.09 m |
 * | 0.10 | 0.33× | 0.84 m |
 * | 0.30 | 1.00× | 0.38 m |
 * | 1.00 | 3.33× | 1.25 m |
 *
 * 即 **R 偏小（过度自信）会发散，R 偏大（保守）只是略慢**。这一不对称是
 * 下述配置规则的依据。
 *
 * @par 配置规则：全局 R 取「最大噪声源的 σ」
 *
 * 噪声相差 6 倍（光流 0.05 / GPS 0.30）：
 *
 * | 全局 R | 光流场景 | GPS 场景 |
 * |---|---|---|
 * | 0.05 | 0.05 m | 4.09 m |
 * | **0.30** | **0.02 m** | **0.38 m** |
 * | 0.50 | 0.03 m | 0.26 m |
 *
 * 噪声相差 30 倍（RTK 0.02 / 廉价 GPS 0.60）：
 *
 * | 全局 R | RTK 场景 | GPS 场景 |
 * |---|---|---|
 * | 0.02 | 0.03 m | **20.01 m（发散）** |
 * | 0.10 | 0.01 m | **20.00 m（发散）** |
 * | **0.30** | **0.01 m** | **1.14 m** |
 *
 * 可见取最大噪声源的 σ 可同时适配两类传感器，**且在 30 倍差异下依然成立**；
 * 代价极小（RTK 场景反而由 0.03 m 改善到 0.01 m）。而按最小或平均取值会让
 * 差传感器发散。
 *
 * @par 局限（诚实标注）
 *
 * 该规则是「次优但安全」的退路，并非真正的融合：取大 R 使高精度传感器被
 * 低估、信息未被充分利用（本例中 RTK 的潜在精度未体现）。真正的按源定 R
 * 需要 HAL 提供传感器标识与各自的噪声水平，当前架构不具备。
 */

#include "FlightControlLoop.h"
#include "HalSimulator.h"
#include "ImuModel.h"
#include "SixDofPidController.h"
#include "SixDofSimulator.h"
#include "TensorUtils.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

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
    double rms_xy_err = 0.0;
    int n = 0;
    bool diverged = false;
};

/// @param actual_sigma 真实的量测噪声
/// @param assumed_sigma 配置给滤波器的 R（kalman_pos_noise）
Out run(double actual_sigma, double assumed_sigma, int steps = 8000) {
    SixDofConfig cfg;
    SixDofState init{makeVec3(0.0f, 0.0f, -5.0f), makeVec3(0.0f, 0.0f, 0.0f),
                     Tensor{1.0f, 0.0f, 0.0f, 0.0f}, makeVec3(0.0f, 0.0f, 0.0f)};
    SixDofSimulator sim(cfg, init);

    ImuConfig ic;
    ic.explicit_bias = true;
    ImuModel imu(ic, 20260918u);

    PositionImperfection imp;
    imp.sigma = actual_sigma;
    imp.delay_steps = 0;
    SimSensorReader sensors(&sim, &imu, {0.0, 0.0, cfg.base.gravity}, 10, imp);
    SimActuatorWriter actuators(&sim);
    FixedSetpointSource setpoint(makeVec3(0.0f, 0.0f, -5.0f));
    SixDofPidController ctrl(cfg, {});

    FlightControlConfig fc;
    fc.estimator.sensor_health.enabled = true;
    fc.estimator.pos_filter = PosFilterKind::Kalman;
    fc.estimator.kalman_pos_noise = assumed_sigma;
    fc.hover_thrust = static_cast<double>(cfg.base.mass) * static_cast<double>(cfg.base.gravity);

    FlightControlLoop loop(fc, &sensors, &actuators, &setpoint, &ctrl);
    loop.init();

    Out o;
    double sum_sq = 0.0;
    for (int k = 0; k < steps; ++k) {
        if (!loop.runOneCycle()) {
            break;
        }
        const auto p = toVector(sim.state().pos);
        const double alt = -p[2];
        o.final_alt = alt;
        o.max_alt_dev = std::max(o.max_alt_dev, std::fabs(alt - 5.0));
        const double xy = std::hypot(static_cast<double>(p[0]), static_cast<double>(p[1]));
        o.max_xy = std::max(o.max_xy, xy);
        // 位置估计误差（对悬停而言真值应恒为 0）
        const auto pe = toVector(loop.estimator().state().pos);
        sum_sq += static_cast<double>(pe[0]) * static_cast<double>(pe[0]) +
                  static_cast<double>(pe[1]) * static_cast<double>(pe[1]);
        ++o.n;
        if (alt <= 0.0 || std::fabs(alt - 5.0) > 20.0) {
            o.diverged = true;
            break;
        }
    }
    o.rms_xy_err = o.n > 0 ? std::sqrt(sum_sq / static_cast<double>(o.n)) : 0.0;
    return o;
}

} // namespace

int main() {
    std::printf("=== SensorRMismatchTest: R 取值策略（多传感器下的全局配置）===\n\n");

    // ---- 1. 危险方向：R 偏小（乐观）会发散 ----
    std::printf("真实 σ=0.30 时改变配置 R：\n");
    const Out too_small = run(0.30, 0.02);
    const Out matched = run(0.30, 0.30);
    const Out too_large = run(0.30, 1.00);
    std::printf("  R=0.02（0.07×）: %.2f m%s\n", too_small.max_alt_dev,
                too_small.diverged ? "  [发散]" : "");
    std::printf("  R=0.30（1.00×）: %.2f m\n", matched.max_alt_dev);
    std::printf("  R=1.00（3.33×）: %.2f m\n\n", too_large.max_alt_dev);

    // 注：倾角限幅修正（竖直分量守恒）后，该场景不再发散 ——
    // 实测 20.01 m → 0.79 m。原断言「导致发散」已不成立，改为确认当前的有界性。
    check(!too_small.diverged && too_small.max_alt_dev < 3.0,
          "危险方向：R 远小于真实噪声时仍有界（限幅修正后实测 0.79 m）");
    check(!matched.diverged && matched.max_alt_dev < 2.0,
          "基准：R 等于真实噪声时工作正常");
    check(!too_large.diverged && too_large.max_alt_dev < 3.0,
          "安全方向：R 远大于真实噪声（保守）仍可用，仅略慢");

    // ---- 2. 策略验证：全局 R 取最大噪声源的 σ ----
    // 噪声差 6 倍
    const Out flow_r_max = run(0.05, 0.30);
    const Out gps_r_max = run(0.30, 0.30);
    std::printf("噪声差 6 倍，全局 R=0.30（取大）：光流 %.2f m，GPS %.2f m\n",
                flow_r_max.max_alt_dev, gps_r_max.max_alt_dev);
    check(!flow_r_max.diverged && flow_r_max.max_alt_dev < 1.0,
          "策略：取大 R 时高精度传感器不受影响（光流场景 <1 m）");
    check(!gps_r_max.diverged && gps_r_max.max_alt_dev < 2.0,
          "策略：取大 R 时低精度传感器可用（GPS 场景 <2 m）");

    // 噪声差 30 倍 —— 检验策略的边界
    const Out rtk_small = run(0.02, 0.02);
    const Out gps_small = run(0.60, 0.02);
    const Out rtk_large = run(0.02, 0.30);
    const Out gps_large = run(0.60, 0.30);
    std::printf("噪声差 30 倍：\n");
    std::printf("  全局 R=0.02（按 RTK）：RTK %.2f m，GPS %.2f m%s\n", rtk_small.max_alt_dev,
                gps_small.max_alt_dev, gps_small.diverged ? "  [发散]" : "");
    std::printf("  全局 R=0.30（取大）  ：RTK %.2f m，GPS %.2f m%s\n", rtk_large.max_alt_dev,
                gps_large.max_alt_dev, gps_large.diverged ? "  [发散]" : "");

    // 同上：限幅修正后该场景由 20.01 m 改善到 1.10 m，不再发散。
    check(!gps_small.diverged && gps_small.max_alt_dev < 3.0,
          "边界：按最小噪声设定 R 时差传感器仍有界（修正后实测 1.10 m）");
    check(!rtk_large.diverged && rtk_large.max_alt_dev < 1.0,
          "边界：取大 R 时高精度传感器仍正常（30 倍差异下）");
    check(!gps_large.diverged && gps_large.max_alt_dev < 3.0,
          "边界：取大 R 时差传感器可用（30 倍差异下）");

    // ---- 3. 代价评估：保守策略不劣化高精度场景 ----
    std::printf("\n保守策略的代价：RTK 场景 %.3f m（按自身 σ 设）→ %.3f m（取大）\n",
                rtk_small.max_alt_dev, rtk_large.max_alt_dev);
    check(rtk_large.max_alt_dev <= rtk_small.max_alt_dev * 3.0 + 0.01,
          "代价评估：保守取值不显著劣化高精度传感器");

    std::printf("\n========================================\n");
    std::printf("%d / %d checks passed\n", g_pass, g_pass + g_fail);
    std::printf("========================================\n");
    return g_fail == 0 ? 0 : 1;
}
