/**
 * @file HardFaultRecoveryTest.cpp
 * @brief 硬故障（加速度计掉线）恢复：三处缺陷的回归测试
 *
 * @par 背景
 *
 * 代码注释声称「掉线/卡死是硬故障，恢复后允许清除（传感器可能只是瞬时断连）」，
 * 但实测发现从 Failed 状态**永远无法恢复**：加速度计第 2000 周期归零、
 * 第 5000 周期恢复（模长回到 9.765 m/s²），到第 11000 周期仍为 Failed/AccelDead。
 * 真机后果是瞬时断连（接触不良、EMI）会让飞控永久处于降级返航状态。
 *
 * 追查过程中共发现三处缺陷，本测试对三者一并回归：
 *
 * 1. **恢复路径被完全排除**：状态落实处写成
 *    `else if (_report.accel != SensorStatus::Failed)`，即 Failed 状态永不进入
 *    恢复分支。改为「Failed 需连续 confirm_steps 帧数据正常才解除」（对称去抖），
 *    其余状态照旧立即解除。
 *
 * 2. **健康计数被判据死锁**：恢复计数原先用 `!accel_bad_now` 递增，而
 *    `accel_bad_now` 包含 CUSUM 的**累积报警**。CUSUM 是锁存判据、需显式清除，
 *    于是「报警 → 恒不正常 → 健康计数恒为 0 → 无法恢复」形成死锁。
 *    改为按**原始数据质量**（模长正常、未冻结）计数 —— 当前数据正常即可确认
 *    恢复，历史累积报警不该阻止这一点。
 *
 * 3. **CUSUM 松弛量过低导致自我维持的假阳性**：`cusum_drift` 原为 0.01，
 *    只高于正常稳态残差（0.003~0.006），却低于「姿态尚未收敛」时的残差
 *    （0.012~0.018）。掉线期间方向校正失效、姿态漂移，恢复后残差偏高即被
 *    误判为 AccelBias → 判为 Degraded → 降级策略关闭方向校正 → 姿态更无法
 *    收敛 → Degraded 无限延长。实测恢复后 Degraded 占比高达 76.7%。
 *    松弛量提高到 0.03（高于未收敛水平 0.018，远低于真实偏置 0.1426）。
 *
 * @par 断言设计
 *
 * 除「能否恢复」外，还断言**恢复耗时**与 `confirm_steps` 一致（防止缺少去抖
 * 的立即恢复），以及恢复后的**状态稳定性占比**（防止「能恢复但反复抖动」——
 * 第 3 处缺陷正是这种情况，仅断言「最终为 Healthy」无法发现）。
 */

#include "DroneTypes.h"
#include "ImuModel.h"
#include "SensorHealth.h"
#include "SixDofSimulator.h"
#include "StateEstimator.h"
#include "TensorUtils.h"

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
const char *statusName(SensorStatus s) {
    switch (s) {
    case SensorStatus::Unknown: return "Unknown";
    case SensorStatus::Healthy: return "Healthy";
    case SensorStatus::Degraded: return "Degraded";
    case SensorStatus::Failed: return "Failed";
    }
    return "?";
}
const char *faultName(ImuFault f) {
    switch (f) {
    case ImuFault::None: return "None";
    case ImuFault::AccelDead: return "AccelDead";
    case ImuFault::AccelFrozen: return "AccelFrozen";
    case ImuFault::AccelBias: return "AccelBias";
    case ImuFault::GyroDead: return "GyroDead";
    case ImuFault::GyroFrozen: return "GyroFrozen";
    case ImuFault::GyroBias: return "GyroBias";
    }
    return "?";
}
} // namespace

int main() {
    SixDofConfig cfg;
    const double dt = cfg.base.dt;
    SixDofState init{makeVec3(0.0f, 0.0f, -5.0f), makeVec3(0.0f, 0.0f, 0.0f),
                     Tensor{1.0f, 0.0f, 0.0f, 0.0f}, makeVec3(0.0f, 0.0f, 0.0f)};
    SixDofSimulator sim(cfg, init);

    ImuConfig ic;
    ic.explicit_bias = true;
    ImuModel imu(ic, 20260918u);

    EstimatorConfig etc;
    etc.sensor_health.enabled = true;
    StateEstimator est(etc);
    est.reset();

    // 0-1999 正常；2000-4999 加速度计归零（掉线）；5000+ 恢复
    constexpr int dead_on = 2000;
    constexpr int dead_off = 5000;
    constexpr int steps = 12000;

    int first_failed = -1;
    int first_healthy_after = -1;
    // 统计恢复区间内残差超过 recovery_residual 的帧数与机动帧数
    long long over_thresh = 0;
    long long total_after = 0;
    long long maneuver_frames = 0;
    double max_resid_after = 0.0;
    long long stable_frames = 0;
    long long stable_healthy = 0;
    long long stable_degraded = 0;
    long long stable_failed = 0;

    std::printf("%6s %12s %14s %10s %14s\n", "周期", "加速度计", "故障类型", "加速度", "样本内计数");
    for (int k = 0; k < steps; ++k) {
        ImuSample s = imu.measure(sim.state(), {0.0, 0.0, -cfg.base.gravity}, dt);
        if (k >= dead_on && k < dead_off) {
            s.accel = {0.0, 0.0, 0.0}; // 掉线
        }
        est.updateImu(s, dt);
        if (k % 10 == 0) {
            const auto p = toVector(sim.state().pos);
            est.updatePosition({p[0], p[1], p[2]}, dt * 10.0);
        }

        const auto &h = est.sensorHealth();
        if (k >= dead_off && first_healthy_after < 0) {
            ++total_after;
            if (h.residual >= 0.02) {
                ++over_thresh;
            }
            if (h.maneuvering) {
                ++maneuver_frames;
            }
            max_resid_after = std::max(max_resid_after, h.residual);
        }
        if (h.accel == SensorStatus::Failed && first_failed < 0) {
            first_failed = k;
        }
        // 恢复稳定后（首次恢复后 2000 帧起）统计各状态占比
        if (first_healthy_after > 0 && k > first_healthy_after + 2000) {
            ++stable_frames;
            if (h.accel == SensorStatus::Healthy) ++stable_healthy;
            if (h.accel == SensorStatus::Degraded) ++stable_degraded;
            if (h.accel == SensorStatus::Failed) ++stable_failed;
        }
        if (first_failed > 0 && k > dead_off && h.accel == SensorStatus::Healthy &&
            first_healthy_after < 0) {
            first_healthy_after = k;
        }

        const bool show = (k % 1000 == 0) || k == dead_on || k == dead_off ||
                          (first_healthy_after > 0 && k == first_healthy_after);
        // 密集输出恢复区间，用于定位是什么在阻止恢复
        const bool dense = (k > dead_off && k < dead_off + 900 && k % 50 == 0);
        if (show || dense) {
            const double mag = std::sqrt(s.accel[0] * s.accel[0] + s.accel[1] * s.accel[1] +
                                         s.accel[2] * s.accel[2]);
            std::printf("%6d %12s %14s 残差=%9.5f 模长=%7.3f\n", k, statusName(h.accel),
                        faultName(h.fault), h.residual, h.accel_magnitude);
        }
    }

    std::printf("\n掉线开始:     第 %d 周期\n", dead_on);
    std::printf("首次 Failed:  第 %d 周期\n", first_failed);
    std::printf("掉线结束:     第 %d 周期\n", dead_off);
    std::printf("恢复 Healthy: %s\n",
                first_healthy_after > 0 ? (std::to_string(first_healthy_after) + " 周期").c_str()
                                        : "未恢复");
    const auto &h = est.sensorHealth();
    std::printf("末态:         accel=%s fault=%s\n", statusName(h.accel), faultName(h.fault));
    std::printf("\n--- 恢复区间统计（第 %d 周期起）---\n", dead_off);
    std::printf("总帧数:             %lld\n", total_after);
    std::printf("残差 >= 0.02 帧数:  %lld (%.2f%%)\n", over_thresh,
                total_after > 0 ? 100.0 * static_cast<double>(over_thresh) / static_cast<double>(total_after) : 0.0);
    std::printf("机动帧数:           %lld (%.2f%%)\n", maneuver_frames,
                total_after > 0 ? 100.0 * static_cast<double>(maneuver_frames) / static_cast<double>(total_after) : 0.0);
    std::printf("残差最大值:         %.5f\n", max_resid_after);
    std::printf("末态是否 Healthy:   %s\n", h.accel == SensorStatus::Healthy ? "是" : "否");
    const double healthy_ratio =
        stable_frames ? static_cast<double>(stable_healthy) / static_cast<double>(stable_frames)
                      : 0.0;
    std::printf("\n--- 恢复后稳定性（首次恢复 2000 帧之后）---\n");
    std::printf("总帧数 %lld：Healthy %lld (%.1f%%)、Degraded %lld (%.1f%%)、Failed %lld (%.1f%%)\n",
                stable_frames, stable_healthy, healthy_ratio * 100.0, stable_degraded,
                stable_frames ? 100.0 * static_cast<double>(stable_degraded) / static_cast<double>(stable_frames) : 0.0,
                stable_failed,
                stable_frames ? 100.0 * static_cast<double>(stable_failed) / static_cast<double>(stable_frames) : 0.0);
    std::printf("\n");

    // ---- 1. 激励确认 ----
    check(first_failed > 0, "激励确认：掉线被检出（进入 Failed）");
    check(first_failed >= dead_on, "激励确认：Failed 发生在掉线之后（无提前误报）");

    // ---- 2. 【核心】硬故障必须能恢复 ----
    // 缺陷 1 与 2 存在时此断言失败：状态永久停在 Failed。
    check(first_healthy_after > 0, "核心：掉线恢复后状态回到 Healthy");

    // ---- 3. 恢复耗时须与 confirm_steps 一致（防止缺少去抖的立即恢复）----
    const int rec_time = first_healthy_after - dead_off;
    std::printf("  恢复耗时 %d 周期（confirm_steps 配置为 100）\n", rec_time);
    check(rec_time >= 80 && rec_time <= 200,
          "恢复耗时与 confirm_steps 一致（存在去抖，非立即恢复）");

    // ---- 4. 【核心】恢复后状态须稳定 ----
    // 缺陷 3 存在时此断言失败：能恢复，但随后大部分时间被判为 Degraded
    // （实测 Degraded 占比 76.7%）。仅断言「最终为 Healthy」无法发现该问题。
    std::printf("  恢复后 Healthy 占比 %.1f%%\n", healthy_ratio * 100.0);
    check(healthy_ratio > 0.95, "核心：恢复后状态稳定为 Healthy（占比 >95%）");
    check(stable_degraded == 0, "核心：恢复后不再出现 Degraded 假阳性");

    // ---- 5. 传感器数据确实正常（佐证判据基于真实数据而非侥幸）----
    check(h.accel_magnitude > 9.0 && h.accel_magnitude < 10.5,
          "前置条件：末态加速度模长正常（≈g）");

    std::printf("\n========================================\n");
    std::printf("%d / %d checks passed\n", g_pass, g_pass + g_fail);
    std::printf("========================================\n");
    return g_fail == 0 ? 0 : 1;
}
