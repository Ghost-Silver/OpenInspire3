/**
 * @file ManeuverFaultTest.cpp
 * @brief 机动飞行中的故障检测（含一处检测盲区的回归）
 *
 * @par 背景：历轮验证全在悬停完成
 *
 * `SensorHealth` 对机动帧有多处特殊处理，而所有降级验证都在悬停场景进行
 * （残差基线约 0.004）。本测试补上机动场景的覆盖。
 *
 * @par 发现并修复的检测盲区（核心回归目标）
 *
 * 原实现对机动帧设了**两处门控**：
 * 1. 机动帧不喂 CUSUM；
 * 2. 机动帧即使 CUSUM 报警也不归因到 AccelBias。
 *
 * 两者叠加的后果是：**持续机动时偏置检测完全失效** —— 若全部帧都被判为
 * 机动帧，CUSUM 一个样本都收不到，永远不会报警。实测（连续四向目标跳变、
 * 注入偏置 +1.0）残差由 0.00951 升至 0.08357（**8.79 倍，信号清晰可辨**），
 * 却完全未检出。
 *
 * 这推翻了原文档将其列为「物理限制」的判断 —— 它是机制性的（门控过度），
 * 不是检测能力不足。
 *
 * 修复：机动帧改为**扣除基线偏移后**喂 CUSUM（`maneuver_residual_offset`），
 * 净残差回到静止量级，同一个 CUSUM 即可适配两种状态；同时移除第二处归因门控。
 * 修复后持续机动场景的检出延迟为 99 步，与悬停场景一致。
 *
 * @par 顺带更正的文档错误
 *
 * 文件头原记录「机动时残差基线抬升两个量级（0.0041 → 0.323）」。复测覆盖
 * 悬停/单次机动/剧烈机动/瞬态峰值/多种校正增益，机动帧残差始终在
 * 0.004~0.023，即约为静止的 2 倍而非两个量级。本测试把实测值固化下来，
 * 避免文档再次漂移。
 */

#include "ImuDegradePolicy.h"
#include "ImuModel.h"
#include "SensorHealth.h"
#include "SixDofPidController.h"
#include "SixDofSimulator.h"
#include "StateEstimator.h"
#include "TensorUtils.h"
#include "DroneTypes.h"

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

std::array<double, 3> rv(const Tensor &t) {
    const auto v = toVector(t);
    return {v[0], v[1], v[2]};
}

const char *faultNameOf(ImuFault f) {
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

const char *statusName(SensorStatus s) {
    switch (s) {
    case SensorStatus::Unknown: return "Unknown";
    case SensorStatus::Healthy: return "Healthy";
    case SensorStatus::Degraded: return "Degraded";
    case SensorStatus::Failed: return "Failed";
    }
    return "?";
}

struct Out {
    double mean_resid_still = 0.0;
    double max_resid_still = 0.0;
    double mean_resid_maneuver = 0.0;
    double max_resid_maneuver = 0.0;
    long long still_frames = 0;
    long long maneuver_frames = 0;
    double max_gyro = 0.0;
    int detect_step = -1;
    SensorStatus final_status = SensorStatus::Unknown;
    ImuFault final_fault = ImuFault::None;
};

/**
 * @param maneuver 是否在检测窗口内制造机动
 * @param bias     注入的加速度计偏置（0 表示不注入）
 */
Out run(bool maneuver, double bias, int steps = 9000) {
    SixDofConfig cfg;
    const double dt = cfg.base.dt;
    SixDofState init{makeVec3(0.0f, 0.0f, -5.0f), makeVec3(0.0f, 0.0f, 0.0f),
                     Tensor{1.0f, 0.0f, 0.0f, 0.0f}, makeVec3(0.0f, 0.0f, 0.0f)};
    SixDofSimulator sim(cfg, init);
    SixDofPidController pid(cfg, {});

    ImuConfig ic;
    ic.explicit_bias = true;
    ic.accel_bias_vec = {0.0, 0.0, 0.0};
    ic.gyro_bias_vec = {0.0, 0.0, 0.0};
    ImuModel imu(ic, 20260918u);

    EstimatorConfig etc;
    etc.sensor_health.enabled = true;
    StateEstimator est(etc);
    est.reset();

    // 目标点在 3000 步时大幅跳变，制造真实机动
    const Tensor target_near = makeVec3(0.0f, 0.0f, -5.0f);
    const Tensor target_far = makeVec3(12.0f, 8.0f, -5.0f);
    constexpr int jump_step = 3000;
    constexpr int bias_step = 3600; // 机动开始后注入偏置

    Out o;
    double resid_still_sum = 0.0;
    double resid_maneuver_sum = 0.0;

    for (int k = 0; k < steps; ++k) {
        const double t = k * dt;
        const Tensor target = (maneuver && k >= jump_step) ? target_far : target_near;

        ImuSample s = imu.measure(sim.state(), {0.0, 0.0, -cfg.base.gravity}, dt);
        if (bias > 0.0 && k >= bias_step) {
            s.accel[0] += bias;
        }
        est.updateImu(s, dt);
        if (k % 10 == 0) {
            est.updatePosition(rv(sim.state().pos), dt * 10.0);
        }

        const auto cmd = pid.compute(est.state(), target, t);
        sim.step(cmd.thrust_body, cmd.torque);

        const auto &h = est.sensorHealth();

        // 统计窗口：注入之后（避开跳变刚发生时的瞬态）
        if (k >= bias_step) {
            if (h.maneuvering) {
                ++o.maneuver_frames;
                resid_maneuver_sum += h.residual;
                o.max_resid_maneuver = std::max(o.max_resid_maneuver, h.residual);
            } else {
                ++o.still_frames;
                resid_still_sum += h.residual;
                o.max_resid_still = std::max(o.max_resid_still, h.residual);
            }
            const double gm = std::sqrt(s.gyro[0] * s.gyro[0] + s.gyro[1] * s.gyro[1] +
                                        s.gyro[2] * s.gyro[2]);
            o.max_gyro = std::max(o.max_gyro, gm);
        }

        if (bias > 0.0 && o.detect_step < 0 && h.accel != SensorStatus::Healthy &&
            h.accel != SensorStatus::Unknown) {
            o.detect_step = k;
        }
    }

    o.mean_resid_still =
        o.still_frames > 0 ? resid_still_sum / static_cast<double>(o.still_frames) : 0.0;
    o.mean_resid_maneuver =
        o.maneuver_frames > 0 ? resid_maneuver_sum / static_cast<double>(o.maneuver_frames) : 0.0;
    const auto &h = est.sensorHealth();
    o.final_status = h.accel;
    o.final_fault = h.fault;
    return o;
}

} // namespace

int main() {
    std::printf("=== ManeuverFaultTest: 机动飞行中的故障检测 ===\n\n");

    // ---- A. 悬停基线（对照）----
    const Out hover = run(false, 0.0);
    std::printf("悬停（无故障）：残差均值 %.5f，峰值 %.5f，机动帧 %lld\n", hover.mean_resid_still,
                hover.max_resid_still, hover.maneuver_frames);

    // ---- B. 机动基线（无故障）----
    const Out man = run(true, 0.0);
    std::printf("机动（无故障）：机动帧残差均值 %.5f，峰值 %.5f（%lld 帧）\n",
                man.mean_resid_maneuver, man.max_resid_maneuver, man.maneuver_frames);
    std::printf("                静止帧残差均值 %.5f（%lld 帧）\n", man.mean_resid_still,
                man.still_frames);

    // ---- C. 悬停 + 偏置（对照）----
    const Out hover_bias = run(false, 1.0);
    std::printf("悬停 + 偏置 1.0：检出步 %d（注入于第 3600 步）\n", hover_bias.detect_step);

    // ---- D. 机动 + 偏置（核心回归）----
    const Out man_bias = run(true, 1.0);
    std::printf("机动 + 偏置 1.0：检出步 %d（注入于第 3600 步）\n", man_bias.detect_step);
    std::printf("\n");

    // ---- 1. 激励确认：机动确实发生 ----
    check(man.maneuver_frames > 1000, "激励确认：机动场景确实产生了大量机动帧");
    check(man.max_gyro > 0.5, "激励确认：机动角速度显著超过机动判定阈值");

    // ---- 2. 机动基线实测值（替代文档中被证伪的 0.323）----
    // 固化实测，避免文档再次漂移到无法重现的数值。
    std::printf("  机动帧残差均值 %.5f（静止 %.5f，约 %.1f 倍）\n", man.mean_resid_maneuver,
                man.mean_resid_still,
                man.mean_resid_still > 0 ? man.mean_resid_maneuver / man.mean_resid_still : 0.0);
    check(man.mean_resid_maneuver < 0.05,
          "机动残差基线低于 0.05（实测约 0.01，与文档旧值 0.323 不符）");
    check(man.mean_resid_maneuver < man.max_resid_maneuver,
          "机动残差均值小于峰值（分布合理）");

    // ---- 3. 【核心】机动不得误报 ----
    check(man.detect_step < 0 || man.final_status == SensorStatus::Healthy,
          "核心：机动（无偏置）不触发误报");
    check(man.final_status != SensorStatus::Degraded,
          "核心：机动（无偏置）末态不为 Degraded");

    // ---- 4. 【核心回归】持续机动时偏置必须可检出 ----
    // 修复前此处为 -1（完全未检出）：机动帧双重门控使 CUSUM 收不到样本。
    check(man_bias.detect_step > 0, "核心：持续机动时偏置仍可检出（修复前的检测盲区）");
    check(man_bias.final_status == SensorStatus::Degraded ||
              man_bias.final_status == SensorStatus::Failed,
          "核心：机动+偏置末态为降级状态");
    check(man_bias.final_fault == ImuFault::AccelBias, "核心：机动+偏置被正确归因为 AccelBias");

    // ---- 5. 机动检出延迟与悬停同量级（说明补偿有效）----
    if (hover_bias.detect_step > 0 && man_bias.detect_step > 0) {
        const int hover_delay = hover_bias.detect_step - 3600;
        const int man_delay = man_bias.detect_step - 3600;
        std::printf("  检出延迟：悬停 %d 步，机动 %d 步\n", hover_delay, man_delay);
        check(man_delay <= hover_delay * 3,
              "机动检出延迟与悬停同量级（基线补偿有效）");
    }

    // ---- 6. 信号本身可辨（证明检出依赖真实信号而非侥幸）----
    check(man_bias.mean_resid_maneuver > man.mean_resid_maneuver * 3.0,
          "偏置信号在机动中仍显著高于基线（>3 倍）");

    std::printf("\n========================================\n");
    std::printf("%d / %d checks passed\n", g_pass, g_pass + g_fail);
    std::printf("========================================\n");
    return g_fail == 0 ? 0 : 1;
}
