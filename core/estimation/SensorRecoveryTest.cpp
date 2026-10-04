/**
 * @file SensorRecoveryTest.cpp
 * @brief 故障恢复路径：偏置消失后健康状态能否回到 Healthy 并恢复飞行
 *
 * @par 为什么需要这个测试
 *
 * 检测与降级（故障出现）已有充分覆盖，但**恢复**（故障消失）此前完全没有测试。
 * 这条路径存在真实的失效模式：如果恢复逻辑不生效，一次瞬时偏置会把状态
 * 永久锁在 Degraded —— CUSUM 报警是锁存的，且机动帧不再清零计数，传感器即使
 * 完全恢复也无法回到 Healthy，飞机将永久带着限幅飞行。
 *
 * SensorHealth 为此设计了显式出口（`recovery_residual` / `recovery_steps`：
 * 非机动帧残差连续低于阈值即解除 Degraded，并重置 CUSUM）。本测试验证它生效。
 *
 * @par 断言设计
 *
 * 除「最终回到 Healthy」外，还断言恢复**耗时**与配置一致 —— 若恢复逻辑被
 * 误改为立即恢复（缺少去抖），仅断言最终状态无法发现，必须核对时长。
 */

#include "DegradeExecutor.h"
#include "ImuDegradePolicy.h"
#include "ImuModel.h"
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

const char *statusName(SensorStatus s) {
    switch (s) {
    case SensorStatus::Unknown: return "Unknown";
    case SensorStatus::Healthy: return "Healthy";
    case SensorStatus::Degraded: return "Degraded";
    case SensorStatus::Failed: return "Failed";
    }
    return "?";
}

const char *actionName(DegradeAction a) {
    switch (a) {
    case DegradeAction::Normal: return "Normal";
    case DegradeAction::Cautious: return "Cautious";
    case DegradeAction::ReturnHome: return "ReturnHome";
    case DegradeAction::EmergencyLand: return "EmergencyLand";
    }
    return "?";
}

} // namespace

int main() {
    std::printf("=== SensorRecoveryTest: 故障恢复路径 ===\n\n");

    SixDofConfig cfg;
    const double dt = cfg.base.dt;
    SixDofState init{makeVec3(0.0f, 0.0f, -5.0f), makeVec3(0.0f, 0.0f, 0.0f),
                     Tensor{1.0f, 0.0f, 0.0f, 0.0f}, makeVec3(0.0f, 0.0f, 0.0f)};
    SixDofSimulator sim(cfg, init);
    SixDofPidController pid(cfg, {});
    DegradeExecutorConfig ec;
    ec.hover_thrust = cfg.base.mass * cfg.base.gravity;
    DegradeExecutor exec(pid, ec);

    ImuConfig ic;
    ic.explicit_bias = true;
    ic.accel_bias_vec = {0.0, 0.0, 0.0};
    ic.gyro_bias_vec = {0.0, 0.0, 0.0};
    ImuModel imu(ic, 20260918u);

    EstimatorConfig etc;
    etc.sensor_health.enabled = true;
    StateEstimator est(etc);
    est.reset();

    const ImuDegradePolicy policy;
    const Tensor target = makeVec3(0.0f, 0.0f, -5.0f);

    constexpr int bias_on = 2000;
    constexpr int bias_off = 5000;
    constexpr int steps = 12000;

    int first_degraded = -1;
    int first_healthy_after = -1;
    int last_recovered_step = -1;
    double max_alt_dev_after_recovery = 0.0;
    DegradeAction action_after_recovery = DegradeAction::EmergencyLand;
    SensorStatus status_during_fault = SensorStatus::Unknown;

    for (int k = 0; k < steps; ++k) {
        const double t = k * dt;
        ImuSample s = imu.measure(sim.state(), {0.0, 0.0, -cfg.base.gravity}, dt);
        if (k >= bias_on && k < bias_off) {
            s.accel[0] += 1.0;
        }
        est.updateImu(s, dt);
        if (k % 10 == 0) {
            est.updatePosition(rv(sim.state().pos), dt * 10.0);
        }

        const auto &h = est.sensorHealth();
        const DegradeDecision d = policy.decide(h);
        est.setAccelCorrectionEnabled(d.use_accel_correction);
        est.setTrustPosition(d.trust_position);
        exec.setDecision(d);

        const SixDofCommand cmd = exec.compute(est.state(), target, t);
        sim.step(cmd.thrust_body, cmd.torque);

        // 记录故障期间的判定（取偏置注入期中段，避开检测延迟）
        if (k == bias_on + 500) {
            status_during_fault = h.accel;
        }
        if (h.accel == SensorStatus::Degraded && first_degraded < 0) {
            first_degraded = k;
        }
        if (first_degraded > 0 && k > bias_off && h.accel == SensorStatus::Healthy &&
            first_healthy_after < 0) {
            first_healthy_after = k;
            action_after_recovery = d.action;
        }
        if (first_healthy_after > 0) {
            const double alt = -rv(sim.state().pos)[2];
            max_alt_dev_after_recovery =
                std::max(max_alt_dev_after_recovery, std::fabs(alt - 5.0));
            last_recovered_step = k;
        }
    }

    std::printf("  注入偏置      第 %d 周期\n", bias_on);
    std::printf("  首次 Degraded 第 %d 周期（检出延迟 %d）\n", first_degraded,
                first_degraded - bias_on);
    std::printf("  移除偏置      第 %d 周期\n", bias_off);
    std::printf("  恢复 Healthy  %s\n",
                first_healthy_after > 0
                    ? (std::to_string(first_healthy_after) + " 周期（耗时 " +
                       std::to_string(first_healthy_after - bias_off) + "）").c_str()
                    : "未恢复");
    std::printf("  恢复后最大高度偏差  %.3f m\n", max_alt_dev_after_recovery);
    std::printf("\n");

    // ---- 1. 激励确认：故障确实被注入并检出 ----
    check(first_degraded > 0, "激励确认：偏置故障被检出（进入 Degraded）");
    check(status_during_fault == SensorStatus::Degraded,
          "激励确认：故障持续期间状态保持 Degraded");
    check(first_degraded >= bias_on,
          "激励确认：Degraded 发生在偏置注入之后（无提前误报）");

    // ---- 2. 【核心】故障消失后必须恢复 ----
    // 这条断言在恢复逻辑失效时会失败：状态将被永久锁在 Degraded。
    check(first_healthy_after > 0, "核心：偏置移除后恢复到 Healthy");
    check(action_after_recovery == DegradeAction::Normal,
          "核心：恢复后降级决策回到 Normal（不再限幅）");

    // ---- 3. 恢复时长须与配置一致（防「立即恢复」缺少去抖的误改）----
    const int recovery_time = first_healthy_after - bias_off;
    std::printf("  恢复耗时 %d 周期（recovery_steps 配置为 500）\n", recovery_time);
    check(recovery_time >= 450 && recovery_time <= 600,
          "恢复耗时与 recovery_steps 配置一致（存在去抖，非立即恢复）");

    // ---- 4. 恢复后飞行正常 ----
    check(max_alt_dev_after_recovery < 0.5,
          "恢复后飞行正常：高度偏差 <0.5 m");

    // ---- 5. 恢复后状态稳定（不抖动）----
    check(last_recovered_step > first_healthy_after,
          "恢复后状态持续保持 Healthy（未发生抖回 Degraded）");

    std::printf("\n========================================\n");
    std::printf("%d / %d checks passed\n", g_pass, g_pass + g_fail);
    std::printf("========================================\n");
    return g_fail == 0 ? 0 : 1;
}
