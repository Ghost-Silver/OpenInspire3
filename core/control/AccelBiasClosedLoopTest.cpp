/**
 * @file AccelBiasClosedLoopTest.cpp
 * @brief 加速度计偏置：检测后关闭方向反馈是否改善闭环状态
 *
 * 本测试验证一条此前缺失的因果链：
 *   加速度计偏置注入 → SensorHealth 检出 Degraded →
 *   ImuDegradePolicy 要求关闭方向反馈 → StateEstimator 实际停止反馈，
 *   同时继续更新健康检测 → 估计状态不再被错误方向持续污染。
 *
 * 只验证最终姿态误差是不够的：如果检测根本没有触发，或者开关没有被消费，
 * 结果可能只是偶然较好。因此测试同时记录故障注入、检测时刻、策略输出、
 * 开关状态，以及检测后的估计误差。
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
#include <array>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace oi3;

namespace {
int g_pass = 0;
int g_fail = 0;

void check(bool condition, const std::string &name) {
    if (condition) {
        std::printf("[ ok ] %s\n", name.c_str());
        ++g_pass;
    } else {
        std::printf("[FAIL] %s\n", name.c_str());
        ++g_fail;
    }
}

std::array<double, 3> readVec(const Tensor &tensor) {
    const auto values = toVector(tensor);
    return {values[0], values[1], values[2]};
}

double attitudeErrorDeg(const std::array<double, 4> &estimated, const Tensor &truth) {
    const auto values = toVector(truth);
    double dot = 0.0;
    for (int i = 0; i < 4; ++i) {
        dot += estimated[static_cast<std::size_t>(i)] * values[static_cast<std::size_t>(i)];
    }
    return 2.0 * std::acos(std::min(1.0, std::fabs(dot))) * 180.0 / M_PI;
}

struct RunResult {
    int detect_step = -1;
    double max_error_before_detection = 0.0;
    double max_error_after_detection = 0.0;
    double final_error = 0.0;
    double max_horizontal_error = 0.0;
    double final_altitude = 0.0;
    bool correction_was_disabled = false;
    SensorHealthReport health;
    DegradeDecision decision;
};

enum class Mode { NoDegrade, ApplyPolicy };

RunResult run(Mode mode, double accel_bias, int steps = 8000) {
    SixDofConfig config;
    SixDofState initial{makeVec3(0.0f, 0.0f, -5.0f), makeVec3(0.0f, 0.0f, 0.0f),
                        Tensor{1.0f, 0.0f, 0.0f, 0.0f}, makeVec3(0.0f, 0.0f, 0.0f)};
    SixDofSimulator simulator(config, initial);
    SixDofPidController pid(config, {});

    DegradeExecutorConfig executor_config;
    executor_config.hover_thrust = config.base.mass * config.base.gravity;
    DegradeExecutor executor(pid, executor_config);

    ImuConfig imu_config;
    imu_config.explicit_bias = true;
    imu_config.accel_bias_vec = {0.0, 0.0, 0.0};
    imu_config.gyro_bias_vec = {0.0, 0.0, 0.0};
    ImuModel imu(imu_config, 20260918u);

    EstimatorConfig estimator_config;
    estimator_config.sensor_health.enabled = true;
    StateEstimator estimator(estimator_config);
    estimator.reset();

    const ImuDegradePolicy policy;
    const Tensor target = makeVec3(0.0f, 0.0f, -5.0f);
    constexpr int fault_start = 2000;
    RunResult result;

    for (int step = 0; step < steps; ++step) {
        const double dt = config.base.dt;
        const double time = step * dt;
        ImuSample sample = imu.measure(simulator.state(), {0.0, 0.0, -config.base.gravity}, dt);
        if (step >= fault_start) {
            sample.accel[0] += accel_bias;
        }

        estimator.updateImu(sample, dt);
        if (step % 10 == 0) {
            estimator.updatePosition(readVec(simulator.state().pos), dt * 10.0);
        }

        const DegradeDecision decision = policy.decide(estimator.sensorHealth());
        if (mode == Mode::ApplyPolicy) {
            estimator.setAccelCorrectionEnabled(decision.use_accel_correction);
            estimator.setTrustPosition(decision.trust_position);
            executor.setDecision(decision);
            if (!decision.use_accel_correction) {
                result.correction_was_disabled = true;
            }
        } else {
            DegradeDecision nominal = decision;
            nominal.action = DegradeAction::Normal;
            nominal.use_accel_correction = true;
            nominal.trust_position = true;
            executor.setDecision(nominal);
        }

        const SixDofCommand command = executor.compute(estimator.state(), target, time);
        simulator.step(command.thrust_body, command.torque);

        const double error = attitudeErrorDeg(estimator.attitude(), simulator.state().quat);
        if (result.detect_step < 0 && estimator.sensorHealth().accel == SensorStatus::Degraded) {
            result.detect_step = step;
        }
        if (result.detect_step < 0) {
            result.max_error_before_detection = std::max(result.max_error_before_detection, error);
        } else {
            result.max_error_after_detection = std::max(result.max_error_after_detection, error);
        }
        result.final_error = error;

        const auto position = readVec(simulator.state().pos);
        result.max_horizontal_error =
            std::max(result.max_horizontal_error, std::hypot(position[0], position[1]));
        result.final_altitude = -position[2];
    }

    result.health = estimator.sensorHealth();
    result.decision = policy.decide(result.health);
    return result;
}
} // namespace

int main() {
    std::printf("=== AccelBiasClosedLoopTest: 偏置检测后的闭环降级 ===\n\n");
    constexpr double bias = 1.0;
    const RunResult baseline = run(Mode::NoDegrade, bias);
    const RunResult degraded = run(Mode::ApplyPolicy, bias);

    std::printf("基线：detect=%.3f s, final_est_error=%.3f deg, max_xy=%.3f m\n",
                baseline.detect_step * 0.001, baseline.final_error, baseline.max_horizontal_error);
    std::printf("降级：detect=%.3f s, final_est_error=%.3f deg, max_xy=%.3f m, correction=%s\n\n",
                degraded.detect_step * 0.001, degraded.final_error, degraded.max_horizontal_error,
                degraded.correction_was_disabled ? "OFF" : "ON");

    check(degraded.detect_step >= 2000, "故障只在注入后被检测（未提前误报）");
    check(degraded.health.accel == SensorStatus::Degraded,
          "最终健康状态仍为 Degraded（关闭反馈没有关闭监测）");
    check(degraded.decision.action == DegradeAction::Cautious,
          "偏置故障决策为 Cautious");
    check(!degraded.decision.use_accel_correction,
          "偏置故障决策要求关闭加速度计方向反馈");
    check(degraded.correction_was_disabled,
          "StateEstimator 实际执行了关闭方向反馈");
    check(degraded.final_error < baseline.final_error * 0.5,
          "降级后最终姿态估计误差至少降低一半");
    check(degraded.max_horizontal_error < 1.0,
          "降级后水平位置误差保持在 1 m 内");
    check(degraded.final_altitude > 3.0 && degraded.final_altitude < 8.0,
          "降级后高度保持在合理范围");

    std::printf("\n========================================\n");
    std::printf("%d / %d checks passed\n", g_pass, g_pass + g_fail);
    std::printf("========================================\n");
    return g_fail == 0 ? 0 : 1;
}
