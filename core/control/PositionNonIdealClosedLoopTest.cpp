/**
 * @file PositionNonIdealClosedLoopTest.cpp
 * @brief 位置量测非理想（噪声 + 延迟）下的降级策略消融验证
 *
 * @par 它回答的问题
 *
 * 此前所有降级闭环对照都用**理想位置量测**（每 10 步取真值，无噪声、无延迟）。
 * 真机的 GPS/光流既有噪声又有延迟，三个既有结论可能在非理想量测下不再成立：
 *
 *   1. 「加速度计偏置能在 ~0.25 s 内检出」（理想量测下 2.0 s 注入、2.246 s 检出）；
 *   2. 「关闭方向反馈」改善偏置污染（理想量测下 5.197° → 1.113°）；
 *   3. 「加速度计失效后保留位置预积分」（初版关闭预积分让高度从 5.35 m 飘到
 *      16.14 m）—— 该结论的对照组用的是理想量测。
 *
 * 本测试把位置量测换成「噪声 + 延迟」模型重跑闭环对照，并做四组消融：
 *   - trust_position：保留预积分（现策略）vs 关闭预积分（被推翻的初版）；
 *   - 偏置检测的机动门限：默认 0.05 rad/s vs 放宽到 0.5 rad/s，分离
 *     「检测灵敏度」与「策略有效性」两个问题；
 *   - 放宽门限的误报检查：无故障 + 目标切换瞬态下不得误报偏置
 *     （低角速度高加速度阶段的残差抬升是运动不是故障）；
 *   - 稳定包络归因：噪声单独 / 延迟单独 / 两者叠加（无故障），确认
 *     非理想量测下闭环失稳（若有）的因果来源。
 *
 * @par 为什么场景里必须有机动
 *
 * 纯悬停时位置近似不动，p(t−τ) ≈ p(t)，**延迟几乎不产生激励** —— 那样跑出的
 * 「延迟无害」是激励没到达机制，不是真的无害（本项目铁律之二）。因此场景在
 * 4.0 s 时把目标点从 (0,0,−5) 切到 (2,0,−5)：飞行中 v≠0，延迟量测系统性落后
 * v·τ，激励才真正到达校正回路。
 *
 * @par 量测模型
 *
 *   pos_meas(k) = pos_truth(k − delay) + N(0, σ²)（逐轴独立）
 *
 * 估计器仍按 100 Hz 收到量测并认为它是「当前」的 —— 即量测**内容陈旧**而
 * 时间戳被认为是现在的，这是廉价 GPS 的典型失效形式。噪声序列由固定种子的
 * mt19937 生成，对照组之间使用同一种子，保证逐拍同噪声（配对比较）。
 *
 * 两档参数：
 *   - 光流级：σ = 0.05 m，延迟 20 ms
 *   - GPS 级：σ = 0.30 m，延迟 100 ms
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
#include <deque>
#include <random>
#include <string>

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

/// 带噪声与延迟的位置量测模型（见文件头）。
/// 同时记录「激励是否真的到达」所需的三个量：注入噪声的实测 RMS、
/// 吃到陈旧样本的次数、以及陈旧样本与当前真值的最大落后距离。
///
/// 注意单位：构造参数 delay 以 IMU 步给出（与场景描述一致），内部按
/// 量测周期（每 10 个 IMU 步一次）换算成样本数。若把 IMU 步数直接当
/// 样本数用，延迟会被放大 10 倍 —— 初版就踩了这个坑。
class PositionSensor {
  public:
    PositionSensor(double sigma, int delay_imu_steps, uint32_t seed)
        : _sigma(sigma), _delay_samples(std::max(0, delay_imu_steps) / 10), _rng(seed) {}

    /// 推进一步真值并返回量测：延迟样本 + 高斯噪声
    std::array<double, 3> measure(const std::array<double, 3> &truth) {
        _history.push_back(truth);
        const std::size_t lag =
            _history.size() > static_cast<std::size_t>(_delay_samples)
                ? static_cast<std::size_t>(_delay_samples)
                : _history.size() - 1;
        const auto &src = _history[_history.size() - 1 - lag];
        if (_history.size() > static_cast<std::size_t>(_delay_samples)) {
            ++_stale_served; // 真正吃到延迟样本的次数
        }
        // 陈旧样本与当前真值的距离：延迟激励的直接量度。
        // 静止时 ≈0（延迟不可观测），机动时 ≈ v·τ。
        const double d_lag = std::sqrt(
            (src[0] - truth[0]) * (src[0] - truth[0]) +
            (src[1] - truth[1]) * (src[1] - truth[1]) +
            (src[2] - truth[2]) * (src[2] - truth[2]));
        _max_lag = std::max(_max_lag, d_lag);

        std::array<double, 3> meas{};
        for (int i = 0; i < 3; ++i) {
            const double n = _sigma * _gauss(_rng);
            _injected_sq += n * n;
            ++_injected_n;
            meas[static_cast<std::size_t>(i)] =
                src[static_cast<std::size_t>(i)] + n;
        }
        return meas;
    }

    [[nodiscard]] long long staleServed() const { return _stale_served; }
    [[nodiscard]] double maxLag() const { return _max_lag; }
    /// 注入噪声的实测 RMS（直接统计噪声抽样本身，不与延迟落后混淆）
    [[nodiscard]] double injectedNoiseRms() const {
        return _injected_n > 0 ? std::sqrt(_injected_sq / _injected_n) : 0.0;
    }

  private:
    double _sigma;
    int _delay_samples; ///< 延迟（位置样本数），= IMU 步数 / 10
    std::mt19937 _rng;
    std::normal_distribution<double> _gauss{0.0, 1.0};
    std::deque<std::array<double, 3>> _history;
    long long _stale_served = 0;
    double _max_lag = 0.0;
    double _injected_sq = 0.0;
    long long _injected_n = 0;
};

struct RunResult {
    int detect_step = -1;
    double final_att_error = 0.0;   ///< 末态姿态估计误差（度）
    double max_tilt_true = 0.0;     ///< 真值倾角峰值（度）
    double final_alt = 0.0;         ///< 末态高度（m）
    double final_target_err = 0.0;  ///< 末态相对当前目标的水平误差（m）
    double max_xy_from_origin = 0.0;///< 相对起飞点的最大水平距离（m）
    bool crashed = false;
    bool correction_was_disabled = false;
    bool trust_position_final = true;
    // 量测激励记录（铁律之二：断言前先确认激励到达了机制）
    double sensor_noise_rms = 0.0;     ///< 实际注入噪声的 RMS（m）
    long long sensor_stale_served = 0; ///< 吃到陈旧量测的次数
    double sensor_max_lag = 0.0;       ///< 陈旧量测与当前真值的最大落后（m）
    // 检测窗口诊断（故障注入后 1 s 内）：用于检测灵敏度问题的因果归因
    double maneuver_fraction_window = 0.0; ///< 机动标记为真的时间占比
    double residual_mean_window = 0.0;     ///< 方向残差均值
    SensorHealthReport health;
    DegradeDecision decision;
};

enum class FaultKind { None, AccelBias, AccelDead };
enum class Mode {
    NoDegrade,            ///< 检出也不执行（对照）
    ApplyPolicy,          ///< 决策同时作用于估计器与执行器（现策略）
    ApplyPolicyNoPreinteg ///< 同 ApplyPolicy，但强制关闭位置预积分（初版被推翻的设计）
};

struct SensorSpec {
    double sigma;
    int delay_steps;
    const char *label;
};

struct RunOptions {
    /// 覆盖 SensorHealthConfig::maneuver_gyro；<0 表示用默认值 0.05
    double maneuver_gyro_override = -1.0;
};

RunResult run(FaultKind fault, Mode mode, const SensorSpec &spec, uint32_t seed,
              const RunOptions &opts = {}, int steps = 10000) {
    SixDofConfig config;
    const double dt = config.base.dt;
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
    if (opts.maneuver_gyro_override >= 0.0) {
        estimator_config.sensor_health.maneuver_gyro = opts.maneuver_gyro_override;
    }
    StateEstimator estimator(estimator_config);
    estimator.reset();

    PositionSensor sensor(spec.sigma, spec.delay_steps, seed);
    const ImuDegradePolicy policy;

    constexpr int fault_start = 2000;   // 2.0 s 注入故障
    constexpr int target_switch = 4000; // 4.0 s 目标点切换，激发延迟
    const Tensor target_a = makeVec3(0.0f, 0.0f, -5.0f);
    const Tensor target_b = makeVec3(2.0f, 0.0f, -5.0f);

    RunResult result;
    long long window_steps = 0;
    long long window_maneuver = 0;
    double window_residual_sum = 0.0;

    for (int step = 0; step < steps; ++step) {
        const double time = step * dt;
        ImuSample sample = imu.measure(simulator.state(), {0.0, 0.0, -config.base.gravity}, dt);
        if (fault != FaultKind::None && step >= fault_start) {
            if (fault == FaultKind::AccelBias) {
                sample.accel[0] += 1.0; // +1.0 m/s² 偏置，与既有闭环测试一致
            } else {
                sample.accel = {0.0, 0.0, 0.0}; // 彻底失效
            }
        }

        estimator.updateImu(sample, dt);
        if (step % 10 == 0) {
            const auto truth = readVec(simulator.state().pos);
            const auto meas = sensor.measure(truth);
            estimator.updatePosition(meas, dt * 10.0);
        }

        const Tensor &target = step < target_switch ? target_a : target_b;

        const DegradeDecision decision = policy.decide(estimator.sensorHealth());
        if (mode == Mode::NoDegrade) {
            DegradeDecision nominal = decision;
            nominal.action = DegradeAction::Normal;
            nominal.use_accel_correction = true;
            nominal.trust_position = true;
            executor.setDecision(nominal);
        } else {
            estimator.setAccelCorrectionEnabled(decision.use_accel_correction);
            estimator.setTrustPosition(decision.trust_position);
            executor.setDecision(decision);
            if (mode == Mode::ApplyPolicyNoPreinteg) {
                estimator.setTrustPosition(false); // 消融：强制初版设计
            }
            if (!decision.use_accel_correction) {
                result.correction_was_disabled = true;
            }
        }

        const SixDofCommand command = executor.compute(estimator.state(), target, time);
        simulator.step(command.thrust_body, command.torque);

        const double error = attitudeErrorDeg(estimator.attitude(), simulator.state().quat);
        result.final_att_error = error;
        if (result.detect_step < 0 && fault != FaultKind::None) {
            const bool detected = fault == FaultKind::AccelBias
                                      ? estimator.sensorHealth().accel == SensorStatus::Degraded
                                      : estimator.sensorHealth().accel == SensorStatus::Failed;
            if (detected) {
                result.detect_step = step;
            }
        }
        // 检测窗口诊断：故障注入后 1 s
        if (fault != FaultKind::None && step >= fault_start && step < fault_start + 1000) {
            ++window_steps;
            if (estimator.sensorHealth().maneuvering) {
                ++window_maneuver;
            }
            window_residual_sum += estimator.sensorHealth().residual;
        }

        const auto position = readVec(simulator.state().pos);
        result.max_xy_from_origin =
            std::max(result.max_xy_from_origin, std::hypot(position[0], position[1]));
        result.final_alt = -position[2];
        const double tx = step < target_switch ? 0.0 : 2.0;
        result.final_target_err = std::hypot(position[0] - tx, position[1]);

        const auto q = toVector(simulator.state().quat);
        const double r33 = 1.0 - 2.0 * (static_cast<double>(q[1]) * q[1] +
                                        static_cast<double>(q[2]) * q[2]);
        const double tilt = std::acos(std::max(-1.0, std::min(1.0, r33))) * 180.0 / M_PI;
        result.max_tilt_true = std::max(result.max_tilt_true, tilt);
        if (tilt > 60.0) {
            result.crashed = true;
        }
    }

    result.health = estimator.sensorHealth();
    result.decision = policy.decide(result.health);
    result.trust_position_final = estimator.trustPosition();
    result.sensor_noise_rms = sensor.injectedNoiseRms();
    result.sensor_stale_served = sensor.staleServed();
    result.sensor_max_lag = sensor.maxLag();
    if (window_steps > 0) {
        result.maneuver_fraction_window =
            static_cast<double>(window_maneuver) / static_cast<double>(window_steps);
        result.residual_mean_window = window_residual_sum / static_cast<double>(window_steps);
    }
    return result;
}

void printRow(const char *label, const RunResult &r, double dt) {
    std::printf("%-26s detect=%6.3fs att_err=%7.3f° tilt=%6.2f° alt=%7.2fm "
                "tgt_err=%6.2fm lag=%5.3fm%s\n",
                label, r.detect_step * dt, r.final_att_error, r.max_tilt_true,
                r.final_alt, r.final_target_err, r.sensor_max_lag,
                r.crashed ? "  [失控]" : "");
}

} // namespace

int main() {
    std::printf("=== PositionNonIdealClosedLoopTest: 非理想位置量测下的降级消融 ===\n\n");
    std::printf("场景：5 m 悬停，2.0 s 注入故障，4.0 s 目标点切到 (2,0,−5) 激发延迟，\n");
    std::printf("      全程 10 s。位置量测 = 100 Hz 真值 + 噪声 + 延迟。\n\n");

    const SensorSpec tiers[2] = {
        {0.05, 20, "光流级 σ=0.05m τ=20ms"},
        {0.30, 100, "GPS级 σ=0.30m τ=100ms"},
    };
    constexpr uint32_t seed = 20261001u;
    const double dt = SixDofConfig{}.base.dt;

    for (const SensorSpec &spec : tiers) {
        std::printf("────────── %s ──────────\n", spec.label);

        // ---- 无故障基线：量测非理想本身的影响 ----
        const auto base = run(FaultKind::None, Mode::ApplyPolicy, spec, seed);
        printRow("无故障基线", base, dt);

        // ---- 偏置场景：检测灵敏度与策略有效性的分离 ----
        const auto bias_off = run(FaultKind::AccelBias, Mode::NoDegrade, spec, seed);
        const auto bias_pol = run(FaultKind::AccelBias, Mode::ApplyPolicy, spec, seed);
        RunOptions relaxed;
        relaxed.maneuver_gyro_override = 0.5; // 放宽机动门限，隔离「检测」与「策略」
        const auto bias_rel = run(FaultKind::AccelBias, Mode::ApplyPolicy, spec, seed, relaxed);
        // 放宽门限的误报检查：无故障时 4.0 s 目标切换瞬态（低角速度但高加速度的
        // 阶段残差抬升）不得被误判为偏置。这是放宽门限能否成立的守门断言。
        const auto base_rel = run(FaultKind::None, Mode::ApplyPolicy, spec, seed, relaxed);
        printRow("偏置·不降级", bias_off, dt);
        printRow("偏置·策略(默认门限)", bias_pol, dt);
        std::printf("    ↳ 检测窗口诊断：机动标记占比 %.1f%%，残差均值 %.4f\n",
                    bias_pol.maneuver_fraction_window * 100.0, bias_pol.residual_mean_window);
        printRow("偏置·策略(门限0.5)", bias_rel, dt);
        printRow("无故障(门限0.5,误报检查)", base_rel, dt);

        // ---- 失效场景：trust_position 消融 ----
        const auto dead_off = run(FaultKind::AccelDead, Mode::NoDegrade, spec, seed);
        const auto dead_pol = run(FaultKind::AccelDead, Mode::ApplyPolicy, spec, seed);
        const auto dead_nop = run(FaultKind::AccelDead, Mode::ApplyPolicyNoPreinteg, spec, seed);
        printRow("失效·不降级", dead_off, dt);
        printRow("失效·执行策略", dead_pol, dt);
        printRow("失效·关预积分(消融)", dead_nop, dt);
        std::printf("\n");

        // ================================================================
        // 断言 0：激励到达确认（铁律之二）
        // ================================================================
        {
            char buf[200];
            std::snprintf(buf, sizeof(buf),
                          "[%s] 噪声确实注入：实测 RMS %.4f m 与设定 σ %.2f m 一致(±10%%)",
                          spec.label, base.sensor_noise_rms, spec.sigma);
            check(base.sensor_noise_rms > 0.9 * spec.sigma &&
                      base.sensor_noise_rms < 1.1 * spec.sigma,
                  buf);

            const long long total_updates = 1000;
            const long long warmup_updates = spec.delay_steps / 10;
            std::snprintf(buf, sizeof(buf),
                          "[%s] 延迟确实生效：%lld/%lld 次量测吃到陈旧样本",
                          spec.label, base.sensor_stale_served,
                          total_updates - warmup_updates);
            check(base.sensor_stale_served == total_updates - warmup_updates, buf);

            std::snprintf(buf, sizeof(buf),
                          "[%s] 机动发生：无故障基线最大水平位移 %.2f m > 1.5 m",
                          spec.label, base.max_xy_from_origin);
            check(base.max_xy_from_origin > 1.5, buf);
            std::snprintf(buf, sizeof(buf),
                          "[%s] 延迟激励可观测：陈旧量测最大落后 %.3f m > 0",
                          spec.label, base.sensor_max_lag);
            check(base.sensor_max_lag > 1e-3, buf);
        }

        // ================================================================
        // 断言 1：偏置场景
        // ================================================================
        {
            char buf[200];
            // 保守性：无论检测与否，故障注入前不得误报（两档噪声下都要守住）
            std::snprintf(buf, sizeof(buf), "[%s] 偏置注入前无误报（默认门限）", spec.label);
            check(bias_pol.detect_step < 0 || bias_pol.detect_step >= 2000, buf);

            // 因果分离：放宽机动门限后，检测应能触发，且策略仍然有效。
            // 这回答 P1 的核心问题之一——策略有效性是否在非理想量测下保持。
            std::snprintf(buf, sizeof(buf),
                          "[%s] 放宽机动门限后偏置被检出（检测窗口机动占比 %.1f%%）",
                          spec.label, bias_rel.maneuver_fraction_window * 100.0);
            check(bias_rel.detect_step >= 2000 &&
                      bias_rel.health.accel == SensorStatus::Degraded,
                  buf);
            std::snprintf(buf, sizeof(buf),
                          "[%s] 检出后策略关闭方向反馈并实际执行", spec.label);
            check(bias_rel.decision.action == DegradeAction::Cautious &&
                      !bias_rel.decision.use_accel_correction &&
                      bias_rel.correction_was_disabled,
                  buf);
            std::snprintf(buf, sizeof(buf),
                          "[%s] 非理想量测下策略仍有效：误差 %.3f° < 不降级 %.3f° 的一半",
                          spec.label, bias_rel.final_att_error, bias_off.final_att_error);
            check(bias_rel.final_att_error < bias_off.final_att_error * 0.5, buf);

            // 放宽门限的代价边界：无故障全程（含 4.0 s 目标切换瞬态）不得误报偏置。
            // 若此断言失败，说明 0.5 门限把机动瞬态的低角速度高加速度阶段误当成了
            // 「可归因偏置的静止」，放宽门限方案不成立，只能退回默认门限并接受
            // 「噪声量测下偏置漏检」这一限制。
            std::snprintf(buf, sizeof(buf),
                          "[%s] 放宽门限无误报：无故障全程未检出偏置（末态残差 %.4f）",
                          spec.label, base_rel.health.residual);
            check(base_rel.detect_step < 0 && base_rel.health.accel == SensorStatus::Healthy,
                  buf);
        }

        // ================================================================
        // 断言 2：失效场景——降级不劣化 + trust_position 消融
        // ================================================================
        {
            char buf[200];
            std::snprintf(buf, sizeof(buf), "[%s] 失效被检出且决策为 ReturnHome", spec.label);
            check(dead_pol.detect_step >= 2000 && dead_pol.health.accel == SensorStatus::Failed &&
                      dead_pol.decision.action == DegradeAction::ReturnHome,
                  buf);
            std::snprintf(buf, sizeof(buf),
                          "[%s] 现策略保留预积分且估计器确实信任位置", spec.label);
            check(dead_pol.decision.trust_position && dead_pol.trust_position_final, buf);

            // 「降级不劣化」按配对比较断言：同量测条件下，执行策略的各项指标
            // 不得比不降级显著更差。不用绝对阈值——非理想量测本身的劣化
            // （如 GPS 级的闭环失稳）不是降级策略的责任。
            std::snprintf(buf, sizeof(buf),
                          "[%s] 降级不劣化：倾角峰值 %.2f° ≤ 不降级 %.2f° + 5°",
                          spec.label, dead_pol.max_tilt_true, dead_off.max_tilt_true);
            check(dead_pol.max_tilt_true <= dead_off.max_tilt_true + 5.0, buf);
            std::snprintf(buf, sizeof(buf),
                          "[%s] 降级不劣化：高度偏差 %.2f m ≤ 不降级 %.2f m + 1 m",
                          spec.label, std::fabs(dead_pol.final_alt - 5.0),
                          std::fabs(dead_off.final_alt - 5.0));
            check(std::fabs(dead_pol.final_alt - 5.0) <=
                      std::fabs(dead_off.final_alt - 5.0) + 1.0,
                  buf);

            // trust_position 消融：非理想量测下，关闭预积分是否翻转
            // 「保留预积分更好」的结论？理想量测下关闭预积分让高度 5.35→16.14 m。
            const double dev_keep = std::fabs(dead_pol.final_alt - 5.0);
            const double dev_stop = std::fabs(dead_nop.final_alt - 5.0);
            std::snprintf(buf, sizeof(buf),
                          "[%s] 保留预积分不劣于关闭（高度偏差 %.2f m vs %.2f m）",
                          spec.label, dev_keep, dev_stop);
            check(dev_keep <= dev_stop + 0.5, buf);
        }

        // ================================================================
        // 断言 3：无故障基线——光流级量测下闭环应保持悬停品质
        // （GPS 级的稳定性边界在下方包络探针中单独量化）
        // ================================================================
        if (spec.sigma < 0.1) {
            char buf[200];
            std::snprintf(buf, sizeof(buf),
                          "[%s] 光流级量测下悬停稳定：末态高度偏差 %.2f m < 0.5 m",
                          spec.label, std::fabs(base.final_alt - 5.0));
            check(std::fabs(base.final_alt - 5.0) < 0.5, buf);
            std::snprintf(buf, sizeof(buf),
                          "[%s] 光流级量测下能飞到切换后的目标点（末态水平误差 <1 m）",
                          spec.label);
            check(base.final_target_err < 1.0, buf);
        }
        std::printf("\n");
    }

    // ====================================================================
    // 包络探针：GPS 级参数下闭环失稳（若有）的因果归因
    // 噪声单独 / 延迟单独 / 两者叠加，无故障、不降级，同种子配对。
    // 打印全部测量值，并把支撑 P1 结论的两条归因固化为断言。
    // ====================================================================
    std::printf("────────── 包络探针（无故障，GPS 级参数归因）──────────\n");
    const SensorSpec none{0.0, 0, "理想(参照)"};
    const SensorSpec noise_only{0.30, 0, "仅噪声 σ=0.30m"};
    const SensorSpec delay_only{0.0, 100, "仅延迟 τ=100ms"};
    const SensorSpec both{0.30, 100, "噪声+延迟"};
    const RunResult env[4] = {
        run(FaultKind::None, Mode::NoDegrade, none, seed),
        run(FaultKind::None, Mode::NoDegrade, noise_only, seed),
        run(FaultKind::None, Mode::NoDegrade, delay_only, seed),
        run(FaultKind::None, Mode::NoDegrade, both, seed),
    };
    const SensorSpec *specs[4] = {&none, &noise_only, &delay_only, &both};
    for (int i = 0; i < 4; ++i) {
        std::printf("%-18s 末态高度 %8.2f m（偏差 %7.2f m），末态目标误差 %6.2f m%s\n",
                    specs[i]->label, env[i].final_alt, std::fabs(env[i].final_alt - 5.0),
                    env[i].final_target_err, env[i].crashed ? "  [失控]" : "");
    }
    {
        char buf[200];
        // 归因之一：100 ms 延迟单独作用不破坏高度通道。这保证「GPS 级失稳」
        // 的因果可以归给噪声而不是延迟——本测试的延迟模型因此是可分离的。
        std::snprintf(buf, sizeof(buf),
                      "包络归因：仅延迟 τ=100ms 不破坏高度通道（偏差 %.2f m < 0.5 m）",
                      std::fabs(env[2].final_alt - 5.0));
        check(std::fabs(env[2].final_alt - 5.0) < 0.5, buf);

        // 归因之二：σ=0.30 m 噪声单独作用即令高度通道发散。这是当前
        // 估计器 + PID 的稳定包络边界（速度校正增益 b/dt 把 0.3 m 噪声放大成
        // 每拍 ~0.5 m/s 的速度脉冲）。钉住它：将来若改进滤波器使其稳定，
        // 必须连同本断言与文档结论一起更新，而不是悄悄改变行为。
        std::snprintf(buf, sizeof(buf),
                      "包络归因：仅噪声 σ=0.30m 已超出稳定包络（偏差 %.2f m > 10 m）",
                      std::fabs(env[1].final_alt - 5.0));
        check(std::fabs(env[1].final_alt - 5.0) > 10.0, buf);
    }

    std::printf("\n========================================\n");
    std::printf("%d / %d checks passed\n", g_pass, g_pass + g_fail);
    std::printf("========================================\n");
    return g_fail == 0 ? 0 : 1;
}
