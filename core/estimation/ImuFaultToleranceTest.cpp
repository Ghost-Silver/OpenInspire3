/**
 * @file ImuFaultToleranceTest.cpp
 * @brief 端到端：注入真实 IMU 故障 → 检测 → 降级决策
 *
 * @par 为什么需要这个文件（而不是只测两个模块）
 *
 * 本项目有一条反复被验证的教训：**单个模块正确不能推出组合正确**。
 * 实测过两个各自正确的补偿机制叠加后，稳态误差比完全不补偿还差。
 *
 * 本测试因此把整条链接起来验证：故障注入 → StateEstimator → SensorHealth
 * → ImuDegradePolicy → 最终动作。只有这样才能发现「检测对了但决策错了」
 * 或「决策对了但检测根本没触发」这类跨模块问题。
 *
 * @par 最关心的一条链
 *
 * 陀螺失效 → EmergencyLand。这是安全关键的最后一环：若这条链断在任意一处
 * （检测没触发、状态映射错、决策给了返航），真机上就是翻机。
 * 因此本测试对这条链做了**双重确认**：既断言检测状态，也断言最终动作，
 * 并额外断言「动作不是 ReturnHome」——因为返航需要持续姿态机动，
 * 而姿态环正是陀螺失效后失去的东西。
 *
 * @par 关于「激励确认」
 *
 * 每个场景先断言故障确实改变了系统行为，再断言检测结果与决策。
 * 这一步防止「故障没注入成功却以为检测通过了」——本项目已有多次
 * 此类教训（水平风激发不到轴向入流、把物理开关设为 0 而关掉效应本身、
 * 差分跨段边界、验证脚本系数写错）。
 */

#include "ImuDegradePolicy.h"
#include "ImuModel.h"
#include "SensorHealth.h"
#include "SixDofTypes.h"
#include "StateEstimator.h"
#include "DroneTypes.h"

#include <array>
#include <cmath>
#include <cstdio>
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

double attErrorDeg(const std::array<double, 4> &qe, const std::array<double, 4> &qt) {
    const double we = qe[0], xe = qe[1], ye = qe[2], ze = qe[3];
    const double wt = qt[0], xt = -qt[1], yt = -qt[2], zt = -qt[3];
    const double w = we * wt - xe * xt - ye * yt - ze * zt;
    return 2.0 * std::acos(std::min(1.0, std::fabs(w))) * 180.0 / M_PI;
}

enum class Fault { None, AccelZero, AccelBias, GyroZero };

/// 端到端运行：返回最终的健康报告、决策与观测
struct E2EOut {
    SensorHealthReport health;
    DegradeDecision decision;
    double max_att_err_deg = 0.0;
    double min_accel_mag = 1e9;
};

E2EOut runE2E(Fault fault, bool maneuvering, int fault_start = 8000, int steps = 20000) {
    ImuConfig icfg;
    icfg.explicit_bias = true;
    icfg.accel_bias_vec = {0.0, 0.0, 0.0};
    icfg.gyro_bias_vec = {0.0, 0.0, 0.0};
    ImuModel imu(icfg, 2024u);

    EstimatorConfig ecfg;
    ecfg.sensor_health.enabled = true;
    StateEstimator est(ecfg);
    est.reset();

    const ImuDegradePolicy policy;
    const double dt = 0.001;
    E2EOut out;

    for (int k = 0; k < steps; ++k) {
        const double t = k * dt;
        const double roll_true =
            maneuvering ? (15.0 * M_PI / 180.0) * std::sin(2.0 * M_PI * 0.5 * t) : 0.0;
        const double cr = std::cos(roll_true / 2.0), sr = std::sin(roll_true / 2.0);
        const std::array<double, 4> q_true = {cr, sr, 0.0, 0.0};
        const double w_true =
            maneuvering
                ? (15.0 * M_PI / 180.0) * 2.0 * M_PI * 0.5 * std::cos(2.0 * M_PI * 0.5 * t)
                : 0.0;

        Tensor q_t = Tensor{static_cast<float>(q_true[0]), static_cast<float>(q_true[1]),
                            static_cast<float>(q_true[2]), static_cast<float>(q_true[3])};
        Tensor om_t = makeVec3(static_cast<float>(w_true), 0.0f, 0.0f);
        Tensor z3 = makeVec3(0.0f, 0.0f, 0.0f);
        const SixDofState truth{q_t, z3, q_t, om_t};

        ImuSample s = imu.measure(truth, {0.0, 0.0, -9.81}, dt);
        if (k >= fault_start) {
            if (fault == Fault::AccelZero) {
                s.accel = {0.0, 0.0, 0.0};
            } else if (fault == Fault::AccelBias) {
                s.accel = {s.accel[0] + 1.0, s.accel[1], s.accel[2]};
            } else if (fault == Fault::GyroZero) {
                s.gyro = {0.0, 0.0, 0.0};
            }
        }
        est.updateImu(s, dt);

        const double ae = attErrorDeg(est.attitude(), q_true);
        out.max_att_err_deg = std::max(out.max_att_err_deg, ae);
        const double am = std::sqrt(s.accel[0] * s.accel[0] + s.accel[1] * s.accel[1] +
                                    s.accel[2] * s.accel[2]);
        out.min_accel_mag = std::min(out.min_accel_mag, am);
    }

    out.health = est.sensorHealth();
    out.decision = policy.decide(out.health);
    return out;
}

} // namespace

int main() {
    std::printf("=== ImuFaultToleranceTest: 端到端（注入 → 检测 → 决策）===\n\n");

    // ================================================================
    std::printf("--- 1. 基线：正常飞行必须放行 ---\n");
    // ================================================================
    {
        const auto r = runE2E(Fault::None, false);
        check(r.health.accel == SensorStatus::Healthy && r.health.gyro == SensorStatus::Healthy,
              "基线：双传感器 Healthy");
        check(r.decision.action == DegradeAction::Normal, "基线：决策 Normal（不误触发降级）");
        std::printf("       基线姿态误差峰值 = %.4f 度\n", r.max_att_err_deg);
    }

    // ================================================================
    std::printf("\n--- 2. 【安全关键】陀螺归零 → 必须紧急降落 ---\n");
    // ================================================================
    {
        // 机动场景下危害才显现，故用机动
        const auto r = runE2E(Fault::GyroZero, true);
        // 激励确认
        check(r.max_att_err_deg > 5.0, "激励确认：故障使姿态误差显著增大");
        // 检测
        check(r.health.gyro == SensorStatus::Failed, "检测：陀螺 Failed");
        // 决策（双重确认）
        check(r.decision.action == DegradeAction::EmergencyLand, "决策：EmergencyLand");
        check(r.decision.action != DegradeAction::ReturnHome,
              "决策：绝不能是 ReturnHome（返航需姿态机动，会翻）");
        check(r.decision.max_tilt_deg == 0.0, "决策：停止一切姿态机动");
        std::printf("       姿态误差峰值 = %.3f 度，动作 = EmergencyLand\n", r.max_att_err_deg);
    }

    // ================================================================
    std::printf("\n--- 3. 加速度计归零 → 返航（而非紧急降落）---\n");
    // ================================================================
    {
        const auto r = runE2E(Fault::AccelZero, false);
        check(r.min_accel_mag < 0.01, "激励确认：加速度计模长趋零");
        check(r.health.accel == SensorStatus::Failed, "检测：加速度计 Failed");
        check(r.decision.action == DegradeAction::ReturnHome, "决策：ReturnHome");
        // 关键区分：加速度计失效**不该**升级为紧急降落
        // —— 姿态仍可由陀螺积分维持，这是分钟级退化
        check(r.decision.action != DegradeAction::EmergencyLand,
              "决策：不升级为 EmergencyLand（姿态仍可由陀螺维持）");
        check(!r.decision.use_accel_correction, "决策：停止加速度计校正");
        std::printf("       动作 = ReturnHome，加速度计校正已关闭\n");
    }

    // ================================================================
    std::printf("\n--- 4. 加速度计偏置「说谎」 → 谨慎限幅 ---\n");
    // ================================================================
    {
        const auto base = runE2E(Fault::None, false);
        const auto r = runE2E(Fault::AccelBias, false);
        check(r.max_att_err_deg > 10.0 * base.max_att_err_deg,
              "激励确认：偏置造成姿态误差比基线大一个量级");
        check(r.health.accel == SensorStatus::Degraded, "检测：加速度计 Degraded");
        check(r.decision.action == DegradeAction::Cautious, "决策：Cautious（限幅）");
        check(r.decision.max_tilt_deg < 35.0, "决策：倾角上限被压低");
        std::printf("       姿态误差峰值 = %.3f 度（基线 %.4f），动作 = Cautious\n",
                    r.max_att_err_deg, base.max_att_err_deg);
    }

    // ================================================================
    std::printf("\n--- 5. 默认关闭时：既有行为不得改变 ---\n");
    // ================================================================
    {
        ImuConfig icfg;
        icfg.explicit_bias = true;
        ImuModel imu(icfg, 2024u);
        EstimatorConfig ecfg; // sensor_health.enabled 默认 false
        StateEstimator est(ecfg);
        est.reset();
        const Tensor q = Tensor{1.0f, 0.0f, 0.0f, 0.0f};
        const Tensor z3 = makeVec3(0.0f, 0.0f, 0.0f);
        const SixDofState truth{q, z3, q, z3};
        for (int k = 0; k < 10000; ++k) {
            const ImuSample s = imu.measure(truth, {0.0, 0.0, -9.81}, 0.001);
            est.updateImu(s, 0.001);
        }
        const ImuDegradePolicy policy;
        const auto d = policy.decide(est.sensorHealth());
        check(est.sensorHealth().samples == 0, "默认关闭：不积累样本");
        // Unknown 应给出 Cautious 而非 Normal —— 无证据时不假设健康
        check(d.action == DegradeAction::Cautious,
              "默认关闭：决策为 Cautious（无证据不假设健康）");
    }

    std::printf("\n========================================\n");
    std::printf("%d / %d checks passed\n", g_pass, g_pass + g_fail);
    std::printf("========================================\n");
    return (g_fail == 0) ? 0 : 1;
}
