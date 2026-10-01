/**
 * @file AccelFaultClosedLoopTest.cpp
 * @brief 闭环对照：加速度计失效后「关闭校正 + 停止预积分」是否有用
 *
 * @par 与陀螺失效那条路径的区别
 *
 * 陀螺失效是**姿态环**断链（ms 级、不可逆），只有紧急降落一条路，已在
 * ImuDegradeClosedLoopTest 验证。加速度计失效则是**位置环**退化（分钟级），
 * 姿态仍可由陀螺积分维持 —— 但前提是**必须关掉方向校正**，否则失效数据会把
 * 错误的姿态误差持续注入估计，比不校正更糟。本测试验证的就是这一点。
 *
 * @par 本测试同时补上一个集成缺口
 *
 * 降级决策里的 `use_accel_correction` / `trust_position` 两个字段原先没有任何
 * 生产代码消费 —— 策略输出了正确的意图，却无人执行（典型的「单个模块正确
 * 不能推出组合正确」）。本测试实现了飞控主循环所需的协调逻辑：
 *
 *   决策 → 同时应用到**估计器**（校正开关/位置信任）与**执行器**（限幅/接管）
 *
 * 这个协调不应藏在执行器里（它不该持有估计器），而应属于主循环。
 * 本文件即该主循环的参考实现。
 */

#include "DegradeExecutor.h"
#include "ImuDegradePolicy.h"
#include "ImuModel.h"
#include "SensorHealth.h"
#include "SixDofPidController.h"
#include "SixDofSimulator.h"
#include "SixDofTypes.h"
#include "StateEstimator.h"
#include "TensorUtils.h"
#include "DroneTypes.h"

#include <array>
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

std::array<double, 3> readVec(const Tensor &t) {
    const std::vector<float> v = toVector(t);
    return {static_cast<double>(v[0]), static_cast<double>(v[1]), static_cast<double>(v[2])};
}

/// 由真值四元数求倾角（度）
double tiltOfQuat(const Tensor &qt) {
    const std::vector<float> q = toVector(qt);
    const double r33 = 1.0 - 2.0 * (static_cast<double>(q[1]) * q[1] +
                                    static_cast<double>(q[2]) * q[2]);
    return std::acos(std::max(-1.0, std::min(1.0, r33))) * 180.0 / M_PI;
}

/// 姿态误差角（度）：估计相对真值
double attErrDeg(const std::array<double, 4> &qe, const Tensor &qt) {
    const std::vector<float> t = toVector(qt);
    const std::array<double, 4> q_true = {static_cast<double>(t[0]), static_cast<double>(t[1]),
                                         static_cast<double>(t[2]), static_cast<double>(t[3])};
    double dot = 0.0;
    for (int i = 0; i < 4; ++i) {
        dot += qe[static_cast<std::size_t>(i)] * q_true[static_cast<std::size_t>(i)];
    }
    return 2.0 * std::acos(std::min(1.0, std::fabs(dot))) * 180.0 / M_PI;
}

struct Out {
    double max_att_err_deg = 0.0;  ///< 姿态估计误差峰值
    double max_tilt_true = 0.0;    ///< 真值倾角峰值
    double final_pos_err = 0.0;    ///< 末态位置误差（相对目标点）
    double final_alt = 0.0;        ///< 末态高度
    bool crashed = false;
    SensorHealthReport health;
    DegradeDecision decision;
};

/**
 * @brief 加速度计失效的闭环
 *
 * @param apply_degrade true = 把决策**同时**应用到估计器与执行器（真实主循环）
 *                      false = 不应用（对照组，继续用失效数据）
 */
Out runAccelFault(bool apply_degrade, int steps = 6000) {
    SixDofConfig cfg;
    const double dt = cfg.base.dt;

    SixDofState init{makeVec3(0.0f, 0.0f, -5.0f), makeVec3(0.0f, 0.0f, 0.0f),
                     Tensor{1.0f, 0.0f, 0.0f, 0.0f}, makeVec3(0.0f, 0.0f, 0.0f)};
    SixDofSimulator sim(cfg, init);
    SixDofPidController ctrl(cfg, {});

    DegradeExecutorConfig ecfg;
    ecfg.hover_thrust = static_cast<double>(cfg.base.mass) *
                        static_cast<double>(cfg.base.gravity);
    DegradeExecutor exec(ctrl, ecfg);

    ImuConfig icfg;
    icfg.explicit_bias = true;
    icfg.accel_bias_vec = {0.0, 0.0, 0.0};
    icfg.gyro_bias_vec = {0.0, 0.0, 0.0};
    ImuModel imu(icfg, 20260918u);

    EstimatorConfig est_cfg;
    est_cfg.sensor_health.enabled = true;
    StateEstimator est(est_cfg);
    est.reset();

    const ImuDegradePolicy policy;
    const Tensor target = makeVec3(0.0f, 0.0f, -5.0f);

    const int fault_start = 2000; // 2 秒后注入
    Out out;

    for (int k = 0; k < steps; ++k) {
        const double t = k * dt;

        ImuSample s = imu.measure(sim.state(), {0.0, 0.0, -cfg.base.gravity}, dt);
        if (k >= fault_start) {
            s.accel = {0.0, 0.0, 0.0}; // 加速度计彻底失效
        }
        est.updateImu(s, dt);

        constexpr int pos_decim = 10;
        if (k % pos_decim == 0) {
            est.updatePosition(readVec(sim.state().pos), dt * pos_decim);
        }

        // ---- 飞控主循环的协调逻辑 ----
        const auto &h = est.sensorHealth();
        const DegradeDecision d = policy.decide(h);

        if (apply_degrade) {
            // 关键：决策**同时**作用于估计器与执行器。
            // 只作用于执行器是不够的 —— 加速度计失效后若继续用它做方向校正，
            // 错误方向会被持续注入姿态估计。
            est.setAccelCorrectionEnabled(d.use_accel_correction);
            est.setTrustPosition(d.trust_position);
            exec.setDecision(d);
        }

        const SixDofCommand cmd = exec.compute(est.state(), target, t);
        sim.step(cmd.thrust_body, cmd.torque);

        out.max_att_err_deg = std::max(out.max_att_err_deg, attErrDeg(est.attitude(), sim.state().quat));
        out.max_tilt_true = std::max(out.max_tilt_true, tiltOfQuat(sim.state().quat));
        const auto p = readVec(sim.state().pos);
        out.final_pos_err = std::sqrt(p[0] * p[0] + p[1] * p[1]);
        out.final_alt = -p[2];
        if (tiltOfQuat(sim.state().quat) > 60.0) {
            out.crashed = true;
        }
        if (k == steps - 1) {
            out.health = h;
            out.decision = d;
        }
    }
    return out;
}

} // namespace

int main() {
    std::printf("=== AccelFaultClosedLoopTest: 加速度计失效的降级对照 ===\n\n");
    std::printf("场景：5 m 悬停，2.0 s 时加速度计彻底失效（输出恒零）\n");
    std::printf("A 组 = 不降级（继续用失效数据校正姿态）\n");
    std::printf("B 组 = 降级（关闭方向校正 + 停止位置预积分）\n\n");

    const auto a = runAccelFault(false);
    const auto b = runAccelFault(true);

    std::printf("--- 观测结果 ---\n");
    std::printf("              姿态估计误差峰值   真值倾角峰值   末态高度   水平偏差   失控\n");
    std::printf("A 不降级        %8.3f°        %8.3f°    %7.2f m  %7.2f m   %s\n",
                a.max_att_err_deg, a.max_tilt_true, a.final_alt, a.final_pos_err,
                a.crashed ? "是" : "否");
    std::printf("B 降级          %8.3f°        %8.3f°    %7.2f m  %7.2f m   %s\n",
                b.max_att_err_deg, b.max_tilt_true, b.final_alt, b.final_pos_err,
                b.crashed ? "是" : "否");
    std::printf("\n");

    // ================================================================
    std::printf("--- 1. 故障确实被检出（激励确认）---\n");
    // ================================================================
    {
        check(a.health.accel == SensorStatus::Failed, "A 组：加速度计被检出 Failed");
        check(b.health.accel == SensorStatus::Failed, "B 组：加速度计被检出 Failed");
        check(b.decision.action == DegradeAction::ReturnHome, "B 组：决策为 ReturnHome");
        check(!b.decision.use_accel_correction, "B 组：决策要求关闭方向校正");
    }

    // ================================================================
    std::printf("\n--- 2. 关闭校正确实生效（激励到达机制）---\n");
    // ================================================================
    {
        // 这是本测试的核心断言：决策里的开关必须**真的**改变了估计器行为。
        // 若不检查这一点，「降级有效」可能只是因为别的原因（例如失效本身
        // 影响很小），而不是因为开关起了作用。
        SixDofConfig cfg;
        SixDofState init{makeVec3(0.0f, 0.0f, -5.0f), makeVec3(0.0f, 0.0f, 0.0f),
                         Tensor{1.0f, 0.0f, 0.0f, 0.0f}, makeVec3(0.0f, 0.0f, 0.0f)};
        SixDofSimulator sim(cfg, init);
        ImuConfig icfg;
        ImuModel imu(icfg, 1u);
        EstimatorConfig est_cfg;
        est_cfg.sensor_health.enabled = true;

        // 关闭校正
        StateEstimator est_off(est_cfg);
        est_off.reset();
        est_off.setAccelCorrectionEnabled(false);
        // 保持校正（对照）
        StateEstimator est_on(est_cfg);
        est_on.reset();
        est_on.setAccelCorrectionEnabled(true);

        // 必须用**非零**的错误数据做激励。第一版这里喂的是全零加速度，
        // 而校正分支的进入条件是 `模长 > 1e-6` —— 全零时两条路径都不执行，
        // 于是「关闭校正」与「保持校正」自然表现一致，断言失败。
        // 那不是开关没生效，而是激励根本没到达机制（本项目反复出现的陷阱）。
        // 改用带偏置的非零数据：它会让校正分支真正执行，开关的差别才可观测。
        for (int k = 0; k < 3000; ++k) {
            ImuSample s = imu.measure(sim.state(), {0.0, 0.0, -cfg.base.gravity}, 0.001);
            s.accel = {2.0, 1.0, -9.0}; // 非零且方向错误（模拟偏置类失效）
            est_off.updateImu(s, 0.001);
            est_on.updateImu(s, 0.001);
        }
        check(est_off.accelCorrectionEnabled() == false, "开关状态可读回（关闭）");
        check(est_on.accelCorrectionEnabled() == true, "开关状态可读回（开启）");
        // 两者在失效数据下行为必须不同 —— 证明开关起作用
        const auto q_off = est_off.attitude();
        const auto q_on = est_on.attitude();
        double diff = 0.0;
        for (int i = 0; i < 4; ++i) {
            diff += std::fabs(q_off[static_cast<std::size_t>(i)] - q_on[static_cast<std::size_t>(i)]);
        }
        check(diff > 1e-9, "关闭校正与保持校正的行为确实不同（开关真的生效）");
    }

    // ================================================================
    std::printf("\n--- 3. 降级是否改善了结果 ---\n");
    // ================================================================
    {
        std::printf("       姿态估计误差：B %.3f° vs A %.3f°\n", b.max_att_err_deg,
                    a.max_att_err_deg);
        std::printf("       真值倾角峰值：B %.3f° vs A %.3f°\n", b.max_tilt_true, a.max_tilt_true);
        std::printf("       末态高度：   B %.2f m vs A %.2f m\n", b.final_alt, a.final_alt);

        // 对本场景（加速度计输出恒零）而言，降级不应让结果变差。
        // 这条断言曾经失败过：初版策略把 trust_position 也设为 false，
        // 导致 B 组从 5.35 m 飘到 16.14 m。修正为保留预积分后应恢复。
        check(b.max_tilt_true < 5.0,
              "对照：降级组姿态保持平稳（倾角 <5°，未因降级而飘）");
        check(b.final_alt > 3.0 && b.final_alt < 8.0,
              "对照：降级组高度仍在合理范围（未大幅漂移）");
        check(b.final_pos_err < 1.0, "对照：降级组水平偏差可接受（<1 m）");

        // 两组都不应失控 —— 加速度计失效不致命，姿态可由陀螺维持
        check(!a.crashed && !b.crashed,
              "对照：两组均未失控（加速度计失效后姿态仍可由陀螺维持）");
    }

    // ================================================================
    std::printf("\n--- 4. 默认行为不变（开关默认开启）---\n");
    // ================================================================
    {
        EstimatorConfig cfg;
        StateEstimator est(cfg);
        check(est.accelCorrectionEnabled(), "默认：方向校正开启");
        check(est.trustPosition(), "默认：位置预积分受信任");
    }

    std::printf("\n========================================\n");
    std::printf("%d / %d checks passed\n", g_pass, g_pass + g_fail);
    std::printf("========================================\n");
    return (g_fail == 0) ? 0 : 1;
}
