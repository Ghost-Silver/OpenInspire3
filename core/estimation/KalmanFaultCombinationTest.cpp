/**
 * @file KalmanFaultCombinationTest.cpp
 * @brief 组合验证：卡尔曼位置滤波器 × IMU 故障降级
 *
 * @par 为什么需要这个测试
 *
 * 既有的三个容错测试（ImuFaultToleranceTest / AccelFaultClosedLoopTest /
 * AccelBiasClosedLoopTest）都只覆盖 AlphaBeta 路径，而位置滤波器与降级开关
 * 存在真实的耦合：`trust_position` 同时影响位置预测与卡尔曼协方差预测。
 *
 * 该耦合已经造成过一处缺陷 —— 协方差预测原先嵌在预积分条件块内，
 * 关闭 `trust_position` 后增益单调衰减至原值的 1/11，滤波器对量测失去响应
 * （详见 docs/position-filter.md 第五节）。缺陷只在组合时暴露，
 * 两个模块各自的测试都是通过的。因此组合行为必须单独验证。
 *
 * @par 断言设计
 *
 * 对三种故障分别断言「降级决策正确」与「未失控」，并断言卡尔曼路径的
 * 姿态误差与 AlphaBeta 同量级（不超过 2 倍）。不预设卡尔曼在故障场景下
 * 更优 —— 故障期间的主要行为由降级策略决定，滤波器差异属次要因素。
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

double tiltOf(const Tensor &qt) {
    const auto q = toVector(qt);
    const double r33 = 1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2]);
    return std::acos(std::clamp(r33, -1.0, 1.0)) * 180.0 / M_PI;
}

double attErr(const std::array<double, 4> &qe, const Tensor &qt) {
    const auto t = toVector(qt);
    double d = 0.0;
    for (int i = 0; i < 4; ++i) {
        d += qe[static_cast<std::size_t>(i)] * t[static_cast<std::size_t>(i)];
    }
    return 2.0 * std::acos(std::min(1.0, std::fabs(d))) * 180.0 / M_PI;
}

struct Out {
    double max_att_err = 0.0;
    double final_att_err = 0.0;
    double max_tilt = 0.0;
    double max_xy = 0.0;
    double final_alt = 5.0;
    double touchdown_tilt = 0.0;
    bool landed = false;
    bool crashed = false;
    SensorStatus accel_h = SensorStatus::Unknown;
    SensorStatus gyro_h = SensorStatus::Unknown;
    DegradeAction action = DegradeAction::Normal;
};

enum class Fault { None, GyroZero, AccelZero, AccelBias };

Out run(PosFilterKind filter, Fault fault, int steps = 8000) {
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
    etc.pos_filter = filter;
    StateEstimator est(etc);
    est.reset();

    const ImuDegradePolicy policy;
    const Tensor target = makeVec3(0.0f, 0.0f, -5.0f);
    constexpr int fault_start = 2000;
    Out o;

    for (int k = 0; k < steps; ++k) {
        const double t = k * dt;
        ImuSample s = imu.measure(sim.state(), {0.0, 0.0, -cfg.base.gravity}, dt);
        if (k >= fault_start) {
            if (fault == Fault::GyroZero) {
                s.gyro = {0.0, 0.0, 0.0};
            } else if (fault == Fault::AccelZero) {
                s.accel = {0.0, 0.0, 0.0};
            } else if (fault == Fault::AccelBias) {
                s.accel[0] += 1.0;
            }
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

        const double ae = attErr(est.attitude(), sim.state().quat);
        o.max_att_err = std::max(o.max_att_err, ae);
        o.final_att_err = ae;
        const double tl = tiltOf(sim.state().quat);
        o.max_tilt = std::max(o.max_tilt, tl);
        if (tl > 60.0) {
            o.crashed = true;
        }
        const auto p = rv(sim.state().pos);
        o.max_xy = std::max(o.max_xy, std::hypot(p[0], p[1]));
        o.final_alt = -p[2];
        // 触地即停止（仿真无地面碰撞模型）
        if (o.final_alt <= 0.0 && !o.landed) {
            o.landed = true;
            o.touchdown_tilt = tl;
            o.accel_h = h.accel;
            o.gyro_h = h.gyro;
            o.action = d.action;
            break;
        }
        if (k == steps - 1) {
            o.accel_h = h.accel;
            o.gyro_h = h.gyro;
            o.action = d.action;
        }
    }
    return o;
}

const char *fname(Fault f) {
    switch (f) {
    case Fault::None: return "无故障";
    case Fault::GyroZero: return "陀螺失效";
    case Fault::AccelZero: return "加速度计失效";
    case Fault::AccelBias: return "加速度计偏置";
    }
    return "?";
}

const char *aname(DegradeAction a) {
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
    std::printf("=== KalmanFaultCombinationTest: 卡尔曼 × IMU 故障 ===\n\n");

    // 先取四种场景在两种滤波器下的结果（同仿真、同噪声种子）
    const Out none_ab = run(PosFilterKind::AlphaBeta, Fault::None);
    const Out none_kf = run(PosFilterKind::Kalman, Fault::None);
    const Out gyro_ab = run(PosFilterKind::AlphaBeta, Fault::GyroZero);
    const Out gyro_kf = run(PosFilterKind::Kalman, Fault::GyroZero);
    const Out accel_ab = run(PosFilterKind::AlphaBeta, Fault::AccelZero);
    const Out accel_kf = run(PosFilterKind::Kalman, Fault::AccelZero);
    const Out bias_ab = run(PosFilterKind::AlphaBeta, Fault::AccelBias);
    const Out bias_kf = run(PosFilterKind::Kalman, Fault::AccelBias);

    std::printf("%-16s %-10s %10s %10s %8s %-14s\n", "场景", "滤波器", "姿态误差", "真值倾角",
                "水平", "决策");
    struct Row { const char *name; const Out *ab; const Out *kf; };
    const Row rows[4] = {
        {"无故障", &none_ab, &none_kf},
        {"陀螺失效", &gyro_ab, &gyro_kf},
        {"加速度计失效", &accel_ab, &accel_kf},
        {"加速度计偏置", &bias_ab, &bias_kf},
    };
    for (const auto &r : rows) {
        std::printf("%-16s %-10s %9.3f° %9.3f° %7.2f m %-14s\n", r.name, "AlphaBeta",
                    r.ab->max_att_err, r.ab->max_tilt, r.ab->max_xy, aname(r.ab->action));
        std::printf("%-16s %-10s %9.3f° %9.3f° %7.2f m %-14s\n", "", "Kalman",
                    r.kf->max_att_err, r.kf->max_tilt, r.kf->max_xy, aname(r.kf->action));
    }
    std::printf("\n");

    // ---- 1. 无故障：卡尔曼不得引入退化 ----
    check(none_kf.action == DegradeAction::Normal, "无故障：卡尔曼路径决策为 Normal");
    check(!none_kf.landed || none_kf.final_alt > 4.0, "无故障：卡尔曼路径维持悬停高度");
    check(none_kf.max_att_err < 0.5, "无故障：卡尔曼路径姿态误差正常（<0.5°）");

    // ---- 2. 陀螺失效：必须仍判紧急降落，且触地姿态平稳 ----
    check(gyro_kf.action == DegradeAction::EmergencyLand,
          "陀螺失效：卡尔曼路径决策为 EmergencyLand");
    check(gyro_kf.touchdown_tilt < 5.0,
          "陀螺失效：卡尔曼路径以平稳姿态触地（倾角 <5°）");
    check(gyro_kf.max_att_err <= gyro_ab.max_att_err * 2.0 + 0.1,
          "陀螺失效：卡尔曼路径姿态误差与 AlphaBeta 同量级");

    // ---- 3. 加速度计失效：必须仍判返航（不升级为紧急降落）----
    check(accel_kf.action == DegradeAction::ReturnHome,
          "加速度计失效：卡尔曼路径决策为 ReturnHome");
    check(accel_kf.action != DegradeAction::EmergencyLand,
          "加速度计失效：卡尔曼路径不升级为紧急降落");
    check(accel_kf.final_alt > 3.0 && accel_kf.final_alt < 8.0,
          "加速度计失效：卡尔曼路径维持高度（未大幅漂移）");

    // ---- 4. 偏置：必须仍判谨慎飞行 ----
    check(bias_kf.action == DegradeAction::Cautious, "加速度计偏置：卡尔曼路径决策为 Cautious");
    check(bias_kf.final_alt > 3.0 && bias_kf.final_alt < 8.0,
          "加速度计偏置：卡尔曼路径维持高度");

    // ---- 5. 全局：卡尔曼路径在任何故障下都不得失控 ----
    check(!none_kf.crashed && !gyro_kf.crashed && !accel_kf.crashed && !bias_kf.crashed,
          "卡尔曼路径在四种场景下均未失控");

    // ---- 6. 故障确实被检出（激励确认，防止「没激励也算通过」）----
    check(gyro_kf.gyro_h == SensorStatus::Failed, "激励确认：陀螺失效被检出");
    check(accel_kf.accel_h == SensorStatus::Failed, "激励确认：加速度计失效被检出");
    check(bias_kf.accel_h == SensorStatus::Degraded, "激励确认：加速度计偏置被检出");

    std::printf("\n========================================\n");
    std::printf("%d / %d checks passed\n", g_pass, g_pass + g_fail);
    std::printf("========================================\n");
    return g_fail == 0 ? 0 : 1;
}
