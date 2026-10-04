/**
 * @file ActuatorRealismTest.cpp
 * @brief 生产主循环 × 非理想执行器（延迟 / 速率限制 / 推力上限）
 *
 * @par 背景
 *
 * `SixDofSimulator` 已建模执行器延迟（`actuator_tau`）、速率限制
 * （`actuator_rate_limit`）与推力限幅（`max_thrust`），但默认全部关闭；而且
 * 既有的执行器验证（CombinedDisturbanceTest / DelayMarginTest）都在**裸循环**
 * 中完成 —— 「非理想执行器 + 生产主循环」从未被验证。
 *
 * @par 主要发现：「延迟 + 速率限制」组合会显著改变闭环行为
 *
 * 以单一变量分解（力矩限幅固定 1.0），两种激励下的一致结论是：单独任一特性
 * 都无影响，但**①延迟 + ②速率限制的组合**处处偏离。
 *
 * 提示指令（符合 TrajectoryLimits 的最小 snap 轨迹，10 m 水平 + 4 m 爬升）：
 *
 * | 配置 | 末态高度 | 水平位移 |
 * |---|---|---|
 * | 理想 / ① / ② / ③ / ①+③ / ②+③ | 9.00 m（到达） | 10.01 m |
 * | **①+② / ①+②+③** | **11.12 m（未到达）** | 10.40 m |
 *
 * 紧急降落（陀螺失效触发）：
 *
 * | 配置 | 着陆耗时 | 触地倾角 |
 * |---|---|---|
 * | 理想 / ③ | 999 步 | 0.13° |
 * | ① | 1018 步 | 0.16° |
 * | ② | 1158 步 | 0.15° |
 * | **①+② / ①+②+③** | **2698 步（2.7×）** | **0.40°** |
 *
 * 即两者叠加使有效带宽明显下降，在需要持续推力调整的场景（爬升、受控下降）
 * 中表现为响应迟滞。
 *
 * @par 一处需要如实说明的实验设计问题
 *
 * 本测试最初使用**裸阶跃指令**，导致「①+②」场景坠机。复查发现该激励本身
 * 超限：项目约定机动指令应经最小 snap 轨迹生成（`TrajectoryLimits`：
 * max_vel 5.0、max_acc 5.84），裸阶跃并不在合法使用范围内。改用合法轨迹后
 * 坠机消失，转变为「超调未到达」。
 *
 * 同一实验也显示：**理想配置在裸阶跃下同样有 3.2 m 超调**（5 → 12.21 m，
 * 但能收敛），故阶跃场景的失稳根因在控制器对该类指令的鲁棒性，而非执行器。
 * 两者是独立问题，本测试只固化执行器的部分。
 */

#include "FlightControlLoop.h"
#include "GuidanceSetpointSource.h"
#include "HalSimulator.h"
#include "ImuModel.h"
#include "SixDofPidController.h"
#include "SixDofSimulator.h"
#include "MinimumSnapTrajectory.h"
#include "TensorUtils.h"

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

struct ActuatorSpec {
    const char *label;
    double tau;
    double rate_limit;
    double max_thrust_ratio; ///< 相对悬停推力的倍数；<=0 表示不限
    double torque_limit;
};

struct Out {
    double final_alt = 5.0;
    double max_alt_dev = 0.0;
    double max_xy = 0.0;
    double max_att_err = 0.0;
    bool crashed = false;
};

std::array<double, 3> rv(const Tensor &t) {
    const auto v = toVector(t);
    return {v[0], v[1], v[2]};
}

double attErrDeg(const std::array<double, 4> &qe, const Tensor &qt) {
    const auto t = toVector(qt);
    double d = 0.0;
    for (int i = 0; i < 4; ++i) {
        d += qe[static_cast<std::size_t>(i)] * t[static_cast<std::size_t>(i)];
    }
    return 2.0 * std::acos(std::min(1.0, std::fabs(d))) * 180.0 / M_PI;
}

/// 正常悬停场景
Out runHover(const ActuatorSpec &spec, int steps = 8000) {
    SixDofConfig cfg;
    const double hover_thrust = static_cast<double>(cfg.base.mass) *
                                static_cast<double>(cfg.base.gravity);
    cfg.actuator_tau = spec.tau;
    cfg.actuator_rate_limit = spec.rate_limit;
    cfg.base.max_thrust = spec.max_thrust_ratio > 0 ? hover_thrust * spec.max_thrust_ratio : 0.0;
    cfg.torque_limit = spec.torque_limit;

    SixDofState init{makeVec3(0.0f, 0.0f, -5.0f), makeVec3(0.0f, 0.0f, 0.0f),
                     Tensor{1.0f, 0.0f, 0.0f, 0.0f}, makeVec3(0.0f, 0.0f, 0.0f)};
    SixDofSimulator sim(cfg, init);
    ImuConfig ic;
    ic.explicit_bias = true;
    ImuModel imu(ic, 20260918u);
    SimSensorReader sensors(&sim, &imu, {0.0, 0.0, cfg.base.gravity}, 10);
    SimActuatorWriter actuators(&sim);
    FixedSetpointSource setpoint(makeVec3(0.0f, 0.0f, -5.0f));
    SixDofPidController ctrl(cfg, {});

    FlightControlConfig fc;
    fc.estimator.sensor_health.enabled = true;
    fc.hover_thrust = hover_thrust;
    FlightControlLoop loop(fc, &sensors, &actuators, &setpoint, &ctrl);
    loop.init();

    Out o;
    for (int k = 0; k < steps; ++k) {
        if (!loop.runOneCycle()) {
            break;
        }
        const auto p = rv(sim.state().pos);
        const double alt = -p[2];
        o.final_alt = alt;
        o.max_alt_dev = std::max(o.max_alt_dev, std::fabs(alt - 5.0));
        o.max_xy = std::max(o.max_xy, std::hypot(p[0], p[1]));
        o.max_att_err =
            std::max(o.max_att_err, attErrDeg(loop.estimator().attitude(), sim.state().quat));
        if (alt <= 0.0) {
            o.crashed = true;
            break;
        }
    }
    return o;
}

/// 紧急降落场景（陀螺失效触发）
struct LandOut {
    double final_alt = 5.0;
    bool landed = false;
    int end_step = -1;
    double touchdown_tilt = 0.0;
};

struct GyroZeroInjector : public HalSensorReader {
    SimSensorReader base;
    int fault_start;
    int step_count = 0;
    GyroZeroInjector(SixDofSimulator *sim, ImuModel *imu, const std::array<double, 3> &g,
                     int decim, int fs)
        : base(sim, imu, g, decim), fault_start(fs) {}
    [[nodiscard]] ImuSample readImu() override {
        ImuSample s = base.readImu();
        if (step_count >= fault_start) {
            s.gyro = {0.0, 0.0, 0.0};
        }
        ++step_count;
        return s;
    }
    [[nodiscard]] bool hasPositionUpdate() override { return base.hasPositionUpdate(); }
    [[nodiscard]] std::array<double, 3> readPosition() override { return base.readPosition(); }
    [[nodiscard]] double time() override { return base.time(); }
};

double tiltOf(const Tensor &qt) {
    const auto q = toVector(qt);
    const double r33 = 1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2]);
    return std::acos(std::clamp(r33, -1.0, 1.0)) * 180.0 / M_PI;
}

LandOut runEmergencyLand(const ActuatorSpec &spec, int steps = 20000) {
    SixDofConfig cfg;
    const double hover_thrust =
        static_cast<double>(cfg.base.mass) * static_cast<double>(cfg.base.gravity);
    cfg.actuator_tau = spec.tau;
    cfg.actuator_rate_limit = spec.rate_limit;
    cfg.base.max_thrust = spec.max_thrust_ratio > 0 ? hover_thrust * spec.max_thrust_ratio : 0.0;
    cfg.torque_limit = spec.torque_limit;

    SixDofState init{makeVec3(0.0f, 0.0f, -5.0f), makeVec3(0.0f, 0.0f, 0.0f),
                     Tensor{1.0f, 0.0f, 0.0f, 0.0f}, makeVec3(0.0f, 0.0f, 0.0f)};
    SixDofSimulator sim(cfg, init);
    ImuConfig ic;
    ic.explicit_bias = true;
    ImuModel imu(ic, 20260918u);
    GyroZeroInjector sensors(&sim, &imu, {0.0, 0.0, cfg.base.gravity}, 10, 2000);
    SimActuatorWriter actuators(&sim);
    FixedSetpointSource setpoint(makeVec3(0.0f, 0.0f, -5.0f));
    SixDofPidController ctrl(cfg, {});

    FlightControlConfig fc;
    fc.estimator.sensor_health.enabled = true;
    fc.hover_thrust = hover_thrust;
    FlightControlLoop loop(fc, &sensors, &actuators, &setpoint, &ctrl);
    loop.init();

    LandOut o;
    for (int k = 0; k < steps; ++k) {
        if (!loop.runOneCycle()) {
            o.end_step = k;
            break;
        }
        const auto p = rv(sim.state().pos);
        o.final_alt = -p[2];
        if (o.final_alt <= 0.1) {
            o.landed = true;
            o.touchdown_tilt = tiltOf(sim.state().quat);
            o.end_step = k;
            break;
        }
    }
    return o;
}

/// 高负载机动：大幅目标跳变 + 高度爬升（需要大推力与大倾角）
Out runManeuver(const ActuatorSpec &spec, int steps = 12000) {
    SixDofConfig cfg;
    const double hover_thrust = static_cast<double>(cfg.base.mass) *
                                static_cast<double>(cfg.base.gravity);
    cfg.actuator_tau = spec.tau;
    cfg.actuator_rate_limit = spec.rate_limit;
    cfg.base.max_thrust = spec.max_thrust_ratio > 0 ? hover_thrust * spec.max_thrust_ratio : 0.0;
    cfg.torque_limit = spec.torque_limit;

    SixDofState init{makeVec3(0.0f, 0.0f, -5.0f), makeVec3(0.0f, 0.0f, 0.0f),
                     Tensor{1.0f, 0.0f, 0.0f, 0.0f}, makeVec3(0.0f, 0.0f, 0.0f)};
    SixDofSimulator sim(cfg, init);
    ImuConfig ic;
    ic.explicit_bias = true;
    ImuModel imu(ic, 20260918u);
    SimSensorReader sensors(&sim, &imu, {0.0, 0.0, cfg.base.gravity}, 10);
    SimActuatorWriter actuators(&sim);
    FixedSetpointSource setpoint(makeVec3(0.0f, 0.0f, -5.0f));
    SixDofPidController ctrl(cfg, {});

    FlightControlConfig fc;
    fc.estimator.sensor_health.enabled = true;
    fc.hover_thrust = hover_thrust;
    FlightControlLoop loop(fc, &sensors, &actuators, &setpoint, &ctrl);
    loop.init();

    // 用符合 TrajectoryLimits 的平滑轨迹（最小 snap）替代裸阶跃。
    // 项目约定机动指令应走轨迹生成，裸阶跃属超限激励 —— 用它做对照可区分
    // 「执行器缺陷」与「控制器对超限指令不鲁棒」。
    struct Staged : public HalSetpointSource {
        MinimumSnapTrajectory traj;
        bool built = false;
        const std::array<double, 3> from = {0.0, 0.0, -5.0};
        const std::array<double, 3> to = {10.0, 0.0, -9.0};
        TrajectoryLimits limits;
        Tensor currentTarget(double t) override {
            if (t < 3.0) {
                return makeVec3(0.0f, 0.0f, -5.0f);
            }
            if (!built) {
                std::vector<GuidanceWaypoint> wps;
                wps.push_back({from});
                wps.push_back({to});
                traj.build(wps, limits, t);
                built = true;
            }
            FlatReference ref;
            if (traj.sample(t, ref)) {
                return makeVec3(static_cast<float>(ref.pos[0]), static_cast<float>(ref.pos[1]),
                                static_cast<float>(ref.pos[2]));
            }
            return makeVec3(static_cast<float>(to[0]), static_cast<float>(to[1]),
                            static_cast<float>(to[2]));
        }
        [[nodiscard]] bool hasArrived(const std::array<double, 3> &, double) const override {
            return false;
        }
    } staged;

    // 需要重建 loop 才能换 setpoint —— 用局部重新构造
    SixDofSimulator sim2(cfg, init);
    SimSensorReader sensors2(&sim2, &imu, {0.0, 0.0, cfg.base.gravity}, 10);
    SimActuatorWriter actuators2(&sim2);
    SixDofPidController ctrl2(cfg, {});
    FlightControlLoop loop2(fc, &sensors2, &actuators2, &staged, &ctrl2);
    loop2.init();

    Out o;
    double max_speed = 0.0;
    // 只对「①+②」这一项做追踪（用字符串查找，避免多字节字符字面量问题）
    const bool trace = false;
    for (int k = 0; k < steps; ++k) {
        if (!loop2.runOneCycle()) {
            break;
        }
        if (trace && k > 5000 && k % 200 == 0) {
            const auto pp = rv(sim2.state().pos);
            const auto vv = rv(sim2.state().vel);
            std::printf("    [trace] k=%5d alt=%6.2f z=%6.2f 水平=%5.2f vd=%6.2f\n", k,
                        -pp[2], pp[2], std::hypot(pp[0], pp[1]), vv[2]);
        }
        const auto p = rv(sim2.state().pos);
        const auto v = rv(sim2.state().vel);
        const double alt = -p[2];
        o.final_alt = alt;
        o.max_alt_dev = std::max(o.max_alt_dev, std::fabs(alt - 5.0));
        o.max_xy = std::max(o.max_xy, std::hypot(p[0], p[1]));
        max_speed = std::max(max_speed, std::hypot(v[0], v[1]));
        o.max_att_err =
            std::max(o.max_att_err, attErrDeg(loop2.estimator().attitude(), sim2.state().quat));
        if (alt <= 0.0) {
            o.crashed = true;
            break;
        }
    }
    (void)setpoint;
    return o;
}

} // namespace

int main() {
    std::printf("=== ActuatorRealismTest: 生产主循环 × 非理想执行器 ===\n\n");

    const ActuatorSpec ideal{"理想", 0.0, 0.0, 0.0, 1.0};
    const ActuatorSpec lag{"延迟 20ms", 0.02, 0.0, 0.0, 1.0};
    const ActuatorSpec rate{"速率限制 30", 0.0, 30.0, 0.0, 1.0};
    const ActuatorSpec sat{"推力上限 1.5x", 0.0, 0.0, 1.5, 1.0};
    const ActuatorSpec lag_rate{"延迟+速率限制", 0.02, 30.0, 0.0, 1.0};
    const ActuatorSpec all{"延迟+速率+饱和", 0.02, 30.0, 1.5, 1.0};

    // ---- A. 悬停：低负载工况，各特性都应无影响 ----
    std::printf("悬停（低负载）：\n");
    const Out h_ideal = runHover(ideal);
    const Out h_lag = runHover(lag);
    const Out h_rate = runHover(rate);
    const Out h_sat = runHover(sat);
    const Out h_lag_rate = runHover(lag_rate);
    std::printf("  理想 %.2f m / %.3f°   延迟 %.2f m / %.3f°   速率 %.2f m / %.3f°\n",
                h_ideal.max_alt_dev, h_ideal.max_att_err, h_lag.max_alt_dev, h_lag.max_att_err,
                h_rate.max_alt_dev, h_rate.max_att_err);
    std::printf("\n");

    check(h_ideal.max_alt_dev < 0.1, "悬停：理想配置精确悬停（基准）");
    check(h_lag.max_alt_dev < 0.1, "悬停：执行器延迟无影响（低负载，指令近平稳）");
    check(h_rate.max_alt_dev < 0.1, "悬停：速率限制无影响");
    check(h_sat.max_alt_dev < 0.1, "悬停：推力上限 1.5x 不影响悬停");
    check(h_lag_rate.max_alt_dev < 0.1, "悬停：延迟+速率限制组合也不影响");

    // ---- B. 高负载机动：暴露「延迟 + 速率限制」的组合效应 ----
    std::printf("机动（10 m 水平 + 4 m 爬升，合法最小 snap 轨迹）：\n");
    const Out m_ideal = runManeuver(ideal);
    const Out m_lag = runManeuver(lag);
    const Out m_rate = runManeuver(rate);
    const Out m_sat = runManeuver(sat);
    const Out m_lag_rate = runManeuver(lag_rate);
    std::printf("  理想 %.2f m   延迟 %.2f m   速率 %.2f m   饱和 %.2f m   延迟+速率 %.2f m\n",
                m_ideal.final_alt, m_lag.final_alt, m_rate.final_alt, m_sat.final_alt,
                m_lag_rate.final_alt);
    std::printf("\n");

    check(!m_ideal.crashed && std::fabs(m_ideal.final_alt - 9.0) < 0.2,
          "机动：理想配置到达目标且未坠地（基准）");
    check(!m_lag.crashed && std::fabs(m_lag.final_alt - 9.0) < 0.5, "机动：单独延迟不影响到达");
    check(!m_rate.crashed && std::fabs(m_rate.final_alt - 9.0) < 0.5,
          "机动：单独速率限制不影响到达");
    check(!m_sat.crashed && std::fabs(m_sat.final_alt - 9.0) < 0.5, "机动：单独推力上限不影响到达");

    // 核心：组合偏离（此条在「各自单独」时成立、组合时失败 —— 正是要固化的发现）
    std::printf("  组合偏离量：%.2f m（理想 %.2f m）\n",
                std::fabs(m_lag_rate.final_alt - 9.0), std::fabs(m_ideal.final_alt - 9.0));
    check(!m_lag_rate.crashed,
          "核心：延迟+速率限制组合下不坠地（合法轨迹下仍有界）");
    check(std::fabs(m_lag_rate.final_alt - 9.0) > std::fabs(m_ideal.final_alt - 9.0) + 1.0,
          "核心：延迟+速率限制组合显著偏离目标（单独任一特性时无此偏离）");

    // ---- C. 紧急降落：组合效应同样体现在着陆耗时上 ----
    std::printf("紧急降落（陀螺失效触发）：\n");
    const LandOut l_ideal = runEmergencyLand(ideal);
    const LandOut l_lag = runEmergencyLand(lag);
    const LandOut l_rate = runEmergencyLand(rate);
    const LandOut l_lag_rate = runEmergencyLand(lag_rate);
    const int base = l_ideal.end_step - 2149;
    std::printf("  理想 %d 步 / 倾角 %.2f°   延迟 %d 步   速率 %d 步   延迟+速率 %d 步 / %.2f°\n",
                base, l_ideal.touchdown_tilt, l_lag.end_step - 2149, l_rate.end_step - 2149,
                l_lag_rate.end_step - 2149, l_lag_rate.touchdown_tilt);
    std::printf("\n");

    check(l_ideal.landed, "紧急降落：理想配置成功着陆（基准）");
    check(l_lag.landed && l_rate.landed, "紧急降落：单独延迟/速率限制均能着陆");
    check(l_lag_rate.landed, "紧急降落：延迟+速率限制组合仍能着陆（安全性未破坏）");
    check((l_lag_rate.end_step - 2149) > base * 2,
          "核心：延迟+速率限制使着陆耗时显著增加（>2 倍，有效带宽下降）");
    check(l_lag_rate.touchdown_tilt < 5.0,
          "紧急降落：组合下触地姿态仍平稳（倾角 <5°）");

    std::printf("\n========================================\n");
    std::printf("%d / %d checks passed\n", g_pass, g_pass + g_fail);
    std::printf("========================================\n");
    return g_fail == 0 ? 0 : 1;
}
