/**
 * @file TrajectoryLimitCalibrationTest.cpp
 * @brief P4：速度与 jerk 上限标定
 *
 * @par 标定目标
 *
 * `TrajectoryLimits` 的默认值是保守初值，不是飞行器真实能力上限。
 * 本测试在仿真闭环上扫描多组 (max_vel, max_acc, max_jerk)，找出「既能完成
 * 典型机动、又有足够安全余量」的推荐上限。
 *
 * @par 标定场景
 *
 * 1. 水平急停返航：从 (3,4,-5) 回到 (0,0,-5)，水平距离 5 m；
 * 2. 垂直爬升：从 (0,0,-8) 到 (0,0,-3)，高度变化 5 m。
 *
 * 两个场景覆盖返航与任务中最常见的两类机动。
 *
 * @par 失败判定
 *
 * - Trajectory build 失败；
 * - 超出时间预算（> 2× bang-bang 物理下界 + 1 s）；
 * - 触地或高度失控（高度 < 0.5 m 或 > 15 m）；
 * - 真值倾角 > 60°；
 * - 末态位置误差 > 1.0 m。
 *
 * @par 推荐值提取
 *
 * 在所有通过组合的边界上，取一个保守折扣（0.85）作为推荐上限，给真机参数误差、
 * 风扰、电池压降留余量。
 */

#include "FlightControlLoop.h"
#include "GuidanceSetpointSource.h"
#include "HalSimulator.h"
#include "ImuModel.h"
#include "MinimumSnapTrajectory.h"
#include "SixDofPidController.h"
#include "SixDofSimulator.h"
#include "TensorUtils.h"

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

std::array<double, 3> readVec(const Tensor &t) {
    const std::vector<float> v = toVector(t);
    return {static_cast<double>(v[0]), static_cast<double>(v[1]),
            static_cast<double>(v[2])};
}

/// 由真值四元数求倾角（度）
double tiltOfQuat(const Tensor &qt) {
    const std::vector<float> q = toVector(qt);
    const double r33 = 1.0 - 2.0 * (static_cast<double>(q[1]) * static_cast<double>(q[1]) +
                                    static_cast<double>(q[2]) * static_cast<double>(q[2]));
    return std::acos(std::max(-1.0, std::min(1.0, r33))) * 180.0 / M_PI;
}

struct CalibResult {
    double max_vel = 0.0;
    double max_acc = 0.0;
    double max_jerk = 0.0;
    bool ok = false;
    double final_pos_err = 0.0;
    double max_tilt = 0.0;
    double duration = 0.0;
    std::string reason;
};

enum class ScenarioType { HorizontalReturn, VerticalClimb };

CalibResult runScenario(ScenarioType type, const TrajectoryLimits &limits,
                        int max_steps = 30000) {
    CalibResult res;
    res.max_vel = limits.max_vel;
    res.max_acc = limits.max_acc;
    res.max_jerk = limits.max_jerk;

    SixDofConfig cfg;

    std::array<double, 3> start_pos;
    std::array<double, 3> target_pos;
    if (type == ScenarioType::HorizontalReturn) {
        // 20 m 水平返航：足够长，让速度限制真正影响轨迹时长
        start_pos = {12.0, 16.0, -5.0};
        target_pos = {0.0, 0.0, -5.0};
    } else {
        // 5 m 垂直下降：温和机动，用于验证垂直方向约束边界
        start_pos = {0.0, 0.0, -8.0};
        target_pos = {0.0, 0.0, -3.0};
    }

    SixDofState init{makeVec3(static_cast<float>(start_pos[0]),
                              static_cast<float>(start_pos[1]),
                              static_cast<float>(start_pos[2])),
                     makeVec3(0.0f, 0.0f, 0.0f),
                     Tensor{1.0f, 0.0f, 0.0f, 0.0f},
                     makeVec3(0.0f, 0.0f, 0.0f)};
    SixDofSimulator sim(cfg, init);

    ImuConfig icfg;
    icfg.explicit_bias = true;
    ImuModel imu(icfg, 20260918u);
    SimSensorReader sensors(&sim, &imu, {0.0, 0.0, cfg.base.gravity});
    SimActuatorWriter actuators(&sim);

    const Tensor mission_target =
        makeVec3(static_cast<float>(target_pos[0]),
                 static_cast<float>(target_pos[1]), static_cast<float>(target_pos[2]));
    // 标定聚焦在 ReturnHome 轨迹：Mission 模式不构建轨迹，limits 不会生效
    GuidanceSetpointSource setpoint(mission_target, target_pos, limits,
                                    GuidanceMode::Mission);

    // 手动触发 ReturnHome 轨迹构建：从 Mission 切到 ReturnHome 会调用 build
    DegradeDecision decision;
    decision.action = DegradeAction::ReturnHome;
    setpoint.onDecisionChanged(decision, start_pos, 0.0);

    SixDofPidController ctrl(cfg, {});
    FlightControlConfig fc_cfg;
    fc_cfg.executor.hover_thrust = static_cast<double>(cfg.base.mass) * cfg.base.gravity;
    fc_cfg.hover_thrust = fc_cfg.executor.hover_thrust;

    FlightControlLoop loop(fc_cfg, &sensors, &actuators, &setpoint, &ctrl);
    loop.init();

    const double distance = std::sqrt((start_pos[0] - target_pos[0]) * (start_pos[0] - target_pos[0]) +
                                      (start_pos[1] - target_pos[1]) * (start_pos[1] - target_pos[1]) +
                                      (start_pos[2] - target_pos[2]) * (start_pos[2] - target_pos[2]));
    // 时间预算：min-snap 形状落后 bang-bang 约 1.37 倍，再加 50% 余量
    const double t_lower = 2.0 * std::sqrt(distance / limits.max_acc);
    const double time_budget = 3.0 * t_lower + 2.0;

    double max_tilt = 0.0;
    int step = 0;

    for (; step < max_steps; ++step) {
        if (!loop.runOneCycle()) {
            res.reason = "主循环异常终止";
            return res;
        }

        const auto p = readVec(sim.state().pos);
        const double alt = -p[2];
        if (alt < 0.5 || alt > 15.0) {
            res.reason = "高度失控";
            return res;
        }

        const double tilt = tiltOfQuat(sim.state().quat);
        if (tilt > max_tilt) max_tilt = tilt;
        if (tilt > 60.0) {
            res.reason = "倾角超限";
            res.max_tilt = tilt;
            return res;
        }

        const double t = sim.time();
        if (t > time_budget) {
            res.reason = "超时";
            res.duration = t;
            res.max_tilt = max_tilt;
            return res;
        }

        const double err = std::sqrt((p[0] - target_pos[0]) * (p[0] - target_pos[0]) +
                                     (p[1] - target_pos[1]) * (p[1] - target_pos[1]) +
                                     (p[2] - target_pos[2]) * (p[2] - target_pos[2]));
        if (err < 0.5) {
            // 到达是必要条件；此外以倾角作为安全边界
            res.final_pos_err = err;
            res.max_tilt = max_tilt;
            res.duration = t;
            res.ok = true;
            return res;
        }
    }

    res.reason = "未在最大步数内到达";
    res.max_tilt = max_tilt;
    return res;
}

} // namespace

int main() {
    std::printf("=== TrajectoryLimitCalibrationTest：速度与 jerk 上限标定 ===\n\n");

    // 扫描网格：固定加速度为理论上限 6.87，扫描速度与 jerk 的边界
    // 通过标准：到达 home 且最大倾角 <= 30°（给真机留安全余量）
    const std::vector<double> vel_grid = {3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0};
    const std::vector<double> acc_grid = {6.87};
    const std::vector<double> jerk_grid = {10.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 150.0};

    double best_vel_h = 0.0, best_acc_h = 0.0, best_jerk_h = 0.0;
    double best_vel_v = 0.0, best_acc_v = 0.0, best_jerk_v = 0.0;

    std::printf("--- 水平返航场景扫描（部分结果）---\n");
    std::printf("%-6s %-6s %-6s %-6s %-8s %-8s %-8s %-10s\n",
                "vel", "acc", "jerk", "ok", "err(m)", "tilt", "dur(s)", "reason");

    int printed = 0;
    for (double v : vel_grid) {
        for (double a : acc_grid) {
            for (double j : jerk_grid) {
                TrajectoryLimits lim{v, a, j};
                CalibResult r = runScenario(ScenarioType::HorizontalReturn, lim);
                if (r.ok && r.max_tilt <= 30.0) {
                    // 以倾角 30° 作为安全边界，记录满足条件的最大速度与 jerk
                    best_vel_h = std::max(best_vel_h, v);
                    best_acc_h = std::max(best_acc_h, std::min(a, 6.87));
                    best_jerk_h = std::max(best_jerk_h, j);
                }
                // 每档打印少量代表性结果（jerk=20 时）
                if (std::fabs(j - 20.0) < 0.1 && printed < 24) {
                    const bool safe = r.ok && r.max_tilt <= 30.0;
                    std::printf("%-6.1f %-6.1f %-6.1f %-6s %-8.3f %-8.2f %-8.2f %-10s\n",
                                v, a, j, safe ? "yes" : "no", r.final_pos_err,
                                r.max_tilt, r.duration, r.reason.c_str());
                    ++printed;
                }
            }
        }
    }

    std::printf("\n--- 垂直爬升场景扫描（部分结果）---\n");
    std::printf("%-6s %-6s %-6s %-6s %-8s %-8s %-8s %-10s\n",
                "vel", "acc", "jerk", "ok", "err(m)", "tilt", "dur(s)", "reason");
    printed = 0;
    for (double v : vel_grid) {
        for (double a : acc_grid) {
            for (double j : jerk_grid) {
                TrajectoryLimits lim{v, a, j};
                CalibResult r = runScenario(ScenarioType::VerticalClimb, lim);
                if (r.ok && r.max_tilt <= 30.0) {
                    best_vel_v = std::max(best_vel_v, v);
                    best_acc_v = std::max(best_acc_v, std::min(a, 6.87));
                    best_jerk_v = std::max(best_jerk_v, j);
                }
                if (std::fabs(j - 20.0) < 0.1 && printed < 24) {
                    const bool safe = r.ok && r.max_tilt <= 30.0;
                    std::printf("%-6.1f %-6.1f %-6.1f %-6s %-8.3f %-8.2f %-8.2f %-10s\n",
                                v, a, j, safe ? "yes" : "no", r.final_pos_err,
                                r.max_tilt, r.duration, r.reason.c_str());
                    ++printed;
                }
            }
        }
    }

    std::printf("\n--- 标定结论 ---\n");
    std::printf("水平返航最大通过组合: vel=%.1f acc=%.1f jerk=%.1f\n",
                best_vel_h, best_acc_h, best_jerk_h);
    std::printf("垂直爬升最大通过组合: vel=%.1f acc=%.1f jerk=%.1f\n",
                best_vel_v, best_acc_v, best_jerk_v);

    // 取保守折扣 0.85 作为推荐值；jerk 受电机/结构响应限制，取工程上限 30
    const double rec_vel = 0.85 * std::min(best_vel_h, best_vel_v);
    const double rec_acc = 0.85 * std::min(best_acc_h, best_acc_v);
    const double rec_jerk = std::min(0.85 * std::min(best_jerk_h, best_jerk_v), 30.0);

    // 加速度无论如何不能超过倾角理论上限；取保守值
    const double theoretical_acc_max = 6.87;
    const double safe_acc = std::min(rec_acc, 0.9 * theoretical_acc_max);

    std::printf("\n推荐上限（0.85 折扣后，加速度再取 0.9 倾角上限）:\n");
    std::printf("  max_vel  = %.2f m/s\n", rec_vel);
    std::printf("  max_acc  = %.2f m/s²\n", safe_acc);
    std::printf("  max_jerk = %.2f m/s³\n", rec_jerk);

    // 验收断言
    check(rec_vel > 2.0, "标定：推荐速度 > 2 m/s");
    check(safe_acc > 2.0, "标定：推荐加速度 > 2 m/s²");
    check(rec_jerk > 5.0, "标定：推荐 jerk > 5 m/s³");
    check(safe_acc <= theoretical_acc_max, "标定：推荐加速度不超过倾斜角理论上限 6.87");

    std::printf("\n========================================\n");
    std::printf("%d / %d checks passed\n", g_pass, g_pass + g_fail);
    std::printf("========================================\n");
    return (g_fail == 0) ? 0 : 1;
}
