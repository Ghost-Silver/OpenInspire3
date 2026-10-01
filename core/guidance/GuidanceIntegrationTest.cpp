/**
 * @file GuidanceIntegrationTest.cpp
 * @brief P3 制导层与容错层联调：返航/紧急降落轨迹生成与执行
 *
 * @par 验证目标
 *
 * 降级动作（返航、紧急降落）此前只输出限幅与动作码，没有对应的轨迹执行。
 * 本测试验证 GuidanceSetpointSource 能把降级决策转化为实际的时空轨迹：
 * 1. Mission 模式等价于 FixedSetpointSource（不引入回归）；
 * 2. ReturnHome 生成从当前位置回到起点的 MinimumSnap 轨迹；
 * 3. EmergencyLand 生成垂直下降轨迹，用于着陆检测；
 * 4. 模式切换往返：故障恢复后回到 Mission。
 */

#include "FlightControlLoop.h"
#include "GuidanceSetpointSource.h"
#include "HalSimulator.h"
#include "ImuModel.h"
#include "SixDofPidController.h"
#include "SixDofSimulator.h"
#include "TensorUtils.h"

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

std::array<double, 3> readVec(const Tensor &t) {
    const std::vector<float> v = toVector(t);
    return {static_cast<double>(v[0]), static_cast<double>(v[1]),
            static_cast<double>(v[2])};
}

/// 加速度计归零注入器
struct AccelZeroInjector : public HalSensorReader {
    SimSensorReader base;
    int fault_start_step;
    int step_count = 0;
    bool fault_active = false;
    AccelZeroInjector(SixDofSimulator *sim, ImuModel *imu,
                      const std::array<double, 3> &g, int decim, int fault_start)
        : base(sim, imu, g, decim), fault_start_step(fault_start) {}
    ImuSample readImu() override {
        ++step_count;
        ImuSample s = base.readImu();
        if (!fault_active && step_count >= fault_start_step) {
            fault_active = true;
        }
        if (fault_active) {
            s.accel = {0.0, 0.0, 0.0};
        }
        return s;
    }
    void resetFault() { fault_active = false; }
    bool hasPositionUpdate() override { return base.hasPositionUpdate(); }
    std::array<double, 3> readPosition() override { return base.readPosition(); }
    double time() override { return base.time(); }
};

/// 陀螺归零注入器
struct GyroZeroInjector : public HalSensorReader {
    SimSensorReader base;
    int fault_start_step;
    int step_count = 0;
    GyroZeroInjector(SixDofSimulator *sim, ImuModel *imu,
                     const std::array<double, 3> &g, int decim, int fault_start)
        : base(sim, imu, g, decim), fault_start_step(fault_start) {}
    ImuSample readImu() override {
        ++step_count;
        ImuSample s = base.readImu();
        if (step_count >= fault_start_step) {
            s.gyro = {0.0, 0.0, 0.0};
        }
        return s;
    }
    bool hasPositionUpdate() override { return base.hasPositionUpdate(); }
    std::array<double, 3> readPosition() override { return base.readPosition(); }
    double time() override { return base.time(); }
};

} // namespace

int main() {
    std::printf("=== GuidanceIntegrationTest: 制导层与容错层联调 ===\n\n");

    // ================================================================
    // 1. Mission 模式：等价于 FixedSetpointSource，不引入回归
    // ================================================================
    std::printf("--- 1. Mission 模式（正常悬停）---\n");
    {
        SixDofConfig cfg;
        const double dt = cfg.base.dt;
        SixDofState init{makeVec3(0.0f, 0.0f, -5.0f), makeVec3(0.0f, 0.0f, 0.0f),
                         Tensor{1.0f, 0.0f, 0.0f, 0.0f}, makeVec3(0.0f, 0.0f, 0.0f)};
        SixDofSimulator sim(cfg, init);

        ImuConfig icfg;
        icfg.explicit_bias = true;
        ImuModel imu(icfg, 20260918u);
        SimSensorReader sensors(&sim, &imu, {0.0, 0.0, cfg.base.gravity});
        SimActuatorWriter actuators(&sim);

        const Tensor mission_target = makeVec3(0.0f, 0.0f, -5.0f);
        const std::array<double, 3> home_pos = {0.0, 0.0, -5.0};
        TrajectoryLimits limits;
        GuidanceSetpointSource setpoint(mission_target, home_pos, limits);

        SixDofPidController ctrl(cfg, {});
        FlightControlConfig fc_cfg;
        fc_cfg.executor.hover_thrust = static_cast<double>(cfg.base.mass) * cfg.base.gravity;
        fc_cfg.hover_thrust = fc_cfg.executor.hover_thrust;

        FlightControlLoop loop(fc_cfg, &sensors, &actuators, &setpoint, &ctrl);
        loop.init();

        for (int k = 0; k < 3000; ++k) {
            loop.runOneCycle();
        }

        const auto p = readVec(sim.state().pos);
        const double alt = -p[2];
        const double horiz = std::sqrt(p[0] * p[0] + p[1] * p[1]);

        std::printf("    末态高度 %.2f m，水平偏差 %.3f m\n", alt, horiz);
        check(alt > 4.5 && alt < 5.5, "Mission：高度保持在目标 ±0.5 m 内");
        check(horiz < 0.1, "Mission：水平漂移 < 0.1 m");
        check(setpoint.mode() == GuidanceMode::Mission, "Mission：制导源保持 Mission 模式");
    }

    // ================================================================
    // 2. ReturnHome：加速度计失效后生成返航轨迹并跟踪
    // ================================================================
    std::printf("\n--- 2. ReturnHome（加速度计归零触发返航）---\n");
    {
        SixDofConfig cfg;
        const double dt = cfg.base.dt;
        // 从远离 home 的位置开始（水平 5 m）
        SixDofState init{makeVec3(3.0f, 4.0f, -5.0f), makeVec3(0.0f, 0.0f, 0.0f),
                         Tensor{1.0f, 0.0f, 0.0f, 0.0f}, makeVec3(0.0f, 0.0f, 0.0f)};
        SixDofSimulator sim(cfg, init);

        ImuConfig icfg;
        icfg.explicit_bias = true;
        ImuModel imu(icfg, 1u);
        AccelZeroInjector sensors(&sim, &imu, {0.0, 0.0, cfg.base.gravity}, 10, 2000);
        SimActuatorWriter actuators(&sim);

        const Tensor mission_target = makeVec3(3.0f, 4.0f, -5.0f);
        const std::array<double, 3> home_pos = {0.0, 0.0, -5.0};
        TrajectoryLimits limits;
        GuidanceSetpointSource setpoint(mission_target, home_pos, limits);

        SixDofPidController ctrl(cfg, {});
        FlightControlConfig fc_cfg;
        fc_cfg.estimator.sensor_health.enabled = true;
        fc_cfg.executor.hover_thrust = static_cast<double>(cfg.base.mass) * cfg.base.gravity;
        fc_cfg.hover_thrust = fc_cfg.executor.hover_thrust;

        FlightControlLoop loop(fc_cfg, &sensors, &actuators, &setpoint, &ctrl);
        loop.init();

        bool seen_return_home = false;
        double traj_duration = 0.0;
        int return_home_step = -1;

        for (int k = 0; k < 10000; ++k) {
            loop.runOneCycle();

            if (setpoint.mode() == GuidanceMode::ReturnHome) {
                if (!seen_return_home) {
                    seen_return_home = true;
                    return_home_step = k;
                    traj_duration = setpoint.trajectoryDuration();
                }
            }

            // 到达 home 附近即停止
            const auto p = readVec(sim.state().pos);
            const double dist_home =
                std::sqrt((p[0] - home_pos[0]) * (p[0] - home_pos[0]) +
                          (p[1] - home_pos[1]) * (p[1] - home_pos[1]) +
                          (p[2] - home_pos[2]) * (p[2] - home_pos[2]));
            if (dist_home < 0.5) {
                break;
            }
        }

        const auto p = readVec(sim.state().pos);
        const double dist_home =
            std::sqrt((p[0] - home_pos[0]) * (p[0] - home_pos[0]) +
                      (p[1] - home_pos[1]) * (p[1] - home_pos[1]) +
                      (p[2] - home_pos[2]) * (p[2] - home_pos[2]));

        std::printf("    触发 ReturnHome 步数: %d\n", return_home_step);
        std::printf("    返航轨迹时长: %.2f s\n", traj_duration);
        std::printf("    末态距 home: %.2f m\n", dist_home);

        check(seen_return_home, "ReturnHome：触发返航模式");
        check(return_home_step >= 2050 && return_home_step <= 2300,
              "ReturnHome：触发步数在预期窗口");
        check(traj_duration > 0.0, "ReturnHome：轨迹成功构建");
        check(dist_home < 1.0, "ReturnHome：最终到达 home 附近（< 1 m）");
    }

    // ================================================================
    // 3. EmergencyLand：陀螺失效后生成垂直下降轨迹
    // ================================================================
    std::printf("\n--- 3. EmergencyLand（陀螺归零触发紧急降落）---\n");
    {
        SixDofConfig cfg;
        SixDofState init{makeVec3(0.0f, 0.0f, -5.0f), makeVec3(0.0f, 0.0f, 0.0f),
                         Tensor{1.0f, 0.0f, 0.0f, 0.0f}, makeVec3(0.0f, 0.0f, 0.0f)};
        SixDofSimulator sim(cfg, init);

        ImuConfig icfg;
        icfg.explicit_bias = true;
        ImuModel imu(icfg, 1u);
        GyroZeroInjector sensors(&sim, &imu, {0.0, 0.0, cfg.base.gravity}, 10, 2000);
        SimActuatorWriter actuators(&sim);

        const Tensor mission_target = makeVec3(0.0f, 0.0f, -5.0f);
        const std::array<double, 3> home_pos = {0.0, 0.0, -5.0};
        TrajectoryLimits limits;
        GuidanceSetpointSource setpoint(mission_target, home_pos, limits);

        SixDofPidController ctrl(cfg, {});
        FlightControlConfig fc_cfg;
        fc_cfg.estimator.sensor_health.enabled = true;
        fc_cfg.executor.hover_thrust = static_cast<double>(cfg.base.mass) * cfg.base.gravity;
        fc_cfg.hover_thrust = fc_cfg.executor.hover_thrust;

        FlightControlLoop loop(fc_cfg, &sensors, &actuators, &setpoint, &ctrl);
        loop.init();

        bool seen_emergency = false;
        bool seen_emergency_mode = false;
        double traj_duration = 0.0;

        for (int k = 0; k < 4000; ++k) {
            if (!loop.runOneCycle()) {
                break;
            }

            if (loop.currentDecision().action == DegradeAction::EmergencyLand) {
                seen_emergency = true;
            }
            if (setpoint.mode() == GuidanceMode::EmergencyLand) {
                if (!seen_emergency_mode) {
                    seen_emergency_mode = true;
                    traj_duration = setpoint.trajectoryDuration();
                }
            }
        }

        std::printf("    触发 EmergencyLand: %s\n", seen_emergency ? "是" : "否");
        std::printf("    制导源切到 EmergencyLand: %s\n", seen_emergency_mode ? "是" : "否");
        std::printf("    垂直下降轨迹时长: %.2f s\n", traj_duration);

        check(seen_emergency, "EmergencyLand：触发紧急降落决策");
        check(seen_emergency_mode, "EmergencyLand：制导源切换到 EmergencyLand 模式");
        check(traj_duration > 0.0, "EmergencyLand：垂直下降轨迹成功构建");
    }

    // ================================================================
    // 4. 模式切换往返：ReturnHome -> Mission
    // ================================================================
    std::printf("\n--- 4. 模式切换往返 ---\n");
    {
        // 直接测试 GuidanceSetpointSource 的状态机，不经过完整闭环
        //（闭环中故障恢复路径长，不适合做单元测试）
        const Tensor mission_target = makeVec3(1.0f, 2.0f, -3.0f);
        const std::array<double, 3> home_pos = {0.0, 0.0, -3.0};
        TrajectoryLimits limits;
        GuidanceSetpointSource setpoint(mission_target, home_pos, limits);

        check(setpoint.mode() == GuidanceMode::Mission, "往返：初始为 Mission");

        DegradeDecision d;
        d.action = DegradeAction::ReturnHome;
        setpoint.onDecisionChanged(d, {3.0, 4.0, -3.0}, 0.0);
        check(setpoint.mode() == GuidanceMode::ReturnHome,
              "往返：决策 ReturnHome 后切到返航模式");
        check(setpoint.trajectoryDuration() > 0.0, "往返：返航轨迹已构建");

        d.action = DegradeAction::Normal;
        setpoint.onDecisionChanged(d, {3.0, 4.0, -3.0}, 1.0);
        check(setpoint.mode() == GuidanceMode::Mission,
              "往返：决策恢复 Normal 后回到 Mission");

        // EmergencyLand -> Normal
        d.action = DegradeAction::EmergencyLand;
        setpoint.onDecisionChanged(d, {0.0, 0.0, -5.0}, 2.0);
        check(setpoint.mode() == GuidanceMode::EmergencyLand,
              "往返：决策 EmergencyLand 后切到紧急降落模式");
        check(setpoint.trajectoryDuration() > 0.0, "往返：紧急降落轨迹已构建");

        d.action = DegradeAction::Normal;
        setpoint.onDecisionChanged(d, {0.0, 0.0, -5.0}, 3.0);
        check(setpoint.mode() == GuidanceMode::Mission,
              "往返：EmergencyLand 恢复后回到 Mission");
    }

    std::printf("\n========================================\n");
    std::printf("%d / %d checks passed\n", g_pass, g_pass + g_fail);
    std::printf("========================================\n");
    return (g_fail == 0) ? 0 : 1;
}
