/**
 * @file EmergencyLandingTest.cpp
 * @brief 组合验证：紧急降落是否真的执行了下降过程
 *
 * @par 为什么需要这个测试
 *
 * 这是「单个模块正确不能推出组合正确」的一个安全相关实例，而且三个模块
 * 各自的测试**全是绿的**：
 *
 * - 制导层：`GuidanceIntegrationTest` 断言降落轨迹成功构建 ✓
 * - 执行器：`DegradeExecutorTest` 断言紧急降落时接管并输出下降推力 ✓
 * - 主循环：`FlightControlLoopTest` 断言陀螺失效后主循环终止 ✓
 *
 * 但组合起来，降落从未真正发生：主循环在检测到紧急降落后标记 `_emergency`，
 * 下一周期入口守卫即 `return false`，循环终止 —— 而降落需要持续多拍。
 * 实测陀螺失效后主循环于第 2150 周期终止，飞机仍停在 5.000 m 原高度。
 *
 * 三个测试之所以都没抓到，是因为它们各自只验证「准备好了」这一层：
 * 轨迹建好了、推力算对了、循环停下来了。没有一条断言「飞机真的落到地面」。
 *
 * @par 断言设计
 *
 * 核心断言是末态高度接近地面（触地），而非「循环是否终止」—— 后者在缺陷
 * 存在时同样成立，无法区分「安全降落」与「悬停等死」。
 */

#include "FlightControlLoop.h"
#include "GuidanceSetpointSource.h"
#include "HalSimulator.h"
#include "ImuModel.h"
#include "SixDofPidController.h"
#include "SixDofSimulator.h"
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

/// 2.0 s 后把陀螺输出强制归零
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

/**
 * @brief 陀螺失效 + 位置高度被钉住（模拟高度通道失效）
 *
 * 用于验证超时保护：触地判定永远无法满足时，主循环必须依靠超时退出，
 * 而不是无限运行。
 */
struct FaultyHeightReader : public HalSensorReader {
    SimSensorReader base;
    int gyro_fault_start;
    double stuck_altitude;
    int step_count = 0;

    FaultyHeightReader(SixDofSimulator *sim, ImuModel *imu, const std::array<double, 3> &g,
                       int decim, int fs, double stuck)
        : base(sim, imu, g, decim), gyro_fault_start(fs), stuck_altitude(stuck) {}

    [[nodiscard]] ImuSample readImu() override {
        ImuSample s = base.readImu();
        if (step_count >= gyro_fault_start) {
            s.gyro = {0.0, 0.0, 0.0};
        }
        ++step_count;
        return s;
    }
    [[nodiscard]] bool hasPositionUpdate() override { return base.hasPositionUpdate(); }

    /// 水平位置保持真实，高度被钉在固定值 —— 触地判定永不可能满足
    [[nodiscard]] std::array<double, 3> readPosition() override {
        auto p = base.readPosition();
        if (step_count >= gyro_fault_start) {
            p[2] = -stuck_altitude;
        }
        return p;
    }

    [[nodiscard]] double time() override { return base.time(); }
};

} // namespace

int main() {
    std::printf("=== EmergencyLandingTest: 紧急降落是否真的执行 ===\n\n");

    SixDofConfig cfg;
    SixDofState init{makeVec3(0.0f, 0.0f, -5.0f), makeVec3(0.0f, 0.0f, 0.0f),
                     Tensor{1.0f, 0.0f, 0.0f, 0.0f}, makeVec3(0.0f, 0.0f, 0.0f)};
    SixDofSimulator sim(cfg, init);

    ImuConfig icfg;
    icfg.explicit_bias = true;
    ImuModel imu(icfg, 1u);
    GyroZeroInjector sensors(&sim, &imu, {0.0, 0.0, cfg.base.gravity}, 10, 2000);
    SimActuatorWriter actuators(&sim);
    SixDofPidController ctrl(cfg, {});

    const Tensor mission_target = makeVec3(0.0f, 0.0f, -5.0f);
    const std::array<double, 3> home = {0.0, 0.0, -5.0};
    TrajectoryLimits limits;
    GuidanceSetpointSource setpoint(mission_target, home, limits);

    FlightControlConfig fc;
    fc.estimator.sensor_health.enabled = true;
    fc.hover_thrust = cfg.base.mass * cfg.base.gravity;

    FlightControlLoop loop(fc, &sensors, &actuators, &setpoint, &ctrl);
    loop.init();

    constexpr int fault_step = 2000;
    bool saw_emergency = false;
    bool saw_emergency_mode = false;
    int emergency_start = -1;
    int end_step = -1;
    double alt_at_emergency = -1.0;
    double max_alt_after_emergency = -1.0;

    for (int k = 0; k < 40000; ++k) {
        if (!loop.runOneCycle()) {
            end_step = k;
            break;
        }
        const auto p = toVector(sim.state().pos);
        const double alt = -p[2];
        if (loop.isEmergency()) {
            if (!saw_emergency) {
                saw_emergency = true;
                emergency_start = k;
                alt_at_emergency = alt;
            }
            max_alt_after_emergency = std::max(max_alt_after_emergency, alt);
        }
        if (setpoint.mode() == GuidanceMode::EmergencyLand) {
            saw_emergency_mode = true;
        }
    }

    const auto fin = toVector(sim.state().pos);
    const double final_alt = -fin[2];

    std::printf("  陀螺失效注入:      第 %d 周期\n", fault_step);
    std::printf("  进入紧急降落:      第 %d 周期（高度 %.3f m）\n", emergency_start,
                alt_at_emergency);
    std::printf("  主循环结束:        第 %d 周期\n", end_step);
    std::printf("  末态高度:          %.3f m\n", final_alt);
    std::printf("  制导切到 EmergencyLand: %s\n", saw_emergency_mode ? "是" : "否");
    std::printf("\n");

    // ---- 1. 故障确实触发了紧急降落（激励确认）----
    check(saw_emergency, "激励确认：陀螺失效后进入紧急降落状态");
    check(saw_emergency_mode, "激励确认：制导源切换到 EmergencyLand 模式");
    check(emergency_start >= fault_step,
          "激励确认：紧急降落发生在故障注入之后（无提前误报）");

    // ---- 2. 【核心】降落必须真的执行 ----
    // 这条断言在缺陷存在时会失败：彼时末态高度是 5.000 m。
    // 它不能被「循环是否终止」替代 —— 那个在缺陷下同样为真。
    check(final_alt <= 0.2,
          "核心：末态高度接近地面（触地），降落确实执行");

    // 高度必须从进入紧急降落时的水平显著下降，排除「一开始就在地面」的假通过
    check(alt_at_emergency > 3.0,
          "前置条件：进入紧急降落时飞机确实在空中（高度 >3 m）");
    check(final_alt < alt_at_emergency - 3.0,
          "核心：高度从进入紧急降落时下降超过 3 m");

    // ---- 3. 降落过程可持续多拍（而非一拍即止）----
    const int descent_cycles = end_step - emergency_start;
    std::printf("  降落过程持续:      %d 周期（%.2f s）\n", descent_cycles,
                descent_cycles * 0.001);
    check(descent_cycles > 500,
          "核心：降落过程持续超过 500 周期（排除「一拍即停」的旧缺陷）");

    // ---- 4. 状态语义区分 ----
    check(loop.isEmergencyComplete(), "结束状态：isEmergencyComplete 为真");
    check(loop.isEmergency(), "结束状态：isEmergency 仍为真（区别于 complete）");

    // ================================================================
    // 场景 2：超时保护 —— 触地判定无法满足时必须退出
    // ================================================================
    std::printf("\n--- 场景 2：高度通道失效下的超时保护 ---\n");
    {
        SixDofConfig cfg2;
        SixDofState init2{makeVec3(0.0f, 0.0f, -5.0f), makeVec3(0.0f, 0.0f, 0.0f),
                          Tensor{1.0f, 0.0f, 0.0f, 0.0f}, makeVec3(0.0f, 0.0f, 0.0f)};
        SixDofSimulator sim2(cfg2, init2);

        ImuConfig icfg2;
        icfg2.explicit_bias = true;
        ImuModel imu2(icfg2, 1u);
        // 位置高度被钉在 50 m：触地判定永远不满足
        FaultyHeightReader sensors2(&sim2, &imu2, {0.0, 0.0, cfg2.base.gravity}, 10, 2000, 50.0);
        SimActuatorWriter actuators2(&sim2);
        SixDofPidController ctrl2(cfg2, {});

        const Tensor mission_target2 = makeVec3(0.0f, 0.0f, -5.0f);
        const std::array<double, 3> home2 = {0.0, 0.0, -5.0};
        TrajectoryLimits limits2;
        GuidanceSetpointSource setpoint2(mission_target2, home2, limits2);

        FlightControlConfig fc2;
        fc2.estimator.sensor_health.enabled = true;
        fc2.hover_thrust =
            static_cast<double>(cfg2.base.mass) * static_cast<double>(cfg2.base.gravity);
        // 用较小的超时便于观察，同时验证该配置确实被消费
        fc2.max_emergency_cycles = 3000;

        FlightControlLoop loop2(fc2, &sensors2, &actuators2, &setpoint2, &ctrl2);
        loop2.init();

        int end2 = -1;
        int emg2 = -1;
        for (int k = 0; k < 200000; ++k) {
            if (!loop2.runOneCycle()) {
                end2 = k;
                break;
            }
            if (loop2.isEmergency() && emg2 < 0) {
                emg2 = k;
            }
        }

        std::printf("  进入紧急降落: 第 %d 周期\n", emg2);
        std::printf("  主循环结束:   第 %d 周期（超时上限 %d）\n", end2, fc2.max_emergency_cycles);
        std::printf("  降落持续:     %d 周期\n", end2 - emg2);

        check(end2 > 0, "超时保护：主循环最终退出（未无限运行）");
        check(emg2 > 0 && emg2 < end2, "超时保护：确实进入过紧急降落");
        // 超时上限必须被真正消费：结束点应落在上限附近（不能远超，否则保护无效）
        const int descent = end2 - emg2;
        check(descent >= fc2.max_emergency_cycles &&
                  descent <= fc2.max_emergency_cycles + 50,
              "超时保护：降落时长与配置上限一致（配置被实际消费）");
        check(loop2.isEmergencyComplete(), "超时保护：结束状态标记为 complete");
    }

    std::printf("\n========================================\n");
    std::printf("%d / %d checks passed\n", g_pass, g_pass + g_fail);
    std::printf("========================================\n");
    return g_fail == 0 ? 0 : 1;
}
