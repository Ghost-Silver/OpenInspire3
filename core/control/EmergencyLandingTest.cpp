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

    std::printf("\n========================================\n");
    std::printf("%d / %d checks passed\n", g_pass, g_pass + g_fail);
    std::printf("========================================\n");
    return g_fail == 0 ? 0 : 1;
}
