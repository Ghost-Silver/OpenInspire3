/**
 * @file LoopKalmanCombinationTest.cpp
 * @brief 组合验证：生产主循环（FlightControlLoop）× 位置滤波器
 *
 * @par 为什么需要
 *
 * 卡尔曼路径此前只在两个测试的裸循环里验证（PositionNonIdealClosedLoopTest
 * 与 KalmanFaultCombinationTest），而 **FlightControlLoop 是生产主循环**，
 * 从未跑过该路径。裸循环测试会自己组织传感器读取、估计更新与控制计算，
 * 主循环则通过 HAL 抽象层完成同样的事 —— 两者是不同代码，组合行为需分别验证。
 *
 * @par 已知限制（据实标注）
 *
 * `SimSensorReader` 的位置量测取自仿真真值，无噪声、无延迟。因此本测试对
 * 卡尔曼是**偏乐观**的：卡尔曼的核心优势正是处理含噪量测，理想量测下它
 * 与 α-β 的表现本就应该接近。本测试验证的是「生产路径可用、降级链路正确」，
 * 而非「卡尔曼更优」—— 后者由 PositionNonIdealClosedLoopTest 在有噪声的
 * 场景下验证。
 */

#include "FlightControlLoop.h"
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

enum class Fault { None, GyroZero, AccelZero };

/// 包装 SimSensorReader，在第 fault_start 步后注入故障
class FaultInjectingReader : public HalSensorReader {
  public:
    FaultInjectingReader(SimSensorReader *inner, Fault fault, int fault_start)
        : _inner(inner), _fault(fault), _fault_start(fault_start) {}

    [[nodiscard]] ImuSample readImu() override {
        ImuSample s = _inner->readImu();
        ++_steps;
        if (_steps >= _fault_start) {
            if (_fault == Fault::GyroZero) {
                s.gyro = {0.0, 0.0, 0.0};
            } else if (_fault == Fault::AccelZero) {
                s.accel = {0.0, 0.0, 0.0};
            }
        }
        return s;
    }

    [[nodiscard]] bool hasPositionUpdate() override { return _inner->hasPositionUpdate(); }
    [[nodiscard]] std::array<double, 3> readPosition() override { return _inner->readPosition(); }
    [[nodiscard]] double time() override { return _inner->time(); }

  private:
    SimSensorReader *_inner;
    Fault _fault;
    int _fault_start;
    int _steps = 0;
};

struct Out {
    double final_alt = 5.0;
    double max_xy = 0.0;
    bool emergency = false;
    bool landed = false;
    int cycles = 0;
    std::string action;
};

std::string actionName(DegradeAction a) {
    switch (a) {
    case DegradeAction::Normal: return "Normal";
    case DegradeAction::Cautious: return "Cautious";
    case DegradeAction::ReturnHome: return "ReturnHome";
    case DegradeAction::EmergencyLand: return "EmergencyLand";
    }
    return "?";
}

Out runLoop(PosFilterKind filter, Fault fault, int steps = 6000) {
    SixDofConfig cfg;
    SixDofState init{makeVec3(0.0f, 0.0f, -5.0f), makeVec3(0.0f, 0.0f, 0.0f),
                     Tensor{1.0f, 0.0f, 0.0f, 0.0f}, makeVec3(0.0f, 0.0f, 0.0f)};
    SixDofSimulator sim(cfg, init);

    ImuConfig ic;
    ic.explicit_bias = true;
    ic.accel_bias_vec = {0.0, 0.0, 0.0};
    ic.gyro_bias_vec = {0.0, 0.0, 0.0};
    ImuModel imu(ic, 20260918u);

    SimSensorReader raw_sensors(&sim, &imu, {0.0, 0.0, cfg.base.gravity}, 10);
    FaultInjectingReader sensors(&raw_sensors, fault, 2000);
    SimActuatorWriter actuators(&sim);
    FixedSetpointSource setpoint(makeVec3(0.0f, 0.0f, -5.0f));
    SixDofPidController ctrl(cfg, {});

    FlightControlConfig fc;
    fc.estimator.sensor_health.enabled = true;
    fc.estimator.pos_filter = filter;
    fc.hover_thrust = static_cast<double>(cfg.base.mass) * static_cast<double>(cfg.base.gravity);

    FlightControlLoop loop(fc, &sensors, &actuators, &setpoint, &ctrl);
    loop.init();

    Out o;
    for (int k = 0; k < steps; ++k) {
        if (!loop.runOneCycle()) {
            break;
        }
        ++o.cycles;
        const auto p = toVector(sim.state().pos);
        o.final_alt = -p[2];
        o.max_xy = std::max(o.max_xy, std::hypot(static_cast<double>(p[0]),
                                                 static_cast<double>(p[1])));
        if (o.final_alt <= 0.0 && !o.landed) {
            o.landed = true;
            break;
        }
    }
    o.emergency = loop.isEmergency();
    o.action = actionName(loop.currentDecision().action);
    return o;
}

const char *fname(Fault f) {
    switch (f) {
    case Fault::None: return "无故障";
    case Fault::GyroZero: return "陀螺失效";
    case Fault::AccelZero: return "加速度计失效";
    }
    return "?";
}

} // namespace

int main() {
    std::printf("=== LoopKalmanCombinationTest: 生产主循环 × 位置滤波器 ===\n\n");

    const Out none_ab = runLoop(PosFilterKind::AlphaBeta, Fault::None);
    const Out none_kf = runLoop(PosFilterKind::Kalman, Fault::None);
    const Out gyro_ab = runLoop(PosFilterKind::AlphaBeta, Fault::GyroZero);
    const Out gyro_kf = runLoop(PosFilterKind::Kalman, Fault::GyroZero);
    const Out accel_ab = runLoop(PosFilterKind::AlphaBeta, Fault::AccelZero);
    const Out accel_kf = runLoop(PosFilterKind::Kalman, Fault::AccelZero);

    struct Row { const char *name; const Out *ab; const Out *kf; };
    const Row rows[3] = {
        {"无故障", &none_ab, &none_kf},
        {"陀螺失效", &gyro_ab, &gyro_kf},
        {"加速度计失效", &accel_ab, &accel_kf},
    };
    std::printf("%-14s %-10s %8s %10s %8s %-14s\n", "场景", "滤波器", "周期", "末态高度",
                "水平峰值", "决策");
    for (const auto &r : rows) {
        std::printf("%-14s %-10s %8d %9.2f m %7.2f m %-14s\n", r.name, "AlphaBeta",
                    r.ab->cycles, r.ab->final_alt, r.ab->max_xy, r.ab->action.c_str());
        std::printf("%-14s %-10s %8d %9.2f m %7.2f m %-14s\n", "", "Kalman",
                    r.kf->cycles, r.kf->final_alt, r.kf->max_xy, r.kf->action.c_str());
    }
    std::printf("\n");

    // ---- 1. 正常飞行：卡尔曼路径必须与 AlphaBeta 行为一致 ----
    check(none_kf.action == "Normal", "无故障：卡尔曼路径决策为 Normal");
    check(!none_kf.emergency, "无故障：卡尔曼路径未进入紧急状态");
    check(none_kf.cycles == none_ab.cycles,
          "无故障：卡尔曼路径周期数与 AlphaBeta 一致（主循环行为等价）");
    check(std::fabs(none_kf.final_alt - none_ab.final_alt) < 0.1,
          "无故障：卡尔曼路径末态高度与 AlphaBeta 一致");
    check(none_kf.max_xy < 0.1, "无故障：卡尔曼路径水平漂移可忽略");

    // ---- 2. 陀螺失效：紧急降落链路必须在主循环下正确 ----
    check(gyro_kf.action == "EmergencyLand", "陀螺失效：卡尔曼路径决策为 EmergencyLand");
    check(gyro_kf.emergency, "陀螺失效：卡尔曼路径进入紧急状态");
    check(gyro_kf.cycles == gyro_ab.cycles,
          "陀螺失效：卡尔曼路径周期数与 AlphaBeta 一致");

    // ---- 3. 加速度计失效：返航判定不得升级为紧急降落 ----
    check(accel_kf.action == "ReturnHome", "加速度计失效：卡尔曼路径决策为 ReturnHome");
    check(!accel_kf.emergency, "加速度计失效：卡尔曼路径未误入紧急状态");
    check(accel_kf.final_alt > 3.0 && accel_kf.final_alt < 8.0,
          "加速度计失效：卡尔曼路径维持高度");

    // ---- 4. 循环计数非零（防止「一次都没跑」也通过）----
    check(none_kf.cycles > 1000 && gyro_kf.cycles > 1000 && accel_kf.cycles > 1000,
          "激励确认：三个场景的主循环确实运行了足够周期");

    std::printf("\n========================================\n");
    std::printf("%d / %d checks passed\n", g_pass, g_pass + g_fail);
    std::printf("========================================\n");
    return g_fail == 0 ? 0 : 1;
}
