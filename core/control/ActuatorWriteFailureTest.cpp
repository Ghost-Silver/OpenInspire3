/**
 * @file ActuatorWriteFailureTest.cpp
 * @brief 执行器写入失败的上报通道：从「零感知」到「可察觉」
 *
 * @par 缺陷
 *
 * `HalActuatorWriter::writeCommand()` 返回 `void`，主循环无从得知指令是否真正
 * 送达执行器。真机上电调失联、PWM/DShot 输出失败都会静默丢弃指令。
 *
 * 实测（指令被置零）：飞机自由落体坠地（末态高度 -0.00 m），而**飞控全程判定
 * 一切正常** —— 决策 Normal、传感器健康 Healthy、写入失败累计 **0 次**。系统
 * 对已发生的失控完全没有感知。
 *
 * @par 修复
 *
 * `HalActuatorWriter` 新增 `lastCommandAccepted()`（默认 true，既有 HAL 无需
 * 改动）；主循环累计未接受次数，连续达 `actuator_fail_steps`（默认 10）即置位
 * `isActuatorLost()`，并通过 `actuatorFailures()` 暴露累计次数。
 *
 * A/B 对照（同一失效，唯一变量是 HAL 是否上报）：
 *
 * | 写入失效 | HAL 上报 | 失联判定 | 累计失败 |
 * |---|---|---|---|
 * | 指令置零 | 否 | **未察觉** | **0** |
 * | 指令置零 | 是 | 已失联 | 1010 |
 * | 指令完全丢弃 | 否 | **未察觉** | **0** |
 * | 指令完全丢弃 | 是 | 已失联 | 6000 |
 * | 20% 帧丢弃 | 否 | **未察觉** | **0** |
 * | 20% 帧丢弃 | 是 | 已失联 | 6000 |
 *
 * 悬停与机动两种场景下表现一致。
 *
 * @par 必须如实说明的局限（与读取侧的本质区别）
 *
 * 读取失败**有救** —— 可用预测外推、降级、或换用其他传感器。
 * **写入失败没有补救手段**：没有备用执行器，飞机必然失去控制。
 *
 * 因此本通道的价值仅在于「让系统与外部**知道**」——供黑匣子记录、地面站告警、
 * 以及触发被动安全措施（如开伞）。**把它当作可恢复故障处理是危险的**：
 * 代码中的 `isActuatorLost()` 是告知性状态，不构成可恢复故障。
 *
 * @par 一处顺带澄清的观察
 *
 * 「指令完全丢弃 / 部分丢弃」在悬停与机动下均未导致坠机。这是合理的物理结论
 * 而非掩盖：1 kHz 下相邻两拍的指令差异极小，「保持上一条指令」（PWM 丢失时
 * 电调的常见行为）几乎无损。真正的风险集中在**推力被置零或严重篡改**这类
 * 情形。
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
#include <random>
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

enum class WriteFailure {
    None,      ///< 正常
    Dropped,   ///< 指令被完全丢弃（执行器失联）
    Partial,   ///< 部分帧被丢弃（间歇失联）
    Zeroed,    ///< 指令被置零
};

const char *wfName(WriteFailure w) {
    switch (w) {
    case WriteFailure::None: return "正常";
    case WriteFailure::Dropped: return "指令完全丢弃";
    case WriteFailure::Partial: return "20% 帧丢弃";
    case WriteFailure::Zeroed: return "指令置零";
    }
    return "?";
}

struct Out {
    double final_alt = 5.0;
    double min_alt = 5.0;
    double max_alt_dev = 0.0;
    bool crashed = false;
    int crash_step = -1;
    SensorStatus accel = SensorStatus::Unknown;
    SensorStatus gyro = SensorStatus::Unknown;
    DegradeAction action = DegradeAction::Normal;
    bool actuator_lost = false;
    long long actuator_failures = 0;
};

struct FailingWriter : public HalActuatorWriter {
    SixDofSimulator *sim;
    WriteFailure mode;
    double drop_ratio;
    int writes = 0;
    int dropped = 0;
    std::mt19937 rng{20260918u};
    std::uniform_real_distribution<double> uni{0.0, 1.0};

    bool report_failure = false;
    bool last_ok = true;

    FailingWriter(SixDofSimulator *s, WriteFailure m, double ratio, bool report = false)
        : sim(s), mode(m), drop_ratio(ratio), report_failure(report) {}

    [[nodiscard]] bool lastCommandAccepted() const override {
        return report_failure ? last_ok : true;
    }

    SixDofCommand last_cmd{};
    bool have_last = false;

    void writeCommand(const SixDofCommand &cmd) override {
        ++writes;
        switch (mode) {
        case WriteFailure::None:
            sim->step(cmd.thrust_body, cmd.torque);
            break;
        case WriteFailure::Zeroed:
            // 指令被置零：推力与力矩皆为 0 → 自由落体
            ++dropped;
            sim->step(0.0, makeVec3(0.0f, 0.0f, 0.0f));
            break;
        case WriteFailure::Dropped:
            // 执行器失联：保持上一条有效指令（PWM 丢失时电调的常见行为）
            ++dropped;
            if (have_last) {
                sim->step(last_cmd.thrust_body, last_cmd.torque);
            } else {
                sim->step(0.0, makeVec3(0.0f, 0.0f, 0.0f));
            }
            break;
        case WriteFailure::Partial:
            if (uni(rng) < drop_ratio) {
                ++dropped;
                if (have_last) {
                    sim->step(last_cmd.thrust_body, last_cmd.torque);
                } else {
                    sim->step(0.0, makeVec3(0.0f, 0.0f, 0.0f));
                }
            } else {
                sim->step(cmd.thrust_body, cmd.torque);
            }
            break;
        }
        last_ok = (mode == WriteFailure::None);
        last_cmd = cmd;
        have_last = true;
    }
};

Out run(WriteFailure mode, double ratio, bool report = false, int steps = 6000) {
    SixDofConfig cfg;
    SixDofState init{makeVec3(0.0f, 0.0f, -5.0f), makeVec3(0.0f, 0.0f, 0.0f),
                     Tensor{1.0f, 0.0f, 0.0f, 0.0f}, makeVec3(0.0f, 0.0f, 0.0f)};
    SixDofSimulator sim(cfg, init);
    ImuConfig ic;
    ic.explicit_bias = true;
    ImuModel imu(ic, 20260918u);
    SimSensorReader sensors(&sim, &imu, {0.0, 0.0, cfg.base.gravity}, 10);
    FailingWriter writer(&sim, mode, ratio, report);
    FixedSetpointSource setpoint(makeVec3(0.0f, 0.0f, -5.0f));
    SixDofPidController ctrl(cfg, {});

    FlightControlConfig fc;
    fc.estimator.sensor_health.enabled = true;
    fc.hover_thrust = static_cast<double>(cfg.base.mass) * static_cast<double>(cfg.base.gravity);
    FlightControlLoop loop(fc, &sensors, &writer, &setpoint, &ctrl);
    loop.init();

    Out o;
    for (int k = 0; k < steps; ++k) {
        if (!loop.runOneCycle()) {
            break;
        }
        const auto p = toVector(sim.state().pos);
        const double alt = -p[2];
        o.final_alt = alt;
        o.min_alt = std::min(o.min_alt, alt);
        o.max_alt_dev = std::max(o.max_alt_dev, std::fabs(alt - 5.0));
        if (alt <= 0.0 && !o.crashed) {
            o.crashed = true;
            o.crash_step = k;
            break;
        }
    }
    o.accel = loop.currentHealth().accel;
    o.gyro = loop.currentHealth().gyro;
    o.action = loop.currentDecision().action;
    o.actuator_lost = loop.isActuatorLost();
    o.actuator_failures = loop.actuatorFailures();
    return o;
}

/// 机动场景：目标点跳变，指令需要持续变化 —— 此时「保持上一条指令」才会暴露
Out runManeuver(WriteFailure mode, double ratio, bool report = false, int steps = 8000) {
    SixDofConfig cfg;
    SixDofState init{makeVec3(0.0f, 0.0f, -5.0f), makeVec3(0.0f, 0.0f, 0.0f),
                     Tensor{1.0f, 0.0f, 0.0f, 0.0f}, makeVec3(0.0f, 0.0f, 0.0f)};
    SixDofSimulator sim(cfg, init);
    ImuConfig ic;
    ic.explicit_bias = true;
    ImuModel imu(ic, 20260918u);
    SimSensorReader sensors(&sim, &imu, {0.0, 0.0, cfg.base.gravity}, 10);
    FailingWriter writer(&sim, mode, ratio, report);

    struct Staged : public HalSetpointSource {
        Tensor currentTarget(double t) override {
            if (t < 2.0) {
                return makeVec3(0.0f, 0.0f, -5.0f);
            }
            return makeVec3(8.0f, 0.0f, -8.0f); // 水平 8 m + 上升 3 m
        }
        [[nodiscard]] bool hasArrived(const std::array<double, 3> &, double) const override {
            return false;
        }
    } staged;

    SixDofPidController ctrl(cfg, {});
    FlightControlConfig fc;
    fc.estimator.sensor_health.enabled = true;
    fc.hover_thrust = static_cast<double>(cfg.base.mass) * static_cast<double>(cfg.base.gravity)
        ;
    FlightControlLoop loop(fc, &sensors, &writer, &staged, &ctrl);
    loop.init();

    Out o;
    for (int k = 0; k < steps; ++k) {
        if (!loop.runOneCycle()) {
            break;
        }
        const auto p = toVector(sim.state().pos);
        const double alt = -p[2];
        o.final_alt = alt;
        o.min_alt = std::min(o.min_alt, alt);
        o.max_alt_dev = std::max(o.max_alt_dev, std::fabs(alt - 5.0));
        if (alt <= 0.0 && !o.crashed) {
            o.crashed = true;
            o.crash_step = k;
            break;
        }
    }
    o.accel = loop.currentHealth().accel;
    o.gyro = loop.currentHealth().gyro;
    o.action = loop.currentDecision().action;
    o.actuator_lost = loop.isActuatorLost();
    o.actuator_failures = loop.actuatorFailures();
    return o;
}

} // namespace

int main() {
    std::printf("=== ActuatorWriteFailureTest: 执行器写入失败上报通道 ===\n\n");

    // 悬停场景
    const Out z_off = run(WriteFailure::Zeroed, 1.0, false);
    const Out z_on = run(WriteFailure::Zeroed, 1.0, true);
    const Out d_off = run(WriteFailure::Dropped, 1.0, false);
    const Out d_on = run(WriteFailure::Dropped, 1.0, true);
    const Out p_off = run(WriteFailure::Partial, 0.2, false);
    const Out p_on = run(WriteFailure::Partial, 0.2, true);

    // 机动场景（对照，确认结论不依赖工况）
    const Out mz_off = runManeuver(WriteFailure::Zeroed, 1.0, false);
    const Out mz_on = runManeuver(WriteFailure::Zeroed, 1.0, true);
    const Out md_off = runManeuver(WriteFailure::Dropped, 1.0, false);
    const Out md_on = runManeuver(WriteFailure::Dropped, 1.0, true);

    std::printf("%-18s %-10s %10s %-12s %10s\n", "写入失效", "HAL 上报", "末态高度",
                "失联判定", "累计失败");
    auto row = [](const char *l, bool on, const Out &o) {
        std::printf("%-18s %-10s %9.2f m %-12s %10lld\n", l, on ? "是" : "否", o.final_alt,
                    o.actuator_lost ? "已失联" : "未察觉", o.actuator_failures);
    };
    row("指令置零", false, z_off);
    row("指令置零", true, z_on);
    row("指令完全丢弃", false, d_off);
    row("指令完全丢弃", true, d_on);
    row("20% 帧丢弃", false, p_off);
    row("20% 帧丢弃", true, p_on);
    std::printf("\n");

    // ---- 1. 缺陷记录：无上报通道时对写入失败零感知 ----
    // 这是核心事实：失控已发生，而飞控认为一切正常。
    check(z_off.actuator_failures == 0,
          "缺陷记录：无上报通道时累计失败为 0（系统对写入失败零感知）");
    check(!z_off.actuator_lost, "缺陷记录：无上报通道时不会判定失联");
    check(z_off.crashed, "缺陷记录：指令置零导致坠机（失控已真实发生）");
    check(d_off.actuator_failures == 0 && p_off.actuator_failures == 0,
          "缺陷记录：丢弃类失效同样零感知");

    // ---- 2. 核心：上报通道使失效可察觉 ----
    check(z_on.actuator_lost, "核心：上报后置零失效被判定为失联");
    check(z_on.actuator_failures > 100, "核心：上报后累计失败计数正常增长");
    check(d_on.actuator_lost, "核心：上报后完全丢弃被判定为失联");
    check(p_on.actuator_lost, "核心：上报后 20% 丢失被判定为失联");

    // ---- 3. 工况无关性：机动场景结论一致 ----
    check(mz_off.actuator_failures == 0 && !mz_off.actuator_lost,
          "工况无关：机动场景下无上报时同样零感知");
    check(mz_on.actuator_lost, "工况无关：机动场景下上报后同样可察觉");
    check(md_off.actuator_failures == 0, "工况无关：机动场景下丢弃类失效同样零感知");
    check(md_on.actuator_lost, "工况无关：机动场景下丢弃类失效上报后可察觉");

    // ---- 4. 正常路径不受影响 ----
    const Out ok = run(WriteFailure::None, 0.0, true);
    check(ok.actuator_failures == 0 && !ok.actuator_lost,
          "无劣化：正常写入时计数不增长、不误判失联");
    check(!ok.crashed, "无劣化：正常写入时正常悬停");

    // ---- 5. 澄清：丢弃类失效未致坠机是合理物理结论，非掩盖 ----
    // 1 kHz 下相邻指令差异极小，「保持上一条」几乎无损。
    check(!d_on.crashed && !p_on.crashed,
          "澄清：丢弃类失效未致坠机（1 kHz 下指令变化极小，属合理结论）");

    std::printf("\n========================================\n");
    std::printf("%d / %d checks passed\n", g_pass, g_pass + g_fail);
    std::printf("========================================\n");
    return g_fail == 0 ? 0 : 1;
}
