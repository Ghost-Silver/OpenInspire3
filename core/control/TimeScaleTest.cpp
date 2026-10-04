/**
 * @file TimeScaleTest.cpp
 * @brief 时间尺度：主循环原先硬编码 dt，使系统只能在 1 kHz 附近工作
 *
 * @par 缺陷
 *
 * `FlightControlLoop` 把 `dt` 硬编码为 0.001（1 kHz），注释自述「将来可从
 * 时间戳推导」；而 `ImuSample` 不含时间戳，HAL 也无从提供真实时间信息。
 * 于是采样率一旦偏离 1 kHz，积分与滤波器参数同时错位。
 *
 * 实测（以仿真步长模拟真机频率，主循环假定 1 ms）：
 *
 * | 真机频率 | 末态高度 | 姿态误差峰值 |
 * |---|---|---|
 * | 2 kHz | 5.00 m | 0.88° |
 * | 1 kHz | 5.00 m | 1.27° |
 * | **500 Hz** | **22.91 m** | **76.15°** |
 * | 250 Hz | 坠地 | 93.00° |
 * | 125 Hz | 坠地 | 131.08° |
 *
 * 500 Hz 在低成本飞控上很常见，因此该缺口会直接导致真机不可用。
 *
 * @par 修复
 *
 * `HalSensorReader` 新增 `imuTimestamp()`（默认返回 -1 表示不可用）。
 * 主循环改为用相邻时间戳差分得到 dt，并做合理性检查（非正、过小、过大一律
 * 忽略并回退）。默认实现不提供时间戳，故既有 HAL 与测试行为逐位不变。
 *
 * 修复后：
 *
 * | 真机频率 | 修复前 | 修复后 |
 * |---|---|---|
 * | 500 Hz | 22.91 m | **5.00 m** |
 * | 250 Hz | 坠地 | **5.64 m** |
 * | 125 Hz | 坠地 | 仍发散（见下） |
 *
 * @par 125 Hz 仍发散：属控制带宽限制，非时间尺度问题
 *
 * 该场景下姿态误差仅 5.26°（正常），说明积分已正确 —— 发散来自位置环在低
 * 采样率下相位裕度不足（PID 增益按 1 kHz 整定）。这与时间戳无关，属独立的
 * 整定问题，本测试将其与时间尺度问题**明确区分**并以断言记录，避免将来被
 * 误归因。
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

struct Out {
    double final_alt = 5.0;
    double max_alt_dev = 0.0;
    double max_xy = 0.0;
    double final_att_err = 0.0;
    double max_att_err = 0.0;
    int cycles = 0;
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

/**
 * @param sim_dt   仿真器步长（= 真机的实际采样间隔）
 * @param real_time_scale 用于把仿真步数换算成真实秒数（仅用于日志）
 * @param steps    主循环调用次数
 */
Out run(double sim_dt, bool use_timestamp, int steps = 8000) {
    SixDofConfig cfg;
    cfg.base.dt = sim_dt; // 仿真器按此步长推进
    SixDofState init{makeVec3(0.0f, 0.0f, -5.0f), makeVec3(0.0f, 0.0f, 0.0f),
                     Tensor{1.0f, 0.0f, 0.0f, 0.0f}, makeVec3(0.0f, 0.0f, 0.0f)};
    SixDofSimulator sim(cfg, init);

    ImuConfig ic;
    ic.explicit_bias = true;
    ImuModel imu(ic, 20260918u);

    SimSensorReader sensors(&sim, &imu, {0.0, 0.0, cfg.base.gravity}, 10, {}, use_timestamp);
    SimActuatorWriter actuators(&sim);
    FixedSetpointSource setpoint(makeVec3(0.0f, 0.0f, -5.0f));
    SixDofPidController ctrl(cfg, {});

    FlightControlConfig fc;
    fc.estimator.sensor_health.enabled = true;
    fc.hover_thrust = static_cast<double>(cfg.base.mass) * static_cast<double>(cfg.base.gravity);

    FlightControlLoop loop(fc, &sensors, &actuators, &setpoint, &ctrl);
    loop.init();

    Out o;
    for (int k = 0; k < steps; ++k) {
        if (!loop.runOneCycle()) {
            break;
        }
        ++o.cycles;
        const auto p = rv(sim.state().pos);
        const double alt = -p[2];
        o.final_alt = alt;
        o.max_alt_dev = std::max(o.max_alt_dev, std::fabs(alt - 5.0));
        o.max_xy = std::max(o.max_xy, std::hypot(p[0], p[1]));
        const double ae = attErrDeg(loop.estimator().attitude(), sim.state().quat);
        o.max_att_err = std::max(o.max_att_err, ae);
        o.final_att_err = ae;
        if (alt <= 0.0) {
            break;
        }
    }
    return o;
}

} // namespace

int main() {
    std::printf("=== TimeScaleTest: 采样率与步长推导 ===\n\n");

    const double dt_2k = 0.0005, dt_1k = 0.001, dt_500 = 0.002, dt_250 = 0.004,
                 dt_125 = 0.008;

    const Out no_ts_1k = run(dt_1k, false);
    const Out no_ts_500 = run(dt_500, false);
    const Out no_ts_250 = run(dt_250, false);

    const Out ts_2k = run(dt_2k, true);
    const Out ts_1k = run(dt_1k, true);
    const Out ts_500 = run(dt_500, true);
    const Out ts_250 = run(dt_250, true);
    const Out ts_125 = run(dt_125, true);

    std::printf("%-14s %14s %14s\n", "真机频率", "无时间戳", "有时间戳");
    std::printf("%-14s %13.2f m %13.2f m\n", "2 kHz", run(dt_2k, false).final_alt,
                ts_2k.final_alt);
    std::printf("%-14s %13.2f m %13.2f m\n", "1 kHz", no_ts_1k.final_alt, ts_1k.final_alt);
    std::printf("%-14s %13.2f m %13.2f m\n", "500 Hz", no_ts_500.final_alt, ts_500.final_alt);
    std::printf("%-14s %13.2f m %13.2f m\n", "250 Hz", no_ts_250.final_alt, ts_250.final_alt);
    std::printf("\n");

    // ---- 1. 缺陷记录：无时间戳时偏离 1 kHz 即失控 ----
    check(std::fabs(no_ts_1k.final_alt - 5.0) < 0.5, "无时间戳：1 kHz 正常工作（基准）");
    check(std::fabs(no_ts_500.final_alt - 5.0) > 10.0,
          "缺陷记录：无时间戳时 500 Hz 失控（高度偏差 >10 m）");
    check(no_ts_500.max_att_err > 30.0,
          "缺陷记录：无时间戳时 500 Hz 姿态误差极大（>30°）");
    check(std::fabs(no_ts_250.final_alt - 5.0) > 4.0,
          "缺陷记录：无时间戳时 250 Hz 已无法维持高度");

    // ---- 2. 核心：有时间戳后 500 Hz / 250 Hz 可正常工作 ----
    // 这两条在修复前会失败（22.91 m 与坠地）。
    check(std::fabs(ts_500.final_alt - 5.0) < 1.0,
          "核心：有时间戳时 500 Hz 正常悬停（修复前为 22.91 m）");
    check(ts_500.max_att_err < 10.0, "核心：有时间戳时 500 Hz 姿态误差正常（<10°）");
    check(std::fabs(ts_250.final_alt - 5.0) < 2.0,
          "核心：有时间戳时 250 Hz 正常悬停（修复前为坠地）");
    check(ts_250.max_att_err < 10.0, "核心：有时间戳时 250 Hz 姿态误差正常");

    // ---- 3. 高频不劣化 ----
    check(std::fabs(ts_2k.final_alt - 5.0) < 0.5, "有时间戳：2 kHz 正常工作");

    // ---- 4. 既有行为不变：1 kHz 下两种模式结果一致 ----
    check(ts_1k.max_alt_dev == no_ts_1k.max_alt_dev,
          "兼容性：1 kHz 下有时间戳与无时间戳结果一致（既有行为不变）");

    // ---- 5. 125 Hz 的发散归因：控制带宽限制，非时间尺度问题 ----
    // 姿态误差正常说明积分正确，问题在位置环增益整定 —— 与时间戳无关。
    std::printf("  125 Hz：高度偏差 %.2f m，姿态误差峰值 %.2f°\n",
                std::fabs(ts_125.final_alt - 5.0), ts_125.max_att_err);
    check(ts_125.max_att_err < 10.0,
          "归因：125 Hz 下姿态误差仍正常（说明积分正确，非时间尺度问题）");
    check(std::fabs(ts_125.final_alt - 5.0) > 10.0,
          "归因：125 Hz 仍发散，属位置环低采样率带宽不足（独立问题，已标注）");

    std::printf("\n========================================\n");
    std::printf("%d / %d checks passed\n", g_pass, g_pass + g_fail);
    std::printf("========================================\n");
    return g_fail == 0 ? 0 : 1;
}
