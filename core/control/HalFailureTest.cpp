/**
 * @file HalFailureTest.cpp
 * @brief HAL 失败上报通道：修复一个「数据合理但完全错误」的检测盲区
 *
 * @par 缺陷
 *
 * `HalSensorReader` 原先没有失败语义：`readImu()` 返回 `ImuSample` 值类型，
 * HAL 即便遇到 I2C 超时/总线错误也只能返回某个值。而估计器的健康检测全部
 * 基于**数据特征**，因此对「数据看起来合理但完全错误」无能为力。
 *
 * 实测：HAL 持续返回随机值（模长接近正常）时，姿态误差由 1.45° 一路涨到
 * 180°，而**传感器健康状态全程保持 Healthy**，最终坠机。三类数据特征检测
 * 各有失效原因：
 *
 * - 掉线检测看模长 —— 随机值的模长恰好接近正常；
 * - 卡死检测看是否变化 —— 随机值每帧都在变；
 * - CUSUM 看残差 —— 姿态估计被陀螺垃圾污染后，残差反而自洽。
 *
 * 关键在于：**HAL 自己知道读取失败**（它就是在 I2C 上失败的），却无法上报。
 * 这类信息应当直接传递，而不是让估计器从数据里去猜。
 *
 * @par 修复
 *
 * 1. `HalSensorReader` 新增 `lastImuValid()`（默认 true，既有 HAL 无需改动）；
 * 2. `SensorHealth` 的 `update()` 新增 `imu_read_ok` 参数（默认 true），
 *    连续失败达到 `external_fail_steps`（默认 10 帧）即判定失效；
 * 3. 主循环把 HAL 的有效标志透传给估计器。
 *
 * @par 效果（A/B 对照：同一份失败数据，唯一变量是是否上报）
 *
 * | 失败模式 | HAL 上报 | 末态高度 | 姿态误差 | 坠地 |
 * |---|---|---|---|---|
 * | 持续随机垃圾 | 否 | -0.01 m | **179.984°** | **是** |
 * | 持续随机垃圾 | **是** | **0.10 m** | **1.578°** | 否 |
 *
 * 姿态误差改善约 114 倍。其余三类失败（全零 / 上次值 / 5% 随机）本就被数据
 * 特征检测兜住，上报后无劣化（均正常降落）。
 *
 * @par 诚实标注：这修复的是「可知的失败」，不是「不可知的错误」
 *
 * 本修复的价值在于把 HAL **已知**的失败传递上来。若传感器返回的是「自洽的
 * 错误数据」（例如标定错误、或未报告失败的静默数据损坏），仍无外部参照可判
 * —— 那需要第二条独立传感器或外部观测，属物理限制，不在本修复范围内。
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

enum class FailureMode { None, Zero, Stale, Garbage };

const char *modeName(FailureMode m) {
    switch (m) {
    case FailureMode::None: return "无失败";
    case FailureMode::Zero: return "返回全零";
    case FailureMode::Stale: return "返回上次值";
    case FailureMode::Garbage: return "随机垃圾";
    }
    return "?";
}

struct Out {
    double final_alt = 5.0;
    double max_alt_dev = 0.0;
    double max_att_err = 0.0;
    double final_att_err = 0.0;
    bool crashed = false;
    SensorStatus accel = SensorStatus::Unknown;
    SensorStatus gyro = SensorStatus::Unknown;
    DegradeAction action = DegradeAction::Normal;
    int fail_start = -1;
    int detect_step = -1;
    int crash_step = -1;
    int end_step = -1;
};

/// @param fail_ratio 失败帧占比（1.0 表示持续失败）
struct FailingReader : public HalSensorReader {
    SimSensorReader base;
    FailureMode mode;
    double fail_ratio;
    int step = 0;
    std::mt19937 rng{20260918u};
    std::uniform_real_distribution<double> uni{0.0, 1.0};
    std::normal_distribution<double> gauss{0.0, 1.0};
    ImuSample last{};
    bool report_failure = false; ///< 是否通过 lastImuValid() 上报失败
    bool last_ok = true;

    FailingReader(SixDofSimulator *sim, ImuModel *imu, const std::array<double, 3> &g, int decim,
                  FailureMode m, double ratio, bool report)
        : base(sim, imu, g, decim), mode(m), fail_ratio(ratio), report_failure(report) {}

    [[nodiscard]] bool lastImuValid() const override {
        return report_failure ? last_ok : true;
    }

    [[nodiscard]] ImuSample readImu() override {
        ImuSample s = base.readImu();
        ++step;
        if (mode == FailureMode::None || uni(rng) > fail_ratio) {
            last = s;
            last_ok = true;
            return s;
        }
        last_ok = false;
        switch (mode) {
        case FailureMode::Zero:
            return ImuSample{{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
        case FailureMode::Stale:
            return last;
        case FailureMode::Garbage: {
            ImuSample g2;
            // 垃圾值：量级接近正常（模长约 9.8）但方向随机 —— 这正是
            // 「掉线/卡死」两类检测都看不出的情形。
            for (int i = 0; i < 3; ++i) {
                g2.accel[static_cast<std::size_t>(i)] = 5.6 * gauss(rng);
                g2.gyro[static_cast<std::size_t>(i)] = 0.4 * gauss(rng);
            }
            return g2;
        }
        default:
            return s;
        }
    }
    [[nodiscard]] bool hasPositionUpdate() override { return base.hasPositionUpdate(); }
    [[nodiscard]] std::array<double, 3> readPosition() override { return base.readPosition(); }
    [[nodiscard]] double time() override { return base.time(); }
};

double attErrDeg(const std::array<double, 4> &qe, const Tensor &qt) {
    const auto t = toVector(qt);
    double d = 0.0;
    for (int i = 0; i < 4; ++i) {
        d += qe[static_cast<std::size_t>(i)] * t[static_cast<std::size_t>(i)];
    }
    return 2.0 * std::acos(std::min(1.0, std::fabs(d))) * 180.0 / M_PI;
}

Out run(FailureMode mode, double fail_ratio, bool report_failure, int steps = 8000) {
    SixDofConfig cfg;
    SixDofState init{makeVec3(0.0f, 0.0f, -5.0f), makeVec3(0.0f, 0.0f, 0.0f),
                     Tensor{1.0f, 0.0f, 0.0f, 0.0f}, makeVec3(0.0f, 0.0f, 0.0f)};
    SixDofSimulator sim(cfg, init);
    ImuConfig ic;
    ic.explicit_bias = true;
    ImuModel imu(ic, 20260918u);
    FailingReader sensors(&sim, &imu, {0.0, 0.0, cfg.base.gravity}, 10, mode, fail_ratio,
                          report_failure);
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
        const auto p = toVector(sim.state().pos);
        const double alt = -p[2];
        o.final_alt = alt;
        o.max_alt_dev = std::max(o.max_alt_dev, std::fabs(alt - 5.0));
        const double ae = attErrDeg(loop.estimator().attitude(), sim.state().quat);
        o.max_att_err = std::max(o.max_att_err, ae);
        o.final_att_err = ae;
        // 对「持续随机垃圾」做追踪：观察检测后姿态与决策的演化
        if (false && mode == FailureMode::Garbage && fail_ratio >= 1.0 &&
            (k % 200 == 0 || (k > 180 && k < 420 && k % 40 == 0))) {
            const char *act = "Normal";
            switch (loop.currentDecision().action) {
            case DegradeAction::Cautious: act = "Cautious"; break;
            case DegradeAction::ReturnHome: act = "ReturnHome"; break;
            case DegradeAction::EmergencyLand: act = "EmergencyLand"; break;
            default: break;
            }
            std::printf("    [trace] k=%5d 姿态误差=%8.2f° 高度=%6.2f m 决策=%-14s 健康=%d\n", k,
                        ae, -p[2], act, static_cast<int>(loop.currentHealth().gyro));
        }
        if (mode != FailureMode::None && o.detect_step < 0 &&
            loop.currentHealth().accel != SensorStatus::Healthy &&
            loop.currentHealth().accel != SensorStatus::Unknown) {
            o.detect_step = k;
        }
        if (alt <= 0.0) {
            o.crashed = true;
            o.crash_step = k;
            break;
        }
        o.end_step = k;
    }
    o.accel = loop.currentHealth().accel;
    o.gyro = loop.currentHealth().gyro;
    o.action = loop.currentDecision().action;
    return o;
}

} // namespace

int main() {
    std::printf("=== HalFailureTest: HAL 失败上报通道 ===\n\n");

    // ---- 无上报通道时的表现 ----
    const Out garbage_off = run(FailureMode::Garbage, 1.0, false);
    const Out zero_off = run(FailureMode::Zero, 1.0, false);
    const Out stale_off = run(FailureMode::Stale, 1.0, false);
    const Out sparse_off = run(FailureMode::Garbage, 0.05, false);

    // ---- 有上报通道时的表现 ----
    const Out garbage_on = run(FailureMode::Garbage, 1.0, true);
    const Out zero_on = run(FailureMode::Zero, 1.0, true);
    const Out stale_on = run(FailureMode::Stale, 1.0, true);
    const Out sparse_on = run(FailureMode::Garbage, 0.05, true);

    std::printf("%-20s %-10s %10s %12s %8s\n", "失败模式", "HAL 上报", "末态高度", "姿态误差",
                "坠地");
    auto row = [](const char *label, bool on, const Out &o) {
        std::printf("%-20s %-10s %9.2f m %11.3f° %8s\n", label, on ? "是" : "否", o.final_alt,
                    o.max_att_err, o.crashed ? "是" : "否");
    };
    row("持续随机垃圾", false, garbage_off);
    row("持续随机垃圾", true, garbage_on);
    row("持续返回全零", false, zero_off);
    row("持续返回全零", true, zero_on);
    row("持续返回上次值", false, stale_off);
    row("持续返回上次值", true, stale_on);
    row("5% 帧随机垃圾", false, sparse_off);
    row("5% 帧随机垃圾", true, sparse_on);
    std::printf("\n");

    // ---- 1. 缺陷记录：无上报通道时，垃圾数据导致坠机 ----
    check(garbage_off.crashed, "缺陷记录：无上报通道时持续垃圾数据导致坠机");
    check(garbage_off.max_att_err > 90.0,
          "缺陷记录：无上报通道时姿态估计完全失效（误差 >90°）");
    check(garbage_off.accel == SensorStatus::Failed,
          "缺陷记录：兜底检测最终仍判 Failed，但已在坠机之后（检测无效用于挽救）");

    // ---- 2. 核心：上报通道修复该场景 ----
    // 此两条在修复前失败。
    check(!garbage_on.crashed, "核心：上报失败后同一份垃圾数据不再坠机");
    check(garbage_on.max_att_err < 10.0, "核心：上报失败后姿态误差保持正常（<10°）");
    check(garbage_on.accel == SensorStatus::Failed, "核心：上报失败后健康状态正确判为 Failed");
    check(garbage_on.max_att_err < garbage_off.max_att_err * 0.1,
          "核心：姿态误差改善一个量级以上（实测约 114 倍）");

    // ---- 3. 其余三类失败：本就被数据特征兜住，上报后不得劣化 ----
    check(!zero_off.crashed && !stale_off.crashed && !sparse_off.crashed,
          "基线：全零 / 上次值 / 稀疏垃圾在无上报时也能被数据特征兜住");
    check(!zero_on.crashed, "无劣化：全零 + 上报仍正常降落");
    check(!stale_on.crashed, "无劣化：上次值 + 上报仍正常降落");
    check(!sparse_on.crashed, "无劣化：稀疏垃圾 + 上报仍正常降落");
    check(std::fabs(zero_on.final_alt - zero_off.final_alt) < 0.5,
          "无劣化：全零场景末态高度与无上报时一致量级");

    // ---- 4. 激励确认：失败确实发生了 ----
    check(garbage_on.accel == SensorStatus::Failed || garbage_on.gyro == SensorStatus::Failed,
          "激励确认：上报通道确实触发了失效判定");

    std::printf("\n========================================\n");
    std::printf("%d / %d checks passed\n", g_pass, g_pass + g_fail);
    std::printf("========================================\n");
    return g_fail == 0 ? 0 : 1;
}
