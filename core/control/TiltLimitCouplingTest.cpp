/**
 * @file TiltLimitCouplingTest.cpp
 * @brief 倾角限幅与推力模长的不一致：一个被长期忽略的竖直耦合缺陷
 *
 * @par 缺陷
 *
 * `solveCommand` 先由期望合力算出推力模长 `f_norm = norm(f_des)` 与方向
 * `z_des = −f_des/f_norm`，再对方向做倾角限幅。**但限幅只改方向、不改模长**——
 * 两者在限幅触发后不一致。代码注释「倾斜时自动增大以维持竖直分量」只在方向
 * 未被限幅时成立。
 *
 * 后果：方向被压回 `max_tilt` 后，竖直分量由 `f_norm·cos(θ_ideal)` 变为
 * `f_norm·cos(max_tilt)`（**更大**），飞机被推高。
 *
 * @par 实测（10 m 水平 + 4 m 上升，竖直超调）
 *
 * | 倾角上限 | 修复前 | 修复后 |
 * |---|---|---|
 * | 20° | 13.593 m | 0.076 m |
 * | 35°（默认） | **3.181 m** | **0.141 m** |
 * | 45° | 1.333 m | 0.209 m |
 * | 70° | 0.592 m | 0.518 m |
 *
 * 严格单调地随倾角上限变化，确认它是主导因素。作为对照，**姿态增益缩放
 * （0.5×/1×/2×）对结果毫无影响** —— 排除了「姿态环带宽不足」这一假设。
 *
 * @par 修复
 *
 * 限幅后按新方向反推模长，使竖直分量守恒：
 * @verbatim
 *   thrust_norm·(−z_des.z) = f_des.z   ⇒   thrust_norm = −f_des.z / z_des.z
 * @endverbatim
 * 注意负号 —— 漏掉会使推力变成负值（首次实现即犯此错，被 LargeManeuverTest 的
 * 既有断言捕获：高度偏差由「上升 0.63 m」变为「下沉 0.11 m」）。
 *
 * @par 连带改善（远超预期）
 *
 * 该修正消除了一个长期存在的**错误激励源**（限幅时推力过大 → 高度被顶起 →
 * 位置环反向修正 → 与水平机动耦合 → 振荡放大），因而多处既有问题一并消失：
 *
 * | 场景 | 修复前 | 修复后 |
 * |---|---|---|
 * | 位置环 GPS 级噪声（裸循环） | 46 m | **0.46 m** |
 * | 主循环 GPS 级噪声（α-β） | 45.56 m | **5.42 m** |
 * | 无时间戳 500 Hz | 22.91 m | **4.75 m** |
 * | R 过小（过度自信） | 20.01 m | **0.79 m** |
 *
 * 相应地，多个原先把「发散」编码为预期行为的断言已更新为「确认有界」。
 *
 * @par 关节说明
 *
 * `LargeManeuverTest` 早已识别到此副作用（其注释写明「限幅只旋转方向不缩放
 * 大小，竖直分量会超过 g，导致上升」，并给出「同时缩放水平加速度需求」的修正
 * 方向）。本文件把它从「观察记录」推进为「修复 + 回归断言」。
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
    double final_err = 0.0;   ///< 末态位置误差
    double peak_overshoot = 0.0;
    double settle_step = -1;  ///< 回到 5% 带内的步数（-1 表示未收敛）
    int max_accel_sat = 0;    ///< 期望加速度饱和次数
    double h_overshoot = 0.0; ///< 水平方向超调
    double v_overshoot = 0.0; ///< 垂直方向超调
    bool crashed = false;
};

/// @param step_size 阶跃幅度（米）
/// @param vertical  true 表示垂直方向（高度变化），false 表示水平方向
Out runStep(double step_size, bool vertical, int steps = 12000) {
    SixDofConfig cfg;
    SixDofState init{makeVec3(0.0f, 0.0f, -5.0f), makeVec3(0.0f, 0.0f, 0.0f),
                     Tensor{1.0f, 0.0f, 0.0f, 0.0f}, makeVec3(0.0f, 0.0f, 0.0f)};
    SixDofSimulator sim(cfg, init);
    ImuConfig ic;
    ic.explicit_bias = true;
    ImuModel imu(ic, 20260918u);
    SimSensorReader sensors(&sim, &imu, {0.0, 0.0, cfg.base.gravity}, 10);
    SimActuatorWriter actuators(&sim);
    SixDofPidController ctrl(cfg, {});

    // 阶跃：t >= 1.0 s 后目标跳到 step_size
    struct StepSource : public HalSetpointSource {
        double size;
        bool vert;
        StepSource(double s, bool v) : size(s), vert(v) {}
        Tensor currentTarget(double t) override {
            if (t < 1.0) {
                return makeVec3(0.0f, 0.0f, -5.0f);
            }
            // 垂直：高度 = 5 − size（NED 下 z 为负）
            return vert ? makeVec3(0.0f, 0.0f, static_cast<float>(-5.0 - size))
                        : makeVec3(static_cast<float>(size), 0.0f, -5.0f);
        }
        [[nodiscard]] bool hasArrived(const std::array<double, 3> &, double) const override {
            return false;
        }
    } src(step_size, vertical);

    FlightControlConfig fc;
    fc.estimator.sensor_health.enabled = true;
    fc.hover_thrust = static_cast<double>(cfg.base.mass) * static_cast<double>(cfg.base.gravity);
    FlightControlLoop loop(fc, &sensors, &actuators, &src, &ctrl);
    loop.init();

    Out o;
    constexpr double band = 0.05; // 5% 带
    bool entered = false;
    for (int k = 0; k < steps; ++k) {
        if (!loop.runOneCycle()) {
            break;
        }
        const auto p = toVector(sim.state().pos);
        const double alt = -p[2];
        if (alt <= 0.0) {
            o.crashed = true;
            break;
        }
        // 以「朝目标方向推进的进度」为度量，两个方向统一
        // 垂直方向的「进度」= 高度增量（alt 由 5.0 起算），水平方向 = 位移
        const double progress = vertical ? (alt - 5.0) : static_cast<double>(p[0]);
        o.final_err = std::fabs(progress - step_size);
        if (progress > step_size) {
            o.peak_overshoot = std::max(o.peak_overshoot, progress - step_size);
        }
        // 首次进入 5% 带
        if (k > 1000 && !entered && o.final_err < step_size * band) {
            entered = true;
            o.settle_step = k;
        } else if (entered && o.final_err >= step_size * band) {
            entered = false;
            o.settle_step = -1; // 又出去了
        }
    }
    return o;
}

/// 组合阶跃：水平与垂直同时跳变（复现第 8 轮观察到的超大超调）
Out runCombined(double horiz, double vert, double att_scale = 1.0,
                double max_tilt = 35.0, int steps = 12000) {
    SixDofConfig cfg;
    SixDofState init{makeVec3(0.0f, 0.0f, -5.0f), makeVec3(0.0f, 0.0f, 0.0f),
                     Tensor{1.0f, 0.0f, 0.0f, 0.0f}, makeVec3(0.0f, 0.0f, 0.0f)};
    SixDofSimulator sim(cfg, init);
    ImuConfig ic;
    ic.explicit_bias = true;
    ImuModel imu(ic, 20260918u);
    SimSensorReader sensors(&sim, &imu, {0.0, 0.0, cfg.base.gravity}, 10);
    SimActuatorWriter actuators(&sim);
    SixDofPidGains gains;
    gains.att_kp = 0.9 * att_scale;
    gains.att_kd = 0.25 * att_scale;
    gains.max_tilt_deg = max_tilt;
    SixDofPidController ctrl(cfg, gains);

    struct ComboSource : public HalSetpointSource {
        double h, v;
        ComboSource(double hh, double vv) : h(hh), v(vv) {}
        Tensor currentTarget(double t) override {
            if (t < 1.0) {
                return makeVec3(0.0f, 0.0f, -5.0f);
            }
            return makeVec3(static_cast<float>(h), 0.0f, static_cast<float>(-5.0 - v));
        }
        [[nodiscard]] bool hasArrived(const std::array<double, 3> &, double) const override {
            return false;
        }
    } src(horiz, vert);

    FlightControlConfig fc;
    fc.estimator.sensor_health.enabled = true;
    fc.hover_thrust = static_cast<double>(cfg.base.mass) * static_cast<double>(cfg.base.gravity);
    FlightControlLoop loop(fc, &sensors, &actuators, &src, &ctrl);
    loop.init();

    Out o;
    double max_alt = 0.0, max_x = 0.0;
    o.h_overshoot = 0.0;
    o.v_overshoot = 0.0;
    for (int k = 0; k < steps; ++k) {
        if (!loop.runOneCycle()) {
            break;
        }
        const auto p = toVector(sim.state().pos);
        const double alt = -p[2];
        if (alt <= 0.0) {
            o.crashed = true;
            break;
        }
        const double x = static_cast<double>(p[0]);
        max_alt = std::max(max_alt, alt);
        max_x = std::max(max_x, x);
        o.final_err = std::hypot(x - horiz, alt - (5.0 + vert));
    }
    o.h_overshoot = std::max(0.0, max_x - horiz);
    o.v_overshoot = std::max(0.0, max_alt - (5.0 + vert));
    o.peak_overshoot = std::max(o.h_overshoot, o.v_overshoot);
    return o;
}

} // namespace

int main() {
    std::printf("=== TiltLimitCouplingTest: 倾角限幅 × 推力模长一致性 ===\n\n");

    // ---- 1. 缺陷与修复的核心指标（10m 水平 + 4m 上升）----
    const Out def = runCombined(10.0, 4.0, 1.0, 35.0);
    std::printf("默认倾角 35°：水平超调 %.3f m，垂直超调 %.3f m，未达量 %.3f m\n",
                def.h_overshoot, def.v_overshoot, def.final_err);

    check(!def.crashed, "基准：组合机动未坠地");
    check(def.v_overshoot < 0.5,
          "核心：竖直分量守恒 —— 垂直超调 <0.5 m（修复前为 3.181 m）");
    check(def.final_err < 0.1, "核心：组合机动最终到达目标");

    // ---- 2. 单调性：倾角上限是主导因素 ----
    std::printf("\n倾角上限对垂直超调的影响：\n");
    double v20 = 0.0, v70 = 0.0;
    for (double tilt : {20.0, 35.0, 45.0, 70.0}) {
        const Out o = runCombined(10.0, 4.0, 1.0, tilt);
        std::printf("  %5.0f°: 垂直超调 %.3f m\n", tilt, o.v_overshoot);
        if (tilt == 20.0) {
            v20 = o.v_overshoot;
        }
        if (tilt == 70.0) {
            v70 = o.v_overshoot;
        }
    }
    check(v20 < 1.0, "核心：20° 严格限幅下仍不发散（修复前 13.593 m）");

    // ---- 3. 对照：姿态增益无影响（排除带宽假设）----
    std::printf("\n对照：姿态增益缩放\n");
    double v_lo = 0.0, v_hi = 0.0;
    for (double sc : {0.5, 1.0, 2.0}) {
        const Out o = runCombined(10.0, 4.0, sc, 35.0);
        std::printf("  %.1fx: 垂直超调 %.3f m\n", sc, o.v_overshoot);
        if (sc == 0.5) {
            v_lo = o.v_overshoot;
        }
        if (sc == 2.0) {
            v_hi = o.v_overshoot;
        }
    }
    check(v_hi == v_lo,
          "对照：姿态增益缩放对垂直超调无影响（逐位相同，排除带宽假设）");

    // ---- 4. 纯方向阶跃仍正常（确认未破坏基础响应）----
    const Out h_step = runStep(8.0, false);
    const Out v_step = runStep(8.0, true);
    std::printf("\n基础阶跃（8 m）：水平超调 %.3f m，垂直超调 %.3f m\n", h_step.peak_overshoot,
                v_step.peak_overshoot);
    check(h_step.final_err < 0.1, "无劣化：8 m 水平阶跃仍精确到位");
    check(v_step.final_err < 0.1, "无劣化：8 m 垂直阶跃仍精确到位");
    check(v_step.peak_overshoot < 0.5, "无劣化：8 m 垂直阶跃超调小（未限幅，行为不变）");

    std::printf("\n========================================\n");
    std::printf("%d / %d checks passed\n", g_pass, g_pass + g_fail);
    std::printf("========================================\n");
    return g_fail == 0 ? 0 : 1;
}
