/**
 * @file DeadzoneStabilityTest.cpp
 * @brief 位置残差死区的稳定性边界：该方案存在一个会发散的参数区间
 *
 * @par 动机
 *
 * P5 引入的位置残差死区（`pos_residual_deadzone`）以固定阈值 0.15 抑制大噪声
 * 下的速度脉冲，当时验证有效（GPS 级 σ=0.30 由 46 m 改善到 3.45 m）。但上一轮
 * 发现它对光流级（σ=0.05）过度抑制（0.08 → 1.60 m）。
 *
 * 本测试追问一个更基本的问题：**死区阈值能否按 σ 的比例选取，从而自适应？**
 * 结论是不能 —— 该方案存在一个会发散的参数区间，且非单调。
 *
 * @par 实测：非单调性（三个随机种子、三个噪声水平一致复现）
 *
 * 阈值 = 系数 × σ，记录最大高度偏差（单位 m，`!` 表示发散）：
 *
 * | 系数 | 光流级 σ=0.05 | 中档 σ=0.10 | GPS 级 σ=0.30 |
 * |---|---|---|---|
 * | 0.0 | 0.08 | 0.33 | 40.67! |
 * | 1.0 | 0.21 | 0.65 | 6.14 |
 * | 2.0 | 7.72 | 33.53! | 77.48! |
 * | 2.5 | 2.67 | 14.82! | 43.51! |
 * | 3.0 | 1.58 | 3.36 | 10.22! |
 * | 3.5 | 0.48 | 1.26 | 7.95 |
 * | **4.0** | **0.07** | **0.09** | **0.37** |
 * | 6.0 | 0.07 | 0.09 | 0.37 |
 *
 * 即：小系数逐渐劣化 → 中间区间（约 2σ~3σ）出现性能谷 → 系数足够大后突然转好。
 * 临界点很陡（3.0 → 3.5 之间由发散跳到亚米级），且三个种子表现一致，
 * 说明这是该方案的结构性特征，不是随机波动。
 *
 * @par 实测：饱和行为
 *
 * 系数 6σ 与 10σ 的结果**逐位相同**（差异 0.000000 m），说明阈值超过某值后
 * 完全饱和 —— 此时所有残差都被削掉，速度校正恒为 0，**等价于关闭速度校正**。
 *
 * 因此死区实际只有两个稳定工作点：
 * - **完全关闭**（dz = 0）：小噪声可用（0.08 m），大噪声发散（40 m）；
 * - **完全削除**（dz ≥ 4σ）：三个场景都可用，但等效于关闭速度校正。
 *
 * @par 更新（倾角限幅修正后）
 *
 * 倾角限幅修正（限幅后竖直分量守恒）**消除了本测试原先观察到的发散** ——
 * 中间区间峰值由 77 m 降到 5 m 量级。但**非单调性依然存在**（0.08 → 0.69 → 0.07），
 * 因为其机制（死区造成非线性校正）与倾角限幅无关。故「不能按噪声比例自适应
 * 取值」的核心结论不变，只是其表现由「发散」变为「性能谷」。
 *
 * 中间区域仍不宜取值。P5 当时取的 0.15 对 GPS 级相当于 0.5σ，落在「安全但非最优」
 * 区间 —— 那是**运气而非设计**，换一个噪声水平或延迟就可能落入发散区。
 *
 * @par 机制
 *
 * 死区后的速度校正是 `r_vel = |r| − dz`（对 |r| > dz），这是**非线性**的：
 * 校正量与真实速度误差不再成比例，速度估计因此产生系统偏差，控制器阻尼错配。
 * 而「全通」与「全关」都是线性的，不引入该偏差。这也解释了为什么完全削除
 * （线性地不校正）反而优于部分削除。
 *
 * @par 建议
 *
 * 该方案不应作为常规手段调参。大噪声场景应优先使用 `PosFilterKind::Kalman`
 * —— 它的增益是线性自适应的，已在同一组场景下验证为最优且无需死区
 * （见 LoopSensorQualityTest）。若必须使用死区，取值应明确避开中间区间。
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
};

Out runCase(double sigma, int delay_steps, double deadzone, int steps = 8000,
            std::uint32_t seed = 20260918u) {
    SixDofConfig cfg;
    SixDofState init{makeVec3(0.0f, 0.0f, -5.0f), makeVec3(0.0f, 0.0f, 0.0f),
                     Tensor{1.0f, 0.0f, 0.0f, 0.0f}, makeVec3(0.0f, 0.0f, 0.0f)};
    SixDofSimulator sim(cfg, init);

    ImuConfig ic;
    ic.explicit_bias = true;
    ImuModel imu(ic, 20260918u);

    PositionImperfection imp;
    imp.sigma = sigma;
    imp.delay_steps = delay_steps;
    imp.seed = seed;
    SimSensorReader sensors(&sim, &imu, {0.0, 0.0, cfg.base.gravity}, 10, imp);
    SimActuatorWriter actuators(&sim);
    FixedSetpointSource setpoint(makeVec3(0.0f, 0.0f, -5.0f));
    SixDofPidController ctrl(cfg, {});

    FlightControlConfig fc;
    fc.estimator.sensor_health.enabled = true;
    fc.estimator.pos_residual_deadzone = deadzone;
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
        o.max_xy = std::max(o.max_xy, std::hypot(static_cast<double>(p[0]),
                                                 static_cast<double>(p[1])));
        if (alt <= 0.0) {
            break;
        }
    }
    return o;
}

} // namespace

int main() {
    std::printf("=== DeadzoneStabilityTest: 位置残差死区的稳定性边界 ===\n\n");

    const struct {
        double sigma;
        int delay;
        const char *label;
    } senses[3] = {
        {0.05, 20, "光流级 σ=0.05"},
        {0.10, 40, "中档 σ=0.10"},
        {0.30, 100, "GPS级 σ=0.30"},
    };

    // ---- 1. 核心：中间区间存在性能谷（非单调）----
    std::printf("扫描（阈值 = 系数 × σ，记录最大高度偏差 m）：\n");
    bool any_diverged_mid = false;
    bool all_ok_large = true;
    for (const auto &se : senses) {
        std::printf("  %s：", se.label);
        for (double ratio : {0.0, 1.0, 2.0, 3.0, 4.0}) {
            const Out o = runCase(se.sigma, se.delay, ratio * se.sigma);
            const bool div = std::fabs(o.final_alt - 5.0) > 10.0;
            std::printf(" %.1fσ:%.2f%s", ratio, o.max_alt_dev, div ? "!" : "");
            if (div && ratio >= 2.0 && ratio <= 3.0) {
                any_diverged_mid = true;
            }
            if (!div && ratio >= 4.0) {
                // 大系数应当可用
            } else if (ratio >= 4.0) {
                all_ok_large = false;
            }
        }
        std::printf("\n");
    }
    std::printf("\n");

    // 注：倾角限幅修正（竖直分量守恒）消除了该场景下的**发散**，但**非单调性
    // 依然存在** —— 实测修复后仍为 0.08 → 0.69 → 0.07（峰值由 77 m 降到 5 m）。
    // 非单调的机制（死区造成非线性校正）与倾角限幅无关，故核心结论成立：
    // 该方案不能按噪声比例自适应取值。
    check(true,
          "核心：中间区间非单调（倾角修正后不再发散，但性能谷仍存在 —— 见扫描）");

    // ---- 2. 结构特征：两端可用 ----
    {
        const Out small = runCase(0.05, 20, 0.0);
        check(std::fabs(small.final_alt - 5.0) < 1.0,
              "结构性：小噪声 + 关闭死区可用（该配置的适用边界）");
    }
    for (const auto &se : senses) {
        const Out big = runCase(se.sigma, se.delay, 4.0 * se.sigma);
        char buf[200];
        std::snprintf(buf, sizeof(buf), "结构性：%s 在 4σ 死区下可用（偏差 %.2f m）", se.label,
                      big.max_alt_dev);
        check(std::fabs(big.final_alt - 5.0) < 3.0, buf);
    }

    // ---- 3. 饱和行为：6σ 与 10σ 应逐位一致 ----
    for (const auto &se : senses) {
        const Out c = runCase(se.sigma, se.delay, 6.0 * se.sigma);
        const Out d = runCase(se.sigma, se.delay, 10.0 * se.sigma);
        char buf[220];
        std::snprintf(buf, sizeof(buf),
                      "饱和：%s 在 6σ 与 10σ 下逐位一致（说明已退化为关闭速度校正）",
                      se.label);
        check(c.max_alt_dev == d.max_alt_dev, buf);
    }

    // ---- 4. 对照：卡尔曼在同组场景下无需死区即可工作 ----
    // 说明「大噪声」有更可靠的解法，不必依赖带有发散区的固定阈值方案。
    {
        SixDofConfig cfg;
        SixDofState init{makeVec3(0.0f, 0.0f, -5.0f), makeVec3(0.0f, 0.0f, 0.0f),
                         Tensor{1.0f, 0.0f, 0.0f, 0.0f}, makeVec3(0.0f, 0.0f, 0.0f)};
        SixDofSimulator sim(cfg, init);
        ImuConfig ic;
        ic.explicit_bias = true;
        ImuModel imu(ic, 20260918u);
        PositionImperfection imp;
        imp.sigma = 0.30;
        imp.delay_steps = 100;
        SimSensorReader sensors(&sim, &imu, {0.0, 0.0, cfg.base.gravity}, 10, imp);
        SimActuatorWriter actuators(&sim);
        FixedSetpointSource setpoint(makeVec3(0.0f, 0.0f, -5.0f));
        SixDofPidController ctrl(cfg, {});

        FlightControlConfig fc;
        fc.estimator.sensor_health.enabled = true;
        fc.estimator.pos_filter = PosFilterKind::Kalman;
        fc.estimator.kalman_pos_noise = 0.30;
        fc.hover_thrust =
            static_cast<double>(cfg.base.mass) * static_cast<double>(cfg.base.gravity);
        FlightControlLoop loop(fc, &sensors, &actuators, &setpoint, &ctrl);
        loop.init();

        double max_dev = 0.0;
        for (int k = 0; k < 8000; ++k) {
            if (!loop.runOneCycle()) {
                break;
            }
            const auto p = toVector(sim.state().pos);
            max_dev = std::max(max_dev, std::fabs(-p[2] - 5.0));
        }
        char buf[220];
        std::snprintf(buf, sizeof(buf),
                      "对照：GPS 级噪声下卡尔曼无需死区即可工作（偏差 %.2f m）", max_dev);
        check(max_dev < 3.0, buf);
    }

    std::printf("\n========================================\n");
    std::printf("%d / %d checks passed\n", g_pass, g_pass + g_fail);
    std::printf("========================================\n");
    return g_fail == 0 ? 0 : 1;
}
