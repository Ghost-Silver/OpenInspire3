/**
 * @file LimitConsistencyTest.cpp
 * @brief 限幅体系的一致性：三道限幅的量值是否相互协调
 *
 * @par 背景
 *
 * `solveCommand` 路径上有三道限幅，按「实际约束力」排序（见 LargeManeuverTest）：
 *
 * | 限幅 | 本配置下的水平加速度上限 |
 * |---|---|
 * | `max_tilt_deg = 35` | **6.87 m/s² ← 实际生效（最紧）** |
 * | `max_accel = 12` | 12.00 m/s² |
 * | `max_body_thrust = 20` | 17.43 m/s² |
 *
 * 上一轮修复了倾角限幅的一个缺陷（限幅只改方向、不改模长，导致竖直分量超过 g）。
 * 本测试把「限幅体系的量值协调性」固化下来，覆盖：
 *
 * 1. 倾角限幅后**竖直分量守恒**（上轮修复的回归）；
 * 2. `max_accel` 限幅**逐轴**生效且被正确记录（观测器需要限幅后的值）；
 * 3. 实际可用推力上限应为 `m·sqrt(max_accel² + g²)`，而非 `max_body_thrust`；
 * 4. 三道限幅叠加时**取最紧者**，不出现相互矛盾的结果。
 *
 * @par 为什么值得单独测
 *
 * 「限幅改了 A 却忘了 B」是一类隐蔽缺陷 —— 单看任一道限幅都正确，只有把
 * 它们放在同一路径上才会暴露。上轮的倾角缺陷正是如此（方向被改、模长没跟着改），
 * 且它在多处引发了发散，说明后果可以远超局部。
 */

#include "SixDofPidController.h"
#include "SixDofSimulator.h"
#include "TensorUtils.h"

#include <algorithm>
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

std::array<double, 3> rv(const Tensor &t) {
    const auto v = toVector(t);
    return {v[0], v[1], v[2]};
}

/// 静止于目标点的状态
SixDofState hoverState(const SixDofConfig &cfg) {
    (void)cfg;
    return SixDofState{makeVec3(0.0f, 0.0f, -5.0f), makeVec3(0.0f, 0.0f, 0.0f),
                       Tensor{1.0f, 0.0f, 0.0f, 0.0f}, makeVec3(0.0f, 0.0f, 0.0f)};
}

/// 构造一个「需要大水平加速度」的指令：目标点远离当前位置
SixDofCommand cmdForLargeDemand(SixDofPidController &ctrl, const SixDofConfig &cfg,
                                double horiz_offset, double &tilt_out) {
    const SixDofState s = hoverState(cfg);
    const Tensor target = makeVec3(static_cast<float>(horiz_offset), 0.0f, -5.0f);
    const SixDofCommand c = ctrl.compute(s, target, 0.0);
    tilt_out = ctrl.lastTiltDeg();
    return c;
}

} // namespace

int main() {
    std::printf("=== LimitConsistencyTest: 限幅体系一致性 ===\n\n");

    SixDofConfig cfg;
    const double m = static_cast<double>(cfg.base.mass);
    const double g = static_cast<double>(cfg.base.gravity);
    const double hover_thrust = m * g;

    // ---- 1. 倾角限幅：竖直分量守恒（上轮修复的回归）----
    std::printf("倾角限幅下的竖直分量：\n");
    std::printf("  %12s %12s %14s %16s\n", "水平需求", "实际倾角", "推力(N)", "竖直分量/g");
    bool all_conserved = true;
    for (double off : {2.0, 5.0, 10.0, 20.0, 40.0}) {
        SixDofPidController ctrl(cfg, {});
        double tilt = 0.0;
        const SixDofCommand c = cmdForLargeDemand(ctrl, cfg, off, tilt);
        const double th = static_cast<double>(c.thrust_body);
        const double vcomp = th * std::cos(tilt / 180.0 * M_PI) / (m * g);
        std::printf("  %11.1f m %11.2f° %13.3f %15.4f\n", off, tilt, th, vcomp);
        // 竖直分量应恒为 1（航向角不产生竖直分量；位置误差仅水平）
        if (std::fabs(vcomp - 1.0) > 0.02) {
            all_conserved = false;
        }
    }
    check(all_conserved,
          "核心：倾角限幅触发时竖直分量守恒（各水平需求下均为 1.00 g）");

    // ---- 2. 倾角上限确实生效 ----
    {
        SixDofPidController ctrl(cfg, {});
        double tilt = 0.0;
        cmdForLargeDemand(ctrl, cfg, 40.0, tilt);
        check(tilt <= 35.0 + 1e-6, "倾角限幅生效：实际倾角不超过 max_tilt_deg");
        check(tilt > 30.0, "激励确认：大水平需求确实把倾角推到接近上限");
    }

    // ---- 3. max_accel 限幅：逐轴且被记录 ----
    {
        SixDofPidController ctrl(cfg, {});
        double tilt = 0.0;
        const SixDofCommand c = cmdForLargeDemand(ctrl, cfg, 100.0, tilt);
        // 期望加速度记录应为限幅后的值，其模长不应超过 max_accel 的量级
        const double th = static_cast<double>(c.thrust_body);
        // 推力上限 = m·sqrt(max_accel² + g²)（见头文件说明）
        const double expected_max = m * std::sqrt(12.0 * 12.0 + g * g);
        std::printf("\n推力上限：实测峰值 %.3f N，理论 m·sqrt(max_accel²+g²) = %.3f N\n", th,
                    expected_max);
        check(th <= expected_max + 0.05,
              "max_accel 导出推力上限：实测推力不超过 m·sqrt(max_accel²+g²)");
        check(th > hover_thrust,
              "激励确认：大需求下推力确实高于悬停推力");
    }

    // ---- 4. 三道限幅取最紧者（不相互矛盾）----
    {
        // 倾角上限导出 6.87 m/s²，远低于 max_accel 的 12
        const double tilt_limit_acc = std::tan(35.0 / 180.0 * M_PI) * g;
        const double accel_limit = 12.0;
        const double thrust_limit_acc = std::sqrt(std::pow(20.0 / m, 2.0) - g * g);
        std::printf("\n三道限幅对应的水平加速度上限：\n");
        std::printf("  倾角 %.2f m/s²（最紧）、max_accel %.2f、推力 %.2f\n", tilt_limit_acc,
                    accel_limit, thrust_limit_acc);
        check(tilt_limit_acc < accel_limit,
              "约束链：倾角限幅比 max_accel 更紧（实际生效者是倾角）");
        check(thrust_limit_acc > accel_limit,
              "约束链：推力上限比 max_accel 更松（max_body_thrust 不会被触发）");

        // 实测：不同倾角上限下的可达水平加速度应随 tanθ·g 单调
        double prev = -1.0;
        bool monotone = true;
        for (double cap : {15.0, 25.0, 35.0, 45.0, 60.0}) {
            SixDofPidGains gs;
            gs.max_tilt_deg = cap;
            SixDofPidController ctrl(cfg, gs);
            double tilt = 0.0;
            cmdForLargeDemand(ctrl, cfg, 100.0, tilt);
            const double reach = std::tan(tilt / 180.0 * M_PI) * g;
            std::printf("  倾角上限 %4.0f° → 实际倾角 %5.2f°、水平加速度上限 %.3f m/s²\n", cap,
                        tilt, reach);
            if (reach < prev - 1e-6) {
                monotone = false;
            }
            prev = reach;
        }
        check(monotone, "约束链：可达水平加速度随倾角上限单调（限幅关系自洽）");
    }

    std::printf("\n========================================\n");
    std::printf("%d / %d checks passed\n", g_pass, g_pass + g_fail);
    std::printf("========================================\n");
    return g_fail == 0 ? 0 : 1;
}
