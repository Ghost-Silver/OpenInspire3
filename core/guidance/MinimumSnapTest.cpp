/**
 * @file MinimumSnapTest.cpp
 * @brief Minimum-Snap 轨迹生成器验收测试
 *
 * @par 判据从哪来（对应任务书 §6 的七项验收）
 *
 * 1. 通过航点：位置误差 < 1e-6 m。接口未暴露航点时刻，测试用「到航点的最近
 *    通过时刻」（稠密扫描定区间 + 黄金分割精化）定位通过瞬间 —— 若轨迹真的
 *    经过航点，最近距离为 0；若不经过，最近距离就是实际偏差。该断言会失败。
 * 2. C³ 连续：内部航点左右各偏 1e-9 s 采样，比较 pos/vel/acc/jerk。
 *    偏移量的选取：jerk 的左右差 ~ 2δ·|snap|，snap 量级 ≤ 本场景的 jerk 限 20
 *    的数倍，δ=1e-9 时差异 < 2e-7，离 1e-6 判据有一个数量级余量；
 *    而浮点评估误差 ~1e-14，不会污染判据。
 * 3. 约束满足：每段 1000 点稠密采样，三轴分别抽查 vel/acc/jerk 上限。
 * 4. 边界条件：首尾速度、加速度为零（构造精确满足，断言挡回归）。
 * 5. 退化情形：单航点 / 重合航点 / 共线航点 / 非法约束 / NaN。
 * 6. 时间最优性：单段直线 d 米从静止到静止、|a|≤a_max 时的物理下界
 *    T >= 2√(d/a_max)（bang-bang 全加速-全减速），本测试同时断言
 *    不低于该下界（说明约束没被悄悄放宽）与不超过其 2 倍（说明时间不浪费）。
 * 7. 采样边界：闭区间两端返回 true；越界返回 false 且不修改 out。
 *
 * @par 两个「数学之外」的强校验
 *
 * - 闭式形状对照：单段静止-到达 7 次多项式有精确闭式解
 *   s(τ) = 35τ⁴ − 84τ⁵ + 70τ⁶ − 20τ⁷（及各阶导数，见实现文件头注释）。
 *   测试用该闭式独立核对采样值，验证基矩阵求逆与系数还原无错误。
 * - 约束绑比例：时间分配的目标是「最紧的约束恰好顶到 1.0」。
 *   若实现算错了峰值或缩放阶数，轨迹会「可行但无故变慢」——
 *   这类错误不会违反任何上界断言，只有绑比例断言抓得到。
 */

#include "MinimumSnapTrajectory.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>
#include <utility>
#include <vector>

using namespace oi3;

namespace {

int g_checks = 0;
int g_failed = 0;

void checkTrue(const char *name, bool cond) {
    ++g_checks;
    if (!cond) ++g_failed;
    std::cout << "  [" << (cond ? " ok " : "FAIL") << "] " << name << "\n";
}

/// 单段静止-到达 7 次多项式的闭式形状及其一、二、三阶导数（τ ∈ [0,1]）
void closedFormShape(double t, double out[4]) {
    const double v = 1.0 - t;
    out[0] = t * t * t * t * (35.0 + t * (-84.0 + t * (70.0 - 20.0 * t)));
    out[1] = 140.0 * t * t * t * v * v * v;
    out[2] = 420.0 * t * t * v * v * (1.0 - 2.0 * t);
    out[3] = 840.0 * t * v * (5.0 * t * t - 5.0 * t + 1.0);
}

/// [lo, hi] 上到目标点最近通过的（时刻， 距离）。D 在通过瞬间为 0 且局部光滑，
/// 稠密扫描只负责定位到正确邻域，黄金分割把时刻精化到 ~1e-14 s。
std::pair<double, double> closestPass(const MinimumSnapTrajectory &traj, double lo, double hi,
                                      const std::array<double, 3> &target) {
    auto d2 = [&](double t) {
        FlatReference r;
        if (!traj.sample(t, r)) return std::numeric_limits<double>::infinity();
        double s = 0.0;
        for (int a = 0; a < 3; ++a) {
            const double dx = r.pos[a] - target[a];
            s += dx * dx;
        }
        return s;
    };
    constexpr int kGrid = 8192;
    const double step = (hi - lo) / kGrid;
    int bi = 0;
    double best = std::numeric_limits<double>::infinity();
    for (int i = 0; i <= kGrid; ++i) {
        const double val = d2(lo + step * i);
        if (val < best) { best = val; bi = i; }
    }
    double a = lo + step * std::max(0, bi - 1);
    double b = lo + step * std::min(kGrid, bi + 1);
    const double gr = (std::sqrt(5.0) - 1.0) / 2.0;
    double c = b - gr * (b - a), d = a + gr * (b - a);
    double fc = d2(c), fd = d2(d);
    for (int it = 0; it < 200; ++it) {
        // 最小化：fc < fd ⇒ 极小在 [a, d]，丢弃 (d, b]
        if (fc < fd) { b = d; d = c; fd = fc; c = b - gr * (b - a); fc = d2(c); }
        else         { a = c; c = d; fc = fd; d = a + gr * (b - a); fd = d2(d); }
    }
    const double t_star = 0.5 * (c + d);
    return {t_star, std::sqrt(d2(t_star))};
}

/// out 是否被 sample 动过：与哨兵值逐位相等才算「未修改」
bool untouched(const FlatReference &r, double sentinel) {
    for (int a = 0; a < 3; ++a) {
        if (r.pos[a] != sentinel || r.vel[a] != sentinel || r.acc[a] != sentinel ||
            r.jerk[a] != sentinel)
            return false;
    }
    return r.yaw == sentinel && r.yaw_rate == sentinel;
}

void fillSentinel(FlatReference &r, double v) {
    r.pos = {v, v, v};
    r.vel = {v, v, v};
    r.acc = {v, v, v};
    r.jerk = {v, v, v};
    r.yaw = v;
    r.yaw_rate = v;
}

double maxAbs3(const std::array<double, 3> &v) {
    return std::max({std::fabs(v[0]), std::fabs(v[1]), std::fabs(v[2])});
}

} // namespace

int main() {
    std::cout << "========================================\n";
    std::cout << " MinimumSnapTrajectory 验收测试\n";
    std::cout << "========================================\n";

    // ------------------------------------------------------------------
    // A. 单段直线：闭式解对照 + 时间最优性 + 边界条件
    // ------------------------------------------------------------------
    std::cout << "[A] 单段直线段（有解析答案的情形）\n";
    const double A_d = 3.0;
    const TrajectoryLimits A_lim{10.0, 6.87, 100.0}; // 加速度恰为绑定约束
    const double A_t0 = 2.5;
    MinimumSnapTrajectory trA;
    const bool A_ok = trA.build({GuidanceWaypoint{{0.0, 0.0, 0.0}},
                                 GuidanceWaypoint{{A_d, 0.0, 0.0}}},
                                A_lim, A_t0, 0.35);
    checkTrue("A1 单段 build 成功", A_ok);

    const double A_T = trA.duration();
    const double A_lower = 2.0 * std::sqrt(A_d / A_lim.max_acc); // bang-bang 物理下界
    checkTrue("A2 总时长不低于物理下界 (2*sqrt(d/a_max))", A_T >= A_lower - 1e-9);
    checkTrue("A3 总时长不超过解析下界 2 倍", A_T <= 2.0 * A_lower + 1e-9);
    if (!A_ok) {
        std::cout << "单段构建失败，后续无意义，直接退出。\n";
        std::cout << g_checks - g_failed << " / " << g_checks << " checks passed\n";
        return 1;
    }

    {   // 闭式形状核对：p = d·s(τ), v = d/T·s', a = d/T²·s'', j = d/T³·s'''
        double worst = 0.0;
        bool s_ok = true;
        for (double tau : {0.1, 0.25, 0.5, 0.75, 0.9}) {
            FlatReference r;
            s_ok = trA.sample(A_t0 + tau * A_T, r) && s_ok;
            double sf[4];
            closedFormShape(tau, sf);
            const double scales[4] = {A_d, A_d / A_T, A_d / (A_T * A_T),
                                      A_d / (A_T * A_T * A_T)};
            const std::array<double, 3> meas[4] = {r.pos, r.vel, r.acc, r.jerk};
            for (int k = 0; k < 4; ++k)
                worst = std::max(worst, std::fabs(meas[k][0] - scales[k] * sf[k]));
        }
        checkTrue("A4 采样值与闭式 s(tau) 及其导数一致 (err < 1e-7)", s_ok && worst < 1e-7);
    }

    {   // 正交两轴固定值恒为零，其解必须恒为零（解唯一 ⇒ 系数全零）
        double worst = 0.0;
        bool s_ok = true;
        for (int i = 0; i <= 200; ++i) {
            FlatReference r;
            s_ok = trA.sample(A_t0 + A_T * (i / 200.0), r) && s_ok;
            worst = std::max(worst, std::fabs(r.pos[1]));
            worst = std::max(worst, std::fabs(r.pos[2]));
            worst = std::max(worst, std::fabs(r.vel[2]));
        }
        checkTrue("A5 未激活轴保持精确零", s_ok && worst < 1e-12);
    }

    {
        FlatReference r0, r1;
        const bool b0 = trA.sample(A_t0, r0);
        const bool b1 = trA.sample(A_t0 + A_T, r1);
        checkTrue("A6 起点速度/加速度为零",
                  b0 && maxAbs3(r0.vel) < 1e-9 && maxAbs3(r0.acc) < 1e-9);
        checkTrue("A7 终点速度/加速度为零",
                  b1 && maxAbs3(r1.vel) < 1e-9 && maxAbs3(r1.acc) < 1e-9);
        checkTrue("A8 起点/终点位置落在航点上",
                  b0 && b1 && std::fabs(r0.pos[0]) < 1e-12 &&
                      std::fabs(r1.pos[0] - A_d) < 1e-12);
    }
    checkTrue("A9 feasible() 为真", trA.feasible());

    {   // 加速度是绑定约束：密采 |a| 峰值应恰顶住 6.87（±5% 判据挡缩放错阶）
        double pk = 0.0;
        bool s_ok = true;
        for (int i = 0; i <= 1000; ++i) {
            FlatReference r;
            s_ok = trA.sample(A_t0 + A_T * (i / 1000.0), r) && s_ok;
            pk = std::max(pk, std::fabs(r.acc[0]));
        }
        checkTrue("A10 加速度峰值顶住限值 (时间没有被浪费)", s_ok && pk > 0.95 * A_lim.max_acc);
    }

    {   // 约束变松/变紧，时长必须严格单调跟随 —— 证明限制确实在起作用
        MinimumSnapTrajectory looser, tighter;
        looser.build({GuidanceWaypoint{{0.0, 0.0, 0.0}}, GuidanceWaypoint{{A_d, 0.0, 0.0}}},
                     {10.0, 12.0, 500.0}, 0.0);
        tighter.build({GuidanceWaypoint{{0.0, 0.0, 0.0}}, GuidanceWaypoint{{A_d, 0.0, 0.0}}},
                      {10.0, 3.0, 100.0}, 0.0);
        checkTrue("A11 时长随加速度上限单调 (紧则更慢, 松则更快)",
                  looser.duration() < trA.duration() - 1e-9 &&
                      tighter.duration() > trA.duration() + 1e-9);
    }
    std::printf("    [info] 单段时长 = %.6f s, 物理下界 = %.6f s, 比值 = %.3f\n", A_T,
                A_lower, A_T / A_lower);

    // ------------------------------------------------------------------
    // B. 四维航点的多段轨迹：通过性、C3 连续、三轴约束、偏航字段
    // ------------------------------------------------------------------
    std::cout << "[B] 多航点三维轨迹\n";
    const std::vector<GuidanceWaypoint> Bwps{
        {{0.0, 0.0, 0.0}}, {{1.5, 0.8, -0.5}}, {{3.2, 0.1, 0.4}}, {{4.0, 1.6, 0.0}}};
    const TrajectoryLimits B_lim{}; // 默认 5 / 6.87 / 20
    const double B_t0 = 10.0;
    const double B_yaw = 0.7;
    MinimumSnapTrajectory trB;
    const bool B_ok = trB.build(Bwps, B_lim, B_t0, B_yaw);
    checkTrue("B1 多段 build 成功", B_ok);
    checkTrue("B2 feasible() 为真", B_ok && trB.feasible());
    const double B_T = trB.duration();

    {   // 通过全部航点（含端点）
        double worst = 0.0;
        std::vector<double> t_wp;
        for (const auto &w : Bwps) {
            auto [t, dist] = closestPass(trB, B_t0, B_t0 + B_T, w.pos);
            t_wp.push_back(t);
            worst = std::max(worst, dist);
        }
        checkTrue("B3 轨迹通过每个航点 (dist < 1e-6 m)", worst < 1e-6);

        // C3 连续：内部两个航点时刻左右采样样条四阶输出
        bool s_ok = true;
        double jump = 0.0;
        for (size_t k = 1; k + 1 < Bwps.size(); ++k) {
            FlatReference L, R;
            const double eps = 1e-9;
            s_ok = trB.sample(t_wp[k] - eps, L) && s_ok;
            s_ok = trB.sample(t_wp[k] + eps, R) && s_ok;
            for (int a = 0; a < 3; ++a) {
                jump = std::max(jump, std::fabs(L.pos[a] - R.pos[a]));
                jump = std::max(jump, std::fabs(L.vel[a] - R.vel[a]));
                jump = std::max(jump, std::fabs(L.acc[a] - R.acc[a]));
                jump = std::max(jump, std::fabs(L.jerk[a] - R.jerk[a]));
            }
        }
        checkTrue("B4 内部航点 C3 连续 (左右极限差 < 1e-6)", s_ok && jump < 1e-6);
    }

    {
        FlatReference r0, r1;
        const bool b0 = trB.sample(B_t0, r0);
        const bool b1 = trB.sample(B_t0 + B_T, r1);
        checkTrue("B5 起点速度/加速度为零",
                  b0 && maxAbs3(r0.vel) < 1e-9 && maxAbs3(r0.acc) < 1e-9);
        checkTrue("B6 终点速度/加速度为零",
                  b1 && maxAbs3(r1.vel) < 1e-9 && maxAbs3(r1.acc) < 1e-9);
    }

    {   // 三轴分别、每段 1000 点密采的约束检查（判据按构造余量留到 1+1e-9）
        double pv = 0.0, pa = 0.0, pj = 0.0;
        bool s_ok = true;
        const int n_seg = static_cast<int>(Bwps.size()) - 1;
        for (int i = 0; i < 1000 * n_seg; ++i) {
            // 端点重复计入无妨；避开精确端点以防浮点取段歧义影响统计
            const double t = B_t0 + B_T * ((i + 0.5) / (1000.0 * n_seg));
            FlatReference r;
            s_ok = trB.sample(t, r) && s_ok;
            for (int a = 0; a < 3; ++a) {
                pv = std::max(pv, std::fabs(r.vel[a]));
                pa = std::max(pa, std::fabs(r.acc[a]));
                pj = std::max(pj, std::fabs(r.jerk[a]));
            }
        }
        checkTrue("B7 速度三轴不超限", s_ok && pv <= B_lim.max_vel * (1.0 + 1e-9));
        checkTrue("B8 加速度三轴不超限", pa <= B_lim.max_acc * (1.0 + 1e-9));
        checkTrue("B9 jerk 三轴不超限", pj <= B_lim.max_jerk * (1.0 + 1e-9));
        const double rho = std::max({pv / B_lim.max_vel, pa / B_lim.max_acc,
                                     pj / B_lim.max_jerk});
        checkTrue("B10 最紧约束被顶住 ( rho >= 0.95, 时长未浪费 )", rho >= 0.95);
        std::printf("    [info] 多段时长 = %.4f s, rho_v = %.3f, rho_a = %.3f, rho_j = %.3f\n",
                    B_T, pv / B_lim.max_vel, pa / B_lim.max_acc, pj / B_lim.max_jerk);
    }

    {   // 偏航字段必须原样透传：FlatReference 直接喂控制器，这里映射错会静默
        FlatReference r;
        const bool s_ok = trB.sample(B_t0 + 0.5 * B_T, r);
        checkTrue("B11 yaw 恒为调用方指定值, yaw_rate 为 0",
                  s_ok && r.yaw == B_yaw && r.yaw_rate == 0.0);
    }

    // ------------------------------------------------------------------
    // C. 边界与退化情形
    // ------------------------------------------------------------------
    std::cout << "[C] 边界与退化情形\n";

    {   // 未构建对象
        MinimumSnapTrajectory empty;
        FlatReference r;
        fillSentinel(r, -777.0);
        const bool sampled = empty.sample(1.0, r);
        checkTrue("C1 未构建时 sample 返回 false 且不修改 out", !sampled && untouched(r, -777.0));
        checkTrue("C2 未构建时 duration 为 0", empty.duration() == 0.0);
        checkTrue("C3 未构建时 feasible() 为 false", !empty.feasible());
    }

    {   // 航点不足
        MinimumSnapTrajectory t;
        const bool ok = t.build({GuidanceWaypoint{{1.0, 2.0, 3.0}}}, B_lim, 0.0);
        FlatReference r;
        checkTrue("C4 单航点 build 返回 false", !ok);
        checkTrue("C5 失败构建之后状态等同未构建", t.duration() == 0.0 && !t.sample(0.0, r));
    }

    {   // 相邻重合
        MinimumSnapTrajectory t;
        const bool ok = t.build(
            {GuidanceWaypoint{{1.0, 2.0, 3.0}}, GuidanceWaypoint{{1.0, 2.0, 3.0}}},
            B_lim, 0.0);
        checkTrue("C6 相邻航点重合 build 返回 false", !ok);
    }

    {   // 非法与 NaN 输入
        MinimumSnapTrajectory t1, t2, t3;
        const std::vector<GuidanceWaypoint> w{GuidanceWaypoint{{0.0, 0.0, 0.0}},
                                              GuidanceWaypoint{{1.0, 0.0, 0.0}}};
        checkTrue("C7 max_vel = 0 时 build 返回 false",
                  !t1.build(w, TrajectoryLimits{0.0, 6.87, 20.0}, 0.0));
        checkTrue("C8 max_acc < 0 时 build 返回 false",
                  !t2.build(w, TrajectoryLimits{5.0, -1.0, 20.0}, 0.0));
        const std::vector<GuidanceWaypoint> wnan{GuidanceWaypoint{{0.0, 0.0, 0.0}},
                                                 GuidanceWaypoint{{std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0}}};
        checkTrue("C9 航点含 NaN 时 build 返回 false", !t3.build(wnan, B_lim, 0.0));
    }

    {   // 共线四航点：正确退化为一维运动
        const std::vector<GuidanceWaypoint> w{GuidanceWaypoint{{0.0, 0.0, 0.0}},
                                              GuidanceWaypoint{{1.0, 0.0, 0.0}},
                                              GuidanceWaypoint{{2.0, 0.0, 0.0}},
                                              GuidanceWaypoint{{3.0, 0.0, 0.0}}};
        MinimumSnapTrajectory t;
        const bool ok = t.build(w, B_lim, 30.0);
        checkTrue("C10 共线航点 build 成功且可行", ok && t.feasible());
        double off_axis = 0.0, worst_wp = 0.0;
        bool s_ok = true;
        for (int i = 0; i <= 3000 && ok; ++i) {
            FlatReference r;
            s_ok = t.sample(30.0 + t.duration() * (i / 3000.0), r) && s_ok;
            off_axis = std::max(off_axis, std::fabs(r.pos[1]));
            off_axis = std::max(off_axis, std::fabs(r.pos[2]));
        }
        for (const auto &wp : w) {
            if (ok) worst_wp = std::max(worst_wp,
                                        closestPass(t, 30.0, 30.0 + t.duration(), wp.pos).second);
        }
        checkTrue("C11 共线场景横向分量精确为零", !ok || (s_ok && off_axis < 1e-12));
        checkTrue("C12 共线场景仍通过每个航点", !ok || worst_wp < 1e-6);
    }

    {   // 采样闭区间与越界保护（用 trB：start_time 非零）
        FlatReference r;
        fillSentinel(r, 9e9);
        const bool at_start = trB.sample(B_t0, r);
        const bool at_end = trB.sample(B_t0 + trB.duration(), r);
        checkTrue("C13 t == start_time 与 t == start_time+duration 均返回 true",
                  at_start && at_end);

        const double sentinel = 4.25e11;
        FlatReference o1, o2, o3;
        fillSentinel(o1, sentinel);
        fillSentinel(o2, sentinel);
        fillSentinel(o3, sentinel);
        const bool b1 = trB.sample(B_t0 - 1e-9, o1);
        const bool b2 = trB.sample(B_t0 + trB.duration() + 1e-9, o2);
        const bool b3 = trB.sample(std::numeric_limits<double>::quiet_NaN(), o3);
        checkTrue("C14 越界/NaN 采样返回 false 且不修改 out",
                  !b1 && !b2 && !b3 && untouched(o1, sentinel) &&
                      untouched(o2, sentinel) && untouched(o3, sentinel));
    }

    {   // reset 与重复 build：都不允许残留旧状态
        MinimumSnapTrajectory t;
        t.build(Bwps, B_lim, 0.0, 0.0);
        const double d_first = t.duration();
        t.build(Bwps, B_lim, 0.0, 0.0); // 不 reset 直接重复 build
        checkTrue("C15 重复 build 结果与首次一致 (无状态残留)", t.duration() == d_first);
        t.reset();
        FlatReference r;
        checkTrue("C16 reset 后回到未构建状态", t.duration() == 0.0 && !t.sample(0.0, r));
        const bool reb = t.build(Bwps, B_lim, 0.0, 0.0);
        checkTrue("C17 reset 后可重新构建", reb && t.duration() == d_first);
    }

    std::cout << "========================================\n";
    std::cout << g_checks - g_failed << " / " << g_checks << " checks passed\n";
    if (g_failed > 0) std::cout << g_failed << " FAILED\n";
    std::cout << "========================================\n";
    return g_failed == 0 ? 0 : 1;
}
