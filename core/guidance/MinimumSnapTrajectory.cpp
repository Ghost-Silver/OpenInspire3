/**
 * @file MinimumSnapTrajectory.cpp
 * @brief Minimum-Snap 轨迹生成器：分段七次多项式 + 全局时间缩放
 *
 * @par 数学路线：Hermite 形式 + 无约束二次规划
 *
 * 每段是 7 次多项式，由段两端的「位置、速度、加速度、加加速度」8 个端点状态
 * 唯一确定（Hermite 形式）。于是：
 *
 *  - 内部航点的位置固定 → 每端点剩 v/a/j 三个自由量；
 *  - 分段多项式在内部航点天然 C³ 连续（左右端点状态共享同一组自由量）；
 *  - 每段代价 ∫(p⁗)²dt 是端点状态的二次型，全体拼起来是自由导数 d 的
 *    无约束二次规划 min dᵀHd + 2gᵀd，解为线性方程组 H d = −g。
 *
 * 由样条最优性定理，∫(p⁗)² 的全最优解是 7 次自然样条（内部航点 C⁶），
 * 它就在上述 C³ 分段七次族内，所以 Hermite 族上的最小解即全局 minimum-snap，
 * 与 Mellinger & Kumar (2011) 的端点导数法完全等价，但实现更短、约束永不违反。
 *
 * @par 时间分配：全局均匀缩放（精确成立，非迭代启发）
 *
 * 对任意时刻组 {T_i} 的最优解 p(t)，把自变量换成 t/s 即得到 {s·T_i} 时刻组的
 * 最优解（代价只差一个公共因子 s⁻⁷，minimizer 一一对应、且解唯一）。因此
 * 缩放时间是线性变换：速度 ×1/s，加速度 ×1/s²，jerk ×1/s³。
 * 先用「各段独立静止-到达」的形状因子估计出比例合理的初始时刻，解一次，
 * 再按最吃紧的约束一个比例缩到底 —— 恰好顶住约束边界，一步完成，无需迭代。
 *
 * @par 三个形状常数（7 次单段静止-到达轨迹 s(τ) 的峰值因子）
 *
 * 单段 d 米、T 秒、两端 v/a/j 为零的解具有闭式
 *   s(τ)  = 35τ⁴ − 84τ⁵ + 70τ⁶ − 20τ⁷
 *   s'(τ) = 140 τ³(1−τ)³                    峰值 35/16    = 2.1875 （τ=1/2）
 *   s''(τ) = 420 τ²(1−τ)²(1−2τ)              峰值 84√5/25 ≈ 7.513  （τ=(5−√5)/10）
 *   s'''(τ)= 840 τ(1−τ)(5τ²−5τ+1)            峰值 52.5               （τ=1/2）
 * 即 v_peak = 2.1875·d/T，a_peak = 7.513·d/T²，j_peak = 52.5·d/T³。
 * 时间最优性量级对照：同一约束下最优 bang-bang 需 2√(d/a) 秒，而 min-snap
 * 形状给出 √(7.513·d/a) ≈ 2.74·√(d/a) —— 恒落后于解析下界 1.37 倍，
 * 好远于验收所要求的 2 倍。
 */

#include "MinimumSnapTrajectory.h"

#include <algorithm>
#include <cmath>
#include <functional>

namespace oi3 {

namespace {

/// 升幂系数个数（7 次多项式 8 系数）
constexpr int kNCoeff = 8;
/// 单段端点状态向量 e 的维数：[pL vL aL jL pR vR aR jR]
constexpr int kState = 8;

/// 初始时间分配用的形状因子（推导见文件头注释）
constexpr double kShapeVel = 35.0 / 16.0;   ///< max s'  = 2.1875
constexpr double kShapeAcc = 7.5131884043993; ///< max|s''| = 84√5/25 = 420/(25√5)
constexpr double kShapeJerk = 52.5;         ///< max|s'''|

using Mat4 = std::array<std::array<double, 4>, 4>;
using Mat4x8 = std::array<std::array<double, 8>, 4>;
using Mat8 = std::array<std::array<double, 8>, 8>;

/**
 * @brief 小维度高斯消元（部分主元），解 G(4x4) · X = R(4x8)
 *
 * 4x4 的规模不需要任何库；手写消元避免引入外部依赖，且行为完全确定。
 * 返回 false 表示奇异（段时长非正时才会发生，属于调用方传入的退化输入）。
 */
bool solve4(const Mat4 &G, const Mat4x8 &R, Mat4x8 &X) {
    // 增广 [G | R]，逐行消去
    double aug[4][12];
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) aug[i][j] = G[i][j];
        for (int j = 0; j < 8; ++j) aug[i][4 + j] = R[i][j];
    }
    for (int col = 0; col < 4; ++col) {
        int piv = col;
        for (int i = col + 1; i < 4; ++i) {
            if (std::fabs(aug[i][col]) > std::fabs(aug[piv][col])) piv = i;
        }
        if (std::fabs(aug[piv][col]) < 1e-300) return false;
        if (piv != col) {
            for (int j = col; j < 12; ++j) std::swap(aug[piv][j], aug[col][j]);
        }
        const double d = aug[col][col];
        for (int i = col + 1; i < 4; ++i) {
            const double f = aug[i][col] / d;
            if (f == 0.0) continue;
            for (int j = col; j < 12; ++j) aug[i][j] -= f * aug[col][j];
        }
    }
    for (int col = 3; col >= 0; --col) {
        for (int j = 0; j < 8; ++j) {
            double s = aug[col][4 + j];
            for (int k = col + 1; k < 4; ++k) s -= aug[col][k] * X[k][j];
            X[col][j] = s / aug[col][col];
        }
    }
    return true;
}

/**
 * @brief 段基矩阵的逆 W(T)：c = W·e（e 为两端 [p v a j] 状态，c 为升幂系数）
 *
 * t=0 端给出 c0..c3 的平凡解；t=T 端剩 4×4，解全时段都用它。
 */
bool basisInverse(double T, Mat8 &W) {
    for (auto &row : W) row.fill(0.0);
    W[0][0] = 1.0;                 // c0 = pL
    W[1][1] = 1.0;                 // c1 = vL
    W[2][2] = 0.5;                 // c2 = aL/2
    W[3][3] = 1.0 / 6.0;           // c3 = jL/6

    const double T2 = T * T, T3 = T2 * T, T4 = T3 * T;
    const double T5 = T4 * T, T6 = T5 * T, T7 = T6 * T;
    const Mat4 G{{{T4, T5, T6, T7},
                  {4 * T3, 5 * T4, 6 * T5, 7 * T6},
                  {12 * T2, 20 * T3, 30 * T4, 42 * T5},
                  {24 * T, 60 * T2, 120 * T3, 210 * T4}}};
    // 右端条件扣除 c0..c3（由左端状态直接确定的部分）后的残差，按 e 的 8 个分量展开
    Mat4x8 R{};
    for (auto &row : R) row.fill(0.0);
    R[0][0] = -1.0; R[0][1] = -T;     R[0][2] = -T2 / 2; R[0][3] = -T3 / 6; R[0][4] = 1.0;
    R[1][1] = -1.0; R[1][2] = -T;     R[1][3] = -T2 / 2;                    R[1][5] = 1.0;
    R[2][2] = -1.0; R[2][3] = -T;                                        R[2][6] = 1.0;
    R[3][3] = -1.0;                                                        R[3][7] = 1.0;

    Mat4x8 X{}; // X = G⁻¹R，即 W 的第 4..7 行
    if (!solve4(G, R, X)) return false;
    for (int i = 0; i < 4; ++i) W[4 + i] = X[i];
    return true;
}

/**
 * @brief 段代价二次型 M：∫(p⁗)²dt = eᵀ M e，e 为端点状态
 *
 * ∫(p⁗)²dt = cᵀQc，Q 只在 c4..c7 上非零：
 *   Q[i][j] = π_i π_j T^(i+j−7)/(i+j−7),  π = {24, 120, 360, 840}
 * 再经 c = W e 换元，M = Wᵀ Q W（有效部分只需 W 的 4..7 行）。
 */
bool segCostMatrix(double T, Mat8 &M) {
    Mat8 W{};
    if (!basisInverse(T, W)) return false;
    const double pi[4] = {24.0, 120.0, 360.0, 840.0};
    double q[4][4];
    std::array<double, 8> Tpow{};
    Tpow[0] = 1.0;
    for (int k = 1; k < 8; ++k) Tpow[k] = Tpow[k - 1] * T;
    // T^{i+j-7} for i,j>=4：i+j-7 ∈ 1..7
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            const int p = (i + 4) + (j + 4) - 7; // 1..7
            q[i][j] = pi[i] * pi[j] * Tpow[p] / static_cast<double>(p);
        }
    }
    // M = Aᵀ Q A，A = W 的第 4..7 行（4×8）
    for (int a = 0; a < 8; ++a) {
        for (int b = 0; b < 8; ++b) {
            double s = 0.0;
            for (int i = 0; i < 4; ++i) {
                double acc = 0.0;
                for (int j = 0; j < 4; ++j) acc += q[i][j] * W[4 + j][b];
                s += W[4 + i][a] * acc;
            }
            M[a][b] = s;
        }
    }
    return true;
}

/**
 * @brief Cholesky 解对称正定方程 H d = g（H 是样条能量矩阵，理论上恒正定；
 *        分解遇非正对角元时返回 false 作为「求解失败」的明确出口）
 */
std::vector<double> solveSpd(const std::vector<std::vector<double>> &H,
                             const std::vector<double> &g, bool &ok) {
    const int n = static_cast<int>(g.size());
    ok = false;
    if (n == 0) { ok = true; return {}; }
    std::vector<std::vector<double>> L(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            double s = H[i][j];
            for (int k = 0; k < j; ++k) s -= L[i][k] * L[j][k];
            if (i == j) {
                if (!(s > 0.0) || !std::isfinite(s)) return {};
                L[i][j] = std::sqrt(s);
            } else {
                L[i][j] = s / L[j][j];
            }
        }
    }
    std::vector<double> d(n, 0.0);
    for (int i = 0; i < n; ++i) {           // 前代 L y = g
        double s = g[i];
        for (int k = 0; k < i; ++k) s -= L[i][k] * d[k];
        d[i] = s / L[i][i];
    }
    for (int i = n - 1; i >= 0; --i) {      // 回代 Lᵀ d = y
        double s = d[i];
        for (int k = i + 1; k < n; ++k) s -= L[k][i] * d[k];
        d[i] = s / L[i][i];
    }
    ok = true;
    return d;
}

/// 按 Horner 法求 p 的 0..3 阶导数（数值稳定，避免逐项 pow 的放大误差）
void evalDerivs(const std::array<double, 8> &c, double t, double out[4]) {
    double p = c[7], v = 7 * c[7], a = 42 * c[7], j = 210 * c[7];
    // 逐次乘 t 再折叠：p(t), p'(t), p''(t), p'''(t) 的系数序列
    for (int k = 6; k >= 0; --k) {
        p = p * t + c[k];
        if (k >= 1) v = v * t + k * c[k];
        if (k >= 2) a = a * t + (k * (k - 1)) * c[k];
        if (k >= 3) j = j * t + (k * (k - 1) * (k - 2)) * c[k];
    }
    out[0] = p; out[1] = v; out[2] = a; out[3] = j;
}

/**
 * @brief 区间上 |f| 峰值的可靠估计：粗扫定位 + 黄金分割精化
 *
 * 粗扫单独不可靠：多抽一段网格会把峰值低估 O((Δt)²·f''/f)，
 * 峰值又直接决定「约束被顶到多满」。加一次黄金分割精化后，
 * 误差降到 1e-9 量级，时间缩放才敢精确顶到约束边界。
 */
double peakAbs(const std::function<double(double)> &f, double lo, double hi) {
    constexpr int kGrid = 256;
    double best = -1.0;
    double step = (hi - lo) / kGrid;
    int bi = 0;
    for (int i = 0; i <= kGrid; ++i) {
        const double val = std::fabs(f(lo + step * i));
        if (val > best) { best = val; bi = i; }
    }
    if (best <= 0.0) return 0.0;
    double a = lo + step * std::max(0, bi - 1);
    double b = lo + step * std::min(kGrid, bi + 1);
    const double gr = (std::sqrt(5.0) - 1.0) / 2.0;
    double c = b - gr * (b - a), d = a + gr * (b - a);
    double fc = std::fabs(f(c)), fd = std::fabs(f(d));
    for (int it = 0; it < 60; ++it) {
        if (fc < fd) { a = c; c = d; fc = fd; d = a + gr * (b - a); fd = std::fabs(f(d)); }
        else         { b = d; d = c; fd = fc; c = b - gr * (b - a); fc = std::fabs(f(c)); }
    }
    for (double t : {a, b, c, d}) best = std::max(best, std::fabs(f(t)));
    return best;
}

} // namespace

bool MinimumSnapTrajectory::build(const std::vector<GuidanceWaypoint> &waypoints,
                                  const TrajectoryLimits &limits, double start_time,
                                  double fixed_yaw) {
    // 任何一次 build（成功或失败）都从干净状态开始：
    // 失败的 build 之后的 sample() 必须返回 false，而不是吐出半条旧轨迹。
    reset();
    limits_ = limits;

    // ---- 输入校验（对应契约：航点不足、重合航点、非法约束 → false）----
    const int n_wp = static_cast<int>(waypoints.size());
    if (n_wp < 2) return false;
    if (!(limits.max_vel > 0.0) || !(limits.max_acc > 0.0) || !(limits.max_jerk > 0.0))
        return false;
    if (!std::isfinite(limits.max_vel) || !std::isfinite(limits.max_acc) ||
        !std::isfinite(limits.max_jerk) || !std::isfinite(start_time) ||
        !std::isfinite(fixed_yaw))
        return false;
    for (const auto &w : waypoints) {
        for (double v : w.pos) {
            if (!std::isfinite(v)) return false;
        }
    }
    const int m = n_wp - 1; // 段数
    std::vector<double> seg_dist(m, 0.0);
    for (int i = 0; i < m; ++i) {
        double d2 = 0.0;
        for (int a = 0; a < 3; ++a) {
            const double dx = waypoints[i + 1].pos[a] - waypoints[i].pos[a];
            d2 += dx * dx;
        }
        // 用平方距离与 1e-12 m 比较：零长度段会让 Hermite 系统病态；
        // 比这更短的段对调用方而言就是「重合」而不是「想飞一纳米」。
        if (d2 < 1e-24) return false;
        seg_dist[i] = std::sqrt(d2);
    }

    // ---- 初始时间分配 ----
    // 以「该段若为独立的静止-到达运动」的闭式形状因子为每段估计可行时刻的下界。
    // 多段轨迹内非静止的内节点只会更快，这一估计整体偏保守，
    // 之后的全局缩放只需向下压到第一个约束顶上 —— 比例分配仍保持合理。
    std::vector<double> seg_T(m, 0.0);
    for (int i = 0; i < m; ++i) {
        const double d = seg_dist[i];
        const double t_v = kShapeVel * d / limits.max_vel;
        const double t_a = std::sqrt(kShapeAcc * d / limits.max_acc);
        const double t_j = std::cbrt(kShapeJerk * d / limits.max_jerk);
        seg_T[i] = std::max({t_v, t_a, t_j});
    }

    // ---- 组装并求解自由导数（QP min dᵀHd + 2gᵀd）----
    // 自由量：内部航点 i（1..m-1）的 (v,a,j)，索引 3(i-1)+{0,1,2}。
    // 固定值：航点位置（随轴变化）；首尾 v/a/j = 0（不随轴变化，对 g 无贡献）。
    const int n_free = 3 * (m - 1);
    std::vector<std::vector<double>> H(n_free, std::vector<double>(n_free, 0.0));

    // 每段的 8 个端点条目 → 自由变量索引（-1 表示固定条目）
    // 条目顺序：[pL vL aL jL pR vR aR jR]
    std::vector<std::array<int, 8>> free_map(m);
    for (int i = 0; i < m; ++i) {
        free_map[i].fill(-1);
        for (int k = 0; k < 3; ++k) {
            if (i > 0) free_map[i][1 + k] = 3 * i - 3 + k; // 左端 v/a/j：前一航点的自由量
            if (i < m - 1) free_map[i][5 + k] = 3 * i + k; // 右端 v/a/j：本航点的自由量
        }
    }

    std::vector<Mat8> seg_M(m);
    for (int i = 0; i < m; ++i) {
        if (!segCostMatrix(seg_T[i], seg_M[i])) return false;
        for (int a = 0; a < kState; ++a) {
            const int fa = free_map[i][a];
            if (fa < 0) continue;
            for (int b = 0; b < kState; ++b) {
                const int fb = free_map[i][b];
                if (fb < 0) continue;
                H[fa][fb] += seg_M[i][a][b];
            }
        }
    }

    struct AxisSolution {
        std::array<double, 3> off{}; ///< 内部航点的自由 v/a/j（线性方程的解）
        bool ok = false;
    };
    // 三轴共用同一个 H（只取决于段时间与条目映射），只解一次结构、三次右端项。
    std::array<std::vector<double>, 3> free_vals; // [axis][free index]
    for (int axis = 0; axis < 3; ++axis) {
        std::vector<double> g(n_free, 0.0);
        for (int i = 0; i < m; ++i) {
            for (int a = 0; a < kState; ++a) {
                const int fa = free_map[i][a];
                if (fa < 0) continue;
                // J 写为 dᵀHd + 2gᵀd + const → g_f = Σ M[f][固定]·固定值，
                // 固定条目里只有位置（条目 0/4）非零。
                g[fa] += seg_M[i][a][0] * waypoints[i].pos[axis];
                g[fa] += seg_M[i][a][4] * waypoints[i + 1].pos[axis];
            }
        }
        for (double &v : g) v = -v; // 最优条件 H d = −g
        bool ok = false;
        free_vals[axis] = solveSpd(H, g, ok);
        if (!ok) return false;
    }

    // ---- 由各轴端点状态还原段系数 ----
    segs_.resize(m);
    for (int i = 0; i < m; ++i) {
        Mat8 W{};
        if (!basisInverse(seg_T[i], W)) return false;
        for (int axis = 0; axis < 3; ++axis) {
            std::array<double, 8> e{};
            e[0] = waypoints[i].pos[axis];
            e[4] = waypoints[i + 1].pos[axis];
            for (int k = 0; k < 3; ++k) {
                if (i > 0) e[1 + k] = free_vals[axis][3 * i - 3 + k];
                if (i < m - 1) e[5 + k] = free_vals[axis][3 * i + k];
            }
            auto &c = segs_[i].coeff[axis];
            for (int r = 0; r < kNCoeff; ++r) {
                double s = 0.0;
                for (int q = 0; q < kState; ++q) s += W[r][q] * e[q];
                c[r] = s;
            }
        }
        segs_[i].duration = seg_T[i];
        segs_[i].start = 0.0;
    }

    // ---- 约束检验与全局时间缩放 ----
    // 等比缩放时间 s：各阶导数峰值 ×s^{-k}（严格成立，见文件头）。
    // 最紧的一档约束恰好顶到 1.0；乘 (1+1e-6) 是给峰值估计的数值误差留的安全缝。
    auto peak_of = [&](int deriv) {
        double worst = 0.0;
        for (const auto &seg : segs_) {
            for (int axis = 0; axis < 3; ++axis) {
                const auto &c = seg.coeff[axis];
                double lim_lo = 0.0, lim_hi = seg.duration;
                worst = std::max(worst,
                                 peakAbs([&c, deriv](double t) {
                                     double d[4];
                                     evalDerivs(c, t, d);
                                     return d[deriv];
                                 },
                                         lim_lo, lim_hi));
            }
        }
        return worst;
    };
    const double pv = peak_of(1), pa = peak_of(2), pj = peak_of(3);
    if (!std::isfinite(pv) || !std::isfinite(pa) || !std::isfinite(pj)) return false;
    const double rho_v = pv / limits.max_vel;
    const double rho_j = pj / limits.max_jerk;
    double s = std::max({rho_v, std::sqrt(std::max(pa, 0.0) / limits.max_acc),
                         std::cbrt(std::max(rho_j, 0.0))});
    if (!(s > 0.0) || !std::isfinite(s)) return false; // 全零运动已被重合检查排除，此处只挡数值事故
    s *= 1.0 + 1e-6;

    double cum = 0.0;
    for (auto &seg : segs_) {
        const double old_T = seg.duration;
        seg.duration = old_T * s;
        seg.start = cum;
        cum += seg.duration;
        // q(t) = p(t/s)：t^k 系数除以 s^k。
        double scale = 1.0;
        for (int k = 0; k < kNCoeff; ++k) {
            for (int axis = 0; axis < 3; ++axis) seg.coeff[axis][k] *= scale;
            scale /= s;
        }
    }

    start_time_ = start_time;
    yaw_ = fixed_yaw;
    duration_ = cum;
    built_ = true;

    // feasible() 独立重采样验证（与测试同一标准），通不过算构建失败：
    // 构造自洽性与采样可行性应当一致，不一致说明峰值估计出错，宁报错不放行。
    if (!feasible()) { reset(); return false; }
    return true;
}

bool MinimumSnapTrajectory::sample(double t, FlatReference &out) const {
    // 闭区间判定写成全取反形式：NaN 会落在「范围外」、返回 false 且不动 out。
    if (!built_ || !(t >= start_time_ && t <= start_time_ + duration_)) return false;
    const double tau = t - start_time_;

    // 段定位：航点时刻归属左段（tau > 段终点才右移），左右数值一致，
    // 取值无突变；最后一段允许 local == duration（闭区间右端）。
    int i = 0;
    while (i + 1 < static_cast<int>(segs_.size()) && tau > segs_[i].start + segs_[i].duration)
        ++i;
    const auto &seg = segs_[i];
    const double local = tau - seg.start;

    double d[4];
    for (int axis = 0; axis < 3; ++axis) {
        evalDerivs(seg.coeff[axis], local, d);
        out.pos[axis] = d[0];
        out.vel[axis] = d[1];
        out.acc[axis] = d[2];
        out.jerk[axis] = d[3];
    }
    out.yaw = yaw_;
    out.yaw_rate = 0.0;
    return true;
}

double MinimumSnapTrajectory::duration() const { return built_ ? duration_ : 0.0; }

bool MinimumSnapTrajectory::feasible() const {
    if (!built_) return false;
    // 真采样检查（契约要求），精度与构造时缩放所用的估计互相独立：
    // 缩放到位 → 各峰值恰在边界（1+1e-6) 以内；这里给 1e-9 的判据余量。
    constexpr int kPerSeg = 1000;
    const double tol = 1.0 + 1e-9;
    for (const auto &seg : segs_) {
        for (int k = 0; k <= kPerSeg; ++k) {
            const double t = seg.duration * (static_cast<double>(k) / kPerSeg);
            double d[4];
            for (int axis = 0; axis < 3; ++axis) {
                evalDerivs(seg.coeff[axis], t, d);
                if (std::fabs(d[1]) > limits_.max_vel * tol) return false;
                if (std::fabs(d[2]) > limits_.max_acc * tol) return false;
                if (std::fabs(d[3]) > limits_.max_jerk * tol) return false;
            }
        }
    }
    return true;
}

void MinimumSnapTrajectory::reset() {
    segs_.clear();
    limits_ = TrajectoryLimits{};
    start_time_ = 0.0;
    duration_ = 0.0;
    yaw_ = 0.0;
    built_ = false;
}

} // namespace oi3
