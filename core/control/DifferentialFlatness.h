/**
 * @file DifferentialFlatness.h
 * @brief 微分平坦前馈：从轨迹解析计算推力、姿态与角速度
 * @author GhostFace
 * @date 2026/9/18
 *
 * @par 为什么这是机动性的天花板
 *
 * 四旋翼有一个漂亮的结构性质：它是**微分平坦**的。取四个平坦输出
 * （位置 x、y、z 与偏航 ψ），则全状态 —— 姿态、角速度、推力 —— 都可以由
 * 这四者及其有限阶导数**解析地**算出，不需要任何反馈去「追」。
 *
 * 现有控制器用的是级联 PID：位置误差 → 期望加速度 → 期望倾角 → 姿态误差
 * → 力矩。这条链路上每一环都靠**误差**驱动，所以：
 *
 *  - 姿态指令只用到加速度，**没用 jerk** —— 而姿态变化率由 jerk 决定；
 *  - 角速度靠姿态环的 D 项「追」出来，而不是算出来；
 *  - 机动越快，误差越大，直到撞上力矩限幅。
 *
 * 微分平坦前馈把这些量**直接算出来**：给定一条光滑轨迹，期望推力方向、推力
 * 大小、以及三个轴的期望角速度全部解析可得。反馈只负责修正模型误差与扰动，
 * 不再负责「产生」机动 —— 这是响应速度与机动性的根本来源。
 *
 * @par 平坦映射
 *
 * 设期望加速度 `a`（NED），则推力须提供 `a − g·e_z` 方向的支持力：
 *
 * @verbatim
 *   z_b = (a − g·e_z) / ‖a − g·e_z‖        机体 z 轴（推力反方向）
 *   F   = m·‖a − g·e_z‖                    推力大小
 *
 *   由偏航 ψ 定义中间轴  x_c = [cosψ, sinψ, 0]
 *   y_b = z_b × x_c / ‖z_b × x_c‖
 *   x_b = y_b × z_b
 * @endverbatim
 *
 * 角速度由加速度的导数（jerk）决定。对 `ṡ = j − (j·z_b)z_b`（jerk 在
 * 垂直于推力方向上的分量）：
 *
 * @verbatim
 *   p = −ṡ·y_b / F·m        roll  角速度
 *   q =  ṡ·x_b / F·m        pitch 角速度
 *   r =  ψ̇·(x_c·x_b)         yaw   角速度
 * @endverbatim
 *
 * @note 本文件只做前馈计算，不含反馈。它与级联 PID 组合成「前馈 + 反馈」的
 *       混合结构：前馈负责产生机动，反馈负责修正残差。
 */

#ifndef OI3_DIFFERENTIAL_FLATNESS_H
#define OI3_DIFFERENTIAL_FLATNESS_H

#include <algorithm>
#include <array>
#include <cmath>

namespace oi3 {

/// 三维向量的小工具（纯数值，不涉及张量）
struct FlatVec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;

    [[nodiscard]] FlatVec3 operator+(const FlatVec3 &o) const {
        return {x + o.x, y + o.y, z + o.z};
    }
    [[nodiscard]] FlatVec3 operator-(const FlatVec3 &o) const {
        return {x - o.x, y - o.y, z - o.z};
    }
    [[nodiscard]] FlatVec3 operator*(double s) const { return {x * s, y * s, z * s}; }
    [[nodiscard]] double dot(const FlatVec3 &o) const { return x * o.x + y * o.y + z * o.z; }
    [[nodiscard]] FlatVec3 cross(const FlatVec3 &o) const {
        return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
    }
    [[nodiscard]] double norm() const { return std::sqrt(x * x + y * y + z * z); }
    [[nodiscard]] FlatVec3 normalized() const {
        const double n = norm();
        return (n > 1e-12) ? FlatVec3{x / n, y / n, z / n} : FlatVec3{0.0, 0.0, 0.0};
    }
};

/**
 * @brief 平坦输出的参考轨迹（位置、偏航及其导数）
 *
 * jerk 是必需的：姿态变化率与角速度由它决定。只给加速度的话，前馈只能算出
 * 静态的推力方向，算不出「机身要以多快的角速度转过去」。
 */
struct FlatReference {
    std::array<double, 3> pos{};  ///< 位置（NED，米）
    std::array<double, 3> vel{};  ///< 速度（米/秒）
    std::array<double, 3> acc{};  ///< 加速度（米/秒²）
    std::array<double, 3> jerk{}; ///< 加加速度（米/秒³）
    double yaw = 0.0;             ///< 偏航角（弧度）
    double yaw_rate = 0.0;        ///< 偏航角速度（弧度/秒）
};

/// 平坦前馈的输出
struct FlatOutput {
    double thrust = 0.0;             ///< 期望推力大小（牛顿，沿机体 −z）
    std::array<double, 4> quat{};    ///< 期望姿态四元数 (w,x,y,z)，机体系 → NED
    std::array<double, 3> omega{};   ///< 期望机体角速度（rad/s）
    bool valid = false;              ///< 输入退化（如自由落体）时为 false
};

namespace flat_detail {

/// 由旋转矩阵（列向量 x_b, y_b, z_b）构造四元数
inline std::array<double, 4> quatFromBasis(const FlatVec3 &xb, const FlatVec3 &yb,
                                           const FlatVec3 &zb) {
    // 标准 Shepperd 方法：取迹最大的分支，避免除以接近零的数
    const double m00 = xb.x, m01 = yb.x, m02 = zb.x;
    const double m10 = xb.y, m11 = yb.y, m12 = zb.y;
    const double m20 = xb.z, m21 = yb.z, m22 = zb.z;
    const double tr = m00 + m11 + m22;

    std::array<double, 4> q{};
    if (tr > 0.0) {
        const double s = std::sqrt(tr + 1.0) * 2.0;
        q[0] = 0.25 * s;
        q[1] = (m21 - m12) / s;
        q[2] = (m02 - m20) / s;
        q[3] = (m10 - m01) / s;
    } else if (m00 > m11 && m00 > m22) {
        const double s = std::sqrt(1.0 + m00 - m11 - m22) * 2.0;
        q[0] = (m21 - m12) / s;
        q[1] = 0.25 * s;
        q[2] = (m01 + m10) / s;
        q[3] = (m02 + m20) / s;
    } else if (m11 > m22) {
        const double s = std::sqrt(1.0 + m11 - m00 - m22) * 2.0;
        q[0] = (m02 - m20) / s;
        q[1] = (m01 + m10) / s;
        q[2] = 0.25 * s;
        q[3] = (m12 + m21) / s;
    } else {
        const double s = std::sqrt(1.0 + m22 - m00 - m11) * 2.0;
        q[0] = (m10 - m01) / s;
        q[1] = (m02 + m20) / s;
        q[2] = (m12 + m21) / s;
        q[3] = 0.25 * s;
    }
    const double n = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    if (n > 1e-12) {
        for (double &v : q) {
            v /= n;
        }
    }
    // 规范化符号：保证 w ≥ 0，避免同一姿态出现两种表示
    if (q[0] < 0.0) {
        for (double &v : q) {
            v = -v;
        }
    }
    return q;
}

} // namespace flat_detail

/**
 * @brief 由平坦输出计算前馈推力、姿态与角速度
 *
 * @param ref  参考轨迹（含 jerk 与偏航导数）
 * @param mass 质量（kg）
 * @param g    重力加速度（m/s²）
 */
[[nodiscard]] inline FlatOutput computeFlatFeedforward(const FlatReference &ref, double mass,
                                                       double g = 9.81) {
    FlatOutput out;

    // ---- 推力方向 ----
    //
    // 运动方程：m·a = m·g_ned + F·(−z_b)，其中 g_ned = [0,0,+g]（NED 的 z 朝下），
    // 推力沿机体 −z_b。整理得
    //
    //     z_b = (g_ned − a) / ‖g_ned − a‖,      F = m·‖g_ned − a‖
    //
    // 悬停自检：a = 0 ⇒ z_b = [0,0,+1]，即机体 z 轴与 NED 的 z 轴同向 —— 这与
    // 项目其余部分的约定一致（推力 [0,0,−T] 抵消重力 [0,0,+mg]）。
    //
    // 第一版写成 (a − g_ned)，符号反了：悬停时 z_b 变成 [0,0,−1]，倾角算出
    // 168.477°（= 180° − 11.523°，正好是补角）。
    const FlatVec3 acc{ref.acc[0], ref.acc[1], ref.acc[2]};
    const FlatVec3 gvec{0.0, 0.0, g};
    const FlatVec3 fdir = gvec - acc;
    const double fmag = fdir.norm(); // = F/m，即所需加速度模长
    if (fmag < 1e-6) {
        return out; // 自由落体：推力方向未定义
    }
    const FlatVec3 zb = fdir.normalized();
    out.thrust = mass * fmag;

    // ---- 机体三轴 ----
    //
    // 由偏航角定义中间轴 x_c，再正交化得到 y_b、x_b。
    const FlatVec3 xc{std::cos(ref.yaw), std::sin(ref.yaw), 0.0};
    const FlatVec3 xc_dot =
        FlatVec3{-std::sin(ref.yaw), std::cos(ref.yaw), 0.0} * ref.yaw_rate;

    const FlatVec3 n = zb.cross(xc);
    const double nn = n.norm();
    if (nn < 1e-6) {
        return out; // 推力方向与偏航轴共线，姿态未定义
    }
    const FlatVec3 yb = n * (1.0 / nn);
    const FlatVec3 xb = yb.cross(zb);

    out.quat = flat_detail::quatFromBasis(xb, yb, zb);

    // ---- 角速度 ----
    //
    // 关键：`ż_b = ω × z_b` 只决定 ω **垂直于** z_b 的两个分量，ω 沿 z_b 的
    // 分量（偏航角速度）**无法**由它确定。圆周飞行时推力方向绕世界竖直轴
    // 进动，这个进动在机体系里恰有一个 z 分量 —— 凭直觉写「ω_z = 偏航率」
    // 会完全漏掉它（实测漏掉 0.0407 rad/s）。
    //
    // 正确做法是用旋转矩阵的导数：[ω]× = Ṙ·Rᵀ，其分量为
    //     ω_x = ẏ_b·z_b,   ω_y = ż_b·x_b,   ω_z = ẋ_b·y_b
    // 因此需要 x_b、y_b 的导数，而它们可以解析求出（x_c 的导数已知）。
    const FlatVec3 jerk{ref.jerk[0], ref.jerk[1], ref.jerk[2]};

    // ż_b：z_b = fdir/‖fdir‖，而 ḟdir = −jerk（gvec 为常量）
    //   ż_b = (−jerk + (jerk·z_b)z_b) / ‖fdir‖
    const FlatVec3 sdot = jerk - zb * (jerk.dot(zb));
    const FlatVec3 zb_dot = sdot * (-1.0 / fmag);

    // ṅ = ż_b × x_c + z_b × ẋ_c
    const FlatVec3 n_dot = zb_dot.cross(xc) + zb.cross(xc_dot);

    // ẏ_b = ṅ/‖n‖ − n·(ṅ·n)/‖n‖³
    const FlatVec3 yb_dot = n_dot * (1.0 / nn) - n * ((n_dot.dot(n)) / (nn * nn * nn));

    // ẋ_b = ẏ_b × z_b + y_b × ż_b
    const FlatVec3 xb_dot = yb_dot.cross(zb) + yb.cross(zb_dot);

    out.omega[0] = yb_dot.dot(zb);
    out.omega[1] = zb_dot.dot(xb);
    out.omega[2] = xb_dot.dot(yb);

    out.valid = true;
    return out;
}

} // namespace oi3

#endif // OI3_DIFFERENTIAL_FLATNESS_H
