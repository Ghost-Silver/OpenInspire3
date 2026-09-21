/**
 * @file FlatnessTest.cpp
 * @brief 微分平坦前馈的验证：推力、姿态与角速度的解析映射
 * @author GhostFace
 * @date 2026/9/18
 *
 * @par 验证策略
 *
 * 四旋翼的坐标系约定叠在一起时符号极易出错：NED 的 z 轴**朝下**、机体 z 轴
 * 与之同向、推力沿机体 **−z**、偏航由水平中间轴定义。任何一处搞反都会得到
 * 「看起来合理但完全错误」的结果。
 *
 * 所以本测试不靠肉眼检查，而是用两类**独立**的判据：
 *
 * 1. **解析预期**：悬停、常值加速度、匀速圆周这三种情形都有闭式解，
 *    可以直接对照（倾角、推力大小、角速度）。
 *
 * 2. **四元数数值微分**：姿态角速度 ω 满足 `q̇ = ½·q ⊗ ω`。在 t±h 处求出
 *    解析四元数后用中心差分得到 ω，与平坦映射**独立**算出的 ω 对照。
 *    这一条能抓住任何符号错误 —— 因为它不依赖推导，只用四元数运动学。
 *
 * @par 三种典型情形的闭式解
 *
 * - **悬停**：`a = 0` ⇒ 推力 `F = m·g`，姿态单位四元数，`ω = 0`。
 * - **常值水平加速度** `a = (a_x, 0, 0)`：倾角 `θ = atan(a_x/g)`，
 *   `F = m·√(a_x² + g²)`，`ω = 0`（姿态不变）。
 * - **匀速圆周**（半径 R、角速度 Ω）：向心加速度 `Ω²R` 指向圆心，
 *   倾角 `θ = atan(Ω²R/g)`。若偏航固定，则机体 z 轴绕世界竖直轴以 Ω 进动，
 *   而机体系角速度**不等于** Ω —— 这一点正是最容易被直觉搞错的地方，
 *   必须由数值微分来裁决。
 */

#include "DifferentialFlatness.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>

using namespace oi3;

namespace {

int g_checks = 0;
int g_failed = 0;

void checkTrue(const char *name, bool cond) {
    ++g_checks;
    if (!cond) {
        ++g_failed;
    }
    std::cout << "  [" << (cond ? " ok " : "FAIL") << "] " << name << "\n";
}

void checkNear(const char *name, double got, double want, double tol) {
    ++g_checks;
    const bool ok = std::isfinite(got) && std::fabs(got - want) <= tol;
    if (!ok) {
        ++g_failed;
    }
    std::cout << "  [" << (ok ? " ok " : "FAIL") << "] " << name << "（实测 " << std::setprecision(6)
              << got << "，解析 " << want << "，差 " << std::fabs(got - want) << "）\n";
}

/// 四元数乘法（Hamilton，与项目其余部分一致）
std::array<double, 4> quatMul(const std::array<double, 4> &a, const std::array<double, 4> &b) {
    return {a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
            a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
            a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
            a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]};
}

std::array<double, 4> quatConj(const std::array<double, 4> &q) {
    return {q[0], -q[1], -q[2], -q[3]};
}

/// 由四元数取机体 z 轴在 NED 中的方向（旋转矩阵第三列）
FlatVec3 bodyZFromQuat(const std::array<double, 4> &q) {
    // R 的第三列为 [2(xz+wy), 2(yz-wx), 1-2(x²+y²)]
    return {2.0 * (q[1] * q[3] + q[0] * q[2]), 2.0 * (q[2] * q[3] - q[0] * q[1]),
            1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2])};
}

/// 圆周轨迹的解析参考点
FlatReference circularRef(double t, double radius, double omega, double yaw) {
    FlatReference r;
    const double a = omega * t;
    r.pos = {radius * std::cos(a), radius * std::sin(a), 0.0};
    r.vel = {-radius * omega * std::sin(a), radius * omega * std::cos(a), 0.0};
    r.acc = {-radius * omega * omega * std::cos(a), -radius * omega * omega * std::sin(a), 0.0};
    r.jerk = {radius * omega * omega * omega * std::sin(a),
              -radius * omega * omega * omega * std::cos(a), 0.0};
    r.yaw = yaw;
    r.yaw_rate = 0.0;
    return r;
}

} // namespace

int main() {
    const double mass = 1.0;
    const double g = 9.81;

    std::cout << "========================================\n";
    std::cout << "微分平坦前馈验证\n";
    std::cout << "========================================\n";

    // ---- 1. 悬停 ----
    std::cout << "\n[1] 悬停（a = 0）\n";
    std::cout << "  预期：F = mg，姿态单位四元数，ω = 0。\n";
    std::cout << "  注意 NED 的 z 轴朝下：机体 z 轴在水平时应为 [0,0,+1]。\n\n";
    {
        FlatReference ref;
        const FlatOutput out = computeFlatFeedforward(ref, mass, g);
        const FlatVec3 zb = bodyZFromQuat(out.quat);

        std::cout << "  推力 " << std::setprecision(6) << out.thrust << " N（预期 "
                  << mass * g << "）\n";
        std::cout << "  机体 z 轴 = [" << zb.x << ", " << zb.y << ", " << zb.z << "]\n";
        std::cout << "  ω = [" << out.omega[0] << ", " << out.omega[1] << ", " << out.omega[2]
                  << "]\n";

        checkTrue("悬停时输出有效", out.valid);
        checkNear("推力 = mg", out.thrust, mass * g, 1e-9);
        checkNear("机体 z 轴指向 NED +z（水平姿态）", zb.z, 1.0, 1e-9);
        checkNear("机体 z 轴水平分量为零", std::sqrt(zb.x * zb.x + zb.y * zb.y), 0.0, 1e-9);
        checkNear("悬停角速度为零（roll）", out.omega[0], 0.0, 1e-9);
        checkNear("悬停角速度为零（pitch）", out.omega[1], 0.0, 1e-9);
        checkNear("悬停角速度为零（yaw）", out.omega[2], 0.0, 1e-9);
    }

    // ---- 2. 常值水平加速度 ----
    std::cout << "\n[2] 常值水平加速度 a = (2, 0, 0)\n";
    std::cout << "  预期：倾角 θ = atan(a/g)，F = m·√(a²+g²)，姿态恒定故 ω = 0。\n\n";
    {
        FlatReference ref;
        ref.acc = {2.0, 0.0, 0.0};
        const FlatOutput out = computeFlatFeedforward(ref, mass, g);
        const FlatVec3 zb = bodyZFromQuat(out.quat);

        const double theta_pred = std::atan2(2.0, g) * 180.0 / M_PI;
        const double theta_got = std::acos(std::max(-1.0, std::min(1.0, zb.z))) * 180.0 / M_PI;
        const double F_pred = mass * std::sqrt(2.0 * 2.0 + g * g);

        std::cout << "  倾角 " << std::setprecision(6) << theta_got << " deg（预期 " << theta_pred
                  << "）\n";
        std::cout << "  推力 " << out.thrust << " N（预期 " << F_pred << "）\n";
        std::cout << "  机体 z 轴 = [" << zb.x << ", " << zb.y << ", " << zb.z << "]\n";

        checkNear("倾角 = atan(a/g)", theta_got, theta_pred, 1e-6);
        checkNear("推力 = m√(a²+g²)", out.thrust, F_pred, 1e-9);
        // 加速方向为 +x（NED 北），机体 z 轴应朝 −x 倾斜（推力反方向指向 +x）
        checkTrue("机体 z 轴朝 −x 倾斜（推力指向 +x）", zb.x < 0.0);
        checkNear("姿态恒定时角速度为零（roll）", out.omega[0], 0.0, 1e-9);
        checkNear("姿态恒定时角速度为零（pitch）", out.omega[1], 0.0, 1e-9);
    }

    // ---- 3. 匀速圆周：倾角与角速度 ----
    std::cout << "\n[3] 匀速圆周（半径 2 m，角速度 1 rad/s，偏航固定）\n";
    std::cout << "  预期：倾角 = atan(Ω²R/g) 指向圆心。\n";
    std::cout << "  角速度**不等于** Ω —— 这是最容易凭直觉搞错的地方，用四元数\n";
    std::cout << "  数值微分独立裁决。\n\n";
    {
        const double R = 2.0, Om = 1.0;
        const double h = 1e-6;
        const double t = 0.0;

        const FlatReference ref = circularRef(t, R, Om, 0.0);
        const FlatOutput out = computeFlatFeedforward(ref, mass, g);

        const double theta_pred = std::atan2(Om * Om * R, g) * 180.0 / M_PI;
        const FlatVec3 zb = bodyZFromQuat(out.quat);
        const double theta_got = std::acos(std::max(-1.0, std::min(1.0, zb.z))) * 180.0 / M_PI;

        std::cout << "  倾角 " << std::setprecision(6) << theta_got << " deg（预期 "
                  << theta_pred << "）\n";
        std::cout << "  机体 z 轴 = [" << zb.x << ", " << zb.y << ", " << zb.z << "]\n";
        std::cout << "  映射给出的 ω = [" << out.omega[0] << ", " << out.omega[1] << ", "
                  << out.omega[2] << "]\n";

        // 独立求 ω：在 t±h 处求解析四元数，中心差分
        const FlatOutput op = computeFlatFeedforward(circularRef(t + h, R, Om, 0.0), mass, g);
        const FlatOutput om = computeFlatFeedforward(circularRef(t - h, R, Om, 0.0), mass, g);
        std::array<double, 4> qp = op.quat;
        std::array<double, 4> qm = om.quat;
        // 消除四元数符号歧义（q 与 −q 表示同一姿态）
        double dot = 0.0;
        for (int i = 0; i < 4; ++i) {
            dot += qp[static_cast<std::size_t>(i)] * qm[static_cast<std::size_t>(i)];
        }
        if (dot < 0.0) {
            for (double &v : qm) {
                v = -v;
            }
        }
        std::array<double, 4> dq{};
        for (int i = 0; i < 4; ++i) {
            dq[static_cast<std::size_t>(i)] =
                (qp[static_cast<std::size_t>(i)] - qm[static_cast<std::size_t>(i)]) / (2.0 * h);
        }
        const std::array<double, 4> qc = quatConj(out.quat);
        const std::array<double, 4> qd = quatMul(qc, dq);
        const std::array<double, 3> omega_fd = {2.0 * qd[1], 2.0 * qd[2], 2.0 * qd[3]};

        std::cout << "  数值微分 ω = [" << omega_fd[0] << ", " << omega_fd[1] << ", "
                  << omega_fd[2] << "]\n";

        checkNear("圆周倾角 = atan(Ω²R/g)", theta_got, theta_pred, 1e-6);
        // 向心加速度指向圆心（t=0 时在 +x 处，故指向 −x）；推力沿机体 −z_b，
        // 要产生 −x 方向的加速度就必须让 z_b 有 **+x** 分量，即机身朝 +x 倾。
        checkTrue("机体 z 轴朝外（+x）倾斜，使推力指向圆心", zb.x > 0.0);
        checkNear("ω_x 与数值微分一致", out.omega[0], omega_fd[0], 1e-4);
        checkNear("ω_y 与数值微分一致", out.omega[1], omega_fd[1], 1e-4);
        checkNear("ω_z 与数值微分一致", out.omega[2], omega_fd[2], 1e-4);
    }

    // ---- 4. 带偏航速率的圆周 ----
    std::cout << "\n[4] 带偏航速率的圆周（偏航 0.5 rad/s）\n";
    std::cout << "  偏航在转时 ω_z 应非零，且必须与数值微分一致。\n\n";
    {
        const double R = 1.5, Om = 0.8, yaw_rate = 0.5, h = 1e-6, t = 0.0;
        auto refWithYaw = [&](double tt) {
            FlatReference r = circularRef(tt, R, Om, yaw_rate * tt);
            r.yaw_rate = yaw_rate;
            return r;
        };
        const FlatOutput out = computeFlatFeedforward(refWithYaw(t), mass, g);
        const FlatOutput op = computeFlatFeedforward(refWithYaw(t + h), mass, g);
        const FlatOutput om = computeFlatFeedforward(refWithYaw(t - h), mass, g);

        std::array<double, 4> qp = op.quat, qm = om.quat;
        double dot = 0.0;
        for (int i = 0; i < 4; ++i) {
            dot += qp[static_cast<std::size_t>(i)] * qm[static_cast<std::size_t>(i)];
        }
        if (dot < 0.0) {
            for (double &v : qm) {
                v = -v;
            }
        }
        std::array<double, 4> dq{};
        for (int i = 0; i < 4; ++i) {
            dq[static_cast<std::size_t>(i)] =
                (qp[static_cast<std::size_t>(i)] - qm[static_cast<std::size_t>(i)]) / (2.0 * h);
        }
        const std::array<double, 3> omega_fd = [&] {
            const std::array<double, 4> qd = quatMul(quatConj(out.quat), dq);
            return std::array<double, 3>{2.0 * qd[1], 2.0 * qd[2], 2.0 * qd[3]};
        }();

        std::cout << "  映射给出的 ω = [" << std::setprecision(6) << out.omega[0] << ", "
                  << out.omega[1] << ", " << out.omega[2] << "]\n";
        std::cout << "  数值微分 ω = [" << omega_fd[0] << ", " << omega_fd[1] << ", "
                  << omega_fd[2] << "]\n";

        checkTrue("偏航转动时 ω_z 非零", std::fabs(out.omega[2]) > 1e-3);
        checkNear("ω_x 与数值微分一致", out.omega[0], omega_fd[0], 1e-4);
        checkNear("ω_y 与数值微分一致", out.omega[1], omega_fd[1], 1e-4);
        checkNear("ω_z 与数值微分一致", out.omega[2], omega_fd[2], 1e-4);
    }

    // ---- 5. 退化情形 ----
    std::cout << "\n[5] 退化情形：自由落体（a = g·e_z）\n";
    std::cout << "  推力方向未定义，应返回 invalid 而非 NaN。\n\n";
    {
        FlatReference ref;
        ref.acc = {0.0, 0.0, g};
        const FlatOutput out = computeFlatFeedforward(ref, mass, g);
        std::cout << "  valid = " << (out.valid ? "true" : "false") << "\n";
        checkTrue("自由落体时标记为无效（不产生 NaN）", !out.valid);
    }

    std::cout << "\n[结论]\n";
    std::cout << "  1. 悬停与常值加速度的闭式解全部吻合，说明推力大小与方向的映射正确。\n";
    std::cout << "  2. 圆周与带偏航情形下，映射给出的角速度与**四元数数值微分**独立算出\n";
    std::cout << "     的结果一致 —— 这一条不依赖推导，能抓住任何符号错误。\n";
    std::cout << "  3. 由此确认：机体角速度不能凭「绕竖直轴转 Ω 所以 ω_z = Ω」的直觉，\n";
    std::cout << "     必须由 jerk 与偏航率解析计算。\n";

    std::cout << "\n========================================\n";
    std::cout << g_checks - g_failed << " / " << g_checks << " checks passed\n";
    if (g_failed > 0) {
        std::cout << g_failed << " FAILED\n";
    }
    std::cout << "========================================\n";
    return g_failed == 0 ? 0 : 1;
}
