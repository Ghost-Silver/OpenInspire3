/**
 * @file ModelCalibrationTest.cpp
 * @brief 模型校准的收益：参数辨识如何改善轨迹预测
 * @author GhostFace
 * @date 2026/9/18
 *
 * @par 这个测试回答什么
 *
 * 参数辨识本身能把参数拟合出来（见 ParameterIdTest），但真正要紧的是**拟合出来的
 * 参数有没有用**。这里用同一段真值轨迹，比较三套参数的开环预测误差：
 *
 *   - 手填典型值：OI3 里原本的默认（质量 1.0、惯量 {0.01,0.01,0.02}、无阻力）；
 *   - 辨识值：ParameterIdTest 从轨迹里恢复出来的（带辨识残差）；
 *   - 真值：生成数据时使用的参数（这组在真机上不可得，只作下界参考）。
 *
 * 之所以用**开环预测误差**而不是闭环控制性能作为主要指标：预测误差是模型精度的
 * 直接度量，不受控制器结构的干扰；而 MPC 等基于模型的方法正是把预测当输入用的，
 * 预测准不准会原样传导到控制上。闭环对比成本高且受视野等参数影响，
 * 不适合作为模型精度的判据。
 *
 * @note 预测误差在真值轨迹噪声为零时，下界就是辨识残差带来的模型误差。
 *       本测试的数据为无噪合成数据，因此「辨识值」与「真值」的差距只反映
 *       参数残差，而不含观测噪声的影响。
 */

#include "ParameterIdentification.h"
#include "TensorUtils.h"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

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

SixDofState makeState(double vx, double vy, double vz, double wx, double wy, double wz) {
    Tensor p(ShapeTag{}, {3});
    Tensor v(ShapeTag{}, {3});
    Tensor q(ShapeTag{}, {4});
    Tensor w(ShapeTag{}, {3});
    p.data_write<float>()[0] = 0.0f;
    p.data_write<float>()[1] = 0.0f;
    p.data_write<float>()[2] = -5.0f;
    v.data_write<float>()[0] = static_cast<float>(vx);
    v.data_write<float>()[1] = static_cast<float>(vy);
    v.data_write<float>()[2] = static_cast<float>(vz);
    v.data_write<float>()[2] = static_cast<float>(vz);
    q.data_write<float>()[0] = 1.0f; // (w,x,y,z) 单位四元数
    w.data_write<float>()[0] = static_cast<float>(wx);
    w.data_write<float>()[1] = static_cast<float>(wy);
    w.data_write<float>()[2] = static_cast<float>(wz);
    return SixDofState{p, v, q, w};
}

/// 两条轨迹的均方根偏差（位置与速度合起来）
double trajectoryRms(const IdentTrajectory &a, const IdentTrajectory &b) {
    const std::size_t n = std::min(a.pos.size(), b.pos.size());
    if (n == 0) {
        return 0.0;
    }
    double acc = 0.0;
    for (std::size_t k = 0; k < n; ++k) {
        for (int i = 0; i < 3; ++i) {
            const double dp = a.pos[k][static_cast<std::size_t>(i)] -
                              b.pos[k][static_cast<std::size_t>(i)];
            const double dv = a.vel[k][static_cast<std::size_t>(i)] -
                              b.vel[k][static_cast<std::size_t>(i)];
            acc += dp * dp + dv * dv;
        }
    }
    return std::sqrt(acc / static_cast<double>(n * 6));
}

} // namespace

int main() {
    CtorchError::setPrintLevel(PrintLevel::MINIUM);

    Config base;
    base.dt = 0.001;
    base.mass = 1.0;
    base.gravity = 9.81;
    base.drag_coeff = 0.0;

    // 真值：模拟「真机」的参数
    ParamSet truth;
    truth.mass = 1.35;
    truth.inertia[0] = 0.015;
    truth.inertia[1] = 0.018;
    truth.inertia[2] = 0.028;
    truth.drag_coeff = 0.12;

    // 手填典型值：OI3 里的默认（质量按 1.0、惯量按小四轴典型值、阻力未建模）
    ParamSet handfilled;
    handfilled.mass = 1.0;
    handfilled.inertia[0] = 0.01;
    handfilled.inertia[1] = 0.01;
    handfilled.inertia[2] = 0.02;
    handfilled.drag_coeff = 0.0;

    // 辨识值：ParameterIdTest 实测输出（含辨识残差）
    ParamSet identified;
    identified.mass = 1.324213;
    identified.inertia[0] = 0.012410;
    identified.inertia[1] = 0.014678;
    identified.inertia[2] = 0.022391;
    identified.drag_coeff = 0.102422;

    // 一段有激励的飞行：非零初速（激励阻力）+ 三轴力矩（激励惯量）
    const int steps = 60;
    std::vector<double> thrust(steps, 0.0);
    std::vector<std::array<double, 3>> torque(steps, {0.0, 0.0, 0.0});
    for (int k = 0; k < steps; ++k) {
        const double t = static_cast<double>(k);
        thrust[k] = 9.81 * truth.mass * (1.0 + 0.5 * std::sin(0.25 * t));
        torque[k] = {0.080 * std::cos(0.40 * t), 0.070 * std::sin(0.35 * t),
                     0.060 * std::cos(0.30 * t)};
    }
    const SixDofState init = makeState(5.0, -4.0, 3.0, 1.5, -1.2, 1.0);

    std::cout << "========================================\n";
    std::cout << "模型校准收益：参数精度 → 预测精度\n";
    std::cout << "========================================\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "\n参数对比：\n";
    std::cout << "  真值    m=" << truth.mass << " I=(" << truth.inertia[0] << ","
              << truth.inertia[1] << "," << truth.inertia[2]
              << ") c=" << truth.drag_coeff << "\n";
    std::cout << "  手填    m=" << handfilled.mass << " I=(" << handfilled.inertia[0] << ","
              << handfilled.inertia[1] << "," << handfilled.inertia[2]
              << ") c=" << handfilled.drag_coeff << "\n";
    std::cout << "  辨识    m=" << identified.mass << " I=(" << identified.inertia[0] << ","
              << identified.inertia[1] << "," << identified.inertia[2]
              << ") c=" << identified.drag_coeff << "\n";

    // 真值轨迹（模拟真机观测）
    const IdentTrajectory ref = simulateTrajectory(init, thrust, torque, truth, base);

    // 三套参数的预测
    const IdentTrajectory pred_hand = simulateTrajectory(init, thrust, torque, handfilled, base);
    const IdentTrajectory pred_ident = simulateTrajectory(init, thrust, torque, identified, base);
    const IdentTrajectory pred_truth = simulateTrajectory(init, thrust, torque, truth, base);

    const double rms_hand = trajectoryRms(ref, pred_hand);
    const double rms_ident = trajectoryRms(ref, pred_ident);
    const double rms_truth = trajectoryRms(ref, pred_truth);

    std::cout << "\n开环预测 RMS 偏差（位置+速度，60 ms / 60 步）：\n";
    std::cout << "  手填典型值    " << rms_hand << "\n";
    std::cout << "  辨识值        " << rms_ident << "\n";
    std::cout << "  真值          " << rms_truth << "（下界，真机不可得）\n";
    if (rms_ident > 0.0) {
        std::cout << "\n  校准使预测误差降低 " << (rms_hand / rms_ident) << " 倍\n";
    }

    checkTrue("三套参数均产生有限轨迹",
              std::isfinite(rms_hand) && std::isfinite(rms_ident) && std::isfinite(rms_truth));
    checkTrue("真值自比为零（验证口径）", rms_truth < 1e-9);
    // 判据用「相对手填值的改善倍数」来表达，而不是「与真值下界的距离」：
    // 真值自比的偏差恒为 0，而辨识残差必然大于 0，任何形如「靠近下界若干倍」的
    // 断言都不可能成立 —— 那是断言写错了，不是结果不好。
    checkTrue("辨识值显著优于手填典型值（至少 3 倍）", rms_hand > rms_ident * 3.0);
    checkTrue("校准使预测误差至少降低 10 倍", rms_hand > rms_ident * 10.0);
    checkTrue("辨识残差本身处于小量级（< 0.01）", rms_ident < 0.01);

    std::cout << "\n注：这里量的是**模型精度**。基于模型的方法（MPC、EKF、"
              "轨迹规划）都把预测当输入，\n";
    std::cout << "    预测误差会原样传导到它们的结果上 —— 这正是参数辨识的价值所在。\n";

    std::cout << "\n========================================\n";
    std::cout << g_checks - g_failed << " / " << g_checks << " checks passed\n";
    if (g_failed > 0) {
        std::cout << g_failed << " FAILED\n";
    }
    std::cout << "========================================\n";
    return g_failed == 0 ? 0 : 1;
}
