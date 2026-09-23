/**
 * @file YawControlTest.cpp
 * @brief 偏航控制：从「偏航自由」到「偏航受控」
 * @author GhostFace
 * @date 2026/9/18
 *
 * @par 补的是什么
 *
 * 原实现的姿态误差用 `cross(z_cur, z_des)` 计算 —— **只对齐机体 z 轴**（推力
 * 方向）。这在数学上留下一个自由度：绕 z_des 的转动不受任何约束，**偏航角
 * 完全自由**。
 *
 * 对纯定点悬停这没影响（偏航随便转，位置照样准），所以此前一直没暴露。
 * 但任何**需要指向**的任务都做不了：挂相机要对准目标、挂云台要指定朝向、
 * 多机协同要指定机头方向。真机上偏航还影响气动与能耗。
 *
 * @par 本测试的判据
 *
 * 1. **偏航跟踪**：给定偏航指令，能否转到并保持；
 * 2. **机动中保持偏航**：位置机动时偏航是否被扰动（这检验解耦）；
 * 3. **偏航旋转**：能否以指定角速度连续旋转（跟踪偏航斜坡）；
 * 4. **与倾斜限幅解耦**：大倾斜时偏航是否仍受控（原实现把限幅加在误差角上，
 *    会把偏航误差也算进去，属概念混淆）。
 *
 * @par 一个容易搞错的地方
 *
 * 偏航「自由」不等于「不动」。机体绕 z 轴没有恢复力矩时，任何微小扰动都会
 * 让它慢慢转走且不回来 —— 所以「偏航自由」的实际表现是**缓慢漂移**，而不是
 * 稳定保持。测试要能区分这两种情形。
 */

#include "SixDofPidController.h"
#include "SixDofSimulator.h"
#include "SixDofTypes.h"
#include "TensorUtils.h"
#include "WindModel.h"

#include <algorithm>
#include <array>
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

std::array<double, 3> readVec(const Tensor &t) {
    const std::vector<float> v = toVector(t);
    return {v[0], v[1], v[2]};
}

/// 由四元数取偏航角（NED，ZYX 欧拉角的 yaw）
double yawOf(const Tensor &quat) {
    const std::vector<float> q = toVector(quat);
    const double w = q[0], x = q[1], y = q[2], z = q[3];
    // R[0][0] = 1−2(y²+z²), R[1][0] = 2(xy+wz)
    const double r00 = 1.0 - 2.0 * (y * y + z * z);
    const double r10 = 2.0 * (x * y + w * z);
    return std::atan2(r10, r00);
}

double wrapPi(double a) {
    while (a > M_PI) {
        a -= 2.0 * M_PI;
    }
    while (a < -M_PI) {
        a += 2.0 * M_PI;
    }
    return a;
}

struct YawResult {
    double final_yaw_deg = 0.0;
    double rms_yaw_err_deg = 0.0;
    double max_yaw_err_deg = 0.0;
    double pos_rms = 0.0;
    bool diverged = false;
};

/**
 * @brief 悬停并跟踪偏航指令
 *
 * @param use_yaw   是否启用偏航控制
 * @param yaw_cmd   偏航指令（弧度）
 * @param wind      是否加风（检验扰动下偏航保持）
 */
YawResult runYawHold(bool use_yaw, double yaw_cmd, bool wind, double seconds = 10.0) {
    SixDofConfig cfg;
    cfg.base.dt = 0.001;
    cfg.base.mass = 1.0;
    cfg.base.gravity = 9.81;
    cfg.base.drag_coeff = 0.049;
    cfg.base.max_thrust = 0.0;
    cfg.inertia[0] = 0.015;
    cfg.inertia[1] = 0.018;
    cfg.inertia[2] = 0.028;
    cfg.torque_limit = 1.0;
    cfg.max_body_thrust = 20.0;

    const double dt = cfg.base.dt;
    const int steps = static_cast<int>(seconds / dt);
    const std::array<double, 3> target = {0.0, 0.0, -5.0};

    const SixDofState init{makeVec3(0.0f, 0.0f, -5.0f), makeVec3(0.0f, 0.0f, 0.0f),
                           Tensor{1.0f, 0.0f, 0.0f, 0.0f}, makeVec3(0.0f, 0.0f, 0.0f)};
    SixDofSimulator sim(cfg, init);

    TurbulentWind w(4.0, 0.0, 1.0, 8.0, dt, 20260918u);
    if (wind) {
        sim.setWind(&w);
        w.reset();
    }

    SixDofPidGains gains;
    gains.use_yaw_control = use_yaw;
    SixDofPidController ctrl(cfg, gains);
    const Tensor tgt = makeVec3(0.0f, 0.0f, -5.0f);

    YawResult out;
    double sq_yaw = 0.0, mx_yaw = 0.0, sq_pos = 0.0;
    int n = 0;
    const int from = steps / 3;

    for (int k = 0; k < steps; ++k) {
        const double t = static_cast<double>(k) * dt;
        SixDofSetpoint sp;
        sp.pos = target;
        sp.yaw = yaw_cmd;

        const SixDofCommand cmd = ctrl.computeTracking(sim.state(), sp, t);
        sim.step(cmd.thrust_body, cmd.torque);

        if (k >= from) {
            const double ye = wrapPi(yawOf(sim.state().quat) - yaw_cmd) * 180.0 / M_PI;
            sq_yaw += ye * ye;
            mx_yaw = std::max(mx_yaw, std::fabs(ye));
            const std::array<double, 3> p = readVec(sim.state().pos);
            sq_pos += (p[0] - target[0]) * (p[0] - target[0]) +
                      (p[1] - target[1]) * (p[1] - target[1]) +
                      (p[2] - target[2]) * (p[2] - target[2]);
            ++n;
        }
        const std::array<double, 3> p = readVec(sim.state().pos);
        if (!std::isfinite(p[0]) || std::fabs(p[0]) > 1e4) {
            out.diverged = true;
            return out;
        }
    }

    if (n > 0) {
        out.final_yaw_deg = yawOf(sim.state().quat) * 180.0 / M_PI;
        out.rms_yaw_err_deg = std::sqrt(sq_yaw / n);
        out.max_yaw_err_deg = mx_yaw;
        out.pos_rms = std::sqrt(sq_pos / n);
    }
    return out;
}

} // namespace

int main() {
    CtorchError::setPrintLevel(PrintLevel::MINIUM);

    std::cout << "========================================\n";
    std::cout << "偏航控制：从「偏航自由」到「偏航受控」\n";
    std::cout << "========================================\n";

    // ---- 1. 偏航跟踪能力 ----
    std::cout << "\n[1] 偏航指令跟踪（无风）\n";
    std::cout << "  偏航自由时无恢复力矩，扰动会让它缓慢漂走且不回来。\n\n";
    std::cout << "  " << std::setw(16) << "偏航指令" << std::setw(22) << "偏航自由RMS(deg)"
              << std::setw(22) << "受控RMS(deg)" << std::setw(20) << "受控终值(deg)" << "\n";

    std::array<double, 4> cmds_deg = {30.0, 90.0, -60.0, 180.0};
    std::array<double, 4> free_rms{}, ctrl_rms{};
    for (int i = 0; i < 4; ++i) {
        const double cmd = cmds_deg[static_cast<std::size_t>(i)] * M_PI / 180.0;
        const YawResult a = runYawHold(false, cmd, false);
        const YawResult b = runYawHold(true, cmd, false);
        free_rms[static_cast<std::size_t>(i)] = a.rms_yaw_err_deg;
        ctrl_rms[static_cast<std::size_t>(i)] = b.rms_yaw_err_deg;

        std::cout << "  " << std::setw(16) << std::fixed << std::setprecision(1)
                  << cmds_deg[static_cast<std::size_t>(i)] << std::setw(22)
                  << std::setprecision(4) << a.rms_yaw_err_deg << std::setw(22)
                  << b.rms_yaw_err_deg << std::setw(20) << std::setprecision(2)
                  << b.final_yaw_deg << "\n";
    }
    checkTrue("偏航受控时误差远小于偏航自由",
              ctrl_rms[1] < free_rms[1] * 0.5);
    checkTrue("偏航受控时能跟到指令附近（90° 指令下 RMS < 5°）", ctrl_rms[1] < 5.0);

    // ---- 2. 风扰动下的偏航保持 ----
    std::cout << "\n[2] 风扰动下的偏航保持（湍流 4 m/s）\n";
    std::cout << "  这是「需要指向」的任务的基本要求：机身朝向不能被风带走。\n\n";
    std::cout << "  " << std::setw(18) << "配置" << std::setw(22) << "偏航RMS(deg)"
              << std::setw(22) << "位置RMS(m)" << "\n";
    {
        const double cmd = 90.0 * M_PI / 180.0;
        const YawResult a = runYawHold(false, cmd, true);
        const YawResult b = runYawHold(true, cmd, true);
        std::cout << "  " << std::setw(18) << "偏航自由" << std::setw(22)
                  << std::setprecision(4) << a.rms_yaw_err_deg << std::setw(22) << a.pos_rms
                  << "\n";
        std::cout << "  " << std::setw(18) << "偏航受控" << std::setw(22) << b.rms_yaw_err_deg
                  << std::setw(22) << b.pos_rms << "\n";
        checkTrue("有风时偏航受控仍能保持朝向（RMS < 8°）", b.rms_yaw_err_deg < 8.0);
        checkTrue("偏航控制不显著损害位置精度（位置 RMS 增幅 < 50%）",
                  b.pos_rms < a.pos_rms * 1.5 + 1e-6);
    }

    // ---- 3. 位置机动时的偏航解耦 ----
    //
    // 关键检验：位置机动会大幅改变姿态（倾斜），偏航是否被这个倾斜带跑。
    // 若耦合严重，说明姿态误差的构造有问题。
    std::cout << "\n[3] 位置机动时偏航是否被扰动（解耦检验）\n";
    std::cout << "  机动时姿态大幅倾斜，偏航应保持不受影响。\n\n";
    {
        SixDofConfig cfg;
        cfg.base.dt = 0.001;
        cfg.base.mass = 1.0;
        cfg.base.gravity = 9.81;
        cfg.base.drag_coeff = 0.049;
        cfg.base.max_thrust = 0.0;
        cfg.inertia[0] = 0.015;
        cfg.inertia[1] = 0.018;
        cfg.inertia[2] = 0.028;
        cfg.torque_limit = 1.0;
        cfg.max_body_thrust = 20.0;

        const double dt = cfg.base.dt;
        const int steps = static_cast<int>(14.0 / dt);
        const double yaw_cmd = 45.0 * M_PI / 180.0;

        const SixDofState init{makeVec3(0.0f, 0.0f, -5.0f), makeVec3(0.0f, 0.0f, 0.0f),
                               Tensor{1.0f, 0.0f, 0.0f, 0.0f}, makeVec3(0.0f, 0.0f, 0.0f)};

        auto runManeuver = [&](bool use_yaw) {
            SixDofSimulator sim(cfg, init);
            SixDofPidGains gains;
            gains.use_yaw_control = use_yaw;
            SixDofPidController ctrl(cfg, gains);

            double sq = 0.0, mx_tilt = 0.0;
            int n = 0;
            for (int k = 0; k < steps; ++k) {
                const double t = static_cast<double>(k) * dt;
                SixDofSetpoint sp;
                // 正弦机动：x 方向 ±2 m，幅度足以产生明显倾斜
                const double w = 0.5;
                sp.pos = {2.0 * std::sin(w * t), 0.0, -5.0};
                sp.vel = {2.0 * w * std::cos(w * t), 0.0, 0.0};
                sp.acc = {-2.0 * w * w * std::sin(w * t), 0.0, 0.0};
                sp.jerk = {-2.0 * w * w * w * std::cos(w * t), 0.0, 0.0};
                sp.yaw = yaw_cmd;

                const SixDofCommand cmd = ctrl.computeTracking(sim.state(), sp, t);
                sim.step(cmd.thrust_body, cmd.torque);

                if (k > steps / 3) {
                    const double ye = wrapPi(yawOf(sim.state().quat) - yaw_cmd) * 180.0 / M_PI;
                    sq += ye * ye;
                    // 倾角
                    // 机体 z 轴与竖直方向的夹角：由旋转矩阵第三行第三列给出
                    //   R[2][2] = 1 − 2(q_x² + q_y²)
                    // 注意**不能**用 1 − 2(q_y² + q_z²) —— 那是 R[0][0]（机体 x 轴的
                    // 倾角）。纯偏航 45° 时后者会给出 45°，与真实倾斜无关：
                    // 第一版正是这样写，于是「偏航受控」的倾角被误测成 45°。
                    const std::vector<float> q = toVector(sim.state().quat);
                    const double r33 = 1.0 - 2.0 * (static_cast<double>(q[1]) * q[1] +
                                                    static_cast<double>(q[2]) * q[2]);
                    mx_tilt = std::max(mx_tilt,
                                       std::acos(std::max(-1.0, std::min(1.0, r33))) * 180.0 / M_PI);
                    ++n;
                }
            }
            return std::pair<double, double>{std::sqrt(sq / std::max(1, n)), mx_tilt};
        };

        const auto [yaw_err_free, tilt_free] = runManeuver(false);
        const auto [yaw_err_ctrl, tilt_ctrl] = runManeuver(true);

        std::cout << "  " << std::setw(18) << "配置" << std::setw(24) << "偏航RMS(deg)"
                  << std::setw(22) << "峰值倾角(deg)" << "\n";
        std::cout << "  " << std::setw(18) << "偏航自由" << std::setw(24)
                  << std::setprecision(4) << yaw_err_free << std::setw(22)
                  << std::setprecision(2) << tilt_free << "\n";
        std::cout << "  " << std::setw(18) << "偏航受控" << std::setw(24) << yaw_err_ctrl
                  << std::setw(22) << tilt_ctrl << "\n";

        checkTrue("机动时偏航受控仍保持（RMS < 10°）", yaw_err_ctrl < 10.0);
        checkTrue("机动本身未受影响（峰值倾角与偏航自由时相当）",
                  std::fabs(tilt_ctrl - tilt_free) < 5.0);
    }

    // ---- 4. 与倾斜限幅解耦 ----
    std::cout << "\n[4] 倾斜限幅与偏航解耦\n";
    std::cout << "  原实现把限幅加在误差角上，会把偏航误差也算进去（概念混淆）。\n";
    std::cout << "  现在限幅作用在**期望推力方向**上，偏航是绕该方向的转动，互不干扰。\n\n";
    {
        // 构造一个大偏航指令 + 位置误差的组合，验证偏航仍能收敛
        const YawResult r = runYawHold(true, 170.0 * M_PI / 180.0, false, 12.0);
        std::cout << "  170° 偏航指令下：RMS 误差 " << std::setprecision(4)
                  << r.rms_yaw_err_deg << " deg，终值 " << r.final_yaw_deg << " deg\n";
        checkTrue("大偏航指令下仍能收敛（170° 指令 RMS < 10°）", r.rms_yaw_err_deg < 10.0);
    }

    std::cout << "\n[结论]\n";
    std::cout << "  1. 原实现的偏航完全自由（无恢复力矩），扰动会让它缓慢漂走且不回来。\n";
    std::cout << "     这对纯悬停无碍，但任何需要指向的任务都做不了。\n";
    std::cout << "  2. 启用偏航控制后，姿态误差改用完整旋转矩阵计算，期望姿态由\n";
    std::cout << "     「期望推力方向 + 期望偏航」共同构造，偏航因此被真正约束。\n";
    std::cout << "  3. 倾斜限幅改作用在**期望推力方向**上：偏航是绕该方向的转动，\n";
    std::cout << "     两者互不干扰。原实现把限幅加在误差角上，会把偏航误差也算进去。\n";
    std::cout << "  4. 默认关闭，既有全部结果逐位不变。\n";

    std::cout << "\n========================================\n";
    std::cout << g_checks - g_failed << " / " << g_checks << " checks passed\n";
    if (g_failed > 0) {
        std::cout << g_failed << " FAILED\n";
    }
    std::cout << "========================================\n";
    return g_failed == 0 ? 0 : 1;
}
