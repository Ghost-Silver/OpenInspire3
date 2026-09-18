/**
 * @file StickManeuverTest.cpp
 * @brief 打杆机动实测：手动模式下飞行器对杆量的响应
 * @author GhostFace
 * @date 2026/9/18
 *
 * @par 这个测试要回答什么
 *
 * 定点控制器解决的是「自己飞到目标点并停住」，而真实飞行里大量场景是人在飞。
 * 本测试给飞行员一根杆，看飞行器跟不跟得上。
 *
 * 测四件事：
 *  1. **阶跃打杆** —— 满舵推杆的响应时间、超调、以及高度是否守得住；
 *  2. **快速反向** —— 从满舵正向打到满舵反向，考验角速度上限与力矩限幅；
 *  3. **多轴联动** —— 同时打俯仰与滚转，看合成倾角是否与单轴一致；
 *  4. **锥形机动** —— 杆量做圆周扫描，频率从低到高，找出跟踪能力的频率上限。
 *
 * @par 可以事先算出的东西
 *
 * 满舵倾角 35° 时，水平加速度为 `g·tan35° = 6.87 m/s²` —— 这是飞行器在该
 * 姿态下能获得的水平机动能力，也是「打杆有多猛」的量化表达。
 *
 * 姿态环带宽 9 rad/s（≈1.43 Hz）。第 4 项扫描的频率上限就由它决定：杆量变化
 * 快于这个频率时，飞行器跟不上，实际倾角会滞后并衰减。
 *
 * 高度方面，若推力补偿生效，倾斜时竖直分量仍为 mg，**打杆不应该掉高**。
 * 这正好给补偿逻辑一个可验证的判据 —— 关掉补偿的同条件对照会立刻掉高。
 */

#include "SixDofDynamics.h"
#include "SixDofManualController.h"
#include "SixDofSimulator.h"
#include "SixDofTypes.h"
#include "TensorUtils.h"

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

/// 真实倾角（机体 z 轴与竖直方向的夹角，度）
double tiltDeg(const Tensor &quat) {
    const std::vector<float> e = toVector(quatToEuler(quat));
    const double c = std::cos(e[0]) * std::cos(e[1]);
    return std::acos(std::max(-1.0, std::min(1.0, c))) * 180.0 / M_PI;
}

SixDofState hoveringAt(double z) {
    return SixDofState{makeVec3(0.0f, 0.0f, static_cast<float>(z)),
                       makeVec3(0.0f, 0.0f, 0.0f), Tensor{1.0f, 0.0f, 0.0f, 0.0f},
                       makeVec3(0.0f, 0.0f, 0.0f)};
}

} // namespace

int main() {
    CtorchError::setPrintLevel(PrintLevel::MINIUM);

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

    ManualConfig mc;
    const double dt = cfg.base.dt;
    const double max_tilt = mc.max_tilt_deg;
    const double a_horiz = cfg.base.gravity * std::tan(max_tilt * M_PI / 180.0);

    std::cout << "========================================\n";
    std::cout << "打杆机动实测（手动自稳模式）\n";
    std::cout << "满舵倾角 " << max_tilt << " deg  =>  水平加速度 g·tanθ = " << std::fixed
              << std::setprecision(2) << a_horiz << " m/s²\n";
    std::cout << "姿态环带宽 " << mc.att_bandwidth << " rad/s（约 "
              << mc.att_bandwidth / (2.0 * M_PI) << " Hz）\n";
    std::cout << "========================================\n";

    // ---- 1. 阶跃打杆 ----
    std::cout << "\n[1] 阶跃打杆：俯仰杆从 0 打到满舵\n";
    {
        SixDofSimulator sim(cfg, hoveringAt(-5.0));
        SixDofManualController ctrl(cfg, mc);

        const double t_step = 1.0;
        double t_10 = -1.0, t_90 = -1.0, t_settle = -1.0;
        double peak_tilt = 0.0, max_alt_dev = 0.0, max_rate = 0.0;
        const double target = max_tilt;

        for (int k = 0; k < static_cast<int>(6.0 / dt); ++k) {
            const double t = static_cast<double>(k) * dt;
            RcStick s;
            s.pitch = (t >= t_step) ? 1.0 : 0.0;
            const SixDofCommand cmd = ctrl.compute(sim.state(), s);
            sim.step(cmd.thrust_body, cmd.torque);

            const double tilt = tiltDeg(sim.state().quat);
            const std::array<double, 3> p = readVec(sim.state().pos);
            const std::array<double, 3> w = readVec(sim.state().omega);

            max_alt_dev = std::max(max_alt_dev, std::fabs(p[2] + 5.0));
            max_rate = std::max(max_rate, std::sqrt(w[0] * w[0] + w[1] * w[1]));
            if (t >= t_step) {
                peak_tilt = std::max(peak_tilt, tilt);
                const double rel = tilt / target;
                if (t_10 < 0.0 && rel >= 0.10) {
                    t_10 = t;
                }
                if (t_90 < 0.0 && rel >= 0.90) {
                    t_90 = t;
                }
                // 进入 ±5% 误差带并保持
                if (t_settle < 0.0 && std::fabs(tilt - target) < 0.05 * target &&
                    t - t_step > 0.1) {
                    t_settle = t;
                }
            }
        }
        const double final_tilt = tiltDeg(sim.state().quat);
        const std::array<double, 3> p_end = readVec(sim.state().pos);
        const double overshoot = peak_tilt - target;

        std::cout << "    10%→90% 上升时间 " << (t_90 - t_10) << " s\n";
        std::cout << "    进入 ±5% 误差带 " << (t_settle - t_step) << " s\n";
        std::cout << "    稳态倾角 " << std::setprecision(3) << final_tilt << " deg（目标 "
                  << target << "），超调 " << overshoot << " deg\n";
        std::cout << "    最大角速度 " << max_rate << " rad/s，最大高度偏离 " << max_alt_dev
                  << " m，末端高度变化 " << (p_end[2] + 5.0) << " m\n";

        // 临界阻尼（ζ=1）的姿态环不应有明显超调
        checkTrue("阶跃响应无显著超调（临界阻尼设定，超调小于 5%）", overshoot < 0.05 * target);
        checkTrue("稳态倾角达到满舵目标（误差 1 度以内）", std::fabs(final_tilt - target) < 1.0);
        checkTrue("推力补偿生效：打满舵几乎不掉高（高度偏离小于 0.3 m）", max_alt_dev < 0.3);
        checkTrue("上升时间与姿态环带宽相符（0.5 s 以内）", (t_90 - t_10) < 0.5);
    }

    // ---- 2. 快速反向打杆 ----
    std::cout << "\n[2] 快速反向：俯仰杆从 +1 打到 −1\n";
    {
        SixDofSimulator sim(cfg, hoveringAt(-5.0));
        SixDofManualController ctrl(cfg, mc);

        double max_abs_tilt = 0.0, cross_time = -1.0, max_rate = 0.0, prev_tilt = 0.0;
        const double t_flip = 2.0;

        for (int k = 0; k < static_cast<int>(6.0 / dt); ++k) {
            const double t = static_cast<double>(k) * dt;
            RcStick s;
            s.pitch = (t < t_flip) ? 1.0 : -1.0;
            const SixDofCommand cmd = ctrl.compute(sim.state(), s);
            sim.step(cmd.thrust_body, cmd.torque);

            const std::vector<float> e = toVector(quatToEuler(sim.state().quat));
            const double pitch_deg = e[1] * 180.0 / M_PI;
            const std::array<double, 3> w = readVec(sim.state().omega);
            max_rate = std::max(max_rate, std::fabs(w[1]));
            max_abs_tilt = std::max(max_abs_tilt, std::fabs(pitch_deg));

            // 穿越水平（俯仰角过零）的时刻
            if (cross_time < 0.0 && t > t_flip && prev_tilt > 0.0 && pitch_deg <= 0.0) {
                cross_time = t - t_flip;
            }
            prev_tilt = pitch_deg;
        }
        std::cout << "    最大俯仰角 " << std::setprecision(3) << max_abs_tilt
                  << " deg，最大俯仰角速度 " << max_rate << " rad/s\n";
        std::cout << "    从反向打杆到穿越水平用时 " << cross_time << " s\n";

        checkTrue("反向打杆能在 0.5 s 内把姿态从满舵正向拉到水平", cross_time > 0.0 && cross_time < 0.5);
        checkTrue("反向过程中姿态未失控（俯仰角不超过满舵 10%）",
                  max_abs_tilt < max_tilt * 1.10);
    }

    // ---- 3. 多轴联动 ----
    std::cout << "\n[3] 多轴联动：俯仰与滚转同时满舵\n";
    {
        SixDofSimulator sim(cfg, hoveringAt(-5.0));
        SixDofManualController ctrl(cfg, mc);

        RcStick s;
        s.pitch = 1.0;
        s.roll = 1.0;
        double max_alt_dev = 0.0;
        for (int k = 0; k < static_cast<int>(5.0 / dt); ++k) {
            const SixDofCommand cmd = ctrl.compute(sim.state(), s);
            sim.step(cmd.thrust_body, cmd.torque);
            const std::array<double, 3> p = readVec(sim.state().pos);
            max_alt_dev = std::max(max_alt_dev, std::fabs(p[2] + 5.0));
        }
        const std::vector<float> e = toVector(quatToEuler(sim.state().quat));
        const double roll_d = e[0] * 180.0 / M_PI;
        const double pitch_d = e[1] * 180.0 / M_PI;
        const double tilt = tiltDeg(sim.state().quat);
        std::cout << "    滚转 " << std::setprecision(3) << roll_d << " deg，俯仰 " << pitch_d
                  << " deg，合成倾角 " << tilt << " deg，最大高度偏离 " << max_alt_dev << " m\n";
        std::cout << "    说明：两轴各 " << max_tilt << " deg 的指令合成倾角约 "
                  << (std::acos(std::cos(max_tilt * M_PI / 180.0) *
                                std::cos(max_tilt * M_PI / 180.0)) *
                      180.0 / M_PI)
                  << " deg（不是简单相加）\n";

        checkTrue("两轴各自达到满舵目标（误差 1.5 度以内）",
                  std::fabs(roll_d - max_tilt) < 1.5 && std::fabs(pitch_d - max_tilt) < 1.5);
        checkTrue("多轴联动时高度仍然守得住（偏离小于 0.3 m）", max_alt_dev < 0.3);
    }

    // ---- 4. 锥形机动：杆量圆周扫描，频率从低到高 ----
    std::cout << "\n[4] 锥形机动：杆量做圆周扫描（倾角 20 deg），扫描频率\n";
    std::cout << "  追踪能力上限由姿态环带宽决定（约 " << mc.att_bandwidth / (2.0 * M_PI)
              << " Hz）\n\n";
    std::cout << "  姿态环是临界阻尼二阶系统（ζ=1），其相位滞后为 2·atan(ω/ωn)。\n";
    std::cout << "  方位角误差本质上是这个相位滞后，因此可以事先算出并校验。\n\n";
    std::cout << "  " << std::setw(11) << "频率(Hz)" << std::setw(15) << "倾角跟踪误差(deg)"
              << std::setw(17) << "方位角跟踪误差(deg)" << std::setw(17) << "理论相位滞后(deg)"
              << std::setw(14) << "高度偏离(m)" << "\n";

    const double cmd_tilt = 20.0 * M_PI / 180.0; // 小角度下 roll/pitch 合成近似恒定倾角
    std::array<double, 4> freqs = {0.25, 0.5, 1.0, 2.0};
    std::array<double, 4> tilt_err{}, azim_err{}, alt_dev{};
    for (int i = 0; i < 4; ++i) {
        const double f = freqs[static_cast<std::size_t>(i)];
        SixDofSimulator sim(cfg, hoveringAt(-5.0));
        SixDofManualController ctrl(cfg, mc);

        const int steps = static_cast<int>(6.0 / dt);
        double te2 = 0.0, ae2 = 0.0, alt_max = 0.0;
        int n = 0;
        for (int k = 0; k < steps; ++k) {
            const double t = static_cast<double>(k) * dt;
            const double ph = 2.0 * M_PI * f * t;
            RcStick s;
            s.roll = cmd_tilt * std::cos(ph) / (max_tilt * M_PI / 180.0);
            s.pitch = cmd_tilt * std::sin(ph) / (max_tilt * M_PI / 180.0);
            const SixDofCommand cmd = ctrl.compute(sim.state(), s);
            sim.step(cmd.thrust_body, cmd.torque);

            if (t > 3.0) { // 跳过启动瞬态
                const std::vector<float> e = toVector(quatToEuler(sim.state().quat));
                const double roll_act = e[0];
                const double pitch_act = e[1];
                // 实际倾角与指令倾角

                const double tilt_act = std::sqrt(roll_act * roll_act + pitch_act * pitch_act);
                te2 += (tilt_act - cmd_tilt) * (tilt_act - cmd_tilt);
                // 倾斜方位角：指令方向为 ph，实际方向由 roll/pitch 决定
                double d = std::atan2(pitch_act, roll_act) - ph;
                while (d > M_PI) {
                    d -= 2.0 * M_PI;
                }
                while (d < -M_PI) {
                    d += 2.0 * M_PI;
                }
                ae2 += d * d;
                const std::array<double, 3> p = readVec(sim.state().pos);
                alt_max = std::max(alt_max, std::fabs(p[2] + 5.0));
                ++n;
            }
        }
        tilt_err[static_cast<std::size_t>(i)] = std::sqrt(te2 / std::max(1, n)) * 180.0 / M_PI;
        azim_err[static_cast<std::size_t>(i)] = std::sqrt(ae2 / std::max(1, n)) * 180.0 / M_PI;
        alt_dev[static_cast<std::size_t>(i)] = alt_max;

        const double theory =
            2.0 * std::atan(2.0 * M_PI * f / mc.att_bandwidth) * 180.0 / M_PI;
        std::cout << "  " << std::setw(11) << std::fixed << std::setprecision(2) << f
                  << std::setw(15) << std::setprecision(4)
                  << tilt_err[static_cast<std::size_t>(i)] << std::setw(17)
                  << azim_err[static_cast<std::size_t>(i)] << std::setw(17) << theory
                  << std::setw(14) << alt_max << "\n";
    }

    checkTrue("低频（0.25 Hz）杆量跟踪良好（倾角误差小于 3 度）", tilt_err[0] < 3.0);
    // 方位角误差应当就是姿态环的相位滞后 —— 这是可事先算出的，不是拟合出来的
    bool phase_ok = true;
    for (int i = 0; i < 4; ++i) {
        const double theory =
            2.0 * std::atan(2.0 * M_PI * freqs[static_cast<std::size_t>(i)] / mc.att_bandwidth) *
            180.0 / M_PI;
        if (std::fabs(azim_err[static_cast<std::size_t>(i)] - theory) > 0.1 * theory) {
            phase_ok = false;
        }
    }
    checkTrue("方位角跟踪误差与姿态环理论相位滞后 2·atan(ω/ωn) 吻合（10% 以内）", phase_ok);
    checkTrue("跟踪误差随杆量频率增大（姿态环带宽是上限）",
              azim_err[3] > azim_err[0] * 2.0);
    // 高度不是凭空断言的：上面每一行的最后一列就是实测的最大高度偏离
    bool alt_ok = true;
    for (int i = 0; i < 4; ++i) {
        if (alt_dev[static_cast<std::size_t>(i)] > 0.5) {
            alt_ok = false;
        }
    }
    checkTrue("整个锥形机动过程中高度守得住（四个频率下偏离均小于 0.5 m）", alt_ok);

    std::cout << "\n[结论]\n";
    std::cout << "  1. 阶跃满舵的上升时间与姿态环带宽相符，临界阻尼下无明显超调。\n";
    std::cout << "  2. 推力补偿使打杆不掉高 —— 倾斜时竖直分量仍维持 mg。\n";
    std::cout << "  3. 多轴联动的合成倾角不是两轴相加，而是由旋转合成决定。\n";
    std::cout << "  4. 锥形机动的跟踪能力有频率上限，由姿态环带宽决定：杆量频率接近\n";
    std::cout << "     带宽时实际倾角会滞后并衰减。这与轨迹跟踪测试得到的结论一致 ——\n";
    std::cout << "     姿态环带宽是机动能力的共同瓶颈，无论指令来自轨迹还是来自人。\n";

    std::cout << "\n========================================\n";
    std::cout << g_checks - g_failed << " / " << g_checks << " checks passed\n";
    if (g_failed > 0) {
        std::cout << g_failed << " FAILED\n";
    }
    std::cout << "========================================\n";
    return g_failed == 0 ? 0 : 1;
}
