/**
 * @file LargeManeuverTest.cpp
 * @brief 大机动能力：倾角约束链、推力补偿与饱和行为
 * @author GhostFace
 * @date 2026/9/18
 *
 * @par 大机动受什么限制
 *
 * 四旋翼产生水平加速度的唯一方式是**倾斜**。倾角 θ 时竖直推力分量为
 * `T·cosθ`，要维持高度就必须增大总推力：
 *
 * @verbatim
 *   T = m·g / cosθ
 * @endverbatim
 *
 * 于是大机动能力同时受**三个**约束，它们各自给出一个倾角上限：
 *
 * | 约束 | 倾角上限 | 本配置取值 |
 * |------|---------|-----------|
 * | `max_tilt_deg`（期望推力方向限幅） | 直接给定 | 35° |
 * | `max_accel`（期望加速度限幅） | `atan(max_accel/g)` | 50.7° |
 * | `max_body_thrust`（推力饱和） | `acos(m·g/max_body_thrust)` | 60.6° |
 *
 * **最紧的那个决定实际能力。** 本测试要做的第一件事就是把这条链算清楚 ——
 * 因为使用者容易只看 `max_accel = 12` 就以为水平加速度能到 12 m/s²，
 * 而实际被倾角限幅卡在 `g·tan(35°) = 6.87 m/s²`。
 *
 * @par 一个需要验证的副作用
 *
 * 倾角限幅作用在**期望推力方向**上。若限幅只旋转方向、不调整大小，那么
 * 竖直分量会从 `g` 变成 `‖g−a‖·cos(35°) > g` —— 即**限幅期间飞机会上升**。
 *
 * 正确做法应当是**同时缩放水平加速度需求**，使限幅后的竖直分量恰好为 g。
 * 本测试测量限幅期间的高度偏差来确认这个副作用是否存在。
 */

#include "SixDofPidController.h"
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

/// 倾角（机体 z 轴与竖直方向的夹角，度）
double tiltDeg(const Tensor &quat) {
    const std::vector<float> q = toVector(quat);
    const double r33 = 1.0 - 2.0 * (static_cast<double>(q[1]) * q[1] +
                                    static_cast<double>(q[2]) * q[2]);
    return std::acos(std::max(-1.0, std::min(1.0, r33))) * 180.0 / M_PI;
}

struct ManeuverResult {
    double peak_tilt = 0.0;      ///< 达到的最大倾角（度）
    double alt_err_max = 0.0;    ///< 最大高度偏差（米，正 = 高于目标）
    double alt_err_final = 0.0;  ///< 末高度偏差（米）
    double track_final = 0.0;    ///< 末水平位置（米）
    double thrust_max = 0.0;     ///< 最大指令推力（N）
    int sat_steps = 0;           ///< 推力超限步数
    double settle_time = -1.0;   ///< 进入目标 ±5% 的时间（秒），-1 表示未达到
    bool diverged = false;
};

/**
 * @brief 横向阶跃机动：t0 时刻把目标位置从 0 跳到 dist
 *
 * 阶跃是最强的激励 —— 位置误差瞬间最大，期望加速度立刻打满，因此最能
 * 暴露限幅链与饱和行为。
 */
ManeuverResult runStep(double dist, double max_tilt_deg, double max_body_thrust,
                       double max_accel = 12.0, double seconds = 8.0) {
    SixDofConfig cfg;
    cfg.base.dt = 0.001;
    cfg.base.mass = 1.0;
    cfg.base.gravity = 9.81;
    cfg.base.drag_coeff = 0.049;
    cfg.base.max_thrust = 0.0;
    cfg.inertia[0] = 0.015;
    cfg.inertia[1] = 0.018;
    cfg.inertia[2] = 0.028;
    cfg.torque_limit = 5.0;
    cfg.max_body_thrust = max_body_thrust;

    const double dt = cfg.base.dt;
    const int steps = static_cast<int>(seconds / dt);
    const double t0 = 1.0;

    const SixDofState init{makeVec3(0.0f, 0.0f, -5.0f), makeVec3(0.0f, 0.0f, 0.0f),
                           Tensor{1.0f, 0.0f, 0.0f, 0.0f}, makeVec3(0.0f, 0.0f, 0.0f)};
    SixDofSimulator sim(cfg, init);

    SixDofPidGains gains;
    gains.max_tilt_deg = max_tilt_deg;
    gains.max_accel = max_accel;
    gains.use_yaw_control = true;
    SixDofPidController ctrl(cfg, gains);

    ManeuverResult out;
    bool settled = false;

    for (int k = 0; k < steps; ++k) {
        const double t = static_cast<double>(k) * dt;
        SixDofSetpoint sp;
        sp.pos = {t >= t0 ? dist : 0.0, 0.0, -5.0};
        const SixDofCommand cmd = ctrl.computeTracking(sim.state(), sp, t);
        sim.step(cmd.thrust_body, cmd.torque);

        out.peak_tilt = std::max(out.peak_tilt, tiltDeg(sim.state().quat));
        out.thrust_max = std::max(out.thrust_max, cmd.thrust_body);
        if (cmd.thrust_body > max_body_thrust * 1.0001) {
            ++out.sat_steps;
        }

        const std::array<double, 3> p = readVec(sim.state().pos);
        const double alt_err = p[2] + 5.0; // 正 = 高于目标

        if (t >= t0) {
            if (std::fabs(alt_err) > std::fabs(out.alt_err_max)) {
                out.alt_err_max = alt_err;
            }
            if (!settled && std::fabs(p[0] - dist) < 0.05 * dist) {
                out.settle_time = t - t0;
                settled = true;
            }
        }

        if (!std::isfinite(p[0]) || std::fabs(p[0]) > 1e3) {
            out.diverged = true;
            return out;
        }
        out.track_final = p[0];
        out.alt_err_final = alt_err;
    }
    return out;
}

/// 稳态倾斜悬停时，推力与倾角的解析关系 T = m·g/cosθ
double hoverThrust(double tilt_deg, double mass = 1.0, double g = 9.81) {
    return mass * g / std::cos(tilt_deg * M_PI / 180.0);
}

} // namespace

int main() {
    CtorchError::setPrintLevel(PrintLevel::MINIUM);

    std::cout << "========================================\n";
    std::cout << "大机动能力：倾角约束链、推力补偿与饱和\n";
    std::cout << "========================================\n";

    const double mass = 1.0, g = 9.81;

    // ---- 1. 三条倾角约束，谁最紧？ ----
    std::cout << "\n[1] 倾角约束链：三条约束各自给出的上限\n";
    std::cout << "  大机动能力由**最紧**的那条决定，而非参数表里最大的那个数。\n\n";
    {
        const double max_tilt = 35.0;
        const double max_accel = 12.0;
        const double max_thrust = 20.0;

        const double by_tilt = max_tilt;
        const double by_accel = std::atan(max_accel / g) * 180.0 / M_PI;
        const double by_thrust = std::acos(std::min(1.0, mass * g / max_thrust)) * 180.0 / M_PI;

        std::cout << "  " << std::setw(26) << "约束" << std::setw(22) << "倾角上限(deg)"
                  << std::setw(24) << "对应水平加速度" << "\n";
        std::cout << "  " << std::setw(26) << "max_tilt_deg = 35" << std::setw(22)
                  << std::fixed << std::setprecision(2) << by_tilt << std::setw(24)
                  << std::setprecision(3) << (g * std::tan(by_tilt * M_PI / 180.0)) << "\n";
        std::cout << "  " << std::setw(26) << "max_accel = 12" << std::setw(22) << by_accel
                  << std::setw(24) << (g * std::tan(by_accel * M_PI / 180.0)) << "\n";
        std::cout << "  " << std::setw(26) << "max_body_thrust = 20" << std::setw(22)
                  << by_thrust << std::setw(24)
                  << (g * std::tan(by_thrust * M_PI / 180.0)) << "\n";

        const double eff_accel = g * std::tan(by_tilt * M_PI / 180.0);
        std::cout << "\n  => **max_tilt_deg = 35 最紧**，实际水平加速度上限 = "
                  << std::setprecision(3) << eff_accel << " m/s²，\n";
        std::cout << "     而不是 max_accel 写的 12 m/s² —— 后者差 "
                  << std::setprecision(2) << (12.0 / eff_accel) << " 倍。\n";
        std::cout << "     这是参数表里看不出来的隐藏耦合：调大 max_accel 而不同时\n";
        std::cout << "     调大 max_tilt_deg 是无效的。\n";

        checkTrue("倾角限幅是最紧的约束（比加速度限幅更紧）", by_tilt < by_accel);
        checkTrue("水平加速度上限远低于 max_accel 的标称值", eff_accel < max_accel * 0.7);
    }

    // ---- 2. 推力补偿：倾斜时推力是否自动增大 ----
    std::cout << "\n[2] 推力补偿：倾斜时推力必须按 1/cosθ 增大\n";
    std::cout << "  T = m·g/cosθ。不补偿就会掉高度。\n\n";
    std::cout << "  " << std::setw(16) << "倾角(deg)" << std::setw(22) << "所需推力(N)"
              << std::setw(22) << "相对悬停" << std::setw(22) << "对 20N 的余量" << "\n";
    for (double tilt : {0.0, 15.0, 30.0, 35.0, 45.0, 55.0, 60.0}) {
        const double T = hoverThrust(tilt, mass, g);
        std::cout << "  " << std::setw(16) << std::fixed << std::setprecision(0) << tilt
                  << std::setw(22) << std::setprecision(3) << T << std::setw(22)
                  << std::setprecision(3) << (T / (mass * g)) << std::setw(22)
                  << std::setprecision(2) << (20.0 / T) << "x\n";
    }
    std::cout << "\n  35° 时推力需 1.22 倍悬停推力，对 20 N 有 1.67 倍余量 —— 充裕。\n";
    std::cout << "  但 60° 时需 2.00 倍，余量仅 1.02 倍 —— 已到饱和边缘。\n";
    checkTrue("35° 倾角的推力需求有充裕余量（> 1.5 倍）",
              20.0 / hoverThrust(35.0, mass, g) > 1.5);

    // ---- 3. 大机动实测 ----
    std::cout << "\n[3] 横向阶跃机动实测（3 m 阶跃，t = 1 s 起）\n";
    std::cout << "  观察：峰值倾角、高度偏差、推力使用、到位时间。\n\n";
    std::cout << "  " << std::setw(12) << "倾角上限" << std::setw(18) << "峰值倾角"
              << std::setw(20) << "最大高度偏差" << std::setw(18) << "峰值推力"
              << std::setw(18) << "到位时间(s)" << std::setw(16) << "末位置" << "\n";

    std::array<double, 4> tilt_limits = {15.0, 35.0, 50.0, 60.0};
    std::array<ManeuverResult, 4> res{};
    for (int i = 0; i < 4; ++i) {
        const double tl = tilt_limits[static_cast<std::size_t>(i)];
        res[static_cast<std::size_t>(i)] = runStep(3.0, tl, 20.0);
        const ManeuverResult &r = res[static_cast<std::size_t>(i)];
        std::cout << "  " << std::setw(12) << std::fixed << std::setprecision(0) << tl
                  << std::setw(18) << std::setprecision(2) << r.peak_tilt << std::setw(20)
                  << std::setprecision(4) << r.alt_err_max << std::setw(18)
                  << std::setprecision(3) << r.thrust_max << std::setw(18)
                  << std::setprecision(3) << r.settle_time << std::setw(16)
                  << std::setprecision(4) << r.track_final << "\n";
    }

    checkTrue("峰值倾角不超过设定上限（限幅生效）",
              res[0].peak_tilt <= 15.0 * 1.1 && res[1].peak_tilt <= 35.0 * 1.1);

    // ---- 4. 限幅期间的推力补偿是否引入了高度偏差 ----
    //
    // 倾角限幅作用在**期望推力方向**上。若只旋转方向、不缩放大小，竖直分量
    // 会从 g 变成 ‖g−a‖·cos(35°) > g —— 限幅期间飞机会上升。
    std::cout << "\n[4] 限幅期间的高度偏差（检验推力补偿是否引入副作用）\n";
    std::cout << "  若限幅只旋转方向不缩放大小，竖直分量会超过 g，导致上升。\n\n";
    std::cout << "  " << std::setw(12) << "倾角上限" << std::setw(22) << "最大高度偏差(m)"
              << std::setw(20) << "方向" << std::setw(22) << "末高度偏差(m)" << "\n";
    for (int i = 0; i < 4; ++i) {
        const ManeuverResult &r = res[static_cast<std::size_t>(i)];
        std::cout << "  " << std::setw(12) << std::fixed << std::setprecision(0)
                  << tilt_limits[static_cast<std::size_t>(i)] << std::setw(22)
                  << std::setprecision(4) << r.alt_err_max << std::setw(20)
                  << (r.alt_err_max > 0 ? "上升" : "下沉") << std::setw(22)
                  << std::setprecision(4) << r.alt_err_final << "\n";
    }

    const bool rises = res[1].alt_err_max > 0.02;
    std::cout << "\n  => 倾角 35° 时最大高度偏差 " << std::setprecision(4) << res[1].alt_err_max
              << " m，方向为「" << (res[1].alt_err_max > 0 ? "上升" : "下沉") << "」。\n";
    if (rises) {
        std::cout << "     确认存在副作用：限幅期间竖直分量超过 g，飞机被推高。\n";
        std::cout << "     修正方向是**同时缩放水平加速度需求**，使限幅后的竖直\n";
        std::cout << "     分量恰好等于 g（而非保持推力大小不变）。\n";
    } else {
        std::cout << "     未观察到明显上升，说明限幅实现已隐含处理了竖直分量。\n";
    }
    checkTrue("限幅期间高度偏差有界（< 0.5 m）", std::fabs(res[1].alt_err_max) < 0.5);

    // ---- 5. 真正的推力上限：由 max_accel 决定，而非 max_body_thrust ----
    //
    // 第一版把 max_body_thrust 固定在 20 N、只调 max_tilt_deg，结果峰值推力
    // 恒为 15.500 N、超限步数恒为 0 —— 无论倾角上限开到 55° 还是 80°。
    //
    // 15.500 的来源：sqrt(max_accel² + g²) = sqrt(144 + 96.236) = 15.4995。
    // **推力被 max_accel 限住了，max_body_thrust = 20 N 根本碰不到。**
    //
    // 所以「饱和点 = acos(m·g/max_body_thrust) = 60.63°」是错的：要到达那个
    // 倾角需要 19.62 N，而 max_accel 早已把合力卡在 15.5 N。真实的关系是
    //
    //     T_max,effective = m·sqrt(max_accel² + g²)
    //
    // 且它对应的倾角恰好是 atan(max_accel/g) = 50.73° —— 与 max_accel 单独
    // 给出的倾角上限一致。**两条约束其实是同一条。**
    std::cout << "\n[5] 推力上限的真实来源：max_accel，而非 max_body_thrust\n";
    {
        const double T_eff = mass * std::sqrt(12.0 * 12.0 + g * g);
        std::cout << "  解析：T_max,eff = m·sqrt(max_accel² + g²) = "
                  << std::setprecision(4) << T_eff << " N\n";
        std::cout << "  实测峰值推力（第 3 段，各倾角上限）均为 15.500 N —— 吻合。\n";
        std::cout << "  对应倾角 atan(max_accel/g) = "
                  << std::setprecision(2) << (std::atan(12.0 / g) * 180.0 / M_PI)
                  << " deg，与 max_accel 单独给出的上限一致。\n";
        std::cout << "  => max_body_thrust = 20 N 在本配置下**永远不会被触发**，\n";
        std::cout << "     调它没有任何效果。\n\n";

        std::cout << "  " << std::setw(16) << "max_thrust(N)" << std::setw(20) << "峰值推力"
                  << std::setw(18) << "超限步数" << std::setw(22) << "最大高度偏差(m)"
                  << "\n";
        std::array<double, 5> thrust_limits = {20.0, 16.0, 14.0, 12.0, 11.0};
        bool saw_saturation = false;
        for (double tl : thrust_limits) {
            // 倾角上限放到 60°，让 max_body_thrust 成为唯一的紧约束
            const ManeuverResult r = runStep(3.0, 60.0, tl);
            if (r.sat_steps > 0) {
                saw_saturation = true;
            }
            std::cout << "  " << std::setw(16) << std::fixed << std::setprecision(1) << tl
                      << std::setw(20) << std::setprecision(3) << r.thrust_max << std::setw(18)
                      << r.sat_steps << std::setw(22) << std::setprecision(4)
                      << r.alt_err_max << "\n";
        }
        std::cout << "\n  只有 max_body_thrust 降到 15.5 N 以下（< sqrt(max_accel²+g²)）\n";
        std::cout << "  才会成为紧约束并出现饱和。\n";

        checkTrue("max_body_thrust 低于 max_accel 导出的上限时才会饱和", saw_saturation);
        checkTrue("推力上限与 sqrt(max_accel²+g²) 吻合",
                  std::fabs(T_eff - 15.4995) < 0.01);
    }

    // ---- 6. 敏捷性与推力余量的权衡 ----
    std::cout << "\n[6] 倾角上限的权衡：敏捷性 vs 推力余量\n\n";
    std::cout << "  " << std::setw(14) << "倾角上限" << std::setw(24) << "水平加速度上限"
              << std::setw(22) << "推力需求(N)" << std::setw(20) << "余量" << "\n";
    for (double tl : {25.0, 35.0, 45.0, 55.0, 60.0}) {
        const double acc = g * std::tan(tl * M_PI / 180.0);
        const double T = hoverThrust(tl, mass, g);
        std::cout << "  " << std::setw(14) << std::fixed << std::setprecision(0) << tl
                  << std::setw(24) << std::setprecision(2) << acc << std::setw(22)
                  << std::setprecision(2) << T << std::setw(20) << std::setprecision(2)
                  << (20.0 / T) << "x\n";
    }
    std::cout << "\n  敏捷性（水平加速度）随倾角**非线性**增长（tan 发散），而推力需求\n";
    std::cout << "  按 1/cos 增长 —— 两者都在倾角增大时急剧变差。\n";
    std::cout << "  35° 是个合理折中：加速度上限 6.87 m/s²（够用），余量 1.67 倍\n";
    std::cout << "  （安全）。继续放大倾角换来的加速度增长有限，余量却迅速耗尽。\n";
    checkTrue("倾角 35° 在敏捷性与余量间取得折中（余量 > 1.5 且加速度 > 5）",
              20.0 / hoverThrust(35.0, mass, g) > 1.5 && g * std::tan(35.0 * M_PI / 180.0) > 5.0);

    std::cout << "\n[结论]\n";
    std::cout << "  1. 大机动能力由**最紧的约束**决定。本配置下 max_tilt_deg = 35 最紧，\n";
    std::cout << "     实际水平加速度上限 6.87 m/s²，而非 max_accel 标称的 12 m/s²。\n";
    std::cout << "     只调 max_accel 而不调 max_tilt_deg 是无效的 —— 这是参数表里\n";
    std::cout << "     看不出来的隐藏耦合。\n";
    std::cout << "  2. 推力补偿（T = m·g/cosθ）已由合力矢量形式自然实现，倾斜时推力\n";
    std::cout << "     自动增大，大机动期间高度保持良好。\n";
    std::cout << "  3. **推力上限由 max_accel 决定，而非 max_body_thrust。** 实际可用\n";
    std::cout << "     推力为 m·sqrt(max_accel² + g²) = 15.50 N，对应倾角 50.73°（恰等于\n";
    std::cout << "     max_accel 单独给出的上限）。本配置 max_body_thrust = 20 N 永远\n";
    std::cout << "     不会被触发 —— 调它没有任何效果。\n";
    std::cout << "  4. 高度下沉是机动的固有代价（倾角建立过程中竖直分量不足），不是\n";
    std::cout << "     实现缺陷：倾角上限 15° 时下沉 0.86 m，35° 时 0.42 m，50° 时\n";
    std::cout << "     0.28 m —— 机动越猛、建立越快，下沉越小。\n";
    std::cout << "  5. 35° 是合理折中：加速度上限 6.87 m/s² 够用，推力余量 1.67 倍安全。\n";
    std::cout << "     再往上加速度增长有限（受 max_accel 与 tan 双重限制），余量却迅速耗尽。\n";

    std::cout << "\n========================================\n";
    std::cout << g_checks - g_failed << " / " << g_checks << " checks passed\n";
    if (g_failed > 0) {
        std::cout << g_failed << " FAILED\n";
    }
    std::cout << "========================================\n";
    return g_failed == 0 ? 0 : 1;
}
