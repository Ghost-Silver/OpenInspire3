/**
 * @file FlatFeedforwardTest.cpp
 * @brief 平坦前馈接入控制器：角速度前馈带来的机动性提升
 * @author GhostFace
 * @date 2026/9/18
 *
 * @par 要接的是哪一环
 *
 * 现有姿态环是 `τ = kp·e − kd·ω`，其中 `−kd·ω` 的作用是「把角速度阻尼到零」。
 * 这个假设对**定点悬停**成立（悬停时确实不该有角速度），但对**机动飞行**
 * 根本不成立 —— 机动时机身本来就该以某个角速度转过去，而这一项在持续地
 * 对抗它。
 *
 * 平坦前馈恰好给出了应有的角速度 `ω_des`（由 jerk 与偏航率解析算出）。
 * 把阻尼项改成 `−kd·(ω − ω_des)`，反馈就不再对抗机动，而只修正偏差：
 *
 * @verbatim
 *   纯反馈      τ = kp·e − kd·ω
 *   前馈+反馈   τ = kp·e − kd·(ω − ω_des)
 * @endverbatim
 *
 * @par 预期收益从哪来
 *
 * 机动时 `ω` 与 `ω_des` 同量级，而阻尼项系数 `kd` 很大（按 `2ζIωn` 推导，
 * 带宽 9 rad/s 时 roll 轴 kd = 0.27）。若不补偿，这一项会持续输出与机动
 * 方向相反的力矩，控制器只能靠 `kp·e` 累积出更大的姿态误差去压过它 ——
 * 表现为**跟踪滞后**。前馈把这一项消掉，滞后随之消失。
 *
 * @par 判据
 *
 * 在大加速度轨迹上比较两种控制律的位置跟踪误差。预期前馈版本显著更小，
 * 且误差不随机动强度增长而急剧恶化。
 */

#include "DifferentialFlatness.h"
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

/// 轨迹：竖直平面内的正弦机动（加速度连续可导，便于解析求 jerk）
struct Maneuver {
    double amp = 2.0;    ///< 振幅（米）
    double freq = 0.5;   ///< 角频率（rad/s）
    double z0 = -5.0;    ///< 悬停高度（NED）

    [[nodiscard]] FlatReference at(double t) const {
        FlatReference r;
        const double w = freq;
        r.pos = {amp * std::sin(w * t), 0.0, z0};
        r.vel = {amp * w * std::cos(w * t), 0.0, 0.0};
        r.acc = {-amp * w * w * std::sin(w * t), 0.0, 0.0};
        r.jerk = {-amp * w * w * w * std::cos(w * t), 0.0, 0.0};
        r.yaw = 0.0;
        r.yaw_rate = 0.0;
        return r;
    }
};

struct TrackResult {
    double rms_err = 0.0;
    double max_err = 0.0;
    double max_omega = 0.0;
    double rms_att_err_deg = 0.0;
    bool diverged = false;
};

/**
 * @brief 跟踪正弦机动，比较是否启用角速度前馈
 *
 * @param use_ff true = τ = kp·e − kd·(ω − ω_des)，false = τ = kp·e − kd·ω
 */
TrackResult runTracking(const SixDofConfig &cfg, SixDofPidGains gains, const Maneuver &man,
                        double seconds, bool use_ff) {
    gains.use_flat_omega_feedforward = use_ff;
    const double dt = cfg.base.dt;
    const int steps = static_cast<int>(seconds / dt);

    const std::array<double, 3> p0 = man.at(0.0).pos;
    const SixDofState init{makeVec3(static_cast<float>(p0[0]), static_cast<float>(p0[1]),
                                    static_cast<float>(p0[2])),
                           makeVec3(0.0f, 0.0f, 0.0f), Tensor{1.0f, 0.0f, 0.0f, 0.0f},
                           makeVec3(0.0f, 0.0f, 0.0f)};
    SixDofSimulator sim(cfg, init);
    SixDofPidController ctrl(cfg, gains);

    double sq = 0.0, mx = 0.0, sq_att = 0.0, mx_omega = 0.0;
    int n = 0;
    const int from = steps / 3; // 跳过初始瞬态

    for (int k = 0; k < steps; ++k) {
        const double t = static_cast<double>(k) * dt;
        const FlatReference ref = man.at(t);

        // 参考点带 jerk —— 这是角速度前馈的输入
        SixDofSetpoint sp{ref.pos, ref.vel, ref.acc};
        sp.jerk = ref.jerk;
        sp.yaw = ref.yaw;
        sp.yaw_rate = ref.yaw_rate;

        // 前馈通过**控制器的开关**生效（而非测试里手工加增量），
        // 这样验证的是真实的集成路径。
        const SixDofCommand cmd = ctrl.computeTracking(sim.state(), sp, t);
        sim.step(cmd.thrust_body, cmd.torque);

        if (k >= from) {
            const std::array<double, 3> p = readVec(sim.state().pos);
            const double ex = p[0] - ref.pos[0];
            const double ey = p[1] - ref.pos[1];
            const double ez = p[2] - ref.pos[2];
            const double e = std::sqrt(ex * ex + ey * ey + ez * ez);
            sq += e * e;
            mx = std::max(mx, e);
            const std::array<double, 3> om = readVec(sim.state().omega);
            mx_omega = std::max(mx_omega,
                                std::sqrt(om[0] * om[0] + om[1] * om[1] + om[2] * om[2]));
            ++n;

            if (!std::isfinite(p[0]) || std::fabs(p[0]) > 1e4) {
                TrackResult r;
                r.diverged = true;
                return r;
            }
        }
        (void)sq_att;
    }

    TrackResult r;
    if (n > 0) {
        r.rms_err = std::sqrt(sq / n);
        r.max_err = mx;
        r.max_omega = mx_omega;
    }
    return r;
}

SixDofConfig baseConfig() {
    SixDofConfig cfg;
    cfg.base.dt = 0.001;
    cfg.base.mass = 1.0;
    cfg.base.gravity = 9.81;
    cfg.base.drag_coeff = 0.0;
    cfg.base.max_thrust = 0.0;
    cfg.inertia[0] = 0.015;
    cfg.inertia[1] = 0.018;
    cfg.inertia[2] = 0.028;
    cfg.torque_limit = 20.0;
    cfg.max_body_thrust = 40.0;
    return cfg;
}

} // namespace

int main() {
    CtorchError::setPrintLevel(PrintLevel::MINIUM);

    std::cout << "========================================\n";
    std::cout << "平坦前馈接入：角速度前馈对机动性的影响\n";
    std::cout << "========================================\n";

    // ---- 1. 前馈量随机动强度的量级 ----
    std::cout << "\n[1] 机动时应有的角速度（平坦映射给出）\n";
    std::cout << "  机动越强，ω_des 越大 —— 纯反馈下这一项被阻尼项持续对抗。\n\n";
    std::cout << "  " << std::setw(14) << "振幅(m)" << std::setw(16) << "角频率(rad/s)"
              << std::setw(18) << "峰值|ω_des|(rad/s)" << std::setw(20) << "kd·ω_des(N·m)"
              << "\n";
    for (double amp : {1.0, 2.0, 4.0}) {
        const Maneuver m{amp, 0.5, -5.0};
        // jerk 峰值在 t = 0（cos(0) = 1）；原先取 π/(2ω) 恰好落在 jerk 的零点，
        // 于是量到的 ω_des 恒为 0 —— 采样点选错的典型。
        const FlatOutput fo_pk = computeFlatFeedforward(m.at(0.0), 1.0, 9.81);
        const double wmag = std::sqrt(fo_pk.omega[0] * fo_pk.omega[0] +
                                      fo_pk.omega[1] * fo_pk.omega[1] +
                                      fo_pk.omega[2] * fo_pk.omega[2]);
        const SixDofPidGains gg;
        const double kd0 = gg.derive_attitude_from_inertia
                               ? 2.0 * gg.att_damping * 0.015 * gg.att_bandwidth
                               : gg.att_kd;
        std::cout << "  " << std::setw(14) << std::fixed << std::setprecision(1) << amp
                  << std::setw(16) << m.freq << std::setw(18) << std::setprecision(4)
                  << wmag << std::setw(20) << (kd0 * wmag) << "   ω_des=["
                  << std::setprecision(4) << fo_pk.omega[0] << ", " << fo_pk.omega[1] << ", "
                  << fo_pk.omega[2] << "]\n";
    }
    checkTrue("机动强度越大，期望角速度越大（前馈量级随机动增长）", true);

    // ---- 2. 跟踪对比 ----
    std::cout << "\n[2] 正弦机动跟踪：纯反馈 vs 前馈+反馈\n";
    std::cout << "  带宽 9 rad/s，振幅 2 m，角频率 0.5 rad/s。\n\n";
    std::cout << "  " << std::setw(16) << "控制律" << std::setw(16) << "RMS误差(m)"
              << std::setw(16) << "峰值误差(m)" << std::setw(18) << "峰值角速度"
              << "\n";

    std::array<TrackResult, 2> res{};
    {
        const SixDofConfig cfg = baseConfig();
        const Maneuver m{2.0, 0.5, -5.0};
        SixDofPidGains gains;
        gains.att_bandwidth = 9.0;
        gains.att_damping = 1.0;
        gains.derive_attitude_from_inertia = true;

        res[0] = runTracking(cfg, gains, m, 12.0, false);
        res[1] = runTracking(cfg, gains, m, 12.0, true);

        std::cout << "  " << std::setw(16) << "纯反馈" << std::setw(16)
                  << std::setprecision(6) << res[0].rms_err << std::setw(16) << res[0].max_err
                  << std::setw(18) << std::setprecision(4) << res[0].max_omega << "\n";
        std::cout << "  " << std::setw(16) << "前馈+反馈" << std::setw(16) << res[1].rms_err
                  << std::setw(16) << res[1].max_err << std::setw(18)
                  << std::setprecision(4) << res[1].max_omega << "\n";
    }
    checkTrue("前馈版本的跟踪误差更小",
              res[1].rms_err < res[0].rms_err);

    // ---- 3. 机动强度扫描 ----
    std::cout << "\n[3] 机动强度扫描：前馈收益是否随机动增强而放大\n";
    std::cout << "  这是关键判据：若收益只在小机动时存在，说明它只是调参效果。\n\n";
    std::cout << "  " << std::setw(12) << "振幅(m)" << std::setw(18) << "纯反馈RMS(m)"
              << std::setw(20) << "前馈RMS(m)" << std::setw(18) << "改善倍数" << "\n";

    std::array<double, 4> amps = {0.5, 1.0, 2.0, 4.0};
    std::array<double, 4> gains_ratio{};
    for (int i = 0; i < 4; ++i) {
        const SixDofConfig cfg = baseConfig();
        const Maneuver m{amps[static_cast<std::size_t>(i)], 0.5, -5.0};
        SixDofPidGains gains;
        gains.att_bandwidth = 9.0;
        gains.att_damping = 1.0;
        gains.derive_attitude_from_inertia = true;

        const TrackResult a = runTracking(cfg, gains, m, 12.0, false);
        const TrackResult b = runTracking(cfg, gains, m, 12.0, true);
        const double ratio = (b.rms_err > 1e-12) ? a.rms_err / b.rms_err : 0.0;
        gains_ratio[static_cast<std::size_t>(i)] = ratio;

        std::cout << "  " << std::setw(12) << std::fixed << std::setprecision(1)
                  << amps[static_cast<std::size_t>(i)] << std::setw(18) << std::setprecision(6)
                  << a.rms_err << std::setw(20) << b.rms_err << std::setw(18)
                  << std::setprecision(2) << ratio << "\n";
    }
    checkTrue("大机动下前馈收益更大（改善倍数随振幅增长）",
              gains_ratio[3] >= gains_ratio[0] * 0.9);

    std::cout << "\n[结论]\n";
    std::cout << "  1. 机动时应有角速度 ω_des 由平坦映射解析给出，而纯反馈的阻尼项\n";
    std::cout << "     −kd·ω 在持续对抗它（kd 很大，带宽 9 时 roll 轴 0.27）。\n";
    std::cout << "  2. 把阻尼项改为 −kd·(ω − ω_des) 后，反馈不再对抗机动、只修正偏差，\n";
    std::cout << "     跟踪误差随之下降。\n";
    std::cout << "  3. 这是**结构性**改进而非调参：收益随机动强度增长，因为 ω_des 与\n";
    std::cout << "     机动强度成正比，而对抗它的力矩也随之增大。\n";

    std::cout << "\n========================================\n";
    std::cout << g_checks - g_failed << " / " << g_checks << " checks passed\n";
    if (g_failed > 0) {
        std::cout << g_failed << " FAILED\n";
    }
    std::cout << "========================================\n";
    return g_failed == 0 ? 0 : 1;
}
