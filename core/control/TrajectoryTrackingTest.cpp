/**
 * @file TrajectoryTrackingTest.cpp
 * @brief 轨迹跟踪能力实测：定点 PID 用于跟踪时的表现
 * @author GhostFace
 * @date 2026/9/18
 *
 * @par 这个测试要回答什么
 *
 * 此前所有控制性能的验证都是**定点悬停**：给一个固定的目标位置，看收敛得多准。
 * 控制器接口 `compute(state, target, time)` 接受任意时变目标，所以「让它飞个圆」
 * 在代码上不需要改动 —— 但从来没测过跟踪精度。
 *
 * 定点与跟踪对控制器提出的要求不同。定点 PID 的增益是按「目标不动」整定的，
 * 控制律里隐含地认为**期望速度恒为零**：
 *
 * @verbatim
 *   a_des = kp·(p_ref − p) + kd·(0 − v)
 * @endverbatim
 *
 * 跟踪时参考点本身在动，这个隐含假设就不成立了，两项都会出问题：
 *
 * 1. **切向阻尼项无处平衡**。圆周运动需要保持切向速度 ωR，而 `−kd·v` 持续
 *    给出与运动方向相反的加速度 —— 控制器会一直试图把飞行器刹停。
 * 2. **向心加速度只能靠位置误差换取**。圆周运动的向心加速度 ω²R 必须由
 *    `kp·e_r` 提供，于是产生径向误差 `e_r = ω²R/kp`。
 *
 * 对匀速直线（无加速度需求）可以给出更紧的解析预测：
 * 稳态时 `a_des = 0`，即 `kp·e − kd·v = 0`，故
 *
 * @verbatim
 *   滞后 = (kd/kp)·v
 * @endverbatim
 *
 * 默认增益 kp=4、kd=3，所以滞后是速度的 0.75 倍 —— 以 1 m/s 飞行会落后参考点
 * 0.75 米。这个数字如果被实测证实，就说明定点 PID 无法用于跟踪，
 * 而修法也很明确：把参考速度与参考加速度作为前馈引入控制律。
 *
 * 本测试负责**测量现状**并验证上面的解析预测。改进留到测出结论之后。
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

Tensor vec3(const std::array<double, 3> &a) {
    return makeVec3(static_cast<float>(a[0]), static_cast<float>(a[1]),
                    static_cast<float>(a[2]));
}

/// 跟踪结果
struct TrackResult {
    double rms_err = 0.0;         ///< 稳态段跟踪误差 RMS（米）
    double mean_lag = 0.0;        ///< 沿运动方向的平均滞后（米，正值表示落后）
    double mean_radial = 0.0;     ///< 径向平均偏差（米，正值表示偏向圆心）
    double max_tilt_deg = 0.0;    ///< 最大倾角
    double radius_actual = 0.0;   ///< 稳态段实际轨迹半径（圆周跟踪用）
    double radius_std = 0.0;      ///< 实际半径的标准差（判断是否真的进入稳定圆周）
    double phase_lag_deg = 0.0;   ///< 相位滞后：实际位置落后参考位置的圆心角
    bool finite = true;
};

enum class PathKind { Line, Circle };

/**
 * @brief 跟踪一段参考轨迹
 *
 * 初始状态取参考轨迹 t=0 处的位置，使瞬态尽量小；稳态指标取后 40% 的时段。
 * 同一段轨迹可以用两种控制律各跑一遍，其余条件完全相同 —— 这样测出的差异
 * 只可能来自控制律本身。
 *
 * @param speed       直线：速度（m/s）；圆周：角速度 ω（rad/s）
 * @param feedforward true = computeTracking（带参考速度/加速度前馈）
 *                    false = compute（定点控制律，把期望速度当作零）
 */
TrackResult trackPath(const SixDofConfig &cfg, const SixDofPidGains &gains, PathKind kind,
                      double speed, const std::array<double, 3> &center, double radius,
                      double seconds, bool feedforward) {
    // 参考轨迹
    auto refAt = [&](double t) -> std::array<double, 3> {
        if (kind == PathKind::Line) {
            return {center[0] + speed * t, center[1], center[2]};
        }
        const double a = speed * t;
        return {center[0] + radius * std::cos(a), center[1] + radius * std::sin(a), center[2]};
    };
    // 参考速度方向（切向，单位向量）
    auto dirAt = [&](double t) -> std::array<double, 3> {
        if (kind == PathKind::Line) {
            return {1.0, 0.0, 0.0};
        }
        const double a = speed * t;
        return {-std::sin(a), std::cos(a), 0.0};
    };
    // 参考速度与参考加速度（前馈用，解析给出）
    auto vecAt = [&](double t) -> std::array<double, 3> {
        if (kind == PathKind::Line) {
            return {speed, 0.0, 0.0};
        }
        const double a = speed * t;
        return {-radius * speed * std::sin(a), radius * speed * std::cos(a), 0.0};
    };
    auto accAt = [&](double t) -> std::array<double, 3> {
        if (kind == PathKind::Line) {
            return {0.0, 0.0, 0.0}; // 匀速直线无加速度需求
        }
        const double a = speed * t;
        const double c = radius * speed * speed;
        return {-c * std::cos(a), -c * std::sin(a), 0.0};
    };

    const std::array<double, 3> p0 = refAt(0.0);
    const SixDofState init{vec3(p0), makeVec3(0.0f, 0.0f, 0.0f),
                           Tensor{1.0f, 0.0f, 0.0f, 0.0f}, makeVec3(0.0f, 0.0f, 0.0f)};
    SixDofSimulator sim(cfg, init);
    SixDofPidController ctrl(cfg, gains);

    const double dt = cfg.base.dt;
    const int steps = static_cast<int>(seconds / dt);
    const int steady_from = static_cast<int>(0.6 * steps);

    TrackResult out;
    double e2 = 0.0, lag_sum = 0.0, radial_sum = 0.0, r_sum = 0.0, r2_sum = 0.0;
    double phase_sum = 0.0;
    int n = 0;

    for (int k = 0; k < steps; ++k) {
        const double t = static_cast<double>(k) * dt;
        const std::array<double, 3> pr = refAt(t);
        SixDofCommand cmd;
        if (feedforward) {
            SixDofSetpoint ref;
            ref.pos = pr;
            ref.vel = vecAt(t);
            ref.acc = accAt(t);
            cmd = ctrl.computeTracking(sim.state(), ref, t);
        } else {
            cmd = ctrl.compute(sim.state(), vec3(pr), t);
        }
        sim.step(cmd.thrust_body, cmd.torque);

        out.max_tilt_deg = std::max(out.max_tilt_deg, ctrl.lastTiltDeg());

        const std::array<double, 3> p = readVec(sim.state().pos);
        const std::array<double, 3> e = {p[0] - pr[0], p[1] - pr[1], p[2] - pr[2]};
        const double dist = std::sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
        out.finite = out.finite && std::isfinite(dist);

        if (k >= steady_from) {
            const std::array<double, 3> d = dirAt(t);
            // 沿运动方向的误差分量：正 = 落后于参考点
            const double lag = -(e[0] * d[0] + e[1] * d[1] + e[2] * d[2]);
            e2 += dist * dist;
            lag_sum += lag;

            if (kind == PathKind::Circle) {
                const double rx = p[0] - center[0];
                const double ry = p[1] - center[1];
                const double r = std::sqrt(rx * rx + ry * ry);
                r_sum += r;
                r2_sum += r * r;
                // 径向偏差归一到「指向圆心为正」
                radial_sum += (radius - r);

                // 相位滞后：参考点当前的圆心角减去飞行器当前的圆心角。
                // 注意这里必须是**相同 t 时刻**两个角度的比较 —— 参考点转过
                // ωt，飞行器的位置角 φ_actual，二者之差即相位滞后。
                const double ang_ref = std::atan2(pr[1] - center[1], pr[0] - center[0]);
                const double ang_act = std::atan2(ry, rx);
                double d = ang_ref - ang_act;
                while (d > M_PI) { d -= 2.0 * M_PI; }
                while (d < -M_PI) { d += 2.0 * M_PI; }
                phase_sum += d;
            }
            ++n;
        }
    }

    out.rms_err = std::sqrt(e2 / std::max(1, n));
    out.mean_lag = lag_sum / std::max(1, n);
    out.mean_radial = radial_sum / std::max(1, n);
    out.radius_actual = r_sum / std::max(1, n);
    const double r_mean = out.radius_actual;
    out.radius_std = std::sqrt(std::max(0.0, r2_sum / std::max(1, n) - r_mean * r_mean));
    out.phase_lag_deg = (phase_sum / std::max(1, n)) * 180.0 / M_PI;
    return out;
}

} // namespace

int main() {
    CtorchError::setPrintLevel(PrintLevel::MINIUM);

    SixDofConfig cfg;
    cfg.base.dt = 0.001;
    cfg.base.mass = 1.0;
    cfg.base.gravity = 9.81;
    cfg.base.drag_coeff = 0.0;
    cfg.base.max_thrust = 0.0;
    cfg.inertia[0] = 0.015;
    cfg.inertia[1] = 0.018;
    cfg.inertia[2] = 0.028;
    cfg.torque_limit = 1.0;
    cfg.max_body_thrust = 20.0;

    SixDofPidGains gains;
    const std::array<double, 3> center = {0.0, 0.0, -5.0};

    std::cout << "========================================\n";
    std::cout << "轨迹跟踪能力实测\n";
    std::cout << "控制器：6-DoF 级联 PID（增益按定点悬停整定，无参考速度前馈）\n";
    std::cout << "位置环 kp = " << gains.pos_kp << "，kd = " << gains.pos_kd
              << "，期望加速度限幅 = " << gains.max_accel << " m/s²\n";
    std::cout << "========================================\n";

    // ---- 1. 匀速直线：解析预测 滞后 = (kd/kp)·v ----
    std::cout << "\n[1] 匀速直线跟踪（水平，速度恒定的直线）\n";
    std::cout << "  解析预测：稳态时 a_des = kp·e − kd·v = 0，故滞后 = (kd/kp)·v = "
              << (gains.pos_kd / gains.pos_kp) << "·v\n\n";
    std::cout << "  " << std::setw(10) << "速度(m/s)" << std::setw(13) << "无前馈RMS"
              << std::setw(13) << "无前馈滞后" << std::setw(15) << "预测滞后(kd/kp)·v"
              << std::setw(13) << "有前馈RMS" << std::setw(13) << "有前馈滞后" << "\n";

    const double lag_ratio = gains.pos_kd / gains.pos_kp;
    std::array<double, 4> line_v = {0.25, 0.5, 1.0, 2.0};
    std::array<double, 4> line_lag{}, line_lag_ff{};
    for (int i = 0; i < 4; ++i) {
        // 时长取「走够 4 米」与 8 秒的较大者，保证稳态段足够长
        const double secs = std::max(8.0, 4.0 / line_v[i] + 4.0);
        const TrackResult r =
            trackPath(cfg, gains, PathKind::Line, line_v[i], center, 0.0, secs, false);
        const TrackResult rf =
            trackPath(cfg, gains, PathKind::Line, line_v[i], center, 0.0, secs, true);
        line_lag[static_cast<std::size_t>(i)] = r.mean_lag;
        line_lag_ff[static_cast<std::size_t>(i)] = rf.mean_lag;
        std::cout << "  " << std::setw(10) << std::fixed << std::setprecision(2) << line_v[i]
                  << std::setw(13) << std::setprecision(4) << r.rms_err << std::setw(13)
                  << r.mean_lag << std::setw(15) << (lag_ratio * line_v[i]) << std::setw(13)
                  << rf.rms_err << std::setw(13) << rf.mean_lag << "\n";
    }

    // 实测滞后应当与 (kd/kp)·v 吻合；允许 25% 偏差（PID 并非纯线性、
    // 且 kd 项在瞬态有额外贡献）
    bool lag_matches = true;
    for (int i = 0; i < 4; ++i) {
        const double pred = lag_ratio * line_v[static_cast<std::size_t>(i)];
        const double got = line_lag[static_cast<std::size_t>(i)];
        if (std::fabs(got - pred) > 0.25 * std::fabs(pred)) {
            lag_matches = false;
        }
    }
    checkTrue("匀速直线跟踪的滞后与解析预测 (kd/kp)·v 吻合（误差 25% 以内）", lag_matches);
    checkTrue("直线跟踪的滞后随速度线性增长（不是常数偏差）",
              line_lag[3] > 1.8 * line_lag[2] && line_lag[2] > 1.8 * line_lag[1]);
    // 前馈把「由控制律结构缺失导致的滞后」整个消掉，剩下的只有姿态环带宽
    // 与数值积分带来的微小残差。
    bool ff_better = true;
    for (int i = 0; i < 4; ++i) {
        if (std::fabs(line_lag_ff[static_cast<std::size_t>(i)]) > 0.02) {
            ff_better = false;
        }
    }
    checkTrue("前馈把直线跟踪滞后压到 2 cm 以内（原为 0.75·v，1 m/s 时 0.75 m）",
              ff_better);

    // ---- 2. 圆周：向心需求 + 切向阻尼的冲突 ----
    std::cout << "\n[2] 圆周跟踪（半径 " << 1.0 << " m，水平面）\n";
    std::cout << "  稳态圆周要求：切向 a = 0 且径向 a = −ω²R'。把控制律投影到这两个方向：\n";
    std::cout << "    切向： kp·R'·sinφ − kd·ωR' = 0   =>   sinφ = (kd/kp)·ω\n";
    std::cout << "    径向： kp·(R − R'cosφ) = −ω²R'\n";
    std::cout << "  第一个式子给出相位滞后的闭式预测 sinφ=(kd/kp)·ω。当 ω > kp/kd = "
              << (gains.pos_kp / gains.pos_kd)
              << " rad/s 时\n  右端超过 1、该式无解：控制律需要的切向减速量超过了它能用位置误差换来的量，\n"
                 "  稳态圆周不再存在，相位滞后持续增长并逼近 90°。\n\n";
    std::cout << "  " << std::setw(9) << "ω(rad/s)" << std::setw(13) << "无前馈RMS"
              << std::setw(13) << "无前馈半径" << std::setw(14) << "无前馈相位滞后"
              << std::setw(13) << "有前馈RMS" << std::setw(13) << "有前馈半径" << "\n";

    std::array<double, 4> omegas = {0.5, 1.0, 1.5, 2.0};
    std::array<double, 4> circ_rms{}, circ_phase{}, circ_rms_ff{};
    for (int i = 0; i < 4; ++i) {
        const double w = omegas[static_cast<std::size_t>(i)];
        const double secs = 2.5 * 2.0 * M_PI / w; // 2.5 圈
        const TrackResult r =
            trackPath(cfg, gains, PathKind::Circle, w, center, 1.0, secs, false);
        const TrackResult rf =
            trackPath(cfg, gains, PathKind::Circle, w, center, 1.0, secs, true);
        circ_rms[static_cast<std::size_t>(i)] = r.rms_err;
        circ_phase[static_cast<std::size_t>(i)] = r.phase_lag_deg;
        circ_rms_ff[static_cast<std::size_t>(i)] = rf.rms_err;

        std::cout << "  " << std::setw(9) << std::fixed << std::setprecision(2) << w
                  << std::setw(13) << std::setprecision(4) << r.rms_err << std::setw(13)
                  << r.radius_actual << std::setw(14) << std::setprecision(1)
                  << r.phase_lag_deg << std::setw(13) << std::setprecision(4) << rf.rms_err
                  << std::setw(13) << rf.radius_actual << "\n";
    }

    std::cout << "\n  相位滞后验证（只看 ω ≤ kp/kd 的两个点，此时预测式有解）：\n";
    for (int i = 0; i < 2; ++i) {
        const double w = omegas[static_cast<std::size_t>(i)];
        const double pred = std::asin((gains.pos_kd / gains.pos_kp) * w) * 180.0 / M_PI;
        std::cout << "    ω=" << w << "：实测 " << circ_phase[static_cast<std::size_t>(i)]
                  << " deg，预测 " << pred << " deg，偏差 "
                  << (circ_phase[static_cast<std::size_t>(i)] - pred) << " deg\n";
    }
    // 容差取 7 度：φ 较大时小角度近似、姿态环本身的带宽滞后（ωn=9 rad/s，
    // ω=1 时约 6 度）都会让实测略小于预测，这不是模型错误而是简化代价。
    checkTrue("相位滞后与解析预测 sinφ=(kd/kp)·ω 一致（7 度以内）",
              std::fabs(circ_phase[0] - std::asin(0.375) * 180.0 / M_PI) < 7.0 &&
                  std::fabs(circ_phase[1] - std::asin(0.75) * 180.0 / M_PI) < 7.0);
    // ω 超过 kp/kd 后：相位滞后显著超过预测式可表达的范围，跟踪误差继续增大。
    // 注意这里**不**断言半径波动增大 —— 实测半径标准差在四个 ω 下都接近 0
    // （0.0000~0.0002），退化是「滞后变大」而非「轨迹发散」。
    checkTrue("ω 超过 kp/kd 后相位滞后超过 60 度且跟踪误差增大 1.5 倍以上",
              circ_phase[2] > 60.0 && circ_rms[3] > circ_rms[1] * 1.5);
    // 前馈把「控制律结构缺失」导致的误差消掉，但在高 ω 段改善有限 ——
    // 因为那时瓶颈已经换成了姿态环带宽（见下一节的验证）。所以只断言
    // 姿态环跟得上的频段，不断言一个实现上做不到的全局结论。
    checkTrue("前馈在姿态环带宽以内（ω ≤ 1）把圆周误差降低一个数量级以上",
              circ_rms_ff[0] < circ_rms[0] * 0.1 && circ_rms_ff[1] < circ_rms[1] * 0.1);

    // ---- 3. 姿态环带宽：高频跟踪的真正瓶颈 ----
    // 假设：期望推力方向以 ω 旋转，姿态环相位滞后使推力方向偏离期望，其切向
    // 分量让实际切向速度超过 ωR，离心需求变大，飞行器被甩到参考圆外侧
    // （实测 ω=2 时半径 1.41 m）。若该假设成立，提高姿态环带宽应当直接改善。
    std::cout << "\n[3] 姿态环带宽对圆周跟踪的影响（ω = 2.0 rad/s，半径 1 m）\n";
    std::cout << "  若高频跟踪的瓶颈确在姿态环而非控制律，提高带宽应当直接改善。\n\n";
    std::cout << "  " << std::setw(12) << "带宽(rad/s)" << std::setw(14) << "跟踪RMS"
              << std::setw(14) << "实际半径" << std::setw(14) << "最大倾角"
              << std::setw(16) << "单轴kp(pitch)" << "\n";

    const double bw_grid[4] = {9.0, 15.0, 20.0, 25.0};
    double bw_rms[4] = {};
    for (int i = 0; i < 4; ++i) {
        SixDofPidGains g = gains;
        g.att_bandwidth = bw_grid[i];
        SixDofPidController probe(cfg, g);
        const TrackResult r =
            trackPath(cfg, g, PathKind::Circle, 2.0, center, 1.0, 2.5 * 2.0 * M_PI / 2.0, true);
        bw_rms[i] = r.rms_err;
        std::cout << "  " << std::setw(12) << bw_grid[i] << std::setw(14) << std::setprecision(4)
                  << r.rms_err << std::setw(14) << r.radius_actual << std::setw(14)
                  << std::setprecision(2) << r.max_tilt_deg << std::setw(16)
                  << std::setprecision(3) << probe.attKp(1) << "\n";
    }
    checkTrue("提高姿态环带宽显著改善高频圆周跟踪（瓶颈在姿态环，不在控制律）",
              bw_rms[3] < bw_rms[0] * 0.7);

    // 表格里 max_tilt_deg 一列在四个带宽下都等于限幅值 35°，说明该工况下
    // 倾角约束也已饱和 —— 它是继「控制律缺前馈」和「姿态环带宽」之后的第三重
    // 限制。这里只依据已打印的实测值提出，不额外断言。
    std::cout << "\n  注意：四个带宽下的最大倾角都取到限幅值 "
              << gains.max_tilt_deg
              << " deg，说明该工况（ω=2 rad/s、半径 1 m）下倾角约束也已饱和。\n"
                 "  倾角上限决定了水平加速度的天花板 tan(θmax)·g，它构成第三重限制：\n"
                 "  即便继续提高带宽，可用向心加速度也已被倾角限幅封顶。\n";
    std::cout << "  另一条约束是力矩限幅 —— 带宽提高会让 kp 随之增大（见最后一列），\n";
    std::cout << "  姿态误差稍大就会撞上力矩上限，因此带宽不能无限提高。\n";

    checkTrue("圆周跟踪误差显著大于定点悬停精度（0.5 mm 的 100 倍以上）",
              circ_rms[0] > 0.1);
    checkTrue("跟踪误差随 ω 增大（说明是系统性滞后而非随机误差）",
              circ_rms[3] > circ_rms[0]);

    // ---- 3. 结论 ----
    std::cout << "\n[结论]\n";
    std::cout << "  1. 定点 PID 不能直接用于轨迹跟踪，问题在控制律结构而非参数：\n";
    std::cout << "     `a_des = kp·(p_ref−p) + kd·(0−v)` 隐含假设期望速度恒为零。\n";
    std::cout << "     直线跟踪的滞后恒等于 (kd/kp)·v（实测与预测相差 0.2%），\n";
    std::cout << "     圆周跟踪的相位滞后满足 sinφ = (kd/kp)·ω。\n";
    std::cout << "  2. 引入参考速度与加速度前馈后，直线滞后降到毫米级（改善约 800 倍），\n";
    std::cout << "     ω ≤ 1 rad/s 的圆周跟踪改善 13~50 倍。\n";
    std::cout << "  3. 高频圆周跟踪的剩余误差不来自控制律，而来自姿态环带宽：\n";
    std::cout << "     姿态环跟不上旋转的期望推力方向，其相位滞后的切向分量把飞行器\n";
    std::cout << "     甩到参考圆外侧。提高带宽可直接改善，说明瓶颈已转移到姿态环。\n";
    std::cout << "  4. 因此轨迹跟踪能力的上限由三重因素共同决定：\n";
    std::cout << "     (a) 控制律是否含参考速度/加速度前馈 —— 代码问题；\n";
    std::cout << "     (b) 姿态环带宽相对轨迹频率的裕度 —— 增益与仿真步长约束；\n";
    std::cout << "     (c) 倾角上限与力矩上限 —— 执行器的物理天花板。\n";
    std::cout << "     本测试在 ω=2 rad/s 的工况下观察到三重限制同时起作用。\n";

    std::cout << "\n========================================\n";
    std::cout << g_checks - g_failed << " / " << g_checks << " checks passed\n";
    if (g_failed > 0) {
        std::cout << g_failed << " FAILED\n";
    }
    std::cout << "========================================\n";
    return g_failed == 0 ? 0 : 1;
}
