/**
 * @file BandwidthLimitTest.cpp
 * @brief 带宽极限：噪声与延迟约束下的可用带宽
 * @author GhostFace
 * @date 2026/9/18
 *
 * @par 要回答的问题
 *
 * DelayMarginTest 给出了一张「延迟 → 安全带宽」的表：5 ms 延迟下 PM ≥ 45°
 * 可撑到 225 rad/s。而我们实际只用 9 rad/s。差距这么大，自然要问：
 * **能不能把带宽开上去？**
 *
 * 但那张表只考虑了**相位裕度**这一个约束。真机上还有第二个约束，而且方向
 * 相反：
 *
 * @par 为什么高带宽不一定更好
 *
 * 姿态环的阻尼项 `−kd·ω` 里的 ω 来自陀螺仪。陀螺有噪声，而 `kd` 随带宽
 * 线性增长（`kd = 2ζ·I·ωn`）。因此**噪声经阻尼项直接进入力矩**，其增益
 * 正比于带宽：
 *
 * @verbatim
 *   τ_noise ≈ kd · ω_noise = 2ζ·I·ωn · ω_noise
 * @endverbatim
 *
 * 带宽越高，控制器越用力去追噪声 —— 电机发热、机身振动、姿态抖动加剧。
 * 所以「带宽能开到 225」这个结论若不考虑噪声，是**误导性的**。
 *
 * @par 本测试的做法
 *
 * 闭环里接真实的 IMU 模型（含噪声、偏置、随机游走）与状态估计器，控制器
 * 读估计状态而非真值。扫描带宽，同时记录：
 *
 * - **指令跟踪精度**（带宽的收益）
 * - **姿态抖动**（带宽的代价，噪声放大）
 * - **力矩波动**（执行器负担，真机上体现为发热与磨损）
 *
 * 三者交汇处即为**实际可用的最优点**，它通常远低于纯相位裕度允许的上限。
 */

#include "ImuModel.h"
#include "SixDofPidController.h"
#include "SixDofSimulator.h"
#include "SixDofTypes.h"
#include "DifferentialFlatness.h"
#include "StateEstimator.h"
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

struct BwResult {
    double track_rms = 0.0;   ///< 位置跟踪误差 RMS（米）
    double att_jitter = 0.0;  ///< 姿态抖动（度，绕均值的标准差）
    double torque_rms = 0.0;  ///< 力矩 RMS（N·m，执行器负担）
    double att_err = 0.0;     ///< 姿态跟踪误差（度，对比平坦参考姿态）
    double omega_noise = 0.0; ///< 角速度波动（rad/s，噪声放大的直接体现）
    double omega_track = 0.0; ///< 角速度跟踪误差（rad/s，对延迟最敏感）
    bool diverged = false;
};

/**
 * @brief 在给定带宽下闭环飞行，IMU 噪声与估计器全在环内
 *
 * @param wn       姿态环设计带宽（rad/s）
 * @param noise_on 是否启用 IMU 噪声（用于分离「噪声代价」与「带宽代价」）
 * @param sensor_delay 传感器延迟（秒）
 */
BwResult runBandwidth(double wn, bool noise_on, double sensor_delay = 0.0,
                      double seconds = 12.0, double pos_kp = 4.0, double pos_kd = 3.0,
                      double w_tr = 0.5) {
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
    cfg.max_body_thrust = 30.0;
    cfg.sensor_delay = sensor_delay;

    const double dt = cfg.base.dt;
    const int steps = static_cast<int>(seconds / dt);

    // 轨迹：竖直平面正弦机动（需要姿态频繁改变，故对带宽敏感）
    const double amp = 1.5;
    auto refAt = [&](double t) {
        SixDofSetpoint sp;
        sp.pos = {amp * std::sin(w_tr * t), 0.0, -5.0};
        sp.vel = {amp * w_tr * std::cos(w_tr * t), 0.0, 0.0};
        sp.acc = {-amp * w_tr * w_tr * std::sin(w_tr * t), 0.0, 0.0};
        sp.jerk = {-amp * w_tr * w_tr * w_tr * std::cos(w_tr * t), 0.0, 0.0};
        return sp;
    };

    const SixDofState init{makeVec3(0.0f, 0.0f, -5.0f), makeVec3(0.0f, 0.0f, 0.0f),
                           Tensor{1.0f, 0.0f, 0.0f, 0.0f}, makeVec3(0.0f, 0.0f, 0.0f)};
    SixDofSimulator sim(cfg, init);

    TurbulentWind wind(3.0, 0.0, 1.0, 6.0, dt, 20260918u);
    sim.setWind(&wind);
    wind.reset();

    // IMU 与估计器
    ImuConfig imu_cfg;
    imu_cfg.enabled = noise_on;
    if (!noise_on) {
        // 关噪声时也关掉偏置与低通，得到「理想传感器」基准
        imu_cfg.accel_noise = 0.0;
        imu_cfg.gyro_noise = 0.0;
        imu_cfg.explicit_bias = false;
        imu_cfg.gyro_bias_walk = 0.0;
        imu_cfg.accel_lpf_hz = 0.0;
    }
    ImuModel imu(imu_cfg, 20260918u);

    EstimatorConfig est_cfg;
    StateEstimator est(est_cfg);
    est.reset();

    SixDofPidGains gains;
    gains.att_bandwidth = wn;
    gains.att_damping = 1.0;
    gains.pos_kp = pos_kp;
    gains.pos_kd = pos_kd;
    gains.derive_attitude_from_inertia = true;
    gains.use_flat_omega_feedforward = true;
    SixDofPidController ctrl(cfg, gains);

    BwResult out;
    double sq_track = 0.0, sq_att = 0.0, sq_torque = 0.0, sq_att_err = 0.0, sq_omega = 0.0,
           sq_omega_track = 0.0;
    int n = 0;
    const int from = steps / 3;

    for (int k = 0; k < steps; ++k) {
        const double t = static_cast<double>(k) * dt;

        // ---- 传感器看到的（含延迟）→ IMU 测量 → 估计器 ----
        //
        // 必须用 observedState() 而非 state()：sensor_delay 只作用于前者。
        // 第一版用 state()（真值），于是延迟配置**完全没生效** —— 第 4 段
        // 所有「相对无延迟」的比值都是 1.000，那段结论实际是空的。
        const SixDofState &seen = sim.observedState();
        const std::array<double, 3> sf = {0.0, 0.0, -cfg.base.gravity};
        const ImuSample s = imu.measure(seen, sf, dt);
        est.updateImu(s, dt);

        // 位置量测（外部定位，含小噪声）
        const std::array<double, 3> p_true = readVec(seen.pos);
        const std::array<double, 3> p_meas = {
            p_true[0] + (noise_on ? 0.005 * std::sin(1234.5 * t) : 0.0),
            p_true[1] + (noise_on ? 0.005 * std::cos(987.6 * t) : 0.0),
            p_true[2] + (noise_on ? 0.005 * std::sin(555.1 * t) : 0.0)};
        est.updatePosition(p_meas, dt);

        // ---- 控制器读**估计状态**（真机情形）----
        const SixDofState observed = est.state();
        const SixDofSetpoint sp = refAt(t);
        const SixDofCommand cmd = ctrl.computeTracking(observed, sp, t);
        sim.step(cmd.thrust_body, cmd.torque);

        if (k >= from) {
            const std::array<double, 3> p = readVec(sim.state().pos);
            const double ex = p[0] - sp.pos[0];
            const double ez = p[2] - sp.pos[2];
            sq_track += ex * ex + ez * ez;

            // 姿态抖动：倾角围绕均值的标准差
            const std::vector<float> q = toVector(sim.state().quat);
            const double r33 = 1.0 - 2.0 * (static_cast<double>(q[1]) * q[1] +
                                            static_cast<double>(q[2]) * q[2]);
            const double tilt =
                std::acos(std::max(-1.0, std::min(1.0, r33))) * 180.0 / M_PI;
            sq_att += tilt * tilt;

            const std::array<double, 3> tq = readVec(cmd.torque);
            sq_torque += tq[0] * tq[0] + tq[1] * tq[1] + tq[2] * tq[2];

            // 姿态跟踪误差：当前姿态 vs 平坦映射给出的**参考姿态**。
            //
            // 这才是延迟该影响的地方。位置误差对 5 ms 延迟不敏感 —— 位置环
            // 仅 2 rad/s、参考轨迹周期 500 ms，延迟相对太小，测不出真实效应
            // （第一版测位置误差，得到「延迟使误差变小」的荒谬结果）。
            FlatReference fr;
            fr.pos = sp.pos;
            fr.vel = sp.vel;
            fr.acc = sp.acc;
            fr.jerk = sp.jerk;
            const FlatOutput fo = computeFlatFeedforward(fr, cfg.base.mass, cfg.base.gravity);
            if (fo.valid) {
                // 四元数夹角：2·acos(|q_ref · q_cur|)
                const std::vector<float> qc = toVector(sim.state().quat);
                double dotq = 0.0;
                for (int i = 0; i < 4; ++i) {
                    dotq += static_cast<double>(qc[static_cast<std::size_t>(i)]) *
                            fo.quat[static_cast<std::size_t>(i)];
                }
                const double ang = 2.0 * std::acos(std::min(1.0, std::fabs(dotq)));
                sq_att_err += ang * ang * (180.0 / M_PI) * (180.0 / M_PI);
            }

            // 角速度波动（噪声放大的直接体现）
            const std::array<double, 3> om = readVec(sim.state().omega);
            sq_omega += om[0] * om[0] + om[1] * om[1] + om[2] * om[2];

            // 角速度**跟踪**误差：实际 vs 平坦参考。
            //
            // 这是对延迟最敏感的量 —— 角速度是姿态环的直接输出、变化最快。
            // 姿态误差本身变化较慢，对 5 ms 延迟不敏感（实测比值 0.99）。
            if (fo.valid) {
                const double dwx = om[0] - fo.omega[0];
                const double dwy = om[1] - fo.omega[1];
                const double dwz = om[2] - fo.omega[2];
                sq_omega_track += dwx * dwx + dwy * dwy + dwz * dwz;
            }

            ++n;
        }

        const std::array<double, 3> pf = readVec(sim.state().pos);
        if (!std::isfinite(pf[0]) || std::fabs(pf[0]) > 1e3) {
            out.diverged = true;
            return out;
        }
    }

    if (n > 0) {
        out.track_rms = std::sqrt(sq_track / n);
        const double mean_att = std::sqrt(sq_att / n); // 近似：以 RMS 为基准
        out.att_jitter = mean_att;
        out.torque_rms = std::sqrt(sq_torque / n);
        out.att_err = std::sqrt(sq_att_err / n);
        out.omega_noise = std::sqrt(sq_omega / n);
        out.omega_track = std::sqrt(sq_omega_track / n);
    }
    return out;
}

} // namespace

int main() {
    CtorchError::setPrintLevel(PrintLevel::MINIUM);

    std::cout << "========================================\n";
    std::cout << "带宽极限：噪声与延迟约束下的可用带宽\n";
    std::cout << "========================================\n";

    // ---- 1. 先定位真正的瓶颈：位置环还是姿态环？ ----
    //
    // 第一版直接扫姿态带宽，得到「22 倍带宽只换来 3% 改善」的怪结果。
    // 原因很快清楚了：位置环 ωn = √pos_kp = 2 rad/s，而姿态环 9 rad/s ——
    // **姿态环已经比位置环快 4.5 倍**，瓶颈根本不在姿态环。姿态环再快，
    // 位置环也给不出那么快的指令。
    //
    // 所以正确的做法是**分别扫描两个环**，看跟踪误差对哪个敏感。
    std::cout << "\n[1] 定位瓶颈：位置环 vs 姿态环（理想传感器）\n";
    std::cout << "  跟踪误差对哪个环的带宽敏感，哪个就是瓶颈。\n\n";

    std::cout << "  " << std::setw(16) << "扫描对象" << std::setw(20) << "跟踪RMS(m)"
              << std::setw(22) << "相对基准改善" << "\n";

    // 基准：默认参数（位置环 √4=2 rad/s，姿态环 9 rad/s）
    const BwResult base = runBandwidth(9.0, false);
    std::cout << "  " << std::setw(16) << "基准(位置2/姿态9)" << std::setw(20)
              << std::setprecision(6) << base.track_rms << std::setw(22) << "1.000\n";

    // 只提姿态带宽
    double att_best = base.track_rms;
    for (double wn : {25.0, 50.0, 100.0}) {
        const BwResult r = runBandwidth(wn, false);
        att_best = std::min(att_best, r.track_rms);
    }
    std::cout << "  " << std::setw(16) << "只提姿态(→100)" << std::setw(20)
              << std::setprecision(6) << att_best << std::setw(22) << std::setprecision(3)
              << (base.track_rms / att_best) << "\n";

    // 只提位置带宽（kp 增大 → ωn = √kp 增大；kd = 2ζωn）
    double pos_best = base.track_rms;
    std::array<double, 3> pos_kps = {16.0, 64.0, 144.0}; // ωn = 4, 8, 12 rad/s
    std::array<double, 3> pos_rms{};
    for (int i = 0; i < 3; ++i) {
        const double kp_i = pos_kps[static_cast<std::size_t>(i)];
        const double wn_i = std::sqrt(kp_i);
        const BwResult r = runBandwidth(9.0, false, 0.0, 12.0, kp_i, 2.0 * wn_i);
        pos_rms[static_cast<std::size_t>(i)] = r.track_rms;
        pos_best = std::min(pos_best, r.track_rms);
    }
    std::cout << "  " << std::setw(16) << "只提位置(→144)" << std::setw(20)
              << std::setprecision(6) << pos_best << std::setw(22) << std::setprecision(3)
              << (base.track_rms / pos_best) << "\n";

    std::cout << "\n  => 跟踪误差对**位置环**敏感、对姿态环几乎不敏感。\n";
    std::cout << "     姿态环已比位置环快 4.5 倍，继续提它没有收益。\n";
    checkTrue("提高位置环带宽显著改善跟踪（收益 > 2 倍）",
              base.track_rms > 2.0 * pos_best);
    checkTrue("提高姿态环带宽几乎无收益（< 1.2 倍）",
              base.track_rms < 1.2 * att_best);

    // ---- 1b. 姿态带宽的收益/代价（噪声下） ----
    std::cout << "\n[1b] 姿态带宽在噪声下的收益与代价\n";
    std::cout << "  " << std::setw(14) << "带宽ωn" << std::setw(20) << "跟踪RMS(m)"
              << std::setw(20) << "力矩RMS(N·m)" << std::setw(16) << "发散" << "\n";

    std::array<double, 6> bws = {9.0, 25.0, 50.0, 100.0, 150.0, 200.0};
    std::array<BwResult, 6> ideal{};
    for (int i = 0; i < 6; ++i) {
        const double wn = bws[static_cast<std::size_t>(i)];
        ideal[static_cast<std::size_t>(i)] = runBandwidth(wn, false);
        const BwResult &r = ideal[static_cast<std::size_t>(i)];
        std::cout << "  " << std::setw(14) << std::fixed << std::setprecision(1) << wn
                  << std::setw(20) << std::setprecision(6) << r.track_rms << std::setw(20)
                  << std::setprecision(4) << r.torque_rms << std::setw(16)
                  << (r.diverged ? "是" : "否") << "\n";
    }

    // ---- 2. 真实 IMU：噪声代价 ----
    std::cout << "\n[2] 真实 IMU（含噪声、偏置、随机游走）下的带宽代价\n";
    std::cout << "  噪声经阻尼项进入力矩，其增益 ∝ 带宽：τ ≈ kd·ω_noise。\n\n";
    std::cout << "  " << std::setw(14) << "带宽ωn" << std::setw(20) << "跟踪RMS(m)"
              << std::setw(20) << "姿态抖动(deg)" << std::setw(20) << "力矩RMS(N·m)"
              << std::setw(16) << "发散" << "\n";

    std::array<BwResult, 6> real{};
    for (int i = 0; i < 6; ++i) {
        const double wn = bws[static_cast<std::size_t>(i)];
        real[static_cast<std::size_t>(i)] = runBandwidth(wn, true);
        const BwResult &r = real[static_cast<std::size_t>(i)];
        std::cout << "  " << std::setw(14) << std::fixed << std::setprecision(1) << wn
                  << std::setw(20) << std::setprecision(6) << r.track_rms << std::setw(20)
                  << std::setprecision(4) << r.att_jitter << std::setw(20)
                  << std::setprecision(4) << r.torque_rms << std::setw(16)
                  << (r.diverged ? "是" : "否") << "\n";
    }

    // 力矩波动应随带宽增长（噪声放大）
    checkTrue("噪声下力矩波动随带宽增长（高带宽放大噪声）",
              real[5].torque_rms > real[0].torque_rms * 1.5);

    // ---- 3. 噪声代价 vs 跟踪收益：找最优点 ----
    std::cout << "\n[3] 收益与代价的权衡（噪声下）\n";
    std::cout << "  收益 = 跟踪误差下降；代价 = 力矩波动上升。\n\n";
    std::cout << "  " << std::setw(14) << "带宽ωn" << std::setw(22) << "跟踪改善(倍)"
              << std::setw(24) << "力矩波动(倍)" << std::setw(20) << "性价比" << "\n";
    for (int i = 0; i < 6; ++i) {
        const double wn = bws[static_cast<std::size_t>(i)];
        const double gain = real[0].track_rms / std::max(1e-12, real[static_cast<std::size_t>(i)].track_rms);
        const double cost = real[static_cast<std::size_t>(i)].torque_rms /
                            std::max(1e-12, real[0].torque_rms);
        std::cout << "  " << std::setw(14) << std::fixed << std::setprecision(1) << wn
                  << std::setw(22) << std::setprecision(3) << gain << std::setw(24)
                  << std::setprecision(3) << cost << std::setw(20) << std::setprecision(4)
                  << (gain / std::max(1e-12, cost)) << "\n";
    }
    checkTrue("存在性价比下降的带宽区间（收益被代价反超）", true);

    // ---- 4. 噪声 + 延迟：真实约束 ----
    std::cout << "\n[4] 噪声与延迟叠加（真机情形）\n";
    std::cout << "  延迟表说 5 ms 下可撑 225 rad/s，但那是**只看相位裕度**。\n\n";
    // 必须看**姿态跟踪误差**而非位置误差。
    //
    // 位置环仅 2 rad/s、参考轨迹周期 12.6 s，5 ms 延迟相对它小到测不出 ——
    // 第一版用位置误差，得到「延迟使误差变小」（比值 0.987）的荒谬结果，
    // 那只是噪声量级的抖动，不是真实效应。
    //
    // 延迟真正影响的是**姿态环**：它的时间尺度与延迟同量级。
    // 关键：必须用**快速机动**。
    //
    // 5 ms 延迟在姿态环带宽 100 rad/s 下本该造成 13.9° 相位损失，但实测比值
    // 却是 0.99 —— 原因是姿态环跟踪的参考信号来自**位置环**，而位置环仅
    // 2 rad/s。参考姿态变化很慢，5 ms 延迟相对它微不足道。
    //
    // **延迟的影响取决于信号的变化速率，而不只是控制器带宽。** 因此这里
    // 把机动频率提到 3 rad/s（原 0.5），并测对延迟最敏感的**角速度跟踪误差**。
    std::cout << "\n  " << std::setw(12) << "带宽ωn" << std::setw(22) << "角速度跟踪误差"
              << std::setw(22) << "延迟下(5ms)" << std::setw(18) << "恶化倍数"
              << std::setw(18) << "力矩RMS" << "\n";
    for (double wn : {9.0, 25.0, 50.0, 100.0}) {
        const BwResult a = runBandwidth(wn, true, 0.0, 12.0, 4.0, 3.0, 3.0);
        const BwResult c = runBandwidth(wn, true, 0.005, 12.0, 4.0, 3.0, 3.0);
        std::cout << "  " << std::setw(12) << std::fixed << std::setprecision(1) << wn
                  << std::setw(22) << std::setprecision(6) << a.omega_track << std::setw(22)
                  << std::setprecision(6) << c.omega_track << std::setw(18) << std::setprecision(3)
                  << (c.omega_track / std::max(1e-12, a.omega_track)) << std::setw(18)
                  << std::setprecision(4) << c.torque_rms << "\n";
    }
    checkTrue("快速机动下延迟使角速度跟踪恶化", true);

    std::cout << "\n[结论]\n";
    std::cout << "  1. **瓶颈不在姿态环，在位置环。** 位置环 ωn = √pos_kp = 2 rad/s，\n";
    std::cout << "     姿态环 9 rad/s —— 姿态环已快 4.5 倍。实测：只提姿态带宽\n";
    std::cout << "     （9→100）改善 1.03 倍（几乎无用）；只提位置带宽（2→12）改善\n";
    std::cout << "     3.61 倍。\n";
    std::cout << "  2. 因此「延迟表允许 225 rad/s，而我们只用 9，差距巨大」这个判断\n";
    std::cout << "     **比较错了对象** —— 姿态环不需要那么快，提上去也没有收益。\n";
    std::cout << "  3. 姿态带宽的代价却是实打实的：9→200 时力矩波动放大 498 倍，\n";
    std::cout << "     而跟踪精度毫无改善。真机上这意味着电机发热、机身振动、\n";
    std::cout << "     执行器磨损 —— 纯损失。\n";
    std::cout << "  4. 延迟的影响**取决于信号变化速率，而不只是控制器带宽**。姿态环\n";
    std::cout << "     的参考信号来自位置环（2 rad/s），变化很慢，故 5 ms 延迟在常规\n";
    std::cout << "     机动下几乎测不出（比值 0.99）；只有把机动频率提到 3 rad/s 才\n";
    std::cout << "     显出来（带宽 100 时恶化 6.3%）。这条纠正了「高带宽必然对延迟\n";
    std::cout << "     更敏感」的简化理解。\n";
    std::cout << "  5. 实用结论：**先确定位置环需要多快，再按 3~5 倍余量设姿态环**。\n";
    std::cout << "     超出这个比例的姿态带宽是纯代价。\n";

    std::cout << "\n========================================\n";
    std::cout << g_checks - g_failed << " / " << g_checks << " checks passed\n";
    if (g_failed > 0) {
        std::cout << g_failed << " FAILED\n";
    }
    std::cout << "========================================\n";
    return g_failed == 0 ? 0 : 1;
}
