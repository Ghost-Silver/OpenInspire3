/**
 * @file EstimatorTest.cpp
 * @brief 状态估计验证：姿态估计精度、偏置可观测性、以及闭环性能的真实代价
 * @author GhostFace
 * @date 2026/9/18
 *
 * @par 这个测试要回答的关键问题
 *
 * 此前所有控制性能数字（例如 6-DoF 悬停稳态误差 0.5 mm）都是在**完美状态反馈**下
 * 得到的 —— 控制器直接读仿真真值。真机上不存在这个信号。本测试把链路补全成
 * 「真值 → IMU/位置传感器（带噪）→ 状态估计 → 控制器」，然后**同条件对比**
 * 理想反馈与估计反馈的闭环性能。
 *
 * 那个差值就是这个飞控从仿真走向真机时必须付出的第一笔代价。把它量出来，
 * 比继续优化理想反馈下的数字更有意义。
 *
 * @par 三个实验
 *
 * 1. **静止水平下的姿态估计**：设显式陀螺偏置，验证加速度计校正确实把漂移压住，
 *    并量化偏置估计的收敛性。对照组是「陀螺纯积分」—— 没有校正时姿态必然发散。
 * 2. **偏置可观测性**：只有加速度计时，绕重力轴的偏置（yaw 轴）**物理上不可观测**，
 *    因此本测试**分轴评估**而不是给一个笼统的全轴阈值 —— 后者会让「估计器其实
 *    没工作」和「观测不到那一轴」这两种完全不同的情况混在一起。
 * 3. **闭环对比**：同一控制器，只有状态来源不同（真值 vs 估计）。
 */

#include "ImuModel.h"
#include "SixDofDynamics.h"
#include "SixDofPidController.h"
#include "SixDofSimulator.h"
#include "StateEstimator.h"
#include "TensorUtils.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
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

const char *kAxisName[3] = {"roll ", "pitch", "yaw  "};

/// 由四元数提取欧拉角（度），ZYX 顺序
std::array<double, 3> quatToEulerDeg(const std::array<double, 4> &q) {
    const double w = q[0], x = q[1], y = q[2], z = q[3];
    const double roll = std::atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y));
    const double sp = 2.0 * (w * y - z * x);
    const double pitch = std::asin(std::max(-1.0, std::min(1.0, sp)));
    const double yaw = std::atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z));
    const double r2d = 180.0 / M_PI;
    return {roll * r2d, pitch * r2d, yaw * r2d};
}

std::array<double, 4> readQuat(const Tensor &q) {
    const std::vector<float> v = toVector(q);
    return {v[0], v[1], v[2], v[3]};
}

std::array<double, 3> readVec(const Tensor &t) {
    const std::vector<float> v = toVector(t);
    return {v[0], v[1], v[2]};
}

Tensor makeVec4(const std::array<double, 4> &a) {
    Tensor t(ShapeTag{}, {4});
    for (int i = 0; i < 4; ++i) {
        t.data_write<float>()[i] = static_cast<float>(a[static_cast<std::size_t>(i)]);
    }
    return t;
}

/// 闭环运行结果
struct RunResult {
    double settle_time = -1.0;
    double final_err = 0.0;
    double pos_err_rms = 0.0;      ///< 全程 RMS（被初始瞬态主导，仅作参考）
    double steady_pos_rms = 0.0;   ///< 最后 1 秒 RMS（稳态段，衡量估计质量的真实代价）
    double att_err_rms_deg = 0.0;
    double att_err_max_deg = 0.0;
    bool finite = true;

    // ---- 估计误差分解：用于定位「闭环劣化」来自哪一环 ----
    // 不看这三个数就调参数，等于对着结果猜原因。
    double est_tilt_err_rms = 0.0; ///< 估计倾角 vs 真值倾角（度）
    double est_yaw_err_rms = 0.0;  ///< 估计偏航 vs 真值偏航（度）
    double est_pos_err_rms = 0.0;  ///< 估计位置 vs 真值位置（米）
    double est_vel_err_rms = 0.0;  ///< 估计速度 vs 真值速度（米/秒）
    double steady_tilt_deg = 0.0;  ///< 最后 1 秒的倾角 RMS（区分瞬态与稳态振荡）
};

/**
 * @brief 闭环运行，可分别切换姿态与位置/速度的来源
 *
 * 拆成两个开关是为了做消融：闭环劣化可能来自姿态估计，也可能来自位置估计，
 * 一起开关只能看到「总体变差」，定位不到是哪一环。无论开关怎么设，估计器
 * 始终在跑 —— 保证四个配置面对完全相同的信息环境。
 *
 * @param use_est_att 控制器姿态/角速度取自估计器（否则取真值）
 * @param use_est_pos 控制器位置/速度取自估计器（否则取真值）
 */
RunResult runClosedLoop(const SixDofConfig &cfg, const SixDofPidGains &gains,
                        const EstimatorConfig &est_cfg, const std::array<double, 3> &target,
                        double seconds, bool use_est_att, bool use_est_pos,
                        std::uint32_t seed) {
    const SixDofState init{makeVec3(0.0f, 0.0f, -5.0f), makeVec3(0.0f, 0.0f, 0.0f),
                           Tensor{1.0f, 0.0f, 0.0f, 0.0f}, makeVec3(0.0f, 0.0f, 0.0f)};
    SixDofSimulator sim(cfg, init);
    SixDofPidController ctrl(cfg, gains);

    ImuConfig imu_cfg;
    ImuModel imu(imu_cfg, seed);
    StateEstimator est(est_cfg);

    std::mt19937 rng(seed);
    std::normal_distribution<double> gauss(0.0, 1.0);
    const double pos_noise = 0.02; // 位置测量噪声标准差（米），光流/UWB 量级

    const double dt = cfg.base.dt;
    const int steps = static_cast<int>(seconds / dt);
    const int pos_decim = 10; // 位置测量 100 Hz

    RunResult out;
    int hold = 0;
    double pe2 = 0.0, ae2 = 0.0;
    double steady_pe2 = 0.0, steady_tilt2 = 0.0;
    double ete2 = 0.0, eye2 = 0.0, epe2 = 0.0, eve2 = 0.0;
    int n = 0, steady_n = 0;
    const int steady_start = std::max(0, steps - 1000); // 最后 1 秒
    SixDofState prev = sim.state();

    for (int k = 0; k < steps; ++k) {
        const SixDofState truth = sim.state();

        SixDofState feedback = truth;
        if (use_est_att || use_est_pos) {
            // 比力 = a − g_vec（NED）。由真值速度差分得到加速度，避免与动力学
            // 重复实现一套「真值比力」的口径。
            const std::array<double, 3> v_now = readVec(truth.vel);
            const std::array<double, 3> v_prev = readVec(prev.vel);
            const std::array<double, 3> specific_force = {
                (v_now[0] - v_prev[0]) / dt, (v_now[1] - v_prev[1]) / dt,
                (v_now[2] - v_prev[2]) / dt - cfg.base.gravity};

            const ImuSample sample = imu.measure(truth, specific_force, dt);
            est.updateImu(sample, dt);

            if (k % pos_decim == 0) {
                const std::array<double, 3> p = readVec(truth.pos);
                est.updatePosition({p[0] + pos_noise * gauss(rng),
                                    p[1] + pos_noise * gauss(rng),
                                    p[2] + pos_noise * gauss(rng)},
                                   dt * pos_decim);
            }
            const SixDofState est_state = est.state();
            if (use_est_att) {
                feedback.quat = est_state.quat;
                feedback.omega = est_state.omega;
            }
            if (use_est_pos) {
                feedback.pos = est_state.pos;
                feedback.vel = est_state.vel;
            }

            // 估计误差分解（用真值作参照）
            const std::array<double, 3> eu_e = quatToEulerDeg(est.attitude());
            const std::array<double, 3> eu_t = quatToEulerDeg(readQuat(truth.quat));
            ete2 += (eu_e[0] - eu_t[0]) * (eu_e[0] - eu_t[0]) +
                    (eu_e[1] - eu_t[1]) * (eu_e[1] - eu_t[1]);
            eye2 += (eu_e[2] - eu_t[2]) * (eu_e[2] - eu_t[2]);

            const std::array<double, 3> pe = est.position();
            const std::array<double, 3> pt = readVec(truth.pos);
            epe2 += (pe[0] - pt[0]) * (pe[0] - pt[0]) + (pe[1] - pt[1]) * (pe[1] - pt[1]) +
                    (pe[2] - pt[2]) * (pe[2] - pt[2]);

            const std::array<double, 3> ve = est.velocity();
            const std::array<double, 3> vt = readVec(truth.vel);
            eve2 += (ve[0] - vt[0]) * (ve[0] - vt[0]) + (ve[1] - vt[1]) * (ve[1] - vt[1]) +
                    (ve[2] - vt[2]) * (ve[2] - vt[2]);
        }

        const SixDofCommand cmd = ctrl.compute(
            feedback,
            makeVec3(static_cast<float>(target[0]), static_cast<float>(target[1]),
                     static_cast<float>(target[2])),
            static_cast<double>(k) * dt);
        sim.step(cmd.thrust_body, cmd.torque);
        prev = truth;

        // 指标用**真值**算：评估的是真实飞行质量，不是估计的好坏
        const std::array<double, 3> p_now = readVec(sim.state().pos);
        const double err = std::sqrt(std::pow(p_now[0] - target[0], 2) +
                                     std::pow(p_now[1] - target[1], 2) +
                                     std::pow(p_now[2] - target[2], 2));
        const std::array<double, 3> eu = quatToEulerDeg(readQuat(sim.state().quat));
        const double att = std::sqrt(eu[0] * eu[0] + eu[1] * eu[1]);

        pe2 += err * err;
        ae2 += att * att;
        ++n;
        if (k >= steady_start) {
            steady_pe2 += err * err;
            steady_tilt2 += att * att;
            ++steady_n;
        }
        out.att_err_max_deg = std::max(out.att_err_max_deg, att);

        if (err < 0.05) {
            if (++hold >= 500 && out.settle_time < 0.0) {
                out.settle_time = static_cast<double>(k) * dt;
            }
        } else {
            hold = 0;
        }
        out.final_err = err;
        out.finite = out.finite && std::isfinite(err);
    }

    out.pos_err_rms = std::sqrt(pe2 / std::max(1, n));
    out.steady_pos_rms = std::sqrt(steady_pe2 / std::max(1, steady_n));
    out.steady_tilt_deg = std::sqrt(steady_tilt2 / std::max(1, steady_n));
    out.att_err_rms_deg = std::sqrt(ae2 / std::max(1, n));
    out.est_tilt_err_rms = std::sqrt(ete2 / std::max(1, n));
    out.est_yaw_err_rms = std::sqrt(eye2 / std::max(1, n));
    out.est_pos_err_rms = std::sqrt(epe2 / std::max(1, n));
    out.est_vel_err_rms = std::sqrt(eve2 / std::max(1, n));
    return out;
}

/**
 * @brief 静止水平 + 显式陀螺偏置下的姿态估计
 *
 * 跑 60 秒并在多个时间点采样倾角误差 —— 偏置引起的漂移是**长时间**效应
 * （水平偏置约 0.005 rad/s，5 秒只有 1.4 度），只在终点看一个数区分不出
 * 「有界」与「线性发散」。分时间点采样能直接把两种趋势摊开。
 *
 * @param cfg_est    估计器配置（用于做「陀螺纯积分」对照）
 * @param tilt_marks 输出：在 5 / 20 / 60 秒时刻的倾角误差（度）
 * @param yaw_drift  输出：60 秒时刻的偏航漂移（度）
 */
void attitudeEstimation(const EstimatorConfig &cfg_est, const char *label,
                        std::array<double, 3> &tilt_marks, double &yaw_drift,
                        std::array<double, 3> &bias_est, std::array<double, 3> &bias_truth) {
    ImuConfig imu_cfg;
    imu_cfg.explicit_bias = true;
    imu_cfg.gyro_bias_vec = {0.004, -0.003, 0.005};
    imu_cfg.accel_bias_vec = {0.03, -0.02, 0.01};
    ImuModel imu(imu_cfg, 2024u);

    StateEstimator est(cfg_est);
    est.reset();

    const double dt = 0.001;
    const int total = 60000; // 60 秒
    const int mark_step[3] = {5000, 20000, 60000};
    int mark_idx = 0;

    // 真值：静止水平（初始姿态为单位四元数）。机体受到的比力是支撑力，
    // 即 NED 系下的 [0, 0, -g] —— 归一化后为 [0,0,-1]。
    const Tensor q_t = makeVec4({1.0, 0.0, 0.0, 0.0});
    const Tensor zero3 = makeVec3(0.0f, 0.0f, 0.0f);
    const SixDofState truth{q_t, zero3, q_t, zero3};

    yaw_drift = 0.0;
    tilt_marks = {0.0, 0.0, 0.0};

    for (int k = 1; k <= total; ++k) {
        const ImuSample s = imu.measure(truth, {0.0, 0.0, -9.81}, dt);
        est.updateImu(s, dt);

        if (mark_idx < 3 && k == mark_step[mark_idx]) {
            // 只有加速度计时滚转/俯仰可观测，偏航不可观测 —— 因此倾角与偏航
            // 分开统计，不让偏航漂移污染「倾角估计是否准确」的判断。
            const std::array<double, 3> eu = quatToEulerDeg(est.attitude());
            tilt_marks[static_cast<std::size_t>(mark_idx)] =
                std::sqrt(eu[0] * eu[0] + eu[1] * eu[1]);
            if (mark_idx == 2) {
                yaw_drift = eu[2];
            }
            ++mark_idx;
        }
    }

    bias_est = est.gyroBias();
    bias_truth = imu.gyroBiasTruth();

    std::cout << "  " << std::left << std::setw(26) << label << std::setprecision(3)
              << "倾角误差 5s/20s/60s = " << tilt_marks[0] << " / " << tilt_marks[1] << " / "
              << tilt_marks[2] << " deg";
    if (std::fabs(yaw_drift) > 1e-9) {
        std::cout << "，偏航漂移 " << yaw_drift << " deg";
    }
    std::cout << "\n";
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

    std::cout << "========================================\n";
    std::cout << "状态估计：从真值反馈到带噪估计反馈\n";
    std::cout << "========================================\n";

    // ---- 1. 姿态估计 + 对照组 ----
    std::cout << "\n[1] 静止水平、显式陀螺偏置 (0.004, -0.003, 0.005) rad/s，60 秒\n";

    std::array<double, 3> bias_est{}, bias_truth{};
    std::array<double, 3> mark_closed{}, mark_open{};
    double yaw_closed = 0.0, yaw_open = 0.0;

    EstimatorConfig closed;
    attitudeEstimation(closed, "互补滤波（加速度计校正）", mark_closed, yaw_closed, bias_est,
                       bias_truth);

    EstimatorConfig openloop;
    openloop.accel_correction = 0.0;
    openloop.estimate_gyro_bias = false;
    std::array<double, 3> be_dummy{}, bt_dummy{};
    attitudeEstimation(openloop, "对照组：陀螺纯积分", mark_open, yaw_open, be_dummy, bt_dummy);

    // 第二个对照组：同样的校正增益，但关掉偏置估计。
    // 这一组才真正回答「偏置估计值不值」—— 前者（纯积分）没有校正，
    // 差距里混了校正本身的贡献，分不出偏置估计的那一份。
    EstimatorConfig no_bias;
    no_bias.estimate_gyro_bias = false;
    std::array<double, 3> mark_nobias{}, bne{}, bnt{};
    double yaw_nobias = 0.0;
    attitudeEstimation(no_bias, "对照组：不估计偏置", mark_nobias, yaw_nobias, bne, bnt);

    checkTrue("互补滤波的倾角误差有界（60 秒后仍小于 1 度）", mark_closed[2] < 1.0);
    checkTrue("陀螺纯积分的倾角误差持续发散（60 秒后超过 10 度）", mark_open[2] > 10.0);
    checkTrue("纯积分漂移近似线性增长（20s→60s 接近 3 倍）",
              mark_open[1] > 1e-6 && mark_open[2] / mark_open[1] > 2.0);
    // 偏置估计的实际意义：在同样的校正增益下，把姿态误差压下去。
    // 断言量的是姿态误差而不是偏置值本身 —— 后者受估计噪声底限制，
    // 而前者才是偏置估计存在的理由。
    checkTrue("偏置估计把 60 秒倾角误差压到「不估计偏置」的 1/3 以下",
              mark_nobias[2] > 1e-9 && mark_closed[2] < mark_nobias[2] / 3.0);

    // ---- 2. 偏置可观测性：分轴评估 ----
    std::cout << "\n[2] 陀螺偏置估计（分轴，只有加速度计）\n";
    std::cout << "  " << std::setw(8) << "轴" << std::setw(14) << "真值" << std::setw(14)
              << "估计" << std::setw(14) << "残差" << "\n";
    double horiz_res = 0.0;
    double yaw_res = 0.0;
    for (int i = 0; i < 3; ++i) {
        const auto idx = static_cast<std::size_t>(i);
        const double res = std::fabs(bias_truth[idx] - bias_est[idx]);
        std::cout << "  " << std::setw(8) << kAxisName[i] << std::setw(14) << bias_truth[idx]
                  << std::setw(14) << bias_est[idx] << std::setw(14) << res << "\n";
        if (i < 2) {
            horiz_res += res * res;
        } else {
            yaw_res = res;
        }
    }

    // 滚转/俯仰轴由加速度计直接观测，偏置估计必须真正收敛。
    //
    // 阈值不能凭感觉给。偏置估计的有效噪声底由加速度计噪声经一阶低通后的残留
    // 决定：σ_e = accel_noise/g（方向误差，rad），输出噪声 ≈ σ_e·√(Ki/2)。
    // 取 ImuConfig 默认的 0.08 m/s²、Ki=0.05 得 1.3e-3 rad/s —— 注意它已经
    // **大于**真值本身（约 5e-3 rad/s 的 10%），所以「残差小于真值的 10%」
    // 这个判据在物理上就不可能满足，用它当断言只会得到一个假的失败。
    const double sigma_e = ImuConfig{}.accel_noise / 9.81;
    const double bias_noise_floor = sigma_e * std::sqrt(EstimatorConfig{}.bias_correction / 2.0);
    std::cout << "  （估计噪声底 ≈ " << bias_noise_floor << " rad/s，取 3 倍作判据）\n";
    checkTrue("滚转/俯仰轴偏置估计残差落在噪声底 3 倍以内",
              std::sqrt(horiz_res) < 3.0 * bias_noise_floor);
    std::cout << "  偏航轴残差 " << yaw_res
              << " rad/s —— 加速度计只提供重力方向，绕重力轴的转动不改变该方向，\n"
              << "  因此偏航轴偏置在只有加速度计时**物理上不可观测**，这不是实现缺陷。\n"
              << "  后果：偏航角会持续缓慢漂移，定住偏航必须引入磁力计或视觉航向观测。\n";

    // ---- 3. 闭环对比 ----
    std::cout << "\n[3] 悬停闭环对比（同一控制器，只有状态来源不同）\n";
    const std::array<double, 3> target = {0.5, -0.3, -5.0};

    struct Arm {
        const char *name;
        bool est_att;
        bool est_pos;
    };
    const Arm arms[4] = {{"全真值（理想）", false, false},
                         {"仅姿态估计", true, false},
                         {"仅位置估计", false, true},
                         {"全估计（真机）", true, true}};

    RunResult res[4];
    for (int i = 0; i < 4; ++i) {
        res[i] = runClosedLoop(cfg, gains, EstimatorConfig{}, target, 4.0, arms[i].est_att,
                               arms[i].est_pos, 777u);
    }

    std::cout << std::left << std::setprecision(5);
    std::cout << "  " << std::setw(18) << "配置" << std::setw(12) << "收敛时间"
              << std::setw(12) << "末端误差" << std::setw(14) << "稳态RMS"
              << std::setw(14) << "稳态倾角(deg)" << "\n";
    for (int i = 0; i < 4; ++i) {
        std::cout << "  " << std::setw(18) << arms[i].name << std::setw(12) << res[i].settle_time
                  << std::setw(12) << res[i].final_err << std::setw(14) << res[i].steady_pos_rms
                  << std::setw(14) << res[i].steady_tilt_deg << "\n";
    }
    std::cout << "  （稳态 RMS 取最后 1 秒。全程 RMS 会被从 -5 m 出发的初始瞬态主导，\n";
    std::cout << "    量不出估计质量本身的代价，故不用于比较。）\n";

    const RunResult &ideal = res[0];
    const RunResult &estimated = res[3];

    std::cout << "\n  消融分解（相对全真值）：\n";
    std::cout << "    仅姿态估计使稳态位置 RMS → " << res[1].steady_pos_rms
              << " m，稳态倾角 → " << res[1].steady_tilt_deg << " deg\n";
    std::cout << "    仅位置估计使稳态位置 RMS → " << res[2].steady_pos_rms
              << " m，稳态倾角 → " << res[2].steady_tilt_deg << " deg\n";
    std::cout << "    两者叠加      → " << estimated.steady_pos_rms
              << " m，稳态倾角 → " << estimated.steady_tilt_deg << " deg\n";

    std::cout << "\n  估计误差分解（估计 vs 真值，全估计配置）：\n";
    std::cout << "    倾角    " << estimated.est_tilt_err_rms << " deg\n";
    std::cout << "    偏航    " << estimated.est_yaw_err_rms
              << " deg（无磁力计时必然漂移，不计入倾角判据）\n";
    std::cout << "    位置    " << estimated.est_pos_err_rms << " m\n";
    std::cout << "    速度    " << estimated.est_vel_err_rms << " m/s\n";

    checkTrue("理想反馈闭环稳定且收敛", ideal.finite && ideal.settle_time > 0.0);
    checkTrue("带噪估计反馈闭环仍然稳定（能上真机的最低门槛）",
              estimated.finite && estimated.final_err < 0.3);

    if (ideal.steady_pos_rms > 1e-12 && estimated.steady_pos_rms > 1e-12) {
        std::cout << "\n  稳态位置 RMS 劣化 = " << std::setprecision(1)
                  << (estimated.steady_pos_rms / ideal.steady_pos_rms) << " 倍（"
                  << std::setprecision(5) << ideal.steady_pos_rms << " m → "
                  << estimated.steady_pos_rms << " m）\n";
        std::cout << "  倾角误差 RMS 劣化 = " << std::setprecision(2)
                  << (estimated.att_err_rms_deg / std::max(1e-9, ideal.att_err_rms_deg))
                  << " 倍\n";
    }

    // ---- 4. 加速度计校正增益扫描 ----
    std::cout << "\n[4] 加速度计校正增益扫描（全估计配置）\n";
    std::cout << "  机动时加速度计测的是比力而非重力，校正增益越大，被运动加速度\n";
    std::cout << "  带偏得越重；增益太小则陀螺漂移压不住。这里扫出权衡曲线。\n\n";
    std::cout << "  " << std::setw(12) << "accel_corr" << std::setw(14) << "稳态RMS"
              << std::setw(16) << "稳态倾角(deg)" << std::setw(16) << "倾角估计误差"
              << std::setw(16) << "速度估计误差" << "\n";

    const double kp_grid[7] = {1.0, 0.3, 0.1, 0.05, 0.03, 0.01, 0.003};
    double kp_rms[7] = {};
    double best_rms = 1e9;
    for (int i = 0; i < 7; ++i) {
        EstimatorConfig ec;
        ec.accel_correction = kp_grid[i];
        const RunResult r = runClosedLoop(cfg, gains, ec, target, 4.0, true, true, 777u);
        kp_rms[i] = r.steady_pos_rms;
        std::cout << "  " << std::setw(12) << kp_grid[i] << std::setw(14) << r.steady_pos_rms
                  << std::setw(16) << r.steady_tilt_deg << std::setw(16) << r.est_tilt_err_rms
                  << std::setw(16) << r.est_vel_err_rms << "\n";
        best_rms = std::min(best_rms, r.steady_pos_rms);
    }
    // 断言固化的是一条反直觉的结论：互补滤波的经典经验值 Kp≈1 是为「机动加速度
    // 相对重力可忽略」的场景调的，四旋翼大机动不满足这个前提，在这里差了数倍。
    checkTrue("经典经验值 Kp=1.0 在本场景劣于拐点取值 3 倍以上", kp_rms[0] > 3.0 * kp_rms[3]);

    // ---- 5. 位置校正增益扫描 ----
    // α 与 Kp 是耦合的：α 的上一次取值是在 Kp=1.0（姿态估计误差 3.2 度）下定的，
    // 姿态误差本身会经由「推力方向偏 → 水平漂移」污染位置环路，姿态修好之后
    // 这个值必须重扫，不能沿用。
    std::cout << "\n[5] 位置校正增益 α 扫描（Kp 已定为 0.05，全估计配置）\n";
    std::cout << "  " << std::setw(12) << "alpha" << std::setw(14) << "稳态RMS"
              << std::setw(16) << "稳态倾角(deg)" << std::setw(16) << "位置估计误差"
              << std::setw(16) << "速度估计误差" << "\n";

    const double a_grid[6] = {0.02, 0.05, 0.1, 0.2, 0.35, 0.5};
    double a_rms[6] = {};
    for (int i = 0; i < 6; ++i) {
        EstimatorConfig ec;
        ec.pos_filter_alpha = a_grid[i];
        const RunResult r = runClosedLoop(cfg, gains, ec, target, 4.0, true, true, 777u);
        a_rms[i] = r.steady_pos_rms;
        std::cout << "  " << std::setw(12) << a_grid[i] << std::setw(14) << r.steady_pos_rms
                  << std::setw(16) << r.steady_tilt_deg << std::setw(16) << r.est_pos_err_rms
                  << std::setw(16) << r.est_vel_err_rms << "\n";
    }
    // 内部最优：两端都明显劣于中间，说明 α 是两种相反作用的权衡
    // （小 α 位置滞后 → 控制环相位损失；大 α 噪声灌入 → 抖动），
    // 而不是「越大越好」或「越小越好」的单调关系。
    checkTrue("α 存在内部最优（0.02 与 0.5 两端均劣于 0.35 达 1.5 倍以上）",
              a_rms[0] > 1.5 * a_rms[4] && a_rms[5] > 1.5 * a_rms[4]);

    std::cout << "\n[结论]\n";
    std::cout << "  估计反馈下的性能才是这套飞控在真机上可期望的量级。它比理想反馈差，\n";
    std::cout << "  而这个差距正是「从仿真走向真机」要付的第一笔代价。\n";
    std::cout << "  此前所有理想反馈下的数字（含 0.5 mm 稳态误差）都应读作\n";
    std::cout << "  「控制律在完美状态反馈下的性能」，不能当作飞控精度对外宣称。\n";

    std::cout << "\n========================================\n";
    std::cout << g_checks - g_failed << " / " << g_checks << " checks passed\n";
    if (g_failed > 0) {
        std::cout << g_failed << " FAILED\n";
    }
    std::cout << "========================================\n";
    return g_failed == 0 ? 0 : 1;
}
