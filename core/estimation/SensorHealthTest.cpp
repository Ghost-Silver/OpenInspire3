/**
 * @file SensorHealthTest.cpp
 * @brief IMU 传感器健康监测的验证
 *
 * @par 本测试的组织原则
 *
 * 每个场景都遵循「先证明故障生效，再断言检测结果」两步：
 *
 *   1. **激励确认**：先断言故障确实改变了估计器的行为（例如加速度计归零后
 *      姿态误差显著增大）。这一步若失败，说明故障根本没注入成功，
 *      后面的「检测到了」就毫无意义。
 *   2. **检测断言**：再断言 SensorHealth 给出了正确的状态与故障类型。
 *
 * 这不是形式主义。本项目已有多次「以为在测 A，实际在测 B」的实例
 * （水平风激发不到轴向入流、把物理效应开关设为 0 而关掉了效应本身、
 * 差分跨段边界、验证脚本系数写错）。本测试中「激励确认」这一步
 * 就是为了防止重蹈覆辙。
 *
 * @par 关于「不误报」的断言
 *
 * 正常场景下的断言（期望 Healthy）与故障场景下的断言同等重要。
 * 一个对什么都报警的检测器毫无价值。因此正常场景也做了长时间
 * （含机动）的观察，确认不会误触发。
 */

#include "ImuModel.h"
#include "SensorHealth.h"
#include "SixDofTypes.h"
#include "StateEstimator.h"
#include "DroneTypes.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace oi3;

namespace {

int g_pass = 0;
int g_fail = 0;

void check(bool cond, const std::string &name) {
    if (cond) {
        std::printf("[ ok ] %s\n", name.c_str());
        ++g_pass;
    } else {
        std::printf("[FAIL] %s\n", name.c_str());
        ++g_fail;
    }
}

/// 四元数 -> 姿态误差角（度）：估计相对真值的偏差
double attErrorDeg(const std::array<double, 4> &qe, const std::array<double, 4> &qt) {
    const double we = qe[0], xe = qe[1], ye = qe[2], ze = qe[3];
    const double wt = qt[0], xt = -qt[1], yt = -qt[2], zt = -qt[3];
    const double w = we * wt - xe * xt - ye * yt - ze * zt;
    const double c = std::min(1.0, std::fabs(w));
    return 2.0 * std::acos(c) * 180.0 / M_PI;
}

/// 故障注入类型
enum class Fault { None, AccelZero, AccelFrozen, AccelBias, GyroZero, GyroFrozen, GyroBias };

/// 一次完整运行的观测结果
struct RunOut {
    double max_att_err_deg = 0.0;  ///< 峰值姿态误差（用于激励确认）
    double final_att_err_deg = 0.0;
    double mean_resid = 0.0;       ///< 平均方向残差
    double min_accel_mag = 1e9;    ///< 最小加速度计模长
    SensorHealthReport health;
};

/**
 * @brief 跑一段带故障注入的估计
 *
 * @param fault        注入的故障类型
 * @param maneuvering  真值是否做滚转振荡（机动激励）
 * @param fault_start  从第几步开始注入
 * @param steps        总步数
 */
RunOut runFault(Fault fault, bool maneuvering, int fault_start, int steps = 20000) {
    ImuConfig icfg;
    icfg.explicit_bias = true;
    icfg.accel_bias_vec = {0.0, 0.0, 0.0};
    icfg.gyro_bias_vec = {0.0, 0.0, 0.0};
    ImuModel imu(icfg, 2024u);

    EstimatorConfig ecfg;
    ecfg.sensor_health.enabled = true; // 启用监测
    StateEstimator est(ecfg);
    est.reset();

    const double dt = 0.001;
    RunOut out;
    double resid_sum = 0.0;
    int resid_n = 0;

    // 冻结用的固定值：取一个「看起来合理」但不再变化的读数
    const std::array<double, 3> frozen_accel = {0.1, 0.2, -9.7};
    const std::array<double, 3> frozen_gyro = {0.001, -0.002, 0.003};

    for (int k = 0; k < steps; ++k) {
        const double t = k * dt;
        const double roll_true =
            maneuvering ? (15.0 * M_PI / 180.0) * std::sin(2.0 * M_PI * 0.5 * t) : 0.0;
        const double cr = std::cos(roll_true / 2.0), sr = std::sin(roll_true / 2.0);
        const std::array<double, 4> q_true = {cr, sr, 0.0, 0.0};
        const double w_true =
            maneuvering
                ? (15.0 * M_PI / 180.0) * 2.0 * M_PI * 0.5 * std::cos(2.0 * M_PI * 0.5 * t)
                : 0.0;

        Tensor q_t = Tensor{static_cast<float>(q_true[0]), static_cast<float>(q_true[1]),
                            static_cast<float>(q_true[2]), static_cast<float>(q_true[3])};
        Tensor om_t = makeVec3(static_cast<float>(w_true), 0.0f, 0.0f);
        Tensor z3 = makeVec3(0.0f, 0.0f, 0.0f);
        const SixDofState truth{q_t, z3, q_t, om_t};
        const std::array<double, 3> spec = {0.0, 0.0, -9.81};

        ImuSample s = imu.measure(truth, spec, dt);

        if (k >= fault_start) {
            switch (fault) {
            case Fault::AccelZero:
                s.accel = {0.0, 0.0, 0.0};
                break;
            case Fault::AccelFrozen:
                s.accel = frozen_accel; // 值合理但恒定不变
                break;
            case Fault::AccelBias:
                s.accel = {s.accel[0] + 1.0, s.accel[1], s.accel[2]}; // 约 10% g
                break;
            case Fault::GyroZero:
                s.gyro = {0.0, 0.0, 0.0};
                break;
            case Fault::GyroFrozen:
                s.gyro = frozen_gyro;
                break;
            case Fault::GyroBias:
                s.gyro = {s.gyro[0] + 0.05, s.gyro[1], s.gyro[2]};
                break;
            default:
                break;
            }
        }

        est.updateImu(s, dt);

        const double ae = attErrorDeg(est.attitude(), q_true);
        out.max_att_err_deg = std::max(out.max_att_err_deg, ae);
        if (k == steps - 1) {
            out.final_att_err_deg = ae;
        }
        const double am = std::sqrt(s.accel[0] * s.accel[0] + s.accel[1] * s.accel[1] +
                                    s.accel[2] * s.accel[2]);
        out.min_accel_mag = std::min(out.min_accel_mag, am);
        if (k > steps / 5) {
            resid_sum += est.sensorHealth().residual;
            ++resid_n;
        }
    }
    out.mean_resid = (resid_n > 0) ? (resid_sum / resid_n) : 0.0;
    out.health = est.sensorHealth();
    return out;
}

} // namespace

int main() {
    std::printf("=== SensorHealthTest: IMU 传感器健康监测 ===\n\n");

    const int fault_start = 8000; // 8 秒后注入
    const int steps = 20000;      // 总 20 秒

    // ================================================================
    std::printf("--- 1. 正常场景：不得误报（含机动）---\n");
    // ================================================================
    {
        const RunOut base_static = runFault(Fault::None, false, fault_start, steps);
        check(base_static.health.accel == SensorStatus::Healthy,
              "静止正常：加速度计判定 Healthy（无误报）");
        check(base_static.health.gyro == SensorStatus::Healthy,
              "静止正常：陀螺判定 Healthy（无误报）");
        check(base_static.health.fault == ImuFault::None, "静止正常：无故障类型");
        check(base_static.health.samples > 0, "静止正常：确实在监测（samples>0）");

        const RunOut base_man = runFault(Fault::None, true, fault_start, steps);
        check(base_man.health.accel == SensorStatus::Healthy,
              "机动正常：加速度计仍 Healthy（机动不误报）");
        check(base_man.health.maneuvering, "机动正常：正确识别为机动状态");
        // 机动时残差基线抬升两个量级，这是已知物理现象（加速度计读比力非重力）
        check(base_man.mean_resid > base_static.mean_resid,
              "机动正常：残差基线确实高于静止（机动污染存在）");
    }

    // ================================================================
    std::printf("\n--- 2. 加速度计彻底归零（掉线）---\n");
    // ================================================================
    {
        const RunOut r = runFault(Fault::AccelZero, false, fault_start, steps);

        // --- 激励确认：故障必须真的改变了系统行为 ---
        check(r.min_accel_mag < 0.01, "激励确认：加速度计模长确实趋零");
        check(r.final_att_err_deg > 0.05,
              "激励确认：姿态误差显著增大（说明故障影响到估计）");

        // --- 检测断言 ---
        check(r.health.accel == SensorStatus::Failed, "检测：加速度计判定 Failed");
        check(r.health.fault == ImuFault::AccelDead, "检测：故障类型 AccelDead");
        check(!r.health.usable(), "检测：整体不可用（usable()==false）");
    }

    // ================================================================
    std::printf("\n--- 3. 加速度计冻结（卡死在非零值）---\n");
    // ================================================================
    {
        const RunOut r = runFault(Fault::AccelFrozen, false, fault_start, steps);
        check(r.health.accel == SensorStatus::Failed, "检测：加速度计判定 Failed");
        check(r.health.fault == ImuFault::AccelFrozen, "检测：故障类型 AccelFrozen");
    }

    // ================================================================
    std::printf("\n--- 4. 加速度计偏置「说谎」（静止，最危险的一类）---\n");
    // ================================================================
    {
        const RunOut base = runFault(Fault::None, false, fault_start, steps);
        const RunOut r = runFault(Fault::AccelBias, false, fault_start, steps);

        // --- 激励确认：偏置必须真的污染了姿态 ---
        check(r.final_att_err_deg > 1.0, "激励确认：偏置造成姿态误差 > 1 度");
        check(r.final_att_err_deg > 10.0 * base.final_att_err_deg,
              "激励确认：姿态误差比基线大一个量级以上");
        // 关键：模长判据**测不出**偏置（偏置对模长是二阶影响），
        // 但残差能测出。这里同时断言两个事实，防止将来误改检测量。
        check(r.mean_resid > 5.0 * base.mean_resid, "激励确认：方向残差显著抬升");

        // --- 检测断言 ---
        check(r.health.accel == SensorStatus::Degraded,
              "检测：加速度计判定 Degraded（仍可用但精度受损）");
        check(r.health.fault == ImuFault::AccelBias, "检测：故障类型 AccelBias");
        check(r.health.confidence > 0.0, "检测：给出了非零置信度");
    }

    // ================================================================
    std::printf("\n--- 5. 陀螺冻结 ---\n");
    // ================================================================
    {
        const RunOut r = runFault(Fault::GyroFrozen, true, fault_start, steps);
        check(r.health.gyro == SensorStatus::Failed, "检测：陀螺判定 Failed");
        check(r.health.fault == ImuFault::GyroFrozen, "检测：故障类型 GyroFrozen");
    }

    // ================================================================
    std::printf("\n--- 6. 陀螺偏置（机动激励下）---\n");
    // ================================================================
    {
        const RunOut base = runFault(Fault::None, true, fault_start, steps);
        const RunOut r = runFault(Fault::GyroBias, true, fault_start, steps);
        check(r.max_att_err_deg > 5.0 * base.max_att_err_deg,
              "激励确认：陀螺偏置使姿态误差显著增大");
        // 陀螺偏置会让残差抬升（姿态被带偏后加速度计与之不符）
        check(r.mean_resid > base.mean_resid, "激励确认：残差高于基线");
    }

    // ================================================================
    std::printf("\n--- 7. 陀螺归零：静止时靠「无噪声」特征检出 ---\n");
    // ================================================================
    {
        // 一条容易想错的地方：静止时真值角速度本就是零，直觉上会以为
        // 「归零前后数据一致、不可观测」。但真实陀螺带噪声（gyro_noise=0.004），
        // 正常读数在零附近持续抖动；而失效后输出是**精确零、逐位不变**。
        // 因此「输出完全冻结」本身就是死传感器的指纹 —— 静止时也可检出。
        // 第一版测试曾反过来断言「静止时检测不到」，是被直觉带偏了，
        // 实测数据（姿态误差无差别但冻结判据命中）修正了这个认知。
        const RunOut base = runFault(Fault::None, false, fault_start, steps);
        const RunOut r = runFault(Fault::GyroZero, false, fault_start, steps);

        // 姿态误差确实无差别（静止时归零不影响姿态）——这是物理事实
        check(std::fabs(r.max_att_err_deg - base.max_att_err_deg) < 0.1,
              "静止时陀螺归零：姿态误差与基线无差别（对姿态确实无影响）");
        // 但冻结判据能抓到它
        check(r.health.gyro == SensorStatus::Failed,
              "静止时陀螺归零：由冻结判据检出 Failed（死传感器无噪声）");
        check(r.health.fault == ImuFault::GyroFrozen, "故障类型 GyroFrozen");

        // 机动时危害顯现：姿态跟不上真值
        const RunOut rm = runFault(Fault::GyroZero, true, fault_start, steps);
        check(rm.max_att_err_deg > 5.0, "机动时陀螺归零：姿态误差显著增大（危害显现）");
    }

    // ================================================================
    std::printf("\n--- 8. 未启用监测时的行为（默认关闭）---\n");
    // ================================================================
    {
        ImuConfig icfg;
        icfg.explicit_bias = true;
        ImuModel imu(icfg, 2024u);
        EstimatorConfig ecfg; // 默认 sensor_health.enabled = false
        StateEstimator est(ecfg);
        est.reset();
        const Tensor q = Tensor{1.0f, 0.0f, 0.0f, 0.0f};
        const Tensor z3 = makeVec3(0.0f, 0.0f, 0.0f);
        const SixDofState truth{q, z3, q, z3};
        for (int k = 0; k < 5000; ++k) {
            const ImuSample s = imu.measure(truth, {0.0, 0.0, -9.81}, 0.001);
            est.updateImu(s, 0.001);
        }
        check(est.sensorHealth().samples == 0,
              "默认关闭：不积累样本（既有行为逐位不变）");
        check(est.sensorHealth().accel == SensorStatus::Unknown,
              "默认关闭：状态为 Unknown 而非 Healthy（避免误读为健康）");
    }

    std::printf("\n========================================\n");
    std::printf("%d / %d checks passed\n", g_pass, g_pass + g_fail);
    std::printf("========================================\n");
    return (g_fail == 0) ? 0 : 1;
}
