/**
 * @file ImuDegradeClosedLoopTest.cpp
 * @brief 闭环对照：陀螺失效后「执行降级」是否真的比「不降级」更好
 *
 * @par 为什么必须有这个测试
 *
 * 前面的测试只证明了两件事：检测器能报出故障、决策表能给出动作。
 * 但**没有证明这个动作真的有用**。一个永远喊「紧急降落」而对结果毫无改善的
 * 机制，与没有这个机制等价——甚至更糟（它给了虚假的安全感）。
 *
 * 因此这里做对照实验：同样的陀螺失效，一组执行降级、一组不执行，
 * 比较最终结果。这是「激励到达机制」的最后一道验证。
 *
 * @par 一个重要的诚实声明
 *
 * 四旋翼是**开环不稳定**系统，姿态控制完全依赖角速度反馈。陀螺彻底失效后，
 * 纯软件手段能否避免坠机，事前并不显然：
 *
 * - 若紧急降落能让飞机以较小倾角落地 → 降级有效（减小伤害）
 * - 若两组都翻掉 → 说明纯软件无法挽救，必须靠硬件冗余（双 IMU）
 *
 * **两种结果都有价值**，且第二种更重要——它给出「软件容错的边界在哪」。
 * 本测试如实报告观测到的结果，不预设结论。
 *
 * @par 观测指标
 *
 * - 峰值倾角：衡量姿态发散程度（>60° 视为已翻）
 * - 触地时的倾角与垂直速度：决定撞击严重程度
 * - 是否安全着陆：触地时倾角小且下降率可控
 */

#include "ImuDegradePolicy.h"
#include "ImuModel.h"
#include "SensorHealth.h"
#include "SixDofPidController.h"
#include "SixDofSimulator.h"
#include "SixDofTypes.h"
#include "StateEstimator.h"
#include "TensorUtils.h"
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

/// 四元数转欧拉角（度），返回 {roll, pitch, yaw}
std::array<double, 3> quatToEulerDeg(const std::array<double, 4> &q) {
    const double w = q[0], x = q[1], y = q[2], z = q[3];
    const double sp = 2.0 * (w * y - z * x);
    const double roll = std::atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y));
    const double pitch = (std::fabs(sp) >= 1.0) ? std::copysign(M_PI / 2.0, sp) : std::asin(sp);
    constexpr double R2D = 180.0 / M_PI;
    return {roll * R2D, pitch * R2D, 0.0};
}

/// 倾角大小（度）
double tiltDeg(const std::array<double, 4> &q) {
    const auto e = quatToEulerDeg(q);
    return std::sqrt(e[0] * e[0] + e[1] * e[1]);
}

/// 读张量为数组
std::array<double, 3> readVec(const Tensor &t) {
    const std::vector<float> v = toVector(t);
    return {static_cast<double>(v[0]), static_cast<double>(v[1]), static_cast<double>(v[2])};
}

/// 由真值四元数张量求倾角（度）
double tiltDegOf(const Tensor &qt) {
    const std::vector<float> q = toVector(qt);
    // 倾角 = 机体 z 轴与 NED 竖直方向的夹角；r33 = 1-2(x²+y²)
    const double r33 = 1.0 - 2.0 * (static_cast<double>(q[1]) * q[1] +
                                    static_cast<double>(q[2]) * q[2]);
    return std::acos(std::max(-1.0, std::min(1.0, r33))) * 180.0 / M_PI;
}

struct ClosedLoopOut {
    double max_tilt_deg = 0.0;   ///< 峰值倾角
    double final_alt = 0.0;      ///< 最终高度（米，向上为正）
    double final_tilt_deg = 0.0; ///< 触地/结束时的倾角
    double touchdown_speed = 0.0;///< 触地时的垂直速度
    double touchdown_tilt_deg = 0.0; ///< 触地时的倾角
    bool crashed = false;        ///< 是否失控（倾角超 60 度）
    bool landed = false;         ///< 是否触地
    SensorHealthReport health;
    DegradeDecision decision;
};

/**
 * @brief 跑一次闭环
 *
 * @param degrade 陀螺失效后是否执行降级（紧急降落）
 * @param steps   总步数
 */
ClosedLoopOut runClosedLoop(bool degrade, int steps = 8000) {
    SixDofConfig cfg;
    const double dt = cfg.base.dt;

    // 初始：5 米高度悬停（NED 系 z = -5）
    SixDofState init{makeVec3(0.0f, 0.0f, -5.0f), makeVec3(0.0f, 0.0f, 0.0f),
                     Tensor{1.0f, 0.0f, 0.0f, 0.0f}, makeVec3(0.0f, 0.0f, 0.0f)};
    SixDofSimulator sim(cfg, init);
    SixDofPidController ctrl(cfg, {});

    ImuConfig icfg;
    icfg.explicit_bias = true;
    icfg.accel_bias_vec = {0.0, 0.0, 0.0};
    icfg.gyro_bias_vec = {0.0, 0.0, 0.0};
    ImuModel imu(icfg, 20260918u);

    EstimatorConfig ecfg;
    ecfg.sensor_health.enabled = true;
    StateEstimator est(ecfg);
    est.reset();

    const ImuDegradePolicy policy;
    const Tensor target = makeVec3(0.0f, 0.0f, -5.0f); // 悬停目标点

    const int fault_start = 3000; // 3 秒后注入陀螺失效
    ClosedLoopOut out;

    for (int k = 0; k < steps; ++k) {
        const double t = k * dt;
        const SixDofState &truth = sim.state();

        // --- IMU 采样（真实值）---
        // 比力：NED 下的支撑力，悬停时约 [0,0,-g]
        const std::array<double, 3> spec = {0.0, 0.0, -cfg.base.gravity};
        ImuSample s = imu.measure(truth, spec, dt);
        if (k >= fault_start) {
            s.gyro = {0.0, 0.0, 0.0}; // 陀螺彻底失效
        }
        est.updateImu(s, dt);

        // --- 位置测量（100 Hz 降采样）---
        constexpr int pos_decim = 10;
        if (k % pos_decim == 0) {
            const auto p = readVec(sim.state().pos);
            est.updatePosition(p, dt * pos_decim);
        }

        // --- 控制：使用估计状态 ---
        SixDofCommand cmd;
        const auto &h = est.sensorHealth();
        const DegradeDecision d = policy.decide(h);
        const bool emergency = (d.action == DegradeAction::EmergencyLand);

        if (emergency && degrade) {
            // 紧急降落：放弃姿态控制与安全高度保持，直接降推力使其下降。
            // 陀螺已失效，姿态反馈不可信，继续做姿态修正只会加剧发散。
            cmd.thrust_body = cfg.base.mass * cfg.base.gravity * 0.6f; // 低于悬停推力
            cmd.torque = makeVec3(0.0f, 0.0f, 0.0f);                   // 零力矩
        } else {
            cmd = ctrl.compute(est.state(), target, t);
        }

        sim.step(cmd.thrust_body, cmd.torque);

        // --- 观测 ---
        const double td = tiltDegOf(sim.state().quat);
        out.max_tilt_deg = std::max(out.max_tilt_deg, td);
        const double alt = -readVec(sim.state().pos)[2]; // NED z 取负得高度
        out.final_alt = alt;
        out.final_tilt_deg = td;
        if (td > 60.0) {
            out.crashed = true;
        }
        // 触地即停止积分。仿真模型没有地面碰撞，若继续积分飞机会「穿过地面」
        // 无限下落（第一版即如此，末态高度 −41 m 完全是穿地后的假结果）。
        // 触地后的一切数据都无物理意义，故在此中断并以触地瞬间作为终态。
        if (alt <= 0.0 && !out.landed) {
            out.landed = true;
            out.touchdown_speed = std::fabs(readVec(sim.state().vel)[2]);
            out.touchdown_tilt_deg = td;
            out.final_alt = 0.0;
            // 触地中断时**也必须**记录健康状态与决策。第一版把这两行放在
            // `if (k == steps-1)` 内，而 break 提前退出使该条件永不成立，
            // 于是 health 停留在默认构造值（Unknown/None），检测断言全灭。
            out.health = est.sensorHealth();
            out.decision = policy.decide(out.health);
            break;
        }
        if (k == steps - 1) {
            out.health = h;
            out.decision = d;
        }
    }
    return out;
}

} // namespace

int main() {
    std::printf("=== ImuDegradeClosedLoopTest: 降级动作是否有用（对照实验）===\n\n");
    std::printf("场景：5 m 悬停，3.0 s 时陀螺彻底失效（输出恒零）\n");
    std::printf("A 组 = 不降级（继续悬停任务）；B 组 = 执行紧急降落\n\n");

    const auto a = runClosedLoop(false);
    const auto b = runClosedLoop(true);

    std::printf("--- 观测结果 ---\n");
    std::printf("              峰值倾角    末态高度    触地倾角   触地垂速   失控\n");
    std::printf("A 不降级    %8.2f°  %8.2f m  %8.2f°  %8.2f    %s\n", a.max_tilt_deg, a.final_alt,
                a.touchdown_tilt_deg, a.touchdown_speed, a.crashed ? "是" : "否");
    std::printf("B 紧急降落  %8.2f°  %8.2f m  %8.2f°  %8.2f    %s\n", b.max_tilt_deg, b.final_alt,
                b.touchdown_tilt_deg, b.touchdown_speed, b.crashed ? "是" : "否");
    std::printf("\n");

    // ================================================================
    std::printf("--- 1. 故障确实被检出（激励确认）---\n");
    // ================================================================
    {
        check(a.health.gyro == SensorStatus::Failed, "A 组：陀螺被检出 Failed");
        check(b.health.gyro == SensorStatus::Failed, "B 组：陀螺被检出 Failed");
        check(a.decision.action == DegradeAction::EmergencyLand, "A 组：决策给出 EmergencyLand");
        check(b.decision.action == DegradeAction::EmergencyLand, "B 组：决策给出 EmergencyLand");
    }

    // ================================================================
    std::printf("\n--- 2. 故障确实造成危害（激励确认）---\n");
    // ================================================================
    {
        // 若不降级时飞机毫无反应，说明陀螺失效根本没影响到闭环，
        // 后面的「降级有用」就没有意义。
        check(a.max_tilt_deg > 10.0, "A 组：陀螺失效确实造成姿态发散（>10°）");
    }

    // ================================================================
    std::printf("\n--- 3. 降级是否改善了结果 ---\n");
    // ================================================================
    {
        // 核心对照：紧急降落组的峰值倾角应更小（或至少不更差）
        const bool improved = (b.max_tilt_deg < a.max_tilt_deg);
        std::printf("       B 峰值倾角 %.2f° vs A %.2f° → %s\n", b.max_tilt_deg, a.max_tilt_deg,
                    improved ? "改善" : "未改善");
        check(improved, "对照：紧急降落组的峰值倾角小于不降级组");

        // 触地姿态：这是决定「撞击严重程度」的量，比峰值倾角更贴近后果
        std::printf("       触地倾角：B %.2f° vs A %.2f°\n", b.touchdown_tilt_deg,
                    a.touchdown_tilt_deg);
        check(b.touchdown_tilt_deg < a.touchdown_tilt_deg,
              "对照：紧急降落组触地姿态更平（撞击更轻）");
        check(b.touchdown_tilt_deg < 30.0,
              "对照：紧急降落组以接近水平的姿态触地（<30°）");

        // 紧急降落应让飞机真正下降（否则「降落」是空话）
        const bool descended = (b.final_alt <= a.final_alt);
        std::printf("       B 末态高度 %.2f m vs A %.2f m → %s\n", b.final_alt, a.final_alt,
                    descended ? "确实下降" : "未下降");
        check(descended, "对照：紧急降落组确实降低了高度");
    }

    std::printf("\n========================================\n");
    std::printf("%d / %d checks passed\n", g_pass, g_pass + g_fail);
    std::printf("========================================\n");
    return (g_fail == 0) ? 0 : 1;
}
