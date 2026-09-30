/**
 * @file ImuDegradePolicyTest.cpp
 * @brief 降级策略映射的验证
 *
 * @par 本测试关注「状态 → 动作」的映射是否正确，尤其是两处最易写错的地方
 *
 * 1. **Unknown 不等于 Healthy**。若把样本不足当成健康，等于在没有任何证据时
 *    假设传感器正常——这在启动阶段尤其危险。此处显式断言 Unknown 走 Cautious。
 * 2. **陀螺失效必须是 EmergencyLand 而非 ReturnHome**。返航需要持续做姿态机动，
 *   而姿态环正是陀螺失效后失去的东西。若给出返航，飞机会在返航途中翻掉。
 *    这一条是「分级」的意义所在，单独断言以固定认知。
 *
 * @par 为什么不复用 SensorHealth 的仿真场景
 *
 * 那是端到端验证（故障 → 检测 → 决策）；本测试只验证映射表本身，
 * 直接构造健康报告作为输入。两个层次分开测，才能定位问题是出在
 * 「检测错了」还是「决策错了」——这正是本项目「单个模块正确不能推出
 * 组合正确」的应对方式：先各自正确，再单独验证组合。
 */

#include "ImuDegradePolicy.h"
#include "SensorHealth.h"

#include <cstdio>
#include <string>

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

/// 构造一个指定状态的健康报告
SensorHealthReport makeReport(SensorStatus accel, SensorStatus gyro) {
    SensorHealthReport h;
    h.accel = accel;
    h.gyro = gyro;
    h.samples = 1000; // 足够样本，避免走 Unknown 分支
    return h;
}

} // namespace

int main() {
    std::printf("=== ImuDegradePolicyTest: 降级策略映射 ===\n\n");
    const ImuDegradePolicy policy;

    // ================================================================
    std::printf("--- 1. 正常状态 ---\n");
    // ================================================================
    {
        const auto d = policy.decide(makeReport(SensorStatus::Healthy, SensorStatus::Healthy));
        check(d.action == DegradeAction::Normal, "双健康 → Normal");
        check(d.max_tilt_deg == 35.0, "正常：倾角上限取标称 35 度");
        check(d.use_accel_correction, "正常：使用加速度计校正");
        check(d.trust_position, "正常：信任位置估计");
    }

    // ================================================================
    std::printf("\n--- 2. 陀螺失效（最紧急）---\n");
    // ================================================================
    {
        const auto d = policy.decide(makeReport(SensorStatus::Healthy, SensorStatus::Failed));
        check(d.action == DegradeAction::EmergencyLand, "陀螺失效 → EmergencyLand");
        // 关键：不能是 ReturnHome。返航要持续姿态机动，而姿态环正是失去的东西。
        check(d.action != DegradeAction::ReturnHome,
              "陀螺失效：不得给出 ReturnHome（返航需姿态机动，会翻）");
        check(d.max_tilt_deg == 0.0, "陀螺失效：不再做姿态机动（倾角上限 0）");
        check(!d.use_accel_correction, "陀螺失效：停止加速度计校正");
        check(!d.trust_position, "陀螺失效：不再信任位置估计");

        // 即使加速度计同时失效，陀螺失效仍应主导
        const auto d2 = policy.decide(makeReport(SensorStatus::Failed, SensorStatus::Failed));
        check(d2.action == DegradeAction::EmergencyLand, "双失效：仍为 EmergencyLand（陀螺主导）");
    }

    // ================================================================
    std::printf("\n--- 3. 加速度计失效（可降级）---\n");
    // ================================================================
    {
        const auto d = policy.decide(makeReport(SensorStatus::Failed, SensorStatus::Healthy));
        check(d.action == DegradeAction::ReturnHome, "加速度计失效 → ReturnHome");
        check(d.action != DegradeAction::EmergencyLand,
              "加速度计失效：不升级为紧急降落（姿态仍可由陀螺维持）");
        check(!d.use_accel_correction,
              "加速度计失效：停止方向校正（否则把错误方向烧进姿态）");
        check(!d.trust_position, "加速度计失效：位置预积分不可信");
        check(d.max_tilt_deg > 0.0, "加速度计失效：仍保留机动能力（可返航）");
    }

    // ================================================================
    std::printf("\n--- 4. 降级（偏置类，仍可用）---\n");
    // ================================================================
    {
        const auto d = policy.decide(makeReport(SensorStatus::Degraded, SensorStatus::Healthy));
        check(d.action == DegradeAction::Cautious, "加速度计降级 → Cautious");
        check(d.max_tilt_deg < 35.0, "降级：倾角上限被压低（抑制误差放大）");
        check(d.max_speed < 5.0, "降级：速度上限被压低");
        check(d.use_accel_correction, "降级：仍使用校正（数据可用，只是有偏差）");

        const auto d2 = policy.decide(makeReport(SensorStatus::Healthy, SensorStatus::Degraded));
        check(d2.action == DegradeAction::Cautious, "陀螺降级 → Cautious");
    }

    // ================================================================
    std::printf("\n--- 5. Unknown 的处理（易错点）---\n");
    // ================================================================
    {
        // 样本不足时若按 Healthy 放行，等于无证据即假设正常。
        const auto d = policy.decide(makeReport(SensorStatus::Unknown, SensorStatus::Unknown));
        check(d.action == DegradeAction::Cautious, "Unknown → Cautious（不盲目信任）");
        check(d.action != DegradeAction::Normal,
              "Unknown：不得按 Normal 放行（无证据即假设健康是错的）");
        check(d.action != DegradeAction::EmergencyLand,
              "Unknown：也不得过度反应为紧急降落");

        // 单侧 Unknown 同样处理
        const auto d2 = policy.decide(makeReport(SensorStatus::Healthy, SensorStatus::Unknown));
        check(d2.action == DegradeAction::Cautious, "陀螺 Unknown → Cautious");
    }

    // ================================================================
    std::printf("\n--- 6. 限幅的单调性 ---\n");
    // ================================================================
    {
        const auto n = policy.decide(makeReport(SensorStatus::Healthy, SensorStatus::Healthy));
        const auto c = policy.decide(makeReport(SensorStatus::Degraded, SensorStatus::Healthy));
        const auto r = policy.decide(makeReport(SensorStatus::Failed, SensorStatus::Healthy));
        const auto e = policy.decide(makeReport(SensorStatus::Healthy, SensorStatus::Failed));
        // 越紧急，机动能力越小
        check(n.max_tilt_deg > c.max_tilt_deg, "倾角上限：正常 > 谨慎");
        check(r.max_tilt_deg >= e.max_tilt_deg, "倾角上限：返航 >= 紧急降落");
        check(e.max_tilt_deg == 0.0, "紧急降落：倾角上限为 0");
    }

    std::printf("\n========================================\n");
    std::printf("%d / %d checks passed\n", g_pass, g_pass + g_fail);
    std::printf("========================================\n");
    return (g_fail == 0) ? 0 : 1;
}
