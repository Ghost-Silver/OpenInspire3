/**
 * @file DegradeExecutorTest.cpp
 * @brief 降级执行器的验证：透传正确性、接管正确性、限幅正确性
 *
 * @par 最重要的一条断言
 *
 * **Normal 模式下必须与内层控制器逐位一致**。执行器包在 PID 外面，若它在
 * 正常飞行时对指令有任何改动，就等于改动了整个飞控的既有行为——这违反项目
 * 「新效应默认不改变既有结果」的铁律，而且这种改动极难发现（数值只差一点）。
 * 因此这里用逐位比较（而非容差比较）来固定这条性质。
 *
 * @par 关于「激励确认」
 *
 * 验证紧急降落接管时，必须同时确认内层控制器**确实没有被调用**（而非调用了
 * 但结果恰好相同）。这里用一个可观测的桩控制器：它被调用时会累加计数并输出
 * 一个特征值。若断言只看输出，无法区分「没调用」与「调用了但输出相同」。
 */

#include "DegradeExecutor.h"
#include "SixDofPidController.h"
#include "SixDofSimulator.h"
#include "SixDofTypes.h"
#include "TensorUtils.h"
#include "DroneTypes.h"

#include <cmath>
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

std::array<double, 3> readVec(const Tensor &t) {
    const std::vector<float> v = toVector(t);
    return {static_cast<double>(v[0]), static_cast<double>(v[1]), static_cast<double>(v[2])};
}

/// 可观测的桩控制器：记录被调用次数，输出特征值
class ProbeController : public SixDofController {
  public:
    [[nodiscard]] SixDofCommand compute(const SixDofState &, const Tensor &, double) override {
        ++calls;
        SixDofCommand c;
        c.thrust_body = 12.345;                  // 特征推力
        c.torque = makeVec3(1.0f, -2.0f, 3.0f);  // 特征力矩
        return c;
    }
    [[nodiscard]] const char *name() const override { return "ProbeController"; }

    int calls = 0;
};

DegradeDecision makeDecision(DegradeAction a, double tilt) {
    DegradeDecision d;
    d.action = a;
    d.max_tilt_deg = tilt;
    return d;
}

} // namespace

int main() {
    std::printf("=== DegradeExecutorTest: 降级执行器 ===\n\n");

    const Tensor q = Tensor{1.0f, 0.0f, 0.0f, 0.0f};
    const Tensor z3 = makeVec3(0.0f, 0.0f, 0.0f);
    const SixDofState st{q, z3, q, z3};
    const Tensor target = makeVec3(0.0f, 0.0f, -5.0f);

    DegradeExecutorConfig cfg;
    cfg.hover_thrust = 5.0 * 9.81; // 按 5 kg 机体算

    // ================================================================
    std::printf("--- 1. Normal：必须逐位透传（不得改变既有行为）---\n");
    // ================================================================
    {
        ProbeController probe;
        DegradeExecutor exec(probe, cfg);
        // 默认即 Normal，无需显式设置
        const SixDofCommand c1 = exec.compute(st, target, 0.0);
        const SixDofCommand c2 = probe.compute(st, target, 0.0);

        check(probe.calls == 2, "Normal：内层控制器确实被调用（两次）");
        // 逐位比较，不用容差——任何数值改动都应被这条断言拦住
        check(c1.thrust_body == c2.thrust_body, "Normal：推力逐位一致");
        const auto t1 = readVec(c1.torque);
        const auto t2 = readVec(c2.torque);
        check(t1[0] == t2[0] && t1[1] == t2[1] && t1[2] == t2[2], "Normal：力矩逐位一致");

        // 显式设为 Normal 也应透传
        exec.setDecision(makeDecision(DegradeAction::Normal, 35.0));
        const SixDofCommand c3 = exec.compute(st, target, 0.0);
        const SixDofCommand c4 = probe.compute(st, target, 0.0);
        check(c3.thrust_body == c4.thrust_body, "显式 Normal：推力仍逐位一致");
    }

    // ================================================================
    std::printf("\n--- 2. EmergencyLand：完全接管 ---\n");
    // ================================================================
    {
        ProbeController probe;
        DegradeExecutor exec(probe, cfg);
        exec.setDecision(makeDecision(DegradeAction::EmergencyLand, 0.0));

        const int calls_before = probe.calls;
        const SixDofCommand c = exec.compute(st, target, 0.0);

        // 激励确认：内层控制器**不得**被调用。
        // 只断言输出无法区分「没调用」与「调用了但输出恰好相同」，
        // 故必须检查调用计数——这是本测试用桩控制器的原因。
        check(probe.calls == calls_before,
              "紧急降落：内层控制器未被调用（用调用计数确认，非看输出）");
        check(exec.emergency(), "紧急降落：状态标志正确");

        // 推力应低于悬停，使飞机下降
        check(c.thrust_body < cfg.hover_thrust, "紧急降落：推力低于悬停推力（会下降）");
        check(std::fabs(c.thrust_body - cfg.hover_thrust * cfg.emergency_thrust_ratio) < 1e-9,
              "紧急降落：推力等于配置的紧急系数");

        // 力矩必须为零——这正是让飞机保持姿态的关键
        const auto tq = readVec(c.torque);
        check(tq[0] == 0.0 && tq[1] == 0.0 && tq[2] == 0.0,
              "紧急降落：力矩为零（切断姿态控制是保持姿态的关键）");
    }

    // ================================================================
    std::printf("\n--- 3. Cautious / ReturnHome：调用内层但限幅 ---\n");
    // ================================================================
    {
        ProbeController probe;
        DegradeExecutor exec(probe, cfg);
        exec.setDecision(makeDecision(DegradeAction::Cautious, 15.0)); // 标称 35 度

        const int calls_before = probe.calls;
        const SixDofCommand c = exec.compute(st, target, 0.0);
        check(probe.calls == calls_before + 1, "谨慎：内层控制器被调用");

        // 15/35 的比例缩放
        const double ratio = 15.0 / 35.0;
        const auto tq = readVec(c.torque);
        check(std::fabs(tq[0] - 1.0 * ratio) < 1e-6, "谨慎：力矩按倾角比例缩放（x 轴）");
        check(std::fabs(tq[1] - (-2.0) * ratio) < 1e-6, "谨慎：力矩按倾角比例缩放（y 轴）");
        // 推力不参与限幅（限的是机动能力而非升力）
        check(c.thrust_body == 12.345, "谨慎：推力不受限幅影响");

        // 倾角上限为 0 时力矩应完全归零
        exec.setDecision(makeDecision(DegradeAction::ReturnHome, 0.0));
        const SixDofCommand c2 = exec.compute(st, target, 0.0);
        const auto tq2 = readVec(c2.torque);
        check(tq2[0] == 0.0 && tq2[1] == 0.0 && tq2[2] == 0.0,
              "倾角上限 0：力矩完全归零");
    }

    // ================================================================
    std::printf("\n--- 4. 限幅的单调性 ---\n");
    // ================================================================
    {
        ProbeController probe;
        DegradeExecutor exec(probe, cfg);
        auto magOf = [&](double tilt) {
            exec.setDecision(makeDecision(DegradeAction::Cautious, tilt));
            const auto tq = readVec(exec.compute(st, target, 0.0).torque);
            return std::sqrt(tq[0] * tq[0] + tq[1] * tq[1] + tq[2] * tq[2]);
        };
        const double m10 = magOf(10.0);
        const double m20 = magOf(20.0);
        const double m35 = magOf(35.0);
        check(m10 < m20 && m20 < m35, "限幅单调：倾角上限越大，力矩越大");
        // 超过标称值时截断到 1（不放大）
        const double m50 = magOf(50.0);
        check(std::fabs(m50 - m35) < 1e-6, "限幅截断：超过标称倾角不再放大");
    }

    std::printf("\n========================================\n");
    std::printf("%d / %d checks passed\n", g_pass, g_pass + g_fail);
    std::printf("========================================\n");
    return (g_fail == 0) ? 0 : 1;
}
