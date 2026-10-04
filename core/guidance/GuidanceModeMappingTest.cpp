/**
 * @file GuidanceModeMappingTest.cpp
 * @brief 制导模式映射完整性：四个降级动作 × 三个起始模式
 *
 * @par 为什么需要
 *
 * `GuidanceSetpointSource::onDecisionChanged` 原先把 `DegradeAction` 的
 * **四个枚举值只处理了三个** —— `Cautious` 因「不匹配任何分支」而隐式保持
 * 当前模式。行为恰好正确，但语义是隐式的：枚举新增取值或调整分级时会静默
 * 出错（表现为制导模式与降级决策不一致，且没有任何提示）。
 *
 * 本测试固化完整映射矩阵，把设计意图变成可执行的约束：
 *
 * | 决策 | 制导模式 |
 * |---|---|
 * | `Normal` | Mission |
 * | `Cautious` | 保持当前模式 |
 * | `ReturnHome` | ReturnHome |
 * | `EmergencyLand` | EmergencyLand |
 *
 * 其中 `Cautious` 保持当前模式是**有意设计**而非疏漏：`Cautious` 是轻降级，
 * 可能出现在返航途中（故障由 Failed 减轻为 Degraded）。已开始的安全动作
 * 应当执行完，中途折返任务点会让飞机停在半路，反而更不安全。
 */

#include "GuidanceSetpointSource.h"
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
const char *modeName(GuidanceMode m) {
    switch (m) {
    case GuidanceMode::Mission: return "Mission";
    case GuidanceMode::ReturnHome: return "ReturnHome";
    case GuidanceMode::EmergencyLand: return "EmergencyLand";
    }
    return "?";
}
const char *actionName(DegradeAction a) {
    switch (a) {
    case DegradeAction::Normal: return "Normal";
    case DegradeAction::Cautious: return "Cautious";
    case DegradeAction::ReturnHome: return "ReturnHome";
    case DegradeAction::EmergencyLand: return "EmergencyLand";
    }
    return "?";
}
} // namespace

int main() {
    std::printf("=== GuidanceModeMappingTest: 决策 → 制导模式映射矩阵 ===\n\n");

    const Tensor mission = makeVec3(0.0f, 0.0f, -5.0f);
    const std::array<double, 3> home = {0.0, 0.0, -5.0};
    TrajectoryLimits limits;

    // 辅助：把制导源置入指定起始模式
    auto makeAt = [&](GuidanceMode start) {
        if (start == GuidanceMode::Mission) {
            return GuidanceSetpointSource(mission, home, limits);
        }
        GuidanceSetpointSource sp(mission, home, limits);
        DegradeDecision d;
        d.action = (start == GuidanceMode::ReturnHome) ? DegradeAction::ReturnHome
                                                       : DegradeAction::EmergencyLand;
        sp.onDecisionChanged(d, {3.0, 4.0, -5.0}, 0.0);
        return sp;
    };

    struct Case {
        GuidanceMode start;
        DegradeAction action;
        GuidanceMode expect;
    };
    const Case cases[12] = {
        {GuidanceMode::Mission, DegradeAction::Normal, GuidanceMode::Mission},
        {GuidanceMode::Mission, DegradeAction::Cautious, GuidanceMode::Mission},
        {GuidanceMode::Mission, DegradeAction::ReturnHome, GuidanceMode::ReturnHome},
        {GuidanceMode::Mission, DegradeAction::EmergencyLand, GuidanceMode::EmergencyLand},

        {GuidanceMode::ReturnHome, DegradeAction::Normal, GuidanceMode::Mission},
        {GuidanceMode::ReturnHome, DegradeAction::Cautious, GuidanceMode::ReturnHome},
        {GuidanceMode::ReturnHome, DegradeAction::ReturnHome, GuidanceMode::ReturnHome},
        {GuidanceMode::ReturnHome, DegradeAction::EmergencyLand, GuidanceMode::EmergencyLand},

        {GuidanceMode::EmergencyLand, DegradeAction::Normal, GuidanceMode::Mission},
        {GuidanceMode::EmergencyLand, DegradeAction::Cautious, GuidanceMode::EmergencyLand},
        {GuidanceMode::EmergencyLand, DegradeAction::ReturnHome, GuidanceMode::ReturnHome},
        {GuidanceMode::EmergencyLand, DegradeAction::EmergencyLand, GuidanceMode::EmergencyLand},
    };

    std::printf("%-14s + %-14s → %-14s %s\n", "起始模式", "决策", "实际", "期望");
    for (const auto &c : cases) {
        GuidanceSetpointSource sp = makeAt(c.start);
        DegradeDecision d;
        d.action = c.action;
        sp.onDecisionChanged(d, {2.0, 3.0, -5.0}, 1.0);
        const bool ok = (sp.mode() == c.expect);
        std::printf("%-14s + %-14s → %-14s %s\n", modeName(c.start), actionName(c.action),
                    modeName(sp.mode()), ok ? "✓" : "✗");

        char buf[220];
        std::snprintf(buf, sizeof(buf), "映射：%s + %s → %s", modeName(c.start),
                      actionName(c.action), modeName(c.expect));
        check(ok, buf);
    }

    // ---- 关键单点：Cautious 保持当前模式（有意设计）----
    std::printf("\n");
    for (GuidanceMode start : {GuidanceMode::ReturnHome, GuidanceMode::EmergencyLand}) {
        GuidanceSetpointSource sp = makeAt(start);
        DegradeDecision d;
        d.action = DegradeAction::Cautious;
        sp.onDecisionChanged(d, {2.0, 3.0, -5.0}, 1.0);

        char buf[260];
        std::snprintf(buf, sizeof(buf),
                      "Cautious 保持当前模式 %s（安全动作执行完，不中途折返）",
                      modeName(start));
        check(sp.mode() == start, buf);
    }

    // ---- Normal 必须能退出任何降级模式（恢复链路的关键）----
    std::printf("\n");
    for (GuidanceMode start : {GuidanceMode::ReturnHome, GuidanceMode::EmergencyLand}) {
        GuidanceSetpointSource sp = makeAt(start);
        DegradeDecision d;
        d.action = DegradeAction::Normal;
        sp.onDecisionChanged(d, {2.0, 3.0, -5.0}, 1.0);

        char buf[260];
        std::snprintf(buf, sizeof(buf), "Normal 能从 %s 退出回到 Mission", modeName(start));
        check(sp.mode() == GuidanceMode::Mission, buf);
    }

    std::printf("\n========================================\n");
    std::printf("%d / %d checks passed\n", g_pass, g_pass + g_fail);
    std::printf("========================================\n");
    return g_fail == 0 ? 0 : 1;
}
