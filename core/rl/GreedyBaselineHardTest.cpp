/**
 * @file GreedyBaselineHardTest.cpp
 * @brief 贪心覆盖基线（hard）：4 架无人机、80×80×20、5 个目标
 * @author GhostFace
 * @date 2026/4/4
 *
 * @note 2026/9/16 重构：公共评测骨架已抽到 Baseline.h/cpp，此处仅保留场景选择。
 *       悬停权重沿用原实现的本文件口径 0.3（easy 为 0.5）。
 */

#include "Baseline.h"
#include <iostream>

int main() {
    using namespace oi3::rl;

    std::cout << "=== 高难度贪心覆盖基线测试 ===" << std::endl;
    const Scenario scenario = makeHardScenario();
    const EvalResult result = evaluate(scenario, makeGreedyPolicy(0.3));
    printResult(scenario, result);
    return 0;
}
