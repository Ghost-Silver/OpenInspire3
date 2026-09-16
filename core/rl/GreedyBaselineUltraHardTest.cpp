/**
 * @file GreedyBaselineUltraHardTest.cpp
 * @brief 贪心覆盖基线（ultra-hard）：6 架无人机、120×120×30、含障碍物
 * @author GhostFace
 * @date 2026/4/4
 *
 * @note 2026/9/16 重构：公共评测骨架已抽到 Baseline.h/cpp。原先本文件（464 行）
 *       自带的 Obstacle 定义、障碍物生成、AABB 转换与进阶贪心策略，
 *       现已分别落到 Baseline.h 的 Obstacle / Scenario::obstacles /
 *       advancedGreedySelect 中，此处仅保留场景选择与输出。
 */

#include "Baseline.h"
#include <iostream>

int main() {
    using namespace oi3::rl;

    std::cout << "=== 超难度贪心覆盖基线测试 ===" << std::endl;
    const Scenario scenario = makeUltraHardScenario();
    const EvalResult result = evaluate(scenario, advancedGreedySelect);
    printResult(scenario, result);
    std::cout << "\n预期: AM-MAPPO 应在此场景下显著优于贪心策略" << std::endl;
    return 0;
}
