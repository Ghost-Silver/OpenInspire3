/**
 * @file GreedyBaselineTest.cpp
 * @brief 贪心覆盖基线（easy）：2 架无人机、无障碍、50×50×15
 * @author GhostFace
 * @date 2026/4/4
 *
 * @note 2026/9/16 重构：原先本文件自带一份「环境循环 + 统计」骨架（227 行），
 *       与 Hard / UltraHard / AM-MAPPO 三个测试高度重复。公共部分已抽到
 *       Baseline.h/cpp，此处仅保留场景选择与输出。
 */

#include "Baseline.h"
#include <iostream>

int main() {
    using namespace oi3::rl;

    std::cout << "=== 贪心覆盖基线测试 ===" << std::endl;
    const Scenario scenario = makeEasyScenario();
    const EvalResult result = evaluate(scenario, makeGreedyPolicy());
    printResult(scenario, result);
    return 0;
}
