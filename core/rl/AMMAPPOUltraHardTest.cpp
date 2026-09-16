/**
 * @file AMMAPPOUltraHardTest.cpp
 * @brief 超难度 AM-MAPPO 训练测试：在 ultra-hard 场景上训练 PPO，并与贪心基线对比
 * @author GhostFace
 * @date 2026/4/4
 *
 * @note 2026/9/16 重构：
 *   1. 场景参数、障碍物列表与 AABB 转换改为复用 Baseline.h —— 原先本文件
 *      自行复制了一份 Obstacle 定义、一份硬编码障碍物列表与一段 AABB 转换，
 *      与 GreedyBaselineUltraHardTest 完全重复。
 *   2. 删除 generateSparseTargets()：其返回值被直接丢弃，从未参与环境构造。
 *   3. 与贪心基线的对比数据改为现场跑一遍得到 —— 原先是硬编码的字面量，
 *      其数值（平均奖励 8330.64）与实测口径不符，会导致对比结论失真。
 *   4. 成功判据由「总奖励 > 800」改为「全部目标被发现」。
 */

#include "Baseline.h"
#include "EnvManager.h"
#include "PPOAgent.h"

#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <tuple>
#include <vector>

namespace {

using oi3::rl::EvalResult;

/// PPO 超参数
constexpr float kLearningRate = 1e-3f;
constexpr int kHiddenDim = 64;
/// 每 N 个 episode 执行一次参数更新（沿用原实现：攒若干回合再学，降低更新频率）
constexpr int kUpdateInterval = 3;
/// 离散动作数
constexpr int kActionDim = 7;

/// 单回合记录
struct EpisodeRecord {
    double reward = 0.0;
    int steps = 0;
    bool success = false;
};

void printHyperparameters() {
    std::cout << "PPO 超参数:" << std::endl;
    std::cout << "  学习率: " << kLearningRate << std::endl;
    std::cout << "  隐藏层维度: " << kHiddenDim << std::endl;
    std::cout << "  更新间隔: 每 " << kUpdateInterval << " 个 episode" << std::endl;
    std::cout << std::endl;
}

/// 由逐回合记录汇总为评测结果（与 evaluate() 的统计口径一致）
EvalResult summarize(const std::vector<EpisodeRecord> &records) {
    EvalResult result;
    result.episodes = static_cast<int>(records.size());
    if (records.empty()) {
        return result;
    }

    for (const auto &r : records) {
        result.avg_reward += r.reward;
        result.avg_steps += static_cast<double>(r.steps);
        if (r.success) {
            ++result.successes;
        }
    }

    const double n = static_cast<double>(records.size());
    result.avg_reward /= n;
    result.avg_steps /= n;
    result.success_rate = 100.0 * static_cast<double>(result.successes) / n;

    double reward_var = 0.0;
    double steps_var = 0.0;
    for (const auto &r : records) {
        reward_var += (r.reward - result.avg_reward) * (r.reward - result.avg_reward);
        steps_var += (static_cast<double>(r.steps) - result.avg_steps) *
                     (static_cast<double>(r.steps) - result.avg_steps);
    }
    result.reward_std = std::sqrt(reward_var / n);
    result.steps_std = std::sqrt(steps_var / n);
    return result;
}

} // namespace

int main() {
    using namespace oi3::rl;

    const Scenario scenario = makeUltraHardScenario();

    std::cout << "=== 超难度 AM-MAPPO 测试 ===" << std::endl;
    std::cout << "地图 " << scenario.map_x << "x" << scenario.map_y << "x" << scenario.map_z
              << "，无人机 " << scenario.start_positions.size() << "，目标 " << scenario.num_targets
              << "，障碍 " << scenario.obstacles.size() << std::endl;
    std::cout << "传感器半径 " << scenario.sensor_radius << "，最大步数 " << scenario.max_steps
              << "，episode " << scenario.num_episodes << std::endl;
    printHyperparameters();

    // ---- 环境 ----
    EnvManager env(scenario.map_x, scenario.map_y, scenario.map_z, scenario.resolution,
                   scenario.sensor_radius, scenario.start_positions, scenario.max_steps,
                   toAABBs(scenario.obstacles));
    env.set_num_targets(scenario.num_targets);

    // 先 reset 一次并读取观测，用于确定观测维度（与基线测试保持同一调用序）
    env.reset();
    const auto initial_obs = env.get_local_observations();

    const int num_uavs = static_cast<int>(scenario.start_positions.size());
    const int obs_dim = static_cast<int>(initial_obs[0].size());

    std::cout << "观测维度: " << obs_dim << "，动作维度: " << kActionDim << std::endl << std::endl;

    // ---- 智能体 ----
    // reserve 避免容器扩容搬移元素。PPOAgent 的 Optimizer 指向自身网络参数，
    // 搬移必须依赖其移动构造函数完成重新绑定（见 PPOAgent.cpp）。
    std::vector<PPOAgent> agents;
    agents.reserve(static_cast<std::size_t>(num_uavs));
    for (int i = 0; i < num_uavs; ++i) {
        agents.emplace_back(obs_dim, kHiddenDim, kActionDim, kLearningRate);
    }

    std::vector<EpisodeRecord> records;
    records.reserve(static_cast<std::size_t>(scenario.num_episodes));

    const auto start_time = std::chrono::high_resolution_clock::now();
    int update_counter = 0;

    for (int episode = 0; episode < scenario.num_episodes; ++episode) {
        env.reset();

        std::vector<double> rewards(static_cast<std::size_t>(num_uavs), 0.0);
        bool done = false;
        int steps = 0;

        for (int step = 0; step < scenario.max_steps; ++step) {
            const auto observations = env.get_local_observations();

            std::vector<Action> actions;
            std::vector<std::tuple<int, float, float>> samples;
            actions.reserve(static_cast<std::size_t>(num_uavs));
            samples.reserve(static_cast<std::size_t>(num_uavs));

            for (int i = 0; i < num_uavs; ++i) {
                const auto sample =
                    agents[static_cast<std::size_t>(i)].select_action(observations[i]);
                samples.push_back(sample);
                actions.push_back(static_cast<Action>(std::get<0>(sample)));
            }

            const auto step_result = env.step(actions);
            const auto &step_rewards = std::get<0>(step_result);
            done = std::get<2>(step_result);

            for (int i = 0; i < num_uavs; ++i) {
                Experience exp;
                exp.obs = makeObsTensor(observations[i]);
                exp.action = Tensor({static_cast<float>(std::get<0>(samples[i]))});
                exp.log_prob = Tensor({std::get<1>(samples[i])});
                exp.value = Tensor({std::get<2>(samples[i])});
                exp.reward = Tensor({step_rewards[i]});
                exp.done = Tensor({done ? 1.0f : 0.0f});
                agents[static_cast<std::size_t>(i)].store_experience(exp);

                rewards[static_cast<std::size_t>(i)] += step_rewards[i];
            }

            steps = step + 1;
            if (done) {
                break;
            }
        }

        // 成功判据：全部目标被发现（全局状态中的目标发现比例达到 1）。
        // 原实现用「总奖励 > 800」，而本场景单回合总奖励量级在 1e4 以上，
        // 该阈值恒为真，成功率恒为 100%，不具备区分度。
        const auto global_state = env.get_global_state();
        const auto found_ratio_idx = static_cast<std::size_t>(num_uavs) * 3 + 1;
        const bool success = found_ratio_idx < global_state.size() &&
                             global_state[found_ratio_idx] >= 1.0f;

        const double total_reward = std::accumulate(rewards.begin(), rewards.end(), 0.0);
        records.push_back(EpisodeRecord{total_reward, steps, success});

        if (++update_counter % kUpdateInterval == 0) {
            for (auto &agent : agents) {
                agent.update();
            }
        }

        std::cout << " Episode " << (episode + 1) << ": 总奖励 = " << total_reward
                  << ", 成功 = " << (success ? "是" : "否") << ", 步数 = " << steps << std::endl;
    }

    const auto end_time = std::chrono::high_resolution_clock::now();
    const auto elapsed =
        std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();

    // ---- 结果 ----
    Scenario ppo_scenario = scenario;
    ppo_scenario.name = "超难度 AM-MAPPO（ultra-hard）";

    std::cout << "\n总训练时间: " << elapsed << " 秒" << std::endl;
    printResult(ppo_scenario, summarize(records));

    // ---- 与贪心基线对比 ----
    // 现场运行同一场景的进阶贪心基线，避免硬编码对比数据随时间失真。
    std::cout << "\n运行贪心基线以作对比（同一场景、同一回合数）..." << std::endl;
    Scenario baseline_scenario = scenario;
    baseline_scenario.name = "进阶贪心基线（ultra-hard）";
    const EvalResult baseline = evaluate(baseline_scenario, advancedGreedySelect, false);

    const EvalResult am_mappo = summarize(records);
    printResult(baseline_scenario, baseline);

    std::cout << "\n========== 对比结论 ==========" << std::endl;
    std::cout << "进阶贪心: 奖励 " << baseline.avg_reward << "，成功率 "
              << baseline.success_rate << "%" << std::endl;
    std::cout << "AM-MAPPO: 奖励 " << am_mappo.avg_reward << "，成功率 "
              << am_mappo.success_rate << "%" << std::endl;
    if (am_mappo.avg_reward >= baseline.avg_reward) {
        std::cout << "AM-MAPPO 平均奖励不低于进阶贪心基线。" << std::endl;
    } else {
        std::cout << "AM-MAPPO 平均奖励低于进阶贪心基线，需继续训练或调参。" << std::endl;
    }

    return 0;
}
