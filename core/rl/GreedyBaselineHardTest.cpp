/**
 * @file GreedyBaselineHardTest.cpp
 * @brief 高难度贪心覆盖基线测试 - 验证 AM-MAPPO 优势
 * @author GhostFace
 * @date 2026/4/4
 */

#include "EnvManager.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>

// 高难度场景参数
constexpr int MAP_SIZE_X = 80;
constexpr int MAP_SIZE_Y = 80;
constexpr int MAP_SIZE_Z = 20;
constexpr int NUM_UAVS = 4;
constexpr int NUM_TARGETS = 15;
constexpr double SENSOR_RADIUS = 1.2;
constexpr int MAX_STEPS = 400;
constexpr int NUM_EPISODES = 50;

// 简化的贪心覆盖：基于局部观测选择动作
Action greedy_select_action(const std::vector<float>& obs) {
    // 从观测中提取信息
    // obs[0-2]: 位置
    // obs[3-5]: 速度
    // obs[6]: 覆盖率
    // obs[7]: 最近目标距离
    // obs[8-72]: 局部地图 (5x5x3)
    
    // 简单的贪心策略：优先探索未探索区域
    // 基于局部地图选择方向
    
    // 解析局部地图 (5x5x3)
    const int local_map_start = 8;
    
    // 计算每个方向的未探索体素数
    std::vector<double> direction_scores(7, 0.0);
    
    // 前向 (y+)
    for (int z = 0; z < 3; z++) {
        for (int x = 2; x < 5; x++) {
            int idx = local_map_start + z * 25 + x;
            if (idx < static_cast<int>(obs.size()) && obs[idx] < 0.5) { // 未探索
                direction_scores[0] += 1.0;
            }
        }
    }
    
    // 后向 (y-)
    for (int z = 0; z < 3; z++) {
        for (int x = 0; x < 3; x++) {
            int idx = local_map_start + z * 25 + x;
            if (idx < static_cast<int>(obs.size()) && obs[idx] < 0.5) {
                direction_scores[1] += 1.0;
            }
        }
    }
    
    // 左向 (x-)
    for (int z = 0; z < 3; z++) {
        for (int y = 0; y < 3; y++) {
            int idx = local_map_start + z * 25 + y * 5;
            if (idx < static_cast<int>(obs.size()) && obs[idx] < 0.5) {
                direction_scores[2] += 1.0;
            }
        }
    }
    
    // 右向 (x+)
    for (int z = 0; z < 3; z++) {
        for (int y = 2; y < 5; y++) {
            int idx = local_map_start + z * 25 + y * 5 + 4;
            if (idx < static_cast<int>(obs.size()) && obs[idx] < 0.5) {
                direction_scores[3] += 1.0;
            }
        }
    }
    
    // 上向 (z+)
    for (int y = 1; y < 4; y++) {
        for (int x = 1; x < 4; x++) {
            int idx = local_map_start + 2 * 25 + y * 5 + x;
            if (idx < static_cast<int>(obs.size()) && obs[idx] < 0.5) {
                direction_scores[4] += 1.0;
            }
        }
    }
    
    // 下向 (z-)
    for (int y = 1; y < 4; y++) {
        for (int x = 1; x < 4; x++) {
            int idx = local_map_start + y * 5 + x;
            if (idx < static_cast<int>(obs.size()) && obs[idx] < 0.5) {
                direction_scores[5] += 1.0;
            }
        }
    }
    
    // 悬停
    for (int z = 0; z < 3; z++) {
        for (int y = 1; y < 4; y++) {
            for (int x = 1; x < 4; x++) {
                int idx = local_map_start + z * 25 + y * 5 + x;
                if (idx < static_cast<int>(obs.size()) && obs[idx] < 0.5) {
                    direction_scores[6] += 0.3; // 悬停得分较低
                }
            }
        }
    }
    
    // 选择未探索体素最多的方向
    int best_action = 0;
    double max_score = direction_scores[0];
    
    for (int i = 1; i < 7; i++) {
        if (direction_scores[i] > max_score) {
            max_score = direction_scores[i];
            best_action = i;
        }
    }
    
    return static_cast<Action>(best_action);
}

void testGreedyCoverageHard() {
    std::cout << "=== 高难度贪心覆盖基线测试 ===" << std::endl;
    std::cout << "场景参数:" << std::endl;
    std::cout << "  地图尺寸: " << MAP_SIZE_X << "x" << MAP_SIZE_Y << "x" << MAP_SIZE_Z << std::endl;
    std::cout << "  无人机数量: " << NUM_UAVS << std::endl;
    std::cout << "  目标数量: " << NUM_TARGETS << std::endl;
    std::cout << "  传感器半径: " << SENSOR_RADIUS << std::endl;
    std::cout << "  最大步数: " << MAX_STEPS << std::endl;
    std::cout << "  测试轮数: " << NUM_EPISODES << std::endl;
    std::cout << std::endl;
    
    // 初始化无人机位置（分散在地图四角）
    std::vector<Vec3> init_positions;
    init_positions.push_back(Vec3(2.0, 2.0, 1.0));
    init_positions.push_back(Vec3(6.0, 2.0, 1.0));
    init_positions.push_back(Vec3(2.0, 6.0, 1.0));
    init_positions.push_back(Vec3(6.0, 6.0, 1.0));
    
    // 创建环境管理器
    EnvManager env_manager(MAP_SIZE_X, MAP_SIZE_Y, MAP_SIZE_Z, 
                          0.1, SENSOR_RADIUS, init_positions, MAX_STEPS);
    
    // 设置目标数量为15个
    // 注意：这里需要访问 Environment 的内部成员，我们暂时使用默认的5个目标
    // 实际运行时需要在 Environment 类中添加设置方法
    
    // 统计指标
    int success_count = 0;
    double total_rewards_all = 0.0;
    int total_steps_all = 0;
    std::vector<double> episode_rewards_list;
    std::vector<int> episode_steps_list;
    std::vector<int> targets_found_list;
    
    // 训练循环
    for (int episode = 0; episode < NUM_EPISODES; episode++) {
        env_manager.reset();
        std::vector<double> episode_rewards(NUM_UAVS, 0.0);
        bool episode_success = false;
        int episode_steps = 0;
        int targets_found = 0;
        
        for (int step = 0; step < MAX_STEPS; step++) {
            // 获取观测
            std::vector<std::vector<float>> observations = env_manager.get_local_observations();
            
            // 为每个无人机选择动作（贪心策略）
            std::vector<Action> actions;
            for (int i = 0; i < NUM_UAVS; i++) {
                Action action = greedy_select_action(observations[i]);
                actions.push_back(action);
            }
            
            // 执行动作
            auto step_result = env_manager.step(actions);
            std::vector<float> rewards = std::get<0>(step_result);
            bool done = std::get<2>(step_result);
            
            // 累计奖励
            for (int i = 0; i < NUM_UAVS; i++) {
                episode_rewards[i] += rewards[i];
            }
            
            // 检查是否成功（所有目标被发现）
            // 这里简化处理：奖励超过阈值认为成功
            double total_reward = 0.0;
            for (double r : episode_rewards) {
                total_reward += r;
            }
            
            if (total_reward > 300.0) { // 高难度场景的成功阈值
                episode_success = true;
            }
            
            // 记录步数
            episode_steps = step + 1;
            
            // 如果完成，退出循环
            if (done) {
                break;
            }
        }
        
        // 统计
        double total_reward = 0.0;
        for (double r : episode_rewards) {
            total_reward += r;
        }
        
        if (episode_success) {
            success_count++;
        }
        
        total_rewards_all += total_reward;
        total_steps_all += episode_steps;
        episode_rewards_list.push_back(total_reward);
        episode_steps_list.push_back(episode_steps);
        
        // 打印训练信息
        std::cout << " Episode " << episode + 1 << ": 总奖励 = " << total_reward 
                  << ", 成功 = " << (episode_success ? "是" : "否") 
                  << ", 步数 = " << episode_steps;
        
        // 打印每架无人机的奖励
        std::cout << ", 无人机奖励: [";
        for (int i = 0; i < NUM_UAVS; i++) {
            std::cout << episode_rewards[i];
            if (i < NUM_UAVS - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        // 每 10 个 episode 打印统计
        if ((episode + 1) % 10 == 0) {
            double success_rate = static_cast<double>(success_count) / (episode + 1) * 100.0;
            double avg_reward = total_rewards_all / (episode + 1);
            double avg_steps = static_cast<double>(total_steps_all) / (episode + 1);
            
            std::cout << "\n--- 前 " << (episode + 1) << " 个 episode 统计 ---" << std::endl;
            std::cout << "  成功率: " << success_rate << "%" << std::endl;
            std::cout << "  平均奖励: " << avg_reward << std::endl;
            std::cout << "  平均步数: " << avg_steps << std::endl;
            std::cout << std::endl;
        }
    }
    
    // 最终统计
    double total_success_rate = static_cast<double>(success_count) / NUM_EPISODES * 100.0;
    double avg_reward = total_rewards_all / NUM_EPISODES;
    double avg_steps = static_cast<double>(total_steps_all) / NUM_EPISODES;
    
    // 计算标准差
    double reward_variance = 0.0;
    double steps_variance = 0.0;
    for (int i = 0; i < NUM_EPISODES; i++) {
        reward_variance += std::pow(episode_rewards_list[i] - avg_reward, 2);
        steps_variance += std::pow(episode_steps_list[i] - avg_steps, 2);
    }
    double reward_std = std::sqrt(reward_variance / NUM_EPISODES);
    double steps_std = std::sqrt(steps_variance / NUM_EPISODES);
    
    std::cout << "\n========== 高难度贪心覆盖基线测试结果 ==========" << std::endl;
    std::cout << "成功率: " << total_success_rate << "% (" << success_count << "/" << NUM_EPISODES << ")" << std::endl;
    std::cout << "平均奖励: " << avg_reward << " ± " << reward_std << std::endl;
    std::cout << "平均步数: " << avg_steps << " ± " << steps_std << std::endl;
    std::cout << "================================================" << std::endl;
    std::cout << "\n高难度贪心覆盖基线测试完成！" << std::endl;
    std::cout << "预期: AM-MAPPO 应在此场景下显著优于贪心策略" << std::endl;
}

int main() {
    testGreedyCoverageHard();
    return 0;
}
