/**
 * @file GreedyBaselineTest.cpp
 * @brief 贪心覆盖基线测试
 * @author GhostFace
 * @date 2026/4/4
 */

#include "EnvManager.h"
#include <iostream>
#include <vector>
#include <algorithm>

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
    const int local_map_size = 5 * 5 * 3; // 75
    
    // 计算每个方向的未探索体素数
    std::vector<double> direction_scores(7, 0.0);
    
    // 前向 (y+)
    for (int z = 0; z < 3; z++) {
        for (int x = 2; x < 5; x++) {
            int idx = local_map_start + z * 25 + x;
            if (idx < obs.size() && obs[idx] < 0.5) { // 未探索
                direction_scores[0] += 1.0;
            }
        }
    }
    
    // 后向 (y-)
    for (int z = 0; z < 3; z++) {
        for (int x = 0; x < 3; x++) {
            int idx = local_map_start + z * 25 + x;
            if (idx < obs.size() && obs[idx] < 0.5) {
                direction_scores[1] += 1.0;
            }
        }
    }
    
    // 左向 (x-)
    for (int z = 0; z < 3; z++) {
        for (int y = 0; y < 3; y++) {
            int idx = local_map_start + z * 25 + y * 5;
            if (idx < obs.size() && obs[idx] < 0.5) {
                direction_scores[2] += 1.0;
            }
        }
    }
    
    // 右向 (x+)
    for (int z = 0; z < 3; z++) {
        for (int y = 2; y < 5; y++) {
            int idx = local_map_start + z * 25 + y * 5 + 4;
            if (idx < obs.size() && obs[idx] < 0.5) {
                direction_scores[3] += 1.0;
            }
        }
    }
    
    // 上向 (z+)
    for (int y = 1; y < 4; y++) {
        for (int x = 1; x < 4; x++) {
            int idx = local_map_start + 2 * 25 + y * 5 + x;
            if (idx < obs.size() && obs[idx] < 0.5) {
                direction_scores[4] += 1.0;
            }
        }
    }
    
    // 下向 (z-)
    for (int y = 1; y < 4; y++) {
        for (int x = 1; x < 4; x++) {
            int idx = local_map_start + y * 5 + x;
            if (idx < obs.size() && obs[idx] < 0.5) {
                direction_scores[5] += 1.0;
            }
        }
    }
    
    // 悬停：如果当前位置周围有很多未探索区域
    for (int z = 0; z < 3; z++) {
        for (int y = 1; y < 4; y++) {
            for (int x = 1; x < 4; x++) {
                int idx = local_map_start + z * 25 + y * 5 + x;
                if (idx < obs.size() && obs[idx] < 0.5) {
                    direction_scores[6] += 0.5; // 悬停得分较低
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

void testGreedyCoverage() {
    std::cout << "=== 贪心覆盖基线测试 ===" << std::endl;
    
    // 初始化无人机位置（2 架无人机）
    std::vector<Vec3> init_positions;
    init_positions.push_back(Vec3(1.0, 1.0, 0.5)); // 左上角
    init_positions.push_back(Vec3(3.0, 3.0, 0.5)); // 右下角
    
    // 创建环境管理器
    EnvManager env_manager(50, 50, 15, 0.1, 1.5, init_positions, 300);
    
    // 获取基本信息
    env_manager.reset();
    std::vector<std::vector<float>> initial_obs = env_manager.get_local_observations();
    int obs_dim = initial_obs[0].size();
    int num_uavs = init_positions.size();
    
    std::cout << "无人机数量: " << num_uavs << std::endl;
    std::cout << "地图尺寸: 50x50x15" << std::endl;
    std::cout << "目标数量: 5" << std::endl;
    std::cout << "测试 episode: 50" << std::endl;
    
    // 训练参数
    int num_episodes = 50;
    int max_steps_per_episode = 300;
    int success_count = 0;
    
    // 训练循环
    for (int episode = 0; episode < num_episodes; episode++) {
        env_manager.reset();
        std::vector<double> episode_rewards(num_uavs, 0.0);
        bool episode_success = false;
        int episode_steps = 0;
        
        for (int step = 0; step < max_steps_per_episode; step++) {
            // 获取观测
            std::vector<std::vector<float>> observations = env_manager.get_local_observations();
            
            // 为每个无人机选择动作（贪心策略）
            std::vector<Action> actions;
            for (int i = 0; i < num_uavs; i++) {
                Action action = greedy_select_action(observations[i]);
                actions.push_back(action);
            }
            
            // 执行动作
            auto step_result = env_manager.step(actions);
            std::vector<float> rewards = std::get<0>(step_result);
            bool done = std::get<2>(step_result);
            
            // 累计奖励
            for (int i = 0; i < num_uavs; i++) {
                episode_rewards[i] += rewards[i];
            }
            
            // 检查是否成功
            bool all_successful = true;
            for (double reward : episode_rewards) {
                if (reward < 100.0) {
                    all_successful = false;
                    break;
                }
            }
            if (all_successful) {
                episode_success = true;
            }
            
            // 记录步数
            episode_steps = step + 1;
            
            // 如果完成，退出循环
            if (done) {
                break;
            }
        }
        
        // 统计成功次数
        if (episode_success) {
            success_count++;
        }
        
        // 计算总奖励
        double total_reward = 0.0;
        for (double reward : episode_rewards) {
            total_reward += reward;
        }
        
        // 打印训练信息
        std::cout << " Episode " << episode + 1 << ": 总奖励 = " << total_reward 
                  << ", 成功 = " << (episode_success ? "是" : "否") 
                  << ", 步数 = " << episode_steps << std::endl;
        
        // 每 10 个 episode 打印一次成功率
        if ((episode + 1) % 10 == 0) {
            double success_rate = static_cast<double>(success_count) / (episode + 1) * 100.0;
            std::cout << "\n--- 前 " << (episode + 1) << " 个 episode 成功率: " << success_rate << "% ---\n" << std::endl;
        }
    }
    
    // 计算总成功率
    double total_success_rate = static_cast<double>(success_count) / num_episodes * 100.0;
    std::cout << "\n=== 贪心覆盖基线测试完成 ===" << std::endl;
    std::cout << "总成功率: " << total_success_rate << "%" << std::endl;
    std::cout << "\n贪心覆盖基线测试完成！" << std::endl;
}

int main() {
    testGreedyCoverage();
    return 0;
}
