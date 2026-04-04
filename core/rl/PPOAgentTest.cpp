/**
 * @file PPOAgentTest.cpp
 * @brief PPO 智能体测试，验证梯度传播和收敛性
 * @author GhostFace
 * @date 2026/4/4
 */

#include "PPOAgent.h"
#include "EnvManager.h"
#include <iostream>
#include <vector>

void testPPOAgent() {
    std::cout << "=== PPO 智能体测试 ===" << std::endl;
    
    // 初始化无人机位置（2 架无人机）
    std::vector<Vec3> init_positions;
    init_positions.push_back(Vec3(1.0, 1.0, 0.5)); // 左上角
    init_positions.push_back(Vec3(3.0, 3.0, 0.5)); // 右下角
    
    // 创建环境管理器（更大地图和更多目标）
    EnvManager env_manager(50, 50, 15, 0.1, 1.5, init_positions, 300);
    
    // 获取观测维度
    env_manager.reset();
    std::vector<std::vector<float>> initial_obs = env_manager.get_local_observations();
    int obs_dim = initial_obs[0].size();
    int action_dim = 7; // 7 个离散动作
    int hidden_dim = 64;
    int num_uavs = init_positions.size();
    
    std::cout << "观测维度: " << obs_dim << std::endl;
    std::cout << "动作维度: " << action_dim << std::endl;
    std::cout << "无人机数量: " << num_uavs << std::endl;
    std::cout << "地图尺寸: 50x50x15" << std::endl;
    std::cout << "目标数量: 5" << std::endl;
    
    // 创建 PPO 智能体（每个无人机一个）
    std::vector<PPOAgent> agents;
    for (int i = 0; i < num_uavs; i++) {
        agents.emplace_back(obs_dim, hidden_dim, action_dim, 1e-4); // 降低学习率
    }
    
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
            
            // 为每个无人机选择动作
            std::vector<Action> actions;
            for (int i = 0; i < num_uavs; i++) {
                auto [action_idx, log_prob, value] = agents[i].select_action(observations[i]);
                actions.push_back(static_cast<Action>(action_idx));
            }
            
            // 执行动作
            auto step_result = env_manager.step(actions);
            std::vector<float> rewards = std::get<0>(step_result);
            std::vector<std::vector<float>> next_observations = std::get<1>(step_result);
            bool done = std::get<2>(step_result);
            
            // 存储经验
            for (int i = 0; i < num_uavs; i++) {
                // 创建持久化的张量，确保其生命周期足够长
                Tensor obs_tensor(ShapeTag(), {1, static_cast<size_t>(obs_dim)});
                try {
                    float* obs_data = obs_tensor.data<float>();
                    for (size_t j = 0; j < observations[i].size(); j++) {
                        obs_data[j] = observations[i][j];
                    }
                } catch (const std::exception& e) {
                    std::cerr << "创建观测张量失败: " << e.what() << std::endl;
                    continue;
                }
                
                Tensor action_tensor(ShapeTag(), {1});
                action_tensor.data<float>()[0] = static_cast<int>(actions[i]);
                
                // 重新选择动作以获取 log_prob 和 value
                auto [action_idx, log_prob, value] = agents[i].select_action(observations[i]);
                
                Tensor log_prob_tensor(log_prob);
                Tensor value_tensor(value);
                Tensor reward_tensor(rewards[i]);
                Tensor done_tensor(done ? 1.0f : 0.0f);
                
                Experience exp = {
                    obs_tensor,
                    action_tensor,
                    log_prob_tensor,
                    value_tensor,
                    reward_tensor,
                    done_tensor
                };
                agents[i].store_experience(exp);
                
                // 累计奖励
                episode_rewards[i] += rewards[i];
            }
            
            // 检查是否成功（所有无人机的奖励都大于阈值）
            bool all_successful = true;
            for (double reward : episode_rewards) {
                if (reward < 100.0) { // 提高成功阈值
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
        
        // 更新策略
        for (int i = 0; i < num_uavs; i++) {
            agents[i].update();
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
        
        // 打印详细训练信息
        std::cout << " Episode " << episode + 1 << ": 总奖励 = " << total_reward << ", 成功 = " << (episode_success ? "是" : "否") << ", 步数 = " << episode_steps << std::endl;
        for (int i = 0; i < num_uavs; i++) {
            std::cout << "   无人机 " << i + 1 << " 奖励: " << episode_rewards[i] << std::endl;
        }
        
        // 每 10 个 episode 打印一次成功率
        if ((episode + 1) % 10 == 0) {
            double success_rate = static_cast<double>(success_count) / (episode + 1) * 100.0;
            std::cout << "\n--- 前 " << (episode + 1) << " 个 episode 成功率: " << success_rate << "% ---\n" << std::endl;
        }
    }
    
    // 计算总成功率
    double total_success_rate = static_cast<double>(success_count) / num_episodes * 100.0;
    std::cout << "\n=== 训练完成 ===" << std::endl;
    std::cout << "总成功率: " << total_success_rate << "%" << std::endl;
    std::cout << "预期成功率 > 80%: " << (total_success_rate > 80.0 ? "达成" : "未达成") << std::endl;
    
    std::cout << "\nPPO 智能体测试完成！" << std::endl;
}

int main() {
    testPPOAgent();
    return 0;
}
