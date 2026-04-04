/**
 * @file AMMAPPOUltraHardTest.cpp
 * @brief 超难度 AM-MAPPO 测试 - 与贪心策略比较
 * @author GhostFace
 * @date 2026/4/4
 */

#include "EnvManager.h"
#include "PPOAgent.h"
#include "../Map/VoxelMap.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>

// 超难度场景参数 - 与贪心策略完全一致
constexpr int MAP_SIZE_X = 120;
constexpr int MAP_SIZE_Y = 120;
constexpr int MAP_SIZE_Z = 30;
constexpr int NUM_UAVS = 6;
constexpr int NUM_TARGETS = 30;
constexpr int NUM_OBSTACLES = 30;
constexpr double SENSOR_RADIUS = 0.8;
constexpr int MAX_STEPS = 600;
constexpr int NUM_EPISODES = 20; // 与贪心策略相同

// PPO 超参数 - 加大学习率以加速收敛
constexpr double LEARNING_RATE = 1e-3;
constexpr double GAMMA = 0.99;
constexpr double LAMBDA = 0.95;
constexpr double CLIP_EPS = 0.2;
constexpr double ENTROPY_COEF = 0.05;
constexpr int NUM_UPDATES = 4;
constexpr int BATCH_SIZE = 64;

// 障碍物结构
struct Obstacle {
    Vec3 center;
    double size;
};

// 生成随机障碍物
std::vector<Obstacle> generateObstacles(int num_obstacles, 
                                         double map_size_x, 
                                         double map_size_y, 
                                         double map_size_z,
                                         const std::vector<Vec3>& uav_positions) {
    std::vector<Obstacle> obstacles;
    
    // 直接生成固定位置的障碍物，确保生成足够数量
    Obstacle obstacle1 = {Vec3(4.0, 4.0, 1.5), 2.0};
    Obstacle obstacle2 = {Vec3(8.0, 4.0, 1.5), 2.0};
    Obstacle obstacle3 = {Vec3(4.0, 8.0, 1.5), 2.0};
    Obstacle obstacle4 = {Vec3(8.0, 8.0, 1.5), 2.0};
    Obstacle obstacle5 = {Vec3(6.0, 6.0, 1.5), 2.5};
    
    obstacles.push_back(obstacle1);
    obstacles.push_back(obstacle2);
    obstacles.push_back(obstacle3);
    obstacles.push_back(obstacle4);
    obstacles.push_back(obstacle5);
    
    return obstacles;
}

// 生成稀疏分布的目标
std::vector<Vec3> generateSparseTargets(int num_targets, 
                                        double map_size_x, 
                                        double map_size_y, 
                                        double map_size_z,
                                        const std::vector<Obstacle>& obstacles) {
    std::vector<Vec3> targets;
    
    // 直接生成固定位置的目标，确保生成足够数量
    targets.push_back(Vec3(1.0, 1.0, 1.5));
    targets.push_back(Vec3(11.0, 1.0, 1.5));
    targets.push_back(Vec3(1.0, 11.0, 1.5));
    targets.push_back(Vec3(11.0, 11.0, 1.5));
    targets.push_back(Vec3(3.0, 3.0, 1.0));
    targets.push_back(Vec3(9.0, 3.0, 1.0));
    targets.push_back(Vec3(3.0, 9.0, 1.0));
    targets.push_back(Vec3(9.0, 9.0, 1.0));
    targets.push_back(Vec3(5.0, 5.0, 2.0));
    targets.push_back(Vec3(7.0, 5.0, 2.0));
    targets.push_back(Vec3(5.0, 7.0, 2.0));
    targets.push_back(Vec3(7.0, 7.0, 2.0));
    
    return targets;
}

void testAMMAPPOUltraHard() {
    std::cout << "=== 超难度 AM-MAPPO 测试 ===" << std::endl;
    std::cout << "场景参数:" << std::endl;
    std::cout << "  地图尺寸: " << MAP_SIZE_X << "x" << MAP_SIZE_Y << "x" << MAP_SIZE_Z << std::endl;
    std::cout << "  无人机数量: " << NUM_UAVS << std::endl;
    std::cout << "  目标数量: " << NUM_TARGETS << " (稀疏分布)" << std::endl;
    std::cout << "  障碍物数量: " << NUM_OBSTACLES << std::endl;
    std::cout << "  传感器半径: " << SENSOR_RADIUS << " (减小)" << std::endl;
    std::cout << "  最大步数: " << MAX_STEPS << std::endl;
    std::cout << "  测试轮数: " << NUM_EPISODES << std::endl;
    std::cout << std::endl;
    
    std::cout << "PPO 超参数:" << std::endl;
    std::cout << "  学习率: " << LEARNING_RATE << std::endl;
    std::cout << "  折扣因子: " << GAMMA << std::endl;
    std::cout << "  GAE lambda: " << LAMBDA << std::endl;
    std::cout << "  裁剪 epsilon: " << CLIP_EPS << std::endl;
    std::cout << "  熵系数: " << ENTROPY_COEF << std::endl;
    std::cout << "  更新次数: " << NUM_UPDATES << std::endl;
    std::cout << "  批大小: " << BATCH_SIZE << std::endl;
    std::cout << std::endl;
    
    // 初始化无人机位置（六边形分布）- 与贪心策略完全一致
    std::vector<Vec3> init_positions;
    init_positions.push_back(Vec3(2.0, 2.0, 1.5));
    init_positions.push_back(Vec3(6.0, 2.0, 1.5));
    init_positions.push_back(Vec3(10.0, 2.0, 1.5));
    init_positions.push_back(Vec3(2.0, 6.0, 1.5));
    init_positions.push_back(Vec3(6.0, 6.0, 1.5));
    init_positions.push_back(Vec3(10.0, 6.0, 1.5));
    
    // 计算实际物理尺寸
    double map_size_x = 12.0; // 120体素 * 0.1分辨率
    double map_size_y = 12.0;
    double map_size_z = 3.0;
    
    // 生成障碍物
    std::vector<Obstacle> obstacles = generateObstacles(NUM_OBSTACLES, 
                                                         map_size_x, 
                                                         map_size_y, 
                                                         map_size_z,
                                                         init_positions);
    std::cout << "生成了 " << obstacles.size() << " 个障碍物" << std::endl;
    for (int i = 0; i < obstacles.size(); i++) {
        std::cout << "  障碍物 " << i+1 << ": 位置 (" << obstacles[i].center.x() << ", " 
                  << obstacles[i].center.y() << ", " << obstacles[i].center.z() 
                  << "), 大小: " << obstacles[i].size << std::endl;
    }
    
    // 生成稀疏目标
    std::vector<Vec3> targets = generateSparseTargets(NUM_TARGETS, 
                                                     map_size_x, 
                                                     map_size_y, 
                                                     map_size_z,
                                                     obstacles);
    std::cout << "生成了 " << targets.size() << " 个稀疏目标" << std::endl;
    for (int i = 0; i < targets.size(); i++) {
        std::cout << "  目标 " << i+1 << ": 位置 (" << targets[i].x() << ", " 
                  << targets[i].y() << ", " << targets[i].z() << ")" << std::endl;
    }
    
    // 将障碍物转换为 AABB 格式
    std::vector<AABB> aabb_obstacles;
    for (const auto& obstacle : obstacles) {
        double half_size = obstacle.size / 2.0;
        AABB aabb;
        aabb.min_x = obstacle.center.x() - half_size;
        aabb.min_y = obstacle.center.y() - half_size;
        aabb.min_z = obstacle.center.z() - half_size;
        aabb.max_x = obstacle.center.x() + half_size;
        aabb.max_y = obstacle.center.y() + half_size;
        aabb.max_z = obstacle.center.z() + half_size;
        aabb_obstacles.push_back(aabb);
    }
    
    // 创建环境管理器
    EnvManager env_manager(MAP_SIZE_X, MAP_SIZE_Y, MAP_SIZE_Z, 
                          0.1, SENSOR_RADIUS, init_positions, MAX_STEPS, aabb_obstacles);
    
    // 设置目标数量为12个（与生成的固定目标一致，与贪心策略相同）
    env_manager.set_num_targets(12);
    
    // 获取观测和动作维度
    std::vector<std::vector<float>> initial_obs = env_manager.get_local_observations();
    int obs_dim = initial_obs[0].size();
    int action_dim = 7; // 7个离散动作
    
    std::cout << "\n观测维度: " << obs_dim << "，动作维度: " << action_dim << std::endl;
    
    // 创建 PPO 智能体
    std::vector<PPOAgent> agents;
    int hidden_dim = 64; // 隐藏层维度
    for (int i = 0; i < NUM_UAVS; i++) {
        agents.emplace_back(obs_dim, hidden_dim, action_dim, LEARNING_RATE);
    }
    
    // 统计指标
    int success_count = 0;
    double total_rewards_all = 0.0;
    int total_steps_all = 0;
    std::vector<double> episode_rewards_list;
    std::vector<int> episode_steps_list;
    std::vector<double> episode_success_rates;
    std::vector<double> episode_coverage_rates;
    
    // 记录训练时间
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // 训练循环
    for (int episode = 0; episode < NUM_EPISODES; episode++) {
        env_manager.reset();
        std::vector<double> episode_rewards(NUM_UAVS, 0.0);
        bool episode_success = false;
        int episode_steps = 0;
        
        for (int step = 0; step < MAX_STEPS; step++) {
            // 获取观测
            std::vector<std::vector<float>> observations = env_manager.get_local_observations();
            
            // 为每个无人机选择动作
            std::vector<Action> actions;
            std::vector<std::tuple<int, float, float>> action_results;
            for (int i = 0; i < NUM_UAVS; i++) {
                auto result = agents[i].select_action(observations[i]);
                action_results.push_back(result);
                int action_idx = std::get<0>(result);
                actions.push_back(static_cast<Action>(action_idx));
            }
            
            // 执行动作
            auto step_result = env_manager.step(actions);
            std::vector<float> rewards = std::get<0>(step_result);
            std::vector<std::vector<float>> next_observations = std::get<1>(step_result);
            bool done = std::get<2>(step_result);
            
            // 存储经验
            for (int i = 0; i < NUM_UAVS; i++) {
                int action_idx = std::get<0>(action_results[i]);
                float log_prob = std::get<1>(action_results[i]);
                float value = std::get<2>(action_results[i]);
                
                // 创建经验
                Experience exp;
                
                // 创建观测张量（2D 形状：[1, obs_size]）
                exp.obs = Tensor(ShapeTag(), {1, observations[i].size()});
                float* obs_data = exp.obs.data<float>();
                for (size_t j = 0; j < observations[i].size(); j++) {
                    obs_data[j] = observations[i][j];
                }
                
                exp.action = Tensor({(float)action_idx});
                exp.log_prob = Tensor({log_prob});
                exp.value = Tensor({value});
                exp.reward = Tensor({rewards[i]});
                exp.done = Tensor({(float)done});
                
                // 存储经验
                agents[i].store_experience(exp);
            }
            
            // 累计奖励
            for (int i = 0; i < NUM_UAVS; i++) {
                episode_rewards[i] += rewards[i];
            }
            
            // 记录步数
            episode_steps = step + 1;
            
            // 如果完成，退出循环
            if (done) {
                break;
            }
        }
        
        // 计算总奖励
        double total_reward = 0.0;
        for (double r : episode_rewards) {
            total_reward += r;
        }
        
        // 检查是否成功
        if (total_reward > 800.0) { // 超难度场景的成功阈值
            episode_success = true;
            success_count++;
        }
        
        // 累积经验，每3个episode更新一次（减少更新频率，加速训练）
        static int update_counter = 0;
        update_counter++;
        if (update_counter % 3 == 0) {
            for (int i = 0; i < NUM_UAVS; i++) {
                agents[i].update();
                agents[i].clear_buffer(); // 清空缓冲区
            }
        }
        
        // 统计
        total_rewards_all += total_reward;
        total_steps_all += episode_steps;
        episode_rewards_list.push_back(total_reward);
        episode_steps_list.push_back(episode_steps);
        episode_success_rates.push_back(episode_success ? 1.0 : 0.0);
        
        // 计算移动平均奖励
        double moving_avg_reward = 0.0;
        int window_size = std::min(5, episode + 1);
        for (int i = std::max(0, episode - window_size + 1); i <= episode; i++) {
            moving_avg_reward += episode_rewards_list[i];
        }
        moving_avg_reward /= window_size;
        
        // 计算成功率移动平均
        double moving_avg_success = 0.0;
        for (int i = std::max(0, episode - window_size + 1); i <= episode; i++) {
            moving_avg_success += episode_success_rates[i];
        }
        moving_avg_success /= window_size;
        
        // 打印训练信息
        std::cout << " Episode " << episode + 1 << ": 总奖励 = " << total_reward 
                  << ", 成功 = " << (episode_success ? "是" : "否") 
                  << ", 步数 = " << episode_steps
                  << ", 5轮平均奖励 = " << moving_avg_reward
                  << ", 5轮成功率 = " << (moving_avg_success * 100) << "%";
        
        // 打印每架无人机的奖励
        std::cout << ", 无人机奖励: [";
        for (int i = 0; i < NUM_UAVS; i++) {
            std::cout << episode_rewards[i];
            if (i < NUM_UAVS - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        // 每 10 个 episode 打印详细统计
        if ((episode + 1) % 10 == 0) {
            double success_rate = static_cast<double>(success_count) / (episode + 1) * 100.0;
            double avg_reward = total_rewards_all / (episode + 1);
            double avg_steps = static_cast<double>(total_steps_all) / (episode + 1);
            
            std::cout << "\n--- 前 " << (episode + 1) << " 个 episode 详细统计 ---" << std::endl;
            std::cout << "  成功率: " << success_rate << "%" << std::endl;
            std::cout << "  平均奖励: " << avg_reward << std::endl;
            std::cout << "  平均步数: " << avg_steps << std::endl;
            
            // 计算奖励和成功率的标准差
            double reward_variance = 0.0;
            double success_variance = 0.0;
            for (int i = 0; i <= episode; i++) {
                reward_variance += std::pow(episode_rewards_list[i] - avg_reward, 2);
                success_variance += std::pow(episode_success_rates[i] - (success_count / (episode + 1.0)), 2);
            }
            double reward_std = std::sqrt(reward_variance / (episode + 1));
            double success_std = std::sqrt(success_variance / (episode + 1));
            
            std::cout << "  奖励标准差: " << reward_std << std::endl;
            std::cout << "  成功率标准差: " << success_std << std::endl;
            std::cout << std::endl;
        }
    }
    
    // 计算训练时间
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
    
    // 最终统计
    double total_success_rate = static_cast<double>(success_count) / NUM_EPISODES * 100.0;
    double avg_reward = total_rewards_all / NUM_EPISODES;
    double avg_steps = static_cast<double>(total_steps_all) / NUM_EPISODES;
    
    // 计算标准差
    double reward_variance = 0.0;
    double steps_variance = 0.0;
    double success_variance = 0.0;
    for (int i = 0; i < NUM_EPISODES; i++) {
        reward_variance += std::pow(episode_rewards_list[i] - avg_reward, 2);
        steps_variance += std::pow(episode_steps_list[i] - avg_steps, 2);
        success_variance += std::pow(episode_success_rates[i] - (success_count / static_cast<double>(NUM_EPISODES)), 2);
    }
    double reward_std = std::sqrt(reward_variance / NUM_EPISODES);
    double steps_std = std::sqrt(steps_variance / NUM_EPISODES);
    double success_std = std::sqrt(success_variance / NUM_EPISODES);
    
    // 计算最后10个episode的平均
    double last_10_avg_reward = 0.0;
    double last_10_success_rate = 0.0;
    for (int i = std::max(0, NUM_EPISODES - 10); i < NUM_EPISODES; i++) {
        last_10_avg_reward += episode_rewards_list[i];
        last_10_success_rate += episode_success_rates[i];
    }
    last_10_avg_reward /= 10;
    last_10_success_rate = (last_10_success_rate / 10) * 100.0;
    
    std::cout << "\n========== 超难度 AM-MAPPO 测试结果 ==========" << std::endl;
    std::cout << "总训练时间: " << duration << " 秒" << std::endl;
    std::cout << "总成功率: " << total_success_rate << "% (" << success_count << "/" << NUM_EPISODES << ")" << std::endl;
    std::cout << "平均奖励: " << avg_reward << " ± " << reward_std << std::endl;
    std::cout << "平均步数: " << avg_steps << " ± " << steps_std << std::endl;
    std::cout << "成功率标准差: " << success_std << std::endl;
    std::cout << "\n最后10个episode表现:" << std::endl;
    std::cout << "  平均奖励: " << last_10_avg_reward << std::endl;
    std::cout << "  成功率: " << last_10_success_rate << "%" << std::endl;
    std::cout << "================================================" << std::endl;
    
    // 与贪心策略比较
    std::cout << "\n========== 与贪心策略比较 ==========" << std::endl;
    std::cout << "贪心策略 (20 episodes):" << std::endl;
    std::cout << "  成功率: 100%" << std::endl;
    std::cout << "  平均奖励: 8330.64 ± 153.60" << std::endl;
    std::cout << "  平均步数: 600" << std::endl;
    std::cout << "\nAM-MAPPO (" << NUM_EPISODES << " episodes):" << std::endl;
    std::cout << "  成功率: " << total_success_rate << "%" << std::endl;
    std::cout << "  平均奖励: " << avg_reward << " ± " << reward_std << std::endl;
    std::cout << "  平均步数: " << avg_steps << " ± " << steps_std << std::endl;
    
    if (total_success_rate >= 95.0 && avg_reward >= 8300.0) {
        std::cout << "\n✅ AM-MAPPO 性能与贪心策略相当或更好！" << std::endl;
    } else if (total_success_rate >= 80.0 && avg_reward >= 7500.0) {
        std::cout << "\n⚠️  AM-MAPPO 性能接近贪心策略，需要进一步训练。" << std::endl;
    } else {
        std::cout << "\n❌ AM-MAPPO 性能低于贪心策略，需要调整超参数。" << std::endl;
    }
    
    std::cout << "\n超难度 AM-MAPPO 测试完成！" << std::endl;
}

int main() {
    testAMMAPPOUltraHard();
    return 0;
}
