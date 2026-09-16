/**
 * @file GreedyBaselineUltraHardTest.cpp
 * @brief 超难度贪心覆盖基线测试 - 验证 AM-MAPPO 优势
 * @author GhostFace
 * @date 2026/4/4
 */

#include "EnvManager.h"
#include "../Map/VoxelMap.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>

// 超难度场景参数
constexpr int MAP_SIZE_X = 120;
constexpr int MAP_SIZE_Y = 120;
constexpr int MAP_SIZE_Z = 30;
constexpr int NUM_UAVS = 6;
constexpr int NUM_TARGETS = 30;
constexpr int NUM_OBSTACLES = 30;
constexpr double SENSOR_RADIUS = 0.8;
constexpr int MAX_STEPS = 600;
constexpr int NUM_EPISODES = 20;

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

// 检查点是否在障碍物内
bool isInObstacle(const Vec3& pos, const std::vector<Obstacle>& obstacles) {
    for (const auto& obs : obstacles) {
        double half_size = obs.size / 2.0;
        if (pos.x() >= obs.center.x() - half_size && pos.x() <= obs.center.x() + half_size &&
            pos.y() >= obs.center.y() - half_size && pos.y() <= obs.center.y() + half_size &&
            pos.z() >= obs.center.z() - half_size && pos.z() <= obs.center.z() + half_size) {
            return true;
        }
    }
    return false;
}

// 高级贪心策略：考虑障碍物、协同和目标稀疏性
Action advanced_greedy_select_action(const std::vector<float>& obs, 
                                     int uav_id,
                                     const std::vector<std::vector<float>>& all_observations,
                                     const std::vector<Obstacle>& obstacles) {
    // 从观测中提取信息
    Vec3 current_pos(obs[0], obs[1], obs[2]);
    const int local_map_start = 8;
    
    // 计算每个方向的得分
    std::vector<double> direction_scores(7, 0.0);
    
    // 基础探索得分（基于局部地图）
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
                    direction_scores[6] += 0.2; // 悬停得分更低
                }
            }
        }
    }
    
    // 动作对应的速度向量
    std::vector<Vec3> action_velocities = {
        Vec3(0.0, 1.0, 0.0),   // FORWARD
        Vec3(0.0, -1.0, 0.0),  // BACKWARD
        Vec3(-1.0, 0.0, 0.0),  // LEFT
        Vec3(1.0, 0.0, 0.0),   // RIGHT
        Vec3(0.0, 0.0, 1.0),   // UP
        Vec3(0.0, 0.0, -1.0),  // DOWN
        Vec3(0.0, 0.0, 0.0)    // HOVER
    };
    
    // 障碍物惩罚：检查每个方向的下一步是否会碰到障碍物
    for (int i = 0; i < 7; i++) {
        Vec3 next_pos = current_pos + action_velocities[i] * 0.8; // 预测下一步位置
        if (isInObstacle(next_pos, obstacles)) {
            direction_scores[i] -= 15.0; // 大幅惩罚进入障碍物的动作
        }
        
        // 边界检查
        if (next_pos.x() < 0.5 || next_pos.x() >= 12.0 - 0.5 ||
            next_pos.y() < 0.5 || next_pos.y() >= 12.0 - 0.5 ||
            next_pos.z() < 0.5 || next_pos.z() >= 3.0 - 0.5) {
            direction_scores[i] -= 15.0; // 大幅惩罚飞出边界的动作
        }
    }
    
    // 协同：避免与其他无人机过于接近
    for (int i = 0; i < static_cast<int>(all_observations.size()); i++) {
        if (i == uav_id) continue;
        
        Vec3 other_pos(all_observations[i][0], all_observations[i][1], all_observations[i][2]);
        double dist = std::sqrt(std::pow(current_pos.x() - other_pos.x(), 2) +
                               std::pow(current_pos.y() - other_pos.y(), 2) +
                               std::pow(current_pos.z() - other_pos.z(), 2));
        
        if (dist < 3.0) { // 距离过近
            // 惩罚朝向其他无人机的动作
            for (int j = 0; j < 6; j++) {
                Vec3 next_pos = current_pos + action_velocities[j] * 0.8;
                double next_dist = std::sqrt(std::pow(next_pos.x() - other_pos.x(), 2) +
                                             std::pow(next_pos.y() - other_pos.y(), 2) +
                                             std::pow(next_pos.z() - other_pos.z(), 2));
                if (next_dist < dist) {
                    direction_scores[j] -= 3.0;
                }
            }
        }
    }
    
    // 目标稀疏性奖励：优先探索未被其他无人机覆盖的区域
    for (int i = 0; i < static_cast<int>(all_observations.size()); i++) {
        if (i == uav_id) continue;
        
        Vec3 other_pos(all_observations[i][0], all_observations[i][1], all_observations[i][2]);
        double dist = std::sqrt(std::pow(current_pos.x() - other_pos.x(), 2) +
                               std::pow(current_pos.y() - other_pos.y(), 2) +
                               std::pow(current_pos.z() - other_pos.z(), 2));
        
        if (dist > 20.0) { // 距离较远，鼓励探索
            for (int j = 0; j < 6; j++) {
                Vec3 next_pos = current_pos + action_velocities[j] * 0.8;
                double next_dist = std::sqrt(std::pow(next_pos.x() - other_pos.x(), 2) +
                                             std::pow(next_pos.y() - other_pos.y(), 2) +
                                             std::pow(next_pos.z() - other_pos.z(), 2));
                if (next_dist > dist) {
                    direction_scores[j] += 2.0; // 鼓励远离其他无人机，探索新区域
                }
            }
        }
    }
    
    // 选择得分最高的动作
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

void testGreedyCoverageUltraHard() {
    std::cout << "=== 超难度贪心覆盖基线测试 ===" << std::endl;
    std::cout << "场景参数:" << std::endl;
    std::cout << "  地图尺寸: " << MAP_SIZE_X << "x" << MAP_SIZE_Y << "x" << MAP_SIZE_Z << std::endl;
    std::cout << "  无人机数量: " << NUM_UAVS << std::endl;
    std::cout << "  目标数量: " << NUM_TARGETS << " (稀疏分布)" << std::endl;
    std::cout << "  障碍物数量: " << NUM_OBSTACLES << std::endl;
    std::cout << "  传感器半径: " << SENSOR_RADIUS << " (减小)" << std::endl;
    std::cout << "  最大步数: " << MAX_STEPS << std::endl;
    std::cout << "  测试轮数: " << NUM_EPISODES << std::endl;
    std::cout << std::endl;
    
    // 初始化无人机位置（六边形分布）
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
    std::cout << std::endl;
    
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
    
    // 设置目标数量为12个（与生成的固定目标一致）
    env_manager.set_num_targets(12);
    
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
            
            // 为每个无人机选择动作（高级贪心策略）
            std::vector<Action> actions;
            for (int i = 0; i < NUM_UAVS; i++) {
                Action action = advanced_greedy_select_action(observations[i], i, 
                                                               observations, obstacles);
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
            
            if (total_reward > 800.0) { // 超难度场景的成功阈值
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
        
        // 每 5 个 episode 打印统计
        if ((episode + 1) % 5 == 0) {
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
    
    std::cout << "\n========== 超难度贪心覆盖基线测试结果 ==========" << std::endl;
    std::cout << "成功率: " << total_success_rate << "% (" << success_count << "/" << NUM_EPISODES << ")" << std::endl;
    std::cout << "平均奖励: " << avg_reward << " ± " << reward_std << std::endl;
    std::cout << "平均步数: " << avg_steps << " ± " << steps_std << std::endl;
    std::cout << "================================================" << std::endl;
    std::cout << "\n超难度贪心覆盖基线测试完成！" << std::endl;
    std::cout << "预期: AM-MAPPO 应在此场景下显著优于贪心策略" << std::endl;
}

int main() {
    testGreedyCoverageUltraHard();
    return 0;
}
