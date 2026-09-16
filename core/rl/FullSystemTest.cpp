/**
 * @file FullSystemTest.cpp
 * @brief 完整系统测试：测试所有模块的协同工作
 * @author GhostFace
 * @date 2026/4/4
 */

#include "ActionSpace.h"
#include "Environment.h"
#include "../Map/VoxelMap.h"
#include <iostream>

void testFullSystem() {
    std::cout << "=== 完整系统测试 ===" << std::endl;
    
    // 测试 1: VoxelMap 功能测试
    std::cout << "\n测试 1: VoxelMap 功能测试" << std::endl;
    VoxelMap map(30, 30, 10, 0.1, 1.0);
    map.initTargets(5, true, 42);
    
    // 测试坐标转换
    int gx, gy, gz;
    bool valid = map.worldToGrid(1.5, 1.5, 0.5, gx, gy, gz);
    std::cout << "坐标转换: (1.5, 1.5, 0.5) → (" << gx << ", " << gy << ", " << gz << ") 有效: " << (valid ? "是" : "否") << std::endl;
    
    double x, y, z;
    map.gridToWorld(gx, gy, gz, x, y, z);
    std::cout << "坐标转换: (" << gx << ", " << gy << ", " << gz << ") → (" << x << ", " << y << ", " << z << ")" << std::endl;
    
    // 测试传感器更新
    int new_cells = map.updateExploration(1.5, 1.5, 0.5, 0);
    std::cout << "传感器更新: 新探索体素数 = " << new_cells << std::endl;
    
    // 测试目标发现
    int new_targets = map.checkNewTargets();
    std::cout << "目标发现: 新发现目标数 = " << new_targets << std::endl;
    
    // 测试局部地图提取
    std::vector<float> local_map = map.getLocalMap(1.5, 1.5, 0.5, 2, 1);
    std::cout << "局部地图大小: " << local_map.size() << " (5x5x3)" << std::endl;
    
    // 测试 2: UAV 类测试
    std::cout << "\n测试 2: UAV 类测试" << std::endl;
    UAV uav(1.0);
    uav.setPosition(Vec3(0.5, 0.5, 0.5));
    uav.setVelocity(Vec3(0.0, 0.0, 0.0));
    uav.setAcceleration(Vec3(0.0, 5.0, 0.0));
    
    std::cout << "初始位置: (" << uav.getPosition().x() << ", " << uav.getPosition().y() << ", " << uav.getPosition().z() << ")" << std::endl;
    std::cout << "初始速度: (" << uav.getVelocity().x() << ", " << uav.getVelocity().y() << ", " << uav.getVelocity().z() << ")" << std::endl;
    
    // 更新状态
    uav.update(0.03);
    std::cout << "更新后位置: (" << uav.getPosition().x() << ", " << uav.getPosition().y() << ", " << uav.getPosition().z() << ")" << std::endl;
    std::cout << "更新后速度: (" << uav.getVelocity().x() << ", " << uav.getVelocity().y() << ", " << uav.getVelocity().z() << ")" << std::endl;
    
    // 测试 3: UAV 管理器测试
    std::cout << "\n测试 3: UAV 管理器测试" << std::endl;
    std::vector<Vec3> init_positions;
    init_positions.push_back(Vec3(0.5, 0.5, 0.5));
    init_positions.push_back(Vec3(1.5, 1.5, 0.5));
    init_positions.push_back(Vec3(2.5, 2.5, 0.5));
    
    UAVManager uav_manager(3);
    uav_manager.reset(init_positions);
    
    std::vector<Vec3> accelerations;
    accelerations.push_back(Vec3(0.0, 5.0, 0.0));
    accelerations.push_back(Vec3(5.0, 0.0, 0.0));
    accelerations.push_back(Vec3(0.0, 0.0, 5.0));
    
    uav_manager.updateAll(0.03, accelerations);
    
    std::vector<Vec3> positions = uav_manager.getPositions();
    std::vector<Vec3> velocities = uav_manager.getVelocities();
    
    for (size_t i = 0; i < positions.size(); i++) {
        std::cout << "无人机 " << i << " 位置: (" << positions[i].x() << ", " << positions[i].y() << ", " << positions[i].z() << ")" << std::endl;
        std::cout << "无人机 " << i << " 速度: (" << velocities[i].x() << ", " << velocities[i].y() << ", " << velocities[i].z() << ")" << std::endl;
    }
    
    // 测试 4: 强化学习环境测试
    std::cout << "\n测试 4: 强化学习环境测试" << std::endl;
    SearchEnvironment env(30, 30, 10, 0.1, 1.0, init_positions, 20);
    env.reset();
    
    // 测试初始观测
    std::vector<std::vector<float>> initial_obs = env.getLocalObservations();
    std::cout << "初始观测数量: " << initial_obs.size() << std::endl;
    for (size_t i = 0; i < initial_obs.size(); i++) {
        std::cout << "无人机 " << i << " 观测大小: " << initial_obs[i].size() << std::endl;
    }
    
    // 测试全局状态
    std::vector<float> global_state = env.getGlobalState();
    std::cout << "全局状态大小: " << global_state.size() << std::endl;
    
    // 测试动作掩码
    std::vector<std::vector<bool>> action_masks = env.getActionMasks();
    std::cout << "动作掩码数量: " << action_masks.size() << std::endl;
    
    // 测试步进
    std::vector<Action> actions;
    actions.push_back(Action::FORWARD);
    actions.push_back(Action::RIGHT);
    actions.push_back(Action::UP);
    
    auto result = env.step(actions);
    std::vector<float> rewards = std::get<0>(result);
    std::vector<std::vector<float>> next_obs = std::get<1>(result);
    bool done = std::get<2>(result);
    std::vector<float> next_global_state = std::get<3>(result);
    
    std::cout << "步进结果: " << std::endl;
    std::cout << "奖励: ";
    for (float reward : rewards) {
        std::cout << reward << " ";
    }
    std::cout << std::endl;
    std::cout << "是否终止: " << (done ? "是" : "否") << std::endl;
    std::cout << "下一观测大小: " << next_obs.size() << std::endl;
    std::cout << "下一全局状态大小: " << next_global_state.size() << std::endl;
    
    // 测试 5: 多步进测试
    std::cout << "\n测试 5: 多步进测试" << std::endl;
    env.reset();
    int total_steps = 10;
    double total_reward = 0.0;
    
    for (int step = 0; step < total_steps; step++) {
        // 生成随机动作
        std::vector<Action> step_actions;
        for (int i = 0; i < 3; i++) {
            int action_idx = rand() % 7;
            step_actions.push_back(static_cast<Action>(action_idx));
        }
        
        // 执行步进
        auto step_result = env.step(step_actions);
        std::vector<float> step_rewards = std::get<0>(step_result);
        bool step_done = std::get<2>(step_result);
        
        // 计算总奖励
        for (float reward : step_rewards) {
            total_reward += reward;
        }
        
        // 打印步进信息
        std::cout << "步进 " << step + 1 << ": 奖励 = [";
        for (size_t i = 0; i < step_rewards.size(); i++) {
            std::cout << step_rewards[i];
            if (i < step_rewards.size() - 1) std::cout << ", ";
        }
        std::cout << "], 终止 = " << (step_done ? "是" : "否") << std::endl;
        
        // 如果终止，退出循环
        if (step_done) {
            break;
        }
    }
    
    std::cout << "总奖励: " << total_reward << std::endl;
    
    std::cout << "\n完整系统测试完成！" << std::endl;
}

int main() {
    // 设置随机种子
    srand(42);
    testFullSystem();
    return 0;
}
