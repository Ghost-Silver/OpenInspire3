/**
 * @file IntegrationTest.cpp
 * @brief 综合测试：仿真器与 RK4、VoxelMap、UAV 管理器的配合
 * @author GhostFace
 * @date 2026/4/4
 */

#include "ActionSpace.h"
#include "Environment.h"
#include "../Map/VoxelMap.h"
#include <iostream>

void testIntegration() {
    std::cout << "=== 综合测试：UAV 管理器与 VoxelMap 的配合 ===" << std::endl;
    
    // 测试 1: UAV 管理器与 VoxelMap
    std::cout << "\n测试 1: UAV 管理器与 VoxelMap" << std::endl;
    
    // 初始化无人机位置
    std::vector<Vec3> init_positions;
    init_positions.push_back(Vec3(0.5, 0.5, 0.5));
    init_positions.push_back(Vec3(1.5, 1.5, 0.5));
    init_positions.push_back(Vec3(2.5, 2.5, 0.5));
    
    // 创建 VoxelMap
    VoxelMap map(30, 30, 10, 0.1, 1.0);
    map.initTargets(5, true, 42);
    
    // 创建 UAV 管理器
    UAVManager uav_manager(3);
    uav_manager.reset(init_positions);
    
    // 测试 UAV 位置更新
    double dt = 0.03;
    std::vector<Vec3> accelerations;
    accelerations.push_back(Vec3(0.0, 5.0, 0.0));  // 加速度 (m/s²)
    accelerations.push_back(Vec3(5.0, 0.0, 0.0));
    accelerations.push_back(Vec3(0.0, 0.0, 5.0));
    
    uav_manager.updateAll(dt, accelerations);
    
    const std::vector<Vec3>& positions = uav_manager.getPositions();
    std::cout << "更新后无人机位置:" << std::endl;
    for (size_t i = 0; i < positions.size(); i++) {
        std::cout << "无人机 " << i << ": x=" << positions[i].x() << ", y=" << positions[i].y() << ", z=" << positions[i].z() << std::endl;
    }
    
    // 测试 3: 环境与物理引擎集成
    std::cout << "\n测试 3: 环境与物理引擎集成" << std::endl;
    
    // 创建环境
    SearchEnvironment env(30, 30, 10, 0.1, 1.0, init_positions, 20);
    env.reset();
    
    // 执行一些步进
    for (int step = 0; step < 5; step++) {
        // 生成动作
        std::vector<Action> actions;
        actions.push_back(Action::FORWARD);
        actions.push_back(Action::RIGHT);
        actions.push_back(Action::UP);
        
        // 执行步进
        auto result = env.step(actions);
        std::vector<float> rewards = std::get<0>(result);
        bool done = std::get<2>(result);
        
        // 打印结果
        std::cout << "步进 " << step + 1 << ": 奖励 = [";
        for (size_t i = 0; i < rewards.size(); i++) {
            std::cout << rewards[i];
            if (i < rewards.size() - 1) std::cout << ", ";
        }
        std::cout << "], 终止 = " << (done ? "是" : "否") << std::endl;
    }
    
    // 测试 4: 多无人机协同搜索
    std::cout << "\n测试 4: 多无人机协同搜索" << std::endl;
    
    // 创建新的环境
    SearchEnvironment search_env(30, 30, 10, 0.1, 1.0, init_positions, 50);
    search_env.reset();
    
    // 模拟协同搜索
    int total_reward = 0;
    int steps = 0;
    bool done = false;
    
    while (!done && steps < 50) {
        // 生成随机动作
        std::vector<Action> actions;
        for (int i = 0; i < 3; i++) {
            int action_idx = rand() % 7;
            actions.push_back(static_cast<Action>(action_idx));
        }
        
        // 执行步进
        auto result = search_env.step(actions);
        std::vector<float> rewards = std::get<0>(result);
        done = std::get<2>(result);
        
        // 累加奖励
        for (float reward : rewards) {
            total_reward += reward;
        }
        
        steps++;
    }
    
    std::cout << "协同搜索完成: 步数 = " << steps << ", 总奖励 = " << total_reward << std::endl;
    
    std::cout << "\n综合测试完成！" << std::endl;
}

int main() {
    // 设置随机种子
    srand(42);
    testIntegration();
    return 0;
}
