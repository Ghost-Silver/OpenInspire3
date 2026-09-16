/**
 * @file EnvironmentTestAdvanced.cpp
 * @brief 强化学习环境高级测试
 * @author GhostFace
 * @date 2026/4/4
 */

#include "Environment.h"
#include <iostream>

void testAdvancedEnvironment() {
    std::cout << "=== 测试强化学习环境（高级） ===" << std::endl;
    
    // 初始化无人机位置
    std::vector<Vec3> init_positions;
    init_positions.push_back(Vec3(0.5, 0.5, 0.5));
    init_positions.push_back(Vec3(1.5, 1.5, 0.5));
    init_positions.push_back(Vec3(2.5, 2.5, 0.5));
    
    // 创建环境
    SearchEnvironment env(30, 30, 10, 0.1, 1.0, init_positions, 20);
    
    // 重置环境
    env.reset();
    std::cout << "环境重置完成" << std::endl;
    
    // 测试多个步进
    std::cout << "\n测试多个步进..." << std::endl;
    int total_steps = 10;
    double total_reward = 0.0;
    
    for (int step = 0; step < total_steps; step++) {
        // 生成随机动作
        std::vector<Action> actions;
        for (int i = 0; i < 3; i++) {
            int action_idx = rand() % 7;
            actions.push_back(static_cast<Action>(action_idx));
        }
        
        // 执行步进
        auto result = env.step(actions);
        std::vector<float> rewards = std::get<0>(result);
        bool done = std::get<2>(result);
        
        // 计算总奖励
        for (float reward : rewards) {
            total_reward += reward;
        }
        
        // 打印步进信息
        std::cout << "步进 " << step + 1 << ": 奖励 = [";
        for (size_t i = 0; i < rewards.size(); i++) {
            std::cout << rewards[i];
            if (i < rewards.size() - 1) std::cout << ", ";
        }
        std::cout << "], 终止 = " << (done ? "是" : "否") << std::endl;
        
        // 如果终止，退出循环
        if (done) {
            break;
        }
    }
    
    std::cout << "总奖励: " << total_reward << std::endl;
    
    // 测试边界动作掩码
    std::cout << "\n测试边界动作掩码..." << std::endl;
    // 创建一个靠近边界的无人机位置
    std::vector<Vec3> boundary_positions;
    boundary_positions.push_back(Vec3(0.1, 0.1, 0.1)); // 靠近左下角
    boundary_positions.push_back(Vec3(2.9, 2.9, 0.9)); // 靠近右上角
    
    SearchEnvironment boundary_env(30, 30, 10, 0.1, 1.0, boundary_positions, 10);
    boundary_env.reset();
    
    std::vector<std::vector<bool>> masks = boundary_env.getActionMasks();
    for (size_t i = 0; i < masks.size(); i++) {
        std::cout << "边界无人机 " << i << " 动作掩码: ";
        for (bool mask : masks[i]) {
            std::cout << (mask ? "1" : "0") << " ";
        }
        std::cout << std::endl;
    }
    
    // 测试目标发现
    std::cout << "\n测试目标发现..." << std::endl;
    SearchEnvironment target_env(30, 30, 10, 0.1, 1.0, init_positions, 10);
    target_env.reset();
    
    // 执行一些动作，尝试发现目标
    for (int step = 0; step < 5; step++) {
        std::vector<Action> actions;
        for (int i = 0; i < 3; i++) {
            actions.push_back(Action::FORWARD);
        }
        
        auto result = target_env.step(actions);
        std::vector<float> rewards = std::get<0>(result);
        
        std::cout << "目标发现测试步进 " << step + 1 << ": 奖励 = [";
        for (size_t i = 0; i < rewards.size(); i++) {
            std::cout << rewards[i];
            if (i < rewards.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
    
    std::cout << "强化学习环境高级测试完成！" << std::endl;
}

int main() {
    // 设置随机种子
    srand(42);
    testAdvancedEnvironment();
    return 0;
}
