/**
 * @file EnvironmentTest.cpp
 * @brief 强化学习环境测试
 * @author GhostFace
 * @date 2026/4/4
 */

#include "Environment.h"
#include <iostream>

void testEnvironment() {
    std::cout << "=== 测试强化学习环境 ===" << std::endl;
    
    // 初始化无人机位置
    std::vector<Vec3> init_positions;
    init_positions.push_back(Vec3(0.5, 0.5, 0.5));
    init_positions.push_back(Vec3(1.5, 1.5, 0.5));
    init_positions.push_back(Vec3(2.5, 2.5, 0.5));
    
    // 创建环境
    SearchEnvironment env(30, 30, 10, 0.1, 1.0, init_positions, 100);
    
    // 重置环境
    env.reset();
    std::cout << "环境重置完成" << std::endl;
    
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
    for (size_t i = 0; i < action_masks.size(); i++) {
        std::cout << "无人机 " << i << " 动作掩码: ";
        for (bool mask : action_masks[i]) {
            std::cout << (mask ? "1" : "0") << " ";
        }
        std::cout << std::endl;
    }
    
    // 测试步进
    std::vector<Action> actions;
    actions.push_back(Action::FORWARD);
    actions.push_back(Action::RIGHT);
    actions.push_back(Action::UP);
    
    std::cout << "执行步进测试..." << std::endl;
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
    
    std::cout << "强化学习环境测试完成！" << std::endl;
}

int main() {
    testEnvironment();
    return 0;
}
