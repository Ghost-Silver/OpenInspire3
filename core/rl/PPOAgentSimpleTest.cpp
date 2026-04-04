/**
 * @file PPOAgentSimpleTest.cpp
 * @brief 简单的 PPO 智能体测试，只测试基本的网络功能
 * @author GhostFace
 * @date 2026/4/4
 */

#include "PPOAgent.h"
#include <iostream>
#include <vector>

void testPPOAgentSimple() {
    std::cout << "=== 简单 PPO 智能体测试 ===" << std::endl;
    
    // 测试参数
    int obs_dim = 75;  // 观测维度
    int hidden_dim = 64;  // 隐藏层维度
    int action_dim = 7;  // 动作维度
    
    std::cout << "观测维度: " << obs_dim << std::endl;
    std::cout << "动作维度: " << action_dim << std::endl;
    
    // 创建 PPO 智能体
    PPOAgent agent(obs_dim, hidden_dim, action_dim, 3e-4);
    
    // 生成随机观测
    std::vector<float> obs(obs_dim);
    for (int i = 0; i < obs_dim; i++) {
        obs[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    
    // 测试选择动作
    std::cout << "测试选择动作..." << std::endl;
    auto [action_idx, log_prob, value] = agent.select_action(obs);
    std::cout << "选择的动作: " << action_idx << std::endl;
    std::cout << "对数概率: " << log_prob << std::endl;
    std::cout << "状态价值: " << value << std::endl;
    
    // 测试存储经验
    std::cout << "\n测试存储经验..." << std::endl;
    Tensor obs_tensor(ShapeTag(), {1, static_cast<size_t>(obs_dim)});
    float* obs_data = obs_tensor.data<float>();
    for (size_t i = 0; i < obs.size(); i++) {
        obs_data[i] = obs[i];
    }
    
    Tensor action_tensor(ShapeTag(), {1});
    action_tensor.data<float>()[0] = action_idx;
    
    Tensor log_prob_tensor(log_prob);
    Tensor value_tensor(value);
    Tensor reward_tensor(1.0f);
    Tensor done_tensor(0.0f);
    
    Experience exp = {
        obs_tensor,
        action_tensor,
        log_prob_tensor,
        value_tensor,
        reward_tensor,
        done_tensor
    };
    agent.store_experience(exp);
    std::cout << "经验存储成功" << std::endl;
    
    // 测试更新策略
    std::cout << "\n测试更新策略..." << std::endl;
    try {
        agent.update();
        std::cout << "策略更新成功" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "策略更新失败: " << e.what() << std::endl;
    }
    
    std::cout << "\n简单 PPO 智能体测试完成！" << std::endl;
}

int main() {
    // 设置随机种子
    srand(42);
    testPPOAgentSimple();
    return 0;
}
