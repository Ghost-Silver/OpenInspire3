/**
 * @file PPOAgent.h
 * @brief PPO 智能体类，封装网络、优化器、经验缓冲、GAE 和更新逻辑
 * @author GhostFace
 * @date 2026/4/4
 */

#ifndef PPOAGENT_H
#define PPOAGENT_H

#include "ActionSpace.h"
#include "../CTorch/include/Tensor.h"
#include <vector>
#include <tuple>

// 经验样本结构体
struct Experience {
    Tensor obs;           // 观测
    Tensor action;        // 动作
    Tensor log_prob;      // 动作对数概率
    Tensor value;         // 状态价值
    Tensor reward;        // 奖励
    Tensor done;          // 终止标志
};

// 神经网络结构
class Network {
private:
    // 策略网络参数
    Tensor fc1_w;         // 第一层权重
    Tensor fc1_b;         // 第一层偏置
    Tensor fc2_w;         // 第二层权重
    Tensor fc2_b;         // 第二层偏置
    Tensor policy_out_w;  // 策略输出层权重
    Tensor policy_out_b;  // 策略输出层偏置
    Tensor value_out_w;   // 价值输出层权重
    Tensor value_out_b;   // 价值输出层偏置

    // 网络配置
    int obs_dim;          // 观测维度
    int hidden_dim;       // 隐藏层维度
    int action_dim;       // 动作维度

public:
    Network(int obs_dim, int hidden_dim, int action_dim);

    // 前向传播
    std::tuple<Tensor, Tensor> forward(const Tensor& obs);

    // 获取策略分布
    Tensor get_policy(const Tensor& obs);

    // 获取价值
    Tensor get_value(const Tensor& obs);

    // 获取所有参数
    std::vector<Tensor> get_parameters();

    // 重置参数
    void reset_parameters();
};

// 优化器
class Optimizer {
private:
    std::vector<Tensor> parameters;
    float learning_rate;
    float momentum;
    float weight_decay;

public:
    Optimizer(std::vector<Tensor> params, float lr = 3e-4, float mom = 0.9, float wd = 0.0001);

    // 零梯度
    void zero_grad();

    // 梯度下降
    void step();
};

// PPO 智能体
class PPOAgent {
private:
    Network network;
    Optimizer optimizer;
    std::vector<Experience> buffer;
    float gamma;          // 折扣因子
    float gae_lambda;     // GAE 参数
    float clip_epsilon;   // PPO 裁剪参数
    int batch_size;       // 批次大小
    int update_epochs;    // 更新轮数

public:
    PPOAgent(int obs_dim, int hidden_dim, int action_dim, float lr = 3e-4);

    // 选择动作
    std::tuple<int, float, float> select_action(const std::vector<float>& obs);

    // 存储经验
    void store_experience(const Experience& exp);

    // 计算 GAE
    std::vector<Tensor> compute_gae(const std::vector<Tensor>& rewards, const std::vector<Tensor>& values, const std::vector<Tensor>& dones);

    // 更新策略
    void update();

    // 清空缓冲区
    void clear_buffer();

    // 获取网络
    Network& get_network();
};

#endif // PPOAGENT_H
