/**
 * @file PPOAgent.cpp
 * @brief PPO 智能体类实现
 * @author GhostFace
 * @date 2026/4/4
 */

#include "PPOAgent.h"
#include "AutoDiff.h"
#include <random>

// 神经网络实现
Network::Network(int obs_dim, int hidden_dim, int action_dim) 
    : obs_dim(obs_dim), hidden_dim(hidden_dim), action_dim(action_dim) {
    // 初始化权重和偏置
    fc1_w = Tensor(ShapeTag(), {static_cast<size_t>(hidden_dim), static_cast<size_t>(obs_dim)});
    fc1_w.rand();
    fc1_w.requires_grad(true);
    
    fc1_b = Tensor(ShapeTag(), {static_cast<size_t>(hidden_dim)});
    fc1_b.zero();
    fc1_b.requires_grad(true);
    
    fc2_w = Tensor(ShapeTag(), {static_cast<size_t>(hidden_dim), static_cast<size_t>(hidden_dim)});
    fc2_w.rand();
    fc2_w.requires_grad(true);
    
    fc2_b = Tensor(ShapeTag(), {static_cast<size_t>(hidden_dim)});
    fc2_b.zero();
    fc2_b.requires_grad(true);
    
    policy_out_w = Tensor(ShapeTag(), {static_cast<size_t>(action_dim), static_cast<size_t>(hidden_dim)});
    policy_out_w.rand();
    policy_out_w.requires_grad(true);
    
    policy_out_b = Tensor(ShapeTag(), {static_cast<size_t>(action_dim)});
    policy_out_b.zero();
    policy_out_b.requires_grad(true);
    
    value_out_w = Tensor(ShapeTag(), {1, static_cast<size_t>(hidden_dim)});
    value_out_w.rand();
    value_out_w.requires_grad(true);
    
    value_out_b = Tensor(ShapeTag(), {1});
    value_out_b.zero();
    value_out_b.requires_grad(true);
}

// 前向传播
std::tuple<Tensor, Tensor> Network::forward(const Tensor& obs) {
    // 第一层
    Tensor h1 = obs.matmul(fc1_w.t()) + fc1_b;
    h1 = h1.relu();
    
    // 第二层
    Tensor h2 = h1.matmul(fc2_w.t()) + fc2_b;
    h2 = h2.relu();
    
    // 策略输出
    Tensor policy = h2.matmul(policy_out_w.t()) + policy_out_b;
    policy = policy.softmax(1);
    
    // 价值输出
    Tensor value = h2.matmul(value_out_w.t()) + value_out_b;
    
    return std::make_tuple(policy, value);
}

// 获取策略分布
Tensor Network::get_policy(const Tensor& obs) {
    auto [policy, _] = forward(obs);
    return policy;
}

// 获取价值
Tensor Network::get_value(const Tensor& obs) {
    auto [_, value] = forward(obs);
    return value;
}

// 获取所有参数
std::vector<Tensor> Network::get_parameters() {
    return {
        fc1_w, fc1_b, fc2_w, fc2_b,
        policy_out_w, policy_out_b, value_out_w, value_out_b
    };
}

// 重置参数
void Network::reset_parameters() {
    fc1_w.rand();
    fc1_b.zero();
    fc2_w.rand();
    fc2_b.zero();
    policy_out_w.rand();
    policy_out_b.zero();
    value_out_w.rand();
    value_out_b.zero();
}

// 优化器实现
Optimizer::Optimizer(std::vector<Tensor> params, float lr, float mom, float wd) 
    : parameters(params), learning_rate(lr), momentum(mom), weight_decay(wd) {}

// 零梯度
void Optimizer::zero_grad() {
    // CTorch 会自动处理梯度的清零
}

// 梯度下降
void Optimizer::step() {
    for (auto& param : parameters) {
        if (param.requires_grad()) {
            // 简单的梯度下降更新 - 使用全局 grad 函数
            Tensor grad_tensor = grad(param);
            if (!grad_tensor.is_cleared()) {
                param = param - learning_rate * grad_tensor;
            }
        }
    }
}

// PPO 智能体实现
PPOAgent::PPOAgent(int obs_dim, int hidden_dim, int action_dim, float lr) 
    : network(obs_dim, hidden_dim, action_dim),
      optimizer(network.get_parameters(), lr),
      gamma(0.99),
      gae_lambda(0.95),
      clip_epsilon(0.2),
      batch_size(64),
      update_epochs(2) {}  // 减少更新轮次从4到2，加速训练

// 选择动作
std::tuple<int, float, float> PPOAgent::select_action(const std::vector<float>& obs) {
    // 将观测转换为张量
    Tensor obs_tensor(ShapeTag(), {1, static_cast<size_t>(obs.size())});
    float* data = obs_tensor.data<float>();
    for (size_t i = 0; i < obs.size(); i++) {
        data[i] = obs[i];
    }
    
    // 获取策略和价值
    auto [policy, value] = network.forward(obs_tensor);
    
    // 从策略分布中采样动作
    float* policy_data = policy.data<float>();
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> dist(policy_data, policy_data + policy.size(1));
    int action = dist(gen);
    
    // 计算动作的对数概率
    float log_prob = std::log(policy_data[action]);
    
    // 获取价值
    float value_val = value.data<float>()[0];
    
    return std::make_tuple(action, log_prob, value_val);
}

// 存储经验
void PPOAgent::store_experience(const Experience& exp) {
    buffer.push_back(exp);
}

// 计算 GAE
std::vector<Tensor> PPOAgent::compute_gae(const std::vector<Tensor>& rewards, const std::vector<Tensor>& values, const std::vector<Tensor>& dones) {
    std::vector<Tensor> advantages(rewards.size());
    Tensor advantage = Tensor(0.0f);
    
    for (int i = rewards.size() - 1; i >= 0; i--) {
        // 确保所有张量都是标量
        float reward_val = rewards[i].data<float>()[0];
        float value_val = values[i].data<float>()[0];
        float next_value_val = values[i + 1].data<float>()[0];
        float done_val = dones[i].data<float>()[0];
        
        float delta = reward_val + gamma * next_value_val * (1 - done_val) - value_val;
        advantage = Tensor(delta) + gamma * gae_lambda * (1 - done_val) * advantage;
        advantages[i] = advantage;
    }
    
    return advantages;
}

// 更新策略
void PPOAgent::update() {
    if (buffer.empty()) {
        return;
    }
    
    // 提取经验（一次性提取，减少内存操作）
    std::vector<Tensor> obs_list, action_list, log_prob_list, value_list, reward_list, done_list;
    obs_list.reserve(buffer.size());
    action_list.reserve(buffer.size());
    log_prob_list.reserve(buffer.size());
    value_list.reserve(buffer.size());
    reward_list.reserve(buffer.size());
    done_list.reserve(buffer.size());
    
    for (const auto& exp : buffer) {
        obs_list.push_back(exp.obs);
        action_list.push_back(exp.action);
        log_prob_list.push_back(exp.log_prob);
        value_list.push_back(exp.value);
        reward_list.push_back(exp.reward);
        done_list.push_back(exp.done);
    }
    
    // 计算 GAE
    value_list.push_back(Tensor(0.0f)); // 最后一个状态的价值为 0
    std::vector<Tensor> advantages = compute_gae(reward_list, value_list, done_list);
    
    // 计算回报
    std::vector<Tensor> returns(advantages.size());
    for (size_t i = 0; i < advantages.size(); i++) {
        returns[i] = advantages[i] + value_list[i];
    }
    
    // 执行多个更新轮次
    for (int epoch = 0; epoch < update_epochs; epoch++) {
        // 创建自动微分上下文（每个更新轮次创建一次）
        AutoDiff auto_diff;
        AutoDiffContext::Guard guard(&auto_diff);
        
        // 零梯度
        optimizer.zero_grad();
        
        // 计算损失
        Tensor policy_loss(0.0f);
        Tensor value_loss(0.0f);
        
        for (size_t i = 0; i < buffer.size(); i++) {
            // 获取当前经验
            Tensor obs = obs_list[i];
            Tensor action = action_list[i];
            Tensor old_log_prob = log_prob_list[i];
            Tensor old_value = value_list[i];
            Tensor advantage = advantages[i];
            Tensor return_ = returns[i];
            
            // 快速跳过无效张量
            if (obs.is_cleared() || !obs.check_storage_offset()) {
                continue;
            }
            
            // 前向传播
            auto [policy, value] = network.forward(obs);
            
            // 快速跳过无效张量
            if (policy.is_cleared() || !policy.check_storage_offset()) {
                continue;
            }
            
            // 计算新的对数概率
            int action_idx = static_cast<int>(action.data<float>()[0]);
            float action_prob = 0.0f;
            
            // 直接访问数据，减少异常处理开销
            float* policy_data = policy.data<float>();
            size_t policy_size = policy.size(1);
            if (action_idx >= 0 && static_cast<size_t>(action_idx) < policy_size) {
                action_prob = policy_data[action_idx];
            } else {
                action_prob = 0.0001f; // 防止除零错误
            }
            
            if (action_prob <= 0) {
                action_prob = 0.0001f; // 防止除零错误
            }
            
            Tensor new_log_prob(std::log(action_prob));
            
            // 计算概率比率
            Tensor ratio = (new_log_prob - old_log_prob).exp();
            
            // 计算 PPO 损失
            // 手动实现 clamp
            Tensor clipped_ratio = ratio;
            float* ratio_data = clipped_ratio.data<float>();
            float min_val = 1 - clip_epsilon;
            float max_val = 1 + clip_epsilon;
            size_t numel = clipped_ratio.numel();
            for (size_t j = 0; j < numel; j++) {
                if (ratio_data[j] < min_val) ratio_data[j] = min_val;
                else if (ratio_data[j] > max_val) ratio_data[j] = max_val;
            }
            
            Tensor surr1 = ratio * advantage;
            Tensor surr2 = clipped_ratio * advantage;
            
            // 手动实现 min
            Tensor min_surr = surr1;
            float* surr1_data = surr1.data<float>();
            float* surr2_data = surr2.data<float>();
            float* min_data = min_surr.data<float>();
            for (size_t j = 0; j < numel; j++) {
                min_data[j] = std::min(surr1_data[j], surr2_data[j]);
            }
            
            policy_loss = policy_loss + (-min_surr).mean();
            
            // 计算价值损失
            value_loss = value_loss + (return_ - value).square().mean();
        }
        
        // 总损失
        Tensor total_loss = policy_loss + 0.5 * value_loss;
        
        // 反向传播
        total_loss.backward();
        
        // 梯度下降
        optimizer.step();
    }
    
    // 清空缓冲区
    clear_buffer();
}

// 清空缓冲区
void PPOAgent::clear_buffer() {
    buffer.clear();
}

// 获取网络
Network& PPOAgent::get_network() {
    return network;
}
