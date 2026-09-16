/**
 * @file PPOAgent.h
 * @brief PPO 智能体：网络、优化器、经验缓冲、GAE 与更新逻辑
 * @author GhostFace
 * @date 2026/4/4
 *
 * @note 2026/9/16 迁移至当前 CTorch 接口，三处结构性修正：
 *
 *  1. **autograd 接口**：移除对 `AutoDiff` / `AutoDiffContext::Guard` /
 *     `Tensor::backward()` 的依赖（这些在当前 CTorch 中并不存在），改用
 *     `AutoGrad::backward(tensor.getRelatedNode(), retain_graph)`。
 *
 *  2. **参数所有权**：旧版 `get_parameters()` 返回按值拷贝的 vector，
 *     Optimizer 于是持有一份副本，`step()` 的更新只作用于副本、从不写回
 *     网络——训练实际是无效的。现改为参数集中存放、Optimizer 持引用。
 *
 *  3. **保图运算**：动作概率的选取、ratio 的 clamp、surrogate 的逐元素 min
 *     原先都用 `data<float>()` 手写循环实现，既不注册 grad_fn，又会被
 *     `Tensor clamped = ratio;` 这类拷贝切断上游。现全部改为 Tensor 算子。
 */

#ifndef PPOAGENT_H
#define PPOAGENT_H

#include "ActionSpace.h"
#include "Tensor.h"
#include <cstddef>
#include <tuple>
#include <vector>

/// 经验样本
struct Experience {
    Tensor obs;      ///< 观测
    Tensor action;   ///< 动作索引
    Tensor log_prob; ///< 旧策略下的动作对数概率
    Tensor value;    ///< 状态价值
    Tensor reward;   ///< 奖励
    Tensor done;     ///< 终止标志
};

/// 由观测向量构造 [1, n] 张量（供调用方构造 Experience 使用）
Tensor makeObsTensor(const std::vector<float> &obs);

/**
 * @class Network
 * @brief 策略/价值网络：两层 MLP 主干 + 策略头 + 价值头
 *
 * 参数集中存放在 `_params` 中，按 ParamIndex 索引。
 *
 * 之所以不把每个权重做成独立成员：Optimizer 需要拿到参数的**引用**才能把
 * 更新写回网络。若沿用旧版的按值返回，optimizer 持有的是副本，
 * `param = param - lr * grad` 只会改副本，网络权重永远不变。
 */
class Network {
  public:
    /// 参数在 _params 中的固定下标
    enum ParamIndex : std::size_t {
        FC1_W = 0,
        FC1_B,
        FC2_W,
        FC2_B,
        POLICY_W,
        POLICY_B,
        VALUE_W,
        VALUE_B,
        PARAM_COUNT
    };

    Network(int obs_dim, int hidden_dim, int action_dim);

    /// 前向传播，返回 (策略分布 [1, action_dim], 状态价值 [1])
    std::tuple<Tensor, Tensor> forward(const Tensor &obs);

    Tensor get_policy(const Tensor &obs);
    Tensor get_value(const Tensor &obs);

    /// 全部参数（引用：可就地更新）
    std::vector<Tensor> &get_parameters() { return _params; }
    const std::vector<Tensor> &get_parameters() const { return _params; }

    Tensor &param(ParamIndex index) { return _params[index]; }
    const Tensor &param(ParamIndex index) const { return _params[index]; }

    int obsDim() const { return _obs_dim; }
    int hiddenDim() const { return _hidden_dim; }
    int actionDim() const { return _action_dim; }

    void reset_parameters();

  private:
    std::vector<Tensor> _params;
    int _obs_dim;
    int _hidden_dim;
    int _action_dim;
};

/**
 * @class Optimizer
 * @brief 朴素 SGD（保留动量/权重衰减字段，当前 step 仅使用学习率）
 *
 * 持有 Network 参数 vector 的**引用**，因此 step() 的更新直接作用于网络。
 */
class Optimizer {
  public:
    explicit Optimizer(std::vector<Tensor> &params, float lr = 3e-4f,
                       float mom = 0.9f, float wd = 0.0001f);

    /// 重新绑定参数向量（供持有者在自身被移动后修正指向）
    void rebind(std::vector<Tensor> &params) { _params = &params; }

    void zero_grad();
    void step();

  private:
    /// 指针而非引用：Optimizer 必须能在其持有者被移动后重新绑定。
    /// C++ 引用一经绑定不可更换，若用引用，则任何移动 PPOAgent 的容器操作
    /// （如 std::vector 扩容）都会让这里指向已析构的旧对象。
    std::vector<Tensor> *_params;
    float _lr;
    float _momentum;
    float _weight_decay;
};

/**
 * @class PPOAgent
 * @brief PPO 智能体
 */
class PPOAgent {
  public:
    PPOAgent(int obs_dim, int hidden_dim, int action_dim, float lr = 3e-4f);

    /// 内部 Optimizer 指向 _network 的参数，拷贝会产生悬垂引用，故禁止拷贝。
    PPOAgent(const PPOAgent &) = delete;
    PPOAgent &operator=(const PPOAgent &) = delete;

    /// 移动后需要把 Optimizer 重新绑定到自身（而非源对象）的网络参数
    PPOAgent(PPOAgent &&other) noexcept;
    PPOAgent &operator=(PPOAgent &&other) noexcept;

    /// 采样动作，返回 (动作索引, 对数概率, 状态价值)
    std::tuple<int, float, float> select_action(const std::vector<float> &obs);

    void store_experience(const Experience &exp);

    /// 广义优势估计（无梯度需求，按标量计算）
    std::vector<Tensor> compute_gae(const std::vector<Tensor> &rewards,
                                    const std::vector<Tensor> &values,
                                    const std::vector<Tensor> &dones);

    /// 执行 _update_epochs 轮策略更新
    void update();

    void clear_buffer();

    Network &get_network() { return _network; }

    std::size_t bufferSize() const { return _buffer.size(); }

  private:
    Network _network;
    Optimizer _optimizer;
    std::vector<Experience> _buffer;
    float _gamma;
    float _gae_lambda;
    float _clip_epsilon;
    int _batch_size;
    int _update_epochs;
};

#endif // PPOAGENT_H
