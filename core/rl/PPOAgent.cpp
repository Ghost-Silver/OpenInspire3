/**
 * @file PPOAgent.cpp
 * @brief PPO 智能体实现
 * @author GhostFace
 * @date 2026/4/4
 *
 * @note 2026/9/16 迁移说明见 PPOAgent.h 顶部注释。本文件相对 4 月版本的
 *       实质性改动集中在下述位置，均以 [迁移] 标注。
 */

#include "PPOAgent.h"
#include "AutoGrad.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <random>

namespace {

/**
 * @brief He 风格权重初始化
 *
 * CTorch 的 `Tensor::rand()` 产生 U[0, 1) —— 均值为 0.5 而非零均值。
 * 若直接拿它当权重，每层输出期望会被放大约 fan_in/4 倍：本网络两层隐层
 * 的 fan_in 分别为 75 与 64，信号逐层放大约 18 倍与 16 倍，累计近三百倍，
 * 输出头再放大一次就达到 1e4 量级，softmax 直接饱和（表现为 log_prob 恒为 0、
 * 策略梯度消失），训练无法进行。
 *
 * 故先平移到零均值，再按 He 初始化缩放：令方差 = 2 / fan_in，
 * 对应均匀分布的半宽 k/2 满足 (k/2)^2 / 3 = 2 / fan_in，即 k = sqrt(24 / fan_in)。
 *
 * @note 初始化发生在 requires_grad 置位之前，此处写入叶子张量数据是安全的。
 */
void initHeWeight(Tensor &w, std::size_t fan_in) {
    w.rand(); // U[0, 1)
    const float k = std::sqrt(24.0f / static_cast<float>(fan_in));
    float *p = w.data<float>();
    const std::size_t n = w.numel();
    for (std::size_t i = 0; i < n; ++i) {
        p[i] = (p[i] - 0.5f) * k; // → 零均值，方差 ≈ 2 / fan_in
    }
}

/**
 * @brief 构造 one-hot 行向量 [1, n]
 *
 * 用途：从策略分布中按索引取出所执行动作的概率，同时**保持计算图连接**。
 * 旧版直接 `float p = policy_data[idx]` 再从 float 构造张量，该张量与 policy
 * 没有任何梯度关联，导致策略梯度恒为零。改用 one-hot 与逐元素乘再求和，
 * 全程走 Tensor 算子。
 */
Tensor makeOneHot(int index, int n) {
    Tensor t(ShapeTag{}, {1, static_cast<std::size_t>(n)});
    if (index >= 0 && index < n) {
        t.data<float>()[index] = 1.0f;
    }
    return t;
}

} // namespace

/**
 * @brief 把观测向量转为 [1, n] 张量
 * @note 这里写入的是**新建叶子张量**的数据，属于常量输入，不涉及梯度。
 */
Tensor makeObsTensor(const std::vector<float> &obs) {
    Tensor t(ShapeTag{}, {1, obs.size()});
    float *p = t.data<float>();
    for (std::size_t i = 0; i < obs.size(); ++i) {
        p[i] = obs[i];
    }
    return t;
}

// ============================== Network ==============================

Network::Network(int obs_dim, int hidden_dim, int action_dim)
    : _obs_dim(obs_dim), _hidden_dim(hidden_dim), _action_dim(action_dim) {
    const auto h = static_cast<std::size_t>(hidden_dim);
    const auto o = static_cast<std::size_t>(obs_dim);
    const auto a = static_cast<std::size_t>(action_dim);

    // 权重一律按 [fan_in, fan_out] 布局存放，前向直接 matmul，无需转置。
    //
    // 为什么不按教科书那样存 [fan_out, fan_in] 再左乘 W^T：
    // CTorch 的 Tensor::transpose() / t() 不参与 autograd ——
    // AutoGrad/Nodes 下没有 TransposeNode，且 transpose 内部是
    // `Tensor result(*this)`（拷贝构造，节点被替换为 GradAccumulator），
    // 因此梯度无法穿过转置。改布局即可绕开，还省掉一次转置拷贝。
    _params.resize(PARAM_COUNT);
    _params[FC1_W] = Tensor(ShapeTag{}, {o, h});    // [obs, hidden]
    _params[FC1_B] = Tensor(ShapeTag{}, {h});
    _params[FC2_W] = Tensor(ShapeTag{}, {h, h});    // [hidden, hidden]
    _params[FC2_B] = Tensor(ShapeTag{}, {h});
    _params[POLICY_W] = Tensor(ShapeTag{}, {h, a}); // [hidden, action]
    _params[POLICY_B] = Tensor(ShapeTag{}, {a});
    _params[VALUE_W] = Tensor(ShapeTag{}, {h, 1});  // [hidden, 1]
    _params[VALUE_B] = Tensor(ShapeTag{}, {1});

    reset_parameters();

    // 参数需要梯度：只有 requires_grad=true 的叶子张量才会在算子分派时
    // 注册计算图节点（见 AutoGrad::dispatch 中的 requires_grad 判定）。
    for (auto &p : _params) {
        p.requires_grad(true);
    }
}

void Network::reset_parameters() {
    const auto h = static_cast<std::size_t>(_hidden_dim);
    const auto o = static_cast<std::size_t>(_obs_dim);
    const auto a = static_cast<std::size_t>(_action_dim);

    // 权重按各自 fan_in 做 He 初始化，偏置置零
    initHeWeight(_params[FC1_W], o);
    initHeWeight(_params[FC2_W], h);
    initHeWeight(_params[POLICY_W], h);
    initHeWeight(_params[VALUE_W], h);

    _params[FC1_B].zero();
    _params[FC2_B].zero();
    _params[POLICY_B].zero();
    _params[VALUE_B].zero();

    (void)a;
}

std::tuple<Tensor, Tensor> Network::forward(const Tensor &obs) {
    // 局部引用只是为了让下面的表达式保持可读
    const Tensor &fc1_w = _params[FC1_W];
    const Tensor &fc1_b = _params[FC1_B];
    const Tensor &fc2_w = _params[FC2_W];
    const Tensor &fc2_b = _params[FC2_B];
    const Tensor &policy_w = _params[POLICY_W];
    const Tensor &policy_b = _params[POLICY_B];
    const Tensor &value_w = _params[VALUE_W];
    const Tensor &value_b = _params[VALUE_B];

    Tensor h1 = (obs.matmul(fc1_w) + fc1_b).relu();
    Tensor h2 = (h1.matmul(fc2_w) + fc2_b).relu();

    Tensor policy = (h2.matmul(policy_w) + policy_b).softmax(1);
    // value 归约为标量：价值头输出 [1,1]，而 PPO 的回报/优势都是标量（0 维）。
    // 调度器的逐元素算子要求形状严格一致、不做隐式广播，[1,1] 与 {} 直接相减会
    // 报 "Tensor形状不一致"。归约后形状统一为标量，语义不变（[1,1] 的 sum 即其自身）。
    Tensor value = (h2.matmul(value_w) + value_b).sum();

    // 必须用 std::move 构造返回的 tuple。此处 policy / value 是具名局部变量（左值），
    // 直接 `return {policy, value}` 会触发 Tensor 的**拷贝构造**，而拷贝构造会把副本的
    // autograd 节点替换成新建的 GradAccumulator（见 Tensor(const Tensor&) 中的
    // createGradAccumulator），副本与上游就此断开 —— 表现就是前向一切正常、
    // 但 backward 时所有参数的梯度恒为零。移动构造保留节点，只 rebind 弱引用。
    return {std::move(policy), std::move(value)};
}

Tensor Network::get_policy(const Tensor &obs) {
    auto [policy, value] = forward(obs);
    (void)value;
    return policy;
}

Tensor Network::get_value(const Tensor &obs) {
    auto [policy, value] = forward(obs);
    (void)policy;
    return value;
}

// ============================== Optimizer ==============================

Optimizer::Optimizer(std::vector<Tensor> &params, float lr, float mom, float wd)
    : _params(&params), _lr(lr), _momentum(mom), _weight_decay(wd) {}

void Optimizer::zero_grad() {
    for (auto &p : *_params) {
        p.zero_grad();
    }
}

void Optimizer::step() {
    // 采用**原地写入**更新参数，与 CTorch 自身的 MNIST 训练保持一致
    // （见 core/CTorch/mnist/mnist.cpp 的 sgd_step）。
    //
    // 不能用 `p = p - lr * g`：Tensor 的移动赋值会把 `p - lr*g` 这个**计算图节点**
    // 搬进参数槽位，参数于是不再是有 GradAccumulator 的叶子张量，而是一个指向
    // 旧参数节点的中间节点。下一轮 forward 时梯度会沿这条历史链继续上溯，
    // 图结构逐轮膨胀，多轮 epoch 后触发反向传播的形状断言而崩溃。
    //
    // 原地写入保持参数张量对象本身不变（node 仍为 GradAccumulator），
    // 数值更新效果等价。
    for (auto &p : *_params) {
        float *gp = p.grad_ptr();
        if (gp == nullptr) {
            continue;
        }
        float *pp = p.data_write<float>();
        const std::size_t n = p.numel();
        for (std::size_t i = 0; i < n; ++i) {
            pp[i] -= gp[i] * _lr;
        }
    }
}

// ============================== PPOAgent ==============================

PPOAgent::PPOAgent(int obs_dim, int hidden_dim, int action_dim, float lr)
    : _network(obs_dim, hidden_dim, action_dim),
      _optimizer(_network.get_parameters(), lr), _rng(std::random_device{}()),
      _gamma(0.99f), _gae_lambda(0.95f), _clip_epsilon(0.2f), _batch_size(64),
      _update_epochs(2) {}

// [2026/9/16 修复] 移动语义必须重新绑定优化器。
//
// Optimizer 内部指向 _network 的参数向量。编译器隐式生成的移动构造只做逐成员
// 搬移，_optimizer 会连同指针一起指向**源对象**的 _network 参数；源对象析构后
// 该指针悬垂，update() 里 step() 一写就崩。
//
// 这不是理论风险：`std::vector<PPOAgent> agents; agents.emplace_back(...)` 在
// 扩容时会移动已有元素，因此只要无人机数量超过首次扩容容量就必然触发。
PPOAgent::PPOAgent(PPOAgent &&other) noexcept
    : _network(std::move(other._network)), _optimizer(std::move(other._optimizer)),
      _buffer(std::move(other._buffer)), _rng(std::move(other._rng)),
      _gamma(other._gamma),
      _gae_lambda(other._gae_lambda), _clip_epsilon(other._clip_epsilon),
      _batch_size(other._batch_size), _update_epochs(other._update_epochs) {
    _optimizer.rebind(_network.get_parameters());
}

PPOAgent &PPOAgent::operator=(PPOAgent &&other) noexcept {
    if (this != &other) {
        _network = std::move(other._network);
        _optimizer = std::move(other._optimizer);
        _buffer = std::move(other._buffer);
        _rng = std::move(other._rng);
        _gamma = other._gamma;
        _gae_lambda = other._gae_lambda;
        _clip_epsilon = other._clip_epsilon;
        _batch_size = other._batch_size;
        _update_epochs = other._update_epochs;
        _optimizer.rebind(_network.get_parameters());
    }
    return *this;
}

std::tuple<int, float, float> PPOAgent::select_action(const std::vector<float> &obs) {
    Tensor obs_tensor = makeObsTensor(obs);

    // [迁移] 采样是推理行为，不需要计算图。暂时关闭梯度记录：
    // 既省掉无用的图构建开销，也使随后按指针读取 policy 数据是安全的
    // （此时 policy 不含 grad_fn，不会有节点引用悬挂的顾虑）。
    const bool saved_enable_grad = AutoGrad::EnableGrad;
    AutoGrad::EnableGrad = false;
    auto [policy, value] = _network.forward(obs_tensor);
    AutoGrad::EnableGrad = saved_enable_grad;

    const int action_dim = _network.actionDim();
    const float *policy_data = policy.data<float>();

    std::discrete_distribution<> dist(policy_data, policy_data + action_dim);
    const int action = dist(_rng);

    const float prob = policy_data[action];
    const float log_prob = std::log(std::max(prob, 1e-12f));
    const float value_val = value.data<float>()[0];

    return {action, log_prob, value_val};
}

void PPOAgent::store_experience(const Experience &exp) {
    _buffer.push_back(exp);
}

std::vector<Tensor> PPOAgent::compute_gae(const std::vector<Tensor> &rewards,
                                          const std::vector<Tensor> &values,
                                          const std::vector<Tensor> &dones) {
    // GAE 只用于构造优势估计的数值，不参与梯度，按标量递推即可
    std::vector<Tensor> advantages(rewards.size());
    Tensor advantage(0.0f);

    for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(rewards.size()) - 1; i >= 0;
         --i) {
        const float reward_val = rewards[static_cast<std::size_t>(i)].data<float>()[0];
        const float value_val = values[static_cast<std::size_t>(i)].data<float>()[0];
        const float next_value_val =
            values[static_cast<std::size_t>(i) + 1].data<float>()[0];
        const float done_val = dones[static_cast<std::size_t>(i)].data<float>()[0];

        const float delta =
            reward_val + _gamma * next_value_val * (1.0f - done_val) - value_val;
        advantage = Tensor(delta) + advantage * (_gamma * _gae_lambda * (1.0f - done_val));
        advantages[static_cast<std::size_t>(i)] = advantage;
    }

    return advantages;
}

void PPOAgent::update() {
    if (_buffer.empty()) {
        return;
    }

    // 取出缓冲区（避免在更新循环中反复访问 vector 元素）
    std::vector<Tensor> obs_list, action_list, log_prob_list, value_list, reward_list,
        done_list;
    const std::size_t n = _buffer.size();
    obs_list.reserve(n);
    action_list.reserve(n);
    log_prob_list.reserve(n);
    value_list.reserve(n);
    reward_list.reserve(n);
    done_list.reserve(n);

    for (const auto &exp : _buffer) {
        obs_list.push_back(exp.obs);
        action_list.push_back(exp.action);
        log_prob_list.push_back(exp.log_prob);
        value_list.push_back(exp.value);
        reward_list.push_back(exp.reward);
        done_list.push_back(exp.done);
    }

    // 最后一个状态的价值为 0，供 GAE 的 bootstrap 项使用
    value_list.push_back(Tensor(0.0f));
    std::vector<Tensor> advantages = compute_gae(reward_list, value_list, done_list);

    // 回报 = 优势 + 状态价值
    std::vector<Tensor> returns(advantages.size());
    for (std::size_t i = 0; i < advantages.size(); ++i) {
        returns[i] = advantages[i] + value_list[i];
    }

    const int action_dim = _network.actionDim();

    // 按 minibatch 分组更新。
    //
    // 原实现把整段缓冲区（回合步数 × 累积回合数，实测可达 1800 条）逐个累加进同一个
    // policy_loss / value_loss，再对这一个巨型图做一次反传。图遍历深度与中间激活内存
    // 都随 n 线性增长，是训练耗时的主要来源（ultra-hard 场景 20 回合约 21 分钟）。
    // `_batch_size` 成员此前从未被使用，正是为分组预留的。
    //
    // 洗牌使用固定种子：采样本身已带随机性，此处固定种子只是让「数据顺序」这一项
    // 在排查问题时可控。
    constexpr unsigned kShuffleSeed = 20260916u;
    const std::size_t batch_size =
        std::min(n, static_cast<std::size_t>(std::max(1, _batch_size)));

    std::vector<std::size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(kShuffleSeed);

    for (int epoch = 0; epoch < _update_epochs; ++epoch) {
        std::shuffle(indices.begin(), indices.end(), rng);

        for (std::size_t batch_begin = 0; batch_begin < n; batch_begin += batch_size) {
            const std::size_t batch_end = std::min(batch_begin + batch_size, n);

            _optimizer.zero_grad();

            Tensor policy_loss(0.0f);
            Tensor value_loss(0.0f);

            for (std::size_t k = batch_begin; k < batch_end; ++k) {
                const std::size_t i = indices[k];
                const Tensor &obs = obs_list[i];
                const Tensor &action = action_list[i];
                const Tensor &old_log_prob = log_prob_list[i];
                const Tensor &advantage = advantages[i];
                const Tensor &target_return = returns[i];

                auto [policy, value] = _network.forward(obs);

                // [迁移] 关键修复：按索引取出所执行动作的对数概率，且保持计算图连接。
                //
                // 旧版做法是
                //     float p = policy_data[action_idx];
                //     Tensor new_log_prob(std::log(p));
                // —— new_log_prob 由 float 构造，是独立叶子张量，与 policy 之间
                // 没有任何梯度路径，于是策略梯度恒为零。改为 one-hot 掩码：
                //     (log_policy * one_hot).sum()  →  标量，且梯度能回传到 policy。
                const int action_idx = static_cast<int>(action.data<float>()[0]);
                const Tensor one_hot = makeOneHot(action_idx, action_dim);
                const Tensor log_policy = policy.clamp(1e-12f, 1.0f).log();
                const Tensor new_log_prob = (log_policy * one_hot).sum();

                // 概率比率 r = exp(log π_new − log π_old)
                const Tensor ratio = (new_log_prob - old_log_prob).exp();

                // [迁移] clamp 改用算子：一行替代原先的 11 行指针循环，
                // 且 clamp 会注册节点，梯度可正常回传。
                const Tensor clipped_ratio =
                    ratio.clamp(1.0f - _clip_epsilon, 1.0f + _clip_epsilon);

                const Tensor surr1 = ratio * advantage;
                const Tensor surr2 = clipped_ratio * advantage;

                // [迁移] 逐元素 min 改用算子（原为拷贝 + 指针循环，两条路径都断图）
                const Tensor min_surr = surr1.min(surr2);

                policy_loss = policy_loss + (-min_surr).mean();
                // 价值损失：0.5 * (R - V)^2，系数在外层统一乘
                value_loss = value_loss + (target_return - value).square().mean();
            }

            const Tensor total_loss = policy_loss + value_loss * 0.5f;

            // 每个 minibatch 单独反传与更新：图不跨组复用，故 retain_graph = false。
            AutoGrad::backward(total_loss.getRelatedNode(), false);
            _optimizer.step();
        }
    }

    clear_buffer();
}

void PPOAgent::clear_buffer() {
    _buffer.clear();
}
