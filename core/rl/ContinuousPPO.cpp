/**
 * @file ContinuousPPO.cpp
 * @brief 连续动作 PPO（高斯策略）实现
 * @author GhostFace
 * @date 2026/9/16
 */

#include "ContinuousPPO.h"
#include "AutoGrad.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace oi3::rl {

namespace {

/// 把观测向量转为 [1, n] 张量（新建叶子张量，属常量输入）
Tensor makeObsTensor(const std::vector<float> &obs) {
    Tensor t(ShapeTag{}, {1, obs.size()});
    float *p = t.data<float>();
    for (std::size_t i = 0; i < obs.size(); ++i) {
        p[i] = obs[i];
    }
    return t;
}

/// 策略输出层的额外缩放系数
///
/// 动作幅度直接对应推力偏离悬停量的比例（动作 1.0 等于一整个 m*g）。若输出层
/// 与隐层采用同一尺度，初始策略就会输出 O(1) 的动作，等效于持续施加接近满量
/// 程的推力偏置，无人机在数十秒内即飞出数百米并落到训练分布之外，训练无法
/// 从合理工作点开始。缩小后初始策略近似「保持悬停」。
constexpr float kPolicyOutputScale = 0.01f;

/// log_std 的允许区间
///
/// 必须设下界：策略损失会把 log_std 往小推，标准差趋零后 z = (a-μ)/σ 迅速
/// 变大，z² 更大，链路形成正反馈并最终产生 NaN；一旦出现 NaN，参数再也无法
/// 恢复。上界则防止采样噪声淹没信号、并避免 exp 溢出。
constexpr float kLogStdMin = -3.0f;
constexpr float kLogStdMax = 1.0f;

/// He 风格初始化：CTorch 的 rand() 为 U[0,1)（均值 0.5），先平移再按 fan_in 缩放
void initHeWeight(Tensor &w, std::size_t fan_in) {
    w.rand();
    const float k = std::sqrt(24.0f / static_cast<float>(fan_in));
    float *p = w.data<float>();
    const std::size_t n = w.numel();
    for (std::size_t i = 0; i < n; ++i) {
        p[i] = (p[i] - 0.5f) * k;
    }
}

/**
 * @brief 按索引重排并截取区间，组装成 [B, dim] 常量张量
 *
 * 每个 minibatch 重新构造一次批量张量，而不是先建大张量再切片：构造开销仅是
 * B*dim 次浮点拷贝，而切片是否保留计算图取决于其具体实现，不值得为此引入
 * 不确定性。
 *
 * @param use_action true 取动作（[B, action_dim]），false 取观测（[B, obs_dim]）
 */
Tensor gatherMatrix(const std::vector<Transition> &buffer,
                    const std::vector<std::size_t> &order, std::size_t begin,
                    std::size_t end, std::size_t dim, bool use_action) {
    const std::size_t rows = end - begin;
    Tensor t(ShapeTag{}, {rows, dim});
    float *p = t.data<float>();
    for (std::size_t i = 0; i < rows; ++i) {
        const Transition &tr = buffer[order[begin + i]];
        for (std::size_t j = 0; j < dim; ++j) {
            p[i * dim + j] = use_action ? tr.action[j] : tr.obs[j];
        }
    }
    return t;
}

/// 组装 [B] 形状的常量张量（取自外部标量数组，按 order 重排）
Tensor gatherValues(const std::vector<float> &values,
                    const std::vector<std::size_t> &order, std::size_t begin,
                    std::size_t end) {
    const std::size_t rows = end - begin;
    Tensor t(ShapeTag{}, {rows});
    float *p = t.data<float>();
    for (std::size_t i = 0; i < rows; ++i) {
        p[i] = values[order[begin + i]];
    }
    return t;
}

/// 组装 [B] 形状的常量张量（取自 Transition 的标量字段）
Tensor gatherField(const std::vector<Transition> &buffer,
                   const std::vector<std::size_t> &order, std::size_t begin,
                   std::size_t end, bool use_log_prob) {
    const std::size_t rows = end - begin;
    Tensor t(ShapeTag{}, {rows});
    float *p = t.data<float>();
    for (std::size_t i = 0; i < rows; ++i) {
        const Transition &tr = buffer[order[begin + i]];
        p[i] = use_log_prob ? tr.log_prob : tr.value;
    }
    return t;
}

} // namespace

ContinuousPPO::ContinuousPPO(PpoConfig config, std::uint32_t seed)
    : _cfg(config), _rng(seed) {
    _params.resize(PARAM_COUNT);

    const auto o = static_cast<std::size_t>(_cfg.obs_dim);
    const auto h = static_cast<std::size_t>(_cfg.hidden_dim);
    const auto a = static_cast<std::size_t>(_cfg.action_dim);

    // 权重按 [fan_in, fan_out] 布局：CTorch 的 transpose 不参与 autograd，
    // 因此不采用「存 [fan_out, fan_in] 再左乘 W^T」的教科书写法。
    _params[FC1_W] = Tensor(ShapeTag{}, {o, h});
    _params[FC1_B] = Tensor(ShapeTag{}, {h});
    _params[FC2_W] = Tensor(ShapeTag{}, {h, h});
    _params[FC2_B] = Tensor(ShapeTag{}, {h});
    _params[MEAN_W] = Tensor(ShapeTag{}, {h, a});
    _params[MEAN_B] = Tensor(ShapeTag{}, {a});
    _params[V_W] = Tensor(ShapeTag{}, {h, 1});
    _params[V_B] = Tensor(ShapeTag{}, {1});
    _params[LOG_STD] = Tensor(ShapeTag{}, {1, a});

    initParameters();

    for (auto &p : _params) {
        p.requires_grad(true);
    }
}

void ContinuousPPO::initParameters() {
    const auto o = static_cast<std::size_t>(_cfg.obs_dim);
    const auto h = static_cast<std::size_t>(_cfg.hidden_dim);

    initHeWeight(_params[FC1_W], o);
    initHeWeight(_params[FC2_W], h);
    initHeWeight(_params[V_W], h);
    initHeWeight(_params[MEAN_W], h);

    // 策略输出层额外缩小，理由见 kPolicyOutputScale 的说明
    {
        float *p = _params[MEAN_W].data<float>();
        const std::size_t n = _params[MEAN_W].numel();
        for (std::size_t i = 0; i < n; ++i) {
            p[i] *= kPolicyOutputScale;
        }
    }

    _params[FC1_B].zero();
    _params[FC2_B].zero();
    _params[MEAN_B].zero();
    _params[V_B].zero();

    // log_std 用常量填充（每个动作维度相同）
    float *ls = _params[LOG_STD].data<float>();
    for (int i = 0; i < _cfg.action_dim; ++i) {
        ls[i] = _cfg.init_log_std;
    }
}

std::tuple<Tensor, Tensor> ContinuousPPO::forward(const Tensor &obs) {
    const Tensor &fc1_w = _params[FC1_W];
    const Tensor &fc1_b = _params[FC1_B];
    const Tensor &fc2_w = _params[FC2_W];
    const Tensor &fc2_b = _params[FC2_B];
    const Tensor &mean_w = _params[MEAN_W];
    const Tensor &mean_b = _params[MEAN_B];
    const Tensor &v_w = _params[V_W];
    const Tensor &v_b = _params[V_B];

    // obs 支持 [B, obs_dim]：批量更新时 B 为 minibatch 大小，采样时 B = 1
    Tensor h1 = (obs.matmul(fc1_w) + fc1_b).relu();
    Tensor h2 = (h1.matmul(fc2_w) + fc2_b).relu();

    Tensor mean = h2.matmul(mean_w) + mean_b; // [B, action_dim]
    // 价值：[B,1] 沿最后一维归约成 [B]。调度器对逐元素算子做广播，
    // 因此 [B] 形状的回报可以直接与之相减。
    Tensor value = (h2.matmul(v_w) + v_b).sum(1);

    // 必须移动构造：具名局部变量若直接 return，会触发 Tensor 拷贝构造，
    // 副本的 autograd 节点被替换为新建 GradAccumulator，与上游断开。
    return {std::move(mean), std::move(value)};
}

Tensor ContinuousPPO::logProbFrom(const Tensor &mean, const std::array<float, 3> &action) {
    // 限幅后使用：训练与采样必须采用同一个有效标准差，否则新旧策略的对数概率
    // 不可比，ratio 会失真
    const Tensor std_t = _params[LOG_STD].clamp(kLogStdMin, kLogStdMax).exp();

    // 采样得到的动作作为常量参与计算：策略梯度来自 log π 对 μ 与 σ 的偏导，
    // 不需要穿过「采样」这一步本身，故动作张量是叶子常量。
    Tensor a_t(ShapeTag{}, {1, static_cast<std::size_t>(_cfg.action_dim)});
    float *ap = a_t.data<float>();
    for (int i = 0; i < _cfg.action_dim; ++i) {
        ap[i] = action[static_cast<std::size_t>(i)];
    }

    const Tensor z = (a_t - mean) / std_t;
    // 常数项 -0.5*ln(2π)*dim 在 ratio 中相消，故省略
    return (z.square() * -0.5f - std_t.log()).sum();
}

ActionSample ContinuousPPO::selectAction(const std::vector<float> &obs) {
    const Tensor obs_t = makeObsTensor(obs);

    // 采样是推理行为，无需构建计算图
    const bool saved = AutoGrad::EnableGrad;
    AutoGrad::EnableGrad = false;
    auto [mean, value] = forward(obs_t);
    const float value_scalar = value.data<float>()[0];
    AutoGrad::EnableGrad = saved;

    const float *mu = mean.data<float>();
    const Tensor effective_log_std = _params[LOG_STD].clamp(kLogStdMin, kLogStdMax);
    const float *log_std = effective_log_std.data<float>();

    ActionSample sample;
    double log_prob_sum = 0.0;
    for (int i = 0; i < _cfg.action_dim; ++i) {
        const float sigma = std::exp(log_std[i]);
        const float noise = _normal(_rng);
        const float a = mu[i] + sigma * noise;
        sample.action[static_cast<std::size_t>(i)] = a;

        const float z = (a - mu[i]) / sigma;
        log_prob_sum += -0.5 * static_cast<double>(z) * z - std::log(static_cast<double>(sigma));
    }
    sample.log_prob = static_cast<float>(log_prob_sum);
    sample.value = value_scalar;
    return sample;
}

std::array<float, 3> ContinuousPPO::actMean(const std::vector<float> &obs) {
    const Tensor obs_t = makeObsTensor(obs);
    const bool saved = AutoGrad::EnableGrad;
    AutoGrad::EnableGrad = false;
    auto [mean, value] = forward(obs_t);
    (void)value;
    const float *mu = mean.data<float>();
    AutoGrad::EnableGrad = saved;

    std::array<float, 3> out{};
    for (int i = 0; i < _cfg.action_dim; ++i) {
        out[static_cast<std::size_t>(i)] = mu[i];
    }
    return out;
}

void ContinuousPPO::store(const Transition &transition) {
    _buffer.push_back(transition);
}

float ContinuousPPO::meanLogStd() const {
    const Tensor effective = _params[LOG_STD].clamp(kLogStdMin, kLogStdMax);
    const float *ls = effective.data<float>();
    float sum = 0.0f;
    for (int i = 0; i < _cfg.action_dim; ++i) {
        sum += ls[i];
    }
    return sum / static_cast<float>(_cfg.action_dim);
}

void ContinuousPPO::clipGradients(float max_norm) {
    float total = 0.0f;
    for (auto &p : _params) {
        const float *gp = p.grad_ptr();
        if (gp == nullptr) {
            continue;
        }
        const std::size_t n = p.numel();
        for (std::size_t i = 0; i < n; ++i) {
            total += gp[i] * gp[i];
        }
    }

    const float norm = std::sqrt(total);
    if (!(norm > max_norm)) { // 同时排除 NaN
        return;
    }

    const float scale = max_norm / (norm + 1e-6f);
    for (auto &p : _params) {
        float *gp = p.grad_ptr();
        if (gp == nullptr) {
            continue;
        }
        const std::size_t n = p.numel();
        for (std::size_t i = 0; i < n; ++i) {
            gp[i] *= scale;
        }
    }
}

void ContinuousPPO::clearBuffer() {
    _buffer.clear();
}

void ContinuousPPO::zeroGrad() {
    for (auto &p : _params) {
        p.zero_grad();
    }
}

void ContinuousPPO::sgdStep() {
    // 先裁剪梯度范数：任何一次异常大的梯度都可能把 log_std 推到边界之外或
    // 产生 NaN，而 NaN 一旦进入参数就再也无法恢复
    clipGradients(1.0f);

    // 原地写入更新，与 CTorch mnist 的 sgd_step 一致。
    // 不能用 `p = p - lr*g`：移动赋值会把计算图节点搬进参数槽位，
    // 参数不再是带 GradAccumulator 的叶子张量，图结构逐轮膨胀。
    for (auto &p : _params) {
        float *gp = p.grad_ptr();
        if (gp == nullptr) {
            continue;
        }
        float *pp = p.data_write<float>();
        const std::size_t n = p.numel();
        for (std::size_t i = 0; i < n; ++i) {
            pp[i] -= gp[i] * _cfg.learning_rate;
        }
    }
}

void ContinuousPPO::update() {
    if (_buffer.empty()) {
        return;
    }

    const std::size_t n = _buffer.size();

    // ---- 广义优势估计（标量递推，不参与梯度）----
    std::vector<float> advantages(n, 0.0f);
    float running = 0.0f;
    for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(n) - 1; i >= 0; --i) {
        const auto idx = static_cast<std::size_t>(i);
        const float next_value =
            (idx + 1 < n) ? _buffer[idx + 1].value : 0.0f;
        const float not_done = 1.0f - _buffer[idx].done;

        const float delta =
            _buffer[idx].reward + _cfg.gamma * next_value * not_done - _buffer[idx].value;
        running = delta + _cfg.gamma * _cfg.gae_lambda * not_done * running;
        advantages[idx] = running;
    }

    // 回报必须在优势标准化之前算：returns = 优势(原始) + 状态价值。
    //
    // 若先标准化优势再相加，得到的量已不是真实回报，价值网络的回归目标会被
    // 错误缩放（缩放后的优势量纲与价值完全不同），进而污染下一轮的优势估计，
    // 表现为策略越训越差。
    std::vector<float> returns(n, 0.0f);
    for (std::size_t i = 0; i < n; ++i) {
        returns[i] = advantages[i] + _buffer[i].value;
    }

    // 优势标准化：PPO 的常规做法，显著改善策略更新的稳定性。
    // 只作用于策略损失，不影响上面的回报。
    float mean_adv = 0.0f;
    for (float x : advantages) {
        mean_adv += x;
    }
    mean_adv /= static_cast<float>(n);

    float var_adv = 0.0f;
    for (float x : advantages) {
        var_adv += (x - mean_adv) * (x - mean_adv);
    }
    var_adv /= static_cast<float>(n);
    const float std_adv = std::sqrt(var_adv) + 1e-8f;
    for (float &x : advantages) {
        x = (x - mean_adv) / std_adv;
    }

    // ---- minibatch 批量更新 ----
    //
    // 逐样本累加会把 B 条经验拼成一张 B 倍大的图：反传要遍历全部节点，且每条
    // 经验都要走一遍完整的算子调度。实测该路径下训练耗时的 99% 花在参数更新上
    // （采样不到 1%）。改为把整个 minibatch 组装成 [B, obs_dim] 一次前向、一次
    // 反传，调用次数与图节点数同时下降近 B 倍。
    const std::size_t obs_dim = static_cast<std::size_t>(_cfg.obs_dim);
    const std::size_t act_dim = static_cast<std::size_t>(_cfg.action_dim);
    const auto batch = std::min(n, static_cast<std::size_t>(std::max(1, _cfg.batch_size)));
    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), 0);

    for (int epoch = 0; epoch < _cfg.update_epochs; ++epoch) {
        std::shuffle(order.begin(), order.end(), _rng);

        for (std::size_t begin = 0; begin < n; begin += batch) {
            const std::size_t end = std::min(begin + batch, n);

            zeroGrad();

            const Tensor obs_b = gatherMatrix(_buffer, order, begin, end, obs_dim, false);
            const Tensor act_b = gatherMatrix(_buffer, order, begin, end, act_dim, true);
            const Tensor old_log_p = gatherField(_buffer, order, begin, end, true);
            const Tensor adv_b = gatherValues(advantages, order, begin, end);
            const Tensor ret_b = gatherValues(returns, order, begin, end);

            auto [mean, value] = forward(obs_b); // mean [B, action_dim]，value [B]

            const Tensor std_t = _params[LOG_STD].clamp(kLogStdMin, kLogStdMax).exp();
            const Tensor z = (act_b - mean) / std_t; // std_t 为 [1,a]，广播到 [B,a]
            const Tensor log_p = (z.square() * -0.5f - std_t.log()).sum(1); // [B]

            const Tensor ratio = (log_p - old_log_p).exp();
            const Tensor clipped =
                ratio.clamp(1.0f - _cfg.clip_epsilon, 1.0f + _cfg.clip_epsilon);

            const Tensor surr1 = ratio * adv_b;
            const Tensor surr2 = clipped * adv_b;
            const Tensor min_surr = surr1.min(surr2);

            const Tensor policy_loss = (-min_surr).mean();
            const Tensor value_loss = (ret_b - value).square().mean();
            const Tensor entropy = std_t.log().mean();

            const Tensor total_loss =
                policy_loss + value_loss * _cfg.value_coef - entropy * _cfg.entropy_coef;

            AutoGrad::backward(total_loss.getRelatedNode(), false);
            sgdStep();
        }
    }

    _buffer.clear();
}

} // namespace oi3::rl
