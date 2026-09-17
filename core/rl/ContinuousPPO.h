/**
 * @file ContinuousPPO.h
 * @brief 连续动作 PPO（高斯策略）
 * @author GhostFace
 * @date 2026/9/16
 *
 * 用于飞行控制的策略学习：动作是三维连续推力，而不是离散动作索引。
 *
 * 与 core/rl/PPOAgent（离散，服务于体素搜索环境）并存：两者的动作分布、
 * 对数概率与梯度路径都不同，各自保持独立与简单，不做强行统一。
 *
 * 高斯策略的对数概率：
 * @verbatim
 *   log π(a|s) = Σ_i [ -0.5 * ((a_i - μ_i)/σ_i)^2 - log σ_i ] + const
 * @endverbatim
 * 常数项 -0.5*ln(2π)*dim 在 PPO 的 ratio 中相消（新旧策略用同一公式），
 * 故实现中省略，不影响梯度。
 *
 * 标准差由**可学习的独立参数** log_std 给出（不依赖状态），这是 PPO 的
 * 常用简化：策略只需在网络中拟合均值，探索幅度单独学习。
 */

#ifndef OI3_CONTINUOUS_PPO_H
#define OI3_CONTINUOUS_PPO_H

#include "Tensor.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <random>
#include <tuple>
#include <vector>

namespace oi3::rl {

/// 连续 PPO 超参数
struct PpoConfig {
    int obs_dim = 6;             ///< 观测维度（位置误差 3 + 速度 3）
    int action_dim = 3;          ///< 动作维度（三维推力）
    int hidden_dim = 64;         ///< 隐层宽度
    float learning_rate = 3e-4f; ///< SGD 学习率
    float gamma = 0.99f;         ///< 折扣因子
    float gae_lambda = 0.95f;    ///< GAE 的 lambda
    float clip_epsilon = 0.2f;   ///< PPO 裁剪范围
    int update_epochs = 4;       ///< 每次 update 的轮数
    int batch_size = 64;         ///< minibatch 大小
    float init_log_std = -0.5f;  ///< 初始对数标准差（std ≈ 0.61）
    float value_coef = 0.5f;     ///< 价值损失权重
    /// 熵奖励系数。高斯策略的熵为 Σ(log σ + 0.5*ln(2πe))，其中常数项不影响
    /// 梯度方向。缺少熵项时，探索幅度只会被策略损失挤压而单调收缩，早期若
    /// 探索不够便再也走不出次优解。
    float entropy_coef = 0.01f;
};

/// 单条经验
struct Transition {
    std::vector<float> obs;
    std::array<float, 3> action;
    float log_prob = 0.0f; ///< 采样时旧策略的对数概率
    float value = 0.0f;
    float reward = 0.0f;
    float done = 0.0f;
};

/// 一次动作采样的结果
struct ActionSample {
    std::array<float, 3> action;
    float log_prob = 0.0f;
    float value = 0.0f;
};

/// 一次 update 结束后的统计量（诊断用）
///
/// 这些量是判断 PPO 是否在有效更新的基本依据：approx_kl 过大说明步长过大
/// 或数据复用过度；clip_fraction 长期为 0 说明裁剪从未生效（ratio 始终在带内，
/// 更新可能过小）；mean_advantage 为 0 说明优势估计失去区分度。
struct UpdateStats {
    float policy_loss = 0.0f;
    float value_loss = 0.0f;
    float entropy = 0.0f;
    float approx_kl = 0.0f;
    float clip_fraction = 0.0f;
    float mean_advantage = 0.0f;
    float mean_return = 0.0f;
    float mean_value = 0.0f;
    float grad_norm = 0.0f; ///< 最近一次裁剪前的全局梯度范数
};

/**
 * @class ContinuousPPO
 * @brief 连续动作 PPO 智能体
 *
 * 网络结构：obs -> fc1 -> ReLU -> fc2 -> ReLU，随后分两路
 *   - 策略头 -> 动作均值 μ
 *   - 价值头 -> 状态价值 V（标量）
 * 另有独立可学习参数 log_std。
 *
 * 参数集中存放在 _params 中，SGD 按原地写入更新（`p = p - lr*g` 会把计算图
 * 节点搬进参数槽位，导致图逐轮膨胀）。
 */
class ContinuousPPO {
  public:
    explicit ContinuousPPO(PpoConfig config = {}, std::uint32_t seed = 12345u);

    /// 采样动作（训练用）
    [[nodiscard]] ActionSample selectAction(const std::vector<float> &obs);

    /// 取策略均值（评估与部署用，确定性动作）
    [[nodiscard]] std::array<float, 3> actMean(const std::vector<float> &obs);

    /**
     * @brief 行为克隆：一步监督学习，让策略均值拟合给定动作（诊断用）
     *
     * 用途是把「网络与反向传播是否可用」从「PPO 的梯度估计是否正确」中分离出来。
     * 若连用专家动作做监督学习都无法降低损失，则问题在网络或反传路径，与 PPO
     * 的超参数无关。
     *
     * @return 本步的均方误差
     */
    float behaviorCloneStep(const std::vector<std::vector<float>> &obs_batch,
                            const std::vector<std::array<float, 3>> &action_batch);

    void store(const Transition &transition);
    void clearBuffer();
    [[nodiscard]] std::size_t bufferSize() const { return _buffer.size(); }

    /// 执行 update_epochs 轮 minibatch 更新，结束后清空缓冲区
    void update();

    [[nodiscard]] const PpoConfig &config() const { return _cfg; }
    [[nodiscard]] const std::vector<Tensor> &parameters() const { return _params; }

    /// 当前 log_std 的均值（诊断用：观察探索幅度是否失控）
    [[nodiscard]] float meanLogStd() const;

    /// 最近一次 update 的统计量
    [[nodiscard]] const UpdateStats &lastStats() const { return _last_stats; }

    /// 最近一次 sgdStep 中拿到非空梯度的参数个数（诊断用）
    [[nodiscard]] int paramsWithGrad() const { return _last_params_with_grad; }

    /// 参数总数
    [[nodiscard]] int paramCount() const { return static_cast<int>(_params.size()); }

    /// 各参数的梯度范数（诊断用；参数顺序同 ParamIndex，空梯度记 0）
    [[nodiscard]] std::vector<float> gradNorms();

    /// 各参数的数据范数（诊断用，用于判断是否真的发生了变化）
    [[nodiscard]] std::vector<float> paramNorms() const;

  private:
    /// 参数在 _params 中的下标
    enum ParamIndex : std::size_t {
        FC1_W = 0,
        FC1_B,
        FC2_W,
        FC2_B,
        MEAN_W,
        MEAN_B,
        V_W,
        V_B,
        LOG_STD,
        PARAM_COUNT
    };

    /// 前向：返回 (动作均值 [1, action_dim], 状态价值 标量)
    [[nodiscard]] std::tuple<Tensor, Tensor> forward(const Tensor &obs);

    /// 由策略均值重算给定动作的对数概率（保持计算图连接）
    /// @note 接收已算好的 mean 而非 obs，避免在训练循环中重复前向
    [[nodiscard]] Tensor logProbFrom(const Tensor &mean, const std::array<float, 3> &action);

    void initParameters();
    void zeroGrad();
    /// 执行一步 SGD，返回裁剪前的全局梯度范数（诊断用）
    float sgdStep();
    /// 按全局范数裁剪梯度（PPO 的标准稳定措施），返回裁剪前的范数
    float clipGradients(float max_norm);

    PpoConfig _cfg;
    std::vector<Tensor> _params;
    std::vector<Transition> _buffer;
    UpdateStats _last_stats;
    int _last_params_with_grad = 0;
    std::mt19937 _rng;
    std::normal_distribution<float> _normal{0.0f, 1.0f};
};

} // namespace oi3::rl

#endif // OI3_CONTINUOUS_PPO_H
