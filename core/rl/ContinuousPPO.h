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

    void store(const Transition &transition);
    void clearBuffer();
    [[nodiscard]] std::size_t bufferSize() const { return _buffer.size(); }

    /// 执行 update_epochs 轮 minibatch 更新，结束后清空缓冲区
    void update();

    [[nodiscard]] const PpoConfig &config() const { return _cfg; }
    [[nodiscard]] const std::vector<Tensor> &parameters() const { return _params; }

    /// 当前 log_std 的均值（诊断用：观察探索幅度是否失控）
    [[nodiscard]] float meanLogStd() const;

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
    void sgdStep();

    PpoConfig _cfg;
    std::vector<Tensor> _params;
    std::vector<Transition> _buffer;
    std::mt19937 _rng;
    std::normal_distribution<float> _normal{0.0f, 1.0f};
};

} // namespace oi3::rl

#endif // OI3_CONTINUOUS_PPO_H
