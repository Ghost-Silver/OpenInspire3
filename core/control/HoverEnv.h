/**
 * @file HoverEnv.h
 * @brief 定点悬停环境：把 DroneSimulator 包装为强化学习环境
 * @author GhostFace
 * @date 2026/9/16
 *
 * 观测为「位置误差 + 速度」共 6 维；动作为三维推力，以悬停推力为中心按
 * 比例偏移。这样策略只需学习相对悬停的修正量，而不必从头学出 m*g 这个
 * 已知常量，显著降低学习难度。
 *
 * 奖励是位置误差、速度与推力偏离悬停量的加权平方和取负。三项都是有界的
 * 控制代价，不存在可以无限累积的正向项，因此不会出现「刷分」式退化解
 * （对比：以新探索格子数给正奖励时，最优策略会变成反复巡视而非完成任务）。
 */

#ifndef OI3_HOVER_ENV_H
#define OI3_HOVER_ENV_H

#include "DroneSimulator.h"
#include "DroneTypes.h"

#include <array>
#include <cstdint>
#include <random>
#include <vector>

namespace oi3 {

/// 定点悬停环境配置
struct HoverEnvConfig {
    Config plant; ///< 被控对象配置（质量、重力、阻力、推力限幅）

    double episode_seconds = 4.0; ///< 单回合时长（秒）

    /**
     * @brief 每个 RL 步对应的仿真步数
     *
     * 仿真仍以 `plant.dt`（默认 1 kHz）积分以保持数值精度，而控制以更低频率
     * 执行。这与真实飞控的内外环分层一致（内环 1 kHz 稳姿态，外环 50~100 Hz
     * 做位置与轨迹），同时把 RL 的步数按同一比例减少，否则 1 kHz 下一回合
     * 就是数千步，训练代价无法接受。
     */
    int control_decimation = 10;

    double target_range = 1.5;    ///< 目标位置各轴采样范围（±，米）
    double start_range = 0.3;     ///< 初始位置各轴采样范围（±，米）
    double vel_scale = 2.0;       ///< 观测中速度的归一化尺度（米/秒）

    /// 动作到推力的缩放：thrust = hover + a * thrust_scale * m * g
    /// a 会被限幅到 [-1, 1]
    double thrust_scale = 1.0;

    // 奖励权重
    double w_pos = 1.0;     ///< 位置误差平方权重
    double w_vel = 0.2;     ///< 速度平方权重
    double w_thrust = 0.005;///< 推力偏离悬停量的平方权重

    double success_tolerance = 0.1; ///< 到位判定半径（米）
    double success_speed = 0.5;     ///< 到位判定速度上限（米/秒）

    /**
     * @brief 回合内发散判据：位置误差超过该半径即终止本回合
     *
     * 训练早期策略尚不成熟时，无人机可能持续朝错误方向加速并迅速飞远。
     * 此时继续采样既无信息量，又会让观测超出训练分布（归一化后数值发散），
     * 使策略更难恢复。提前终止并给一次性惩罚是标准做法。
     */
    double abort_radius = 5.0;
    double abort_penalty = 10.0; ///< 发散终止时的一次性惩罚

    /// 观测各维限幅（防止发散时数值爆炸）
    double obs_clip = 5.0;

    /**
     * @brief 奖励整体缩放系数
     *
     * 回合总回报的量级需要与价值网络的学习能力匹配：若每步奖励为 O(1)、
     * 回合长度数百步，则回报量级在数十，价值损失与策略损失的梯度尺度才相当。
     * 否则价值损失会经由共享主干主导训练，策略本身难以改善。
     */
    double reward_scale = 1.0;
};

/**
 * @class HoverEnv
 * @brief 定点悬停环境
 */
class HoverEnv {
  public:
    static constexpr int kObsDim = 6;    ///< [位置误差(3), 速度(3)]
    static constexpr int kActionDim = 3; ///< 三维推力

    HoverEnv(HoverEnvConfig config, std::uint32_t seed);

    /// 开始新回合，返回初始观测
    [[nodiscard]] std::vector<float> reset();

    struct StepResult {
        std::vector<float> obs;
        float reward = 0.0f;
        bool done = false;
        bool success = false; ///< 仅在 done 时有意义
    };

    /// 执行一步：动作 a ∈ [-1,1]^3，映射为推力后送入仿真
    [[nodiscard]] StepResult step(const std::array<float, 3> &action);

    [[nodiscard]] int stepsPerEpisode() const { return _steps_per_episode; }
    [[nodiscard]] const HoverEnvConfig &config() const { return _cfg; }

    /// 最近一次回合的末端位置误差（米），供训练日志使用
    [[nodiscard]] double lastPositionError() const { return _last_pos_error; }

    /// 当前回合的目标位置（NED，米）
    [[nodiscard]] std::array<float, 3> target() const {
        return {static_cast<float>(_target[0]), static_cast<float>(_target[1]),
                static_cast<float>(_target[2])};
    }

    /// 当前飞行器状态（供解析控制器生成专家数据使用）
    [[nodiscard]] const DroneState &state() const { return _sim.state(); }

    /**
     * @brief 把推力换算为归一化动作（HoverEnv::step 的逆映射）
     *
     * a_i = (thrust_i - hover_i) / (thrust_scale * m * g)，并限幅到 [-1,1]。
     * 用于把解析控制器（如 PID）的输出转成策略的动作表示，以便生成专家数据。
     */
    [[nodiscard]] std::array<float, 3> thrustToAction(
        const std::array<double, 3> &thrust) const;

  private:
    [[nodiscard]] std::vector<float> makeObservation(double pos_err_n, double pos_err_e,
                                                     double pos_err_d) const;

    HoverEnvConfig _cfg;
    std::mt19937 _rng;
    std::uniform_real_distribution<double> _uniform{-1.0, 1.0};

    int _steps_per_episode = 0;
    int _step_index = 0;
    double _last_pos_error = 0.0;

    // 当前回合的目标位置（NED，米）；状态直接取自 _sim
    double _target[3] = {0.0, 0.0, 0.0};

    DroneSimulator _sim;
};

} // namespace oi3

#endif // OI3_HOVER_ENV_H
