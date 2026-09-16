/**
 * @file Baseline.h
 * @brief 基线策略与统一评测驱动
 * @author GhostFace
 * @date 2026/9/16
 *
 * @details 原先 4 个基线测试（GreedyBaselineTest / HardTest / UltraHardTest /
 *          AMMAPPOUltraHardTest）各自复制了一份「环境循环 + 统计」骨架，差异仅
 *          在场景参数与所用策略上，合计约 1400 行且互不同步。此外还存在：
 *            - 策略签名不一致（简单贪心只收 obs，进阶贪心另收 uav_id /
 *              全部观测 / 障碍物）；
 *            - `Obstacle`（中心 + 尺寸）在每个文件里本地定义，并各自手写一遍
 *              到 `AABB` 的转换；
 *            - `generateObstacles()` 签名带 5 个参数却忽略全部实参、硬编码返回
 *              同样 5 个障碍物。
 *
 *          此处抽出三件事：
 *            1. **策略**：与环境解耦的纯函数，统一签名（协同所需信息经
 *               PolicyContext 传入，不用的策略直接忽略）
 *            2. **场景**：一份参数结构体，新增难度只需加一份配置
 *            3. **evaluate()**：统一评测驱动（内部处理 Obstacle → AABB 转换）
 *
 *          各测试因此退化为「选场景 + 指定策略 + 打印结果」的薄壳。
 */

#ifndef OI3_RL_BASELINE_H
#define OI3_RL_BASELINE_H

#include "ActionSpace.h"
#include <functional>
#include <string>
#include <vector>

/// 轴对齐包围盒，定义于 core/Map/VoxelMap.h（全局命名空间）
struct AABB;

namespace oi3::rl {

/// 圆形障碍物（以中心与直径描述）
struct Obstacle {
    Vec3 center;
    double size;
};

/// Obstacle（中心 + 尺寸）→ AABB
/// 供需要自行构造 EnvManager 的调用方复用（如 PPO 训练驱动）
std::vector<AABB> toAABBs(const std::vector<Obstacle> &obstacles);

/// 评测场景参数
struct Scenario {
    std::string name = "scenario";

    // 地图与传感器
    int map_x = 50;
    int map_y = 50;
    int map_z = 15;
    int num_targets = 5;
    double resolution = 0.1;
    double sensor_radius = 1.5;

    // 回合设置
    int max_steps = 300;
    int num_episodes = 50;

    /// 进入 episode 循环前的额外 reset 次数
    /// 原各测试在循环前的准备代码不同（easy 会先 reset 一次以读取观测维度），
    /// 而目标生成会随 reset 次数推进，因此该值会影响逐 episode 结果。
    /// 此处按原实现的实际调用序声明，以保证与既有实测数据逐一对齐。
    int warmup_resets = 0;

    /// 单机累计奖励达到该值即认为本机成功（沿用原判据）
    double success_reward = 100.0;

    /// 起始位置；其数量即为无人机数量
    std::vector<Vec3> start_positions;

    /// 障碍物（为空则无障碍）
    std::vector<Obstacle> obstacles;
};

/// 评测结果统计
struct EvalResult {
    int episodes = 0;
    int successes = 0;
    double success_rate = 0.0;
    double avg_reward = 0.0;
    double reward_std = 0.0;
    double avg_steps = 0.0;
    double steps_std = 0.0;
};

/// 策略运行时上下文（仅多机协同策略需要）
struct PolicyContext {
    int uav_id = 0;
    const std::vector<std::vector<float>> *all_observations = nullptr;
    const std::vector<Obstacle> *obstacles = nullptr;
};

/// 策略签名：给定本机观测与上下文，返回动作
using Policy = std::function<Action(const std::vector<float> &, const PolicyContext &)>;

// ---------------------------------------------------------------------------
// 内置基线策略
// ---------------------------------------------------------------------------

/// 简单贪心策略工厂：选择局部地图中未探索体素最多的方向
/// @param hover_weight 悬停方向的权重。原实现中 easy 用 0.5、hard 用 0.3，
///        两份副本口径不同，此处把该差异显式化为参数，各自保持原值。
Policy makeGreedyPolicy(double hover_weight = 0.5);

/// 进阶贪心：在简单贪心的基础上加入障碍规避、边界惩罚与多机去冲突
Action advancedGreedySelect(const std::vector<float> &obs, const PolicyContext &ctx);

// ---------------------------------------------------------------------------
// 评测驱动
// ---------------------------------------------------------------------------

/**
 * @brief 按场景跑 num_episodes 轮评测
 * @param scenario 场景参数
 * @param policy 待评测策略
 * @param verbose 是否打印每轮明细
 * @return 统计结果
 */
EvalResult evaluate(const Scenario &scenario, const Policy &policy, bool verbose = true);

/// 打印场景与结果摘要
void printResult(const Scenario &scenario, const EvalResult &result);

// ---------------------------------------------------------------------------
// 预置场景
// ---------------------------------------------------------------------------

Scenario makeEasyScenario();
Scenario makeHardScenario();
Scenario makeUltraHardScenario();

} // namespace oi3::rl

#endif // OI3_RL_BASELINE_H
