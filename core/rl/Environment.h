/**
 * @file Environment.h
 * @brief 强化学习搜索环境
 * @author GhostFace
 * @date 2026/4/4
 */

#ifndef ENVIRONMENT_H
#define ENVIRONMENT_H

#include "ActionSpace.h"
#include "../Map/VoxelMap.h"
#include <vector>
#include <tuple>
// 3D 向量类（已在 ActionSpace.h 中定义）

class SearchEnvironment {
public:
    // 构造函数：地图尺寸（体素数量）、分辨率、传感器半径、无人机初始位置列表、最大步数等
    SearchEnvironment(int nx, int ny, int nz, double resolution, 
                     double sensor_radius, const std::vector<Vec3>& init_positions,
                     int max_steps, const std::vector<AABB>& obstacles = std::vector<AABB>());

    // 重置环境：重置地图、无人机位置、步数计数器，返回初始观测
    void reset();

    // 执行一步：输入每个无人机的动作（Action枚举值），
    // 输出（奖励列表, 下一局部观测列表, 是否终止, 全局状态）
    std::tuple<std::vector<float>,                     // 每个无人机的奖励（标量）
               std::vector<std::vector<float>>,        // 每个无人机的局部观测（展平）
               bool,                                   // 终止标志
               std::vector<float>>                     // 全局状态（用于集中式Critic）
    step(const std::vector<Action>& actions);

    // 获取当前全局状态（用于Critic）
    std::vector<float> getGlobalState() const;

    // 获取每个无人机的局部观测（用于Actor）
    std::vector<std::vector<float>> getLocalObservations() const;

    // 获取每个无人机的动作掩码（合法动作列表）
    std::vector<std::vector<bool>> getActionMasks() const;

    // 设置目标数量
    void setNumTargets(int num_targets) { num_targets_ = num_targets; }

private:
    VoxelMap map_;              // 体素地图
    UAVManager uav_manager_;    // 无人机管理器
    int current_step_;          // 当前步数
    int max_steps_;             // 最大步数
    double time_penalty_;       // 每步负奖励
    double explore_reward_;     // 每个新探索体素奖励
    double target_reward_;      // 每个新发现目标奖励
    double proximity_reward_;   // 接近目标奖励
    double dt_;                 // 仿真步长
    std::vector<Vec3> init_positions_; // 初始位置
    int num_uavs_;              // 无人机数量
    int num_targets_ = 5;       // 目标数量（默认可配置）
    std::vector<AABB> obstacles_; // 障碍物信息

    // 辅助：将动作转换为加速度
    std::vector<Vec3> actionsToAccelerations(const std::vector<Action>& actions) const;

    // 辅助：计算奖励（所有无人机共享总奖励）
    std::vector<float> computeRewards(int new_cells, int new_targets);

    // 辅助：检查位置是否在边界内
    bool isPositionValid(const Vec3& pos) const;
};

#endif // ENVIRONMENT_H
