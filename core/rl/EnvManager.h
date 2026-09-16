/**
 * @file EnvManager.h
 * @brief 环境管理器类，复用 SearchEnvironment 类
 * @author GhostFace
 * @date 2026/4/4
 */

#ifndef ENVMANAGER_H
#define ENVMANAGER_H

#include "Environment.h"
#include "ActionSpace.h"
#include <vector>
#include <tuple>

class EnvManager {
private:
    SearchEnvironment env;
    int num_uavs;

public:
    EnvManager(int nx, int ny, int nz, double resolution, double sensor_radius, const std::vector<Vec3>& init_positions, int max_steps = 500, const std::vector<AABB>& obstacles = std::vector<AABB>());

    // 重置环境
    void reset();

    // 执行一步
    std::tuple<std::vector<float>, std::vector<std::vector<float>>, bool, std::vector<float>> step(const std::vector<Action>& actions);

    // 获取局部观测
    std::vector<std::vector<float>> get_local_observations() const;

    // 获取全局状态
    std::vector<float> get_global_state() const;

    // 获取动作掩码
    std::vector<std::vector<bool>> get_action_masks() const;

    // 获取无人机数量
    int get_num_uavs() const;
    
    // 设置目标数量
    void set_num_targets(int num_targets);
};

#endif // ENVMANAGER_H
