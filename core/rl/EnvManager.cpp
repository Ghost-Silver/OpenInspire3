/**
 * @file EnvManager.cpp
 * @brief 环境管理器类实现
 * @author GhostFace
 * @date 2026/4/4
 */

#include "EnvManager.h"

EnvManager::EnvManager(int nx, int ny, int nz, double resolution, double sensor_radius, const std::vector<Vec3>& init_positions, int max_steps, const std::vector<AABB>& obstacles) 
    : env(nx, ny, nz, resolution, sensor_radius, init_positions, max_steps, obstacles),
      num_uavs(init_positions.size()) {
}

void EnvManager::reset() {
    env.reset();
}

std::tuple<std::vector<float>, std::vector<std::vector<float>>, bool, std::vector<float>> EnvManager::step(const std::vector<Action>& actions) {
    return env.step(actions);
}

std::vector<std::vector<float>> EnvManager::get_local_observations() const {
    return env.getLocalObservations();
}

std::vector<float> EnvManager::get_global_state() const {
    return env.getGlobalState();
}

std::vector<std::vector<bool>> EnvManager::get_action_masks() const {
    return env.getActionMasks();
}

int EnvManager::get_num_uavs() const {
    return num_uavs;
}

void EnvManager::set_num_targets(int num_targets) {
    env.setNumTargets(num_targets);
}
