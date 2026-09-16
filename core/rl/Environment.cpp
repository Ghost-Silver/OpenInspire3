/**
 * @file Environment.cpp
 * @brief 强化学习搜索环境实现
 * @author GhostFace
 * @date 2026/4/4
 */

#include "Environment.h"
#include <cmath>
#include <algorithm>

// 构造函数
SearchEnvironment::SearchEnvironment(int nx, int ny, int nz, double resolution, 
                                   double sensor_radius, const std::vector<Vec3>& init_positions,
                                   int max_steps, const std::vector<AABB>& obstacles) 
    : map_(nx, ny, nz, resolution, sensor_radius),
      uav_manager_(init_positions.size()),
      current_step_(0),
      max_steps_(max_steps),
      time_penalty_(-0.005),
      explore_reward_(0.5),  // 合理的探索奖励，与贪心策略公平对比
      target_reward_(50.0),  // 提高目标奖励
      proximity_reward_(0.2), // 新增接近奖励
      dt_(0.03),
      init_positions_(init_positions),
      num_uavs_(init_positions.size()),
      num_targets_(5),     // 默认5个目标
      obstacles_(obstacles) {     // 存储障碍物信息
    // 初始化地图障碍物
    if (!obstacles.empty()) {
        map_.initObstacles(obstacles);
    }
    // 初始化地图目标（多目标）
    map_.initTargets(num_targets_, true, 42);
}

// 重置环境
void SearchEnvironment::reset() {
    map_.reset();
    // 重新初始化障碍物
    if (!obstacles_.empty()) {
        map_.initObstacles(obstacles_);
    }
    uav_manager_.reset(init_positions_);
    current_step_ = 0;
    // 重新初始化目标（多目标）
    map_.initTargets(num_targets_, true, 42);
}

// 执行一步
std::tuple<std::vector<float>, std::vector<std::vector<float>>, bool, std::vector<float>>
SearchEnvironment::step(const std::vector<Action>& actions) {
    // 1. 将动作转换为加速度
    std::vector<Vec3> accelerations = actionsToAccelerations(actions);
    
    // 2. 检查并调整加速度，避免碰撞
    std::vector<Vec3> adjusted_accelerations = accelerations;
    const std::vector<Vec3>& current_positions = uav_manager_.getPositions();
    
    for (int i = 0; i < num_uavs_; i++) {
        const Vec3& current_pos = current_positions[i];
        Vec3 velocity = ACTION_VELOCITY[static_cast<int>(actions[i])];
        Vec3 new_pos = current_pos + velocity * dt_;
        
        // 检查新位置是否有效
        if (!isPositionValid(new_pos)) {
            // 无效位置，调整为悬停
            adjusted_accelerations[i] = Vec3(0.0, 0.0, 0.0);
        }
    }
    
    // 3. 更新无人机状态
    uav_manager_.updateAll(dt_, adjusted_accelerations);
    
    // 4. 更新地图探索
    int total_new_cells = 0;
    const std::vector<Vec3>& positions = uav_manager_.getPositions();
    
    for (const Vec3& pos : positions) {
        total_new_cells += map_.updateExploration(pos.x(), pos.y(), pos.z(), current_step_);
    }
    
    // 5. 检查目标发现
    int new_targets = map_.checkNewTargets();
    
    // 6. 计算奖励
    std::vector<float> rewards = computeRewards(total_new_cells, new_targets);
    
    // 7. 判断终止条件
    bool done = (map_.getSearchedTarget() >= map_.getTotalTarget()) || (current_step_ >= max_steps_);
    
    // 8. 增加步数
    current_step_++;
    
    // 9. 返回结果
    return std::make_tuple(
        rewards,
        getLocalObservations(),
        done,
        getGlobalState()
    );
}

// 获取全局状态
std::vector<float> SearchEnvironment::getGlobalState() const {
    std::vector<float> state;
    
    // 1. 所有无人机的位置（归一化）
    const std::vector<Vec3>& positions = uav_manager_.getPositions();
    for (const Vec3& pos : positions) {
        state.push_back(pos.x() / 3.0); // 假设地图大小为 3x3x1
        state.push_back(pos.y() / 3.0);
        state.push_back(pos.z() / 1.0);
    }
    
    // 2. 全局覆盖率
    state.push_back(map_.getCoverageRatio());
    
    // 3. 已发现目标数 / 总目标数
    int total_target = map_.getTotalTarget();
    if (total_target > 0) {
        state.push_back(static_cast<float>(map_.getSearchedTarget()) / total_target);
    } else {
        state.push_back(0.0f);
    }
    
    // 4. 当前步数 / 最大步数
    state.push_back(static_cast<float>(current_step_) / max_steps_);
    
    return state;
}

// 获取每个无人机的局部观测
std::vector<std::vector<float>> SearchEnvironment::getLocalObservations() const {
    std::vector<std::vector<float>> observations;
    const std::vector<Vec3>& positions = uav_manager_.getPositions();
    
    for (const Vec3& pos : positions) {
        // 获取 5x5x3 的局部地图
        std::vector<float> local_map = map_.getLocalMap(pos.x(), pos.y(), pos.z(), 2, 1);
        observations.push_back(local_map);
    }
    
    return observations;
}

// 获取每个无人机的动作掩码
std::vector<std::vector<bool>> SearchEnvironment::getActionMasks() const {
    std::vector<std::vector<bool>> masks;
    const std::vector<Vec3>& positions = uav_manager_.getPositions();
    
    for (const Vec3& pos : positions) {
        std::vector<bool> mask(7, true); // 7个动作，默认都合法
        
        // 检查边界
        for (int i = 1; i < 7; i++) { // 跳过悬停
            Vec3 velocity = ACTION_VELOCITY[i];
            Vec3 new_pos = pos + velocity * dt_;
            if (!isPositionValid(new_pos)) {
                mask[i] = false;
            }
        }
        
        masks.push_back(mask);
    }
    
    return masks;
}

// 辅助：将动作转换为加速度
std::vector<Vec3> SearchEnvironment::actionsToAccelerations(const std::vector<Action>& actions) const {
    std::vector<Vec3> accelerations;
    for (Action action : actions) {
        int action_idx = static_cast<int>(action);
        // 动作对应的加速度 (m/s²)
        Vec3 acc = ACTION_VELOCITY[action_idx] * 10.0; // 假设加速度是速度的10倍
        accelerations.push_back(acc);
    }
    return accelerations;
}

// 辅助：计算奖励
std::vector<float> SearchEnvironment::computeRewards(int new_cells, int new_targets) {
    std::vector<float> rewards(num_uavs_, 0.0f);
    
    // 基础奖励 - 优化：直接使用标量计算
    float base_reward = time_penalty_ + new_cells * explore_reward_ + new_targets * target_reward_;
    
    // 获取无人机位置
    const std::vector<Vec3>& positions = uav_manager_.getPositions();
    
    // 获取未发现的目标位置 - 只在有目标时计算
    const auto& undiscovered_targets = map_.getUndiscoveredTargets();
    
    // 如果没有未发现的目标，直接返回基础奖励
    if (undiscovered_targets.empty()) {
        std::fill(rewards.begin(), rewards.end(), base_reward);
        return rewards;
    }
    
    // 计算接近奖励 - 优化：减少重复计算
    for (int i = 0; i < num_uavs_; i++) {
        const Vec3& pos = positions[i];
        double min_distance_sq = 100000000.0; // 使用平方距离避免开方
        
        // 找到最近的未发现目标
        for (const auto& target : undiscovered_targets) {
            double dx = pos.x() - std::get<0>(target);
            double dy = pos.y() - std::get<1>(target);
            double dz = pos.z() - std::get<2>(target);
            
            double dist_sq = dx*dx + dy*dy + dz*dz;
            if (dist_sq < min_distance_sq) {
                min_distance_sq = dist_sq;
            }
        }
        
        // 计算接近奖励（距离越近奖励越高）- 使用平方距离比较
        float proximity_reward = 0.0f;
        if (min_distance_sq < 25.0) { // 5米平方
            proximity_reward = proximity_reward_ * (5.0 - std::sqrt(min_distance_sq)) / 5.0;
        }
        
        // 总奖励
        rewards[i] = base_reward + proximity_reward;
    }
    
    return rewards;
}

// 辅助：检查位置是否有效（边界和障碍物）
bool SearchEnvironment::isPositionValid(const Vec3& pos) const {
    // 检查边界
    if (pos.x() < 0.0 || pos.x() > 12.0 || // 120体素 * 0.1分辨率
        pos.y() < 0.0 || pos.y() > 12.0 ||
        pos.z() < 0.0 || pos.z() > 3.0) {
        return false;
    }
    
    // 检查障碍物
    int gx, gy, gz;
    if (map_.worldToGrid(pos.x(), pos.y(), pos.z(), gx, gy, gz)) {
        const Cell& cell = map_.cellAt(gx, gy, gz);
        if (cell.isObs()) {
            return false;
        }
    }
    
    return true;
}
