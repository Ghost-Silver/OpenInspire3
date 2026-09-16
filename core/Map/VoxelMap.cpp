/**
 * @file VoxelMap.cpp
 * @brief OpenInspire3 空间块地图建模实现
 * @author GhostFace
 * @date 2026/3/29
 */

#include "VoxelMap.h"
#include <cmath>
#include <random>

// 构造函数
VoxelMap::VoxelMap(const int &width, const int &height, const int &depth,
                   const double &resolution, const double &sensor_r) 
    : _nx(width), _ny(height), _nz(depth), _resolution(resolution), 
      _ox(0), _oy(0), _oz(0), _sensor_r(sensor_r),
      _total_target(0), _searched_tar(0), _searched_voxel(0) {
    _voxels.resize(_nx * _ny * _nz);
    buildSphereOffsets();
}

// 坐标转换：世界坐标转体素索引
bool VoxelMap::worldToGrid(double x, double y, double z, int& gx, int& gy, int& gz) const {
    gx = static_cast<int>((x - _ox) / _resolution);
    gy = static_cast<int>((y - _oy) / _resolution);
    gz = static_cast<int>((z - _oz) / _resolution);
    return isValid(gx, gy, gz);
}

// 坐标转换：体素索引转世界坐标
void VoxelMap::gridToWorld(int gx, int gy, int gz, double& x, double& y, double& z) const {
    x = _ox + (gx + 0.5) * _resolution;
    y = _oy + (gy + 0.5) * _resolution;
    z = _oz + (gz + 0.5) * _resolution;
}

// 构建球形传感器偏移表
void VoxelMap::buildSphereOffsets() {
    _rank2off.clear();
    int radius = static_cast<int>(std::ceil(_sensor_r / _resolution));
    double radius_sq = (_sensor_r / _resolution) * (_sensor_r / _resolution);
    
    for (int dz = -radius; dz <= radius; ++dz) {
        for (int dy = -radius; dy <= radius; ++dy) {
            for (int dx = -radius; dx <= radius; ++dx) {
                double dist_sq = dx*dx + dy*dy + dz*dz;
                if (dist_sq <= radius_sq) {
                    _rank2off.push_back({dx, dy, dz});
                }
            }
        }
    }
}

// 传感器更新（核心）
int VoxelMap::updateExploration(double x, double y, double z, int current_step) {
    int cx, cy, cz;
    if (!worldToGrid(x, y, z, cx, cy, cz)) {
        return 0;
    }
    
    int new_explored = 0;
    for (const auto& offset : _rank2off) {
        int gx = cx + offset.dx;
        int gy = cy + offset.dy;
        int gz = cz + offset.dz;
        
        if (isValid(gx, gy, gz)) {
            Cell& cell = cellAt(gx, gy, gz);
            if (!cell.is_searched) {
                cell.is_searched = 1;
                cell.last_visit_time = current_step;
                new_explored++;
                _searched_voxel++;
            }
        }
    }
    
    return new_explored;
}

// 目标发现
int VoxelMap::checkNewTargets() {
    int new_found = 0;
    for (const auto& target : _target_voxels) {
        int gx = std::get<0>(target);
        int gy = std::get<1>(target);
        int gz = std::get<2>(target);
        
        if (isValid(gx, gy, gz)) {
            Cell& cell = cellAt(gx, gy, gz);
            if (cell.has_target && !cell.target_found && cell.is_searched) {
                cell.target_found = 1;
                new_found++;
                _searched_tar++;
            }
        }
    }
    
    return new_found;
}

// 局部地图提取（用于神经网络）
std::vector<float> VoxelMap::getLocalMap(double x, double y, double z, int radius_xy, int radius_z) const {
    int cx, cy, cz;
    if (!worldToGrid(x, y, z, cx, cy, cz)) {
        // 如果坐标转换失败，返回一个全为 -1.0 的向量
        int size_x = 2 * radius_xy + 1;
        int size_y = 2 * radius_xy + 1;
        int size_z = 2 * radius_z + 1;
        return std::vector<float>(size_x * size_y * size_z, -1.0f);
    }
    
    int size_x = 2 * radius_xy + 1;
    int size_y = 2 * radius_xy + 1;
    int size_z = 2 * radius_z + 1;
    std::vector<float> local_map(size_x * size_y * size_z, -1.0f);
    
    int idx = 0;
    for (int dz = -radius_z; dz <= radius_z; ++dz) {
        for (int dy = -radius_xy; dy <= radius_xy; ++dy) {
            for (int dx = -radius_xy; dx <= radius_xy; ++dx) {
                int gx = cx + dx;
                int gy = cy + dy;
                int gz = cz + dz;
                
                if (isValid(gx, gy, gz)) {
                    const Cell& cell = cellAt(gx, gy, gz);
                    if (cell.is_obs) {
                        local_map[idx] = 4.0f;
                    } else if (!cell.is_searched) {
                        local_map[idx] = 0.0f;
                    } else if (cell.has_target) {
                        local_map[idx] = cell.target_found ? 3.0f : 2.0f;
                    } else {
                        local_map[idx] = 1.0f;
                    }
                }
                idx++;
            }
        }
    }
    
    return local_map;
}

// 全局覆盖地图
std::vector<float> VoxelMap::getCoverageMap() const {
    std::vector<float> coverage_map(_nx * _ny * _nz);
    for (int i = 0; i < _voxels.size(); ++i) {
        coverage_map[i] = _voxels[i].is_searched ? 1.0f : 0.0f;
    }
    return coverage_map;
}

// 未发现目标位置
std::vector<std::tuple<double,double,double>> VoxelMap::getUndiscoveredTargets() const {
    std::vector<std::tuple<double,double,double>> targets;
    for (const auto& target : _target_voxels) {
        int gx = std::get<0>(target);
        int gy = std::get<1>(target);
        int gz = std::get<2>(target);
        
        if (isValid(gx, gy, gz)) {
            const Cell& cell = cellAt(gx, gy, gz);
            if (!cell.target_found) {
                double x, y, z;
                gridToWorld(gx, gy, gz, x, y, z);
                targets.emplace_back(x, y, z);
            }
        }
    }
    return targets;
}

// 重置
void VoxelMap::reset() {
    for (auto& cell : _voxels) {
        cell.is_searched = 0;
        cell.target_found = 0;
        cell.last_visit_time = -1;
    }
    _searched_tar = 0;
    _searched_voxel = 0;
}

// 目标初始化
void VoxelMap::initTargets(int num_targets, bool random, unsigned int seed) {
    _target_voxels.clear();
    _total_target = num_targets;
    
    if (random) {
        std::mt19937 rng(seed);
        std::uniform_int_distribution<int> dist_x(0, _nx - 1);
        std::uniform_int_distribution<int> dist_y(0, _ny - 1);
        std::uniform_int_distribution<int> dist_z(0, _nz - 1);
        
        int count = 0;
        while (count < num_targets) {
            int gx = dist_x(rng);
            int gy = dist_y(rng);
            int gz = dist_z(rng);
            
            Cell& cell = cellAt(gx, gy, gz);
            if (!cell.is_obs && !cell.has_target) {
                cell.has_target = 1;
                _target_voxels.emplace_back(gx, gy, gz);
                count++;
            }
        }
    }
}

// 障碍物初始化
void VoxelMap::initObstacles(const std::vector<AABB>& obstacles) {
    for (const auto& obstacle : obstacles) {
        int min_gx, min_gy, min_gz;
        int max_gx, max_gy, max_gz;
        
        worldToGrid(obstacle.min_x, obstacle.min_y, obstacle.min_z, min_gx, min_gy, min_gz);
        worldToGrid(obstacle.max_x, obstacle.max_y, obstacle.max_z, max_gx, max_gy, max_gz);
        
        for (int gz = std::max(0, min_gz); gz <= std::min(_nz - 1, max_gz); ++gz) {
            for (int gy = std::max(0, min_gy); gy <= std::min(_ny - 1, max_gy); ++gy) {
                for (int gx = std::max(0, min_gx); gx <= std::min(_nx - 1, max_gx); ++gx) {
                    cellAt(gx, gy, gz).is_obs = 1;
                }
            }
        }
    }
}
