
/**
 * @file VoxelMap.h
 * @brief OpenInspire3 空间块地图建模
 * @author GhostFace
 * @date 2026/3/29
 */

#ifndef VOXELMAP_h
#define VOXELMAP_h
#include <stdint.h>
#include <vector>
// AABB 结构体定义
struct AABB {
    double min_x, min_y, min_z;
    double max_x, max_y, max_z;
};

struct Cell { // 单个Voxel
    uint8_t is_searched : 1;
    uint8_t has_target : 1;
    uint8_t is_obs : 1;  // 是否存在障碍物
    uint8_t target_found : 1; // 目标奖励是否已发放
    int last_visit_time; // 上次访问时间
    Cell() : is_searched(0), has_target(0), is_obs(0), target_found(0), last_visit_time(-1) {};

    bool isSearched() const { return is_searched; }
    bool hasTarget() const { return has_target; }
    bool isObs() const { return is_obs; }
    bool targetFound() const { return target_found; }

    int lastVisitTm() const { return last_visit_time; }
};
struct Offset {
    int dx, dy, dz;
};
class VoxelMap {
  private:
    int _nx, _ny, _nz;    // 三维体素数量（亦即步长）
    double _resolution;   // 分辨率（米/体素）
    double _ox, _oy, _oz; // 坐标原点
    double _sensor_r;     // 传感器半径
    int _total_target;    // 总目标数量
    int _searched_tar;    // 搜索到的目标数量
    int _searched_voxel;  // 搜索到的体素数量
    std::vector<Cell> _voxels;
    std::vector<Offset> _rank2off;
    std::vector<std::tuple<int, int, int>> _target_voxels; // 目标体素坐标

  public:
    VoxelMap(const int &width, const int &height, const int &depth,
             const double &resolution, const double &sensor_r = 5.00);
    
    // 坐标转换
    bool worldToGrid(double x, double y, double z, int& gx, int& gy, int& gz) const;
    void gridToWorld(int gx, int gy, int gz, double& x, double& y, double& z) const;
    
    // 传感器偏移表
    void buildSphereOffsets();
    
    // 传感器更新
    int updateExploration(double x, double y, double z, int current_step);
    
    // 目标发现
    int checkNewTargets();
    
    // 局部地图提取
    std::vector<float> getLocalMap(double x, double y, double z, int radius_xy, int radius_z) const;
    
    // 全局覆盖地图
    std::vector<float> getCoverageMap() const;
    
    // 未发现目标位置
    std::vector<std::tuple<double,double,double>> getUndiscoveredTargets() const;
    
    // 重置
    void reset();
    
    // 目标/障碍物初始化
    void initTargets(int num_targets, bool random, unsigned int seed);
    void initObstacles(const std::vector<AABB>& obstacles);
    
    // 辅助函数
    bool isValid(int gx, int gy, int gz) const {
        return gx >= 0 && gx < _nx && gy >= 0 && gy < _ny && gz >= 0 && gz < _nz;
    }
    
    Cell& cellAt(int gx, int gy, int gz) {
        return _voxels[gx + gy * _nx + gz * _nx * _ny];
    }
    
    const Cell& cellAt(int gx, int gy, int gz) const {
        return _voxels[gx + gy * _nx + gz * _nx * _ny];
    }
    
    // 获取已发现目标数
    int getSearchedTarget() const {
        return _searched_tar;
    }
    
    // 获取总目标数
    int getTotalTarget() const {
        return _total_target;
    }
    
    // 获取覆盖率
    double getCoverageRatio() const {
        return static_cast<double>(_searched_voxel) / (_nx * _ny * _nz);
    }
};
#endif // VoxelMap.h
