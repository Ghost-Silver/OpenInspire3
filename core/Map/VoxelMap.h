
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
struct Cell { // 单个Voxel
    uint8_t is_searched : 1;
    uint8_t has_target : 1;
    uint8_t is_obs : 1;  // 是否存在障碍物
    int last_visit_time; // 上次访问时间
    Cell() : is_searched(0), has_target(0), is_obs(0), last_visit_time(-1) {};

    bool isSearched() { return is_searched; }
    bool hasTarget() { return has_target; }
    bool isObs() { return is_obs; }

    int lastVisitTm() { return last_visit_time; }
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

  public:
    VoxelMap(const int &width, const int &height, const int &depth,
             const double &resolution, const double &sensor_r = 5.00)
        : _nx(width), _ny(height), _nz(depth), _resolution(resolution), _ox(0),
          _oy(0), _oz(0), _sensor_r(sensor_r) {
        _voxels.resize(_nx * _ny * _nz);
    };
};
#endif // VoxelMap.h
