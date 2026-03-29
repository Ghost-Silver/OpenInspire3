
/**
 * @file GridMap.h
 * @brief OpenInspire3 空间块地图建模
 * @author GhostFace
 * @date 2026/3/29
 */

#ifndef GridMap_h
#define GridMap_h
#include <vector>
struct Cell {
    bool is_searched;
    bool has_target;
    bool is_obs;
    int last_visit_time;
    std::vector<int> visited_drone;
    std::vector<int> visited_time;
};
class GridMap {
  private:
    std::vector<Cell> grids;
};
#endif // GridMap.h
