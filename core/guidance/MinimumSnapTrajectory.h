#ifndef OI3_MINIMUM_SNAP_TRAJECTORY_H
#define OI3_MINIMUM_SNAP_TRAJECTORY_H

#include "DifferentialFlatness.h"   // FlatReference

#include <array>
#include <vector>

namespace oi3 {

/// 制导航点
struct GuidanceWaypoint {
    std::array<double, 3> pos{};  ///< 位置（NED，米）
};

/**
 * @brief 轨迹约束
 *
 * @warning 默认值是**保守初值**，不是飞行器的真实能力上限。
 *          调用方必须显式传入实测值。尤其 max_acc 的真实上限是 6.87 m/s²
 *          （受 max_tilt_deg = 35° 限制），**不是** max_accel 配置里的 12.0。
 *          详见 §4。
 */
struct TrajectoryLimits {
    double max_vel = 5.0;    ///< 最大速度（m/s）
    double max_acc = 6.87;   ///< 最大加速度（m/s²）
    double max_jerk = 20.0;  ///< 最大加加速度（m/s³），保守初值，待实测确认
};

/**
 * @class MinimumSnapTrajectory
 * @brief 分段七次多项式轨迹，最小化 snap 的积分
 */
class MinimumSnapTrajectory {
  public:
    MinimumSnapTrajectory() = default;

    /**
     * @brief 构建轨迹
     * @param waypoints  航点序列（至少 2 个），NED 坐标
     * @param limits     约束
     * @param start_time 轨迹起始时刻（秒），sample() 的 t 以此为零点基准
     * @param fixed_yaw  恒定偏航角（弧度），第一版不处理偏航机动
     * @return 构建成功返回 true；航点不足、求解失败、不可行时返回 false
     *
     * @note 调用方需保证相邻航点不重合；若重合，返回 false 而非静默处理。
     */
    bool build(const std::vector<GuidanceWaypoint> &waypoints,
               const TrajectoryLimits &limits, double start_time, double fixed_yaw = 0.0);

    /**
     * @brief 采样
     * @param t   绝对时刻（秒）
     * @param out 输出，直接填充 FlatReference
     * @return t 在 [start_time, start_time + duration()] 内返回 true，
     *         超出范围返回 false 且不修改 out
     *
     * @warning 边界条件：t == start_time 与 t == start_time + duration()
     *         都必须返回 true（闭区间）。
     */
    [[nodiscard]] bool sample(double t, FlatReference &out) const;

    /// 轨迹总时长（秒）。未构建时返回 0
    [[nodiscard]] double duration() const;

    /**
     * @brief 是否满足全部约束
     *
     * 需实际采样检查，不能仅凭构造时的假设返回 true。
     * 判据：max|vel| <= limits.max_vel 等，三轴分别检查。
     */
    [[nodiscard]] bool feasible() const;

    /// 清空状态，可重新 build
    void reset();

  private:
    /// 一段多项式：三轴系数 coeff[axis][k]（局部时间 t^k 升幂）+ 段时长与全局起点偏移
    struct Segment {
        std::array<std::array<double, 8>, 3> coeff{};
        double duration = 0.0; ///< 段时长 T（秒）
        double start = 0.0;    ///< 段起点相对 start_time 的偏移（秒）
    };

    std::vector<Segment> segs_;
    TrajectoryLimits limits_{};
    double start_time_ = 0.0;
    double duration_ = 0.0;
    double yaw_ = 0.0;
    bool built_ = false;
};

} // namespace oi3

#endif // OI3_MINIMUM_SNAP_TRAJECTORY_H
