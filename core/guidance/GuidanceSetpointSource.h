/**
 * @file GuidanceSetpointSource.h
 * @brief 任务模式制导源：让降级动作有轨迹可执行
 *
 * @par 与 FixedSetpointSource 的区别
 *
 * FixedSetpointSource 只返回一个固定位置，适合悬停场景。
 * 本类管理三种任务模式：
 * - Mission：返回预设任务目标（与 FixedSetpointSource 等价）；
 * - ReturnHome：触发时生成从当前位置回到起点的 MinimumSnap 轨迹，
 *   每周期采样跟踪；
 * - EmergencyLand：生成垂直下降到地面的轨迹，用于着陆检测与到达判定。
 *
 * @par 为什么 EmergencyLand 也要生成轨迹
 *
 * 紧急降落的实际控制由 DegradeExecutor 处理（切断力矩、输出固定下降推力）。
 * 但「没有轨迹」意味着系统无法判断「何时着陆完成」——只能凭超时或外部传感器。
 * 垂直下降轨迹提供了一个时间基准：轨迹执行完毕即认为已着陆，主循环据此终止。
 */

#ifndef OI3_GUIDANCE_SETPOINT_SOURCE_H
#define OI3_GUIDANCE_SETPOINT_SOURCE_H

#include "HalAbstraction.h"
#include "MinimumSnapTrajectory.h"
#include "TensorUtils.h"

#include <cmath>

namespace oi3 {

/// 制导任务模式
enum class GuidanceMode {
    Mission,       ///< 正常执行任务
    ReturnHome,    ///< 返航
    EmergencyLand, ///< 紧急降落
};

/**
 * @class GuidanceSetpointSource
 * @brief 根据降级决策自动切换制导模式的任务源
 */
class GuidanceSetpointSource : public HalSetpointSource {
  public:
    /**
     * @param mission_target 正常任务目标位置（NED）
     * @param home_pos       起飞点 / 返航目标（NED）
     * @param limits         轨迹约束（返航时使用）
     */
    GuidanceSetpointSource(const Tensor &mission_target,
                           const std::array<double, 3> &home_pos,
                           const TrajectoryLimits &limits,
                           GuidanceMode initial_mode = GuidanceMode::Mission)
        : _mission_target(mission_target), _home_pos(home_pos), _limits(limits),
          _mode(initial_mode) {}

    /**
     * @brief 降级决策变化时切换制导模式
     *
     * 由 FlightControlLoop 每周期在 applyDegradation 后调用。
     * 模式切换只发生在决策**变化**时，避免重复构建轨迹。
     */
    void onDecisionChanged(const DegradeDecision &d,
                           const std::array<double, 3> &current_pos,
                           double time) override {
        if (d.action == DegradeAction::ReturnHome && _mode != GuidanceMode::ReturnHome) {
            _mode = GuidanceMode::ReturnHome;
            buildReturnHomeTrajectory(current_pos, time);
        } else if (d.action == DegradeAction::EmergencyLand &&
                   _mode != GuidanceMode::EmergencyLand) {
            _mode = GuidanceMode::EmergencyLand;
            buildEmergencyLandTrajectory(current_pos, time);
        } else if (d.action == DegradeAction::Normal && _mode != GuidanceMode::Mission) {
            // 故障恢复后回到任务模式
            _mode = GuidanceMode::Mission;
            _trajectory.reset();
            _landed = false;
        }
    }

    [[nodiscard]] Tensor currentTarget(double time) override {
        _last_time = time;
        switch (_mode) {
        case GuidanceMode::Mission:
            return _mission_target;

        case GuidanceMode::ReturnHome: {
            FlatReference ref;
            if (_trajectory.sample(time, ref)) {
                return makeVec3(static_cast<float>(ref.pos[0]),
                                static_cast<float>(ref.pos[1]),
                                static_cast<float>(ref.pos[2]));
            }
            // 轨迹结束后悬停在 home
            return makeVec3(static_cast<float>(_home_pos[0]),
                            static_cast<float>(_home_pos[1]),
                            static_cast<float>(_home_pos[2]));
        }

        case GuidanceMode::EmergencyLand: {
            FlatReference ref;
            if (_trajectory.sample(time, ref)) {
                return makeVec3(static_cast<float>(ref.pos[0]),
                                static_cast<float>(ref.pos[1]),
                                static_cast<float>(ref.pos[2]));
            }
            // 轨迹结束 → 已着陆
            _landed = true;
            return makeVec3(static_cast<float>(_emergency_end_pos[0]),
                            static_cast<float>(_emergency_end_pos[1]),
                            static_cast<float>(_emergency_end_pos[2]));
        }
        }
        return _mission_target;
    }

    [[nodiscard]] bool hasArrived(const std::array<double, 3> &pos,
                                  double tolerance) const override {
        switch (_mode) {
        case GuidanceMode::Mission: {
            const auto t = toVector(_mission_target);
            const double dx = pos[0] - static_cast<double>(t[0]);
            const double dy = pos[1] - static_cast<double>(t[1]);
            const double dz = pos[2] - static_cast<double>(t[2]);
            return std::sqrt(dx * dx + dy * dy + dz * dz) < tolerance;
        }
        case GuidanceMode::ReturnHome: {
            const double dx = pos[0] - _home_pos[0];
            const double dy = pos[1] - _home_pos[1];
            const double dz = pos[2] - _home_pos[2];
            return std::sqrt(dx * dx + dy * dy + dz * dz) < tolerance;
        }
        case GuidanceMode::EmergencyLand:
            return isLanded();
        }
        return false;
    }

    [[nodiscard]] bool isLanded() const override {
        if (_mode != GuidanceMode::EmergencyLand) return false;
        return _landed;
    }

    [[nodiscard]] GuidanceMode mode() const { return _mode; }

    /// 当前轨迹总时长（秒），未构建时返回 0
    [[nodiscard]] double trajectoryDuration() const { return _trajectory.duration(); }

  private:
    void buildReturnHomeTrajectory(const std::array<double, 3> &from, double time) {
        _trajectory.reset();
        std::vector<GuidanceWaypoint> wps;
        wps.push_back({from});
        wps.push_back({_home_pos});
        _trajectory_start_time = time;
        if (!_trajectory.build(wps, _limits, time)) {
            _trajectory.reset();
        }
    }

    void buildEmergencyLandTrajectory(const std::array<double, 3> &from, double time) {
        _trajectory.reset();
        std::vector<GuidanceWaypoint> wps;
        wps.push_back({from});
        std::array<double, 3> ground = from;
        ground[2] = 0.0; // NED 地面 z = 0
        _emergency_end_pos = ground;
        wps.push_back({ground});
        _trajectory_start_time = time;

        // 紧急降落使用更保守的约束：缓慢下降
        TrajectoryLimits conservative = _limits;
        conservative.max_vel = 1.5;
        conservative.max_acc = 2.0;
        conservative.max_jerk = 5.0;

        if (!_trajectory.build(wps, conservative, time)) {
            _trajectory.reset();
        }
        _landed = false;
    }

    Tensor _mission_target;
    std::array<double, 3> _home_pos;
    TrajectoryLimits _limits;

    GuidanceMode _mode = GuidanceMode::Mission;
    MinimumSnapTrajectory _trajectory;
    double _trajectory_start_time = 0.0;

    mutable bool _landed = false;
    mutable double _last_time = -1.0;
    std::array<double, 3> _emergency_end_pos{};
};

} // namespace oi3

#endif // OI3_GUIDANCE_SETPOINT_SOURCE_H
