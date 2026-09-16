/**
 * @file DroneSimulator.h
 * @brief OpenInspire3 飞行仿真器
 * @author GhostFace
 * @date 2026/9/16
 */

#ifndef OI3_DRONE_SIMULATOR_H
#define OI3_DRONE_SIMULATOR_H

#include "DroneDynamics.h"
#include "DroneTypes.h"

namespace oi3 {

/**
 * @class DroneSimulator
 * @brief 单机飞行仿真器
 *
 * 职责边界：持有状态与配置、推进时间、提供状态访问。
 * 动力学与积分步进在 `DroneDynamics.h`，本类只负责把它们串成仿真循环。
 *
 * 相对旧 Engine 的改动：
 *  - 去掉了 `_numl` + `std::vector<Tensor> _state` 这层未被使用的多体抽象
 *    （旧实现只为 rank 0 服务，却把状态存成 vector 并到处按索引访问）
 *  - 状态改用 `DroneState` 复合类型，不再是裸的 6 元扁平向量
 *  - 推进时不再每次构造 `std::function` 包装的 lambda，消除类型擦除开销
 *  - 不再持有独立的积分器对象：步长来自配置，直接传给 `rk4StepDrone`
 */
class DroneSimulator {
  public:
    DroneSimulator(Config config, DroneState initial_state);

    /**
     * @brief 推进一个步长
     * @param thrust 该步的合推力 {3}（NED 系，牛顿）
     */
    void step(const Tensor &thrust);

    /// 连续推进 n 步（推力保持不变）
    void step(const Tensor &thrust, int n);

    /// 重置到指定初始状态，时间与步数计数归零
    void reset(DroneState initial_state);

    [[nodiscard]] const DroneState &state() const { return _state; }
    [[nodiscard]] const Config &config() const { return _config; }
    [[nodiscard]] double time() const { return _time; }
    [[nodiscard]] long long stepCount() const { return _step_count; }

    /// 打印当前状态到 stdout
    void print() const;

  private:
    Config _config;
    DroneState _state;
    double _time = 0.0;
    long long _step_count = 0;
};

} // namespace oi3

#endif // OI3_DRONE_SIMULATOR_H
