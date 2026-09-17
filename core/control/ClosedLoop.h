/**
 * @file ClosedLoop.h
 * @brief 闭环飞行仿真驱动与飞行品质评价
 * @author GhostFace
 * @date 2026/9/16
 *
 * 把「控制器 → 推力 → 动力学 → 新状态 → 控制器」串成闭环，并给出可比较的
 * 定量指标。控制器的差异只体现在 computeThrust 的实现上，任务、积分器与
 * 评价口径完全一致。
 */

#ifndef OI3_CLOSED_LOOP_H
#define OI3_CLOSED_LOOP_H

#include "Controller.h"
#include "DroneTypes.h"
#include "Tensor.h"

#include <iostream>
#include <string>
#include <vector>

namespace oi3 {

/**
 * @struct FlightTask
 * @brief 闭环飞行任务：从初始状态飞抵并保持在目标点
 */
struct FlightTask {
    std::string name = "hover";

    Tensor target;      ///< 期望位置 {3}（NED，米）
    Tensor initial_pos; ///< 初始位置 {3}（NED，米）
    Tensor initial_vel; ///< 初始速度 {3}（NED，米/秒）

    double duration = 5.0;   ///< 仿真时长（秒）
    double tolerance = 0.05; ///< 收敛带宽（米）：位置误差模小于该值视为到位
    double hold_ratio = 0.1; ///< 稳态量取末段该比例时长内的平均
};

/**
 * @struct FlightTrace
 * @brief 闭环仿真轨迹（按固定间隔采样，避免逐微秒存点）
 */
struct FlightTrace {
    std::vector<double> time;
    std::vector<float> pos_n, pos_e, pos_d;
    std::vector<float> vel_n, vel_e, vel_d;
    std::vector<float> thrust_n, thrust_e, thrust_d;
};

/**
 * @struct FlightMetrics
 * @brief 飞行品质指标
 */
struct FlightMetrics {
    double settle_time = 0.0;   ///< 收敛时间（秒）：最后一次离开容差带之后
    double steady_error = 0.0;  ///< 稳态误差（米）：末段位置误差模的平均
    double max_overshoot = 0.0; ///< 最大超调（米）：越过目标点的最大距离
    double max_thrust = 0.0;    ///< 峰值推力模（牛顿）
    double control_energy = 0.0;///< 控制代价 ∫|F|²dt（N²·s）
    double terminal_error = 0.0;///< 终止时刻的位置误差模（米）
    bool converged = false;     ///< 是否在时限内收敛（末段稳定落在容差带内）
};

/**
 * @brief 运行一次闭环仿真
 * @param cfg        仿真配置（步长、质量、重力）
 * @param task       飞行任务
 * @param controller 控制器（内部会被 reset）
 * @return 采样轨迹
 */
[[nodiscard]] FlightTrace runClosedLoop(const Config &cfg, const FlightTask &task,
                                        Controller &controller);

/**
 * @brief 由轨迹计算飞行品质指标
 */
[[nodiscard]] FlightMetrics evaluateTrace(const FlightTask &task, const FlightTrace &trace);

/**
 * @brief 打印任务、控制器与指标
 */
void printMetrics(const FlightTask &task, const Controller &controller,
                  const FlightMetrics &metrics, std::ostream &os = std::cout);

/**
 * @brief 打印轨迹的定点摘要（每隔若干采样点一行）
 */
void printTrace(const FlightTrace &trace, std::size_t max_rows = 10,
                std::ostream &os = std::cout);

} // namespace oi3

#endif // OI3_CLOSED_LOOP_H
