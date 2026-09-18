/**
 * @file ShootingMpc.h
 * @brief 滚动时域控制器（MPC）：用可微动力学在线求解最优控制
 * @author GhostFace
 * @date 2026/9/18
 *
 * @par 为什么是 MPC 而不是「打靶法生成专家」
 *
 * 打靶法作为**离线专家生成器**已被实测否定（见 ShootingExpert.h 与 skills 文档）：
 * 要给出能真正完成任务的演示，窗口需长到 1.2 s 量级，单样本约 200 s，而行为克隆
 * 需要成百上千条演示；短窗口虽然便宜，但专家本身就完不成任务，用它初始化策略反而
 * 比朴素基线更差（-17.09 vs -5.11）。
 *
 * 换成**滚动时域**就成立：每个控制周期只需解**一次**短窗口问题、只执行第一步（或
 * 前几步）、下一周期重新规划。它不生成数据集，因而不受上述成本约束；而可微动力学
 * 恰好让「每步解一次带约束的最优控制」变得可行。
 *
 * @par 两个关键设计
 *
 * **热启动**：相邻两次规划的问题只差一小段时间平移，用上一次的解（向前平移几段）
 * 作为初值，能把所需迭代数压到很低 —— 这是让在线重规划可行的前提。
 *
 * **控制时域 < 预测时域**：规划 N 段但只执行前 M 段（M < N）再重规划。这是标准 MPC
 * 的折中：既利用了预测能力，又不必每个周期都重算。
 *
 * @par 与 PID 的关系
 *
 * PID 是解析的、零求解成本，在定点悬停这类任务上表现很好；MPC 的价值在于能显式
 * 处理约束（推力限幅、倾角上限）并对未来有预测。本模块的评估脚本会把两者放在同一
 * 环境、同一初始条件下对比，用数据说明各自适用面，而不是假设谁更优。
 */

#ifndef OI3_SHOOTING_MPC_H
#define OI3_SHOOTING_MPC_H

#include "HoverEnv.h"
#include "ShootingExpert.h"

#include <array>
#include <cstddef>
#include <vector>

namespace oi3 {

/**
 * @class ShootingMpc
 * @brief 基于打靶法的滚动时域控制器
 */
class ShootingMpc {
public:
    /// MPC 配置
    struct Config {
        /// 预测时域（段），每段对应一个控制周期
        int horizon = 20;

        /// 控制时域（段）：执行多少段之后重新规划，必须 <= horizon
        int control_horizon = 5;

        /// 每次规划的迭代上限（配合热启动，通常不需要很多）
        int iters = 12;

        /// 归一化梯度的初始步长（动作量纲）
        float step = 0.05f;

        /// 代价中沿途中段误差的权重（增大可抑制「末端对了、中途飞远」）
        double w_traj = 0.05;
    };

    ShootingMpc(HoverEnvConfig env, Config cfg);

    /**
     * @brief 以给定状态为起点重新规划
     * @param st     当前状态（读位置与速度）
     * @param target 目标位置（NED）
     * @return 规划是否成功（数值有效）
     *
     * 用上一次的解向前平移作为热启动；首次调用则冷启动（从悬停推力出发）。
     * 规划完成后 `hasPlan()` 为真，可连续调用 `nextAction()` 取控制时域内的动作。
     */
    bool replan(const DroneState &st, const std::array<double, 3> &target);

    /// 取下一个动作。计划耗尽时返回 false，调用方应重规划。
    bool nextAction(std::array<float, 3> &action);

    /// 当前是否还有可执行的动作
    [[nodiscard]] bool hasPlan() const { return _cursor < _actions.size(); }

    /// 剩余可执行动作数
    [[nodiscard]] std::size_t remaining() const { return _actions.size() - _cursor; }

    /// 上一次规划的代价（诊断用）
    [[nodiscard]] double lastCost() const { return _last_cost; }

    /// 累计规划次数（诊断用，用于评估在线求解成本）
    [[nodiscard]] int planCount() const { return _plan_count; }

    /// 累计规划耗时（秒，诊断用）
    [[nodiscard]] double planSeconds() const { return _plan_seconds; }

private:
    HoverEnvConfig _env;
    Config _cfg;

    /// 待执行的动作序列（长度 = control_horizon）
    std::vector<std::array<float, 3>> _actions;

    /// 上一次的推力解（热启动用）
    std::vector<std::array<double, 3>> _warm;

    std::size_t _cursor = 0;
    double _last_cost = 0.0;
    int _plan_count = 0;
    double _plan_seconds = 0.0;
};

} // namespace oi3

#endif // OI3_SHOOTING_MPC_H
