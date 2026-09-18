/**
 * @file ShootingExpert.h
 * @brief 打靶法专家：用可微动力学求一段最优推力序列，供行为克隆使用
 * @author GhostFace
 * @date 2026/9/18
 *
 * @par 为什么需要它
 *
 * PPO 冷启动很慢：在悬停任务上需要上千回合才能从随机策略爬到一个可用的解，而
 * 每一回合都要跑完整仿真。打靶法（shooting）换了一条路 —— 把控制序列本身当作
 * 参数，直接对轨迹末端的代价求导。它利用的是 CTorch 的可微动力学：一次前向 + 一次
 * 反向就能拿到整段序列的梯度，信噪比远高于用采样估计策略梯度。
 *
 * 但打靶法给出的是**开环序列**（针对某一个初始状态），而飞控需要的是**闭环策略**。
 * 两者的桥是行为克隆：用打靶法在多个初始状态上求解，把沿途的
 * (状态观测, 控制动作) 作为专家演示，监督训练策略网络。策略学到的不是某一条轨迹，
 * 而是「看到这样的状态该往哪推」这一映射 —— 于是它自带闭环纠偏能力，
 * 即使初始状态不在训练集里也能工作。
 *
 * @par 与 PID 专家的区别
 *
 * 项目里已有一条「PID 专家 → 行为克隆」的链路（BcDiagnostic）。PID 是**启发式**：
 * 它按固定增益把误差映射成推力，在非线性段（大误差、有阻力、限幅附近）并非最优。
 * 打靶法是**对给定代价函数最优**的：它会把「到达目标 + 末速归零 + 少用推力」
 * 一起权衡。因此它提供的教师信号理论上质量更高 —— 本模块的存在就是为了把这个
 * 判断变成可测量的对比，而不是停留在直觉上。
 *
 * @par 与环境的严格一致性
 *
 * 专家数据必须与 HoverEnv 的观测/动作契约**逐项一致**，否则策略学到的东西在环境里
 * 无法执行。为此本模块直接复用同一套换算：
 *   - 观测：位置误差 / target_range、速度 / vel_scale，再按 obs_clip 限幅；
 *   - 动作：`a = (thrust - hover) / (thrust_scale * m * g)`，限幅到 [-1, 1]；
 *   - 时间：一个控制段 = control_decimation 个仿真步（与环境的控制周期一致）。
 */

#ifndef OI3_SHOOTING_EXPERT_H
#define OI3_SHOOTING_EXPERT_H

#include "HoverEnv.h"

#include <array>
#include <vector>

namespace oi3 {

/// 打靶法求解配置
struct ShootingConfig {
    /// 控制段数（每段对应一个 RL 控制周期）
    int segments = 12;

    /// 梯度下降轮数
    int iters = 25;

    /**
     * @brief 每轮沿梯度方向的移动量，单位是**动作量纲**（即悬停推力的 thrust_scale 倍）
     *
     * 用符号梯度而非比例梯度：这条链的代价梯度量级随轨迹长度剧烈变化，
     * 固定比例步长要么走不动、要么直接发散；符号步长把「走多远」与梯度尺度解耦，
     * 只需保证代价函数本身是下降的。
     */
    float step = 0.2f;

    /// 代价权重
    double w_pos = 1.0;    ///< 末端位置误差平方
    double w_vel = 0.5;    ///< 末端速度平方
    double w_traj = 0.02;  ///< 沿途位置误差平方（防止「末端对了、中途飞出去」）
    double w_thrust = 0.002; ///< 推力偏离悬停量的平方（省能量、避免贴限幅）
};

/// 一次打靶法求解的结果
struct ShootingResult {
    /// 每段的推力（NED 系，牛顿）
    std::vector<std::array<double, 3>> thrust_seq;

    /// 每段起点的观测（维度与 HoverEnv::kObsDim 一致）
    std::vector<std::vector<float>> obs_seq;

    /// 每段施加的动作（维度 3，已按环境契约归一化并限幅）
    std::vector<std::array<float, 3>> action_seq;

    /// 收敛后的代价
    double cost = 0.0;

    /// 代价是否有限（发散时为 false，调用方应丢弃该样本）
    bool finite = true;

    /// 末段结束时的位置误差（米），用于评估专家质量
    double final_pos_error = 0.0;

    /// 初始位置误差（米）。与 final_pos_error 一起才能判断求解器是否真的改善了
    /// 轨迹：只有末值无法区分「收敛」与「原地不动」。
    double initial_pos_error = 0.0;
};

/**
 * @brief 用打靶法求解「从 pos0/vel0 飞到 target 并停住」的最优推力序列
 *
 * @param pos0     初始位置（NED，米）
 * @param vel0     初始速度（NED，米/秒）
 * @param target   目标位置（NED，米）
 * @param env      环境配置（必须与训练环境一致，观测/动作换算直接取自它）
 * @param sc       求解配置
 * @param warm_seq 热启动初值（可选）。滚动时域控制里每一次重规划的问题只比上一次
 *                 略作平移，用上一次解作为初值能把迭代数压到很低；传空则从悬停推力
 *                 出发冷启动。长度不足 segments 的部分按悬停推力补齐。
 * @return 推力序列与配套的 (观测, 动作) 演示数据
 */
[[nodiscard]] ShootingResult shootHover(const std::array<double, 3> &pos0,
                                        const std::array<double, 3> &vel0,
                                        const std::array<double, 3> &target,
                                        const HoverEnvConfig &env,
                                        const ShootingConfig &sc,
                                        const std::vector<std::array<double, 3>>
                                            &warm_seq = {});

} // namespace oi3

#endif // OI3_SHOOTING_EXPERT_H
