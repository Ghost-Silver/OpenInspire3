/**
 * @file ParameterIdentification.h
 * @brief 用可微仿真从实测轨迹辨识物理参数
 * @author GhostFace
 * @date 2026/9/18
 *
 * @par 为什么这是可微仿真的合理落点
 *
 * 前面两条路径都已用数据排除：离线专家生成（长窗口成本不可接受、短窗口专家完不成
 * 任务）与在线 MPC（收敛所需视野使单次求解成本过高，实时性差两个数量级）。两者的
 * 共同问题是**计算模式的时间尺度不匹配** —— 全轨迹反向是秒级，而闭环控制要毫秒级。
 *
 * 参数辨识没有这个矛盾：它**离线、一次性、不需要实时**，而且恰恰需要「整条轨迹的
 * 梯度」。质量、三轴转动惯量、阻力系数这些量在真实平台上通常是估的或按典型值填的，
 * 而它们直接决定仿真与真机的差距。有了梯度就能用**少量实测轨迹**做最小二乘拟合，
 * 而不是在高维参数空间里网格搜索。
 *
 * @par 与「打靶法优化控制序列」的关系
 *
 * 两者共用同一套可微动力学与同一套优化器（归一化梯度 + 回溯线搜索），区别只在优化
 * 对象：打靶法优化 N×3 维的控制序列，本模块优化**个位数维**的物理参数。维度低得
 * 多，因此同样的计算预算能跑更多轮、更容易收敛。
 *
 * @par 可辨识性的前提
 *
 * 参数只有在轨迹**激励**了它对应的物理效应时才可辨识：辨识阻力需要轨迹里速度明显
 * 变化，辨识惯量需要角速度明显变化。静止悬停轨迹对任何参数都不敏感 —— 那不是算法
 * 失败，而是数据里没有信息。本模块的测试会显式对比「有激励」与「无激励」两种数据，
 * 把这一点作为结论的一部分。
 */

#ifndef OI3_PARAMETER_IDENTIFICATION_H
#define OI3_PARAMETER_IDENTIFICATION_H

#include "SixDofTypes.h"

#include <array>
#include <vector>

namespace oi3 {

/**
 * @struct ParamSet
 * @brief 待辨识的物理参数集合
 *
 * 只放「真实平台上通常估不准」的量。重力加速度 g 是物理常数、无需辨识；
 * 力臂长度对转角加速度的影响已被惯量吸收，也不单独辨识。
 */
struct ParamSet {
    double mass = 1.0;                ///< 质量（千克）
    double inertia[3] = {0.01, 0.01, 0.02}; ///< 三轴转动惯量（kg·m²）
    double drag_coeff = 0.05;         ///< 逐轴二次阻力系数（N·s²/m²）
};

/// 一条用作辨识输入的控制-状态序列
struct IdentTrajectory {
    /// 初始状态
    SixDofState init;

    /// 每一步施加的控制（推力 {1} 牛顿、力矩 {3} N·m），长度 = steps
    std::vector<double> thrust;
    std::vector<std::array<double, 3>> torque;

    /// 每一步**之后**的真实状态观测（带噪），长度 = steps
    std::vector<std::array<double, 3>> pos;
    std::vector<std::array<double, 3>> vel;
    std::vector<std::array<double, 4>> quat;
    std::vector<std::array<double, 3>> omega;
};

/// 辨识配置
struct IdentConfig {
    int iters = 40;      ///< 迭代上限
    double step = 0.05;  ///< 归一化梯度的初始步长（相对参数初值的比例）
    double rel_tol = 1e-6; ///< 代价相对下降阈值 → 收敛
};

/// 辨识结果
struct IdentResult {
    ParamSet params;          ///< 辨识出的参数
    double initial_loss = 0.0;
    double final_loss = 0.0;
    int iters = 0;            ///< 实际迭代轮数
    bool converged = false;   ///< 是否因收敛判据退出（而非跑满迭代）
    bool finite = true;       ///< 过程中数值是否有效
};

/**
 * @brief 从一条或多条轨迹辨识物理参数
 *
 * @param data  观测轨迹（可多条：不同工况合起来能更好地激励各参数）
 * @param guess 参数初值（通常是当前的估算值）
 * @param cfg   辨识配置
 * @return 辨识结果，含收敛曲线端点与实际迭代数
 */
[[nodiscard]] IdentResult identifyParameters(const std::vector<IdentTrajectory> &data,
                                             const ParamSet &guess,
                                             const IdentConfig &cfg);

/**
 * @brief 用给定参数把一条轨迹的初始状态推进出来（不开梯度）
 *
 * 用于生成合成观测数据与评估拟合质量；与辨识内部走同一套方程。
 */
[[nodiscard]] IdentTrajectory simulateTrajectory(const SixDofState &init,
                                                 const std::vector<double> &thrust,
                                                 const std::vector<std::array<double, 3>> &torque,
                                                 const ParamSet &params,
                                                 const Config &base);

} // namespace oi3

#endif // OI3_PARAMETER_IDENTIFICATION_H
