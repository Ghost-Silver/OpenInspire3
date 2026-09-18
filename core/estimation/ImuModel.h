/**
 * @file ImuModel.h
 * @brief 惯性测量单元（IMU）误差模型
 * @author GhostFace
 * @date 2026/9/18
 *
 * @par 为什么需要它
 *
 * 在此之前，控制器直接读仿真真值（`ctrl.compute(sim.state(), ...)`）—— 真机上不存在
 * 这个信号。要验证状态估计器、也要让控制器的性能数字变得诚实，就必须先有传感器模型：
 * 真值 → 加噪测量 → 估计器 → 控制器。
 *
 * @par 建模了什么，没建模什么
 *
 * 建模（这几项是 IMU 误差的主导项）：
 *   - **白噪声**：加速度计与陀螺仪的测量噪声；
 *   - **常值偏置**：装配误差与温漂的常值分量；
 *   - **偏置随机游走**：陀螺偏置随时间缓慢漂移，这是纯积分姿态必然发散的原因，
 *     也是加速度计校正必须存在的理由；
 *   - **加速度计低通**：真实 IMU 内部有抗混叠滤波，会引入相位滞后。
 *
 * 未建模（需要更完整的器件手册或实测数据）：标度因数误差、三轴非正交、
 * 非线性、温度特性、量化噪声、采样抖动。这些属于「器件级建模」，
 * 对验证互补滤波/EKF 的鲁棒性不是必需 —— 前面那几项已经足以暴露估计器的核心问题。
 *
 * @note 加速度计测量的是**比力**（specific force），不是运动加速度：静止悬停时
 *       它读出的是「支撑力方向」，即 -g 方向。这是姿态估计能用加速度计定姿的物理基础。
 */

#ifndef OI3_IMU_MODEL_H
#define OI3_IMU_MODEL_H

#include "SixDofTypes.h"

#include <array>
#include <cstdint>
#include <random>

namespace oi3 {

/// IMU 误差参数（量级参考消费级 MEMS，如 MPU6000/ICM42688 一档）
struct ImuConfig {
    /// 加速度计白噪声标准差（m/s²）
    double accel_noise = 0.08;

    /// 陀螺仪白噪声标准差（rad/s）
    double gyro_noise = 0.004;

    /// 常值偏置模式：false = 按下面的幅值随机采样方向（每次构造都不同）；
    /// true = 使用显式的三轴取值。后者是为了让「偏置估计」这类断言可复现 ——
    /// 随机偏置下断言阈值只能按最坏情况放宽，反而验证不了估计器是否真的在工作。
    bool explicit_bias = false;

    /// 加速度计常值偏置幅值（m/s²），随机方向时使用
    double accel_bias = 0.03;

    /// 陀螺仪常值偏置幅值（rad/s），随机方向时使用
    double gyro_bias = 0.002;

    /// 显式三轴偏置（explicit_bias = true 时生效）
    std::array<double, 3> accel_bias_vec{0.03, -0.02, 0.01};
    std::array<double, 3> gyro_bias_vec{0.002, -0.0015, 0.0025};

    /// 陀螺偏置随机游走速率（rad/s per sqrt(s)）
    double gyro_bias_walk = 2e-5;

    /// 加速度计低通截止频率（Hz），0 表示不滤波
    double accel_lpf_hz = 60.0;

    /// 是否启用测量（便于做「理想传感器」对照）
    bool enabled = true;
};

/// 一次 IMU 采样
struct ImuSample {
    std::array<double, 3> accel{}; ///< 比力（机体坐标系，m/s²）
    std::array<double, 3> gyro{};  ///< 角速度（机体坐标系，rad/s）
};

/**
 * @class ImuModel
 * @brief 由真值状态生成带误差的 IMU 测量
 *
 * 状态无关的偏置在构造时采样一次（模拟出厂标定后的残余偏置），
 * 陀螺偏置在每次测量时做随机游走累积。
 */
class ImuModel {
public:
    ImuModel(ImuConfig cfg, std::uint32_t seed);

    /**
     * @brief 生成一次测量
     * @param truth 真值状态（读姿态与角速度；位置/速度用于构造比力）
     * @param specific_force_ned 真值比力（NED 系，m/s²）。由调用方从动力学给出；
     *        这里显式传入而不是内部推算，是为了避免与动力学重复实现导致口径漂移。
     * @param dt 采样间隔（秒），用于偏置随机游走与低通滤波
     */
    ImuSample measure(const SixDofState &truth,
                      const std::array<double, 3> &specific_force_ned, double dt);

    /// 当前陀螺偏置真值（用于评估估计器的偏置估计）
    [[nodiscard]] const std::array<double, 3> &gyroBiasTruth() const { return _gyro_bias; }

    void reset();

private:
    /// 采样常值偏置（构造与复位共用，保证两条路径的偏置来源一致）
    void sampleBiases();

    ImuConfig _cfg;
    std::mt19937 _rng;
    std::normal_distribution<double> _unit{0.0, 1.0};

    std::array<double, 3> _accel_bias{};
    std::array<double, 3> _gyro_bias{};

    /// 加速度计低通状态
    std::array<double, 3> _accel_lpf_state{};
    bool _lpf_initialized = false;
};

} // namespace oi3

#endif // OI3_IMU_MODEL_H
