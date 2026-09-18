/**
 * @file StateEstimator.h
 * @brief 状态估计：互补滤波姿态估计 + 位置/速度滤波
 * @author GhostFace
 * @date 2026/9/18
 *
 * @par 它补上的是什么
 *
 * 此前控制器直接读仿真真值（`ctrl.compute(sim.state(), ...)`）—— 真机上没有这个信号。
 * 本模块从 IMU 与位置传感器**估计**出控制器需要的状态，于是整条链路变成
 * 「真值 → 传感器 → 估计 → 控制」，控制器的性能数字也才有真机参考意义。
 *
 * @par 姿态：互补滤波
 *
 * 陀螺仪短时精确但积分会漂移（偏置随机游走使它必然发散）；加速度计长期无漂移
 * 但噪声大、且在机动时测的是比力而非重力。互补滤波把两者的频段拼起来：
 *
 * @verbatim
 *   ω_c = ω_gyro + Kp · (f̂_meas × f̂_est)      // 用加速度计方向修正陀螺
 *   q̇  = ½ · q ⊗ [0, ω_c]                     // 积分
 * @endverbatim
 *
 * 其中 f̂_meas 是归一化的加速度计读数（机体坐标系下的比力方向），f̂_est 是由当前
 * 姿态估计推出的同一方向。叉积给出修正旋转轴，Kp 决定「相信加速度计多少」。
 *
 * **Kp 的物理含义是截止频率**：低频段信任加速度计（无漂移），高频段信任陀螺
 * （无噪声）。Kp 太大则机动时被比力污染，太小则漂移收敛慢。
 *
 * @par 陀螺偏置：由互补滤波的修正量隐式估计
 *
 * 修正项 `Kp·e` 在稳态下恰好抵消陀螺偏置造成的漂移，因此把它的低通值取出来就是
 * 偏置估计。这一步不额外增加传感器，却能让姿态在有偏置的情况下仍然收敛 ——
 * 也是「不用磁力计能否定住 yaw」的关键（见下）。
 *
 * @par 位置/速度：IMU 预积分 + 固定增益校正
 *
 * 预测用高频 IMU（1 kHz），校正用低频位置测量（100 Hz）：
 *
 * @verbatim
 *   // 每个 IMU 周期（预测）：
 *   a_ned = R(q)·f_body + [0, 0, g]          // 比力换算成惯性加速度
 *   p ← p + v·dt + ½·a_ned·dt²
 *   v ← v + a_ned·dt
 *
 *   // 每个位置周期（校正）：
 *   r = p_meas − p
 *   p ← p + α·r
 *   v ← v + (β/dt)·r        β = α²/(2−α)（临界阻尼）
 * @endverbatim
 *
 * 结构与 EKF 一致（IMU 预测 / 观测校正），差别只在增益是固定值而非由协方差
 * 递推得到的时变增益。
 *
 * @par 两条走过弯路，都记在这里免得再犯
 *
 * 1. **「位置低通 + 差分求速度」是错的，不是简化**。位置低通后相邻样本仍高度
 *    相关（α=0.35 时相关系数 0.65），差分除以 dt 把残差噪声放大 1/dt 倍 ——
 *    位置噪声 0.02 m 时速度估计噪声高达 0.77 m/s。控制器拿它做阻尼，闭环直接
 *    起极限环（稳态位置误差 0.2 m，比位置噪声本身还大一个量级）。
 * 2. **只用 α-β（匀速预测）仍然不够**。速度由位置残差驱动，机动时 α-β 假设的
 *    「匀速」与实际不符，实测速度估计误差仍有 0.34 m/s，稳态倾角被抬到 9.9 度
 *    （理想反馈下 0.26 度）。加速度计本来就带着运动加速度信息，把它接进预测步
 *    才是对的。
 */

#ifndef OI3_STATE_ESTIMATOR_H
#define OI3_STATE_ESTIMATOR_H

#include "ImuModel.h"
#include "SixDofTypes.h"

#include <array>

namespace oi3 {

/// 估计器配置
struct EstimatorConfig {
    /**
     * @brief 加速度计校正增益 Kp（1/s），互补滤波的截止频率
     *
     * 这个增益的经典取值是 0.5~2.0，但那个经验值的前提是**机动加速度相对重力
     * 可忽略**。四旋翼做大机动时并非如此：位置环限幅 12 m/s²，比力模长可比 g
     * 大出一倍以上，此时加速度计读的是比力而不是重力，把它当作重力方向去校正
     * 姿态，会把运动加速度当成姿态误差「烧」进姿态估计里。
     *
     * 实测（EstimatorTest 的增益扫描，全估计闭环 4 秒悬停）：
     * @verbatim
     *   Kp     稳态位置 RMS   稳态倾角   倾角估计误差
     *   1.0    0.200 m        8.8°       3.2°
     *   0.3    0.042 m        1.0°       0.68°
     *   0.1    0.031 m        0.64°      0.36°
     *   0.05   0.029 m        0.61°      0.32°
     *   ≤0.03  0.028 m        0.59°      0.31°   ← 饱和：已到位置噪声底
     * @endverbatim
     * 经典值在这里差了 7 倍。0.05 以下不再改善，因为位置测量本身有 2 cm 噪声，
     * 闭环精度已经触到传感器噪声底。
     *
     * 默认取拐点附近的 0.05。Kp 偏小只会减慢姿态初始对准与漂移收敛，不损害
     * 稳态精度（加速度计自身的常值偏置造成的方向误差与 Kp 无关）。
     */
    double accel_correction = 0.05;

    /// 偏置估计增益（1/s²）。取修正量的低通，用于估计陀螺零偏。
    double bias_correction = 0.05;

    /**
     * @brief 位置校正增益 α（0~1）：越大越信任位置测量，越小越信任 IMU 预测
     *
     * 这个值不能随手取。校正注入速度的增益是 `β/dt = α²/((2−α)·dt)`，与 α 近似
     * 平方关系，而位置噪声正比地进入速度估计：α=0.35、位置噪声 0.02 m 时
     * 每 10 ms 往速度里注入 0.148 m/s 的噪声 —— 控制器用它做阻尼（kd=3）会被
     * 放大三倍。取 α=0.1 时该增益降到 0.53 /s，速度噪声降到约 0.011 m/s。
     *
     * 但这只是「降低速度噪声」这一个方向的考虑。实测扫描（EstimatorTest，
     * 姿态通路修好之后重扫）显示 α 的主导影响其实是**位置估计的相位滞后**：
     * @verbatim
     *   α      稳态位置 RMS   位置估计误差   速度估计误差
     *   0.02   0.077 m        0.023 m        0.054 m/s   ← 校正太慢，位置反馈滞后→振荡
     *   0.05   0.045 m        0.0093 m       0.032 m/s
     *   0.1    0.029 m        0.0095 m       0.046 m/s
     *   0.2    0.023 m        0.014 m        0.13 m/s
     *   0.35   0.022 m        0.019 m        0.33 m/s    ← 最佳：位置跟得上，速度噪声被控制环滤掉
     *   0.5    0.041 m        0.024 m        0.66 m/s    ← 校正过强，噪声灌进控制环
     * @endverbatim
     * 注意 α=0.35 时速度估计误差（0.33 m/s）比 α=0.05 时（0.032 m/s）**大一个
     * 量级**，闭环却好一倍 —— 说明「速度估计误差」不是闭环性能的好预测指标，
     * 控制环本身会滤掉一部分速度噪声，而位置滞后是无法被滤掉的纯相位损失。
     * 因此 α 按「位置反馈滞后」来选，默认 0.35。
     */
    double pos_filter_alpha = 0.35;

    /// 重力加速度（m/s²），用于把加速度计的比力换算成惯性加速度
    double gravity = 9.81;

    /// 是否启用偏置估计
    bool estimate_gyro_bias = true;
};

/**
 * @class StateEstimator
 * @brief 从 IMU 与位置测量估计控制器所需的状态
 */
class StateEstimator {
public:
    explicit StateEstimator(EstimatorConfig cfg = {});

    /// 复位到给定姿态（通常取初始加速度计读数对齐）
    void reset(const std::array<double, 4> &initial_quat = {1.0, 0.0, 0.0, 0.0});

    /**
     * @brief IMU 更新（高频，与控制器同频）
     * @param imu IMU 测量
     * @param dt  采样间隔（秒）
     */
    void updateImu(const ImuSample &imu, double dt);

    /**
     * @brief 位置测量更新（低频，例如 100 Hz）
     * @param pos_meas 位置测量（NED，米）
     * @param dt       距上次位置更新的时间（秒）
     */
    void updatePosition(const std::array<double, 3> &pos_meas, double dt);

    /// 用给定四元数直接设定姿态（仅用于初始化对齐，不用于运行时）
    void setAttitude(const std::array<double, 4> &quat);

    /// 估计出的状态，供控制器使用
    [[nodiscard]] SixDofState state() const;

    /// 估计姿态（w,x,y,z）
    [[nodiscard]] const std::array<double, 4> &attitude() const { return _quat; }

    /// 去掉偏置后的角速度估计（机体系，rad/s）
    [[nodiscard]] const std::array<double, 3> &angularRate() const { return _omega; }

    /// 陀螺偏置估计（rad/s）
    [[nodiscard]] const std::array<double, 3> &gyroBias() const { return _gyro_bias; }

    /// 位置估计（NED，米）
    [[nodiscard]] const std::array<double, 3> &position() const { return _pos; }

    /// 速度估计（NED，m/s）
    [[nodiscard]] const std::array<double, 3> &velocity() const { return _vel; }

    /// 位置测量是否已经接入（未接入时位置保持初值）
    [[nodiscard]] bool hasPosition() const { return _has_pos; }

private:
    EstimatorConfig _cfg;

    std::array<double, 4> _quat{1.0, 0.0, 0.0, 0.0};
    std::array<double, 3> _omega{};
    std::array<double, 3> _gyro_bias{};
    std::array<double, 3> _pos{};
    std::array<double, 3> _vel{};
    bool _has_pos = false;
};

} // namespace oi3

#endif // OI3_STATE_ESTIMATOR_H
