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
#include "SensorHealth.h"
#include "SixDofTypes.h"

#include <array>

namespace oi3 {

/**
 * @brief 位置滤波器类型
 *
 * `AlphaBeta` 为固定增益 α-β 滤波器（历史实现）；
 * `Kalman` 为位置-速度卡尔曼滤波器，增益随过程/量测噪声自动缩放（**默认**）。
 *
 * @par 默认值已于 2026-10-02 由 AlphaBeta 切换为 Kalman
 *
 * 依据是 `FilterShowdownTest` 的全面对照（悬停稳态位置误差 RMS，对真值）：
 *
 * | 场景 | α-β | α-β+死区 | 卡尔曼 |
 * |---|---|---|---|
 * | 理想量测 | 0.0229 m | 0.2884 m | 0.0234 m |
 * | 光流级 σ=0.05 | 0.0725 m | 1.2140 m | **0.0509 m** |
 * | 中档 σ=0.15 | 0.2904 m | 0.7080 m | **0.1259 m** |
 * | GPS 级 σ=0.30 | 0.4677 m | 0.6141 m | **0.3072 m** |
 * | 延迟 100 ms | 0.1016 m | 1.6848 m | **0.0613 m** |
 * | 机动 + GPS 级 | 1.8098 m | 2.0425 m | **1.0374 m** |
 *
 * 三条理由：
 *
 * 1. **卡尔曼在噪声、延迟、机动三个维度全面占优**，且优势随工况恶化而扩大
 *    （光流级 1.43 倍 → 机动+大噪声 1.74 倍）。这正是「增益自适应」的价值 ——
 *    越恶劣的工况越依赖它。
 * 2. **唯一持平的是理想量测场景**（0.0229 vs 0.0234，差异 <3%，由控制滞后主导）。
 *    而真机不会有理想量测，该场景不构成保留旧默认值的理由。
 * 3. **死区补丁方案三者最差**（小噪声下劣化 17 倍），已被排除。
 *
 * @par 兼容性说明
 *
 * 切换会改变依赖位置估计的闭环结果。实测影响面：仅 `EstimatorTest` 需要
 * 显式绑定 AlphaBeta（该测试验证的正是 α 参数特性）；其余测试在清洁构建后
 * 全部通过。若需复现旧行为，显式设 `pos_filter = PosFilterKind::AlphaBeta` 即可。
 */
enum class PosFilterKind {
    AlphaBeta = 0, ///< 固定增益 α-β（历史实现，保留以支持对照与回退）
    Kalman,        ///< 位置-速度卡尔曼（**默认**，增益自适应）
};

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

    /**
     * @brief IMU 传感器健康监测配置
     *
     * 默认**关闭**。原因与本项目所有新效应一致：接入不得改变既有结果。
     * 关闭时 updateImu 完全不触碰健康监测，既有 29 个测试逐位不变。
     * 启用后每步多算一次残差模长（三次乘法与一次开方），代价可忽略。
     */
    SensorHealthConfig sensor_health{};

    /**
     * @brief 位置残差死区（米），仅作用于速度校正
     *
     * 大噪声位置量测（如廉价 GPS）会把脉冲灌入速度估计：速度校正增益
     * β/dt = α²/((2−α)·dt) 在 α=0.35、dt=0.01 s 时约为 7.4 /s，0.3 m 噪声
     * 直接产生 ~2.2 m/s 的速度脉冲，控制器 kd=3 再放大，闭环发散。
     *
     * 死区只裁剪进入速度校正的残差，位置校正仍用完整残差，因此：
     * - 小残差（在死区内）不触发速度修正，抑制纯噪声带来的速度抖动；
     * - 大残差去掉死区后仍驱动速度，保证机动时速度估计能跟上；
     * - 位置估计不受影响，仍能利用完整量测做快速位置修正。
     *
     * 默认 0.0 表示关闭，不改变既有行为。光流级量测（σ≈0.05 m）通常不需要
     * 死区；GPS 级量测（σ≈0.30 m）可取 0.1~0.2 m。
     */
    double pos_residual_deadzone = 0.0;

    /**
     * @brief 位置滤波器类型
     *
     * 默认 `AlphaBeta` 保持既有行为逐位不变；`Kalman` 为可选升级路径。
     */
    PosFilterKind pos_filter = PosFilterKind::Kalman;

    /**
     * @brief 卡尔曼滤波的过程噪声：IMU 加速度不确定度（m/s²）
     *
     * 物理含义是「预积分所用加速度的可信度」。取值偏大则滤波器更信任量测
     * （增益升高、跟踪更快、抗噪变差），偏小则更信任模型（平滑但滞后）。
     *
     * @par 默认值 100 的来源（由对照实验标定，非理论推导）
     *
     * 该值偏离加速度计本身的噪声量级（约 0.08 m/s²）达三个数量级，原因是
     * **预积分误差并非白噪声**：姿态误差经重力投影产生虚假加速度、陀螺偏置
     * 积分、控制推力响应滞后等，都表现为长相关误差，其增长速度远快于
     * 「白噪声加速度」模型的假设。标准解法（加速度偏置随机游走模型）会引入
     * 额外状态与调参维度，故这里用增大的过程噪声补偿模型失配。
     *
     * 标定实验（`PositionNonIdealClosedLoopTest` 的滤波器对照章节，扫描 q 从
     * 0.1 到 500，覆盖光流级 σ=0.05 与 GPS 级 σ=0.30 两种场景）：
     *
     * | q | 光流级高度偏差 | GPS 级高度偏差 |
     * |---|---|---|
     * | 0.5 | 48.97 m（发散） | — |
     * | 10 | 0.04 m | 46.78 m（发散） |
     * | 50 | 0.01 m | 11.09 m |
     * | **100** | **0.03 m** | **0.25 m** |
     * | 200 | 0.11 m | 0.35 m |
     * | 500 | 0.53 m | 2.89 m |
     *
     * q 过小时滤波器过度信任预积分，速度估计失去量测校正而漂移，闭环发散；
     * q 过大则等同不信任模型，退化为直接使用含噪量测。100 是两种噪声水平下
     * 同时可用的区间，其等效位置增益（0.299）与经过验证的 α-β 增益（0.35）
     * 同量级，即「带宽匹配既有设计，同时获得噪声自适应性」。
     */
    double kalman_accel_noise = 100.0;

    /**
     * @brief 卡尔曼滤波的量测噪声：位置传感器标准差（米）
     *
     * 这一项是卡尔曼相对 α-β 的核心优势所在 —— 它使增益随量测噪声自动缩放。
     * 光流级（σ≈0.05 m）与 GPS 级（σ≈0.30 m）只需改这一个数，无需为不同
     * 传感器重新标定增益，也不需要死区补丁。
     */
    double kalman_pos_noise = 0.05;
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
    /**
     * @param imu_read_ok HAL 是否成功读到本次 IMU 数据（默认 true）。
     *        为 false 时数据不可信，健康检测走外部失败路径
     *        （见 SensorHealthConfig::external_fail_steps）。
     */
    void updateImu(const ImuSample &imu, double dt, bool imu_read_ok = true);

    /**
     * @brief 位置测量更新（低频，例如 100 Hz）
     * @param pos_meas 位置测量（NED，米）
     * @param dt       距上次位置更新的时间（秒）
     */
    void updatePosition(const std::array<double, 3> &pos_meas, double dt);

    /// 用给定四元数直接设定姿态（仅用于初始化对齐，不用于运行时）
    void setAttitude(const std::array<double, 4> &quat);

    /**
     * @brief 运行时开关：是否使用加速度计做姿态方向校正
     *
     * 供降级策略使用。加速度计失效后**必须关闭**：失效数据会让方向校正把
     * 错误的姿态误差持续注入姿态估计，比不校正更糟。
     *
     * @note 关闭后姿态仅靠陀螺积分维持 —— 短期可用，但会随陀螺偏置缓慢漂移，
     *       因此这是一个「争取时间」而非「长期可用」的状态。
     */
    void setAccelCorrectionEnabled(bool on) { _use_accel_correction = on; }

    /// 当前是否启用加速度计方向校正
    [[nodiscard]] bool accelCorrectionEnabled() const { return _use_accel_correction; }

    /**
     * @brief 运行时开关：是否使用加速度计做位置/速度预积分
     *
     * 供降级策略使用。加速度计失效后预积分不可信，须关闭；
     * 此时位置只能靠外部位置测量（GPS/光流/视觉）校正。
     */
    void setTrustPosition(bool on) { _trust_position = on; }

    /// 当前是否信任位置预积分
    [[nodiscard]] bool trustPosition() const { return _trust_position; }

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

    /**
     * @brief IMU 健康报告（仅在 sensor_health.enabled 时有意义）
     *
     * 未启用时返回默认构造的报告（各项为 Unknown/None），调用方应先检查
     * report().samples > 0 再据其决策，避免把「未监测」误读为「健康」。
     */
    [[nodiscard]] const SensorHealthReport &sensorHealth() const { return _health.report(); }

    /**
     * @brief 当前卡尔曼增益（位置分量），仅 pos_filter == Kalman 时有意义
     *
     * 暴露该值用于验证「增益确实随噪声参数缩放」这一核心性质 —— 否则无法
     * 从外部区分「卡尔曼生效」与「只是碰巧数值接近」。
     */
    [[nodiscard]] const std::array<double, 3> &kalmanGainPos() const { return _k_pos; }

    /// 当前卡尔曼增益（速度分量）
    [[nodiscard]] const std::array<double, 3> &kalmanGainVel() const { return _k_vel; }

private:
    EstimatorConfig _cfg;

    std::array<double, 4> _quat{1.0, 0.0, 0.0, 0.0};
    std::array<double, 3> _omega{};
    std::array<double, 3> _gyro_bias{};
    std::array<double, 3> _pos{};
    std::array<double, 3> _vel{};
    bool _has_pos = false;

    /// 传感器健康监测（默认关闭，见 EstimatorConfig::sensor_health）
    SensorHealth _health;

    /// 运行时降级开关（默认为真，保证不改变既有行为）
    bool _use_accel_correction = true;
    bool _trust_position = true;

    /**
     * @brief 每轴 2×2 协方差（对称，仅存三个独立分量）
     *
     * 状态为 [p, v]，故 P = [[p_pp, p_pv], [p_pv, p_vv]]。
     * 三轴解耦处理：位置量测按轴独立，IMU 加速度虽经姿态旋转产生轴间耦合，
     * 但耦合项远小于各轴自身不确定度，解耦可显著降低复杂度而精度损失可忽略。
     */
    struct AxisCovariance {
        double p_pp = 0.0; ///< 位置方差
        double p_pv = 0.0; ///< 位置-速度协方差
        double p_vv = 0.0; ///< 速度方差
    };
    std::array<AxisCovariance, 3> _pos_cov{};

    /// 最近一次更新得到的卡尔曼增益（供测试验证增益随噪声缩放）
    std::array<double, 3> _k_pos{};
    std::array<double, 3> _k_vel{};
};

} // namespace oi3

#endif // OI3_STATE_ESTIMATOR_H
