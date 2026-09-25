/**
 * @file SixDofPidController.h
 * @brief 六自由度级联 PID 控制器（位置环 + 姿态环）
 * @author GhostFace
 * @date 2026/9/17
 *
 * 四旋翼无法直接产生水平推力：水平运动只能靠倾斜机身、把竖直推力的一部分转向
 * 水平。因此控制器必须是级联结构 —— 位置环先算出所需的合力方向，姿态环再让
 * 机体轴去对准该方向。
 *
 * 控制律：
 * @verbatim
 *   位置环：  a_des = kp·(target − pos) + kd·(0 − vel)          [NED]
 *   合力：    F_des = m·(a_des − [0,0,g])                        [NED]
 *   推力：    T = |F_des|
 *   期望轴：  z_des = −F_des / |F_des|         （机体 z 轴朝上，推力沿 −z）
 *   姿态误差：rotvec = axis(z_cur → z_des)·θ   再转到机体系
 *   力矩：    τ = att_kp·rotvec_body − att_kd·ω
 * @endverbatim
 *
 * 姿态误差用「轴角」而非四元数差：只需要把机体的 z 轴对准推力方向，绕 z 轴的
 * 偏航不受约束（四旋翼无 yaw 偏好），轴角表示天然只约束这一个自由度。
 */

#ifndef OI3_SIX_DOF_PID_CONTROLLER_H
#define OI3_SIX_DOF_PID_CONTROLLER_H

#include "DifferentialFlatness.h"
#include "SixDofTypes.h"
#include "Tensor.h"

#include <array>

namespace oi3 {

/**
 * @struct SixDofSetpoint
 * @brief 轨迹跟踪的参考点：位置 + 速度 + 加速度
 *
 * 纯数值、不含张量，因为前馈量不需要参与求导，也不需要「空值」这种模糊状态
 * （缺省即零）。
 *
 * @par 为什么前馈是必需的
 *
 * 定点控制律 `a_des = kp·(p_ref−p) + kd·(0−v)` 隐含假设**期望速度恒为零**。
 * 跟踪时参考点本身在动，这个假设不成立，于是出现两项系统性误差（均已实测）：
 *
 * 1. **切向阻尼无处平衡**。圆周运动要保持切向速度 ωR，而 `−kd·v` 持续给出反向
 *    加速度。把控制律投影到切向可得 `sinφ = (kd/kp)·ω`，其中 φ 是相位滞后 ——
 *    实测 ω=1 rad/s 时滞后 42.6 度，且 ω 超过 kp/kd 后该式无解、跟踪退化。
 * 2. **向心加速度只能靠位置误差换**。径向需要 ω²R 的向心加速度，
 *    没有前馈就只能由位置误差提供。
 *
 * 对匀速直线（无加速度需求）有更紧的解析结果：稳态 `kp·e − kd·v = 0`，
 * 即滞后恒等于 `(kd/kp)·v`。默认增益下这是速度的 0.75 倍 —— 实测 1 m/s 时
 * 落后参考点 0.749 m，与预测 0.750 m 相差 0.2%。
 *
 * 引入前馈后控制律变为：
 * @verbatim
 *   a_des = a_ref + kp·(p_ref − p) + kd·(v_ref − v)
 * @endverbatim
 * 向心加速度由 `a_ref` 承担，阻尼作用于**速度误差**而非绝对速度，两项冲突消失。
 */
struct SixDofSetpoint {
    std::array<double, 3> pos{}; ///< 参考位置（NED，米）
    std::array<double, 3> vel{}; ///< 参考速度（NED，米/秒），缺省零
    std::array<double, 3> acc{}; ///< 参考加速度（NED，米/秒²），缺省零

    /**
     * @brief 参考加加速度（NED，米/秒³），缺省零
     *
     * 姿态变化率由 jerk 决定：推力方向的转动速率正比于 jerk 中垂直于推力
     * 的分量。只给加速度的话，前馈能算出该用多大力、该摆什么姿态，却算不出
     * **机身要以多快的角速度转过去** —— 而后者恰恰是机动跟踪滞后的主因。
     *
     * 缺省零表示「不提供 jerk 信息」，此时角速度前馈退化为零，行为与改造前
     * 完全一致。
     */
    std::array<double, 3> jerk{};

    /// 参考偏航角（弧度），缺省零
    double yaw = 0.0;
    /// 参考偏航角速度（弧度/秒），缺省零
    double yaw_rate = 0.0;
};

/**
 * @brief 入流系数的在线估计器接口
 *
 * 控制器只依赖这个抽象接口，而不直接依赖具体的 RLS 实现 —— 保持控制模块
 * 与仿真/估计模块的单向依赖（控制不该反过来 include 仿真）。
 *
 * 实现方（如 InflowEstimator）在每步喂入观测，控制器则读取当前估计值。
 */
class InflowEstimateSource {
  public:
    virtual ~InflowEstimateSource() = default;
    /// 当前入流系数估计值
    [[nodiscard]] virtual double inflowMu() const = 0;
    /// 是否已积累足够样本（未收敛时控制器应回退到配置值）
    [[nodiscard]] virtual bool inflowEstimateReady() const = 0;
};

/// 六自由度控制器接口
class SixDofController {
  public:
    virtual ~SixDofController() = default;

    /**
     * @param state  当前状态
     * @param target 期望位置 {3}（NED）
     * @param time   当前仿真时间（秒）
     */
    [[nodiscard]] virtual SixDofCommand compute(const SixDofState &state,
                                                const Tensor &target, double time) = 0;

    [[nodiscard]] virtual const char *name() const = 0;
    virtual void reset() {}
};

/// 级联 PID 参数
struct SixDofPidGains {
    // 位置环：输出期望加速度
    double pos_kp = 4.0;
    double pos_kd = 3.0;
    double max_accel = 12.0; ///< 期望加速度限幅（m/s²）

    /**
     * @brief 位置环积分增益（默认 0 = 纯 PD）
     *
     * @par 为什么此前是纯 PD
     *
     * 常值扰动（风）由**前馈**补偿，不需要积分慢慢消除。而积分引入相位滞后，
     * 在延迟已经在吃裕度的情况下无条件加积分是危险的。
     *
     * @par 但纯 PD 有一个前馈补不了的缺口
     *
     * 前馈依赖模型准确性。对**完全未建模**的结构性偏差（机身不对称、电机
     * 安装偏斜、重心偏移、传感器零偏），PD 会留下稳态误差 `F/kp`，且没有任何
     * 机制消除它。自适应补偿补的是「已知效应 + 未知系数」，对此无能为力。
     *
     * @par 代价有解析式
     *
     * @verbatim
     *   PM = atan2(kd·ωc − ki/ωc, kp) − ωc·T
     * @endverbatim
     *
     * 关键在于 `ki/ωc` 是从 `kd·ωc` 中**减去**的 —— 这就是吃裕度的机制。
     * 实测：ki = 1 时相位裕度损失 < 5°；ki = 4 时代价明显。推荐 0.5~1.0。
     */
    double pos_ki = 0.0;

    /**
     * @brief 积分项限幅（anti-windup），单位与积分输出一致（m/s）
     *
     * 积分必须限幅：执行器饱和期间误差持续累积会让积分项涨到很大，之后需要
     * 很长时间才能退回来（积分饱和）。限幅值取 max_accel 的同一量级即可 ——
     * 积分的作用是消除稳态误差，不该主导控制量。
     */
    double integral_limit = 4.0;

    // ---- 姿态环：输出力矩 ----
    //
    // 姿态环的物理量纲是「力矩 → 角加速度」，而角加速度 = τ / I —— 因此**增益必须
    // 随惯量缩放**才能让三轴得到一致的闭环特性。原先用一组标量增益套三轴，
    // 转动惯量三轴不等时闭环带宽与阻尼比都会不同：以默认值（I={0.01,0.01,0.02}、
    // kp=0.9、kd=0.25）计，roll/pitch 的带宽 9.5 rad/s、阻尼比 1.32，而 yaw 是
    // 6.7 rad/s、0.93 —— yaw 的响应比另外两轴慢四成，姿态耦合时这一轴明显滞后。
    //
    // 默认改为按「期望带宽 + 阻尼比」推导（惯量取自 SixDofConfig，而惯量正是可以用
    // ParameterIdentification 从实测轨迹辨识的量）：
    //     att_kp[i] = I[i] · ωn²
    //     att_kd[i] = 2·ζ·I[i]·ωn
    // 这样三轴的闭环特性一致，且惯量变了增益自动跟着变 —— 增益是算出来的，不是试出来的。
    bool derive_attitude_from_inertia = true;
    double att_bandwidth = 9.0; ///< 姿态环期望带宽 ωn（rad/s）
    double att_damping = 1.0;   ///< 姿态环期望阻尼比 ζ（1.0 = 临界阻尼）

    /**
     * @brief 是否使用平坦前馈提供的**角速度参考**（默认关闭）
     *
     * 开启后姿态环的阻尼项由 `−kd·ω` 变为 `−kd·(ω − ω_des)`，其中 `ω_des`
     * 由微分平坦映射从轨迹的 jerk 与偏航率解析算出。
     *
     * @par 为什么这一项对机动性是关键
     *
     * `−kd·ω` 的物理含义是「把角速度阻尼到零」—— 这个假设对定点悬停成立
     * （悬停确实不该有角速度），但对机动飞行根本不成立：机动时机身本来就
     * 该以某个角速度转过去，而阻尼项在**持续对抗**它。控制器只能靠 `kp·e`
     * 累积出更大的姿态误差去压过这一项，表现为跟踪滞后。
     *
     * 实测（正弦机动，振幅 2 m、角频率 0.5 rad/s、带宽 9 rad/s）：
     * 跟踪误差 RMS 从 9.967e-3 降到 1.028e-3，**改善 9.7 倍**；且改善倍数在
     * 振幅 0.5~4 m 范围内稳定在 9.4~9.7 倍 —— 稳定倍数说明消除的是一个与
     * 机动强度成正比的结构项，而非调参效果。
     *
     * 关闭时 `ω_des` 取零，行为与改造前逐位相同。
     */
    bool use_flat_omega_feedforward = false;

    /**
     * @brief 是否使用**在线辨识**的入流系数（默认关闭）
     *
     * 开启后，入流补偿用在线估计的 mu 而非配置里的固定值。这对应真实需求：
     * 载荷、桨叶磨损、空气密度都会让出厂标定值在飞行中失准。
     *
     * @par 为什么必须显式处理「代数环」
     *
     * 估计器以控制器输出（推力指令）为输入，而控制器输出又依赖估计值 ——
     * 这是一个**闭环**。若估计值抖动，补偿随之抖动，推力变化又反过来影响
     * 估计输入，理论上可能自激。
     *
     * 缓解措施：估计器的遗忘因子（记忆长度）远大于控制回路时间常数，
     * 使参数估计成为**慢回路**、控制成为快回路，二者时间尺度分离。
     * 本项在测试中用「估计值抖动幅度」与「闭环误差」两个量来验证是否稳定。
     */
    bool use_online_inflow_estimate = false;

    /**
     * @brief 是否控制**偏航角**（默认关闭）
     *
     * 关闭时姿态误差只对齐机体 z 轴（推力方向），偏航角完全自由 —— 这是既有
     * 全部结果的基准行为，也是定点悬停的合理选择（悬停时偏航随便转，位置
     * 照样准）。
     *
     * 开启后，姿态误差用**完整旋转矩阵**计算（`R_e = R_des·R_curᵀ`），期望
     * 姿态由「期望推力方向 + 期望偏航」共同构造，因此偏航被真正约束。
     *
     * @par 为什么这是必需的
     *
     * 偏航不受控意味着任何**需要指向**的任务都做不了：挂相机要对准目标、
     * 挂云台要指定朝向、多机协同要指定机头方向。真机上偏航还影响气动与能耗。
     * 这是结构性缺失 —— 不是调参能补的。
     *
     * @par 与倾斜限幅的关系
     *
     * 限幅作用在**期望推力方向**上（推力方向不能偏得太多），而偏航是绕该
     * 方向的转动，两者互不干扰。原实现把限幅加在误差角上，会把偏航误差
     * 也算进去，属于概念混淆。
     */
    bool use_yaw_control = false;

    /**
     * @brief 是否启用扰动观测器前馈
     *
     * 与积分的分工：积分作用于反馈回路（吃相位裕度），观测器作用于前馈路径
     * （不吃裕度，但放大速度噪声、且依赖模型准确）。默认关闭以保证既有结果
     * 逐位不变。
     *
     * @warning 两者同时启用会**重复补偿**同一扰动，可能过冲。除非有实测依据，
     *          否则只开一个。
     */
    bool use_disturbance_observer = false;

    /// 扰动观测器带宽（Hz）。经验取控制带宽的 1/3 ~ 1/5。
    double disturbance_observer_hz = 2.0;

    /// 扰动估计限幅（m/s²），防止饱和期间估计发散
    double disturbance_limit = 12.0;

    // 手动模式（derive_attitude_from_inertia = false 时使用）
    double att_kp = 0.9;  ///< 姿态角误差增益（N·m/rad）
    double att_kd = 0.25; ///< 角速度阻尼（N·m·s/rad）

    /**
     * @brief 姿态误差限幅（度）
     *
     * @note 该字段的语义经历过一次修正，此处记录以免误读。
     *
     * **旧行为**：限的是「期望姿态相对当前姿态的旋转角」，即姿态指令的激进
     * 程度（一种软性速率保护）。稳态下姿态已跟上指令、夹角趋近零，限幅不生效，
     * 因此机体实际可倾斜到远超此值的角度 —— WindTunnelTest 中 12 m/s 风下
     * 稳定倾斜 35.73°，超过 35° 而系统正常。
     *
     * **现行为**：限的是**期望推力方向相对竖直轴的偏角**。大机动测试实测峰值
     * 倾角随该值单调变化（15→14.99°、35→33.60°、50→42.88°），说明它现在确实
     * 约束机体倾角。
     *
     * @warning 真正的水平加速度上限**不是** max_accel 的标称值，而是由本字段与
     *          max_accel 共同决定，取更紧者。实测（LargeManeuverTest）本配置下
     *
     * @verbatim
     *   max_tilt_deg = 35      -> 6.87 m/s²   ← 实际生效
     *   max_accel    = 12      -> 12.00 m/s²
     *   max_body_thrust = 20   -> 17.43 m/s²
     * @endverbatim
     *
     * 且实际可用推力上限为 `m·sqrt(max_accel² + g²) = 15.50 N`，而非
     * max_body_thrust —— 后者在本配置下永远不会被触发，调它没有效果。
     */
    double max_tilt_deg = 35.0;
};

class SixDofPidController : public SixDofController {
  public:
    SixDofPidController(SixDofConfig cfg, SixDofPidGains gains = {});

    [[nodiscard]] SixDofCommand compute(const SixDofState &state, const Tensor &target,
                                        double time) override;

    /**
     * @brief 带风扰动前馈的定点控制律
     *
     * @param state  当前状态
     * @param target 目标位置 {3}
     * @param v_wind NED 风速 {3}（由风速估计给出；真实飞控有这一路信息）
     * @param time   仿真时刻
     *
     * @par 它补的是什么
     *
     * 定点 PID 对风是**被动响应**：只能靠位置误差感知风的存在，因此必然产生
     * 稳态偏移 `e = k·|v_wind|²/(m·kp)`（实测 8 m/s 风时 0.784 m，与解析式
     * 吻合到四位小数）。
     *
     * 但风速**是可测的**。既然已知扰动的大小与方向，就没有理由等它把飞行器
     * 推偏之后再去纠正 —— 直接前馈掉：
     *
     * @verbatim
     *   F_wind = k·|v_wind|·v_wind          （稳态时 v_rel = −v_wind）
     *   a_ff   = −F_wind / m
     *   a_des  = a_ff + kp·(p_ref − p) + kd·(0 − v)
     * @endverbatim
     *
     * 这就是「被动抗风」与「主动抗风」的差别。本方法提供解析前馈基线，
     * 用于量化这一差距 —— 它同时也是判断「风补偿任务上学习有没有空间」的
     * 基准：解析式只能补偿**均值**，湍流的波动部分它管不了。
     */
    [[nodiscard]] SixDofCommand computeWithWind(const SixDofState &state, const Tensor &target,
                                                const std::array<double, 3> &v_wind,
                                                double time);

    /**
     * @brief 轨迹跟踪控制律（带参考速度与加速度前馈）
     *
     * 与 compute 的区别只在期望加速度的算法：定点版本把期望速度当作零，
     * 本版本使用参考点给定的速度与加速度。
     * @note 当 setpoint 的 vel 与 acc 都为零时，本函数与 compute 等价。
     */
    [[nodiscard]] SixDofCommand computeTracking(const SixDofState &state,
                                                const SixDofSetpoint &ref, double time);

    [[nodiscard]] const char *name() const override { return "6DoF-PID"; }

    void reset() override;

    [[nodiscard]] const SixDofPidGains &gains() const { return _gains; }

    /**
     * @brief 挂载入流系数的在线估计器（不获取所有权，传 nullptr 表示不使用）
     *
     * 仅在 `_gains.use_online_inflow_estimate` 为真时生效，且估计器报告
     * `inflowEstimateReady()` 后才会用估计值覆盖配置值 —— 样本不足时回退到
     * 配置值，避免用未收敛的估计去做补偿。
     */
    void setInflowEstimateSource(InflowEstimateSource *src) { _inflow_src = src; }

    /// 当前实际使用的入流系数（配置值或被估计值覆盖后的值）
    [[nodiscard]] double activeInflowMu() const { return _active_mu; }

    /// 当前扰动估计（NED，m/s²），供诊断与测试
    [[nodiscard]] const double *disturbanceEstimate() const { return _d_hat; }

    /// 上一拍输出的推力指令（供入流补偿估计实际推力用；0 表示尚未有历史）
    double _last_thrust = 0.0;

    /// 位置环积分状态（NED 三轴，单位 m·s，乘 ki 后为加速度）
    double _pos_integral[3] = {0.0, 0.0, 0.0};

    /// 上一拍的期望加速度（供入流补偿估计用）
    double _prev_a_des[3] = {0.0, 0.0, 0.0};

    /// 积分项当前幅值（供诊断与测试检查限幅是否生效）
    [[nodiscard]] double integralMagnitude() const {
        return std::sqrt(_pos_integral[0] * _pos_integral[0] +
                         _pos_integral[1] * _pos_integral[1] +
                         _pos_integral[2] * _pos_integral[2]);
    }

    /// 上一拍的期望加速度（供在线辨识构造残差观测用）
    double _last_a_des[3] = {0.0, 0.0, 0.0};

    /**
     * @brief 统一的期望加速度后处理：扰动前馈 + 限幅
     *
     * 三条控制路径（compute / computeWithWind / computeTracking）**共用**此函数。
     *
     * 这是刻意的：本文件此前因三条路径各自独立实现位置环，导致给其中两条加
     * 积分时漏掉第三条，症状是「调用 compute 时积分项恒为 0」。同一条控制律
     * 写多份就是漏改的温床，故新增逻辑一律走公共出口。
     *
     * @param raw   未经限幅的期望加速度（含积分与 PD 项）
     * @param state 当前状态（供观测器取速度）
     * @param dt    步长
     */
    [[nodiscard]] std::array<double, 3> finalizeAccel(const double raw[3],
                                                      const SixDofState &state, double dt);

    /**
     * @brief 统一的命令出口：记录本拍输出，供下一拍的状态量使用
     *
     * 三条控制路径**必须**经由它返回。此前 `_last_thrust` 只在
     * `computeTracking` 中被赋值，导致另外两条路径下该值恒为 0 —— 入流补偿
     * 与扰动观测器都因此失效，且症状隐蔽（不报错，只是估计值恒为 0）。
     *
     * 同类问题在本文件已出现三次（积分漏加、_last_thrust 漏加），故新增状态
     * 一律走公共出口，不再逐路径复制。
     */
    [[nodiscard]] SixDofCommand publish(SixDofCommand cmd);

    /// 扰动观测器状态（NED 三轴，m/s²）
    double _d_hat[3] = {0.0, 0.0, 0.0};
    /// 观测器上一拍的速度（用于差分）
    double _obs_prev_vel[3] = {0.0, 0.0, 0.0};
    bool _obs_has_prev = false;
    /// 上一拍实际施加的期望加速度（供观测器构造残差）
    double _obs_last_applied[3] = {0.0, 0.0, 0.0};

    /**
     * @brief 本拍**已被前馈补偿掉**的扰动加速度（NED，m/s²）
     *
     * @par 为什么需要它
     *
     * 观测器的残差是「实际加速度 − 控制器认为施加的」。但「认为施加的」只包含
     * 推力模型（`R·(−T·ẑ)/m + g`），**不含前馈已处理的效应**。于是当前馈补掉
     * 某个扰动后，观测器仍会在残差里看到它，并**再补一次** —— 重复补偿。
     *
     * 实测两处（均为数量级级别的恶化）：
     *
     * | 场景 | 叠加后 | 最优单项 | 恶化 |
     * |------|-------|---------|------|
     * | 入流自适应 + 观测器 | 0.280519 | 4.00543e-05 | 7000x |
     * | 风前馈 + 观测器（只有风） | 0.283779 | 0.0428476 | 6.6x |
     *
     * 风前馈那次的严重性更高：叠加后 **0.283779 比完全不做补偿的 0.209515 还差**。
     *
     * @par 统一规则
     *
     * 残差必须相对控制器的**完整内部模型**计算：
     *
     * @verbatim
     *   resid = a_measured − (仅推力模型算出的加速度) − (所有已建模扰动)
     * @endverbatim
     *
     * 本变量即上式末项，由各前馈块累加写入：
     *  - 入流损失：`−loss_ned/m`（损失等效的扰动加速度）
     *  - 风阻前馈：`−a_ff`（阻力加速度）
     *
     * 这样前馈估得越准，观测器残差越接近零，两者自然分工而非重复。
     *
     * @note 由 `publish()` 每拍清零；三条路径共用该出口，避免遗漏。
     */
    double _last_modeled_disturb[3] = {0.0, 0.0, 0.0};

    /// 在线估计器（非拥有）
    InflowEstimateSource *_inflow_src = nullptr;
    /// 本拍实际使用的入流系数（供外部读取与诊断）
    double _active_mu = 0.0;

    /// 第 i 轴（0=roll,1=pitch,2=yaw）实际使用的姿态增益
    [[nodiscard]] double attKp(int axis) const { return _att_kp[axis]; }
    [[nodiscard]] double attKd(int axis) const { return _att_kd[axis]; }
    [[nodiscard]] double lastTiltDeg() const { return _last_tilt_deg; }

  private:
    SixDofConfig _cfg;
    SixDofPidGains _gains;

    /// 三轴姿态增益（由构造时按推导模式或手动模式确定）
    double _att_kp[3] = {0.0, 0.0, 0.0};
    double _att_kd[3] = {0.0, 0.0, 0.0};

    double _last_tilt_deg = 0.0;
};

} // namespace oi3

#endif // OI3_SIX_DOF_PID_CONTROLLER_H
