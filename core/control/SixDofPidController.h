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

#include "SixDofTypes.h"
#include "Tensor.h"

#include <array>

namespace oi3 {

/// 六自由度控制指令
struct SixDofCommand {
    double thrust_body = 0.0; ///< 机体 z 轴推力（牛顿，向上为正）
    Tensor torque;            ///< 机体三轴力矩 {3}（N·m）
};

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

    // 手动模式（derive_attitude_from_inertia = false 时使用）
    double att_kp = 0.9;  ///< 姿态角误差增益（N·m/rad）
    double att_kd = 0.25; ///< 角速度阻尼（N·m·s/rad）

    /// 期望倾角上限（度）。倾角越大，竖直方向可用推力越小：cos(60°)=0.5 意味着
    /// 一半推力被用于水平机动，超过这个范围容易掉高度。
    double max_tilt_deg = 35.0;
};

class SixDofPidController : public SixDofController {
  public:
    SixDofPidController(SixDofConfig cfg, SixDofPidGains gains = {});

    [[nodiscard]] SixDofCommand compute(const SixDofState &state, const Tensor &target,
                                        double time) override;

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
