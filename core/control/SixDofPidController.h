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

namespace oi3 {

/// 六自由度控制指令
struct SixDofCommand {
    double thrust_body = 0.0; ///< 机体 z 轴推力（牛顿，向上为正）
    Tensor torque;            ///< 机体三轴力矩 {3}（N·m）
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

    // 姿态环：输出力矩
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

    [[nodiscard]] const char *name() const override { return "6DoF-PID"; }

    void reset() override;

    [[nodiscard]] const SixDofPidGains &gains() const { return _gains; }
    [[nodiscard]] double lastTiltDeg() const { return _last_tilt_deg; }

  private:
    SixDofConfig _cfg;
    SixDofPidGains _gains;
    double _last_tilt_deg = 0.0;
};

} // namespace oi3

#endif // OI3_SIX_DOF_PID_CONTROLLER_H
