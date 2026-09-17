/**
 * @file PidController.h
 * @brief 位置 PID 控制器（闭环基线）
 * @author GhostFace
 * @date 2026/9/16
 *
 * 作为闭环的第一个可运行控制器，它承担两个作用：
 *  1. 验证被控对象与接口：若 PID 都无法稳定，问题在动力学或符号约定，
 *     而不在后续接入的学习型控制器
 *  2. 作为对照基线：学习型控制器必须至少达到或超过它的控制品质才有意义
 */

#ifndef OI3_PID_CONTROLLER_H
#define OI3_PID_CONTROLLER_H

#include "Controller.h"
#include "DroneTypes.h"

namespace oi3 {

/**
 * @struct PidGains
 * @brief PID 增益与限幅参数
 *
 * @par 控制律（NED 系，向下为正）
 * @verbatim
 *   e_p = target - pos
 *   e_v = -vel
 *   I   = clamp(I + e_p*dt, ±integral_limit)
 *   a_des = clamp(kp*e_p + kd*e_v + ki*I, ±max_accel)
 *   thrust = mass * (a_des - [0, 0, g])
 * @endverbatim
 *
 * 最后一步是重力补偿：悬停时 a_des = 0，推力为 [0, 0, -mass*g]，即一个抵消
 * 重力的向上推力，与 DroneSimulator 约定的一致（见 main.cpp 的悬停场景）。
 *
 * @note ki 默认为 0。位置环在没有持续外扰时本就没有稳态误差，先不加积分项，
 *       避免引入积分饱和这一额外变量；需要抗常值风扰时再打开。
 */
struct PidGains {
    double kp = 4.0;          ///< 位置误差比例增益（1/s²）
    double kd = 3.0;          ///< 速度阻尼增益（1/s）
    double ki = 0.0;          ///< 位置误差积分增益（1/s³）
    double max_accel = 15.0;  ///< 期望加速度限幅（m/s²），等效于推力上限
    double integral_limit = 5.0; ///< 积分项限幅（抗饱和）
};

/**
 * @class PidController
 * @brief 三轴独立的位置 PID + 重力补偿
 *
 * 三轴解耦控制：质点模型的三个平动轴之间没有耦合项，因此每个轴各用一组
 * 相同的增益即可。姿态相关的耦合要等引入 6 自由度后才需要考虑。
 *
 * 数值计算全部用 double 在标量域完成，最后再构造推力张量。控制器是闭环的
 * 输出端，其内部不需要 autograd 计算图，这样既避开张量拷贝断图的问题，
 * 也便于直接用解析解核对控制律。
 */
class PidController : public Controller {
  public:
    PidController(Config cfg, PidGains gains = {});

    [[nodiscard]] Tensor computeThrust(const DroneState &state, const Tensor &target,
                                       double time) override;

    [[nodiscard]] const char *name() const override { return "PID"; }

    /// 清除积分累积
    void reset() override;

    [[nodiscard]] const PidGains &gains() const { return _gains; }

  private:
    Config _cfg;
    PidGains _gains;
    double _integral[3] = {0.0, 0.0, 0.0};
};

} // namespace oi3

#endif // OI3_PID_CONTROLLER_H
