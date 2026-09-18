/**
 * @file SixDofManualController.h
 * @brief 手动模式（自稳）控制器：把遥控杆量映射为姿态指令
 * @author GhostFace
 * @date 2026/9/18
 *
 * @par 为什么需要它
 *
 * 定点控制器解决的是「自己飞到目标点并停住」，而真实飞行中大量场景是**人在飞**：
 * 打杆改变方向、做机动动作。这两者对控制器的要求不同 —— 定点控制器的目标是
 * 让位置误差归零，手动控制器则是**忠实地跟随杆量**，把人的输入变成姿态指令。
 *
 * 本模块模拟的是最常见的自稳（Stabilize）模式：
 *
 * @verbatim
 *   右杆（roll/pitch）  → 目标倾角（满舵 = max_tilt_deg）
 *   左杆水平（yaw）     → 目标偏航角速度（速率模式）
 *   左杆垂直（throttle）→ 相对悬停油门的推力偏移
 * @endverbatim
 *
 * @par 两个与定点控制不同的地方
 *
 * 1. **姿态误差必须包含偏航**。定点控制器只需把机体 z 轴对准推力方向，绕 z 轴的
 *    偏航不受约束（四旋翼没有 yaw 偏好），所以它用「轴角对齐 z 轴」就够了。
 *    手动模式下偏航是被直接操纵的自由度，必须用完整的四元数误差。
 *
 * 2. **倾斜时要补偿推力**。倾角 θ 时竖直分量是 `T·cosθ`，若不补偿，打杆越猛掉高
 *    越快 —— 这是新手最容易察觉的现象。补偿后 `T = mg/cosθ`，竖直分量恒为 mg，
 *    于是「打杆只改变方向、不改变高度」。
 */

#ifndef OI3_SIX_DOF_MANUAL_CONTROLLER_H
#define OI3_SIX_DOF_MANUAL_CONTROLLER_H

#include "SixDofTypes.h"
#include "Tensor.h"

#include <array>

namespace oi3 {

/**
 * @struct RcStick
 * @brief 遥控杆量，各通道归一化到 [−1, 1]
 *
 * 符号约定与真实遥控器一致（从飞手视角）：
 */
struct RcStick {
    double roll = 0.0;     ///< 右杆左右：正 = 向右滚转（右倾）
    double pitch = 0.0;    ///< 右杆前后：正 = 向前飞（机头下压）
    double yaw = 0.0;      ///< 左杆左右：正 = 向右偏航（俯视顺时针）
    double throttle = 0.0; ///< 左杆上下：正 = 相对悬停油门上升

    /// 杆量是否在中位
    [[nodiscard]] bool centered(double tol = 1e-6) const {
        return roll * roll + pitch * pitch + yaw * yaw + throttle * throttle < tol * tol;
    }
};

/// 手动模式配置
struct ManualConfig {
    /// 满舵对应的目标倾角（度）
    double max_tilt_deg = 35.0;

    /// 满舵对应的偏航角速度（rad/s）
    double yaw_rate_max = 2.0;

    /// 油门杆满量程对应的推力变化（牛顿），以悬停推力为中性点
    double throttle_range = 4.0;

    /// 姿态环期望带宽（rad/s）与阻尼比，增益由惯量推导
    double att_bandwidth = 9.0;
    double att_damping = 1.0;

    /// 倾斜时补偿推力使竖直分量恒为 mg（关掉可对比「打杆掉高」现象）
    bool thrust_compensation = true;

    /// 倾角保护：限制杆量映射出的目标倾角，防止姿态环指令过于激进
    bool enable_tilt_limit = true;
};

/**
 * @class SixDofManualController
 * @brief 把杆量翻译成推力与力矩
 *
 * @note 它不实现 `SixDofController` 接口 —— 那个接口的语义是「给定目标位置」，
 *       而手动模式的目标是姿态，两者不是同一类问题。强行套用会让两者都别扭。
 */
class SixDofManualController {
  public:
    SixDofManualController(SixDofConfig cfg, ManualConfig mc = {});

    /**
     * @brief 由杆量算出控制指令
     * @param state 当前状态
     * @param stick 杆量
     */
    [[nodiscard]] SixDofCommand compute(const SixDofState &state, const RcStick &stick);

    [[nodiscard]] const char *name() const { return "6DoF-Manual"; }

    void reset();

    /// 上一次算出的目标倾角（度），用于日志与测试
    [[nodiscard]] double targetTiltDeg() const { return _target_tilt_deg; }

    /// 第 i 轴（0=roll,1=pitch,2=yaw）实际使用的姿态增益
    [[nodiscard]] double attKp(int axis) const { return _att_kp[axis]; }
    [[nodiscard]] double attKd(int axis) const { return _att_kd[axis]; }

    [[nodiscard]] const ManualConfig &config() const { return _mc; }

  private:
    SixDofConfig _cfg;
    ManualConfig _mc;

    double _att_kp[3] = {0.0, 0.0, 0.0};
    double _att_kd[3] = {0.0, 0.0, 0.0};

    double _target_tilt_deg = 0.0;
};

} // namespace oi3

#endif // OI3_SIX_DOF_MANUAL_CONTROLLER_H
