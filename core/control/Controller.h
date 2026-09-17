/**
 * @file Controller.h
 * @brief OpenInspire3 飞行控制器抽象接口
 * @author GhostFace
 * @date 2026/9/16
 *
 * 控制器把「当前状态 + 期望位置」映射为合推力，是闭环仿真中的可替换部件。
 * PID 与神经网络等实现共享同一接口，因而可以在同一套任务与评价指标下
 * 直接对比；同时，用经典控制器先行跑通闭环，可以先把被控对象与接口本身
 * 验证正确，再排除控制器一侧的变量。
 */

#ifndef OI3_CONTROLLER_H
#define OI3_CONTROLLER_H

#include "DroneTypes.h"
#include "Tensor.h"

namespace oi3 {

/**
 * @class Controller
 * @brief 飞行控制器接口
 *
 * 约定：
 *  - 输入状态为 NED 系；期望位置同为 NED 系（相对仿真原点）
 *  - 输出为该步的合推力 {3}，单位牛顿，NED 系，可直接传入
 *    `DroneSimulator::step`，中间不做任何坐标或单位变换
 *  - 输出形状必须为 {3}
 */
class Controller {
  public:
    virtual ~Controller() = default;

    /**
     * @param state  当前飞行器状态
     * @param target 期望位置 {3}（NED）
     * @param time   当前仿真时间（秒），供时变参考轨迹的实现使用
     * @return 合推力 {3}（NED，牛顿）
     */
    [[nodiscard]] virtual Tensor computeThrust(const DroneState &state,
                                               const Tensor &target,
                                               double time) = 0;

    /// 实现名，用于日志与结果对比
    [[nodiscard]] virtual const char *name() const = 0;

    /// 开始新回合前的复位（清积分项、内部历史等）
    virtual void reset() {}
};

} // namespace oi3

#endif // OI3_CONTROLLER_H
