/**
 * @file HalAbstraction.h
 * @brief 硬件抽象层（HAL）接口：让主循环与底层硬件解耦
 *
 * @par 为什么需要 HAL
 *
 * 飞控主循环的核心逻辑（读传感器 → 估计 → 决策 → 控制 → 写执行器）
 * 是平台无关的，但传感器与执行器的具体访问方式因平台而异：
 * - 仿真平台：从 SixDofSimulator 读状态、用 ImuModel 生成测量、调用 sim.step()
 * - 真机平台：从 SPI/I²C 总线读 MPU6000/ICM42688，通过 PWM 输出到电调
 * - 半实物仿真：传感器是真机、执行器是仿真，或反之
 *
 * 把这两侧用接口隔开，主循环就能以同一份代码跑在三种平台上，
 * 只需要换 HAL 实现即可。
 *
 * @par 接口粒度
 *
 * 故意不做「所有传感器统一接口」的过度抽象：
 * - IMU 是高频必备（1 kHz），位置测量是低频可选（100 Hz），
 *   两者的调用节奏、错误处理、超时恢复完全不同；
 * - 真机上 IMU 和 GPS 通常走不同总线、有不同驱动；
 * - 统一成 `readSensor(id)` 会让调用方失去类型安全，也让 HAL 实现
 *   内部充满 switch-case。
 *
 * 因此每种传感器/执行器各给一个细粒度接口，主循环按需组合。
 */

#ifndef OI3_HAL_ABSTRACTION_H
#define OI3_HAL_ABSTRACTION_H

#include "ImuModel.h"
#include "SixDofTypes.h"
#include "Tensor.h"

#include <array>

namespace oi3 {

/**
 * @class HalSensorReader
 * @brief 传感器读取接口
 *
 * 主循环每周期调用一次 readImu()；位置测量是低频的，
 * 用 hasPositionUpdate() 轮询，有新数据时再读。
 */
class HalSensorReader {
  public:
    virtual ~HalSensorReader() = default;

    /// 读取当前 IMU 采样（机体坐标系）
    [[nodiscard]] virtual ImuSample readImu() = 0;

    /// 本周期是否有新的位置测量可用
    [[nodiscard]] virtual bool hasPositionUpdate() = 0;

    /// 读取位置测量（NED，米）。仅在 hasPositionUpdate() 为 true 时调用。
    [[nodiscard]] virtual std::array<double, 3> readPosition() = 0;

    /// 当前时间（秒），单调递增
    [[nodiscard]] virtual double time() = 0;
};

/**
 * @class HalActuatorWriter
 * @brief 执行器写入接口
 */
class HalActuatorWriter {
  public:
    virtual ~HalActuatorWriter() = default;

    /// 输出控制指令。主循环每周期调用一次。
    virtual void writeCommand(const SixDofCommand &cmd) = 0;
};

/**
 * @class HalSetpointSource
 * @brief 目标点/制导指令来源
 *
 * 定点悬停时返回固定位置；轨迹跟踪时返回随时间变化的参考点；
 * 返航模式下返回起飞点。主循环不判断「现在该飞去哪」，只问
 * 「当前目标点是哪」—— 任务切换由实现方负责。
 */
class HalSetpointSource {
  public:
    virtual ~HalSetpointSource() = default;

    /// 当前期望位置（NED，米）
    [[nodiscard]] virtual Tensor currentTarget(double time) = 0;

    /// 是否已到达当前目标（容差内）
    [[nodiscard]] virtual bool hasArrived(const std::array<double, 3> &pos,
                                          double tolerance) const = 0;
};

} // namespace oi3

#endif // OI3_HAL_ABSTRACTION_H
