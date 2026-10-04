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

#include "ImuDegradePolicy.h"
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

    /**
     * @brief 最近一次 readImu() 是否成功
     *
     * @par 为什么需要
     *
     * 本接口原先没有失败语义：`readImu()` 返回 `ImuSample` 值类型，HAL 即便
     * 遇到 I2C 超时/总线错误也无法上报，只能返回某个值。而估计器的健康检测
     * 全部基于**数据特征**，对「数据看起来合理但完全错误」无能为力。
     *
     * 实测（持续返回随机值、模长接近正常）：姿态误差由 1.45° 涨到 180°，
     * 健康状态却**全程保持 Healthy**，最终坠机 —— 检测并非不及时，而是根本
     * 判不出来（模长正常、每帧在变、残差因姿态被污染而自洽）。
     *
     * 而 HAL 本身**知道**读取失败。这类信息应当直接传递，不该让估计器从数据
     * 里去猜。
     *
     * @par 约定
     *
     * 默认实现返回 true，故既有 HAL（如 `SimSensorReader`）无需改动。
     * 返回 false 时主循环会把该帧标记为不可信，健康检测按连续失败计数处理
     * （阈值见 `SensorHealthConfig::external_fail_steps`）。
     */
    [[nodiscard]] virtual bool lastImuValid() const { return true; }

    /**
     * @brief 最近一次 readImu() 对应的时间戳（秒）
     *
     * 主循环用它**差分得到真实的采样间隔 dt**，而不是依赖硬编码步长。
     *
     * @par 为什么这个接口是必要的
     *
     * 主循环原先把 `dt` 硬编码为 0.001（1 kHz），而 `ImuSample` 不含时间戳，
     * HAL 也就无法提供真实时间信息。后果是系统**只能在 1 kHz 附近工作** ——
     * 实测（以仿真步长模拟真机频率，主循环仍假定 1 ms）：
     *
     * | 真机频率 | 末态高度 | 姿态误差峰值 |
     * |---|---|---|
     * | 2 kHz | 5.00 m | 0.88° |
     * | 1 kHz | 5.00 m | 1.27° |
     * | **500 Hz** | **22.91 m** | **76.15°** |
     * | 250 Hz | 坠地 | 93.00° |
     * | 125 Hz | 坠地 | 131.08° |
     *
     * 500 Hz 在低成本飞控上很常见，故该缺口会直接导致真机不可用 ——
     * 时间尺度失配使积分与滤波器参数同时错位。
     *
     * @par 约定
     *
     * 返回 `<= 0` 表示**时间戳不可用**，主循环随即回退到默认步长
     * （`FlightControlConfig::default_dt`）。默认实现返回 -1，故既有实现
     * （如 `SimSensorReader`）无需改动，行为逐位不变。
     *
     * 时间戳应单调递增；主循环会做合理性检查（见 `FlightControlConfig::max_dt`），
     * 异常值（非正、倒流、跳跃过大）将被忽略并回退到默认步长。
     */
    [[nodiscard]] virtual double imuTimestamp() { return -1.0; }
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

    /**
     * @brief 降级决策变化时的通知
     *
     * 制导源据此切换任务模式（如返航、紧急降落）。
     * 默认空实现：不响应决策变化的源（如固定悬停点）无需重写。
     */
    virtual void onDecisionChanged(const DegradeDecision &d,
                                   const std::array<double, 3> &current_pos,
                                   double time) {
        (void)d;
        (void)current_pos;
        (void)time;
    }

    /**
     * @brief 是否已完成着陆
     *
     * 紧急降落模式下，制导源根据垂直下降轨迹判断飞机是否已触地。
     * 默认返回 false。
     */
    [[nodiscard]] virtual bool isLanded() const { return false; }
};

} // namespace oi3

#endif // OI3_HAL_ABSTRACTION_H
