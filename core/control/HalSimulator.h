/**
 * @file HalSimulator.h
 * @brief 仿真 HAL 实现：让主循环跑在 SixDofSimulator 上
 *
 * 这是 HAL 接口的「仿真后端」：
 * - 传感器：用 SixDofSimulator 的真值状态，经 ImuModel 加噪后输出；
 * - 位置测量：直接取真值位置（可配噪声/延迟，此处先取理想化版本）；
 * - 执行器：把推力/力矩写回 SixDofSimulator::step()；
 * - 时间：取仿真器的累积时间。
 *
 * @par 与测试代码的区别
 *
 * 此前测试文件（如 AccelFaultClosedLoopTest）把传感器读取、估计更新、
 * 控制计算、仿真推进全部写在一个大循环里。HalSimulator 把这些职责拆开：
 * 主循环只通过 HAL 接口操作，不再直接碰仿真器。这让「测试里的主循环」
 * 与「真机上的主循环」是同一份代码。
 */

#ifndef OI3_HAL_SIMULATOR_H
#define OI3_HAL_SIMULATOR_H

#include "HalAbstraction.h"
#include "ImuModel.h"
#include "SixDofSimulator.h"

namespace oi3 {

/**
 * @class SimSensorReader
 * @brief 仿真传感器读取：从 SixDofSimulator + ImuModel 生成测量
 */
class SimSensorReader : public HalSensorReader {
  public:
    /**
     * @param sim         六自由度仿真器（不拥有）
     * @param imu         IMU 误差模型（不拥有）
     * @param gravity_ned NED 系重力加速度 {0, 0, g}
     * @param pos_decim   位置测量分频（每多少 IMU 步输出一次位置）
     */
    SimSensorReader(SixDofSimulator *sim, ImuModel *imu,
                    const std::array<double, 3> &gravity_ned, int pos_decim = 10)
        : _sim(sim), _imu(imu), _gravity_ned(gravity_ned), _pos_decim(pos_decim),
          _step_count(-1) {}

    [[nodiscard]] ImuSample readImu() override {
        ++_step_count; // 须在 hasPositionUpdate() 之前调用，以匹配 k % decim == 0 语义
        // 比力 = a_ned − g_ned。悬停时 a_ned = 0，比力 = −g_ned。
        // ImuModel::measure() 内部自己做 nedToBody 旋转，此处只需传 NED 系值。
        const std::array<double, 3> specific_force_ned = {0.0, 0.0, -_gravity_ned[2]};
        return _imu->measure(_sim->state(), specific_force_ned, _sim->config().base.dt);
    }

    [[nodiscard]] bool hasPositionUpdate() override {
        return (_step_count % _pos_decim) == 0;
    }

    [[nodiscard]] std::array<double, 3> readPosition() override {
        const auto p = toVector(_sim->state().pos);
        return {static_cast<double>(p[0]), static_cast<double>(p[1]),
                static_cast<double>(p[2])};
    }

    [[nodiscard]] double time() override { return _sim->time(); }

  private:
    SixDofSimulator *_sim;
    ImuModel *_imu;
    std::array<double, 3> _gravity_ned;
    int _pos_decim;
    int _step_count;
};

/**
 * @class SimActuatorWriter
 * @brief 仿真执行器写入：把指令推给 SixDofSimulator::step()
 */
class SimActuatorWriter : public HalActuatorWriter {
  public:
    explicit SimActuatorWriter(SixDofSimulator *sim) : _sim(sim) {}

    void writeCommand(const SixDofCommand &cmd) override {
        _sim->step(cmd.thrust_body, cmd.torque);
    }

  private:
    SixDofSimulator *_sim;
};

/**
 * @class FixedSetpointSource
 * @brief 固定目标点（悬停场景）
 */
class FixedSetpointSource : public HalSetpointSource {
  public:
    explicit FixedSetpointSource(const Tensor &target) : _target(target) {}

    [[nodiscard]] Tensor currentTarget(double /*time*/) override { return _target; }

    [[nodiscard]] bool hasArrived(const std::array<double, 3> &pos,
                                  double tolerance) const override {
        const auto t = toVector(_target);
        const double dx = pos[0] - static_cast<double>(t[0]);
        const double dy = pos[1] - static_cast<double>(t[1]);
        const double dz = pos[2] - static_cast<double>(t[2]);
        return std::sqrt(dx * dx + dy * dy + dz * dz) < tolerance;
    }

  private:
    Tensor _target;
};

} // namespace oi3

#endif // OI3_HAL_SIMULATOR_H
