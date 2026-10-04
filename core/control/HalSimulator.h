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

#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

namespace oi3 {

/**
 * @struct PositionImperfection
 * @brief 位置量测的非理想特性
 *
 * 默认全部关闭（sigma=0、delay_steps=0），此时 `SimSensorReader::readPosition()`
 * 直接返回仿真真值 —— 与改动前逐位一致，既有测试不受影响。
 *
 * @par 为什么需要这个配置
 *
 * `SimSensorReader` 原先无条件返回真值位置，造成一个验证缺口：
 * `PositionNonIdealClosedLoopTest` 覆盖了非理想量测，但它走**裸循环**；
 * 而走 HAL 的**生产主循环**测试（FlightControlLoopTest / GuidanceIntegrationTest
 * / EmergencyLandingTest）全部建立在理想量测之上。「非理想量测 + 生产主循环」
 * 这一组合从未被验证，而真机上 GPS/光流必然带噪声与延迟。
 *
 * 噪声与延迟的语义与 `PositionNonIdealClosedLoopTest` 保持一致以便对照：
 * 量测每 pos_decim 个 IMU 步抽取一次；延迟以 IMU 步数给出（内部换算为量测样本数）。
 *
 * 注：定义在 namespace 级而非嵌套于 `SimSensorReader` 内 —— 后者会触发
 * 「嵌套类型的默认成员初始化器不可用于同类构造函数的默认参数」这一 C++ 限制。
 */
struct PositionImperfection {
    /// 位置量测噪声标准差（米），逐轴独立高斯；0 表示无噪声
    double sigma = 0.0;
    /// 位置量测延迟（IMU 步数）；0 表示无延迟
    int delay_steps = 0;
    /// 噪声随机种子
    std::uint32_t seed = 20260918u;
};

/**
 * @class SimSensorReader
 * @brief 仿真传感器读取：从 SixDofSimulator + ImuModel 生成测量
 */
class SimSensorReader : public HalSensorReader {
  public:
    /**
     * @param sim          六自由度仿真器（不拥有）
     * @param imu          IMU 误差模型（不拥有）
     * @param gravity_ned  NED 系重力加速度 {0, 0, g}
     * @param pos_decim    位置测量分频（每多少 IMU 步输出一次位置）
     * @param imperfection 位置量测的非理想特性（默认理想）
     */
    SimSensorReader(SixDofSimulator *sim, ImuModel *imu,
                    const std::array<double, 3> &gravity_ned, int pos_decim = 10,
                    const PositionImperfection &imperfection = {})
        : _sim(sim), _imu(imu), _gravity_ned(gravity_ned), _pos_decim(pos_decim),
          _step_count(-1), _sigma(imperfection.sigma),
          _delay_samples(std::max(0, imperfection.delay_steps) / std::max(1, pos_decim)),
          _rng(imperfection.seed), _gauss(0.0, 1.0) {}

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
        const std::array<double, 3> truth = {static_cast<double>(p[0]),
                                             static_cast<double>(p[1]),
                                             static_cast<double>(p[2])};

        // 理想路径：逐位等同于改动前（不引入额外浮点运算）
        if (_delay_samples <= 0 && _sigma <= 0.0) {
            return truth;
        }

        // 延迟：真值入队，队首即 delay_samples 拍之前的值
        std::array<double, 3> measured = truth;
        if (_delay_samples > 0) {
            _history.push_back(truth);
            const auto d = static_cast<std::size_t>(_delay_samples);
            if (_history.size() > d) {
                measured = _history.front();
                _history.erase(_history.begin());
            }
            // 历史尚未填满时沿用当前真值（等效于启动阶段无延迟），
            // 避免用零值初始化导致估计器被错误牵引。
        }

        if (_sigma > 0.0) {
            for (int i = 0; i < 3; ++i) {
                measured[static_cast<std::size_t>(i)] += _sigma * _gauss(_rng);
            }
        }
        return measured;
    }

    [[nodiscard]] double time() override { return _sim->time(); }

  private:
    SixDofSimulator *_sim;
    ImuModel *_imu;
    std::array<double, 3> _gravity_ned;
    int _pos_decim;
    int _step_count;

    double _sigma;
    int _delay_samples;
    std::vector<std::array<double, 3>> _history;
    std::mt19937 _rng;
    std::normal_distribution<double> _gauss;
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
