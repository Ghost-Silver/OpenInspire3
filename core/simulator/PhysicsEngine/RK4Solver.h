/**
 * @file RK4Solver.h
 * @brief OpenInspire3 通用四阶龙格-库塔积分器
 * @author GhostFace
 * @date 2026/9/16
 */

#ifndef OI3_RK4_SOLVER_H
#define OI3_RK4_SOLVER_H

#include <concepts>

namespace oi3 {

/**
 * @concept RK4State
 * @brief 可作为 RK4 状态的类型约束
 *
 * 状态只需支持加法与标量乘，这是 RK4 线性组合
 * `y + (k1 + 2*k2 + 2*k3 + k4) / 6` 的全部要求。
 * Tensor 与 DroneState 都满足。
 */
template <typename S>
concept RK4State = requires(const S &a, const S &b, float k) {
    { a + b } -> std::convertible_to<S>;
    { a * k } -> std::convertible_to<S>;
};

/**
 * @class RK4Integrator
 * @brief 经典四阶龙格-库塔积分器（定步长）
 *
 * 单步公式（k 已含步长）：
 * @verbatim
 *   k1 = dt * f(t,      y)
 *   k2 = dt * f(t+dt/2, y + k1/2)
 *   k3 = dt * f(t+dt/2, y + k2/2)
 *   k4 = dt * f(t+dt,   y + k3)
 *   y' = y + (k1 + 2*k2 + 2*k3 + k4) / 6
 * @endverbatim
 *
 * 局部截断误差 O(dt^5)，全局误差 O(dt^4)。
 *
 * **为什么用模板而非 std::function**：旧实现把右端函数包装成
 * `std::function<Tensor(double, const Tensor&)>`，每次求值都要经过
 * 类型擦除与间接调用。模板化后右端函数被直接内联，同时让积分器可以服务
 * 任意满足 RK4State 的状态类型（标量 ODE 测试、DroneState 仿真共用一份实现）。
 */
class RK4Integrator {
  public:
    explicit RK4Integrator(double step_size) : _dt(step_size) {}

    /// 当前步长（秒）
    [[nodiscard]] double dt() const { return _dt; }

    /**
     * @brief 推进单个步长
     * @tparam State 状态类型，需满足 RK4State
     * @tparam Func  右端函数类型，调用形式 `State(double t, const State &y)`
     * @param f 微分方程右端，返回 dy/dt
     * @param t 当前时刻
     * @param y 当前状态
     * @return 步进后的状态
     */
    template <RK4State State, typename Func>
    [[nodiscard]] State step(const Func &f, double t, const State &y) const {
        const double half = _dt * 0.5;
        const float h = static_cast<float>(_dt);

        const State k1 = f(t, y) * h;
        const State k2 = f(t + half, y + k1 * 0.5f) * h;
        const State k3 = f(t + half, y + k2 * 0.5f) * h;
        const State k4 = f(t + _dt, y + k3) * h;

        return y + (k1 + k2 * 2.0f + k3 * 2.0f + k4) * (1.0f / 6.0f);
    }

    /**
     * @brief 从 t_start 积分到 t_end
     *
     * 完整步长走定步进；不足一步的余量用等长的临时积分器补足，
     * 保证积分终点不被截断（旧实现通过临时改写成员 dt 再改回来实现，
     * 存在被异常打断后步长无法复原的风险）。
     */
    template <RK4State State, typename Func>
    [[nodiscard]] State integrate(const Func &f, double t_start, double t_end,
                                  const State &y0) const {
        State y = y0;
        double t = t_start;

        const auto full_steps = static_cast<long long>((t_end - t_start) / _dt);
        for (long long i = 0; i < full_steps; ++i) {
            y = step(f, t, y);
            t += _dt;
        }

        const double remainder =
            (t_end - t_start) - static_cast<double>(full_steps) * _dt;
        if (remainder > 1e-12) {
            y = RK4Integrator(remainder).step(f, t, y);
        }
        return y;
    }

  private:
    double _dt;
};

} // namespace oi3

#endif // OI3_RK4_SOLVER_H
