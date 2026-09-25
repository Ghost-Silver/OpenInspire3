/**
 * @file DisturbanceObserver.h
 * @brief 扰动观测器：无风速传感器时估计并前馈补偿未知扰动
 * @author GhostFace
 * @date 2026/9/18
 *
 * @par 它填补的是哪一格
 *
 * 扰动补偿按「对扰动的知情程度」排列有三种途径：
 *
 * | 途径 | 前提 | 精度 | 相位代价 |
 * |------|------|------|---------|
 * | 前馈 | 知道扰动（需传感器或准确模型） | 精确 | 无 |
 * | 观测器 | 能从残差推断扰动 | 收敛后精确 | 很小 |
 * | 积分 | 都不知道 | 慢，需误差累积 | -90° 量级 |
 *
 * 本项目此前只有**前馈**（`computeWithWind` 需要外部喂入风速）与**积分**
 * （`pos_ki`）。真机上没有风速传感器，前馈那条路走不通；积分则以牺牲相位
 * 裕度为代价。观测器这一格是空的，本类填补它。
 *
 * @par 原理
 *
 * 控制器发出的期望加速度 `a_des` 与实际发生的加速度 `a_actual` 之差，正是
 * **未被模型解释的那部分力**：
 *
 * @verbatim
 *   a_actual = a_des + d      =>      d = a_actual − a_des
 * @endverbatim
 *
 * 用一阶低通滤掉噪声即得估计 `d_hat`：
 *
 * @verbatim
 *   d_hat += (dt / (Tau + dt)) · (d − d_hat),     Tau = 1/(2π·f_obs)
 * @endverbatim
 *
 * 补偿时**减去**估计值（`a_des = a_ref − d_hat`），使
 * `a_actual = a_ref − d_hat + d → a_ref`。
 *
 * @par 为什么它可能优于积分
 *
 * 观测器作用于**前馈路径**，不在反馈回路内。理想情况下
 *
 * @verbatim
 *   a_actual = a_ref + (1 − Q(s))·d,     Q(s) = 1/(Tau·s + 1)
 * @endverbatim
 *
 * 回路增益（`a_ref` 到位置）不变，因此相位裕度代价远小于积分的 -90°；
 * 而 `1 − Q(0) = 0`，直流扰动仍被完全抑制。该判断需实测确认（见
 * DisturbanceObserverTest），不可仅凭推导采信。
 *
 * @par 代价与局限
 *
 * 1. **噪声放大**：`d` 由速度差分得到，速度噪声经 `Q(s)` 进入前馈。观测带宽
 *    越高、噪声穿透越多 —— 与高控制带宽放大传感器噪声同理。
 * 2. **依赖模型**：`a_des` 是控制器**认为**发出的加速度。若执行器饱和或模型
 *    参数（质量、推力系数）不准，残差里会混入这些误差并被误认为扰动。故
 *    调用方应传入**限幅后**的实际值，而非原始指令。
 * 3. **需要速度**：无速度观测时退化为位置差分，噪声更大。
 */

#ifndef OI3_DISTURBANCE_OBSERVER_H
#define OI3_DISTURBANCE_OBSERVER_H

#include <algorithm>
#include <cmath>

namespace oi3 {

/**
 * @class DisturbanceObserver
 * @brief 估计 NED 系下的扰动加速度（m/s²），供前馈补偿使用
 */
class DisturbanceObserver {
  public:
    /**
     * @param bandwidth_hz 观测器带宽（Hz）。经验取控制带宽的 1/3 ~ 1/5：
     *                     过高则放大噪声并可能与控制环耦合，过低则响应慢。
     * @param limit        估计值限幅（m/s²），防止饱和期间估计值发散。
     */
    explicit DisturbanceObserver(double bandwidth_hz = 2.0, double limit = 12.0)
        : _limit(limit) {
        setBandwidth(bandwidth_hz);
    }

    /// 设置观测器带宽（Hz）；<= 0 表示禁用（估计值恒为 0）
    void setBandwidth(double hz) {
        _Tau = (hz > 1e-9) ? (1.0 / (2.0 * M_PI * hz)) : 0.0;
    }

    [[nodiscard]] double bandwidthHz() const {
        return (_Tau > 1e-9) ? (1.0 / (2.0 * M_PI * _Tau)) : 0.0;
    }

    void setLimit(double limit) { _limit = limit; }

    void reset() {
        for (int i = 0; i < 3; ++i) {
            _d_hat[i] = 0.0;
            _prev_vel[i] = 0.0;
        }
        _has_prev = false;
        _ready = false;
    }

    /**
     * @brief 推进一拍
     * @param vel         当前速度（NED，m/s），来自状态估计器
     * @param a_intended  控制器**认为**已施加的加速度（NED，m/s²），须含饱和修正
     * @param dt          步长（秒）
     */
    void update(const double vel[3], const double a_intended[3], double dt) {
        if (dt <= 0.0 || _Tau <= 1e-9) {
            return;
        }
        if (!_has_prev) {
            for (int i = 0; i < 3; ++i) {
                _prev_vel[i] = vel[i];
            }
            _has_prev = true;
            return;
        }

        const double alpha = dt / (_Tau + dt);
        for (int i = 0; i < 3; ++i) {
            const double a_meas = (vel[i] - _prev_vel[i]) / dt;
            const double resid = a_meas - a_intended[i];
            _d_hat[i] += alpha * (resid - _d_hat[i]);
            _d_hat[i] = std::max(-_limit, std::min(_limit, _d_hat[i]));
            _prev_vel[i] = vel[i];
        }
        _ready = true;
    }

    /// 当前扰动加速度估计（NED，m/s²）
    [[nodiscard]] const double *estimate() const { return _d_hat; }

    /// 是否已积累至少一拍（未 ready 时调用方应回退到不补偿）
    [[nodiscard]] bool ready() const { return _ready; }

  private:
    double _Tau = 1.0 / (2.0 * M_PI * 2.0);
    double _limit = 12.0;
    double _d_hat[3] = {0.0, 0.0, 0.0};
    double _prev_vel[3] = {0.0, 0.0, 0.0};
    bool _has_prev = false;
    bool _ready = false;
};

} // namespace oi3

#endif // OI3_DISTURBANCE_OBSERVER_H
