/**
 * @file ExcitationMonitor.h
 * @brief 激励充分性监测与主动激励注入
 * @author GhostFace
 * @date 2026/9/18
 *
 * @par 要解决的问题
 *
 * OnlineIdTest 揭示了一个硬约束：**若回归量恒为零，参数在数学上不可辨** ——
 * 无激励跑 3000 步，估计值精确停在初值上。危险在于它不报错、不震荡、
 * 不发散，只是无声地失效。
 *
 * 悬停无风时 `v_axial ≈ 0`，而入流系数 mu 的回归量正是 `φ = T·v_axial`。
 * 也就是说：**飞行器越平稳，越估不出自己的气动参数** —— 这是个结构性矛盾，
 * 不能靠调参绕过，只能显式处理。
 *
 * @par 本文件提供两件事
 *
 * 1. **监测**：由累积信息量判断当前估计是否可信。理论依据是最小二乘的
 *    协方差 —— 参数估计标准差为
 *
 *    @verbatim
 *      std(θ̂) = σ / √R,     R = Σ λ^(k−i)·φ_i²
 *    @endverbatim
 *
 *    即估计精度由**累积激励**决定。R 小则不可信。这个式子给出的是**可预测的**
 *    误差量级，而不是事后才知道的结果 —— 前者才能用于在线决策。
 *
 * 2. **主动激励**：当激励不足时注入小幅机动，主动创造信息。代价是偏离任务
 *    轨迹，所以必须权衡「估计精度」与「任务扰动」。
 *
 * @par 为什么不能一直开着激励
 *
 * 主动激励本身是扰动：为了估计气动系数而让飞行器做多余机动，会牺牲悬停
 * 精度、增加能耗。合理策略是**按需激励** —— 只在信息量不足时注入，且幅度
 * 与所需信息量挂钩。这也是本文件把「监测」与「注入」分开的原因：前者是
 * 决策依据，后者是执行手段。
 */

#ifndef OI3_EXCITATION_MONITOR_H
#define OI3_EXCITATION_MONITOR_H

#include <algorithm>
#include <cmath>
#include <limits>

namespace oi3 {

/**
 * @brief 激励充分性监测器
 *
 * 累积回归量的平方和（带遗忘），据此给出参数估计的可预测标准差。
 */
class ExcitationMonitor {
  public:
    /**
     * @param forgetting 遗忘因子，应与估计器保持一致
     */
    explicit ExcitationMonitor(double forgetting = 1.0) : _lambda(forgetting) {}

    /// 喂入一步回归量
    void update(double phi) {
        _R = _lambda * _R + phi * phi;
        _samples += 1;
    }

    /// 重置（例如切换估计目标时）
    void reset() {
        _R = 0.0;
        _samples = 0;
    }

    /// 累积信息量 R = Σ λ^(k−i)·φ_i²
    [[nodiscard]] double information() const { return _R; }

    [[nodiscard]] long long samples() const { return _samples; }

    /**
     * @brief 预测参数估计的标准差
     *
     * @param obs_sigma 观测噪声标准差 σ
     * @return σ/√R；R 为零时返回无穷（完全不可辨）
     */
    [[nodiscard]] double predictedStdDev(double obs_sigma) const {
        if (_R <= 1e-18) {
            return std::numeric_limits<double>::infinity();
        }
        return obs_sigma / std::sqrt(_R);
    }

    /**
     * @brief 当前激励是否足以把估计精度做到目标水平
     *
     * @param obs_sigma  观测噪声标准差
     * @param target_std 期望的估计标准差
     */
    [[nodiscard]] bool sufficient(double obs_sigma, double target_std) const {
        return predictedStdDev(obs_sigma) <= target_std;
    }

    /**
     * @brief 达到目标精度**还差多少信息量**
     *
     * @return 需要补足的 R 增量；已达标时返回 0
     *
     * 用于决定主动激励的强度：缺口越大，需要注入的激励越强。
     */
    [[nodiscard]] double informationDeficit(double obs_sigma, double target_std) const {
        if (target_std <= 1e-18) {
            return 0.0;
        }
        const double r_needed = (obs_sigma * obs_sigma) / (target_std * target_std);
        return std::max(0.0, r_needed - _R);
    }

  private:
    double _lambda = 1.0;
    double _R = 0.0;
    long long _samples = 0;
};

/**
 * @brief 主动激励信号发生器
 *
 * 在目标轨迹上叠加一个小的正弦扰动，制造回归量的变化。
 *
 * @par 为什么用正弦而不是阶跃
 *
 * 正弦的均值为零，长期看**不引入位置偏移** —— 激励结束后飞行器自然回到
 * 原轨迹。阶跃则会让飞行器偏出去，必须额外规划回程。对「按需激励」而言，
 * 零均值是重要性质。
 *
 * @par 频率选择
 *
 * 频率过低则一个周期内信息量太少、收敛慢；过高则被姿态环带宽衰减、实际
 * 产生的 `v_axial` 反而变小。合理区间在位置环带宽与姿态环带宽之间 ——
 * 本实现默认 1.5 rad/s（远低于姿态环 9 rad/s，高于位置环 2 rad/s）。
 */
class ExcitationInjector {
  public:
    ExcitationInjector(double freq = 1.5, double amp = 0.3)
        : _freq(freq), _amp(amp) {}

    /// 开启/关闭激励
    void setActive(bool on) {
        if (on && !_active) {
            _phase0 = _t; // 记录起始相位，保证从零开始（避免阶跃）
        }
        _active = on;
    }

    [[nodiscard]] bool active() const { return _active; }

    /// 推进时间并返回本步应叠加的位移（NED z 方向）
    double step(double dt) {
        _t += dt;
        if (!_active) {
            return 0.0;
        }
        // 从零相位开始的 1−cos 包络，保证开关瞬间无阶跃
        const double tau = _t - _phase0;
        const double ramp = std::min(1.0, tau / 0.5); // 0.5 秒渐入
        return _amp * ramp * std::sin(_freq * tau);
    }

    /// 当前激励幅度（供诊断）
    [[nodiscard]] double amplitude() const { return _amp; }
    [[nodiscard]] double frequency() const { return _freq; }

  private:
    double _freq = 1.5;
    double _amp = 0.3;
    double _t = 0.0;
    double _phase0 = 0.0;
    bool _active = false;
};

} // namespace oi3

#endif // OI3_EXCITATION_MONITOR_H
