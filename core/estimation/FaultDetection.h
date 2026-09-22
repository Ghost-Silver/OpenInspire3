/**
 * @file FaultDetection.h
 * @brief 基于残差的故障检测与辨识（FDI）
 * @author GhostFace
 * @date 2026/9/18
 *
 * @par 为什么建立在残差上
 *
 * 在线辨识（OnlineIdentification.h）已经在算「预测值 − 实测值」这个残差。
 * 参数正常时它只是噪声，**参数突变时它立刻变大** —— 也就是说，故障检测所需
 * 的信息本来就在手上，不需要另起一套机制。
 *
 * 这与项目既有结论一脉相承：可验证的因果链优于黑箱判断。残差是有物理含义
 * 的量（模型与现实的差），它的量级直接对应「模型偏离现实多远」。
 *
 * @par 三个层次的问题
 *
 * 1. **检测**：现在是否异常？—— 用残差的统计量做假设检验；
 * 2. **定位**：异常从何时开始？—— 用累积和（CUSUM）找突变点；
 * 3. **辨识**：是哪种异常、多严重？—— 由参数估计值反推物理量。
 *
 * @par 为什么用 CUSUM 而不是简单阈值
 *
 * 单点阈值对**缓变**故障不敏感：缓慢退化的电机，每一拍的残差都在噪声范围内，
 * 但累积起来会显著偏离零。CUSUM 累积对数似然比，对小而持续的偏差敏感，
 * 同时对单个野点不敏感 —— 这正是缓变故障与突发故障都需要的性质。
 *
 * @par 与容错控制的关系
 *
 * 项目里容错旋转那条线一直卡着（姿态保持已做成、位置那半未完成）。FDI 是
 * 容错的前置：**必须先知道出了什么故障，才能谈如何重构**。本文件只做检测与
 * 辨识，不做重构 —— 但给出了重构所需的信息（哪一轴、多大程度）。
 */

#ifndef OI3_FAULT_DETECTION_H
#define OI3_FAULT_DETECTION_H

#include <algorithm>
#include <cmath>
#include <deque>
#include <vector>

namespace oi3 {

/**
 * @brief 残差监测器：由滑动窗口给出统计量并判定异常
 *
 * 残差定义为 `预测 − 实测`。正常时它应当零均值、方差有限；故障时均值偏离零。
 */
class ResidualMonitor {
  public:
    /**
     * @param window    滑动窗口长度（用于估计噪声水平）
     * @param threshold 判定阈值，单位为「噪声标准差」的倍数
     */
    explicit ResidualMonitor(int window = 200, double threshold = 4.0)
        : _window(std::max(10, window)), _threshold(threshold) {}

    /// 喂入一步残差
    void update(double residual) {
        _buf.push_back(residual);
        _sum += residual;
        _sum_sq += residual * residual;
        // 弹出时**必须同时扣除**它的贡献。第一版只 push/累加、弹出后没有减，
        // 导致 _sum 无限累积（窗口已满但和仍在增长），均值随之失真。
        if (static_cast<int>(_buf.size()) > _window) {
            const double old = _buf.front();
            _buf.pop_front();
            _sum -= old;
            _sum_sq -= old * old;
        }
    }

    /// 当前窗口内的残差均值
    [[nodiscard]] double mean() const {
        return _buf.empty() ? 0.0 : _sum / static_cast<double>(_buf.size());
    }

    /// 当前窗口内的残差标准差（作为噪声水平估计）
    [[nodiscard]] double stdDev() const {
        const double n = static_cast<double>(_buf.size());
        if (n < 2.0) {
            return 0.0;
        }
        const double m = mean();
        const double var = std::max(0.0, _sum_sq / n - m * m);
        return std::sqrt(var);
    }

    /// 当前残差相对噪声水平的偏离（单位：σ）
    [[nodiscard]] double deviationSigma() const {
        const double s = stdDev();
        if (s < 1e-12) {
            // 噪声极小时用绝对下限，避免除零后给出无穷大的假警报
            return std::fabs(mean()) / 1e-9;
        }
        return std::fabs(mean()) / s;
    }

    /// 是否判定为异常
    [[nodiscard]] bool isFaulted() const { return deviationSigma() > _threshold; }

    [[nodiscard]] int samples() const { return static_cast<int>(_buf.size()); }

    void reset() {
        _buf.clear();
        _sum = 0.0;
        _sum_sq = 0.0;
    }

  private:
    int _window;
    double _threshold;
    std::deque<double> _buf;
    double _sum = 0.0;
    double _sum_sq = 0.0;
};

/**
 * @brief CUSUM 突变检测器
 *
 * 累积和检验：对「残差均值是否偏离零」做序贯检验。相比单点阈值，
 * 它对**小而持续**的偏差敏感（缓变故障），对单个野点不敏感。
 *
 * @verbatim
 *   S⁺_k = max(0, S⁺_{k−1} + r_k − k_d)
 *   S⁻_k = max(0, S⁻_{k−1} − r_k − k_d)
 * @endverbatim
 *
 * `k_d` 为松弛量（通常取噪声的 0.5~1 倍），使零均值噪声不会误触发；
 * 两侧累积和任一超过阈值 `h` 即报警。
 */
class CusumDetector {
  public:
    /**
     * @param drift   松弛量 k_d：小于此幅度的偏差不被累积（抑制噪声误报）
     * @param thresh  报警阈值 h
     */
    explicit CusumDetector(double drift = 0.02, double thresh = 0.5)
        : _drift(drift), _thresh(thresh) {}

    /**
     * @brief 按噪声水平构造（推荐用法）
     *
     * 松弛量与阈值都应当**随噪声缩放** —— 硬编码常数在不同噪声水平下没有
     * 可比性：噪声大时会频繁误报，噪声小时又迟钝得几乎不报。
     *
     * 标准取法：松弛量 `k_d = 0.5σ`（抑制零均值噪声的累积），
     * 报警阈值 `h = 5σ`（在 H₀ 下极少误报，同时对 σ 量级的持续偏移敏感）。
     */
    [[nodiscard]] static CusumDetector forNoise(double sigma, double k_sigma = 0.5,
                                                double h_sigma = 5.0) {
        return CusumDetector(k_sigma * sigma, h_sigma * sigma);
    }

    void update(double residual) {
        _sp = std::max(0.0, _sp + residual - _drift);
        _sm = std::max(0.0, _sm - residual - _drift);
        if (!_alarmed && (_sp > _thresh || _sm > _thresh)) {
            _alarmed = true;
            _alarm_index = _k;
        }
        ++_k;
    }

    [[nodiscard]] bool alarmed() const { return _alarmed; }

    /// 首次报警发生的步数（未报警时返回 −1）
    [[nodiscard]] long long alarmIndex() const { return _alarm_index; }

    /// 报警方向：+1 表示残差偏正，−1 表示偏负，0 表示未报警
    [[nodiscard]] int direction() const {
        if (!_alarmed) {
            return 0;
        }
        return (_sp > _sm) ? 1 : -1;
    }

    [[nodiscard]] double positiveSum() const { return _sp; }
    [[nodiscard]] double negativeSum() const { return _sm; }

    void reset() {
        _sp = 0.0;
        _sm = 0.0;
        _k = 0;
        _alarmed = false;
        _alarm_index = -1;
    }

  private:
    double _drift;
    double _thresh;
    double _sp = 0.0;
    double _sm = 0.0;
    long long _k = 0;
    bool _alarmed = false;
    long long _alarm_index = -1;
};

/**
 * @brief 故障类型
 */
enum class FaultType {
    None = 0,
    ThrustLoss,     ///< 推力损失（电机退化、桨叶损伤）
    IncreasedDrag,  ///< 阻力增大（机身损伤、外物附着）
    MassChange,     ///< 质量变化（载荷投放、电池消耗）
    Unknown,
};

/// 故障辨识结果
struct FaultEstimate {
    FaultType type = FaultType::None;
    double severity = 0.0; ///< 严重程度：相对标称值的比例（0.1 = 10% 退化）
    double confidence = 0.0; ///< 置信度 0~1，由残差信噪比给出
    bool detected = false;
};

/**
 * @brief 由参数估计值辨识故障类型与严重程度
 *
 * 判据：比较当前参数估计与标称值
 *  - 质量显著变化 → MassChange（严重度 = |Δm|/m₀）
 *  - 阻力系数显著增大 → IncreasedDrag
 *  - 质量与阻力都基本不变但推力残差显著 → ThrustLoss
 */
class FaultIdentifier {
  public:
    FaultIdentifier(double nominal_mass, double nominal_drag)
        : _m0(nominal_mass), _k0(nominal_drag) {}

    /**
     * @param mass_est 当前质量估计
     * @param drag_est 当前阻力系数估计
     * @param thrust_residual 推力方向的残差（牛顿），用于识别推力损失
     * @param noise_level 残差噪声水平，用于计算置信度
     */
    [[nodiscard]] FaultEstimate identify(double mass_est, double drag_est,
                                         double thrust_residual,
                                         double noise_level) const {
        FaultEstimate out;

        const double dm = (_m0 > 1e-9) ? std::fabs(mass_est - _m0) / _m0 : 0.0;
        const double dk = (_k0 > 1e-9) ? (drag_est - _k0) / _k0 : 0.0;

        // 置信度：残差超出噪声的倍数（饱和到 1）
        out.confidence = std::min(1.0, std::fabs(thrust_residual) /
                                            std::max(1e-9, 3.0 * noise_level));

        // 判据阈值取 5%：低于此量级难以与噪声区分
        const double thresh = 0.05;
        if (dm > thresh && dm >= std::fabs(dk)) {
            out.type = FaultType::MassChange;
            out.severity = dm;
            out.detected = true;
        } else if (dk > thresh) {
            out.type = FaultType::IncreasedDrag;
            out.severity = dk;
            out.detected = true;
        } else if (std::fabs(thrust_residual) > 3.0 * noise_level) {
            out.type = FaultType::ThrustLoss;
            out.severity = std::fabs(thrust_residual) / (_m0 * 9.81);
            out.detected = true;
        } else {
            out.type = FaultType::None;
            out.severity = 0.0;
            out.detected = false;
        }
        return out;
    }

  private:
    double _m0;
    double _k0;
};

} // namespace oi3

#endif // OI3_FAULT_DETECTION_H
