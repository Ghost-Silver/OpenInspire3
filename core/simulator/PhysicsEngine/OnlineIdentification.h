/**
 * @file OnlineIdentification.h
 * @brief 在线参数估计：递推最小二乘（RLS）
 * @author GhostFace
 * @date 2026/9/18
 *
 * @par 为什么需要在线辨识
 *
 * 项目已有离线辨识（ParameterIdentification，从整条轨迹批量拟合）。但离线
 * 有个前提：飞行前就能拿到数据、且参数在飞行中不变。真实情况不是这样 ——
 *
 *  - **载荷变化**：挂载相机、换电池都会改变质量与惯量；
 *  - **气动变化**：桨叶磨损、机身损伤、结冰；
 *  - **环境变化**：空气密度随高度与温度改变，直接影响气动系数。
 *
 * 这些变化让「出厂标定 + 离线辨识」的初值在飞行中逐渐失准。在线辨识边飞边估，
 * 是自适应控制的前提。
 *
 * @par 为什么用递推最小二乘
 *
 * 关键前提：**待辨识量在回归形式下是线性的**。以桨盘入流为例，推力损失为
 *
 * @verbatim
 *   F_loss = T·mu·v_axial = mu · (T·v_axial)
 * @endverbatim
 *
 * 即回归量 `phi = T·v_axial`、待估参数 `θ = mu`，关于 θ 线性。这类问题
 * 有闭式递推解（RLS），不需要梯度迭代、不需要存储历史数据，每步只做几次
 * 矩阵运算 —— 适合跑在 1 kHz 的控制回路里。
 *
 * 对质量、阻力系数等同样是线性的：
 *
 * @verbatim
 *   m·a = F_thrust + m·g − k·|v_rel|·v_rel
 *   ⇒ a − g = (1/m)·F_thrust − (k/m)·|v_rel|·v_rel
 * @endverbatim
 *
 * 令 θ = [1/m, k/m]，phi = [F_thrust, −|v_rel|·v_rel]，同样是线性回归。
 *
 * @par 遗忘因子的作用
 *
 * 标准 RLS 假设参数恒定，协方差矩阵 P 单调收缩、增益趋于零 —— 最终「学不动」，
 * 无法跟踪后续变化。引入遗忘因子 λ < 1 让旧数据权重指数衰减：
 *
 * @verbatim
 *   P_k = (P_{k-1} − K·phiᵀ·P_{k-1}) / λ
 * @endverbatim
 *
 * λ = 1 退化为标准 RLS（恒定参数）；λ 越小跟踪越快，但稳态方差越大。
 * 这是典型的「跟踪速度 vs 估计精度」权衡，本文件用测试把它量化出来。
 */

#ifndef OI3_ONLINE_IDENTIFICATION_H
#define OI3_ONLINE_IDENTIFICATION_H

#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

namespace oi3 {

/**
 * @brief 递推最小二乘估计器（支持遗忘因子）
 *
 * 求解 `y_k = phi_kᵀ·θ + noise` 中的 θ，逐样本递推更新。
 *
 * @tparam N 参数个数
 */
template <std::size_t N>
class RecursiveLeastSquares {
  public:
    /**
     * @param forgetting  遗忘因子 λ ∈ (0, 1]，1 表示不遗忘（恒定参数）
     * @param p_init      协方差初值，通常取较大值（表示对初值不确定）
     */
    explicit RecursiveLeastSquares(double forgetting = 1.0, double p_init = 100.0)
        : _lambda(forgetting) {
        _P.fill(0.0);
        for (std::size_t i = 0; i < N; ++i) {
            _P[i * N + i] = p_init;
        }
    }

    /// 设置参数初值
    void setTheta(const std::array<double, N> &theta) { _theta = theta; }

    [[nodiscard]] const std::array<double, N> &theta() const { return _theta; }

    /// 当前参数估计的方差（P 的对角元），可用于置信度判断
    [[nodiscard]] double variance(std::size_t i) const { return _P[i * N + i]; }

    /**
     * @brief 一步递推更新
     *
     * @param phi 回归量
     * @param y   观测值
     */
    void update(const std::array<double, N> &phi, double y) {
        // ---- 1. 计算增益 K = P·phi / (λ + phiᵀ·P·phi) ----
        std::array<double, N> Pphi{};
        for (std::size_t i = 0; i < N; ++i) {
            double s = 0.0;
            for (std::size_t j = 0; j < N; ++j) {
                s += _P[i * N + j] * phi[j];
            }
            Pphi[i] = s;
        }
        double denom = _lambda;
        for (std::size_t i = 0; i < N; ++i) {
            denom += phi[i] * Pphi[i];
        }
        if (std::fabs(denom) < 1e-18) {
            return; // 数值退化，跳过本步（不更新）
        }
        std::array<double, N> K{};
        for (std::size_t i = 0; i < N; ++i) {
            K[i] = Pphi[i] / denom;
        }

        // ---- 2. 参数更新 θ += K·(y − phiᵀ·θ) ----
        double pred = 0.0;
        for (std::size_t i = 0; i < N; ++i) {
            pred += phi[i] * _theta[i];
        }
        const double err = y - pred;
        _last_residual = err;
        for (std::size_t i = 0; i < N; ++i) {
            _theta[i] += K[i] * err;
        }

        // ---- 3. 协方差更新 P = (P − K·phiᵀ·P) / λ ----
        std::array<double, N * N> Pnew{};
        for (std::size_t i = 0; i < N; ++i) {
            for (std::size_t j = 0; j < N; ++j) {
                double s = 0.0;
                for (std::size_t k = 0; k < N; ++k) {
                    s += K[i] * phi[k] * _P[k * N + j];
                }
                Pnew[i * N + j] = (_P[i * N + j] - s) / _lambda;
            }
        }
        _P = Pnew;

        ++_count;
    }

    [[nodiscard]] long long count() const { return _count; }

    /**
     * @brief 最近一步的预测残差 `y − φᵀθ`
     *
     * 这是故障检测的原始信号：参数正常时它只是噪声，参数突变时它立刻变大。
     * 因此 FDI 不需要另起一套机制 —— 检测所需的信息本来就在估计器手上。
     */
    [[nodiscard]] double lastResidual() const { return _last_residual; }

  private:
    double _last_residual = 0.0;
    std::array<double, N> _theta{};
    std::array<double, N * N> _P{};
    double _lambda = 1.0;
    long long _count = 0;
};

/**
 * @brief 桨盘入流系数的在线估计
 *
 * 回归形式：`F_loss = mu · (T·v_axial)`，其中
 *  - `T` 为推力指令（牛顿）
 *  - `v_axial` 为轴向相对气流（机体 z 分量，米/秒）
 *  - `F_loss` 为由此造成的推力损失（牛顿）
 *
 * 单参数，用 1 维 RLS 即可。实际使用时把「推力损失」换成可观测量：
 * 由加速度残差反推 —— 但那是间接的。本类提供两种输入方式：
 *  1. `updateWithLoss()`：直接给损失（仿真中可用真值，验证估计器本身）；
 *  2. `updateFromResidual()`：给加速度残差，内部换算成损失。
 */
class InflowEstimator {
  public:
    explicit InflowEstimator(double forgetting = 0.998) : _rls(forgetting) {}

    /// 直接以「推力损失」为观测（仿真验证用）
    void updateWithLoss(double thrust, double v_axial, double loss) {
        const std::array<double, 1> phi{thrust * v_axial};
        _rls.update(phi, loss);
    }

    /**
     * @brief 由加速度残差估计
     *
     * 沿推力方向的加速度残差 `da_z` 对应损失力 `m·da_z`，故
     *     loss = m·da_z
     */
    void updateFromResidual(double thrust, double v_axial, double da_along_thrust,
                            double mass) {
        updateWithLoss(thrust, v_axial, mass * da_along_thrust);
    }

    /// 设置 mu 的初值（用于验证「无激励时停在初值」这一失效模式）
    void setTheta0(double mu0) { _rls.setTheta({mu0}); }

    [[nodiscard]] double mu() const { return _rls.theta()[0]; }
    [[nodiscard]] double variance() const { return _rls.variance(0); }
    [[nodiscard]] long long count() const { return _rls.count(); }
    [[nodiscard]] double lastResidual() const { return _rls.lastResidual(); }

  private:
    RecursiveLeastSquares<1> _rls;
};

/**
 * @brief 质量与阻力系数的在线估计
 *
 * 由竖直通道的运动方程
 *
 * @verbatim
 *   m·a_z = F_thrust·(z_b·(−e_z)) + m·g − k·|v_rel|·v_rel_z
 * @endverbatim
 *
 * 在悬停、姿态接近水平时 `z_b·(−e_z) ≈ 1`，简化为
 *
 * @verbatim
 *   a_z − g = (1/m)·F_thrust − (k/m)·|v_rel|·v_rel_z
 * @endverbatim
 *
 * 令 θ = [1/m, k/m]，回归量 φ = [F_thrust, −|v_rel|·v_rel_z]。
 * 注意估计出的是**比值**，需要 `m = 1/θ₀`、`k = θ₁·m` 换算。
 */
class MassDragEstimator {
  public:
    explicit MassDragEstimator(double forgetting = 1.0) : _rls(forgetting) {}

    void update(double thrust, double v_rel_z, double a_z, double g) {
        const double speed = std::fabs(v_rel_z);
        const std::array<double, 2> phi{thrust, -speed * v_rel_z};
        _rls.update(phi, a_z - g);
    }

    /// 估计的质量（θ₀ = 1/m）
    [[nodiscard]] double mass() const {
        const double inv = _rls.theta()[0];
        return (std::fabs(inv) > 1e-9) ? 1.0 / inv : 0.0;
    }

    /// 估计的阻力系数（θ₁ = k/m）
    [[nodiscard]] double dragCoeff() const {
        const double inv = _rls.theta()[0];
        if (std::fabs(inv) < 1e-9) {
            return 0.0;
        }
        return _rls.theta()[1] / inv;
    }

    [[nodiscard]] long long count() const { return _rls.count(); }
    [[nodiscard]] double lastResidual() const { return _rls.lastResidual(); }

  private:
    RecursiveLeastSquares<2> _rls;
};

} // namespace oi3

#endif // OI3_ONLINE_IDENTIFICATION_H
