/**
 * @file WindModel.h
 * @brief 风场模型：常值风、阵风、大气湍流
 * @author GhostFace
 * @date 2026/9/18
 *
 * @par 为什么需要它
 *
 * 在此之前，仿真里的大气是完全静止的。控制器只需要抵抗重力和自己的加速度，
 * 而真实飞行中最大的外部扰动来自风 —— 它才是把「仿真里飞得好」和「真机上飞得
 * 住」区分开的东西。
 *
 * @par 风是怎么作用到机体上的
 *
 * 四旋翼本体没有机翼，气动力主要来自**相对气流**产生的阻力（机身、桨盘、起落架）。
 * 关键在于阻力取决于**相对速度**而不是地速：
 *
 * @verbatim
 *   v_rel = v_body − v_wind
 *   F_aero = −k · |v_rel| · v_rel        （逐轴二次形式，与既有阻力模型同口径）
 * @endverbatim
 *
 * 顺风飞行时相对气流小、阻力小；逆风时阻力大，且需要额外的推力分量来抵消风力。
 * 风速为零时模型退化回原有的阻力项，因此**不会改变任何既有结果**。
 *
 * @par 抗风能力的物理上限
 *
 * 四旋翼靠**倾斜**产生水平力：倾角 θ 时水平力为 `T·sinθ`，而竖直分量 `T·cosθ`
 * 必须仍等于重力。于是水平可用力上限为
 *
 * @verbatim
 *   F_h_max = tan(θ_max) · m · g
 * @endverbatim
 *
 * 需要平衡风阻 `k·v²`，因此可悬停的最大风速是
 *
 * @verbatim
 *   v_max = sqrt( tan(θ_max)·m·g / k )
 * @endverbatim
 *
 * 默认参数（m=1 kg、θ_max=35°、k=0.049）给出约 11.8 m/s。**超过这个风速无法
 * 悬停**，不是控制器不够好，而是推力方向被倾角限幅卡死了 —— 这个数字由
 * WindTunnelTest 实测校验。
 */

#ifndef OI3_WIND_MODEL_H
#define OI3_WIND_MODEL_H

#include <array>
#include <cstdint>
#include <random>
#include <string>

namespace oi3 {

/// NED 系风速（米/秒）：x 向北、y 向东、z 向下
using WindVec = std::array<double, 3>;

/**
 * @class WindModel
 * @brief 风场接口：给定时刻返回 NED 风速
 *
 * 只依赖时间而不依赖位置。真实大气有空间变化（风切变、湍流场），但飞行器的
 * 尺度只有几米，远小于湍流场的空间相关长度，因此在单机仿真里时间模型足够。
 * 多机编队若需要模拟空间相关性，再扩展接口即可。
 */
class WindModel {
  public:
    virtual ~WindModel() = default;

    /// 时刻 t（秒）的风速（NED，米/秒）
    [[nodiscard]] virtual WindVec at(double t) = 0;

    /// 供日志与测试报告使用的名称
    [[nodiscard]] virtual std::string name() const = 0;

    /// 复位内部状态（随机数发生器与滤波器状态）
    virtual void reset() {}
};

/// 无风：模型退化回原有动力学，用于对照组
class NoWind : public WindModel {
  public:
    [[nodiscard]] WindVec at(double /*t*/) override { return {0.0, 0.0, 0.0}; }
    [[nodiscard]] std::string name() const override { return "无风"; }
};

/// 由风速与方位角构造 NED 风速向量
/// @param speed        风速（米/秒，非负）
/// @param to_deg       风**吹向**的方位角（度）：0 = 吹向北，90 = 吹向东
/// @param vertical     垂直分量（米/秒，正为下沉气流）
[[nodiscard]] WindVec windFromSpeedDirection(double speed, double to_deg,
                                             double vertical = 0.0);

/**
 * @class SteadyWind
 * @brief 常值风：风速风向恒定
 *
 * 最基础的抗风测试场景。悬停时飞行器必须**持续保持一个倾角**来抵消风力，
 * 因此与控制器的定点悬停不同，此时姿态不是水平的 —— 这一点是判断抗风是否
 * 真正生效的直观标志。
 */
class SteadyWind : public WindModel {
  public:
    SteadyWind(double speed, double to_deg, double vertical = 0.0)
        : _v(windFromSpeedDirection(speed, to_deg, vertical)) {}

    [[nodiscard]] WindVec at(double /*t*/) override { return _v; }

    [[nodiscard]] std::string name() const override;

  private:
    WindVec _v;
};

/**
 * @class GustWind
 * @brief 常值风 + 1-cos 离散阵风
 *
 * 1-cos 型阵风是飞行品质规范里的标准阵风形式（连续阵风的离散化）：风速在
 * 持续时间内平滑地升降，没有阶跃，因此不会引入规范之外的频率成分。相比直接用
 * 阶跃，它能测出控制器对付**渐变扰动**的能力，而不只是冲击响应。
 *
 * @verbatim
 *   w(τ) = (A/2)·(1 − cos(2π·τ/T))     τ ∈ [0, T]
 * @endverbatim
 * τ=0 与 τ=T 时值为 0，τ=T/2 时达到峰值 A，两端与常值风平滑衔接。
 */
class GustWind : public WindModel {
  public:
    /**
     * @param steady       背景常值风（米/秒）
     * @param steady_to_deg 背景风方位角（度）
     * @param gust_amp     阵风峰值幅度（米/秒）
     * @param gust_to_deg  阵风方位角（度）
     * @param start        阵风起始时刻（秒）
     * @param duration     阵风持续时间（秒）
     */
    GustWind(double steady, double steady_to_deg, double gust_amp, double gust_to_deg,
             double start, double duration);

    [[nodiscard]] WindVec at(double t) override;

    [[nodiscard]] std::string name() const override;

    /// 当前阵风增益系数（0~1），供测试检查时间曲线
    [[nodiscard]] double gustFactor(double t) const;

  private:
    WindVec _steady;
    WindVec _gust_dir; ///< 单位向量
    double _amp;
    double _start;
    double _duration;
};

/**
 * @class TurbulentWind
 * @brief 常值风 + Dryden 连续湍流
 *
 * Dryden 谱是 MIL-F-8785C 采用的大气湍流模型，工程上常用它的**一阶成形滤波器**
 * 实现：把白噪声通过一阶低通，得到功率谱密度与 Dryden 谱一致的时间序列。
 *
 * @verbatim
 *   H(s) = σ·sqrt(2·τ/π) / (1 + τ·s),    τ = L / V
 * @endverbatim
 *
 * 其中 σ 是湍流强度（均方根风速），L 是湍流尺度长度，V 是飞行速度。
 *
 * 实现上就是三个轴各跑一个一阶低通，输入为高斯白噪声。滤波器的输入噪声幅度按
 * `σ_in = σ_out / sqrt(a/(2−a))` 反推，使输出标准差恰好等于设定的湍流强度 ——
 * 否则参数就失去物理含义，调 σ 只能「看着差不多」。
 *
 * @note 与阵风的区别：阵风是确定性的、可重复的扰动；湍流是随机过程。前者用来
 *       测瞬态响应，后者用来测统计意义上的平均性能与鲁棒性。
 */
class TurbulentWind : public WindModel {
  public:
    /**
     * @param steady        背景常值风（米/秒）
     * @param steady_to_deg 背景风方位角（度）
     * @param intensity     湍流强度 σ（米/秒，各轴的均方根风速）
     * @param scale_length  湍流尺度长度 L（米）
     * @param dt            采样步长（秒），用于滤波器离散化
     * @param seed          随机种子（固定种子保证可复现）
     */
    TurbulentWind(double steady, double steady_to_deg, double intensity,
                  double scale_length, double dt, std::uint32_t seed);

    [[nodiscard]] WindVec at(double t) override;

    [[nodiscard]] std::string name() const override;

    void reset() override;

    /// 推进一个采样步并返回当前风速（供仿真器按步长调用）
    [[nodiscard]] WindVec step();

    /// 湍流强度设定值（米/秒）
    [[nodiscard]] double intensity() const { return _intensity; }

  private:
    /// 推进一个采样步
    void advance();

    WindVec _steady;
    double _intensity;
    double _scale_length;
    double _dt;
    std::uint32_t _seed;

    std::mt19937 _rng;
    std::normal_distribution<double> _unit{0.0, 1.0};

    std::array<double, 3> _state{}; ///< 三轴成形滤波器状态
    double _alpha = 0.0;            ///< 一阶低通系数
    double _sigma_in = 0.0;         ///< 输入噪声幅度（已按输出方差反推）
    double _last_t = 0.0;           ///< 上次查询时刻，防止同一时刻被推进两次
};

} // namespace oi3

#endif // OI3_WIND_MODEL_H
