/**
 * @file WindModel.cpp
 * @brief 风场模型实现
 * @author GhostFace
 * @date 2026/9/18
 */

#include "WindModel.h"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace oi3 {

namespace {
constexpr double kPi = 3.14159265358979323846;

/// 保留两位小数的字符串化，供 name() 拼装可读描述
std::string fmt(double v) {
    std::ostringstream os;
    os.setf(std::ios::fixed);
    os.precision(2);
    os << v;
    return os.str();
}
} // namespace

WindVec windFromSpeedDirection(double speed, double to_deg, double vertical) {
    // 方位角按气象习惯度量，但语义是「风**吹向**哪里」：
    //   0°   = 吹向北（NED 的 +x）
    //   90°  = 吹向东（NED 的 +y）
    //   180° = 吹向南（NED 的 −x）
    // 之所以显式写成「吹向」，是因为「北风」这类气象术语指的是**来向**，
    // 两者差 180°，是建模时最容易搞反的地方。
    const double rad = to_deg * kPi / 180.0;
    return {speed * std::cos(rad), speed * std::sin(rad), vertical};
}

std::string SteadyWind::name() const {
    const double sp = std::sqrt(_v[0] * _v[0] + _v[1] * _v[1]);
    std::ostringstream os;
    os << "常值风 " << fmt(sp) << " m/s";
    if (std::fabs(_v[2]) > 1e-9) {
        os << "（垂直 " << fmt(_v[2]) << " m/s）";
    }
    return os.str();
}

// ---------------------------------------------------------------------------
// 阵风
// ---------------------------------------------------------------------------

GustWind::GustWind(double steady, double steady_to_deg, double gust_amp,
                   double gust_to_deg, double start, double duration)
    : _steady(windFromSpeedDirection(steady, steady_to_deg)), _amp(gust_amp), _start(start),
      _duration(std::max(1e-3, duration)) {
    _gust_dir = windFromSpeedDirection(1.0, gust_to_deg);
}

double GustWind::gustFactor(double t) const {
    if (t <= _start || t >= _start + _duration) {
        return 0.0;
    }
    const double tau = (t - _start) / _duration; // 归一化到 [0,1]
    return 0.5 * (1.0 - std::cos(2.0 * kPi * tau));
}

WindVec GustWind::at(double t) {
    const double a = _amp * gustFactor(t);
    return {_steady[0] + _gust_dir[0] * a, _steady[1] + _gust_dir[1] * a,
            _steady[2] + _gust_dir[2] * a};
}

std::string GustWind::name() const {
    const double sp = std::sqrt(_steady[0] * _steady[0] + _steady[1] * _steady[1]);
    std::ostringstream os;
    os << "常值风 " << fmt(sp) << " m/s + 阵风 " << fmt(_amp) << " m/s（"
       << fmt(_start) << "s 起持续 " << fmt(_duration) << "s）";
    return os.str();
}

// ---------------------------------------------------------------------------
// 湍流
// ---------------------------------------------------------------------------

TurbulentWind::TurbulentWind(double steady, double steady_to_deg, double intensity,
                             double scale_length, double dt, std::uint32_t seed)
    : _steady(windFromSpeedDirection(steady, steady_to_deg)), _intensity(intensity),
      _scale_length(std::max(1e-3, scale_length)), _dt(std::max(1e-6, dt)), _seed(seed),
      _rng(seed) {
    // Dryden 成形滤波器的时间常数 τ = L / V。这里的 V 取背景风速而不是飞行速度：
    // 悬停时飞行速度为零，若直接代入会让 τ 发散。取 1 m/s 作为下限，对应
    // τ = L 秒（L 为尺度长度），与「低速悬停时湍流时间尺度约等于尺度长度」的
    // 物理直觉一致。
    const double v_ref = std::max(1.0, std::sqrt(_steady[0] * _steady[0] +
                                                 _steady[1] * _steady[1] +
                                                 _steady[2] * _steady[2]));
    const double tau = _scale_length / v_ref;
    _alpha = _dt / (tau + _dt);

    // 一阶低通对白噪声输入的输出方差是 a/(2−a) 倍的输入方差，反推输入幅度，
    // 使**输出**标准差恰好等于设定的湍流强度 σ。
    // 不做这一步的话，σ 就只是个没有物理含义的旋钮，调参只能靠试。
    _sigma_in = _intensity * std::sqrt((2.0 - _alpha) / _alpha);
}

void TurbulentWind::reset() {
    _rng.seed(_seed);
    _state = {0.0, 0.0, 0.0};
    _last_t = 0.0;
}

void TurbulentWind::advance() {
    for (int i = 0; i < 3; ++i) {
        const auto idx = static_cast<std::size_t>(i);
        const double u = _sigma_in * _unit(_rng);
        _state[idx] += _alpha * (u - _state[idx]);
    }
}

WindVec TurbulentWind::step() {
    advance();
    _last_t += _dt;
    return {_steady[0] + _state[0], _steady[1] + _state[1], _steady[2] + _state[2]};
}

WindVec TurbulentWind::at(double t) {
    // 按时间推进而非按调用次数推进：仿真器可能在同一时刻多次查询风速
    // （例如 RK4 的中间级），按调用次数推进会凭空多走滤波器状态。
    constexpr long long kMaxAdvance = 200000; // 防止 t 跳跃导致长时间循环
    const long long n = static_cast<long long>((t - _last_t) / _dt + 1e-9);
    const long long steps = std::max(0LL, std::min(n, kMaxAdvance));
    for (long long i = 0; i < steps; ++i) {
        advance();
    }
    _last_t += static_cast<double>(n) * _dt;

    return {_steady[0] + _state[0], _steady[1] + _state[1], _steady[2] + _state[2]};
}

std::string TurbulentWind::name() const {
    const double sp = std::sqrt(_steady[0] * _steady[0] + _steady[1] * _steady[1]);
    std::ostringstream os;
    os << "常值风 " << fmt(sp) << " m/s + Dryden 湍流 σ=" << fmt(_intensity)
       << " m/s（尺度 " << fmt(_scale_length) << " m）";
    return os.str();
}

} // namespace oi3
