/**
 * @file ImuModel.cpp
 * @brief IMU 误差模型实现
 * @author GhostFace
 * @date 2026/9/18
 */

#include "ImuModel.h"

#include "SixDofDynamics.h"
#include "TensorUtils.h"

#include <algorithm>
#include <cmath>

namespace oi3 {

namespace {

/// 把 NED 系向量旋转到机体坐标系（与 SixDofDynamics 的 rotateNedToBody 同口径）
std::array<double, 3> nedToBody(const Tensor &quat, const std::array<double, 3> &v) {
    const std::vector<float> q = toVector(quat);
    const double w = q[0], x = q[1], y = q[2], z = q[3];

    // 旋转矩阵的转置（机体 ← NED）
    const double r00 = 1.0 - 2.0 * (y * y + z * z);
    const double r01 = 2.0 * (x * y - w * z);
    const double r02 = 2.0 * (x * z + w * y);
    const double r10 = 2.0 * (x * y + w * z);
    const double r11 = 1.0 - 2.0 * (x * x + z * z);
    const double r12 = 2.0 * (y * z - w * x);
    const double r20 = 2.0 * (x * z - w * y);
    const double r21 = 2.0 * (y * z + w * x);
    const double r22 = 1.0 - 2.0 * (x * x + y * y);

    // R^T · v
    return {r00 * v[0] + r10 * v[1] + r20 * v[2],
            r01 * v[0] + r11 * v[1] + r21 * v[2],
            r02 * v[0] + r12 * v[1] + r22 * v[2]};
}

} // namespace

ImuModel::ImuModel(ImuConfig cfg, std::uint32_t seed) : _cfg(cfg), _rng(seed) {
    sampleBiases();
}

void ImuModel::reset() {
    _accel_lpf_state = {0.0, 0.0, 0.0};
    _lpf_initialized = false;
    sampleBiases();
}

void ImuModel::sampleBiases() {
    if (!_cfg.enabled) {
        _accel_bias = {0.0, 0.0, 0.0};
        _gyro_bias = {0.0, 0.0, 0.0};
        return;
    }
    if (_cfg.explicit_bias) {
        _accel_bias = _cfg.accel_bias_vec;
        _gyro_bias = _cfg.gyro_bias_vec;
        return;
    }
    // 出厂标定后的残余偏置：一次采样后固定
    for (int i = 0; i < 3; ++i) {
        _accel_bias[static_cast<std::size_t>(i)] = _unit(_rng) * _cfg.accel_bias;
        _gyro_bias[static_cast<std::size_t>(i)] = _unit(_rng) * _cfg.gyro_bias;
    }
}

ImuSample ImuModel::measure(const SixDofState &truth,
                            const std::array<double, 3> &specific_force_ned, double dt) {
    ImuSample out;

    // 真值：把 NED 系的比力转到机体坐标系（加速度计装在机体上）
    const std::array<double, 3> f_body = nedToBody(truth.quat, specific_force_ned);
    const std::vector<float> w = toVector(truth.omega);

    out.accel = f_body;
    out.gyro = {w[0], w[1], w[2]};

    if (!_cfg.enabled) {
        return out;
    }

    // 陀螺偏置随机游走：Bias 随时间的扩散，纯积分姿态会因此持续漂移
    const double walk = _cfg.gyro_bias_walk * std::sqrt(std::max(0.0, dt));
    for (int i = 0; i < 3; ++i) {
        _gyro_bias[static_cast<std::size_t>(i)] += _unit(_rng) * walk;
    }

    // 加偏置与白噪声
    for (int i = 0; i < 3; ++i) {
        const auto idx = static_cast<std::size_t>(i);
        out.accel[idx] += _accel_bias[idx] + _unit(_rng) * _cfg.accel_noise;
        out.gyro[idx] += _gyro_bias[idx] + _unit(_rng) * _cfg.gyro_noise;
    }

    // 加速度计低通（一阶），模拟器件内部的抗混叠滤波带来的相位滞后
    if (_cfg.accel_lpf_hz > 0.0 && dt > 0.0) {
        const double rc = 1.0 / (2.0 * M_PI * _cfg.accel_lpf_hz);
        const double alpha = dt / (rc + dt);
        if (!_lpf_initialized) {
            _accel_lpf_state = out.accel;
            _lpf_initialized = true;
        } else {
            for (int i = 0; i < 3; ++i) {
                const auto idx = static_cast<std::size_t>(i);
                _accel_lpf_state[idx] += alpha * (out.accel[idx] - _accel_lpf_state[idx]);
            }
        }
        out.accel = _accel_lpf_state;
    }

    return out;
}

} // namespace oi3
