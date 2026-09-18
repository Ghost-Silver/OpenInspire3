/**
 * @file StateEstimator.cpp
 * @brief 状态估计实现
 * @author GhostFace
 * @date 2026/9/18
 *
 * 姿态部分用互补滤波（陀螺积分 + 加速度计方向修正 + 偏置隐式估计），
 * 位置/速度为显式简化（一阶低通 + 差分）。全部用标量计算：估计器不参与求导，
 * 且处在 1 kHz 热路径上，标量实现既快又便于用解析解核对。
 */

#include "StateEstimator.h"

#include "TensorUtils.h"

#include <algorithm>
#include <cmath>

namespace oi3 {

namespace {

/// 四元数归一化
void normalize(std::array<double, 4> &q) {
    const double n = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    if (n > 1e-12) {
        for (double &v : q) {
            v /= n;
        }
    } else {
        q = {1.0, 0.0, 0.0, 0.0};
    }
}

/// 由四元数把 NED 系向量转到机体坐标系（R^T · v）
std::array<double, 3> rotateNedToBody(const std::array<double, 4> &q,
                                      const std::array<double, 3> &v) {
    const double w = q[0], x = q[1], y = q[2], z = q[3];
    const double r00 = 1.0 - 2.0 * (y * y + z * z);
    const double r01 = 2.0 * (x * y - w * z);
    const double r02 = 2.0 * (x * z + w * y);
    const double r10 = 2.0 * (x * y + w * z);
    const double r11 = 1.0 - 2.0 * (x * x + z * z);
    const double r12 = 2.0 * (y * z - w * x);
    const double r20 = 2.0 * (x * z - w * y);
    const double r21 = 2.0 * (y * z + w * x);
    const double r22 = 1.0 - 2.0 * (x * x + y * y);

    return {r00 * v[0] + r10 * v[1] + r20 * v[2],
            r01 * v[0] + r11 * v[1] + r21 * v[2],
            r02 * v[0] + r12 * v[1] + r22 * v[2]};
}

/// 由四元数把机体向量转到 NED 系（R · v）
std::array<double, 3> rotateBodyToNed(const std::array<double, 4> &q,
                                      const std::array<double, 3> &v) {
    const double w = q[0], x = q[1], y = q[2], z = q[3];
    const double r00 = 1.0 - 2.0 * (y * y + z * z);
    const double r01 = 2.0 * (x * y - w * z);
    const double r02 = 2.0 * (x * z + w * y);
    const double r10 = 2.0 * (x * y + w * z);
    const double r11 = 1.0 - 2.0 * (x * x + z * z);
    const double r12 = 2.0 * (y * z - w * x);
    const double r20 = 2.0 * (x * z - w * y);
    const double r21 = 2.0 * (y * z + w * x);
    const double r22 = 1.0 - 2.0 * (x * x + y * y);

    return {r00 * v[0] + r01 * v[1] + r02 * v[2],
            r10 * v[0] + r11 * v[1] + r12 * v[2],
            r20 * v[0] + r21 * v[1] + r22 * v[2]};
}

/// 四元数按角速度积分：q ← q + ½·(q ⊗ [0,ω])·dt
void integrateQuat(std::array<double, 4> &q, const std::array<double, 3> &w, double dt) {
    const double qw = q[0], qx = q[1], qy = q[2], qz = q[3];
    const double wx = w[0], wy = w[1], wz = w[2];

    // q ⊗ [0, ω]
    const double dw = -(qx * wx + qy * wy + qz * wz);
    const double dx = qw * wx + (qy * wz - qz * wy);
    const double dy = qw * wy + (qz * wx - qx * wz);
    const double dz = qw * wz + (qx * wy - qy * wx);

    const double h = 0.5 * dt;
    q[0] += h * dw;
    q[1] += h * dx;
    q[2] += h * dy;
    q[3] += h * dz;
    normalize(q);
}

} // namespace

StateEstimator::StateEstimator(EstimatorConfig cfg) : _cfg(cfg) {}

void StateEstimator::reset(const std::array<double, 4> &initial_quat) {
    _quat = initial_quat;
    normalize(_quat);
    _omega = {0.0, 0.0, 0.0};
    _gyro_bias = {0.0, 0.0, 0.0};
    _pos = {0.0, 0.0, 0.0};
    _vel = {0.0, 0.0, 0.0};
    _has_pos = false;
}

void StateEstimator::setAttitude(const std::array<double, 4> &quat) {
    _quat = quat;
    normalize(_quat);
}

void StateEstimator::updateImu(const ImuSample &imu, double dt) {
    if (dt <= 0.0) {
        return;
    }

    // ---- 1. 去偏置 ----
    std::array<double, 3> w{};
    for (int i = 0; i < 3; ++i) {
        const auto idx = static_cast<std::size_t>(i);
        w[idx] = imu.gyro[idx] - _gyro_bias[idx];
    }

    // ---- 2. 加速度计方向校正 ----
    // 归一化的测量比力方向（机体坐标系）
    double an = 0.0;
    for (double a : imu.accel) {
        an += a * a;
    }
    an = std::sqrt(an);
    if (an > 1e-6) {
        const std::array<double, 3> f_meas = {imu.accel[0] / an, imu.accel[1] / an,
                                              imu.accel[2] / an};

        // 估计的比力方向：静止时加速度计读数为「支撑力」，即 NED 系下的 [0,0,-1]·g，
        // 归一化后就是 [0,0,-1]（向上），旋转到机体得到 f_est
        const std::array<double, 3> f_est = rotateNedToBody(_quat, {0.0, 0.0, -1.0});

        // 误差：叉积给出修正旋转轴（f_meas × f_est）
        const std::array<double, 3> e = {
            f_meas[1] * f_est[2] - f_meas[2] * f_est[1],
            f_meas[2] * f_est[0] - f_meas[0] * f_est[2],
            f_meas[0] * f_est[1] - f_meas[1] * f_est[0]};

        // 修正角速度；同时把修正量的低通作为陀螺偏置估计 ——
        // 稳态下这项恰好抵消偏置造成的漂移，因此它本身就是偏置的观测量。
        //
        // 符号必须取负：设真实偏置 b > 0（测得角速度偏大），姿态估计会超前真值，
        // 记姿态误差为 δ，则叉积误差 e ≈ −δ。要让 bias_est 追向 +b，就必须让
        // bias_est 沿 −e 的方向增长。写成 += 会让估计值朝真值的反方向走，
        // 表现为「偏置估计不收敛且符号相反」—— 这个错误由 EstimatorTest 的分轴
        // 断言捕获，不是靠读代码看出来的。
        for (int i = 0; i < 3; ++i) {
            const auto idx = static_cast<std::size_t>(i);
            const double corr = _cfg.accel_correction * e[idx];
            w[idx] += corr;
            if (_cfg.estimate_gyro_bias) {
                // 标准 Mahony 形式：ḃ = −Ki·e，用未经 Kp 缩放的原始误差。
                // 若把 Kp·e 代进来，Ki 的有效值就变成 Ki·Kp，两个增益不再可独立
                // 调节 —— 想靠降低 Kp 抑制机动污染时，偏置收敛会跟着一起变慢。
                _gyro_bias[idx] -= _cfg.bias_correction * e[idx] * dt;
            }
        }
    }

    _omega = w;

    // ---- 3. 积分姿态 ----
    integrateQuat(_quat, w, dt);

    // ---- 4. 位置/速度预测（IMU 预积分）----
    // 加速度计测的是比力，加回重力才得到惯性加速度：a_ned = R(q)·f_body + g_vec。
    // NED 系下 g_vec = [0, 0, +g]。这一步用上了高频 IMU 携带的运动信息，
    // 位置测量只需要做校正，不必再靠差分去「估」速度。
    if (_has_pos) {
        const std::array<double, 3> a_ned = rotateBodyToNed(_quat, imu.accel);
        for (int i = 0; i < 3; ++i) {
            const auto idx = static_cast<std::size_t>(i);
            const double a = a_ned[idx] + (i == 2 ? _cfg.gravity : 0.0);
            _pos[idx] += _vel[idx] * dt + 0.5 * a * dt * dt;
            _vel[idx] += a * dt;
        }
    }
}

void StateEstimator::updatePosition(const std::array<double, 3> &pos_meas, double dt) {
    if (!_has_pos) {
        _pos = pos_meas;
        _vel = {0.0, 0.0, 0.0};
        _has_pos = true;
        return;
    }
    if (dt <= 1e-9) {
        return;
    }

    // 校正步：预测由 updateImu 每周期推进，这里只用位置残差修正位置与速度。
    // 速度由残差驱动，不含差分带来的 1/dt 噪声放大。
    const double a = std::max(1e-6, std::min(1.0, _cfg.pos_filter_alpha));
    const double b = a * a / (2.0 - a); // 临界阻尼约束 β = α²/(2−α)

    for (int i = 0; i < 3; ++i) {
        const auto idx = static_cast<std::size_t>(i);
        const double r = pos_meas[idx] - _pos[idx];
        _pos[idx] += a * r;
        _vel[idx] += (b / dt) * r;
    }
}

SixDofState StateEstimator::state() const {
    Tensor pos(ShapeTag{}, {3});
    Tensor vel(ShapeTag{}, {3});
    Tensor quat(ShapeTag{}, {4});
    Tensor omega(ShapeTag{}, {3});

    for (int i = 0; i < 3; ++i) {
        const auto idx = static_cast<std::size_t>(i);
        pos.data_write<float>()[i] = static_cast<float>(_pos[idx]);
        vel.data_write<float>()[i] = static_cast<float>(_vel[idx]);
        omega.data_write<float>()[i] = static_cast<float>(_omega[idx]);
    }
    for (int i = 0; i < 4; ++i) {
        quat.data_write<float>()[i] = static_cast<float>(_quat[static_cast<std::size_t>(i)]);
    }

    return SixDofState{pos, vel, quat, omega};
}

} // namespace oi3
