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

StateEstimator::StateEstimator(EstimatorConfig cfg) : _cfg(cfg), _health(cfg.sensor_health) {}

void StateEstimator::reset(const std::array<double, 4> &initial_quat) {
    _quat = initial_quat;
    normalize(_quat);
    _omega = {0.0, 0.0, 0.0};
    _gyro_bias = {0.0, 0.0, 0.0};
    _pos = {0.0, 0.0, 0.0};
    _vel = {0.0, 0.0, 0.0};
    _has_pos = false;
    // 健康监测随估计器一同复位：换场地或重启后应重新积累统计，
    // 否则上一段数据的故障判定会残留到新数据上。
    _health.reset();

    // 卡尔曼协方差复位。若沿用上一次运行的收敛协方差，重启后滤波器会
    // 过度自信，对最初几帧量测几乎不做修正 —— 故在此清零，由首次量测重建。
    for (auto &cov : _pos_cov) {
        cov = AxisCovariance{};
    }
    _k_pos = {0.0, 0.0, 0.0};
    _k_vel = {0.0, 0.0, 0.0};
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
    // 只要加速度计有方向信息，就独立计算方向残差。这里刻意把「检测」与
    // 「校正」拆开：降级策略可以关闭校正，但不能因此让健康监测也停止，
    // 否则检测到 AccelBias 后，下一拍就失去继续监测的能力。
    if (an > 1e-6) {
        const std::array<double, 3> f_meas = {imu.accel[0] / an, imu.accel[1] / an,
                                              imu.accel[2] / an};

        // 估计的比力方向：静止时加速度计读数为「支撑力」，即 NED 系下的 [0,0,-1]·g，
        // 归一化后就是 [0,0,-1]（向上），旋转到机体得到 f_est。
        const std::array<double, 3> f_est = rotateNedToBody(_quat, {0.0, 0.0, -1.0});

        // 误差：叉积给出修正旋转轴（f_meas × f_est）。
        const std::array<double, 3> e = {
            f_meas[1] * f_est[2] - f_meas[2] * f_est[1],
            f_meas[2] * f_est[0] - f_meas[0] * f_est[2],
            f_meas[0] * f_est[1] - f_meas[1] * f_est[0]};

        // 方向残差是健康检测的观测量，必须无条件更新（只受 enabled 控制），
        // 与是否将它反馈进姿态积分完全独立。
        if (_cfg.sensor_health.enabled) {
            const double resid = std::sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
            _health.update(imu.accel, imu.gyro, resid);
        }

        // 加速度计方向校正。降级策略可关闭它——失效数据会让这里把错误的
        // 姿态误差持续注入估计，比不校正更糟。关闭校正不影响上面的检测。
        if (_use_accel_correction) {
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
                    // 调节——想靠降低 Kp 抑制机动污染时，偏置收敛会跟着一起变慢。
                    _gyro_bias[idx] -= _cfg.bias_correction * e[idx] * dt;
                }
            }
        }
    }

    // 加速度计归零（an <= 1e-6）时上面的分支不进入，此处补一次 update，
    // 残差按 0 计。这一补很关键：模长判据是检测「彻底掉线」的唯一手段，
    // 若此时不调用 update，该判据永远没有执行机会 —— 第一版实现就是漏了
    // 这一处，导致加速度计归零被完全漏报（测试第 2 节捕获）。
    // 掉线由 accel_mag_min 判据命中，不依赖残差，故 resid=0 不影响判定。
    if (_cfg.sensor_health.enabled && an <= 1e-6) {
        _health.update(imu.accel, imu.gyro, 0.0);
    }

    _omega = w;

    // ---- 3. 积分姿态 ----
    integrateQuat(_quat, w, dt);

    // ---- 4. 位置/速度预测（IMU 预积分）----
    // 加速度计测的是比力，加回重力才得到惯性加速度：a_ned = R(q)·f_body + g_vec。
    // NED 系下 g_vec = [0, 0, +g]。这一步用上了高频 IMU 携带的运动信息，
    // 位置测量只需要做校正，不必再靠差分去「估」速度。
    if (_has_pos && _trust_position) {
        const std::array<double, 3> a_ned = rotateBodyToNed(_quat, imu.accel);
        for (int i = 0; i < 3; ++i) {
            const auto idx = static_cast<std::size_t>(i);
            const double a = a_ned[idx] + (i == 2 ? _cfg.gravity : 0.0);
            _pos[idx] += _vel[idx] * dt + 0.5 * a * dt * dt;
            _vel[idx] += a * dt;
        }

        // 卡尔曼协方差预测：P ← F·P·Fᵀ + Q，其中 F = [[1, dt], [0, 1]]。
        // 展开后（利用 P 对称，只需维护三个独立分量）：
        //   p_pp ← p_pp + 2·dt·p_pv + dt²·p_vv + Q_pp
        //   p_pv ← p_pv + dt·p_vv             + Q_pv
        //   p_vv ← p_vv                        + Q_vv
        // 过程噪声 Q 由 IMU 加速度不确定度 q 驱动（连续白噪声加速度模型）：
        //   Q_pp = q²·dt⁴/4，Q_pv = q²·dt³/2，Q_vv = q²·dt²
        // 即「加速度噪声经两次积分进入位置」。q 取配置值与量测噪声的相对关系
        // 决定了滤波器更信任模型还是更信任量测。
        if (_cfg.pos_filter == PosFilterKind::Kalman) {
            const double q = _cfg.kalman_accel_noise;
            const double dt2 = dt * dt;
            const double dt3 = dt2 * dt;
            const double dt4 = dt2 * dt2;
            const double q_pp = q * q * dt4 * 0.25;
            const double q_pv = q * q * dt3 * 0.5;
            const double q_vv = q * q * dt2;
            for (auto &cov : _pos_cov) {
                const double pp = cov.p_pp;
                const double pv = cov.p_pv;
                const double vv = cov.p_vv;
                cov.p_pp = pp + 2.0 * dt * pv + dt2 * vv + q_pp;
                cov.p_pv = pv + dt * vv + q_pv;
                cov.p_vv = vv + q_vv;
            }
        }
    }
}

void StateEstimator::updatePosition(const std::array<double, 3> &pos_meas, double dt) {
    if (!_has_pos) {
        _pos = pos_meas;
        _vel = {0.0, 0.0, 0.0};
        _has_pos = true;

        // 卡尔曼初值：位置不确定度取量测噪声量级（首帧量测的可信度即如此），
        // 速度不确定度取 1 m/s（悬停启动时速度未知，但不至于完全无界）。
        // 不建这个初值而让协方差保持为零，滤波器会「过度自信」——
        // 新息几乎不产生修正，估计器将长期滞后于真实状态。
        if (_cfg.pos_filter == PosFilterKind::Kalman) {
            const double R = _cfg.kalman_pos_noise * _cfg.kalman_pos_noise;
            for (auto &cov : _pos_cov) {
                cov.p_pp = R;
                cov.p_pv = 0.0;
                cov.p_vv = 1.0;
            }
        }
        return;
    }
    if (dt <= 1e-9) {
        return;
    }

    // ---- 卡尔曼分支（可选升级路径）----
    // 标准 2×2 卡尔曼更新：状态 [p, v]，量测 p。
    //   新息    y = z − p
    //   新息方差 S = p_pp + R
    //   增益    K = [p_pp/S, p_pv/S]ᵀ
    //   协方差  P ← (I − K·H)·P
    // 与 α-β 的关键区别：增益不是常数，而由协方差与量测噪声 R 实时算出。
    // 量测噪声增大时 S 增大、增益自动下降，无需为不同传感器重新标定增益，
    // 也不需要死区补丁 —— 这正是升级的动机。
    if (_cfg.pos_filter == PosFilterKind::Kalman) {
        const double R = _cfg.kalman_pos_noise * _cfg.kalman_pos_noise;
        for (int i = 0; i < 3; ++i) {
            const auto idx = static_cast<std::size_t>(i);
            auto &cov = _pos_cov[idx];
            const double S = cov.p_pp + R;
            // S 理论上恒为正（p_pp ≥ 0、R > 0）；此处仍做保护，
            // 避免 R 被配置为 0 时出现除零。
            if (S <= 1e-15) {
                continue;
            }
            const double k_p = cov.p_pp / S;
            const double k_v = cov.p_pv / S;
            const double y = pos_meas[idx] - _pos[idx];
            _pos[idx] += k_p * y;
            _vel[idx] += k_v * y;

            const double pp = cov.p_pp;
            const double pv = cov.p_pv;
            const double vv = cov.p_vv;
            cov.p_pp = (1.0 - k_p) * pp;
            cov.p_pv = (1.0 - k_p) * pv;
            cov.p_vv = vv - k_v * pv;

            _k_pos[idx] = k_p;
            _k_vel[idx] = k_v;
        }
        return;
    }

    // 校正步：预测由 updateImu 每周期推进，这里只用位置残差修正位置与速度。
    // 速度由残差驱动，不含差分带来的 1/dt 噪声放大。
    const double a = std::max(1e-6, std::min(1.0, _cfg.pos_filter_alpha));
    const double b = a * a / (2.0 - a); // 临界阻尼约束 β = α²/(2−α)
    const double dz = std::max(0.0, _cfg.pos_residual_deadzone);

    for (int i = 0; i < 3; ++i) {
        const auto idx = static_cast<std::size_t>(i);
        const double r = pos_meas[idx] - _pos[idx];
        _pos[idx] += a * r;

        // 死区仅作用于速度校正，位置校正仍用完整残差。
        // 这样既抑制大噪声脉冲灌入速度，又不牺牲位置估计的响应速度。
        double r_vel = r;
        if (dz > 0.0) {
            if (r_vel > dz) {
                r_vel -= dz;
            } else if (r_vel < -dz) {
                r_vel += dz;
            } else {
                r_vel = 0.0;
            }
        }
        _vel[idx] += (b / dt) * r_vel;
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
