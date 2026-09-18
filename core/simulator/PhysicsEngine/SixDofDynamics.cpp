/**
 * @file SixDofDynamics.cpp
 * @brief 六自由度飞行器动力学实现
 * @author GhostFace
 * @date 2026/9/17
 *
 * 姿态相关的量按头文件说明用标量计算后构造常量张量，不保证可微。
 */

#include "SixDofDynamics.h"
#include "TensorUtils.h"

#include <cmath>

namespace oi3 {

namespace {

struct Vec3d {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

struct Quatd {
    double w = 1.0;
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

Vec3d readVec3(const Tensor &t) {
    const float *p = t.data<float>();
    return {p[0], p[1], p[2]};
}

Quatd readQuat(const Tensor &t) {
    const float *p = t.data<float>();
    return {p[0], p[1], p[2], p[3]};
}

/// 由单位四元数构造机体系 -> NED 的旋转矩阵
void rotationMatrix(const Quatd &q, double r[3][3]) {
    const double w = q.w, x = q.x, y = q.y, z = q.z;
    r[0][0] = 1.0 - 2.0 * (y * y + z * z);
    r[0][1] = 2.0 * (x * y - w * z);
    r[0][2] = 2.0 * (x * z + w * y);

    r[1][0] = 2.0 * (x * y + w * z);
    r[1][1] = 1.0 - 2.0 * (x * x + z * z);
    r[1][2] = 2.0 * (y * z - w * x);

    r[2][0] = 2.0 * (x * z - w * y);
    r[2][1] = 2.0 * (y * z + w * x);
    r[2][2] = 1.0 - 2.0 * (x * x + y * y);
}

Quatd conjugate(const Quatd &q) {
    return {q.w, -q.x, -q.y, -q.z};
}

} // namespace

Tensor normalizeQuat(const Tensor &quat) {
    const Quatd q = readQuat(quat);
    const double n = std::sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
    const double inv = (n > 1e-12) ? (1.0 / n) : 1.0;
    return Tensor{static_cast<float>(q.w * inv), static_cast<float>(q.x * inv),
                  static_cast<float>(q.y * inv), static_cast<float>(q.z * inv)};
}

Tensor rotateBodyToNed(const Tensor &quat, const Tensor &v_body) {
    const Quatd q = readQuat(quat);
    const Vec3d v = readVec3(v_body);

    double r[3][3];
    rotationMatrix(q, r);

    return makeVec3(static_cast<float>(r[0][0] * v.x + r[0][1] * v.y + r[0][2] * v.z),
                    static_cast<float>(r[1][0] * v.x + r[1][1] * v.y + r[1][2] * v.z),
                    static_cast<float>(r[2][0] * v.x + r[2][1] * v.y + r[2][2] * v.z));
}

Tensor rotateNedToBody(const Tensor &quat, const Tensor &v_ned) {
    const Quatd q = readQuat(quat);
    const Vec3d v = readVec3(v_ned);

    double r[3][3];
    rotationMatrix(q, r);

    // 旋转矩阵的转置即逆（正交矩阵）
    return makeVec3(static_cast<float>(r[0][0] * v.x + r[1][0] * v.y + r[2][0] * v.z),
                    static_cast<float>(r[0][1] * v.x + r[1][1] * v.y + r[2][1] * v.z),
                    static_cast<float>(r[0][2] * v.x + r[1][2] * v.y + r[2][2] * v.z));
}

Tensor quatToEuler(const Tensor &quat) {
    const Quatd q = readQuat(quat);

    // Z-Y-X（yaw-pitch-roll）
    const double sinr_cosp = 2.0 * (q.w * q.x + q.y * q.z);
    const double cosr_cosp = 1.0 - 2.0 * (q.x * q.x + q.y * q.y);
    const double roll = std::atan2(sinr_cosp, cosr_cosp);

    const double sinp = 2.0 * (q.w * q.y - q.z * q.x);
    const double pitch = (std::fabs(sinp) >= 1.0)
                             ? std::copysign(M_PI / 2.0, sinp)
                             : std::asin(sinp);

    const double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
    const double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
    const double yaw = std::atan2(siny_cosp, cosy_cosp);

    return makeVec3(static_cast<float>(roll), static_cast<float>(pitch),
                    static_cast<float>(yaw));
}

Tensor sixDofAcceleration(const Tensor &vel, const Tensor &quat, double thrust_body,
                          const Config &cfg, const Tensor *v_wind) {
    const double m = cfg.mass;
    const double g = cfg.gravity;

    // 重力（NED 系向下为正）
    const Tensor f_gravity = makeVec3(0.0f, 0.0f, static_cast<float>(m * g));

    // 推力沿机体 -z：机体 z 轴朝上，故正向推力对应机体系的 -z 分量。
    // 姿态水平时旋转矩阵为单位阵，推力为 [0,0,-T]，恰好抵消重力。
    const Tensor f_thrust_body = makeVec3(0.0f, 0.0f, static_cast<float>(-thrust_body));
    const Tensor f_thrust_ned = rotateBodyToNed(quat, f_thrust_body);

    Tensor f_total = f_gravity + f_thrust_ned;

    // 气动阻力（NED 系，逐轴二次形式，与三自由度版本口径一致）。
    // 有风时按**相对气流**计算：v_rel = v − v_wind。无风时 v_wind 为空，
    // 表达式与改造前逐字相同，因此所有既有结果逐位不变。
    if (cfg.drag_coeff > 0.0) {
        Tensor v_rel = vel;
        if (v_wind != nullptr) {
            v_rel = vel - *v_wind;
        }
        const Tensor abs_vel = v_rel.abs();
        f_total = f_total + (abs_vel * v_rel) * static_cast<float>(-cfg.drag_coeff);
    }

    return f_total / static_cast<float>(m);
}

Tensor angularAcceleration(const Tensor &omega, const Tensor &torque,
                           const SixDofConfig &cfg) {
    const Vec3d w = readVec3(omega);
    const Vec3d t = readVec3(torque);

    const double ix = cfg.inertia[0];
    const double iy = cfg.inertia[1];
    const double iz = cfg.inertia[2];

    // 陀螺项 ω × (Iω)
    const double iw[3] = {ix * w.x, iy * w.y, iz * w.z};
    const double cx = w.y * iw[2] - w.z * iw[1];
    const double cy = w.z * iw[0] - w.x * iw[2];
    const double cz = w.x * iw[1] - w.y * iw[0];

    // 气动转动阻尼 τ = −k·ω。默认 0 时下面三项与改造前逐字相同，
    // 因此既有结果逐位不变。
    const double kd = cfg.rot_damping;
    const double dx = kd * w.x;
    const double dy = kd * w.y;
    const double dz = kd * w.z;

    return makeVec3(static_cast<float>((t.x - cx - dx) / ix),
                    static_cast<float>((t.y - cy - dy) / iy),
                    static_cast<float>((t.z - cz - dz) / iz));
}

Tensor quatDerivative(const Tensor &quat, const Tensor &omega) {
    const Quatd q = readQuat(quat);
    const Vec3d w = readVec3(omega);

    // q̇ = ½ · q ⊗ [0, ω]
    // (w1,v1) ⊗ (0,ω) = (−v1·ω, w1·ω + v1×ω)
    const double dw = -(q.x * w.x + q.y * w.y + q.z * w.z);
    const double dx = q.w * w.x + (q.y * w.z - q.z * w.y);
    const double dy = q.w * w.y + (q.z * w.x - q.x * w.z);
    const double dz = q.w * w.z + (q.x * w.y - q.y * w.x);

    const float h = 0.5f;
    return Tensor{static_cast<float>(h * dw), static_cast<float>(h * dx),
                  static_cast<float>(h * dy), static_cast<float>(h * dz)};
}

SixDofState rk4StepSixDof(const SixDofState &y, double thrust_body, const Tensor &torque,
                          const SixDofConfig &cfg, double dt, const Tensor *v_wind) {
    const float h = static_cast<float>(dt);
    const float half = h * 0.5f;
    const float sixth = h / 6.0f;

    const auto derivative = [&](const SixDofState &s) -> SixDofState {
        return SixDofState{
            s.vel,                                                    // dpos/dt = vel
            sixDofAcceleration(s.vel, s.quat, thrust_body, cfg.base, v_wind), // dvel/dt
            quatDerivative(s.quat, s.omega),                           // dquat/dt
            angularAcceleration(s.omega, torque, cfg)};                // domega/dt
    };

    const SixDofState k1 = derivative(y);
    const SixDofState k2 = derivative(y + k1 * half);
    const SixDofState k3 = derivative(y + k2 * half);
    const SixDofState k4 = derivative(y + k3 * h);

    SixDofState out = y + (k1 + k2 * 2.0f + k3 * 2.0f + k4) * sixth;

    // 四元数积分会缓慢偏离单位模长，每步归一化抑制漂移
    out.quat = normalizeQuat(out.quat);
    return out;
}

} // namespace oi3
