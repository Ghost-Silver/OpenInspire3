/**
 * @file SixDofDynamicsDiff.cpp
 * @brief 六自由度动力学的可微张量实现
 * @author GhostFace
 * @date 2026/9/17
 *
 * 全部姿态量都用 CTorch 张量算子表达，因此前向过程即建图过程，梯度可穿回初值与
 * 控制量。方程与 SixDofDynamics.cpp 逐项对应，便于测试做逐步数值对照。
 */

#include "SixDofDynamicsDiff.h"

#include "AutoGrad.h"

#include <cmath>
#include <cstddef>

namespace oi3 {

namespace {

/// 取第 i 个分量的 {1} 视图（可微：slice 已注册反向节点）
Tensor comp(const Tensor &t, std::size_t i) { return t.slice(0, i, 1); }

/// 构造 {1} 常量张量（不参与建图，作为常数项使用）
Tensor scalarConst(float v) {
    Tensor t(ShapeTag{}, {1});
    t.data_write<float>()[0] = v;
    return t;
}

/// 逐项矩阵-向量乘法：out_k = Σ_j R[k][j] · v_j
Tensor applyMatrix(const std::array<Tensor, 9> &r, const Tensor &v) {
    const Tensor vx = comp(v, 0);
    const Tensor vy = comp(v, 1);
    const Tensor vz = comp(v, 2);
    return stackVec3(r[0] * vx + r[1] * vy + r[2] * vz,
                     r[3] * vx + r[4] * vy + r[5] * vz,
                     r[6] * vx + r[7] * vy + r[8] * vz);
}

} // namespace

Tensor stackVec3(const Tensor &x, const Tensor &y, const Tensor &z) {
    return x.concat(y, 0).concat(z, 0);
}

Tensor stackVec4(const Tensor &w, const Tensor &x, const Tensor &y, const Tensor &z) {
    return w.concat(x, 0).concat(y, 0).concat(z, 0);
}

std::array<Tensor, 9> rotationMatrixEntries(const Tensor &quat) {
    const Tensor w = comp(quat, 0);
    const Tensor x = comp(quat, 1);
    const Tensor y = comp(quat, 2);
    const Tensor z = comp(quat, 3);

    const Tensor one = scalarConst(1.0f);
    const Tensor two = scalarConst(2.0f);

    // 与标量版 rotationMatrix 逐项一致（行主序）
    std::array<Tensor, 9> r;
    r[0] = one - (y * y + z * z) * two;
    r[1] = (x * y - w * z) * two;
    r[2] = (x * z + w * y) * two;

    r[3] = (x * y + w * z) * two;
    r[4] = one - (x * x + z * z) * two;
    r[5] = (y * z - w * x) * two;

    r[6] = (x * z - w * y) * two;
    r[7] = (y * z + w * x) * two;
    r[8] = one - (x * x + y * y) * two;
    return r;
}

Tensor rotationMatrixFromQuat(const Tensor &quat) {
    const std::array<Tensor, 9> r = rotationMatrixEntries(quat);
    Tensor flat = r[0];
    for (std::size_t i = 1; i < 9; ++i) {
        flat = flat.concat(r[i], 0);
    }
    // flat 是 {9} 的新分配连续张量，reshape 重算 strides 合法；
    // reshape 参与建图，梯度经 ReshapeNode 还原为输入形状。
    return flat.reshape({3, 3});
}

Tensor normalizeQuatDiff(const Tensor &quat) {
    // 逐分量求模长平方，不用 sum()：
    //   1. 与标量版公式逐项对应，便于数值对照与定位差异；
    //   2. **避开 0 维标量**。sum() 的结果是 0 维张量 `{}`（numel=1 但 strides
    //      为空），它参与 `{1}` 广播运算时结果不可靠 —— 实测在长时间积分中
    //      出现「第 0 个分量正常、其余分量变 NaN」的现象，即分母只吸收到部分
    //      分量的贡献。逐分量运算的每一步都是等形状的普通逐元素算子，没有
    //      0 维张量也没有广播。
    const Tensor qw = comp(quat, 0);
    const Tensor qx = comp(quat, 1);
    const Tensor qy = comp(quat, 2);
    const Tensor qz = comp(quat, 3);

    const Tensor norm_sq = qw * qw + qx * qx + qy * qy + qz * qz;
    const Tensor n = (norm_sq + scalarConst(1e-12f)).sqrt();

    // 同样按分量相除，避免 {4}/{1} 广播
    return stackVec4(qw / n, qx / n, qy / n, qz / n);
}

Tensor rotateBodyToNedDiff(const Tensor &quat, const Tensor &v_body) {
    return applyMatrix(rotationMatrixEntries(quat), v_body);
}

Tensor sixDofAccelerationDiff(const Tensor &vel, const Tensor &quat, const Tensor &thrust,
                              const Config &cfg) {
    const float m = static_cast<float>(cfg.mass);
    const float g = static_cast<float>(cfg.gravity);

    // 重力（NED 系向下为正）：常量项，不参与建图
    const Tensor f_gravity = makeVec3(0.0f, 0.0f, m * g);

    // 推力沿机体 -z：机体 z 轴朝上，故正向推力对应机体系 -z 分量
    const Tensor f_thrust_body =
        stackVec3(scalarConst(0.0f), scalarConst(0.0f), thrust * scalarConst(-1.0f));
    const Tensor f_thrust_ned = rotateBodyToNedDiff(quat, f_thrust_body);

    Tensor f_total = f_gravity + f_thrust_ned;

    // 气动阻力（NED 系，逐轴二次形式，与三自由度版本口径一致）
    if (cfg.drag_coeff > 0.0) {
        const Tensor abs_vel = vel.abs();
        f_total = f_total + (abs_vel * vel) * scalarConst(static_cast<float>(-cfg.drag_coeff));
    }

    return f_total / scalarConst(m);
}

Tensor angularAccelerationDiff(const Tensor &omega, const Tensor &torque,
                               const SixDofConfig &cfg) {
    const Tensor wx = comp(omega, 0);
    const Tensor wy = comp(omega, 1);
    const Tensor wz = comp(omega, 2);

    const Tensor tx = comp(torque, 0);
    const Tensor ty = comp(torque, 1);
    const Tensor tz = comp(torque, 2);

    const float ix = static_cast<float>(cfg.inertia[0]);
    const float iy = static_cast<float>(cfg.inertia[1]);
    const float iz = static_cast<float>(cfg.inertia[2]);

    // 陀螺项 ω × (Iω)
    const Tensor iw_x = wx * scalarConst(ix);
    const Tensor iw_y = wy * scalarConst(iy);
    const Tensor iw_z = wz * scalarConst(iz);

    const Tensor c_x = wy * iw_z - wz * iw_y;
    const Tensor c_y = wz * iw_x - wx * iw_z;
    const Tensor c_z = wx * iw_y - wy * iw_x;

    return stackVec3((tx - c_x) / scalarConst(ix), (ty - c_y) / scalarConst(iy),
                     (tz - c_z) / scalarConst(iz));
}

Tensor quatDerivativeDiff(const Tensor &quat, const Tensor &omega) {
    const Tensor qw = comp(quat, 0);
    const Tensor qx = comp(quat, 1);
    const Tensor qy = comp(quat, 2);
    const Tensor qz = comp(quat, 3);

    const Tensor wx = comp(omega, 0);
    const Tensor wy = comp(omega, 1);
    const Tensor wz = comp(omega, 2);

    // q̇ = ½ · q ⊗ [0, ω]
    const Tensor dw = -(qx * wx + qy * wy + qz * wz);
    const Tensor dx = qw * wx + (qy * wz - qz * wy);
    const Tensor dy = qw * wy + (qz * wx - qx * wz);
    const Tensor dz = qw * wz + (qx * wy - qy * wx);

    const Tensor half = scalarConst(0.5f);
    return stackVec4(dw * half, dx * half, dy * half, dz * half);
}

SixDofState rk4StepSixDofDiff(const SixDofState &y, const Tensor &thrust,
                              const Tensor &torque, const SixDofConfig &cfg, float dt) {
    const float h = dt;
    const float half = h * 0.5f;
    const float sixth = h / 6.0f;

    const auto derivative = [&](const SixDofState &s) -> SixDofState {
        return SixDofState{
            s.vel,                                                       // dpos/dt = vel
            sixDofAccelerationDiff(s.vel, s.quat, thrust, cfg.base),      // dvel/dt
            quatDerivativeDiff(s.quat, s.omega),                          // dquat/dt
            angularAccelerationDiff(s.omega, torque, cfg)};               // domega/dt
    };

    const SixDofState k1 = derivative(y);
    const SixDofState k2 = derivative(y + k1 * half);
    const SixDofState k3 = derivative(y + k2 * half);
    const SixDofState k4 = derivative(y + k3 * h);

    SixDofState out = y + (k1 + k2 * 2.0f + k3 * 2.0f + k4) * sixth;

    // 四元数积分会缓慢偏离单位模长，每步归一化抑制漂移
    out.quat = normalizeQuatDiff(out.quat);
    return out;
}

} // namespace oi3
