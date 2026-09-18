/**
 * @file ParameterIdentification.cpp
 * @brief 参数辨识实现
 * @author GhostFace
 * @date 2026/9/18
 *
 * 实现要点：把待辨识参数做成 requires_grad 的 {1} 叶子，动力学里所有用到它们的
 * 地方都改成张量运算（不能再用 cfg 里的 double），于是「轨迹预测误差」可以一路
 * 反向传播到参数上。优化器复用与打靶法相同的「归一化梯度 + 回溯线搜索」——
 * 后者保证代价单调不增，从根本上避免在最优点附近震荡。
 *
 * 参数的正性约束（质量、惯量、阻力系数都必须为正）通过每步更新后投影到可行域
 * 实现，而不是靠罚项：罚项需要调权重，而投影是精确的。
 */

#include "ParameterIdentification.h"

#include "SixDofDynamicsDiff.h"
#include "TensorUtils.h"

#include "AutoGrad.h"

#include <algorithm>
#include <cmath>

namespace oi3 {

namespace {

/// {1} 常量张量
Tensor scalarConst(double v) {
    Tensor t(ShapeTag{}, {1});
    t.data_write<float>()[0] = static_cast<float>(v);
    return t;
}

/// 由四元数分量拼装 {4}（与 SixDofDynamicsDiff 的 stackVec4 同构，避免跨 TU 依赖）
Tensor stackQ(const Tensor &w, const Tensor &x, const Tensor &y, const Tensor &z) {
    return w.concat(x, 0).concat(y, 0).concat(z, 0);
}

/// 取第 i 个分量的 {1} 视图
Tensor comp(const Tensor &t, std::size_t i) { return t.slice(0, i, 1); }

/**
 * @brief 参数化的平动加速度
 *
 * 与 sixDofAccelerationDiff 的方程一致，但质量与阻力系数是**张量**（可微），
 * 而非常量 —— 这正是让轨迹误差能反传到参数上的关键。
 */
Tensor accelParametric(const Tensor &vel, const Tensor &quat, const Tensor &thrust,
                       const Tensor &mass_t, const Tensor &drag_t, double gravity) {
    const Tensor g_t = scalarConst(gravity);
    const Tensor f_gravity = stackVec3(scalarConst(0.0), scalarConst(0.0), mass_t * g_t);

    const Tensor f_thrust_body =
        stackVec3(scalarConst(0.0), scalarConst(0.0), thrust * scalarConst(-1.0));
    Tensor f_total = f_gravity + rotateBodyToNedDiff(quat, f_thrust_body);

    // 阻力：F = -c * |v| * v（c 为可训练参数）
    f_total = f_total + (vel.abs() * vel) * (drag_t * scalarConst(-1.0));

    return f_total / mass_t;
}

/// 参数化的角加速度：ω̇ = I⁻¹(τ − ω × (Iω))，惯量为可训练参数
Tensor angularAccelParametric(const Tensor &omega, const Tensor &torque, const Tensor &ix,
                             const Tensor &iy, const Tensor &iz) {
    const Tensor wx = comp(omega, 0);
    const Tensor wy = comp(omega, 1);
    const Tensor wz = comp(omega, 2);
    const Tensor tx = comp(torque, 0);
    const Tensor ty = comp(torque, 1);
    const Tensor tz = comp(torque, 2);

    const Tensor iwx = wx * ix;
    const Tensor iwy = wy * iy;
    const Tensor iwz = wz * iz;

    const Tensor cx = wy * iwz - wz * iwy;
    const Tensor cy = wz * iwx - wx * iwz;
    const Tensor cz = wx * iwy - wy * iwx;

    return stackVec3((tx - cx) / ix, (ty - cy) / iy, (tz - cz) / iz);
}

/// 参数化的单步 RK4（与 SixDofDynamicsDiff::rk4StepSixDofDiff 同方程）
SixDofState rk4Parametric(const SixDofState &y, const Tensor &thrust, const Tensor &torque,
                          const Tensor &mass_t, const Tensor &drag_t, const Tensor &ix,
                          const Tensor &iy, const Tensor &iz, double gravity, float dt) {
    const float h = dt;
    const float half = h * 0.5f;
    const float sixth = h / 6.0f;

    const auto deriv = [&](const SixDofState &s) -> SixDofState {
        return SixDofState{
            s.vel,
            accelParametric(s.vel, s.quat, thrust, mass_t, drag_t, gravity),
            quatDerivativeDiff(s.quat, s.omega),
            angularAccelParametric(s.omega, torque, ix, iy, iz)};
    };

    const SixDofState k1 = deriv(y);
    const SixDofState k2 = deriv(y + k1 * half);
    const SixDofState k3 = deriv(y + k2 * half);
    const SixDofState k4 = deriv(y + k3 * half * 2.0f);

    SixDofState out = y + (k1 + k2 * 2.0f + k3 * 2.0f + k4) * sixth;
    out.quat = normalizeQuatDiff(out.quat);
    return out;
}

/// 从张量读第 i 个标量
double at(const Tensor &t, std::size_t i) {
    const float *p = t.data<float>();
    return p == nullptr ? 0.0 : static_cast<double>(p[i]);
}

/// 参数快照
struct ParamSnapshot {
    double mass = 1.0;
    double inertia[3] = {0.01, 0.01, 0.02};
    double drag = 0.05;
};

ParamSnapshot saveSnapshot(const std::vector<Tensor> &pt) {
    ParamSnapshot s;
    s.mass = at(pt[0], 0);
    s.inertia[0] = at(pt[1], 0);
    s.inertia[1] = at(pt[2], 0);
    s.inertia[2] = at(pt[3], 0);
    s.drag = at(pt[4], 0);
    return s;
}

void restoreSnapshot(std::vector<Tensor> &pt, const ParamSnapshot &s) {
    pt[0].data_write<float>()[0] = static_cast<float>(s.mass);
    pt[1].data_write<float>()[0] = static_cast<float>(s.inertia[0]);
    pt[2].data_write<float>()[0] = static_cast<float>(s.inertia[1]);
    pt[3].data_write<float>()[0] = static_cast<float>(s.inertia[2]);
    pt[4].data_write<float>()[0] = static_cast<float>(s.drag);
}

/// 正性投影：质量、惯量、阻力系数都必须为正
void projectPositive(std::vector<Tensor> &pt) {
    const double floors[5] = {1e-4, 1e-6, 1e-6, 1e-6, 1e-6};
    for (std::size_t i = 0; i < pt.size(); ++i) {
        float *v = pt[i].data_write<float>();
        if (!(v[0] > floors[i])) {
            v[0] = static_cast<float>(floors[i]);
        }
    }
}

/// 轨迹预测误差（含初值）。返回标量代价。
Tensor trajectoryLoss(const IdentTrajectory &tr, const std::vector<Tensor> &pt,
                      const Config &base) {
    // 初值：用常量张量起步（观测到的初始状态不参与求导）
    //
    // 注意四元数是 **4 维**，必须用 {4} 构造。早先这里误用 makeVec3（{3}）再写
    // q[3]，是一次堆越界写 —— 表现为进程 abort，且错误信息与「参数辨识」毫无关联。
    Tensor q0(ShapeTag{}, {4});
    for (int i = 0; i < 4; ++i) {
        q0.data_write<float>()[i] = static_cast<float>(at(tr.init.quat, i));
    }
    SixDofState st{
        makeVec3(static_cast<float>(at(tr.init.pos, 0)),
                 static_cast<float>(at(tr.init.pos, 1)),
                 static_cast<float>(at(tr.init.pos, 2))),
        makeVec3(static_cast<float>(at(tr.init.vel, 0)),
                 static_cast<float>(at(tr.init.vel, 1)),
                 static_cast<float>(at(tr.init.vel, 2))),
        q0,
        makeVec3(static_cast<float>(at(tr.init.omega, 0)),
                 static_cast<float>(at(tr.init.omega, 1)),
                 static_cast<float>(at(tr.init.omega, 2)))};

    Tensor cost = scalarConst(0.0);
    const std::size_t steps = tr.thrust.size();
    for (std::size_t k = 0; k < steps; ++k) {
        const Tensor th = scalarConst(tr.thrust[k]);
        Tensor tq(ShapeTag{}, {3});
        {
            float *p = tq.data_write<float>();
            p[0] = static_cast<float>(tr.torque[k][0]);
            p[1] = static_cast<float>(tr.torque[k][1]);
            p[2] = static_cast<float>(tr.torque[k][2]);
        }

        st = rk4Parametric(st, th, tq, pt[0], pt[4], pt[1], pt[2], pt[3], base.gravity,
                           static_cast<float>(base.dt));

        // 观测残差：位置 + 速度（姿态/角速度也纳入，惯量才可辨识）
        const Tensor dp = st.pos - makeVec3(static_cast<float>(tr.pos[k][0]),
                                            static_cast<float>(tr.pos[k][1]),
                                            static_cast<float>(tr.pos[k][2]));
        const Tensor dv = st.vel - makeVec3(static_cast<float>(tr.vel[k][0]),
                                            static_cast<float>(tr.vel[k][1]),
                                            static_cast<float>(tr.vel[k][2]));
        const Tensor dw = st.omega - makeVec3(static_cast<float>(tr.omega[k][0]),
                                              static_cast<float>(tr.omega[k][1]),
                                              static_cast<float>(tr.omega[k][2]));
        cost = cost + (dp * dp).sum() + (dv * dv).sum() + (dw * dw).sum();
    }
    return cost;
}

} // namespace

IdentTrajectory simulateTrajectory(const SixDofState &init,
                                   const std::vector<double> &thrust,
                                   const std::vector<std::array<double, 3>> &torque,
                                   const ParamSet &params, const Config &base) {
    IdentTrajectory tr;
    tr.init = init;
    tr.thrust = thrust;
    tr.torque = torque;

    SixDofState st = init;
    for (std::size_t k = 0; k < thrust.size(); ++k) {
        const Tensor th = scalarConst(thrust[k]);
        Tensor tq(ShapeTag{}, {3});
        {
            float *p = tq.data_write<float>();
            p[0] = static_cast<float>(torque[k][0]);
            p[1] = static_cast<float>(torque[k][1]);
            p[2] = static_cast<float>(torque[k][2]);
        }
        const Tensor mass_t = scalarConst(params.mass);
        const Tensor drag_t = scalarConst(params.drag_coeff);
        const Tensor ix = scalarConst(params.inertia[0]);
        const Tensor iy = scalarConst(params.inertia[1]);
        const Tensor iz = scalarConst(params.inertia[2]);

        st = rk4Parametric(st, th, tq, mass_t, drag_t, ix, iy, iz, base.gravity,
                           static_cast<float>(base.dt));

        const std::vector<float> p = toVector(st.pos);
        const std::vector<float> v = toVector(st.vel);
        const std::vector<float> q = toVector(st.quat);
        const std::vector<float> w = toVector(st.omega);
        tr.pos.push_back({p[0], p[1], p[2]});
        tr.vel.push_back({v[0], v[1], v[2]});
        tr.quat.push_back({q[0], q[1], q[2], q[3]});
        tr.omega.push_back({w[0], w[1], w[2]});
    }
    return tr;
}

IdentResult identifyParameters(const std::vector<IdentTrajectory> &data,
                               const ParamSet &guess, const IdentConfig &cfg) {
    IdentResult out;

    Config base;
    base.dt = 0.001;
    base.mass = guess.mass;
    base.gravity = 9.81;
    base.drag_coeff = guess.drag_coeff;

    // 待辨识参数：顺序 [mass, ix, iy, iz, drag]
    std::vector<Tensor> pt;
    const double init_vals[5] = {guess.mass, guess.inertia[0], guess.inertia[1],
                                 guess.inertia[2], guess.drag_coeff};
    for (double v : init_vals) {
        Tensor t(ShapeTag{}, {1});
        t.data_write<float>()[0] = static_cast<float>(v);
        t.requires_grad(true);
        pt.push_back(std::move(t));
    }

    auto totalLoss = [&]() {
        Tensor acc = scalarConst(0.0);
        for (const auto &tr : data) {
            acc = acc + trajectoryLoss(tr, pt, base);
        }
        return acc;
    };

    // 初始代价
    {
        Tensor l0 = totalLoss();
        out.initial_loss = at(l0, 0);
        if (!std::isfinite(out.initial_loss)) {
            out.finite = false;
            return out;
        }
    }

    double cur_loss = out.initial_loss;
    double step = cfg.step * std::max(1e-6, std::abs(guess.mass)); // 步长按参数初值量级

    for (int it = 0; it < cfg.iters; ++it) {
        for (auto &t : pt) {
            t.zero_grad();
        }
        Tensor l = totalLoss();
        if (!std::isfinite(at(l, 0))) {
            out.finite = false;
            break;
        }
        AutoGrad::backward(l.getRelatedNode(), false);

        // 归一化梯度方向
        double norm2 = 0.0;
        for (auto &t : pt) {
            const float *gr = t.grad_ptr();
            if (gr != nullptr && std::isfinite(gr[0])) {
                norm2 += static_cast<double>(gr[0]) * gr[0];
            }
        }
        if (!(norm2 > 0.0)) {
            out.converged = true;
            break;
        }
        const double inv_norm = 1.0 / std::sqrt(norm2);

        // 试探步（参数尺度差异大，各维按自身初值缩放后再走同一步长）
        ParamSnapshot snap = saveSnapshot(pt);
        for (std::size_t i = 0; i < pt.size(); ++i) {
            const float *gr = pt[i].grad_ptr();
            float *v = pt[i].data_write<float>();
            if (gr == nullptr || !std::isfinite(gr[0])) {
                continue;
            }
            const double scale_i = std::max(1e-9, std::abs(init_vals[i]));
            v[0] = static_cast<float>(static_cast<double>(v[0]) -
                                      step * static_cast<double>(gr[0]) * inv_norm * scale_i);
        }
        projectPositive(pt);

        Tensor lt = totalLoss();
        const double trial = at(lt, 0);
        if (std::isfinite(trial) && trial < cur_loss) {
            const double rel = (cur_loss - trial) / std::max(1e-12, std::abs(cur_loss));
            cur_loss = trial;
            step *= 1.2;
            out.iters = it + 1;
            if (rel < cfg.rel_tol) {
                out.converged = true;
                break;
            }
        } else {
            restoreSnapshot(pt, snap);
            step *= 0.5;
            out.iters = it + 1;
            if (step < 1e-9 * std::max(1e-6, std::abs(guess.mass))) {
                out.converged = true;
                break;
            }
        }
    }

    const ParamSnapshot fin = saveSnapshot(pt);
    out.params.mass = fin.mass;
    out.params.inertia[0] = fin.inertia[0];
    out.params.inertia[1] = fin.inertia[1];
    out.params.inertia[2] = fin.inertia[2];
    out.params.drag_coeff = fin.drag;

    Tensor lf = totalLoss();
    out.final_loss = at(lf, 0);
    out.finite = out.finite && std::isfinite(out.final_loss);
    return out;
}

} // namespace oi3
