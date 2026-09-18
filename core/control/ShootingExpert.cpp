/**
 * @file ShootingExpert.cpp
 * @brief 打靶法专家实现
 * @author GhostFace
 * @date 2026/9/18
 *
 * 实现要点：把每段的推力当作 requires_grad 的 {3} 叶子张量，用可微的 RK4 推进
 * control_decimation 个仿真步，末端与沿途的误差构成代价；一次反向即可得到整段
 * 序列的梯度，随后按符号梯度更新（见 ShootingConfig::step 的说明）。
 */

#include "ShootingExpert.h"

#include "DroneDynamics.h"
#include "TensorUtils.h"

#include "AutoGrad.h"

#include <algorithm>
#include <cmath>

namespace oi3 {

namespace {

/// 三分量常量张量（作为代价里的参考点，不参与求导）
Tensor makeConst3(double n, double e, double d) {
    Tensor t(ShapeTag{}, {3});
    float *p = t.data_write<float>();
    p[0] = static_cast<float>(n);
    p[1] = static_cast<float>(e);
    p[2] = static_cast<float>(d);
    return t;
}

/// 按 HoverEnv 的契约构造观测：位置误差 / target_range、速度 / vel_scale，再限幅
std::vector<float> makeObs(const DroneState &st, const std::array<double, 3> &target,
                           const HoverEnvConfig &env) {
    const std::vector<float> pos = toVector(st.pos);
    const std::vector<float> vel = toVector(st.vel);

    const double pos_norm = std::max(1e-6, env.target_range);
    const double vel_norm = std::max(1e-6, env.vel_scale);

    std::vector<float> obs(static_cast<std::size_t>(HoverEnv::kObsDim), 0.0f);
    for (int i = 0; i < 3; ++i) {
        const double err =
            target[static_cast<std::size_t>(i)] - static_cast<double>(pos[static_cast<std::size_t>(i)]);
        obs[static_cast<std::size_t>(i)] = static_cast<float>(err / pos_norm);
        obs[static_cast<std::size_t>(i + 3)] =
            static_cast<float>(static_cast<double>(vel[static_cast<std::size_t>(i)]) / vel_norm);
    }
    const float clip = static_cast<float>(env.obs_clip);
    for (float &v : obs) {
        v = std::max(-clip, std::min(clip, v));
    }
    return obs;
}

/// 按 HoverEnv 的契约把推力换成动作：a = (thrust - hover) / (thrust_scale * m * g)
std::array<float, 3> thrustToAction(const Tensor &thrust, const HoverEnvConfig &env) {
    const std::vector<float> t = toVector(thrust);
    const double m = env.plant.mass;
    const double g = env.plant.gravity;
    const double hover[3] = {0.0, 0.0, -m * g};
    const double scale = std::max(1e-9, env.thrust_scale * m * g);

    std::array<float, 3> action{};
    for (int i = 0; i < 3; ++i) {
        const double a =
            (static_cast<double>(t[static_cast<std::size_t>(i)]) - hover[i]) / scale;
        action[static_cast<std::size_t>(i)] =
            static_cast<float>(std::max(-1.0, std::min(1.0, a)));
    }
    return action;
}

} // namespace

ShootingResult shootHover(const std::array<double, 3> &pos0,
                          const std::array<double, 3> &vel0,
                          const std::array<double, 3> &target, const HoverEnvConfig &env,
                          const ShootingConfig &sc) {
    ShootingResult result;

    const Config &plant = env.plant;
    const double m = plant.mass;
    const double g = plant.gravity;
    const double hover_d = -m * g;
    const double scale = std::max(1e-9, env.thrust_scale * m * g);
    const int decim = std::max(1, env.control_decimation);
    const int segments = std::max(1, sc.segments);

    const Tensor target_t = makeConst3(target[0], target[1], target[2]);
    const Tensor hover_t = makeConst3(0.0, 0.0, hover_d);

    // 参数：每段一个 {3} 叶子，初值取悬停推力（比零初值更接近解，收敛更快）
    std::vector<Tensor> seg_thrust;
    seg_thrust.reserve(static_cast<std::size_t>(segments));
    for (int s = 0; s < segments; ++s) {
        Tensor t(ShapeTag{}, {3});
        float *p = t.data_write<float>();
        p[0] = 0.0f;
        p[1] = 0.0f;
        p[2] = static_cast<float>(hover_d);
        t.requires_grad(true);
        seg_thrust.push_back(std::move(t));
    }

    result.obs_seq.assign(static_cast<std::size_t>(segments), {});
    result.action_seq.assign(static_cast<std::size_t>(segments), {});
    result.thrust_seq.assign(static_cast<std::size_t>(segments), {});

    // 一次完整滚动：建图并返回代价。record=true 时顺带记录每段起点的观测。
    auto rollout = [&](bool record) {
        DroneState st{makeVec3(static_cast<float>(pos0[0]), static_cast<float>(pos0[1]),
                               static_cast<float>(pos0[2])),
                      makeVec3(static_cast<float>(vel0[0]), static_cast<float>(vel0[1]),
                               static_cast<float>(vel0[2]))};

        Tensor cost = makeConst3(0.0, 0.0, 0.0).sum() * 0.0f; // 标量零，保持图连通
        for (int s = 0; s < segments; ++s) {
            if (record) {
                result.obs_seq[static_cast<std::size_t>(s)] = makeObs(st, target, env);
            }
            const Tensor &th = seg_thrust[static_cast<std::size_t>(s)];

            for (int k = 0; k < decim; ++k) {
                st = rk4StepDrone(st, th, plant, plant.dt);
            }

            // 沿途位置误差（防止末端对了但中途飞远）
            Tensor d = st.pos - target_t;
            cost = cost + (d * d).sum() * static_cast<float>(sc.w_traj);

            // 推力偏离悬停量的惩罚
            Tensor dev = th - hover_t;
            cost = cost + (dev * dev).sum() * static_cast<float>(sc.w_thrust);
        }

        Tensor dp = st.pos - target_t;
        Tensor dv = st.vel;
        cost = cost + (dp * dp).sum() * static_cast<float>(sc.w_pos) +
               (dv * dv).sum() * static_cast<float>(sc.w_vel);

        if (record) {
            const std::vector<float> pos = toVector(st.pos);
            double e2 = 0.0;
            for (int i = 0; i < 3; ++i) {
                const double e = target[static_cast<std::size_t>(i)] -
                                 static_cast<double>(pos[static_cast<std::size_t>(i)]);
                e2 += e * e;
            }
            result.final_pos_error = std::sqrt(e2);
        }
        return std::make_pair(cost, st);
    };

    // ---- 梯度下降：符号步长 ----
    float last_cost = 0.0f;
    for (int it = 0; it < sc.iters; ++it) {
        for (auto &t : seg_thrust) {
            t.zero_grad();
        }
        auto [cost, st_unused] = rollout(false);
        (void)st_unused;
        last_cost = cost.data<float>()[0];
        if (!std::isfinite(last_cost)) {
            result.finite = false;
            return result;
        }

        AutoGrad::backward(cost.getRelatedNode(), false);

        for (int s = 0; s < segments; ++s) {
            Tensor &t = seg_thrust[static_cast<std::size_t>(s)];
            const float *gr = t.grad_ptr();
            if (gr == nullptr) {
                continue;
            }
            float *v = t.data_write<float>();
            for (int i = 0; i < 3; ++i) {
                if (!std::isfinite(gr[i])) {
                    continue;
                }
                const float dir = (gr[i] > 0.0f) ? 1.0f : ((gr[i] < 0.0f) ? -1.0f : 0.0f);
                v[i] -= sc.step * static_cast<float>(scale) * dir;
            }
            // 把推力约束在**动作能够表达的范围内**（hover ± scale，对应 action ∈ [-1,1]）。
            //
            // 这一步不可省：打靶法本身只关心代价，符号步长累积起来会给出远超动作
            // 范围的推力；而策略经动作输出，超出部分会在环境里被限幅 —— 于是专家
            // 演示与策略实际能执行的动作**系统性不一致**，行为克隆学到的是饱和动作，
            // 闭环直接发散（实测：末端误差从 0.5 m 恶化到 3.6 m）。
            // 约束范围而非事后检查：让专家只在可执行的解空间里寻优。
            const double hover[3] = {0.0, 0.0, hover_d};
            for (int i = 0; i < 3; ++i) {
                const double lo = hover[i] - scale;
                const double hi = hover[i] + scale;
                v[i] = static_cast<float>(std::max(lo, std::min(hi, static_cast<double>(v[i]))));
            }
        }
    }

    // ---- 用最终参数记录轨迹与动作 ----
    auto [final_cost, st_final] = rollout(true);
    (void)st_final;
    result.cost = static_cast<double>(final_cost.data<float>()[0]);
    result.finite = result.finite && std::isfinite(result.cost);

    for (int s = 0; s < segments; ++s) {
        const Tensor &t = seg_thrust[static_cast<std::size_t>(s)];
        const std::vector<float> tv = toVector(t);
        result.thrust_seq[static_cast<std::size_t>(s)] = {tv[0], tv[1], tv[2]};
        result.action_seq[static_cast<std::size_t>(s)] = thrustToAction(t, env);
    }

    return result;
}

} // namespace oi3
