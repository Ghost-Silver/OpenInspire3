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
                          const ShootingConfig &sc,
                          const std::vector<std::array<double, 3>> &warm_seq) {
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
        if (static_cast<std::size_t>(s) < warm_seq.size()) {
            // 热启动：用上一次的解。滚动时域控制里相邻两次规划的问题只差一小段
            // 时间平移，热启动能把所需迭代数压到很低。
            p[0] = static_cast<float>(warm_seq[static_cast<std::size_t>(s)][0]);
            p[1] = static_cast<float>(warm_seq[static_cast<std::size_t>(s)][1]);
            p[2] = static_cast<float>(warm_seq[static_cast<std::size_t>(s)][2]);
        } else {
            p[0] = 0.0f;
            p[1] = 0.0f;
            p[2] = static_cast<float>(hover_d);
        }
        t.requires_grad(true);
        seg_thrust.push_back(std::move(t));
    }

    // 初始位置误差（与末段误差一起输出，用于判断求解器是否真的在改善 ——
    // 只看末值无法区分「收敛到最优」与「根本没动」）
    {
        const double d0n = pos0[0] - target[0];
        const double d0e = pos0[1] - target[1];
        const double d0d = pos0[2] - target[2];
        result.initial_pos_error = std::sqrt(d0n * d0n + d0e * d0e + d0d * d0d);
    }

    result.obs_seq.assign(static_cast<std::size_t>(segments), {});
    result.action_seq.assign(static_cast<std::size_t>(segments), {});
    result.thrust_seq.assign(static_cast<std::size_t>(segments), {});

    // rollout 的返回类型。
    //
    // 这里必须用一个具名结构体 + 移动语义，不能图省事返回 std::make_pair：
    // Tensor 的**拷贝**构造会重置 autograd 节点（见 CTorch/AutoGrad.h 的说明），
    // 而 pair 的构造是拷贝 —— 调用方拿到的 cost 与计算图已经断开，
    // backward 从一个孤立的 GradAccumulator 开始，梯度恒为零、参数从不更新。
    // 症状极具误导性：专家解看起来「没动」（末段误差 ≈ 初值），而调整步长、
    // 迭代轮数都毫无效果。用移动构造则保留节点（Tensor 的 move 会 rebind）。
    struct RolloutOut {
        Tensor cost;
        DroneState state;
    };

    // 一次完整滚动：建图并返回代价。record=true 时顺带记录每段起点的观测。
    auto rollout = [&](bool record) -> RolloutOut {
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
        return RolloutOut{std::move(cost), std::move(st)};
    };

    // ---- 优化：归一化梯度 + 回溯线搜索 ----
    //
    // 起初这里用的是「符号梯度 + 固定步长」，它在本问题上**不收敛**：符号梯度每步
    // 移动的距离与梯度无关（固定为 step × scale），一旦接近最优点，很小的梯度仍然
    // 对应同样大的位移，于是解在最优点两侧反复横跳，净位移趋于零。实测把迭代数从
    // 15 加到 200 毫无改善（末段误差 0.3600 / 0.3719 / 0.3719 m），正是这个原因。
    //
    // 现在改为：
    //   1. 方向取**归一化梯度**（g / |g|）—— 各维同量纲（都是力），整体方向比逐维
    //      符号更能表达「往哪走」，也不受梯度绝对尺度影响；
    //   2. 步长用**回溯线搜索**（试探 → 代价下降则接受并略增，否则回退并减半）——
    //      这样不需要预知梯度量级，且**保证代价单调不增**，从根本上排除震荡；
    //   3. 收敛判据改为「步长缩到阈值以下」或「代价相对下降低于阈值」，
    //      而不是「跑满固定迭代数」。
    //
    // 推力始终约束在**动作可表达的范围内**（hover ± scale，对应 action ∈ [-1,1]）：
    // 打靶法本身只关心代价，会给出远超动作范围的解；而策略经动作输出时超出部分会
    // 在环境里被限幅，于是专家演示与策略实际能执行的动作系统性不一致，行为克隆
    // 学到饱和动作、闭环发散（实测末端误差从 0.5 m 恶化到 3.6 m）。
    const double hover_ref[3] = {0.0, 0.0, hover_d};
    auto clampToActionRange = [&](Tensor &t) {
        float *v = t.data_write<float>();
        for (int i = 0; i < 3; ++i) {
            const double lo = hover_ref[i] - scale;
            const double hi = hover_ref[i] + scale;
            v[i] = static_cast<float>(std::max(lo, std::min(hi, static_cast<double>(v[i]))));
        }
    };

    // 参数快照（回溯失败时回退用）
    std::vector<std::array<float, 3>> snapshot(static_cast<std::size_t>(segments));
    auto saveParams = [&]() {
        for (int s = 0; s < segments; ++s) {
            const std::vector<float> tv = toVector(seg_thrust[static_cast<std::size_t>(s)]);
            snapshot[static_cast<std::size_t>(s)] = {tv[0], tv[1], tv[2]};
        }
    };
    auto restoreParams = [&]() {
        for (int s = 0; s < segments; ++s) {
            float *v = seg_thrust[static_cast<std::size_t>(s)].data_write<float>();
            const auto &sn = snapshot[static_cast<std::size_t>(s)];
            v[0] = sn[0];
            v[1] = sn[1];
            v[2] = sn[2];
        }
    };

    double step = static_cast<double>(sc.step) * scale; // 初始步长（牛顿）
    double cur_cost = 0.0;
    {
        RolloutOut out0 = rollout(false);
        cur_cost = static_cast<double>(out0.cost.data<float>()[0]);
        if (!std::isfinite(cur_cost)) {
            result.finite = false;
            return result;
        }
    }

    const double step_min = 1e-4 * scale;      // 步长下界 → 视为收敛
    const double rel_tol = 1e-5;               // 代价相对下降阈值

    for (int it = 0; it < sc.iters; ++it) {
        // 1) 求梯度方向
        for (auto &t : seg_thrust) {
            t.zero_grad();
        }
        RolloutOut grad_out = rollout(false);
        Tensor &cost_for_grad = grad_out.cost;
        if (!std::isfinite(cost_for_grad.data<float>()[0])) {
            result.finite = false;
            return result;
        }
        AutoGrad::backward(cost_for_grad.getRelatedNode(), false);

        // 归一化梯度方向（整体归一，保留各维相对比例）
        double norm2 = 0.0;
        for (int s = 0; s < segments; ++s) {
            const float *gr = seg_thrust[static_cast<std::size_t>(s)].grad_ptr();
            if (gr == nullptr) {
                continue;
            }
            for (int i = 0; i < 3; ++i) {
                if (std::isfinite(gr[i])) {
                    norm2 += static_cast<double>(gr[i]) * gr[i];
                }
            }
        }
        if (!(norm2 > 0.0)) {
            break; // 梯度为零 → 已到驻点
        }
        const double inv_norm = 1.0 / std::sqrt(norm2);

        // 2) 试探步
        saveParams();
        for (int s = 0; s < segments; ++s) {
            Tensor &t = seg_thrust[static_cast<std::size_t>(s)];
            const float *gr = t.grad_ptr();
            float *v = t.data_write<float>();
            if (gr == nullptr) {
                continue;
            }
            for (int i = 0; i < 3; ++i) {
                const double g = std::isfinite(gr[i]) ? static_cast<double>(gr[i]) : 0.0;
                v[i] = static_cast<float>(static_cast<double>(v[i]) - step * g * inv_norm);
            }
            clampToActionRange(t);
        }

        RolloutOut trial_out = rollout(false);
        const double trial_cost = static_cast<double>(trial_out.cost.data<float>()[0]);

        if (std::isfinite(trial_cost) && trial_cost < cur_cost) {
            const double rel = (cur_cost - trial_cost) / std::max(1e-12, std::abs(cur_cost));
            cur_cost = trial_cost;
            step *= 1.2; // 成功 → 略增，加快后续推进
            if (rel < rel_tol) {
                break; // 代价已基本不再下降
            }
        } else {
            restoreParams(); // 失败 → 回退并减半
            step *= 0.5;
            if (step < step_min) {
                break; // 步长缩到阈值以下 → 收敛
            }
        }
    }

    // ---- 用最终参数记录轨迹与动作 ----
    RolloutOut final_out = rollout(true);
    (void)final_out.state;
    result.cost = static_cast<double>(final_out.cost.data<float>()[0]);
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
