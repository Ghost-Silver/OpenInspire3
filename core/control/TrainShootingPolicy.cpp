/**
 * @file TrainShootingPolicy.cpp
 * @brief 用打靶法专家数据做行为克隆，并做闭环评估
 * @author GhostFace
 * @date 2026/9/18
 *
 * 链路：随机初始状态 → 打靶法求最优推力序列（可微动力学 + 梯度下降）
 *       → 取沿途 (观测, 动作) 作为专家演示 → 监督训练策略网络
 *       → 把策略放回环境跑闭环，测稳态误差与成功率。
 *
 * 为什么这样接：打靶法给的是**开环序列**（只针对一个初始状态），而飞控要的是
 * **闭环策略**。行为克隆把前者转成后者 —— 策略学的是「看到偏差该往哪推」，
 * 因而具备纠偏能力；这也是打靶法作为「教师」的用法。
 *
 * 对照组：未经训练的策略（同一网络结构、随机权重），用于确认性能提升确实来自
 * 专家数据而不是评估口径。
 *
 * 用法：TrainShootingPolicy [专家样本数] [BC 训练步数] [评估回合数] [PPO 微调回合数]
 */

#include "ContinuousPPO.h"
#include "HoverEnv.h"
#include "ShootingExpert.h"
#include "TensorUtils.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

using namespace oi3;
using oi3::rl::ContinuousPPO;
using oi3::rl::PpoConfig;

namespace {

/// 与环境训练脚本一致的被控对象与任务配置
HoverEnvConfig makeEnvConfig() {
    HoverEnvConfig cfg;
    cfg.plant.dt = 0.001;
    cfg.plant.mass = 1.0;
    cfg.plant.gravity = 9.81;
    cfg.plant.drag_coeff = 0.3;
    cfg.plant.max_thrust = 20.0;
    cfg.episode_seconds = 2.0;
    cfg.control_decimation = 10;
    cfg.thrust_scale = 0.8;
    cfg.reward_scale = 0.1;
    return cfg;
}

PpoConfig makePpoConfig() {
    PpoConfig cfg;
    cfg.obs_dim = HoverEnv::kObsDim;
    cfg.action_dim = HoverEnv::kActionDim;
    cfg.hidden_dim = 64;
    cfg.learning_rate = 1e-3f;
    return cfg;
}

/// 闭环评估：返回 {平均末端位置误差, 成功率, 平均回报}
struct EvalResult {
    double mean_final_error = 0.0;
    double success_rate = 0.0;
    double mean_return = 0.0;
};

EvalResult evaluate(ContinuousPPO &agent, const HoverEnvConfig &env_cfg, int episodes,
                    std::uint32_t seed) {
    EvalResult out;
    double err_sum = 0.0;
    double ret_sum = 0.0;
    int success = 0;

    for (int ep = 0; ep < episodes; ++ep) {
        HoverEnv env(env_cfg, seed + static_cast<std::uint32_t>(ep) * 7919u);
        std::vector<float> obs = env.reset();
        double ep_return = 0.0;
        bool done = false;
        bool ep_success = false;
        while (!done) {
            // 用策略均值（确定性动作）评估，不注入探索噪声
            const std::array<float, 3> a = agent.actMean(obs);
            const HoverEnv::StepResult r = env.step(a);
            obs = r.obs;
            done = r.done;
            ep_return += static_cast<double>(r.reward);
            if (r.success) {
                ep_success = true;
            }
        }
        err_sum += env.lastPositionError();
        ret_sum += ep_return;
        if (ep_success) {
            ++success;
        }
    }

    const double n = std::max(1, episodes);
    out.mean_final_error = err_sum / n;
    out.success_rate = static_cast<double>(success) / n;
    out.mean_return = ret_sum / n;
    return out;
}

} // namespace

int main(int argc, char **argv) {
    const int expert_states = argc > 1 ? std::atoi(argv[1]) : 30;
    const int bc_steps = argc > 2 ? std::atoi(argv[2]) : 800;
    const int eval_episodes = argc > 3 ? std::atoi(argv[3]) : 20;
    const int ppo_ep = argc > 4 ? std::atoi(argv[4]) : 0;

    const HoverEnvConfig env_cfg = makeEnvConfig();
    const PpoConfig ppo_cfg = makePpoConfig();

    // 窗口长度是这里的关键参数：段数 × 控制周期 = 专家可见的时间跨度。
    // 窗口太短（例如 12 段 ≈ 0.12 s）时，专家在窗口内根本来不及消除初始偏差，
    // 末段误差几乎等于初值 —— 那样提供的教师信号只说明「朝目标推」，
    // 与一个 PD 控制器无异。这里取 25 段 ≈ 0.25 s，让专家有空间展示真正的修正过程。
    ShootingConfig sc;
    sc.segments = 25;
    sc.iters = 15;
    sc.step = 0.2f;

    std::cout << "========================================\n";
    std::cout << "打靶法专家 -> 行为克隆 -> 闭环评估\n";
    std::cout << "专家样本状态数 = " << expert_states << "，BC 步数 = " << bc_steps
              << "，评估回合 = " << eval_episodes << "，PPO 微调回合 = " << ppo_ep << "\n";
    std::cout << "打靶法：段数 = " << sc.segments << "，迭代 = " << sc.iters
              << "，步长 = " << sc.step << "\n";
    std::cout << "========================================\n";

    // ---- 1. 生成专家数据 ----
    // 初始状态在目标附近按正态采样：策略需要见到的不是「一条轨迹上的点」，
    // 而是「各种偏差下该怎么修正」，所以覆盖状态分布比覆盖时间更重要。
    std::mt19937 rng(20260918u);
    std::normal_distribution<double> pos_noise(0.0, 0.3);
    std::normal_distribution<double> vel_noise(0.0, 0.25);

    std::vector<std::vector<float>> obs_batch;
    std::vector<std::array<float, 3>> act_batch;
    obs_batch.reserve(static_cast<std::size_t>(expert_states * sc.segments));
    act_batch.reserve(static_cast<std::size_t>(expert_states * sc.segments));

    int used = 0;
    double expert_err_sum = 0.0;
    for (int i = 0; i < expert_states; ++i) {
        const std::array<double, 3> pos0 = {pos_noise(rng), pos_noise(rng), pos_noise(rng)};
        const std::array<double, 3> vel0 = {vel_noise(rng), vel_noise(rng), vel_noise(rng)};
        const std::array<double, 3> target = {0.0, 0.0, 0.0};

        const ShootingResult r = shootHover(pos0, vel0, target, env_cfg, sc);
        if (!r.finite) {
            continue;
        }
        ++used;
        expert_err_sum += r.final_pos_error;

        const std::size_t n = std::min(r.obs_seq.size(), r.action_seq.size());
        for (std::size_t k = 0; k < n; ++k) {
            obs_batch.push_back(r.obs_seq[k]);
            act_batch.push_back(r.action_seq[k]);
        }
    }

    std::cout << "\n[专家]\n";
    std::cout << "  成功求解 " << used << " / " << expert_states << " 个初始状态\n";
    std::cout << "  收集到 " << obs_batch.size() << " 条 (观测, 动作) 演示\n";
    if (used > 0) {
        std::cout << "  专家末段平均位置误差 = " << (expert_err_sum / used) << " m\n";
    }

    if (obs_batch.size() < 16) {
        std::cout << "  演示样本过少，终止\n";
        return 1;
    }

    // ---- 2. 行为克隆 ----
    ContinuousPPO agent(ppo_cfg);
    std::cout << "\n[行为克隆]\n";
    float first_loss = 0.0f;
    float last_loss = 0.0f;
    const std::size_t batch = 64;
    for (int step = 0; step < bc_steps; ++step) {
        // 小批量采样（带放回），避免固定顺序带来的偏置
        std::vector<std::vector<float>> bo;
        std::vector<std::array<float, 3>> ba;
        bo.reserve(batch);
        ba.reserve(batch);
        for (std::size_t k = 0; k < batch; ++k) {
            const std::size_t idx =
                static_cast<std::size_t>(rng()) % obs_batch.size();
            bo.push_back(obs_batch[idx]);
            ba.push_back(act_batch[idx]);
        }
        const float loss = agent.behaviorCloneStep(bo, ba);
        if (step == 0) {
            first_loss = loss;
        }
        last_loss = loss;
        if ((step + 1) % 200 == 0) {
            std::cout << "  step " << (step + 1) << "  MSE = " << loss << "\n";
        }
    }
    std::cout << "  MSE: " << first_loss << " -> " << last_loss << "\n";

    // ---- 3. 闭环评估 ----
    std::cout << "\n[闭环评估]（确定性动作，无探索噪声）\n";
    const EvalResult trained = evaluate(agent, env_cfg, eval_episodes, 1234u);

    // 对照组：同一结构、未训练的随机策略
    ContinuousPPO untrained(ppo_cfg);
    const EvalResult random_policy = evaluate(untrained, env_cfg, eval_episodes, 1234u);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  " << std::left << std::setw(14) << "指标" << std::setw(16) << "未训练"
              << std::setw(16) << "打靶法专家BC" << "\n";
    std::cout << "  " << std::setw(14) << "末端位置误差" << std::setw(16)
              << random_policy.mean_final_error << std::setw(16)
              << trained.mean_final_error << "\n";
    std::cout << "  " << std::setw(14) << "成功率" << std::setw(16)
              << random_policy.success_rate << std::setw(16) << trained.success_rate
              << "\n";
    std::cout << "  " << std::setw(14) << "平均回报" << std::setw(16)
              << random_policy.mean_return << std::setw(16) << trained.mean_return
              << "\n";

    const bool improved =
        trained.mean_final_error < random_policy.mean_final_error * 0.5 ||
        trained.success_rate > random_policy.success_rate;
    std::cout << "\n  [" << (improved ? " ok " : "FAIL")
              << "] 行为克隆后的策略优于未训练策略\n";

    // ---- 4. PPO 微调 ----
    //
    // 行为克隆单独跑闭环是不够的，这一点必须说清楚：专家演示只覆盖窗口内的状态
    // （本配置 25 段 ≈ 0.25 s），而回合有数百个控制周期；一旦策略把飞行器带到
    // 训练分布之外，它就没有可依据的示范了 —— 误差沿回合累积放大，这就是行为
    // 克隆的经典复合误差问题。打靶法专家在这里的价值不是「替代控制器」，
    // 而是**给 PPO 一个好得多的起点**，把探索从「两秒内就漂出发散边界」变成
    // 「已经会悬停，只需精调」。
    const int ppo_episodes = std::max(0, ppo_ep);
    std::cout << "\n[PPO 微调]（暖启动于打靶法专家，回合数 " << ppo_episodes << "）\n";
    if (ppo_episodes > 0) {
        std::uint32_t seed = 777u;
        // 直接在 BC 后的 agent 上继续训练（不做对象拷贝：策略内部持有优化器与
        // 参数张量的绑定，拷贝构造后仍需重绑定才正确，而这层语义不该由调用方承担）
        ContinuousPPO &fine = agent;
        double ret_window = 0.0;
        for (int ep = 0; ep < ppo_episodes; ++ep) {
            HoverEnv env(env_cfg, seed + static_cast<std::uint32_t>(ep));
            std::vector<float> obs = env.reset();
            double ep_return = 0.0;
            while (true) {
                const oi3::rl::ActionSample sample = fine.selectAction(obs);
                const HoverEnv::StepResult r = env.step(sample.action);
                oi3::rl::Transition tr;
                tr.obs = obs;
                tr.action = sample.action;
                tr.log_prob = sample.log_prob;
                tr.value = sample.value;
                tr.reward = r.reward;
                tr.done = r.done ? 1.0f : 0.0f;
                fine.store(tr);
                obs = r.obs;
                ep_return += static_cast<double>(r.reward);
                if (r.done) {
                    break;
                }
            }
            fine.update();
            ret_window += ep_return;
            if ((ep + 1) % 100 == 0) {
                std::cout << "  episode " << (ep + 1)
                          << "  近 100 回合平均回报 = " << (ret_window / 100.0) << "\n";
                ret_window = 0.0;
            }
        }

        const EvalResult finetuned = evaluate(fine, env_cfg, eval_episodes, 1234u);
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "  " << std::left << std::setw(14) << "指标" << std::setw(16)
                  << "仅 BC" << std::setw(16) << "BC + PPO 微调" << "\n";
        std::cout << "  " << std::setw(14) << "末端位置误差" << std::setw(16)
                  << trained.mean_final_error << std::setw(16)
                  << finetuned.mean_final_error << "\n";
        std::cout << "  " << std::setw(14) << "成功率" << std::setw(16)
                  << trained.success_rate << std::setw(16) << finetuned.success_rate
                  << "\n";
        std::cout << "  " << std::setw(14) << "平均回报" << std::setw(16)
                  << trained.mean_return << std::setw(16) << finetuned.mean_return
                  << "\n";
        std::cout << "\n  [" << (finetuned.mean_final_error < trained.mean_final_error
                                      ? " ok "
                                      : "FAIL")
                  << "] PPO 微调进一步降低末端误差\n";
        return finetuned.mean_final_error < trained.mean_final_error ? 0 : 1;
    }

    return improved ? 0 : 1;
}
