/**
 * @file BcDiagnostic.cpp
 * @brief 行为克隆诊断：分离「网络/反向传播是否可用」与「PPO 梯度估计是否正确」
 * @author GhostFace
 * @date 2026/9/16
 *
 * 逻辑：用 PID 在悬停任务上生成专家动作，再让策略网络做纯监督学习去拟合。
 *   - 损失能稳定下降 -> 网络结构与反向传播都是通的，PPO 不收敛的原因在算法侧
 *   - 损失不下降     -> 问题在网络或反传路径，与 PPO 的超参数无关
 *
 * 用法：BcDiagnostic [专家回合数] [训练步数]
 */

#include "ContinuousPPO.h"
#include "HoverEnv.h"
#include "PidController.h"
#include "TensorUtils.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

using namespace oi3;

namespace {

/// 与训练脚本一致的被控对象与环境配置
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

} // namespace

int main(int argc, char **argv) {
    CtorchError::setPrintLevel(PrintLevel::MINIUM);

    const int expert_episodes = argc > 1 ? std::atoi(argv[1]) : 20;
    const int train_steps = argc > 2 ? std::atoi(argv[2]) : 3000;

    const HoverEnvConfig env_cfg = makeEnvConfig();

    // ---- 第一步：用 PID 生成专家数据 ----
    HoverEnv env(env_cfg, 7u);
    PidController pid(env_cfg.plant);

    std::vector<std::vector<float>> obs_data;
    std::vector<std::array<float, 3>> act_data;

    for (int ep = 0; ep < expert_episodes; ++ep) {
        std::vector<float> obs = env.reset();

        for (int t = 0; t < env.stepsPerEpisode(); ++t) {
            const std::array<float, 3> tgt = env.target();
            const Tensor target_t = makeVec3(tgt[0], tgt[1], tgt[2]);

            const Tensor thrust = pid.computeThrust(env.state(), target_t, 0.0);
            const std::vector<float> tf = toVector(thrust);
            const std::array<double, 3> td = {tf[0], tf[1], tf[2]};
            const std::array<float, 3> expert_action = env.thrustToAction(td);

            obs_data.push_back(obs);
            act_data.push_back(expert_action);

            const HoverEnv::StepResult result = env.step(expert_action);
            obs = result.obs;
            if (result.done) {
                break;
            }
        }
    }

    std::cout << "===== 行为克隆诊断 =====\n";
    std::cout << "专家数据 " << obs_data.size() << " 条（PID 控制器，"
              << expert_episodes << " 回合）\n";

    // 专家动作的统计：若几乎恒为 0，说明 PID 基本只输出悬停推力，
    // 那么行为克隆本身也学不到有信息量的映射
    double act_abs_mean = 0.0;
    double act_abs_max = 0.0;
    for (const auto &a : act_data) {
        for (float v : a) {
            act_abs_mean += std::fabs(v);
            act_abs_max = std::max(act_abs_max, static_cast<double>(std::fabs(v)));
        }
    }
    act_abs_mean /= static_cast<double>(act_data.size() * 3);
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "专家动作 |a| 均值 " << act_abs_mean << "，最大值 " << act_abs_max
              << "\n\n";

    // ---- 第二步：监督学习拟合 ----
    rl::PpoConfig ppo_cfg;
    ppo_cfg.obs_dim = HoverEnv::kObsDim;
    ppo_cfg.action_dim = HoverEnv::kActionDim;
    ppo_cfg.hidden_dim = 64;
    ppo_cfg.learning_rate = 1e-3f;

    rl::ContinuousPPO agent(ppo_cfg, 99u);

    const std::size_t batch = 64;
    std::mt19937 shuffle_rng(2024u);
    std::vector<std::size_t> order(obs_data.size());
    std::iota(order.begin(), order.end(), 0);

    std::cout << "训练步数 " << train_steps << "，batch " << batch << "\n";
    double first_loss = 0.0;
    double last_loss = 0.0;

    for (int step = 0; step < train_steps; ++step) {
        std::shuffle(order.begin(), order.end(), shuffle_rng);

        std::vector<std::vector<float>> obs_batch;
        std::vector<std::array<float, 3>> act_batch;
        obs_batch.reserve(batch);
        act_batch.reserve(batch);
        for (std::size_t i = 0; i < batch && i < order.size(); ++i) {
            obs_batch.push_back(obs_data[order[i]]);
            act_batch.push_back(act_data[order[i]]);
        }

        const float loss = agent.behaviorCloneStep(obs_batch, act_batch);
        if (step == 0) {
            first_loss = loss;
        }
        last_loss = loss;
        if (step == 0 || step == 599) {
            const char *names[] = {"FC1_W", "FC1_B", "FC2_W", "FC2_B", "MEAN_W",
                                   "MEAN_B", "V_W",   "V_B",   "LOG_STD"};
            const auto gn = agent.gradNorms();
            const auto pn = agent.paramNorms();
            std::cout << "  [step " << step << "] 参数梯度范数 / 数据范数:\n";
            for (std::size_t k = 0; k < gn.size(); ++k) {
                std::cout << "    " << std::setw(8) << names[k] << "  grad "
                          << std::setw(12) << gn[k] << "   data " << pn[k] << "\n";
            }
        }

        if ((step + 1) % 300 == 0) {
            std::cout << " step " << std::setw(5) << (step + 1) << "  MSE " << loss
                      << std::endl;
        }
    }

    std::cout << "\n初始 MSE " << first_loss << " -> 最终 MSE " << last_loss << "\n";
    const bool learned = last_loss < first_loss * 0.5;
    std::cout << "结论: " << (learned ? "网络与反向传播可用（损失显著下降）"
                                      : "网络未能拟合专家动作，问题在网络或反传路径")
              << std::endl;
    return learned ? 0 : 1;
}
