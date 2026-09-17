/**
 * @file TrainHoverPolicy.cpp
 * @brief 用连续 PPO 训练定点悬停策略，并与 PID 基线对比
 * @author GhostFace
 * @date 2026/9/16
 *
 * 闭环的两条路径在此汇合：
 *   - PID：解析控制器，输出推力，作为对照基线
 *   - ContinuousPPO：学习型控制器，通过 NeuralController 适配到同一 Controller 接口
 * 两者在同一个任务与同一套飞行品质指标下评估。
 *
 * 用法：TrainHoverPolicy [episodes]
 */

#include "ClosedLoop.h"
#include "ContinuousPPO.h"
#include "HoverEnv.h"
#include "PidController.h"
#include "TensorUtils.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace oi3;

namespace {

double clampToUnit(double v) {
    return std::max(-1.0, std::min(1.0, v));
}

/**
 * @class NeuralController
 * @brief 把训练好的连续 PPO 策略适配为 Controller
 *
 * 观测的构造必须与 HoverEnv 完全一致，否则训练与部署之间存在分布偏移。
 * 为避免两处实现漂移，观测布局在 HoverEnv 中集中定义（kObsDim 与归一化尺度），
 * 此处按同一规则重算。
 */
class NeuralController : public Controller {
  public:
    NeuralController(rl::ContinuousPPO &agent, HoverEnvConfig env_cfg)
        : _agent(agent), _cfg(std::move(env_cfg)) {}

    [[nodiscard]] Tensor computeThrust(const DroneState &state, const Tensor &target,
                                       double /*time*/) override {
        const std::vector<float> pos = toVector(state.pos);
        const std::vector<float> vel = toVector(state.vel);
        const std::vector<float> tgt = toVector(target);

        const double pos_norm = std::max(1e-6, _cfg.target_range);
        const double vel_norm = std::max(1e-6, _cfg.vel_scale);

        std::vector<float> obs(static_cast<std::size_t>(HoverEnv::kObsDim), 0.0f);
        obs[0] = static_cast<float>((static_cast<double>(tgt[0]) - pos[0]) / pos_norm);
        obs[1] = static_cast<float>((static_cast<double>(tgt[1]) - pos[1]) / pos_norm);
        obs[2] = static_cast<float>((static_cast<double>(tgt[2]) - pos[2]) / pos_norm);
        obs[3] = static_cast<float>(vel[0] / vel_norm);
        obs[4] = static_cast<float>(vel[1] / vel_norm);
        obs[5] = static_cast<float>(vel[2] / vel_norm);

        const std::array<float, 3> a = _agent.actMean(obs);

        const double m = _cfg.plant.mass;
        const double g = _cfg.plant.gravity;
        const double scale = _cfg.thrust_scale * m * g;

        // 与 HoverEnv::step 相同的映射：以悬停推力为中心，动作限幅到 [-1,1]
        return makeVec3(static_cast<float>(clampToUnit(a[0]) * scale),
                        static_cast<float>(clampToUnit(a[1]) * scale),
                        static_cast<float>(-m * g + clampToUnit(a[2]) * scale));
    }

    [[nodiscard]] const char *name() const override { return "ContinuousPPO"; }

  private:
    rl::ContinuousPPO &_agent;
    HoverEnvConfig _cfg;
};

void printHeading(const char *title) {
    std::cout << "\n========================================\n";
    std::cout << title << "\n";
    std::cout << "========================================\n";
}

} // namespace

int main(int argc, char **argv) {
    CtorchError::setPrintLevel(PrintLevel::MINIUM);

    const int episodes = argc > 1 ? std::atoi(argv[1]) : 150;

    // ---- 被控对象：带阻力与推力限幅，非线性且受限 ----
    HoverEnvConfig env_cfg;
    env_cfg.plant.dt = 0.001;
    env_cfg.plant.mass = 1.0;
    env_cfg.plant.gravity = 9.81;
    env_cfg.plant.drag_coeff = 0.3;
    // 推重比约 2:1（真实四旋翼的常见水平）。此前设为 12 N 时推重比仅 1.22:1，
    // 向上机动余量只剩 2.19 N 而向下有 4.9 N，动作空间严重不对称，
    // 目标在上方时策略几乎无计可施。
    env_cfg.plant.max_thrust = 20.0;
    env_cfg.episode_seconds = 2.0;
    env_cfg.control_decimation = 10; // 控制 100 Hz，仿真 1 kHz
    // 动作范围取 0.8 倍 m*g：与限幅（约 2 倍 m*g）共同界定可用的推力区间
    env_cfg.thrust_scale = 0.8;
    // 奖励缩放到每步 O(0.1)：回合回报量级约数十，价值损失与策略损失尺度相当
    env_cfg.reward_scale = 0.1;

    rl::PpoConfig ppo_cfg;
    ppo_cfg.obs_dim = HoverEnv::kObsDim;
    ppo_cfg.action_dim = HoverEnv::kActionDim;
    ppo_cfg.hidden_dim = 64;
    ppo_cfg.learning_rate = 3e-4f;
    ppo_cfg.update_epochs = 4;
    ppo_cfg.batch_size = 64;
    ppo_cfg.init_log_std = -0.5f; // 初始 std ≈ 0.61，保证早期探索能覆盖有效动作区间

    printHeading("OpenInspire3 神经网络飞控训练（定点悬停）");
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "质量 " << env_cfg.plant.mass << " kg，重力 " << env_cfg.plant.gravity
              << " m/s^2\n";
    std::cout << "阻力系数 " << env_cfg.plant.drag_coeff << "，推力限幅 "
              << env_cfg.plant.max_thrust << " N（悬停需 "
              << env_cfg.plant.mass * env_cfg.plant.gravity << " N）\n";
    std::cout << "仿真步长 " << env_cfg.plant.dt << " s，控制抽稀 "
              << env_cfg.control_decimation << " -> 控制频率 "
              << (1.0 / (env_cfg.plant.dt * env_cfg.control_decimation)) << " Hz\n";
    std::cout << "回合时长 " << env_cfg.episode_seconds << " s（每回合 RL 步数 "
              << static_cast<int>(env_cfg.episode_seconds /
                                  (env_cfg.plant.dt * env_cfg.control_decimation))
              << "）\n";
    std::cout << "训练回合数 " << episodes << "\n";

    HoverEnv env(env_cfg, 42u);
    rl::ContinuousPPO agent(ppo_cfg, 12345u);

    FlightTask task;
    task.name = "定点悬停 (0,0,0) -> (0.8, -0.6, -0.9)";
    task.initial_pos = makeVec3(0.0f, 0.0f, 0.0f);
    task.initial_vel = makeVec3(0.0f, 0.0f, 0.0f);
    task.target = makeVec3(0.8f, -0.6f, -0.9f);
    task.duration = 5.0;
    task.tolerance = 0.05;

    printHeading("训练前评估（初始策略）");
    {
        NeuralController initial_policy(agent, env_cfg);
        const FlightTrace trace = runClosedLoop(env_cfg.plant, task, initial_policy);
        printMetrics(task, initial_policy, evaluateTrace(task, trace));
    }

    printHeading("训练");
    const auto train_start = std::chrono::steady_clock::now();

    int success_count = 0;
    double recent_reward = 0.0;
    int recent_count = 0;
    double sample_seconds = 0.0;
    double update_seconds = 0.0;

    for (int ep = 0; ep < episodes; ++ep) {
        std::vector<float> obs = env.reset();
        double episode_reward = 0.0;
        bool success = false;

        const auto sample_begin = std::chrono::steady_clock::now();
        for (int t = 0; t < env.stepsPerEpisode(); ++t) {
            const rl::ActionSample sample = agent.selectAction(obs);
            const HoverEnv::StepResult result = env.step(sample.action);

            rl::Transition transition;
            transition.obs = obs;
            transition.action = sample.action;
            transition.log_prob = sample.log_prob;
            transition.value = sample.value;
            transition.reward = result.reward;
            transition.done = result.done ? 1.0f : 0.0f;
            agent.store(transition);

            obs = result.obs;
            episode_reward += result.reward;

            if (result.done) {
                success = result.success;
                break;
            }
        }
        const auto sample_end = std::chrono::steady_clock::now();
        sample_seconds +=
            std::chrono::duration<double>(sample_end - sample_begin).count();

        const auto update_begin = std::chrono::steady_clock::now();
        agent.update();
        const auto update_end = std::chrono::steady_clock::now();
        update_seconds +=
            std::chrono::duration<double>(update_end - update_begin).count();

        // 数值崩溃早停：log_std 一旦变成 NaN 就再也无法恢复，继续跑只是浪费算力
        if (std::isnan(agent.meanLogStd())) {
            std::cerr << "\n[中止] 参数出现 NaN，训练在第 " << (ep + 1) << " 回合终止\n";
            return 2;
        }

        if (success) {
            ++success_count;
        }
        recent_reward += episode_reward;
        ++recent_count;

        if ((ep + 1) % 10 == 0) {
            std::cout << " Episode " << std::setw(4) << (ep + 1)
                      << " | 近10回合平均奖励 " << std::setw(10)
                      << (recent_reward / recent_count) << " | 累计到位率 "
                      << std::setw(6) << (100.0 * success_count / (ep + 1)) << "%"
                      << " | 末端误差 " << std::setw(8) << env.lastPositionError()
                      << " m | log_std " << std::setw(8) << agent.meanLogStd() << "\n";
            recent_reward = 0.0;
            recent_count = 0;
        }
    }

    const auto train_end = std::chrono::steady_clock::now();
    const auto train_seconds =
        std::chrono::duration_cast<std::chrono::seconds>(train_end - train_start).count();
    std::cout << "\n训练用时 " << train_seconds << " s，最终到位率 "
              << (100.0 * success_count / episodes) << "%\n";
    std::cout << "耗时分解: 采样 " << sample_seconds << " s ("
              << (100.0 * sample_seconds / (sample_seconds + update_seconds))
              << "%)，参数更新 " << update_seconds << " s ("
              << (100.0 * update_seconds / (sample_seconds + update_seconds))
              << "%)" << std::endl;

    // ---- 评估：与 PID 在同一任务、同一指标下对比 ----
    printHeading("闭环对比评估");

    PidController pid(env_cfg.plant);
    NeuralController neural(agent, env_cfg);

    const FlightTrace pid_trace = runClosedLoop(env_cfg.plant, task, pid);
    printMetrics(task, pid, evaluateTrace(task, pid_trace));

    std::cout << "\n";
    const FlightTrace nn_trace = runClosedLoop(env_cfg.plant, task, neural);
    printMetrics(task, neural, evaluateTrace(task, nn_trace));

    return 0;
}
