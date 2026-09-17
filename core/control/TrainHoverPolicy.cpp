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

/// 课程学习档位：按训练进度推进目标采样范围
///
/// 直接使用大范围时，随机初始化的策略几乎每回合都撞发散边界，所有样本回报趋同、
/// 优势失去区分度。从小范围起步、随策略成熟逐步扩大，是标准的课程学习做法。
struct CurriculumStage {
    double progress;     ///< 达到该训练进度后生效
    double target_range; ///< 目标位置各轴采样范围（±，米）
};

const std::vector<CurriculumStage> kCurriculum = {
    {0.00, 0.3}, {0.25, 0.6}, {0.50, 1.0}, {0.75, 1.5},
};

CurriculumStage stageAt(double progress) {
    CurriculumStage current = kCurriculum.front();
    for (const auto &stage : kCurriculum) {
        if (progress >= stage.progress) {
            current = stage;
        }
    }
    return current;
}

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
    // 起点为课程学习第一档，训练过程中由 kCurriculum 逐步扩大（见训练循环）
    env_cfg.target_range = 0.3;
    env_cfg.start_range = 0.05;
    env_cfg.abort_radius = 2.0;
    env_cfg.plant.dt = 0.001;
    env_cfg.plant.mass = 1.0;
    env_cfg.plant.gravity = 9.81;
    env_cfg.plant.drag_coeff = 0.3;
    // 推重比约 2:1（真实四旋翼的常见水平）。此前设为 12 N 时推重比仅 1.22:1，
    // 向上机动余量只剩 2.19 N 而向下有 4.9 N，动作空间严重不对称，
    // 目标在上方时策略几乎无计可施。
    env_cfg.plant.max_thrust = 20.0;
    // 回合时长 3 秒：PID 在同类任务上需 1.73 秒收敛，2 秒对策略过紧；
    // 而 4 秒会让策略退化为「前段收敛、后段漂移」的开环轨迹（实测稳态误差
    // 由 0.08m 劣化到 0.74m），取 3 秒为折中。
    env_cfg.episode_seconds = 3.0;
    env_cfg.control_decimation = 10; // 控制 100 Hz，仿真 1 kHz
    // 动作范围取 0.8 倍 m*g：与限幅（约 2 倍 m*g）共同界定可用的推力区间
    env_cfg.thrust_scale = 0.8;
    // 奖励缩放。关键在于：策略损失因优势被标准化而与奖励尺度无关，但价值损失
    // 正比于回报的平方 —— 回合回报越大，价值侧梯度越会通过共享主干淹没策略侧。
    // 实测 reward_scale=0.1 时回合回报约 -45，value_loss 达 2000 量级，而
    // policy_loss 只有 0.05，主干实际在拟合价值函数而非策略特征。
    // 取 0.01 把回报压到个位数，两者尺度才可比。
    env_cfg.reward_scale = 0.01;

    rl::PpoConfig ppo_cfg;
    ppo_cfg.obs_dim = HoverEnv::kObsDim;
    ppo_cfg.action_dim = HoverEnv::kActionDim;
    ppo_cfg.hidden_dim = 64;
    // 高斯策略的梯度含 1/σ 因子：σ 越小梯度越大。σ=0.05 时被放大 20 倍，
    // 配合 1e-3 学习率会一步就把策略推飞（实测 approx_kl 达 32）。
    ppo_cfg.learning_rate = 3e-4f;
    ppo_cfg.update_epochs = 4;
    ppo_cfg.batch_size = 64;
    // 初始 std ≈ 0.22。参照 PID 的实测动作幅值（|a| 均值仅 0.073），探索噪声若
    // 远大于信号本身，策略梯度会被噪声淹没；早期用 -0.5（std 0.61）时正是如此。
    ppo_cfg.init_log_std = -0.5f; // std ≈ 0.61，标准探索幅度
    // 去掉熵奖励：高斯策略下熵项是恒定推高 log_std 的力，会阻止策略收窄探索，
    // 实测 log_std 由 -0.5 一路涨到 -0.36、策略越来越随机，回报随之恶化。
    ppo_cfg.entropy_coef = 0.001f;
    ppo_cfg.train_log_std = true;

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

    // 评估任务落在课程学习终点档位的范围内（±1.5m，该任务误差 1.35m）。
    // 注意：评估任务的误差范围必须与训练配置一致 —— 曾用 1.35m 的任务去评测
    // 在 ±0.3m 上训练的策略，属分布外测试，并据此连续误判「策略发散」。
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

    // ---- 行为克隆预热：把策略校准到「维持悬停」这个工作点 ----
    //
    // 随机初始化时，ReLU 输出非负（均值约 0.5 而非 0），即使权重零均值，策略
    // 动作也会带有约 ±0.7 的标准差，折算成 ±5.5 N 的持续推力偏置 —— 无人机两秒
    // 内就会漂出发散边界。此时所有回合的奖励完全相同（都撞边界），优势失去区分度，
    // PPO 无从学习。
    //
    // 先用零动作做一段监督学习，把策略初始化到「输出悬停推力」，既避免上述问题，
    // 又不牺牲输出层权重尺度（缩权重会掐断隐层梯度，见 ContinuousPPO 中的说明）。
    {
        std::vector<std::vector<float>> warmup_obs;
        std::vector<std::array<float, 3>> warmup_act;
        const int warmup_samples = 256;
        warmup_obs.reserve(warmup_samples);
        warmup_act.reserve(warmup_samples);

        HoverEnv warm_env(env_cfg, 1234u);
        std::vector<float> o = warm_env.reset();
        for (int i = 0; i < warmup_samples; ++i) {
            warmup_obs.push_back(o);
            warmup_act.push_back({0.0f, 0.0f, 0.0f});
            o = warm_env.step({0.0f, 0.0f, 0.0f}).obs;
        }

        std::cout << "\n行为克隆预热（目标：输出零动作 = 维持悬停）...\n";
        for (int i = 0; i < 400; ++i) {
            const float loss = agent.behaviorCloneStep(warmup_obs, warmup_act);
            if ((i + 1) % 100 == 0) {
                std::cout << "  预热 step " << (i + 1) << "  MSE " << loss << std::endl;
            }
        }
        std::cout << "预热后策略动作均值应接近 0。" << std::endl;
    }

    printHeading("训练");
    const auto train_start = std::chrono::steady_clock::now();

    int success_count = 0;
    double recent_reward = 0.0;
    int recent_count = 0;
    double sample_seconds = 0.0;
    double update_seconds = 0.0;

    for (int ep = 0; ep < episodes; ++ep) {
        // 课程学习：随训练进度扩大目标范围，发散边界同步放开
        {
            const double progress = static_cast<double>(ep) / std::max(1, episodes);
            const CurriculumStage stage = stageAt(progress);
            if (std::abs(stage.target_range - env.targetRange()) > 1e-9) {
                env.setCurriculum(stage.target_range, 3.0 * stage.target_range);
                std::cout << "  [课程] 第 " << (ep + 1) << " 回合：目标范围扩至 ±"
                          << stage.target_range << " m，发散边界 "
                          << env.abortRadius() << " m" << std::endl;
            }
        }

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

            if ((ep + 1) % 50 == 0) {
                const rl::UpdateStats &s = agent.lastStats();
                std::cout << "          policy_loss " << std::setw(10) << s.policy_loss
                          << " | value_loss " << std::setw(12) << s.value_loss
                          << " | approx_kl " << std::setw(10) << s.approx_kl
                          << " | entropy " << s.entropy << "\n";
                std::cout << "          mean_return " << std::setw(10) << s.mean_return
                          << " | mean_value " << std::setw(10) << s.mean_value
                          << " | mean_advantage " << s.mean_advantage
                          << " | grad_norm " << s.grad_norm << "\n"
                          << "          ratio_mean " << s.ratio_mean
                          << " | ratio_max " << s.ratio_max
                          << " | logp_first " << s.logp_first
                          << " | oldlogp_first " << s.oldlogp_first << std::endl;
            }
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

    // ---- 参照：用 PID 在同一环境下跑出平均回报 ----
    // 这是判断奖励设计是否与控制目标一致的基准。若 PID（已知能把误差压到毫米级）
    // 的环境回报反而不高于未经训练的策略，说明奖励函数本身没有在奖励「朝目标飞」，
    // 那样无论怎么调 PPO 都学不出来。
    {
        HoverEnv ref_env(env_cfg, 7u);
        PidController ref_pid(env_cfg.plant);
        double pid_return = 0.0;
        const int ref_episodes = 20;

        for (int ep = 0; ep < ref_episodes; ++ep) {
            std::vector<float> obs = ref_env.reset();
            for (int t = 0; t < ref_env.stepsPerEpisode(); ++t) {
                const std::array<float, 3> tgt = ref_env.target();
                const Tensor thrust = ref_pid.computeThrust(
                    ref_env.state(), makeVec3(tgt[0], tgt[1], tgt[2]), 0.0);
                const std::vector<float> tf = toVector(thrust);
                const HoverEnv::StepResult r =
                    ref_env.step(ref_env.thrustToAction({tf[0], tf[1], tf[2]}));
                pid_return += r.reward;
                if (r.done) {
                    break;
                }
            }
        }
        std::cout << "\n参照：PID 在该环境下的平均回合回报 = "
                  << (pid_return / ref_episodes)
                  << "（对比训练日志中的策略回报）" << std::endl;
    }

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
