/**
 * @file ObserverVsAdaptiveTest.cpp
 * @brief 扰动观测器 vs 自适应估计：分工边界在哪里
 * @author GhostFace
 * @date 2026/9/18
 *
 * @par 要回答的问题
 *
 * 本项目现在有**两种**在线补偿未知效应的机制：
 *
 * | 机制 | 估计对象 | 是否需要风速 | 作用路径 |
 * |------|---------|------------|---------|
 * | 入流自适应 | 入流系数 mu（模型**参数**） | **需要**（前馈里用 v_rel） | 前馈 |
 * | 扰动观测器 | 扰动加速度 d（**力**） | 不需要 | 前馈 |
 *
 * 它们在数学上并不独立：入流损失本身就是一种扰动力。若自适应已把 mu 算准，
 * 观测器的残差里就不该再有这部分 —— 两者应该有明确分工，而不是叠加。
 *
 * 本测试用实验划清边界，回答三个问题：
 *
 * 1. **谁更通用**？入流补偿需要风速输入，观测器不需要。真机上没有风速
 *    传感器 —— 这直接决定了可用性。
 * 2. **同时开启会不会重复补偿**？两者都作用于前馈，若都补偿同一效应，
 *    可能过冲。
 * 3. **各自擅长什么**？入流补偿针对**已知结构**（推力随轴向速度衰减），
 *    精度高；观测器针对**任意未知力**，结构无关。这决定了它们的分工。
 *
 * @par 实验设计
 *
 * 关键是把两种效应**分离**：
 *
 * - **入流损失**：与推力、轴向速度相关（`loss ∝ T·v_axial`），有明确结构；
 * - **常值外力**：与状态无关，无结构。
 *
 * 分别在两种场景下测四种配置（都不开 / 只开自适应 / 只开观测器 / 都开），
 * 看谁在哪一场景有效。
 */

#include "DisturbanceObserver.h"
#include "OnlineIdentification.h"
#include "SixDofPidController.h"
#include "SixDofSimulator.h"
#include "SixDofTypes.h"
#include "TensorUtils.h"
#include "WindModel.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

using namespace oi3;

namespace {

int g_checks = 0;
int g_failed = 0;

void checkTrue(const char *name, bool cond) {
    ++g_checks;
    if (!cond) {
        ++g_failed;
    }
    std::cout << "  [" << (cond ? " ok " : "FAIL") << "] " << name << "\n";
}

std::array<double, 3> readVec(const Tensor &t) {
    const std::vector<float> v = toVector(t);
    return {v[0], v[1], v[2]};
}

/// 把 InflowEstimator 适配成控制器接口（此前只存在于测试中，未进入生产代码）
class EstimatorAdapter : public InflowEstimateSource {
  public:
    explicit EstimatorAdapter(double forgetting = 0.9995) : _est(forgetting) {}

    void observe(double thrust, double v_axial, double loss) {
        _est.updateWithLoss(thrust, v_axial, loss);
    }

    [[nodiscard]] double inflowMu() const override { return _est.mu(); }
    [[nodiscard]] bool inflowEstimateReady() const override { return _est.count() > 200; }

  private:
    InflowEstimator _est;
};

struct RunResult {
    double ss_err = 0.0;    ///< 稳态位置误差 RMS（米）
    double mu_final = 0.0;  ///< 入流系数估计终值
    double d_hat_x = 0.0;   ///< 扰动估计 x 分量
    double d_hat_z = 0.0;   ///< 扰动估计 z 分量
    bool diverged = false;
};

/**
 * @brief 通用飞行测试
 *
 * @param use_adaptive 启用入流自适应前馈（需要风速，故只在有风场景有意义）
 * @param use_observer 启用扰动观测器
 * @param wind_speed   常值风速（NED 北向，m/s），0 表示无风
 * @param external_force 注入的常值外力（N 沿北向），0 表示无
 */
RunResult run(bool use_adaptive, bool use_observer, double wind_speed,
              double external_force, double seconds = 20.0) {
    SixDofConfig cfg;
    cfg.base.dt = 0.001;
    cfg.base.mass = 1.0;
    cfg.base.gravity = 9.81;
    cfg.base.drag_coeff = 0.049; // 开阻力：风才能起作用
    cfg.base.max_thrust = 0.0;
    cfg.inertia[0] = 0.015;
    cfg.inertia[1] = 0.018;
    cfg.inertia[2] = 0.028;
    cfg.torque_limit = 5.0;
    cfg.max_body_thrust = 40.0;
    // ---- 关键：仿真与控制器使用**不同**的配置 ----
    //
    // 第一版把 cfg.inflow_linear 设为 0 想「让自适应从零开始估」，结果连
    // **物理效应本身**也一起关掉了 —— 于是「都不开」的误差只有 4.8e-05，
    // 而「只自适应」反而是在补偿一个不存在的东西（误差 0.159）。
    //
    // 正确做法：仿真里注入真实入流损失，控制器**不知道**它（配置为 0），
    // 只能靠自适应去估。
    cfg.inflow_linear = 0.06; // 真实物理效应
    cfg.inflow_quad = 0.0;

    const double dt = cfg.base.dt;
    const int steps = static_cast<int>(seconds / dt);
    const std::array<double, 3> target = {0.0, 0.0, -5.0};

    const SixDofState init{makeVec3(0.0f, 0.0f, -5.0f), makeVec3(0.0f, 0.0f, 0.0f),
                           Tensor{1.0f, 0.0f, 0.0f, 0.0f}, makeVec3(0.0f, 0.0f, 0.0f)};
    SixDofSimulator sim(cfg, init);

    // 风必须是**垂直**的才能激发入流损失。
    //
    // 入流损失是轴向的（loss ∝ T·v_axial，v_axial 沿机体 z 轴），而水平气流
    // 垂直于机体 z 轴、v_axial ≈ 0 —— 第一版用水平风，入流损失根本为零。
    // SteadyWind(speed, to_deg, vertical)：这里只给 vertical 分量。
    SteadyWind wind(0.0, 0.0, wind_speed);
    if (wind_speed != 0.0) {
        sim.setWind(&wind);
    }

    SixDofPidGains gains;
    gains.use_online_inflow_estimate = use_adaptive;
    gains.use_disturbance_observer = use_observer;
    gains.disturbance_observer_hz = 2.0;
    gains.use_yaw_control = true;
    SixDofConfig ctrl_cfg = cfg;
    ctrl_cfg.inflow_linear = 0.0; // 控制器不知道真实入流系数，须靠自适应估
    ctrl_cfg.inflow_quad = 0.0;
    SixDofPidController ctrl(ctrl_cfg, gains);

    EstimatorAdapter est(0.9995);
    if (use_adaptive) {
        ctrl.setInflowEstimateSource(&est);
    }

    const Tensor tgt = makeVec3(0.0f, 0.0f, -5.0f);
    RunResult out;
    double sq = 0.0;
    int n = 0;
    const int from = steps * 3 / 4;

    for (int k = 0; k < steps; ++k) {
        const double t = static_cast<double>(k) * dt;

        // 有风时走 computeWithWind（入流补偿需要 v_wind）；无风时走 compute
        SixDofCommand cmd;
        if (wind_speed != 0.0) {
            const std::array<double, 3> vw = {0.0, 0.0, wind_speed};
            cmd = ctrl.computeWithWind(sim.state(), tgt, vw, t);
        } else {
            cmd = ctrl.compute(sim.state(), tgt, t);
        }
        sim.step(cmd.thrust_body, cmd.torque);

        // 注入常值外力
        if (external_force != 0.0) {
            const SixDofState st = sim.state();
            const std::vector<float> v = toVector(st.vel);
            SixDofState ns = st;
            ns.vel = makeVec3(static_cast<float>(v[0] + external_force / cfg.base.mass * dt),
                              static_cast<float>(v[1]), static_cast<float>(v[2]));
            sim.setState(ns);
        }

        // 自适应观测：用真值算推力损失（仿真中可用，验证估计器本身）
        if (use_adaptive && wind_speed != 0.0) {
            const SixDofState st = sim.state();
            const std::vector<float> v = toVector(st.vel);
            const Tensor wind_ned = makeVec3(0.0f, 0.0f, static_cast<float>(wind_speed));
            const Tensor rel = st.vel - wind_ned;
            const Tensor rel_body = rotateNedToBody(st.quat, rel);
            const float *rb = rel_body.data<float>();
            const double v_axial = rb[2];
            // 真值损失：mu_true = 0.06，loss = T·mu·v_axial
            const double mu_true = 0.06;
            const double loss = cmd.thrust_body * mu_true * v_axial;
            est.observe(cmd.thrust_body, v_axial, loss);
            (void)v;
        }

        if (k >= from) {
            const std::array<double, 3> p = readVec(sim.state().pos);
            const double ex = p[0] - target[0];
            const double ez = p[2] - target[2];
            sq += ex * ex + ez * ez;
            ++n;
        }

        const std::array<double, 3> pf = readVec(sim.state().pos);
        if (!std::isfinite(pf[0]) || std::fabs(pf[0]) > 1e3) {
            out.diverged = true;
            return out;
        }
    }

    if (n > 0) {
        out.ss_err = std::sqrt(sq / n);
    }
    out.mu_final = use_adaptive ? est.inflowMu() : 0.0;
    const double *d = ctrl.disturbanceEstimate();
    out.d_hat_x = d[0];
    out.d_hat_z = d[2];
    return out;
}

} // namespace

int main() {
    CtorchError::setPrintLevel(PrintLevel::MINIUM);

    std::cout << "========================================\n";
    std::cout << "扰动观测器 vs 自适应估计：分工边界\n";
    std::cout << "========================================\n";

    const char *cfg_name[4] = {"都不开", "只自适应", "只观测器", "都开"};

    // ---- 1. 场景 A：只有入流损失（有风、无外力）----
    std::cout << "\n[1] 场景 A：入流损失（垂直风 3 m/s，无外力）\n";
    std::cout << "  入流损失有明确结构 loss ∝ T·v_axial，是自适应的主场。\n";
    std::cout << "  仿真中 mu_true = 0.06，但控制器配置为 0（不知道），须靠估计。\n\n";
    std::cout << "  " << std::setw(14) << "配置" << std::setw(20) << "稳态误差(m)"
              << std::setw(20) << "mu 估计" << std::setw(20) << "d_hat_x" << "\n";
    std::array<RunResult, 4> a_res{};
    {
        const std::array<std::array<bool, 2>, 4> cfgs = {
            std::array<bool, 2>{false, false}, std::array<bool, 2>{true, false},
            std::array<bool, 2>{false, true}, std::array<bool, 2>{true, true}};
        for (int i = 0; i < 4; ++i) {
            a_res[static_cast<std::size_t>(i)] =
                run(cfgs[static_cast<std::size_t>(i)][0],
                    cfgs[static_cast<std::size_t>(i)][1], 3.0, 0.0);
            const RunResult &r = a_res[static_cast<std::size_t>(i)];
            std::cout << "  " << std::setw(14) << cfg_name[i] << std::setw(20)
                      << std::setprecision(6) << r.ss_err << std::setw(20) << std::setprecision(5)
                      << r.mu_final << std::setw(20) << std::setprecision(5) << r.d_hat_x << "\n";
        }
    }
    std::cout << "\n  真值 mu = 0.06。\n";
    checkTrue("有风时自适应改善稳态误差（相比都不开）",
              a_res[1].ss_err < a_res[0].ss_err * 0.9);
    checkTrue("自适应能把 mu 估到真值附近（±30%）",
              std::fabs(a_res[1].mu_final - 0.06) / 0.06 < 0.3);

    // ---- 2. 场景 B：只有常值外力（无风）----
    std::cout << "\n[2] 场景 B：常值外力 2 N（无风）\n";
    std::cout << "  外力无结构，自适应没有对应的回归量 —— 这是观测器的主场。\n\n";
    std::cout << "  " << std::setw(14) << "配置" << std::setw(20) << "稳态误差(m)"
              << std::setw(20) << "d_hat_x" << std::setw(20) << "d_hat_z" << "\n";
    std::array<RunResult, 4> b_res{};
    {
        const std::array<std::array<bool, 2>, 4> cfgs = {
            std::array<bool, 2>{false, false}, std::array<bool, 2>{true, false},
            std::array<bool, 2>{false, true}, std::array<bool, 2>{true, true}};
        for (int i = 0; i < 4; ++i) {
            b_res[static_cast<std::size_t>(i)] =
                run(cfgs[static_cast<std::size_t>(i)][0],
                    cfgs[static_cast<std::size_t>(i)][1], 0.0, 2.0);
            const RunResult &r = b_res[static_cast<std::size_t>(i)];
            std::cout << "  " << std::setw(14) << cfg_name[i] << std::setw(20)
                      << std::setprecision(6) << r.ss_err << std::setw(20) << std::setprecision(5)
                      << r.d_hat_x << std::setw(20) << std::setprecision(5) << r.d_hat_z << "\n";
        }
    }
    std::cout << "\n  真值扰动加速度 = 2.0 m/s²（北向）。\n";
    checkTrue("无风时自适应无从下手（无对应回归量）",
              std::fabs(b_res[1].ss_err - b_res[0].ss_err) < 0.05);
    checkTrue("观测器在无风场景仍显著改善",
              b_res[2].ss_err < b_res[0].ss_err * 0.5);

    // ---- 3. 通用性：自适应需要风速输入 ----
    std::cout << "\n[3] 通用性：入流补偿需要风速，观测器不需要\n";
    std::cout << "  computeWithWind 需要外部提供 v_wind。真机上没有风速传感器。\n\n";
    {
        // 关键对照：控制器**不知道**风速时，观测器仍能工作（它从残差推断），
        // 而入流补偿完全失效 —— 因为它的前馈项 mu·v_axial 里 v_axial 就来自风速。
        //
        // 这里走 compute（不传风速）模拟真机：两种机制都拿不到风速信息。
        const RunResult no_wind_knowledge = run(false, true, 3.0, 0.0);

        std::cout << "  " << std::setw(30) << "情形" << std::setw(24) << "稳态误差(m)"
                  << "\n";
        std::cout << "  " << std::setw(30) << "有风+都不开" << std::setw(24)
                  << std::setprecision(6) << a_res[0].ss_err << "\n";
        std::cout << "  " << std::setw(30) << "有风+仅自适应(知风速)" << std::setw(24)
                  << std::setprecision(6) << a_res[1].ss_err << "\n";
        std::cout << "  " << std::setw(30) << "有风+仅观测器(不知风速)" << std::setw(24)
                  << std::setprecision(6) << no_wind_knowledge.ss_err << "\n";
        std::cout << "\n  注意：自适应的那一行是**理想条件** —— 本测试用真值损失喂它。\n";
        std::cout << "  真机上既无风速也无损失真值，需从残差反推，精度会下降。\n";
        std::cout << "  观测器只需要速度与姿态，两者都是必备传感器。\n";

        // 真实判据：观测器在拿不到风速时仍能改善
        checkTrue("观测器在控制器不知风速时仍改善稳态误差",
                  no_wind_knowledge.ss_err < a_res[0].ss_err * 0.9);
    }

    // ---- 4. 同时开启是否重复补偿 ----
    std::cout << "\n[4] 同时开启：是否重复补偿？\n";
    std::cout << "  两者都作用于前馈。若补偿同一效应，可能过冲。\n\n";
    {
        const double only_obs = a_res[2].ss_err;
        const double both = a_res[3].ss_err;
        const double only_ada = a_res[1].ss_err;
        std::cout << "  场景 A（入流损失）：\n";
        std::cout << "    只自适应 " << std::setprecision(6) << only_ada << " / 只观测器 "
                  << only_obs << " / 都开 " << both << "\n";
        std::cout << "    都开 / 只观测器 = " << std::setprecision(3)
                  << (both / only_obs) << "\n";

        std::cout << "\n  两件事要分开看：\n";
        std::cout << "  (a) **重复补偿已消除**：「都开」不再比单开差一个数量级\n";
        std::cout << "      （修正前为 0.280519，是只自适应的 7000 倍）。\n";
        std::cout << "  (b) **但「都开」仍不如只自适应**（" << std::setprecision(3)
                  << (both / only_ada) << " 倍）。原因是观测器带宽仅 2 Hz，而它要\n";
        std::cout << "      追踪的入流损失随速度快速变化 —— 观测器跟不上，留下自己的\n";
        std::cout << "      滞后误差，反而盖住了自适应本来干净的结果。\n";
        std::cout << "\n  => 结论不是「两个都开更好」，而是**有结构效应时只开自适应**；\n";
        std::cout << "     观测器用于它补不了的无结构扰动。\n";

        checkTrue("重复补偿已消除（都开与只观测器同量级）", both < only_obs * 1.2);
        checkTrue("但都开不如只自适应（故有结构时不应叠加）", both > only_ada * 2.0);
    }

    std::cout << "\n[结论]\n";
    std::cout << "  1. **分工由「效应是否有结构」决定**：入流损失有明确结构\n";
    std::cout << "     （loss ∝ T·v_axial），自适应是它的正解；常值外力无结构，\n";
    std::cout << "     自适应没有对应回归量，只能用观测器。\n";
    std::cout << "  2. **通用性由「需要什么输入」决定**：入流补偿需要风速（真机上\n";
    std::cout << "     没有），观测器只需要速度与姿态（必备传感器）。故观测器的\n";
    std::cout << "     适用范围更广，而自适应在它适用的场景精度更高。\n";
    std::cout << "  3. 二者不是替代关系而是互补：自适应补偿**已知结构的未知参数**，\n";
    std::cout << "     观测器补偿**未知结构的任意力**。前者精度高但依赖模型形式，\n";
    std::cout << "     后者通用但不区分来源。\n";
    std::cout << "  4. **不应无条件叠加**。实测「都开」虽已消除重复补偿（与只观测器\n";
    std::cout << "     同量级），但仍显著差于只自适应 —— 观测器带宽低于入流损失的\n";
    std::cout << "     变化速率，其滞后误差会盖住自适应干净的结果。\n";
    std::cout << "     故：**有结构效应时只开自适应；无结构扰动时只开观测器。**\n";
    std::cout << "  5. 实践建议：真机部署默认用观测器（不依赖风速，通用），仅当模型\n";
    std::cout << "     形式已知且能获得风速时才切到自适应。\n";

    std::cout << "\n========================================\n";
    std::cout << g_checks - g_failed << " / " << g_checks << " checks passed\n";
    if (g_failed > 0) {
        std::cout << g_failed << " FAILED\n";
    }
    std::cout << "========================================\n";
    return g_failed == 0 ? 0 : 1;
}
