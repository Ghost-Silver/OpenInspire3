/**
 * @file test_ppo_grad.cpp
 * @brief 验证 PPO 迁移后确实能训练（梯度可回传、参数被更新）
 * @author GhostFace
 * @date 2026/9/16
 *
 * 4 月版本存在三处使训练实际无效的缺陷，本测试正是针对它们的回归：
 *
 *   1. autograd 接口不存在（AutoDiff / AutoDiffContext::Guard /
 *      Tensor::backward()）—— 代码从未编译成功。
 *   2. 动作概率由 float 构造：`Tensor new_log_prob(std::log(p))`，
 *      该张量与 policy 之间没有梯度路径，**策略梯度恒为零**；
 *      clamp 与 min 又用「拷贝 + 指针循环」实现，同样断图。
 *   3. Optimizer 持有参数副本（get_parameters 按值返回），
 *      step() 的更新写不回网络。
 *
 * 因此判据不是「能编译」「不崩溃」，而是：
 *      update() 之后网络参数的数值必须真的发生变化。
 *
 * ---------------------------------------------------------------------------
 * 附：CTorch 侧已修复的缺陷（本测试的 0c / 0d 为其回归用例）
 *
 *   `Tensor::transpose()` / `t()` 原先不参与 autograd：`AutoGrad/Nodes/` 下没有
 *   TransposeNode，且 transpose 内部是 `Tensor result(*this)`（拷贝构造，节点被
 *   替换为 GradAccumulator），梯度无法穿过转置；多个转置反向叠加后还会破坏内部
 *   状态，导致后续用例形状断言崩溃。
 *
 *   修复方式：新增 `TransposeNode`（转置自逆，反向即再转置一次），并在
 *   `Tensor::transpose()` 中补节点注册；前向仍是纯元数据操作、不经调度器，
 *   因此无需新增 op 枚举项，避开 op 顺序与 kCount 静态断言两条红线。
 * ---------------------------------------------------------------------------
 */

#include "AutoGrad.h"
#include "PPOAgent.h"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

namespace {

int g_checks = 0;
int g_failures = 0;

void expectTrue(bool cond, const std::string &what) {
    ++g_checks;
    if (cond) {
        std::cout << "  [ ok ] " << what << "\n";
    } else {
        ++g_failures;
        std::cout << "  [FAIL] " << what << "\n";
    }
}

/// 只报告不计数：用于记录 CTorch 侧已知缺陷的哨兵用例
void infoCheck(const char *label, bool ok, double value) {
    std::cout << "  [info] " << label << "  (max|grad| = " << value << ")"
              << (ok ? "" : "  <- CTorch 已知缺陷，见文末说明") << "\n";
}

void expectNear(double got, double want, double tol, const std::string &what) {
    ++g_checks;
    const double err = std::fabs(got - want);
    if (err <= tol) {
        std::cout << "  [ ok ] " << what << "  (|err| = " << err << ")\n";
    } else {
        ++g_failures;
        std::cout << "  [FAIL] " << what << ": got " << got << ", want " << want
                  << ", |err| = " << err << "\n";
    }
}

/// 参数张量的数值快照
std::vector<float> snapshot(const Tensor &t) {
    const float *p = t.data<float>();
    return std::vector<float>(p, p + static_cast<std::ptrdiff_t>(t.numel()));
}

double maxAbsDiff(const std::vector<float> &a, const std::vector<float> &b) {
    double m = 0.0;
    const std::size_t n = a.size() < b.size() ? a.size() : b.size();
    for (std::size_t i = 0; i < n; ++i) {
        m = std::max(m, static_cast<double>(std::fabs(a[i] - b[i])));
    }
    return m;
}

Experience makeExperience(const std::vector<float> &obs, int action, float log_prob,
                           float value, float reward, float done) {
    Tensor obs_t(ShapeTag{}, {1, obs.size()});
    float *op = obs_t.data<float>();
    for (std::size_t i = 0; i < obs.size(); ++i) {
        op[i] = obs[i];
    }

    Tensor action_t(ShapeTag{}, {1});
    action_t.data<float>()[0] = static_cast<float>(action);

    return Experience{obs_t, action_t, Tensor(log_prob), Tensor(value), Tensor(reward),
                      Tensor(done)};
}

// ============================ Test 0 ============================
// 逐层拆解，定位梯度的断点（PPO 网络前向由这几个构件组成）
double maxAbsGrad(const Tensor &t) {
    const Tensor g = t.grad();
    const float *gp = g.data<float>();
    double m = 0.0;
    for (std::size_t i = 0; i < g.numel(); ++i) {
        m = std::max(m, static_cast<double>(std::fabs(gp[i])));
    }
    return m;
}

void subCheck(const char *label, bool ok, double value) {
    ++g_checks;
    if (ok) {
        std::cout << "  [ ok ] " << label << "  (max|grad| = " << value << ")\n";
    } else {
        ++g_failures;
        std::cout << "  [FAIL] " << label << "  (max|grad| = " << value << ")\n";
    }
}

void testMinimalMatMulGradient() {
    std::cout << "\n[Test 0] Gradient breakpoint localisation\n";

    {   // 0a: 元素级加法 + sum
        Tensor a(ShapeTag{}, {3}); a.rand(); a.requires_grad(true);
        Tensor b(ShapeTag{}, {3}); b.rand();
        Tensor loss = (a + b).sum();
        if (loss.getRelatedNode()) AutoGrad::backward(loss.getRelatedNode(), false);
        const double g = maxAbsGrad(a);
        subCheck("0a  (a+b).sum()", g > 1e-9, g);
    }
    {   // 0b: matmul 不转置
        Tensor x(ShapeTag{}, {1, 3}); x.rand();
        Tensor w(ShapeTag{}, {3, 4}); w.rand(); w.requires_grad(true);
        Tensor loss = x.matmul(w).sum();
        if (loss.getRelatedNode()) AutoGrad::backward(loss.getRelatedNode(), false);
        const double g = maxAbsGrad(w);
        subCheck("0b  x.matmul(w).sum()", g > 1e-9, g);
    }
            {   // 0f: matmul + 加偏置
        Tensor x(ShapeTag{}, {1, 3}); x.rand();
        Tensor w(ShapeTag{}, {3, 4}); w.rand(); w.requires_grad(true);
        Tensor b(ShapeTag{}, {4}); b.rand();
        Tensor loss = (x.matmul(w) + b).sum();
        if (loss.getRelatedNode()) AutoGrad::backward(loss.getRelatedNode(), false);
        subCheck("0f  (x.matmul(w)+b).sum()", maxAbsGrad(w) > 1e-9, maxAbsGrad(w));
    }
    {   // 0g: + relu
        Tensor x(ShapeTag{}, {1, 3}); x.rand();
        Tensor w(ShapeTag{}, {3, 4}); w.rand(); w.requires_grad(true);
        Tensor b(ShapeTag{}, {4}); b.rand();
        Tensor loss = (x.matmul(w) + b).relu().sum();
        if (loss.getRelatedNode()) AutoGrad::backward(loss.getRelatedNode(), false);
        subCheck("0g  (...).relu().sum()", maxAbsGrad(w) > 1e-9, maxAbsGrad(w));
    }
    {   // 0h: 经 std::vector 持有的参数（PPO 的实际存放方式）
        std::vector<Tensor> params;
        params.push_back(Tensor(ShapeTag{}, {3, 4}));
        params[0].rand();
        params[0].requires_grad(true);
        Tensor x(ShapeTag{}, {1, 3}); x.rand();
        const Tensor &w = params[0];
        Tensor loss = x.matmul(w).sum();
        if (loss.getRelatedNode()) AutoGrad::backward(loss.getRelatedNode(), false);
        subCheck("0h  param via vector", maxAbsGrad(params[0]) > 1e-9, maxAbsGrad(params[0]));
    }
    {   // 0i: 与 0h 相同，但多加一层 relu（复刻 PPO 首层）
        std::vector<Tensor> params;
        params.push_back(Tensor(ShapeTag{}, {3, 8}));
        params[0].rand();
        params[0].requires_grad(true);
        Tensor b(ShapeTag{}, {8}); b.rand();
        Tensor x(ShapeTag{}, {1, 3}); x.rand();
        const Tensor &w = params[0];
        Tensor loss = (x.matmul(w) + b).relu().sum();
        if (loss.getRelatedNode()) AutoGrad::backward(loss.getRelatedNode(), false);
        subCheck("0i  vector param + relu", maxAbsGrad(params[0]) > 1e-9, maxAbsGrad(params[0]));
    }

    // --- 以下两项用于记录 CTorch 的 transpose autograd 缺陷，置后以免污染前序状态 ---
{   // 0c: 仅转置
        Tensor w(ShapeTag{}, {4, 3}); w.rand(); w.requires_grad(true);
        Tensor loss = w.t().sum();
        if (loss.getRelatedNode()) AutoGrad::backward(loss.getRelatedNode(), false);
        const double g = maxAbsGrad(w);
        subCheck("0c  w.t().sum()", g > 1e-9, g);
    }
{   // 0d: matmul + 转置（PPO 网络的实际写法）
        Tensor x(ShapeTag{}, {1, 3}); x.rand();
        Tensor w(ShapeTag{}, {4, 3}); w.rand(); w.requires_grad(true);
        Tensor loss = x.matmul(w.t()).sum();
        if (loss.getRelatedNode()) AutoGrad::backward(loss.getRelatedNode(), false);
        const double g = maxAbsGrad(w);
        subCheck("0d  x.matmul(w.t()).sum()", g > 1e-9, g);
    }
    {   // 0e: 两层网络形态（权重按 [in, out] 布局，与迁移后的 PPO 一致）
        Tensor x(ShapeTag{}, {1, 3}); x.rand();
        Tensor w1(ShapeTag{}, {3, 4}); w1.rand(); w1.requires_grad(true);
        Tensor w2(ShapeTag{}, {4, 5}); w2.rand(); w2.requires_grad(true);
        Tensor h = x.matmul(w1).relu();
        Tensor loss = h.matmul(w2).sum();
        if (loss.getRelatedNode()) AutoGrad::backward(loss.getRelatedNode(), false);
        subCheck("0e  two-layer grad -> w2", maxAbsGrad(w2) > 1e-9, maxAbsGrad(w2));
        subCheck("0e  two-layer grad -> w1", maxAbsGrad(w1) > 1e-9, maxAbsGrad(w1));
    }
}

// ============================ Test 1 ============================
// 初始化健康度：softmax 不应饱和。旧版用 U[0,1) 直接当权重，逐层放大后
// 策略分布退化为 one-hot（log_prob 恒为 0），梯度随之消失。
void testInitialisationHealth() {
    std::cout << "\n[Test 1] Initialisation health\n";

    const int obs_dim = 75, hidden = 64, action_dim = 7;
    PPOAgent agent(obs_dim, hidden, action_dim, 3e-4f);

    std::vector<float> obs(obs_dim, 0.5f);
    auto [action, log_prob, value] = agent.select_action(obs);

    std::cout << "    action = " << action << ", log_prob = " << log_prob
              << ", value = " << value << "\n";

    expectTrue(action >= 0 && action < action_dim, "action index within range");
    // 均匀策略下 log(1/7) ≈ -1.946；饱和时 log_prob 会趋近 0
    expectTrue(log_prob < -0.05f, "policy not saturated (log_prob notably < 0)");
    expectTrue(std::fabs(value) < 50.0f, "value output of sane magnitude");
}

// ============================ Test 2 ============================
// 梯度流转：update() 后各层参数都必须发生数值变化。
// 若策略梯度断链（旧缺陷 2）或 optimizer 写不回参数（旧缺陷 3），
// FC1_W / POLICY_W 将保持不变。
void testParametersActuallyUpdate() {
    std::cout << "\n[Test 2] Parameters update after training step\n";

    const int obs_dim = 8, hidden = 16, action_dim = 4;
    PPOAgent agent(obs_dim, hidden, action_dim, 0.05f);
    Network &net = agent.get_network();

    for (int i = 0; i < 4; ++i) {
        std::vector<float> obs(static_cast<std::size_t>(obs_dim),
                               0.1f * static_cast<float>(i + 1));
        auto [action, log_prob, value] = agent.select_action(obs);
        // 让旧策略的 log_prob 与「重新前向」得到的值略有差异，制造非退化的 ratio。
        //
        // 偏移量必须小：PPO 的 clip 区间是 [1-ε, 1+ε] = [0.8, 1.2]（ε=0.2）。
        // 若偏移过大（例如 -0.5，ratio ≈ 1.65），ratio 越界后 surr2 被 clip 成常数，
        // min(surr1, surr2) 取到被 clip 的那一支，梯度被**设计性地**截断为 0 ——
        // 那是 PPO 限制策略更新幅度的正确行为，不是梯度链断裂。
        // 取 0.05（ratio ≈ 1.05，落在区间内）才能真实检验策略梯度是否连通。
        agent.store_experience(
            makeExperience(obs, action, log_prob - 0.05f, value, 1.0f, 0.0f));
    }

    const std::vector<float> fc1_before = snapshot(net.param(Network::FC1_W));
    const std::vector<float> pol_before = snapshot(net.param(Network::POLICY_W));
    const std::vector<float> val_before = snapshot(net.param(Network::VALUE_W));

    agent.update();

    // 诊断：update() 之后参数上的梯度是否非零
    {
        const Tensor g = net.param(Network::FC1_W).grad();
        const float *gp = g.data<float>();
        double gm = 0.0;
        for (std::size_t i = 0; i < g.numel(); ++i) gm = std::max(gm, (double)std::fabs(gp[i]));
        std::cout << "    [diag] FC1_W.grad numel=" << g.numel() << ", max|g|=" << gm << "\n";
        std::cout << "    [diag] FC1_W node=" << (net.param(Network::FC1_W).getRelatedNode() ? "yes" : "none")
                  << ", requires_grad=" << net.param(Network::FC1_W).requires_grad() << "\n";
    }

    const double d_fc1 = maxAbsDiff(fc1_before, snapshot(net.param(Network::FC1_W)));
    const double d_pol = maxAbsDiff(pol_before, snapshot(net.param(Network::POLICY_W)));
    const double d_val = maxAbsDiff(val_before, snapshot(net.param(Network::VALUE_W)));

    std::cout << "    max |Δ| : FC1_W = " << d_fc1 << ", POLICY_W = " << d_pol
              << ", VALUE_W = " << d_val << "\n";

    expectTrue(d_fc1 > 1e-9, "FC1_W changed (trunk receives gradient)");
    expectTrue(d_pol > 1e-9, "POLICY_W changed (policy gradient path intact)");
    expectTrue(d_val > 1e-9, "VALUE_W changed (value loss path intact)");
    expectTrue(agent.bufferSize() == 0, "buffer cleared after update");
}

// ============================ Test 3 ============================
// 策略梯度确实沿动作方向：advantage 为正且只采样同一个动作时，
// 该动作的 log 概率应当上升。这是对「梯度来自 policy 而非恒零」的进一步确认。
void testPolicyPrefersRewardedAction() {
    std::cout << "\n[Test 3] Policy gradient direction\n";

    const int obs_dim = 6, hidden = 12, action_dim = 3;
    PPOAgent agent(obs_dim, hidden, action_dim, 0.05f);

    const std::vector<float> obs(static_cast<std::size_t>(obs_dim), 0.3f);
    const int target_action = 1;

    auto logProbOf = [&](int a) {
        Network &net = agent.get_network();
        Tensor obs_t(ShapeTag{}, {1, obs.size()});
        float *op = obs_t.data<float>();
        for (std::size_t i = 0; i < obs.size(); ++i) {
            op[i] = obs[i];
        }
        Tensor policy = net.get_policy(obs_t);
        return static_cast<double>(policy.data<float>()[a]);
    };

    const double p_before = logProbOf(target_action);

    // 反复采样同一动作并给予正奖励，PPO 应当提升该动作的概率
    for (int round = 0; round < 8; ++round) {
        auto [action, log_prob, value] = agent.select_action(obs);
        (void)action;
        // 人为固定在 target_action 上构造正优势
        agent.store_experience(
            makeExperience(obs, target_action, log_prob, value, 1.0f, 0.0f));
        agent.update();
    }

    const double p_after = logProbOf(target_action);
    std::cout << "    P(action=" << target_action << "): " << p_before << " -> "
              << p_after << "\n";

    expectTrue(p_after > p_before,
               "probability of the rewarded action increases");
}

} // namespace

int main() {
    CtorchError::setPrintLevel(PrintLevel::MINIUM);

    std::cout << "================================================\n";
    std::cout << "PPO migration regression tests\n";
    std::cout << "================================================\n";

    testMinimalMatMulGradient();
    testInitialisationHealth();
    testParametersActuallyUpdate();
    testPolicyPrefersRewardedAction();

    std::cout << "\n================================================\n";
    std::cout << (g_checks - g_failures) << " / " << g_checks << " checks passed\n";
    if (g_failures != 0) {
        std::cout << g_failures << " FAILED\n";
    }
    std::cout << "================================================\n";
    return g_failures == 0 ? 0 : 1;
}
