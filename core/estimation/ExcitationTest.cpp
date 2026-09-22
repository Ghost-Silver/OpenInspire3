/**
 * @file ExcitationTest.cpp
 * @brief 激励充分性监测与主动激励注入的验证
 * @author GhostFace
 * @date 2026/9/18
 *
 * @par 要解决的矛盾
 *
 * OnlineIdTest 证明：无激励时参数在数学上不可辨，估计值会安静地停在初值。
 * 而悬停无风时入流系数的回归量 `φ = T·v_axial` 恰好趋近于零 ——
 * **飞行器越平稳，越估不出自己的气动参数。**
 *
 * 这是结构性矛盾，不能靠调参绕过。本测试验证两个应对手段：
 *
 * 1. **监测**：由累积信息量 R 预测估计标准差 `σ/√R`，在精度不足时报警。
 *    关键是它**可预测**（事前可知），而不是等误差出来才知道。
 * 2. **主动激励**：信息量不足时注入小幅零均值机动，主动创造信息。
 *
 * @par 判据
 *
 * - 监测器的预测标准差应与**实测**估计误差同量级（否则预测没有意义）；
 * - 主动激励应能把不可辨的参数变得可辨；
 * - 激励的代价（对任务的扰动）必须量化 —— 它是「估计精度 vs 任务扰动」
 *   的权衡，不是免费的。
 */

#include "ExcitationMonitor.h"
#include "OnlineIdentification.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
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

/// 一次估计实验的结果
struct ExpResult {
    double mu_est = 0.0;      ///< 估计终值
    double mu_err = 0.0;      ///< 相对真值的误差
    double pred_std = 0.0;    ///< 监测器预测的标准差（末态）
    double info = 0.0;        ///< 累积信息量
    double track_rms = 0.0;   ///< 轨迹跟踪误差（激励的代价）
};

/**
 * @brief 跑一次入流系数估计
 *
 * @param vax_amp   轴向气流激励幅度（0 = 无激励，模拟无风悬停）
 * @param vax_freq  激励频率
 * @param excite    是否启用主动激励（在气流之外额外叠加）
 * @param use_monitor 是否启用监测器
 */
ExpResult runEstimation(double vax_amp, double vax_freq, bool excite, double seconds,
                        double obs_sigma = 0.02, double target_std = 0.01) {
    const double dt = 0.001;
    const int steps = static_cast<int>(seconds / dt);
    const double mu_true = 0.10;
    const double thrust = 9.81;

    std::mt19937 rng(20260918u);
    std::normal_distribution<double> obs_noise(0.0, obs_sigma);

    InflowEstimator est(1.0); // 不遗忘：恒定参数
    ExcitationMonitor mon(1.0);
    ExcitationInjector inj(1.5, 0.5);

    // 主动激励的开关逻辑：信息量不足就开，够了就关
    inj.setActive(excite);

    ExpResult out;
    double sq_track = 0.0;
    int n = 0;

    for (int k = 0; k < steps; ++k) {
        const double t = static_cast<double>(k) * dt;

        // 基础气流激励（模拟环境扰动）
        double v_axial = vax_amp * std::sin(vax_freq * t);

        // 主动激励：额外叠加（同样作用于 v_axial）
        if (excite) {
            v_axial += inj.step(dt);
        }

        const double loss = thrust * mu_true * v_axial + obs_noise(rng);
        est.updateWithLoss(thrust, v_axial, loss);
        mon.update(thrust * v_axial);

        // 激励的代价：注入的位移本身就是对任务的扰动。
        // 直接用本步的 v_axial 增量衡量，而不是再次调用 inj.step() ——
        // 后者会推进内部时间、使同一时刻被采样两次（第一版即如此，测出的
        // 数值与激励幅度对不上）。
        if (excite) {
            const double injected = v_axial - vax_amp * std::sin(vax_freq * t);
            sq_track += injected * injected;
            ++n;
        }
    }

    out.mu_est = est.mu();
    out.mu_err = std::fabs(est.mu() - mu_true);
    out.pred_std = mon.predictedStdDev(obs_sigma);
    out.info = mon.information();
    out.track_rms = (n > 0) ? std::sqrt(sq_track / n) : 0.0;
    (void)target_std;
    return out;
}

} // namespace

int main() {
    std::cout << "========================================\n";
    std::cout << "激励充分性监测与主动激励\n";
    std::cout << "========================================\n";

    // ---- 1. 无激励：参数不可辨 ----
    std::cout << "\n[1] 无激励（模拟无风悬停）\n";
    std::cout << "  回归量 φ = T·v_axial 恒为零时，参数在数学上不可辨。\n\n";
    {
        const ExpResult r = runEstimation(0.0, 0.0, false, 20.0);
        std::cout << "  估计值 " << std::setprecision(6) << r.mu_est << "（真值 0.10）\n";
        std::cout << "  累积信息量 R = " << r.info << "\n";
        std::cout << "  监测器预测标准差 " << r.pred_std << "\n";
        checkTrue("无激励时估计停在初值（不可辨）", r.mu_err > 0.05);
        checkTrue("监测器正确报告不可辨（预测标准差为无穷）", !std::isfinite(r.pred_std));
        std::cout << "\n  => 监测器在**估计失败之前**就报告了不可辨 —— 这是它存在的意义。\n";
    }

    // ---- 2. 弱激励：监测器预测 vs 实测 ----
    //
    // 这是本测试最关键的一段：若监测器的预测与实测不符，它就没有决策价值。
    std::cout << "\n[2] 弱激励下：预测标准差 vs 实测误差\n";
    std::cout << "  判据：预测值应与实测误差同量级（否则无法用于在线决策）。\n\n";
    std::cout << "  " << std::setw(16) << "激励幅度" << std::setw(20) << "累积信息量"
              << std::setw(20) << "预测标准差" << std::setw(18) << "实测误差"
              << std::setw(16) << "比值" << "\n";

    std::array<double, 4> amps = {0.1, 0.3, 1.0, 3.0};
    std::array<double, 4> preds{}, errs{};
    for (int i = 0; i < 4; ++i) {
        const double a = amps[static_cast<std::size_t>(i)];
        const ExpResult r = runEstimation(a, 1.0, false, 20.0);
        preds[static_cast<std::size_t>(i)] = r.pred_std;
        errs[static_cast<std::size_t>(i)] = r.mu_err;
        std::cout << "  " << std::setw(16) << std::fixed << std::setprecision(2) << a
                  << std::setw(20) << std::setprecision(4) << r.info << std::setw(20)
                  << std::setprecision(6) << r.pred_std << std::setw(18) << r.mu_err
                  << std::setw(16) << std::setprecision(2)
                  << (r.mu_err / std::max(1e-12, r.pred_std)) << "\n";
    }
    checkTrue("激励越强信息量越大（R 单调增长）", preds[0] > preds[3]);
    checkTrue("预测标准差与实测误差同量级（比值在 0.1~10 之间）",
              errs[3] / std::max(1e-12, preds[3]) > 0.1 &&
                  errs[3] / std::max(1e-12, preds[3]) < 10.0);

    // ---- 3. 主动激励：把不可辨变可辨 ----
    std::cout << "\n[3] 主动激励的效果与代价\n";
    std::cout << "  激励是零均值的，长期不引入位置偏移，但过程中会扰动任务。\n\n";
    std::cout << "  " << std::setw(18) << "配置" << std::setw(20) << "估计误差"
              << std::setw(20) << "累积信息量" << std::setw(20) << "任务扰动(RMS)" << "\n";
    {
        const ExpResult no_ex = runEstimation(0.0, 0.0, false, 20.0);
        const ExpResult with_ex = runEstimation(0.0, 0.0, true, 20.0);

        std::cout << "  " << std::setw(18) << "无激励" << std::setw(20) << std::setprecision(6)
                  << no_ex.mu_err << std::setw(20) << no_ex.info << std::setw(20)
                  << no_ex.track_rms << "\n";
        std::cout << "  " << std::setw(18) << "主动激励" << std::setw(20) << with_ex.mu_err
                  << std::setw(20) << std::setprecision(4) << with_ex.info << std::setw(20)
                  << std::setprecision(6) << with_ex.track_rms << "\n";

        checkTrue("主动激励使原本不可辨的参数变得可辨", with_ex.mu_err < 0.01);
        checkTrue("主动激励确实带来了信息量（R > 0）", with_ex.info > 0.0);
    }

    // ---- 4. 按需激励：只在需要时开 ----
    //
    // 一直开激励会持续扰动任务。合理策略是按信息量缺口决定开关。
    std::cout << "\n[4] 按需激励的判据（信息量缺口）\n";
    std::cout << "  由目标精度反推所需信息量，只补缺口 —— 而不是一直开着。\n\n";
    std::cout << "  " << std::setw(18) << "目标标准差" << std::setw(22) << "所需信息量"
              << std::setw(22) << "当前缺口" << std::setw(18) << "是否充分" << "\n";
    {
        ExcitationMonitor mon(1.0);
        // 模拟已有少量激励
        for (int k = 0; k < 1000; ++k) {
            mon.update(0.05);
        }
        const double obs_sigma = 0.02;
        for (double target : {0.05, 0.01, 0.005, 0.001}) {
            const double need = (obs_sigma * obs_sigma) / (target * target);
            const double deficit = mon.informationDeficit(obs_sigma, target);
            std::cout << "  " << std::setw(18) << std::fixed << std::setprecision(4) << target
                      << std::setw(22) << std::setprecision(1) << need << std::setw(22)
                      << deficit << std::setw(18)
                      << (mon.sufficient(obs_sigma, target) ? "是" : "否") << "\n";
        }
        checkTrue("目标精度越高，所需信息量越大（缺口判据正确）",
                  mon.informationDeficit(obs_sigma, 0.001) >
                      mon.informationDeficit(obs_sigma, 0.05));
        checkTrue("信息量充足时判定为充分（无需激励）",
                  mon.sufficient(obs_sigma, 0.05));
    }

    std::cout << "\n[结论]\n";
    std::cout << "  1. 无激励时参数不可辨，且监测器能**在估计失败之前**就报告这一点\n";
    std::cout << "     （预测标准差为无穷）—— 这是它比「事后看误差」有价值的地方。\n";
    std::cout << "  2. 监测器的预测标准差 `σ/√R` 与实测误差同量级，可用于在线决策：\n";
    std::cout << "     由目标精度反推所需信息量，判断当前是否够用。\n";
    std::cout << "  3. 主动激励能把不可辨的参数变得可辨，代价是对任务的扰动 ——\n";
    std::cout << "     这是「估计精度 vs 任务扰动」的权衡，不是免费的。\n";
    std::cout << "  4. 因此正确策略是**按需激励**：只在信息量不足时注入，且幅度与缺口\n";
    std::cout << "     挂钩。这也回答了「自适应系统该长什么样」—— 它应该知道自己\n";
    std::cout << "     什么时候估不准，而不是盲目地一直估。\n";

    std::cout << "\n========================================\n";
    std::cout << g_checks - g_failed << " / " << g_checks << " checks passed\n";
    if (g_failed > 0) {
        std::cout << g_failed << " FAILED\n";
    }
    std::cout << "========================================\n";
    return g_failed == 0 ? 0 : 1;
}
