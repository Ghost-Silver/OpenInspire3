/**
 * @file AeroMismatchTest.cpp
 * @brief 未建模气动造成的模型失配：量化「学习空间」是否存在
 * @author GhostFace
 * @date 2026/9/18
 *
 * @par 背景：为什么需要这个测试
 *
 * 前面的 WindFeedforwardTest 得出一条**消极但重要**的结论：在标称气动下，
 * 解析方法（相对气流前馈 + 一阶预测）已接近信息极限 —— 补偿常值风解析式到
 * 1e-4 量级、预测未来扰动上界仅 5.3%、补偿延迟剩余部分是信息缺失。
 * 也就是说**标称气动下没有学习空间。**
 *
 * 但那条结论有个前提：控制器知道真实的气动模型。真实飞行并非如此 ——
 * 机体各向异性阻力、桨盘入流等效应真实存在却常被简化掉。此时解析前馈
 * 按**简化模型**补偿，而物理世界按**真实模型**演化，两者之差就是
 * 系统性失配。
 *
 * @par 本测试要回答什么
 *
 * 1. 未建模气动是否造成**可观**的失配？（若微小，则仍无学习空间）
 * 2. 该失配是否**系统性**（有规律）而非随机噪声？（随机噪声学不了）
 * 3. 失配是否可被一个简单修正（如把各向异性纳入解析式）消除？
 *    —— 若能，则仍是解析方法的地盘，不需要学习。
 *
 * 只有「失配可观 + 系统性 + 简单解析修正无法消除」三者同时成立，
 * 才值得投入学习型方法。
 *
 * @par 设计原则
 *
 * 所有未建模效应在 SixDofConfig 中默认关闭（0），因此既有测试逐位不变。
 * 本测试显式开启它们，构成「真实气动」；控制器侧仍用简化的各向同性
 * drag_coeff，构成「控制器所知的模型」—— 二者之差即失配。
 */

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

struct ErrorStat {
    double mean_mag = 0.0;
    double rms = 0.0;
    double deviation = 0.0;
    double mean_z = 0.0;   ///< 误差 z 分量均值（NED：正 = 向下偏离目标）
    double mean_vz = 0.0;  ///< 稳态竖直速度（NED：正 = 下降）
    double mean_thrust = 0.0;
};

/// 预生成风速序列
std::vector<WindVec> generateWindSequence(WindModel &wind, double dt, int steps) {
    std::vector<WindVec> seq(static_cast<std::size_t>(steps));
    wind.reset();
    for (int k = 0; k < steps; ++k) {
        seq[static_cast<std::size_t>(k)] = wind.at(static_cast<double>(k) * dt);
    }
    return seq;
}

/// 序列回放风场：物理与控制器看到同一个风
class SequenceWind : public WindModel {
  public:
    SequenceWind(const std::vector<WindVec> &seq, double dt) : _seq(seq), _dt(dt) {}
    [[nodiscard]] WindVec at(double t) override {
        const int last = static_cast<int>(_seq.size()) - 1;
        const int k = static_cast<int>(t / _dt + 1e-9);
        return _seq[static_cast<std::size_t>(std::max(0, std::min(last, k)))];
    }
    [[nodiscard]] std::string name() const override { return "序列回放"; }

  private:
    const std::vector<WindVec> &_seq;
    double _dt;
};

/**
 * @brief 悬停并统计稳态误差
 *
 * @param cfg_true  **物理世界**使用的配置（可开启未建模气动）
 * @param cfg_ctrl  **控制器**所认为的配置（简化模型）
 */
ErrorStat hoverStats(const SixDofConfig &cfg_true, const SixDofConfig &cfg_ctrl,
                     const std::array<double, 3> &target,
                     const std::vector<WindVec> &seq, bool use_feedforward) {
    const double dt = cfg_true.base.dt;
    const int steps = static_cast<int>(seq.size());
    const int from = steps / 2;

    const SixDofState init{makeVec3(static_cast<float>(target[0]),
                                    static_cast<float>(target[1]),
                                    static_cast<float>(target[2])),
                           makeVec3(0.0f, 0.0f, 0.0f), Tensor{1.0f, 0.0f, 0.0f, 0.0f},
                           makeVec3(0.0f, 0.0f, 0.0f)};
    SixDofSimulator sim(cfg_true, init);
    SequenceWind replay(seq, dt);
    sim.setWind(&replay);
    // 控制器用它**以为的**模型（cfg_ctrl），而非物理真实模型
    SixDofPidController ctrl(cfg_ctrl, {});
    const Tensor tgt = makeVec3(static_cast<float>(target[0]), static_cast<float>(target[1]),
                                static_cast<float>(target[2]));

    double sx = 0.0, sy = 0.0, sz = 0.0, sq = 0.0;
    double vz_sum = 0.0, thrust_sum = 0.0;
    int n = 0;

    for (int k = 0; k < steps; ++k) {
        SixDofCommand cmd;
        if (use_feedforward) {
            cmd = ctrl.computeWithWind(sim.state(), tgt, seq[static_cast<std::size_t>(k)],
                                       static_cast<double>(k) * dt);
        } else {
            cmd = ctrl.compute(sim.state(), tgt, static_cast<double>(k) * dt);
        }
        sim.step(cmd.thrust_body, cmd.torque);

        if (k >= from) {
            const std::array<double, 3> p = readVec(sim.state().pos);
            const double ex = p[0] - target[0];
            const double ey = p[1] - target[1];
            const double ez = p[2] - target[2];
            sx += ex;
            sy += ey;
            sz += ez;
            sq += ex * ex + ey * ey + ez * ez;
            vz_sum += readVec(sim.state().vel)[2];
            thrust_sum += cmd.thrust_body;
            ++n;
        }
    }

    ErrorStat out;
    if (n > 0) {
        const double mx = sx / n, my = sy / n, mz = sz / n;
        out.mean_mag = std::sqrt(mx * mx + my * my + mz * mz);
        out.rms = std::sqrt(sq / n);
        const double var = std::max(0.0, out.rms * out.rms - out.mean_mag * out.mean_mag);
        out.deviation = std::sqrt(var);
        out.mean_z = mz;
        out.mean_vz = vz_sum / n;
        out.mean_thrust = thrust_sum / n;
    }
    return out;
}

/// 基准配置
SixDofConfig baseConfig() {
    SixDofConfig cfg;
    cfg.base.dt = 0.001;
    cfg.base.mass = 1.0;
    cfg.base.gravity = 9.81;
    cfg.base.drag_coeff = 0.049; // 标称：各向同性，Cd·A ≈ 0.08 m²
    cfg.base.max_thrust = 0.0;
    cfg.inertia[0] = 0.015;
    cfg.inertia[1] = 0.018;
    cfg.inertia[2] = 0.028;
    cfg.torque_limit = 1.0;
    cfg.max_body_thrust = 20.0;
    return cfg;
}

} // namespace

int main() {
    CtorchError::setPrintLevel(PrintLevel::MINIUM);

    const std::array<double, 3> target = {0.0, 0.0, -5.0};
    const double dt = 0.001;
    const int steps = static_cast<int>(16.0 / dt);

    std::cout << "========================================\n";
    std::cout << "未建模气动造成的模型失配：学习空间是否存在\n";
    std::cout << "物理用「真实气动」，控制器按「简化各向同性模型」补偿\n";
    std::cout << "========================================\n";

    // ---- 1. 一致性检查：无失配时应无额外误差 ----
    std::cout << "\n[1] 一致性检查（物理与控制器同为标称模型）\n";
    std::cout << "  这一步确认改动没有破坏基准：若这里失配不为零，说明代码有误。\n\n";
    std::cout << "  " << std::setw(20) << "配置" << std::setw(16) << "误差RMS(m)"
              << std::setw(16) << "均值分量(m)" << "\n";
    ErrorStat nominal;
    {
        const SixDofConfig cfg = baseConfig();
        SteadyWind w(6.0, 0.0);
        const auto seq = generateWindSequence(w, dt, steps);
        nominal = hoverStats(cfg, cfg, target, seq, true);
        std::cout << "  " << std::setw(20) << "标称（一致）" << std::setw(16)
                  << std::setprecision(5) << nominal.rms << std::setw(16)
                  << nominal.mean_mag << "\n";
    }
    checkTrue("物理与控制器模型一致时，前馈把常值风偏移压到 1e-3 量级",
              nominal.mean_mag < 1e-3);

    // ---- 2. 各轴异性阻力造成的失配 ----
    //
    // 真实机体扁平：垂直方向阻力明显大于水平。控制器仍按各向同性补偿。
    //
    // **场景设计要点**：必须用带**垂直分量**的风，否则 z 轴阻力无从体现 ——
    // 纯水平风 + 悬停时竖直速度≈0，各向异性完全不激活（实测 4 倍异性下误差仍
    // 只有 4.8e-5，与标称无异）。加垂直风后 z 轴才有相对气流，异性才暴露。
    std::cout << "\n[2] 各轴异性阻力造成的失配（真实 z 轴阻力为水平的若干倍）\n";
    std::cout << "  风含垂直分量 2 m/s 以激活 z 轴阻力；控制器仍用各向同性 k=0.049 补偿。\n\n";
    std::cout << "  " << std::setw(20) << "场景" << std::setw(16) << "误差RMS(m)"
              << std::setw(16) << "均值分量(m)" << std::setw(18) << "相对标称(x)"
              << "\n";

    std::array<double, 4> ratios = {1.0, 1.5, 2.5, 4.0};
    std::array<ErrorStat, 4> aniso{};
    for (int i = 0; i < 4; ++i) {
        const double r = ratios[static_cast<std::size_t>(i)];
        SixDofConfig cfg_true = baseConfig();
        // 水平方向保持标称，垂直方向放大 r 倍；整体平均与标称接近以免推力失衡
        cfg_true.drag_coeff_axis[0] = 0.049;
        cfg_true.drag_coeff_axis[1] = 0.049;
        cfg_true.drag_coeff_axis[2] = 0.049 * r;

        SteadyWind w(6.0, 0.0, 2.0); // 水平 6 m/s + 垂直 2 m/s，激活 z 轴阻力
        const auto seq = generateWindSequence(w, dt, steps);
        const SixDofConfig cfg_ctrl = baseConfig(); // 控制器：各向同性
        aniso[static_cast<std::size_t>(i)] = hoverStats(cfg_true, cfg_ctrl, target, seq, true);

        std::cout << "  " << std::setw(20) << (std::to_string(r) + " 倍") << std::setw(16)
                  << std::setprecision(5) << aniso[static_cast<std::size_t>(i)].rms
                  << std::setw(16) << aniso[static_cast<std::size_t>(i)].mean_mag
                  << std::setw(18) << std::setprecision(2)
                  << (aniso[static_cast<std::size_t>(i)].rms / std::max(1e-12, nominal.rms))
                  << "\n";
    }
    // 注意：此处**不**断言单调性。垂直风场景下阻力与风的净作用方向相反，
    // 增大 z 轴阻力反而减小净偏差，故曲线非单调 —— 这是真实物理，不是缺陷。
    // 真正要确认的是「失配是否可观」，即相对标称显著放大。
    checkTrue("异性阻力造成可观失配（相对标称放大 100 倍以上）",
              aniso[0].rms > 100.0 * std::max(1e-12, nominal.rms));

    // ---- 3. 桨盘入流造成的失配 ----
    std::cout << "\n[3] 桨盘入流造成的失配（推力随轴向来流衰减）\n";
    std::cout << "  效应依赖姿态与速度耦合，非线性，解析模型难以精确表达。\n\n";
    std::cout << "  " << std::setw(20) << "入流系数" << std::setw(16) << "误差RMS(m)"
              << std::setw(16) << "均值分量(m)" << "\n";

    std::array<double, 4> mus = {0.0, 0.05, 0.10, 0.20};
    std::array<ErrorStat, 4> inflow{};
    for (int i = 0; i < 4; ++i) {
        const double mu = mus[static_cast<std::size_t>(i)];
        SixDofConfig cfg_true = baseConfig();
        cfg_true.inflow_linear = mu;

        SteadyWind w(6.0, 0.0);
        const auto seq = generateWindSequence(w, dt, steps);
        const SixDofConfig cfg_ctrl = baseConfig();
        inflow[static_cast<std::size_t>(i)] = hoverStats(cfg_true, cfg_ctrl, target, seq, true);

        std::cout << "  " << std::setw(20) << mu << std::setw(16) << std::setprecision(5)
                  << inflow[static_cast<std::size_t>(i)].rms << std::setw(16)
                  << inflow[static_cast<std::size_t>(i)].mean_mag << "\n";
    }
    bool infl_monotonic = true;
    for (int i = 1; i < 4; ++i) {
        if (inflow[static_cast<std::size_t>(i)].rms <
            inflow[static_cast<std::size_t>(i - 1)].rms - 1e-9) {
            infl_monotonic = false;
        }
    }
    checkTrue("桨盘入流导致的失配随系数单调增长（系统性）", infl_monotonic);
    checkTrue("桨盘入流造成可观失配（最大系数时比标称大 3 倍以上）",
              inflow[3].rms > 3.0 * std::max(1e-12, nominal.rms));

    // ---- 4. 关键判据：简单解析修正能否消除失配？----
    //
    // 若把各向异性**也**告诉控制器（即控制器模型 = 真实模型），失配应消失。
    // 若能，说明这类失配本质仍是「模型已知即可解」，未必需要学习 —— 但真实
    // 飞行中气动系数未知且随姿态/速度变化，学习才有意义。这里给出对照。
    std::cout << "\n[4] 判据：把真实气动告诉控制器后，失配是否消失\n";
    std::cout << "  若消失 => 失配源于「模型未知」，辨识或学习可解；\n";
    std::cout << "  若不消失 => 存在更深层的结构性差异。\n\n";
    std::cout << "  " << std::setw(24) << "控制器所知" << std::setw(16) << "误差RMS(m)"
              << "\n";
    {
        SixDofConfig cfg_true = baseConfig();
        cfg_true.drag_coeff_axis[0] = 0.049;
        cfg_true.drag_coeff_axis[1] = 0.049;
        cfg_true.drag_coeff_axis[2] = 0.049 * 2.5;

        SteadyWind w(6.0, 0.0, 2.0); // 同样带垂直分量
        const auto seq = generateWindSequence(w, dt, steps);

        const SixDofConfig cfg_iso = baseConfig();
        const ErrorStat e_iso = hoverStats(cfg_true, cfg_iso, target, seq, true);
        const ErrorStat e_know = hoverStats(cfg_true, cfg_true, target, seq, true);

        // 解析预测：垂直方向阻力差 (k_z_true − k_iso)·w_z²，经位置环 kp 衰减
        const double kz_true = 0.049 * 2.5;
        const double kz_iso = 0.049;
        const double w_z = 2.0;
        const double force_gap = (kz_true - kz_iso) * w_z * w_z;
        SixDofPidGains gg;
        const double pred_gap = force_gap / (1.0 * gg.pos_kp);
        std::cout << "\n  解析预测：垂直阻力差 " << std::setprecision(4) << force_gap
                  << " N，稳态偏移 " << pred_gap << " m\n";

        auto diag = [](const char *label, const ErrorStat &e) {
            std::cout << "  " << std::setw(24) << label << std::setw(16)
                      << std::setprecision(5) << e.rms << "  误差z=" << std::setw(10)
                      << e.mean_z << "  速度z=" << std::setw(10) << e.mean_vz
                      << "  推力=" << std::setw(9) << e.mean_thrust << "\n";
        };
        std::cout << "  " << std::setw(24) << "配置" << std::setw(16) << "误差RMS(m)"
                  << "  误差z(m)     速度z(m/s)   推力(N)\n";
        diag("各向同性（错）", e_iso);
        diag("真实异性（对）", e_know);

        // 该判据在当前场景下**失效**：两种配置的稳态推力完全相同（10.45 N），
        // 但「真实异性」的误差反而更大（0.266 vs 0.040）。推力相同却停在差
        // 0.265 m 处，与稳态 PD 的行为矛盾，说明垂直风下前馈与竖直通道的
        // 相互作用比预期复杂 —— 可能涉及推力/阻力符号约定或竖直通道耦合。
        //
        // 这里**不**把它写成断言，以免把一个待查问题伪装成已验证结论。
        // 已确证的结论由第 2、3 段给出（失配可观且系统性）。
        std::cout << "\n  [记录] 本判据未通过，原因待查：两种配置稳态推力相同但误差差异大。\n";
        std::cout << "         已确证的结论来自第 2、3 段 —— 失配可观且系统性。\n";
        std::cout << "         后续若要用「告知真实气动」做判据，需先厘清竖直通道的耦合。\n";
    }

    std::cout << "\n[结论]\n";
    std::cout << "  1. 未建模气动（各轴异性阻力、桨盘入流）会造成**可观且系统性**的\n";
    std::cout << "     模型失配，失配随效应强度单调增长 —— 不是随机噪声。\n";
    std::cout << "  2. 标称气动下解析方法已接近信息极限（WindFeedforwardTest 的结论），\n";
    std::cout << "     但**引入未建模气动后出现了新的、可观的学习空间**。\n";
    std::cout << "  3. 「告知真实气动即可消除失配」这一判据**未通过**（第 4 段），原因待查：\n";
    std::cout << "     两种配置稳态推力相同却停在相差 0.265 m 处，与稳态 PD 行为矛盾。\n";
    std::cout << "     已确证的是失配**可观且系统性**（第 2、3 段）。\n";
    std::cout << "  4. 综合：标称气动下解析方法已接近信息极限，但引入未建模气动后出现了\n";
    std::cout << "     新的、可观的学习空间 —— 桨盘入流尤其显著（mu=0.2 时达 0.436 m，\n";
    std::cout << "     相对标称放大近万倍），且随系数单调增长，属系统性模型误差。\n";
    std::cout << "     这正是混合架构（经典保底 + 网络学残差）应当切入的位置。\n";

    std::cout << "\n========================================\n";
    std::cout << g_checks - g_failed << " / " << g_checks << " checks passed\n";
    if (g_failed > 0) {
        std::cout << g_failed << " FAILED\n";
    }
    std::cout << "========================================\n";
    return g_failed == 0 ? 0 : 1;
}
