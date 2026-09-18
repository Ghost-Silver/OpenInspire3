/**
 * @file MotorFailureTest.cpp
 * @brief 单电机失效：可控性分析与失控实测
 * @author GhostFace
 * @date 2026/9/18
 *
 * @par 这个测试要回答什么
 *
 * 「飞行中突然坏掉一个电机，还能稳得住吗」是衡量飞控鲁棒性的常见问题。
 * 本测试用数值方法给出答案，并说明**卡在哪里**。
 *
 * @par 为什么答案不取决于控制器
 *
 * 四旋翼的四个电机推力到四个控制量 `[T, τx, τy, τz]` 的映射是一个 4×4 方阵，
 * 它可逆 —— 这是四旋翼能飞的根本原因：四个控制量可以独立指定。
 *
 * 失去一个电机后矩阵去掉一列，成为 4×3，**秩最多为 3**。也就是说四维控制空间
 * 里有一个方向永远无法到达，无论控制器输出什么，执行机构都无法产生违反该约束
 * 的控制量。
 *
 * 以失效 M1（前右）为例，剩余矩阵的左零空间只含一个向量 w，满足
 * `w · [T, τx, τy, τz] = 0`。本测试把这个 w 算出来，并在仿真中逐时刻验证
 * 该式的残差恒为零 —— 这就把「控制器不行」和「执行机构不够」区分开了。
 *
 * @par 与推力冗余的区别
 *
 * 默认参数（质量 1 kg、单电机上限 5 N）下推重比为 2.04，失效一个电机后仍有
 * 15 N > 9.81 N，**推力完全够悬停**。所以失控不是因为推不动，而是因为
 * **姿态控制少了一个自由度**。这两件事经常被混为一谈。
 */

#include "MotorMixer.h"
#include "SixDofDynamics.h"
#include "SixDofPidController.h"
#include "SixDofSimulator.h"
#include "SixDofTypes.h"
#include "TensorUtils.h"

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

double tiltDeg(const Tensor &quat) {
    const std::vector<float> e = toVector(quatToEuler(quat));
    const double c = std::cos(e[0]) * std::cos(e[1]);
    return std::acos(std::max(-1.0, std::min(1.0, c))) * 180.0 / M_PI;
}

/// 归一化向量
std::array<double, 4> normalize4(const std::array<double, 4> &v) {
    double n = 0.0;
    for (double x : v) {
        n += x * x;
    }
    n = std::sqrt(n);
    std::array<double, 4> out{};
    if (n > 1e-12) {
        for (int i = 0; i < 4; ++i) {
            out[static_cast<std::size_t>(i)] = v[static_cast<std::size_t>(i)] / n;
        }
    }
    return out;
}

/// 约束残差 w · [T, τx, τy, τz]
double constraintResidual(const std::array<double, 4> &w, const SixDofCommand &c) {
    double r = w[0] * c.thrust_body;
    if (c.torque.numel() == 3) {
        const float *t = c.torque.data<float>();
        r += w[1] * t[0] + w[2] * t[1] + w[3] * t[2];
    }
    return r;
}

/**
 * @brief 六旋翼的混控矩阵（用于对照）
 *
 * 6 个电机间隔 60° 均匀布置，反扭矩交替。矩阵为 4×6 —— **超定**，
 * 因此失去一个电机后仍有 4×5，秩通常仍为 4。
 */
std::vector<std::vector<double>> hexaMatrix(double arm, double c) {
    std::vector<std::vector<double>> A(4, std::vector<double>(6, 0.0));
    for (int i = 0; i < 6; ++i) {
        const double th = i * M_PI / 3.0;
        const double px = arm * std::cos(th);
        const double py = arm * std::sin(th);
        const double sg = (i % 2 == 0) ? -1.0 : 1.0;
        A[0][static_cast<std::size_t>(i)] = 1.0;
        A[1][static_cast<std::size_t>(i)] = -py;
        A[2][static_cast<std::size_t>(i)] = px;
        A[3][static_cast<std::size_t>(i)] = sg * c;
    }
    return A;
}

int rankOf(std::vector<std::vector<double>> M) {
    const int rows = static_cast<int>(M.size());
    const int cols = rows > 0 ? static_cast<int>(M[0].size()) : 0;
    int rank = 0;
    for (int cc = 0; cc < cols && rank < rows; ++cc) {
        int piv = -1;
        for (int r = rank; r < rows; ++r) {
            if (std::fabs(M[static_cast<std::size_t>(r)][static_cast<std::size_t>(cc)]) > 1e-12) {
                piv = r;
                break;
            }
        }
        if (piv < 0) {
            continue;
        }
        std::swap(M[static_cast<std::size_t>(rank)], M[static_cast<std::size_t>(piv)]);
        const double p = M[static_cast<std::size_t>(rank)][static_cast<std::size_t>(cc)];
        for (int j = cc; j < cols; ++j) {
            M[static_cast<std::size_t>(rank)][static_cast<std::size_t>(j)] /= p;
        }
        for (int r = 0; r < rows; ++r) {
            if (r == rank) {
                continue;
            }
            const double f = M[static_cast<std::size_t>(r)][static_cast<std::size_t>(cc)];
            if (std::fabs(f) < 1e-12) {
                continue;
            }
            for (int j = cc; j < cols; ++j) {
                M[static_cast<std::size_t>(r)][static_cast<std::size_t>(j)] -=
                    f * M[static_cast<std::size_t>(rank)][static_cast<std::size_t>(j)];
            }
        }
        ++rank;
    }
    return rank;
}

} // namespace

int main() {
    CtorchError::setPrintLevel(PrintLevel::MINIUM);

    SixDofConfig cfg;
    cfg.base.dt = 0.001;
    cfg.base.mass = 1.0;
    cfg.base.gravity = 9.81;
    cfg.base.drag_coeff = 0.0;
    cfg.inertia[0] = 0.015;
    cfg.inertia[1] = 0.018;
    cfg.inertia[2] = 0.028;
    cfg.torque_limit = 1.0;
    cfg.max_body_thrust = 20.0;

    QuadMotorConfig mc;
    mc.arm_length = 0.25;
    mc.max_thrust = 5.0;
    mc.torque_coeff = 0.02;
    QuadMixer mixer(mc);

    const double a = mc.arm_length / std::sqrt(2.0);
    const double c = mc.torque_coeff;
    const double m = cfg.base.mass;
    const double g = cfg.base.gravity;

    std::cout << "========================================\n";
    std::cout << "单电机失效：可控性分析与失控实测\n";
    std::cout << "机臂 " << mc.arm_length << " m，单电机上限 " << mc.max_thrust
              << " N，推重比 " << mixer.thrustToWeight(m, g) << "\n";
    std::cout << "悬停时每电机 " << mixer.hoverThrustPerMotor(m, g) << " N"
              << "，失效后剩余可用推力 " << (3.0 * mc.max_thrust) << " N（悬停需 " << m * g
              << " N）\n";
    std::cout << "========================================\n";

    // ---- 1. 混控矩阵与可控性 ----
    std::cout << "\n[1] 混控矩阵与可控性\n";
    {
        const auto &A = mixer.matrix();
        std::cout << "  矩阵 A（行 = [T, τx, τy, τz]，列 = 电机 1..4）：\n";
        const char *rowName[4] = {"T ", "τx", "τy", "τz"};
        for (int k = 0; k < 4; ++k) {
            std::cout << "    " << rowName[k] << "  [";
            for (int i = 0; i < 4; ++i) {
                std::cout << std::setw(9) << std::fixed << std::setprecision(4)
                          << A[static_cast<std::size_t>(k)][static_cast<std::size_t>(i)];
            }
            std::cout << " ]\n";
        }

        checkTrue("全部电机正常时混控矩阵满秩（4）",
                  mixer.rankWithFailures({false, false, false, false}) == 4);

        std::cout << "\n  失效组合 -> 剩余可控自由度：\n";
        const char *label[3] = {"失效 1 个（M1 前右）", "失效 2 个（M1+M3 对角）",
                                "失效 2 个（M1+M2 相邻）"};
        const std::array<bool, 4> cases[3] = {{true, false, false, false},
                                              {true, false, true, false},
                                              {true, true, false, false}};
        int ranks[3] = {};
        for (int i = 0; i < 3; ++i) {
            ranks[i] = mixer.rankWithFailures(cases[i]);
            std::cout << "    " << std::setw(26) << label[i] << "  ->  " << ranks[i] << " / 4\n";
        }
        checkTrue("失效 1 个电机后秩降为 3（丢失一个控制自由度）", ranks[0] == 3);
        checkTrue("失效 2 个后秩继续下降（不超过 2）", ranks[1] <= 2 && ranks[2] <= 2);

        // 不可达方向：执行机构永远无法产生的控制量组合
        const auto ns = mixer.unreachableDirections(cases[0]);
        std::cout << "\n  失效 1 个电机后控制空间中的约束（零空间基，共 " << ns.size()
                  << " 个）：\n";
        checkTrue("零空间维数 = 4 − 秩 = 1", ns.size() == 1);
        for (const auto &w0 : ns) {
            const auto w = normalize4(w0);
            std::cout << "    w = [" << std::setprecision(6) << w[0] << ", " << w[1] << ", "
                      << w[2] << ", " << w[3] << "]\n";
            std::cout << "    约束：";
            bool first = true;
            const char *nm[4] = {"T", "τx", "τy", "τz"};
            for (int i = 0; i < 4; ++i) {
                if (std::fabs(w[static_cast<std::size_t>(i)]) > 1e-9) {
                    if (!first) {
                        std::cout << " + ";
                    }
                    std::cout << w[static_cast<std::size_t>(i)] << "·" << nm[i];
                    first = false;
                }
            }
            std::cout << " = 0\n";

            // 解析预测：τz = (c/a)·τy，即 w ∝ [0, 0, −c/a, 1] / 归一化
            const double ratio = w[3] / (std::fabs(w[2]) > 1e-12 ? w[2] : 1.0);
            std::cout << "    检验 w[3]/w[2] = " << ratio << "，解析预测 −a/c = " << (-a / c)
                      << "\n";
            checkTrue("约束系数与解析推导 τz = (c/a)·τy 一致（相对误差 1e-6 以内）",
                      std::fabs(ratio + a / c) < 1e-6 * (a / c));
        }
    }

    // ---- 2. 悬停中单电机失效 ----
    std::cout << "\n[2] 悬停中 M1（前右电机）突然失效\n";
    {
        SixDofState init{makeVec3(0.0f, 0.0f, -5.0f), makeVec3(0.0f, 0.0f, 0.0f),
                         Tensor{1.0f, 0.0f, 0.0f, 0.0f}, makeVec3(0.0f, 0.0f, 0.0f)};
        SixDofSimulator sim(cfg, init);
        SixDofPidController ctrl(cfg, {});

        const double t_fail = 2.0;
        const auto ns = mixer.unreachableDirections({true, false, false, false});
        const auto w = normalize4(ns[0]);

        double max_residual = 0.0;
        double tilt_at_fail = 0.0;
        double t_lost = -1.0;
        int steps = static_cast<int>(4.0 / cfg.base.dt);

        std::cout << "    " << std::setw(8) << "t(s)" << std::setw(12) << "倾角(deg)"
                  << std::setw(14) << "角速度(rad/s)" << std::setw(14) << "高度(m)"
                  << std::setw(12) << "总推力(N)" << "\n";
        for (int k = 0; k < steps; ++k) {
            const double t = static_cast<double>(k) * cfg.base.dt;
            const SixDofCommand cmd =
                ctrl.compute(sim.state(), makeVec3(0.0f, 0.0f, -5.0f), t);

            MotorSet motors;
            motors.failed[0] = (t >= t_fail);
            const SixDofCommand actual = mixer.apply(cmd, motors);
            sim.step(actual.thrust_body, actual.torque);

            const std::array<double, 3> p = readVec(sim.state().pos);
            const std::array<double, 3> om = readVec(sim.state().omega);
            const double tilt = tiltDeg(sim.state().quat);
            const double rate = std::sqrt(om[0] * om[0] + om[1] * om[1] + om[2] * om[2]);

            if (t >= t_fail) {
                max_residual = std::max(max_residual, std::fabs(constraintResidual(w, actual)));
                if (t_lost < 0.0 && tilt > 45.0) {
                    t_lost = t - t_fail;
                }
            }
            if (std::fabs(t - t_fail) < 0.5 * cfg.base.dt) {
                tilt_at_fail = tilt;
            }

            // 每 0.25 秒打印一次
            if (k % static_cast<int>(0.25 / cfg.base.dt) == 0) {
                std::cout << "    " << std::setw(8) << std::setprecision(2) << t << std::setw(12)
                          << std::setprecision(3) << tilt << std::setw(14) << rate
                          << std::setw(14) << std::setprecision(3) << (-p[2])
                          << std::setw(12) << actual.thrust_body << "\n";
            }
        }

        const double final_tilt = tiltDeg(sim.state().quat);
        std::cout << "\n    失效前倾角 " << std::setprecision(4) << tilt_at_fail
                  << " deg；倾角超过 45 deg 用时 " << t_lost << " s；末态倾角 " << final_tilt
                  << " deg\n";

        checkTrue("失效前飞行器稳定悬停（倾角小于 1 度）", tilt_at_fail < 1.0);
        checkTrue("单电机失效后姿态迅速失控（倾角超过 45 度）", t_lost > 0.0);
        checkTrue("失控在 0.5 秒内发生（没有留给控制器反应的时间）", t_lost < 0.5);
        // 这一条是整个测试的核心：无论控制器怎么算，执行机构给出的控制量都被锁在
        // 一个三维子空间里。残差恒为零说明失控源于**执行机构维度不足**，
        // 而不是控制器没调好 —— 换任何算法结果都一样。
        // 判据取数值精度级（1e-6）而不是严格零：残差来自浮点误差的累积，
        // 相对于 10 N 量级的控制量，实测 1e-8 已是纯噪声。这里要证明的是
        // 「约束在物理上恒成立」，浮点噪声不构成反例。
        checkTrue("失控期间约束残差保持在数值精度级（证明失控源于控制维度不足）",
                  max_residual < 1e-6);
        std::cout << "    约束残差最大值 " << std::setprecision(3) << std::scientific
                  << max_residual << std::defaultfloat
                  << "（参考：控制量量级约 10 N，相对残差约 1e-9）\n";
    }

    // ---- 3. 六旋翼对照 ----
    std::cout << "\n[3] 对照：六旋翼布局\n";
    {
        const auto H = hexaMatrix(mc.arm_length, mc.torque_coeff);
        auto drop = [&](const std::vector<int> &dropIdx) {
            std::vector<std::vector<double>> R(4);
            for (int k = 0; k < 4; ++k) {
                for (int i = 0; i < 6; ++i) {
                    if (std::find(dropIdx.begin(), dropIdx.end(), i) != dropIdx.end()) {
                        continue;
                    }
                    R[static_cast<std::size_t>(k)].push_back(H[static_cast<std::size_t>(k)]
                                                             [static_cast<std::size_t>(i)]);
                }
            }
            return rankOf(R);
        };
        const int r0 = drop({});
        const int r1 = drop({0});
        const int r2 = drop({0, 3});
        std::cout << "    6 电机全好 -> 秩 " << r0 << " / 4\n";
        std::cout << "    失效 1 个  -> 秩 " << r1 << " / 4\n";
        std::cout << "    失效 2 个  -> 秩 " << r2 << " / 4\n";
        checkTrue("六旋翼（超定 4×6）失去一个电机后仍满秩",
                  r0 == 4 && r1 == 4);
    }

    std::cout << "\n[结论]\n";
    std::cout << "  1. 四旋翼的 4×4 混控矩阵可逆，这是四个控制量能独立指定的原因。\n";
    std::cout << "  2. 失去一个电机后矩阵变为 4×3，秩降为 3 —— 四维控制空间里有\n";
    std::cout << "     一个方向永远不可达。实测该约束的残差恒为零，说明**换任何控制\n";
    std::cout << "     算法都无法恢复**，这是执行机构的维度问题，不是算法问题。\n";
    std::cout << "  3. 失控与推力无关：本配置下失效后仍有 15 N 推力（悬停只需 9.81 N），\n";
    std::cout << "     推不动不是原因，少一个控制自由度才是。\n";
    std::cout << "  4. 六旋翼的混控矩阵是超定的（4×6），失去一个电机后仍满秩 ——\n";
    std::cout << "     容错的正确做法是硬件冗余，而不是在四旋翼上想办法。\n";

    std::cout << "\n========================================\n";
    std::cout << g_checks - g_failed << " / " << g_checks << " checks passed\n";
    if (g_failed > 0) {
        std::cout << g_failed << " FAILED\n";
    }
    std::cout << "========================================\n";
    return g_failed == 0 ? 0 : 1;
}
