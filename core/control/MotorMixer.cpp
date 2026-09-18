/**
 * @file MotorMixer.cpp
 * @brief 四旋翼混控器实现
 * @author GhostFace
 * @date 2026/9/18
 *
 * 混控矩阵只有 4×4，直接给出逆矩阵的解析构造不如做一次数值求逆省事，而且数值
 * 求逆让「失去电机后秩降为 3」这件事可以用同一套代码算出来，不必另写一份推导。
 */

#include "MotorMixer.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace oi3 {

namespace {

constexpr double kTol = 1e-12;

using Mat = std::vector<std::vector<double>>;

/// 高斯-约当消元化为行最简形，返回秩，pivotCols 输出主元列
int toRref(Mat &M, std::vector<int> &pivotCols) {
    const int rows = static_cast<int>(M.size());
    const int cols = rows > 0 ? static_cast<int>(M[0].size()) : 0;
    int rank = 0;
    for (int c = 0; c < cols && rank < rows; ++c) {
        int piv = -1;
        for (int r = rank; r < rows; ++r) {
            if (std::fabs(M[r][c]) > kTol) {
                piv = r;
                break;
            }
        }
        if (piv < 0) {
            continue;
        }
        std::swap(M[rank], M[piv]);
        const double p = M[rank][c];
        for (int j = c; j < cols; ++j) {
            M[rank][j] /= p;
        }
        for (int r = 0; r < rows; ++r) {
            if (r == rank) {
                continue;
            }
            const double f = M[r][c];
            if (std::fabs(f) < kTol) {
                continue;
            }
            for (int j = c; j < cols; ++j) {
                M[r][j] -= f * M[rank][j];
            }
        }
        pivotCols.push_back(c);
        ++rank;
    }
    return rank;
}

/// 矩阵的秩
int matrixRank(Mat M) {
    std::vector<int> piv;
    return toRref(M, piv);
}

/// 矩阵的零空间基
std::vector<std::vector<double>> nullSpace(Mat M) {
    const int cols = M.empty() ? 0 : static_cast<int>(M[0].size());
    std::vector<int> pivotCols;
    const int rank = toRref(M, pivotCols);

    std::vector<int> freeCols;
    for (int c = 0; c < cols; ++c) {
        if (std::find(pivotCols.begin(), pivotCols.end(), c) == pivotCols.end()) {
            freeCols.push_back(c);
        }
    }

    std::vector<std::vector<double>> basis;
    for (int fc : freeCols) {
        std::vector<double> v(static_cast<std::size_t>(cols), 0.0);
        v[static_cast<std::size_t>(fc)] = 1.0;
        for (int r = 0; r < rank; ++r) {
            v[static_cast<std::size_t>(pivotCols[static_cast<std::size_t>(r)])] =
                -M[static_cast<std::size_t>(r)][static_cast<std::size_t>(fc)];
        }
        basis.push_back(v);
    }
    return basis;
}

/// 4×4 求逆（伴随矩阵法，4×4 规模下手写比通用消元更直接）
bool invert4(const std::array<std::array<double, 4>, 4> &m,
             std::array<std::array<double, 4>, 4> &out) {
    double a[4][8];
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            a[i][j] = m[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
            a[i][j + 4] = (i == j) ? 1.0 : 0.0;
        }
    }
    for (int c = 0; c < 4; ++c) {
        int piv = -1;
        for (int r = c; r < 4; ++r) {
            if (std::fabs(a[r][c]) > kTol) {
                piv = r;
                break;
            }
        }
        if (piv < 0) {
            return false;
        }
        for (int j = 0; j < 8; ++j) {
            std::swap(a[c][j], a[piv][j]);
        }
        const double p = a[c][c];
        for (int j = 0; j < 8; ++j) {
            a[c][j] /= p;
        }
        for (int r = 0; r < 4; ++r) {
            if (r == c) {
                continue;
            }
            const double f = a[r][c];
            if (std::fabs(f) < kTol) {
                continue;
            }
            for (int j = 0; j < 8; ++j) {
                a[r][j] -= f * a[c][j];
            }
        }
    }
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            out[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = a[i][j + 4];
        }
    }
    return true;
}

} // namespace

QuadMixer::QuadMixer(QuadMotorConfig cfg) : _cfg(cfg) {
    // X 型布局：电机位于 (±a, ±a)，a = arm_length/√2
    const double a = _cfg.arm_length / std::sqrt(2.0);
    //            M1(前右)  M2(后右)  M3(后左)  M4(前左)
    const double px[4] = {a, -a, -a, a};  // 机体系 x（向前为正）
    const double py[4] = {a, a, -a, -a};  // 机体系 y（向右为正）
    // 反扭矩符号：对角线上两个电机同向旋转，相邻反向
    const double sgn[4] = {-1.0, 1.0, -1.0, 1.0};
    const double c = _cfg.torque_coeff;

    // τ = r × F，推力沿 −z：τx = −y·f，τy = x·f
    for (int i = 0; i < 4; ++i) {
        _A[0][static_cast<std::size_t>(i)] = 1.0;
        _A[1][static_cast<std::size_t>(i)] = -py[i];
        _A[2][static_cast<std::size_t>(i)] = px[i];
        _A[3][static_cast<std::size_t>(i)] = sgn[i] * c;
    }
    invert4(_A, _Ainv);
}

MotorSet QuadMixer::mix(const SixDofCommand &cmd) const {
    double w[4] = {cmd.thrust_body, 0.0, 0.0, 0.0};
    if (cmd.torque.numel() == 3) {
        const float *tp = cmd.torque.data<float>();
        w[1] = tp[0];
        w[2] = tp[1];
        w[3] = tp[2];
    }

    MotorSet out;
    for (int i = 0; i < 4; ++i) {
        double f = 0.0;
        for (int k = 0; k < 4; ++k) {
            f += _Ainv[static_cast<std::size_t>(i)][static_cast<std::size_t>(k)] * w[k];
        }
        // 逐电机限幅：电机只能正转，且不超过最大推力。限幅在这里而不是在控制器里，
        // 因为它是执行机构的物理属性 —— 控制器无从知道分配后的结果是超限的。
        out.thrust[static_cast<std::size_t>(i)] =
            std::max(_cfg.min_thrust, std::min(_cfg.max_thrust, f));
    }
    return out;
}

SixDofCommand QuadMixer::unmix(const MotorSet &motors) const {
    const auto &f = motors.thrust;
    auto row = [&](int k) {
        const std::size_t kk = static_cast<std::size_t>(k);
        return _A[kk][0] * f[0] + _A[kk][1] * f[1] + _A[kk][2] * f[2] + _A[kk][3] * f[3];
    };

    SixDofCommand out;
    out.thrust_body = row(0);
    out.torque = makeVec3(static_cast<float>(row(1)), static_cast<float>(row(2)),
                          static_cast<float>(row(3)));
    return out;
}

SixDofCommand QuadMixer::apply(const SixDofCommand &cmd, MotorSet &motors) const {
    MotorSet m = mix(cmd);
    for (int i = 0; i < 4; ++i) {
        const auto idx = static_cast<std::size_t>(i);
        if (motors.failed[idx]) {
            m.thrust[idx] = 0.0; // 失效电机不产生推力，也不产生反扭矩
        }
    }
    m.failed = motors.failed;
    motors = m;
    return unmix(m);
}

int QuadMixer::rankWithFailures(const std::array<bool, 4> &failed) const {
    Mat B;
    for (int k = 0; k < 4; ++k) {
        std::vector<double> row;
        for (int i = 0; i < 4; ++i) {
            if (!failed[static_cast<std::size_t>(i)]) {
                row.push_back(_A[static_cast<std::size_t>(k)][static_cast<std::size_t>(i)]);
            }
        }
        if (row.empty()) {
            return 0; // 四个电机全失效
        }
        B.push_back(std::move(row));
    }
    return matrixRank(B);
}

std::vector<std::array<double, 4>>
QuadMixer::unreachableDirections(const std::array<bool, 4> &failed) const {
    std::vector<std::array<double, 4>> out;

    // 构造 A_eff（4 × 未失效电机数）
    Mat Ae;
    for (int k = 0; k < 4; ++k) {
        std::vector<double> row;
        for (int i = 0; i < 4; ++i) {
            if (!failed[static_cast<std::size_t>(i)]) {
                row.push_back(_A[static_cast<std::size_t>(k)][static_cast<std::size_t>(i)]);
            }
        }
        if (row.empty()) {
            return out;
        }
        Ae.push_back(std::move(row));
    }

    // 约束 w 满足 wᵀ·A_eff = 0，即 A_effᵀ·w = 0 —— 求 A_effᵀ 的零空间
    const int cols = static_cast<int>(Ae[0].size());
    Mat At(static_cast<std::size_t>(cols), std::vector<double>(4, 0.0));
    for (int k = 0; k < 4; ++k) {
        for (int i = 0; i < cols; ++i) {
            At[static_cast<std::size_t>(i)][static_cast<std::size_t>(k)] =
                Ae[static_cast<std::size_t>(k)][static_cast<std::size_t>(i)];
        }
    }

    for (const auto &v : nullSpace(At)) {
        std::array<double, 4> w{};
        for (int i = 0; i < 4; ++i) {
            w[static_cast<std::size_t>(i)] = v[static_cast<std::size_t>(i)];
        }
        out.push_back(w);
    }
    return out;
}

} // namespace oi3
