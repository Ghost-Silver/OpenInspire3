/**
 * @file MotorMixer.h
 * @brief 四旋翼混控器：控制指令与电机推力之间的双向映射
 * @author GhostFace
 * @date 2026/9/18
 *
 * @par 它补上的是哪一层
 *
 * 此前仿真器直接接受 `(thrust_body, torque)` —— 相当于假设执行机构能**任意**
 * 产生指令要求的推力与力矩。真实飞控不是这样：它算出的指令要先经过**混控**
 * 分配到四个电机，每个电机有推力上限，而且电机可能失效。
 *
 * 加上这一层之后，「控制器想给什么」与「执行机构真正能给出什么」才区分开来 ——
 * 这正是讨论容错的前提。
 *
 * @par 混控矩阵与可控性
 *
 * X 型布局，四个电机在机体系的位置为 (±a, ±a)，a = arm_length/√2。
 * 推力沿机体 −z，力矩 τ = r × F，反扭矩与推力成正比。于是
 *
 * @verbatim
 *   [ T  ]   [  1    1    1    1  ] [ f1 ]
 *   [ τx ] = [ -y1  -y2  -y3  -y4 ] [ f2 ]
 *   [ τy ]   [  x1   x2   x3   x4 ] [ f3 ]
 *   [ τz ]   [  s1c  s2c  s3c  s4c] [ f4 ]
 * @endverbatim
 *
 * 这个 4×4 矩阵**可逆**，所以正常状态下四个控制量可以独立指定 —— 这是四旋翼
 * 能飞的根本原因。
 *
 * @par 失去一个电机意味着什么
 *
 * 去掉一列后变成 4×3，秩最多为 3：四维控制空间里**有一个方向永远无法到达**。
 * 这不是控制器好坏的问题，而是执行机构的维度不够 —— 无论用什么算法，
 * 都有一个姿态自由度不可控。
 *
 * 以失效 M1（前右）为例，剩余矩阵的左零空间给出约束
 *
 * @verbatim
 *   τz = (c / a) · τy
 * @endverbatim
 *
 * 即**偏航力矩被俯仰力矩锁定**，两者不能独立指定。本模块提供 `rankWithFailures`
 * 与零空间计算，把这件事从定性判断变成可验证的数值结论。
 */

#ifndef OI3_MOTOR_MIXER_H
#define OI3_MOTOR_MIXER_H

#include "SixDofTypes.h"

#include <array>
#include <vector>

namespace oi3 {

/// 四旋翼电机与布局参数
struct QuadMotorConfig {
    /// 机臂长度（机体中心到电机的距离，米）。X 型布局下电机位于 (±a, ±a)，
    /// a = arm_length / √2
    double arm_length = 0.25;

    /// 单电机最大推力（牛顿）
    double max_thrust = 5.0;

    /// 单电机最小推力（牛顿），电机有怠速
    double min_thrust = 0.0;

    /// 反扭矩系数 c（N·m per N）：电机反扭矩 τz = c · f
    double torque_coeff = 0.02;
};

/// 四个电机的推力与失效标志
struct MotorSet {
    std::array<double, 4> thrust{};
    /// 失效标志：为 true 时该电机推力恒为 0，且不受混控结果影响
    std::array<bool, 4> failed{};

    /// 是否全部正常
    [[nodiscard]] bool allHealthy() const {
        return !failed[0] && !failed[1] && !failed[2] && !failed[3];
    }

    [[nodiscard]] int failedCount() const {
        int n = 0;
        for (bool f : failed) {
            n += f ? 1 : 0;
        }
        return n;
    }
};

/**
 * @class QuadMixer
 * @brief 控制指令 ⇄ 电机推力
 *
 * 电机编号（俯视，机头朝上）：
 * @verbatim
 *        M4(前左)   M1(前右)
 *             \\     /
 *              [机体]
 *             /     \\
 *        M3(后左)   M2(后右)
 * @endverbatim
 */
class QuadMixer {
  public:
    explicit QuadMixer(QuadMotorConfig cfg = {});

    /**
     * @brief 控制指令 → 四电机推力
     *
     * @param cmd          控制指令
     * @param failed       失效标志
     * @param redistribute 是否有容错重分配逻辑，见下
     *
     * @par redistribute 的含义（这是一个真实的工程分界）
     *
     * **false = 飞控没有容错逻辑**：仍按四个电机求解（4×4 逆），再把失效电机的
     * 推力丢弃。后果是剩余电机的分配**完全错误** —— 正常悬停时每电机 2.45 N，
     * 失效一个后剩三个各 2.45 N，总推力仅 7.36 N 而需要 9.81 N，飞行器直接坠落。
     * 这是「没有做失效检测与重构」的真实表现。
     *
     * **true = 飞控检测到失效并重构**：改用剩余电机重新求解。此时只放弃不可达
     * 的 τz，用 T、τx、τy 三行配三个健康电机求解 —— 该 3×3 子矩阵行列式为
     * 4a² ≠ 0，**恰好可逆**，所以三个控制量仍能精确实现。
     *
     * 两种模式都有意义：前者是失效容错缺位时的对照，后者是容错接管后的行为。
     */
    [[nodiscard]] MotorSet mix(const SixDofCommand &cmd,
                               const std::array<bool, 4> &failed = {},
                               bool redistribute = true) const;

    /// 四电机推力 → 实际可实现的推力与力矩（不做任何限幅）
    [[nodiscard]] SixDofCommand unmix(const MotorSet &motors) const;

    /**
     * @brief 完整执行链：混控 → 逐电机限幅 → 失效置零 → 反算实际指令
     *
     * @param cmd    控制器给出的指令
     * @param motors 输入输出：更新为各电机实际推力（含失效）
     * @return 执行机构**真正实现**的推力与力矩
     */
    [[nodiscard]] SixDofCommand apply(const SixDofCommand &cmd, MotorSet &motors,
                                      bool redistribute = true) const;

    /// 混控矩阵 A：行 = [T, τx, τy, τz]，列 = 电机 1..4
    [[nodiscard]] const std::array<std::array<double, 4>, 4> &matrix() const { return _A; }

    /// 指定失效组合下剩余混控矩阵的秩，即**仍可控的自由度数量**
    [[nodiscard]] int rankWithFailures(const std::array<bool, 4> &failed) const;

    /**
     * @brief 失效后控制空间中不可达方向的约束系数
     *
     * 返回零空间的一组基向量，每个长度 4、对应 [T, τx, τy, τz]，满足
     * `w · [T, τx, τy, τz] = 0` —— 即执行机构永远无法产生违反该式的控制量。
     *
     * 基的个数 = 4 − 秩 = 丢失的控制自由度。全部电机正常时返回空。
     */
    [[nodiscard]] std::vector<std::array<double, 4>> unreachableDirections(
        const std::array<bool, 4> &failed) const;

    [[nodiscard]] const QuadMotorConfig &config() const { return _cfg; }

    /// 悬停时每个电机的推力
    [[nodiscard]] double hoverThrustPerMotor(double mass, double gravity) const {
        return mass * gravity / 4.0;
    }

    /// 推重比（四个电机总推力 / 重力）
    [[nodiscard]] double thrustToWeight(double mass, double gravity) const {
        return 4.0 * _cfg.max_thrust / (mass * gravity);
    }

  private:
    QuadMotorConfig _cfg;
    std::array<std::array<double, 4>, 4> _A{};
    std::array<std::array<double, 4>, 4> _Ainv{};
};

} // namespace oi3

#endif // OI3_MOTOR_MIXER_H
