/**
 * @file DroneTypes.h
 * @brief OpenInspire3 飞行器基础类型（仿真配置与状态表示）
 * @author GhostFace
 * @date 2026/9/16
 */

#ifndef OI3_DRONE_TYPES_H
#define OI3_DRONE_TYPES_H

#include "Tensor.h"

namespace oi3 {

/**
 * @struct Config
 * @brief 仿真配置
 */
struct Config {
    double dt = 0.001;     ///< 仿真步长（秒），默认对应 1kHz
    double mass = 1.0;     ///< 质量（千克）
    double gravity = 9.81; ///< 重力加速度（米/秒²）
};

/**
 * @struct DroneState
 * @brief 飞行器状态（NED 坐标系）
 *
 * 状态按物理量拆分为独立张量，而不是拼接成单一的扁平向量。两个原因：
 *
 * 1. **计算图完整性**。CTorch 当前没有 concat / stack 这类张量算子，
 *    把多个分量拼成一个向量只能靠裸指针写入新张量，而新建张量是叶子节点
 *    （构造函数会执行 `_autograd_meta._node.reset()`），写入操作不建立
 *    任何梯度关联，会切断 autograd 计算图。拆分后所有运算都走现有算子，
 *    梯度自然回流。
 *
 * 2. **语义清晰**。位置与速度量纲不同，分开存放更贴近物理建模，
 *    也便于按量纲分别做归一化和调试。
 *
 * 扩展到 6 自由度时，在此追加姿态四元数与机体系角速度字段即可，
 * 动力学函数与积分器接口无需改动。
 *
 * @note 逐分量运算符是 RK4 线性组合的前提：积分器需要计算
 *       `y + (k1 + 2*k2 + 2*k3 + k4) / 6`，因此 +、* 必须可用。
 */
struct DroneState {
    Tensor pos; ///< 位置 {3}：NED 系 [north, east, down]，单位米
    Tensor vel; ///< 速度 {3}：NED 系 [vn, ve, vd]，单位米/秒

    /// 逐分量加法
    [[nodiscard]] DroneState operator+(const DroneState &other) const {
        return DroneState{pos + other.pos, vel + other.vel};
    }

    /// 逐分量标量乘
    [[nodiscard]] DroneState operator*(float scalar) const {
        return DroneState{pos * scalar, vel * scalar};
    }
};

/// 标量左乘，写法与 Tensor 保持一致
[[nodiscard]] inline DroneState operator*(float scalar, const DroneState &state) {
    return state * scalar;
}

/**
 * @brief 构造 NED 系三分量常量向量
 * @note 用 initializer_list 构造函数，不触碰底层数据指针。
 */
[[nodiscard]] inline Tensor makeVec3(float north, float east, float down) {
    return Tensor{north, east, down};
}

} // namespace oi3

#endif // OI3_DRONE_TYPES_H
