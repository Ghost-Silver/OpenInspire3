/**
 * @file SixDofTypes.h
 * @brief 六自由度飞行器类型定义
 * @author GhostFace
 * @date 2026/9/17
 *
 * 与三自由度质点模型并存，不替换后者：三自由度版本已经完成闭环验证，保留它可以
 * 作为六自由度实现的对照基准（同样的 PID 参数与任务，两者应当给出可解释的差异）。
 *
 * 姿态用四元数而非欧拉角：欧拉角在俯仰 ±90° 处存在万向节死锁，而四旋翼做大角度
 * 机动时必然经过该区域。
 */

#ifndef OI3_SIX_DOF_TYPES_H
#define OI3_SIX_DOF_TYPES_H

#include "DroneTypes.h"
#include "Tensor.h"

namespace oi3 {

/**
 * @struct SixDofCommand
 * @brief 六自由度控制指令：推力 + 三轴力矩
 *
 * 定义在类型头而非某个控制器里 —— 它是各级控制器共同产出的接口类型
 * （定点 PID、手动模式、将来的学习型控制器都用它），放在具体控制器下会让
 * 其他控制器不得不反过来包含那个头文件。
 */
struct SixDofCommand {
    double thrust_body = 0.0; ///< 机体 z 轴推力（牛顿，向上为正）
    Tensor torque;            ///< 机体三轴力矩 {3}（N·m）

    /// 力矩是否已初始化
    [[nodiscard]] bool valid() const { return torque.numel() == 3; }
};

/**
 * @struct SixDofState
 * @brief 六自由度状态：平动 + 转动
 *
 * 张量按物理量拆分（而非拼成单一扁平向量），原因与三自由度版本相同：CTorch 当前
 * 没有 concat / stack 算子，拼接只能靠裸指针写入新张量，而新建张量是叶子节点，
 * 会切断 autograd 计算图。
 */
struct SixDofState {
    Tensor pos;   ///< {3} NED 位置（米）
    Tensor vel;   ///< {3} NED 速度（米/秒）
    Tensor quat;  ///< {4} 姿态四元数 (w, x, y, z)，机体系 -> NED
    Tensor omega; ///< {3} 机体系角速度 (p, q, r)（弧度/秒）

    /// 逐分量加法（RK4 线性组合的前提）
    [[nodiscard]] SixDofState operator+(const SixDofState &other) const {
        return {pos + other.pos, vel + other.vel, quat + other.quat, omega + other.omega};
    }

    /// 逐分量标量乘
    [[nodiscard]] SixDofState operator*(float scalar) const {
        return {pos * scalar, vel * scalar, quat * scalar, omega * scalar};
    }
};

/// 标量左乘，写法与 Tensor 保持一致
[[nodiscard]] inline SixDofState operator*(float scalar, const SixDofState &state) {
    return state * scalar;
}

/**
 * @struct SixDofConfig
 * @brief 六自由度配置
 *
 * `base` 复用三自由度版本的 Config（质量、重力、步长、逐轴二次阻力）。
 * 阻力按 NED 系建模：真实气动阻力沿速度反方向，在惯性系计算更直接。
 */
struct SixDofConfig {
    Config base;

    /**
     * @brief 机体转动惯量对角元 (kg·m²)，对应 roll / pitch / yaw 轴
     *
     * @note 下面的默认值是「小四轴典型值」，仅供缺省可用；真实平台的惯量与阻力
     *       直接决定仿真与真机的差距。可用 ParameterIdentification 从实测轨迹
     *       辨识（离线一次性、不需要实时，正是可微仿真的合理落点）：
     *       在 60 ms 的激励轨迹上实测可把质量辨识到 1.9%、阻力 14.6%、惯量约 18%，
     *       折算成开环预测精度是**误差降低 11 倍**（相对手填典型值），
     *       详见 ModelCalibrationTest。
     */
    double inertia[3] = {0.01, 0.01, 0.02};

    /// 力臂长度（米），仅用于记录机体几何；力矩由控制器直接给出
    double arm_length = 0.25;

    /// 各轴力矩限幅（N·m），<= 0 表示不限幅
    double torque_limit = 1.0;

    /// 推力上限（牛顿），对应机体 z 轴；<= 0 表示不限幅
    /// 注：与 base.max_thrust 共同存在，前者约束总推力标量、后者约束三轴分量
    double max_body_thrust = 20.0;
};

} // namespace oi3

#endif // OI3_SIX_DOF_TYPES_H
