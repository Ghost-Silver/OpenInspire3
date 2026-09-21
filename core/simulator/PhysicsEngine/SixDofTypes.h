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

    /**
     * @brief 转动阻尼系数（N·m·s/rad），0 表示不建模
     *
     * 角阻尼力矩 `τ = −k·ω`，主要来自桨叶与机身的空气阻力。这一项在分析
     * 电机失效后的自旋行为时是**必需的**，否则会得到与现实不符的结论：
     *
     * 失去一个电机后偏航力矩不可控，飞行器持续加速自旋。没有阻尼时角速度
     * 线性增长（实测 6 秒到 37 rad/s），角动量随之增大到 `I·ω ≈ 1.04
     * kg·m²/s`。此时飞行器像陀螺一样被角动量锁住姿态，而控制力矩上限
     * （torque_limit = 1 N·m）根本掰不动它 —— 表现为倾角卡在 138° 不再
     * 收敛，旋转容错随之失效。
     *
     * 真实飞行器的气动阻尼会让自旋趋于一个终速（τ_z = k·ω_final），也就
     * 给旋转容错留下了一个**时间窗**：自旋起来之前必须把姿态建立好，
     * 否则角动量一大就锁死了。
     */
    double rot_damping = 0.0;

    /**
     * @brief 各轴异性阻力系数（N·s²/m²），0 表示回退到标量形式
     *
     * 机体并非各向同性：机身扁平，垂直方向的迎风面积与阻力系数明显大于水平
     * 方向，真实小四轴常达 2~3 倍。现有模型只有一个标量 `drag_coeff`（三轴
     * 等同），这是**结构性简化**而非参数误差 —— 单靠调参数无法表达。
     *
     * 启用方式：三个分量均 > 0 时按逐轴计算；否则回退到 `base.drag_coeff` 的
     * 标量形式，行为与改造前逐字相同。
     *
     * @note 这一项与「桨盘入流」共同构成**未建模气动**：它们真实存在但被简化掉。
     *       解析前馈若仍按简化的各向同性模型补偿，就会产生系统性失配 ——
     *       这正是学习型方法可能占优的地方。标称气动下解析方法已接近信息极限
     *       （见 WindFeedforwardTest），只有引入未建模效应才能创造出学习空间。
     */
    double drag_coeff_axis[3] = {0.0, 0.0, 0.0};

    /**
     * @brief 桨盘入流系数：推力随轴向来流的衰减
     *
     * 螺旋桨产生的推力取决于桨盘处的**相对气流**，而非指令值。设轴向（机体 z）
     * 相对气流速度为 `v_axial`，则
     *
     * @verbatim
     *   T_eff = T_cmd · (1 − mu_lin·v_axial − mu_quad·|v_axial|)
     * @endverbatim
     *
     * 该效应依赖姿态与速度的耦合，呈非线性，解析模型难以精确表达。
     * 两个系数默认 0（关闭），保证既有结果逐位不变。
     */
    double inflow_linear = 0.0;
    double inflow_quad = 0.0;

    /// 各轴力矩限幅（N·m），<= 0 表示不限幅
    double torque_limit = 1.0;

    /// 推力上限（牛顿），对应机体 z 轴；<= 0 表示不限幅
    /// 注：与 base.max_thrust 共同存在，前者约束总推力标量、后者约束三轴分量
    double max_body_thrust = 20.0;

    /**
     * @brief 执行机构一阶滞后时间常数（秒），0 表示理想（无滞后）
     *
     * 真实电机 + 桨从收到指令到推力到位有时间常数（小四轴典型 20~50 ms）。
     * 这是一个**纯延迟性质的相位损失源**：一阶滞后 `1/(1 + τ·s)` 在频率 ω 处
     * 贡献 `−atan(ω·τ)` 的相位滞后，且随频率增长。
     *
     * @par 为什么这一项重要
     *
     * 频域分析（FrequencyResponseTest）得出一条修正直觉的结论：本项目的姿态环
     * 结构（两个极点、无零点）**相位裕度只由阻尼比 ζ 决定、与带宽无关** ——
     * 提高带宽并不损失裕度。那么真机上「带宽越高越不稳」的经验从何而来？
     * 答案就是**未建模延迟**：它随频率线性吃掉相位，带宽一高，同一延迟占用的
     * 裕度比例就大。
     *
     * 把这一项建出来，才能回答真机参数选择的关键问题：
     * **在给定的延迟水平下，多大带宽仍然安全？**
     *
     * 默认 0（理想执行器），保证既有结果逐位不变。
     */
    double actuator_tau = 0.0;

    /**
     * @brief 执行机构速率限幅（每秒相对变化率），0 表示不限幅
     *
     * 电机推力不能瞬时跳变。与时间常数不同，速率限幅是**非线性**的：
     * 小幅指令下线性响应良好，大幅机动时才暴露 —— 这是大机动场景下
     * 模型失配的来源之一。
     */
    double actuator_rate_limit = 0.0;

    /**
     * @brief 传感器测量延迟（秒），0 表示无延迟
     *
     * 真实飞控从物理量到控制器可用数据存在延迟，来源包括：IMU 采样与滤波、
     * 总线传输、姿态解算、控制周期本身。典型量级 1~10 ms（IMU 滤波可到 20 ms）。
     *
     * @note 本项作用于**控制器看到的状态**，与 `actuator_tau`（作用于执行输出）
     *       是两个独立的相位损失源，真机上两者叠加。
     */
    double sensor_delay = 0.0;

    /**
     * @brief 传感器采样与控制周期的频率比（>0），0 或 1 表示同频
     *
     * 真机上 IMU 通常跑 1~8 kHz，而控制回路 400 Hz~1 kHz —— 控制器每周期
     * 拿到的是若干帧之前的最新样本，这构成额外的**量化式延迟**（平均半帧到
     * 一帧）。同频建模会低估这一效应，默认 0 保持既有行为不变。
     */
    double sensor_rate_ratio = 0.0;
};

} // namespace oi3

#endif // OI3_SIX_DOF_TYPES_H
