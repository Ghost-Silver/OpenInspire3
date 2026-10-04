/**
 * @file SensorHealth.h
 * @brief IMU 传感器健康监测：检测「掉线」「卡死」「说谎」三类失效
 *
 * @par 为什么需要单独一个模块
 *
 * 已有的 FaultDetection.h 做的是**参数故障**（推力损失、阻力增大、质量变化），
 * 建立在在线辨识的残差上——它回答的是「飞机本身的物理特性变了吗」。
 * 而本文件回答的是另一个问题：「**传感器给的数据还能信吗**」。
 * 两者是不同的故障域：机体参数故障下传感器是好的，传感器故障下机体是好的。
 *
 * @par 三类失效，以及为什么「说谎」最难
 *
 * 1. **掉线（Dead）**：输出恒为零或不再更新。最容易测，但有个陷阱——
 *    静止时陀螺读数本来就是零，不能仅凭「值为零」判定掉线。
 * 2. **卡死（Frozen）**：输出冻结在某个值不再变化。与掉线的区别是它不一定为零。
 * 3. **说谎（Lying）**：数据还在更新，但系统性偏离真值（偏置漂移、标度误差）。
 *    这一类是**最危险的**——它不触发任何「无数据」告警，飞控会拿着错误的数据
 *    继续飞，以为自己在平飞，实际在倾斜。
 *
 * @par 检测量的选择：本模块的实测依据
 *
 * 设计前做了两轮探针实测（结果记录于 docs/imu-fault-tolerance.md），
 * 其中两条结论直接决定了这里的实现：
 *
 * - **「加速度计模长 ≈ g」不能检测偏置**。偏置沿某一轴叠加，而模长是各轴平方和
 *   的平方根，偏置对模长的影响是**二阶**的。实测偏置 +1.0 m/s²（约 10% g）时，
 *   模长仅从 9.809 变到 9.860，变化 0.05，完全淹没在 ±0.1 的噪声里。
 *   但模长对**彻底归零**极敏感（9.81 → 0.000），故模长只用于检测掉线。
 *
 * - **方向残差能检测偏置，但阈值必须随机动状态变化**。残差是 Mahony 互补滤波
 *   天然算出的量（测量比力方向与姿态估计方向的叉积）。实测加速度计偏置 +1.0
 *   使残差从 0.0041 升到 0.1426（35 倍）；陀螺偏置 +0.05 使其升到 0.102（25 倍）。
 *   但**机动时残差基线抬升两个量级**（0.0041 → 0.323），因为加速度计读的是比力
 *   而非重力，机动加速度被当成了姿态误差。固定阈值在机动时会全面误报。
 *
 * @par 统计量的机动门控，以及为什么确认计数要「暂停」而非「清零」
 *
 * 由上一条实测推出两条实现约束：
 *
 * - **残差统计量（滑窗与 CUSUM）只喂非机动帧**。机动帧的残差是运动不是故障，
 *   喂进累积检验等于注入垃圾样本；而 CUSUM 报警一经触发不会自动解除，
 *   一次机动就能换来永久误报。
 * - **机动帧不清空偏置确认计数**。机动时残差既不能归因偏置、也不能作为
 *   「健康」的证据。若按「非异常即清零」处理，一帧机动就会清空已累计的确认
 *   计数；位置量测噪声较强时陀螺抖动会频繁越限，状态将在 Healthy/Degraded
 *   间高频翻动，偏置判定永远无法稳定保持（PositionNonIdealClosedLoopTest
 *   实测：检出后状态翻回，方向反馈时开时关，姿态误差只能收敛一半）。
 *   故机动帧暂停计数，偏置的解除改由显式恢复路径负责（见
 *   recovery_residual / recovery_steps）。
 *
 * @par 一个必须诚实说明的物理限制
 *
 * **陀螺彻底归零在静止时不可观测**。静止时真值角速度本就是零，归零前后数据完全
 * 一致（实测姿态误差 0.011° vs 基线 0.025°，残差 0.0041 vs 0.0041）。
 * 这是物理事实而非设计缺陷——要检测它必须引入外部姿态信息（视觉、磁力计、
 * 或至少机动激励）。本模块因此**不为静止状态下的陀螺归零提供保证**，
 * 只在有机动激励时给出判据，并把这一限制显式暴露给调用方。
 *
 * @par 与容错控制的分工
 *
 * 本模块只做「检测与报告」，不做「重构与控制」。理由与 FaultDetection.h 相同：
 * 必须先知道出了什么故障，才能谈如何重构；而重构策略依机型与任务而异，
 * 不应写死在检测模块里。降级决策见 ImuDegradePolicy。
 */

#ifndef OI3_SENSOR_HEALTH_H
#define OI3_SENSOR_HEALTH_H

#include "FaultDetection.h" // 复用 ResidualMonitor / CusumDetector 两个统计量

#include <array>
#include <cmath>

namespace oi3 {

/// 单个传感器的健康状态
enum class SensorStatus {
    Unknown = 0, ///< 尚未收集到足够样本（启动阶段），不应据此决策
    Healthy,     ///< 正常
    Degraded,    ///< 降级：数据仍可用但精度受损（如可补偿的偏置）
    Failed,      ///< 失效：数据不可信
};

/// IMU 故障类型
enum class ImuFault {
    None = 0,
    AccelDead,   ///< 加速度计无有效输出（模长趋零或超量程）
    AccelFrozen, ///< 加速度计输出冻结（长时不变）
    AccelBias,   ///< 加速度计系统性偏置（方向残差持续偏离）
    GyroDead,    ///< 陀螺无有效输出
    GyroFrozen,  ///< 陀螺输出冻结
    GyroBias,    ///< 陀螺系统性偏置
};

/// 健康监测配置
struct SensorHealthConfig {
    /// 是否启用监测。默认关闭——保证接入后既有测试逐位不变。
    bool enabled = false;

    /// 加速度计模长的合理下限（m/s²）。低于此判定掉线。
    /// 悬停时约 9.81，彻底归零时约 0，故取 1.0 有充足余量。
    double accel_mag_min = 1.0;

    /// 加速度计模长的合理上限（m/s²）。高于此判定超量程/异常。
    /// 大机动（12 m/s² 水平）时实测约 15.5，留余量取 30。
    double accel_mag_max = 30.0;

    /// 方向残差的噪声底（静止悬停实测量级，见文件头）。用于置信度归一。
    double residual_noise = 0.004;

    /**
     * @brief CUSUM 松弛量：小于此幅度的残差不累积
     *
     * 必须**高于静止稳态残差均值**，否则正常飞行也会持续累积并误报。
     * 实测静止稳态残差约 0.0041（噪声与常值偏置共同造成，并非零均值）；
     * 若按常见的 0.5σ 取松弛量（0.002），稳态每步净累积 +0.002，
     * 十余步就越过阈值 —— 这正是第一版实现误报的根因。
     *
     * @par 由 0.01 提高到 0.03 的依据（实测修正）
     *
     * 0.01 只高于**稳态**残差，却低于「姿态尚未收敛」时的残差。实测加速度计
     * 掉线后恢复的场景：掉线期间方向校正失效、姿态漂移，恢复后残差升至
     * 0.012~0.018 且长时间不回落 —— 高于 0.01，于是 CUSUM 反复报警，
     * 传感器数据明明已完全正常（模长 9.765 m/s²）却被判为 AccelBias。
     *
     * 更糟的是形成**自我维持的死锁**：判为 Degraded 后降级策略会关闭方向校正
     * （防止偏置污染姿态），而关闭校正又让姿态无法收敛、残差维持高位，
     * 于是 Degraded 被无限延长。实测恢复后 Degraded 占比达 76.7%。
     *
     * 三个水平的实测值：
     *
     * | 状态 | 残差 |
     * |---|---|
     * | 正常稳态 | 0.003 ~ 0.006 |
     * | 姿态未收敛（掉线恢复后） | 0.012 ~ 0.018 |
     * | 真实加速度计偏置 | 0.1426 |
     *
     * 取 0.03：高于未收敛水平（0.018），远低于真实偏置（0.1426）。
     * 对真实偏置的检出不受影响 —— 每帧净累积 0.11，仍会迅速越过阈值 0.05。
     */
    double cusum_drift = 0.03;

    /**
     * @brief CUSUM 报警阈值
     *
     * 取值须落在「静止稳态残差」与「故障残差」之间。实测：
     *   静止稳态 ≈ 0.004 ；加速度计偏置 +1.0 ⇒ 0.143 ；陀螺偏置 +0.05 ⇒ 0.102
     * 取 0.05 可避开稳态、同时远低于两类故障水平。
     */
    double cusum_h = 0.05;

    /// 卡死检测：连续多少步输出完全不变即判定冻结。
    /// 取 50 步（1 kHz 下 50 ms）；真机上传感器噪声会自然抖动，
    /// 完全不变只可能是数字通路卡死。
    int frozen_steps = 50;

    /// 残差统计量窗口长度
    int residual_window = 200;

    /**
     * @brief 判定为「机动中」的角速度水平（rad/s）
     *
     * 必须**独立于方向残差**。第一版用残差判定机动（阈值 0.05），
     * 结果故障本身抬升残差后会被误判为机动，进而触发「机动时不归因偏置」
     * 的规则 —— 故障把自己伪装成了机动，检测链自我关闭。
     * 改用陀螺角速度：静止 ≈ 0（噪声 0.004），机动（滚转 ±15°@0.5Hz）
     * 峰值 ≈ 0.82 rad/s，相差两个量级，区分度充足且不被加速度计故障污染。
     */
    double maneuver_gyro = 0.05;

    /// 进入 Degraded 所需的持续步数（去抖，避免单帧野点误判）
    int confirm_steps = 100;

    /**
     * @brief 恢复判据：视为「故障可能已消失」的方向残差上限
     *
     * @par 取值依据（由实测修正，原值 0.02 不可用）
     *
     * 原注释按**稳态均值**设计阈值：健康静止残差均值约 0.004、偏置故障 ≥0.10，
     * 故取 0.02（5 倍噪声底）。但恢复判据要求的是「**瞬时残差**连续
     * `recovery_steps` 帧低于阈值」—— 而瞬时残差存在波动，实测健康状态下
     * 波动上沿达 0.025，即**原阈值 0.02 落在正常波动带内部**。
     *
     * 后果（实测：加速度计掉线后恢复的场景）：
     *
     * | 项 | 值 |
     * |---|---|
     * | 恢复观察区间总帧数 | 6986 |
     * | 残差 ≥0.02 的帧数 | 108（1.55%） |
     * | 残差最大值 | 0.02483 |
     *
     * 平均每约 65 帧就有一帧越阈，把连续计数清零。要连续 500 帧不越阈几乎
     * 不可能，恢复因此被拖长至 6986 帧（约 7 s），远超设计意图的 0.5 s。
     * 偏置恢复场景耗时正常（499 帧）只是抽样运气 —— 同一机制在掉线场景下
     * 就暴露了。
     *
     * 修正为 0.05：明显高于健康波动上沿（0.025），仍远低于故障水平（≥0.10），
     * 落在两者之间，判据由此对单帧波动免疫。
     *
     * 教训：为「连续 N 帧达标」这类判据选阈值时，必须按**瞬时值的波动上沿**
     * 选取，而非按稳态均值 —— 后者会让判据嵌在噪声带内，形式上成立、实际不可用。
     */
    double recovery_residual = 0.05;

    /// 确认恢复所需的连续非机动低残差步数（去抖）。取 500 步（0.5 s）：
    /// 足以避免单个侥幸低点触发解除，又远短于偏置漂移再现的时间尺度。
    int recovery_steps = 500;
};

/// 健康报告
struct SensorHealthReport {
    SensorStatus accel = SensorStatus::Unknown;
    SensorStatus gyro = SensorStatus::Unknown;
    ImuFault fault = ImuFault::None;

    /// 置信度 0~1：由残差超出噪声的倍数给出，仅对 Degraded 有意义
    double confidence = 0.0;

    /// 当前是否处于机动（影响各故障的可检测性，调用方须据此解读结果）
    bool maneuvering = false;

    /// 当前加速度计模长（m/s²）
    double accel_magnitude = 0.0;

    /// 当前方向残差
    double residual = 0.0;

    /// 已收集样本数
    long long samples = 0;

    /// 整体是否可用：两者都未 Failed
    [[nodiscard]] bool usable() const {
        return accel != SensorStatus::Failed && gyro != SensorStatus::Failed;
    }
};

/**
 * @class SensorHealth
 * @brief IMU 健康监测：融合模长、残差、活动性三类判据
 *
 * 调用时机：每收到一次 IMU 采样后调用 update()，传入原始采样、
 * 当前方向残差与姿态估计所需的上下文。
 */
class SensorHealth {
  public:
    explicit SensorHealth(const SensorHealthConfig &cfg = {}) : _cfg(cfg) {
        // 注意：这里**不用** CusumDetector::forNoise。
        // forNoise 按 k=0.5σ 取松弛量，而本场景的静止稳态残差（0.004）
        // 与 σ 同量级且并非零均值 —— 按 σ 缩放会让松弛量（0.002）低于稳态，
        // 正常飞行即持续累积并误报。故改用显式的 cusum_drift / cusum_h，
        // 两者直接对「稳态残差」与「故障残差」取中间值，见配置注释的实测依据。
        _resid_cusum = CusumDetector(cfg.cusum_drift, cfg.cusum_h);
    }

    void reset() {
        _resid_monitor.reset();
        _resid_cusum = CusumDetector(_cfg.cusum_drift, _cfg.cusum_h);
        _accel_frozen_count = 0;
        _gyro_frozen_count = 0;
        _accel_bad_count = 0;
        _accel_ok_count = 0;
        _accel_fault_latched = ImuFault::None;
        _recovery_count = 0;
        _gyro_bad_count = 0;
        _prev_accel = {};
        _prev_gyro = {};
        _have_prev = false;
        _report = {};
        _samples = 0;
    }

    /**
     * @brief 喂入一次采样
     *
     * @param accel        原始加速度计读数（机体系比力，m/s²）
     * @param gyro         原始陀螺读数（机体系角速度，rad/s）
     * @param residual     当前方向残差（Mahony 叉积误差的模长）
     *
     * @note residual 必须由调用方从**当前姿态估计**算出。它是本模块最主要的
     *       检测量，算错（例如用了未归一化的向量）会让整条检测链失效。
     */
    void update(const std::array<double, 3> &accel, const std::array<double, 3> &gyro,
                double residual) {
        if (!_cfg.enabled) {
            return;
        }

        ++_samples;
        _report.samples = _samples;
        _report.residual = residual;

        // ---- 1. 加速度计模长：检测掉线与超量程 ----
        const double am = std::sqrt(accel[0] * accel[0] + accel[1] * accel[1] + accel[2] * accel[2]);
        _report.accel_magnitude = am;
        const bool accel_mag_bad = (am < _cfg.accel_mag_min) || (am > _cfg.accel_mag_max);

        // ---- 2. 卡死检测：输出完全不变 ----
        if (_have_prev) {
            if (sameAs(accel, _prev_accel)) {
                ++_accel_frozen_count;
            } else {
                _accel_frozen_count = 0;
            }
            if (sameAs(gyro, _prev_gyro)) {
                ++_gyro_frozen_count;
            } else {
                _gyro_frozen_count = 0;
            }
        }
        _prev_accel = accel;
        _prev_gyro = gyro;
        _have_prev = true;

        // ---- 3. 残差统计：检测「说谎」 ----
        // 机动判定先行：用陀螺角速度模长，独立于残差（见配置注释）
        const double gm = std::sqrt(gyro[0] * gyro[0] + gyro[1] * gyro[1] + gyro[2] * gyro[2]);
        _report.maneuvering = (gm > _cfg.maneuver_gyro);

        // 残差统计量只喂非机动帧（见文件头「统计量的机动门控」）：机动帧的残差
        // 是运动不是故障，喂进累积检验等于注入垃圾样本，而 CUSUM 报警锁存不解除，
        // 一次机动就会换来永久误报。恢复证据同样只在非机动帧累计。
        if (!_report.maneuvering) {
            _resid_monitor.update(residual);
            _resid_cusum.update(residual);
            if (residual < _cfg.recovery_residual) {
                ++_recovery_count;
            } else {
                _recovery_count = 0;
            }
        } else {
            _recovery_count = 0;
        }

        // ---- 4. 综合判定 ----
        // 加速度计
        bool accel_bad_now = false;
        ImuFault accel_fault = ImuFault::None;
        if (accel_mag_bad) {
            accel_bad_now = true;
            accel_fault = ImuFault::AccelDead;
        } else if (_accel_frozen_count >= _cfg.frozen_steps) {
            accel_bad_now = true;
            accel_fault = ImuFault::AccelFrozen;
        } else if (_resid_cusum.alarmed()) {
            // 残差持续偏离——在**静止**时才能可靠归因到加速度计偏置；
            // 机动时残差基线本身抬升两个量级，不做此归因（见文件头实测）。
            if (!_report.maneuvering) {
                accel_bad_now = true;
                accel_fault = ImuFault::AccelBias;
            }
        }
        // 故障类型**锁存**：达标判定用的是累计计数，但 accel_fault 是瞬时值。
        // 若达标那一帧恰好因相位或机动标记变化而不再是 AccelBias，
        // 就会漏设状态。故在首次触发时锁存类型，计数归零时清除。
        if (accel_bad_now) {
            // 取本次连续异常期间出现过的**最高优先级**类型，允许后续升级。
            if (faultPriority(accel_fault) > faultPriority(_accel_fault_latched)) {
                _accel_fault_latched = accel_fault;
            }
            ++_accel_bad_count;
            // 注意：这里**不**重置 _accel_ok_count。它由下方独立的新判据管理 ——
            // 若在此处归零，会与紧随其后的递增互相抵消，每帧结果恒为 1，
            // 恢复将永远无法完成（这一处正是实测中「健康计数卡在 1」的原因）。
        } else if (_report.maneuvering) {
            // 机动帧：残差不可归因，既不算故障证据、也不算健康证据。
            // 保持计数不动（暂停而非清零）——否则一帧机动就清空已累计的确认计数，
            // 状态会在 Healthy/Degraded 间高频翻动，见文件头「统计量的机动门控」。
        } else {
            _accel_bad_count = 0;
            _accel_fault_latched = ImuFault::None;
        }

        // 硬故障恢复计数：判据是**原始数据是否正常**（模长正常、未冻结），
        // 而不是「综合异常标志是否为假」。
        //
        // 这一点是实测纠出来的。原先用 `!accel_bad_now` 递增，会把 CUSUM 的
        // **累积报警**也算作「不正常」——而 CUSUM 是锁存判据，报警后必须显式
        // 清除（由 recovery_count 达标触发）。于是形成死锁：
        //
        //   CUSUM 报警 → accel_bad_now 恒真 → 健康计数恒为 0 → 无法从 Failed 恢复
        //   而 recovery_count 每次达标只清一次 CUSUM，残差略高即重新累积
        //
        // 实测：加速度计掉线后第 5000 周期恢复（模长回到 9.765 m/s²），
        // 健康计数始终为 0，到第 11000 周期仍未解除 Failed。
        //
        // 语义上，当前数据正常就够了 —— 历史累积报警是「过去可能出过问题」的
        // 证据，不该阻止系统确认「现在已恢复」。偏置类故障仍由 recovery_count
        // 路径负责解除（它需要连续低残差，与这里互补）。
        if (accel_mag_bad || _accel_frozen_count >= _cfg.frozen_steps) {
            _accel_ok_count = 0;
        } else {
            ++_accel_ok_count;
        }

        // ---- 4b. 偏置恢复路径 ----
        // 偏置是软故障：传感器真正恢复后必须有显式出口，否则一次偏置会把状态
        // 永久锁在 Degraded（CUSUM 报警锁存、且机动帧不再清零计数）。
        // 条件：非机动帧残差连续 recovery_steps 步低于 recovery_residual。
        // 只解除偏置类 Degraded；掉线/卡死等硬故障走「判据消失 → 计数清零」
        // 自己的恢复语义，不经过这里。
        if (_recovery_count >= _cfg.recovery_steps) {
            _resid_cusum.reset();
            _recovery_count = 0;
            if (_report.accel == SensorStatus::Degraded &&
                _report.fault == ImuFault::AccelBias) {
                _accel_bad_count = 0;
                _accel_fault_latched = ImuFault::None;
                _report.accel = SensorStatus::Healthy;
                _report.fault = ImuFault::None;
            }
        }

        // 陀螺
        bool gyro_bad_now = false;
        ImuFault gyro_fault = ImuFault::None;
        if (_gyro_frozen_count >= _cfg.frozen_steps) {
            gyro_bad_now = true;
            gyro_fault = ImuFault::GyroFrozen;
        }
        _gyro_bad_count = gyro_bad_now ? (_gyro_bad_count + 1) : 0;

        // ---- 5. 去抖后落实到状态 ----
        if (_samples < _cfg.residual_window) {
            _report.accel = SensorStatus::Unknown;
            _report.gyro = SensorStatus::Unknown;
            _report.fault = ImuFault::None;
            return;
        }

        // 加速度计：掉线/卡死为 Failed，偏置为 Degraded（仍可用但精度受损）
        if (_accel_bad_count >= _cfg.confirm_steps) {
            if (_accel_fault_latched == ImuFault::AccelBias) {
                if (_report.accel != SensorStatus::Failed) {
                    _report.accel = SensorStatus::Degraded;
                    _report.fault = ImuFault::AccelBias;
                }
            } else {
                _report.accel = SensorStatus::Failed;
                // 用锁存类型而非瞬时值：计数可跨机动帧保持，达标后的帧
                // 可能是不可归因的机动帧（瞬时类型为 None）。
                _report.fault = _accel_fault_latched;
            }
        } else if (_report.accel != SensorStatus::Failed ||
                   _accel_ok_count >= _cfg.confirm_steps) {
            // 判据已消失，可以解除故障。
            //
            // 从 Failed 解除需 `_accel_ok_count` 连续正常达标（去抖）；其余状态
            // 立即解除。曾经的写法是 `else if (_report.accel != Failed)`，即
            // **完全排除** Failure 的恢复路径 —— 结果一次瞬时断连就把状态永久
            // 锁在 Failed。实测：加速度计第 2000 周期归零、第 5000 周期恢复
            // （模长回到 9.765 m/s²），到第 11000 周期仍为 Failed/AccelDead。
            // 代码注释当时已写明「恢复后允许清除」，实现与注释相悖。
            //
            // 真机后果：瞬时断连（接触不良、EMI）会让飞控永久处于降级返航状态，
            // 传感器完全恢复后也不会回到正常飞行。
            _report.accel = SensorStatus::Healthy;
            if (_report.fault == ImuFault::AccelDead || _report.fault == ImuFault::AccelFrozen) {
                // 掉线/卡死是硬故障，恢复后清除类型（传感器可能只是瞬时断连）
                _report.fault = ImuFault::None;
            }
            if (_accel_ok_count >= _cfg.confirm_steps) {
                _accel_ok_count = 0;
            }
        }

        if (_gyro_bad_count >= _cfg.confirm_steps) {
            _report.gyro = SensorStatus::Failed;
            _report.fault = gyro_fault;
        } else if (_report.gyro != SensorStatus::Failed) {
            _report.gyro = SensorStatus::Healthy;
        }

        // 置信度：残差超出噪声的倍数，饱和到 1
        const double sigma = std::max(_cfg.residual_noise, 1e-12);
        _report.confidence = std::min(1.0, std::fabs(_resid_monitor.mean()) / (3.0 * sigma));
    }

    [[nodiscard]] const SensorHealthReport &report() const { return _report; }


    /// 残差的噪声水平估计（窗口标准差），供上层诊断
    [[nodiscard]] double residualStdDev() const { return _resid_monitor.stdDev(); }

  private:
    /**
     * @brief 故障类型优先级：硬故障（掉线/卡死）高于软故障（偏置）
     *
     * 为什么要分级而不是「先到先得」：多个判据会**同时**接近触发条件，
     * 而软故障（CUSUM 累积残差）常常比硬故障（冻结计数）早几步报警。
     * 实测加速度计冻结时，CUSUM 约在第 45 步报警、冻结计数第 50 步达标；
     * 若按首次触发锁存类型，冻结就会被误判成偏置（第一版实现即如此，
     * 由测试第 3 节捕获）。故每次触发取优先级最高者，允许后续升级。
     */
    static int faultPriority(ImuFault f) {
        switch (f) {
        case ImuFault::AccelDead:
            return 3;
        case ImuFault::AccelFrozen:
            return 2;
        case ImuFault::AccelBias:
            return 1;
        default:
            return 0;
        }
    }

    static bool sameAs(const std::array<double, 3> &a, const std::array<double, 3> &b) {
        // 完全相等才算冻结。真机上传感器噪声会让读数持续抖动，
        // 逐位相等只可能出现在数字通路卡死（SPI/I2C 停更）。
        return a[0] == b[0] && a[1] == b[1] && a[2] == b[2];
    }

    SensorHealthConfig _cfg;
    ResidualMonitor _resid_monitor{200, 4.0};
    CusumDetector _resid_cusum;
    std::array<double, 3> _prev_accel{};
    std::array<double, 3> _prev_gyro{};
    bool _have_prev = false;
    int _accel_frozen_count = 0;
    int _gyro_frozen_count = 0;
    int _accel_bad_count = 0;
    /// 加速度计连续健康帧计数（用于解除 Failed 的去抖）
    int _accel_ok_count = 0;
    ImuFault _accel_fault_latched = ImuFault::None;
    int _recovery_count = 0;
    int _gyro_bad_count = 0;
    SensorHealthReport _report{};
    long long _samples = 0;
};

} // namespace oi3

#endif // OI3_SENSOR_HEALTH_H
