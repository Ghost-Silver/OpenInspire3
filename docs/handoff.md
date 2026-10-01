# 接手须知：OpenInspire3 当前状态与待办

本文件面向接手 OI3 开发的新成员。目的是让接手者不必回溯对话历史即可开工。

## 一、项目概况

OpenInspire3（OI3）是开源的四旋翼飞控与仿真平台，以 CTorch 作为可微仿真后端。
仓库布局：

| 路径 | 说明 |
|---|---|
| `core/simulator/PhysicsEngine/` | 六自由度动力学、IMU 模型、仿真器 |
| `core/estimation/` | 状态估计、故障检测、传感器健康 |
| `core/control/` | 各类控制器、降级执行器 |
| `core/guidance/` | 制导层（Minimum-Snap 轨迹生成） |
| `core/CTorch/` | CTorch 子模块（独立仓库 ShengFlow/CTorch） |
| `docs/` | 设计与验证文档 |

构建：

```bash
cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && cmake --build . -j8
```

已知的构建噪音：`tl-ArrayTest` 因缺少 `gtest/gtest.h` 路径而失败。这是环境配置问题，
与源码无关，不影响其余目标。若需要跑该测试，配置 GTest 的 include 路径即可。

## 二、已完成并验证的工作

### 2.1 IMU 失效容错（2026-10-01 完成）

三个模块加一份文档：

- `core/estimation/SensorHealth.h` —— 检测掉线、卡死、偏置漂移三类失效
- `core/estimation/ImuDegradePolicy.h` —— 由健康状态推出分级降级动作
- `core/control/DegradeExecutor.h` —— 把决策落到实际控制指令（包装任意控制器）
- `docs/imu-fault-tolerance.md` —— 设计依据、探针数据、六处自我纠错、已知限制

分级依据是「故障发生在哪个控制环」：陀螺失效是姿态环断链（ms 级、不可逆），
只有紧急降落一条路；加速度计失效是位置环退化（分钟级），给返航；偏置漂移则限幅。

闭环对照的实测结果：

| 场景 | 不降级 | 执行降级 |
|---|---|---|
| 陀螺失效 | 触地倾角 164.61°（倒扣） | 触地倾角 0.39°（水平） |
| 加速度计偏置 | 姿态估计误差 5.197° | 1.113° |

健康监测**默认关闭**（`EstimatorConfig::sensor_health.enabled = false`），
启用前不影响任何既有行为。

### 2.2 制导层（较早完成，P4 完成标定）

`core/guidance/MinimumSnapTrajectory.*` —— 七阶分段多项式最小快照轨迹生成，
39 项断言通过。设计与验证记录见 `docs/guidance-minimum-snap.md`。

P4 在仿真闭环中扫描 `(max_vel, max_acc, max_jerk)` 网格，标定出推荐上限：

| 约束 | 推荐值 | 来源 |
|---|---|---|
| `max_vel` | 5.0 m/s | 标定上限 5.10 m/s 的工程取整 |
| `max_acc` | 5.84 m/s² | `0.9 × g·tan(35°)` |
| `max_jerk` | 30.0 m/s³ | 标定可达 127 m/s³，但受电机/结构响应限制 |

水平返航（20 m）是速度瓶颈：超过 6 m/s 后倾角突破 30° 安全边界；垂直方向未触顶。
详见 `docs/guidance-minimum-snap.md` §十。

### 2.3 非理想位置量测下的降级验证（2026-10-01 完成，P1）

`core/control/PositionNonIdealClosedLoopTest.cpp`（32 项断言）用「延迟样本 +
高斯噪声」位置量测模型重跑闭环对照，两档参数（光流级 σ=0.05 m/20 ms、
GPS 级 σ=0.30 m/100 ms），四组消融。核心结论：

- **降级策略在非理想量测下仍有效**：偏置检出后关闭方向反馈，姿态误差
  7.18° → 0.30°（光流级）；失效场景降级不劣化。
- **保留位置预积分的结论保持且更强**：非理想量测下关闭预积分让高度偏差
  从 0.43 m 恶化到 169.50 m（理想量测下是 5.35 → 16.14 m）。
- **过程中修复了一个新缺陷**：旧实现一帧机动就清零偏置确认计数，噪声量测
  引起的陀螺抖动使状态在 Healthy/Degraded 间翻动。修复为「残差统计量只喂
  非机动帧 + 机动帧暂停计数 + 显式恢复路径」，见 `SensorHealth.h` 与
  `docs/imu-fault-tolerance.md` §6.2。
- **两条诚实的边界**：① 默认机动门限（0.05 rad/s）下，噪声量测使偏置归因
  几乎全程被掩蔽（漏检，非误报）；② GPS 级噪声单独作用即令高度通道发散
  （46 m 偏差），超出当前估计器 + PID 的稳定包络——归因已用「仅噪声 /
  仅延迟 / 叠加」探针分离并固化为断言。

### 2.4 飞控主循环与 HAL 抽象层（2026-10-01 完成，P2）

此前降级通路只存在于测试代码的参考实现中（`AccelFaultClosedLoopTest`），
没有独立的生产代码可供真机接入。P2 交付：

- **`core/control/HalAbstraction.h`**：HAL 接口（传感器读取、执行器写入、
  目标点来源），让主循环与底层硬件解耦。
- **`core/control/FlightControlLoop.h/.cpp`**：独立主循环，把「读传感器 →
  估计 → 决策 → 控制 → 写执行器」串成生产代码。降级协调逻辑（决策同时
  作用于估计器与执行器）内聚在主循环中。
- **`core/control/HalSimulator.h`**：仿真 HAL 实现，用 `SixDofSimulator` +
  `ImuModel` 模拟传感器和执行器，让同一份主循环代码跑在仿真平台上。
- **`core/control/FlightControlLoopTest.cpp`**（16 项断言）验证四条路径：
  正常飞行、默认行为不变（健康监测关闭时与直接闭环逐位一致）、
  降级协调通路（决策→估计器+执行器同时到达）、紧急降落（陀螺归零模拟
  掉线，2149 步触发 EmergencyLand 并终止主循环）。

### 2.5 制导层与容错层联调（2026-10-01 完成，P3）

降级动作（返航、紧急降落）此前只输出限幅与动作码，没有对应的轨迹执行。
P3 把 `MinimumSnapTrajectory` 接入降级通路：

- **`core/guidance/GuidanceSetpointSource.h`**：任务模式制导源，实现 `HalSetpointSource`，
  管理 Mission / ReturnHome / EmergencyLand 三种模式。ReturnHome 时生成从当前位置
  回到起点的 MinimumSnap 轨迹；EmergencyLand 时生成垂直下降轨迹用于着陆检测。
- **`core/control/HalAbstraction.h`**：`HalSetpointSource` 新增 `onDecisionChanged` 接口，
  由主循环在降级决策变化时调用，制导源据此切换模式并构建轨迹。
- **`core/control/FlightControlLoop.cpp`**：在 `applyDegradation` 后通知制导源决策变化，
  实现「降级决策 → 轨迹生成 → 控制跟踪」的闭环。
- **`core/guidance/GuidanceIntegrationTest.cpp`**（17 项断言）验证：Mission 模式与
  `FixedSetpointSource` 等价（不引入回归）、ReturnHome 轨迹生成与跟踪（5 m 水平返航，
  末态距 home 0.50 m）、EmergencyLand 垂直下降轨迹构建（时长 7.29 s）、模式切换往返。

**设计取舍**：EmergencyLand 的实际控制仍由 `DegradeExecutor` 处理（切断力矩、固定推力），
  而非跟踪垂直下降轨迹。陀螺失效后姿态反馈不可信，继续做姿态修正会加剧发散——
  这一点由 `imu-fault-tolerance.md` §6.1 的闭环对照证实。垂直下降轨迹在此仅用于
  着陆检测与地面站显示。

## 三、待办清单

### P0：落地未提交的改动

工作区有 6 处未提交改动（加速度计偏置容错），意图如下：

1. 把「方向残差检测」与「方向反馈校正」解耦。此前两者在同一个 `if` 内，
   关闭校正会连带停止健康检测。修改后检测独立于校正开关。
2. `AccelBias → Cautious` 时关闭加速度计方向反馈，保留位置预积分。
   初版只做力矩限幅却继续反馈偏置，实测降级前后误差都是 5.197°，等于无效。
3. 新增 `core/control/AccelBiasClosedLoopTest.cpp`（8 项断言）。
4. 更新 `ImuDegradePolicyTest` 的相应断言与文档。

验证状态：全量回归通过。**已提交并推送（`f38efa6`）。**

### ~~P1：位置测量非理想下的降级验证~~（已完成，见 §2.3）

结论：降级策略与 `trust_position` 取舍在非理想量测下保持成立；过程中修复了
偏置判定翻动缺陷（`SensorHealth`），并钉住两条包络边界（噪声量测下偏置
漏检、GPS 级噪声超出稳定域）。验证见 `PositionNonIdealClosedLoopTest`（32/32）
与 `docs/imu-fault-tolerance.md` §6.2。

### ~~P2：降级通路接入真实主循环~~（已完成，见 §2.4）

结论：`FlightControlLoop` 主循环已独立为生产代码，HAL 抽象层让同一份代码可跑
仿真与真机；降级协调逻辑（决策同时作用于估计器与执行器）内聚在主循环中；
`FlightControlLoopTest`（16/16）验证正常飞行、默认行为不变、降级协调、紧急降落
四条路径。全量回归 54 个测试 53 过。详见 §2.4。

### ~~P3：制导层与容错层联调~~（已完成，见 §2.5）

结论：`GuidanceSetpointSource` 已接入主循环，降级决策（ReturnHome / EmergencyLand）
自动触发对应轨迹生成；Mission 模式与 `FixedSetpointSource` 逐位等价，不引入回归。
验证见 `GuidanceIntegrationTest`（17/17）与 `docs/guidance-minimum-snap.md` §八。

### ~~P4：速度与 jerk 上限标定~~（已完成，见 §2.2）

结论：`TrajectoryLimitCalibrationTest` 在仿真闭环中扫描 `(max_vel, max_acc, max_jerk)`
网格，覆盖 20 m 水平返航与 5 m 垂直下降两个场景，以「到达 ±0.5 m + 最大倾角 ≤ 30°
+ 不超时」为通过标准，标定出推荐上限 `max_vel=5.0 m/s`、`max_acc=5.84 m/s²`、
`max_jerk=30.0 m/s³`，并已同步到 `MinimumSnapTrajectory.h` 的 `TrajectoryLimits` 默认值。
验证见 `TrajectoryLimitCalibrationTest`（4/4）与 `docs/guidance-minimum-snap.md` §十。

### ~~P5：GPS 级位置噪声稳定域修复~~（已完成）

结论：`StateEstimator` 新增 `pos_residual_deadzone`（位置残差死区，仅作用于速度
校正）。GPS 级量测（σ=0.30 m）启用 0.15 m 死区后，无故障基线高度偏差从 46.04 m
降到 3.45 m，回到有界状态；光流级等低噪声场景死区默认关闭，既有结果逐位不变。
`PositionNonIdealClosedLoopTest` 从 32/32 更新为 36/36，新增四条断言：无死区时
噪声仍发散、延迟单独无害、死区使噪声场景稳定、死区使噪声+延迟场景稳定。
详见 `docs/imu-fault-tolerance.md` §6.2.4。

## 四、硬约束（务必遵守）

### 4.1 三条铁律

1. **不写恒真断言**。`checkTrue("检查", true)` 等于没检查。写不出会失败的断言，
   说明这个检查没有价值。
2. **断言「发现缺陷」之前，先验证激励到达了待测机制**。本项目有多个反例：
   用水平风激发轴向入流损失（`v_axial ≈ 0`）、把 `inflow_linear` 设为 0
   连物理效应本身也关掉、验证脚本差分系数写错且跨了段边界。
3. **单个模块正确，不能推出组合正确**。实测过两个各自正确的补偿机制叠加后，
   稳态误差比完全不补偿还差。新模块必须单独验证组合行为。

### 4.2 默认行为不变

新增效应一律**默认关闭**，且需验证既有测试逐位不变。IMU 容错就是这样接入的。

### 4.3 改动前先确认因果，不凭直觉

偏置降级与位置预积分两处设计都是被实测推翻的：

- 「数据不可信就该关掉」——错。关闭位置预积分导致高度漂 3 倍。
- 「检测到故障就限幅」——不够。不切断错误反馈，限幅无法阻止状态被污染。

降级动作必须用对照实验验证，不能靠推理定案。

### 4.4 文档同步

改动设计决策时同步更新对应文档。`docs/imu-fault-tolerance.md` 记录了大量
「被推翻的初版设计」，这些负面轨迹是有价值的，不要清理。

## 五、工作方式

- 提交前跑全量回归：当前基线为仓库 55 个测试可执行文件中 **54 个通过**；
  唯一失败 `WindTunnelTest`（12 m/s 稳态偏移断言）是既有问题，与容错工作无关。
- 数值结论需附测量环境；小效应量（<5%）在背景负载下不可信。
- 报告与经验沉淀写入 `~/skills/reports/YYYY-MM-DD/`。
- 遇到需要改变公共接口、默认行为或删除代码时，先确认再动手。

## 六、当前未决问题

| 问题 | 影响 | 状态 |
|---|---|---|
| 位置量测理想化 | P1 验证结论可能不适用于真机 | **已验证**（§2.3），附两条包络边界 |
| ~~速度/jerk 上限为估值~~ | ~~轨迹可行域判断不可靠~~ | **已完成**（P4，见 §2.2） |
| ~~降级无轨迹执行~~ | ~~返航/紧急降落只有动作码~~ | **已完成**（P3，见 §2.5） |
| 大机动下偏置检测不可靠 | 机动污染残差基线两个量级 | 物理限制，已标注 |
| 噪声量测下偏置漏检 | 默认机动门限被陀螺抖动掩蔽（§2.3） | 已量化，取保守侧 |
| ~~GPS 级噪声超出稳定域~~ | ~~σ=0.30 m 令高度通道发散，属估计器+PID 层~~ | **已完成**（P5，见 §三 P5） |
| `WindTunnelTest` 12 m/s 断言失败 | 稳态偏移 5.84 m vs 预测 1.76 m | 既有问题，与容错无关，待查 |

其中「大机动下偏置检测不可靠」不是缺陷而是物理限制：机动时加速度计读的是
比力而非重力，残差基线抬升两个量级，偏置信号被淹没。因此机动时刻意不归因
偏置——宁可不报，不可错报。「噪声量测下偏置漏检」是同一原则的延伸代价，
已量化（见 `docs/imu-fault-tolerance.md` §6.2.2）。
