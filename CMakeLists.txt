cmake_minimum_required(VERSION 3.20)
project(OpenInspire3 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 默认 Release：CTorch 的性能依赖其自身的 LTO / -O3 配置，Debug 会显著拖慢仿真。
# 需要断点调试时显式传 -DCMAKE_BUILD_TYPE=Debug。
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# 引入 CTorch 前关闭其 gtest 单元测试组，减少配置负担。
# 注意：CTorch 另有 test_*/bench_* 目标不受 TEST_ENABLED 控制（定义在开关之前），
# 构建时只指定本项目目标即可，CMake 不会连带构建它们。
set(TEST_ENABLED OFF CACHE BOOL "" FORCE)
set(FETCH_GOOGLETEST OFF CACHE BOOL "" FORCE)

# CTorch 子模块
add_subdirectory(core/CTorch)

# ---------------------------------------------------------------------------
# 仿真核心库：动力学 + 积分步进 + 仿真器 + 体素地图
# 抽成库是为了让演示程序与回归测试共用同一份编译产物。
# ---------------------------------------------------------------------------
add_library(oi3_simulator STATIC
        core/simulator/PhysicsEngine/DroneDynamics.cpp
        core/simulator/PhysicsEngine/DroneSimulator.cpp
        core/simulator/PhysicsEngine/TensorUtils.cpp
        core/simulator/PhysicsEngine/SixDofDynamics.cpp
        core/simulator/PhysicsEngine/SixDofDynamicsDiff.cpp
        core/simulator/PhysicsEngine/ParameterIdentification.cpp
        core/simulator/PhysicsEngine/SixDofSimulator.cpp
        core/control/PidController.cpp
        core/control/ShootingExpert.cpp
        core/control/ShootingMpc.cpp
        core/control/SixDofPidController.cpp
        core/control/ClosedLoop.cpp
        core/control/HoverEnv.cpp
        core/rl/ContinuousPPO.cpp
        core/Map/VoxelMap.cpp
)

target_include_directories(oi3_simulator PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/core/CTorch/include
        ${CMAKE_CURRENT_SOURCE_DIR}/core/simulator/PhysicsEngine
        ${CMAKE_CURRENT_SOURCE_DIR}/core/control
        ${CMAKE_CURRENT_SOURCE_DIR}/core/rl
        ${CMAKE_CURRENT_SOURCE_DIR}/core/Map
)

# 编译选项用 target_compile_options 而非 add_compile_options：
# 后者会向子目录（CTorch）传递，给 CTorch 追加 -O0 会破坏其 Release 优化。
target_compile_options(oi3_simulator PRIVATE -Wall -Wextra)

target_link_libraries(oi3_simulator PUBLIC CTorch)

# ---------------------------------------------------------------------------
# 演示程序
# ---------------------------------------------------------------------------
add_executable(DroneSimulation main.cpp)
target_compile_options(DroneSimulation PRIVATE -Wall -Wextra)
target_link_libraries(DroneSimulation PRIVATE oi3_simulator)

# ---------------------------------------------------------------------------
# 回归测试：数值正确性 + 可微性
# ---------------------------------------------------------------------------
add_executable(test_sim tests/test_sim.cpp)
target_compile_options(test_sim PRIVATE -Wall -Wextra)
target_link_libraries(test_sim PRIVATE oi3_simulator)

enable_testing()
add_test(NAME sim_regression COMMAND test_sim)

# ---------------------------------------------------------------------------
# 闭环控制：控制器输出推力 -> 真实动力学积分 -> 状态回馈
# 控制器为可替换部件，PID 是第一个实现，也是后续学习型控制器的对照基线。
# ---------------------------------------------------------------------------
add_executable(PidHoverTest core/control/PidHoverTest.cpp)
target_compile_options(PidHoverTest PRIVATE -Wall -Wextra)
target_link_libraries(PidHoverTest PRIVATE oi3_simulator)
add_test(NAME pid_hover COMMAND PidHoverTest)

# 六自由度：倾斜产生水平分量、四元数积分、级联 PID 定点悬停
add_executable(AttitudeGainTest core/control/AttitudeGainTest.cpp)
target_compile_options(AttitudeGainTest PRIVATE -Wall -Wextra)
target_link_libraries(AttitudeGainTest PRIVATE oi3_simulator)

add_executable(SixDofHoverTest core/control/SixDofHoverTest.cpp)
target_compile_options(SixDofHoverTest PRIVATE -Wall -Wextra)
target_link_libraries(SixDofHoverTest PRIVATE oi3_simulator)
add_test(NAME six_dof_hover COMMAND SixDofHoverTest)

add_executable(MpcHoverTest core/control/MpcHoverTest.cpp)
target_compile_options(MpcHoverTest PRIVATE -Wall -Wextra)
target_link_libraries(MpcHoverTest PRIVATE oi3_simulator)

add_executable(TrainShootingPolicy core/control/TrainShootingPolicy.cpp)
target_compile_options(TrainShootingPolicy PRIVATE -Wall -Wextra)
target_link_libraries(TrainShootingPolicy PRIVATE oi3_simulator)

add_executable(ModelCalibrationTest core/simulator/PhysicsEngine/ModelCalibrationTest.cpp)
target_compile_options(ModelCalibrationTest PRIVATE -Wall -Wextra)
target_link_libraries(ModelCalibrationTest PRIVATE oi3_simulator)

add_executable(ParameterIdTest core/simulator/PhysicsEngine/ParameterIdTest.cpp)
target_compile_options(ParameterIdTest PRIVATE -Wall -Wextra)
target_link_libraries(ParameterIdTest PRIVATE oi3_simulator)

add_executable(SixDofGradTest core/simulator/PhysicsEngine/SixDofGradTest.cpp)
target_compile_options(SixDofGradTest PRIVATE -Wall -Wextra)
target_link_libraries(SixDofGradTest PRIVATE oi3_simulator)
add_test(NAME six_dof_grad COMMAND SixDofGradTest)

# 神经网络飞控训练：连续 PPO 学习定点悬停，并与 PID 在同一指标下对比。
# 训练耗时随回合数增长，默认不加入 ctest（可用 --episodes 调整）。
add_executable(TrainHoverPolicy core/control/TrainHoverPolicy.cpp)
target_compile_options(TrainHoverPolicy PRIVATE -Wall -Wextra)
target_link_libraries(TrainHoverPolicy PRIVATE oi3_simulator)

# 行为克隆诊断：用 PID 生成专家动作让策略网络做监督学习，
# 以分离「网络/反传是否可用」与「PPO 梯度估计是否正确」两类问题
add_executable(BcDiagnostic core/control/BcDiagnostic.cpp)
target_compile_options(BcDiagnostic PRIVATE -Wall -Wextra)
target_link_libraries(BcDiagnostic PRIVATE oi3_simulator)

# ===========================================================================
# core/rl/ —— 强化学习栈（搜索环境 / EnvManager / PPO / AM-MAPPO / 贪心基线）
#
# 迁移进度（2026/9/16）：
#   [x] PPOAgent —— autograd 接口已迁移、参数所有权已修正、数值运算已保图
#   [ ] Environment / EnvManager —— 待接入新的仿真核心
#   [ ] 贪心基线与 AM-MAPPO 测试 —— 待 Environment 就绪后恢复
#
# 原阻塞原因（PPOAgent 部分已解决，留档）：
#   1. 使用当前 CTorch 中不存在的自动微分接口：
#        AutoDiff / AutoDiffContext::Guard / Tensor::backward()
#      已改为 AutoGrad::backward(tensor.getRelatedNode(), retain_graph)。
#   2. 多处用 `float* p = t.data<float>()` 手写 clamp / min 循环，
#      直接改写数据指针不注册 grad_fn，会切断计算图。
#      已改为 ratio.clamp(...) 与 surr1.min(surr2)。
#   3. 依赖 core/simulator/PhysicsEngine/OIEngine.{h,cpp} 与 RK4Solver.cpp
#      —— 这三个文件已被 DroneTypes / DroneDynamics / DroneSimulator /
#      TensorUtils 取代并删除。
#   4. 旧 CMakeLists 引用的 core/Map/VoxelMapMultiDroneTest.cpp 在仓库中不存在。
# ===========================================================================

# PPO 智能体最小测试（不依赖环境层）
add_executable(PPOAgentSimpleTest
        core/rl/PPOAgent.cpp
        core/rl/PPOAgentSimpleTest.cpp)
target_include_directories(PPOAgentSimpleTest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/core/rl)
target_compile_options(PPOAgentSimpleTest PRIVATE -Wall -Wextra)
target_link_libraries(PPOAgentSimpleTest PRIVATE oi3_simulator)

# PPO 迁移回归：验证梯度确实回传到参数（判据不是"能编译"，而是"参数被更新"）
add_executable(test_ppo_grad
        core/rl/PPOAgent.cpp
        tests/test_ppo_grad.cpp)
target_include_directories(test_ppo_grad PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/core/rl)
target_compile_options(test_ppo_grad PRIVATE -Wall -Wextra)
target_link_libraries(test_ppo_grad PRIVATE oi3_simulator)
add_test(NAME ppo_migration COMMAND test_ppo_grad)

# 已移除 -D_GLIBCXX_DEBUG / -D_GLIBCXX_DEBUG_PEDANTIC：
# macOS 使用 libc++，这两个宏是 GCC libstdc++ 专有，实测完全空操作（无调试收益）；
# 在 Linux/GCC 上则会改变 STL 容器 ABI，与未编译该宏的 CTorch 混链时存在不兼容风险。


# 环境层与贪心基线（不涉及张量运算，接口与当前 CTorch 无耦合）
add_executable(EnvironmentTest
        core/Map/VoxelMap.cpp core/rl/Environment.cpp core/rl/EnvironmentTest.cpp)
target_include_directories(EnvironmentTest PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/core/rl ${CMAKE_CURRENT_SOURCE_DIR}/core/Map)
target_link_libraries(EnvironmentTest PRIVATE oi3_simulator)

add_executable(GreedyBaselineTest
        core/Map/VoxelMap.cpp core/rl/Environment.cpp core/rl/EnvManager.cpp
        core/rl/Baseline.cpp core/rl/GreedyBaselineTest.cpp)
target_include_directories(GreedyBaselineTest PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/core/rl ${CMAKE_CURRENT_SOURCE_DIR}/core/Map)
target_link_libraries(GreedyBaselineTest PRIVATE oi3_simulator)

add_executable(GreedyBaselineHardTest
        core/Map/VoxelMap.cpp core/rl/Environment.cpp core/rl/EnvManager.cpp
        core/rl/Baseline.cpp core/rl/GreedyBaselineHardTest.cpp)
target_include_directories(GreedyBaselineHardTest PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/core/rl ${CMAKE_CURRENT_SOURCE_DIR}/core/Map)
target_link_libraries(GreedyBaselineHardTest PRIVATE oi3_simulator)

add_executable(GreedyBaselineUltraHardTest
        core/Map/VoxelMap.cpp core/rl/Environment.cpp core/rl/EnvManager.cpp
        core/rl/Baseline.cpp core/rl/GreedyBaselineUltraHardTest.cpp)
target_include_directories(GreedyBaselineUltraHardTest PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/core/rl ${CMAKE_CURRENT_SOURCE_DIR}/core/Map)
target_link_libraries(GreedyBaselineUltraHardTest PRIVATE oi3_simulator)

# AM-MAPPO 超难度训练测试：在 ultra-hard 场景上训练 PPO，并与贪心基线对比
add_executable(AMMAPPOUltraHardTest
        core/Map/VoxelMap.cpp core/rl/Environment.cpp core/rl/EnvManager.cpp
        core/rl/Baseline.cpp core/rl/PPOAgent.cpp core/rl/AMMAPPOUltraHardTest.cpp)
target_include_directories(AMMAPPOUltraHardTest PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/core/rl ${CMAKE_CURRENT_SOURCE_DIR}/core/Map)
target_compile_options(AMMAPPOUltraHardTest PRIVATE -Wall -Wextra)
target_link_libraries(AMMAPPOUltraHardTest PRIVATE oi3_simulator)

# 环境高级测试（多机、障碍、目标发现）
add_executable(EnvironmentTestAdvanced
        core/Map/VoxelMap.cpp core/rl/Environment.cpp core/rl/EnvironmentTestAdvanced.cpp)
target_include_directories(EnvironmentTestAdvanced PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/core/rl ${CMAKE_CURRENT_SOURCE_DIR}/core/Map)
target_link_libraries(EnvironmentTestAdvanced PRIVATE oi3_simulator)

# 集成测试（UAV 管理器 / VoxelMap / Environment 配合）
add_executable(IntegrationTest
        core/Map/VoxelMap.cpp core/rl/Environment.cpp core/rl/IntegrationTest.cpp)
target_include_directories(IntegrationTest PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/core/rl ${CMAKE_CURRENT_SOURCE_DIR}/core/Map)
target_link_libraries(IntegrationTest PRIVATE oi3_simulator)

# 完整系统测试（各模块协同）
add_executable(FullSystemTest
        core/Map/VoxelMap.cpp core/rl/Environment.cpp core/rl/FullSystemTest.cpp)
target_include_directories(FullSystemTest PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/core/rl ${CMAKE_CURRENT_SOURCE_DIR}/core/Map)
target_link_libraries(FullSystemTest PRIVATE oi3_simulator)

# PPO 智能体端到端测试（环境 + 智能体）
add_executable(PPOAgentTest
        core/Map/VoxelMap.cpp core/rl/Environment.cpp core/rl/EnvManager.cpp
        core/rl/Baseline.cpp core/rl/PPOAgent.cpp core/rl/PPOAgentTest.cpp)
target_include_directories(PPOAgentTest PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/core/rl ${CMAKE_CURRENT_SOURCE_DIR}/core/Map)
target_link_libraries(PPOAgentTest PRIVATE oi3_simulator)
