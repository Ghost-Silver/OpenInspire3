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
        core/Map/VoxelMap.cpp
)

target_include_directories(oi3_simulator PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/core/CTorch/include
        ${CMAKE_CURRENT_SOURCE_DIR}/core/simulator/PhysicsEngine
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

# ===========================================================================
# core/rl/ —— 强化学习栈（搜索环境 / EnvManager / PPO / AM-MAPPO / 贪心基线）
#
# 状态：源文件保留在仓库中，但**暂不纳入构建**。
#
# 阻塞原因（逐条均已核实）：
#   1. 使用当前 CTorch 中不存在的自动微分接口：
#        AutoDiff / AutoDiffContext::Guard / Tensor::backward()
#      当前接口为 AutoGrad::backward(tensor.getRelatedNode(), retain_graph)。
#      见 core/rl/PPOAgent.cpp 的 update()。
#   2. 多处用 `float* p = t.data<float>()` 手写 clamp / min 循环
#      （PPOAgent.cpp 中 clipped_ratio 与 min_surr 两处），
#      直接改写数据指针不建立 grad_fn，会切断计算图。
#   3. 依赖 core/simulator/PhysicsEngine/OIEngine.{h,cpp} 与 RK4Solver.cpp
#      —— 这三个文件已被 DroneTypes / DroneDynamics / DroneSimulator /
#      TensorUtils 取代并删除。
#   4. 旧 CMakeLists 引用的 core/Map/VoxelMapMultiDroneTest.cpp 在仓库中不存在。
#
# 迁移方向：
#   a. 环境层（Environment / EnvManager）改接 DroneSimulator；
#   b. PPO 的动作采样与损失全部改为纯算子表达（保图），
#      clamp 用 Min/Max 算子、min 用逐元素 Min，不再手写指针循环；
#   c. total_loss.backward() 改为 AutoGrad::backward(total_loss.getRelatedNode(), false)。
#
# 迁移完成后再逐条恢复下列 target。
# ===========================================================================
# add_executable(PPOAgentTest ...)
# add_executable(AMMAPPOUltraHardTest ...)
# add_executable(GreedyBaselineTest ...)  等，见 706470b 提交中的原始定义

# 已移除 -D_GLIBCXX_DEBUG / -D_GLIBCXX_DEBUG_PEDANTIC：
# macOS 使用 libc++，这两个宏是 GCC libstdc++ 专有，实测完全空操作（无调试收益）；
# 在 Linux/GCC 上则会改变 STL 容器 ABI，与未编译该宏的 CTorch 混链时存在不兼容风险。
