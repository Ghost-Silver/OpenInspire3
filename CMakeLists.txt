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
# 仿真核心库：动力学 + 积分器 + 仿真器 + 体素地图
# 抽成库是为了让演示程序与回归测试共用同一份编译产物，避免源文件重复编译。
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

# 已移除 -D_GLIBCXX_DEBUG / -D_GLIBCXX_DEBUG_PEDANTIC：
# macOS 使用 libc++，这两个宏是 GCC libstdc++ 专有，实测完全空操作（无调试收益）；
# 在 Linux/GCC 上则会改变 STL 容器 ABI，与未启用该宏编译的 CTorch 混链时存在不兼容风险。
