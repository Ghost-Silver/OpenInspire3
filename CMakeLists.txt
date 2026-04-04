cmake_minimum_required(VERSION 3.20)
project(OpenInspire3)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_BUILD_TYPE Debug)

add_compile_options(
    -Wall 
    -Wextra
    -g
    -O0
)

add_definitions(-D_GLIBCXX_DEBUG)
add_definitions(-D_GLIBCXX_DEBUG_PEDANTIC)

# 添加CTorch子目录
add_subdirectory(core/CTorch)

# 主可执行文件（使用main.cpp）
add_executable(DroneSimulation
        main.cpp
        core/simulator/PhysicsEngine/RK4Solver.cpp
        core/simulator/PhysicsEngine/OIEngine.cpp)

# 链接CTorch库
target_link_libraries(DroneSimulation PRIVATE CTorch)

# 添加CTorch头文件目录
target_include_directories(DroneSimulation PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/core/CTorch/include
        ${CMAKE_CURRENT_SOURCE_DIR}/core/simulator/PhysicsEngine
)

# 原有可执行文件（保留）
add_executable(OpenInspire3
        main.cpp
        core/simulator/PhysicsEngine/OIEngine.cpp
        core/simulator/PhysicsEngine/RK4Solver.cpp)

# 链接CTorch库
target_link_libraries(OpenInspire3 PRIVATE CTorch)

# 添加CTorch头文件目录
target_include_directories(OpenInspire3 PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/core/CTorch/include
)

# 多无人机测试
add_executable(VoxelMapMultiDroneTest
        core/Map/VoxelMap.cpp
        core/Map/VoxelMapMultiDroneTest.cpp
        core/simulator/PhysicsEngine/OIEngine.cpp
        core/simulator/PhysicsEngine/RK4Solver.cpp)

# 链接CTorch库
target_link_libraries(VoxelMapMultiDroneTest PRIVATE CTorch)

# 添加头文件目录
target_include_directories(VoxelMapMultiDroneTest PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/core/CTorch/include
        ${CMAKE_CURRENT_SOURCE_DIR}/core/simulator/PhysicsEngine
        ${CMAKE_CURRENT_SOURCE_DIR}/core/Map
)

# 强化学习环境测试
add_executable(RLEnvironmentTest
        core/Map/VoxelMap.cpp
        core/rl/ActionSpace.h
        core/rl/Environment.h
        core/rl/Environment.cpp
        core/rl/EnvironmentTest.cpp
        core/simulator/PhysicsEngine/OIEngine.cpp
        core/simulator/PhysicsEngine/RK4Solver.cpp)

# 链接CTorch库
target_link_libraries(RLEnvironmentTest PRIVATE CTorch)

# 添加头文件目录
target_include_directories(RLEnvironmentTest PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/core/CTorch/include
        ${CMAKE_CURRENT_SOURCE_DIR}/core/simulator/PhysicsEngine
        ${CMAKE_CURRENT_SOURCE_DIR}/core/Map
        ${CMAKE_CURRENT_SOURCE_DIR}/core/rl
)

# 强化学习环境高级测试
add_executable(RLEnvironmentTestAdvanced
        core/Map/VoxelMap.cpp
        core/rl/ActionSpace.h
        core/rl/Environment.h
        core/rl/Environment.cpp
        core/rl/EnvironmentTestAdvanced.cpp
        core/simulator/PhysicsEngine/OIEngine.cpp
        core/simulator/PhysicsEngine/RK4Solver.cpp)

# 链接CTorch库
target_link_libraries(RLEnvironmentTestAdvanced PRIVATE CTorch)

# 添加头文件目录
target_include_directories(RLEnvironmentTestAdvanced PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/core/CTorch/include
        ${CMAKE_CURRENT_SOURCE_DIR}/core/simulator/PhysicsEngine
        ${CMAKE_CURRENT_SOURCE_DIR}/core/Map
        ${CMAKE_CURRENT_SOURCE_DIR}/core/rl
)

# 综合测试：仿真器与 RK4、VoxelMap、UAV 管理器的配合
add_executable(IntegrationTest
        core/Map/VoxelMap.cpp
        core/rl/ActionSpace.h
        core/rl/Environment.h
        core/rl/Environment.cpp
        core/rl/IntegrationTest.cpp
        core/simulator/PhysicsEngine/OIEngine.cpp
        core/simulator/PhysicsEngine/RK4Solver.cpp)

# 链接CTorch库
target_link_libraries(IntegrationTest PRIVATE CTorch)

# 添加头文件目录
target_include_directories(IntegrationTest PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/core/CTorch/include
        ${CMAKE_CURRENT_SOURCE_DIR}/core/simulator/PhysicsEngine
        ${CMAKE_CURRENT_SOURCE_DIR}/core/Map
        ${CMAKE_CURRENT_SOURCE_DIR}/core/rl
)

# 完整系统测试
add_executable(FullSystemTest
        core/Map/VoxelMap.cpp
        core/rl/ActionSpace.h
        core/rl/Environment.h
        core/rl/Environment.cpp
        core/rl/FullSystemTest.cpp
        core/simulator/PhysicsEngine/OIEngine.cpp
        core/simulator/PhysicsEngine/RK4Solver.cpp)

# 链接CTorch库
target_link_libraries(FullSystemTest PRIVATE CTorch)

# 添加头文件目录
target_include_directories(FullSystemTest PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/core/CTorch/include
        ${CMAKE_CURRENT_SOURCE_DIR}/core/simulator/PhysicsEngine
        ${CMAKE_CURRENT_SOURCE_DIR}/core/Map
        ${CMAKE_CURRENT_SOURCE_DIR}/core/rl
)

# PPO 智能体测试
add_executable(PPOAgentTest
        core/Map/VoxelMap.cpp
        core/rl/ActionSpace.h
        core/rl/Environment.h
        core/rl/Environment.cpp
        core/rl/PPOAgent.h
        core/rl/PPOAgent.cpp
        core/rl/EnvManager.h
        core/rl/EnvManager.cpp
        core/rl/PPOAgentTest.cpp
        core/simulator/PhysicsEngine/OIEngine.cpp
        core/simulator/PhysicsEngine/RK4Solver.cpp)

# 链接CTorch库
target_link_libraries(PPOAgentTest PRIVATE CTorch)

# 添加头文件目录
target_include_directories(PPOAgentTest PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/core/CTorch/include
        ${CMAKE_CURRENT_SOURCE_DIR}/core/simulator/PhysicsEngine
        ${CMAKE_CURRENT_SOURCE_DIR}/core/Map
        ${CMAKE_CURRENT_SOURCE_DIR}/core/rl
)

# 简单 PPO 智能体测试
add_executable(PPOAgentSimpleTest
        core/rl/PPOAgent.h
        core/rl/PPOAgent.cpp
        core/rl/PPOAgentSimpleTest.cpp)

# 链接CTorch库
target_link_libraries(PPOAgentSimpleTest PRIVATE CTorch)

# 添加头文件目录
target_include_directories(PPOAgentSimpleTest PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/core/CTorch/include
        ${CMAKE_CURRENT_SOURCE_DIR}/core/rl
)

# 贪心覆盖基线测试
add_executable(GreedyBaselineTest
        core/Map/VoxelMap.cpp
        core/rl/ActionSpace.h
        core/rl/Environment.h
        core/rl/Environment.cpp
        core/rl/EnvManager.h
        core/rl/EnvManager.cpp
        core/rl/GreedyBaselineTest.cpp
        core/simulator/PhysicsEngine/OIEngine.cpp
        core/simulator/PhysicsEngine/RK4Solver.cpp)

# 链接CTorch库
target_link_libraries(GreedyBaselineTest PRIVATE CTorch)

# 添加头文件目录
target_include_directories(GreedyBaselineTest PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/core/CTorch/include
        ${CMAKE_CURRENT_SOURCE_DIR}/core/simulator/PhysicsEngine
        ${CMAKE_CURRENT_SOURCE_DIR}/core/Map
        ${CMAKE_CURRENT_SOURCE_DIR}/core/rl
)

# 高难度贪心覆盖基线测试
add_executable(GreedyBaselineHardTest
        core/Map/VoxelMap.cpp
        core/rl/ActionSpace.h
        core/rl/Environment.h
        core/rl/Environment.cpp
        core/rl/EnvManager.h
        core/rl/EnvManager.cpp
        core/rl/GreedyBaselineHardTest.cpp
        core/simulator/PhysicsEngine/OIEngine.cpp
        core/simulator/PhysicsEngine/RK4Solver.cpp)

# 链接CTorch库
target_link_libraries(GreedyBaselineHardTest PRIVATE CTorch)

# 添加头文件目录
target_include_directories(GreedyBaselineHardTest PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/core/CTorch/include
        ${CMAKE_CURRENT_SOURCE_DIR}/core/simulator/PhysicsEngine
        ${CMAKE_CURRENT_SOURCE_DIR}/core/Map
        ${CMAKE_CURRENT_SOURCE_DIR}/core/rl
)

# 超难度贪心覆盖基线测试
add_executable(GreedyBaselineUltraHardTest
        core/Map/VoxelMap.cpp
        core/rl/ActionSpace.h
        core/rl/Environment.h
        core/rl/Environment.cpp
        core/rl/EnvManager.h
        core/rl/EnvManager.cpp
        core/rl/GreedyBaselineUltraHardTest.cpp
        core/simulator/PhysicsEngine/OIEngine.cpp
        core/simulator/PhysicsEngine/RK4Solver.cpp)

# 链接CTorch库
target_link_libraries(GreedyBaselineUltraHardTest PRIVATE CTorch)

# 添加头文件目录
target_include_directories(GreedyBaselineUltraHardTest PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/core/CTorch/include
        ${CMAKE_CURRENT_SOURCE_DIR}/core/simulator/PhysicsEngine
        ${CMAKE_CURRENT_SOURCE_DIR}/core/Map
        ${CMAKE_CURRENT_SOURCE_DIR}/core/rl
)

# 超难度 AM-MAPPO 测试
add_executable(AMMAPPOUltraHardTest
        core/Map/VoxelMap.cpp
        core/rl/ActionSpace.h
        core/rl/Environment.h
        core/rl/Environment.cpp
        core/rl/PPOAgent.h
        core/rl/PPOAgent.cpp
        core/rl/EnvManager.h
        core/rl/EnvManager.cpp
        core/rl/AMMAPPOUltraHardTest.cpp
        core/simulator/PhysicsEngine/OIEngine.cpp
        core/simulator/PhysicsEngine/RK4Solver.cpp)

# 链接CTorch库
target_link_libraries(AMMAPPOUltraHardTest PRIVATE CTorch)

# 添加头文件目录
target_include_directories(AMMAPPOUltraHardTest PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/core/CTorch/include
        ${CMAKE_CURRENT_SOURCE_DIR}/core/simulator/PhysicsEngine
        ${CMAKE_CURRENT_SOURCE_DIR}/core/Map
        ${CMAKE_CURRENT_SOURCE_DIR}/core/rl
)
