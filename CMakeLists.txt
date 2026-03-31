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
