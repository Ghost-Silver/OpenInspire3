/**
 * @file PidHoverTest.cpp
 * @brief PID 定点悬停闭环测试
 * @author GhostFace
 * @date 2026/9/16
 *
 * 闭环的第一个端到端用例：控制器输出推力 → 真实动力学(RK4)积分 → 新状态回馈。
 * 退出码反映是否在时限内收敛，可直接接入 CI。
 */

#include "ClosedLoop.h"
#include "PidController.h"
#include "TensorUtils.h"

#include <iostream>

using namespace oi3;

int main() {
    CtorchError::setPrintLevel(PrintLevel::MINIUM);

    Config cfg;
    cfg.dt = 0.001;     // 1 kHz
    cfg.mass = 1.0;     // 1 kg
    cfg.gravity = 9.81; // m/s^2

    // 起飞并平移到目标点：NED 系下 down 为负表示向上
    FlightTask task;
    task.name = "定点悬停 (0,0,0) -> (1,1,-1)";
    task.initial_pos = makeVec3(0.0f, 0.0f, 0.0f);
    task.initial_vel = makeVec3(0.0f, 0.0f, 0.0f);
    task.target = makeVec3(1.0f, 1.0f, -1.0f);
    task.duration = 5.0;
    task.tolerance = 0.05;
    task.hold_ratio = 0.1;

    PidController pid(cfg);

    std::cout << "===== PID 定点悬停闭环 =====\n";
    std::cout << "质量 " << cfg.mass << " kg，重力 " << cfg.gravity
              << " m/s^2，步长 " << cfg.dt << " s\n\n";

    const FlightTrace trace = runClosedLoop(cfg, task, pid);
    const FlightMetrics metrics = evaluateTrace(task, trace);

    printMetrics(task, pid, metrics);

    std::cout << "\n轨迹摘要（每 " << (trace.time.size() / 10) << " 个采样点一行）:\n";
    printTrace(trace, 10);

    std::cout << "\n结果: " << (metrics.converged ? "收敛" : "未收敛") << "\n";
    return metrics.converged ? 0 : 1;
}
