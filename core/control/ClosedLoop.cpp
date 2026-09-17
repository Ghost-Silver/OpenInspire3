/**
 * @file ClosedLoop.cpp
 * @brief 闭环飞行仿真驱动与飞行品质评价实现
 * @author GhostFace
 * @date 2026/9/16
 */

#include "ClosedLoop.h"
#include "DroneSimulator.h"
#include "TensorUtils.h"

#include <algorithm>
#include <cmath>
#include <iomanip>

namespace oi3 {

namespace {

/// 轨迹最多保留的采样点数（超过则按等间隔抽稀）
constexpr std::size_t kMaxSamples = 2000;

void recordSample(FlightTrace &trace, const DroneSimulator &sim, double time,
                  const Tensor &thrust) {
    const std::vector<float> p = toVector(sim.state().pos);
    const std::vector<float> v = toVector(sim.state().vel);
    const std::vector<float> f = toVector(thrust);

    trace.time.push_back(time);
    trace.pos_n.push_back(p[0]);
    trace.pos_e.push_back(p[1]);
    trace.pos_d.push_back(p[2]);
    trace.vel_n.push_back(v[0]);
    trace.vel_e.push_back(v[1]);
    trace.vel_d.push_back(v[2]);
    trace.thrust_n.push_back(f[0]);
    trace.thrust_e.push_back(f[1]);
    trace.thrust_d.push_back(f[2]);
}

double norm3(double a, double b, double c) {
    return std::sqrt(a * a + b * b + c * c);
}

} // namespace

FlightTrace runClosedLoop(const Config &cfg, const FlightTask &task,
                          Controller &controller) {
    controller.reset();

    // 构造初始状态。这里状态张量是仿真初值，不参与 autograd；
    // 仍用移动构造以避免 Tensor 拷贝构造的额外节点替换开销。
    Tensor pos0 = task.initial_pos;
    Tensor vel0 = task.initial_vel;
    DroneSimulator sim(cfg, DroneState{std::move(pos0), std::move(vel0)});

    FlightTrace trace;
    const int total_steps = static_cast<int>(task.duration / cfg.dt);
    const std::size_t stride =
        std::max<std::size_t>(1, static_cast<std::size_t>(total_steps) / kMaxSamples);

    for (int i = 0; i <= total_steps; ++i) {
        const Tensor thrust = controller.computeThrust(sim.state(), task.target, sim.time());
        if (static_cast<std::size_t>(i) % stride == 0) {
            recordSample(trace, sim, sim.time(), thrust);
        }
        if (i == total_steps) {
            break;
        }
        sim.step(thrust);
    }
    return trace;
}

FlightMetrics evaluateTrace(const FlightTask &task, const FlightTrace &trace) {
    FlightMetrics metrics;
    if (trace.time.empty()) {
        return metrics;
    }

    const std::vector<float> tgt = toVector(task.target);
    const std::vector<float> pos0 = toVector(task.initial_pos);

    std::vector<double> err(trace.time.size());
    for (std::size_t i = 0; i < err.size(); ++i) {
        err[i] = norm3(static_cast<double>(trace.pos_n[i]) - tgt[0],
                       static_cast<double>(trace.pos_e[i]) - tgt[1],
                       static_cast<double>(trace.pos_d[i]) - tgt[2]);
    }

    const std::size_t n = err.size();
    metrics.terminal_error = err[n - 1];

    // 收敛时间：从末尾向前找最后一次越出容差带的采样点，其后一点即收敛时刻。
    // 若全程都在带内则收敛时间为 0；若末尾仍在带外则视为未收敛。
    std::ptrdiff_t last_violation = -1;
    for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(n) - 1; i >= 0; --i) {
        if (err[static_cast<std::size_t>(i)] > task.tolerance) {
            last_violation = i;
            break;
        }
    }
    metrics.converged = (last_violation < static_cast<std::ptrdiff_t>(n) - 1);
    metrics.settle_time = metrics.converged
                              ? trace.time[static_cast<std::size_t>(last_violation) + 1]
                              : trace.time[n - 1];

    // 稳态误差：末段 hold_ratio 时长内的位置误差模平均
    const double hold_start = trace.time[n - 1] * (1.0 - task.hold_ratio);
    double err_sum = 0.0;
    int err_count = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (trace.time[i] >= hold_start) {
            err_sum += err[i];
            ++err_count;
        }
    }
    metrics.steady_error = err_count > 0 ? err_sum / err_count : err[n - 1];

    // 最大超调：按轴判断越过目标的方向，取各轴中的最大值
    auto overshootOfAxis = [](const std::vector<float> &axis, double target,
                              double start) -> double {
        const auto hi = static_cast<double>(*std::max_element(axis.begin(), axis.end()));
        const auto lo = static_cast<double>(*std::min_element(axis.begin(), axis.end()));
        if (target > start) { // 向上运动，越过目标即超出最大值
            return std::max(0.0, hi - target);
        }
        if (target < start) {
            return std::max(0.0, target - lo);
        }
        return 0.0; // 该轴无位移
    };
    metrics.max_overshoot =
        std::max({overshootOfAxis(trace.pos_n, tgt[0], pos0[0]),
                  overshootOfAxis(trace.pos_e, tgt[1], pos0[1]),
                  overshootOfAxis(trace.pos_d, tgt[2], pos0[2])});

    // 峰值推力与控制代价
    for (std::size_t i = 0; i < n; ++i) {
        const double f = norm3(trace.thrust_n[i], trace.thrust_e[i], trace.thrust_d[i]);
        metrics.max_thrust = std::max(metrics.max_thrust, f);
    }
    for (std::size_t i = 1; i < n; ++i) {
        const double dt = trace.time[i] - trace.time[i - 1];
        const double f_prev =
            norm3(trace.thrust_n[i - 1], trace.thrust_e[i - 1], trace.thrust_d[i - 1]);
        const double f_cur = norm3(trace.thrust_n[i], trace.thrust_e[i], trace.thrust_d[i]);
        metrics.control_energy += 0.5 * (f_prev * f_prev + f_cur * f_cur) * dt;
    }

    return metrics;
}

void printMetrics(const FlightTask &task, const Controller &controller,
                  const FlightMetrics &metrics, std::ostream &os) {
    os << std::fixed << std::setprecision(4);
    os << "任务: " << task.name << "   控制器: " << controller.name() << "\n";
    os << "  收敛: " << (metrics.converged ? "是" : "否")
       << "    收敛时间: " << metrics.settle_time << " s\n";
    os << "  稳态误差: " << metrics.steady_error << " m"
       << "    终止误差: " << metrics.terminal_error << " m\n";
    os << "  最大超调: " << metrics.max_overshoot << " m"
       << "    峰值推力: " << metrics.max_thrust << " N\n";
    os << "  控制代价: " << metrics.control_energy << " N^2*s\n";
}

void printTrace(const FlightTrace &trace, std::size_t max_rows, std::ostream &os) {
    const std::size_t n = trace.time.size();
    if (n == 0) {
        os << "(空轨迹)\n";
        return;
    }

    const std::size_t rows = std::min(max_rows, n);
    const std::size_t stride = std::max<std::size_t>(1, n / rows);

    os << std::fixed << std::setprecision(4);
    os << "      t      pos_n      pos_e      pos_d      vel_n      vel_e      vel_d"
          "      |F|\n";
    for (std::size_t i = 0; i < n; i += stride) {
        const double f = norm3(trace.thrust_n[i], trace.thrust_e[i], trace.thrust_d[i]);
        os << std::setw(9) << trace.time[i] << std::setw(11) << trace.pos_n[i]
           << std::setw(11) << trace.pos_e[i] << std::setw(11) << trace.pos_d[i]
           << std::setw(11) << trace.vel_n[i] << std::setw(11) << trace.vel_e[i]
           << std::setw(11) << trace.vel_d[i] << std::setw(11) << f << "\n";
    }
    if ((n - 1) % stride != 0) {
        const std::size_t i = n - 1;
        const double f = norm3(trace.thrust_n[i], trace.thrust_e[i], trace.thrust_d[i]);
        os << std::setw(9) << trace.time[i] << std::setw(11) << trace.pos_n[i]
           << std::setw(11) << trace.pos_e[i] << std::setw(11) << trace.pos_d[i]
           << std::setw(11) << trace.vel_n[i] << std::setw(11) << trace.vel_e[i]
           << std::setw(11) << trace.vel_d[i] << std::setw(11) << f << "\n";
    }
}

} // namespace oi3
