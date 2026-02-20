/**
 * @file RK4solver.h
 * @brief OpenInspire3 通用 RK4 求解器
 * @author GhostFace
 * @date 2026/2/19
 */

#ifndef RK4SOLVER_H
#define RK4SOLVER_H

#include "../../CTorch/include/Tensor.h"
#include <functional>

// ==========================================
//  定义通用 RK4 求解器
// ==========================================

class RK4Integrator {
public:
    double dt;

    RK4Integrator(double step_size);

    // 定义 ODE 函数类型：支持时间依赖的函数
    using ODEFunc = std::function<Tensor(double, const Tensor&)>;
    
    // 简化版 ODE 函数类型：不依赖时间
    using TimeIndependentODEFunc = std::function<Tensor(const Tensor&)>;

    // 通用步进函数：支持时间依赖的 ODE
    Tensor step(ODEFunc f, double t, const Tensor& y);

    // 重载步进函数：支持不依赖时间的 ODE
    Tensor step(TimeIndependentODEFunc f, const Tensor& y);

    // 积分函数：从初始时间积分到终止时间
    Tensor integrate(ODEFunc f, double t_start, double t_end, const Tensor& y0);

    // 重载积分函数：不依赖时间的 ODE
    Tensor integrate(TimeIndependentODEFunc f, double t_start, double t_end, const Tensor& y0);
};

// ==========================================
// 辅助工具函数
// ==========================================

// 创建标量 Tensor
Tensor make_scalar_tensor(float val);

// 创建向量 Tensor
Tensor make_vector_tensor(const std::vector<float>& vals);

// 获取标量值
float get_scalar_value(const Tensor& t);

// 获取向量值
std::vector<float> get_vector_value(const Tensor& t);

// 打印 Tensor 值
void print_tensor(const Tensor& t, const std::string& name = "");

#endif // RK4SOLVER_H
