/**
 * @file RK4Solver.cpp
 * @brief OpenInspire3 通用 RK4 求解器
 * @author GhostFace
 * @date 2026/2/19
 */

#include "RK4Solver.h"
#include <cmath>
#include <iostream>

// ==========================================
//  RK4Integrator 类实现
// ==========================================

RK4Integrator::RK4Integrator(double step_size) : dt(step_size) {}

// 通用步进函数：支持时间依赖的 ODE
Tensor RK4Integrator::step(ODEFunc f, double t, const Tensor &y) {
  // 计算 k1-k4 系数
  Tensor k1 = f(t, y) * dt;
  Tensor k2 = f(t + dt / 2, y + k1 * 0.5) * dt;
  Tensor k3 = f(t + dt / 2, y + k2 * 0.5) * dt;
  Tensor k4 = f(t + dt, y + k3) * dt;

  // 计算下一个状态
  Tensor y_new = y + (k1 + k2 * 2.0 + k3 * 2.0 + k4) / 6.0;
  return y_new;
}

// 重载步进函数：支持不依赖时间的 ODE
Tensor RK4Integrator::step(TimeIndependentODEFunc f, const Tensor &y) {
  // 包装成时间依赖的函数
  auto wrapped_func = [&f](double, const Tensor &y_in) -> Tensor {
    return f(y_in);
  };
  return step(wrapped_func, 0.0, y);
}

// 积分函数：从初始时间积分到终止时间
Tensor RK4Integrator::integrate(ODEFunc f, double t_start, double t_end,
                                const Tensor &y0) {
  Tensor y = y0;
  double t = t_start;

  // 计算步数
  int steps = static_cast<int>((t_end - t_start) / dt);

  // 积分循环
  for (int i = 0; i < steps; ++i) {
    y = step(f, t, y);
    t += dt;
  }

  // 处理剩余时间步
  double remaining_time = (t_end - t_start) - steps * dt;
  if (remaining_time > 1e-10) {
    double original_dt = dt;
    dt = remaining_time;
    y = step(f, t, y);
    dt = original_dt;
  }

  return y;
}

// 重载积分函数：不依赖时间的 ODE
Tensor RK4Integrator::integrate(TimeIndependentODEFunc f, double t_start,
                                double t_end, const Tensor &y0) {
  auto wrapped_func = [&f](double, const Tensor &y_in) -> Tensor {
    return f(y_in);
  };
  return integrate(wrapped_func, t_start, t_end, y0);
}

// ==========================================
// 辅助工具函数实现
// ==========================================

// 创建标量 Tensor
Tensor make_scalar_tensor(float val) {
  Tensor t(ShapeTag{}, {1UL}, DType::kFloat, DeviceType::kCPU);
  float *data = const_cast<float *>(t.data<float>());
  data[0] = val;
  return t;
}

// 创建向量 Tensor
Tensor make_vector_tensor(const std::vector<float> &vals) {
  Tensor t(ShapeTag{}, {static_cast<unsigned long>(vals.size())}, DType::kFloat,
           DeviceType::kCPU);
  float *data = const_cast<float *>(t.data<float>());
  for (size_t i = 0; i < vals.size(); ++i) {
    data[i] = vals[i];
  }
  return t;
}

// 获取标量值
float get_scalar_value(const Tensor &t) { return t.data<float>()[0]; }

// 获取向量值
std::vector<float> get_vector_value(const Tensor &t) {
  int size = static_cast<int>(t.shape()[0]);
  std::vector<float> vals(size);
  const float *data = t.data<float>();
  for (int i = 0; i < size; ++i) {
    vals[i] = data[i];
  }
  return vals;
}

// 打印 Tensor 值
void print_tensor(const Tensor &t, const std::string &name) {
  if (!name.empty()) {
    std::cout << name << ": ";
  }

  // 使用 Tensor 类的 << 操作符
  std::cout << t << std::endl;

  // 额外打印张量的具体值
  int size = static_cast<int>(t.shape()[0]);
  const float *data = t.data<float>();

  if (size == 1) {
    std::cout << "Value: " << data[0] << std::endl;
  } else {
    std::cout << "Values: [";
    for (int i = 0; i < size; ++i) {
      std::cout << data[i];
      if (i < size - 1)
        std::cout << ", ";
    }
    std::cout << "]" << std::endl;
  }
}

// ==========================================
// 测试函数
// ==========================================

// 测试 1: 指数增长 dy/dt = y
void test_exponential_growth() {
  std::cout << "\n=== Test 1: Exponential Growth ===" << std::endl;
  std::cout << "ODE: dy/dt = y" << std::endl;
  std::cout << "Initial Condition: y(0) = 1.0" << std::endl;
  std::cout << "Target: y(1.0) = e ~ 2.71828" << std::endl;
  std::cout << "-----------------------------------" << std::endl;

  // 配置
  double dt = 0.01;
  double t_start = 0.0;
  double t_end = 1.0;

  // 初始化状态
  Tensor y0 = make_scalar_tensor(1.0f);

  // 创建求解器
  RK4Integrator solver(dt);

  // 定义 ODE
  auto ode_func = [](double t, const Tensor &y) -> Tensor {
    return y; // dy/dt = y
  };

  // 积分
  Tensor y_final = solver.integrate(ode_func, t_start, t_end, y0);

  // 结果对比
  float numerical_result = get_scalar_value(y_final);
  double analytical_result = std::exp(1.0);

  std::cout << "Final Result (Numerical): " << numerical_result << std::endl;
  std::cout << "Final Result (Analytical): " << analytical_result << std::endl;
  std::cout << "Absolute Error: "
            << std::abs(numerical_result - analytical_result) << std::endl;
}

// 测试 2: 简谐运动 d^2x/dt^2 = -x
void test_harmonic_oscillator() {
  std::cout << "\n=== Test 2: Harmonic Oscillator ===" << std::endl;
  std::cout << "ODE: d^2x/dt^2 = -x" << std::endl;
  std::cout << "Initial Conditions: x(0) = 1.0, v(0) = 0.0" << std::endl;
  std::cout << "Target: x(π) = -1.0, v(π) = 0.0" << std::endl;
  std::cout << "-----------------------------------" << std::endl;

  // 配置
  double dt = 0.01;
  double t_start = 0.0;
  double t_end = M_PI;

  // 初始化状态 [x, v]
  std::vector<float> initial_state = {1.0f, 0.0f};
  Tensor y0 = make_vector_tensor(initial_state);

  // 创建求解器
  RK4Integrator solver(dt);

  // 定义 ODE 系统
  auto ode_func = [](double t, const Tensor &y) -> Tensor {
    float x = y.data<float>()[0];
    float v = y.data<float>()[1];

    // dx/dt = v
    // dv/dt = -x
    std::vector<float> dydt = {v, -x};
    return make_vector_tensor(dydt);
  };

  // 积分
  Tensor y_final = solver.integrate(ode_func, t_start, t_end, y0);

  // 获取结果
  std::vector<float> result = get_vector_value(y_final);
  float x_final = result[0];
  float v_final = result[1];

  // 结果对比
  double analytical_x = -1.0; // cos(π)
  double analytical_v = 0.0;  // -sin(π)

  std::cout << "Final Result (Numerical): x = " << x_final
            << ", v = " << v_final << std::endl;
  std::cout << "Final Result (Analytical): x = " << analytical_x
            << ", v = " << analytical_v << std::endl;
  std::cout << "Absolute Error in x: " << std::abs(x_final - analytical_x)
            << std::endl;
  std::cout << "Absolute Error in v: " << std::abs(v_final - analytical_v)
            << std::endl;
}

// 测试 3: 逻辑斯蒂增长 dy/dt = r*y*(1 - y/K)
void test_logistic_growth() {
  std::cout << "\n=== Test 3: Logistic Growth ===" << std::endl;
  std::cout << "ODE: dy/dt = r*y*(1 - y/K)" << std::endl;
  std::cout << "Parameters: r = 1.0, K = 10.0" << std::endl;
  std::cout << "Initial Condition: y(0) = 1.0" << std::endl;
  std::cout << "Target: y(5.0) ≈ 10.0 (carrying capacity)" << std::endl;
  std::cout << "-----------------------------------" << std::endl;

  // 配置
  double dt = 0.01;
  double t_start = 0.0;
  double t_end = 5.0;
  double r = 1.0;  // 增长率
  double K = 10.0; // 承载能力

  // 初始化状态
  Tensor y0 = make_scalar_tensor(1.0f);

  // 创建求解器
  RK4Integrator solver(dt);

  // 定义 ODE
  auto ode_func = [r, K](double t, const Tensor &y) -> Tensor {
    float y_val = get_scalar_value(y);
    float dydt = r * y_val * (1.0f - y_val / K);
    return make_scalar_tensor(dydt);
  };

  // 积分
  Tensor y_final = solver.integrate(ode_func, t_start, t_end, y0);

  // 结果
  float numerical_result = get_scalar_value(y_final);

  std::cout << "Final Result (Numerical): y = " << numerical_result
            << std::endl;
  std::cout << "Expected Result: y ≈ 10.0" << std::endl;
  std::cout << "Difference: " << std::abs(numerical_result - 10.0) << std::endl;
}
// ==========================================
// 测试 4: 无人机质点物理模型
// ==========================================
void test_drone_physics() {
  std::cout << "\n=== Test 4: Drone Point Mass Physics ===" << std::endl;

  // 物理参数
  const double mass = 1.0; // kg
  const double g = 9.81;   // m/s^2 (NED坐标系向下为正)

  // 配置
  double dt = 0.001; // 1ms 步长
  double t_start = 0.0;
  double t_end = 1.0; // 仿真 1 秒

  // 创建求解器
  RK4Integrator solver(dt);

  // --- 场景 A: 自由落体 (Free Fall) ---
  std::cout << "\n--- Scenario A: Free Fall ---" << std::endl;
  {
    // 初始状态: [0, 0, 0, 0, 0, 0] (位置为0，速度为0)
    std::vector<float> init = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    Tensor y0 = make_vector_tensor(init);

    // 定义无人机 ODE (Lambda 捕获 mass, g 和 thrust)
    // 场景 A: 推力为 0
    std::vector<float> thrust_input = {0.0f, 0.0f, 0.0f};

    auto drone_ode = [=](double t, const Tensor &y) -> Tensor {
      // 提取状态
      const float *y_data = y.data<float>();
      // float pn = y_data[0];
      // float pe = y_data[1];
      // float pd = y_data[2];
      float vn = y_data[3];
      float ve = y_data[4];
      float vd = y_data[5];

      // 计算加速度 a = F/m
      // F = gravity + thrust
      // 注意：NED 坐标系下，重力是正的，推力如果向上是负的
      float an = (0 + thrust_input[0]) / mass;
      float ae = (0 + thrust_input[1]) / mass;
      float ad = (mass * g + thrust_input[2]) / mass; // F_gravity = m*g

      // 组装导数: [vn, ve, vd, an, ae, ad]
      std::vector<float> dydt = {vn, ve, vd, an, ae, ad};
      return make_vector_tensor(dydt);
    };

    // 积分
    Tensor y_final = solver.integrate(drone_ode, t_start, t_end, y0);
    std::vector<float> res = get_vector_value(y_final);

    // 验证: pd 应该等于 0.5 * g * t^2
    double analytical_pd = 0.5 * g * t_end * t_end;
    std::cout << "Simulated Height (Down): " << res[2] << " m" << std::endl;
    std::cout << "Analytical Height: " << analytical_pd << " m" << std::endl;
    std::cout << "Error: " << std::abs(res[2] - analytical_pd) << " m"
              << std::endl;
  }

  // --- 场景 B: 悬停 (Hover) ---
  std::cout << "\n--- Scenario B: Hover ---" << std::endl;
  {
    // 初始状态: 高度 0m，速度 0
    std::vector<float> init = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    Tensor y0 = make_vector_tensor(init);

    // 场景 B: 推力向上，大小等于 -m*g (抵消重力)
    // 注意：NED 向下为正，所以向上的推力是负的
    std::vector<double> thrust_input = {0.0f, 0.0f, -mass * g};

    auto drone_ode = [=](double t, const Tensor &y) -> Tensor {
      const float *y_data = y.data<float>();
      float vn = y_data[3];
      float ve = y_data[4];
      float vd = y_data[5];

      float an = (0 + thrust_input[0]) / mass;
      float ae = (0 + thrust_input[1]) / mass;
      float ad = (mass * g + thrust_input[2]) / mass; // 这一项应该接近 0

      std::vector<float> dydt = {vn, ve, vd, an, ae, ad};
      return make_vector_tensor(dydt);
    };

    Tensor y_final = solver.integrate(drone_ode, t_start, t_end, y0);
    std::vector<float> res = get_vector_value(y_final);

    std::cout << "Simulated Height (should stay ~0): " << res[2] << " m"
              << std::endl;
    std::cout << "Simulated Velocity (should stay ~0): " << res[5] << " m/s"
              << std::endl;
  }
}

// 测试函数已移至单独的测试文件
// ==========================================
// Main 函数已移至 main.cpp
// ==========================================
