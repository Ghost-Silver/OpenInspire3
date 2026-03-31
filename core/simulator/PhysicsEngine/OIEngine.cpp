/**
 * @file OIEngine.cpp
 * @brief OpenInspire3 物理仿真器
 * @author GhostFace
 * @date 2026/2/18
 */

#include "OIEngine.h"

// 创建张量的辅助函数
Tensor make_tensor(const std::vector<float> &vals) {
    Tensor t(ShapeTag{}, {static_cast<unsigned long>(vals.size())},
             DType::kFloat, DeviceType::kCPU);
    float *data = const_cast<float *>(t.data<float>());
    for (size_t i = 0; i < vals.size(); ++i) {
        data[i] = vals[i];
    }
    return t;
}

// f(x, u)
Tensor drone_dynamics(const Tensor &state, const Tensor &thrust,
                      Config config) {
    // state shape: {6}
    // [0]: pn, [1]: pe, [2]: pd
    // [3]: vn, [4]: ve, [5]: vd

    // 1. 提取当前速度
    Tensor vel = state.slice(0, 3, 6); // 取出后3个元素

    // 2. 计算加速度 (Dynamics)
    const double g = config.gravity;
    const double mass = config.mass;

    // 2.1 重力
    Tensor F_grav = make_tensor({0.0f, 0.0f, static_cast<float>(mass * g)});

    // 2.2 合外力
    Tensor F_total = F_grav + thrust; // 这里可以再加 wind

    // 2.3 F = ma => a = F/m
    Tensor acc = F_total / mass;

    // 3. 组装导数 dx/dt
    // dx/dt = [vel, acc]
    Tensor derivative =
        Tensor(ShapeTag{}, {6}, DType::kFloat, DeviceType::kCPU);

    // 直接访问数据指针来修改derivative张量
    float *derivative_data = const_cast<float *>(derivative.data<float>());
    const float *vel_data = vel.data<float>();
    const float *acc_data = acc.data<float>();

    // 前3个导数是速度
    for (int i = 0; i < 3; ++i) {
        derivative_data[i] = vel_data[i];
    }

    // 后3个导数是加速度
    for (int i = 0; i < 3; ++i) {
        derivative_data[i + 3] = acc_data[i];
    }

    return derivative;
}

// Engine类实现

Engine::Engine(Config config, const std::vector<Tensor> &state, const int &numl)
    : _config(config), _rk4(config.dt), _numl(numl) {
    for (int i = 0; i < _numl; i++) {
        _state[i] = state[i];
    }
}

// 使用初始化列表的构造函数
Engine::Engine(Config config, const int &numl)
    : _config(config), _rk4(config.dt) {
    // 初始化默认状态：位置和速度都为0

    for (int i = 0; i < _numl; i++) {
        _state[i] = Tensor(ShapeTag{}, {6}, DType::kFloat, DeviceType::kCPU);
    }
}

// 获取当前状态
Tensor Engine::getState(const int &rank) const { return _state[rank]; }

// 设置状态
void Engine::setState(const Tensor &state, const int &rank) {
    _state[rank] = state;
}

// 获取配置
Config Engine::getConfig() const { return _config; }

// 更新配置
void Engine::setConfig(Config config) {
    _config = config;
    _rk4 = RK4Integrator(config.dt); // 更新RK4求解器的时间步长
}

// 执行单个仿真步（使用RK4方法）
void Engine::step(const Tensor &thrust) {
    // 定义时间依赖的ODE函数
    auto ode_func = [this, &thrust](double t, const Tensor &state) -> Tensor {
        return drone_dynamics(state, thrust, _config);
    };

    // 使用RK4方法更新状态
    for (int i = 0; i < _numl; i++) {
        _state[i] = _rk4.step(ode_func, 0.0, _state[i]);
    }
}
