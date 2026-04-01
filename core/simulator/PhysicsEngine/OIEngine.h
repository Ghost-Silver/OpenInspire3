/**
 * @file OIEngine.h
 * @brief OpenInspire3 物理仿真器
 * @author GhostFace
 * @date 2026/2/18
 */

#ifndef OIENGINE_H
#define OIENGINE_H

#include "RK4Solver.h"

struct Config {
    double dt = 0.001;     ///> 仿真器频率 1kHz
    double mass = 1.0;     ///> 质点质量 1kg
    double gravity = 9.81; ///> 重力加速度 9.81
};

// 创建张量的辅助函数
Tensor make_tensor(const std::vector<float> &vals);

// f(x, u)
Tensor drone_dynamics(const Tensor &state, const Tensor &thrust, Config config);

class Engine {
  private:
    Config _config;
    std::vector<Tensor> _state;
    RK4Integrator _rk4;
    int _numl;
    double _timems = 0.0;

  public:
    Engine(Config config, const std::vector<Tensor> &state, const int &numl);

    // 使用初始化列表的构造函数（推荐）
    Engine(Config config, const int &numl);

    // 获取当前状态
    Tensor getState(const int &rank) const;

    // 设置状态
    void setState(const Tensor &state, const int &rank);

    // 获取配置
    Config getConfig() const;

    // 更新配置
    void setConfig(Config config);

    // 执行单个仿真步（使用RK4方法）
    void step(const Tensor &thrust);

    double getTime();

    void print(const int &rank);
};

#endif // OIENGINE_H
