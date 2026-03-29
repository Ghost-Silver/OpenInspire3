#include "core/simulator/PhysicsEngine/OIEngine.h"
#include "core/simulator/PhysicsEngine/RK4Solver.h"
#include <iostream>

using namespace std;

int main() {

    Ctorch_Error::setPrintLevel(PrintLevel::MINIUM);
    cout << "OpenInspire3 Drone Free Fall Simulation" << endl;
    cout << "====================================" << endl;

    // 1. 创建配置
    Config config;
    config.dt = 0.001;     // 1ms 时间步长
    config.mass = 1.0;     // 1kg 质量
    config.gravity = 9.81; // 重力加速度

    // 2. 创建引擎
    Engine engine(config);

    // 3. 设置初始状态 [pn, pe, pd, vn, ve, vd]
    //    初始位置：原点 (0, 0, 0)
    //    初始速度：静止 (0, 0, 0)
    vector<float> initial_state = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    Tensor initial_state_tensor = make_vector_tensor(initial_state);
    engine.setState(initial_state_tensor);

    // 4. 定义推力输入（自由落体时推力为0）
    vector<float> thrust_input = {0.0f, 0.0f, 0.0f};
    Tensor thrust = make_vector_tensor(thrust_input);

    // 5. 仿真参数
    double simulation_time = 1.0; // 仿真 1 秒
    int steps = static_cast<int>(simulation_time / config.dt);

    // 6. 执行仿真
    cout << "\nStarting free fall simulation..." << endl;
    cout << "Initial state: [pn, pe, pd, vn, ve, vd] = [0, 0, 0, 0, 0, 0]"
         << endl;
    cout << "Thrust: [tn, te, td] = [0, 0, 0]" << endl;
    cout << "Gravity: " << config.gravity << " m/s^2" << endl;
    cout << "Time step: " << config.dt << " s" << endl;
    cout << "Simulation time: " << simulation_time << " s" << endl;
    cout << "====================================" << endl;

    for (int i = 0; i < steps; ++i) {
        // 执行单个仿真步
        engine.step(thrust);

        // 每100步打印一次状态（每0.1秒）
        if (i % 100 == 0) {
            Tensor current_state = engine.getState();
            vector<float> state_vals = get_vector_value(current_state);
            double time = i * config.dt;

            cout << "Time: " << time << " s, "
                 << "Position: [" << state_vals[0] << ", " << state_vals[1]
                 << ", " << state_vals[2] << "], "
                 << "Velocity: [" << state_vals[3] << ", " << state_vals[4]
                 << ", " << state_vals[5] << "]" << endl;
        }
    }

    // 7. 打印最终状态
    Tensor final_state = engine.getState();
    vector<float> final_state_vals = get_vector_value(final_state);
    cout << "====================================" << endl;
    cout << "Final state after " << simulation_time << " s:" << endl;
    cout << "Position: [pn, pe, pd] = [" << final_state_vals[0] << ", "
         << final_state_vals[1] << ", " << final_state_vals[2] << "]" << endl;
    cout << "Velocity: [vn, ve, vd] = [" << final_state_vals[3] << ", "
         << final_state_vals[4] << ", " << final_state_vals[5] << "]" << endl;

    // 8. 计算理论值并比较
    double theoretical_pd =
        0.5 * config.gravity * simulation_time * simulation_time;
    double theoretical_vd = config.gravity * simulation_time;
    cout << "====================================" << endl;
    cout << "Theoretical values (free fall):" << endl;
    cout << "Position (down): " << theoretical_pd << " m" << endl;
    cout << "Velocity (down): " << theoretical_vd << " m/s" << endl;
    cout << "====================================" << endl;
    cout << "Simulation completed!" << endl;

    return 0;
}
