/**
 * @file SixDofGradTest.cpp
 * @brief 六自由度可微动力学测试
 * @author GhostFace
 * @date 2026/9/17
 *
 * 分三部分：
 *   Part A 数值等价 —— 可微张量实现必须与已通过闭环验证的标量实现给出同一结果。
 *          若两者不等价，梯度再漂亮也没有意义：被求导的对象已经不是那个被
 *          验证过的物理模型了。
 *   Part B 梯度正确性 —— 解析梯度（autograd）与中心差分对照，覆盖初始状态与推力。
 *   Part C 端到端可用性 —— 用梯度下降直接优化一段推力序列（打靶法），
 *          验证梯度确实能驱动物理仿真朝目标收敛。
 *
 * 退出码反映全部断言是否通过。
 *
 * @note 两条测试自身的纪律（都是踩过坑之后补的）：
 *  1. **NaN 必须当作「最大差异」**。`std::max(a, fabs(NaN))` 会返回 a，于是
 *     一个全 NaN 的结果会被报告成「偏差 0」。所有比较都要显式判非有限值。
 *  2. **梯度要查叶子张量本身**。`Tensor` 的拷贝赋值会重置 autograd 节点
 *     （`operator=(const Tensor&)` 内 `_node.reset()`），把叶子拷进容器再回读，
 *     拿到的是与计算图无关的新对象，梯度恒为 0。必须用移动语义把叶子传出。
 */

#include "SixDofDynamics.h"
#include "SixDofDynamicsDiff.h"
#include "TensorUtils.h"

#include "AutoGrad.h"

#include <chrono>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

using namespace oi3;

namespace {

int g_checks = 0;
int g_failed = 0;

/// 分阶段计时：可微仿真的代价主要在图规模，超时问题必须能定位到具体段落
using Clock = std::chrono::steady_clock;
Clock::time_point g_t0 = Clock::now();

double elapsedMs() {
    return std::chrono::duration<double, std::milli>(Clock::now() - g_t0).count();
}

void checkTrue(const char *name, bool cond) {
    ++g_checks;
    if (!cond) {
        ++g_failed;
    }
    std::cout << "  [" << (cond ? " ok " : "FAIL") << "] " << name << "\n";
}

void checkNear(const char *name, double got, double want, double tol) {
    ++g_checks;
    const double err = std::fabs(got - want);
    const bool ok = (err <= tol) && std::isfinite(got);
    if (!ok) {
        ++g_failed;
    }
    std::cout << "  [" << (ok ? " ok " : "FAIL") << "] " << name << "  (got " << got
              << ", want " << want << ", |err| = " << err << ")\n";
}

SixDofConfig makeConfig(double drag = 0.0) {
    SixDofConfig cfg;
    cfg.base.dt = 0.001;
    cfg.base.mass = 1.0;
    cfg.base.gravity = 9.81;
    cfg.base.drag_coeff = drag;
    cfg.base.max_thrust = 0.0;
    cfg.inertia[0] = 0.01;
    cfg.inertia[1] = 0.01;
    cfg.inertia[2] = 0.02;
    cfg.arm_length = 0.25;
    cfg.torque_limit = 1.0;
    cfg.max_body_thrust = 20.0;
    return cfg;
}

/// 构造一维叶子张量并写入初值
Tensor leaf1d(std::initializer_list<float> values, bool requires_grad = false) {
    Tensor t(ShapeTag{}, {values.size()});
    float *p = t.data_write<float>();
    size_t i = 0;
    for (float v : values) {
        p[i++] = v;
    }
    if (requires_grad) {
        t.requires_grad(true);
    }
    return t;
}

float at(const Tensor &t, size_t i) { return t.data<float>()[i]; }

/// 张量是否含非有限值
bool hasNonFinite(const Tensor &t) {
    const size_t n = t.numel();
    const float *p = t.data<float>();
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(p[i])) {
            return true;
        }
    }
    return false;
}

void dumpTensor(const Tensor &t, const char *name) {
    std::cout << "        " << name << " = [";
    const size_t n = t.numel();
    for (size_t i = 0; i < n; ++i) {
        std::cout << at(t, i) << (i + 1 < n ? ", " : "");
    }
    std::cout << "]\n";
}

void dumpState(const SixDofState &s, const char *tag) {
    std::cout << "      " << tag << "\n";
    dumpTensor(s.pos, "pos");
    dumpTensor(s.vel, "vel");
    dumpTensor(s.quat, "quat");
    dumpTensor(s.omega, "omega");
}

/**
 * 逐分量比较四个状态张量。
 *
 * @note 任一元素非有限值时直接返回无穷大：`std::max` 遇到 NaN 会保留旧值，
 *       若不显式处理，「两侧都是 NaN」会被误报成「偏差为 0」。
 */
double maxStateDiff(const SixDofState &a, const SixDofState &b) {
    double m = 0.0;
    const Tensor *pa[4] = {&a.pos, &a.vel, &a.quat, &a.omega};
    const Tensor *pb[4] = {&b.pos, &b.vel, &b.quat, &b.omega};
    for (int k = 0; k < 4; ++k) {
        const size_t n = pa[k]->numel();
        for (size_t i = 0; i < n; ++i) {
            const double x = at(*pa[k], i);
            const double y = at(*pb[k], i);
            if (!std::isfinite(x) || !std::isfinite(y)) {
                return std::numeric_limits<double>::infinity();
            }
            m = std::max(m, std::fabs(x - y));
        }
    }
    return m;
}

bool stateFinite(const SixDofState &s) {
    return !hasNonFinite(s.pos) && !hasNonFinite(s.vel) && !hasNonFinite(s.quat) &&
           !hasNonFinite(s.omega);
}

/// 读取梯度（可能为 nullptr，此时按 0 处理）。grad_ptr() 非 const，故收非常量引用。
float gradAt(Tensor &t, size_t i) {
    const float *g = t.grad_ptr();
    return (g == nullptr) ? 0.0f : g[i];
}

} // namespace

int main() {
    std::cout << "========================================\n";
    std::cout << "六自由度可微动力学测试\n";
    std::cout << "========================================\n";

    const SixDofConfig cfg = makeConfig();
    const float dt = static_cast<float>(cfg.base.dt);

    // ---- Part A: 与标量实现的数值等价 ----
    std::cout << "\n[A1] 算子形状探针\n";
    {
        Tensor q = leaf1d({1.0f, 0.0f, 0.0f, 0.0f});
        Tensor norm_sq = (q * q).sum();
        std::cout << "    (q*q).sum() 维度数 = " << norm_sq.sizes().size()
                  << "，numel = " << norm_sq.numel() << "\n";
        Tensor n = (norm_sq + leaf1d({1e-12f})).sqrt();
        std::cout << "    sqrt 后 维度数 = " << n.sizes().size() << "，numel = " << n.numel()
                  << "\n";

        Tensor normalized = q / n;
        checkTrue("归一化结果形状为 {4}",
                  normalized.sizes().size() == 1 && normalized.sizes()[0] == 4);
        checkNear("归一化后第 0 个分量为 1", at(normalized, 0), 1.0, 1e-6);

        Tensor q2 = leaf1d({0.9f, 0.1f, 0.3f, 0.2f});
        Tensor n2 = ((q2 * q2).sum() + leaf1d({1e-12f})).sqrt();
        Tensor q2n = q2 / n2;
        const double norm2 = std::sqrt(static_cast<double>(at(q2n, 0)) * at(q2n, 0) +
                                       static_cast<double>(at(q2n, 1)) * at(q2n, 1) +
                                       static_cast<double>(at(q2n, 2)) * at(q2n, 2) +
                                       static_cast<double>(at(q2n, 3)) * at(q2n, 3));
        checkNear("非平凡四元数归一化后模长为 1", norm2, 1.0, 1e-6);
    }

    std::cout << "\n[A2] 单步数值等价（标量版 vs 可微版）\n";
    {
        SixDofState y0{leaf1d({0.0f, 0.0f, -10.0f}), leaf1d({1.0f, -0.5f, 0.25f}),
                       leaf1d({0.9239f, 0.0f, 0.3827f, 0.0f}), // 俯仰 45°
                       leaf1d({0.05f, -0.03f, 0.02f})};
        const float thrust = 9.81f * 1.2f;
        Tensor torque = leaf1d({0.01f, -0.02f, 0.005f});

        const SixDofState s1 = rk4StepSixDof(y0, static_cast<double>(thrust), torque, cfg, dt);
        std::cout << "    [探针] 标量版单步 OK\n" << std::flush;

        // 逐个算子隔离：崩溃定位时不必反复二分测试文件
        {
            Tensor a = sixDofAccelerationDiff(y0.vel, y0.quat, leaf1d({thrust}), cfg.base);
            std::cout << "    [探针] 平动加速度 OK  = [" << at(a, 0) << ", " << at(a, 1)
                      << ", " << at(a, 2) << "]\n" << std::flush;
        }
        {
            Tensor qd = quatDerivativeDiff(y0.quat, y0.omega);
            std::cout << "    [探针] 四元数导数 OK  = [" << at(qd, 0) << ", " << at(qd, 1)
                      << ", " << at(qd, 2) << ", " << at(qd, 3) << "]\n" << std::flush;
        }
        {
            Tensor al = angularAccelerationDiff(y0.omega, torque, cfg);
            std::cout << "    [探针] 角加速度 OK    = [" << at(al, 0) << ", " << at(al, 1)
                      << ", " << at(al, 2) << "]\n" << std::flush;
        }
        {
            Tensor qn = normalizeQuatDiff(y0.quat);
            std::cout << "    [探针] 四元数归一化 OK= [" << at(qn, 0) << ", " << at(qn, 1)
                      << ", " << at(qn, 2) << ", " << at(qn, 3) << "]\n" << std::flush;
        }
        {
            Tensor r = rotationMatrixFromQuat(y0.quat);
            std::cout << "    [探针] 旋转矩阵 OK    形状 " << r.sizes().size() << "，元素 ["
                      << at(r, 0) << ", " << at(r, 1) << ", " << at(r, 2) << ", " << at(r, 3)
                      << "...]\n" << std::flush;
        }
        {
            Tensor st = stackVec3(leaf1d({1.0f}), leaf1d({2.0f}), leaf1d({3.0f}));
            std::cout << "    [探针] stackVec3 OK   = [" << at(st, 0) << ", " << at(st, 1)
                      << ", " << at(st, 2) << "]\n" << std::flush;
        }

        std::cout << "    [探针] 开始完整单步...\n" << std::flush;
        const SixDofState s2 = rk4StepSixDofDiff(y0, leaf1d({thrust}), torque, cfg, dt);
        std::cout << "    [探针] 完整单步 OK\n" << std::flush;

        const double d = maxStateDiff(s1, s2);
        std::cout << "    最大偏差 = " << d << "\n";
        checkTrue("单步 RK4 与标量版一致（< 1e-5）", d < 1e-5);
        checkTrue("可微版单步结果有限", stateFinite(s2));
    }

    std::cout << "\n[A3] 多步数值等价（含阻力）\n";
    {
        int STEPS = 200;
        if (const char *e = std::getenv("OI3_A3_STEPS")) {
            STEPS = std::atoi(e);
        }
        SixDofState ys{leaf1d({0.0f, 0.0f, -5.0f}), leaf1d({0.3f, -0.2f, 0.1f}),
                       leaf1d({0.9f, 0.1f, 0.3f, 0.2f}), leaf1d({0.1f, 0.05f, -0.02f})};
        ys.quat = normalizeQuat(ys.quat);

        SixDofState yd = ys;
        yd.quat = normalizeQuatDiff(yd.quat);

        const SixDofConfig cfg_drag = makeConfig(0.05);
        const float thrust = 10.5f;
        Tensor torque = leaf1d({0.005f, 0.01f, -0.004f});

        int first_bad = -1;
        for (int step = 0; step < STEPS; ++step) {
            ys = rk4StepSixDof(ys, static_cast<double>(thrust), torque, cfg_drag, dt);
            yd = rk4StepSixDofDiff(yd, leaf1d({thrust}), torque, cfg_drag, dt);

            if (step % 20 == 0) {
                std::cout << "    [进度] 第 " << step << " 步\n" << std::flush;
            }
            if (first_bad < 0 && (!stateFinite(yd) || !stateFinite(ys))) {
                first_bad = step;
                std::cout << "    [诊断] 第 " << step << " 步出现非有限值\n";
                dumpState(yd, "可微版：");
                dumpState(ys, "标量版：");
                break;
            }
        }

        checkTrue("整个积分过程无非有限值", first_bad < 0);

        // 诊断：积分结束后逐个张量报告元数据（越界读表现为 SIGBUS/Crash，
        // 元数据先于比较打印出来才能定位到具体张量）
        {
            auto dumpInfo = [](const Tensor &t, const char *n) {
                std::cout << "      " << n << " 维度数=" << t.sizes().size()
                          << " numel=" << t.numel()
                          << " data=" << static_cast<const void *>(t.data<float>());
                if (!t.sizes().empty()) {
                    std::cout << " shape=[";
                    for (size_t sv : t.sizes()) {
                        std::cout << sv << ",";
                    }
                    std::cout << "]";
                }
                std::cout << "\n" << std::flush;
            };
            std::cout << "    [诊断] 标量版状态：" << std::endl;
            dumpInfo(ys.pos, "pos");
            dumpInfo(ys.vel, "vel");
            dumpInfo(ys.quat, "quat");
            dumpInfo(ys.omega, "omega");
            std::cout << "    [诊断] 可微版状态：" << std::endl;
            dumpInfo(yd.pos, "pos");
            dumpInfo(yd.vel, "vel");
            dumpInfo(yd.quat, "quat");
            dumpInfo(yd.omega, "omega");
        }

        if (first_bad < 0) {
            const double d = maxStateDiff(ys, yd);
            std::cout << "    " << STEPS << " 步后最大偏差 = " << d << "\n";
            checkTrue("多步后仍与标量版一致（< 1e-3）", d < 1e-3);
            checkNear("可微版四元数仍为单位模长",
                      std::sqrt(static_cast<double>(at(yd.quat, 0)) * at(yd.quat, 0) +
                                static_cast<double>(at(yd.quat, 1)) * at(yd.quat, 1) +
                                static_cast<double>(at(yd.quat, 2)) * at(yd.quat, 2) +
                                static_cast<double>(at(yd.quat, 3)) * at(yd.quat, 3)),
                      1.0, 1e-4);
        }
    }

    // ---- Part B: 梯度正确性 ----
    std::cout << "\n[B1] 对初始状态的梯度 vs 中心差分\n";
    {
        const int STEPS = 5;
        const float thrust = 10.0f;
        const float target[3] = {1.0f, -1.0f, -9.0f};

        // 返回值用移动语义传出叶子张量：拷贝赋值会重置 autograd 节点
        struct Rollout {
            Tensor pos_leaf;
            Tensor quat_leaf;
            Tensor loss;
        };

        auto rollout = [&](const std::vector<float> &pos_v, const std::vector<float> &quat_v,
                           bool want_grad) -> Rollout {
            Tensor pos_leaf = leaf1d({pos_v[0], pos_v[1], pos_v[2]}, want_grad);
            Tensor vel_leaf = leaf1d({0.4f, -0.2f, 0.1f}, want_grad);
            Tensor quat_leaf =
                leaf1d({quat_v[0], quat_v[1], quat_v[2], quat_v[3]}, want_grad);
            Tensor omega_leaf = leaf1d({0.02f, -0.01f, 0.005f}, want_grad);
            Tensor torque = leaf1d({0.004f, -0.006f, 0.002f});

            // 状态经一次逐元素乘法从参数派生：RK4 循环里 `s` 的成员会被移动赋值
            // 覆盖，若状态成员直接是叶子对象，覆盖时叶子析构、其 GradAccumulator
            // 的 weak_ptr 失效，梯度静默丢失（grad_ptr() 返回 nullptr 且不报错）。
            // 派生一层之后，图的锚点在参数对象上，而参数由本函数持有并传出。
            const Tensor one = leaf1d({1.0f});
            SixDofState s{pos_leaf * one, vel_leaf * one, quat_leaf * one,
                          omega_leaf * one};
            for (int i = 0; i < STEPS; ++i) {
                s = rk4StepSixDofDiff(s, leaf1d({thrust}), torque, cfg, dt);
            }
            Tensor d = s.pos - makeVec3(target[0], target[1], target[2]);
            Tensor loss = (d * d).sum();
            return Rollout{std::move(pos_leaf), std::move(quat_leaf), std::move(loss)};
        };

        const std::vector<float> pos0 = {0.0f, 0.0f, -10.0f};
        const std::vector<float> quat0 = {0.9239f, 0.0f, 0.3827f, 0.0f};

        Rollout r = rollout(pos0, quat0, true);
        std::cout << "    [t=" << elapsedMs() << "ms] 前向完成，loss = "
                  << r.loss.data<float>()[0] << "\n";
        checkTrue("前向损失有限", std::isfinite(r.loss.data<float>()[0]));

        AutoGrad::backward(r.loss.getRelatedNode(), false);

        const double h = 1e-3;
        double max_err = 0.0;
        bool any_nan = false;
        for (int i = 0; i < 3; ++i) {
            std::vector<float> p = pos0, m = pos0;
            p[i] += static_cast<float>(h);
            m[i] -= static_cast<float>(h);
            const double lp = rollout(p, quat0, false).loss.data<float>()[0];
            const double lm = rollout(m, quat0, false).loss.data<float>()[0];
            const double num = (lp - lm) / (2.0 * h);
            const double ana = static_cast<double>(gradAt(r.pos_leaf, i));
            if (!std::isfinite(num) || !std::isfinite(ana)) {
                any_nan = true;
            }
            max_err = std::max(max_err, std::fabs(num - ana));
            std::cout << "      pos[" << i << "]: 解析 = " << ana << "，差分 = " << num
                      << "\n";
        }
        for (int i = 0; i < 4; ++i) {
            std::vector<float> p = quat0, m = quat0;
            p[i] += static_cast<float>(h);
            m[i] -= static_cast<float>(h);
            const double lp = rollout(pos0, p, false).loss.data<float>()[0];
            const double lm = rollout(pos0, m, false).loss.data<float>()[0];
            const double num = (lp - lm) / (2.0 * h);
            const double ana = static_cast<double>(gradAt(r.quat_leaf, i));
            if (!std::isfinite(num) || !std::isfinite(ana)) {
                any_nan = true;
            }
            max_err = std::max(max_err, std::fabs(num - ana));
            std::cout << "      quat[" << i << "]: 解析 = " << ana << "，差分 = " << num
                      << "\n";
        }
        std::cout << "    最大偏差 = " << max_err << "\n";
        checkTrue("梯度与差分均为有限值", !any_nan);
        checkTrue("∂loss/∂(初始 pos, quat) 与中心差分一致（< 1e-2）", max_err < 1e-2);
    }

    std::cout << "\n[B2] 对推力的梯度 vs 中心差分\n";
    {
        // 步数与推力刻意选得使损失对推力敏感：步长太短时末端速度变化落在 float
        // 分辨率以下，中心差分退化为 0，测试会「通过」但什么都没验证。
        const int STEPS = 60;
        const float thrust_nominal = 12.0f;

        struct ThrustRollout {
            Tensor thrust_leaf;
            Tensor loss;
        };

        auto rollout = [&](float thrust, bool want_grad) -> ThrustRollout {
            Tensor th = leaf1d({thrust}, want_grad);
            SixDofState s{leaf1d({0.0f, 0.0f, -10.0f}), leaf1d({0.0f, 0.0f, 0.0f}),
                          leaf1d({0.98f, 0.0f, 0.199f, 0.0f}),
                          leaf1d({0.0f, 0.0f, 0.0f})};
            Tensor torque = leaf1d({0.0f, 0.0f, 0.0f});
            for (int i = 0; i < STEPS; ++i) {
                s = rk4StepSixDofDiff(s, th, torque, cfg, dt);
            }
            // 用末端速度作损失：推力直接决定加速度，灵敏度高且无需求解位置
            Tensor dv = s.vel - makeVec3(0.0f, 0.0f, 0.0f);
            Tensor loss = (dv * dv).sum();
            return ThrustRollout{std::move(th), std::move(loss)};
        };

        ThrustRollout r = rollout(thrust_nominal, true);
        AutoGrad::backward(r.loss.getRelatedNode(), false);

        const double h = 1e-3;
        const double lp =
            rollout(thrust_nominal + static_cast<float>(h), false).loss.data<float>()[0];
        const double lm =
            rollout(thrust_nominal - static_cast<float>(h), false).loss.data<float>()[0];
        const double num = (lp - lm) / (2.0 * h);
        const double ana = static_cast<double>(gradAt(r.thrust_leaf, 0));
        std::cout << "    解析 = " << ana << "，差分 = " << num << "\n";
        checkTrue("两者均为有限值", std::isfinite(num) && std::isfinite(ana));
        checkNear("∂loss/∂推力 与中心差分一致", ana, num,
                  std::max(1e-3, std::fabs(num) * 1e-2));
    }

    // ---- Part C: 端到端可用性（打靶法）----
    std::cout << "\n[C1] 用梯度下降优化推力序列（打靶法）\n";
    {
        // 规模刻意压小：可微仿真的成本随轨迹长度线性增长（图节点数 = 步数 x
        // 每步算子数），本项只需证明「梯度能驱动轨迹朝目标收敛」，不需要长时域。
        //
        // 另外，单张图的规模还受一个**未解决的框架问题**限制：当图大到一定程度
        // 且反复构建-反传时，C3 反向融合路径会出现越界访问（EXC_BAD_ACCESS /
        // KERN_PROTECTION_FAILURE，故障地址为页对齐边界）。同样规模的图在 eager
        // 路径（B1/B2 的小图、CTorch 侧的 test_deep_graph_backward）下完全正常，
        // 因此这不是可微动力学本身的问题。此处把规模控制在稳定区间内，
        // 并把「大图 + 重复反传」作为单独议题跟踪。
        int SEGMENTS = 8;
        int STEPS_PER_SEG = 5; // 每段 5 ms，总 40 ms
        if (const char *e = std::getenv("OI3_C1_SEGS")) {
            SEGMENTS = std::atoi(e);
        }
        if (const char *e = std::getenv("OI3_C1_STEPS")) {
            STEPS_PER_SEG = std::atoi(e);
        }
        const float target[3] = {0.0f, 0.0f, -10.0f};

        auto rollout = [&](const std::vector<Tensor> &thrusts) {
            SixDofState s{leaf1d({0.0f, 0.0f, -9.5f}), leaf1d({0.0f, 0.0f, 0.0f}),
                          leaf1d({0.98f, 0.0f, 0.199f, 0.0f}), leaf1d({0.0f, 0.0f, 0.0f})};
            Tensor torque = leaf1d({0.0f, 0.0f, 0.0f});
            for (int seg = 0; seg < SEGMENTS; ++seg) {
                for (int k = 0; k < STEPS_PER_SEG; ++k) {
                    s = rk4StepSixDofDiff(s, thrusts[seg], torque, cfg, dt);
                }
            }
            Tensor dp = s.pos - makeVec3(target[0], target[1], target[2]);
            Tensor dv = s.vel - makeVec3(0.0f, 0.0f, 0.0f);
            return (dp * dp).sum() + (dv * dv).sum() * 0.5f;
        };

        // 初始推力刻意远低于悬停所需（悬停约 9.81 N）：这样梯度有明确的方向
        // 与足够的优化空间 —— 若初值已接近最优，符号梯度只会在最优点附近交替
        // 振荡，末态代价与初值几乎相同，判据失去区分度（实测如此）。
        std::vector<Tensor> thrusts;
        thrusts.reserve(SEGMENTS);
        for (int i = 0; i < SEGMENTS; ++i) {
            thrusts.push_back(leaf1d({5.0f}, true));
        }

        std::cout << "    [t=" << elapsedMs() << "ms] 开始 rollout\n";
        Tensor loss0 = rollout(thrusts);
        std::cout << "    [t=" << elapsedMs() << "ms] 首次前向完成\n";
        const double loss_before = static_cast<double>(loss0.data<float>()[0]);
        AutoGrad::backward(loss0.getRelatedNode(), false);

        const float *g0 = thrusts[0].grad_ptr();
        checkTrue("推力参数拿到了梯度", g0 != nullptr);
        checkTrue("前向损失有限", std::isfinite(loss_before));
        if (g0 != nullptr) {
            checkTrue("梯度非零（说明梯度穿过了整条轨迹）",
                      std::isfinite(g0[0]) && std::fabs(g0[0]) > 1e-6f);
        }

        // 步长用符号梯度（signSGD）：本任务的 loss 对推力不敏感（40 ms 轨迹的位移
        // 量级远小于目标距离），普通梯度步长会让一轮之内的变化落在 float 精度之下，
        // 判据无法体现「梯度方向正确」。符号梯度每步固定移动 step_newton，
        // 与梯度尺度无关，一轮即可观察到明确下降。
        const float step_newton = 0.2f;

        // 优化轮数。此前默认只能取 1：CTorch 在「同一进程内反复构建-反传同一形态
        // 的图」时会越界崩溃（根因是 Arena 的对象池与跨轮存活的 shared_ptr 冲突，
        // 已修 —— 见 CTorch 提交 fc13686）。修复后多轮优化正常运行：
        // 30 轮足以让代价下降约 5.7%（推力由 5.0 被推至 9.0，趋近悬停所需的 9.81），
        // 环境变量保留以便调参。
        int EPOCHS = 30;
        if (const char *e = std::getenv("OI3_GRAD_EPOCHS")) {
            EPOCHS = std::atoi(e);
        }

        // 更新参数并重新评估：损失在「参数更新之后」重新计算，单轮也能体现下降
        auto stepOnce = [&]() {
            for (int i = 0; i < SEGMENTS; ++i) {
                thrusts[i].zero_grad();
            }
            Tensor loss = rollout(thrusts);
            AutoGrad::backward(loss.getRelatedNode(), false);
            for (int i = 0; i < SEGMENTS; ++i) {
                const float *gi = thrusts[i].grad_ptr();
                float *ti = thrusts[i].data_write<float>();
                if (gi != nullptr && std::isfinite(gi[0])) {
                    const float dir = (gi[0] > 0.0f) ? 1.0f : ((gi[0] < 0.0f) ? -1.0f : 0.0f);
                    ti[0] -= step_newton * dir;
                    // 螺旋桨只能推不能拉：推力非负
                    if (ti[0] < 0.0f) {
                        ti[0] = 0.0f;
                    }
                }
            }
        };

        for (int epoch = 0; epoch < EPOCHS; ++epoch) {
            stepOnce();
            // 每若干轮打印一次代价轨迹：仅看首末两点无法区分「收敛」与「在最优附近
            // 振荡」——两者都可能表现为终点回到起点，需要的判据不同。
            if ((epoch + 1) % 10 == 0) {
                const double cur = static_cast<double>(rollout(thrusts).data<float>()[0]);
                const float *g0 = thrusts[0].grad_ptr();
                std::cout << "      epoch " << (epoch + 1) << "  loss = " << cur
                          << "  T[0] = " << thrusts[0].data<float>()[0]
                          << "  g[0] = " << (g0 ? g0[0] : 0.0f)
                          << "  T[4] = " << thrusts[4].data<float>()[0]
                          << "  g[4] = "
                          << (thrusts[4].grad_ptr() ? thrusts[4].grad_ptr()[0] : 0.0f)
                          << "\n" << std::flush;
            }
        }

        const double loss_after = static_cast<double>(rollout(thrusts).data<float>()[0]);
        std::cout << "    [t=" << elapsedMs() << "ms] " << EPOCHS << " 轮梯度下降完成\n";
        std::cout << "    loss（更新前->更新后）: " << loss_before << " -> " << loss_after;
        if (loss_before != 0.0) {
            std::cout << "（相对下降 " << (100.0 * (loss_before - loss_after) / loss_before)
                      << "%）";
        }
        std::cout << "\n";
        // 判据是「梯度通路可用且方向正确」：40 ms 的窗口里推力对末端状态的影响本就
        // 有限（外加 8 段之间的更新会部分抵消），因此不设大幅下降的门槛，
        // 只要求代价确实下降。
        checkTrue("梯度下降使末端代价下降",
                  std::isfinite(loss_after) && loss_after < loss_before);
    }

    std::cout << "\n========================================\n";
    std::cout << g_checks - g_failed << " / " << g_checks << " checks passed\n";
    if (g_failed > 0) {
        std::cout << g_failed << " FAILED\n";
    }
    std::cout << "========================================\n";
    return g_failed == 0 ? 0 : 1;
}
