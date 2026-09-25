/**
 * @file SixDofPidController.cpp
 * @brief 六自由度级联 PID 控制器实现
 * @author GhostFace
 * @date 2026/9/17
 */

#include "SixDofPidController.h"
#include "SixDofDynamics.h"

#include <algorithm>
#include <cmath>

namespace oi3 {

namespace {

constexpr double kPi = 3.14159265358979323846;

double clampd(double v, double lo, double hi) {
    return std::max(lo, std::min(hi, v));
}

struct V3d {
    double x, y, z;
};

V3d readV3(const Tensor &t) {
    const float *p = t.data<float>();
    return {p[0], p[1], p[2]};
}

V3d cross(const V3d &a, const V3d &b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

double dot(const V3d &a, const V3d &b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

double norm(const V3d &a) { return std::sqrt(dot(a, a)); }

/// 四元数（w, x, y, z），Hamilton 约定，与 SixDofDynamics 一致
struct Quatd {
    double w, x, y, z;
};

Quatd readQuatD(const Tensor &t) {
    const float *p = t.data<float>();
    return {p[0], p[1], p[2], p[3]};
}

/// 旋转矩阵（body → NED），与 SixDofDynamics::rotationMatrix 同一约定
void rotationMatrixOf(const Quatd &q, double r[3][3]) {
    const double w = q.w, x = q.x, y = q.y, z = q.z;
    r[0][0] = 1.0 - 2.0 * (y * y + z * z);
    r[0][1] = 2.0 * (x * y - w * z);
    r[0][2] = 2.0 * (x * z + w * y);
    r[1][0] = 2.0 * (x * y + w * z);
    r[1][1] = 1.0 - 2.0 * (x * x + z * z);
    r[1][2] = 2.0 * (y * z - w * x);
    r[2][0] = 2.0 * (x * z - w * y);
    r[2][1] = 2.0 * (y * z + w * x);
    r[2][2] = 1.0 - 2.0 * (x * x + y * y);
}

/**
 * @brief 由「期望推力方向 + 期望偏航角」构造完整期望姿态
 *
 * @par 为什么必须补上偏航
 *
 * 原先的姿态误差只用 `cross(z_cur, z_des)` —— 即**只对齐机体 z 轴（推力方向）**。
 * 这在数学上留下一个自由度：绕 z_des 的转动不受任何约束，偏航角完全自由。
 *
 * 对纯定点悬停这没影响（偏航随便转，位置照样准），但一旦挂载相机、云台，
 * 或做任何需要指向的任务，**偏航就是任务要求，不是自由变量**。真机上偏航
 * 还影响气动与能耗。所以这是结构性缺失，不是调参能补的。
 *
 * @par 构造方式
 *
 * 已知期望机体 z 轴 `z_des`（由期望加速度定，保证推力方向正确）与期望偏航
 * `psi_des`（任务给定），按以下方式正交化：
 *
 * @verbatim
 *   x_c = [cos ψ, sin ψ, 0]        偏航方向决定的中间轴
 *   y_b = z_des × x_c / ‖·‖
 *   x_b = y_b × z_des
 *   R_des = [x_b  y_b  z_des]      列为机体系三轴在 NED 中的表示
 * @endverbatim
 *
 * 这与 DifferentialFlatness.h 的构造一致 —— 平坦映射同样是由加速度与偏航
 * 定出完整姿态，两处必须用同一套约定，否则前馈与反馈会互相打架。
 *
 * @note 当 `z_des` 与偏航轴共线（机身竖直指向偏航方向）时退化，此时返回 false，
 *       调用方应回退到只对齐推力方向的处理。
 */
bool buildDesiredAttitude(const V3d &z_des, double yaw_des, double r_des[3][3]) {
    const V3d xc{std::cos(yaw_des), std::sin(yaw_des), 0.0};
    V3d yb = cross(z_des, xc);
    const double ybn = norm(yb);
    if (ybn < 1e-6) {
        return false; // 退化：推力方向与偏航轴共线
    }
    yb = {yb.x / ybn, yb.y / ybn, yb.z / ybn};
    const V3d xb = cross(yb, z_des);

    // 列为机体系三轴（R 的列 = 机体轴在 NED 中的表示）
    r_des[0][0] = xb.x; r_des[0][1] = yb.x; r_des[0][2] = z_des.x;
    r_des[1][0] = xb.y; r_des[1][1] = yb.y; r_des[1][2] = z_des.y;
    r_des[2][0] = xb.z; r_des[2][1] = yb.z; r_des[2][2] = z_des.z;
    return true;
}

/**
 * @brief 由两个旋转矩阵求误差旋转向量（NED 系）
 *
 * 误差定义为 `R_e = R_des · R_curᵀ`：把当前姿态**转到**期望姿态所需的旋转。
 * 取其轴角即得误差向量。
 *
 * @par 为什么用矩阵而不是四元数
 *
 * 四元数误差需要处理符号歧义（q 与 −q 表示同一姿态），在热路径上多一次判断；
 * 而这里只需要旋转向量、不需要插值，矩阵形式更直接。3×3 矩阵乘法在栈上完成，
 * 无堆分配。
 */
V3d rotationErrorVector(const double r_des[3][3], const double r_cur[3][3]) {
    // R_e = R_des · R_curᵀ（R_cur 正交，转置即逆）
    double re[3][3];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            re[i][j] = r_des[i][0] * r_cur[j][0] + r_des[i][1] * r_cur[j][1] +
                       r_des[i][2] * r_cur[j][2];
        }
    }

    // 轴角：sinθ·axis = ½·(R_e − R_eᵀ) 的轴向量
    const double s = 0.5 * std::sqrt(std::max(0.0, (re[2][1] - re[1][2]) * (re[2][1] - re[1][2]) +
                                                  (re[0][2] - re[2][0]) * (re[0][2] - re[2][0]) +
                                                  (re[1][0] - re[0][1]) * (re[1][0] - re[0][1])));
    const double c = clampd(0.5 * (re[0][0] + re[1][1] + re[2][2] - 1.0), -1.0, 1.0);
    const double theta = std::atan2(s, c);

    if (s > 1e-9) {
        // 一般情形：反对称部分直接给出轴
        const double k = theta / s;
        return {k * 0.5 * (re[2][1] - re[1][2]), k * 0.5 * (re[0][2] - re[2][0]),
                k * 0.5 * (re[1][0] - re[0][1])};
    }

    if (c > 0.0) {
        return {0.0, 0.0, 0.0}; // θ ≈ 0：无旋转
    }

    // ---- θ ≈ π 的奇异情形 ----
    //
    // 此时 sinθ ≈ 0，反对称部分**恒为零**，轴无法由它确定。若不特殊处理，
    // 上面的 `s > 1e-9` 分支不成立、而 c < 0 又意味着确实需要转 180°，
    // 结果返回零旋转 —— **完全不产生纠正力矩，静默失效**。
    //
    // 实测：180° 偏航指令下 RMS 误差 180°、终值 0°，飞行器原地不动。
    //
    // 正确做法用**对称部分**：θ = π 时 `R_e = 2·a·aᵀ − I`，故
    //     R_e[i][i] + 1 = 2·a_i²        （对角元给出轴的各分量模长）
    //     R_e[i][j]     = 2·a_i·a_j     （非对角元给出相对符号）
    // 取模长最大的分量作主元以保证数值稳定，其余分量由非对角元定出。
    const double d0 = std::max(0.0, (re[0][0] + 1.0) * 0.5);
    const double d1 = std::max(0.0, (re[1][1] + 1.0) * 0.5);
    const double d2 = std::max(0.0, (re[2][2] + 1.0) * 0.5);

    double a[3] = {0.0, 0.0, 0.0};
    if (d0 >= d1 && d0 >= d2 && d0 > 1e-12) {
        a[0] = std::sqrt(d0);
        a[1] = re[0][1] / (2.0 * a[0]);
        a[2] = re[0][2] / (2.0 * a[0]);
    } else if (d1 >= d2 && d1 > 1e-12) {
        a[1] = std::sqrt(d1);
        a[0] = re[0][1] / (2.0 * a[1]);
        a[2] = re[1][2] / (2.0 * a[1]);
    } else if (d2 > 1e-12) {
        a[2] = std::sqrt(d2);
        a[0] = re[0][2] / (2.0 * a[2]);
        a[1] = re[1][2] / (2.0 * a[2]);
    } else {
        return {0.0, 0.0, 0.0};
    }
    const double an = std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
    if (an < 1e-12) {
        return {0.0, 0.0, 0.0};
    }
    const double k = theta / an;
    return {k * a[0], k * a[1], k * a[2]};
}

} // namespace

SixDofPidController::SixDofPidController(SixDofConfig cfg, SixDofPidGains gains)
    : _cfg(std::move(cfg)), _gains(gains) {
    // 三轴增益在构造时一次性确定：按期望带宽/阻尼比 + 惯量推导，或用一组手动值。
    // 放在构造期而不是每步计算，是因为惯量在一次飞行里不变，而 compute 是 1 kHz 热路径。
    if (_gains.derive_attitude_from_inertia) {
        const double wn = std::max(1e-3, _gains.att_bandwidth);
        const double zeta = std::max(1e-3, _gains.att_damping);
        for (int i = 0; i < 3; ++i) {
            const double I = std::max(1e-9, _cfg.inertia[i]);
            _att_kp[i] = I * wn * wn;
            _att_kd[i] = 2.0 * zeta * I * wn;
        }
    } else {
        for (int i = 0; i < 3; ++i) {
            _att_kp[i] = _gains.att_kp;
            _att_kd[i] = _gains.att_kd;
        }
    }
}

void SixDofPidController::reset() {
    _last_tilt_deg = 0.0;
    _last_thrust = 0.0;
    _pos_integral[0] = _pos_integral[1] = _pos_integral[2] = 0.0;
}

namespace {

/**
 * @brief 由期望加速度解算推力与力矩（姿态环 + 推力分配）
 *
 * 抽出来给定点与跟踪两条入口共用。姿态解算与位置环无关，重复实现只会让两条
 * 路径慢慢漂开。
 *
 * @param att_kp/att_kd 三轴姿态增益（构造期已确定）
 * @param tilt_out      输出：实际使用的倾角（度）
 */
SixDofCommand solveCommand(const SixDofConfig &cfg, const SixDofPidGains &gains,
                           const double att_kp[3], const double att_kd[3],
                           const SixDofState &state, const V3d &a_in, double &tilt_out,
                           double yaw_des = 0.0, bool use_yaw = false) {
    const double m = cfg.base.mass;
    const double g = cfg.base.gravity;

    const double a_des[3] = {a_in.x, a_in.y, a_in.z};

    // ---- 合力与期望推力方向 ----
    // F = m·(a_des − g_vec)，g_vec = [0, 0, g]
    const V3d f_des{m * a_des[0], m * a_des[1], m * (a_des[2] - g)};
    const double f_norm = norm(f_des);

    // 期望机体 z 轴（机体 z 朝上，推力沿 −z，故 z_des 与合力反向）
    V3d z_des{0.0, 0.0, 1.0};
    if (f_norm > 1e-9) {
        z_des = {-f_des.x / f_norm, -f_des.y / f_norm, -f_des.z / f_norm};
    }

    // 倾角限幅：过大的倾角会让竖直可用推力不足以维持高度。
    //
    // 限幅作用在**期望推力方向**上（而非误差），物理含义清楚：推力方向不能
    // 偏得太多。这样限幅与偏航互不干扰 —— 偏航是绕 z_des 的转动，不该被
    // 倾斜约束波及。
    const double max_tilt = gains.max_tilt_deg / 180.0 * kPi;
    {
        const double cz = clampd(z_des.z, -1.0, 1.0);
        const double tilt_now = std::acos(cz);
        if (tilt_now > max_tilt) {
            // 把 z_des 绕「与竖直轴垂直的方向」压回 max_tilt
            const V3d axis{0.0, 0.0, 0.0};
            (void)axis;
            const double s_t = std::sin(tilt_now);
            if (s_t > 1e-9) {
                // z_des 的水平分量方向
                const double hx = z_des.x / s_t;
                const double hy = z_des.y / s_t;
                const double st = std::sin(max_tilt);
                const double ct = std::cos(max_tilt);
                z_des = {hx * st, hy * st, ct};
            }
        }
        tilt_out = std::acos(clampd(z_des.z, -1.0, 1.0)) / kPi * 180.0;
    }

    const V3d omega = readV3(state.omega);
    V3d rotvec_body{0.0, 0.0, 0.0};

    if (use_yaw) {
        // ---- 完整姿态误差（含偏航）----
        //
        // 用「期望推力方向 + 期望偏航」构造完整期望姿态，再与当前姿态作差。
        // 这样偏航角被真正约束住，而不是像原路径那样完全自由。
        double r_des[3][3];
        const bool ok = buildDesiredAttitude(z_des, yaw_des, r_des);
        if (ok) {
            double r_cur[3][3];
            rotationMatrixOf(readQuatD(state.quat), r_cur);
            const V3d rotvec_ned = rotationErrorVector(r_des, r_cur);

            // 误差在 NED，力矩须在机体系施加：绕 NED 的旋转向量转到机体系
            // 只需左乘 Rᵀ（正交矩阵的逆即转置），栈上完成、无堆分配。
            rotvec_body = {r_cur[0][0] * rotvec_ned.x + r_cur[1][0] * rotvec_ned.y +
                               r_cur[2][0] * rotvec_ned.z,
                           r_cur[0][1] * rotvec_ned.x + r_cur[1][1] * rotvec_ned.y +
                               r_cur[2][1] * rotvec_ned.z,
                           r_cur[0][2] * rotvec_ned.x + r_cur[1][2] * rotvec_ned.y +
                               r_cur[2][2] * rotvec_ned.z};
        } else {
            use_yaw = false; // 退化时回退到原路径
        }
    }

    if (!use_yaw) {
        // ---- 原路径：只对齐推力方向（偏航自由）----
        //
        // 保留为默认行为：定点悬停不需要约束偏航，且既有全部测试与结果都基于
        // 这条路径，改动会破坏可比性。需要指向的任务显式开启偏航控制。
        const Tensor z_body_frame = makeVec3(0.0f, 0.0f, 1.0f);
        const V3d z_cur = readV3(rotateBodyToNed(state.quat, z_body_frame));

        const V3d axis_raw = cross(z_cur, z_des);
        const double sin_theta = norm(axis_raw);
        const double cos_theta = clampd(dot(z_cur, z_des), -1.0, 1.0);
        const double theta = std::atan2(sin_theta, cos_theta);

        V3d rotvec_ned{0.0, 0.0, 0.0};
        if (sin_theta > 1e-9) {
            rotvec_ned = {axis_raw.x / sin_theta * theta, axis_raw.y / sin_theta * theta,
                          axis_raw.z / sin_theta * theta};
        }

        const Tensor rotvec_ned_t =
            makeVec3(static_cast<float>(rotvec_ned.x), static_cast<float>(rotvec_ned.y),
                     static_cast<float>(rotvec_ned.z));
        rotvec_body = readV3(rotateNedToBody(state.quat, rotvec_ned_t));
    }

    // ---- 姿态环：PD 力矩 ----
    SixDofCommand cmd;
    cmd.thrust_body = f_norm; // 推力大小取合力模长；倾斜时自动增大以维持竖直分量

    cmd.torque = makeVec3(
        static_cast<float>(att_kp[0] * rotvec_body.x - att_kd[0] * omega.x),
        static_cast<float>(att_kp[1] * rotvec_body.y - att_kd[1] * omega.y),
        static_cast<float>(att_kp[2] * rotvec_body.z - att_kd[2] * omega.z));

    return cmd;
}

} // namespace

/**
 * @brief 期望加速度的统一后处理：扰动前馈 + 限幅
 *
 * 三条控制路径共用此出口。见头文件中的说明 —— 这是为避免「同一控制律写三份
 * 导致漏改」而刻意做的收敛。
 */
SixDofCommand SixDofPidController::publish(SixDofCommand cmd) {
    _last_thrust = cmd.thrust_body; // 供下一拍的入流补偿与扰动观测器使用
    return cmd;
}

std::array<double, 3> SixDofPidController::finalizeAccel(const double raw[3],
                                                        const SixDofState &state,
                                                        double dt) {
    double a[3] = {raw[0], raw[1], raw[2]};

    // ---- 扰动前馈：a_des = a_raw − d_hat ----
    //
    // 观测器估的是「未被模型解释的加速度」d。补偿时**减去**它，使
    // a_actual = a_des + d = (a_raw − d_hat) + d → a_raw。
    //
    // 注意它作用于前馈路径（不在反馈回路内），故不改变回路增益、
    // 不消耗相位裕度 —— 这是相对积分的结构性优势。
    if (_gains.use_disturbance_observer) {
        if (!_obs_has_prev) {
            // 首拍：记录速度初值，本拍不补偿
            const V3d v0 = readV3(state.vel);
            _obs_prev_vel[0] = v0.x;
            _obs_prev_vel[1] = v0.y;
            _obs_prev_vel[2] = v0.z;
            _obs_has_prev = true;
        } else if (dt > 0.0) {
            const double hz = _gains.disturbance_observer_hz;
            if (hz > 1e-9) {
                const double Tau = 1.0 / (2.0 * M_PI * hz);
                const double alpha = dt / (Tau + dt);
                const V3d v = readV3(state.vel);
                const double v_arr[3] = {v.x, v.y, v.z};
                const double lim = _gains.disturbance_limit;

                // ---- 「实际施加的加速度」必须用真实姿态与真实推力算 ----
                //
                // 第一版直接用期望加速度 a_des，结果观测器在 0.5 Hz 正常
                // （估计 1.9999 vs 真值 2.00），但 1 Hz 起符号翻转、2 Hz 撞限幅。
                //
                // 原因是**期望与实际之间隔着姿态环**：控制器要求 5 m/s² 水平
                // 加速度，但那要靠倾斜实现，而姿态环有 9 rad/s 带宽、需要时间
                // 跟上。于是 a_actual ≠ a_des，残差里混入了姿态跟踪误差。观测器
                // 把它当扰动补偿，补偿又改变 a_des —— **正反馈，必然发散**。
                //
                // 正确做法：用机体**实际**产生的力计算，即
                //
                //     a_applied = [0, 0, g] + R_actual · [0, 0, −T_last] / m
                //
                // 其中 R_actual 取当前姿态、T_last 取上一拍实际输出的推力。
                // 这样姿态环的滞后已被包含在 R_actual 里，残差中只剩下真正的
                // 未建模力（阻力、外力、推力损失）。
                const double m = _cfg.base.mass;
                const double g = _cfg.base.gravity;
                const V3d f_body{0.0, 0.0, -_last_thrust};
                const Tensor f_body_t =
                    makeVec3(0.0f, 0.0f, static_cast<float>(-_last_thrust));
                const V3d f_ned = readV3(rotateBodyToNed(state.quat, f_body_t));
                const double a_applied[3] = {f_ned.x / m, f_ned.y / m, f_ned.z / m + g};
                (void)f_body;

                for (int i = 0; i < 3; ++i) {
                    const double a_meas = (v_arr[i] - _obs_prev_vel[i]) / dt;
                    const double resid = a_meas - a_applied[i];
                    _d_hat[i] += alpha * (resid - _d_hat[i]);
                    _d_hat[i] = std::max(-lim, std::min(lim, _d_hat[i]));
                    _obs_prev_vel[i] = v_arr[i];
                }
            }
        }
        for (int i = 0; i < 3; ++i) {
            a[i] -= _d_hat[i];
        }
    }

    // ---- 限幅 ----
    const double lim = _gains.max_accel;
    const std::array<double, 3> out{clampd(a[0], -lim, lim), clampd(a[1], -lim, lim),
                                    clampd(a[2], -lim, lim)};

    // 记录**限幅后**的实际值供下一拍构造残差。
    //
    // 必须用限幅后的值：若执行器饱和，控制器以为发出了更大的加速度，
    // 残差里会混入饱和误差并被误认为扰动 —— 这是观测器的已知陷阱。
    _obs_last_applied[0] = out[0];
    _obs_last_applied[1] = out[1];
    _obs_last_applied[2] = out[2];
    _last_a_des[0] = out[0];
    _last_a_des[1] = out[1];
    _last_a_des[2] = out[2];
    return out;
}


SixDofCommand SixDofPidController::compute(const SixDofState &state, const Tensor &target,
                                           double /*time*/) {
    const V3d pos = readV3(state.pos);
    const V3d vel = readV3(state.vel);
    const V3d tgt = readV3(target);

    // ---- 位置环：期望加速度 ----
    // 定点版本把期望速度与期望加速度都视为零。下面的表达式刻意保持原样
    // （写成 kp*e + kd*(-vel) 而不是 kd*(0-vel)），使定点悬停的全部既有实测
    // 结果逐位不变 —— 改控制律时最忌讳把回归数据悄悄改掉。
    const V3d e_pos{tgt.x - pos.x, tgt.y - pos.y, tgt.z - pos.z};

    // 积分项（默认 ki = 0，完全跳过）。
    //
    // 注意：本函数此前是**独立实现**的位置环，与 computeWithWind /
    // computeTracking 各写一份。给后两者加积分时漏了这里，症状是
    // 「调用 compute 时积分项恒为 0」—— 同一条控制律写三遍，改一处就会漏
    // 另外两处。这里补上后三处逻辑一致。
    double i_term[3] = {0.0, 0.0, 0.0};
    if (_gains.pos_ki != 0.0) {
        const double dt = _cfg.base.dt;
        const double e[3] = {e_pos.x, e_pos.y, e_pos.z};
        const double lim = _gains.integral_limit / std::max(1e-9, _gains.pos_ki);
        for (int i = 0; i < 3; ++i) {
            _pos_integral[i] += e[i] * dt;
            _pos_integral[i] = clampd(_pos_integral[i], -lim, lim);
            i_term[i] = _gains.pos_ki * _pos_integral[i];
        }
    }

    const double raw[3] = {
        i_term[0] + _gains.pos_kp * e_pos.x + _gains.pos_kd * (-vel.x),
        i_term[1] + _gains.pos_kp * e_pos.y + _gains.pos_kd * (-vel.y),
        i_term[2] + _gains.pos_kp * e_pos.z + _gains.pos_kd * (-vel.z),
    };
    // 统一出口：扰动前馈 + 限幅。三条路径共用，避免重复实现导致漏改。
    //
    // dt 取 _cfg.base.dt：控制器不持有仿真时间步，故沿用配置值。
    const std::array<double, 3> a_arr = finalizeAccel(raw, state, _cfg.base.dt);
    const V3d a_des{a_arr[0], a_arr[1], a_arr[2]};

    return publish(
        solveCommand(_cfg, _gains, _att_kp, _att_kd, state, a_des, _last_tilt_deg));
}

SixDofCommand SixDofPidController::computeWithWind(const SixDofState &state,
                                                    const Tensor &target,
                                                    const std::array<double, 3> &v_wind,
                                                    double /*time*/) {
    const double m = _cfg.base.mass;

    const V3d pos = readV3(state.pos);
    const V3d vel = readV3(state.vel);
    const V3d tgt = readV3(target);

    // 风阻前馈。气动力取决于**相对气流**，因此必须用飞行器速度与风速之差：
    //
    //   v_rel = v_aircraft − v_wind
    //   F_aero = −k·|v_rel|·v_rel        （作用在机身上，把它推走）
    //   a_ff   = −F_aero / m = +k·|v_rel|·v_rel / m
    //
    // 第一版写的是 a_ff = −k·|v_wind|·v_wind / m，即把 v_rel 简化成了 −v_wind。
    // 那个式子在**稳态静止**时与上式等价，但飞行器一旦有速度（湍流下始终如此）
    // 就会失配 —— 等于把一个本身带建模误差的基线当成对照，让后续比较失去意义。
    // 阻力系数：优先使用各轴异性（若启用），否则回退到标量。
    // 未支持异性之前，前馈恒按各向同性补偿 —— 于是「把真实气动告诉控制器」
    // 这一操作形同虚设（两种配置下误差逐位相同），第 4 段判据因而失效。
    const bool use_axis = _cfg.drag_coeff_axis[0] > 0.0 && _cfg.drag_coeff_axis[1] > 0.0 &&
                          _cfg.drag_coeff_axis[2] > 0.0;
    const double kx = use_axis ? _cfg.drag_coeff_axis[0] : _cfg.base.drag_coeff;
    const double ky = use_axis ? _cfg.drag_coeff_axis[1] : _cfg.base.drag_coeff;
    const double kz = use_axis ? _cfg.drag_coeff_axis[2] : _cfg.base.drag_coeff;

    V3d a_ff{0.0, 0.0, 0.0};
    if (_cfg.base.drag_coeff > 0.0 || use_axis) {
        const double rx = vel.x - v_wind[0];
        const double ry = vel.y - v_wind[1];
        const double rz = vel.z - v_wind[2];
        const double sp = std::sqrt(rx * rx + ry * ry + rz * rz);
        a_ff = {kx * sp * rx / m, ky * sp * ry / m, kz * sp * rz / m};
    }

    // 桨盘入流补偿：实际推力 T_eff = T·(1 − mu·v_axial)，指令推力被入流打了
    // 折扣，故需补足。
    //
    // 补偿量 = 损失的推力，而损失**沿机体 −z 方向**（推力方向），不是沿 NED 的
    // z 轴。机身倾斜时这个损失在 NED 里有水平分量 —— 只补 z 轴会漏掉它们，
    // 留下与倾角相关的系统性残差。
    //
    // 幅值也不能用 T ≈ m·g 近似：机动时推力可达 mg 的 1.2~1.5 倍，近似会低估
    // 损失。这里用**上一拍的指令推力**作为实际推力的估计（一阶近似，误差远小于
    // 用 mg 替代）。
    //
    // 两处修正前，解析补偿只吃掉了总差距的 95%，残余 0.028 m 与倾角相关
    // （相关系数 0.41）；修正后残余应进一步下降。
    // 入流系数来源：优先用在线估计值（若已挂载估计器且其报告样本充足），
    // 否则回退到配置里的固定值。回退是必要的 —— 估计未收敛时用它会引入
    // 比「不知道」更差的补偿。
    double mu_lin = _cfg.inflow_linear;
    double mu_quad = _cfg.inflow_quad;
    if (_gains.use_online_inflow_estimate && _inflow_src != nullptr &&
        _inflow_src->inflowEstimateReady()) {
        mu_lin = _inflow_src->inflowMu();
        mu_quad = 0.0; // 在线估计当前只覆盖线性项
    }
    _active_mu = mu_lin;

    if (mu_lin != 0.0 || mu_quad != 0.0) {
        const Tensor wind_ned =
            makeVec3(static_cast<float>(v_wind[0]), static_cast<float>(v_wind[1]),
                     static_cast<float>(v_wind[2]));
        const Tensor rel_body = rotateNedToBody(state.quat, state.vel - wind_ned);
        const float *rb = rel_body.data<float>();
        const double v_axial = rb[2];
        const double corr = mu_lin * v_axial + mu_quad * v_axial * std::fabs(v_axial);

        // 损失的推力大小（牛顿）：T_loss = T·corr
        const double t_est = (_last_thrust > 1e-9) ? _last_thrust : _cfg.base.mass * _cfg.base.gravity;
        const double f_loss = t_est * corr;

        // 方向沿机体 −z（推力方向）：在机体中为 [0,0,−1]，转到 NED 后取反
        const Tensor loss_body = makeVec3(0.0f, 0.0f, static_cast<float>(-f_loss));
        const Tensor loss_ned = rotateBodyToNed(state.quat, loss_body);

        // 补偿 = **加上**损失矢量（而非减去）。
        //
        // 用水平姿态这个特例验符号：推力损失等效于一个**向下的额外力** T·corr，
        // 要抵消它须让期望推力增大，即 a_ff.z 要**减小**（NED 中 z 向下为正）。
        // 水平时 loss_body = [0,0,−f_loss] 转到 NED 仍是 [0,0,−f_loss]，故
        // `a_ff += loss_ned/m` 恰好给出 a_ff.z -= f_loss/m ✓
        //
        // 第一版写成 `a_ff -= loss_ned/m`，符号反了：补偿变成「再施加一份同等
        // 损失」，误差从 0.137 涨到 1.284（比完全不补偿还差一倍）。
        const float *ln = loss_ned.data<float>();
        a_ff.x += ln[0] / m;
        a_ff.y += ln[1] / m;
        a_ff.z += ln[2] / m;
    }

    const V3d e_pos{tgt.x - pos.x, tgt.y - pos.y, tgt.z - pos.z};

    // ---- 积分项（默认 ki = 0，即纯 PD）----
    //
    // 作用：消除**模型外**常值偏差造成的稳态误差。前馈补的是已知效应，
    // 自适应补的是「已知效应 + 未知系数」，而机身不对称、电机安装偏斜、
    // 重心偏移这类结构性偏差两者都补不了 —— 只有积分能吃掉。
    //
    // 代价：`PM = atan2(kd·ωc − ki/ωc, kp) − ωc·T`，`ki/ωc` 从 `kd·ωc` 中减去。
    // 实测 ki = 1 时裕度损失 < 5°，可接受。
    //
    // 限幅（anti-windup）：执行器饱和期间误差持续累积会让积分涨到很大、
    // 之后长时间退不回来。限幅值取与 max_accel 同量级。
    double i_term[3] = {0.0, 0.0, 0.0};
    if (_gains.pos_ki != 0.0) {
        const double dt = _cfg.base.dt;
        const double e[3] = {e_pos.x, e_pos.y, e_pos.z};
        const double lim = _gains.integral_limit / std::max(1e-9, _gains.pos_ki);
        for (int i = 0; i < 3; ++i) {
            _pos_integral[i] += e[i] * dt;
            _pos_integral[i] = clampd(_pos_integral[i], -lim, lim);
            i_term[i] = _gains.pos_ki * _pos_integral[i];
        }
    }

    const double raw[3] = {
        a_ff.x + i_term[0] + _gains.pos_kp * e_pos.x + _gains.pos_kd * (-vel.x),
        a_ff.y + i_term[1] + _gains.pos_kp * e_pos.y + _gains.pos_kd * (-vel.y),
        a_ff.z + i_term[2] + _gains.pos_kp * e_pos.z + _gains.pos_kd * (-vel.z),
    };
    // 统一出口：扰动前馈 + 限幅。三条路径共用，避免重复实现导致漏改。
    //
    // dt 取 _cfg.base.dt：控制器不持有仿真时间步，故沿用配置值。
    const std::array<double, 3> a_arr = finalizeAccel(raw, state, _cfg.base.dt);
    const V3d a_des{a_arr[0], a_arr[1], a_arr[2]};

    return publish(solveCommand(_cfg, _gains, _att_kp, _att_kd, state, a_des,
                                _last_tilt_deg, 0.0, _gains.use_yaw_control));
}

SixDofCommand SixDofPidController::computeTracking(const SixDofState &state,
                                                   const SixDofSetpoint &ref,
                                                   double /*time*/) {
    const V3d pos = readV3(state.pos);
    const V3d vel = readV3(state.vel);

    // ---- 位置环：带参考速度与加速度前馈 ----
    //   a_des = a_ref + kp·(p_ref − p) + kd·(v_ref − v)
    // 与定点版本的两点差别：向心/切向加速度由 a_ref 提供（不必再用位置误差换），
    // 阻尼作用在速度误差上（不再把轨迹本身的运动速度当成要消除的量）。
    const V3d e_pos{ref.pos[0] - pos.x, ref.pos[1] - pos.y, ref.pos[2] - pos.z};
    const V3d e_vel{ref.vel[0] - vel.x, ref.vel[1] - vel.y, ref.vel[2] - vel.z};

    // 积分项（与定点路径共用同一组积分状态；ki = 0 时完全跳过）
    double i_term[3] = {0.0, 0.0, 0.0};
    if (_gains.pos_ki != 0.0) {
        const double dt = _cfg.base.dt;
        const double e[3] = {e_pos.x, e_pos.y, e_pos.z};
        const double lim = _gains.integral_limit / std::max(1e-9, _gains.pos_ki);
        for (int i = 0; i < 3; ++i) {
            _pos_integral[i] += e[i] * dt;
            _pos_integral[i] = clampd(_pos_integral[i], -lim, lim);
            i_term[i] = _gains.pos_ki * _pos_integral[i];
        }
    }

    const double raw[3] = {
        ref.acc[0] + i_term[0] + _gains.pos_kp * e_pos.x + _gains.pos_kd * e_vel.x,
        ref.acc[1] + i_term[1] + _gains.pos_kp * e_pos.y + _gains.pos_kd * e_vel.y,
        ref.acc[2] + i_term[2] + _gains.pos_kp * e_pos.z + _gains.pos_kd * e_vel.z,
    };
    // 统一出口：扰动前馈 + 限幅。三条路径共用，避免重复实现导致漏改。
    //
    // dt 取 _cfg.base.dt：控制器不持有仿真时间步，故沿用配置值。
    const std::array<double, 3> a_arr = finalizeAccel(raw, state, _cfg.base.dt);
    const V3d a_des{a_arr[0], a_arr[1], a_arr[2]};

    // 参考偏航角接入控制律（此前只传给平坦前馈，未参与反馈 —— 偏航因此
    // 完全自由，是结构性缺失）。
    SixDofCommand cmd = solveCommand(_cfg, _gains, _att_kp, _att_kd, state, a_des,
                                     _last_tilt_deg, ref.yaw, _gains.use_yaw_control);

    // ---- 平坦前馈：角速度参考 ----
    //
    // 姿态环原本的阻尼项是 `−kd·ω`，语义是「把角速度阻尼到零」—— 这对定点
    // 悬停成立，对机动飞行则是在**持续对抗**机身本该有的转动。目标律是
    //
    //     τ = kp·e − kd·(ω − ω_des)
    //
    // 而 solveCommand 已给出 `τ_old = kp·e − kd·ω`，故**增量恰为 `+kd·ω_des`**。
    // （写成 `+kd·(ω_des − ω)` 会多减一次 kd·ω、把阻尼翻倍，实测使跟踪误差
    // 反而放大 5 倍，且误差与振幅成正比 —— 常值滞后的典型指纹。）
    //
    // 关闭开关或参考未提供 jerk 时 ω_des 为零，增量为零，行为与改造前逐位相同。
    if (_gains.use_flat_omega_feedforward) {
        FlatReference fr;
        fr.pos = ref.pos;
        fr.vel = ref.vel;
        fr.acc = ref.acc;
        fr.jerk = ref.jerk;
        fr.yaw = ref.yaw;
        fr.yaw_rate = ref.yaw_rate;

        const FlatOutput fo = computeFlatFeedforward(fr, _cfg.base.mass, _cfg.base.gravity);
        if (fo.valid) {
            const float *tq = cmd.torque.data<float>();
            const std::array<float, 3> tnew{
                static_cast<float>(tq[0] + _att_kd[0] * fo.omega[0]),
                static_cast<float>(tq[1] + _att_kd[1] * fo.omega[1]),
                static_cast<float>(tq[2] + _att_kd[2] * fo.omega[2])};
            cmd.torque = makeVec3(tnew[0], tnew[1], tnew[2]);
        }
    }

    return publish(cmd);
}

} // namespace oi3
