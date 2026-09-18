/**
 * @file SpinningController.cpp
 * @brief 旋转容错模式控制器实现
 * @author GhostFace
 * @date 2026/9/18
 *
 * 姿态误差仍然用「轴角对齐机体 z 轴」的表示 —— 与定点 PID 相同，而不是手动模式
 * 的完整四元数误差。理由正是本模式的前提：**偏航自由度已经被放弃**，不需要也
 * 不应该去约束它。用完整四元数误差会引入偏航分量，进而通过混控转嫁到 τx/τy 上，
 * 破坏倾角控制。
 */

#include "SpinningController.h"

#include "SixDofDynamics.h"

#include <algorithm>
#include <cmath>

namespace oi3 {

namespace {

constexpr double kPi = 3.14159265358979323846;

double clampd(double v, double lo, double hi) { return std::max(lo, std::min(hi, v)); }

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

/// 从四元数提取偏航角（ZYX 约定）
double yawOf(const Tensor &quat) {
    const float *q = quat.data<float>();
    const double w = q[0], x = q[1], y = q[2], z = q[3];
    return std::atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z));
}

} // namespace

SpinningController::SpinningController(SixDofConfig cfg, SpinConfig sc)
    : _cfg(std::move(cfg)), _sc(sc) {
    const double wn = std::max(1e-3, _sc.att_bandwidth);
    const double zeta = std::max(1e-3, _sc.att_damping);
    for (int i = 0; i < 3; ++i) {
        const double I = std::max(1e-9, _cfg.inertia[i]);
        _att_kp[i] = I * wn * wn;
        _att_kd[i] = 2.0 * zeta * I * wn;
    }
}

SixDofCommand SpinningController::compute(const SixDofState &state) const {
    const double m = _cfg.base.mass;
    const double g = _cfg.base.gravity;

    const V3d pos = readV3(state.pos);
    const V3d vel = readV3(state.vel);
    const V3d omega = readV3(state.omega);

    // ---- 高度环：期望 NED 竖直加速度 ----
    // NED 系 z 向下为正，目标高度是负数（例如 −5 表示 5 米高）。
    const double z_err = _sc.target_z - pos.z;
    const double a_cmd = clampd(_sc.z_kp * z_err - _sc.z_kd * vel.z, -_sc.max_accel,
                                _sc.max_accel);

    // ---- 目标姿态：只约束倾角**大小**，不约束倾斜**方位** ----
    //
    // 这里踩过两个坑，都记下来：
    //
    // 坑一（机体系定义）：把目标写成机体系的 [sinθ, 0, cosθ] 再转到 NED 比较。
    // 但当前机体 z 轴在机体系里恒为 [0,0,1]，两者夹角永远等于 θ —— 控制器
    // 会认为「一直差 θ」，持续输出力矩把飞行器推翻。**姿态误差要求两个量在
    // 同一坐标系中比较**，而重力定义在 NED 系，基准只能选 NED。
    //
    // 坑二（跟随偏航角）：改成用当前偏航角 ψ 定义倾斜方向
    // [sinθ·cosψ, sinθ·sinψ, −cosθ]，结果形成正反馈：倾斜过程本身会带动
    // ψ 变化 → 目标跟着转 → 控制器去追一个旋转的目标 → 翻滚加剧（实测
    // wx 冲到 52 rad/s，而 wz 反而接近零）。
    //
    // 结论：**倾斜方位不能由控制律指定**。方位应由角动量自然决定，控制器
    // 只把倾角的大小拉到目标值 —— 目标方向取当前倾斜方向本身，误差因此只
    // 沿「把倾角拉大/拉小」这一个方向，不产生任何方位分量。
    const double th = _sc.tilt_deg / 180.0 * kPi;
    const Tensor z_axis_body = makeVec3(0.0f, 0.0f, 1.0f);
    const V3d z_cur = readV3(rotateBodyToNed(state.quat, z_axis_body));

    // 注意 NED 系的 z 轴**朝下**，机体 z 轴与之同向：水平时 z_cur = [0,0,+1]，
    // 倾斜 θ 后为 [sinθ·ux, sinθ·uy, **+**cosθ] —— z 分量始终为正。
    //
    // 这里踩过第三个坑：写成 −cosθ 会让目标指向反方向，与水平姿态的夹角变成
    // acos(−cosθ)，θ=25° 时是 155° 而非 25°。控制器于是从一开始就在纠正一个
    // 155° 的假误差，把飞行器直接打翻 —— 表现为「从精确目标倾角出发也保不住」，
    // 而根源只是一个符号。
    const double hnorm = std::sqrt(z_cur.x * z_cur.x + z_cur.y * z_cur.y);
    V3d z_des{};
    if (hnorm > 1e-6) {
        // 保持当前倾斜方向，只把倾角大小拉到 θ
        const double ux = z_cur.x / hnorm;
        const double uy = z_cur.y / hnorm;
        z_des = {ux * std::sin(th), uy * std::sin(th), std::cos(th)};
    } else {
        // 恰好竖直时方向未定义，任取一个作为起始方向
        z_des = {std::sin(th), 0.0, std::cos(th)};
    }

    // ---- 姿态误差：把 z_cur 旋到 z_des 的轴角 ----
    const V3d axis_raw = cross(z_cur, z_des);
    const double sin_t = norm(axis_raw);
    const double cos_t = clampd(dot(z_cur, z_des), -1.0, 1.0);
    const double theta = std::atan2(sin_t, cos_t);

    V3d rotvec_ned{0.0, 0.0, 0.0};
    if (sin_t > 1e-9) {
        rotvec_ned = {axis_raw.x / sin_t * theta, axis_raw.y / sin_t * theta,
                      axis_raw.z / sin_t * theta};
    }

    // 误差旋转向量在 NED 系，力矩须在机体系施加
    const Tensor rotvec_ned_t =
        makeVec3(static_cast<float>(rotvec_ned.x), static_cast<float>(rotvec_ned.y),
                 static_cast<float>(rotvec_ned.z));
    const V3d e = readV3(rotateNedToBody(state.quat, rotvec_ned_t));

    // ---- 力矩 ----
    // 偏航轴留空（τz = 0）：偏航力矩本就不可独立指定，下发非零值只会让混控把它
    // 转嫁到 τx/τy 上，反而破坏倾角控制。
    //
    // 注意到位后姿态误差为零、力矩也为零，于是角动量守恒，自旋得以保持 ——
    // 不需要额外的陀螺前馈项。若将来发现到位后仍在缓慢漂移，才需要补 ω×(Iω)。
    // 陀螺耦合前馈：动力学 I·ω̇ = τ − ω×(Iω) − kω，要得到期望闭环
    // I·ω̇ = kp·e − kd·ω，控制力矩必须加上 ω×(Iω)。
    // 这一项在低速时可忽略（所以定点与手动控制器都没有它），但在自旋模式下
    // ω_z 不再是小量，且耦合是交叉的（驱动 x 轴的是 ω_y），不补偿会形成
    // 两轴互相激励的振荡。
    V3d gyro{0.0, 0.0, 0.0};
    if (_sc.compensate_gyroscopic) {
        const double iw[3] = {_cfg.inertia[0] * omega.x, _cfg.inertia[1] * omega.y,
                              _cfg.inertia[2] * omega.z};
        gyro = cross(V3d{omega.x, omega.y, omega.z}, V3d{iw[0], iw[1], iw[2]});
    }

    SixDofCommand cmd;
    const double tau_z = _sc.control_yaw
                             ? (-_att_kp[2] * 0.0 - _att_kd[2] * omega.z) // 仅阻尼
                             : 0.0;
    cmd.torque = makeVec3(static_cast<float>(_att_kp[0] * e.x - _att_kd[0] * omega.x + gyro.x),
                          static_cast<float>(_att_kp[1] * e.y - _att_kd[1] * omega.y + gyro.y),
                          static_cast<float>(tau_z + gyro.z));

    // ---- 推力 ----
    // 竖直方向：m·a_z = g·m − T·cosθ，要得到期望的 a_cmd 需 T·cosθ = m(g − a_cmd)。
    // 与手动模式同样的教训：补偿必须用**实际**倾角，用目标倾角会在过渡期过推，
    // 而且推力需求同时也是偏航力矩的来源（τz 含 c·T 项），用错会一并影响自旋。
    const double cos_act = clampd(1.0 - 2.0 * (z_cur.x * z_cur.x + z_cur.y * z_cur.y), 0.2,
                                  1.0);
    cmd.thrust_body = std::max(0.0, m * (g - a_cmd) / cos_act);

    return cmd;
}

} // namespace oi3
