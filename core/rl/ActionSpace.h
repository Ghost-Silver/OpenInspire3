/**
 * @file ActionSpace.h
 * @brief 动作空间定义与底层控制器
 * @author GhostFace
 * @date 2026/4/4
 */

#ifndef ACTIONSPACE_H
#define ACTIONSPACE_H

#include <vector>

// 3D 向量类
class Vec3 {
public:
    double x_, y_, z_;

    Vec3(double x = 0.0, double y = 0.0, double z = 0.0) : x_(x), y_(y), z_(z) {}

    // 加法
    Vec3 operator+(const Vec3& other) const {
        return Vec3(x_ + other.x_, y_ + other.y_, z_ + other.z_);
    }

    // 减法
    Vec3 operator-(const Vec3& other) const {
        return Vec3(x_ - other.x_, y_ - other.y_, z_ - other.z_);
    }

    // 乘法（标量）
    Vec3 operator*(double scalar) const {
        return Vec3(x_ * scalar, y_ * scalar, z_ * scalar);
    }

    // 赋值
    Vec3& operator=(const Vec3& other) {
        if (this != &other) {
            x_ = other.x_;
            y_ = other.y_;
            z_ = other.z_;
        }
        return *this;
    }

    // 获取坐标
    double x() const { return x_; }
    double y() const { return y_; }
    double z() const { return z_; }

    // 静态方法：零向量
    static Vec3 Zero() {
        return Vec3(0.0, 0.0, 0.0);
    }
};

// 动作枚举（7个离散动作）
enum class Action {
    HOVER,           // 悬停
    FORWARD,         // 向前
    BACKWARD,        // 向后
    LEFT,            // 向左
    RIGHT,           // 向右
    UP,              // 向上
    DOWN             // 向下
};

// 动作对应的速度矢量 (m/s)
const std::vector<Vec3> ACTION_VELOCITY = {
    Vec3(0.0, 0.0, 0.0),   // HOVER
    Vec3(0.0, 0.5, 0.0),    // FORWARD (y方向)
    Vec3(0.0, -0.5, 0.0),   // BACKWARD (y方向)
    Vec3(-0.5, 0.0, 0.0),   // LEFT (x方向)
    Vec3(0.5, 0.0, 0.0),    // RIGHT (x方向)
    Vec3(0.0, 0.0, 0.5),    // UP (z方向)
    Vec3(0.0, 0.0, -0.5)    // DOWN (z方向)
};

// 无人机类
class UAV {
private:
    Vec3 position_;   // 位置
    Vec3 velocity_;   // 速度
    Vec3 acceleration_; // 加速度
    double mass_;     // 质量

public:
    UAV(double mass = 1.0) : mass_(mass) {
        position_ = Vec3::Zero();
        velocity_ = Vec3::Zero();
        acceleration_ = Vec3::Zero();
    }

    // 获取位置
    const Vec3& getPosition() const {
        return position_;
    }

    // 获取速度
    const Vec3& getVelocity() const {
        return velocity_;
    }

    // 获取加速度
    const Vec3& getAcceleration() const {
        return acceleration_;
    }

    // 获取质量
    double getMass() const {
        return mass_;
    }

    // 设置位置
    void setPosition(const Vec3& position) {
        position_ = position;
    }

    // 设置速度
    void setVelocity(const Vec3& velocity) {
        velocity_ = velocity;
    }

    // 设置加速度
    void setAcceleration(const Vec3& acceleration) {
        acceleration_ = acceleration;
    }

    // 更新状态
    void update(double dt) {
        velocity_ = velocity_ + acceleration_ * dt;
        position_ = position_ + velocity_ * dt;
    }
};

// 无人机管理器
class UAVManager {
private:
    std::vector<UAV> uavs_;  // 无人机列表
    int num_uavs_;            // 无人机数量

public:
    UAVManager(int num_uavs) : num_uavs_(num_uavs) {
        uavs_.resize(num_uavs);
    }

    // 重置无人机位置
    void reset(const std::vector<Vec3>& init_positions) {
        for (int i = 0; i < num_uavs_; i++) {
            uavs_[i].setPosition(init_positions[i]);
            uavs_[i].setVelocity(Vec3::Zero());
            uavs_[i].setAcceleration(Vec3::Zero());
        }
    }

    // 更新所有无人机状态
    void updateAll(double dt, const std::vector<Vec3>& accelerations) {
        for (int i = 0; i < num_uavs_; i++) {
            uavs_[i].setAcceleration(accelerations[i]);
            uavs_[i].update(dt);
        }
    }

    // 获取无人机位置
    std::vector<Vec3> getPositions() const {
        std::vector<Vec3> positions;
        for (const UAV& uav : uavs_) {
            positions.push_back(uav.getPosition());
        }
        return positions;
    }

    // 获取无人机速度
    std::vector<Vec3> getVelocities() const {
        std::vector<Vec3> velocities;
        for (const UAV& uav : uavs_) {
            velocities.push_back(uav.getVelocity());
        }
        return velocities;
    }

    // 获取无人机加速度
    std::vector<Vec3> getAccelerations() const {
        std::vector<Vec3> accelerations;
        for (const UAV& uav : uavs_) {
            accelerations.push_back(uav.getAcceleration());
        }
        return accelerations;
    }

    // 获取无人机
    const UAV& getUAV(int index) const {
        return uavs_[index];
    }

    // 获取无人机数量
    int getNumUAVs() const {
        return num_uavs_;
    }
};

#endif // ACTIONSPACE_H
