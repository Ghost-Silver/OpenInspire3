/**
 * @file Baseline.cpp
 * @brief 基线策略与统一评测驱动实现
 * @author GhostFace
 * @date 2026/9/16
 */

#include "Baseline.h"
#include "EnvManager.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>

namespace oi3::rl {

namespace {

/// 观测向量前 8 维：位置(3) 速度(3) 覆盖率(1) 最近目标距离(1)
constexpr int kLocalMapStart = 8;
/// 局部地图为 5 × 5 × 3
constexpr int kMapXY = 5;
constexpr int kMapZ = 3;
/// 判定「未探索」的阈值（与 VoxelMap::getLocalMap 的编码约定一致）
constexpr float kUnexplored = 0.5f;

/// 多机去冲突的距离阈值
constexpr double kTooClose = 3.0;
constexpr double kTooFar = 20.0;
/// 预测下一步位置时使用的单步位移
constexpr double kStepAhead = 0.8;
/// 障碍 / 边界惩罚与协同权重（沿用原实现）
constexpr double kPenaltyHard = 15.0;
constexpr double kPenaltyCrowd = 3.0;
constexpr double kBonusSpread = 2.0;
constexpr double kHoverWeightAdvanced = 0.2;

/// 统计局部地图中某立方体区域的未探索体素数
double countUnexplored(const std::vector<float> &obs, int z0, int z1, int y0, int y1,
                       int x0, int x1) {
    double n = 0.0;
    const int size = static_cast<int>(obs.size());
    for (int z = z0; z < z1; ++z) {
        for (int y = y0; y < y1; ++y) {
            for (int x = x0; x < x1; ++x) {
                const int idx = kLocalMapStart + z * kMapXY * kMapXY + y * kMapXY + x;
                if (idx < size && obs[idx] < kUnexplored) {
                    n += 1.0;
                }
            }
        }
    }
    return n;
}

/// 7 个动作对应的方向单位向量（顺序与 Action 枚举一致）
const std::vector<Vec3> &actionDirections() {
    static const std::vector<Vec3> dirs = {
        Vec3(0.0, 1.0, 0.0),  // FORWARD
        Vec3(0.0, -1.0, 0.0), // BACKWARD
        Vec3(-1.0, 0.0, 0.0), // LEFT
        Vec3(1.0, 0.0, 0.0),  // RIGHT
        Vec3(0.0, 0.0, 1.0),  // UP
        Vec3(0.0, 0.0, -1.0), // DOWN
        Vec3(0.0, 0.0, 0.0)   // HOVER
    };
    return dirs;
}

/// 各方向的未探索体素计数（基础探索得分）
///
/// 统计范围严格沿用原实现，索引布局为 idx = kLocalMapStart + z*25 + y*5 + x：
/// 四个水平方向各只取局部地图边缘的一条线（y=0 行 / x=0 列 / x=4 列），
/// 垂直方向取顶、底两层中心 3x3 区域，悬停取全部三层的中心 3x3 区域。
/// 注意水平方向的取样范围是「线」而非整个切片，这是原实现的既有口径，
/// 重构阶段不做更改；若需改为整片统计属于策略行为变更，须单独评估。
std::vector<double> baseExplorationScores(const std::vector<float> &obs) {
    std::vector<double> s(7, 0.0);
    s[0] = countUnexplored(obs, 0, kMapZ, 0, 1, 2, kMapXY);      // 前 (y+)
    s[1] = countUnexplored(obs, 0, kMapZ, 0, 1, 0, 3);           // 后 (y-)
    s[2] = countUnexplored(obs, 0, kMapZ, 0, 3, 0, 1);           // 左 (x-)
    s[3] = countUnexplored(obs, 0, kMapZ, 2, kMapXY, 4, kMapXY); // 右 (x+)
    s[4] = countUnexplored(obs, 2, kMapZ, 1, 4, 1, 4);           // 上 (z+)
    s[5] = countUnexplored(obs, 0, 1, 1, 4, 1, 4);               // 下 (z-)
    s[6] = countUnexplored(obs, 0, kMapZ, 1, 4, 1, 4);           // 悬停
    return s;
}

/// 取最高分动作；并列时取下标最小者（与原实现一致）
int argmaxAction(const std::vector<double> &scores) {
    int best = 0;
    double best_score = scores.front();
    for (int i = 1; i < static_cast<int>(scores.size()); ++i) {
        if (scores[i] > best_score) {
            best_score = scores[i];
            best = i;
        }
    }
    return best;
}

double distance(const Vec3 &a, const Vec3 &b) {
    const double dx = a.x() - b.x();
    const double dy = a.y() - b.y();
    const double dz = a.z() - b.z();
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

bool insideObstacle(const Vec3 &pos, const std::vector<Obstacle> &obstacles) {
    for (const auto &ob : obstacles) {
        if (distance(pos, ob.center) < ob.size / 2.0) {
            return true;
        }
    }
    return false;
}

double mean(const std::vector<double> &v) {
    if (v.empty()) {
        return 0.0;
    }
    return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
}

double stddev(const std::vector<double> &v, double m) {
    if (v.empty()) {
        return 0.0;
    }
    double acc = 0.0;
    for (double x : v) {
        acc += (x - m) * (x - m);
    }
    return std::sqrt(acc / static_cast<double>(v.size()));
}

} // namespace

// ---------------------------------------------------------------------------
// 障碍物转换
// ---------------------------------------------------------------------------

std::vector<AABB> toAABBs(const std::vector<Obstacle> &obstacles) {
    std::vector<AABB> out;
    out.reserve(obstacles.size());
    for (const auto &ob : obstacles) {
        const double half = ob.size / 2.0;
        out.push_back(AABB{ob.center.x() - half, ob.center.y() - half, ob.center.z() - half,
                           ob.center.x() + half, ob.center.y() + half, ob.center.z() + half});
    }
    return out;
}

// ---------------------------------------------------------------------------
// 基线策略
// ---------------------------------------------------------------------------

Policy makeGreedyPolicy(double hover_weight) {
    return [hover_weight](const std::vector<float> &obs, const PolicyContext & /*ctx*/) {
        std::vector<double> scores = baseExplorationScores(obs);
        scores[6] *= hover_weight;
        return static_cast<Action>(argmaxAction(scores));
    };
}

Action advancedGreedySelect(const std::vector<float> &obs, const PolicyContext &ctx) {
    std::vector<double> scores = baseExplorationScores(obs);
    scores[6] *= kHoverWeightAdvanced;

    const Vec3 current(obs[0], obs[1], obs[2]);
    const std::vector<Vec3> &dirs = actionDirections();

    // 障碍与边界惩罚
    for (int i = 0; i < 7; ++i) {
        const Vec3 next = current + dirs[i] * kStepAhead;
        if (ctx.obstacles != nullptr && insideObstacle(next, *ctx.obstacles)) {
            scores[i] -= kPenaltyHard;
        }
        if (next.x() < 0.5 || next.x() >= 12.0 - 0.5 || next.y() < 0.5 ||
            next.y() >= 12.0 - 0.5 || next.z() < 0.5 || next.z() >= 3.0 - 0.5) {
            scores[i] -= kPenaltyHard;
        }
    }

    // 多机协同：过近则相互推开，过远则鼓励分散探索
    if (ctx.all_observations != nullptr) {
        const auto &all = *ctx.all_observations;
        for (int i = 0; i < static_cast<int>(all.size()); ++i) {
            if (i == ctx.uav_id) {
                continue;
            }
            const Vec3 other(all[i][0], all[i][1], all[i][2]);
            const double dist = distance(current, other);

            if (dist < kTooClose) {
                for (int j = 0; j < 6; ++j) {
                    if (distance(current + dirs[j] * kStepAhead, other) < dist) {
                        scores[j] -= kPenaltyCrowd;
                    }
                }
            }
            if (dist > kTooFar) {
                for (int j = 0; j < 6; ++j) {
                    if (distance(current + dirs[j] * kStepAhead, other) > dist) {
                        scores[j] += kBonusSpread;
                    }
                }
            }
        }
    }

    return static_cast<Action>(argmaxAction(scores));
}

// ---------------------------------------------------------------------------
// 评测驱动
// ---------------------------------------------------------------------------

EvalResult evaluate(const Scenario &sc, const Policy &policy, bool verbose) {
    const std::vector<AABB> aabbs = toAABBs(sc.obstacles);
    EnvManager env(sc.map_x, sc.map_y, sc.map_z, sc.resolution, sc.sensor_radius,
                   sc.start_positions, sc.max_steps, aabbs);
    if (sc.num_targets > 0) {
        env.set_num_targets(sc.num_targets);
    }

    const int num_uavs = static_cast<int>(sc.start_positions.size());
    std::vector<double> ep_rewards;
    std::vector<double> ep_steps;
    ep_rewards.reserve(sc.num_episodes);
    ep_steps.reserve(sc.num_episodes);
    int successes = 0;

    // 预热：部分原实现在进入 episode 循环前会先 reset 一次并读取初始观测
    // （用于取得观测维度）。目标生成会随 reset 次数推进，故该调用会影响逐
    // episode 结果；此处按场景声明的次数复刻同样的调用序。
    for (int i = 0; i < sc.warmup_resets; ++i) {
        env.reset();
        (void)env.get_local_observations();
    }

    for (int episode = 0; episode < sc.num_episodes; ++episode) {
        env.reset();

        std::vector<double> rewards(static_cast<std::size_t>(num_uavs), 0.0);
        bool success = false;
        int steps = 0;

        for (int step = 0; step < sc.max_steps; ++step) {
            const auto observations = env.get_local_observations();

            std::vector<Action> actions;
            actions.reserve(num_uavs);
            for (int i = 0; i < num_uavs; ++i) {
                PolicyContext ctx;
                ctx.uav_id = i;
                ctx.all_observations = &observations;
                ctx.obstacles = &sc.obstacles;
                actions.push_back(policy(observations[i], ctx));
            }

            auto result = env.step(actions);
            const auto &step_rewards = std::get<0>(result);
            const bool done = std::get<2>(result);

            for (int i = 0; i < num_uavs; ++i) {
                rewards[i] += step_rewards[i];
            }

            // 成功判据：所有无人机累计奖励均达到阈值（沿用原实现）
            bool all_above = true;
            for (double r : rewards) {
                if (r < sc.success_reward) {
                    all_above = false;
                    break;
                }
            }
            if (all_above) {
                success = true;
            }

            steps = step + 1;
            if (done) {
                break;
            }
        }

        const double total = std::accumulate(rewards.begin(), rewards.end(), 0.0);
        if (success) {
            ++successes;
        }
        ep_rewards.push_back(total);
        ep_steps.push_back(static_cast<double>(steps));

        if (verbose) {
            std::cout << " Episode " << (episode + 1) << ": 总奖励 = " << total
                      << ", 成功 = " << (success ? "是" : "否")
                      << ", 步数 = " << steps << std::endl;
            if ((episode + 1) % 10 == 0) {
                std::cout << "\n--- 前 " << (episode + 1) << " 个 episode 成功率: "
                          << (100.0 * successes / (episode + 1)) << "% ---\n"
                          << std::endl;
            }
        }
    }

    EvalResult r;
    r.episodes = sc.num_episodes;
    r.successes = successes;
    r.success_rate =
        sc.num_episodes > 0 ? 100.0 * successes / sc.num_episodes : 0.0;
    r.avg_reward = mean(ep_rewards);
    r.reward_std = stddev(ep_rewards, r.avg_reward);
    r.avg_steps = mean(ep_steps);
    r.steps_std = stddev(ep_steps, r.avg_steps);
    return r;
}

void printResult(const Scenario &sc, const EvalResult &r) {
    std::cout << "\n========== " << sc.name << " ==========" << std::endl;
    std::cout << "地图 " << sc.map_x << "x" << sc.map_y << "x" << sc.map_z
              << "，无人机 " << sc.start_positions.size() << "，目标 " << sc.num_targets
              << "，障碍 " << sc.obstacles.size() << std::endl;
    std::cout << "传感器半径 " << sc.sensor_radius << "，最大步数 " << sc.max_steps
              << "，episode " << sc.num_episodes << std::endl;
    std::cout << "成功率: " << r.success_rate << "% (" << r.successes << "/"
              << r.episodes << ")" << std::endl;
    std::cout << "平均奖励: " << r.avg_reward << " ± " << r.reward_std << std::endl;
    std::cout << "平均步数: " << r.avg_steps << " ± " << r.steps_std << std::endl;
    std::cout << "=================================================" << std::endl;
}

// ---------------------------------------------------------------------------
// 预置场景（参数取自原各测试文件，保持实测口径不变）
// ---------------------------------------------------------------------------

Scenario makeEasyScenario() {
    Scenario s;
    s.name = "贪心覆盖基线（easy）";
    s.map_x = 50;
    s.map_y = 50;
    s.map_z = 15;
    s.num_targets = 5;
    s.sensor_radius = 1.5;
    s.max_steps = 300;
    s.num_episodes = 50;
    // 原实现在循环前先 reset 一次以读取观测维度
    s.warmup_resets = 1;
    s.start_positions = {Vec3(1.0, 1.0, 0.5), Vec3(3.0, 3.0, 0.5)};
    return s;
}

Scenario makeHardScenario() {
    Scenario s;
    s.name = "贪心覆盖基线（hard）";
    s.map_x = 80;
    s.map_y = 80;
    s.map_z = 20;
    // 注：原文件定义 NUM_TARGETS = 15，但生成环境时从未调用 set_num_targets()，
    // 原注释亦写明「暂时使用默认的 5 个目标」，故实际生效值为 5。此处以实际
    // 生效值为准，保持基线的实测口径不变。
    s.num_targets = 5;
    s.sensor_radius = 1.2;
    s.max_steps = 400;
    s.num_episodes = 50;
    s.start_positions = {Vec3(2.0, 2.0, 1.0), Vec3(6.0, 2.0, 1.0),
                         Vec3(2.0, 6.0, 1.0), Vec3(6.0, 6.0, 1.0)};
    return s;
}

Scenario makeUltraHardScenario() {
    Scenario s;
    s.name = "贪心覆盖基线（ultra-hard）";
    s.map_x = 120;
    s.map_y = 120;
    s.map_z = 30;
    // 注：原文件定义 NUM_TARGETS = 30 但实际调用 set_num_targets(12)；
    // 此处以实际生效值为准（12）。
    s.num_targets = 12;
    s.sensor_radius = 0.8;
    s.max_steps = 600;
    s.num_episodes = 20;
    s.start_positions = {Vec3(2.0, 2.0, 1.5),  Vec3(6.0, 2.0, 1.5),
                         Vec3(10.0, 2.0, 1.5), Vec3(2.0, 6.0, 1.5),
                         Vec3(6.0, 6.0, 1.5),  Vec3(10.0, 6.0, 1.5)};
    // 原 generateObstacles() 忽略全部参数、硬编码返回下列 5 个障碍物，
    // 此处将其显式列为场景配置（保持实测口径一致）。
    s.obstacles = {Obstacle{Vec3(4.0, 4.0, 1.5), 2.0}, Obstacle{Vec3(8.0, 4.0, 1.5), 2.0},
                   Obstacle{Vec3(4.0, 8.0, 1.5), 2.0}, Obstacle{Vec3(8.0, 8.0, 1.5), 2.0},
                   Obstacle{Vec3(6.0, 6.0, 1.5), 2.5}};
    return s;
}

} // namespace oi3::rl
