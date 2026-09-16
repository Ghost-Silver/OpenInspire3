/**
 * @file TensorUtils.h
 * @brief OpenInspire3 张量辅助工具
 * @author GhostFace
 * @date 2026/9/16
 */

#ifndef OI3_TENSOR_UTILS_H
#define OI3_TENSOR_UTILS_H

#include "Tensor.h"
#include <iostream>
#include <string>
#include <vector>

namespace oi3 {

/**
 * @brief 把一维张量读成 std::vector（用于打印与断言）
 *
 * 只读访问，不修改张量，也不影响计算图。
 */
[[nodiscard]] std::vector<float> toVector(const Tensor &t);

/**
 * @brief 把一维张量格式化为 "name = [a, b, c]" 输出
 */
void printVector(const Tensor &t, const std::string &name,
                 std::ostream &os = std::cout);

} // namespace oi3

#endif // OI3_TENSOR_UTILS_H
