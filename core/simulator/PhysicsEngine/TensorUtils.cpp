/**
 * @file TensorUtils.cpp
 * @brief OpenInspire3 张量辅助工具实现
 * @author GhostFace
 * @date 2026/9/16
 */

#include "TensorUtils.h"

namespace oi3 {

std::vector<float> toVector(const Tensor &t) {
    const std::vector<size_t> &shape = t.shape();
    const float *data = t.data<float>();

    // 标量张量（shape 为空或 {1}）也走同一条路径
    size_t n = 1;
    if (!shape.empty()) {
        n = shape[0];
    }

    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) {
        out[i] = data[i];
    }
    return out;
}

void printVector(const Tensor &t, const std::string &name, std::ostream &os) {
    const std::vector<float> v = toVector(t);
    os << name << " = [";
    for (size_t i = 0; i < v.size(); ++i) {
        if (i != 0) {
            os << ", ";
        }
        os << v[i];
    }
    os << "]";
}

} // namespace oi3
