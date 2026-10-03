#include "HybridController.h"
#include <iostream>

using namespace oi3;

class DummyNet : public ResidualNetworkSource {
public:
    std::array<double, 3> predictResidualAccel(const SixDofState &, const std::array<double, 3> &, double) override {
        return {0.0, 0.0, 0.0};
    }
};

SixDofConfig baseConfig() {
    SixDofConfig cfg;
    return cfg;
}

int main() {
    SixDofConfig cfg = baseConfig();
    SixDofPidGains gains;
    HybridController ctrl(cfg, gains);
    DummyNet net;
    ctrl.setNetworkSource(&net);
    std::cout << "HybridController compiled and initialized successfully!" << std::endl;
    return 0;
}
