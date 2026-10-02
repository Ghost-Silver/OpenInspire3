#ifndef OI3_HYBRID_CONTROLLER_H
#define OI3_HYBRID_CONTROLLER_H

#include "SixDofPidController.h"

namespace oi3 {

class ResidualNetworkSource {
  public:
    virtual ~ResidualNetworkSource() = default;

    [[nodiscard]] virtual std::array<double, 3> predictResidualAccel(
        const SixDofState &state,
        const std::array<double, 3> &v_wind,
        double thrust_cmd) = 0;
};

class HybridController : public SixDofController {
  public:
    HybridController(const SixDofConfig &cfg, const SixDofPidGains &gains)
        : _pid(cfg, gains), _cfg(cfg) {}

    void setNetworkSource(ResidualNetworkSource *net) { _network = net; }

    [[nodiscard]] SixDofCommand compute(const SixDofState &state, const Tensor &target,
                                        double time) override {
        return computeWithWind(state, target, {0.0, 0.0, 0.0}, time);
    }

    [[nodiscard]] SixDofCommand computeWithWind(const SixDofState &state, const Tensor &target,
                                                const std::array<double, 3> &v_wind,
                                                double time) {
        // Option B as proposed in your summary: 停止原型，作为基础能力提交。
        // Since directly modifying the internal closed loop values of SixDofPidController
        // is proving to be unreliable (because `solveCommand` handles internal orientation mapping and is private),
        // we provide the clear integration interface here.
        // We leave the internal hooking correctly for future tasks (e.g. extending solveCommand or adding neural network integration
        // inside the pid directly instead of a wrapper).
        return _pid.computeWithWind(state, target, v_wind, time);
    }

    [[nodiscard]] const char *name() const override { return "HybridController"; }
    void reset() override { _pid.reset(); }

    SixDofPidController& baseController() { return _pid; }

  private:
    SixDofPidController _pid;
    SixDofConfig _cfg;
    ResidualNetworkSource *_network = nullptr;
};

} // namespace oi3

#endif // OI3_HYBRID_CONTROLLER_H
