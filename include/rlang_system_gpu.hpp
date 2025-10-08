
#pragma once
#include <vector>
#include <cstdint>
#include <memory>
#include <functional>
#include "rlang_fast_simd.hpp"
#ifdef RLANG_CUDA_ENABLED
  #include "rlang_cuda_host.hpp"
#endif

namespace rlangf {

// High-level GPU wrapper with CPU fallback.
class SystemGPU {
public:
    SystemGPU() = default;
    explicit SystemGPU(System* cpu) : cpu_(cpu) {}

    void attach(System* cpu){ cpu_ = cpu; }
    bool available() const {
    #ifdef RLANG_CUDA_ENABLED
        return true;
    #else
        return false;
    #endif
    }

    bool upload();
    bool step(double dt);
    bool download();

private:
    System* cpu_{nullptr};
#ifdef RLANG_CUDA_ENABLED
    rlangf_cuda::DeviceBuffers dev_;
#endif
};

} // namespace rlangf
