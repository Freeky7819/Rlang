
// src/rlang_cuda_host.cpp — v0.1.4 simple CPU<->GPU glue (requires CUDA runtime)
#include "rlang_cuda_host.hpp"
#include <cuda_runtime.h>
#include <cstdio>

extern "C" {
    void kuramoto_accumulate(const double*, const int*, const int*, const double*, const int*, const int*, double*, int);
    void advance_osc(double*, const double*, const double*, const double*, double, int);
}

namespace rlangf_cuda {

#define CUDA_OK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
    std::fprintf(stderr,"CUDA error: %s @%s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); return false; } } while(0)

bool init(DeviceBuffers& dev, int Nosc, int Nedge){
    dev.Nosc=Nosc; dev.Nedge=Nedge;
    CUDA_OK(cudaMalloc(&dev.phase, sizeof(double)*Nosc));
    CUDA_OK(cudaMalloc(&dev.freq,  sizeof(double)*Nosc));
    CUDA_OK(cudaMalloc(&dev.damp,  sizeof(double)*Nosc));
    CUDA_OK(cudaMalloc(&dev.dphi,  sizeof(double)*Nosc));
    CUDA_OK(cudaMalloc(&dev.Xi,    sizeof(double)*Nedge));
    CUDA_OK(cudaMalloc(&dev.a,     sizeof(int)*Nedge));
    CUDA_OK(cudaMalloc(&dev.b,     sizeof(int)*Nedge));
    CUDA_OK(cudaMalloc(&dev.p,     sizeof(int)*Nedge));
    CUDA_OK(cudaMalloc(&dev.q,     sizeof(int)*Nedge));
    return true;
}
bool upload_state(DeviceBuffers& dev, const std::vector<double>& phase,
                  const std::vector<double>& freq,  const std::vector<double>& damp){
    CUDA_OK(cudaMemcpy(dev.phase, phase.data(), sizeof(double)*dev.Nosc, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dev.freq,  freq.data(),  sizeof(double)*dev.Nosc, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dev.damp,  damp.data(),  sizeof(double)*dev.Nosc, cudaMemcpyHostToDevice));
    return true;
}
bool upload_edges(DeviceBuffers& dev, const std::vector<int>& a, const std::vector<int>& b,
                  const std::vector<double>& Xi, const std::vector<int>& p, const std::vector<int>& q){
    CUDA_OK(cudaMemcpy(dev.a,  a.data(),  sizeof(int)*dev.Nedge, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dev.b,  b.data(),  sizeof(int)*dev.Nedge, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dev.Xi, Xi.data(), sizeof(double)*dev.Nedge, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dev.p,  p.data(),  sizeof(int)*dev.Nedge, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dev.q,  q.data(),  sizeof(int)*dev.Nedge, cudaMemcpyHostToDevice));
    return true;
}
bool clear_dphi(DeviceBuffers& dev){
    CUDA_OK(cudaMemset(dev.dphi, 0, sizeof(double)*dev.Nosc));
    return true;
}
bool download_phase(DeviceBuffers& dev, std::vector<double>& phase){
    CUDA_OK(cudaMemcpy(phase.data(), dev.phase, sizeof(double)*dev.Nosc, cudaMemcpyDeviceToHost));
    return true;
}

bool launch_kuramoto(DeviceBuffers& dev, int blocks, int threads){
    kuramoto_accumulate<<<blocks,threads>>>(dev.phase, dev.a, dev.b, dev.Xi, dev.p, dev.q, dev.dphi, dev.Nedge);
    return cudaPeekAtLastError()==cudaSuccess;
}
bool launch_advance(DeviceBuffers& dev, double dt, int blocks, int threads){
    advance_osc<<<blocks,threads>>>(dev.phase, dev.freq, dev.damp, dev.dphi, dt, dev.Nosc);
    return cudaPeekAtLastError()==cudaSuccess;
}
void destroy(DeviceBuffers& dev){
    auto f=[&](void* p){ if(p) cudaFree(p); };
    f(dev.phase); f(dev.freq); f(dev.damp); f(dev.dphi); f(dev.Xi); f(dev.a); f(dev.b); f(dev.p); f(dev.q);
    dev = DeviceBuffers{};
}

} // namespace rlangf_cuda
