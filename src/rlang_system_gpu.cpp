
#include "rlang_system_gpu.hpp"
#ifdef RLANG_CUDA_ENABLED
  #include "rlang_cuda_host.hpp"
#endif
#include <cstdio>

namespace rlangf {

bool SystemGPU::upload(){
#ifdef RLANG_CUDA_ENABLED
    if (!cpu_) return false;
    const int Nosc = (int)cpu_->osc.n;
    const int Nedge= (int)cpu_->k_edges.size();
    if (!rlangf_cuda::init(dev_, Nosc, Nedge)) return false;

    std::vector<int> a(Nedge), b(Nedge), p(Nedge), q(Nedge);
    std::vector<double> Xi(Nedge);
    for (int i=0;i<Nedge;++i){
        a[i]=cpu_->k_edges[i].a; b[i]=cpu_->k_edges[i].b;
        p[i]=cpu_->k_edges[i].p; q[i]=cpu_->k_edges[i].q; Xi[i]=cpu_->k_edges[i].Xi;
    }
    if (!rlangf_cuda::upload_state(dev_, cpu_->osc.phase, cpu_->osc.freq, cpu_->osc.damp)) return false;
    if (!rlangf_cuda::upload_edges(dev_, a,b,Xi,p,q)) return false;
    if (!rlangf_cuda::clear_dphi(dev_)) return false;
    return true;
#else
    return false;
#endif
}

bool SystemGPU::step(double dt){
#ifdef RLANG_CUDA_ENABLED
    if (!cpu_) return false;
    int blocks = (dev_.Nedge + 255)/256, threads=256;
    if (!rlangf_cuda::clear_dphi(dev_)) return false;
    if (!rlangf_cuda::launch_kuramoto(dev_, blocks, threads)) return false;
    blocks = (dev_.Nosc + 255)/256;
    if (!rlangf_cuda::launch_advance(dev_, dt, blocks, threads)) return false;
    cpu_->time += dt;
    return true;
#else
    return false;
#endif
}

bool SystemGPU::download(){
#ifdef RLANG_CUDA_ENABLED
    if (!cpu_) return false;
    return rlangf_cuda::download_phase(dev_, cpu_->osc.phase);
#else
    return false;
#endif
}

} // namespace rlangf
