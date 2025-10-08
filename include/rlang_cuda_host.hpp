
// include/rlang_cuda_host.hpp — v0.1.4 GPU host glue (header-only facade)
#pragma once
#include <vector>
#include <cstdint>

namespace rlangf_cuda {

struct DeviceBuffers {
    double *phase=nullptr, *freq=nullptr, *damp=nullptr, *dphi=nullptr, *Xi=nullptr;
    int *a=nullptr, *b=nullptr, *p=nullptr, *q=nullptr;
    int Nosc=0, Nedge=0;
};

// Allocate/copy helpers (implemented in .cpp to avoid CUDA headers in this header)
bool init(DeviceBuffers& dev, int Nosc, int Nedge);
bool upload_state(DeviceBuffers& dev, const std::vector<double>& phase,
                  const std::vector<double>& freq,  const std::vector<double>& damp);
bool upload_edges(DeviceBuffers& dev, const std::vector<int>& a, const std::vector<int>& b,
                  const std::vector<double>& Xi, const std::vector<int>& p, const std::vector<int>& q);
bool clear_dphi(DeviceBuffers& dev);
bool download_phase(DeviceBuffers& dev, std::vector<double>& phase);

bool launch_kuramoto(DeviceBuffers& dev, int blocks, int threads);
bool launch_advance(DeviceBuffers& dev, double dt, int blocks, int threads);
void destroy(DeviceBuffers& dev);

} // namespace rlangf_cuda
