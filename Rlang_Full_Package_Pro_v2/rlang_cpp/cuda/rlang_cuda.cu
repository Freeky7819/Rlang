#include <cuda_runtime.h>
extern "C" __global__
void rlang_step_kernel(const double* phase, const double* amp, const double* chord,
                       double alpha, double omega, double* out_phase, double* out_amp, int n){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i<n){
    double delta = alpha * sin(phase[i]*omega + chord[i]);
    out_phase[i] = phase[i] + delta;
    out_amp[i]   = amp[i]   + 0.1*delta;
  }
}
