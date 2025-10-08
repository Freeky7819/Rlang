#include "rlang.hpp"
#include <cmath>
#if defined(__AVX2__)
  #include <immintrin.h>
#endif
namespace rlang {
StepResult step_simd_avx2(const State& s, const Profile& p){
#if defined(__AVX2__)
  StepResult out; size_t n = s.phase.size(); out.state.phase.resize(n); out.state.amp.resize(n);
  size_t i=0;
  for (; i+4<=n; i+=4){
    __m256d ph = _mm256_loadu_pd(&s.phase[i]);
    __m256d am = _mm256_loadu_pd(&s.amp[i]);
    __m256d ch = _mm256_loadu_pd(&p.chord[i]);
    __m256d omg= _mm256_set1_pd(p.omega);
    __m256d arg= _mm256_add_pd(_mm256_mul_pd(ph, omg), ch);
    double arg_s[4]; _mm256_storeu_pd(arg_s, arg);
    double d_s[4]; for(int k=0;k<4;++k) d_s[k] = p.alpha * std::sin(arg_s[k]);
    __m256d d = _mm256_loadu_pd(d_s);
    __m256d ph_out = _mm256_add_pd(ph, d);
    __m256d am_out = _mm256_add_pd(am, _mm256_mul_pd(_mm256_set1_pd(0.1), d));
    _mm256_storeu_pd(&out.state.phase[i], ph_out);
    _mm256_storeu_pd(&out.state.amp[i], am_out);
  }
  for (; i<n; ++i){
    double d = p.alpha * std::sin(s.phase[i]*p.omega + p.chord[i]);
    out.state.phase[i] = s.phase[i] + d;
    out.state.amp[i]   = s.amp[i]   + 0.1*d;
  }
  out.state.seed = next_seed(s.seed);
  return out;
#else
  return step_scalar(s,p);
#endif
}
} // namespace rlang
