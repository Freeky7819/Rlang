#include "rlang_scheduler.hpp"
#include <thread>
#include <vector>
#include <algorithm>

namespace rlang {

void parallel_for(size_t n, size_t grain, const std::function<void(size_t,size_t)>& fn){
#if defined(RLANG_USE_OPENMP)
  #pragma omp parallel
  {
    size_t T = omp_get_num_threads();
    size_t tid = omp_get_thread_num();
    size_t chunk = (n + T - 1) / T;
    size_t begin = tid * chunk;
    size_t end   = std::min(begin + chunk, n);
    for (size_t i = begin; i < end; i += grain){
      size_t j = std::min(i + grain, end);
      fn(i, j);
    }
  }
#else
  unsigned T = std::max(1u, std::thread::hardware_concurrency());
  std::vector<std::thread> th; th.reserve(T);
  size_t chunk = (n + T - 1) / T;
  for (unsigned t=0; t<T; ++t){
    th.emplace_back([=,&fn]{
      size_t begin = t*chunk, end = std::min(begin+chunk, n);
      for (size_t i=begin;i<end;i+=grain){
        size_t j = std::min(i+grain,end);
        fn(i,j);
      }
    });
  }
  for (auto& x: th) x.join();
#endif
}

} // namespace rlang
