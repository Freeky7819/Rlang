#include "rlang_scheduler.hpp"
#include <thread>
#include <vector>
#include <algorithm>
namespace rlang {
void parallel_for(size_t n, size_t grain, const std::function<void(size_t,size_t)>& fn){
  unsigned T = std::max(1u, std::thread::hardware_concurrency());
  std::vector<std::thread> th; th.reserve(T);
  size_t chunk = (n + T - 1) / T;
  for (unsigned t=0;t<T;++t){
    th.emplace_back([=,&fn]{
      size_t b=t*chunk, e=std::min(b+chunk,n);
      for (size_t i=b;i<e;i+=grain){ size_t j=std::min(i+grain,e); fn(i,j); }
    });
  }
  for (auto& x: th) x.join();
}
} // namespace rlang
