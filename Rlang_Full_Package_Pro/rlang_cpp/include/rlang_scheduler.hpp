#pragma once
#include <functional>
#include <cstddef>

namespace rlang {
// Very small thread helper; if OpenMP present, we use it.
void parallel_for(size_t n, size_t grain, const std::function<void(size_t,size_t)>& fn);
}