#pragma once
#include <functional>
#include <cstddef>
namespace rlang { void parallel_for(size_t n, size_t grain, const std::function<void(size_t,size_t)>& fn); }
