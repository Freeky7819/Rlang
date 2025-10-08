#pragma once
#include <vector>
#include <cstdint>
#include <string>
#include <functional>
namespace rlang {
struct State { std::vector<double> phase; std::vector<double> amp; uint64_t seed; };
struct Profile { std::vector<double> chord; double alpha; double omega; };
struct StepResult { State state; };
StepResult step_scalar(const State& s, const Profile& p);
StepResult step_simd_avx2(const State& s, const Profile& p);
void parallel_for(size_t n, size_t grain, const std::function<void(size_t,size_t)>& fn);
Profile load_profile_txt(const std::string& path);
State   load_state_txt(const std::string& path);
std::string to_json_like(const StepResult& r);
inline uint64_t next_seed(uint64_t seed){ return (seed * 6364136223846793005ULL + 1ULL); }
} // namespace rlang
