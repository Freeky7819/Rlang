#pragma once
#include <string>
#include <vector>
#include <cstdint>

namespace rlang {

struct State {
  std::vector<double> phase;
  std::vector<double> amp;
  uint64_t seed;
};

struct Profile {
  std::vector<double> chord;
  double alpha;
  double omega;
};

struct StepResult {
  State state;
  double loss;
};

StepResult step(const State& s, const Profile& p);
std::string to_json(const StepResult& r);
State state_from_json(const std::string& j);
Profile profile_from_json(const std::string& j);

}
