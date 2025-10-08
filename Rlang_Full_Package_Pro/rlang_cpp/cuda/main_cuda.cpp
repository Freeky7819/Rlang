#include <iostream>
#include <vector>
#include <cmath>
#include "../include/rlang.hpp"

int main(int argc, char** argv){
  if (argc < 3){
    std::cerr << "Usage: rlang_cuda_cli <profile.txt> <state.txt>\n";
    return 1;
  }
  auto p = rlang::load_profile_txt(argv[1]);
  auto s = rlang::load_state_txt(argv[2]);
  // NOTE: For brevity, this stub just calls scalar on CPU.
  auto out = rlang::step_scalar(s, p);
  std::cout << rlang::to_json_like(out);
  return 0;
}
