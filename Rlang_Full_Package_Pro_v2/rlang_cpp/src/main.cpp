#include "rlang.hpp"
#include <iostream>
int main(int argc, char** argv){
  if (argc < 3){ std::cerr << "Usage: rlang_cli <profile.txt> <state.txt>\n"; return 1; }
  auto p = rlang::load_profile_txt(argv[1]);
  auto s = rlang::load_state_txt(argv[2]);
#if defined(__AVX2__)
  auto out = rlang::step_simd_avx2(s,p);
#else
  auto out = rlang::step_scalar(s,p);
#endif
  std::cout << rlang::to_json_like(out);
  return 0;
}
