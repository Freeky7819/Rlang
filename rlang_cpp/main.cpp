#include "rlang.hpp"
#include <fstream>
#include <iostream>
#include <sstream>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: rlang_cpp <profile.json> <state.json>\n";
        return 1;
    }
    std::ifstream pf(argv[1]), sf(argv[2]);
    std::stringstream pb, sb;
    pb << pf.rdbuf(); sb << sf.rdbuf();

    auto prof = rlang::profile_from_json(pb.str());
    auto state = rlang::state_from_json(sb.str());
    auto result = rlang::step(state, prof);
    std::cout << rlang::to_json(result) << std::endl;
    return 0;
}
