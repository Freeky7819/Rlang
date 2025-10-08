// examples/opt_adam.cpp
#include "rlang_fast_simd.hpp"
#include "rlang_opt_adam.hpp"
#include <cstdio>
#include <cmath>

int main(){
    rlangf::System sys;
    auto C = sys.add_osc(264.0,0.0,0.01,0.0,0.8);
    auto E = sys.add_osc(330.0,0.5,0.01,0.0,0.7);
    auto G = sys.add_osc(396.0,1.0,0.01,0.0,0.7);
    sys.couple_k(C,E,20.0,5,4);
    sys.couple_k(E,G,20.0,6,5);
    sys.couple_k(C,G,20.0,3,2);

    // Optimize the three Xi values to minimize sum of squared phase errors.
    std::vector<double> Xi = {20.0,20.0,20.0};
    auto apply = [&](const std::vector<double>& p){
        sys.k_edges[0].Xi = p[0]; sys.k_edges[1].Xi = p[1]; sys.k_edges[2].Xi = p[2];
    };
    auto loss = [&]()->double{
        // run small chunk to settle a bit
        sys.run(0.02, 1.0/48000.0, true, 0, true);
        double e = 0.0;
        e += std::pow(std::abs(sys.phase_error(C,E,5,4)),2);
        e += std::pow(std::abs(sys.phase_error(E,G,6,5)),2);
        e += std::pow(std::abs(sys.phase_error(C,G,3,2)),2);
        return e;
    };

    rlangf_opt::adam(Xi, apply, loss, 80, 0.1);
    std::printf("Optimized Xi: %.2f, %.2f, %.2f\n", Xi[0],Xi[1],Xi[2]);
    return 0;
}
