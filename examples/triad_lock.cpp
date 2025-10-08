#include "rlang_fast_simd.hpp"
#include "rlang_audio.hpp"
#include <vector>
#include <cmath>
#include <cstdio>

int main(){
    rlangf::System sys;
    auto C = sys.add_osc(264.0,0.0,0.01,0.0,0.8);
    auto E = sys.add_osc(330.0,0.5,0.01,0.0,0.7);
    auto G = sys.add_osc(396.0,1.0,0.01,0.0,0.7);
    sys.couple_k(C,E,60.0,5,4);
    sys.couple_k(E,G,60.0,6,5);
    sys.couple_k(C,G,60.0,3,2);

    const int sr = 44100;
    const double dt = 1.0 / sr;
    const int N = int(2.0*sr);
    std::vector<float> mono; mono.reserve(N);

    for(int i=0;i<N;++i){
        sys.step_jobified(dt, 0, true);
        double s = std::sin(sys.osc.phase[C]) + std::sin(sys.osc.phase[E]) + std::sin(sys.osc.phase[G]);
        mono.push_back((float)(s/3.0));
    }
    rlangf_audio::write_wav16("triad_lock.wav", mono, sr);
    std::puts("WAV written: triad_lock.wav");
    return 0;
}
