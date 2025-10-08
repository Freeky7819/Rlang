// examples/audio_triads.cpp
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
    const double T = 2.0; // 2 seconds
    const int N = int(T*sr);
    std::vector<float> mono; mono.reserve(N);

    for(int i=0;i<N;++i){
        sys.step_jobified(dt, 0, /*avx2*/true);
        // Simple additive synth from oscillator phases
        double s = 0.0;
        s += std::sin(sys.osc.phase[C]);
        s += std::sin(sys.osc.phase[E]);
        s += std::sin(sys.osc.phase[G]);
        mono.push_back((float)(s/3.0));
    }
    rlangf_audio::write_wav16("/mnt/data/triad_lock.wav", mono, sr);
    std::printf("WAV saved to /mnt/data/triad_lock.wav\n");
    return 0;
}
