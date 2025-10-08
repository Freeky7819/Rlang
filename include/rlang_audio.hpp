
#pragma once
#include <vector>
#include <cstdint>
#include <cstdio>
#include <cmath>

namespace rlangf_audio {

inline void write_wav16(const char* path, const std::vector<float>& mono, int sample_rate=44100){
    FILE* f = std::fopen(path, "wb"); if(!f) return;
    auto u32=[&](uint32_t v){ std::fwrite(&v,4,1,f); };
    auto u16=[&](uint16_t v){ std::fwrite(&v,2,1,f); };
    auto ch =[&](const char*s){ std::fwrite(s,1,4,f); };
    uint32_t data_bytes = mono.size()*2;
    // RIFF header
    ch("RIFF"); u32(36 + data_bytes); ch("WAVE");
    // fmt
    ch("fmt "); u32(16); u16(1); u16(1); u32(sample_rate); u32(sample_rate*2); u16(2); u16(16);
    // data
    ch("data"); u32(data_bytes);
    for(float x : mono){
        int16_t s = (int16_t)std::round(std::max(-1.f, std::min(1.f, x)) * 32767.f);
        std::fwrite(&s, 2, 1, f);
    }
    std::fclose(f);
}

} // namespace
