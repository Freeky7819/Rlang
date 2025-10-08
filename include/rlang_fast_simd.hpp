
// rlang_fast_simd.hpp — v0.1.3
// Header‑only, C++17. SoA layout, AVX2 SIMD for Kuramoto accumulation + oscillator advance,
// jobified stepping, profiles, tiny optimizers, simple phase history, and binary log.
//
// Build: -O3 -ffast-math -march=native (GCC/Clang). To enable AVX2 paths, compile with -mavx2 or -march=native.
//
// Optional CUDA backend provided in /cuda as separate .cu (build independently).

#pragma once
#include <cstdint>
#include <vector>
#include <cmath>
#include <atomic>
#include <cstring>
#include <algorithm>
#include <functional>
#include <thread>
#include <random>
#include <chrono>
#include <string>

#if defined(__AVX2__)
  #ifndef RLANG_USE_AVX2
  #define RLANG_USE_AVX2 1
  #endif
#endif

#if RLANG_USE_AVX2
  #include <immintrin.h>
#endif

#ifndef RLANG_ALIGN
#  define RLANG_ALIGN alignas(64)
#endif

namespace rlangf {

constexpr double PI  = 3.14159265358979323846;
constexpr double TAU = 6.28318530717958647692;

struct RLANG_ALIGN OscPool {
    uint32_t n{0};
    std::vector<double> phase, freq, damp, ent, amp;
    void reserve(uint32_t cap){
        phase.resize(cap); freq.resize(cap); damp.resize(cap); ent.resize(cap); amp.resize(cap);
    }
    uint32_t add(double f_hz, double phase0=0.0, double damping=0.0, double ent_sigma=0.0, double amplitude=1.0){
        if (n >= phase.size()) reserve(std::max<uint32_t>(8, n*2));
        phase[n]=phase0; freq[n]=f_hz; damp[n]=damping; ent[n]=ent_sigma; amp[n]=amplitude;
        return n++;
    }
};

struct RLANG_ALIGN IntPool {
    uint32_t n{0};
    std::vector<double> value, decay;
    void reserve(uint32_t cap){ value.resize(cap); decay.resize(cap); }
    uint32_t add(double init, double decay_=0.0){
        if(n>=value.size()) reserve(std::max<uint32_t>(8, n*2));
        value[n]=init; decay[n]=decay_; return n++;
    }
};

struct KuramotoEdge { uint32_t a, b; double Xi; int p, q; };
struct LinearEdge   { uint32_t a, b; double k_ab, k_ba; };

struct RNG { uint64_t s=0x9E3779B97F4A7C15ull;
    inline uint64_t next(){ s^=s<<7; s^=s>>9; return s; }
    inline double gauss(){
        double u = ((next()>>11) * (1.0/9007199254740992.0));
        double v = ((next()>>11) * (1.0/9007199254740992.0));
        return std::sqrt(-2.0*std::log(u+1e-15)) * std::cos(TAU*v);
    }
};

template<size_t CAP=1<<15>
struct BinLog {
    struct Rec { double t; uint32_t code; uint32_t a, b; double x; };
    std::vector<Rec> buf; size_t head{0};
    BinLog(){ buf.resize(CAP); }
    inline void push(double t, uint32_t code, uint32_t a, uint32_t b, double x){
        buf[head % CAP] = Rec{t,code,a,b,x}; ++head;
    }
    void save(const char* path, size_t count=0) const {
        FILE* f = std::fopen(path,"wb");
        if(!f) return;
        uint64_t N = count? count: std::min(head, buf.size());
        std::fwrite(&N, sizeof(uint64_t), 1, f);
        for(uint64_t i=0;i<N;++i){
            const Rec& r = buf[i % buf.size()];
            std::fwrite(&r, sizeof(Rec), 1, f);
        }
        std::fclose(f);
    }
};

// Simple phase history for selected indices (for viewer/debug)
struct PhaseHistory {
    std::vector<uint32_t> indices;
    std::vector<double>   times;
    std::vector<std::vector<double>> samples; // per-index time series
    void setup(const std::vector<uint32_t>& idx){
        indices = idx;
        samples.assign(indices.size(), {});
        times.clear();
    }
    void push(double t, const std::vector<double>& phases){
        times.push_back(t);
        for(size_t k=0;k<indices.size();++k) samples[k].push_back(phases[indices[k]]);
    }
    void save(const char* path) const {
        FILE* f = std::fopen(path,"wb");
        if(!f) return;
        uint64_t K = indices.size(), T = times.size();
        std::fwrite(&K, sizeof(uint64_t), 1, f);
        std::fwrite(indices.data(), sizeof(uint32_t), K, f);
        std::fwrite(&T, sizeof(uint64_t), 1, f);
        std::fwrite(times.data(), sizeof(double), T, f);
        for(size_t k=0;k<K;++k){
            std::fwrite(samples[k].data(), sizeof(double), T, f);
        }
        std::fclose(f);
    }
};

struct System {
    OscPool osc;
    IntPool integ;
    std::vector<KuramotoEdge> k_edges;
    std::vector<LinearEdge>   l_edges;
    RNG rng;
    double time{0.0};
    BinLog<> log;
    PhaseHistory hist;

    // Profiles
    void profile_music_major_triad(const char* noise="low", const char* xi="high", double damp=0.01){
        double noise_cap = (std::string(noise)=="low")?0.1: (std::string(noise)=="high"?0.3:0.2);
        double xi_boost  = (std::string(xi)=="low")?30.0: (std::string(xi)=="high"?60.0:45.0);
        for(uint32_t i=0;i<osc.n;++i){
            osc.ent[i]   = std::min(osc.ent[i], noise_cap);
            osc.damp[i]  = std::max(osc.damp[i], damp);
        }
        for(auto& e: k_edges){ e.Xi = std::max(e.Xi, xi_boost); }
        log.push(time, 1, 0,0, 0.0);
    }
    void profile_neuro_gamma_sync(double target_hz=40.0){
        const double band_lo=30.0, band_hi=80.0;
        for(uint32_t i=0;i<osc.n;++i){
            if (osc.freq[i] >= band_lo && osc.freq[i] <= band_hi){
                osc.freq[i] = 0.8*osc.freq[i] + 0.2*target_hz;
                osc.damp[i] = std::max(osc.damp[i], 0.02);
                osc.ent[i]  = std::min(osc.ent[i], 0.05);
            }
        }
        for(auto& e : k_edges){
            if (e.p==1 && e.q==1) e.Xi = std::max(e.Xi, 70.0);
        }
        log.push(time, 2, 0,0, target_hz);
    }

    // API
    uint32_t add_osc(double f,double ph=0,double d=0,double e=0,double a=1){ return osc.add(f,ph,d,e,a); }
    uint32_t add_int(double init,double decay=0){ return integ.add(init,decay); }
    void couple_k(uint32_t ai,uint32_t bi,double Xi,int p=1,int q=1){ k_edges.push_back({ai,bi,Xi,p,q}); }
    void couple_l(uint32_t ai,uint32_t bi,double kab,double kba){ l_edges.push_back({ai,bi,kab,kba}); }

    // Scalar Kuramoto accumulation
    inline void accumulate_k_scalar(std::vector<double>& dphi){
        for (auto &e : k_edges){
            double err = std::fmod(e.p*osc.phase[e.a] - e.q*osc.phase[e.b] + PI, TAU); if (err<0) err+=TAU; err -= PI;
            double da  = - e.Xi * std::sin(err);
            dphi[e.a] += da;
            dphi[e.b] += - da * (double(e.p)/double(e.q?e.q:1));
        }
    }

    // AVX2 accumulation (batch 4)
    void accumulate_k_avx2(std::vector<double>& dphi){
    #if RLANG_USE_AVX2
        const size_t N = k_edges.size();
        size_t i = 0;
        for (; i + 3 < N; i += 4) {
            uint32_t a0=k_edges[i+0].a, a1=k_edges[i+1].a, a2=k_edges[i+2].a, a3=k_edges[i+3].a;
            uint32_t b0=k_edges[i+0].b, b1=k_edges[i+1].b, b2=k_edges[i+2].b, b3=k_edges[i+3].b;
            double e0 = k_edges[i+0].p*osc.phase[a0] - k_edges[i+0].q*osc.phase[b0];
            double e1 = k_edges[i+1].p*osc.phase[a1] - k_edges[i+1].q*osc.phase[b1];
            double e2 = k_edges[i+2].p*osc.phase[a2] - k_edges[i+2].q*osc.phase[b2];
            double e3 = k_edges[i+3].p*osc.phase[a3] - k_edges[i+3].q*osc.phase[b3];
            double ev[4] = {e0,e1,e2,e3};
            for(int k=0;k<4;++k){ double x = std::fmod(ev[k] + PI, TAU); if (x<0) x+=TAU; ev[k] = x - PI; }
            for(int k=0;k<4;++k) ev[k] = std::sin(ev[k]);
            __m256d s = _mm256_loadu_pd(ev);
            __m256d Xi = _mm256_set_pd(k_edges[i+3].Xi, k_edges[i+2].Xi, k_edges[i+1].Xi, k_edges[i+0].Xi);
            __m256d da = _mm256_mul_pd(_mm256_set1_pd(-1.0), _mm256_mul_pd(Xi, s));
            double dav[4]; _mm256_storeu_pd(dav, da);
            dphi[a0] += dav[0]; dphi[a1] += dav[1]; dphi[a2] += dav[2]; dphi[a3] += dav[3];
            double r0 = (double)k_edges[i+0].p / (double)(k_edges[i+0].q ? k_edges[i+0].q : 1);
            double r1 = (double)k_edges[i+1].p / (double)(k_edges[i+1].q ? k_edges[i+1].q : 1);
            double r2 = (double)k_edges[i+2].p / (double)(k_edges[i+2].q ? k_edges[i+2].q : 1);
            double r3 = (double)k_edges[i+3].p / (double)(k_edges[i+3].q ? k_edges[i+3].q : 1);
            dphi[b0] += -dav[0]*r0; dphi[b1] += -dav[1]*r1; dphi[b2] += -dav[2]*r2; dphi[b3] += -dav[3]*r3;
        }
        for (; i < N; ++i) {
            auto &e = k_edges[i];
            double err = std::fmod(e.p*osc.phase[e.a] - e.q*osc.phase[e.b] + PI, TAU); if (err<0) err+=TAU; err -= PI;
            double da  = - e.Xi * std::sin(err);
            dphi[e.a] += da;
            dphi[e.b] += - da * (double(e.p)/double(e.q?e.q:1));
        }
    #else
        (void)dphi;
    #endif
    }

    // Scalar advance for oscillators
    inline void advance_scalar(double dt, const std::vector<double>& dphi){
        for (uint32_t i=0;i<osc.n;++i){
            double ph = osc.phase[i];
            ph += TAU*osc.freq[i]*dt + dphi[i]*dt - osc.damp[i]*ph*dt;
            if (osc.ent[i]>0.0) ph += rng.gauss()*osc.ent[i]*std::sqrt(dt);
            if (ph >= PI || ph < -PI){
                ph = std::fmod(ph + PI, TAU); if (ph < 0) ph += TAU; ph -= PI;
            }
            osc.phase[i] = ph;
        }
    }

    // AVX2 advance (batch 4), noise handled scalar for correctness
    inline void advance_avx2(double dt, const std::vector<double>& dphi){
    #if RLANG_USE_AVX2
        const size_t N = osc.n;
        size_t i=0;
        __m256d vdt = _mm256_set1_pd(dt);
        __m256d vTAU = _mm256_set1_pd(TAU);
        for(; i+3 < N; i+=4){
            double phv[4] = {osc.phase[i+0], osc.phase[i+1], osc.phase[i+2], osc.phase[i+3]};
            double fqv[4] = {osc.freq[i+0],  osc.freq[i+1],  osc.freq[i+2],  osc.freq[i+3]};
            double dv[4]  = {dphi[i+0],      dphi[i+1],      dphi[i+2],      dphi[i+3]};
            double dampv[4]={osc.damp[i+0],  osc.damp[i+1],  osc.damp[i+2],  osc.damp[i+3]};

            __m256d ph = _mm256_loadu_pd(phv);
            __m256d fq = _mm256_loadu_pd(fqv);
            __m256d d  = _mm256_loadu_pd(dv);
            __m256d dm = _mm256_loadu_pd(dampv);

            ph = _mm256_add_pd(ph, _mm256_mul_pd(_mm256_mul_pd(vTAU,fq), vdt)); // TAU*freq*dt
            ph = _mm256_add_pd(ph, _mm256_mul_pd(d, vdt));                       // + dphi*dt
            // -damp*ph*dt  (approx Euler)
            ph = _mm256_sub_pd(ph, _mm256_mul_pd(_mm256_mul_pd(dm, ph), vdt));

            double out[4]; _mm256_storeu_pd(out, ph);
            for(int k=0;k<4;++k){
                double p = out[k];
                // noise scalar
                if (osc.ent[i+k]>0.0) p += rng.gauss()*osc.ent[i+k]*std::sqrt(dt);
                if (p >= PI || p < -PI){
                    p = std::fmod(p + PI, TAU); if (p < 0) p += TAU; p -= PI;
                }
                osc.phase[i+k] = p;
            }
        }
        for(; i<N; ++i){
            double ph = osc.phase[i];
            ph += TAU*osc.freq[i]*dt + dphi[i]*dt - osc.damp[i]*ph*dt;
            if (osc.ent[i]>0.0) ph += rng.gauss()*osc.ent[i]*std::sqrt(dt);
            if (ph >= PI || ph < -PI){ ph = std::fmod(ph + PI, TAU); if (ph < 0) ph += TAU; ph -= PI; }
            osc.phase[i] = ph;
        }
    #else
        (void)dt; (void)dphi;
    #endif
    }

    // Linear integrators
    inline void advance_integrators(double dt, const std::vector<double>& dint){
        for (uint32_t i=0;i<integ.n;++i){
            double v = integ.value[i];
            v += (-integ.decay[i]*v + dint[i]) * dt;
            integ.value[i] = v;
        }
    }

    // Scalar step
    void step_scalar(double dt, bool record_hist=false){
        std::vector<double> dphi(osc.n, 0.0), dint(integ.n, 0.0);
        accumulate_k_scalar(dphi);
        for (auto &e : l_edges){ double A=integ.value[e.a], B=integ.value[e.b]; dint[e.a]+=e.k_ab*B; dint[e.b]+=e.k_ba*A; }
        advance_scalar(dt, dphi);
        advance_integrators(dt, dint);
        time += dt;
        if (record_hist && !hist.indices.empty()) hist.push(time, osc.phase);
    }

    // Jobified step (optionally AVX2 accumulation + AVX2 advance)
    void step_jobified(double dt, int threads=std::thread::hardware_concurrency(), bool use_avx2=false, bool record_hist=false){
        std::vector<double> dphi(osc.n, 0.0), dint(integ.n, 0.0);
        if (use_avx2) accumulate_k_avx2(dphi);
        else {
            auto work_k = [&](int tid){
                size_t N = k_edges.size(), chunk = (N+threads-1)/threads;
                size_t s = tid*chunk, e = std::min(N, s+chunk);
                std::vector<double> local(osc.n,0.0);
                for(size_t i=s;i<e;++i){
                    auto &ed = k_edges[i];
                    double err = std::fmod(ed.p*osc.phase[ed.a] - ed.q*osc.phase[ed.b] + PI, TAU); if (err<0) err+=TAU; err -= PI;
                    double da  = - ed.Xi * std::sin(err);
                    local[ed.a] += da;
                    local[ed.b] += -da * (double(ed.p)/double(ed.q?ed.q:1));
                }
                for(uint32_t i=0;i<osc.n;++i) if (local[i]!=0.0) __atomic_add_fetch(&dphi[i], local[i], __ATOMIC_RELAXED);
            };
            std::vector<std::thread> pool;
            for (int t=0;t<threads;++t) pool.emplace_back(work_k, t);
            for (auto& th: pool) th.join();
        }
        for (auto &e : l_edges){ double A=integ.value[e.a], B=integ.value[e.b]; dint[e.a]+=e.k_ab*B; dint[e.b]+=e.k_ba*A; }

        if (use_avx2) advance_avx2(dt, dphi); else advance_scalar(dt, dphi);
        advance_integrators(dt, dint);
        time += dt;
        if (record_hist && !hist.indices.empty()) hist.push(time, osc.phase);
    }

    void run(double T, double dt, bool jobified=false, int threads=0, bool use_avx2=false, bool record_hist=false){
        const int steps = int(T/dt + 1e-9);
        if (!jobified){
            for (int i=0;i<steps;++i) step_scalar(dt, record_hist);
        } else {
            if (threads<=0) threads = std::thread::hardware_concurrency();
            for (int i=0;i<steps;++i) step_jobified(dt, threads, use_avx2, record_hist);
        }
    }

    double phase_error(uint32_t ai,uint32_t bi,int p=1,int q=1) const {
        double pa = osc.phase[ai], pb = osc.phase[bi];
        return std::fmod(p*pa - q*pb + PI, TAU) - PI;
    }
};

} // namespace rlangf
