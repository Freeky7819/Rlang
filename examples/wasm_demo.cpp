// examples/wasm_demo.cpp
#include "rlang_fast_simd.hpp"
#include <cstdio>
extern "C" {
    // Minimal C ABI exports for JS glue
    void* rlang_new(){ return new rlangf::System(); }
    void  rlang_free(void* p){ delete (rlangf::System*)p; }
    int   rlang_add_osc(void* p, double f,double ph){ return ((rlangf::System*)p)->add_osc(f,ph,0,0,1); }
    void  rlang_couple(void* p, int a,int b,double Xi,int P,int Q){ ((rlangf::System*)p)->couple_k(a,b,Xi,P,Q); }
    void  rlang_run(void* p, double T,double dt){ ((rlangf::System*)p)->run(T,dt,false); }
    double rlang_phase(void* p, int i){ return ((rlangf::System*)p)->osc.phase[i]; }
}
int main(){ std::puts("WASM demo stub"); return 0; }
