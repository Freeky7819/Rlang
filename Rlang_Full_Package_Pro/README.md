# Rlang — Pro Package (Python + C++)

Extended build with performance and deployment targets:

- ✅ SIMD-accelerated core (AVX2) with scalar fallback
- ✅ Multi-threaded job scheduler (std::thread) + optional OpenMP
- ✅ Optional CUDA backend (GPU)
- ✅ Real-time audio synthesis demo (WAV render)
- ✅ WASM / Emscripten build for browser
- ✅ Profile system for domain heuristics (music, neuro, robotics)
- ✅ Optimizers: Adam (ready) + hooks for CMA‑ES / Levenberg–Marquardt

## Build Matrix

| Target | How |
|-------|-----|
| C++ native | `cmake -S . -B build && cmake --build build -j` |
| AVX2 | `-DRLANG_ENABLE_AVX2=ON` (auto if CPU supports) |
| OpenMP | `-DRLANG_ENABLE_OPENMP=ON` |
| CUDA | `-DRLANG_ENABLE_CUDA=ON` (needs nvcc) |
| WASM | `./scripts/build_wasm.sh` (requires Emscripten) |

## Quick Start

### Python
```bash
python -m rlang_py.cli run-step --profile examples/profile.json --state examples/state.json
```

### C++ (native)
```bash
cmake -S . -B build -DRLANG_ENABLE_AVX2=ON -DRLANG_ENABLE_OPENMP=ON
cmake --build build -j
./build/rlang_cli examples/profile.txt examples/state.txt
```

### CUDA (optional)
```bash
cmake -S . -B build -DRLANG_ENABLE_CUDA=ON
cmake --build build -j
./build/rlang_cuda_cli examples/profile.txt examples/state.txt
```

### WASM
```bash
./scripts/build_wasm.sh
# Serve examples/web to view in browser
```

### Audio Synthesis (WAV)
```bash
python examples/audio_synth.py --profile examples/profile.json --seconds 4 --samplerate 44100 --out out.wav
```

## Project Layout
See inline comments in headers. Scalar fallback is always available.
