# Rlang — Resonant Language (Pro v2, 2025-10-08)

**Goal:** A complete, runnable package showing Rlang (Python + C++) with theory, AI integration, DSL, tests, SIMD/CUDA stubs, WASM script, and docs for public repo use.

## Highlights
- Python reference + C++ port (scalar + AVX2 path, thread scheduler)
- Optional CUDA backend (stub provided), WASM script (Emscripten)
- Real-time audio example (WAV)
- AI agent drift demo + PyTorch/Gym integration stubs
- DSL parser + examples
- Theory doc (derivation + related work) and API reference
- Golden tests (Python vs C++) and CI workflow
- RHL-1.0 license

## Quick Start
### Python
```bash
python -m rlang_py.cli run-step --profile examples/profile.json --state examples/state.json
```
### C++
```bash
cmake -S . -B build -DRLANG_ENABLE_AVX2=ON
cmake --build build -j
./build/rlang_cli examples/profile.txt examples/state.txt
```
### Cross-check
```bash
python tests/run_golden.py
```
### Audio demo
```bash
python examples/audio_synth.py --profile examples/profile.json --seconds 3 --out out.wav
```
### AI drift demo
```bash
python examples/ai_agent_drift_demo.py
```
