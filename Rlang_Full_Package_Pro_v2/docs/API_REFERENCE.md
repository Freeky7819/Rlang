# API REFERENCE

## Python
- rlang_py.interpreter.step(state, profile) -> dict
- rlang_py.cli (run-step)
- rlang_py.parser.parse_rlang(path) -> dict
- rlang_py.profiles: music/neuro/robotics/ai_alignment
- rlang_py.optimizers.adam.adam_update(...)

## C++
- step_scalar, step_simd_avx2, parallel_for
- CLI: rlang_cli <profile.txt> <state.txt> → JSON-like out
