# Tutorial — From Zero to Resonance
1) Python reference:
   python -m rlang_py.cli run-step --profile examples/profile.json --state examples/state.json
2) C++ build:
   cmake -S . -B build -DRLANG_ENABLE_AVX2=ON && cmake --build build -j
3) Golden test:
   python tests/run_golden.py
4) AI drift demo:
   python examples/ai_agent_drift_demo.py
5) DSL:
   python -c "from rlang_py.parser import parse_rlang; print(parse_rlang('examples/dsl/example_agents_v0_4.rlang'))"
