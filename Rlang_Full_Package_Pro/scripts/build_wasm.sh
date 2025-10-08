#!/usr/bin/env bash
set -e
# Requires Emscripten environment (emcmake/emmake)
emcmake cmake -S . -B build-wasm -DCMAKE_BUILD_TYPE=Release
emmake cmake --build build-wasm -j
# Exported CLI as rlang_cli (WASM)
cp build-wasm/rlang_cli examples/web/rlang_cli.wasm 2>/dev/null || true
echo "Built WASM to examples/web/rlang_cli.wasm (serve examples/web/)"
