#!/usr/bin/env bash
set -e
emcmake cmake -S . -B build-wasm -DCMAKE_BUILD_TYPE=Release
emmake cmake --build build-wasm -j
cp build-wasm/rlang_cli examples/web/rlang_cli.wasm 2>/dev/null || true
echo 'Built WASM to examples/web/rlang_cli.wasm'
