
cmake_minimum_required(VERSION 3.18)
project(rlang_fastdrop_015 LANGUAGES CXX)

option(RLANG_AVX2 "Enable AVX2" ON)
option(RLANG_CUDA "Build CUDA backend" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(rlang_fastdrop INTERFACE)
target_include_directories(rlang_fastdrop INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

if (RLANG_AVX2 AND NOT EMSCRIPTEN)
  if (MSVC)
    target_compile_options(rlang_fastdrop INTERFACE /arch:AVX2)
  else()
    target_compile_options(rlang_fastdrop INTERFACE -mavx2 -O3 -ffast-math)
  endif()
endif()

add_executable(audio_triads examples/audio_triads.cpp)
target_link_libraries(audio_triads PRIVATE rlang_fastdrop)

add_executable(opt_adam examples/opt_adam.cpp)
target_link_libraries(opt_adam PRIVATE rlang_fastdrop)

# GPU wrapper (host only; enable with -DRLANG_CUDA=ON and add CUDA sources)
add_library(rlang_sysgpu STATIC src/rlang_system_gpu.cpp)
target_include_directories(rlang_sysgpu PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(rlang_sysgpu PUBLIC rlang_fastdrop)

if (RLANG_CUDA)
  enable_language(CUDA)
  add_library(rlang_cuda OBJECT cuda/rlang_kuramoto_cuda.cu src/rlang_cuda_host.cpp)
  target_include_directories(rlang_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_compile_definitions(rlang_sysgpu PUBLIC RLANG_CUDA_ENABLED=1)
endif()

# WASM example (emscripten builds this normally)
add_executable(wasm_demo examples/wasm_demo.cpp)
target_link_libraries(wasm_demo PRIVATE rlang_fastdrop)
