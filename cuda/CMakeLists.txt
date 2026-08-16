cmake_minimum_required(VERSION 3.24)
project(cuda_opt LANGUAGES CXX CUDA)

# Reference binary
add_executable(reference_bin common/main.cu)
target_include_directories(reference_bin PRIVATE reference common)
set_target_properties(reference_bin PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 17
    CUDA_RUNTIME_LIBRARY Static
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES native
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin
)
target_compile_options(reference_bin PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo>)

# Optimized binary
add_executable(optimized_bin common/main.cu)
target_include_directories(optimized_bin PRIVATE optimized common)
set_target_properties(optimized_bin PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 17
    CUDA_RUNTIME_LIBRARY Static
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES native
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin
)
target_compile_options(optimized_bin PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo>)
