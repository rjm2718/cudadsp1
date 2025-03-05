cmake_minimum_required(VERSION 3.30)
project(cudadsp_sample1)

set(CMAKE_CUDA_STANDARD 23)

include_directories(src)

add_executable(cudadsp_sample1
    src/main.cu
    src/rtpdsp.h)

set_target_properties(cudadsp_sample1 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
