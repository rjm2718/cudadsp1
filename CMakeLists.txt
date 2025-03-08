cmake_minimum_required(VERSION 3.30)
project(cudadsp1 CUDA)

set(CMAKE_CUDA_STANDARD 23)

set(CMAKE_CUDA_ARCHITECTURES 75)


add_executable(cudadsp1 src/main.cu
        src/dsp.cu)

set_target_properties(cudadsp1 PROPERTIES
                      CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(cudadsp1 PRIVATE /home/ryan/nvidia-mathdx-25.01.1/nvidia/mathdx/25.01/include
        /home/ryan/nvidia-mathdx-25.01.1/nvidia/mathdx/25.01/external/cutlass/include)
