cmake_minimum_required(VERSION 3.30)
project(cudadsp1 CUDA)

set(CMAKE_CUDA_STANDARD 23)

add_executable(cudadsp1 src/main.cu)

set_target_properties(cudadsp1 PROPERTIES
                      CUDA_SEPARABLE_COMPILATION ON)
