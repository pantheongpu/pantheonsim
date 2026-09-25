# A CUDA project as it would be written for a real GPU: CMake's own CUDA
# language support, and a test that runs the program directly.
cmake_minimum_required(VERSION 3.20)
project(vectoradd LANGUAGES CXX CUDA)
set(CMAKE_CUDA_ARCHITECTURES 75)
enable_testing()
add_executable(vectoradd vectoradd.cu)
add_test(NAME vectoradd COMMAND vectoradd)
