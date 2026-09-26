# A HIP project as it would be written for a real GPU: CMake's own HIP
# language support, and a test that runs the program directly.
cmake_minimum_required(VERSION 3.21)
project(vectoradd LANGUAGES CXX HIP)
enable_testing()
add_executable(vectoradd vectoradd.hip)
add_test(NAME vectoradd COMMAND vectoradd)
