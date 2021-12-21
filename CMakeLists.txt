cmake_minimum_required(VERSION 3.20)
project(OCLTPTut)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCL REQUIRED)

add_executable(OCLTPTut main.cpp)

target_link_libraries(OCLTPTut OpenCL)
