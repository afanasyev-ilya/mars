cmake_minimum_required(VERSION 3.19)
project(mars)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp -fpermissive")

add_executable(mars main.cpp)