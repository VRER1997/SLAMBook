CMAKE_MINIMUM_REQUIRED(VERSION 3.13)

set(CMAKE_CXX_STANDARD 14)
#include_directories(/usr/local/Cellar/eigen/3.3.7/include/eigen3/)
include_directories(/usr/include/eigen3)

add_executable(eigenMatrix eigenMatrix.cpp)
add_executable(useGeometry useGeometry.cpp)