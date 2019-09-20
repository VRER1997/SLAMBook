CMAKE_MINIMUM_REQUIRED(VERSION 3.13)
project(ch04)

set(CMAKE_CXX_STANDARD 14)

include_directories(/usr/local/include/eigen3)

add_executable(useSophus useSophus.cpp)