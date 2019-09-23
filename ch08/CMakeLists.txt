project(ch08)
cmake_minimum_required(VERSION 3.13)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(useLK useLK.cpp)

target_link_libraries(useLK ${OpenCV_LIBS})