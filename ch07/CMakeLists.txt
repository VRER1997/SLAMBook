project(ch07)
cmake_minimum_required(VERSION 3.13)
set(CMAKE_BUILD_TYPE debug)
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} /usr/include/eigen3)

add_executable(feature_extraction feature_extraction.cpp)
add_executable(pose_estimation_2d2d pose_estimation_2d2d.cpp)
add_executable(triangulation triangulation.cpp)
add_executable(pose_estimation_3d2d pose_estimation_3d2d.cpp)
add_executable(pose_estimation_3d3d pose_estimation_3d3d.cpp)

target_link_libraries(feature_extraction ${OpenCV_LIBS})
target_link_libraries(pose_estimation_2d2d ${OpenCV_LIBS})
target_link_libraries(triangulation ${OpenCV_LIBS})
target_link_libraries(pose_estimation_3d2d ${OpenCV_LIBS})
target_link_libraries(pose_estimation_3d3d ${OpenCV_LIBS})