project(ch06)
cmake_minimum_required(VERSION 3.13)

set(CMAKE_BUILD_TYPE release)

message(${PROJECT_SOURCE_DIR})
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_DIRS})

find_package(G2O REQUIRED)
include_directories(${G2O_INCLUDE_DIRS} /usr/include/eigen3)

add_executable(ceres_fitting ceres_fitting.cpp)
add_executable(g2o_fitting g2o_fitting.cpp)

target_link_libraries(ceres_fitting ${CERES_LIBRARIES} ${OpenCV_LIBS})
target_link_libraries(g2o_fitting ${OpenCV_LIBS} g2o_core g2o_stuff)