project(ch05)

CMAKE_MINIMUM_REQUIRED(VERSION 3.13)
#opencv
find_package(OpenCV REQUIRED)
include_directories(${Opencv_INCLUDE_DIRS})

#eigen3
include_directories(/usr/include/eigen3)

#pcl

find_package(PCL REQUIRED COMPONENT common io)
include_directories(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(imageBasic imageBasic.cpp)
add_executable(joinMap joinMap.cpp)

target_link_libraries(imageBasic ${OpenCV_LIBS})
target_link_libraries(joinMap ${OpenCV_LIBS} ${PCL_LIBRARIES})

