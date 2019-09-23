CMAKE_MINIMUM_REQUIRED(VERSION 3.13)
project(ch04)

set(CMAKE_CXX_STANDARD 14)

include_directories(/usr/include/eigen3)

find_package( Sophus REQUIRED )
include_directories( ${Sophus_INCLUDE_DIRS} )

add_executable(useSophus useSophus.cpp)
target_link_libraries( useSophus ${Sophus_LIBRARIES})