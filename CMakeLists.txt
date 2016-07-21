cmake_minimum_required(VERSION 3.2)
project(Tensor-OpenCV)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
include(GitExternal)

add_subdirectory("${CMAKE_SOURCE_DIR}/external/Tensor" "${CMAKE_CURRENT_BINARY_DIR}/external/Tensor")
include("SionProject.cmake")

find_package(OpenCV REQUIRED)

add_library(sion_tensor_opencv src/Exception.cpp)
target_link_libraries(sion_tensor_opencv ${OpenCV_LIBS})

add_executable(rgb2gray bin/rgb2gray.cpp bin/tinyfiledialogs.c)
target_link_libraries(rgb2gray sion_tensor_opencv)