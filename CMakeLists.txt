cmake_minimum_required(VERSION 3.2)
project(Tensor-OpenCV)

include(ExternalProject)

ExternalProject_Add(
	Tensor
	CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}"
	GIT_REPOSITORY https://github.com/SionProject/Tensor.git
)

include_directories("${CMAKE_BINARY_DIR}/include")

add_executable(imshow src/imshow.cpp)

