cmake_minimum_required(VERSION 3.2)
project(Tensor-OpenCV)

SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

include(ExternalProject)
include(mshadow)

if (DEFINED SION_PROJECT_BINARY_DIR)
	SET(CMAKE_INSTALL_PREFIX "${SION_PROJECT_BINARY_DIR}")
else()
	SET(SION_PROJECT_BINARY_DIR "${CMAKE_BINARY_DIR}")
endif()

SET(EP_PREFIX "${SION_PROJECT_BINARY_DIR}")

ExternalProject_Add(
	Tensor
	CMAKE_ARGS "-DSION_PROJECT_BINARY_DIR=${SION_PROJECT_BINARY_DIR}"
	GIT_REPOSITORY https://github.com/SionProject/Tensor.git
	PREFIX "${SION_PROJECT_BINARY_DIR}"
)

include_directories("${CMAKE_BINARY_DIR}/include")

add_executable(imshow src/imshow.cpp)

