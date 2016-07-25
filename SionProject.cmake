IF(NOT DEFINED SION_TENSOR_OPENCV)
SET(SION_TENSOR_OPENCV true)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
include(GitExternal)

git_external(external/Tensor https://github.com/SionProject/Tensor.git master)
git_external(external/ImProc https://github.com/SionProject/ImProc.git master)

add_library(sion_tensor_opencv src/Exception.cpp)
target_link_libraries(sion_tensor_opencv sion_tensor)
target_link_libraries(sion_tensor_opencv ${OpenCV_LIBS})

add_executable(rgb2gray bin/rgb2gray.cpp bin/tinyfiledialogs.c)
target_link_libraries(rgb2gray sion_tensor_opencv)

include("${CMAKE_SOURCE_DIR}/external/Tensor/SionProject.cmake")
include("${CMAKE_SOURCE_DIR}/external/ImProc/SionProject.cmake")

include_directories("${CMAKE_CURRENT_LIST_DIR}/include")
ENDIF()
