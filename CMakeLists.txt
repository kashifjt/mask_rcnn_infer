cmake_minimum_required(VERSION 2.8.12)

project(demo)

find_package(OpenCV REQUIRED)
find_package(TensorflowCC REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(include)
#include_directories(src)

add_executable(demo src/demo.cpp src/MaskInfer.cpp)
target_link_libraries(demo ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} TensorflowCC::Shared)
find_package(CUDA)
if(CUDA_FOUND)
  target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})
endif()
