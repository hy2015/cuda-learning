cmake_minimum_required(VERSION 3.20)

project(GPU_cache LANGUAGES CXX CUDA)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)


find_package(CUDA 13.2 REQUIRED)
if (CUDA_FOUND)
    message(STATUS "CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
    message(STATUS "CUDA_LIBRARIES: ${CUDA_LIBRARIES}")
    message(STATUS "CUDA_LIBRARY_DIRS: ${CUDA_LIBRARY_DIRS}")
else()
    message(FATAL_ERROR "CUDA not found")
endif()

add_executable(GPU_cache GPU_cache.cu common.cuh)
target_include_directories(GPU_cache PRIVATE ${CUDA_INCLUDE_DIRS})
target_link_libraries(GPU_cache PRIVATE ${CUDA_LIBRARIES})
