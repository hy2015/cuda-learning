cmake_minimum_required(VERSION 3.20)

project(global_memory LANGUAGES CXX CUDA)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)


find_package(CUDA REQUIRED)
if (CUDA_FOUND)
    message(STATUS "CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
    message(STATUS "CUDA_LIBRARIES: ${CUDA_LIBRARIES}")
    message(STATUS "CUDA_LIBRARY_DIRS: ${CUDA_LIBRARY_DIRS}")
else()
    message(FATAL_ERROR "CUDA not found")
endif()

add_executable(global_memory global_memory.cu common.cuh)
target_include_directories(global_memory PRIVATE ${CUDA_INCLUDE_DIRS})
target_link_libraries(global_memory PRIVATE ${CUDA_LIBRARIES})