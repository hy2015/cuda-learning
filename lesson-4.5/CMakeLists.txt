cmake_minimum_required(VERSION 3.20)

project(shared_memory LANGUAGES CXX CUDA)
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

add_executable(static_shared_memory static_shared_memory.cu common.cuh)
target_include_directories(static_shared_memory PRIVATE ${CUDA_INCLUDE_DIRS})
target_link_libraries(static_shared_memory PRIVATE ${CUDA_LIBRARIES})

add_executable(dynamic_shared_memory dynamic_shared_memory.cu common.cuh)
target_include_directories(dynamic_shared_memory PRIVATE ${CUDA_INCLUDE_DIRS})
target_link_libraries(dynamic_shared_memory PRIVATE ${CUDA_LIBRARIES})