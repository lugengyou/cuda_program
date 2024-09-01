cmake_minimum_required(VERSION 3.0)

project(cuda_sample)

find_package(CUDA REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})
# message(STATUS "CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
# include_directories("/usr/local/cuda-11.7/include")

link_libraries(${CUDA_LIBRARIES})
# message(STATUS "CUDA_LIBRARIES: ${CUDA_LIBRARIES}")
# link_libraries("/usr/local/cuda-11.7/lib64")

cuda_add_executable(reduce src/reduce/reduce.cu)
cuda_add_executable(block_reduce src/reduce/block_reduce.cu)
cuda_add_executable(softmax src/softmax/softmax.cu)

