cmake_minimum_required(VERSION 3.0)
###
 # @Author: lugy lugengyou@github.com
 # @Date: 2024-09-07 15:26:15
 # @FilePath: /cuda_program/helloWorld/build.sh
 # @LastEditTime: 2024-09-07 15:27:07
 # @Description: 
### 

project(hello_cuda)

find_package(CUDA REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})
# message(STATUS "CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
# include_directories("/usr/local/cuda-11.7/include")

link_libraries(${CUDA_LIBRARIES})
# message(STATUS "CUDA_LIBRARIES: ${CUDA_LIBRARIES}")
# link_libraries("/usr/local/cuda-11.7/lib64")

cuda_add_executable(hello_cuda hello.cu)


