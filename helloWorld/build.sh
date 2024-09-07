#!/bin/bash
###
 # @Author: lugy lugengyou@github.com
 # @Date: 2024-09-07 15:26:15
 # @FilePath: /cuda_program/helloWorld/build.sh
 # @LastEditTime: 2024-09-07 15:28:12
 # @Description: 
### 

mkdir build

cd build
cmake ..
make

./hello_cuda
