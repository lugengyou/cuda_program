#!/bin/bash
###
 # @Author: lugy lugengyou@github.com
 # @Date: 2024-08-09 21:55:27
 # @FilePath: /cuda_program/interview_samples/build.sh
 # @LastEditTime: 2024-09-01 11:14:49
 # @Description: 
### 


# Build the project

cd build
cmake ..
make

# ./reduce
./block_reduce



