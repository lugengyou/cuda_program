/*
 * @Descripttion: 
 * @version: 
 * @Author: ***
 * @Date: 2024-08-09 21:35:23
 * @LastEditors: gengyou.lu
 * @LastEditTime: 2024-08-10 21:35:56
 */
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>


__global__ void reduce(int *vec, int *block_sum, int n)
{
    uint32_t tid = threadIdx.x;
    uint32_t idx = blockIdx.x * blockDim.x + tid;

    if (idx >= n)
    {
        return ;
    }

    // block address of start
    int *block_start = vec + blockIdx.x * blockDim.x;

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            block_start[tid] += block_start[tid + stride];
        }
        
        // Wait for all threads to finish
        __syncthreads(); 
    }

    if (tid == 0) {
        block_sum[blockIdx.x] = block_start[0];
    }
}


int main(int argc , char** argv) {

    printf("\n******** cuda sample reduce ********\n");

    // 设置数据大小
    int size = 1 << 24;
    
    // 设置线程模型参数
    dim3 block(512);
    dim3 grid((size + block.x - 1) / block.x);
    
    int byte_size = size * sizeof(int);
    int num_block = grid.x;
    int block_size = num_block * sizeof(int);

    // 设备内存申请
    int *h_vec = (int*)malloc(byte_size);
    int *block_sum = (int*)malloc(block_size);

    for (int i = 0; i < size; ++i) {
        h_vec[i] = (int)rand() & 0xff;
    }

    // 设备内存申请
    int *d_vec = NULL, *d_block_sum = NULL;
    cudaMalloc(&d_vec, byte_size);
    cudaMalloc(&d_block_sum, block_size);

    // 数据拷贝
    cudaMemcpy(d_vec, h_vec, byte_size, cudaMemcpyHostToDevice);

    // 调用核函数
    reduce<<<grid, block>>>(d_vec, d_block_sum, size);

    cudaDeviceSynchronize(); // Wait for GPU to finish

    // 数据拷贝
    cudaMemcpy(block_sum, d_block_sum, block_size, cudaMemcpyDeviceToHost);

    // 局部求和
    int sum = 0;
    for (size_t i = 0; i < num_block; i++)
    {
        sum += block_sum[i];
    }
    printf("Cuda Sum: %d\n", sum);

    // 串行求和
    int sum_cpu = 0;
    for (int i = 0; i < size; ++i) {
        sum_cpu += h_vec[i];
    }
    printf("CPU Sum: %d\n", sum_cpu);

    // 释放内存
    free(h_vec);
    free(block_sum);

    cudaFree(d_vec);
    cudaFree(d_block_sum);

    cudaDeviceReset(); // Reset the GPU

    return 0;
}

