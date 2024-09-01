/*
 * @Author: lugy lugengyou@github.com
 * @Date: 2024-09-01 10:21:10
 * @FilePath: /cuda_program/interview_samples/src/reduce/block_reduce.cu
 * @LastEditTime: 2024-09-01 11:11:18
 * @Description: cuda 向量和/乘 block 规约
 */

#include <stdio.h>
#include <stdint.h>
#include "cuda_runtime.h"

__global__ void vec_sum_reduce(float* input, double* res, uint32_t N) {

    extern __shared__ double block_sum[];

    const uint32_t tidx = threadIdx.x;
    uint32_t i = threadIdx.x;

    // block 间规约
    block_sum[tidx] = 0.0;
    for ( ; i < N; i += blockDim.x) {
        block_sum[tidx] += input[i];
    }

    __syncthreads();

    // block 内规约
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tidx < stride) {
            block_sum[tidx] += block_sum[tidx + stride];
        }

        __syncthreads();
    }

    *res = block_sum[0];
}


int main() {

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device name: %s\n", prop.name);
    printf("Shared memory per block: %ld KB\n", prop.sharedMemPerBlock / 1024);

    const uint32_t N = 1024;
    float* d_input;
    double* d_res;
    float* h_input = (float*)malloc(N * sizeof(float));
    double h_res;

    for (uint32_t i = 0; i < N; ++i) {
        h_input[i] = 1.0;
    }

    cudaMalloc(&d_input, N * sizeof(float));
    cudaMalloc(&d_res, sizeof(double));

    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    vec_sum_reduce<<<1, 128, 128>>>(d_input, d_res, N);

    cudaMemcpy(&h_res, d_res, sizeof(double), cudaMemcpyDeviceToHost);

    printf("res: %lf\n", h_res);

    free(h_input);
    cudaFree(d_input);
    cudaFree(d_res);

    cudaDeviceReset();

    return ;
}



