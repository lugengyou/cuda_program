/*
 * @Descripttion: 
 * @version: 
 * @Author: ***
 * @Date: 2024-08-10 16:30:53
 * @LastEditors: gengyou.lu
 * @LastEditTime: 2024-08-10 21:38:47
 */
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
// #include <cmath.h>

float error_range = 1e-8;

bool compare_result(const float *h_res, const float *d_res, uint32_t size) {

    for (uint32_t i = 0; i < size; ++i) {
        if (abs(h_res[i] - d_res[i]) > error_range) {
            printf("result compare false. h_res[%d]:%.8f, d_res[%d]:%.8f.\n", i, h_res[i], i, d_res[i]);
            return false;
        }
    }

    printf("result compare true.\n");
    return true;
}


void softmax_cpu(const float *h_v, float *h_res, uint32_t size) {

    float max = h_v[0];
    for (uint32_t i = 1; i < size; ++i) {
        if (max < h_v[i])
            max = h_v[i];
    }

    // printf("max: %.8f\n", max);

    float exp_sum = 0.0;
    for (uint32_t i = 0; i < size; ++i) {
        h_res[i] = expf(h_v[i] - max);
        exp_sum += h_res[i];
    }

    // printf("exp sum: %.8f\n", exp_sum);

    for (uint32_t i = 0; i < size; ++i) {
        h_res[i] /= exp_sum;
    }
}

__global__ void softmax_v0(const float *d_v, float *d_res, uint32_t size) {
    
    uint32_t tid = threadIdx.x;
    uint32_t idx = tid + blockDim.x * blockIdx.x;

    if (idx >= size) 
        return ;

    float max = d_v[idx];
    for (uint32_t i = 0; i < size; ++i) {
        if (max < d_v[i])
            max = d_v[i];
    }

    float exp_sum = 0.0;
    for (uint32_t i = 0; i < size; ++i) {
        exp_sum += expf(d_v[i] - max);
    }

    d_res[idx] = expf(d_v[idx] - max) / exp_sum;    
}

int main(int argc, char *argv[]) {

    printf("\n******** cuda ops softmax ********\n");

    // 设置数据大小
    uint32_t size = 1 << 14;

    // 设置线程模型参数
    dim3 block(512);
    dim3 grid((size + block.x - 1) / block.x);
    
    // 生成数据
    uint32_t byte_size = size * sizeof(float);
    float *h_v = (float*)malloc(byte_size);
    float *h_res = (float*)malloc(byte_size);
    float *res_from_device = (float*)malloc(byte_size);

    for (uint32_t i = 0; i < size; ++i) {
        h_v[i] = (float)(rand() & 0xFF) / 10.0;
    }

    // 申请 gpu 内存
    float *d_v = NULL, *d_res = NULL;
    cudaMalloc(&d_v, byte_size);
    cudaMalloc(&d_res, byte_size);
    
    // 内存搬运
    cudaMemcpy(d_v, h_v, byte_size, cudaMemcpyHostToDevice);
    
    softmax_v0<<< grid, block >>>(d_v, d_res, size);
    cudaDeviceSynchronize();

    cudaMemcpy(res_from_device, d_res, byte_size, cudaMemcpyDeviceToHost);

    // cpu 计算
    softmax_cpu(h_v, h_res, size);

    // 结果比较
    compare_result(h_res, res_from_device, size);

    // 释放内存
    free(h_v);
    free(h_res);
    free(res_from_device);
    cudaFree(d_v);
    cudaFree(d_res);

    cudaDeviceReset();

    return 0;
}

