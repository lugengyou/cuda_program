#include <stdio.h>

__global__ void helloFromGPU(void) {
    int i = threadIdx.x;
    printf("Hello world from GPU by thread %d.\n", i);
}

int main(void) {

    printf("Hello world from CPU.\n\n");

    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("Device %d: \"%s\"\n", dev, deviceProp.name);

    // 共享内存存储体默认大小
    cudaSharedMemConfig pConfig;
    cudaDeviceGetSharedMemConfig(&pConfig);
    printf("Shared memory configuration: %d\n", pConfig);

    // helloFromGPU<<<1, 10>>>();
    
    cudaDeviceSynchronize();

    cudaDeviceReset();

    return 0;
}


