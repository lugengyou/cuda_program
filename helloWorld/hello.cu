#include <stdio.h>

__global__ void helloFromGPU(void) {
    int i = threadIdx.x;
    printf("Hello world from GPU by thread %d.\n", i);
}

int main(void) {

    printf("Hello world from CPU.\n\n");

    helloFromGPU<<<1, 10>>>();
    // cudaDeviceReset();
    cudaDeviceSynchronize();

    return 0;
}


