#include <cuda_runtime.h>
#include <stdio.h>
#include "common.cuh"

__global__ void kernel(void)
{

}

int main(int argc, char **argv)
{
    int devID = 0;
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, devID));
    printf("Using Device %d: %s\n", devID, deviceProp.name);

    if (deviceProp.globalL1CacheSupported) {
        printf("GPU supports L1 cache\n");
        // printf("L1 cache size: %d KB\n", deviceProp.l1CacheSize / 1024);
    } else {
        printf("GPU does not support L1 cache\n");
    }
    printf("L2 cache size: %d MB\n", deviceProp.l2CacheSize / 1024 / 1024);

    dim3 block(1);
    dim3 grid(1);

    kernel<<<grid, block>>>();
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaDeviceReset());

    return 0;
}