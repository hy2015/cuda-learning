#include <cuda_runtime.h>
#include <stdio.h>
#include "common.cuh"

__constant__ float c_data;
__constant__ float c_data2 = 6.6f;

__global__ void kernel_1()
{
    printf("kernel_1: c_data = %f,\n", c_data);
}

int main(int argc, char **argv)
{
    int devID = 0;
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, devID));
    printf("Using Device %d: %s\n", devID, deviceProp.name);

    float h_data = 3.3f;
    CUDA_CHECK(cudaMemcpyToSymbol(c_data, &h_data, sizeof(float)));

    dim3 block(1);
    dim3 grid(1);

    kernel_1<<<grid, block>>>();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpyFromSymbol(&h_data, c_data2, sizeof(float)));
    printf("constant data h_data = %.2f\n", h_data);

    CUDA_CHECK(cudaDeviceReset());
}