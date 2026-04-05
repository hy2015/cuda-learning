#include <cuda_runtime.h>
#include <stdio.h>
#include "common.cuh"

__global__ void kernel_1(float *d_A, const int nElems)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int idx = bid * blockDim.x + tid;
    __shared__ float s_array[32];

    if(idx < nElems) {
        s_array[tid] = d_A[idx];
    }
    __syncthreads();

    if(tid == 0) {
    
        for(int i = 0; i < 32; i++) {
            printf("kernel_1: %f, blockIdx: %d\n", s_array[i], bid);
        }
    }
}


int main(int argc, char **argv)
{
    int devID = 0;
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, devID));
    printf("Using Device %d: %s\n", devID, deviceProp.name);

    int nElems = 64;
    int nBytes = nElems * sizeof(int);

    float *h_A = nullptr;
    h_A = (float *)malloc(nBytes);
    for (int i = 0; i < nElems; i++) {
        h_A[i] = float(i);
    }

    float *d_A = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&d_A, nBytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));

    dim3 block(32);
    dim3 grid(2);

    kernel_1<<<grid, block>>>(d_A, nElems);

    CUDA_CHECK(cudaFree(d_A));
    free(h_A);
    CUDA_CHECK(cudaDeviceReset());
}