#include <stdio.h>
#include "../tools/common.cuh"

__device__ float add(float a, float b)
{
    return a + b;
}

__global__ void addFromGPU(float *A, float *B, float *C, int N)
{
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int id = tid + bid * blockDim.x;
       
    if(id >= N) return;
    C[id] = add(A[id], B[id]);
}


void initialData(float *addr, int elemCount)
{
    for (int i = 0; i < elemCount; i++)
    {
        addr[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}

int main(void)
{   
    setGPU();

    int iElemCount = 513;
    size_t iSize = iElemCount * sizeof(float);

    // Allocate host memory
    float *h_A = (float *)malloc(iSize);
    float *h_B = (float *)malloc(iSize);
    float *h_C = (float *)malloc(iSize);

    // Initialize host memory
    if (h_A != NULL && h_B != NULL && h_C != NULL)
    {
        memset(h_A, 0, iSize);
        memset(h_B, 0, iSize);
        memset(h_C, 0, iSize);
    }
    else
    {
        printf("Failed to allocate host memory\n");
        exit(-1);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((float **)&d_A, iSize);
    cudaMalloc((float **)&d_B, iSize);
    cudaMalloc((float **)&d_C, iSize);
    if (d_A != NULL && d_B != NULL && d_C != NULL)
    {
        cudaMemset(d_A, 0, iSize);
        cudaMemset(d_B, 0, iSize);
        cudaMemset(d_C, 0, iSize);
    }
    else
    {
        printf("Failed to allocate device memory\n");
        free(h_A);
        free(h_B);
        free(h_C);
        exit(-1);
    }

    // Initialize host data
    srand(666);
    initialData(h_A, iElemCount);
    initialData(h_B, iElemCount);

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, iSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, iSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, iSize, cudaMemcpyHostToDevice);

    // Invoke kernel
    dim3 block(32);
    // dim3 grid(iElemCount / 32);
    dim3 grid((iElemCount + block.x - 1) / block.x);

    addFromGPU<<<grid, block>>>(d_A, d_B, d_C, iElemCount); // 512 threads
    // cudaDeviceSynchronize();

    // Copy data from device to host
    cudaMemcpy(h_C, d_C, iSize, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 10; i++)
    {
        printf("idx=%2d\tmatrixA=%.2f\tmatrixB=%.2f\tmatrixC=%.2f\n", i, h_A[i], h_B[i], h_C[i]);
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    cudaDeviceReset();
    return 0;
}

