#pragma once
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#define CUDA_CHECK(call)           __cudaCheck(call, __FILE__, __LINE__)
#define LAST_KENERL_CHECK(call)    __kernelCheck(call, __FILE__, __LINE__)


static void __cudaCheck(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess) 
    {
        printf("ERROR: %s: %d, ", file, line);
        printf("CODE: %s, DETAIL: %s\n", cudaGetErrorName(err), cudaGetErrorString(err));  
        exit(-1);
    }
}

static void __kernelCheck(const char *file, int line)
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
    {
        printf("ERROR: %s: %d, ", file, line);
        printf("CODE: %s, DETAIL: %s\n", cudaGetErrorName(err), cudaGetErrorString(err));  
        exit(-1);
    }
}