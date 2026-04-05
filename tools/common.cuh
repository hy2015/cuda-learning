#pragma once
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

cudaError_t ErrorCheck(cudaError_t error_code, const char *file, int line)
{
    if (error_code != cudaSuccess) 
    {
        printf("CUDA error:\r\ncode=%d, name=%s, description=%s\r\nfile=%s, line=%d\r\n", error_code, cudaGetErrorName(error_code), cudaGetErrorString(error_code), file, line);
        return error_code;
    }
    return error_code;
}

void setGPU()
{
    // Get the number of CUDA devices
    int iDeviceCount = 0;
    cudaError_t err = ErrorCheck(cudaGetDeviceCount(&iDeviceCount), __FILE__, __LINE__);

    if (err != cudaSuccess || iDeviceCount == 0) 
    {
        printf("Error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
    else 
    {
        printf("Number of CUDA devices: %d\n", iDeviceCount);
    }

    // Set the CUDA device to use
    int iDev = 0;
    err = cudaSetDevice(iDev);
    if (err != cudaSuccess) 
    {
        printf("failed to set device %d: %s\n", iDev, cudaGetErrorString(err));
        exit(-1);
    }
    else 
    {
        printf("set device %d successfully\n", iDev);
    }
}
