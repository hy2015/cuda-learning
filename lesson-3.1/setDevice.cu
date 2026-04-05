#include <stdio.h>

int main(void) {

    // Get the number of CUDA devices
    int iDeviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&iDeviceCount);

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

    return 0;
}