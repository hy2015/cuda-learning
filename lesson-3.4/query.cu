#include "../tools/common.cuh"
#include <stdio.h>

int main(void)
{   
    int device_id = 0;
    ErrorCheck(cudaSetDevice(device_id), __FILE__, __LINE__);

    cudaDeviceProp deviceProp;
    ErrorCheck(cudaGetDeviceProperties(&deviceProp, device_id), __FILE__, __LINE__);
    printf("Device id:                                                           %d\n", 
        device_id);
    printf("Device name:                                                         %s\n", 
        deviceProp.name);
    printf("Amount of global memory:                                             %.2f GB\n", 
        (float)deviceProp.totalGlobalMem / (1024 * 1024 * 1024));
    printf("Amount of const memory:                                              %.2f KB\n", 
        (float)deviceProp.totalConstMem / 1024);
    printf("Maximum grid size:                                                   %d x %d x %d\n", 
        deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
    printf("Maximum block size:                                                  %d x %d x %d\n", 
        deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
    printf("Number of SM:                                                        %d\n", 
        deviceProp.multiProcessorCount);
    printf("Maximum amount of shared memory per block:                           %.2f KB\n", 
        (float)deviceProp.sharedMemPerBlock / 1024);
    printf("Maximum amount of shared memory per SM:                              %.2f KB\n", 
        (float)deviceProp.sharedMemPerMultiprocessor / 1024);
    printf("Maximum number of registers per block:                               %d\n", 
        deviceProp.regsPerBlock);
    printf("Maximum number of registers per SM:                                  %d\n", 
        deviceProp.regsPerMultiprocessor);
    printf("Maximum number of threads per block:                                 %d\n", 
        deviceProp.maxThreadsPerBlock);
    printf("Maximum number of threads per SM:                                    %d\n", 
        deviceProp.maxThreadsPerMultiProcessor);
    return 0;
}