#include <stdio.h>
#include "../tools/common.cuh"

int getSPcores(cudaDeviceProp devProp)
{
    int cores = 0;
    int mp = devProp.multiProcessorCount;
    switch (devProp.major)
    {
        case 2: // Fermi
            if (devProp.minor == 1) cores = mp * 48;
            else cores = mp * 32;
            break;
        case 3: // Kepler
            cores = mp * 192;
            break;
        case 5: // Maxwell
            cores = mp * 128;
            break;
        case 6: // Pascal
            if (devProp.minor == 1 || devProp.minor == 2) cores = mp * 128;
            else if (devProp.minor == 0) cores = mp * 64;
            break;
        case 7: // Volta and Turing
            if (devProp.minor == 0 || devProp.minor == 5) cores = mp * 64;
            break;
        case 8: // Ampere and Ada
            if (devProp.minor == 0) cores = mp * 64; // Ampere
            else if (devProp.minor == 6) cores = mp * 128; // Ada
            break;
        case 9: // Hopper
            if (devProp.minor == 0) cores = mp * 128;
            break;
        case 10: // Lovelace
            if (devProp.minor == 0) cores = mp * 128;
            break;
        case 11: // Grace
            if (devProp.minor == 0) cores = mp * 128;
            break;
        case 12: // Orin
            if (devProp.minor == 0) cores = mp * 128;
            break;
        case 13: // Blackwell
            if (devProp.minor == 0) cores = mp * 128;
            break;
        case 14: // Hopper Next
            if (devProp.minor == 0) cores = mp * 128;
            break;
        default:
            printf("Unknown device type\n");
            break;
    }
    return cores;
}

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
    printf("Number of SM:                                                        %d\n", 
        deviceProp.multiProcessorCount);
    printf("Number of CUDA cores:                                                %d\n", 
        getSPcores(deviceProp));
    return 0;
}