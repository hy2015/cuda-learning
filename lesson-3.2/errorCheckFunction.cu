#include <stdio.h>
#include "../tools/common.cuh"

int main(void)
{
    // Allocate host memory
    float * h_A;
    h_A = (float *)malloc(4);
    memset(h_A, 0, 4);

     // Allocate device memory
    float * d_A;
    cudaError_t err = ErrorCheck(cudaMalloc((float **)&d_A, 4), __FILE__, __LINE__);
    cudaMemset(d_A, 0, 4); // This will cause an error because d_A is not allocated successfully

    // Check for errors
    ErrorCheck(cudaMemcpy(d_A, h_A, 4, cudaMemcpyDeviceToHost), __FILE__, __LINE__); // This will also cause an error because d_A is not allocated successfully


    free(h_A);
    ErrorCheck(cudaFree(d_A), __FILE__, __LINE__); // This will also cause an error because d_A is not allocated successfully
    ErrorCheck(cudaDeviceReset(), __FILE__, __LINE__); // This will also cause an error because the device is not properly reset

    return 0;
}