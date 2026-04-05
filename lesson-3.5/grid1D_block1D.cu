/*
 * grid2D_block2D.cu
 
*/

#include <stdio.h>
#include "../tools/common.cuh"

// Matrix addition: C = A + B
__global__ void addMatrix(int *A, int *B, int *C, const int nx, const int ny)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (ix < nx)
    {
        for (int iy = 0; iy < ny; iy++)
        {
            int idx = iy * nx + ix;
            C[idx] = A[idx] + B[idx];
        }
    }
        
}

int main(void)
{
    // set up device
    setGPU();

    // set up data size
    int nx = 16;
    int ny = 8;
    int nxy = nx * ny;
    size_t nBytes = nxy * sizeof(int);

    // allocate host memory
    int *h_A, *h_B, *h_C;
    h_A = (int *)malloc(nBytes);
    h_B = (int *)malloc(nBytes);
    h_C = (int *)malloc(nBytes);

    if(h_A != NULL && h_B != NULL && h_C != NULL)
    {
        for (int i = 0; i < nxy; i++)
        {
            h_A[i] = i;
            h_B[i] = i + 1;
        }
        memset(h_C, 0, nBytes);
    }
    else
    {
        printf("Failed to allocate host memory\n");
        exit(-1);
    }

    // allocate device memory
    int *d_A, *d_B, *d_C;
    ErrorCheck(cudaMalloc((int **)&d_A, nBytes), __FILE__, __LINE__);
    ErrorCheck(cudaMalloc((int **)&d_B, nBytes), __FILE__, __LINE__);
    ErrorCheck(cudaMalloc((int **)&d_C, nBytes), __FILE__, __LINE__);

    if(d_A != NULL && d_B != NULL && d_C != NULL)
    {
        // transfer data from host to device
        ErrorCheck(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice), __FILE__, __LINE__);
        ErrorCheck(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice), __FILE__, __LINE__);
        ErrorCheck(cudaMemcpy(d_C, h_C, nBytes, cudaMemcpyHostToDevice), __FILE__, __LINE__);
    }
    else
    {
        printf("Failed to allocate device memory\n");
        free(h_A);
        free(h_B);
        free(h_C);
        exit(-1);
    }

    dim3 block(4, 1);
    dim3 grid((nx + block.x - 1) / block.x, 1);
    printf("Thread config:grid:<%d, %d>, block:<%d, %d>\n", grid.x, grid.y, block.x, block.y);

    addMatrix<<<grid, block>>>(d_A, d_B, d_C, nx, ny);

    ErrorCheck(cudaMemcpy(h_C, d_C, nBytes, cudaMemcpyDeviceToHost), __FILE__, __LINE__);
    for (int i = 0; i < nxy; i++)
    {
        printf("id=%d, matrix_A=%d, matrix_B=%d, matrix_C=%d\n", i, h_A[i], h_B[i], h_C[i]);
    }
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}