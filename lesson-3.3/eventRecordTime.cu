#include <stdio.h>
#include "../tools/common.cuh"

#define NUM_REPEATS 10

__device__ float add(float a, float b)
{
    return a + b;
}

__global__ void addFromGPU(float *A, float *B, float *C, int N)
{
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int id = tid + bid * blockDim.x;
        
    if (id >= N) return;
    C[id] = add(A[id], B[id]);
}

void initialData(float *addr, int elemCount)
{
    for (int i = 0; i < elemCount; i++)
    {
        addr[i] = (float)(rand() & 0xFF) / 10.0f;
    }
    return;
}

int main(void)
{   
    setGPU();

    int iElemCount = 4096   ;
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
    ErrorCheck(cudaMalloc((float **)&d_A, iSize), __FILE__, __LINE__);
    ErrorCheck(cudaMalloc((float **)&d_B, iSize), __FILE__, __LINE__);
    ErrorCheck(cudaMalloc((float **)&d_C, iSize), __FILE__, __LINE__);
    if (d_A != NULL && d_B != NULL && d_C != NULL)
    {
        ErrorCheck(cudaMemset(d_A, 0, iSize), __FILE__, __LINE__);
        ErrorCheck(cudaMemset(d_B, 0, iSize), __FILE__, __LINE__);
        ErrorCheck(cudaMemset(d_C, 0, iSize), __FILE__, __LINE__);
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
    ErrorCheck(cudaMemcpy(d_A, h_A, iSize, cudaMemcpyHostToDevice), __FILE__, __LINE__);
    ErrorCheck(cudaMemcpy(d_B, h_B, iSize, cudaMemcpyHostToDevice), __FILE__, __LINE__);
    ErrorCheck(cudaMemcpy(d_C, h_C, iSize, cudaMemcpyHostToDevice), __FILE__, __LINE__);

    // Invoke kernel
    dim3 block(32);
    dim3 grid((iElemCount + block.x - 1) / 32);

    float t_sum = 0.0f;
    for (int i = 0; i < NUM_REPEATS; i++)
    {        
        cudaEvent_t start, stop;
        ErrorCheck(cudaEventCreate(&start), __FILE__, __LINE__);
        ErrorCheck(cudaEventCreate(&stop), __FILE__, __LINE__);
        ErrorCheck(cudaEventRecord(start, 0), __FILE__, __LINE__);
        cudaEventQuery(start); // Ensure the event is recorded before starting the kernel execution

        ErrorCheck(cudaEventRecord(stop, 0), __FILE__, __LINE__);
        ErrorCheck(cudaEventSynchronize(stop), __FILE__, __LINE__);
        float elapsedTime;
        ErrorCheck(cudaEventElapsedTime(&elapsedTime, start, stop), __FILE__, __LINE__);
       
        if (i > 0) // Skip the first iteration for warm-up
        {
            t_sum += elapsedTime;
        }
        
        ErrorCheck(cudaEventDestroy(start), __FILE__, __LINE__);
        ErrorCheck(cudaEventDestroy(stop), __FILE__, __LINE__);
    }


    const float t_avg = t_sum / (NUM_REPEATS - 1);
    printf("Average kernel execution time over %d runs: %g ms\n", NUM_REPEATS - 1, t_avg);

    ErrorCheck(cudaMemcpy(h_C, d_C, iSize, cudaMemcpyDeviceToHost), __FILE__, __LINE__);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);
    // Free device memory
    ErrorCheck(cudaFree(d_A), __FILE__, __LINE__);
    ErrorCheck(cudaFree(d_B), __FILE__, __LINE__);
    ErrorCheck(cudaFree(d_C), __FILE__, __LINE__);


    ErrorCheck(cudaDeviceReset(), __FILE__, __LINE__);
    return 0;
}

