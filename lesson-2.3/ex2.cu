#include <stdio.h>

__global__ void hello_from_gpu() {
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int gid = bid * blockDim.x + tid;

    printf("Hello from GPU block %d and thread %d, global id %d\n", bid, tid, gid);
}

int main() {
    
    // Launch the kernel with 1 block and 10 threads
    hello_from_gpu<<<2, 4>>>();
    
    // Wait for the GPU to finish
    cudaDeviceSynchronize();
    
    return 0;
}