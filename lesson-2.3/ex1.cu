#include <stdio.h>

__global__ void hello_from_gpu() {
    printf("Hello from GPU thread \n");
}

int main() {
    
    // Launch the kernel with 1 block and 10 threads
    hello_from_gpu<<<2, 4>>>();
    
    // Wait for the GPU to finish
    cudaDeviceSynchronize();
    
    return 0;
}