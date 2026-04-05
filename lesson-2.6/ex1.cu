#include <stdio.h>

__global__ void hello_from_gpu() {
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int gid = bid * blockDim.x + tid;

    printf("Hello from GPU block %d and thread %d, global id %d\n", bid, tid, gid);
}

// Compile with:
// nvcc ex1.cu -o ex1_compute86 -arch=compute_86 -code=sm_86,compute_86
// This nvcc command compiles the CUDA source file ex1.cu into an executable named ex1_compute86. 
// The -arch=compute_86 option tells nvcc to generate PTX (the virtual GPU ISA) for the virtual compute capability 8.6, 
// while -code=sm_86,compute_86 requests two code objects be emitted: a native SASS binary for sm_86 (the real GPU ISA) and a PTX fallback for compute_86.
// Including both sm_86 and compute_86 is a common pattern: systems with GPUs that match sm_86 will run the native SASS directly for best performance, 
// and other systems (or future GPUs) can JIT-compile the embedded PTX at load time if the driver supports that PTX version. 
// The -arch flag primarily sets the PTX target/version, and -code controls which concrete binary forms are embedded.
// Gotchas: nvcc will error if you request a compute/sm level your CUDA toolkit doesn’t know 
// (e.g., the earlier “Unsupported gpu architecture 'compute_61'” occurs when your nvcc/CUDA version lacks that target).
//  Fix by using a supported compute/sm pair (check nvcc --help or the CUDA release notes) or upgrade the CUDA toolkit/driver.


/*
nvcc ex1.cu -o ex1_fat \
  -gencode=arch=compute_80,code=sm_80 \
  -gencode=arch=compute_80,code=compute_80 \
  -gencode=arch=compute_90,code=sm_90 \
  -gencode=arch=compute_90,code=compute_90 \
  -gencode=arch=compute_100,code=sm_100 \
  -gencode=arch=compute_100,code=compute_100 \
  -gencode=arch=compute_110,code=sm_110 \
  -gencode=arch=compute_110,code=compute_110
*/

int main() {

    printf("Hello from CPU\n");
    // Launch the kernel with 1 block and 10 threads
    hello_from_gpu<<<2, 2>>>();
    
    // Wait for the GPU to finish
    cudaDeviceSynchronize();
    
    return 0;
}