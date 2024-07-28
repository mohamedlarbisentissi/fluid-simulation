#include <cuda_runtime.h>
#include "structs.h"

// Intialize values
__global__ void initializeValues(data d, int side) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= side * side * side) return;
    if (index == 0) {
        *d.dx = 1e-3f;
        *d.dt = 1e-3f;
        *d.RT = 2494.2f;
        *d.mu = 1.8e-5f;
        *d.g = 9.81f;
        *d.inMain = false;
    }
    d.f1.p[index] = 101'325.0f;
    d.f1.u[index] = 0.0f;
    d.f1.v[index] = 0.0f;
    d.f1.w[index] = 0.0f;
    d.f2.p[index] = 101'325.0f;
    d.f2.u[index] = 0.0f;
    d.f2.v[index] = 0.0f;
    d.f2.w[index] = 0.0f;

    //Non-uniform ICs - pressure gradient in x-direction
    int x = index % side;
    int y = (index / side) % side;
    int z = index / (side * side);
    d.f1.p[index] = 101'325.0f - 100.0f * x / side;
    d.f2.p[index] = 101'325.0f - 100.0f * x / side;
    
}

void init(int numBlocks, int threadsPerBlock, data d, int side) {
    initializeValues<<<numBlocks, threadsPerBlock>>>(d, side);
    cudaDeviceSynchronize();
}

// Flip in main
__global__ void flipInMainKernel(data d) {
    if (*d.inMain) {
        *d.inMain = false;
    } else {
        *d.inMain = true;
    }
}

void flipInMain(data d) {
    flipInMainKernel<<<1, 1>>>(d);
    cudaDeviceSynchronize();
}

// Update field
// *** PYTHON CODE-GENERATED KERNEL DEFINITIONS ***

void step(data d) {
    // *** PYTHON CODE-GENERATED KERNEL CALLS ***
    cudaDeviceSynchronize();
}