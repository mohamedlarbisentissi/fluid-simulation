#include <cuda_runtime.h>
#include "kernel.h"
#include "structs.h"

int main() {
    constexpr int side = 300;
    constexpr int size = side * side * side;
    int t = 0;
    data d;
    cudaMalloc(&d.dx, sizeof(float));
    cudaMalloc(&d.dt, sizeof(float));
    cudaMalloc(&d.RT, size * sizeof(float));
    cudaMalloc(&d.mu, size * sizeof(float));
    cudaMalloc(&d.f1.p, size * sizeof(float));
    cudaMalloc(&d.f1.u, size * sizeof(float));
    cudaMalloc(&d.f1.v, size * sizeof(float));
    cudaMalloc(&d.f1.w, size * sizeof(float));
    cudaMalloc(&d.f2.p, size * sizeof(float));
    cudaMalloc(&d.f2.u, size * sizeof(float));
    cudaMalloc(&d.f2.v, size * sizeof(float));
    cudaMalloc(&d.f2.w, size * sizeof(float));
    cudaMalloc(&d.inMain, sizeof(bool));
    dim3 threadsPerBlock(16, 16, 16);
    dim3 numBlocks((side + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (side + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (side + threadsPerBlock.z - 1) / threadsPerBlock.z);

    initializeValues(numBlocks, threadsPerBlock, d, side);
    cudaDeviceSynchronize();
    while (t < 500) {
        launchKernels(numBlocks, threadsPerBlock, d, side);
        cudaDeviceSynchronize();
        flipInMain(d);
        cudaDeviceSynchronize();
        t++;
    }

    cudaFree(d.dx);
    cudaFree(d.dt);
    cudaFree(d.RT);
    cudaFree(d.mu);
    cudaFree(d.f1.p);
    cudaFree(d.f1.u);
    cudaFree(d.f1.v);
    cudaFree(d.f1.w);
    cudaFree(d.f2.p);
    cudaFree(d.f2.u);
    cudaFree(d.f2.v);
    cudaFree(d.f2.w);
    cudaFree(d.inMain);
    return 0;
}
