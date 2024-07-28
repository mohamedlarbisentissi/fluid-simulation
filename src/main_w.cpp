#include <cuda_runtime.h>
#include <iostream>
#include "kernel.h"
#include "structs.h"
#include "data_saver.h"

int main() {
    constexpr int side = 300;
    constexpr int SAVE_EVERY = 1; //ms
    constexpr int size = side * side * side;
    int t = 0;
    data d;
    cudaMalloc(&d.dx, sizeof(float));
    cudaMalloc(&d.dt, sizeof(float));
    cudaMalloc(&d.RT, size * sizeof(float));
    cudaMalloc(&d.mu, size * sizeof(float));
    cudaMalloc(&d.g, size * sizeof(float));
    cudaMalloc(&d.f1.p, size * sizeof(float));
    cudaMalloc(&d.f1.u, size * sizeof(float));
    cudaMalloc(&d.f1.v, size * sizeof(float));
    cudaMalloc(&d.f1.w, size * sizeof(float));
    cudaMalloc(&d.f2.p, size * sizeof(float));
    cudaMalloc(&d.f2.u, size * sizeof(float));
    cudaMalloc(&d.f2.v, size * sizeof(float));
    cudaMalloc(&d.f2.w, size * sizeof(float));
    cudaMalloc(&d.inMain, sizeof(bool));
    int threadsPerBlock(1024);
    int numBlocks((size + threadsPerBlock - 1) / threadsPerBlock);
    DataSaver dataSaver(side);

    init(numBlocks, threadsPerBlock, d, side);
    while (t < 10) {
        //std::cout << "t = " << t << std::endl;
        if (t % SAVE_EVERY == 0) {
            //std::cout << "Saving data..." << std::endl;
            dataSaver.saveData(t, d, side);
        }
        step(d);
        flipInMain(d);
        t++;
    }

    cudaFree(d.dx);
    cudaFree(d.dt);
    cudaFree(d.RT);
    cudaFree(d.f1.w);
    cudaFree(d.f2.p);
    cudaFree(d.f2.u);
    cudaFree(d.f2.v);
    cudaFree(d.f2.w);
    cudaFree(d.inMain);
    return 0;
}
