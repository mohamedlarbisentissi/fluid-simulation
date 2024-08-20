#include <cuda_runtime.h>
#include <iostream>
#include "kernel.h"
#include "structs.h"
#include "data_saver.h"

int main() {
    // *** PYTHON CODE-GENERATED SIDE VALUE ***
    constexpr int SAVE_EVERY = 300; //ms
    constexpr int size = side * side * side;
    float t = 0.0f;
    data d;
    cudaMalloc(&d.dx, sizeof(float));
    cudaMalloc(&d.dt, sizeof(float));
    cudaMalloc(&d.RT, size * sizeof(float));
    cudaMalloc(&d.mu, size * sizeof(float));
    cudaMalloc(&d.g, size * sizeof(float));
    cudaMalloc(&d.small_dt, sizeof(float));
    d.okayStep.resize(size);
    cudaMalloc(&d.f1.m, size * sizeof(float));
    cudaMalloc(&d.f1.u, size * sizeof(float));
    cudaMalloc(&d.f1.v, size * sizeof(float));
    cudaMalloc(&d.f1.w, size * sizeof(float));
    cudaMalloc(&d.f2.m, size * sizeof(float));
    cudaMalloc(&d.f2.u, size * sizeof(float));
    cudaMalloc(&d.f2.v, size * sizeof(float));
    cudaMalloc(&d.f2.w, size * sizeof(float));
    cudaMalloc(&d.inMain, sizeof(bool));
    int threadsPerBlock(1024);
    int numBlocks((size + threadsPerBlock - 1) / threadsPerBlock);
    DataSaver dataSaver(side);

    init(numBlocks, threadsPerBlock, d, side);
    while (t < 1) {
        std::cout << "t = " << t << std::endl;
        if (static_cast<int>(t/1e-3) % SAVE_EVERY == 0) {
            std::cout << "Saving data..." << std::endl;
            dataSaver.saveData(t, d, side);
        }
        float dt;
        step(numBlocks, threadsPerBlock, d, side, dt);
        flipInMain(d);
        t += dt;
    }

    cudaFree(d.dx);
    cudaFree(d.dt);
    cudaFree(d.RT);
    cudaFree(d.mu);
    cudaFree(d.g);
    cudaFree(d.small_dt);
    cudaFree(d.f1.m);
    cudaFree(d.f1.u);
    cudaFree(d.f1.v);
    cudaFree(d.f1.w);
    cudaFree(d.f2.m);
    cudaFree(d.f2.u);
    cudaFree(d.f2.v);
    cudaFree(d.f2.w);
    cudaFree(d.inMain);
    return 0;
}
