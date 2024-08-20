#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <iostream>
#include "structs.h"

// Intialize values
__global__ void initializeValues(data d, int side) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= side * side * side) return;

    constexpr float g = 9.81f;
    constexpr float P0 = 101'325.0f;
    constexpr float T0 = 288.15f;
    constexpr float R = 287.05f;
    constexpr float dx = 1.0e-3f;

    if (index == 0) {
        *d.dx = dx;
        *d.dt = 1e-4f;
        *d.RT = R*T0;
        *d.mu = 1.8e-5f;
        *d.g = g;
        *d.small_dt = *d.dt * 2;
        *d.inMain = false;
    }

    // OkayStep
    d.okayStep[index] = 0;
    // ICs
    int z_index = index / (side * side);
    float z = z_index * dx;
    d.f1.m[index] = P0 * dx * dx * dx / (R * T0);
    d.f1.u[index] = 0.0f;
    d.f1.v[index] = 0.0f;
    d.f1.w[index] = 0.0f;
    d.f2.m[index] = P0 * dx * dx * dx / (R * T0);
    d.f2.u[index] = 0.0f;
    d.f2.v[index] = 0.0f;
    d.f2.w[index] = 0.0f;

    //Non-uniform ICs - pressure gradient in x-direction
    /*
    int x = index % side;
    int y = (index / side) % side;
    int z = index / (side * side);
    d.f1.p[index] = 101'325.0f - 100.0f * x / side;
    d.f2.p[index] = 101'325.0f - 100.0f * x / side;
    */
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

__global__ void checkSmallDt(int numBlocks, int threadsPerBlock, data d, int side) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= side * side * side) return;
    
    float dt = *d.dt;
    float small_dt = *d.small_dt;
    float dx = *d.dx;
    field f_new = *d.inMain ? d.f1 : d.f2;
    field f_old = *d.inMain ? d.f2 : d.f1;
    float u = f_new.u[index];
    float v = f_new.v[index];
    float w = f_new.w[index];
    float m = f_new.m[index];
    float u_mem = f_old.u[index];
    float v_mem = f_old.v[index];
    float w_mem = f_old.w[index];

    if ((u * u_mem < 0 && fabsf(u) > 1e-5 * dx/dt) || (fabsf(u_mem) > dx/small_dt)) {
        d.okayStep[index] = 1;
        return;
    }
    if ((v * v_mem < 0 && fabsf(v) > 1e-5 * dx/dt) || (fabsf(v_mem) > dx/small_dt)) {
        d.okayStep[index] = 1;
        return;
    }
    if ((w * w_mem < 0 && fabsf(w) > 1e-5 * dx/dt) || (fabsf(w_mem) > dx/small_dt)) {
        d.okayStep[index] = 1;
        return;
    }
    d.okayStep[index] = 0;
    return;
}

__global__ void resetSmallDt(data d) {
    *d.small_dt = *d.dt * 2;
}

__global__ void reduceSmallDt(data d) {
    *d.small_dt = *d.small_dt / 2;
}

float step(int numBlocks, int threadsPerBlock, data d, int side, float& dt) {
    bool okayStep;
    do {
        reduceSmallDt<<<1, 1>>>(d);
        // *** PYTHON CODE-GENERATED KERNEL CALLS ***
        cudaDeviceSynchronize();
        checkSmallDt<<<numBlocks, threadsPerBlock>>>(numBlocks, threadsPerBlock, d, side);
        cudaDeviceSynchronize();
        okayStep = thrust::reduce(d.okayStep.begin(), d.okayStep.end());
    } while(okayStep > 0);
    cudaMemcpy(&dt, d.small_dt, sizeof(float), cudaMemcpyDeviceToHost);
    resetSmallDt<<<1, 1>>>(d);
    cudaDeviceSynchronize();
}