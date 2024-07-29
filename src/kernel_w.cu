#include <cuda_runtime.h>
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
    const float dx = 1e-3f;

    if (index == 0) {
        *d.dx = dx;
        *d.dt = 1e-4f;
        *d.RT = R*T0;
        *d.mu = 1.8e-5f;
        *d.g = g;
        *d.inMain = false;
    }
    // Max allowable speed for stable simulation is ~10m/s if dx = 1e-3 and dt = 1e-4

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

__global__ void updateCore(data d) {
    int totalThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if(totalThreadIndex >= 1) return;

    // Compute indices
    int x = (totalThreadIndex % 1) + 1;
    int y = ((totalThreadIndex / 1) % 1) + 1;
    int z = (totalThreadIndex / (1 * 1)) + 1;
    int index = x + y * 3 + z * 3 * 3;
    int index_xm = (x-1) + y * 3 + z * 3 * 3;
    int index_xp = (x+1) + y * 3 + z * 3 * 3;
    int index_ym = x + (y-1) * 3 + z * 3 * 3;
    int index_yp = x + (y+1) * 3 + z * 3 * 3;
    int index_zm = x + y * 3 + (z-1) * 3 * 3;
    int index_zp = x + y * 3 + (z+1) * 3 * 3;


    // Define origin and destination fields
    field f1 = *d.inMain ? d.f1 : d.f2;
    field f2 = *d.inMain ? d.f2 : d.f1;

    // Define dx, dt and RT
    float dx = *d.dx;
    float dt = *d.dt;
    float RT = *d.RT;
    float mu = *d.mu;
    float g = *d.g;

    // Update mass field

    f2.m[index] = f1.m[index] + dt/dx * (

        + (0.5 * (f1.u[index-1] + f1.u[index])) * ((0.5 * (f1.u[index-1] + f1.u[index])) > 0 ? f1.m[index_xm] : f1.m[index])

        - (0.5 * (f1.u[index+1] + f1.u[index])) * ((0.5 * (f1.u[index+1] + f1.u[index])) > 0 ? f1.m[index] : f1.m[index_xp])

        + (0.5 * (f1.v[index-1] + f1.v[index])) * ((0.5 * (f1.v[index-1] + f1.v[index])) > 0 ? f1.m[index_ym] : f1.m[index])

        - (0.5 * (f1.v[index+1] + f1.v[index])) * ((0.5 * (f1.v[index+1] + f1.v[index])) > 0 ? f1.m[index] : f1.m[index_yp])

        + (0.5 * (f1.w[index-1] + f1.w[index])) * ((0.5 * (f1.w[index-1] + f1.w[index])) > 0 ? f1.m[index_zm] : f1.m[index])

        - (0.5 * (f1.w[index+1] + f1.w[index])) * ((0.5 * (f1.w[index+1] + f1.w[index])) > 0 ? f1.m[index] : f1.m[index_zp])

    );
    // Update velocity field

    f2.u[index] = (f1.m[index] * f1.u[index]) / f2.m[index] + (dt / f2.m[index]) * (
        (1/dx) * (

            + (0.5 * (f1.u[index-1] + f1.u[index])) * ((0.5 * (f1.u[index-1] + f1.u[index])) > 0 ? (f1.m[index_xm] * f1.u[index_xm]) : (f1.m[index] * f1.u[index]))

            - (0.5 * (f1.u[index+1] + f1.u[index])) * ((0.5 * (f1.u[index+1] + f1.u[index])) > 0 ? (f1.m[index] * f1.u[index]) : (f1.m[index_xp] * f1.u[index_xp]))

            + (0.5 * (f1.v[index-1] + f1.v[index])) * ((0.5 * (f1.v[index-1] + f1.v[index])) > 0 ? (f1.m[index_ym] * f1.u[index_ym]) : (f1.m[index] * f1.u[index]))

            - (0.5 * (f1.v[index+1] + f1.v[index])) * ((0.5 * (f1.v[index+1] + f1.v[index])) > 0 ? (f1.m[index] * f1.u[index]) : (f1.m[index_yp] * f1.u[index_yp]))

            + (0.5 * (f1.w[index-1] + f1.w[index])) * ((0.5 * (f1.w[index-1] + f1.w[index])) > 0 ? (f1.m[index_zm] * f1.u[index_zm]) : (f1.m[index] * f1.u[index]))

            - (0.5 * (f1.w[index+1] + f1.w[index])) * ((0.5 * (f1.w[index+1] + f1.w[index])) > 0 ? (f1.m[index] * f1.u[index]) : (f1.m[index_zp] * f1.u[index_zp]))

        ) - (RT/dx) * (

            + f1.m[index_xm]

            - f1.m[index_xp]

        ) + dx * mu * (

            + (f1.u[index_xm] - f1.u[index])

            + (f1.u[index_xp] - f1.u[index])

            + (f1.u[index_ym] - f1.u[index])

        )
    );


    f2.v[index] = (f1.m[index] * f1.v[index]) / f2.m[index] + (dt / f2.m[index]) * (
        (1/dx) * (

            + (0.5 * (f1.u[index-1] + f1.u[index])) * ((0.5 * (f1.u[index-1] + f1.u[index])) > 0 ? (f1.m[index_xm] * f1.v[index_xm]) : (f1.m[index] * f1.v[index]))

            - (0.5 * (f1.u[index+1] + f1.u[index])) * ((0.5 * (f1.u[index+1] + f1.u[index])) > 0 ? (f1.m[index] * f1.v[index]) : (f1.m[index_xp] * f1.v[index_xp]))

            + (0.5 * (f1.v[index-1] + f1.v[index])) * ((0.5 * (f1.v[index-1] + f1.v[index])) > 0 ? (f1.m[index_ym] * f1.v[index_ym]) : (f1.m[index] * f1.v[index]))

            - (0.5 * (f1.v[index+1] + f1.v[index])) * ((0.5 * (f1.v[index+1] + f1.v[index])) > 0 ? (f1.m[index] * f1.v[index]) : (f1.m[index_yp] * f1.v[index_yp]))

            + (0.5 * (f1.w[index-1] + f1.w[index])) * ((0.5 * (f1.w[index-1] + f1.w[index])) > 0 ? (f1.m[index_zm] * f1.v[index_zm]) : (f1.m[index] * f1.v[index]))

            - (0.5 * (f1.w[index+1] + f1.w[index])) * ((0.5 * (f1.w[index+1] + f1.w[index])) > 0 ? (f1.m[index] * f1.v[index]) : (f1.m[index_zp] * f1.v[index_zp]))

        ) - (RT/dx) * (

            + f1.m[index_ym]

            - f1.m[index_yp]

        ) + dx * mu * (

            + (f1.v[index_xm] - f1.v[index])

            + (f1.v[index_xp] - f1.v[index])

            + (f1.v[index_ym] - f1.v[index])

        )
    );


    f2.w[index] = (f1.m[index] * f1.w[index]) / f2.m[index] + (dt / f2.m[index]) * (
        (1/dx) * (

            + (0.5 * (f1.u[index-1] + f1.u[index])) * ((0.5 * (f1.u[index-1] + f1.u[index])) > 0 ? (f1.m[index_xm] * f1.w[index_xm]) : (f1.m[index] * f1.w[index]))

            - (0.5 * (f1.u[index+1] + f1.u[index])) * ((0.5 * (f1.u[index+1] + f1.u[index])) > 0 ? (f1.m[index] * f1.w[index]) : (f1.m[index_xp] * f1.w[index_xp]))

            + (0.5 * (f1.v[index-1] + f1.v[index])) * ((0.5 * (f1.v[index-1] + f1.v[index])) > 0 ? (f1.m[index_ym] * f1.w[index_ym]) : (f1.m[index] * f1.w[index]))

            - (0.5 * (f1.v[index+1] + f1.v[index])) * ((0.5 * (f1.v[index+1] + f1.v[index])) > 0 ? (f1.m[index] * f1.w[index]) : (f1.m[index_yp] * f1.w[index_yp]))

            + (0.5 * (f1.w[index-1] + f1.w[index])) * ((0.5 * (f1.w[index-1] + f1.w[index])) > 0 ? (f1.m[index_zm] * f1.w[index_zm]) : (f1.m[index] * f1.w[index]))

            - (0.5 * (f1.w[index+1] + f1.w[index])) * ((0.5 * (f1.w[index+1] + f1.w[index])) > 0 ? (f1.m[index] * f1.w[index]) : (f1.m[index_zp] * f1.w[index_zp]))

        ) - (RT/dx) * (

            + f1.m[index_zm]

            - f1.m[index_zp]

        ) + dx * mu * (

            + (f1.w[index_xm] - f1.w[index])

            + (f1.w[index_xp] - f1.w[index])

            + (f1.w[index_ym] - f1.w[index])

        ) + f1.m[index] * g
    );


}

__global__ void updateFaceX1(data d) {
    int totalThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if(totalThreadIndex >= 1) return;


    int x = 0;
    int y = (totalThreadIndex % 1) + 1;
    int z = ((totalThreadIndex / 1) % 1) + 1;
    int index = x + y * 3 + z * 3 * 3;
    int index_xp = (x+1) + y * 3 + z * 3 * 3;
    int index_ym = x + (y-1) * 3 + z * 3 * 3;
    int index_yp = x + (y+1) * 3 + z * 3 * 3;
    int index_zm = x + y * 3 + (z-1) * 3 * 3;
    int index_zp = x + y * 3 + (z+1) * 3 * 3;



    // Define origin and destination fields
    field f1 = *d.inMain ? d.f1 : d.f2;
    field f2 = *d.inMain ? d.f2 : d.f1;

    // Define dx, dt and RT
    float dx = *d.dx;
    float dt = *d.dt;
    float RT = *d.RT;
    float mu = *d.mu;
    float g = *d.g;

    // Update mass field

    f2.m[index] = f1.m[index] + dt/dx * (

        - (0.5 * (f1.u[index+1] + f1.u[index])) * ((0.5 * (f1.u[index+1] + f1.u[index])) > 0 ? f1.m[index] : f1.m[index_xp])

        + (0.5 * (f1.v[index-1] + f1.v[index])) * ((0.5 * (f1.v[index-1] + f1.v[index])) > 0 ? f1.m[index_ym] : f1.m[index])

        - (0.5 * (f1.v[index+1] + f1.v[index])) * ((0.5 * (f1.v[index+1] + f1.v[index])) > 0 ? f1.m[index] : f1.m[index_yp])

        + (0.5 * (f1.w[index-1] + f1.w[index])) * ((0.5 * (f1.w[index-1] + f1.w[index])) > 0 ? f1.m[index_zm] : f1.m[index])

        - (0.5 * (f1.w[index+1] + f1.w[index])) * ((0.5 * (f1.w[index+1] + f1.w[index])) > 0 ? f1.m[index] : f1.m[index_zp])

    );
    // Update velocity field
    f2.u[index] = 0;

    f2.v[index] = (f1.m[index] * f1.v[index]) / f2.m[index] + (dt / f2.m[index]) * (
        (1/dx) * (

            - (0.5 * (f1.u[index+1] + f1.u[index])) * ((0.5 * (f1.u[index+1] + f1.u[index])) > 0 ? (f1.m[index] * f1.v[index]) : (f1.m[index_xp] * f1.v[index_xp]))

            + (0.5 * (f1.v[index-1] + f1.v[index])) * ((0.5 * (f1.v[index-1] + f1.v[index])) > 0 ? (f1.m[index_ym] * f1.v[index_ym]) : (f1.m[index] * f1.v[index]))

            - (0.5 * (f1.v[index+1] + f1.v[index])) * ((0.5 * (f1.v[index+1] + f1.v[index])) > 0 ? (f1.m[index] * f1.v[index]) : (f1.m[index_yp] * f1.v[index_yp]))

            + (0.5 * (f1.w[index-1] + f1.w[index])) * ((0.5 * (f1.w[index-1] + f1.w[index])) > 0 ? (f1.m[index_zm] * f1.v[index_zm]) : (f1.m[index] * f1.v[index]))

            - (0.5 * (f1.w[index+1] + f1.w[index])) * ((0.5 * (f1.w[index+1] + f1.w[index])) > 0 ? (f1.m[index] * f1.v[index]) : (f1.m[index_zp] * f1.v[index_zp]))

        ) - (RT/dx) * (

            + f1.m[index_ym]

            - f1.m[index_yp]

        ) + dx * mu * (

            + (f1.v[index_xp] - f1.v[index])

            + (f1.v[index_ym] - f1.v[index])

        )
    );


    f2.w[index] = (f1.m[index] * f1.w[index]) / f2.m[index] + (dt / f2.m[index]) * (
        (1/dx) * (

            - (0.5 * (f1.u[index+1] + f1.u[index])) * ((0.5 * (f1.u[index+1] + f1.u[index])) > 0 ? (f1.m[index] * f1.w[index]) : (f1.m[index_xp] * f1.w[index_xp]))

            + (0.5 * (f1.v[index-1] + f1.v[index])) * ((0.5 * (f1.v[index-1] + f1.v[index])) > 0 ? (f1.m[index_ym] * f1.w[index_ym]) : (f1.m[index] * f1.w[index]))

            - (0.5 * (f1.v[index+1] + f1.v[index])) * ((0.5 * (f1.v[index+1] + f1.v[index])) > 0 ? (f1.m[index] * f1.w[index]) : (f1.m[index_yp] * f1.w[index_yp]))

            + (0.5 * (f1.w[index-1] + f1.w[index])) * ((0.5 * (f1.w[index-1] + f1.w[index])) > 0 ? (f1.m[index_zm] * f1.w[index_zm]) : (f1.m[index] * f1.w[index]))

            - (0.5 * (f1.w[index+1] + f1.w[index])) * ((0.5 * (f1.w[index+1] + f1.w[index])) > 0 ? (f1.m[index] * f1.w[index]) : (f1.m[index_zp] * f1.w[index_zp]))

        ) - (RT/dx) * (

            + f1.m[index_zm]

            - f1.m[index_zp]

        ) + dx * mu * (

            + (f1.w[index_xp] - f1.w[index])

            + (f1.w[index_ym] - f1.w[index])

        ) + f1.m[index] * g
    );


}

__global__ void updateFaceX2(data d) {
    int totalThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if(totalThreadIndex >= 1) return;


    int x = 2;
    int y = (totalThreadIndex % 1) + 1;
    int z = ((totalThreadIndex / 1) % 1) + 1;
    int index = x + y * 3 + z * 3 * 3;
    int index_xm = (x-1) + y * 3 + z * 3 * 3;
    int index_ym = x + (y-1) * 3 + z * 3 * 3;
    int index_yp = x + (y+1) * 3 + z * 3 * 3;
    int index_zm = x + y * 3 + (z-1) * 3 * 3;
    int index_zp = x + y * 3 + (z+1) * 3 * 3;



    // Define origin and destination fields
    field f1 = *d.inMain ? d.f1 : d.f2;
    field f2 = *d.inMain ? d.f2 : d.f1;

    // Define dx, dt and RT
    float dx = *d.dx;
    float dt = *d.dt;
    float RT = *d.RT;
    float mu = *d.mu;
    float g = *d.g;

    // Update mass field

    f2.m[index] = f1.m[index] + dt/dx * (

        + (0.5 * (f1.u[index-1] + f1.u[index])) * ((0.5 * (f1.u[index-1] + f1.u[index])) > 0 ? f1.m[index_xm] : f1.m[index])

        + (0.5 * (f1.v[index-1] + f1.v[index])) * ((0.5 * (f1.v[index-1] + f1.v[index])) > 0 ? f1.m[index_ym] : f1.m[index])

        - (0.5 * (f1.v[index+1] + f1.v[index])) * ((0.5 * (f1.v[index+1] + f1.v[index])) > 0 ? f1.m[index] : f1.m[index_yp])

        + (0.5 * (f1.w[index-1] + f1.w[index])) * ((0.5 * (f1.w[index-1] + f1.w[index])) > 0 ? f1.m[index_zm] : f1.m[index])

        - (0.5 * (f1.w[index+1] + f1.w[index])) * ((0.5 * (f1.w[index+1] + f1.w[index])) > 0 ? f1.m[index] : f1.m[index_zp])

    );
    // Update velocity field
    f2.u[index] = 0;

    f2.v[index] = (f1.m[index] * f1.v[index]) / f2.m[index] + (dt / f2.m[index]) * (
        (1/dx) * (

            + (0.5 * (f1.u[index-1] + f1.u[index])) * ((0.5 * (f1.u[index-1] + f1.u[index])) > 0 ? (f1.m[index_xm] * f1.v[index_xm]) : (f1.m[index] * f1.v[index]))

            + (0.5 * (f1.v[index-1] + f1.v[index])) * ((0.5 * (f1.v[index-1] + f1.v[index])) > 0 ? (f1.m[index_ym] * f1.v[index_ym]) : (f1.m[index] * f1.v[index]))

            - (0.5 * (f1.v[index+1] + f1.v[index])) * ((0.5 * (f1.v[index+1] + f1.v[index])) > 0 ? (f1.m[index] * f1.v[index]) : (f1.m[index_yp] * f1.v[index_yp]))

            + (0.5 * (f1.w[index-1] + f1.w[index])) * ((0.5 * (f1.w[index-1] + f1.w[index])) > 0 ? (f1.m[index_zm] * f1.v[index_zm]) : (f1.m[index] * f1.v[index]))

            - (0.5 * (f1.w[index+1] + f1.w[index])) * ((0.5 * (f1.w[index+1] + f1.w[index])) > 0 ? (f1.m[index] * f1.v[index]) : (f1.m[index_zp] * f1.v[index_zp]))

        ) - (RT/dx) * (

            + f1.m[index_ym]

            - f1.m[index_yp]

        ) + dx * mu * (

            + (f1.v[index_xm] - f1.v[index])

            + (f1.v[index_ym] - f1.v[index])

        )
    );


    f2.w[index] = (f1.m[index] * f1.w[index]) / f2.m[index] + (dt / f2.m[index]) * (
        (1/dx) * (

            + (0.5 * (f1.u[index-1] + f1.u[index])) * ((0.5 * (f1.u[index-1] + f1.u[index])) > 0 ? (f1.m[index_xm] * f1.w[index_xm]) : (f1.m[index] * f1.w[index]))

            + (0.5 * (f1.v[index-1] + f1.v[index])) * ((0.5 * (f1.v[index-1] + f1.v[index])) > 0 ? (f1.m[index_ym] * f1.w[index_ym]) : (f1.m[index] * f1.w[index]))

            - (0.5 * (f1.v[index+1] + f1.v[index])) * ((0.5 * (f1.v[index+1] + f1.v[index])) > 0 ? (f1.m[index] * f1.w[index]) : (f1.m[index_yp] * f1.w[index_yp]))

            + (0.5 * (f1.w[index-1] + f1.w[index])) * ((0.5 * (f1.w[index-1] + f1.w[index])) > 0 ? (f1.m[index_zm] * f1.w[index_zm]) : (f1.m[index] * f1.w[index]))

            - (0.5 * (f1.w[index+1] + f1.w[index])) * ((0.5 * (f1.w[index+1] + f1.w[index])) > 0 ? (f1.m[index] * f1.w[index]) : (f1.m[index_zp] * f1.w[index_zp]))

        ) - (RT/dx) * (

            + f1.m[index_zm]

            - f1.m[index_zp]

        ) + dx * mu * (

            + (f1.w[index_xm] - f1.w[index])

            + (f1.w[index_ym] - f1.w[index])

        ) + f1.m[index] * g
    );


}

__global__ void updateFaceY1(data d) {
    int totalThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if(totalThreadIndex >= 1) return;


    int x = (totalThreadIndex % 1) + 1;
    int y = 0;
    int z = ((totalThreadIndex / 1) % 1) + 1;
    int index = x + y * 3 + z * 3 * 3;
    int index_xm = (x-1) + y * 3 + z * 3 * 3;
    int index_xp = (x+1) + y * 3 + z * 3 * 3;
    int index_yp = x + (y+1) * 3 + z * 3 * 3;
    int index_zm = x + y * 3 + (z-1) * 3 * 3;
    int index_zp = x + y * 3 + (z+1) * 3 * 3;



    // Define origin and destination fields
    field f1 = *d.inMain ? d.f1 : d.f2;
    field f2 = *d.inMain ? d.f2 : d.f1;

    // Define dx, dt and RT
    float dx = *d.dx;
    float dt = *d.dt;
    float RT = *d.RT;
    float mu = *d.mu;
    float g = *d.g;

    // Update mass field

    f2.m[index] = f1.m[index] + dt/dx * (

        + (0.5 * (f1.u[index-1] + f1.u[index])) * ((0.5 * (f1.u[index-1] + f1.u[index])) > 0 ? f1.m[index_xm] : f1.m[index])

        - (0.5 * (f1.u[index+1] + f1.u[index])) * ((0.5 * (f1.u[index+1] + f1.u[index])) > 0 ? f1.m[index] : f1.m[index_xp])

        - (0.5 * (f1.v[index+1] + f1.v[index])) * ((0.5 * (f1.v[index+1] + f1.v[index])) > 0 ? f1.m[index] : f1.m[index_yp])

        + (0.5 * (f1.w[index-1] + f1.w[index])) * ((0.5 * (f1.w[index-1] + f1.w[index])) > 0 ? f1.m[index_zm] : f1.m[index])

        - (0.5 * (f1.w[index+1] + f1.w[index])) * ((0.5 * (f1.w[index+1] + f1.w[index])) > 0 ? f1.m[index] : f1.m[index_zp])

    );
    // Update velocity field

    f2.u[index] = (f1.m[index] * f1.u[index]) / f2.m[index] + (dt / f2.m[index]) * (
        (1/dx) * (

            + (0.5 * (f1.u[index-1] + f1.u[index])) * ((0.5 * (f1.u[index-1] + f1.u[index])) > 0 ? (f1.m[index_xm] * f1.u[index_xm]) : (f1.m[index] * f1.u[index]))

            - (0.5 * (f1.u[index+1] + f1.u[index])) * ((0.5 * (f1.u[index+1] + f1.u[index])) > 0 ? (f1.m[index] * f1.u[index]) : (f1.m[index_xp] * f1.u[index_xp]))

            - (0.5 * (f1.v[index+1] + f1.v[index])) * ((0.5 * (f1.v[index+1] + f1.v[index])) > 0 ? (f1.m[index] * f1.u[index]) : (f1.m[index_yp] * f1.u[index_yp]))

            + (0.5 * (f1.w[index-1] + f1.w[index])) * ((0.5 * (f1.w[index-1] + f1.w[index])) > 0 ? (f1.m[index_zm] * f1.u[index_zm]) : (f1.m[index] * f1.u[index]))

            - (0.5 * (f1.w[index+1] + f1.w[index])) * ((0.5 * (f1.w[index+1] + f1.w[index])) > 0 ? (f1.m[index] * f1.u[index]) : (f1.m[index_zp] * f1.u[index_zp]))

        ) - (RT/dx) * (

            + f1.m[index_xm]

            - f1.m[index_xp]

        ) + dx * mu * (

            + (f1.u[index_xm] - f1.u[index])

            + (f1.u[index_xp] - f1.u[index])

        )
    );

    f2.v[index] = 0;

    f2.w[index] = (f1.m[index] * f1.w[index]) / f2.m[index] + (dt / f2.m[index]) * (
        (1/dx) * (

            + (0.5 * (f1.u[index-1] + f1.u[index])) * ((0.5 * (f1.u[index-1] + f1.u[index])) > 0 ? (f1.m[index_xm] * f1.w[index_xm]) : (f1.m[index] * f1.w[index]))

            - (0.5 * (f1.u[index+1] + f1.u[index])) * ((0.5 * (f1.u[index+1] + f1.u[index])) > 0 ? (f1.m[index] * f1.w[index]) : (f1.m[index_xp] * f1.w[index_xp]))

            - (0.5 * (f1.v[index+1] + f1.v[index])) * ((0.5 * (f1.v[index+1] + f1.v[index])) > 0 ? (f1.m[index] * f1.w[index]) : (f1.m[index_yp] * f1.w[index_yp]))

            + (0.5 * (f1.w[index-1] + f1.w[index])) * ((0.5 * (f1.w[index-1] + f1.w[index])) > 0 ? (f1.m[index_zm] * f1.w[index_zm]) : (f1.m[index] * f1.w[index]))

            - (0.5 * (f1.w[index+1] + f1.w[index])) * ((0.5 * (f1.w[index+1] + f1.w[index])) > 0 ? (f1.m[index] * f1.w[index]) : (f1.m[index_zp] * f1.w[index_zp]))

        ) - (RT/dx) * (

            + f1.m[index_zm]

            - f1.m[index_zp]

        ) + dx * mu * (

            + (f1.w[index_xm] - f1.w[index])

            + (f1.w[index_xp] - f1.w[index])

        ) + f1.m[index] * g
    );


}

__global__ void updateFaceY2(data d) {
    int totalThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if(totalThreadIndex >= 1) return;


    int x = (totalThreadIndex % 1) + 1;
    int y = 2;
    int z = ((totalThreadIndex / 1) % 1) + 1;
    int index = x + y * 3 + z * 3 * 3;
    int index_xm = (x-1) + y * 3 + z * 3 * 3;
    int index_xp = (x+1) + y * 3 + z * 3 * 3;
    int index_ym = x + (y-1) * 3 + z * 3 * 3;
    int index_zm = x + y * 3 + (z-1) * 3 * 3;
    int index_zp = x + y * 3 + (z+1) * 3 * 3;



    // Define origin and destination fields
    field f1 = *d.inMain ? d.f1 : d.f2;
    field f2 = *d.inMain ? d.f2 : d.f1;

    // Define dx, dt and RT
    float dx = *d.dx;
    float dt = *d.dt;
    float RT = *d.RT;
    float mu = *d.mu;
    float g = *d.g;

    // Update mass field

    f2.m[index] = f1.m[index] + dt/dx * (

        + (0.5 * (f1.u[index-1] + f1.u[index])) * ((0.5 * (f1.u[index-1] + f1.u[index])) > 0 ? f1.m[index_xm] : f1.m[index])

        - (0.5 * (f1.u[index+1] + f1.u[index])) * ((0.5 * (f1.u[index+1] + f1.u[index])) > 0 ? f1.m[index] : f1.m[index_xp])

        + (0.5 * (f1.v[index-1] + f1.v[index])) * ((0.5 * (f1.v[index-1] + f1.v[index])) > 0 ? f1.m[index_ym] : f1.m[index])

        + (0.5 * (f1.w[index-1] + f1.w[index])) * ((0.5 * (f1.w[index-1] + f1.w[index])) > 0 ? f1.m[index_zm] : f1.m[index])

        - (0.5 * (f1.w[index+1] + f1.w[index])) * ((0.5 * (f1.w[index+1] + f1.w[index])) > 0 ? f1.m[index] : f1.m[index_zp])

    );
    // Update velocity field

    f2.u[index] = (f1.m[index] * f1.u[index]) / f2.m[index] + (dt / f2.m[index]) * (
        (1/dx) * (

            + (0.5 * (f1.u[index-1] + f1.u[index])) * ((0.5 * (f1.u[index-1] + f1.u[index])) > 0 ? (f1.m[index_xm] * f1.u[index_xm]) : (f1.m[index] * f1.u[index]))

            - (0.5 * (f1.u[index+1] + f1.u[index])) * ((0.5 * (f1.u[index+1] + f1.u[index])) > 0 ? (f1.m[index] * f1.u[index]) : (f1.m[index_xp] * f1.u[index_xp]))

            + (0.5 * (f1.v[index-1] + f1.v[index])) * ((0.5 * (f1.v[index-1] + f1.v[index])) > 0 ? (f1.m[index_ym] * f1.u[index_ym]) : (f1.m[index] * f1.u[index]))

            + (0.5 * (f1.w[index-1] + f1.w[index])) * ((0.5 * (f1.w[index-1] + f1.w[index])) > 0 ? (f1.m[index_zm] * f1.u[index_zm]) : (f1.m[index] * f1.u[index]))

            - (0.5 * (f1.w[index+1] + f1.w[index])) * ((0.5 * (f1.w[index+1] + f1.w[index])) > 0 ? (f1.m[index] * f1.u[index]) : (f1.m[index_zp] * f1.u[index_zp]))

        ) - (RT/dx) * (

            + f1.m[index_xm]

            - f1.m[index_xp]

        ) + dx * mu * (

            + (f1.u[index_xm] - f1.u[index])

            + (f1.u[index_xp] - f1.u[index])

            + (f1.u[index_ym] - f1.u[index])

        )
    );

    f2.v[index] = 0;

    f2.w[index] = (f1.m[index] * f1.w[index]) / f2.m[index] + (dt / f2.m[index]) * (
        (1/dx) * (

            + (0.5 * (f1.u[index-1] + f1.u[index])) * ((0.5 * (f1.u[index-1] + f1.u[index])) > 0 ? (f1.m[index_xm] * f1.w[index_xm]) : (f1.m[index] * f1.w[index]))

            - (0.5 * (f1.u[index+1] + f1.u[index])) * ((0.5 * (f1.u[index+1] + f1.u[index])) > 0 ? (f1.m[index] * f1.w[index]) : (f1.m[index_xp] * f1.w[index_xp]))

            + (0.5 * (f1.v[index-1] + f1.v[index])) * ((0.5 * (f1.v[index-1] + f1.v[index])) > 0 ? (f1.m[index_ym] * f1.w[index_ym]) : (f1.m[index] * f1.w[index]))

            + (0.5 * (f1.w[index-1] + f1.w[index])) * ((0.5 * (f1.w[index-1] + f1.w[index])) > 0 ? (f1.m[index_zm] * f1.w[index_zm]) : (f1.m[index] * f1.w[index]))

            - (0.5 * (f1.w[index+1] + f1.w[index])) * ((0.5 * (f1.w[index+1] + f1.w[index])) > 0 ? (f1.m[index] * f1.w[index]) : (f1.m[index_zp] * f1.w[index_zp]))

        ) - (RT/dx) * (

            + f1.m[index_zm]

            - f1.m[index_zp]

        ) + dx * mu * (

            + (f1.w[index_xm] - f1.w[index])

            + (f1.w[index_xp] - f1.w[index])

            + (f1.w[index_ym] - f1.w[index])

        ) + f1.m[index] * g
    );


}

__global__ void updateFaceZ1(data d) {
    int totalThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if(totalThreadIndex >= 1) return;


    int x = (totalThreadIndex % 1) + 1;
    int y = ((totalThreadIndex / 1) % 1) + 1;
    int z = 0;
    int index = x + y * 3 + z * 3 * 3;
    int index_xm = (x-1) + y * 3 + z * 3 * 3;
    int index_xp = (x+1) + y * 3 + z * 3 * 3;
    int index_ym = x + (y-1) * 3 + z * 3 * 3;
    int index_yp = x + (y+1) * 3 + z * 3 * 3;
    int index_zp = x + y * 3 + (z+1) * 3 * 3;



    // Define origin and destination fields
    field f1 = *d.inMain ? d.f1 : d.f2;
    field f2 = *d.inMain ? d.f2 : d.f1;

    // Define dx, dt and RT
    float dx = *d.dx;
    float dt = *d.dt;
    float RT = *d.RT;
    float mu = *d.mu;
    float g = *d.g;

    // Update mass field

    f2.m[index] = f1.m[index] + dt/dx * (

        + (0.5 * (f1.u[index-1] + f1.u[index])) * ((0.5 * (f1.u[index-1] + f1.u[index])) > 0 ? f1.m[index_xm] : f1.m[index])

        - (0.5 * (f1.u[index+1] + f1.u[index])) * ((0.5 * (f1.u[index+1] + f1.u[index])) > 0 ? f1.m[index] : f1.m[index_xp])

        + (0.5 * (f1.v[index-1] + f1.v[index])) * ((0.5 * (f1.v[index-1] + f1.v[index])) > 0 ? f1.m[index_ym] : f1.m[index])

        - (0.5 * (f1.v[index+1] + f1.v[index])) * ((0.5 * (f1.v[index+1] + f1.v[index])) > 0 ? f1.m[index] : f1.m[index_yp])

        - (0.5 * (f1.w[index+1] + f1.w[index])) * ((0.5 * (f1.w[index+1] + f1.w[index])) > 0 ? f1.m[index] : f1.m[index_zp])

    );
    // Update velocity field

    f2.u[index] = (f1.m[index] * f1.u[index]) / f2.m[index] + (dt / f2.m[index]) * (
        (1/dx) * (

            + (0.5 * (f1.u[index-1] + f1.u[index])) * ((0.5 * (f1.u[index-1] + f1.u[index])) > 0 ? (f1.m[index_xm] * f1.u[index_xm]) : (f1.m[index] * f1.u[index]))

            - (0.5 * (f1.u[index+1] + f1.u[index])) * ((0.5 * (f1.u[index+1] + f1.u[index])) > 0 ? (f1.m[index] * f1.u[index]) : (f1.m[index_xp] * f1.u[index_xp]))

            + (0.5 * (f1.v[index-1] + f1.v[index])) * ((0.5 * (f1.v[index-1] + f1.v[index])) > 0 ? (f1.m[index_ym] * f1.u[index_ym]) : (f1.m[index] * f1.u[index]))

            - (0.5 * (f1.v[index+1] + f1.v[index])) * ((0.5 * (f1.v[index+1] + f1.v[index])) > 0 ? (f1.m[index] * f1.u[index]) : (f1.m[index_yp] * f1.u[index_yp]))

            - (0.5 * (f1.w[index+1] + f1.w[index])) * ((0.5 * (f1.w[index+1] + f1.w[index])) > 0 ? (f1.m[index] * f1.u[index]) : (f1.m[index_zp] * f1.u[index_zp]))

        ) - (RT/dx) * (

            + f1.m[index_xm]

            - f1.m[index_xp]

        ) + dx * mu * (

            + (f1.u[index_xm] - f1.u[index])

            + (f1.u[index_xp] - f1.u[index])

            + (f1.u[index_ym] - f1.u[index])

        )
    );


    f2.v[index] = (f1.m[index] * f1.v[index]) / f2.m[index] + (dt / f2.m[index]) * (
        (1/dx) * (

            + (0.5 * (f1.u[index-1] + f1.u[index])) * ((0.5 * (f1.u[index-1] + f1.u[index])) > 0 ? (f1.m[index_xm] * f1.v[index_xm]) : (f1.m[index] * f1.v[index]))

            - (0.5 * (f1.u[index+1] + f1.u[index])) * ((0.5 * (f1.u[index+1] + f1.u[index])) > 0 ? (f1.m[index] * f1.v[index]) : (f1.m[index_xp] * f1.v[index_xp]))

            + (0.5 * (f1.v[index-1] + f1.v[index])) * ((0.5 * (f1.v[index-1] + f1.v[index])) > 0 ? (f1.m[index_ym] * f1.v[index_ym]) : (f1.m[index] * f1.v[index]))

            - (0.5 * (f1.v[index+1] + f1.v[index])) * ((0.5 * (f1.v[index+1] + f1.v[index])) > 0 ? (f1.m[index] * f1.v[index]) : (f1.m[index_yp] * f1.v[index_yp]))

            - (0.5 * (f1.w[index+1] + f1.w[index])) * ((0.5 * (f1.w[index+1] + f1.w[index])) > 0 ? (f1.m[index] * f1.v[index]) : (f1.m[index_zp] * f1.v[index_zp]))

        ) - (RT/dx) * (

            + f1.m[index_ym]

            - f1.m[index_yp]

        ) + dx * mu * (

            + (f1.v[index_xm] - f1.v[index])

            + (f1.v[index_xp] - f1.v[index])

            + (f1.v[index_ym] - f1.v[index])

        )
    );

    f2.w[index] = 0;

}

__global__ void updateFaceZ2(data d) {
    int totalThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if(totalThreadIndex >= 1) return;


    int x = (totalThreadIndex % 1) + 1;
    int y = ((totalThreadIndex / 1) % 1) + 1;
    int z = 2;
    int index = x + y * 3 + z * 3 * 3;
    int index_xm = (x-1) + y * 3 + z * 3 * 3;
    int index_xp = (x+1) + y * 3 + z * 3 * 3;
    int index_ym = x + (y-1) * 3 + z * 3 * 3;
    int index_yp = x + (y+1) * 3 + z * 3 * 3;
    int index_zm = x + y * 3 + (z-1) * 3 * 3;



    // Define origin and destination fields
    field f1 = *d.inMain ? d.f1 : d.f2;
    field f2 = *d.inMain ? d.f2 : d.f1;

    // Define dx, dt and RT
    float dx = *d.dx;
    float dt = *d.dt;
    float RT = *d.RT;
    float mu = *d.mu;
    float g = *d.g;

    // Update mass field

    f2.m[index] = f1.m[index] + dt/dx * (

        + (0.5 * (f1.u[index-1] + f1.u[index])) * ((0.5 * (f1.u[index-1] + f1.u[index])) > 0 ? f1.m[index_xm] : f1.m[index])

        - (0.5 * (f1.u[index+1] + f1.u[index])) * ((0.5 * (f1.u[index+1] + f1.u[index])) > 0 ? f1.m[index] : f1.m[index_xp])

        + (0.5 * (f1.v[index-1] + f1.v[index])) * ((0.5 * (f1.v[index-1] + f1.v[index])) > 0 ? f1.m[index_ym] : f1.m[index])

        - (0.5 * (f1.v[index+1] + f1.v[index])) * ((0.5 * (f1.v[index+1] + f1.v[index])) > 0 ? f1.m[index] : f1.m[index_yp])

        + (0.5 * (f1.w[index-1] + f1.w[index])) * ((0.5 * (f1.w[index-1] + f1.w[index])) > 0 ? f1.m[index_zm] : f1.m[index])

    );
    // Update velocity field

    f2.u[index] = (f1.m[index] * f1.u[index]) / f2.m[index] + (dt / f2.m[index]) * (
        (1/dx) * (

            + (0.5 * (f1.u[index-1] + f1.u[index])) * ((0.5 * (f1.u[index-1] + f1.u[index])) > 0 ? (f1.m[index_xm] * f1.u[index_xm]) : (f1.m[index] * f1.u[index]))

            - (0.5 * (f1.u[index+1] + f1.u[index])) * ((0.5 * (f1.u[index+1] + f1.u[index])) > 0 ? (f1.m[index] * f1.u[index]) : (f1.m[index_xp] * f1.u[index_xp]))

            + (0.5 * (f1.v[index-1] + f1.v[index])) * ((0.5 * (f1.v[index-1] + f1.v[index])) > 0 ? (f1.m[index_ym] * f1.u[index_ym]) : (f1.m[index] * f1.u[index]))

            - (0.5 * (f1.v[index+1] + f1.v[index])) * ((0.5 * (f1.v[index+1] + f1.v[index])) > 0 ? (f1.m[index] * f1.u[index]) : (f1.m[index_yp] * f1.u[index_yp]))

            + (0.5 * (f1.w[index-1] + f1.w[index])) * ((0.5 * (f1.w[index-1] + f1.w[index])) > 0 ? (f1.m[index_zm] * f1.u[index_zm]) : (f1.m[index] * f1.u[index]))

        ) - (RT/dx) * (

            + f1.m[index_xm]

            - f1.m[index_xp]

        ) + dx * mu * (

            + (f1.u[index_xm] - f1.u[index])

            + (f1.u[index_xp] - f1.u[index])

            + (f1.u[index_ym] - f1.u[index])

        )
    );


    f2.v[index] = (f1.m[index] * f1.v[index]) / f2.m[index] + (dt / f2.m[index]) * (
        (1/dx) * (

            + (0.5 * (f1.u[index-1] + f1.u[index])) * ((0.5 * (f1.u[index-1] + f1.u[index])) > 0 ? (f1.m[index_xm] * f1.v[index_xm]) : (f1.m[index] * f1.v[index]))

            - (0.5 * (f1.u[index+1] + f1.u[index])) * ((0.5 * (f1.u[index+1] + f1.u[index])) > 0 ? (f1.m[index] * f1.v[index]) : (f1.m[index_xp] * f1.v[index_xp]))

            + (0.5 * (f1.v[index-1] + f1.v[index])) * ((0.5 * (f1.v[index-1] + f1.v[index])) > 0 ? (f1.m[index_ym] * f1.v[index_ym]) : (f1.m[index] * f1.v[index]))

            - (0.5 * (f1.v[index+1] + f1.v[index])) * ((0.5 * (f1.v[index+1] + f1.v[index])) > 0 ? (f1.m[index] * f1.v[index]) : (f1.m[index_yp] * f1.v[index_yp]))

            + (0.5 * (f1.w[index-1] + f1.w[index])) * ((0.5 * (f1.w[index-1] + f1.w[index])) > 0 ? (f1.m[index_zm] * f1.v[index_zm]) : (f1.m[index] * f1.v[index]))

        ) - (RT/dx) * (

            + f1.m[index_ym]

            - f1.m[index_yp]

        ) + dx * mu * (

            + (f1.v[index_xm] - f1.v[index])

            + (f1.v[index_xp] - f1.v[index])

            + (f1.v[index_ym] - f1.v[index])

        )
    );

    f2.w[index] = 0;

}

__global__ void updateEdgeX1Y1(data d) {
    int totalThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if(totalThreadIndex >= 1) return;


    int x = 0;
    int y = 0;
    int z = totalThreadIndex + 1;
    int index = x + y * 3 + z * 3 * 3;
    int index_xp = (x+1) + y * 3 + z * 3 * 3;
    int index_yp = x + (y+1) * 3 + z * 3 * 3;
    int index_zm = x + y * 3 + (z-1) * 3 * 3;
    int index_zp = x + y * 3 + (z+1) * 3 * 3;



    // Define origin and destination fields
    field f1 = *d.inMain ? d.f1 : d.f2;
    field f2 = *d.inMain ? d.f2 : d.f1;

    // Define dx, dt and RT
    float dx = *d.dx;
    float dt = *d.dt;
    float RT = *d.RT;
    float mu = *d.mu;
    float g = *d.g;

    // Update mass field

    f2.m[index] = f1.m[index] + dt/dx * (

        - (0.5 * (f1.u[index+1] + f1.u[index])) * ((0.5 * (f1.u[index+1] + f1.u[index])) > 0 ? f1.m[index] : f1.m[index_xp])

        - (0.5 * (f1.v[index+1] + f1.v[index])) * ((0.5 * (f1.v[index+1] + f1.v[index])) > 0 ? f1.m[index] : f1.m[index_yp])

        + (0.5 * (f1.w[index-1] + f1.w[index])) * ((0.5 * (f1.w[index-1] + f1.w[index])) > 0 ? f1.m[index_zm] : f1.m[index])

        - (0.5 * (f1.w[index+1] + f1.w[index])) * ((0.5 * (f1.w[index+1] + f1.w[index])) > 0 ? f1.m[index] : f1.m[index_zp])

    );
    // Update velocity field
    f2.u[index] = 0;
    f2.v[index] = 0;

    f2.w[index] = (f1.m[index] * f1.w[index]) / f2.m[index] + (dt / f2.m[index]) * (
        (1/dx) * (

            - (0.5 * (f1.u[index+1] + f1.u[index])) * ((0.5 * (f1.u[index+1] + f1.u[index])) > 0 ? (f1.m[index] * f1.w[index]) : (f1.m[index_xp] * f1.w[index_xp]))

            - (0.5 * (f1.v[index+1] + f1.v[index])) * ((0.5 * (f1.v[index+1] + f1.v[index])) > 0 ? (f1.m[index] * f1.w[index]) : (f1.m[index_yp] * f1.w[index_yp]))

            + (0.5 * (f1.w[index-1] + f1.w[index])) * ((0.5 * (f1.w[index-1] + f1.w[index])) > 0 ? (f1.m[index_zm] * f1.w[index_zm]) : (f1.m[index] * f1.w[index]))

            - (0.5 * (f1.w[index+1] + f1.w[index])) * ((0.5 * (f1.w[index+1] + f1.w[index])) > 0 ? (f1.m[index] * f1.w[index]) : (f1.m[index_zp] * f1.w[index_zp]))

        ) - (RT/dx) * (

            + f1.m[index_zm]

            - f1.m[index_zp]

        ) + dx * mu * (

            + (f1.w[index_xp] - f1.w[index])

        ) + f1.m[index] * g
    );


}

__global__ void updateEdgeX1Y2(data d) {
    int totalThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if(totalThreadIndex >= 1) return;


    int x = 0;
    int y = 2;
    int z = totalThreadIndex + 1;
    int index = x + y * 3 + z * 3 * 3;
    int index_xp = (x+1) + y * 3 + z * 3 * 3;
    int index_ym = x + (y-1) * 3 + z * 3 * 3;
    int index_zm = x + y * 3 + (z-1) * 3 * 3;
    int index_zp = x + y * 3 + (z+1) * 3 * 3;



    // Define origin and destination fields
    field f1 = *d.inMain ? d.f1 : d.f2;
    field f2 = *d.inMain ? d.f2 : d.f1;

    // Define dx, dt and RT
    float dx = *d.dx;
    float dt = *d.dt;
    float RT = *d.RT;
    float mu = *d.mu;
    float g = *d.g;

    // Update mass field

    f2.m[index] = f1.m[index] + dt/dx * (

        - (0.5 * (f1.u[index+1] + f1.u[index])) * ((0.5 * (f1.u[index+1] + f1.u[index])) > 0 ? f1.m[index] : f1.m[index_xp])

        + (0.5 * (f1.v[index-1] + f1.v[index])) * ((0.5 * (f1.v[index-1] + f1.v[index])) > 0 ? f1.m[index_ym] : f1.m[index])

        + (0.5 * (f1.w[index-1] + f1.w[index])) * ((0.5 * (f1.w[index-1] + f1.w[index])) > 0 ? f1.m[index_zm] : f1.m[index])

        - (0.5 * (f1.w[index+1] + f1.w[index])) * ((0.5 * (f1.w[index+1] + f1.w[index])) > 0 ? f1.m[index] : f1.m[index_zp])

    );
    // Update velocity field
    f2.u[index] = 0;
    f2.v[index] = 0;

    f2.w[index] = (f1.m[index] * f1.w[index]) / f2.m[index] + (dt / f2.m[index]) * (
        (1/dx) * (

            - (0.5 * (f1.u[index+1] + f1.u[index])) * ((0.5 * (f1.u[index+1] + f1.u[index])) > 0 ? (f1.m[index] * f1.w[index]) : (f1.m[index_xp] * f1.w[index_xp]))

            + (0.5 * (f1.v[index-1] + f1.v[index])) * ((0.5 * (f1.v[index-1] + f1.v[index])) > 0 ? (f1.m[index_ym] * f1.w[index_ym]) : (f1.m[index] * f1.w[index]))

            + (0.5 * (f1.w[index-1] + f1.w[index])) * ((0.5 * (f1.w[index-1] + f1.w[index])) > 0 ? (f1.m[index_zm] * f1.w[index_zm]) : (f1.m[index] * f1.w[index]))

            - (0.5 * (f1.w[index+1] + f1.w[index])) * ((0.5 * (f1.w[index+1] + f1.w[index])) > 0 ? (f1.m[index] * f1.w[index]) : (f1.m[index_zp] * f1.w[index_zp]))

        ) - (RT/dx) * (

            + f1.m[index_zm]

            - f1.m[index_zp]

        ) + dx * mu * (

            + (f1.w[index_xp] - f1.w[index])

            + (f1.w[index_ym] - f1.w[index])

        ) + f1.m[index] * g
    );


}

__global__ void updateEdgeX2Y1(data d) {
    int totalThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if(totalThreadIndex >= 1) return;


    int x = 2;
    int y = 0;
    int z = totalThreadIndex + 1;
    int index = x + y * 3 + z * 3 * 3;
    int index_xm = (x-1) + y * 3 + z * 3 * 3;
    int index_yp = x + (y+1) * 3 + z * 3 * 3;
    int index_zm = x + y * 3 + (z-1) * 3 * 3;
    int index_zp = x + y * 3 + (z+1) * 3 * 3;



    // Define origin and destination fields
    field f1 = *d.inMain ? d.f1 : d.f2;
    field f2 = *d.inMain ? d.f2 : d.f1;

    // Define dx, dt and RT
    float dx = *d.dx;
    float dt = *d.dt;
    float RT = *d.RT;
    float mu = *d.mu;
    float g = *d.g;

    // Update mass field

    f2.m[index] = f1.m[index] + dt/dx * (

        + (0.5 * (f1.u[index-1] + f1.u[index])) * ((0.5 * (f1.u[index-1] + f1.u[index])) > 0 ? f1.m[index_xm] : f1.m[index])

        - (0.5 * (f1.v[index+1] + f1.v[index])) * ((0.5 * (f1.v[index+1] + f1.v[index])) > 0 ? f1.m[index] : f1.m[index_yp])

        + (0.5 * (f1.w[index-1] + f1.w[index])) * ((0.5 * (f1.w[index-1] + f1.w[index])) > 0 ? f1.m[index_zm] : f1.m[index])

        - (0.5 * (f1.w[index+1] + f1.w[index])) * ((0.5 * (f1.w[index+1] + f1.w[index])) > 0 ? f1.m[index] : f1.m[index_zp])

    );
    // Update velocity field
    f2.u[index] = 0;
    f2.v[index] = 0;

    f2.w[index] = (f1.m[index] * f1.w[index]) / f2.m[index] + (dt / f2.m[index]) * (
        (1/dx) * (

            + (0.5 * (f1.u[index-1] + f1.u[index])) * ((0.5 * (f1.u[index-1] + f1.u[index])) > 0 ? (f1.m[index_xm] * f1.w[index_xm]) : (f1.m[index] * f1.w[index]))

            - (0.5 * (f1.v[index+1] + f1.v[index])) * ((0.5 * (f1.v[index+1] + f1.v[index])) > 0 ? (f1.m[index] * f1.w[index]) : (f1.m[index_yp] * f1.w[index_yp]))

            + (0.5 * (f1.w[index-1] + f1.w[index])) * ((0.5 * (f1.w[index-1] + f1.w[index])) > 0 ? (f1.m[index_zm] * f1.w[index_zm]) : (f1.m[index] * f1.w[index]))

            - (0.5 * (f1.w[index+1] + f1.w[index])) * ((0.5 * (f1.w[index+1] + f1.w[index])) > 0 ? (f1.m[index] * f1.w[index]) : (f1.m[index_zp] * f1.w[index_zp]))

        ) - (RT/dx) * (

            + f1.m[index_zm]

            - f1.m[index_zp]

        ) + dx * mu * (

            + (f1.w[index_xm] - f1.w[index])

        ) + f1.m[index] * g
    );


}

__global__ void updateEdgeX2Y2(data d) {
    int totalThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if(totalThreadIndex >= 1) return;


    int x = 2;
    int y = 2;
    int z = totalThreadIndex + 1;
    int index = x + y * 3 + z * 3 * 3;
    int index_xm = (x-1) + y * 3 + z * 3 * 3;
    int index_ym = x + (y-1) * 3 + z * 3 * 3;
    int index_zm = x + y * 3 + (z-1) * 3 * 3;
    int index_zp = x + y * 3 + (z+1) * 3 * 3;



    // Define origin and destination fields
    field f1 = *d.inMain ? d.f1 : d.f2;
    field f2 = *d.inMain ? d.f2 : d.f1;

    // Define dx, dt and RT
    float dx = *d.dx;
    float dt = *d.dt;
    float RT = *d.RT;
    float mu = *d.mu;
    float g = *d.g;

    // Update mass field

    f2.m[index] = f1.m[index] + dt/dx * (

        + (0.5 * (f1.u[index-1] + f1.u[index])) * ((0.5 * (f1.u[index-1] + f1.u[index])) > 0 ? f1.m[index_xm] : f1.m[index])

        + (0.5 * (f1.v[index-1] + f1.v[index])) * ((0.5 * (f1.v[index-1] + f1.v[index])) > 0 ? f1.m[index_ym] : f1.m[index])

        + (0.5 * (f1.w[index-1] + f1.w[index])) * ((0.5 * (f1.w[index-1] + f1.w[index])) > 0 ? f1.m[index_zm] : f1.m[index])

        - (0.5 * (f1.w[index+1] + f1.w[index])) * ((0.5 * (f1.w[index+1] + f1.w[index])) > 0 ? f1.m[index] : f1.m[index_zp])

    );
    // Update velocity field
    f2.u[index] = 0;
    f2.v[index] = 0;

    f2.w[index] = (f1.m[index] * f1.w[index]) / f2.m[index] + (dt / f2.m[index]) * (
        (1/dx) * (

            + (0.5 * (f1.u[index-1] + f1.u[index])) * ((0.5 * (f1.u[index-1] + f1.u[index])) > 0 ? (f1.m[index_xm] * f1.w[index_xm]) : (f1.m[index] * f1.w[index]))

            + (0.5 * (f1.v[index-1] + f1.v[index])) * ((0.5 * (f1.v[index-1] + f1.v[index])) > 0 ? (f1.m[index_ym] * f1.w[index_ym]) : (f1.m[index] * f1.w[index]))

            + (0.5 * (f1.w[index-1] + f1.w[index])) * ((0.5 * (f1.w[index-1] + f1.w[index])) > 0 ? (f1.m[index_zm] * f1.w[index_zm]) : (f1.m[index] * f1.w[index]))

            - (0.5 * (f1.w[index+1] + f1.w[index])) * ((0.5 * (f1.w[index+1] + f1.w[index])) > 0 ? (f1.m[index] * f1.w[index]) : (f1.m[index_zp] * f1.w[index_zp]))

        ) - (RT/dx) * (

            + f1.m[index_zm]

            - f1.m[index_zp]

        ) + dx * mu * (

            + (f1.w[index_xm] - f1.w[index])

            + (f1.w[index_ym] - f1.w[index])

        ) + f1.m[index] * g
    );


}

__global__ void updateEdgeX1Z1(data d) {
    int totalThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if(totalThreadIndex >= 1) return;


    int x = 0;
    int y = totalThreadIndex + 1;
    int z = 0;
    int index = x + y * 3 + z * 3 * 3;
    int index_xp = (x+1) + y * 3 + z * 3 * 3;
    int index_ym = x + (y-1) * 3 + z * 3 * 3;
    int index_yp = x + (y+1) * 3 + z * 3 * 3;
    int index_zp = x + y * 3 + (z+1) * 3 * 3;



    // Define origin and destination fields
    field f1 = *d.inMain ? d.f1 : d.f2;
    field f2 = *d.inMain ? d.f2 : d.f1;

    // Define dx, dt and RT
    float dx = *d.dx;
    float dt = *d.dt;
    float RT = *d.RT;
    float mu = *d.mu;
    float g = *d.g;

    // Update mass field

    f2.m[index] = f1.m[index] + dt/dx * (

        - (0.5 * (f1.u[index+1] + f1.u[index])) * ((0.5 * (f1.u[index+1] + f1.u[index])) > 0 ? f1.m[index] : f1.m[index_xp])

        + (0.5 * (f1.v[index-1] + f1.v[index])) * ((0.5 * (f1.v[index-1] + f1.v[index])) > 0 ? f1.m[index_ym] : f1.m[index])

        - (0.5 * (f1.v[index+1] + f1.v[index])) * ((0.5 * (f1.v[index+1] + f1.v[index])) > 0 ? f1.m[index] : f1.m[index_yp])

        - (0.5 * (f1.w[index+1] + f1.w[index])) * ((0.5 * (f1.w[index+1] + f1.w[index])) > 0 ? f1.m[index] : f1.m[index_zp])

    );
    // Update velocity field
    f2.u[index] = 0;

    f2.v[index] = (f1.m[index] * f1.v[index]) / f2.m[index] + (dt / f2.m[index]) * (
        (1/dx) * (

            - (0.5 * (f1.u[index+1] + f1.u[index])) * ((0.5 * (f1.u[index+1] + f1.u[index])) > 0 ? (f1.m[index] * f1.v[index]) : (f1.m[index_xp] * f1.v[index_xp]))

            + (0.5 * (f1.v[index-1] + f1.v[index])) * ((0.5 * (f1.v[index-1] + f1.v[index])) > 0 ? (f1.m[index_ym] * f1.v[index_ym]) : (f1.m[index] * f1.v[index]))

            - (0.5 * (f1.v[index+1] + f1.v[index])) * ((0.5 * (f1.v[index+1] + f1.v[index])) > 0 ? (f1.m[index] * f1.v[index]) : (f1.m[index_yp] * f1.v[index_yp]))

            - (0.5 * (f1.w[index+1] + f1.w[index])) * ((0.5 * (f1.w[index+1] + f1.w[index])) > 0 ? (f1.m[index] * f1.v[index]) : (f1.m[index_zp] * f1.v[index_zp]))

        ) - (RT/dx) * (

            + f1.m[index_ym]

            - f1.m[index_yp]

        ) + dx * mu * (

            + (f1.v[index_xp] - f1.v[index])

            + (f1.v[index_ym] - f1.v[index])

        )
    );

    f2.w[index] = 0;

}

__global__ void updateEdgeX1Z2(data d) {
    int totalThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if(totalThreadIndex >= 1) return;


    int x = 0;
    int y = totalThreadIndex + 1;
    int z = 2;
    int index = x + y * 3 + z * 3 * 3;
    int index_xp = (x+1) + y * 3 + z * 3 * 3;
    int index_ym = x + (y-1) * 3 + z * 3 * 3;
    int index_yp = x + (y+1) * 3 + z * 3 * 3;
    int index_zm = x + y * 3 + (z-1) * 3 * 3;



    // Define origin and destination fields
    field f1 = *d.inMain ? d.f1 : d.f2;
    field f2 = *d.inMain ? d.f2 : d.f1;

    // Define dx, dt and RT
    float dx = *d.dx;
    float dt = *d.dt;
    float RT = *d.RT;
    float mu = *d.mu;
    float g = *d.g;

    // Update mass field

    f2.m[index] = f1.m[index] + dt/dx * (

        - (0.5 * (f1.u[index+1] + f1.u[index])) * ((0.5 * (f1.u[index+1] + f1.u[index])) > 0 ? f1.m[index] : f1.m[index_xp])

        + (0.5 * (f1.v[index-1] + f1.v[index])) * ((0.5 * (f1.v[index-1] + f1.v[index])) > 0 ? f1.m[index_ym] : f1.m[index])

        - (0.5 * (f1.v[index+1] + f1.v[index])) * ((0.5 * (f1.v[index+1] + f1.v[index])) > 0 ? f1.m[index] : f1.m[index_yp])

        + (0.5 * (f1.w[index-1] + f1.w[index])) * ((0.5 * (f1.w[index-1] + f1.w[index])) > 0 ? f1.m[index_zm] : f1.m[index])

    );
    // Update velocity field
    f2.u[index] = 0;

    f2.v[index] = (f1.m[index] * f1.v[index]) / f2.m[index] + (dt / f2.m[index]) * (
        (1/dx) * (

            - (0.5 * (f1.u[index+1] + f1.u[index])) * ((0.5 * (f1.u[index+1] + f1.u[index])) > 0 ? (f1.m[index] * f1.v[index]) : (f1.m[index_xp] * f1.v[index_xp]))

            + (0.5 * (f1.v[index-1] + f1.v[index])) * ((0.5 * (f1.v[index-1] + f1.v[index])) > 0 ? (f1.m[index_ym] * f1.v[index_ym]) : (f1.m[index] * f1.v[index]))

            - (0.5 * (f1.v[index+1] + f1.v[index])) * ((0.5 * (f1.v[index+1] + f1.v[index])) > 0 ? (f1.m[index] * f1.v[index]) : (f1.m[index_yp] * f1.v[index_yp]))

            + (0.5 * (f1.w[index-1] + f1.w[index])) * ((0.5 * (f1.w[index-1] + f1.w[index])) > 0 ? (f1.m[index_zm] * f1.v[index_zm]) : (f1.m[index] * f1.v[index]))

        ) - (RT/dx) * (

            + f1.m[index_ym]

            - f1.m[index_yp]

        ) + dx * mu * (

            + (f1.v[index_xp] - f1.v[index])

            + (f1.v[index_ym] - f1.v[index])

        )
    );

    f2.w[index] = 0;

}

__global__ void updateEdgeX2Z1(data d) {
    int totalThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if(totalThreadIndex >= 1) return;


    int x = 2;
    int y = totalThreadIndex + 1;
    int z = 0;
    int index = x + y * 3 + z * 3 * 3;
    int index_xm = (x-1) + y * 3 + z * 3 * 3;
    int index_ym = x + (y-1) * 3 + z * 3 * 3;
    int index_yp = x + (y+1) * 3 + z * 3 * 3;
    int index_zp = x + y * 3 + (z+1) * 3 * 3;



    // Define origin and destination fields
    field f1 = *d.inMain ? d.f1 : d.f2;
    field f2 = *d.inMain ? d.f2 : d.f1;

    // Define dx, dt and RT
    float dx = *d.dx;
    float dt = *d.dt;
    float RT = *d.RT;
    float mu = *d.mu;
    float g = *d.g;

    // Update mass field

    f2.m[index] = f1.m[index] + dt/dx * (

        + (0.5 * (f1.u[index-1] + f1.u[index])) * ((0.5 * (f1.u[index-1] + f1.u[index])) > 0 ? f1.m[index_xm] : f1.m[index])

        + (0.5 * (f1.v[index-1] + f1.v[index])) * ((0.5 * (f1.v[index-1] + f1.v[index])) > 0 ? f1.m[index_ym] : f1.m[index])

        - (0.5 * (f1.v[index+1] + f1.v[index])) * ((0.5 * (f1.v[index+1] + f1.v[index])) > 0 ? f1.m[index] : f1.m[index_yp])

        - (0.5 * (f1.w[index+1] + f1.w[index])) * ((0.5 * (f1.w[index+1] + f1.w[index])) > 0 ? f1.m[index] : f1.m[index_zp])

    );
    // Update velocity field
    f2.u[index] = 0;

    f2.v[index] = (f1.m[index] * f1.v[index]) / f2.m[index] + (dt / f2.m[index]) * (
        (1/dx) * (

            + (0.5 * (f1.u[index-1] + f1.u[index])) * ((0.5 * (f1.u[index-1] + f1.u[index])) > 0 ? (f1.m[index_xm] * f1.v[index_xm]) : (f1.m[index] * f1.v[index]))

            + (0.5 * (f1.v[index-1] + f1.v[index])) * ((0.5 * (f1.v[index-1] + f1.v[index])) > 0 ? (f1.m[index_ym] * f1.v[index_ym]) : (f1.m[index] * f1.v[index]))

            - (0.5 * (f1.v[index+1] + f1.v[index])) * ((0.5 * (f1.v[index+1] + f1.v[index])) > 0 ? (f1.m[index] * f1.v[index]) : (f1.m[index_yp] * f1.v[index_yp]))

            - (0.5 * (f1.w[index+1] + f1.w[index])) * ((0.5 * (f1.w[index+1] + f1.w[index])) > 0 ? (f1.m[index] * f1.v[index]) : (f1.m[index_zp] * f1.v[index_zp]))

        ) - (RT/dx) * (

            + f1.m[index_ym]

            - f1.m[index_yp]

        ) + dx * mu * (

            + (f1.v[index_xm] - f1.v[index])

            + (f1.v[index_ym] - f1.v[index])

        )
    );

    f2.w[index] = 0;

}

__global__ void updateEdgeX2Z2(data d) {
    int totalThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if(totalThreadIndex >= 1) return;


    int x = 2;
    int y = totalThreadIndex + 1;
    int z = 2;
    int index = x + y * 3 + z * 3 * 3;
    int index_xm = (x-1) + y * 3 + z * 3 * 3;
    int index_ym = x + (y-1) * 3 + z * 3 * 3;
    int index_yp = x + (y+1) * 3 + z * 3 * 3;
    int index_zm = x + y * 3 + (z-1) * 3 * 3;



    // Define origin and destination fields
    field f1 = *d.inMain ? d.f1 : d.f2;
    field f2 = *d.inMain ? d.f2 : d.f1;

    // Define dx, dt and RT
    float dx = *d.dx;
    float dt = *d.dt;
    float RT = *d.RT;
    float mu = *d.mu;
    float g = *d.g;

    // Update mass field

    f2.m[index] = f1.m[index] + dt/dx * (

        + (0.5 * (f1.u[index-1] + f1.u[index])) * ((0.5 * (f1.u[index-1] + f1.u[index])) > 0 ? f1.m[index_xm] : f1.m[index])

        + (0.5 * (f1.v[index-1] + f1.v[index])) * ((0.5 * (f1.v[index-1] + f1.v[index])) > 0 ? f1.m[index_ym] : f1.m[index])

        - (0.5 * (f1.v[index+1] + f1.v[index])) * ((0.5 * (f1.v[index+1] + f1.v[index])) > 0 ? f1.m[index] : f1.m[index_yp])

        + (0.5 * (f1.w[index-1] + f1.w[index])) * ((0.5 * (f1.w[index-1] + f1.w[index])) > 0 ? f1.m[index_zm] : f1.m[index])

    );
    // Update velocity field
    f2.u[index] = 0;

    f2.v[index] = (f1.m[index] * f1.v[index]) / f2.m[index] + (dt / f2.m[index]) * (
        (1/dx) * (

            + (0.5 * (f1.u[index-1] + f1.u[index])) * ((0.5 * (f1.u[index-1] + f1.u[index])) > 0 ? (f1.m[index_xm] * f1.v[index_xm]) : (f1.m[index] * f1.v[index]))

            + (0.5 * (f1.v[index-1] + f1.v[index])) * ((0.5 * (f1.v[index-1] + f1.v[index])) > 0 ? (f1.m[index_ym] * f1.v[index_ym]) : (f1.m[index] * f1.v[index]))

            - (0.5 * (f1.v[index+1] + f1.v[index])) * ((0.5 * (f1.v[index+1] + f1.v[index])) > 0 ? (f1.m[index] * f1.v[index]) : (f1.m[index_yp] * f1.v[index_yp]))

            + (0.5 * (f1.w[index-1] + f1.w[index])) * ((0.5 * (f1.w[index-1] + f1.w[index])) > 0 ? (f1.m[index_zm] * f1.v[index_zm]) : (f1.m[index] * f1.v[index]))

        ) - (RT/dx) * (

            + f1.m[index_ym]

            - f1.m[index_yp]

        ) + dx * mu * (

            + (f1.v[index_xm] - f1.v[index])

            + (f1.v[index_ym] - f1.v[index])

        )
    );

    f2.w[index] = 0;

}

__global__ void updateEdgeZ1X1(data d) {
    int totalThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if(totalThreadIndex >= 1) return;


    int x = 0;
    int y = totalThreadIndex + 1;
    int z = 0;
    int index = x + y * 3 + z * 3 * 3;
    int index_xp = (x+1) + y * 3 + z * 3 * 3;
    int index_ym = x + (y-1) * 3 + z * 3 * 3;
    int index_yp = x + (y+1) * 3 + z * 3 * 3;
    int index_zp = x + y * 3 + (z+1) * 3 * 3;



    // Define origin and destination fields
    field f1 = *d.inMain ? d.f1 : d.f2;
    field f2 = *d.inMain ? d.f2 : d.f1;

    // Define dx, dt and RT
    float dx = *d.dx;
    float dt = *d.dt;
    float RT = *d.RT;
    float mu = *d.mu;
    float g = *d.g;

    // Update mass field

    f2.m[index] = f1.m[index] + dt/dx * (

        - (0.5 * (f1.u[index+1] + f1.u[index])) * ((0.5 * (f1.u[index+1] + f1.u[index])) > 0 ? f1.m[index] : f1.m[index_xp])

        + (0.5 * (f1.v[index-1] + f1.v[index])) * ((0.5 * (f1.v[index-1] + f1.v[index])) > 0 ? f1.m[index_ym] : f1.m[index])

        - (0.5 * (f1.v[index+1] + f1.v[index])) * ((0.5 * (f1.v[index+1] + f1.v[index])) > 0 ? f1.m[index] : f1.m[index_yp])

        - (0.5 * (f1.w[index+1] + f1.w[index])) * ((0.5 * (f1.w[index+1] + f1.w[index])) > 0 ? f1.m[index] : f1.m[index_zp])

    );
    // Update velocity field
    f2.u[index] = 0;

    f2.v[index] = (f1.m[index] * f1.v[index]) / f2.m[index] + (dt / f2.m[index]) * (
        (1/dx) * (

            - (0.5 * (f1.u[index+1] + f1.u[index])) * ((0.5 * (f1.u[index+1] + f1.u[index])) > 0 ? (f1.m[index] * f1.v[index]) : (f1.m[index_xp] * f1.v[index_xp]))

            + (0.5 * (f1.v[index-1] + f1.v[index])) * ((0.5 * (f1.v[index-1] + f1.v[index])) > 0 ? (f1.m[index_ym] * f1.v[index_ym]) : (f1.m[index] * f1.v[index]))

            - (0.5 * (f1.v[index+1] + f1.v[index])) * ((0.5 * (f1.v[index+1] + f1.v[index])) > 0 ? (f1.m[index] * f1.v[index]) : (f1.m[index_yp] * f1.v[index_yp]))

            - (0.5 * (f1.w[index+1] + f1.w[index])) * ((0.5 * (f1.w[index+1] + f1.w[index])) > 0 ? (f1.m[index] * f1.v[index]) : (f1.m[index_zp] * f1.v[index_zp]))

        ) - (RT/dx) * (

            + f1.m[index_ym]

            - f1.m[index_yp]

        ) + dx * mu * (

            + (f1.v[index_xp] - f1.v[index])

            + (f1.v[index_ym] - f1.v[index])

        )
    );

    f2.w[index] = 0;

}

__global__ void updateEdgeZ1X2(data d) {
    int totalThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if(totalThreadIndex >= 1) return;


    int x = 2;
    int y = totalThreadIndex + 1;
    int z = 0;
    int index = x + y * 3 + z * 3 * 3;
    int index_xm = (x-1) + y * 3 + z * 3 * 3;
    int index_ym = x + (y-1) * 3 + z * 3 * 3;
    int index_yp = x + (y+1) * 3 + z * 3 * 3;
    int index_zp = x + y * 3 + (z+1) * 3 * 3;



    // Define origin and destination fields
    field f1 = *d.inMain ? d.f1 : d.f2;
    field f2 = *d.inMain ? d.f2 : d.f1;

    // Define dx, dt and RT
    float dx = *d.dx;
    float dt = *d.dt;
    float RT = *d.RT;
    float mu = *d.mu;
    float g = *d.g;

    // Update mass field

    f2.m[index] = f1.m[index] + dt/dx * (

        + (0.5 * (f1.u[index-1] + f1.u[index])) * ((0.5 * (f1.u[index-1] + f1.u[index])) > 0 ? f1.m[index_xm] : f1.m[index])

        + (0.5 * (f1.v[index-1] + f1.v[index])) * ((0.5 * (f1.v[index-1] + f1.v[index])) > 0 ? f1.m[index_ym] : f1.m[index])

        - (0.5 * (f1.v[index+1] + f1.v[index])) * ((0.5 * (f1.v[index+1] + f1.v[index])) > 0 ? f1.m[index] : f1.m[index_yp])

        - (0.5 * (f1.w[index+1] + f1.w[index])) * ((0.5 * (f1.w[index+1] + f1.w[index])) > 0 ? f1.m[index] : f1.m[index_zp])

    );
    // Update velocity field
    f2.u[index] = 0;

    f2.v[index] = (f1.m[index] * f1.v[index]) / f2.m[index] + (dt / f2.m[index]) * (
        (1/dx) * (

            + (0.5 * (f1.u[index-1] + f1.u[index])) * ((0.5 * (f1.u[index-1] + f1.u[index])) > 0 ? (f1.m[index_xm] * f1.v[index_xm]) : (f1.m[index] * f1.v[index]))

            + (0.5 * (f1.v[index-1] + f1.v[index])) * ((0.5 * (f1.v[index-1] + f1.v[index])) > 0 ? (f1.m[index_ym] * f1.v[index_ym]) : (f1.m[index] * f1.v[index]))

            - (0.5 * (f1.v[index+1] + f1.v[index])) * ((0.5 * (f1.v[index+1] + f1.v[index])) > 0 ? (f1.m[index] * f1.v[index]) : (f1.m[index_yp] * f1.v[index_yp]))

            - (0.5 * (f1.w[index+1] + f1.w[index])) * ((0.5 * (f1.w[index+1] + f1.w[index])) > 0 ? (f1.m[index] * f1.v[index]) : (f1.m[index_zp] * f1.v[index_zp]))

        ) - (RT/dx) * (

            + f1.m[index_ym]

            - f1.m[index_yp]

        ) + dx * mu * (

            + (f1.v[index_xm] - f1.v[index])

            + (f1.v[index_ym] - f1.v[index])

        )
    );

    f2.w[index] = 0;

}

__global__ void updateEdgeZ2X1(data d) {
    int totalThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if(totalThreadIndex >= 1) return;


    int x = 0;
    int y = totalThreadIndex + 1;
    int z = 2;
    int index = x + y * 3 + z * 3 * 3;
    int index_xp = (x+1) + y * 3 + z * 3 * 3;
    int index_ym = x + (y-1) * 3 + z * 3 * 3;
    int index_yp = x + (y+1) * 3 + z * 3 * 3;
    int index_zm = x + y * 3 + (z-1) * 3 * 3;



    // Define origin and destination fields
    field f1 = *d.inMain ? d.f1 : d.f2;
    field f2 = *d.inMain ? d.f2 : d.f1;

    // Define dx, dt and RT
    float dx = *d.dx;
    float dt = *d.dt;
    float RT = *d.RT;
    float mu = *d.mu;
    float g = *d.g;

    // Update mass field

    f2.m[index] = f1.m[index] + dt/dx * (

        - (0.5 * (f1.u[index+1] + f1.u[index])) * ((0.5 * (f1.u[index+1] + f1.u[index])) > 0 ? f1.m[index] : f1.m[index_xp])

        + (0.5 * (f1.v[index-1] + f1.v[index])) * ((0.5 * (f1.v[index-1] + f1.v[index])) > 0 ? f1.m[index_ym] : f1.m[index])

        - (0.5 * (f1.v[index+1] + f1.v[index])) * ((0.5 * (f1.v[index+1] + f1.v[index])) > 0 ? f1.m[index] : f1.m[index_yp])

        + (0.5 * (f1.w[index-1] + f1.w[index])) * ((0.5 * (f1.w[index-1] + f1.w[index])) > 0 ? f1.m[index_zm] : f1.m[index])

    );
    // Update velocity field
    f2.u[index] = 0;

    f2.v[index] = (f1.m[index] * f1.v[index]) / f2.m[index] + (dt / f2.m[index]) * (
        (1/dx) * (

            - (0.5 * (f1.u[index+1] + f1.u[index])) * ((0.5 * (f1.u[index+1] + f1.u[index])) > 0 ? (f1.m[index] * f1.v[index]) : (f1.m[index_xp] * f1.v[index_xp]))

            + (0.5 * (f1.v[index-1] + f1.v[index])) * ((0.5 * (f1.v[index-1] + f1.v[index])) > 0 ? (f1.m[index_ym] * f1.v[index_ym]) : (f1.m[index] * f1.v[index]))

            - (0.5 * (f1.v[index+1] + f1.v[index])) * ((0.5 * (f1.v[index+1] + f1.v[index])) > 0 ? (f1.m[index] * f1.v[index]) : (f1.m[index_yp] * f1.v[index_yp]))

            + (0.5 * (f1.w[index-1] + f1.w[index])) * ((0.5 * (f1.w[index-1] + f1.w[index])) > 0 ? (f1.m[index_zm] * f1.v[index_zm]) : (f1.m[index] * f1.v[index]))

        ) - (RT/dx) * (

            + f1.m[index_ym]

            - f1.m[index_yp]

        ) + dx * mu * (

            + (f1.v[index_xp] - f1.v[index])

            + (f1.v[index_ym] - f1.v[index])

        )
    );

    f2.w[index] = 0;

}

__global__ void updateEdgeZ2X2(data d) {
    int totalThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if(totalThreadIndex >= 1) return;


    int x = 2;
    int y = totalThreadIndex + 1;
    int z = 2;
    int index = x + y * 3 + z * 3 * 3;
    int index_xm = (x-1) + y * 3 + z * 3 * 3;
    int index_ym = x + (y-1) * 3 + z * 3 * 3;
    int index_yp = x + (y+1) * 3 + z * 3 * 3;
    int index_zm = x + y * 3 + (z-1) * 3 * 3;



    // Define origin and destination fields
    field f1 = *d.inMain ? d.f1 : d.f2;
    field f2 = *d.inMain ? d.f2 : d.f1;

    // Define dx, dt and RT
    float dx = *d.dx;
    float dt = *d.dt;
    float RT = *d.RT;
    float mu = *d.mu;
    float g = *d.g;

    // Update mass field

    f2.m[index] = f1.m[index] + dt/dx * (

        + (0.5 * (f1.u[index-1] + f1.u[index])) * ((0.5 * (f1.u[index-1] + f1.u[index])) > 0 ? f1.m[index_xm] : f1.m[index])

        + (0.5 * (f1.v[index-1] + f1.v[index])) * ((0.5 * (f1.v[index-1] + f1.v[index])) > 0 ? f1.m[index_ym] : f1.m[index])

        - (0.5 * (f1.v[index+1] + f1.v[index])) * ((0.5 * (f1.v[index+1] + f1.v[index])) > 0 ? f1.m[index] : f1.m[index_yp])

        + (0.5 * (f1.w[index-1] + f1.w[index])) * ((0.5 * (f1.w[index-1] + f1.w[index])) > 0 ? f1.m[index_zm] : f1.m[index])

    );
    // Update velocity field
    f2.u[index] = 0;

    f2.v[index] = (f1.m[index] * f1.v[index]) / f2.m[index] + (dt / f2.m[index]) * (
        (1/dx) * (

            + (0.5 * (f1.u[index-1] + f1.u[index])) * ((0.5 * (f1.u[index-1] + f1.u[index])) > 0 ? (f1.m[index_xm] * f1.v[index_xm]) : (f1.m[index] * f1.v[index]))

            + (0.5 * (f1.v[index-1] + f1.v[index])) * ((0.5 * (f1.v[index-1] + f1.v[index])) > 0 ? (f1.m[index_ym] * f1.v[index_ym]) : (f1.m[index] * f1.v[index]))

            - (0.5 * (f1.v[index+1] + f1.v[index])) * ((0.5 * (f1.v[index+1] + f1.v[index])) > 0 ? (f1.m[index] * f1.v[index]) : (f1.m[index_yp] * f1.v[index_yp]))

            + (0.5 * (f1.w[index-1] + f1.w[index])) * ((0.5 * (f1.w[index-1] + f1.w[index])) > 0 ? (f1.m[index_zm] * f1.v[index_zm]) : (f1.m[index] * f1.v[index]))

        ) - (RT/dx) * (

            + f1.m[index_ym]

            - f1.m[index_yp]

        ) + dx * mu * (

            + (f1.v[index_xm] - f1.v[index])

            + (f1.v[index_ym] - f1.v[index])

        )
    );

    f2.w[index] = 0;

}

__global__ void updateCornerZ1X1Y1(data d) {

    int x = 0;
    int y = 0;
    int z = 0;
    int index = x + y * 3 + z * 3 * 3;
    int index_xp = (x+1) + y * 3 + z * 3 * 3;
    int index_yp = x + (y+1) * 3 + z * 3 * 3;
    int index_zp = x + y * 3 + (z+1) * 3 * 3;



    // Define origin and destination fields
    field f1 = *d.inMain ? d.f1 : d.f2;
    field f2 = *d.inMain ? d.f2 : d.f1;

    // Define dx, dt and RT
    float dx = *d.dx;
    float dt = *d.dt;
    float RT = *d.RT;
    float mu = *d.mu;
    float g = *d.g;

    // Update mass field

    f2.m[index] = f1.m[index] + dt/dx * (

        - (0.5 * (f1.u[index+1] + f1.u[index])) * ((0.5 * (f1.u[index+1] + f1.u[index])) > 0 ? f1.m[index] : f1.m[index_xp])

        - (0.5 * (f1.v[index+1] + f1.v[index])) * ((0.5 * (f1.v[index+1] + f1.v[index])) > 0 ? f1.m[index] : f1.m[index_yp])

        - (0.5 * (f1.w[index+1] + f1.w[index])) * ((0.5 * (f1.w[index+1] + f1.w[index])) > 0 ? f1.m[index] : f1.m[index_zp])

    );
    // Update velocity field
    f2.u[index] = 0;
    f2.v[index] = 0;
    f2.w[index] = 0;

}

__global__ void updateCornerZ1X1Y2(data d) {

    int x = 0;
    int y = 3 - 1;
    int z = 0;
    int index = x + y * 3 + z * 3 * 3;
    int index_xp = (x+1) + y * 3 + z * 3 * 3;
    int index_ym = x + (y-1) * 3 + z * 3 * 3;
    int index_zp = x + y * 3 + (z+1) * 3 * 3;



    // Define origin and destination fields
    field f1 = *d.inMain ? d.f1 : d.f2;
    field f2 = *d.inMain ? d.f2 : d.f1;

    // Define dx, dt and RT
    float dx = *d.dx;
    float dt = *d.dt;
    float RT = *d.RT;
    float mu = *d.mu;
    float g = *d.g;

    // Update mass field

    f2.m[index] = f1.m[index] + dt/dx * (

        - (0.5 * (f1.u[index+1] + f1.u[index])) * ((0.5 * (f1.u[index+1] + f1.u[index])) > 0 ? f1.m[index] : f1.m[index_xp])

        + (0.5 * (f1.v[index-1] + f1.v[index])) * ((0.5 * (f1.v[index-1] + f1.v[index])) > 0 ? f1.m[index_ym] : f1.m[index])

        - (0.5 * (f1.w[index+1] + f1.w[index])) * ((0.5 * (f1.w[index+1] + f1.w[index])) > 0 ? f1.m[index] : f1.m[index_zp])

    );
    // Update velocity field
    f2.u[index] = 0;
    f2.v[index] = 0;
    f2.w[index] = 0;

}

__global__ void updateCornerZ1X2Y1(data d) {

    int x = 3 - 1;
    int y = 0;
    int z = 0;
    int index = x + y * 3 + z * 3 * 3;
    int index_xm = (x-1) + y * 3 + z * 3 * 3;
    int index_yp = x + (y+1) * 3 + z * 3 * 3;
    int index_zp = x + y * 3 + (z+1) * 3 * 3;



    // Define origin and destination fields
    field f1 = *d.inMain ? d.f1 : d.f2;
    field f2 = *d.inMain ? d.f2 : d.f1;

    // Define dx, dt and RT
    float dx = *d.dx;
    float dt = *d.dt;
    float RT = *d.RT;
    float mu = *d.mu;
    float g = *d.g;

    // Update mass field

    f2.m[index] = f1.m[index] + dt/dx * (

        + (0.5 * (f1.u[index-1] + f1.u[index])) * ((0.5 * (f1.u[index-1] + f1.u[index])) > 0 ? f1.m[index_xm] : f1.m[index])

        - (0.5 * (f1.v[index+1] + f1.v[index])) * ((0.5 * (f1.v[index+1] + f1.v[index])) > 0 ? f1.m[index] : f1.m[index_yp])

        - (0.5 * (f1.w[index+1] + f1.w[index])) * ((0.5 * (f1.w[index+1] + f1.w[index])) > 0 ? f1.m[index] : f1.m[index_zp])

    );
    // Update velocity field
    f2.u[index] = 0;
    f2.v[index] = 0;
    f2.w[index] = 0;

}

__global__ void updateCornerZ1X2Y2(data d) {

    int x = 3 - 1;
    int y = 3 - 1;
    int z = 0;
    int index = x + y * 3 + z * 3 * 3;
    int index_xm = (x-1) + y * 3 + z * 3 * 3;
    int index_ym = x + (y-1) * 3 + z * 3 * 3;
    int index_zp = x + y * 3 + (z+1) * 3 * 3;



    // Define origin and destination fields
    field f1 = *d.inMain ? d.f1 : d.f2;
    field f2 = *d.inMain ? d.f2 : d.f1;

    // Define dx, dt and RT
    float dx = *d.dx;
    float dt = *d.dt;
    float RT = *d.RT;
    float mu = *d.mu;
    float g = *d.g;

    // Update mass field

    f2.m[index] = f1.m[index] + dt/dx * (

        + (0.5 * (f1.u[index-1] + f1.u[index])) * ((0.5 * (f1.u[index-1] + f1.u[index])) > 0 ? f1.m[index_xm] : f1.m[index])

        + (0.5 * (f1.v[index-1] + f1.v[index])) * ((0.5 * (f1.v[index-1] + f1.v[index])) > 0 ? f1.m[index_ym] : f1.m[index])

        - (0.5 * (f1.w[index+1] + f1.w[index])) * ((0.5 * (f1.w[index+1] + f1.w[index])) > 0 ? f1.m[index] : f1.m[index_zp])

    );
    // Update velocity field
    f2.u[index] = 0;
    f2.v[index] = 0;
    f2.w[index] = 0;

}

__global__ void updateCornerZ2X1Y1(data d) {

    int x = 0;
    int y = 0;
    int z = 3 - 1;
    int index = x + y * 3 + z * 3 * 3;
    int index_xp = (x+1) + y * 3 + z * 3 * 3;
    int index_yp = x + (y+1) * 3 + z * 3 * 3;
    int index_zm = x + y * 3 + (z-1) * 3 * 3;



    // Define origin and destination fields
    field f1 = *d.inMain ? d.f1 : d.f2;
    field f2 = *d.inMain ? d.f2 : d.f1;

    // Define dx, dt and RT
    float dx = *d.dx;
    float dt = *d.dt;
    float RT = *d.RT;
    float mu = *d.mu;
    float g = *d.g;

    // Update mass field

    f2.m[index] = f1.m[index] + dt/dx * (

        - (0.5 * (f1.u[index+1] + f1.u[index])) * ((0.5 * (f1.u[index+1] + f1.u[index])) > 0 ? f1.m[index] : f1.m[index_xp])

        - (0.5 * (f1.v[index+1] + f1.v[index])) * ((0.5 * (f1.v[index+1] + f1.v[index])) > 0 ? f1.m[index] : f1.m[index_yp])

        + (0.5 * (f1.w[index-1] + f1.w[index])) * ((0.5 * (f1.w[index-1] + f1.w[index])) > 0 ? f1.m[index_zm] : f1.m[index])

    );
    // Update velocity field
    f2.u[index] = 0;
    f2.v[index] = 0;
    f2.w[index] = 0;

}

__global__ void updateCornerZ2X1Y2(data d) {

    int x = 0;
    int y = 3 - 1;
    int z = 3 - 1;
    int index = x + y * 3 + z * 3 * 3;
    int index_xp = (x+1) + y * 3 + z * 3 * 3;
    int index_ym = x + (y-1) * 3 + z * 3 * 3;
    int index_zm = x + y * 3 + (z-1) * 3 * 3;



    // Define origin and destination fields
    field f1 = *d.inMain ? d.f1 : d.f2;
    field f2 = *d.inMain ? d.f2 : d.f1;

    // Define dx, dt and RT
    float dx = *d.dx;
    float dt = *d.dt;
    float RT = *d.RT;
    float mu = *d.mu;
    float g = *d.g;

    // Update mass field

    f2.m[index] = f1.m[index] + dt/dx * (

        - (0.5 * (f1.u[index+1] + f1.u[index])) * ((0.5 * (f1.u[index+1] + f1.u[index])) > 0 ? f1.m[index] : f1.m[index_xp])

        + (0.5 * (f1.v[index-1] + f1.v[index])) * ((0.5 * (f1.v[index-1] + f1.v[index])) > 0 ? f1.m[index_ym] : f1.m[index])

        + (0.5 * (f1.w[index-1] + f1.w[index])) * ((0.5 * (f1.w[index-1] + f1.w[index])) > 0 ? f1.m[index_zm] : f1.m[index])

    );
    // Update velocity field
    f2.u[index] = 0;
    f2.v[index] = 0;
    f2.w[index] = 0;

}

__global__ void updateCornerZ2X2Y1(data d) {

    int x = 3 - 1;
    int y = 0;
    int z = 3 - 1;
    int index = x + y * 3 + z * 3 * 3;
    int index_xm = (x-1) + y * 3 + z * 3 * 3;
    int index_yp = x + (y+1) * 3 + z * 3 * 3;
    int index_zm = x + y * 3 + (z-1) * 3 * 3;



    // Define origin and destination fields
    field f1 = *d.inMain ? d.f1 : d.f2;
    field f2 = *d.inMain ? d.f2 : d.f1;

    // Define dx, dt and RT
    float dx = *d.dx;
    float dt = *d.dt;
    float RT = *d.RT;
    float mu = *d.mu;
    float g = *d.g;

    // Update mass field

    f2.m[index] = f1.m[index] + dt/dx * (

        + (0.5 * (f1.u[index-1] + f1.u[index])) * ((0.5 * (f1.u[index-1] + f1.u[index])) > 0 ? f1.m[index_xm] : f1.m[index])

        - (0.5 * (f1.v[index+1] + f1.v[index])) * ((0.5 * (f1.v[index+1] + f1.v[index])) > 0 ? f1.m[index] : f1.m[index_yp])

        + (0.5 * (f1.w[index-1] + f1.w[index])) * ((0.5 * (f1.w[index-1] + f1.w[index])) > 0 ? f1.m[index_zm] : f1.m[index])

    );
    // Update velocity field
    f2.u[index] = 0;
    f2.v[index] = 0;
    f2.w[index] = 0;

}

__global__ void updateCornerZ2X2Y2(data d) {

    int x = 3 - 1;
    int y = 3 - 1;
    int z = 3 - 1;
    int index = x + y * 3 + z * 3 * 3;
    int index_xm = (x-1) + y * 3 + z * 3 * 3;
    int index_ym = x + (y-1) * 3 + z * 3 * 3;
    int index_zm = x + y * 3 + (z-1) * 3 * 3;



    // Define origin and destination fields
    field f1 = *d.inMain ? d.f1 : d.f2;
    field f2 = *d.inMain ? d.f2 : d.f1;

    // Define dx, dt and RT
    float dx = *d.dx;
    float dt = *d.dt;
    float RT = *d.RT;
    float mu = *d.mu;
    float g = *d.g;

    // Update mass field

    f2.m[index] = f1.m[index] + dt/dx * (

        + (0.5 * (f1.u[index-1] + f1.u[index])) * ((0.5 * (f1.u[index-1] + f1.u[index])) > 0 ? f1.m[index_xm] : f1.m[index])

        + (0.5 * (f1.v[index-1] + f1.v[index])) * ((0.5 * (f1.v[index-1] + f1.v[index])) > 0 ? f1.m[index_ym] : f1.m[index])

        + (0.5 * (f1.w[index-1] + f1.w[index])) * ((0.5 * (f1.w[index-1] + f1.w[index])) > 0 ? f1.m[index_zm] : f1.m[index])

    );
    // Update velocity field
    f2.u[index] = 0;
    f2.v[index] = 0;
    f2.w[index] = 0;

}


void step(data d) {
    updateCore<<<1, 256>>>(d);
    updateFaceX1<<<1, 256>>>(d);
    updateFaceX2<<<1, 256>>>(d);
    updateFaceY1<<<1, 256>>>(d);
    updateFaceY2<<<1, 256>>>(d);
    updateFaceZ1<<<1, 256>>>(d);
    updateFaceZ2<<<1, 256>>>(d);
    updateEdgeX1Y1<<<1, 256>>>(d);
    updateEdgeX1Y2<<<1, 256>>>(d);
    updateEdgeX2Y1<<<1, 256>>>(d);
    updateEdgeX2Y2<<<1, 256>>>(d);
    updateEdgeX1Z1<<<1, 256>>>(d);
    updateEdgeX1Z2<<<1, 256>>>(d);
    updateEdgeX2Z1<<<1, 256>>>(d);
    updateEdgeX2Z2<<<1, 256>>>(d);
    updateEdgeZ1X1<<<1, 256>>>(d);
    updateEdgeZ1X2<<<1, 256>>>(d);
    updateEdgeZ2X1<<<1, 256>>>(d);
    updateEdgeZ2X2<<<1, 256>>>(d);
    updateCornerZ1X1Y1<<<1, 1>>>(d);
    updateCornerZ1X1Y2<<<1, 1>>>(d);
    updateCornerZ1X2Y1<<<1, 1>>>(d);
    updateCornerZ1X2Y2<<<1, 1>>>(d);
    updateCornerZ2X1Y1<<<1, 1>>>(d);
    updateCornerZ2X1Y2<<<1, 1>>>(d);
    updateCornerZ2X2Y1<<<1, 1>>>(d);
    updateCornerZ2X2Y2<<<1, 1>>>(d);
    cudaDeviceSynchronize();
}