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
}

void init(int numBlocks, int threadsPerBlock, data d, int side) {
    initializeValues<<<numBlocks, threadsPerBlock>>>(d, side);
    cudaDeviceSynchronize();
}

// Flip in main
__global__ void flipInMainKernel(data d) {
    *d.inMain = !*d.inMain;
}

void flipInMain(data d) {
    flipInMainKernel<<<1, 1>>>(d);
    cudaDeviceSynchronize();
}

// Update field

__global__ void updateCore(data d) {
    int totalThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if(totalThreadIndex >= 26463592) return;

    // Compute indices
    int x = (totalThreadIndex % 298) + 1;
    int y = ((totalThreadIndex / 298) % 298) + 1;
    int z = (totalThreadIndex / (298 * 298)) + 1;
    int index = x + y * 300 + z * 300 * 300;
    int index_xm = (x-1) + y * 300 + z * 300 * 300;
    int index_xp = (x+1) + y * 300 + z * 300 * 300;
    int index_ym = x + (y-1) * 300 + z * 300 * 300;
    int index_yp = x + (y+1) * 300 + z * 300 * 300;
    int index_zm = x + y * 300 + (z-1) * 300 * 300;
    int index_zp = x + y * 300 + (z+1) * 300 * 300;


    // Define origin and destination fields
    field f1 = d.inMain ? d.f1 : d.f2;
    field f2 = d.inMain ? d.f2 : d.f1;

    // Define dx, dt and RT
    float dx = *d.dx;
    float dt = *d.dt;
    float RT = *d.RT;
    float mu = *d.mu;
    float g = *d.g;

    // Update velocity field

    f2.u[index] = f1.u[index] + dt *
    (
        - f1.u[index] * (f1.u[index_xp] - f1.u[index_xm]) / (2 * dx)
        - f1.v[index] * (f1.u[index_yp] - f1.u[index_ym]) / (2 * dx)
        - f1.w[index] * (f1.u[index_zp] - f1.u[index_zm]) / (2 * dx)
        - (RT/f1.p[index]) * (f1.p[index_xp] - f1.p[index_xm]) / (2 * dx)
        + mu * (RT/f1.p[index]) * (
            (f1.u[index_xp] - 2 * f1.u[index] + f1.u[index_xm]) / (dx * dx) +
            (f1.u[index_yp] - 2 * f1.u[index] + f1.u[index_ym]) / (dx * dx) +
            (f1.u[index_zp] - 2 * f1.u[index] + f1.u[index_zm]) / (dx * dx)
        )
    );


    f2.v[index] = f1.v[index] + dt *
    (
        - f1.u[index] * (f1.v[index_xp] - f1.v[index_xm]) / (2 * dx)
        - f1.v[index] * (f1.v[index_yp] - f1.v[index_ym]) / (2 * dx)
        - f1.w[index] * (f1.v[index_zp] - f1.v[index_zm]) / (2 * dx)
        - (RT/f1.p[index]) * (f1.p[index_yp] - f1.p[index_ym]) / (2 * dx)
        + mu * (RT/f1.p[index]) * (
            (f1.v[index_xp] - 2 * f1.v[index] + f1.v[index_xm]) / (dx * dx) +
            (f1.v[index_yp] - 2 * f1.v[index] + f1.v[index_ym]) / (dx * dx) +
            (f1.v[index_zp] - 2 * f1.v[index] + f1.v[index_zm]) / (dx * dx)
        )
    );


    f2.w[index] = f1.w[index] + dt *
    (
        - f1.u[index] * (f1.w[index_xp] - f1.w[index_xm]) / (2 * dx)
        - f1.v[index] * (f1.w[index_yp] - f1.w[index_ym]) / (2 * dx)
        - f1.w[index] * (f1.w[index_zp] - f1.w[index_zm]) / (2 * dx)
        - (RT/f1.p[index]) * (f1.p[index_zp] - f1.p[index_zm]) / (2 * dx)
        + mu * (RT/f1.p[index]) * (
            (f1.w[index_xp] - 2 * f1.w[index] + f1.w[index_xm]) / (dx * dx) +
            (f1.w[index_yp] - 2 * f1.w[index] + f1.w[index_ym]) / (dx * dx) +
            (f1.w[index_zp] - 2 * f1.w[index] + f1.w[index_zm]) / (dx * dx)
        )
        - g
    );

    // Update pressure field
    f2.p[index] = f1.p[index] - (dt) * 
    (
        (f1.u[index_xp] - f1.u[index_xm]) / (2 * dx) * f1.p[index] +
        (f1.p[index_xp] - f1.p[index_xm]) / (2 * dx) * f1.u[index] +
        (f1.v[index_yp] - f1.v[index_ym]) / (2 * dx) * f1.p[index] +
        (f1.p[index_yp] - f1.p[index_ym]) / (2 * dx) * f1.v[index] +
        (f1.w[index_zp] - f1.w[index_zm]) / (2 * dx) * f1.p[index] +
        (f1.p[index_zp] - f1.p[index_zm]) / (2 * dx) * f1.w[index]
    );

}



__global__ void updateFaceX1(data d) {
    int totalThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if(totalThreadIndex >= 88804) return;


    int x = 0;
    int y = (totalThreadIndex % 298) + 1;
    int z = ((totalThreadIndex / 298) % 298) + 1;
    int index = x + y * 300 + z * 300 * 300;
    int index_xpp = (x+2) + y * 300 + z * 300 * 300;
    int index_xp = (x+1) + y * 300 + z * 300 * 300;
    int index_ym = x + (y-1) * 300 + z * 300 * 300;
    int index_yp = x + (y+1) * 300 + z * 300 * 300;
    int index_zm = x + y * 300 + (z-1) * 300 * 300;
    int index_zp = x + y * 300 + (z+1) * 300 * 300;



    // Define origin and destination fields
    field f1 = d.inMain ? d.f1 : d.f2;
    field f2 = d.inMain ? d.f2 : d.f1;

    // Define dx, dt and RT
    float dx = *d.dx;
    float dt = *d.dt;
    float RT = *d.RT;
    float mu = *d.mu;
    float g = *d.g;

    // Update velocity field
    f2.u[index] = 0;

    f2.v[index] = f1.v[index] + dt *
    (
        - f1.u[index] * (f1.v[index_xp] - f1.v[index]) / dx
        - f1.v[index] * (f1.v[index_yp] - f1.v[index_ym]) / (2 * dx)
        - f1.w[index] * (f1.v[index_zp] - f1.v[index_zm]) / (2 * dx)
        - (RT/f1.p[index]) * (f1.p[index_yp] - f1.p[index_ym]) / (2 * dx)
        + mu * (RT/f1.p[index]) * (
            (f1.v[index_xpp] - 2 * f1.v[index_xp] + f1.v[index]) / (dx * dx) +
            (f1.v[index_yp] - 2 * f1.v[index] + f1.v[index_ym]) / (dx * dx) +
            (f1.v[index_zp] - 2 * f1.v[index] + f1.v[index_zm]) / (dx * dx)
        )
    );


    f2.w[index] = f1.w[index] + dt *
    (
        - f1.u[index] * (f1.w[index_xp] - f1.w[index]) / dx
        - f1.v[index] * (f1.w[index_yp] - f1.w[index_ym]) / (2 * dx)
        - f1.w[index] * (f1.w[index_zp] - f1.w[index_zm]) / (2 * dx)
        - (RT/f1.p[index]) * (f1.p[index_zp] - f1.p[index_zm]) / (2 * dx)
        + mu * (RT/f1.p[index]) * (
            (f1.w[index_xpp] - 2 * f1.w[index_xp] + f1.w[index]) / (dx * dx) +
            (f1.w[index_yp] - 2 * f1.w[index] + f1.w[index_ym]) / (dx * dx) +
            (f1.w[index_zp] - 2 * f1.w[index] + f1.w[index_zm]) / (dx * dx)
        )
        - g
    );

    // Update pressure field
    f2.p[index] = f1.p[index] - (dt) * 
    (
        (f1.u[index_xp] - f1.u[index]) / dx * f1.p[index] +
        (f1.p[index_xp] - f1.p[index]) / dx * f1.u[index] +
        (f1.v[index_yp] - f1.v[index_ym]) / (2 * dx) * f1.p[index] +
        (f1.p[index_yp] - f1.p[index_ym]) / (2 * dx) * f1.v[index] +
        (f1.w[index_zp] - f1.w[index_zm]) / (2 * dx) * f1.p[index] +
        (f1.p[index_zp] - f1.p[index_zm]) / (2 * dx) * f1.w[index]
    );

}



__global__ void updateFaceX2(data d) {
    int totalThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if(totalThreadIndex >= 88804) return;


    int x = 299;
    int y = (totalThreadIndex % 298) + 1;
    int z = ((totalThreadIndex / 298) % 298) + 1;
    int index = x + y * 300 + z * 300 * 300;
    int index_xm = (x-1) + y * 300 + z * 300 * 300;
    int index_xmm = (x-2) + y * 300 + z * 300 * 300;
    int index_ym = x + (y-1) * 300 + z * 300 * 300;
    int index_yp = x + (y+1) * 300 + z * 300 * 300;
    int index_zm = x + y * 300 + (z-1) * 300 * 300;
    int index_zp = x + y * 300 + (z+1) * 300 * 300;



    // Define origin and destination fields
    field f1 = d.inMain ? d.f1 : d.f2;
    field f2 = d.inMain ? d.f2 : d.f1;

    // Define dx, dt and RT
    float dx = *d.dx;
    float dt = *d.dt;
    float RT = *d.RT;
    float mu = *d.mu;
    float g = *d.g;

    // Update velocity field
    f2.u[index] = 0;

    f2.v[index] = f1.v[index] + dt *
    (
        - f1.u[index] * (f1.v[index] - f1.v[index_xm]) / dx
        - f1.v[index] * (f1.v[index_yp] - f1.v[index_ym]) / (2 * dx)
        - f1.w[index] * (f1.v[index_zp] - f1.v[index_zm]) / (2 * dx)
        - (RT/f1.p[index]) * (f1.p[index_yp] - f1.p[index_ym]) / (2 * dx)
        + mu * (RT/f1.p[index]) * (
            (f1.v[index] - 2 * f1.v[index_xm] + f1.v[index_xmm]) / (dx * dx) +
            (f1.v[index_yp] - 2 * f1.v[index] + f1.v[index_ym]) / (dx * dx) +
            (f1.v[index_zp] - 2 * f1.v[index] + f1.v[index_zm]) / (dx * dx)
        )
    );


    f2.w[index] = f1.w[index] + dt *
    (
        - f1.u[index] * (f1.w[index] - f1.w[index_xm]) / dx
        - f1.v[index] * (f1.w[index_yp] - f1.w[index_ym]) / (2 * dx)
        - f1.w[index] * (f1.w[index_zp] - f1.w[index_zm]) / (2 * dx)
        - (RT/f1.p[index]) * (f1.p[index_zp] - f1.p[index_zm]) / (2 * dx)
        + mu * (RT/f1.p[index]) * (
            (f1.w[index] - 2 * f1.w[index_xm] + f1.w[index_xmm]) / (dx * dx) +
            (f1.w[index_yp] - 2 * f1.w[index] + f1.w[index_ym]) / (dx * dx) +
            (f1.w[index_zp] - 2 * f1.w[index] + f1.w[index_zm]) / (dx * dx)
        )
        - g
    );

    // Update pressure field
    f2.p[index] = f1.p[index] - (dt) * 
    (
        (f1.u[index] - f1.u[index_xm]) / dx * f1.p[index] +
        (f1.p[index] - f1.p[index_xm]) / dx * f1.u[index] +
        (f1.v[index_yp] - f1.v[index_ym]) / (2 * dx) * f1.p[index] +
        (f1.p[index_yp] - f1.p[index_ym]) / (2 * dx) * f1.v[index] +
        (f1.w[index_zp] - f1.w[index_zm]) / (2 * dx) * f1.p[index] +
        (f1.p[index_zp] - f1.p[index_zm]) / (2 * dx) * f1.w[index]
    );

}



__global__ void updateFaceY1(data d) {
    int totalThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if(totalThreadIndex >= 88804) return;


    int x = (totalThreadIndex % 298) + 1;
    int y = 0;
    int z = ((totalThreadIndex / 298) % 298) + 1;
    int index = x + y * 300 + z * 300 * 300;
    int index_xm = (x-1) + y * 300 + z * 300 * 300;
    int index_xp = (x+1) + y * 300 + z * 300 * 300;
    int index_ypp = x + (y+2) * 300 + z * 300 * 300;
    int index_yp = x + (y+1) * 300 + z * 300 * 300;
    int index_zm = x + y * 300 + (z-1) * 300 * 300;
    int index_zp = x + y * 300 + (z+1) * 300 * 300;



    // Define origin and destination fields
    field f1 = d.inMain ? d.f1 : d.f2;
    field f2 = d.inMain ? d.f2 : d.f1;

    // Define dx, dt and RT
    float dx = *d.dx;
    float dt = *d.dt;
    float RT = *d.RT;
    float mu = *d.mu;
    float g = *d.g;

    // Update velocity field

    f2.u[index] = f1.u[index] + dt *
    (
        - f1.u[index] * (f1.u[index_xp] - f1.u[index_xm]) / (2 * dx)
        - f1.v[index] * (f1.u[index_yp] - f1.u[index]) / dx
        - f1.w[index] * (f1.u[index_zp] - f1.u[index_zm]) / (2 * dx)
        - (RT/f1.p[index]) * (f1.p[index_xp] - f1.p[index_xm]) / (2 * dx)
        + mu * (RT/f1.p[index]) * (
            (f1.u[index_xp] - 2 * f1.u[index] + f1.u[index_xm]) / (dx * dx) +
            (f1.u[index_ypp] - 2 * f1.u[index_yp] + f1.u[index]) / (dx * dx) +
            (f1.u[index_zp] - 2 * f1.u[index] + f1.u[index_zm]) / (dx * dx)
        )
    );

    f2.v[index] = 0;

    f2.w[index] = f1.w[index] + dt *
    (
        - f1.u[index] * (f1.w[index_xp] - f1.w[index_xm]) / (2 * dx)
        - f1.v[index] * (f1.w[index_yp] - f1.w[index]) / dx
        - f1.w[index] * (f1.w[index_zp] - f1.w[index_zm]) / (2 * dx)
        - (RT/f1.p[index]) * (f1.p[index_zp] - f1.p[index_zm]) / (2 * dx)
        + mu * (RT/f1.p[index]) * (
            (f1.w[index_xp] - 2 * f1.w[index] + f1.w[index_xm]) / (dx * dx) +
            (f1.w[index_ypp] - 2 * f1.w[index_yp] + f1.w[index]) / (dx * dx) +
            (f1.w[index_zp] - 2 * f1.w[index] + f1.w[index_zm]) / (dx * dx)
        )
        - g
    );

    // Update pressure field
    f2.p[index] = f1.p[index] - (dt) * 
    (
        (f1.u[index_xp] - f1.u[index_xm]) / (2 * dx) * f1.p[index] +
        (f1.p[index_xp] - f1.p[index_xm]) / (2 * dx) * f1.u[index] +
        (f1.v[index_yp] - f1.v[index]) / dx * f1.p[index] +
        (f1.p[index_yp] - f1.p[index]) / dx * f1.v[index] +
        (f1.w[index_zp] - f1.w[index_zm]) / (2 * dx) * f1.p[index] +
        (f1.p[index_zp] - f1.p[index_zm]) / (2 * dx) * f1.w[index]
    );

}



__global__ void updateFaceY2(data d) {
    int totalThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if(totalThreadIndex >= 88804) return;


    int x = (totalThreadIndex % 298) + 1;
    int y = 299;
    int z = ((totalThreadIndex / 298) % 298) + 1;
    int index = x + y * 300 + z * 300 * 300;
    int index_xm = (x-1) + y * 300 + z * 300 * 300;
    int index_xp = (x+1) + y * 300 + z * 300 * 300;
    int index_ym = x + (y-1) * 300 + z * 300 * 300;
    int index_ymm = x + (y-2) * 300 + z * 300 * 300;
    int index_zm = x + y * 300 + (z-1) * 300 * 300;
    int index_zp = x + y * 300 + (z+1) * 300 * 300;



    // Define origin and destination fields
    field f1 = d.inMain ? d.f1 : d.f2;
    field f2 = d.inMain ? d.f2 : d.f1;

    // Define dx, dt and RT
    float dx = *d.dx;
    float dt = *d.dt;
    float RT = *d.RT;
    float mu = *d.mu;
    float g = *d.g;

    // Update velocity field

    f2.u[index] = f1.u[index] + dt *
    (
        - f1.u[index] * (f1.u[index_xp] - f1.u[index_xm]) / (2 * dx)
        - f1.v[index] * (f1.u[index] - f1.u[index_ym]) / dx
        - f1.w[index] * (f1.u[index_zp] - f1.u[index_zm]) / (2 * dx)
        - (RT/f1.p[index]) * (f1.p[index_xp] - f1.p[index_xm]) / (2 * dx)
        + mu * (RT/f1.p[index]) * (
            (f1.u[index_xp] - 2 * f1.u[index] + f1.u[index_xm]) / (dx * dx) +
            (f1.u[index] - 2 * f1.u[index_ym] + f1.u[index_ymm]) / (dx * dx) +
            (f1.u[index_zp] - 2 * f1.u[index] + f1.u[index_zm]) / (dx * dx)
        )
    );

    f2.v[index] = 0;

    f2.w[index] = f1.w[index] + dt *
    (
        - f1.u[index] * (f1.w[index_xp] - f1.w[index_xm]) / (2 * dx)
        - f1.v[index] * (f1.w[index] - f1.w[index_ym]) / dx
        - f1.w[index] * (f1.w[index_zp] - f1.w[index_zm]) / (2 * dx)
        - (RT/f1.p[index]) * (f1.p[index_zp] - f1.p[index_zm]) / (2 * dx)
        + mu * (RT/f1.p[index]) * (
            (f1.w[index_xp] - 2 * f1.w[index] + f1.w[index_xm]) / (dx * dx) +
            (f1.w[index] - 2 * f1.w[index_ym] + f1.w[index_ymm]) / (dx * dx) +
            (f1.w[index_zp] - 2 * f1.w[index] + f1.w[index_zm]) / (dx * dx)
        )
        - g
    );

    // Update pressure field
    f2.p[index] = f1.p[index] - (dt) * 
    (
        (f1.u[index_xp] - f1.u[index_xm]) / (2 * dx) * f1.p[index] +
        (f1.p[index_xp] - f1.p[index_xm]) / (2 * dx) * f1.u[index] +
        (f1.v[index] - f1.v[index_ym]) / dx * f1.p[index] +
        (f1.p[index] - f1.p[index_ym]) / dx * f1.v[index] +
        (f1.w[index_zp] - f1.w[index_zm]) / (2 * dx) * f1.p[index] +
        (f1.p[index_zp] - f1.p[index_zm]) / (2 * dx) * f1.w[index]
    );

}



__global__ void updateFaceZ1(data d) {
    int totalThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if(totalThreadIndex >= 88804) return;


    int x = (totalThreadIndex % 298) + 1;
    int y = ((totalThreadIndex / 298) % 298) + 1;
    int z = 0;
    int index = x + y * 300 + z * 300 * 300;
    int index_xm = (x-1) + y * 300 + z * 300 * 300;
    int index_xp = (x+1) + y * 300 + z * 300 * 300;
    int index_ym = x + (y-1) * 300 + z * 300 * 300;
    int index_yp = x + (y+1) * 300 + z * 300 * 300;
    int index_zpp = x + y * 300 + (z+2) * 300 * 300;
    int index_zp = x + y * 300 + (z+1) * 300 * 300;



    // Define origin and destination fields
    field f1 = d.inMain ? d.f1 : d.f2;
    field f2 = d.inMain ? d.f2 : d.f1;

    // Define dx, dt and RT
    float dx = *d.dx;
    float dt = *d.dt;
    float RT = *d.RT;
    float mu = *d.mu;
    float g = *d.g;

    // Update velocity field

    f2.u[index] = f1.u[index] + dt *
    (
        - f1.u[index] * (f1.u[index_xp] - f1.u[index_xm]) / (2 * dx)
        - f1.v[index] * (f1.u[index_yp] - f1.u[index_ym]) / (2 * dx)
        - f1.w[index] * (f1.u[index_zp] - f1.u[index]) / dx
        - (RT/f1.p[index]) * (f1.p[index_xp] - f1.p[index_xm]) / (2 * dx)
        + mu * (RT/f1.p[index]) * (
            (f1.u[index_xp] - 2 * f1.u[index] + f1.u[index_xm]) / (dx * dx) +
            (f1.u[index_yp] - 2 * f1.u[index] + f1.u[index_ym]) / (dx * dx) +
            (f1.u[index_zpp] - 2 * f1.u[index_zp] + f1.u[index]) / (dx * dx)
        )
    );


    f2.v[index] = f1.v[index] + dt *
    (
        - f1.u[index] * (f1.v[index_xp] - f1.v[index_xm]) / (2 * dx)
        - f1.v[index] * (f1.v[index_yp] - f1.v[index_ym]) / (2 * dx)
        - f1.w[index] * (f1.v[index_zp] - f1.v[index]) / dx
        - (RT/f1.p[index]) * (f1.p[index_yp] - f1.p[index_ym]) / (2 * dx)
        + mu * (RT/f1.p[index]) * (
            (f1.v[index_xp] - 2 * f1.v[index] + f1.v[index_xm]) / (dx * dx) +
            (f1.v[index_yp] - 2 * f1.v[index] + f1.v[index_ym]) / (dx * dx) +
            (f1.v[index_zpp] - 2 * f1.v[index_zp] + f1.v[index]) / (dx * dx)
        )
    );

    f2.w[index] = 0;
    // Update pressure field
    f2.p[index] = f1.p[index] - (dt) * 
    (
        (f1.u[index_xp] - f1.u[index_xm]) / (2 * dx) * f1.p[index] +
        (f1.p[index_xp] - f1.p[index_xm]) / (2 * dx) * f1.u[index] +
        (f1.v[index_yp] - f1.v[index_ym]) / (2 * dx) * f1.p[index] +
        (f1.p[index_yp] - f1.p[index_ym]) / (2 * dx) * f1.v[index] +
        (f1.w[index_zp] - f1.w[index]) / dx * f1.p[index] +
        (f1.p[index_zp] - f1.p[index]) / dx * f1.w[index]
    );

}



__global__ void updateFaceZ2(data d) {
    int totalThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if(totalThreadIndex >= 88804) return;


    int x = (totalThreadIndex % 298) + 1;
    int y = ((totalThreadIndex / 298) % 298) + 1;
    int z = 299;
    int index = x + y * 300 + z * 300 * 300;
    int index_xm = (x-1) + y * 300 + z * 300 * 300;
    int index_xp = (x+1) + y * 300 + z * 300 * 300;
    int index_ym = x + (y-1) * 300 + z * 300 * 300;
    int index_yp = x + (y+1) * 300 + z * 300 * 300;
    int index_zm = x + y * 300 + (z-1) * 300 * 300;
    int index_zmm = x + y * 300 + (z-2) * 300 * 300;



    // Define origin and destination fields
    field f1 = d.inMain ? d.f1 : d.f2;
    field f2 = d.inMain ? d.f2 : d.f1;

    // Define dx, dt and RT
    float dx = *d.dx;
    float dt = *d.dt;
    float RT = *d.RT;
    float mu = *d.mu;
    float g = *d.g;

    // Update velocity field

    f2.u[index] = f1.u[index] + dt *
    (
        - f1.u[index] * (f1.u[index_xp] - f1.u[index_xm]) / (2 * dx)
        - f1.v[index] * (f1.u[index_yp] - f1.u[index_ym]) / (2 * dx)
        - f1.w[index] * (f1.u[index] - f1.u[index_zm]) / dx
        - (RT/f1.p[index]) * (f1.p[index_xp] - f1.p[index_xm]) / (2 * dx)
        + mu * (RT/f1.p[index]) * (
            (f1.u[index_xp] - 2 * f1.u[index] + f1.u[index_xm]) / (dx * dx) +
            (f1.u[index_yp] - 2 * f1.u[index] + f1.u[index_ym]) / (dx * dx) +
            (f1.u[index] - 2 * f1.u[index_zm] + f1.u[index_zmm]) / (dx * dx)
        )
    );


    f2.v[index] = f1.v[index] + dt *
    (
        - f1.u[index] * (f1.v[index_xp] - f1.v[index_xm]) / (2 * dx)
        - f1.v[index] * (f1.v[index_yp] - f1.v[index_ym]) / (2 * dx)
        - f1.w[index] * (f1.v[index] - f1.v[index_zm]) / dx
        - (RT/f1.p[index]) * (f1.p[index_yp] - f1.p[index_ym]) / (2 * dx)
        + mu * (RT/f1.p[index]) * (
            (f1.v[index_xp] - 2 * f1.v[index] + f1.v[index_xm]) / (dx * dx) +
            (f1.v[index_yp] - 2 * f1.v[index] + f1.v[index_ym]) / (dx * dx) +
            (f1.v[index] - 2 * f1.v[index_zm] + f1.v[index_zmm]) / (dx * dx)
        )
    );

    f2.w[index] = 0;
    // Update pressure field
    f2.p[index] = f1.p[index] - (dt) * 
    (
        (f1.u[index_xp] - f1.u[index_xm]) / (2 * dx) * f1.p[index] +
        (f1.p[index_xp] - f1.p[index_xm]) / (2 * dx) * f1.u[index] +
        (f1.v[index_yp] - f1.v[index_ym]) / (2 * dx) * f1.p[index] +
        (f1.p[index_yp] - f1.p[index_ym]) / (2 * dx) * f1.v[index] +
        (f1.w[index] - f1.w[index_zm]) / dx * f1.p[index] +
        (f1.p[index] - f1.p[index_zm]) / dx * f1.w[index]
    );

}



__global__ void updateEdgeX1Y1(data d) {
    int totalThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if(totalThreadIndex >= 298) return;


    int x = 0;
    int y = 0;
    int z = totalThreadIndex + 1;
    int index = x + y * 300 + z * 300 * 300;
    int index_xpp = (x+2) + y * 300 + z * 300 * 300;
    int index_xp = (x+1) + y * 300 + z * 300 * 300;
    int index_ypp = x + (y+2) * 300 + z * 300 * 300;
    int index_yp = x + (y+1) * 300 + z * 300 * 300;
    int index_zm = x + y * 300 + (z-1) * 300 * 300;
    int index_zp = x + y * 300 + (z+1) * 300 * 300;



    // Define origin and destination fields
    field f1 = d.inMain ? d.f1 : d.f2;
    field f2 = d.inMain ? d.f2 : d.f1;

    // Define dx, dt and RT
    float dx = *d.dx;
    float dt = *d.dt;
    float RT = *d.RT;
    float mu = *d.mu;
    float g = *d.g;

    // Update velocity field
    f2.u[index] = 0;
    f2.v[index] = 0;

    f2.w[index] = f1.w[index] + dt *
    (
        - f1.u[index] * (f1.w[index_xp] - f1.w[index]) / dx
        - f1.v[index] * (f1.w[index_yp] - f1.w[index]) / dx
        - f1.w[index] * (f1.w[index_zp] - f1.w[index_zm]) / (2 * dx)
        - (RT/f1.p[index]) * (f1.p[index_zp] - f1.p[index_zm]) / (2 * dx)
        + mu * (RT/f1.p[index]) * (
            (f1.w[index_xpp] - 2 * f1.w[index_xp] + f1.w[index]) / (dx * dx) +
            (f1.w[index_ypp] - 2 * f1.w[index_yp] + f1.w[index]) / (dx * dx) +
            (f1.w[index_zp] - 2 * f1.w[index] + f1.w[index_zm]) / (dx * dx)
        )
        - g
    );

    // Update pressure field
    f2.p[index] = f1.p[index] - (dt) * 
    (
        (f1.u[index_xp] - f1.u[index]) / dx * f1.p[index] +
        (f1.p[index_xp] - f1.p[index]) / dx * f1.u[index] +
        (f1.v[index_yp] - f1.v[index]) / dx * f1.p[index] +
        (f1.p[index_yp] - f1.p[index]) / dx * f1.v[index] +
        (f1.w[index_zp] - f1.w[index_zm]) / (2 * dx) * f1.p[index] +
        (f1.p[index_zp] - f1.p[index_zm]) / (2 * dx) * f1.w[index]
    );

}



__global__ void updateEdgeX1Y2(data d) {
    int totalThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if(totalThreadIndex >= 298) return;


    int x = 0;
    int y = 299;
    int z = totalThreadIndex + 1;
    int index = x + y * 300 + z * 300 * 300;
    int index_xpp = (x+2) + y * 300 + z * 300 * 300;
    int index_xp = (x+1) + y * 300 + z * 300 * 300;
    int index_ym = x + (y-1) * 300 + z * 300 * 300;
    int index_ymm = x + (y-2) * 300 + z * 300 * 300;
    int index_zm = x + y * 300 + (z-1) * 300 * 300;
    int index_zp = x + y * 300 + (z+1) * 300 * 300;



    // Define origin and destination fields
    field f1 = d.inMain ? d.f1 : d.f2;
    field f2 = d.inMain ? d.f2 : d.f1;

    // Define dx, dt and RT
    float dx = *d.dx;
    float dt = *d.dt;
    float RT = *d.RT;
    float mu = *d.mu;
    float g = *d.g;

    // Update velocity field
    f2.u[index] = 0;
    f2.v[index] = 0;

    f2.w[index] = f1.w[index] + dt *
    (
        - f1.u[index] * (f1.w[index_xp] - f1.w[index]) / dx
        - f1.v[index] * (f1.w[index] - f1.w[index_ym]) / dx
        - f1.w[index] * (f1.w[index_zp] - f1.w[index_zm]) / (2 * dx)
        - (RT/f1.p[index]) * (f1.p[index_zp] - f1.p[index_zm]) / (2 * dx)
        + mu * (RT/f1.p[index]) * (
            (f1.w[index_xpp] - 2 * f1.w[index_xp] + f1.w[index]) / (dx * dx) +
            (f1.w[index] - 2 * f1.w[index_ym] + f1.w[index_ymm]) / (dx * dx) +
            (f1.w[index_zp] - 2 * f1.w[index] + f1.w[index_zm]) / (dx * dx)
        )
        - g
    );

    // Update pressure field
    f2.p[index] = f1.p[index] - (dt) * 
    (
        (f1.u[index_xp] - f1.u[index]) / dx * f1.p[index] +
        (f1.p[index_xp] - f1.p[index]) / dx * f1.u[index] +
        (f1.v[index] - f1.v[index_ym]) / dx * f1.p[index] +
        (f1.p[index] - f1.p[index_ym]) / dx * f1.v[index] +
        (f1.w[index_zp] - f1.w[index_zm]) / (2 * dx) * f1.p[index] +
        (f1.p[index_zp] - f1.p[index_zm]) / (2 * dx) * f1.w[index]
    );

}



__global__ void updateEdgeX2Y1(data d) {
    int totalThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if(totalThreadIndex >= 298) return;


    int x = 299;
    int y = 0;
    int z = totalThreadIndex + 1;
    int index = x + y * 300 + z * 300 * 300;
    int index_xm = (x-1) + y * 300 + z * 300 * 300;
    int index_xmm = (x-2) + y * 300 + z * 300 * 300;
    int index_ypp = x + (y+2) * 300 + z * 300 * 300;
    int index_yp = x + (y+1) * 300 + z * 300 * 300;
    int index_zm = x + y * 300 + (z-1) * 300 * 300;
    int index_zp = x + y * 300 + (z+1) * 300 * 300;



    // Define origin and destination fields
    field f1 = d.inMain ? d.f1 : d.f2;
    field f2 = d.inMain ? d.f2 : d.f1;

    // Define dx, dt and RT
    float dx = *d.dx;
    float dt = *d.dt;
    float RT = *d.RT;
    float mu = *d.mu;
    float g = *d.g;

    // Update velocity field
    f2.u[index] = 0;
    f2.v[index] = 0;

    f2.w[index] = f1.w[index] + dt *
    (
        - f1.u[index] * (f1.w[index] - f1.w[index_xm]) / dx
        - f1.v[index] * (f1.w[index_yp] - f1.w[index]) / dx
        - f1.w[index] * (f1.w[index_zp] - f1.w[index_zm]) / (2 * dx)
        - (RT/f1.p[index]) * (f1.p[index_zp] - f1.p[index_zm]) / (2 * dx)
        + mu * (RT/f1.p[index]) * (
            (f1.w[index] - 2 * f1.w[index_xm] + f1.w[index_xmm]) / (dx * dx) +
            (f1.w[index_ypp] - 2 * f1.w[index_yp] + f1.w[index]) / (dx * dx) +
            (f1.w[index_zp] - 2 * f1.w[index] + f1.w[index_zm]) / (dx * dx)
        )
        - g
    );

    // Update pressure field
    f2.p[index] = f1.p[index] - (dt) * 
    (
        (f1.u[index] - f1.u[index_xm]) / dx * f1.p[index] +
        (f1.p[index] - f1.p[index_xm]) / dx * f1.u[index] +
        (f1.v[index_yp] - f1.v[index]) / dx * f1.p[index] +
        (f1.p[index_yp] - f1.p[index]) / dx * f1.v[index] +
        (f1.w[index_zp] - f1.w[index_zm]) / (2 * dx) * f1.p[index] +
        (f1.p[index_zp] - f1.p[index_zm]) / (2 * dx) * f1.w[index]
    );

}



__global__ void updateEdgeX2Y2(data d) {
    int totalThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if(totalThreadIndex >= 298) return;


    int x = 299;
    int y = 299;
    int z = totalThreadIndex + 1;
    int index = x + y * 300 + z * 300 * 300;
    int index_xm = (x-1) + y * 300 + z * 300 * 300;
    int index_xmm = (x-2) + y * 300 + z * 300 * 300;
    int index_ym = x + (y-1) * 300 + z * 300 * 300;
    int index_ymm = x + (y-2) * 300 + z * 300 * 300;
    int index_zm = x + y * 300 + (z-1) * 300 * 300;
    int index_zp = x + y * 300 + (z+1) * 300 * 300;



    // Define origin and destination fields
    field f1 = d.inMain ? d.f1 : d.f2;
    field f2 = d.inMain ? d.f2 : d.f1;

    // Define dx, dt and RT
    float dx = *d.dx;
    float dt = *d.dt;
    float RT = *d.RT;
    float mu = *d.mu;
    float g = *d.g;

    // Update velocity field
    f2.u[index] = 0;
    f2.v[index] = 0;

    f2.w[index] = f1.w[index] + dt *
    (
        - f1.u[index] * (f1.w[index] - f1.w[index_xm]) / dx
        - f1.v[index] * (f1.w[index] - f1.w[index_ym]) / dx
        - f1.w[index] * (f1.w[index_zp] - f1.w[index_zm]) / (2 * dx)
        - (RT/f1.p[index]) * (f1.p[index_zp] - f1.p[index_zm]) / (2 * dx)
        + mu * (RT/f1.p[index]) * (
            (f1.w[index] - 2 * f1.w[index_xm] + f1.w[index_xmm]) / (dx * dx) +
            (f1.w[index] - 2 * f1.w[index_ym] + f1.w[index_ymm]) / (dx * dx) +
            (f1.w[index_zp] - 2 * f1.w[index] + f1.w[index_zm]) / (dx * dx)
        )
        - g
    );

    // Update pressure field
    f2.p[index] = f1.p[index] - (dt) * 
    (
        (f1.u[index] - f1.u[index_xm]) / dx * f1.p[index] +
        (f1.p[index] - f1.p[index_xm]) / dx * f1.u[index] +
        (f1.v[index] - f1.v[index_ym]) / dx * f1.p[index] +
        (f1.p[index] - f1.p[index_ym]) / dx * f1.v[index] +
        (f1.w[index_zp] - f1.w[index_zm]) / (2 * dx) * f1.p[index] +
        (f1.p[index_zp] - f1.p[index_zm]) / (2 * dx) * f1.w[index]
    );

}



__global__ void updateEdgeX1Z1(data d) {
    int totalThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if(totalThreadIndex >= 298) return;


    int x = 0;
    int y = totalThreadIndex + 1;
    int z = 0;
    int index = x + y * 300 + z * 300 * 300;
    int index_xpp = (x+2) + y * 300 + z * 300 * 300;
    int index_xp = (x+1) + y * 300 + z * 300 * 300;
    int index_ym = x + (y-1) * 300 + z * 300 * 300;
    int index_yp = x + (y+1) * 300 + z * 300 * 300;
    int index_zpp = x + y * 300 + (z+2) * 300 * 300;
    int index_zp = x + y * 300 + (z+1) * 300 * 300;



    // Define origin and destination fields
    field f1 = d.inMain ? d.f1 : d.f2;
    field f2 = d.inMain ? d.f2 : d.f1;

    // Define dx, dt and RT
    float dx = *d.dx;
    float dt = *d.dt;
    float RT = *d.RT;
    float mu = *d.mu;
    float g = *d.g;

    // Update velocity field
    f2.u[index] = 0;

    f2.v[index] = f1.v[index] + dt *
    (
        - f1.u[index] * (f1.v[index_xp] - f1.v[index]) / dx
        - f1.v[index] * (f1.v[index_yp] - f1.v[index_ym]) / (2 * dx)
        - f1.w[index] * (f1.v[index_zp] - f1.v[index]) / dx
        - (RT/f1.p[index]) * (f1.p[index_yp] - f1.p[index_ym]) / (2 * dx)
        + mu * (RT/f1.p[index]) * (
            (f1.v[index_xpp] - 2 * f1.v[index_xp] + f1.v[index]) / (dx * dx) +
            (f1.v[index_yp] - 2 * f1.v[index] + f1.v[index_ym]) / (dx * dx) +
            (f1.v[index_zpp] - 2 * f1.v[index_zp] + f1.v[index]) / (dx * dx)
        )
    );

    f2.w[index] = 0;
    // Update pressure field
    f2.p[index] = f1.p[index] - (dt) * 
    (
        (f1.u[index_xp] - f1.u[index]) / dx * f1.p[index] +
        (f1.p[index_xp] - f1.p[index]) / dx * f1.u[index] +
        (f1.v[index_yp] - f1.v[index_ym]) / (2 * dx) * f1.p[index] +
        (f1.p[index_yp] - f1.p[index_ym]) / (2 * dx) * f1.v[index] +
        (f1.w[index_zp] - f1.w[index]) / dx * f1.p[index] +
        (f1.p[index_zp] - f1.p[index]) / dx * f1.w[index]
    );

}



__global__ void updateEdgeX1Z2(data d) {
    int totalThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if(totalThreadIndex >= 298) return;


    int x = 0;
    int y = totalThreadIndex + 1;
    int z = 299;
    int index = x + y * 300 + z * 300 * 300;
    int index_xpp = (x+2) + y * 300 + z * 300 * 300;
    int index_xp = (x+1) + y * 300 + z * 300 * 300;
    int index_ym = x + (y-1) * 300 + z * 300 * 300;
    int index_yp = x + (y+1) * 300 + z * 300 * 300;
    int index_zm = x + y * 300 + (z-1) * 300 * 300;
    int index_zmm = x + y * 300 + (z-2) * 300 * 300;



    // Define origin and destination fields
    field f1 = d.inMain ? d.f1 : d.f2;
    field f2 = d.inMain ? d.f2 : d.f1;

    // Define dx, dt and RT
    float dx = *d.dx;
    float dt = *d.dt;
    float RT = *d.RT;
    float mu = *d.mu;
    float g = *d.g;

    // Update velocity field
    f2.u[index] = 0;

    f2.v[index] = f1.v[index] + dt *
    (
        - f1.u[index] * (f1.v[index_xp] - f1.v[index]) / dx
        - f1.v[index] * (f1.v[index_yp] - f1.v[index_ym]) / (2 * dx)
        - f1.w[index] * (f1.v[index] - f1.v[index_zm]) / dx
        - (RT/f1.p[index]) * (f1.p[index_yp] - f1.p[index_ym]) / (2 * dx)
        + mu * (RT/f1.p[index]) * (
            (f1.v[index_xpp] - 2 * f1.v[index_xp] + f1.v[index]) / (dx * dx) +
            (f1.v[index_yp] - 2 * f1.v[index] + f1.v[index_ym]) / (dx * dx) +
            (f1.v[index] - 2 * f1.v[index_zm] + f1.v[index_zmm]) / (dx * dx)
        )
    );

    f2.w[index] = 0;
    // Update pressure field
    f2.p[index] = f1.p[index] - (dt) * 
    (
        (f1.u[index_xp] - f1.u[index]) / dx * f1.p[index] +
        (f1.p[index_xp] - f1.p[index]) / dx * f1.u[index] +
        (f1.v[index_yp] - f1.v[index_ym]) / (2 * dx) * f1.p[index] +
        (f1.p[index_yp] - f1.p[index_ym]) / (2 * dx) * f1.v[index] +
        (f1.w[index] - f1.w[index_zm]) / dx * f1.p[index] +
        (f1.p[index] - f1.p[index_zm]) / dx * f1.w[index]
    );

}



__global__ void updateEdgeX2Z1(data d) {
    int totalThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if(totalThreadIndex >= 298) return;


    int x = 299;
    int y = totalThreadIndex + 1;
    int z = 0;
    int index = x + y * 300 + z * 300 * 300;
    int index_xm = (x-1) + y * 300 + z * 300 * 300;
    int index_xmm = (x-2) + y * 300 + z * 300 * 300;
    int index_ym = x + (y-1) * 300 + z * 300 * 300;
    int index_yp = x + (y+1) * 300 + z * 300 * 300;
    int index_zpp = x + y * 300 + (z+2) * 300 * 300;
    int index_zp = x + y * 300 + (z+1) * 300 * 300;



    // Define origin and destination fields
    field f1 = d.inMain ? d.f1 : d.f2;
    field f2 = d.inMain ? d.f2 : d.f1;

    // Define dx, dt and RT
    float dx = *d.dx;
    float dt = *d.dt;
    float RT = *d.RT;
    float mu = *d.mu;
    float g = *d.g;

    // Update velocity field
    f2.u[index] = 0;

    f2.v[index] = f1.v[index] + dt *
    (
        - f1.u[index] * (f1.v[index] - f1.v[index_xm]) / dx
        - f1.v[index] * (f1.v[index_yp] - f1.v[index_ym]) / (2 * dx)
        - f1.w[index] * (f1.v[index_zp] - f1.v[index]) / dx
        - (RT/f1.p[index]) * (f1.p[index_yp] - f1.p[index_ym]) / (2 * dx)
        + mu * (RT/f1.p[index]) * (
            (f1.v[index] - 2 * f1.v[index_xm] + f1.v[index_xmm]) / (dx * dx) +
            (f1.v[index_yp] - 2 * f1.v[index] + f1.v[index_ym]) / (dx * dx) +
            (f1.v[index_zpp] - 2 * f1.v[index_zp] + f1.v[index]) / (dx * dx)
        )
    );

    f2.w[index] = 0;
    // Update pressure field
    f2.p[index] = f1.p[index] - (dt) * 
    (
        (f1.u[index] - f1.u[index_xm]) / dx * f1.p[index] +
        (f1.p[index] - f1.p[index_xm]) / dx * f1.u[index] +
        (f1.v[index_yp] - f1.v[index_ym]) / (2 * dx) * f1.p[index] +
        (f1.p[index_yp] - f1.p[index_ym]) / (2 * dx) * f1.v[index] +
        (f1.w[index_zp] - f1.w[index]) / dx * f1.p[index] +
        (f1.p[index_zp] - f1.p[index]) / dx * f1.w[index]
    );

}



__global__ void updateEdgeX2Z2(data d) {
    int totalThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if(totalThreadIndex >= 298) return;


    int x = 299;
    int y = totalThreadIndex + 1;
    int z = 299;
    int index = x + y * 300 + z * 300 * 300;
    int index_xm = (x-1) + y * 300 + z * 300 * 300;
    int index_xmm = (x-2) + y * 300 + z * 300 * 300;
    int index_ym = x + (y-1) * 300 + z * 300 * 300;
    int index_yp = x + (y+1) * 300 + z * 300 * 300;
    int index_zm = x + y * 300 + (z-1) * 300 * 300;
    int index_zmm = x + y * 300 + (z-2) * 300 * 300;



    // Define origin and destination fields
    field f1 = d.inMain ? d.f1 : d.f2;
    field f2 = d.inMain ? d.f2 : d.f1;

    // Define dx, dt and RT
    float dx = *d.dx;
    float dt = *d.dt;
    float RT = *d.RT;
    float mu = *d.mu;
    float g = *d.g;

    // Update velocity field
    f2.u[index] = 0;

    f2.v[index] = f1.v[index] + dt *
    (
        - f1.u[index] * (f1.v[index] - f1.v[index_xm]) / dx
        - f1.v[index] * (f1.v[index_yp] - f1.v[index_ym]) / (2 * dx)
        - f1.w[index] * (f1.v[index] - f1.v[index_zm]) / dx
        - (RT/f1.p[index]) * (f1.p[index_yp] - f1.p[index_ym]) / (2 * dx)
        + mu * (RT/f1.p[index]) * (
            (f1.v[index] - 2 * f1.v[index_xm] + f1.v[index_xmm]) / (dx * dx) +
            (f1.v[index_yp] - 2 * f1.v[index] + f1.v[index_ym]) / (dx * dx) +
            (f1.v[index] - 2 * f1.v[index_zm] + f1.v[index_zmm]) / (dx * dx)
        )
    );

    f2.w[index] = 0;
    // Update pressure field
    f2.p[index] = f1.p[index] - (dt) * 
    (
        (f1.u[index] - f1.u[index_xm]) / dx * f1.p[index] +
        (f1.p[index] - f1.p[index_xm]) / dx * f1.u[index] +
        (f1.v[index_yp] - f1.v[index_ym]) / (2 * dx) * f1.p[index] +
        (f1.p[index_yp] - f1.p[index_ym]) / (2 * dx) * f1.v[index] +
        (f1.w[index] - f1.w[index_zm]) / dx * f1.p[index] +
        (f1.p[index] - f1.p[index_zm]) / dx * f1.w[index]
    );

}



__global__ void updateEdgeZ1X1(data d) {
    int totalThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if(totalThreadIndex >= 298) return;


    int x = 0;
    int y = totalThreadIndex + 1;
    int z = 0;
    int index = x + y * 300 + z * 300 * 300;
    int index_xpp = (x+2) + y * 300 + z * 300 * 300;
    int index_xp = (x+1) + y * 300 + z * 300 * 300;
    int index_ym = x + (y-1) * 300 + z * 300 * 300;
    int index_yp = x + (y+1) * 300 + z * 300 * 300;
    int index_zpp = x + y * 300 + (z+2) * 300 * 300;
    int index_zp = x + y * 300 + (z+1) * 300 * 300;



    // Define origin and destination fields
    field f1 = d.inMain ? d.f1 : d.f2;
    field f2 = d.inMain ? d.f2 : d.f1;

    // Define dx, dt and RT
    float dx = *d.dx;
    float dt = *d.dt;
    float RT = *d.RT;
    float mu = *d.mu;
    float g = *d.g;

    // Update velocity field
    f2.u[index] = 0;

    f2.v[index] = f1.v[index] + dt *
    (
        - f1.u[index] * (f1.v[index_xp] - f1.v[index]) / dx
        - f1.v[index] * (f1.v[index_yp] - f1.v[index_ym]) / (2 * dx)
        - f1.w[index] * (f1.v[index_zp] - f1.v[index]) / dx
        - (RT/f1.p[index]) * (f1.p[index_yp] - f1.p[index_ym]) / (2 * dx)
        + mu * (RT/f1.p[index]) * (
            (f1.v[index_xpp] - 2 * f1.v[index_xp] + f1.v[index]) / (dx * dx) +
            (f1.v[index_yp] - 2 * f1.v[index] + f1.v[index_ym]) / (dx * dx) +
            (f1.v[index_zpp] - 2 * f1.v[index_zp] + f1.v[index]) / (dx * dx)
        )
    );

    f2.w[index] = 0;
    // Update pressure field
    f2.p[index] = f1.p[index] - (dt) * 
    (
        (f1.u[index_xp] - f1.u[index]) / dx * f1.p[index] +
        (f1.p[index_xp] - f1.p[index]) / dx * f1.u[index] +
        (f1.v[index_yp] - f1.v[index_ym]) / (2 * dx) * f1.p[index] +
        (f1.p[index_yp] - f1.p[index_ym]) / (2 * dx) * f1.v[index] +
        (f1.w[index_zp] - f1.w[index]) / dx * f1.p[index] +
        (f1.p[index_zp] - f1.p[index]) / dx * f1.w[index]
    );

}



__global__ void updateEdgeZ1X2(data d) {
    int totalThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if(totalThreadIndex >= 298) return;


    int x = 299;
    int y = totalThreadIndex + 1;
    int z = 0;
    int index = x + y * 300 + z * 300 * 300;
    int index_xm = (x-1) + y * 300 + z * 300 * 300;
    int index_xmm = (x-2) + y * 300 + z * 300 * 300;
    int index_ym = x + (y-1) * 300 + z * 300 * 300;
    int index_yp = x + (y+1) * 300 + z * 300 * 300;
    int index_zpp = x + y * 300 + (z+2) * 300 * 300;
    int index_zp = x + y * 300 + (z+1) * 300 * 300;



    // Define origin and destination fields
    field f1 = d.inMain ? d.f1 : d.f2;
    field f2 = d.inMain ? d.f2 : d.f1;

    // Define dx, dt and RT
    float dx = *d.dx;
    float dt = *d.dt;
    float RT = *d.RT;
    float mu = *d.mu;
    float g = *d.g;

    // Update velocity field
    f2.u[index] = 0;

    f2.v[index] = f1.v[index] + dt *
    (
        - f1.u[index] * (f1.v[index] - f1.v[index_xm]) / dx
        - f1.v[index] * (f1.v[index_yp] - f1.v[index_ym]) / (2 * dx)
        - f1.w[index] * (f1.v[index_zp] - f1.v[index]) / dx
        - (RT/f1.p[index]) * (f1.p[index_yp] - f1.p[index_ym]) / (2 * dx)
        + mu * (RT/f1.p[index]) * (
            (f1.v[index] - 2 * f1.v[index_xm] + f1.v[index_xmm]) / (dx * dx) +
            (f1.v[index_yp] - 2 * f1.v[index] + f1.v[index_ym]) / (dx * dx) +
            (f1.v[index_zpp] - 2 * f1.v[index_zp] + f1.v[index]) / (dx * dx)
        )
    );

    f2.w[index] = 0;
    // Update pressure field
    f2.p[index] = f1.p[index] - (dt) * 
    (
        (f1.u[index] - f1.u[index_xm]) / dx * f1.p[index] +
        (f1.p[index] - f1.p[index_xm]) / dx * f1.u[index] +
        (f1.v[index_yp] - f1.v[index_ym]) / (2 * dx) * f1.p[index] +
        (f1.p[index_yp] - f1.p[index_ym]) / (2 * dx) * f1.v[index] +
        (f1.w[index_zp] - f1.w[index]) / dx * f1.p[index] +
        (f1.p[index_zp] - f1.p[index]) / dx * f1.w[index]
    );

}



__global__ void updateEdgeZ2X1(data d) {
    int totalThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if(totalThreadIndex >= 298) return;


    int x = 0;
    int y = totalThreadIndex + 1;
    int z = 299;
    int index = x + y * 300 + z * 300 * 300;
    int index_xpp = (x+2) + y * 300 + z * 300 * 300;
    int index_xp = (x+1) + y * 300 + z * 300 * 300;
    int index_ym = x + (y-1) * 300 + z * 300 * 300;
    int index_yp = x + (y+1) * 300 + z * 300 * 300;
    int index_zm = x + y * 300 + (z-1) * 300 * 300;
    int index_zmm = x + y * 300 + (z-2) * 300 * 300;



    // Define origin and destination fields
    field f1 = d.inMain ? d.f1 : d.f2;
    field f2 = d.inMain ? d.f2 : d.f1;

    // Define dx, dt and RT
    float dx = *d.dx;
    float dt = *d.dt;
    float RT = *d.RT;
    float mu = *d.mu;
    float g = *d.g;

    // Update velocity field
    f2.u[index] = 0;

    f2.v[index] = f1.v[index] + dt *
    (
        - f1.u[index] * (f1.v[index_xp] - f1.v[index]) / dx
        - f1.v[index] * (f1.v[index_yp] - f1.v[index_ym]) / (2 * dx)
        - f1.w[index] * (f1.v[index] - f1.v[index_zm]) / dx
        - (RT/f1.p[index]) * (f1.p[index_yp] - f1.p[index_ym]) / (2 * dx)
        + mu * (RT/f1.p[index]) * (
            (f1.v[index_xpp] - 2 * f1.v[index_xp] + f1.v[index]) / (dx * dx) +
            (f1.v[index_yp] - 2 * f1.v[index] + f1.v[index_ym]) / (dx * dx) +
            (f1.v[index] - 2 * f1.v[index_zm] + f1.v[index_zmm]) / (dx * dx)
        )
    );

    f2.w[index] = 0;
    // Update pressure field
    f2.p[index] = f1.p[index] - (dt) * 
    (
        (f1.u[index_xp] - f1.u[index]) / dx * f1.p[index] +
        (f1.p[index_xp] - f1.p[index]) / dx * f1.u[index] +
        (f1.v[index_yp] - f1.v[index_ym]) / (2 * dx) * f1.p[index] +
        (f1.p[index_yp] - f1.p[index_ym]) / (2 * dx) * f1.v[index] +
        (f1.w[index] - f1.w[index_zm]) / dx * f1.p[index] +
        (f1.p[index] - f1.p[index_zm]) / dx * f1.w[index]
    );

}



__global__ void updateEdgeZ2X2(data d) {
    int totalThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if(totalThreadIndex >= 298) return;


    int x = 299;
    int y = totalThreadIndex + 1;
    int z = 299;
    int index = x + y * 300 + z * 300 * 300;
    int index_xm = (x-1) + y * 300 + z * 300 * 300;
    int index_xmm = (x-2) + y * 300 + z * 300 * 300;
    int index_ym = x + (y-1) * 300 + z * 300 * 300;
    int index_yp = x + (y+1) * 300 + z * 300 * 300;
    int index_zm = x + y * 300 + (z-1) * 300 * 300;
    int index_zmm = x + y * 300 + (z-2) * 300 * 300;



    // Define origin and destination fields
    field f1 = d.inMain ? d.f1 : d.f2;
    field f2 = d.inMain ? d.f2 : d.f1;

    // Define dx, dt and RT
    float dx = *d.dx;
    float dt = *d.dt;
    float RT = *d.RT;
    float mu = *d.mu;
    float g = *d.g;

    // Update velocity field
    f2.u[index] = 0;

    f2.v[index] = f1.v[index] + dt *
    (
        - f1.u[index] * (f1.v[index] - f1.v[index_xm]) / dx
        - f1.v[index] * (f1.v[index_yp] - f1.v[index_ym]) / (2 * dx)
        - f1.w[index] * (f1.v[index] - f1.v[index_zm]) / dx
        - (RT/f1.p[index]) * (f1.p[index_yp] - f1.p[index_ym]) / (2 * dx)
        + mu * (RT/f1.p[index]) * (
            (f1.v[index] - 2 * f1.v[index_xm] + f1.v[index_xmm]) / (dx * dx) +
            (f1.v[index_yp] - 2 * f1.v[index] + f1.v[index_ym]) / (dx * dx) +
            (f1.v[index] - 2 * f1.v[index_zm] + f1.v[index_zmm]) / (dx * dx)
        )
    );

    f2.w[index] = 0;
    // Update pressure field
    f2.p[index] = f1.p[index] - (dt) * 
    (
        (f1.u[index] - f1.u[index_xm]) / dx * f1.p[index] +
        (f1.p[index] - f1.p[index_xm]) / dx * f1.u[index] +
        (f1.v[index_yp] - f1.v[index_ym]) / (2 * dx) * f1.p[index] +
        (f1.p[index_yp] - f1.p[index_ym]) / (2 * dx) * f1.v[index] +
        (f1.w[index] - f1.w[index_zm]) / dx * f1.p[index] +
        (f1.p[index] - f1.p[index_zm]) / dx * f1.w[index]
    );

}



__global__ void updateCornerZ1X1Y1(data d) {

    int x = 0;
    int y = 0;
    int z = 0;
    int index = x + y * 300 + z * 300 * 300;
    int index_xpp = (x+2) + y * 300 + z * 300 * 300;
    int index_xp = (x+1) + y * 300 + z * 300 * 300;
    int index_ypp = x + (y+2) * 300 + z * 300 * 300;
    int index_yp = x + (y+1) * 300 + z * 300 * 300;
    int index_zpp = x + y * 300 + (z+2) * 300 * 300;
    int index_zp = x + y * 300 + (z+1) * 300 * 300;



    // Define origin and destination fields
    field f1 = d.inMain ? d.f1 : d.f2;
    field f2 = d.inMain ? d.f2 : d.f1;

    // Define dx, dt and RT
    float dx = *d.dx;
    float dt = *d.dt;
    float RT = *d.RT;
    float mu = *d.mu;
    float g = *d.g;

    // Update velocity field
    f2.u[index] = 0;
    f2.v[index] = 0;
    f2.w[index] = 0;
    // Update pressure field
    f2.p[index] = f1.p[index] - (dt) * 
    (
        (f1.u[index_xp] - f1.u[index]) / dx * f1.p[index] +
        (f1.p[index_xp] - f1.p[index]) / dx * f1.u[index] +
        (f1.v[index_yp] - f1.v[index]) / dx * f1.p[index] +
        (f1.p[index_yp] - f1.p[index]) / dx * f1.v[index] +
        (f1.w[index_zp] - f1.w[index]) / dx * f1.p[index] +
        (f1.p[index_zp] - f1.p[index]) / dx * f1.w[index]
    );

}



__global__ void updateCornerZ1X1Y2(data d) {

    int x = 0;
    int y = 300 - 1;
    int z = 0;
    int index = x + y * 300 + z * 300 * 300;
    int index_xpp = (x+2) + y * 300 + z * 300 * 300;
    int index_xp = (x+1) + y * 300 + z * 300 * 300;
    int index_ym = x + (y-1) * 300 + z * 300 * 300;
    int index_ymm = x + (y-2) * 300 + z * 300 * 300;
    int index_zpp = x + y * 300 + (z+2) * 300 * 300;
    int index_zp = x + y * 300 + (z+1) * 300 * 300;



    // Define origin and destination fields
    field f1 = d.inMain ? d.f1 : d.f2;
    field f2 = d.inMain ? d.f2 : d.f1;

    // Define dx, dt and RT
    float dx = *d.dx;
    float dt = *d.dt;
    float RT = *d.RT;
    float mu = *d.mu;
    float g = *d.g;

    // Update velocity field
    f2.u[index] = 0;
    f2.v[index] = 0;
    f2.w[index] = 0;
    // Update pressure field
    f2.p[index] = f1.p[index] - (dt) * 
    (
        (f1.u[index_xp] - f1.u[index]) / dx * f1.p[index] +
        (f1.p[index_xp] - f1.p[index]) / dx * f1.u[index] +
        (f1.v[index] - f1.v[index_ym]) / dx * f1.p[index] +
        (f1.p[index] - f1.p[index_ym]) / dx * f1.v[index] +
        (f1.w[index_zp] - f1.w[index]) / dx * f1.p[index] +
        (f1.p[index_zp] - f1.p[index]) / dx * f1.w[index]
    );

}



__global__ void updateCornerZ1X2Y1(data d) {

    int x = 300 - 1;
    int y = 0;
    int z = 0;
    int index = x + y * 300 + z * 300 * 300;
    int index_xm = (x-1) + y * 300 + z * 300 * 300;
    int index_xmm = (x-2) + y * 300 + z * 300 * 300;
    int index_ypp = x + (y+2) * 300 + z * 300 * 300;
    int index_yp = x + (y+1) * 300 + z * 300 * 300;
    int index_zpp = x + y * 300 + (z+2) * 300 * 300;
    int index_zp = x + y * 300 + (z+1) * 300 * 300;



    // Define origin and destination fields
    field f1 = d.inMain ? d.f1 : d.f2;
    field f2 = d.inMain ? d.f2 : d.f1;

    // Define dx, dt and RT
    float dx = *d.dx;
    float dt = *d.dt;
    float RT = *d.RT;
    float mu = *d.mu;
    float g = *d.g;

    // Update velocity field
    f2.u[index] = 0;
    f2.v[index] = 0;
    f2.w[index] = 0;
    // Update pressure field
    f2.p[index] = f1.p[index] - (dt) * 
    (
        (f1.u[index] - f1.u[index_xm]) / dx * f1.p[index] +
        (f1.p[index] - f1.p[index_xm]) / dx * f1.u[index] +
        (f1.v[index_yp] - f1.v[index]) / dx * f1.p[index] +
        (f1.p[index_yp] - f1.p[index]) / dx * f1.v[index] +
        (f1.w[index_zp] - f1.w[index]) / dx * f1.p[index] +
        (f1.p[index_zp] - f1.p[index]) / dx * f1.w[index]
    );

}



__global__ void updateCornerZ1X2Y2(data d) {

    int x = 300 - 1;
    int y = 300 - 1;
    int z = 0;
    int index = x + y * 300 + z * 300 * 300;
    int index_xm = (x-1) + y * 300 + z * 300 * 300;
    int index_xmm = (x-2) + y * 300 + z * 300 * 300;
    int index_ym = x + (y-1) * 300 + z * 300 * 300;
    int index_ymm = x + (y-2) * 300 + z * 300 * 300;
    int index_zpp = x + y * 300 + (z+2) * 300 * 300;
    int index_zp = x + y * 300 + (z+1) * 300 * 300;



    // Define origin and destination fields
    field f1 = d.inMain ? d.f1 : d.f2;
    field f2 = d.inMain ? d.f2 : d.f1;

    // Define dx, dt and RT
    float dx = *d.dx;
    float dt = *d.dt;
    float RT = *d.RT;
    float mu = *d.mu;
    float g = *d.g;

    // Update velocity field
    f2.u[index] = 0;
    f2.v[index] = 0;
    f2.w[index] = 0;
    // Update pressure field
    f2.p[index] = f1.p[index] - (dt) * 
    (
        (f1.u[index] - f1.u[index_xm]) / dx * f1.p[index] +
        (f1.p[index] - f1.p[index_xm]) / dx * f1.u[index] +
        (f1.v[index] - f1.v[index_ym]) / dx * f1.p[index] +
        (f1.p[index] - f1.p[index_ym]) / dx * f1.v[index] +
        (f1.w[index_zp] - f1.w[index]) / dx * f1.p[index] +
        (f1.p[index_zp] - f1.p[index]) / dx * f1.w[index]
    );

}



__global__ void updateCornerZ2X1Y1(data d) {

    int x = 0;
    int y = 0;
    int z = 300 - 1;
    int index = x + y * 300 + z * 300 * 300;
    int index_xpp = (x+2) + y * 300 + z * 300 * 300;
    int index_xp = (x+1) + y * 300 + z * 300 * 300;
    int index_ypp = x + (y+2) * 300 + z * 300 * 300;
    int index_yp = x + (y+1) * 300 + z * 300 * 300;
    int index_zm = x + y * 300 + (z-1) * 300 * 300;
    int index_zmm = x + y * 300 + (z-2) * 300 * 300;



    // Define origin and destination fields
    field f1 = d.inMain ? d.f1 : d.f2;
    field f2 = d.inMain ? d.f2 : d.f1;

    // Define dx, dt and RT
    float dx = *d.dx;
    float dt = *d.dt;
    float RT = *d.RT;
    float mu = *d.mu;
    float g = *d.g;

    // Update velocity field
    f2.u[index] = 0;
    f2.v[index] = 0;
    f2.w[index] = 0;
    // Update pressure field
    f2.p[index] = f1.p[index] - (dt) * 
    (
        (f1.u[index_xp] - f1.u[index]) / dx * f1.p[index] +
        (f1.p[index_xp] - f1.p[index]) / dx * f1.u[index] +
        (f1.v[index_yp] - f1.v[index]) / dx * f1.p[index] +
        (f1.p[index_yp] - f1.p[index]) / dx * f1.v[index] +
        (f1.w[index] - f1.w[index_zm]) / dx * f1.p[index] +
        (f1.p[index] - f1.p[index_zm]) / dx * f1.w[index]
    );

}



__global__ void updateCornerZ2X1Y2(data d) {

    int x = 0;
    int y = 300 - 1;
    int z = 300 - 1;
    int index = x + y * 300 + z * 300 * 300;
    int index_xpp = (x+2) + y * 300 + z * 300 * 300;
    int index_xp = (x+1) + y * 300 + z * 300 * 300;
    int index_ym = x + (y-1) * 300 + z * 300 * 300;
    int index_ymm = x + (y-2) * 300 + z * 300 * 300;
    int index_zm = x + y * 300 + (z-1) * 300 * 300;
    int index_zmm = x + y * 300 + (z-2) * 300 * 300;



    // Define origin and destination fields
    field f1 = d.inMain ? d.f1 : d.f2;
    field f2 = d.inMain ? d.f2 : d.f1;

    // Define dx, dt and RT
    float dx = *d.dx;
    float dt = *d.dt;
    float RT = *d.RT;
    float mu = *d.mu;
    float g = *d.g;

    // Update velocity field
    f2.u[index] = 0;
    f2.v[index] = 0;
    f2.w[index] = 0;
    // Update pressure field
    f2.p[index] = f1.p[index] - (dt) * 
    (
        (f1.u[index_xp] - f1.u[index]) / dx * f1.p[index] +
        (f1.p[index_xp] - f1.p[index]) / dx * f1.u[index] +
        (f1.v[index] - f1.v[index_ym]) / dx * f1.p[index] +
        (f1.p[index] - f1.p[index_ym]) / dx * f1.v[index] +
        (f1.w[index] - f1.w[index_zm]) / dx * f1.p[index] +
        (f1.p[index] - f1.p[index_zm]) / dx * f1.w[index]
    );

}



__global__ void updateCornerZ2X2Y1(data d) {

    int x = 300 - 1;
    int y = 0;
    int z = 300 - 1;
    int index = x + y * 300 + z * 300 * 300;
    int index_xm = (x-1) + y * 300 + z * 300 * 300;
    int index_xmm = (x-2) + y * 300 + z * 300 * 300;
    int index_ypp = x + (y+2) * 300 + z * 300 * 300;
    int index_yp = x + (y+1) * 300 + z * 300 * 300;
    int index_zm = x + y * 300 + (z-1) * 300 * 300;
    int index_zmm = x + y * 300 + (z-2) * 300 * 300;



    // Define origin and destination fields
    field f1 = d.inMain ? d.f1 : d.f2;
    field f2 = d.inMain ? d.f2 : d.f1;

    // Define dx, dt and RT
    float dx = *d.dx;
    float dt = *d.dt;
    float RT = *d.RT;
    float mu = *d.mu;
    float g = *d.g;

    // Update velocity field
    f2.u[index] = 0;
    f2.v[index] = 0;
    f2.w[index] = 0;
    // Update pressure field
    f2.p[index] = f1.p[index] - (dt) * 
    (
        (f1.u[index] - f1.u[index_xm]) / dx * f1.p[index] +
        (f1.p[index] - f1.p[index_xm]) / dx * f1.u[index] +
        (f1.v[index_yp] - f1.v[index]) / dx * f1.p[index] +
        (f1.p[index_yp] - f1.p[index]) / dx * f1.v[index] +
        (f1.w[index] - f1.w[index_zm]) / dx * f1.p[index] +
        (f1.p[index] - f1.p[index_zm]) / dx * f1.w[index]
    );

}



__global__ void updateCornerZ2X2Y2(data d) {

    int x = 300 - 1;
    int y = 300 - 1;
    int z = 300 - 1;
    int index = x + y * 300 + z * 300 * 300;
    int index_xm = (x-1) + y * 300 + z * 300 * 300;
    int index_xmm = (x-2) + y * 300 + z * 300 * 300;
    int index_ym = x + (y-1) * 300 + z * 300 * 300;
    int index_ymm = x + (y-2) * 300 + z * 300 * 300;
    int index_zm = x + y * 300 + (z-1) * 300 * 300;
    int index_zmm = x + y * 300 + (z-2) * 300 * 300;



    // Define origin and destination fields
    field f1 = d.inMain ? d.f1 : d.f2;
    field f2 = d.inMain ? d.f2 : d.f1;

    // Define dx, dt and RT
    float dx = *d.dx;
    float dt = *d.dt;
    float RT = *d.RT;
    float mu = *d.mu;
    float g = *d.g;

    // Update velocity field
    f2.u[index] = 0;
    f2.v[index] = 0;
    f2.w[index] = 0;
    // Update pressure field
    f2.p[index] = f1.p[index] - (dt) * 
    (
        (f1.u[index] - f1.u[index_xm]) / dx * f1.p[index] +
        (f1.p[index] - f1.p[index_xm]) / dx * f1.u[index] +
        (f1.v[index] - f1.v[index_ym]) / dx * f1.p[index] +
        (f1.p[index] - f1.p[index_ym]) / dx * f1.v[index] +
        (f1.w[index] - f1.w[index_zm]) / dx * f1.p[index] +
        (f1.p[index] - f1.p[index_zm]) / dx * f1.w[index]
    );

}


void step(data d) {
    updateCore<<<103374, 256>>>(d);
    updateFaceX1<<<347, 256>>>(d);
    updateFaceX2<<<347, 256>>>(d);
    updateFaceY1<<<347, 256>>>(d);
    updateFaceY2<<<347, 256>>>(d);
    updateFaceZ1<<<347, 256>>>(d);
    updateFaceZ2<<<347, 256>>>(d);
    updateEdgeX1Y1<<<2, 256>>>(d);
    updateEdgeX1Y2<<<2, 256>>>(d);
    updateEdgeX2Y1<<<2, 256>>>(d);
    updateEdgeX2Y2<<<2, 256>>>(d);
    updateEdgeX1Z1<<<2, 256>>>(d);
    updateEdgeX1Z2<<<2, 256>>>(d);
    updateEdgeX2Z1<<<2, 256>>>(d);
    updateEdgeX2Z2<<<2, 256>>>(d);
    updateEdgeZ1X1<<<2, 256>>>(d);
    updateEdgeZ1X2<<<2, 256>>>(d);
    updateEdgeZ2X1<<<2, 256>>>(d);
    updateEdgeZ2X2<<<2, 256>>>(d);
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