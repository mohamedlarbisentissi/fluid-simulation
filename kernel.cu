#include <cuda_runtime.h>
#include "structs.h"

void initializeValues(dim3 numBlocks, dim3 threadsPerBlock, data d, int side) {
    initializeValuesKernel<<<numBlocks, threadsPerBlock>>>(d, side);
}

__global__ void initializeValuesKernel(data d, int side) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x > side && y > side && z > side) return;
    int index = x + y * side + z * side * side;
    if (index == 0) {
        *d.dx = 1e-3f;
        *d.dt = 1e-3f;
        *d.RT = 2494.2f;
        *d.mu = 1.8e-5f;
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

void launchKernel(dim3 numBlocks, dim3 threadsPerBlock, data d, int side) {
    updateField<<<numBlocks, threadsPerBlock>>>(d, side);
}

__global__ void updateField(data d, int side) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x > side && y > side && z > side) return;

    // Compute index and neighbors
    int index = x + y * side + z * side * side;
    int index_xm = (x - 1) + y * side + z * side * side;
    int index_xp = (x + 1) + y * side + z * side * side;
    int index_ym = x + (y - 1) * side + z * side * side;
    int index_yp = x + (y + 1) * side + z * side * side;
    int index_zm = x + y * side + (z - 1) * side * side;
    int index_zp = x + y * side + (z + 1) * side * side;

    // Define origin and destination fields
    field f1 = d.inMain ? d.f1 : d.f2;
    field f2 = d.inMain ? d.f2 : d.f1;

    // Define dx, dt and RT
    float dx = *d.dx;
    float dt = *d.dt;
    float RT = *d.RT;
    float mu = *d.mu;

    // Update velocity field in all cases
    if (x > 0 && x < side - 1 && y > 0 && y < side - 1 && z > 0 && z < side - 1) {
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
            + (2/3) * mu * (RT/f1.p[index]) * (
                (f1.u[index_xp] - 2 * f1.u[index] + f1.u[index_xm]) / (dx * dx) +
                (f1.v[index_yp] - 2 * f1.u[index] + f1.u[index_ym]) / (dx * dx) +
            )
        );
    }
    // Update pressure field only if it is not a boundary cell
    if (x > 0 && x < side - 1 && y > 0 && y < side - 1 && z > 0 && z < side - 1) {
        f2.p[index] = f1.p[index] + 
            (- dt / (2 * dx)) * 
            (f1.u[index_xp] * f1.p[index_xp] - f1.u[index_xm] * f1.p[index_xm] +
             f1.v[index_yp] * f1.p[index_yp] - f1.v[index_ym] * f1.p[index_ym] +
             f1.w[index_zp] * f1.p[index_zp] - f1.w[index_zm] * f1.p[index_zm]);
    }
}

void flipInMain(data d) {
    flipInMainKernel<<<1, 1>>>(d);
}

__global__ void flipInMainKernel(data d) {
    *d.inMain = !*d.inMain;
}