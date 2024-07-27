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
__device__ float getDerivative(float* field, int requestCode, bool* boundaryFlags, 
    float dx, int index, int* extendedNeighbors) {

    // Define indexes
    int index_xm = extendedNeighbors[0];
    int index_xmm = extendedNeighbors[1];
    int index_xp = extendedNeighbors[2];
    int index_xpp = extendedNeighbors[3];
    int index_ym = extendedNeighbors[4];
    int index_ymm = extendedNeighbors[5];
    int index_yp = extendedNeighbors[6];
    int index_ypp = extendedNeighbors[7];
    int index_zm = extendedNeighbors[8];
    int index_zmm = extendedNeighbors[9];
    int index_zp = extendedNeighbors[10];
    int index_zpp = extendedNeighbors[11];

    switch (requestCode) {
        case 0: // df/dx
            if (boundaryFlags[0]) {
                // Compute forward derivative
                return (field[index_xp] - field[index]) / dx;
            } else if (boundaryFlags[1]) {
                // Compute backward derivative
                return (field[index] - field[index_xm]) / dx;
            } else {
                // Compute central derivative
                return (field[index_xp] - field[index_xm]) / (2 * dx);
            }
            break;
        case 1: // df/dy
            if (boundaryFlags[2]) {
                // Compute forward derivative
                return (field[index_yp] - field[index]) / dx;
            } else if (boundaryFlags[3]) {
                // Compute backward derivative
                return (field[index] - field[index_ym]) / dx;
            } else {
                // Compute central derivative
                return (field[index_yp] - field[index_ym]) / (2 * dx);
            }
            break;
        case 2: // df/dz
            if (boundaryFlags[4]) {
                // Compute forward derivative
                return (field[index_zp] - field[index]) / dx;
            } else if (boundaryFlags[5]) {
                // Compute backward derivative
                return (field[index] - field[index_zm]) / dx;
            } else {
                // Compute central derivative
                return (field[index_zp] - field[index_zm]) / (2 * dx);
            }
            break;
        case 3: // d2f/dx2
            if (boundaryFlags[0]) {
                // Compute forward derivative
                return (field[index_xpp] - 2 * field[index_xp] + field[index]) / (dx * dx);
            } else if (boundaryFlags[1]) {
                // Compute backward derivative
                return (field[index] - 2 * field[index_xm] + field[index_xmm]) / (dx * dx);
            } else {
                // Compute central derivative
                return (field[index_xp] - 2 * field[index] + field[index_xm]) / (dx * dx);
            }
            break;
        case 4: // d2f/dy2
            if (boundaryFlags[2]) {
                // Compute forward derivative
                return (field[index_ypp] - 2 * field[index_yp] + field[index]) / (dx * dx);
            } else if (boundaryFlags[3]) {
                // Compute backward derivative
                return (field[index] - 2 * field[index_ym] + field[index_ymm]) / (dx * dx);
            } else {
                // Compute central derivative
                return (field[index_yp] - 2 * field[index] + field[index_ym]) / (dx * dx);
            }
            break;
        case 5: // d2f/dz2
            if (boundaryFlags[4]) {
                // Compute forward derivative
                return (field[index_zpp] - 2 * field[index_zp] + field[index]) / (dx * dx);
            } else if (boundaryFlags[5]) {
                // Compute backward derivative
                return (field[index] - 2 * field[index_zm] + field[index_zmm]) / (dx * dx);
            } else {
                // Compute central derivative
                return (field[index_zp] - 2 * field[index] + field[index_zm]) / (dx * dx);
            }
            break;
        default:
            break;
    }
}

__global__ void updateField(data d, int side) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= side * side * side) return;

    // Compute neighbors and boundary flags
    int x = index % side;
    int y = (index / side) % side;
    int z = index / (side * side);
    int extendedNeighbors[12] = {
        (x - 1) + y * side + z * side * side,
        (x - 2) + y * side + z * side * side,
        (x + 1) + y * side + z * side * side,
        (x + 2) + y * side + z * side * side,
        x + (y - 1) * side + z * side * side,
        x + (y - 2) * side + z * side * side,
        x + (y + 1) * side + z * side * side,
        x + (y + 2) * side + z * side * side,
        x + y * side + (z - 1) * side * side,
        x + y * side + (z - 2) * side * side,
        x + y * side + (z + 1) * side * side,
        x + y * side + (z + 2) * side * side
    };
    int index_xm = extendedNeighbors[0];
    int index_xp = extendedNeighbors[2];
    int index_ym = extendedNeighbors[4];
    int index_yp = extendedNeighbors[6];
    int index_zm = extendedNeighbors[8];
    int index_zp = extendedNeighbors[10];
    bool boundaryFlags[6] = {
        x == 0,
        x == side - 1,
        y == 0,
        y == side - 1,
        z == 0,
        z == side - 1
    };

    // Define origin and destination fields
    field f1 = d.inMain ? d.f1 : d.f2;
    field f2 = d.inMain ? d.f2 : d.f1;

    // Define dx, dt and RT
    float dx = *d.dx;
    float dt = *d.dt;
    float RT = *d.RT;
    float mu = *d.mu;
    float g = *d.g;

    // Separate boundary conditions from the rest
    if (x > 0 && x < side - 1 && y > 0 && y < side - 1 && z > 0 && z < side - 1) {
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
        f2.v[index] = f2.v[index] + dt *
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
        f2.w[index] = f2.w[index] + dt *
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
        f2.p[index] = f1.p[index] + 
            (- dt / (2 * dx)) * 
            (f1.u[index_xp] * f1.p[index_xp] - f1.u[index_xm] * f1.p[index_xm] +
             f1.v[index_yp] * f1.p[index_yp] - f1.v[index_ym] * f1.p[index_ym] +
             f1.w[index_zp] * f1.p[index_zp] - f1.w[index_zm] * f1.p[index_zm]);
    } else {
        // Update u at boundaries
        if (x == 0 || x == side - 1) {
            f2.u[index] = 0.0f;
        } else {
            f2.u[index] = f1.u[index] + dt *
        (
            - f1.u[index] * getDerivative(f1.u, 0, boundaryFlags, dx, index, extendedNeighbors)
            - f1.v[index] * getDerivative(f1.u, 1, boundaryFlags, dx, index, extendedNeighbors)
            - f1.w[index] * getDerivative(f1.u, 2, boundaryFlags, dx, index, extendedNeighbors)
            - (RT/f1.p[index]) * getDerivative(f1.p, 0, boundaryFlags, dx, index, extendedNeighbors)
            + mu * (RT/f1.p[index]) * (
                getDerivative(f1.u, 3, boundaryFlags, dx, index, extendedNeighbors) +
                getDerivative(f1.u, 4, boundaryFlags, dx, index, extendedNeighbors) +
                getDerivative(f1.u, 5, boundaryFlags, dx, index, extendedNeighbors)
            )
        );
        }
        // Update v at boundaries
        if (y == 0 || y == side - 1) {
            f2.v[index] = 0.0f;
        } else {
            f2.v[index] = f1.v[index] + dt *
        (
            - f1.u[index] * getDerivative(f1.v, 0, boundaryFlags, dx, index, extendedNeighbors)
            - f1.v[index] * getDerivative(f1.v, 1, boundaryFlags, dx, index, extendedNeighbors)
            - f1.w[index] * getDerivative(f1.v, 2, boundaryFlags, dx, index, extendedNeighbors)
            - (RT/f1.p[index]) * getDerivative(f1.p, 1, boundaryFlags, dx, index, extendedNeighbors)
            + mu * (RT/f1.p[index]) * (
                getDerivative(f1.v, 3, boundaryFlags, dx, index, extendedNeighbors) +
                getDerivative(f1.v, 4, boundaryFlags, dx, index, extendedNeighbors) +
                getDerivative(f1.v, 5, boundaryFlags, dx, index, extendedNeighbors)
            )
        );
        }
        // Update w at boundaries
        if (z == 0 || z == side - 1) {
            f2.w[index] = 0.0f;
        } else {
            f2.w[index] = f1.w[index] + dt *
        (
            - f1.u[index] * getDerivative(f1.w, 0, boundaryFlags, dx, index, extendedNeighbors)
            - f1.v[index] * getDerivative(f1.w, 1, boundaryFlags, dx, index, extendedNeighbors)
            - f1.w[index] * getDerivative(f1.w, 2, boundaryFlags, dx, index, extendedNeighbors)
            - (RT/f1.p[index]) * getDerivative(f1.p, 2, boundaryFlags, dx, index, extendedNeighbors)
            + mu * (RT/f1.p[index]) * (
                getDerivative(f1.w, 3, boundaryFlags, dx, index, extendedNeighbors) +
                getDerivative(f1.w, 4, boundaryFlags, dx, index, extendedNeighbors) +
                getDerivative(f1.w, 5, boundaryFlags, dx, index, extendedNeighbors)
            )
        );
        }
        // Update p at boundaries
        f2.p[index] = f1.p[index] - (dt) * 
            (getDerivative(f1.u, 0, boundaryFlags, dx, index, extendedNeighbors) * f1.p[index] +
                getDerivative(f1.p, 0, boundaryFlags, dx, index, extendedNeighbors) * f1.u[index] +
             getDerivative(f1.v, 1, boundaryFlags, dx, index, extendedNeighbors) * f1.p[index] +
                getDerivative(f1.p, 1, boundaryFlags, dx, index, extendedNeighbors) * f1.v[index] +
             getDerivative(f1.w, 2, boundaryFlags, dx, index, extendedNeighbors) * f1.p[index] +
                getDerivative(f1.p, 2, boundaryFlags, dx, index, extendedNeighbors) * f1.w[index]);
    }
}

// *** PYTHON CODE-GENERATED KERNEL DEFINITIONS ***

void step(data d) {
    //updateField<<<numBlocks, threadsPerBlock>>>(d, side);
    // *** PYTHON CODE-GENERATED KERNEL CALLS ***
    cudaDeviceSynchronize();
}