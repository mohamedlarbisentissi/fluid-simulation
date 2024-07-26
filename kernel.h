#include <cuda_runtime.h>

void initializeValues(dim3 numBlocks, dim3 threadsPerBlock, data d, int side);

void launchKernels(dim3 numBlocks, dim3 threadsPerBlock, data d, int side);

void flipInMain(data d);