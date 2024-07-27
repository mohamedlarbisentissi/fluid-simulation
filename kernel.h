#ifndef KERNEL_H
#define KERNEL_H

#include "structs.h"

void init(int numBlocks, int threadsPerBlock, data d, int side);

void step(int numBlocks, int threadsPerBlock, data d, int side);

void flipInMain(data d);

#endif