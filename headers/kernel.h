#ifndef KERNEL_H
#define KERNEL_H

#include "structs.h"

void init(int numBlocks, int threadsPerBlock, data d, int side);

void step(data d);

void flipInMain(data d);

#endif