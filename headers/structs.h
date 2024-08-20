#ifndef STRUCTS_H
#define STRUCTS_H

#include <thrust/device_vector.h>

struct field {
    float* m;
    float* u;
    float* v;
    float* w;
};

struct data {
    float* dx;
    float* dt;
    float* RT;
    float* mu;
    float* g;
    float* small_dt;
    thrust::device_vector<int> okayStep;
    field f1;
    field f2;
    bool* inMain;
};

#endif