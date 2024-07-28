#ifndef STRUCTS_H
#define STRUCTS_H

struct field {
    float* p;
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
    field f1;
    field f2;
    bool* inMain;
};

#endif