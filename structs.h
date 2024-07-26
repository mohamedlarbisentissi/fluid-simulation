struct data {
    float* dx;
    float* dt;
    float* RT;
    float* mu;
    field f1;
    field f2;
    bool* inMain;
};

struct field {
    float* p;
    float* u;
    float* v;
    float* w;
};