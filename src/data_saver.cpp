#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include "data_saver.h"

DataSaver::DataSaver(int side) : side(side) {
    p = new float[side * side * side];
    u = new float[side * side * side];
    v = new float[side * side * side];
    w = new float[side * side * side];
    file.open(filename, std::ios::binary);
}

void DataSaver::saveData(int t, data d, int side) {
    cudaMemcpy(p, d.f1.p, side * side * side * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(u, d.f1.u, side * side * side * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(v, d.f1.v, side * side * side * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(w, d.f1.w, side * side * side * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < side * side * side; i++) {
        std::cout << p[i] << " " << u[i] << " " << v[i] << " " << w[i] << std::endl;
    }
    file.write(reinterpret_cast<char*>(p), side * side * side * sizeof(float));
    file.write(reinterpret_cast<char*>(u), side * side * side * sizeof(float));
    file.write(reinterpret_cast<char*>(v), side * side * side * sizeof(float));
    file.write(reinterpret_cast<char*>(w), side * side * side * sizeof(float));
}

DataSaver::~DataSaver() {
    delete[] p;
    delete[] u;
    delete[] v;
    delete[] w;
    file.close();
}