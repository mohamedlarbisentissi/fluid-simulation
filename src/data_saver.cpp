#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include "data_saver.h"

DataSaver::DataSaver(int side) : side(side) {
    hd.f1.m = new float[side * side * side];
    hd.f1.u = new float[side * side * side];
    hd.f1.v = new float[side * side * side];
    hd.f1.w = new float[side * side * side];
    file.open(filename, std::ios::binary);
}

void DataSaver::saveData(int t, data d, int side) {
    cudaMemcpy(hd.f1.m, d.f1.m, side * side * side * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hd.f1.u, d.f1.u, side * side * side * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hd.f1.v, d.f1.v, side * side * side * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hd.f1.w, d.f1.w, side * side * side * sizeof(float), cudaMemcpyDeviceToHost);
    file.write(reinterpret_cast<char*>(hd.f1.m), side * side * side * sizeof(float));
    file.write(reinterpret_cast<char*>(hd.f1.u), side * side * side * sizeof(float));
    file.write(reinterpret_cast<char*>(hd.f1.v), side * side * side * sizeof(float));
    file.write(reinterpret_cast<char*>(hd.f1.w), side * side * side * sizeof(float));
}

DataSaver::~DataSaver() {
    delete[] hd.f1.m;
    delete[] hd.f1.u;
    delete[] hd.f1.v;
    delete[] hd.f1.w;
    file.close();
}