#ifndef DATA_SAVER_H
#define DATA_SAVER_H
#include "structs.h"
#include <string>
#include <fstream>

class DataSaver {
    int side;
    float* p;
    float* u;
    float* v;
    float* w;
    std::ofstream file;
    const std::string filename = "../visualization/data.bin";

public:
    DataSaver(int side);
    void saveData(int t, data d, int side);
    ~DataSaver();
};

#endif