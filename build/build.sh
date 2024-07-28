#!/bin/bash

# Run pre-compiler
python3 ../build/custom_pre_compiler.py

# Build GPU code
/usr/bin/nvcc -G -c ../src/kernel_w.cu -o ../exec/kernel.o -I../headers

# Build CPU code
/usr/bin/g++ -O3 -c ../src/main_w.cpp -o ../exec/main.o -I../headers
/usr/bin/g++ -O3 -c ../src/data_saver.cpp -o ../exec/data_saver.o -I../headers

# Link
/usr/bin/g++ -O3 ../exec/main.o ../exec/kernel.o ../exec/data_saver.o -o ../exec/fluid_sim -L/usr/local/cuda/lib64 -lcudart