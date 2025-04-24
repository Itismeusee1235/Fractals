#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <iostream>

// Declare the kernel function
__global__ void Calculate(uint32_t *pixels, int maxIter, int d_w, int d_h,
                          double step);

// Host function to call the kernel
void runFractal(uint32_t *pixels, int maxIter, int d_w, int d_h, double c_x,
                double c_y, double step, int type);
