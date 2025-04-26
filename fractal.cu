#include "fractal.cuh"
#include <cstdint>
#include <cuda_runtime.h>
#include <iostream>
#include <thrust/complex.h>

__device__ float lerp(float a, float b, float t) { return a + t * (b - a); }

__device__ void getColorFromPalette(float t, int &r, int &g, int &b) {
  // Clamp t in [0, 1]
  t = fminf(fmaxf(t, 0.0f), 1.0f);

  // Define a palette with 5 anchor points (you can add more)
  const int numColors = 5;
  float palette[numColors][3] = {
      {0.0f, 0.0f, 0.0f}, // black
      {0.2f, 0.0f, 0.5f}, // purple
      {0.0f, 0.8f, 0.8f}, // teal
      {1.0f, 1.0f, 0.0f}, // yellow
      {1.0f, 1.0f, 1.0f}, // white
  };

  float scaled = t * (numColors - 1);
  int idx = int(scaled);
  float localT = scaled - idx;

  if (idx >= numColors - 1) {
    r = int(palette[numColors - 1][0] * 255);
    g = int(palette[numColors - 1][1] * 255);
    b = int(palette[numColors - 1][2] * 255);
    return;
  }

  float *col1 = palette[idx];
  float *col2 = palette[idx + 1];

  r = int(lerp(col1[0], col2[0], localT) * 255);
  g = int(lerp(col1[1], col2[1], localT) * 255);
  b = int(lerp(col1[2], col2[2], localT) * 255);
}

__global__ void CalculateJulia2(uint32_t *pixels, int maxIter, int d_w, int d_h,
                                double c_x, double c_y, double step) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; // origin is bottom left on gpu
  int j = blockIdx.y * blockDim.y + threadIdx.y; // origin is bottom left on gpu

  if (i >= d_w || j >= d_h) {
    return;
  }

  double x = (i - d_w / 2) * 1.0f * step + c_x;
  double y = (j - d_h / 2) * 1.0f * step + c_y;

  thrust::complex<double> c(-0.7, 0.27015);
  thrust::complex<double> z(x, y);

  int iters = 0;
  while (iters < maxIter && thrust::norm(z) < 4) {
    z = z * z * z * z + c;
    iters++;
  }
  int p_i = i;
  int p_j = d_h - j - 1;

  double mod = sqrtf(thrust::norm(z));
  double smooth_iter = double(iters) - log2(max(1.0f, log2(mod)));
  double norm_iter = pow(smooth_iter / double((maxIter)), 0.6f);
  int grad = int(norm_iter * 255.0f);

  int red = (grad * 2) % 256;
  int green = (grad * 5) % 256;
  int blue = (grad * 8) % 256;

  // Store the RGBA color in the pixel array
  pixels[p_i + p_j * d_w] = (0x05 << 24) | (blue << 16) | (green << 8) | red;
}

__global__ void CalculateJulia1(uint32_t *pixels, int maxIter, int d_w, int d_h,
                                double c_x, double c_y, double step) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; // origin is bottom left on gpu
  int j = blockIdx.y * blockDim.y + threadIdx.y; // origin is bottom left on gpu

  if (i >= d_w || j >= d_h) {
    return;
  }

  double x = (i - d_w / 2) * 1.0f * step + c_x;
  double y = (j - d_h / 2) * 1.0f * step + c_y;

  thrust::complex<double> c(-0.7, 0.27015);
  thrust::complex<double> z(x, y);

  int iters = 0;
  while (iters < maxIter && thrust::norm(z) < 1000) {
    z = z * z + c;
    iters++;
  }
  int p_i = i;
  int p_j = d_h - j - 1;

  double mod = sqrtf(thrust::norm(z));
  double smooth_iter = double(iters) - log2(max(1.0f, log2(mod)));
  double norm_iter = pow(smooth_iter / double((maxIter)), 0.6f);
  int grad = int(norm_iter * 255.0f);

  int red = (grad * 2) % 256;
  int green = (grad * 5) % 256;
  int blue = (grad * 8) % 256;

  // Store the RGBA color in the pixel array
  pixels[p_i + p_j * d_w] = (0x0F << 24) | (blue << 16) | (green << 8) | red;
}

__global__ void CalculateMandelBrot(uint32_t *pixels, int maxIter, int d_w,
                                    int d_h, double c_x, double c_y,
                                    double step) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; // origin is bottom left on gpu
  int j = blockIdx.y * blockDim.y + threadIdx.y; // origin is bottom left on gpu

  if (i >= d_w || j >= d_h) {
    return;
  }

  double x = (i - d_w / 2) * 1.0f * step + c_x;
  double y = (j - d_h / 2) * 1.0f * step + c_y;

  thrust::complex<double> c(x, y);
  thrust::complex<double> z(0, 0);

  int iters = 0;
  while (iters < maxIter && thrust::norm(z) < 1000) {
    z = z * z + c;
    iters++;
  }
  int p_i = i;
  int p_j = d_h - j - 1;

  double mod = sqrtf(thrust::norm(z));
  double smooth_iter = double(iters) - log2(max(1.0f, log2(mod)));
  double norm_iter = pow(smooth_iter / double((maxIter)), 0.6f);
  int grad = int(norm_iter * 255.0f);

  int red = (grad * 2) % 256;
  int green = (grad * 5) % 256;
  int blue = (grad * 8) % 256;

  // Store the RGBA color in the pixel array
  pixels[p_i + p_j * d_w] = (0xFF << 24) | (blue << 16) | (green << 8) | red;
}

__global__ void CalculateTricon(uint32_t *pixels, int maxIter, int d_w, int d_h,
                                double c_x, double c_y, double step) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; // origin is bottom left on gpu
  int j = blockIdx.y * blockDim.y + threadIdx.y; // origin is bottom left on gpu

  if (i >= d_w || j >= d_h) {
    return;
  }

  double x = (i - d_w / 2) * 1.0f * step + c_x;
  double y = (j - d_h / 2) * 1.0f * step + c_y;

  thrust::complex<double> c(x, y);
  thrust::complex<double> z(0, 0);

  int iters = 0;
  while (iters < maxIter && thrust::norm(z) < 1000) {
    z = thrust::conj(z) * thrust::conj(z) + c;
    iters++;
  }
  int p_i = i;
  int p_j = d_h - j - 1;

  double mod = sqrtf(thrust::norm(z));
  double smooth_iter = double(iters) - log2(max(1.0f, log2(mod)));
  double norm_iter = pow(smooth_iter / double((maxIter)), 0.6f);
  int grad = int(norm_iter * 255.0f);

  int red = (grad * 2) % 256;
  int green = (grad * 5) % 256;
  int blue = (grad * 8) % 256;

  // Store the RGBA color in the pixel array
  pixels[p_i + p_j * d_w] = (0xFF << 24) | (blue << 16) | (green << 8) | red;
}

__global__ void BurningShip(uint32_t *pixels, int maxIter, int d_w, int d_h,
                            double c_x, double c_y, double step) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i >= d_w || j >= d_h)
    return;

  double x = (i - d_w / 2) * step + c_x;
  double y = (j - d_h / 2) * step + c_y;

  thrust::complex<double> c(x, y);
  thrust::complex<double> z(0, 0);

  int iters = 0;
  while (iters < maxIter && thrust::norm(z) < 1000.0) {
    // Apply abs to real and imag parts
    double zr = fabs(z.real());
    double zi = fabs(z.imag());
    z = thrust::complex<double>(zr, zi);
    z = z * z + c;
    iters++;
  }

  int p_i = i;
  int p_j = d_h - j - 1; // flip vertically

  double mod = thrust::abs(z);
  double smooth_iter = double(iters) - log2(fmax(1.0, log2(mod)));
  double norm_iter = pow(smooth_iter / double(maxIter), 0.6);
  int grad = int(norm_iter * 255.0);

  int red = (grad * 3) % 256;
  int green = (grad * 2) % 256;
  int blue = (grad * 6) % 256;

  pixels[p_i + p_j * d_w] = (0x0F << 24) | (blue << 16) | (green << 8) | red;
}
void runFractal(uint32_t *pixels, int maxIter, int d_w, int d_h, double c_x,
                double c_y, double step, int type) {

  uint32_t *d_pixels;
  cudaMalloc(&d_pixels, d_w * d_h * sizeof(uint32_t));
  dim3 blockDim(16, 16); // Define the block size
  dim3 gridDim((d_w + blockDim.x - 1) / blockDim.x,
               (d_h + blockDim.y - 1) / blockDim.y);

  switch (type) {
  case 0: {

    CalculateMandelBrot<<<gridDim, blockDim>>>(d_pixels, maxIter, d_w, d_h, c_x,
                                               c_y, step);
    break;
  }
  case 1: {

    CalculateJulia1<<<gridDim, blockDim>>>(d_pixels, maxIter, d_w, d_h, c_x,
                                           c_y, step);
    break;
  }
  case 2: {

    CalculateJulia2<<<gridDim, blockDim>>>(d_pixels, maxIter, d_w, d_h, c_x,
                                           c_y, step);
    break;
  }
  case 3: {
    CalculateTricon<<<gridDim, blockDim>>>(d_pixels, maxIter, d_w, d_h, c_x,
                                           c_y, step);
    break;
  }
  case 4: {
    BurningShip<<<gridDim, blockDim>>>(d_pixels, maxIter, d_w, d_h, c_x, c_y,
                                       step);
    break;
  }
  }
  cudaDeviceSynchronize();

  cudaMemcpy(pixels, d_pixels, sizeof(uint32_t) * d_w * d_h,
             cudaMemcpyDeviceToHost);

  cudaFree(d_pixels);
}
