#include "render.hpp"
//#include "heat_lut.hpp"
#include <spdlog/spdlog.h>
#include <cassert>
#include <iostream>

[[gnu::noinline]]
void _abortError(const char* msg, const char* fname, int line)
{
  cudaError_t err = cudaGetLastError();
  spdlog::error("{} ({}, line: {})", msg, fname, line);
  spdlog::error("Error {}: {}", cudaGetErrorName(err), cudaGetErrorString(err));
  std::exit(1);
}

#define abortError(msg) _abortError(msg, __FUNCTION__, __LINE__)

__device__ uchar4 palette(int x)
{
  uint8_t v = 255 * x / 100;
  return {v, v, v, 255};
}

struct rgba8_t {
  std::uint8_t r;
  std::uint8_t g;
  std::uint8_t b;
  std::uint8_t a;
};

rgba8_t heat_lut(float x)
{
  assert(0 <= x && x <= 1);
  float x0 = 1.f / 4.f;
  float x1 = 2.f / 4.f;
  float x2 = 3.f / 4.f;

  if (x < x0)
  {
    auto g = static_cast<std::uint8_t>(x / x0 * 255);
    return rgba8_t{0, g, 255, 255};
  }
  else if (x < x1)
  {
    auto b = static_cast<std::uint8_t>((x1 - x) / x0 * 255);
    return rgba8_t{0, 255, b, 255};
  }
  else if (x < x2)
  {
    auto r = static_cast<std::uint8_t>((x - x1) / x0 * 255);
    return rgba8_t{r, 255, 0, 255};
  }
  else
  {
    auto b = static_cast<std::uint8_t>((1.f - x) / x0 * 255);
    return rgba8_t{255, b, 0, 255};
  }
}

/*__device__ uchar4 heat_lut(double x)
{
  assert(0 <= x && x <= 1);
  double x0 = 1.0 / 4.0;
  double x1 = 2.0 / 4.0;
  double x2 = 3.0 / 4.0;
  double x3 = 4.0 / 4.0;

  if (x < x0)
  {
    auto g = static_cast<uint8_t>(x / x0 * 255);
    return {0, g, 255, 255};
  }
  else if (x < x1)
  {
    auto b = static_cast<uint8_t>((x1 - x) / x0 * 255);
    return {0, 255, b, 255};
  }
  else if (x < x2)
  {
    auto r = static_cast<uint8_t>((x - x1) / x0 * 255);
    return {r, 255, 0, 255};
  }
  else if (x < x3)
  {
    auto b = static_cast<uint8_t>((1.0 - x) / x0 * 255);
    return {255, b, 0, 255};
  }
  else
  {
    return {0, 0, 0, 255};
  }
}*/

// Device code
__global__ void compute_iter(int* buffer, int width, int height, int max_iter)
{
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  int gid = y * width + x;

  double mx0 = ((3.5 / ((double)width - 1.0)) * (double)x) - 2.5; // -2.5 to 1 / (x / width) * 3.5 - 2.5
  double my0 = ((2.0 / ((double)height - 1.0)) * (double)y) - 1.0; // -1 to 1 / (y / height) * 2 - 1

  double mx = 0;
  double my = 0;

  u_int32_t n_iterations = 0;

  while (mx * mx + my * my < 4 && n_iterations < max_iter)
  {
    double mxtemp = mx * mx - my * my + mx0;
    my = 2 * mx * my + my0;
    mx = mxtemp;
    n_iterations++;
  }

  buffer[gid] = n_iterations;
}

/*__global__ void compute_LUT(const uint32_t* buffer, int width, int height, size_t pitch, int max_iter, uchar4* LUT)
{
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  int gid = y * width + x;

  uchar4 color = heat_lut((double)buffer[gid] / (double)max_iter);
  LUT[gid] = color;
}*/

__global__ void apply_LUT(char* buffer,int* iter_buffer, int width, int height, size_t pitch, int max_iter, const rgba8_t* LUT)
{
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  int gid = y * width + x;

  rgba8_t color = LUT[iter_buffer[gid]];
  buffer[gid * 4 + 0] = color.r;
  buffer[gid * 4 + 1] = color.g;
  buffer[gid * 4 + 2] = color.b;
  buffer[gid * 4 + 3] = color.a;
}

void render(char* hostBuffer, int width, int height, std::ptrdiff_t stride, int n_iterations)
{
  cudaError_t err = cudaSuccess;

  int HistogramByteSize = n_iterations * sizeof(int);
  int *histogram = new int[n_iterations];
  memset(histogram, 0, HistogramByteSize);

  int ImageSize = width * height;

  dim3 dimBlock(32, 32);
  int w = std::ceil((float)width / dimBlock.x);
  int h = std::ceil((float)height / dimBlock.y);
  dim3 dimGrid(w, h);

  int *iterBuffer = new int[ImageSize];
  int *dev_iterBuffer;
  err = cudaMalloc((void**)&dev_iterBuffer, ImageSize * sizeof(int));
  if (err != cudaSuccess) abortError("cudaMalloc failed");

  compute_iter<<<dimGrid, dimBlock>>>(dev_iterBuffer, width, height, n_iterations);
  cudaDeviceSynchronize();
  
  err = cudaMemcpy(iterBuffer, dev_iterBuffer, ImageSize * sizeof(int), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) abortError("cudaMemcpy failed");

  for (int i = 0; i < ImageSize; i++)
  {
    histogram[iterBuffer[i]]++;
  }

  rgba8_t *hue = new rgba8_t[n_iterations + 1];
  for (int i = 0; i < n_iterations + 1; i++)
  {
    hue[i] = rgba8_t{0, 0, 0, 255};
  }

  double sum = 0;
  for (int i = 0; i < n_iterations; i++)
    sum += histogram[i];

  double tmp = 0;
  for (int i = 0; i < n_iterations; i++)
  {
    tmp += histogram[i] / sum;
    hue[i] = heat_lut(tmp);
  }

  rgba8_t *devLUT;
  err = cudaMalloc((void **)&devLUT, (n_iterations + 1) * sizeof(rgba8_t));
  if (err != cudaSuccess) abortError("cudaMalloc failed");
  err = cudaMemcpy(devLUT, hue, (n_iterations + 1) * sizeof(rgba8_t), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) abortError("cudaMemcpy failed");

  char* buff;
  err = cudaMalloc(&buff, ImageSize * sizeof(rgba8_t));
  if (err != cudaSuccess) abortError("cudaMalloc failed");

  apply_LUT<<<dimGrid, dimBlock>>>(buff, dev_iterBuffer, width, height, stride, n_iterations, devLUT);
  cudaDeviceSynchronize();

  err = cudaMemcpy(hostBuffer, buff, ImageSize * sizeof(rgba8_t), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) abortError("cudaMemcpy failed");

  err = cudaFree(dev_iterBuffer);
  if (err != cudaSuccess) abortError("cudaFree failed");
  err = cudaFree(devLUT);
  if (err != cudaSuccess) abortError("cudaFree failed");
  err = cudaFree(buff);
  if (err != cudaSuccess) abortError("cudaFree failed");

  delete[] iterBuffer;
  delete[] hue;
  delete[] histogram;
}

/*void render(char* hostBuffer, int width, int height, std::ptrdiff_t stride, int n_iterations)
{
  cudaError_t rc = cudaSuccess;

  // Allocate device memory
  char*  devBuffer;
  size_t pitch;

  rc = cudaMallocPitch(&devBuffer, &pitch, width * sizeof(rgba8_t), height);
  if (rc)
    abortError("Fail buffer allocation");

  // Run the kernel with blocks of size 64 x 64
  {
    int bsize = 32;
    int w     = std::ceil((float)width / bsize);
    int h     = std::ceil((float)height / bsize);

    spdlog::debug("running kernel of size ({},{})", w, h);

    dim3 dimBlock(bsize, bsize);
    dim3 dimGrid(w, h);
    compute_iter<<<dimGrid, dimBlock>>>(devBuffer, width, height, pitch);

    if (cudaPeekAtLastError())
      abortError("Computation Error");
  }

  // Copy back to main memory
  rc = cudaMemcpy2D(hostBuffer, stride, devBuffer, pitch, width * sizeof(rgba8_t), height, cudaMemcpyDeviceToHost);
  if (rc)
    abortError("Unable to copy buffer back to memory");

  // Free
  rc = cudaFree(devBuffer);
  if (rc)
    abortError("Unable to free memory");
}*/
