#pragma once

#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdio>
#include <vector>

#define CHECK_CUDA(call)                                                 \
  do {                                                                   \
    cudaError_t status_ = call;                                          \
    if (status_ != cudaSuccess) {                                        \
      printf("CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__,          \
             cudaGetErrorName(status_), cudaGetErrorString(status_));    \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)


struct Tensor {
  size_t ndim = 0;
  size_t shape[5] = {1, 1, 1, 1, 1};
  float *gpu_buf = nullptr;
  size_t num_elem() {
    size_t s = 1;
    for (size_t i = 0; i < ndim; i++) { s *= shape[i]; }
    return s;
  }
  Tensor() {};
  Tensor(const std::vector<size_t> &shape_);
  Tensor(const std::vector<size_t> &shape_, float *buf_);
  ~Tensor();
  Tensor(const Tensor&) = delete;
  Tensor& operator=(const Tensor&) = delete;
  size_t AllocTensor(const std::vector<size_t> &shape_);
};
