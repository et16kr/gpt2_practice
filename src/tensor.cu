#include"tensor.h"
#include <iostream>

Tensor::Tensor(const std::vector<size_t> &shape_, float *buf_) {
  ndim = shape_.size();
  for (size_t i = 0; i < ndim; i++) { shape[i] = shape_[i]; }
  size_t N_ = num_elem();

  CHECK_CUDA(cudaMalloc((void**)&gpu_buf, N_ * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(gpu_buf, buf_, N_ * sizeof(float), cudaMemcpyHostToDevice));
}

Tensor::~Tensor() {
  if (gpu_buf != nullptr) CHECK_CUDA(cudaFree(gpu_buf));
}

Tensor::Tensor(Tensor&& other) noexcept {
  ndim = other.ndim;
  for (size_t i = 0; i < 5; i++) { shape[i] = other.shape[i]; }
  gpu_buf = other.gpu_buf;

  other.ndim = 0;
  other.gpu_buf = nullptr;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
  if (this == &other) return *this;

  if (gpu_buf != nullptr) CHECK_CUDA(cudaFree(gpu_buf));

  ndim = other.ndim;
  for (size_t i = 0; i < 5; i++) { shape[i] = other.shape[i]; }
  gpu_buf = other.gpu_buf;

  other.ndim = 0;
  other.gpu_buf = nullptr;
  return *this;
}