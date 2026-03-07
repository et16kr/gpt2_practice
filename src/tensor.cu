#include "tensor.h"

Tensor::Tensor(const std::vector<size_t> &shape_, float *buf_) {
  size_t N_ = AllocTensor(shape_);
  CHECK_CUDA(cudaMemcpy(gpu_buf, buf_, N_ * sizeof(float), cudaMemcpyHostToDevice));
}

Tensor::Tensor(const std::vector<size_t> &shape_) {
  AllocTensor(shape_);
}

size_t Tensor::AllocTensor(const std::vector<size_t> &shape_) {
  ndim = shape_.size();
  for (size_t i = 0; i < ndim; i++) { shape[i] = shape_[i]; }
  size_t N_ = num_elem();

  CHECK_CUDA(cudaMalloc((void**)&gpu_buf, N_ * sizeof(float)));
  return N_;
}

Tensor::~Tensor() {
  if (gpu_buf != nullptr) CHECK_CUDA(cudaFree(gpu_buf));
}
