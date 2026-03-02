#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void add_kernel(
    const float* a,
    const float* b,
    float* out,
    int64_t n) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = a[idx] + b[idx];
  }
}

torch::Tensor add_cuda(torch::Tensor a, torch::Tensor b) {
  auto a_contig = a.contiguous();
  auto b_contig = b.contiguous();
  auto out = torch::zeros_like(a_contig);

  const int64_t n = a_contig.numel();
  const int threads = 256;
  const int blocks = static_cast<int>((n + threads - 1) / threads);

  add_kernel<<<blocks, threads, 0, at::cuda::getDefaultCUDAStream()>>>(
      a_contig.data_ptr<float>(),
      b_contig.data_ptr<float>(),
      out.data_ptr<float>(),
      n);

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(
      err == cudaSuccess,
      "add_kernel launch failed: ",
      cudaGetErrorString(err));

  return out.view_as(a);
}
