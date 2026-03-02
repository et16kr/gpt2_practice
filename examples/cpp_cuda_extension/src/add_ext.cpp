#include <torch/extension.h>

torch::Tensor add_cuda(torch::Tensor a, torch::Tensor b);

torch::Tensor add_cpu(torch::Tensor a, torch::Tensor b) {
  return a + b;
}

torch::Tensor add(torch::Tensor a, torch::Tensor b) {
  TORCH_CHECK(a.sizes() == b.sizes(), "a and b must have the same shape");
  TORCH_CHECK(a.scalar_type() == torch::kFloat32, "a must be float32");
  TORCH_CHECK(b.scalar_type() == torch::kFloat32, "b must be float32");
  TORCH_CHECK(a.device() == b.device(), "a and b must be on the same device");

  if (a.is_cuda()) {
    return add_cuda(a, b);
  }
  return add_cpu(a, b);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("add", &add, "Add two float32 tensors (CPU/CUDA)");
}
