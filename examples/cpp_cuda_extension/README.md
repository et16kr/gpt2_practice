# Python -> C++/CUDA Extension Example

This sample shows how Python loads a custom C++/CUDA operator using `torch.utils.cpp_extension.load`.

## Files

- `demo.py`: builds and loads extension, then runs `ext.add(a, b)`
- `src/add_ext.cpp`: CPU path + Python binding
- `src/add_ext_kernel.cu`: CUDA kernel + CUDA path

## Run

```bash
python3 examples/cpp_cuda_extension/demo.py
```

## Requirements

- `pip install torch`
- C++ compiler available (for CPU build)
- CUDA toolkit (`nvcc`) installed if you want CUDA kernel build

If CUDA is unavailable, `demo.py` still works with CPU-only C++ extension.
