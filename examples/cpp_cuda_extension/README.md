# Python -> C++/CUDA Extension Example

This sample shows how Python loads a custom C++/CUDA operator using `torch.utils.cpp_extension.load`.

## Files

- `demo.py`: builds and loads extension, then runs `ext.add(a, b)`
- `src/add_ext.cpp`: CPU path + Python binding
- `src/add_ext_kernel.cu`: CUDA kernel + CUDA path

## Run

`demo.py`를 직접 실행하면, 내부에서 C++/CUDA extension을 먼저 빌드한 뒤 데모를 실행합니다.

```bash
python3 examples/cpp_cuda_extension/demo.py
```

`Makefile`도 동일하게 실행 시(또는 `make build`) extension 빌드를 수행합니다.

```bash
cd examples/cpp_cuda_extension
make build     # extension만 빌드/로드
make run       # 데모 실행
make run-cpu   # GPU가 있어도 CPU 경로 강제
```

## Requirements

- `pip install torch`
- C++ compiler available (for CPU build)
- CUDA toolkit (`nvcc`) installed if you want CUDA kernel build

If CUDA is unavailable, `demo.py` still works with CPU-only C++ extension.
