#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.cpp_extension import CUDA_HOME, load


def build_extension() -> tuple[object, bool]:
    root = Path(__file__).resolve().parent
    src_dir = root / "src"

    use_cuda = torch.cuda.is_available() and CUDA_HOME is not None
    sources = [str(src_dir / "add_ext.cpp")]
    if use_cuda:
        sources.append(str(src_dir / "add_ext_kernel.cu"))

    kwargs = {
        "name": "add_ext_sample",
        "sources": sources,
        "verbose": True,
        "build_directory": str(root / "build"),
        "extra_cflags": ["-O3"],
    }
    if use_cuda:
        kwargs["extra_cuda_cflags"] = ["-O3"]

    ext = load(**kwargs)
    return ext, use_cuda


def main() -> None:
    ext, use_cuda = build_extension()

    device = "cuda" if use_cuda else "cpu"
    print(f"running on device={device}")

    a = torch.randn(4, 4, device=device, dtype=torch.float32)
    b = torch.randn(4, 4, device=device, dtype=torch.float32)

    out = ext.add(a, b)
    ref = a + b
    max_abs_error = (out - ref).abs().max().item()

    print("a:")
    print(a)
    print("b:")
    print(b)
    print("out (extension):")
    print(out)
    print("ref (torch +):")
    print(ref)
    print(f"max_abs_error={max_abs_error:.8f}")


if __name__ == "__main__":
    main()
