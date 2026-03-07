#pragma once

#include <cuda_runtime.h>
#include "tensor.h"


namespace gpt2 {
    void Embedding(Tensor* input_is, Tensor* wte, Tensor* wpe, Tensor* output);
}