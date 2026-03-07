#include "kernels.h"

namespace gpt2 {

    __global__ void Embedding_Kernel(float* input_ids, float* wte, float* wpe, float* output, 
                                     size_t max_pos_emb, size_t vocab_size) {
        size_t b = blockIdx.x;
        size_t s = blockIdx.y;
        size_t h = threadIdx.x;
        size_t H = blockDim.x;
        size_t S = gridDim.y;

        if (s >= S) return;
        size_t idx = input_ids[(b* S) + s];
        if (idx >= vocab_size) return;
        output[ (b* S * H) + (s * H) + h] = wte[idx * H + h] + wpe[s*H + h];

    }
//wpe.weight [ 1024, 768 ]
//wte.weight [ 50257, 768 ]
//input_ids [ 20, 64 ]
// x0 [20, 64, 768]
    void Embedding(Tensor* input_ids, Tensor* wte, Tensor* wpe, Tensor* output) {
        size_t batch_size      = input_ids->shape[0];   // 20
        size_t sequence_length = input_ids->shape[1];   // 64
        size_t max_position_embeddings = wpe->shape[0]; // 1024
        size_t hidden_size     = wpe->shape[1];         // 768
        size_t vocab_size      = wte->shape[0];         // 50257

        dim3 gridDim(batch_size, sequence_length);
        dim3 blockDim(hidden_size);
        Embedding_Kernel<<<gridDim, blockDim>>>(input_ids->gpu_buf, wte->gpu_buf, wpe->gpu_buf, output->gpu_buf, max_position_embeddings, vocab_size);
    }
}