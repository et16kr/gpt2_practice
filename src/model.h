#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include "tensor_loader.h"

struct Layer {
    Tensor* attn_bias = nullptr;
    Tensor* attn_c_attn_bias = nullptr;
    Tensor* attn_c_attn_weight = nullptr;
    Tensor* attn_c_proj_bias = nullptr;
    Tensor* attn_c_proj_weight = nullptr;
    Tensor* ln_1_bias = nullptr;
    Tensor* ln_1_weight = nullptr;
    Tensor* ln_2_bias = nullptr;
    Tensor* ln_2_weight = nullptr;
    Tensor* mlp_c_fc_bias = nullptr;
    Tensor* mlp_c_fc_weight = nullptr;
    Tensor* mlp_c_proj_bias = nullptr;
    Tensor* mlp_c_proj_weight = nullptr;
};

struct Weights {
    Layer layers[12];
    Tensor* ln_f_bias = nullptr;
    Tensor* ln_f_weight = nullptr;
    Tensor* wpe_weight = nullptr;
    Tensor* wte_weight = nullptr;
};

struct Values {
    Tensor* x0 = nullptr;
};

class Model {
public:
    Model(TensorLoader& tl);
    ~Model();
    void Run(TensorLoader& tl);
private:
    Weights weights;
    Values values;
    size_t hidden_size = 0;
    size_t batch_size = 20;
    size_t sequence_size = 64;
};

