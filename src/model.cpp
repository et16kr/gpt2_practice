#include "model.h"

#include <string>
#include "kernels.h"
#include "tensor.h"
#include "tensor_loader.h"

Model::Model(TensorLoader& tl) {
    for(int i = 0 ; i < 12 ; i++) {
        std::string layer_name = "h." + std::to_string(i);
        weights.layers->attn_bias          = tl.GetTensor(layer_name + ".attn.bias");
        weights.layers->attn_c_attn_weight = tl.GetTensor(layer_name + ".attn.c_attn.weight");
        weights.layers->attn_c_attn_bias   = tl.GetTensor(layer_name + ".attn.c_attn.bias");
        weights.layers->attn_c_proj_weight = tl.GetTensor(layer_name + ".attn.c_proj.weight");
        weights.layers->attn_c_proj_bias   = tl.GetTensor(layer_name + ".attn.c_proj.bias");
        weights.layers->ln_1_weight        = tl.GetTensor(layer_name + ".ln_1.weight");
        weights.layers->ln_1_bias          = tl.GetTensor(layer_name + ".ln_1.bias");
        weights.layers->ln_2_weight        = tl.GetTensor(layer_name + ".ln_2.weight");
        weights.layers->ln_2_bias          = tl.GetTensor(layer_name + ".ln_2.bias");
        weights.layers->mlp_c_fc_weight    = tl.GetTensor(layer_name + ".mlp.c_fc.weight");
        weights.layers->mlp_c_fc_bias      = tl.GetTensor(layer_name + ".mlp.c_fc.bias");
        weights.layers->mlp_c_proj_weight  = tl.GetTensor(layer_name + ".mlp.c_proj.weight");
        weights.layers->mlp_c_proj_bias    = tl.GetTensor(layer_name + ".mlp.c_proj.bias");
    }
    weights.ln_f_bias   = tl.GetTensor("ln_f.bias");
    weights.ln_f_weight = tl.GetTensor("ln_f.weight");
    weights.wpe_weight  = tl.GetTensor("wpe.weight");
    weights.wte_weight  = tl.GetTensor("wte.weight");
    hidden_size = weights.wpe_weight->shape[1];
    values.x0 = new Tensor({batch_size, sequence_size, hidden_size});
}
Model::~Model() {
    delete values.x0;
}

void Model::Run(TensorLoader& tl) {
    auto input_ids =  tl.GetTensor("input_ids");
    if (input_ids->shape[0] >= batch_size) return;
    if (input_ids->shape[1] >= sequence_size) return;
    gpt2::Embedding(input_ids, weights.wte_weight, weights.wpe_weight, values.x0);
}