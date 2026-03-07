#include "tensor_loader.h"
#include "model.h"

int main() {
  const char* models_file = "/home/et16/aps/inference_practice/gpt2/model.safetensors";
  const char* inputs_file = "/home/et16/aps/inference_practice/examples/tokenized_cpp/data/gpt2_inputs_20.safetensors";
  TensorLoader model_tensor(models_file);
  TensorLoader input_tensor(inputs_file);
  Model model(model_tensor);
  model.Run(input_tensor);
  //model_tensor.Elements();
  //input_tensor.Elements();
  return 0;
}

