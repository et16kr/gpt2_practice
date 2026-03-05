#include "data_loader.h"

int main() {
  const char* models_file = "/home/et16/aps/inference_practice/gpt2/model.safetensors";
  const char* inputs_file = "/home/et16/aps/inference_practice/examples/tokenized_cpp/data/gpt2_inputs_20.safetensors";
  DataLoader models(models_file);
  DataLoader inputs(inputs_file);

  return 0;
}

