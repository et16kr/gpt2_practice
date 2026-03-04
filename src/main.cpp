#include "data_loader.h"

int main() {
  const char* fname = "/home/et16/aps/inference_practice/gpt2/model.safetensors";
  DataLoader data_loader = DataLoader::LoadData(fname);

  return 0;
}

