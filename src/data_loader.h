#pragma once

#include <unordered_map>
#include <string>
#include "tensor.h"

class DataLoader {
public:
    DataLoader(const char* fname);
    Tensor GetTensor(std::string key);
private:
    std::unordered_map<std::string, Tensor> map;
};