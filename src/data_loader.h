#pragma once

#include <unordered_map>
#include <string>
#include "tensor.h"

class DataLoader {
public:
    DataLoader() {};
    Tensor GetTensor(std::string key);
    static DataLoader LoadData(const char* fname);
private:
    std::unordered_map<std::string, Tensor> map;
};