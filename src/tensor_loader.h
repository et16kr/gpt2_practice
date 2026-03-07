#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include "tensor.h"

class TensorLoader {
public:
    TensorLoader(const char* fname);
    Tensor* GetTensor(const std::string& key) const {
        auto it = map.find(key);
        if (it == map.end()) return nullptr;
        return it->second.get();
    }
    void Elements();
private:
    std::unordered_map<std::string, std::unique_ptr<Tensor>> map;
};