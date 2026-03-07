#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <nlohmann/json.hpp>
#include "tensor_loader.h"

using json = nlohmann::json;

TensorLoader::TensorLoader(const char* fname) {
    int fd = open(fname, O_RDONLY);

    struct stat st;
    fstat(fd, &st);
    
    uint64_t json_size = 0;
    ssize_t ret = pread(fd, &json_size, 8, 0);
    
    char *json_buf = (char*)malloc(json_size +1);
    json_buf[json_size] = 0;
    ret = pread(fd, json_buf, json_size, 8);

    json j = json::parse(json_buf);

    for (auto it = j.begin(); it != j.end(); ++it) {
        const std::string& name = it.key();
        const json& value = it.value();
        if (value.is_object() && value.contains("dtype") && value.contains("shape") &&value.contains("data_offsets")) {
            std::string dtype = value.at("dtype").get<std::string>();
            std::vector<size_t> shape = value.at("shape").get<std::vector<size_t>>();
            std::vector<long long> offs = value.at("data_offsets").get<std::vector<long long>>();
            if (offs.size() < 2) {
                std::cout << "skip invalid data_offsets\n";
                continue;
            }
            size_t tensor_size = offs[1] - offs[0];
            float * buf = (float*)malloc(tensor_size);
            ret = pread(fd, buf, tensor_size, offs[0]+8+json_size);
            if (ret < 0 || static_cast<size_t>(ret) != tensor_size) {
                std::cout << "read: " << ret << " / " << tensor_size << "\n";
            }/*
            if (name == "input_ids") {
                for (int i = 0 ; i < shape[0] ; i++) {
                    std::cout << "[" << i << "] " << buf[i*shape[1]];
                    for (int j = 0 ; j < shape[1] ; j++) {
                        std::cout << ", " << buf[i*shape[1]+j];
                    }
                    std::cout << std::endl;
                }
            }*/
            map[name] = std::make_unique<Tensor>(shape, buf);
            free(buf);
        }
    }
    free(json_buf);
    close(fd);
}

void TensorLoader::Elements() {
    for (auto & [key, value] : map) {
        auto shape = value->shape;
        std::cout << key <<" [ " << shape[0] ;
        
        for (int i = 1; i < 5 ; i++) {
            if (shape[i] <= 1) break;
            std::cout << ", " << shape[i] ;
        }
        std::cout << " ]" << std::endl;
    }
}