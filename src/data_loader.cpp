//#include <cuda_runtime.h>

#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <sys/stat.h>
#include <iostream>
#include <nlohmann/json.hpp>
#include "data_loader.h"

using json = nlohmann::json;

DataLoader DataLoader::LoadData(const char* fname) {
    int fd = open(fname, O_RDONLY);

    struct stat st;
    fstat(fd, &st);
    printf("file_size:%ld\n", st.st_size);
    
    int64_t json_size;
    size_t ret = pread(fd, &json_size, 8, 0);
    printf("read result:%ld\n", ret);

    printf("json_size:%ld\n", json_size);
    
    char *json_buf = (char*)malloc(json_size +1);
    json_buf[json_size] = 0;
    ret = pread(fd, json_buf, json_size, 8);
    printf("read result:%ld\n", ret);

    json j = json::parse(json_buf);
    DataLoader dl;

    for (auto it = j.begin(); it != j.end(); ++it) {
        const std::string& name = it.key();
        std::cout << name << std::endl;
        std::string dtype;
        std::vector<size_t> shape;
        std::vector<long long> offs;
        size_t tensor_size = 0;
        float * buf = nullptr;
        for (auto& [k, v] : it->items()) {
            if (k == "dtype") {
                dtype = v.get<std::string>();
                std::cout << "field: " << k << " -> " << dtype << "\n";
            } else if (k == "shape") {
                shape = v.get<std::vector<size_t>>();
                std::cout << "field: " << k << " -> " << v << "\n";
            } else if (k == "data_offsets") {
                offs = v.get<std::vector<long long>>();
                tensor_size = offs[1] - offs[0];
                std::cout << "field: " << k << " -> " << v << ", size:" << tensor_size << "\n";
                buf = (float*)malloc(tensor_size);
                ret = pread(fd, buf, tensor_size, offs[0]);
                if (ret != tensor_size) 
                    std::cout << "read: " << ret << " / " << tensor_size << "\n";
            }
        }
        std::cout << std::endl;
        if (buf != nullptr) {
            dl.map[name] = Tensor(shape, buf);
            free(buf);
        }
    }
    free(json_buf);
    close(fd);
  
  return dl;
}
