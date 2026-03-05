#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <nlohmann/json.hpp>
#include "data_loader.h"

using json = nlohmann::json;

DataLoader::DataLoader(const char* fname) {
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
            ret = pread(fd, buf, tensor_size, offs[0]);
            if (ret < 0 || static_cast<size_t>(ret) != tensor_size) {
                std::cout << "read: " << ret << " / " << tensor_size << "\n";
            }
            map.try_emplace(name, shape, buf);
            free(buf);
        }
    }
    free(json_buf);
    close(fd);
}
