#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <array>
#include <vector>
#include <cstring>

std::vector<std::string> split(const std::string& str, const std::string& delim);
std::string exec(const char* cmd);
std::string process_result(const std::string& result,const std::string& id_vendor, const std::string& id_product);
