#ifndef UTILS_HPP
#define UTILS_HPP
#endif

#include <iostream>
#include <string>
#include <sys/stat.h>

std::string ensure_dir(std::string location);

std::string get_temp_files_dir();

std::string generate_uuid();