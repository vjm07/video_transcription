#include "utils.hpp"

#include <string>
#include <filesystem>
#include <sys/stat.h>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>

std::string ensure_dir(std::string location) {
    struct stat sb;
    if (stat(location.c_str(), &sb) == 0) {
        return location;
    }

    std::filesystem::create_directories(location);
    return location;
}

std::string get_temp_files_dir() {
    return ensure_dir("./.temp/");
}

std::string generate_uuid() {
    boost::uuids::uuid uuid = boost::uuids::random_generator()();
    std::string uuidStr = boost::uuids::to_string(uuid);
    return uuidStr;
}



