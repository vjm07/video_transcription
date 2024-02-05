#include <hazelcast/client/hazelcast_client.h>

int test_hazelcast() {
    auto hz = hazelcast::new_client().get(); // Connects to the cluster

    auto personnel = hz.get_map("personnel_map").get();
    personnel->put<std::string, std::string>("Alice", "IT").get();
    personnel->put<std::string, std::string>("Bob", "IT").get();
    personnel->put<std::string, std::string>("Clark", "IT").get();
    std::cout << "Added IT personnel. Logging all known personnel" << std::endl;
    for (const auto &entry : personnel->entry_set<std::string, std::string>().get()) {
        std::cout << entry.first << " is in " << entry.second << " department." << std::endl;
    }
    
    return 0;
}

