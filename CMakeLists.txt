cmake_minimum_required(VERSION 3.5)
# HOME PC: /mnt/c/Users/vjmar/Documents/1. Code/vcpkg
# WORK PC: /mnt/c/Users/vjt/documents/Code/vcpkg
set(CMAKE_TOOLCHAIN_FILE "/mnt/c/Users/vjt/documents/Code/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "Vcpkg toolchain file")
# set(CMAKE_TOOLCHAIN_FILE /home/vjt/Documents/Projects/VideoAnalysis/vcpkg/scripts/buildsystems/vcpkg.cmake CACHE STRING "Vcpkg toolchain file")
project(MyNewProject)
set(TARGET_NAME "WhisperTest")

# sudo apt install ffmpeg libavcodec-dev libavformat-dev libavfilter-dev libswscale-dev libavdevice-dev
set(FFMPEG_LIBRARIES "-lavcodec -lavformat -lavutil -lavfilter -lavdevice -lswresample -lswscale -lpng -lz")

# Find the libraries
find_package(whisper REQUIRED)
find_package(SDL2 REQUIRED)
# find_package(hazelcast-cpp-client CONFIG REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/whisper_common)
file(GLOB_RECURSE WHISPER_COMMON_SOURCES 
     "${CMAKE_CURRENT_SOURCE_DIR}/external/whisper_common/*.c"
     "${CMAKE_CURRENT_SOURCE_DIR}/external/whisper_common/*.h"
     "${CMAKE_CURRENT_SOURCE_DIR}/external/whisper_common/*.cpp"
     )

message(STATUS "Whisper common sources: ${WHISPER_COMMON_SOURCES}")

add_executable(${TARGET_NAME} main.cpp
    modules/transcriber.hpp
    modules/transcriber.cpp
    modules/video_handler.cpp 
    modules/video_handler.hpp
    modules/speech_processing.hpp
    modules/speech_processing.cpp
    modules/utils.hpp
    modules/utils.cpp
    ${WHISPER_COMMON_SOURCES}
)  


target_include_directories(${TARGET_NAME} PRIVATE ${SDL2_INCLUDE_DIRS})
target_link_libraries(${TARGET_NAME} PRIVATE whisper::whisper ${SDL2_LIBRARIES} ${FFMPEG_LIBRARIES}) #hazelcast-cpp-client::hazelcast-cpp-client)
target_compile_features(${TARGET_NAME} PRIVATE cxx_std_17)