cmake_minimum_required(VERSION 3.5)
project(MyNewProject)
set(TARGET_NAME "WhisperTest")

# sudo apt install ffmpeg libavcodec-dev libavformat-dev libavfilter-dev libswscale-dev libavdevice-dev
set(FFMPEG_LIBRARIES "-lavcodec -lavformat -lavutil -lavfilter -lavdevice -lswresample -lswscale -lpng -lz")

# Find the libraries
find_package(whisper REQUIRED)
find_package(SDL2 REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external/whisper_common)

file(GLOB_RECURSE WHISPER_COMMON_SOURCES 
     "${CMAKE_CURRENT_SOURCE_DIR}/external/whisper_common/*.c"
     "${CMAKE_CURRENT_SOURCE_DIR}/external/whisper_common/*.h"
     "${CMAKE_CURRENT_SOURCE_DIR}/external/whisper_common/*.cpp"
     )

message(STATUS "Whisper common sources: ${WHISPER_COMMON_SOURCES}")

add_executable(${TARGET_NAME} main.cpp
    transcription.hpp
    transcription.cpp
    video_handler.cpp 
    video_handler.hpp
    ${WHISPER_COMMON_SOURCES}
)  


target_include_directories(${TARGET_NAME} PRIVATE ${SDL2_INCLUDE_DIRS})
target_link_libraries(${TARGET_NAME} PRIVATE whisper::whisper ${SDL2_LIBRARIES} ${FFMPEG_LIBRARIES})
target_compile_features(${TARGET_NAME} PRIVATE cxx_std_17)