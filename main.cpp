// https://learn.microsoft.com/en-us/windows/wsl/connect-usb

// #include <string.h>
// #include "modules/speech_processing.hpp"
// // #include "hazelcast-client.hpp"

// int main(int argc, char* argv[]) { // TODO: Push this code to another file
//     if (argc < 2) {
//         std::cerr << "Usage: " << argv[0] << " <input_file>" << std::endl;
//         return 1;
//     }
//     const std::string input_file = argv[1];

//     auto res = transcribe_video(input_file);
//     std::cout << "Transcription status: " << res.status << std::endl;

//     for (transcription_item i : res.transcriptions) {
//         std::cout << i.timestamp << " - ";
//         std::cout << ">" << i.text << std::endl;
//     }

// }

// // // //###############################################################################################################################
#include "common-sdl.h"
#include "common.h"
#include "whisper.h"

#include <cassert>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>
#include <fstream>

#include "modules/transcriber.hpp"
//  500 -> 00:05.000
// 6000 -> 01:00.000

// command-line parameters


int main(int argc, char ** argv) {
    Transcriber& t = Transcriber::get_instance();

    t.start_whisper_stream(argc, argv);

}
