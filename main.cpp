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

#include "modules/actions.hpp"
//  500 -> 00:05.000
// 6000 -> 01:00.000

// command-line parameters


int main(int argc, char ** argv) {
    
    if (argc < 2) {
        std::cerr << "need to specify and mp4 or wav file." << std::endl;
    }
    std::string file = argv[1];
    whisper_result ws = transcribe_video(file);
    std::cout << ws.status << std::endl;

    // write to file to view results
    std::ofstream out;
    out.open("./transcription.txt");
    if (!out) {
        std::cerr << "could not open file" << std::endl;
        return 1;
    }

    for (transcription_item i : ws.transcriptions) {
        out << i.timestamp << " " << i.text << std::endl;
    }
    out.close();
    
}
