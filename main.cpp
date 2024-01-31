// extern "C" {
//     #include <libavcodec/avcodec.h>
//     #include <libavformat/avformat.h>
//     #include <libavutil/avutil.h>
//     #include <libavfilter/avfilter.h>
//     #include <libavdevice/avdevice.h>
//     #include <libswresample/swresample.h>
//     #include <libswscale/swscale.h>
// }

// #include <chrono>
// #include <iostream>
// #include <filesystem>
// #include <cstdlib>
// #include "transcription.hpp"
// #include "video_management.hpp"

// using namespace std;
// using namespace std::filesystem;

// int main (int argc, char ** argv) {

//     /**
//      * 16KInterview 13m - 60 seconds (3 processors)
//      *                  - 81 seconds (1 processor )
//     */


//     if (!exists("./models/ggml-base.en.bin")) {
//         cout << "Model not found in './models' folder." << endl;
//         return 1;
//     }

//     auto start = std::chrono::high_resolution_clock::now();
//     whisper_result wr = start_whisper( "./audio/jfk.wav", "./models/ggml-base.en.bin", 1, false, false);
//     auto end = std::chrono::high_resolution_clock::now();
//     auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);

//     std::cout << "\n\n\n\n\n";
//     std::cout << "Transcription completed in " << duration.count() << " seconds." << std::endl;

//     for (int i ; i < wr.transcriptions.size(); i ++ ) {
//         std::cout << wr.transcriptions[i].timestamp << "\n";
//         std::cout << wr.transcriptions[i].text << "\n";
//     }

// // Checking if ffmpeg libs are installed correctly...
//     printf("libavcodec version: %d\n", avcodec_version());
//     printf("libavformat version: %d\n", avformat_version());
//     printf("libavutil version: %d\n", avutil_version());
//     printf("libavfilter version: %d\n", avfilter_version());
//     printf("libavdevice version: %d\n", avdevice_version());
//     printf("libswresample version: %d\n", swresample_version());
//     printf("libswscale version: %d\n", swscale_version());

//     extract_wav_from_video("./video/TomH.mp4", "./audio/TomH.wav");




//     return 0;
// }

// ######################################################################################################################################################################################################################################################################################################################################################
#include <chrono>
#include "transcription.hpp"
#include "video_management.hpp"

int main(int argc, char* argv[]) { // TODO: Push this code to another file
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file>" << std::endl;
        return 1;
    }
    const char* input_file = argv[1];
    const char* output_file = "./audio/extracted.wav";

    // Get audio from video
    extract_audio_from_video(input_file, output_file); // can then pass extracted audio for transcription.
    const char* output_16khz = "./audio/extracted_16khz.wav";
    if (!convertTo16kHzWav(output_file, output_16khz)) {
        std::cerr << "Conversion to 16 kHz WAV failed." << std::endl;
        return 1;
    }

    std::cout << "Conversion to 16 kHz WAV completed successfully." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    whisper_result wr = start_whisper( output_16khz, "./models/ggml-base.en.bin", 1, false, false);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
    return 0;
}


// ./WhisperTest ./video/StillHere.mp4
// Below is highly outdated but good to get started
// https://friendlyuser.github.io/posts/tech/cpp/Using_FFmpeg_in_C++_A_Comprehensive_Guide/#integrating-ffmpeg-in-your-cpp-project

