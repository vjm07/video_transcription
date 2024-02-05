#include "speech_processing.hpp"
#include "utils.hpp"

#include <chrono>
#include <ctime>

bool endsWith(const char *str, const char *suffix) {
    if (!str || !suffix) {
        return false;
    }
    size_t str_len = strlen(str);
    size_t suffix_len = strlen(suffix);

    if (suffix_len > str_len) {
        return false;
    }

    return strncmp(str + str_len - suffix_len, suffix, suffix_len) == 0;
}

whisper_result transcribe_video(std::string input_file_location) { 
    whisper_result wr;
    std::string tempfiles = get_temp_files_dir();

    const char* input_file = input_file_location.c_str();
    std::string file_path = tempfiles + generate_uuid() + ".wav";
    const char* output_file = file_path.c_str();

    if (!endsWith(input_file, ".wav") && !endsWith(input_file, ".mp4")) {
        std::cerr << "Not a suitable file type." << std::endl;
        wr.status = "not a suitable file type";
        return wr; 
    }

    std::string timestamp16k = tempfiles + generate_uuid() + "16k.wav";
    const char* output_16khz = timestamp16k.c_str();

    if (endsWith(input_file, ".mp4")) {
        int success = extract_audio_from_video(input_file, output_file); 
        if (success < 0 ) {
            wr.status = "failed to extract audio from video.";
            return wr;
        }
        bool converted = convertTo16kHzWav(output_file, output_16khz);
        std::remove(output_file);
        if (!converted) {
            wr.status = "Conversion to 16 kHz WAV failed.";
            return wr;
        }        
        
    } else {
        if (!convertTo16kHzWav(input_file, output_16khz)) {
            wr.status = "Conversion to 16 kHz WAV failed." ;
            return wr;
        }
    }

    // TODO: put some parameters into some settings.
    auto model_inference_result = start_whisper( output_16khz, "./models/ggml-base.en.bin", 1, false, true);
    std::remove(output_16khz);
    std::string completed_msg = "completed";
    if (model_inference_result.status == completed_msg) {
        wr.status = completed_msg;
        wr.transcriptions = model_inference_result.transcriptions;
    }

    return wr;
}