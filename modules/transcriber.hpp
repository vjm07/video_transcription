#ifndef TRANSCRIBER_HPP
#define TRANSCRIBER_HPP

#include "common.h"
#include "whisper.h"
#include <iostream>
#include <cmath>
#include <fstream>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>
#include <sstream>
#include <cstring>
#include <chrono>
#include <typeindex>
#include <algorithm> // For std::copy

struct transcription_item {
    std::string timestamp;
    std::string text;
};
struct whisper_result {
    std::string status;
    std::vector<transcription_item> transcriptions;
};
struct whisper_params {
    int32_t n_threads    = std::min(4, (int32_t) std::thread::hardware_concurrency());
    int32_t n_processors =  1;
    int32_t offset_t_ms  =  0;
    int32_t offset_n     =  0;
    int32_t duration_ms  =  0;
    int32_t progress_step =  5;
    int32_t max_context  = -1;
    int32_t max_len      =  0;
    int32_t best_of      = whisper_full_default_params(WHISPER_SAMPLING_GREEDY).greedy.best_of;
    int32_t beam_size    = whisper_full_default_params(WHISPER_SAMPLING_BEAM_SEARCH).beam_search.beam_size;

    float word_thold    =  0.01f;
    float entropy_thold =  2.40f;
    float logprob_thold = -1.00f;

    bool speed_up        = false;
    bool debug_mode      = false;
    bool translate       = false;
    bool detect_language = false;
    bool diarize         = false;
    bool tinydiarize     = false;
    bool split_on_word   = false;
    bool no_fallback     = false;
    bool output_txt      = false;
    bool output_vtt      = false;
    bool output_srt      = false;
    bool output_wts      = false;
    bool output_csv      = false;
    bool output_jsn      = false;
    bool output_jsn_full = false;
    bool output_lrc      = false;
    bool no_prints       = false;
    bool print_special   = false;
    bool print_colors    = false;
    bool print_progress  = false;
    bool no_timestamps   = false;
    bool log_score       = false;
    bool use_gpu         = true;

    std::string language  = "en";
    std::string prompt;
    std::string font_path = "./assets/fonts/Courier New Bold.ttf";
    std::string model     = "./models/ggml-base.en.bin";

    // [TDRZ] speaker turn string
    std::string tdrz_speaker_turn = " [SPEAKER_TURN]";
    std::string openvino_encode_device = "CPU";

    std::string fname_inp = "";
    std::vector<std::string> fname_out = {};

    whisper_params(){};
    whisper_params(const whisper_params& other)
        : n_threads(other.n_threads),
          n_processors(other.n_processors),
          offset_t_ms(other.offset_t_ms),
          offset_n(other.offset_n),
          duration_ms(other.duration_ms),
          progress_step(other.progress_step),
          max_context(other.max_context),
          max_len(other.max_len),
          best_of(other.best_of),
          beam_size(other.beam_size),
          word_thold(other.word_thold),
          entropy_thold(other.entropy_thold),
          logprob_thold(other.logprob_thold),
          speed_up(other.speed_up),
          debug_mode(other.debug_mode),
          translate(other.translate),
          detect_language(other.detect_language),
          diarize(other.diarize),
          tinydiarize(other.tinydiarize),
          split_on_word(other.split_on_word),
          no_fallback(other.no_fallback),
          output_txt(other.output_txt),
          output_vtt(other.output_vtt),
          output_srt(other.output_srt),
          output_wts(other.output_wts),
          output_csv(other.output_csv),
          output_jsn(other.output_jsn),
          output_jsn_full(other.output_jsn_full),
          output_lrc(other.output_lrc),
          no_prints(other.no_prints),
          print_special(other.print_special),
          print_colors(other.print_colors),
          print_progress(other.print_progress),
          no_timestamps(other.no_timestamps),
          log_score(other.log_score),
          use_gpu(other.use_gpu),
          language(other.language),
          prompt(other.prompt),
          font_path(other.font_path),
          model(other.model),
          tdrz_speaker_turn(other.tdrz_speaker_turn),
          openvino_encode_device(other.openvino_encode_device),
          fname_inp(other.fname_inp),
          fname_out(other.fname_out) {}

    whisper_params& operator=(const whisper_params& other) {
        if (this != &other) { // Protect against self-assignment
            n_threads = other.n_threads;
            // Repeat for all other members...
            language = other.language;
            prompt = other.prompt;
            font_path = other.font_path;
            model = other.model;
            tdrz_speaker_turn = other.tdrz_speaker_turn;
            openvino_encode_device = other.openvino_encode_device;
            fname_inp = other.fname_inp;
            fname_out = other.fname_out;
        }
        return *this;
    }

};
struct whisper_stream_params {
    int32_t n_threads  = std::min(4, (int32_t) std::thread::hardware_concurrency());
    int32_t step_ms    = 3000;
    int32_t length_ms  = 10000;
    int32_t keep_ms    = 200;
    int32_t capture_id = -1;
    int32_t max_tokens = 32;
    int32_t audio_ctx  = 0;

    float vad_thold    = 0.6f;
    float freq_thold   = 100.0f;

    bool speed_up      = false;
    bool translate     = false;
    bool no_fallback   = false;
    bool print_special = false;
    bool no_context    = true;
    bool no_timestamps = false;
    bool tinydiarize   = false;
    bool save_audio    = false; // save audio to wav file
    bool use_gpu       = true;

    std::string language  = "en";
    std::string model     = "models/ggml-base.en.bin";
    std::string fname_out;
};
struct whisper_print_user_data {
    const whisper_params * params;
    const std::vector<std::vector<float>> * pcmf32s;
    int progress_prev;
};

class Transcriber {
    private:
        whisper_params whisper_p;

        static void whisper_print_segment_callback(struct whisper_context * ctx, struct whisper_state * , int n_new, void * user_data);
        static void cb_log_disable(enum ggml_log_level , const char * , void * ){};
        static std::string to_timestamp(int64_t t, bool comma = false);

        // Streaming functionality - 90% sure will not be needed
        // static bool whisper_params_parse(int argc, char ** argv, whisper_stream_params & params);
        // static void whisper_print_usage(int /*argc*/, char ** argv, const whisper_stream_params & params);
        Transcriber();
        
    public:
        static Transcriber& get_instance();

        bool set_model_location(std::string); 
        std::string get_model_location();
        
        void set_n_processors(int32_t np) {Transcriber::whisper_p.n_processors = np;};
        void set_log_progress(bool log) {Transcriber::whisper_p.no_prints = !log;};
        

        Transcriber(const Transcriber&) = delete;
        void operator=(const Transcriber&) = delete;

        whisper_result start_whisper(std::string file_loc, void (*callback)(struct whisper_context * ctx, struct whisper_state *, int n_new, void * user_data) = whisper_print_segment_callback);
        // whisper_result start_whisper_stream(int argc, char ** argv);
};

#endif
