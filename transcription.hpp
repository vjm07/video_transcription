#include "common.h"
#include "whisper.h"
#include <iostream>
#include <cmath>
#include <fstream>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>
#include <cstring>
#include <sstream>
#include <cstring>
#include <chrono>
#include <typeindex>

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
    std::string model     = "models/ggml-base.en.bin";

    // [TDRZ] speaker turn string
    std::string tdrz_speaker_turn = " [SPEAKER_TURN]";

    std::string openvino_encode_device = "CPU";

    std::vector<std::string> fname_inp = {};
    std::vector<std::string> fname_out = {};
};
struct whisper_print_user_data {
    const whisper_params * params;
    const std::vector<std::vector<float>> * pcmf32s;
    int progress_prev;
};

std::string to_timestamp(int64_t t, bool comma = false) ;
int timestamp_to_sample(int64_t t, int n_samples) ;
void replace_all(std::string & s, const std::string & search, const std::string & replace);
void whisper_print_usage(int argc, char ** argv, const whisper_params & params);
bool whisper_params_parse(int argc, char ** argv, whisper_params & params);
void whisper_print_usage(int /*argc*/, char ** argv, const whisper_params & params);
std::string estimate_diarization_speaker(std::vector<std::vector<float>> pcmf32s, int64_t t0, int64_t t1, bool id_only = false);
void whisper_print_progress_callback(struct whisper_context * /*ctx*/, struct whisper_state * /*state*/, int progress, void * user_data) ;
void whisper_print_segment_callback(struct whisper_context * ctx, struct whisper_state * /*state*/, int n_new, void * user_data);
bool output_txt(struct whisper_context * ctx, const char * fname, const whisper_params & params, std::vector<std::vector<float>> pcmf32s);
bool output_vtt(struct whisper_context * ctx, const char * fname, const whisper_params & params, std::vector<std::vector<float>> pcmf32s);
bool output_srt(struct whisper_context * ctx, const char * fname, const whisper_params & params, std::vector<std::vector<float>> pcmf32s);
char *escape_double_quotes_and_backslashes(const char *str) ;
bool output_csv(struct whisper_context * ctx, const char * fname, const whisper_params & params, std::vector<std::vector<float>> pcmf32s);
bool output_score(struct whisper_context * ctx, const char * fname, const whisper_params & /*params*/, std::vector<std::vector<float>> /*pcmf32s*/) ;
bool output_json(struct whisper_context * ctx, const char * fname, const whisper_params & params, std::vector<std::vector<float>>   pcmf32s, bool   full);
bool output_wts(struct whisper_context * ctx, const char * fname, const char * fname_inp, const whisper_params & params, float t_sec, std::vector<std::vector<float>> pcmf32s);
bool output_lrc(struct whisper_context * ctx, const char * fname, const whisper_params & params, std::vector<std::vector<float>> pcmf32s);
void cb_log_disable(enum ggml_log_level , const char * , void * );
whisper_result start_whisper(std::string file_loc, std::string model_loc, int32_t n_processors, bool diarize, bool no_prints,
void (*callback)(struct whisper_context * ctx, struct whisper_state * /*state*/, int n_new, void * user_data) = whisper_print_segment_callback);
