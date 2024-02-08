#include "common.h"
#include "transcriber.hpp"
#include "whisper.h"
#include "utils.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <thread>
#include <vector>
#include <sstream>
#include <cstring>
#include <chrono>
#include <filesystem>
#include "common-sdl.h"

namespace fs = std::filesystem;

// Terminal color map. 10 colors grouped in ranges [0.0, 0.1, ..., 0.9]
// Lowest is red, middle is yellow, highest is green.
const std::vector<std::string> k_colors = {
    "\033[38;5;196m", "\033[38;5;202m", "\033[38;5;208m", "\033[38;5;214m", "\033[38;5;220m",
    "\033[38;5;226m", "\033[38;5;190m", "\033[38;5;154m", "\033[38;5;118m", "\033[38;5;82m",
};

Transcriber::Transcriber() {
    std::cout << "INITIALISING MODEL PARAMETERS\n";
    whisper_params p;
    Transcriber::whisper_p = p;
    Transcriber::whisper_p.model = "./models/ggml-base.en.bin";
    Transcriber::whisper_p.no_prints = true;
    Transcriber::whisper_p.n_processors = 2;
    Transcriber::whisper_p.diarize = false;
};

Transcriber& Transcriber::get_instance() {
    static Transcriber instance; 
    return instance;
}

bool Transcriber::set_model_location(std::string model_loc) {
    if (!fs::exists(model_loc)) return false;
    fs::path f_path = model_loc;
    std::string extension = f_path.extension().string();
    if (extension != ".bin") return false;
    Transcriber::whisper_p.model = std::string(model_loc);
    return true;
};

std::string Transcriber::get_model_location() {
    return Transcriber::whisper_p.model;
};

std::string Transcriber::to_timestamp(int64_t t, bool comma) {
    int64_t msec = t * 10;
    int64_t hr = msec / (1000 * 60 * 60);
    msec = msec - hr * (1000 * 60 * 60);
    int64_t min = msec / (1000 * 60);
    msec = msec - min * (1000 * 60);
    int64_t sec = msec / 1000;
    msec = msec - sec * 1000;

    char buf[32];
    snprintf(buf, sizeof(buf), "%02d:%02d:%02d%s%03d", (int) hr, (int) min, (int) sec, comma ? "," : ".", (int) msec);

    return std::string(buf);
}

void Transcriber::whisper_print_segment_callback(struct whisper_context * ctx, struct whisper_state * /*state*/, int n_new, void * user_data) {
    const auto & params  = *((whisper_print_user_data *) user_data)->params;
    const auto & pcmf32s = *((whisper_print_user_data *) user_data)->pcmf32s;

    const int n_segments = whisper_full_n_segments(ctx);

    std::string speaker = "";

    int64_t t0 = 0;
    int64_t t1 = 0;

    // print the last n_new segments
    const int s0 = n_segments - n_new;

    if (s0 == 0) {
        printf("\n");
    }

    for (int i = s0; i < n_segments; i++) {
        // TODO: Write here if you want to save responses as they are created!
        // Code...
        if (!params.no_timestamps || params.diarize) {
            
            t0 = whisper_full_get_segment_t0(ctx, i);
            t1 = whisper_full_get_segment_t1(ctx, i);
        }

        if (!params.no_timestamps) {
            printf("[%s --> %s]  ", to_timestamp(t0).c_str(), to_timestamp(t1).c_str());
        }

        if (params.print_colors) {
            for (int j = 0; j < whisper_full_n_tokens(ctx, i); ++j) {
                if (params.print_special == false) {
                    const whisper_token id = whisper_full_get_token_id(ctx, i, j);
                    if (id >= whisper_token_eot(ctx)) {
                        continue;
                    }
                }

                const char * text = whisper_full_get_token_text(ctx, i, j);
                const float  p    = whisper_full_get_token_p   (ctx, i, j);

                const int col = std::max(0, std::min((int) k_colors.size() - 1, (int) (std::pow(p, 3)*float(k_colors.size()))));

                printf("%s%s%s%s", speaker.c_str(), k_colors[col].c_str(), text, "\033[0m");
            }
        } else {
            const char * text = whisper_full_get_segment_text(ctx, i);

            printf("%s%s", speaker.c_str(), text);
        }

        if (params.tinydiarize) {
            if (whisper_full_get_segment_speaker_turn_next(ctx, i)) {
                printf("%s", params.tdrz_speaker_turn.c_str());
            }
        }

        // with timestamps or speakers: each segment on new line
        if (!params.no_timestamps || params.diarize) {
            printf("\n");
        }

        fflush(stdout);
    }
}

whisper_result Transcriber::start_whisper(std::string file_loc, void (*callback)(struct whisper_context * ctx, struct whisper_state * /*state*/, int n_new, void * user_data)) {
    
    whisper_result wr;
    whisper_params params = Transcriber::whisper_p;

    if (file_loc.empty()) {
        wr.status = "no input files specified";
        return wr;
    }

    if (params.language != "auto" && whisper_lang_id(params.language.c_str()) == -1) {
        wr.status = "langauge id does not exist";
        return wr;
    }

    if (params.no_prints) {
        whisper_log_set(cb_log_disable, NULL); // pass empty function to overwrite model log function
        params.print_progress = false;
    }

    // initialise model
    struct whisper_context_params cparams;
    cparams.use_gpu = params.use_gpu;
    struct whisper_context * ctx = whisper_init_from_file_with_params(params.model.c_str(), cparams);

    if (ctx == nullptr) {
        wr.status = "failed to initialize whisper context";
        return wr;
    }

    // initialise openvino encoder - only affects OpenVINO enabled builds 
    whisper_ctx_init_openvino_encoder(ctx, nullptr, params.openvino_encode_device.c_str(), nullptr);
    std::vector<float> pcmf32;               
    std::vector<std::vector<float>> pcmf32s; 

    if (!::read_wav(file_loc, pcmf32, pcmf32s, params.diarize)) {
        std::stringstream ss;
        ss << "failed to read WAV file '" << file_loc.c_str() << "' ";
        wr.status = ss.str();
        return wr;
    }

    if (!whisper_is_multilingual(ctx)) {
        if (params.language != "en" || params.translate) {
            params.language = "en";
            params.translate = false;
            fprintf(stderr, "%s: WARNING: model is not multilingual, ignoring language and translation options\n", __func__);
            wr.status = "model is not multilingual, ignoring language and translation options";
        }
    }
    
    if (params.detect_language) {
        params.language = "auto";
    }

    if (!params.no_prints) {
        // print system information
        fprintf(stderr, "\n");
        fprintf(stderr, "system_info: n_threads = %d / %d | %s\n",
                params.n_threads*params.n_processors, std::thread::hardware_concurrency(), whisper_print_system_info());

        // print some info about the processing
        fprintf(stderr, "\n");
        fprintf(stderr, "%s: processing '%s' (%d samples, %.1f sec), %d threads, %d processors, %d beams + best of %d, lang = %s, task = %s, %stimestamps = %d ...\n",
                __func__, file_loc.c_str(), int(pcmf32.size()), float(pcmf32.size())/WHISPER_SAMPLE_RATE,
                params.n_threads, params.n_processors, params.beam_size, params.best_of,
                params.language.c_str(),
                params.translate ? "translate" : "transcribe",
                params.no_timestamps ? 0 : 1);

        fprintf(stderr, "\n");
    }

    // run the inference
    {
        whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);

        wparams.strategy = params.beam_size > 1 ? WHISPER_SAMPLING_BEAM_SEARCH : WHISPER_SAMPLING_GREEDY;

        wparams.print_realtime   = false;
        wparams.print_progress   = params.print_progress;
        wparams.print_timestamps = !params.no_timestamps;
        wparams.print_special    = params.print_special;
        wparams.translate        = params.translate;
        wparams.language         = params.language.c_str();
        wparams.detect_language  = params.detect_language;
        wparams.n_threads        = params.n_threads;
        wparams.n_max_text_ctx   = params.max_context >= 0 ? params.max_context : wparams.n_max_text_ctx;
        wparams.offset_ms        = params.offset_t_ms;
        wparams.duration_ms      = params.duration_ms;

        wparams.token_timestamps = params.output_wts || params.output_jsn_full || params.max_len > 0;
        wparams.thold_pt         = params.word_thold;
        wparams.max_len          = params.output_wts && params.max_len == 0 ? 60 : params.max_len;
        wparams.split_on_word    = params.split_on_word;

        wparams.speed_up         = params.speed_up;
        wparams.debug_mode       = params.debug_mode;

        wparams.tdrz_enable      = params.tinydiarize; // [TDRZ]

        wparams.initial_prompt   = params.prompt.c_str();

        wparams.greedy.best_of        = params.best_of;
        wparams.beam_search.beam_size = params.beam_size;

        wparams.temperature_inc  = params.no_fallback ? 0.0f : wparams.temperature_inc;
        wparams.entropy_thold    = params.entropy_thold;
        wparams.logprob_thold    = params.logprob_thold;

        whisper_print_user_data user_data = { &params, &pcmf32s, 0 };

        // this callback is called on each new segment - probably not best to use segment, progress is more friendly...
        // ill keep for debugging purposes for now.
        if (!wparams.print_realtime) {
            wparams.new_segment_callback           = whisper_print_segment_callback;
            wparams.new_segment_callback_user_data = &user_data;
        }

        if (wparams.print_progress) {
            wparams.progress_callback           = callback;
            wparams.progress_callback_user_data = &user_data;
        }

        if (whisper_full_parallel(ctx, wparams, pcmf32.data(), pcmf32.size(), params.n_processors) != 0) {
            // fprintf(stderr, "%s: failed to process audio\n", argv[0]);
            wr.status = "failed to process audio";
            whisper_free(ctx);
            return wr;
        }
    }

    // output stuff
    {
        // Populating struct
        const int n_segments = whisper_full_n_segments(ctx);
        for (int i = 0; i < n_segments; ++i) {
            const char * text = whisper_full_get_segment_text(ctx, i);
            std::string speaker = "";

            // Timestamps
            int64_t t0 = whisper_full_get_segment_t0(ctx, i);
            int64_t t1 = whisper_full_get_segment_t1(ctx, i);
            std::string start = to_timestamp(t0).c_str();
            std::string end   = to_timestamp(t1).c_str();

            // Add Text to wr.
            transcription_item a;
            a.text = text;

            // ss << "[" << start << " --> " << end << "]";
            std::stringstream ss;
            ss << "[" << start << " --> " << end << "]";
            a.timestamp = ss.str();
            wr.transcriptions.push_back(a);
        }
    }

    whisper_print_timings(ctx);
    whisper_free(ctx);
    wr.status = "completed";
    return wr;
}


/**
 *   Stream Overloads + Streaming Functionality
    Unlikely to make it into build since this functionality may not be needed as only full video files will apparently be used.
    Keeping just incase people change their minds.
 * 
*/
/*

    void Transcriber::whisper_print_usage(int argc, char ** argv, const whisper_stream_params & params) {};

bool Transcriber::whisper_params_parse(int argc, char ** argv, whisper_stream_params & params) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            whisper_print_usage(argc, argv, params);
            exit(0);
        }
        else if (arg == "-t"    || arg == "--threads")       { params.n_threads     = std::stoi(argv[++i]); }
        else if (                  arg == "--step")          { params.step_ms       = std::stoi(argv[++i]); }
        else if (                  arg == "--length")        { params.length_ms     = std::stoi(argv[++i]); }
        else if (                  arg == "--keep")          { params.keep_ms       = std::stoi(argv[++i]); }
        else if (arg == "-c"    || arg == "--capture")       { params.capture_id    = std::stoi(argv[++i]); }
        else if (arg == "-mt"   || arg == "--max-tokens")    { params.max_tokens    = std::stoi(argv[++i]); }
        else if (arg == "-ac"   || arg == "--audio-ctx")     { params.audio_ctx     = std::stoi(argv[++i]); }
        else if (arg == "-vth"  || arg == "--vad-thold")     { params.vad_thold     = std::stof(argv[++i]); }
        else if (arg == "-fth"  || arg == "--freq-thold")    { params.freq_thold    = std::stof(argv[++i]); }
        else if (arg == "-su"   || arg == "--speed-up")      { params.speed_up      = true; }
        else if (arg == "-tr"   || arg == "--translate")     { params.translate     = true; }
        else if (arg == "-nf"   || arg == "--no-fallback")   { params.no_fallback   = true; }
        else if (arg == "-ps"   || arg == "--print-special") { params.print_special = true; }
        else if (arg == "-kc"   || arg == "--keep-context")  { params.no_context    = false; }
        else if (arg == "-l"    || arg == "--language")      { params.language      = argv[++i]; }
        else if (arg == "-m"    || arg == "--model")         { params.model         = argv[++i]; }
        else if (arg == "-f"    || arg == "--file")          { params.fname_out     = argv[++i]; }
        else if (arg == "-tdrz" || arg == "--tinydiarize")   { params.tinydiarize   = true; }
        else if (arg == "-sa"   || arg == "--save-audio")    { params.save_audio    = true; }
        else if (arg == "-ng"   || arg == "--no-gpu")        { params.use_gpu       = false; }

        else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            whisper_print_usage(argc, argv, params);
            exit(0);
        }
    }

    return true;
}


whisper_result Transcriber::start_whisper_stream(int argc, char ** argv) {
    whisper_result wr;
    whisper_stream_params params;
    params.model = Transcriber::model_location;
    // params.length_ms = 100000; -- experiment with this.

    if (whisper_params_parse(argc, argv, params) == false) {
        wr.status = "couldnt not parse parameters";
        return wr;
    }

    params.keep_ms   = std::min(params.keep_ms,   params.step_ms);
    params.length_ms = std::max(params.length_ms, params.step_ms);

    const int n_samples_step = (1e-3*params.step_ms  )*WHISPER_SAMPLE_RATE;
    const int n_samples_len  = (1e-3*params.length_ms)*WHISPER_SAMPLE_RATE;
    const int n_samples_keep = (1e-3*params.keep_ms  )*WHISPER_SAMPLE_RATE;
    const int n_samples_30s  = (1e-3*30000.0         )*WHISPER_SAMPLE_RATE;

    const bool use_vad = n_samples_step <= 0; // sliding window mode uses VAD

    const int n_new_line = !use_vad ? std::max(1, params.length_ms / params.step_ms - 1) : 1; // number of steps to print new line

    params.no_timestamps  = false; //!use_vad;
    params.no_context    |= use_vad;
    params.max_tokens     = 0;

    // init audio - 
    audio_async audio(params.length_ms);
    if (!audio.init(params.capture_id, WHISPER_SAMPLE_RATE)) {
        wr.status = "could not start microphone.";
        fprintf(stderr, "%s: audio.init() failed!\n", __func__);
        return wr;
    }

    audio.resume();

    // whisper init
    if (params.language != "auto" && whisper_lang_id(params.language.c_str()) == -1){
        fprintf(stderr, "error: unknown language '%s'\n", params.language.c_str());
        whisper_print_usage(argc, argv, params);
        exit(0);
    }

    struct whisper_context_params cparams;
    cparams.use_gpu = params.use_gpu;

    struct whisper_context * ctx = whisper_init_from_file_with_params(params.model.c_str(), cparams);

    std::vector<float> pcmf32    (n_samples_30s, 0.0f);
    std::vector<float> pcmf32_old;
    std::vector<float> pcmf32_new(n_samples_30s, 0.0f);

    std::vector<whisper_token> prompt_tokens;

    int n_iter = 0;

    bool is_running = true;

    std::ofstream fout;
    if (params.fname_out.length() > 0) {
        fout.open(params.fname_out);
        if (!fout.is_open()) {
            wr.status = "could not open output file";
            fprintf(stderr, "%s: failed to open output file '%s'!\n", __func__, params.fname_out.c_str());
            return wr;
        }
    }

    wav_writer wavWriter;
    // save wav file
    if (params.save_audio) {
        // Get current date/time for filename
        time_t now = time(0);
        char buffer[80];
        strftime(buffer, sizeof(buffer), "%Y%m%d%H%M%S", localtime(&now));
        std::string filename = std::string(buffer) + ".wav";

        wavWriter.open(filename, WHISPER_SAMPLE_RATE, 16, 1);
    }
    printf("[Start speaking]\n");
    fflush(stdout);

    auto t_last  = std::chrono::high_resolution_clock::now();
    const auto t_start = t_last;

    // main audio loop
    while (is_running) {
        if (params.save_audio) {
            wavWriter.write(pcmf32_new.data(), pcmf32_new.size());
        }
        // handle Ctrl + C
        is_running = sdl_poll_events();

        if (!is_running) {
            break;
        }

        // process new audio
        if (!use_vad) {
            while (true) {
                audio.get(params.step_ms, pcmf32_new);

                if ((int) pcmf32_new.size() > 2*n_samples_step) {
                    fprintf(stderr, "\n\n%s: WARNING: cannot process audio fast enough, dropping audio ...\n\n", __func__);
                    audio.clear();
                    continue;
                }

                if ((int) pcmf32_new.size() >= n_samples_step) {
                    audio.clear();
                    break;
                }

                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }

            const int n_samples_new = pcmf32_new.size();

            // take up to params.length_ms audio from previous iteration
            const int n_samples_take = std::min((int) pcmf32_old.size(), std::max(0, n_samples_keep + n_samples_len - n_samples_new));

            //printf("processing: take = %d, new = %d, old = %d\n", n_samples_take, n_samples_new, (int) pcmf32_old.size());

            pcmf32.resize(n_samples_new + n_samples_take);

            for (int i = 0; i < n_samples_take; i++) {
                pcmf32[i] = pcmf32_old[pcmf32_old.size() - n_samples_take + i];
            }

            memcpy(pcmf32.data() + n_samples_take, pcmf32_new.data(), n_samples_new*sizeof(float));

            pcmf32_old = pcmf32;
        } else {
            const auto t_now  = std::chrono::high_resolution_clock::now();
            const auto t_diff = std::chrono::duration_cast<std::chrono::milliseconds>(t_now - t_last).count();

            if (t_diff < 2000) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }

            audio.get(2000, pcmf32_new);

            if (::vad_simple(pcmf32_new, WHISPER_SAMPLE_RATE, 1000, params.vad_thold, params.freq_thold, false)) {
                audio.get(params.length_ms, pcmf32);
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));

                continue;
            }

            t_last = t_now;
        }

        // run the inference
        {
            whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);

            wparams.print_progress   = false;
            wparams.print_special    = params.print_special;
            wparams.print_realtime   = false;
            wparams.print_timestamps = !params.no_timestamps;
            wparams.translate        = params.translate;
            wparams.single_segment   = !use_vad;
            wparams.max_tokens       = params.max_tokens;
            wparams.language         = params.language.c_str();
            wparams.n_threads        = params.n_threads;

            wparams.audio_ctx        = params.audio_ctx;
            wparams.speed_up         = params.speed_up;

            wparams.tdrz_enable      = params.tinydiarize; // [TDRZ]

            // disable temperature fallback
            //wparams.temperature_inc  = -1.0f;
            wparams.temperature_inc  = params.no_fallback ? 0.0f : wparams.temperature_inc;

            wparams.prompt_tokens    = params.no_context ? nullptr : prompt_tokens.data();
            wparams.prompt_n_tokens  = params.no_context ? 0       : prompt_tokens.size();

            if (whisper_full(ctx, wparams, pcmf32.data(), pcmf32.size()) != 0) {
                fprintf(stderr, "%s: failed to process audio\n", argv[0]);
                wr.status = "failed to process audio";
                return wr;
            }

            // print result;
            {
                if (!use_vad) {
                    printf("\33[2K\r");

                    // print long empty line to clear the previous line
                    printf("%s", std::string(100, ' ').c_str());

                    printf("\33[2K\r");
                } else {
                    const int64_t t1 = (t_last - t_start).count()/1000000;
                    const int64_t t0 = std::max(0.0, t1 - pcmf32.size()*1000.0/WHISPER_SAMPLE_RATE);

                    printf("\n");
                    printf("### Transcription %d START | t0 = %d ms | t1 = %d ms\n", n_iter, (int) t0, (int) t1);
                    printf("\n");
                }

                const int n_segments = whisper_full_n_segments(ctx);
                for (int i = 0; i < n_segments; ++i) {
                    const char * text = whisper_full_get_segment_text(ctx, i);

                    if (params.no_timestamps) {
                        printf("%s", text);
                        fflush(stdout);

                        if (params.fname_out.length() > 0) {
                            fout << text;
                        }
                    } else {
                        const int64_t t0 = whisper_full_get_segment_t0(ctx, i);
                        const int64_t t1 = whisper_full_get_segment_t1(ctx, i);

                        std::string output = "[" + to_timestamp(t0) + " --> " + to_timestamp(t1) + "]  " + text;

                        if (whisper_full_get_segment_speaker_turn_next(ctx, i)) {
                            output += " [SPEAKER_TURN]";
                        }

                        output += "\n";

                        printf("%s", output.c_str());
                        fflush(stdout);

                        if (params.fname_out.length() > 0) {
                            fout << output;
                        }
                    }
                }

                if (params.fname_out.length() > 0) {
                    fout << std::endl;
                }

                if (use_vad) {
                    printf("\n");
                    printf("### Transcription %d END\n", n_iter);
                }
            }

            ++n_iter;

            if (!use_vad && (n_iter % n_new_line) == 0) {
                printf("\n");

                // keep part of the audio for next iteration to try to mitigate word boundary issues
                pcmf32_old = std::vector<float>(pcmf32.end() - n_samples_keep, pcmf32.end());

                // Add tokens of the last full length segment as the prompt
                if (!params.no_context) {
                    prompt_tokens.clear();

                    const int n_segments = whisper_full_n_segments(ctx);
                    for (int i = 0; i < n_segments; ++i) {
                        const int token_count = whisper_full_n_tokens(ctx, i);
                        for (int j = 0; j < token_count; ++j) {
                            prompt_tokens.push_back(whisper_full_get_token_id(ctx, i, j));
                        }
                    }
                }
            }
            fflush(stdout);
        }
    }

    audio.pause();

    whisper_print_timings(ctx);
    whisper_free(ctx);

    return wr;
}
*/ 
