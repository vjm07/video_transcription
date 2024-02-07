#include <iostream>
#include <fstream>
#include "video_handler.hpp"

extern "C" {
    #include <libavformat/avformat.h>
    #include <libavcodec/avcodec.h>
    #include <libavutil/opt.h>
    #include <libavutil/channel_layout.h>
    #include <libavutil/samplefmt.h>
    #include <libswresample/swresample.h>
}

// Function to write the WAV header
void writeWavHeader(std::ofstream& file, const AVCodecContext* codecCtx, uint32_t dataSize) {
    uint32_t sampleRate = codecCtx->sample_rate;
    uint16_t numChannels = codecCtx->channels;
    uint16_t bitsPerSample = 16;  // We're converting audio to 16-bit PCM

    // RIFF header
    file.write("RIFF", 4);
    uint32_t fileSizeMinus8 = dataSize + 36;
    file.write(reinterpret_cast<const char*>(&fileSizeMinus8), 4);
    file.write("WAVE", 4);

    // fmt subchunk
    file.write("fmt ", 4);
    uint32_t fmtChunkSize = 16;
    file.write(reinterpret_cast<const char*>(&fmtChunkSize), 4);
    uint16_t audioFormat = 1; // PCM
    file.write(reinterpret_cast<const char*>(&audioFormat), 2);
    file.write(reinterpret_cast<const char*>(&numChannels), 2);
    file.write(reinterpret_cast<const char*>(&sampleRate), 4);
    uint32_t byteRate = sampleRate * numChannels * bitsPerSample / 8;
    file.write(reinterpret_cast<const char*>(&byteRate), 4);
    uint16_t blockAlign = numChannels * bitsPerSample / 8;
    file.write(reinterpret_cast<const char*>(&blockAlign), 2);
    file.write(reinterpret_cast<const char*>(&bitsPerSample), 2);

    // data subchunk
    file.write("data", 4);
    file.write(reinterpret_cast<const char*>(&dataSize), 4);
}

int extract_audio_from_video(const char* input_file, const char* output_file) {
    AVFormatContext* formatContext = nullptr;
    avformat_open_input(&formatContext, input_file, nullptr, nullptr);
    avformat_find_stream_info(formatContext, nullptr);

    int audioStreamIndex = -1;
    for (unsigned int i = 0; i < formatContext->nb_streams; i++) {
        if (formatContext->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
            audioStreamIndex = i;
            break;
        }
    }

    if (audioStreamIndex == -1) {
        std::cerr << "Audio stream not found" << std::endl;
        avformat_close_input(&formatContext);
        return -1;
    }

    AVStream* audioStream = formatContext->streams[audioStreamIndex];
    AVCodec* audioDecoder = avcodec_find_decoder(audioStream->codecpar->codec_id);
    AVCodecContext* codecContext = avcodec_alloc_context3(audioDecoder);
    avcodec_parameters_to_context(codecContext, audioStream->codecpar);
    avcodec_open2(codecContext, audioDecoder, nullptr);

    SwrContext* swrCtx = swr_alloc_set_opts(nullptr,
                                            av_get_default_channel_layout(codecContext->channels), AV_SAMPLE_FMT_S16, codecContext->sample_rate,
                                            av_get_default_channel_layout(codecContext->channels), codecContext->sample_fmt, codecContext->sample_rate,
                                            0, nullptr);
    swr_init(swrCtx);

    AVPacket* packet = av_packet_alloc();
    AVFrame* frame = av_frame_alloc();

    std::ofstream wavFile(output_file, std::ios::binary);
    if (!wavFile.is_open()) {
        std::cerr << "Could not open output WAV file for writing." << std::endl;
        return -1;
    }
    wavFile.seekp(44); // Reserve space for the header, to be filled later

    uint32_t totalDataSize = 0;

    while (av_read_frame(formatContext, packet) >= 0) {
        if (packet->stream_index == audioStreamIndex) {
            avcodec_send_packet(codecContext, packet);
            while (avcodec_receive_frame(codecContext, frame) == 0) {
                AVFrame* tempFrame = av_frame_alloc();
                tempFrame->format = AV_SAMPLE_FMT_S16;
                tempFrame->channel_layout = av_get_default_channel_layout(codecContext->channels);
                tempFrame->sample_rate = codecContext->sample_rate;
                tempFrame->nb_samples = frame->nb_samples;
                av_frame_get_buffer(tempFrame, 0);
                swr_convert_frame(swrCtx, tempFrame, frame);

                int bufferSize = av_samples_get_buffer_size(nullptr, codecContext->channels,
                                                            tempFrame->nb_samples, AV_SAMPLE_FMT_S16, 1);
                wavFile.write(reinterpret_cast<char*>(tempFrame->data[0]), bufferSize);
                totalDataSize += bufferSize;

                av_frame_free(&tempFrame);
            }
        }
        av_packet_unref(packet);
    }

    wavFile.seekp(0);
    writeWavHeader(wavFile, codecContext, totalDataSize);

    wavFile.close();
    av_frame_free(&frame);
    av_packet_free(&packet);
    avcodec_close(codecContext);
    avcodec_free_context(&codecContext);
    avformat_close_input(&formatContext);
    swr_free(&swrCtx);

    return 0;
};

void writeWavHeader(std::ofstream& file, const WAVHeader& header) {
    file.write(header.riff, sizeof(header.riff));
    file.write(reinterpret_cast<const char*>(&header.overallSize), sizeof(header.overallSize));
    file.write(header.wave, sizeof(header.wave));
    file.write(header.fmtChunkMarker, sizeof(header.fmtChunkMarker));
    file.write(reinterpret_cast<const char*>(&header.lengthOfFmt), sizeof(header.lengthOfFmt));
    file.write(reinterpret_cast<const char*>(&header.formatType), sizeof(header.formatType));
    file.write(reinterpret_cast<const char*>(&header.channels), sizeof(header.channels));
    file.write(reinterpret_cast<const char*>(&header.sampleRate), sizeof(header.sampleRate));
    file.write(reinterpret_cast<const char*>(&header.byteRate), sizeof(header.byteRate));
    file.write(reinterpret_cast<const char*>(&header.blockAlign), sizeof(header.blockAlign));
    file.write(reinterpret_cast<const char*>(&header.bitsPerSample), sizeof(header.bitsPerSample));
    file.write(header.dataChunkHeader, sizeof(header.dataChunkHeader));
    file.write(reinterpret_cast<const char*>(&header.dataSize), sizeof(header.dataSize));
}

bool convertTo16kHzWav(const char* inputPath, const char* outputPath) {
    avformat_network_init();

    AVFormatContext* formatContext = nullptr;
    if (avformat_open_input(&formatContext, inputPath, nullptr, nullptr) != 0) {
        std::cerr << "Could not open input file: " << inputPath << std::endl;
        return false;
    }

    if (avformat_find_stream_info(formatContext, nullptr) < 0) {
        std::cerr << "Could not find stream information in the file" << std::endl;
        avformat_close_input(&formatContext);
        return false;
    }

    int audioStreamIndex = av_find_best_stream(formatContext, AVMEDIA_TYPE_AUDIO, -1, -1, nullptr, 0);
    if (audioStreamIndex < 0) {
        std::cerr << "Audio stream not found in the input file" << std::endl;
        avformat_close_input(&formatContext);
        return false;
    }

    AVStream* audioStream = formatContext->streams[audioStreamIndex];
    AVCodecContext* codecContext = avcodec_alloc_context3(nullptr);
    avcodec_parameters_to_context(codecContext, audioStream->codecpar);
    AVCodec* codec = avcodec_find_decoder(codecContext->codec_id);
    if (!codec || avcodec_open2(codecContext, codec, nullptr) < 0) {
        std::cerr << "Failed to open codec" << std::endl;
        avcodec_free_context(&codecContext);
        avformat_close_input(&formatContext);
        return false;
    }

    SwrContext* swrCtx = swr_alloc_set_opts(nullptr,
                                            av_get_default_channel_layout(codecContext->channels), AV_SAMPLE_FMT_S16, 16000,
                                            av_get_default_channel_layout(codecContext->channels), codecContext->sample_fmt, codecContext->sample_rate,
                                            0, nullptr);
    swr_init(swrCtx);

    std::ofstream wavFile(outputPath, std::ios::binary);
    if (!wavFile.is_open()) {
        std::cerr << "Could not open output WAV file for writing: " << outputPath << std::endl;
        swr_free(&swrCtx);
        avcodec_close(codecContext);
        avcodec_free_context(&codecContext);
        avformat_close_input(&formatContext);
        return false;
    }

    WAVHeader wavHeader;
    wavHeader.channels = codecContext->channels;
    wavHeader.sampleRate = 16000; // Target sample rate
    wavHeader.bitsPerSample = 16;
    wavHeader.byteRate = wavHeader.sampleRate * wavHeader.channels * wavHeader.bitsPerSample / 8;
    wavHeader.blockAlign = wavHeader.channels * wavHeader.bitsPerSample / 8;
    // Placeholder values for dataSize and overallSize, to be updated later
    wavHeader.dataSize = 0;
    wavHeader.overallSize = 36 + wavHeader.dataSize;
    writeWavHeader(wavFile, wavHeader); // Write initial WAV header with placeholders

    AVPacket packet;
    av_init_packet(&packet);
    AVFrame* frame = av_frame_alloc();
    uint32_t totalDataSize = 0;

    while (av_read_frame(formatContext, &packet) >= 0) {
        if (packet.stream_index == audioStreamIndex) {
            if (avcodec_send_packet(codecContext, &packet) == 0) {
                while (avcodec_receive_frame(codecContext, frame) == 0) {
                    int64_t delay = swr_get_delay(swrCtx, codecContext->sample_rate) + frame->nb_samples;
                    int64_t out_samples = av_rescale_rnd(delay, 16000, codecContext->sample_rate, AV_ROUND_UP);
                    uint8_t* outputBuffer = nullptr;
                    av_samples_alloc(&outputBuffer, nullptr, codecContext->channels, out_samples, AV_SAMPLE_FMT_S16, 0);

                    int frameCount = swr_convert(swrCtx, &outputBuffer, out_samples, (const uint8_t**)frame->data, frame->nb_samples);
                    if (frameCount < 0) {
                        std::cerr << "Resampling failed for the frame" << std::endl;
                        av_freep(&outputBuffer);
                        continue;
                    }

                    int bufferSize = av_samples_get_buffer_size(nullptr, codecContext->channels, frameCount, AV_SAMPLE_FMT_S16, 1);
                    wavFile.write(reinterpret_cast<char*>(outputBuffer), bufferSize);
                    totalDataSize += bufferSize;

                    av_freep(&outputBuffer);
                }
            }
        }
        av_packet_unref(&packet);
    }

    // Update the WAV header with the correct data size
    wavHeader.dataSize = totalDataSize;
    wavHeader.overallSize = 36 + totalDataSize; // 36 bytes for header fields except 'RIFF' and 'overallSize'
    wavFile.seekp(0, std::ios::beg);
    writeWavHeader(wavFile, wavHeader);

    wavFile.close();
    av_frame_free(&frame);
    swr_free(&swrCtx);
    avcodec_close(codecContext);
    avcodec_free_context(&codecContext);
    avformat_close_input(&formatContext);

    return true;
}

