#ifndef VIDEO_HANDLER_HPP
#define VIDEO_HANDLER_HPP
#endif

#include <iostream>
#include <fstream>

extern "C" {
    #include <libavformat/avformat.h>
    #include <libavcodec/avcodec.h>
    #include <libavutil/opt.h>
    #include <libavutil/channel_layout.h>
    #include <libavutil/samplefmt.h>
    #include <libswresample/swresample.h>
}

struct WAVHeader {
    char riff[4] = {'R', 'I', 'F', 'F'};
    uint32_t overallSize;  // Placeholder, to be updated
    char wave[4] = {'W', 'A', 'V', 'E'};
    char fmtChunkMarker[4] = {'f', 'm', 't', ' '};
    uint32_t lengthOfFmt = 16;
    uint16_t formatType = 1; // PCM
    uint16_t channels;
    uint32_t sampleRate;
    uint32_t byteRate;  // Calculated as sampleRate * channels * bitsPerSample / 8
    uint16_t blockAlign;  // Calculated as channels * bitsPerSample / 8
    uint16_t bitsPerSample;
    char dataChunkHeader[4] = {'d', 'a', 't', 'a'};
    uint32_t dataSize;  // Placeholder, to be updated
};

void writeWavHeader(std::ofstream& file, const AVCodecContext* codecCtx, uint32_t dataSize);

int extract_audio_from_video(const char* input_file, const char* output_file);
void writeWavHeader(std::ofstream& file, int numChannels, int sampleRate, int bitsPerSample, int dataSize);
bool convertTo16kHzWav(const char* inputPath, const char* outputPath);