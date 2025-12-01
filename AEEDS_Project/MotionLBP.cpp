#include "MotionLBP.h"
#include <cmath>

using namespace cv;

#define EPS 1e-6f

const unsigned char lookup[256] = {
    0,  1,  2,  3,  4,  58, 5,  6,  7,  58, 58, 58, 8,  58, 9,  10,
    11, 58, 58, 58, 58, 58, 58, 58, 12, 58, 58, 58, 13, 58, 14, 15,
    16, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58,
    17, 58, 58, 58, 58, 58, 58, 58, 18, 58, 58, 58, 19, 58, 20, 21,
    22, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58,
    58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58,
    23, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58,
    24, 58, 58, 58, 58, 58, 58, 58, 25, 58, 58, 58, 26, 58, 27, 28,
    29, 30, 58, 31, 58, 58, 58, 32, 58, 58, 58, 58, 58, 58, 58, 33,
    58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 34,
    58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58,
    58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 35,
    36, 37, 58, 38, 58, 58, 58, 39, 58, 58, 58, 58, 58, 58, 58, 40,
    58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 41,
    42, 43, 58, 44, 58, 58, 58, 45, 58, 58, 58, 58, 58, 58, 58, 46,
    47, 48, 58, 49, 58, 58, 58, 50, 51, 52, 58, 53, 54, 55, 56, 57
};

void mlbp_uniformLBPHistogram(const Mat& src, float* hist)
{
    for (int i = 0; i < ULBP_BIN; ++i) hist[i] = 0.0f;

    Mat gray;
    if (src.channels() == 3)
        cvtColor(src, gray, COLOR_BGR2GRAY);
    else
        gray = src;

    for (int y = 1; y < gray.rows - 1; ++y) {
        const uchar* prev = gray.ptr<uchar>(y - 1);
        const uchar* curr = gray.ptr<uchar>(y);
        const uchar* next = gray.ptr<uchar>(y + 1);
        for (int x = 1; x < gray.cols - 1; ++x) {
            uchar center = curr[x];
            int code = 0;
            if (prev[x] > center) code |= (1 << 0);
            if (prev[x + 1] > center) code |= (1 << 1);
            if (curr[x + 1] > center) code |= (1 << 2);
            if (next[x + 1] > center) code |= (1 << 3);
            if (next[x] > center) code |= (1 << 4);
            if (next[x - 1] > center) code |= (1 << 5);
            if (curr[x - 1] > center) code |= (1 << 6);
            if (prev[x - 1] > center) code |= (1 << 7);
            int bin = lookup[code];
            hist[bin] += 1.0f;
        }
    }

    float norm = 0.0f;
    for (int i = 0; i < ULBP_BIN; ++i) norm += hist[i] * hist[i];
    norm = sqrtf(norm) + EPS;
    for (int i = 0; i < ULBP_BIN; ++i) hist[i] /= norm;
}

float mlbp_cosineSimilarityExp(const float* feat, const float* tpl) {
    if (!feat || !tpl) return 0.0f;
    float dot = 0.0f, normFeat = 0.0f, normTpl = 0.0f;
    for (int i = 0; i < ULBP_BIN; ++i) {
        dot += feat[i] * tpl[i];
        normFeat += feat[i] * feat[i];
        normTpl += tpl[i] * tpl[i];
    }
    float denom = sqrtf(normFeat) * sqrtf(normTpl);
    float cosine = (denom > EPS) ? (dot / denom) : 0.0f;
    return exp(5.0f * cosine);
}

void mlbp_averageTemplates(float** samples, int sampleCount, float* outTpl) {
    if (!samples || sampleCount <= 0 || !outTpl) return;
    for (int i = 0; i < ULBP_BIN; ++i) outTpl[i] = 0.0f;

    for (int s = 0; s < sampleCount; ++s) {
        const float* sample = samples[s];
        if (!sample) continue;
        for (int i = 0; i < ULBP_BIN; ++i) outTpl[i] += sample[i];
    }
    float n = (float)sampleCount;
    for (int i = 0; i < ULBP_BIN; ++i) outTpl[i] /= n;

    float norm = 0.0f;
    for (int i = 0; i < ULBP_BIN; ++i) norm += outTpl[i] * outTpl[i];
    norm = sqrtf(norm) + EPS;
    for (int i = 0; i < ULBP_BIN; ++i) outTpl[i] /= norm;
}