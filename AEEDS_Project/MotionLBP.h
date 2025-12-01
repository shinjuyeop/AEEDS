#pragma once
// MotionLBP.h - C-style interface for motion magnitude LBP based gesture templates.
// Feature vector layout: 59-bin uniform LBP histogram.

#include <opencv2/opencv.hpp>

#ifdef __cplusplus
extern "C" {
#endif

#define ULBP_BIN 59

// Compute uniform LBP histogram (59 bins) for a single-channel 8-bit patch.
void mlbp_uniformLBPHistogram(const cv::Mat& patch, float* hist59);

// Cosine similarity between two 59-dim feature vectors. Returns exp(5*rawCos).
float mlbp_cosineSimilarityExp(const float* a, const float* b);

// Average N 59-dim feature samples into meanOut (allocated by caller).
// samples: pointer to array of pointers, each pointing to 59 floats
void mlbp_averageTemplates(float** samples, int sampleCount, float* meanOut);

#ifdef __cplusplus
}
#endif
