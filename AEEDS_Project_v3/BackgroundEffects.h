#pragma once
#include <opencv2/opencv.hpp>
using namespace cv;

typedef enum {
    BG_MODE_ORIGINAL = 0,
    BG_MODE_GRAY,
    BG_MODE_BLUR,
    BG_MODE_MOSAIC,
    BG_MODE_IMAGE
} BgMode;

// Load/set background image used in Image mode
void be_setBackgroundImage(const Mat& img);

// Create full-frame effects
void be_makeGrayEffect(const Mat& src, Mat& dst);
void be_gaussianBlurEffect(const Mat& src, Mat& dst, int ksize);
void be_mosaicEffect(const Mat& src, Mat& dst, int blockSize);

// Apply selected background effect only to background using a foreground mask
void be_applyBackgroundEffectMask(const Mat& src, Mat& dst, const Mat& fgMask, BgMode mode);
