#pragma once
#include <opencv2/opencv.hpp>

enum class BgMode { Original = 0, Gray = 1, Blur = 2, Mosaic = 3, Image = 4 };

// Load/set background image used in Image mode
void be_setBackgroundImage(const cv::Mat& img);

// Create full-frame effects
void be_makeGrayEffect(const cv::Mat& src, cv::Mat& dst);
void be_gaussianBlurEffect(const cv::Mat& src, cv::Mat& dst, int ksize);
void be_mosaicEffect(const cv::Mat& src, cv::Mat& dst, int blockSize);

// Apply selected background effect only to background using a foreground mask
void be_applyBackgroundEffectMask(const cv::Mat& src, cv::Mat& dst, const cv::Mat& fgMask, BgMode mode);
