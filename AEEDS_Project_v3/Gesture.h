#pragma once
#include <opencv2/opencv.hpp>
using namespace cv;

// Compute hand ROI to the right of face, clipped to frame. Returns false if too small.
bool gest_computeHandROI(const Rect& faceRect, const Size& frameSize, Rect& outROI);

// Extract 59-bin uniform LBP histogram from ROI (grayscale input). Returns empty vector on failure.
//std::vector<float> gest_extractLBPFeature(const cv::Mat& roiGray);
