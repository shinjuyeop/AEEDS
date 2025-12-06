#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

// Compute hand ROI to the right of face, clipped to frame. Returns false if too small.
bool gest_computeHandROI(const cv::Rect& faceRect, const cv::Size& frameSize, cv::Rect& outROI);

// Extract 59-bin uniform LBP histogram from ROI (grayscale input). Returns empty vector on failure.
//std::vector<float> gest_extractLBPFeature(const cv::Mat& roiGray);
