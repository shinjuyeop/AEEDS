#pragma once
#include <opencv2/opencv.hpp>

// Compute foreground mask using GrabCut with expanded region around detected face
void seg_getForegroundMaskWithGrabCut(const cv::Mat& src, cv::Mat& outputMask, cv::Rect faceRect);
