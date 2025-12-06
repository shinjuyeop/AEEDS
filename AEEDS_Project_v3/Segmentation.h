#pragma once
#include <opencv2/opencv.hpp>
using namespace cv;

// Compute foreground mask using GrabCut with expanded region around detected face
void seg_getForegroundMaskWithGrabCut(const Mat& src, Mat& outputMask, Rect faceRect);
