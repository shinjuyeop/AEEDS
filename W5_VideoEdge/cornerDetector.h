#pragma once
#include <opencv2/opencv.hpp>

// Harris Corner Detector
// frame (BGR or Gray) will be modified in-place: green circles drawn at detected corners.
// cornerMap returns normalized (0~255) corner response for visualization/debug.
// thresholdVal: 0~255 intensity on normalized response (higher -> fewer corners).
// radius: circle radius.
// winSize: summation window (3 or 5; larger -> smoother, fewer corners but slower).
// k: Harris detector free parameter (0.04~0.06 typical).
void HarrisCornerDetector(cv::Mat& frame, cv::Mat& cornerMap, int thresholdVal = 200, int radius = 4, int winSize = 3, float k = 0.04f);
