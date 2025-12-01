#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

// window 전체에서59-bin LBP 히스토그램만 추출
void LBP_histogram_window(const cv::Mat& patch, float* hist);

void extractLandmarkLBPFeatures(
 const cv::Mat& image,
 const std::vector<cv::Point>& landmarks,
 const std::vector<int>& selected_indices,
 int window_size,
 std::vector<float>& feature_vector);

// C 스타일: feature vector를 out_feature에 저장, 반환값은 feature float 개수
int extractLandmarkLBPFeatures_C(
 const cv::Mat& image,
 const cv::Point* landmarks,
 const int* selected_indices,
 int num_indices,
 int window_size,
 float* out_feature);

float computeSimilarity(const float* ref, const float* tar);
