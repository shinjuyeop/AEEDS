#ifndef MOTIONLBP_H
#define MOTIONLBP_H

#include <opencv2/opencv.hpp>
using namespace cv;

// Uniform 59
#define GRID_SIZE 3
#define ULBP_BINS 59 // Uniform LBP Bin 개수
// 전체 특징 벡터 크기: 3 * 3 * 59 = 531
#define TOTAL_LBP_DIM (GRID_SIZE * GRID_SIZE * ULBP_BINS) 

// 함수 선언
// 1. 에지(Gradient Magnitude) 계산 (Lecture 3)
void calc_gradient_magnitude(const Mat& srcGray, Mat& dstMag);
// 2. Grid LBP 특징 추출 (Lecture 8 + Resizing)
void LBPdescriptor_Gesture(const Mat& inputROI, float* LBPhist);
// 3. 코사인 유사도 계산
float mlbp_cosineSimilarityExp(const float* feat, const float* tpl, int size);

#endif