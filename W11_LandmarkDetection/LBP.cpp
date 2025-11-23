#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
//#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>

using namespace cv;
using namespace cv::ml;
using namespace std;

// ------------------- 상수 정의 -------------------
#define COSINE      // 코사인 유사도 사용
#define PI      3.14159265358979323846f
#define EPS     1e-6f
#define WIN     17
#define BLOCK (WIN / 4)      // 지금 사용 X
#define STR   (BLOCK / 2)    // 지금 사용 X

// Uniform LBP (8-neighbor) : 59-bin
#define ULBP_BIN 59
#define DIM      (ULBP_BIN * 41)   // 41 points * 59-bin
// -------------------------------------------------

// 8-neighbor Uniform LBP (59-bin) lookup table
// 0~57 : uniform pattern index
// 58   : non-uniform pattern
const unsigned char lookup[256] = {
    0,  1,  2,  3,  4,  58, 5,  6,  7,  58, 58, 58, 8,  58, 9,  10,
    11, 58, 58, 58, 58, 58, 58, 58, 12, 58, 58, 58, 13, 58, 14, 15,
    16, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58,
    17, 58, 58, 58, 58, 58, 58, 58, 18, 58, 58, 58, 19, 58, 20, 21,
    22, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58,
    58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58,
    23, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58,
    24, 58, 58, 58, 58, 58, 58, 58, 25, 58, 58, 58, 26, 58, 27, 28,
    29, 30, 58, 31, 58, 58, 58, 32, 58, 58, 58, 58, 58, 58, 58, 33,
    58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 34,
    58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58,
    58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 35,
    36, 37, 58, 38, 58, 58, 58, 39, 58, 58, 58, 58, 58, 58, 58, 40,
    58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 41,
    42, 43, 58, 44, 58, 58, 58, 45, 58, 58, 58, 58, 58, 58, 58, 46,
    47, 48, 58, 49, 58, 58, 58, 50, 51, 52, 58, 53, 54, 55, 56, 57
};

// --------------- 유사도 계산  -----------------
float computeSimilarity(const float* ref, const float* tar)
{
    float score = 0.0f;

#ifdef COSINE
    float nomi = 0.0f;
    float refMag = 0.0f;
    float tarMag = 0.0f;

    for (int k = 0; k < DIM; ++k) {
        nomi += ref[k] * tar[k];
        refMag += ref[k] * ref[k];
        tarMag += tar[k] * tar[k];
    }

    float denom = sqrt(refMag) * sqrt(tarMag);
    if (denom > EPS)
        score = nomi / denom;
    else
        score = 0.0f;
#endif

    // 코사인 값을 exp로 스케일
    return exp(5.0f * score);
}

// ---------------- LBP descriptor ----------------
void LBPdescriptor(const Mat& input, float* LBPhist,
    int sx, int sy, int face_width, int face_height)
{
    // 출력 히스토그램 초기화
    for (int i = 0; i < DIM; ++i) LBPhist[i] = 0.0f;

    // 1) 얼굴 부분만 자르고 그레이 변환
    int x0 = max(0, sx);
    int y0 = max(0, sy);
    if (x0 >= input.cols || y0 >= input.rows) return;

    int w = min(face_width, input.cols - x0);
    int h = min(face_height, input.rows - y0);
    if (w < 4 || h < 4) return;   // 너무 작은 얼굴은 무시

    Mat face(h, w, CV_8UC3);      // 얼굴 이미지 (color)
    // 얼굴 픽셀 복사
    for (int yy = 0; yy < h; ++yy) {
        for (int xx = 0; xx < w; ++xx) {
            face.at<Vec3b>(yy, xx) = input.at<Vec3b>(y0 + yy, x0 + xx);
        }
    }

    Mat gray;
    cvtColor(face, gray, COLOR_BGR2GRAY);

    // 2) 그레이 영상을 표준 크기(WIN x WIN)로 리사이즈
    Mat gray128;
    resize(gray, gray128, Size(WIN, WIN));

    // 3) 표준 크기에서 LBP 계산
    Mat lbp128(WIN, WIN, CV_8UC1, Scalar(0));
    for (int y = 1; y < WIN - 1; ++y) {
        for (int x = 1; x < WIN - 1; ++x) {
            uchar c = gray128.at<uchar>(y, x);  // 중심 픽셀 값
            int v = 0;
            if (gray128.at<uchar>(y - 1, x) > c) v |= (1 << 0); // 12시
            if (gray128.at<uchar>(y - 1, x + 1) > c) v |= (1 << 1); // 1:30
            if (gray128.at<uchar>(y, x + 1) > c) v |= (1 << 2); // 3시
            if (gray128.at<uchar>(y + 1, x + 1) > c) v |= (1 << 3); // 4:30
            if (gray128.at<uchar>(y + 1, x) > c) v |= (1 << 4); // 6시
            if (gray128.at<uchar>(y + 1, x - 1) > c) v |= (1 << 5); // 7:30
            if (gray128.at<uchar>(y, x - 1) > c) v |= (1 << 6); // 9시
            if (gray128.at<uchar>(y - 1, x - 1) > c) v |= (1 << 7); // 10:30
            lbp128.at<uchar>(y, x) = (uchar)v;  // 0~255 LBP 코드
        }
    }

    // 4) 블록(32) / 스트라이드(16)로 7x7 위치에서 59-bin 히스토그램 (Uniform LBP)
    float* hist = (float*)calloc(ULBP_BIN, sizeof(float));
    int cnt = 0;

    for (int y = 0; y <= WIN - BLOCK; y += STR) {
        for (int x = 0; x <= WIN - BLOCK; x += STR) {

            // 히스토그램 초기화
            for (int i = 0; i < ULBP_BIN; ++i) hist[i] = 0.0f;

            // 블록 내 LBP 누적
            for (int yy = y; yy < y + BLOCK; ++yy) {
                for (int xx = x; xx < x + BLOCK; ++xx) {
                    uchar v = lbp128.at<uchar>(yy, xx);   // 0~255
                    int bin = lookup[v];                  // 0~58
                    hist[bin] += 1.0f;
                }
            }

            // L2 정규화
            float denomi = 0.f;
            for (int b = 0; b < ULBP_BIN; ++b) denomi += hist[b] * hist[b];
            denomi = sqrtf(denomi) + EPS;

            for (int b = 0; b < ULBP_BIN; ++b)
                LBPhist[cnt * ULBP_BIN + b] = hist[b] / denomi;

            cnt++;
        }
    }
    free(hist);
}

// window 전체에서59-bin LBP 히스토그램만 추출
void LBP_histogram_window(const cv::Mat& patch, float* hist)
{
    // patch는 그레이스케일 영상이어야 함
    std::fill(hist, hist + ULBP_BIN,0.0f);
    int rows = patch.rows, cols = patch.cols;
    for (int y =1; y < rows -1; ++y) {
        for (int x =1; x < cols -1; ++x) {
            uchar c = patch.at<uchar>(y, x);
            int v =0;
            if (patch.at<uchar>(y -1, x) > c) v |= (1 <<0);
            if (patch.at<uchar>(y -1, x +1) > c) v |= (1 <<1);
            if (patch.at<uchar>(y, x +1) > c) v |= (1 <<2);
            if (patch.at<uchar>(y +1, x +1) > c) v |= (1 <<3);
            if (patch.at<uchar>(y +1, x) > c) v |= (1 <<4);
            if (patch.at<uchar>(y +1, x -1) > c) v |= (1 <<5);
            if (patch.at<uchar>(y, x -1) > c) v |= (1 <<6);
            if (patch.at<uchar>(y -1, x -1) > c) v |= (1 <<7);
            int bin = lookup[v];
            hist[bin] +=1.0f;
        }
    }
    // L2 정규화
    float norm = EPS;
    for (int i =0; i < ULBP_BIN; ++i) norm += hist[i] * hist[i];
    norm = sqrtf(norm);
    for (int i =0; i < ULBP_BIN; ++i) hist[i] /= norm;
}

// 랜드마크 기반 LBP 특징 추출 함수
void extractLandmarkLBPFeatures(
    const cv::Mat& image,
    const std::vector<cv::Point>& landmarks,
    const std::vector<int>& selected_indices,
    int window_size,
    std::vector<float>& feature_vector)
{
    feature_vector.clear();
    for (int idx : selected_indices) {
        int cx = landmarks[idx].x;
        int cy = landmarks[idx].y;
        int half = window_size /2;
        int x0 = std::max(0, cx - half);
        int y0 = std::max(0, cy - half);
        int w = std::min(window_size, image.cols - x0);
        int h = std::min(window_size, image.rows - y0);
        if (w <4 || h <4) continue;
        cv::Mat patch = image(cv::Rect(x0, y0, w, h));
        cv::Mat gray_patch;
        if (patch.channels() ==3)
            cv::cvtColor(patch, gray_patch, cv::COLOR_BGR2GRAY);
        else
            gray_patch = patch;
        float hist[ULBP_BIN] = {0};
        LBP_histogram_window(gray_patch, hist);
        feature_vector.insert(feature_vector.end(), hist, hist + ULBP_BIN);
    }
}
