#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>

using namespace cv;
using namespace cv::ml;
using namespace std;

// ------------------- 상수 정의 -------------------
#define COSINE       // 코사인 유사도 사용
constexpr float PI = 3.14159265358979323846f;
constexpr float EPS = 1e-6f;

constexpr int WIN = 36;
constexpr int BLOCK = 12;
constexpr int NBINS = 9;
constexpr int STR = BLOCK / 2;                      // 6
constexpr int NX = (WIN - BLOCK) / STR + 1;         // 5
constexpr int NY = (WIN - BLOCK) / STR + 1;         // 5
constexpr int DIM = NBINS * NX * NY;                // 225
// -------------------------------------------------

// --------------- 1) ref 이미지 HOG ---------------
void refHOG(const Mat& input, float* out_hog) {
    int x, y, xx, yy;
    int height = input.rows, width = input.cols;
    const int b_size = 3;
    float conv_x, conv_y;

    int mask_x[9] = { -1, 0, 1,  -1, 0, 1,  -1, 0, 1 };
    int mask_y[9] = { -1,-1,-1,   0, 0, 0,   1, 1, 1 };

    float* magnitude = (float*)calloc(height * width, sizeof(float));
    unsigned char* dir = (unsigned char*)calloc(height * width, sizeof(unsigned char));

    // 1) gradient 계산
    for (y = 0; y < height; ++y) {
        for (x = 0; x < width; ++x) {
            conv_x = conv_y = 0.0f;
            for (yy = y - b_size / 2; yy <= y + b_size / 2; ++yy) {
                if (yy < 0 || yy >= height) continue;
                const unsigned char* prow = input.data + yy * input.step;
                for (xx = x - b_size / 2; xx <= x + b_size / 2; ++xx) {
                    if (xx < 0 || xx >= width) continue;
                    int ky = yy - (y - b_size / 2);
                    int kx = xx - (x - b_size / 2);
                    float I = (float)prow[xx] / 255.0f;
                    conv_x += mask_x[ky * b_size + kx] * I;
                    conv_y += mask_y[ky * b_size + kx] * I;
                }
            }
            float M = sqrt(conv_x * conv_x + conv_y * conv_y);
            float theta = atan2f(conv_y, conv_x) * 180.0f / PI;
            if (theta < 0.0f) theta += 180.0f;
            int b = (int)(theta / (180.0f / NBINS));
            if (b >= NBINS) b = NBINS - 1;
            magnitude[y * width + x] = M;
            dir[y * width + x] = (unsigned char)b;
        }
    }

    // 2) HOG 계산
    int bi = 0;
    for (int by = 0; by <= height - BLOCK; by += STR) {
        for (int bx = 0; bx <= width - BLOCK; bx += STR) {
            vector<float> hist(NBINS, 0.0f);
            for (y = by; y < by + BLOCK; ++y) {
                for (x = bx; x < bx + BLOCK; ++x) {
                    unsigned char b = dir[y * width + x];
                    hist[b] += magnitude[y * width + x];
                }
            }
            float n2 = EPS;
            for (int i = 0; i < NBINS; ++i) n2 += hist[i] * hist[i];
            n2 = 1.0f / sqrt(n2);
            for (int i = 0; i < NBINS; ++i)
                out_hog[(bi * NBINS) + i] = hist[i] * n2;
            bi++;
        }
    }

    free(magnitude);
    free(dir);
}

// --------------- 2) tar 이미지 Gradient, Orientation 계산 ---------------
void tar_mag_ori(const Mat& input, float** out_mag, unsigned char** out_dir) {
    int x, y, xx, yy;
    int height = input.rows, width = input.cols;
    const int b_size = 3;
    float conv_x, conv_y;

    int mask_x[9] = { -1, 0, 1,  -1, 0, 1,  -1, 0, 1 };
    int mask_y[9] = { -1,-1,-1,   0, 0, 0,   1, 1, 1 };

    float* tar_mag = (float*)calloc(height * width, sizeof(float));
    unsigned char* tar_dir = (unsigned char*)calloc(height * width, sizeof(unsigned char));

    for (y = 0; y < height; ++y) {
        for (x = 0; x < width; ++x) {
            conv_x = conv_y = 0.0f;
            for (yy = y - b_size / 2; yy <= y + b_size / 2; ++yy) {
                if (yy < 0 || yy >= height) continue;
                const unsigned char* prow = input.data + yy * input.step;
                for (xx = x - b_size / 2; xx <= x + b_size / 2; ++xx) {
                    if (xx < 0 || xx >= width) continue;
                    int ky = yy - (y - b_size / 2);
                    int kx = xx - (x - b_size / 2);
                    float I = (float)prow[xx] / 255.0f;
                    conv_x += mask_x[ky * b_size + kx] * I;
                    conv_y += mask_y[ky * b_size + kx] * I;
                }
            }
            float M = sqrt(conv_x * conv_x + conv_y * conv_y);
            float theta = atan2f(conv_y, conv_x) * 180.0f / PI;
            if (theta < 0.0f) theta += 180.0f;
            int b = (int)(theta / (180.0f / NBINS));
            if (b >= NBINS) b = NBINS - 1;
            tar_mag[y * width + x] = M;
            tar_dir[y * width + x] = (unsigned char)b;
        }
    }
    *out_mag = tar_mag;
    *out_dir = tar_dir;
}

// --------------- 3) 윈도우 HOG 계산 ---------------
void tarHOG(const float* testMag, const unsigned char* testDir, float* tarHOG_descriptor) {
    int bi = 0;
    for (int by = 0; by <= WIN - BLOCK; by += STR) {
        for (int bx = 0; bx <= WIN - BLOCK; bx += STR) {
            vector<float> hist(NBINS, 0.0f);
            for (int y = by; y < by + BLOCK; ++y) {
                int row = y * WIN;
                for (int x = bx; x < bx + BLOCK; ++x) {
                    unsigned char b = testDir[row + x];
                    hist[b] += testMag[row + x];
                }
            }
            float n2 = EPS;
            for (int i = 0; i < NBINS; ++i) n2 += hist[i] * hist[i];
            n2 = 1.0f / sqrt(n2);
            for (int i = 0; i < NBINS; ++i)
                tarHOG_descriptor[(bi * NBINS) + i] = hist[i] * n2;
            bi++;
        }
    }
}

// --------------- 4) 유사도 계산  -----------------
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

    // 유사도 대비 강조
    return exp(5.0f * score);
}

// --------------- 5) 얼굴 탐색 (픽셀 단위, 70% 이상 마킹) ---------------
void searchFaces(const Mat& tar_img, const float* refHOG_descriptor) {
    float* tar_mag = nullptr;
    unsigned char* tar_dir = nullptr;
    tar_mag_ori(tar_img, &tar_mag, &tar_dir);

    float* testMag = (float*)malloc(WIN * WIN * sizeof(float));
    unsigned char* testDir = (unsigned char*)malloc(WIN * WIN * sizeof(unsigned char));
    float* tarHOG_descriptor = (float*)malloc(DIM * sizeof(float));

    float maxScore = -1.0f;
    Point maxPt(-1, -1);
    int height = tar_img.rows, width = tar_img.cols;

    size_t total = static_cast<size_t>(height) * static_cast<size_t>(width);
    float* scores = (float*)malloc(total * sizeof(float));
    Point* points = (Point*)malloc(total * sizeof(Point));

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = 0;
            for (int yy = y - WIN / 2; yy < y + WIN / 2; ++yy) {
                for (int xx = x - WIN / 2; xx < x + WIN / 2; ++xx) {
                    if (yy >= 0 && yy < height && xx >= 0 && xx < width) {
                        testMag[idx] = tar_mag[yy * width + xx];
                        testDir[idx] = tar_dir[yy * width + xx];
                    }
                    else {
                        testMag[idx] = 0.0f;
                        testDir[idx] = 0;
                    }
                    idx++;
                }
            }

            tarHOG(testMag, testDir, tarHOG_descriptor);
            float score = computeSimilarity(refHOG_descriptor, tarHOG_descriptor);

            size_t p = static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x);
            scores[p] = score;
            points[p].x = x;
            points[p].y = y;

            if (score > maxScore) {
                maxScore = score;
                maxPt = Point(x, y);
            }
        }
    }

    float threshold = 0.7f * maxScore; // 70% 기준
    Mat vis;
    cvtColor(tar_img, vis, COLOR_GRAY2BGR);
    int count = 0;

    for (size_t i = 0; i < total; ++i) {
        if (scores[i] >= threshold) {
            circle(vis, points[i], 0, Scalar(0, 0, 255), FILLED);
            count++;
        }
    }

    //rectangle(vis, Rect(maxPt.x - WIN / 2, maxPt.y - WIN / 2, WIN, WIN), Scalar(0, 255, 0), 2);

    printf("Max similarity: %.6f at (%d, %d)\n", maxScore, maxPt.x, maxPt.y);
    printf("Threshold (70%% of max): %.6f\n", threshold);
    printf("Marked points: %d / %zu\n", count, total);

    imshow("detections", vis);
    waitKey(0);

    free(tar_mag);
    free(tar_dir);
    free(testMag);
    free(testDir);
    free(tarHOG_descriptor);
    free(scores);
    free(points);
}

// --------------- 6) main ---------------
int main() {
    Mat ref_img = imread("C:/Users/shinj/Desktop/3-2/AEEDS/Data/face_ref.bmp", 0);
    Mat tar_img = imread("C:/Users/shinj/Desktop/3-2/AEEDS/Data/face_tar.bmp", 0);

    float refHOG_descriptor[DIM];
    refHOG(ref_img, refHOG_descriptor);
    searchFaces(tar_img, refHOG_descriptor);

    return 0;
}
