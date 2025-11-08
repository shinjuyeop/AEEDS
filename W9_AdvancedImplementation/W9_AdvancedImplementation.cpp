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
#define PI      3.14159265358979323846f
#define EPS     1e-6f
#define WIN     36
#define BLOCK   12
#define NBINS   9
#define STR     (BLOCK /2)                    // 6
#define NX      ((WIN - BLOCK) / STR +1)      // 5
#define NY      ((WIN - BLOCK) / STR +1)      // 5
#define DIM     (NBINS * NX * NY)             // 225
// -------------------------------------------------

// --------------- 1) ref 이미지 HOG ---------------
void refHOG(const Mat& input, float* out_hog) {
    int x, y, xx, yy;
    int height = input.rows, width = input.cols;
    const int b_size = 3;
    float conv_x, conv_y;

    int mask_x[9] = { -1, 0, 1,  -1, 0, 1,  -1, 0, 1 };
    int mask_y[9] = { -1,-1,-1,   0, 0, 0,   1, 1, 1 };

	// magnitude, direction 메모리 할당
    float* magnitude = (float*)calloc(height * width, sizeof(float));
    unsigned char* dir = (unsigned char*)calloc(height * width, sizeof(unsigned char));

	// gradient 계산, magnitude, direction 저장
    for (y = 0; y < height; ++y) {
        for (x = 0; x < width; ++x) {
            conv_x = conv_y = 0.0f;
            for (yy = y - b_size / 2; yy <= y + b_size / 2; ++yy) {
                if (yy < 0 || yy >= height) continue;
                for (xx = x - b_size / 2; xx <= x + b_size / 2; ++xx) {
                    if (xx < 0 || xx >= width) continue;
                    int ky = yy - (y - b_size / 2);
                    int kx = xx - (x - b_size / 2);
                    float I = (float)input.at<uchar>(yy, xx) / 255.0f;
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

    // HOG 계산
    int bi = 0;
    for (int by = 0; by <= height - BLOCK; by += STR) {
        for (int bx = 0; bx <= width - BLOCK; bx += STR) {
            float hist[NBINS];
            for (int i = 0;i < NBINS;++i) hist[i] = 0.0f;
			
            // 히스토그램 계산
            for (y = by; y < by + BLOCK; ++y) {
                for (x = bx; x < bx + BLOCK; ++x) {
                    unsigned char b = dir[y * width + x];
                    hist[b] += magnitude[y * width + x];
                }
            }
			// L2 정규화
            float n2 = EPS;
            for (int i = 0; i < NBINS; ++i) n2 += hist[i] * hist[i];
            n2 = 1.0f / sqrt(n2);
            for (int i = 0; i < NBINS; ++i) out_hog[(bi * NBINS) + i] = hist[i] * n2;
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
                for (xx = x - b_size / 2; xx <= x + b_size / 2; ++xx) {
                    if (xx < 0 || xx >= width) continue;
                    int ky = yy - (y - b_size / 2);
                    int kx = xx - (x - b_size / 2);
                    float I = (float)input.at<uchar>(yy, xx) / 255.0f;
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
            float hist[NBINS];
            for (int i = 0;i < NBINS;++i) hist[i] = 0.0f;

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

    return exp(5.0f * score);
}

// --------------- 5) 얼굴 탐색 (70% 이상 마킹) ----------------
void searchFaces(const Mat& tar_img, const float* refHOG_descriptor) {
	// tar 이미지의 gradient magnitude, orientation 계산
    float* tar_mag;
    unsigned char* tar_dir;
    tar_mag_ori(tar_img, &tar_mag, &tar_dir);

	// 윈도우 HOG 계산용 메모리 할당
    float* testMag = (float*)malloc(WIN * WIN * sizeof(float));
    unsigned char* testDir = (unsigned char*)malloc(WIN * WIN * sizeof(unsigned char));
    float* tarHOG_descriptor = (float*)malloc(DIM * sizeof(float));

    float maxScore = -1.0f;
    Point maxPt(-1, -1);

    int height = tar_img.rows, width = tar_img.cols;
	int total = height * width; // 전체 픽셀 수
	float* scores = (float*)malloc(total * sizeof(float)); // 각 픽셀의 유사도 저장
	Point* points = (Point*)malloc(total * sizeof(Point)); // 각 픽셀의 좌표 저장

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

			// 윈도우 HOG 계산
            tarHOG(testMag, testDir, tarHOG_descriptor);
			// 유사도 계산
            float score = computeSimilarity(refHOG_descriptor, tarHOG_descriptor);

            int p = y * width + x;
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
	Mat vis; // 탐지 시각화
    cvtColor(tar_img, vis, COLOR_GRAY2BGR);
    int count = 0;

	// 기준 이상인 점들 마킹
    for (size_t i = 0; i < total; ++i) {
        if (scores[i] >= threshold) {
            circle(vis, points[i], 0, Scalar(0, 0, 255), FILLED);
            count++;
        }
    }

    // 유사도 맵 생성 (그레이스케일, 밝을수록 유사)
    Mat sim(height, width, CV_8UC1);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int p = y * width + x;
            float s = scores[p];
            float norm = (maxScore > EPS) ? (s / maxScore) : 0.0f;
            if (norm < 0.0f) norm = 0.0f;
            if (norm > 1.0f) norm = 1.0f;
            sim.at<uchar>(y, x) = (unsigned char)(norm * 255.0f + 0.5f);
        }
    }

    printf("Max similarity: %.6f at (%d, %d)\n", maxScore, maxPt.x, maxPt.y);
    printf("Threshold (70%% of max): %.6f\n", threshold);
    printf("Marked points: %d / %zu\n", count, total);

    imshow("detections", vis);
	imshow("similarity map", sim);
    waitKey(0);

    free(tar_mag);free(tar_dir);
    free(testMag);free(testDir);
    free(tarHOG_descriptor);
    free(scores);free(points);
}

// --------------- main ----------------
int main() {
    Mat ref_img = imread("C:/Users/shinj/Desktop/3-2/AEEDS/Data/face_ref.bmp", 0);
    Mat tar_img = imread("C:/Users/shinj/Desktop/3-2/AEEDS/Data/face_tar.bmp", 0);

	// 1) ref 이미지 HOG 계산
    float refHOG_descriptor[DIM];
    refHOG(ref_img, refHOG_descriptor);

	// 2) tar 이미지에서 얼굴 탐색
    searchFaces(tar_img, refHOG_descriptor);

    return 0;
}
