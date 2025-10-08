#include "cornerDetector.h"
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>

using namespace cv;

void HarrisCornerDetector(Mat& frame, Mat& cornerMap, int thresholdVal, int radius, int winSize, float k) {
    if (frame.empty()) return;
    if (thresholdVal < 0) thresholdVal = 0; if (thresholdVal > 255) thresholdVal = 255;
    if (radius < 1) radius = 1; if (winSize != 3 && winSize != 5) winSize = 3; if (k <= 0.f) k = 0.04f;

    Mat gray;
    if (frame.channels() == 3) cvtColor(frame, gray, COLOR_BGR2GRAY); else gray = frame;

    const int width = gray.cols;
    const int height = gray.rows;
    const int rWin = winSize / 2;

    // Prewitt gradient (could be Sobel for more robustness)
    int xmask[3][3] = { {-1,0,1},{-1,0,1},{-1,0,1} };
    int ymask[3][3] = { {-1,-1,-1},{0,0,0},{1,1,1} };

    float* Ix = (float*)calloc(width * height, sizeof(float));
    float* Iy = (float*)calloc(width * height, sizeof(float));
    float* R  = (float*)calloc(width * height, sizeof(float));
    if (!Ix || !Iy || !R) { fprintf(stderr, "메모리 할당 실패\n"); free(Ix); free(Iy); free(R); return; }

    // 1) Gradients
    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            int gx = 0, gy = 0;
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) {
                    uchar pix = gray.at<uchar>(y + dy, x + dx);
                    gx += xmask[dy + 1][dx + 1] * (int)pix;
                    gy += ymask[dy + 1][dx + 1] * (int)pix;
                }
            Ix[y * width + x] = (float)gx;
            Iy[y * width + x] = (float)gy;
        }
    }

    // Optional: Gaussian smoothing of Ixx,Iyy,Ixy could be added for stability.

    float minR = FLT_MAX, maxR = -FLT_MAX;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float Sxx = 0.f, Syy = 0.f, Sxy = 0.f;
            int y0 = std::max(0, y - rWin);
            int y1 = std::min(height - 1, y + rWin);
            int x0 = std::max(0, x - rWin);
            int x1 = std::min(width - 1, x + rWin);
            for (int yy = y0; yy <= y1; ++yy) {
                int row = yy * width;
                for (int xx = x0; xx <= x1; ++xx) {
                    float ixv = Ix[row + xx];
                    float iyv = Iy[row + xx];
                    Sxx += ixv * ixv;
                    Syy += iyv * iyv;
                    Sxy += ixv * iyv;
                }
            }
            float det = Sxx * Syy - Sxy * Sxy;
            float tr  = Sxx + Syy;
            float resp = det - k * tr * tr;
            R[y * width + x] = resp;
            if (resp < minR) minR = resp;
            if (resp > maxR) maxR = resp;
        }
    }

    // 3) Normalize to 0~255 cornerness map
    cornerMap.create(height, width, CV_8UC1);
    cornerMap.setTo(Scalar(0));
    if (maxR > minR) {
        float denom = maxR - minR;
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                float v = 255.f * (R[y * width + x] - minR) / denom;
                v = std::min(255.f, std::max(0.f, v));
                cornerMap.at<uchar>(y, x) = (uchar)(v + 0.5f);
            }
        }
    }

    // 4) Draw local maxima above threshold
    for (int y = 1; y < height - 1; ++y) {
        const uchar* prev = cornerMap.ptr<uchar>(y - 1);
        const uchar* curr = cornerMap.ptr<uchar>(y);
        const uchar* next = cornerMap.ptr<uchar>(y + 1);
        for (int x = 1; x < width - 1; ++x) {
            uchar v = curr[x];
            if (v < thresholdVal) continue;
            if (v < curr[x - 1] || v < curr[x + 1] ||
                v < prev[x - 1] || v < prev[x] || v < prev[x + 1] ||
                v < next[x - 1] || v < next[x] || v < next[x + 1]) continue;
            circle(frame, Point(x, y), radius, Scalar(0,255,0), 1, LINE_AA);
        }
    }

    free(Ix); free(Iy); free(R);
}