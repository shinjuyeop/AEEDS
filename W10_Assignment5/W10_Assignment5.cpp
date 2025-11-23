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
#define WIN     128
#define BLOCK   (WIN / 4)          // 32
#define STR     (BLOCK / 2)        // 16

// Uniform LBP (8-neighbor) : 59-bin
#define ULBP_BIN 59
#define DIM      (ULBP_BIN * 49)   // 7x7 블록 * 59-bin
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
            if (gray128.at<uchar>(y - 1, x    ) > c) v |= (1 << 0); // 12시
            if (gray128.at<uchar>(y - 1, x + 1) > c) v |= (1 << 1); // 1:30
            if (gray128.at<uchar>(y    , x + 1) > c) v |= (1 << 2); // 3시
            if (gray128.at<uchar>(y + 1, x + 1) > c) v |= (1 << 3); // 4:30
            if (gray128.at<uchar>(y + 1, x    ) > c) v |= (1 << 4); // 6시
            if (gray128.at<uchar>(y + 1, x - 1) > c) v |= (1 << 5); // 7:30
            if (gray128.at<uchar>(y    , x - 1) > c) v |= (1 << 6); // 9시
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

// --------------- main ----------------
int main()
{
    int flag = 0;
    int cnt = 0;
    int k;
    float score;
    float th = 50.0f; // threshold (exp 스케일 이후 값)

    VideoCapture capture(0);
    Mat frame;
    if (!capture.isOpened()) {
        printf("Couldn’t open the web camera...\n");
        return -1;
    }

    CascadeClassifier cascade;
    cascade.load("C:/opencv/sources/data/lbpcascades/lbpcascade_frontalface.xml");
    vector<Rect> faces;

    float refLBPhist[DIM];
    float tarLBPhist[DIM];

    while (true) {
        capture >> frame;
        if (frame.empty()) break;

        // 1. 얼굴 검출
        cascade.detectMultiScale(frame, faces, 1.1, 4, 0 | CV_HAAR_SCALE_IMAGE, Size(100, 100));

        // 2. enrollment (ref)
        if (faces.size() == 1 && flag == 0 && cnt > 30) {
            Point lb(faces[0].x + faces[0].width,
                faces[0].y + faces[0].height);
            Point tr(faces[0].x, faces[0].y);

            rectangle(frame, lb, tr, Scalar(0, 255, 0), 3, 8, 0);

            // face verification reference feature 추출
            LBPdescriptor(frame, refLBPhist,
                faces[0].x, faces[0].y,
                faces[0].width, faces[0].height);
            flag = 1;
        }

        // 3. verification (tar)
        if (!faces.empty() && flag != 0 && cnt > 30) {
            for (k = 0; k < (int)faces.size(); ++k) {
                LBPdescriptor(frame, tarLBPhist,
                    faces[k].x, faces[k].y,
                    faces[k].width, faces[k].height);

                score = computeSimilarity(refLBPhist, tarLBPhist);
                printf("score : %f\n", score);

                Point lb(faces[k].x + faces[k].width,
                    faces[k].y + faces[k].height);
                Point tr(faces[k].x, faces[k].y);

                if (score > th)
                    rectangle(frame, lb, tr, Scalar(0, 255, 0), 3, 8, 0);
                else
                    rectangle(frame, lb, tr, Scalar(0, 0, 255), 3, 8, 0);
            }
        }

        imshow("Face", frame);
        if (waitKey(30) >= 0) break;
        cnt++;
    }

    return 0;
}
