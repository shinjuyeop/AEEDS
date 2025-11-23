#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <cmath>
//#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>

using namespace cv;
using namespace cv::ml;
using namespace std;

// ------------------- 상수 정의 -------------------
#define COSINE  // 코사인 유사도 사용
#define PI      3.14159265358979323846f
#define EPS     1e-6f
#define WIN     128
#define BLOCK   (WIN / 4)
#define STR     (BLOCK / 2)
#define DIM     256 * 49
// -------------------------------------------------


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

    return exp(5.0f * score);
}

// LBP 이미지를 생성하는 함수
void makeLBPImage(const Mat& input, Mat& dst) {
    dst = Mat::zeros(input.rows, input.cols, CV_8UC1);
    for (int y = 1; y < input.rows - 1; y++) {
        for (int x = 1; x < input.cols - 1; x++) {
            uchar center = input.at<uchar>(y, x);
            int val = 0;
            if (input.at<uchar>(y - 1, x    ) > center) val |= (1 << 0);
            if (input.at<uchar>(y - 1, x + 1) > center) val |= (1 << 1);
            if (input.at<uchar>(y    , x + 1) > center) val |= (1 << 2);
            if (input.at<uchar>(y + 1, x + 1) > center) val |= (1 << 3);
            if (input.at<uchar>(y + 1, x    ) > center) val |= (1 << 4);
            if (input.at<uchar>(y + 1, x - 1) > center) val |= (1 << 5);
            if (input.at<uchar>(y    , x - 1) > center) val |= (1 << 6);
            if (input.at<uchar>(y - 1, x - 1) > center) val |= (1 << 7);
            dst.at<uchar>(y, x) = val;
        }
    }
}

/*
void LBPdescriptor(const Mat& input, float* LBPhist, int sx, int sy, int face_width, int face_height) {
    memset(LBPhist, 0, sizeof(float) * 256 * 49);

    int x, y, xx, yy;
    int height, width;
    int val, cnt = 0;
    float denomi;
	float* hist = (float*)calloc(256 * 49, sizeof(float));

	height = input.rows;
	width = input.cols;

    Mat inputGray;
	Mat LBPimage(face_height, face_width, CV_8UC1);
	Mat LBPimageResize(WIN, WIN, CV_8UC1);

	cvtColor(input, inputGray, COLOR_BGR2GRAY);

    for (y = sy + 1; y < sy + face_height - 1; y++) {
        for (x = sx + 1; x < sx + face_width - 1; x++) {
            uchar center = inputGray.at<uchar>(y, x);
            int val = 0;
            if (inputGray.at<uchar>(y - 1, x    ) > center) val |= (1 << 0);
            if (inputGray.at<uchar>(y - 1, x + 1) > center) val |= (1 << 1);
            if (inputGray.at<uchar>(y    , x + 1) > center) val |= (1 << 2);
            if (inputGray.at<uchar>(y + 1, x + 1) > center) val |= (1 << 3);
            if (inputGray.at<uchar>(y + 1, x    ) > center) val |= (1 << 4);
            if (inputGray.at<uchar>(y + 1, x - 1) > center) val |= (1 << 5);
            if (inputGray.at<uchar>(y    , x - 1) > center) val |= (1 << 6);
            if (inputGray.at<uchar>(y - 1, x - 1) > center) val |= (1 << 7);

            LBPimage.at<uchar>(y - sy, x - sx) = (uchar)val;
        }
    }
    
	resize(LBPimage, LBPimageResize, Size(WIN, WIN), 1);

	// LBP 히스토그램 계산
    for (y = 0; y <= WIN - BLOCK; y += STR) {
        for (x = 0; x <= WIN - BLOCK; x += STR) {
            // 히스토그램 초기화
            for (yy = 0; yy < 256; yy++) {
                hist[yy] = 0;
			}
            // 블록 내 픽셀 순회
            for (yy = y; yy < y + BLOCK; yy++) {
                for (xx = x; xx < x + BLOCK; xx++) {
					hist[LBPimageResize.at<uchar>(yy, xx)]++; // LBP 값에 해당하는 히스토그램 빈도 증가
                }
            }
			// 히스토그램 정규화
			denomi = 0;
            for (yy = 0; yy < 256; yy++) denomi += hist[yy] * hist[yy];
            denomi = sqrt(denomi) + EPS;

			for (yy = 0; yy < 256; yy++) {
				LBPhist[cnt * 256 + yy] = hist[yy] / denomi;
			}
            cnt++;
        }
	}
	free(hist);
}
*/

/*
Rect roi = Rect(sx, sy, face_width, face_height) & Rect(0, 0, input.cols, input.rows);
if (roi.width < 4 || roi.height < 4) return;

Mat face = input(roi);
Mat gray;
cvtColor(face, gray, COLOR_BGR2GRAY);
*/

void LBPdescriptor(const Mat input, float* LBPhist, int sx, int sy, int face_width, int face_height) {
    // 출력 히스토그램 초기화
    for (int i = 0; i < 256 * 49; i++) LBPhist[i] = 0;

    // 1) 얼굴부분만 자르고 그레이 변환
    // --- 범위 안전하게 보정 ---
    int x0 = max(0, sx);
    int y0 = max(0, sy);
    int w = min(face_width, input.cols - x0);
    int h = min(face_height, input.rows - y0);
    if (w < 4 || h < 4) return;   // 너무 작은 얼굴은 무시

    Mat face(h, w, CV_8UC3);      // 얼굴 이미지 (color)
    // --- 얼굴 픽셀 복사 ---
    for (int yy = 0; yy < h; yy++) {
        for (int xx = 0; xx < w; xx++) {
            face.at<Vec3b>(yy, xx) = input.at<Vec3b>(y0 + yy, x0 + xx);
        }
    }

    Mat gray;
    cvtColor(face, gray, COLOR_BGR2GRAY);


    // 2) 그레이 영상을 표준 크기(WIN x WIN)로 리사이즈
    Mat gray128;
    resize(gray, gray128, Size(WIN, WIN));

    // 3) 표준 크기에서 LBP 계산
    Mat lbp128(WIN, WIN, CV_8UC1);
    for (int y = 1; y < WIN - 1; ++y) {
        for (int x = 1; x < WIN - 1; ++x) {
            uchar c = gray128.at<uchar>(y, x);
            int v = 0;
            if (gray128.at<uchar>(y - 1, x    ) > c) v |= (1 << 0); // 12시
            if (gray128.at<uchar>(y - 1, x + 1) > c) v |= (1 << 1); // 1:30
            if (gray128.at<uchar>(y    , x + 1) > c) v |= (1 << 2); // 3시
            if (gray128.at<uchar>(y + 1, x + 1) > c) v |= (1 << 3); // 4:30
            if (gray128.at<uchar>(y + 1, x    ) > c) v |= (1 << 4); // 6시
            if (gray128.at<uchar>(y + 1, x - 1) > c) v |= (1 << 5); // 7:30
            if (gray128.at<uchar>(y    , x - 1) > c) v |= (1 << 6); // 9시
            if (gray128.at<uchar>(y - 1, x - 1) > c) v |= (1 << 7); // 10:30
            lbp128.at<uchar>(y, x) = (uchar)v;
        }
    }

    // 4) 블록(32) / 스트라이드(16)로 7x7 위치에서 256-bin 히스토그램
    float* hist = (float*)calloc(256, sizeof(float));
    int cnt = 0;

    for (int y = 0; y <= WIN - BLOCK; y += STR) {
        for (int x = 0; x <= WIN - BLOCK; x += STR) {
            for (int i = 0; i < 256; i++) hist[i] = 0;

            for (int yy = y; yy < y + BLOCK; ++yy) {
                for (int xx = x; xx < x + BLOCK; ++xx) {
                    hist[lbp128.at<uchar>(yy, xx)] += 1.0f;
                }
            }

            float denomi = 0.f;
            for (int b = 0; b < 256; ++b) denomi += hist[b] * hist[b];
            denomi = sqrtf(denomi) + EPS;

            for (int b = 0; b < 256; ++b)
                LBPhist[cnt * 256 + b] = hist[b] / denomi;

            cnt++;
        }
    }
    free(hist);
}

// --------------- main ----------------
void main() {

    int flag = 0;
    int cnt = 0;
    int k;
    float score;
    float th = 50; //40

    VideoCapture capture(0);
    Mat frame;
    if (!capture.isOpened()) {
        printf("Couldn’t open the web camera...\n");
        return;
    }

    CascadeClassifier cascade;
    cascade.load("C:/opencv/sources/data/lbpcascades/lbpcascade_frontalface.xml");
    vector<Rect> faces;

    float refLBPhist[256 * 49];
    float tarLBPhist[256 * 49];

    while (true) {
        capture >> frame;
        if (frame.empty()) break;;

        // 1. 얼굴 검출 (frame 사용)
        cascade.detectMultiScale(frame, faces, 1.1, 4, 0 | CV_HAAR_SCALE_IMAGE, Size(100, 100));

        // 2. enrollment(ref)
        if (faces.size() == 1 && flag == 0 && cnt > 30) {
            Point lb(faces[0].x + faces[0].width, faces[0].y + faces[0].height);
            Point tr(faces[0].x, faces[0].y);

			rectangle(frame, lb, tr, Scalar(0, 255, 0), 3, 8, 0);
            
			// face verification
            LBPdescriptor(frame, refLBPhist, faces[0].x, faces[0].y, faces[0].width, faces[0].height);
            flag = 1;
        }

        // 3. verification(tar)
        if (faces.size() > 0 && flag != 0 && cnt > 30) {
            for (k = 0; k < faces.size(); k++) {
                LBPdescriptor(frame, tarLBPhist, faces[k].x, faces[k].y, faces[k].width, faces[k].height);
                score = computeSimilarity(refLBPhist, tarLBPhist);
                printf("score : %f\n", score);

                Point lb(faces[k].x + faces[k].width, faces[k].y + faces[k].height);
                Point tr(faces[k].x, faces[k].y);
                if (score > th) {
                    rectangle(frame, lb, tr, Scalar(0, 255, 0), 3, 8, 0);
                } else {
                    rectangle(frame, lb, tr, Scalar(0, 0, 255), 3, 8, 0);
                }
            }
        }

        imshow("Face", frame);
        if (waitKey(30) >= 0) break;
        cnt++;
    }
    
    /*
    // LBP 테스트: 그레이스케일 변환 후 LBP 이미지 생성 및 저장
    Mat img = imread("C:/Users/shinj/Desktop/3-2/AEEDS/Data/test.jpg", 1); // image read (color)
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    Mat lbpImg;
    makeLBPImage(gray, lbpImg);
    imwrite("lbp_result.png", lbpImg);
    imshow("LBP Image", lbpImg);
    waitKey(0);

	// 얼굴 검출
    cascade.detectMultiScale(img, faces, 1.1, 4, 0 | CV_HAAR_SCALE_IMAGE, Size(10, 10)); 
    int y;
    for (y = 0; y < faces.size(); y++)
    {
        Point lb(faces[y].x + faces[y].width, faces[y].y + faces[y].height);
        Point tr(faces[y].x, faces[y].y);
        rectangle(img, lb, tr, Scalar(0, 255, 0), 3, 8, 0);
    }
    imshow("Face", img);
    imwrite("Face.bmp", img);
    waitKey(50000);
    */
}