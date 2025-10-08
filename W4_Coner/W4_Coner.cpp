/*
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <math.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;

int main()
{
    // 입력 이미지 (그레이스케일)
    Mat imgGray = imread(
        "C:/Users/shinj/Desktop/3-2/AEEDS/Data/checkerboard.png",
        IMREAD_GRAYSCALE);

    if (imgGray.empty()) {
        fprintf(stderr, "이미지 로드 실패\n");
        return -1;
    }

    const int width = imgGray.cols;
    const int height = imgGray.rows;

    // 3x3 gradient mask (Prewitt)
    int xmask[3][3] = { {-1, 0, 1}, {-1, 0, 1}, {-1, 0, 1} };
    int ymask[3][3] = { {-1, -1, -1}, {0, 0, 0}, {1, 1, 1} };

    // Ix, Iy 저장 버퍼
	float* Ix = (float*)calloc(width * height, sizeof(float)); // Gradient X
	float* Iy = (float*)calloc(width * height, sizeof(float)); // Gradient Y
	float* R = (float*)calloc(width * height, sizeof(float));  // Harris response

    if (Ix == NULL || Iy == NULL || R == NULL) {
        fprintf(stderr, "메모리 할당 실패\n");
        free(Ix); free(Iy); free(R);
        return -1;
    }

    // 1) 이미지 그라디언트 계산 (경계 1픽셀 제외)
    int y, x, dy, dx;
    for (y = 1; y < height - 1; ++y) {
        for (x = 1; x < width - 1; ++x) {
            int gx = 0, gy = 0;
            for (dy = -1; dy <= 1; ++dy) {
                for (dx = -1; dx <= 1; ++dx) {
                    unsigned char pix = imgGray.data[(y + dy) * width + (x + dx)];
                    gx += xmask[dy + 1][dx + 1] * (int)pix;
                    gy += ymask[dy + 1][dx + 1] * (int)pix;
                }
            }
            Ix[y * width + x] = (float)gx;
            Iy[y * width + x] = (float)gy;
        }
    }

    // 2) Harris Corner: 윈도우 합으로 Sxx, Syy, Sxy 계산 후 R = det - k*(trace^2)
    const int win_size = 3;
    const int r = win_size / 2;
    const float k = 0.04f;

    float minR = FLT_MAX;
    float maxR = -FLT_MAX;
    
    for (y = 0; y < height; ++y) {
        for (x = 0; x < width; ++x) {
            float Sxx = 0.f, Syy = 0.f, Sxy = 0.f;

            // 윈도우 경계 처리
            int y0 = (y - r > 0) ? (y - r) : 0;
            int y1 = (y + r < height - 1) ? (y + r) : (height - 1);
            int x0 = (x - r > 0) ? (x - r) : 0;
            int x1 = (x + r < width - 1) ? (x + r) : (width - 1);

			// 윈도우 내 합 계산
            int yy, xx;
            for (yy = y0; yy <= y1; ++yy) {
                int row = yy * width;
                for (xx = x0; xx <= x1; ++xx) {
                    float ix = Ix[row + xx];
                    float iy = Iy[row + xx];
                    Sxx += ix * ix;  // IxIx
                    Syy += iy * iy;  // IyIy
                    Sxy += ix * iy;  // IxIy
                }
            }
			// det = ad - bc
			float det = Sxx * Syy - Sxy * Sxy;  // det(M)
			float tr = Sxx + Syy;               // trace(M)
			float rVal = det - k * tr * tr;     // Harris 코너 응답
			R[y * width + x] = rVal;            // R 저장
            if (rVal < minR) minR = rVal;
            if (rVal > maxR) maxR = rVal;
        }
    }

    // 3) 0~255 정규화 및 저장
    Mat out(height, width, CV_8UC1);
    for (int i = 0; i < width * height; ++i) {
        out.data[i] = 0;
    }
    if (maxR > minR) {
		float denom = maxR - minR; // 정규화 분모
        for (y = 0; y < height; ++y) {
            for (x = 0; x < width; ++x) {
				float v = 255.f * (R[y * width + x] - minR) / denom; // 0~255 정규화
                if (v < 0.f) v = 0.f;
                if (v > 255.f) v = 255.f;
                out.data[y * width + x] = (unsigned char)(v + 0.5f); // 반올림 후 저장
            }
        }
    }

    imwrite("Corner.bmp", out);

    // 메모리 해제
    free(Ix);
    free(Iy);
    free(R);

    return 0;
}
*/

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <math.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;

int main()
{
    // 0) 입력 (그레이스케일)
    Mat imgGray = imread(
        "C:/Users/shinj/Desktop/3-2/AEEDS/Data/checkerboard.png",
        IMREAD_GRAYSCALE);

    if (imgGray.empty()) {
        fprintf(stderr, "이미지 로드 실패\n");
        return -1;
    }

    const int width = imgGray.cols;
    const int height = imgGray.rows;

	// 1) 3x3 Prewitt gradient
    int xmask[3][3] = { {-1, 0, 1}, {-1, 0, 1}, {-1, 0, 1} };
    int ymask[3][3] = { {-1,-1,-1}, { 0, 0, 0}, { 1, 1, 1} };

	float* Ix = (float*)calloc(width * height, sizeof(float));  // Gradient X
	float* Iy = (float*)calloc(width * height, sizeof(float));  // Gradient Y
	float* R = (float*)calloc(width * height, sizeof(float));   // Harris response
    if (!Ix || !Iy || !R) {
        fprintf(stderr, "메모리 할당 실패\n");
        free(Ix); free(Iy); free(R);
        return -1;
    }

    int x, y, dx, dy;
    for (y = 1; y < height - 1; ++y) {
        for (x = 1; x < width - 1; ++x) {
            int gx = 0, gy = 0;
            for (dy = -1; dy <= 1; ++dy) {
                for (dx = -1; dx <= 1; ++dx) {
                    unsigned char pix = imgGray.data[(y + dy) * width + (x + dx)];
                    gx += xmask[dy + 1][dx + 1] * (int)pix;
                    gy += ymask[dy + 1][dx + 1] * (int)pix;
                }
            }
            Ix[y * width + x] = (float)gx;
            Iy[y * width + x] = (float)gy;
        }
    }

    // 2) Harris R = det - k*(trace^2)
    const int   win_size = 3;
    const int   r = win_size / 2;
    const float k = 0.04f;

    float minR = FLT_MAX, maxR = -FLT_MAX;

    for (y = 0; y < height; ++y) {
        for (x = 0; x < width; ++x) {
            float Sxx = 0.f, Syy = 0.f, Sxy = 0.f;

            int y0 = (y - r > 0) ? (y - r) : 0;
            int y1 = (y + r < height - 1) ? (y + r) : (height - 1);
            int x0 = (x - r > 0) ? (x - r) : 0;
            int x1 = (x + r < width - 1) ? (x + r) : (width - 1);

            for (int yy = y0; yy <= y1; ++yy) {
                int row = yy * width;
                for (int xx = x0; xx <= x1; ++xx) {
                    float ix = Ix[row + xx];
                    float iy = Iy[row + xx];
                    Sxx += ix * ix;   // Ix^2
                    Syy += iy * iy;   // Iy^2
                    Sxy += ix * iy;   // Ix*Iy
                }
            }

            float det = Sxx * Syy - Sxy * Sxy;
            float tr = Sxx + Syy;
            float rVal = det - k * tr * tr;
            R[y * width + x] = rVal;

            if (rVal < minR) minR = rVal;
            if (rVal > maxR) maxR = rVal;
        }
    }

    // 3) R 맵 정규화 (시각화용) + 저장
    Mat Rviz(height, width, CV_8UC1, Scalar(0));
    if (maxR > minR) {
        float denom = maxR - minR;
        for (y = 0; y < height; ++y)
            for (x = 0; x < width; ++x) {
                float v = 255.f * (R[y * width + x] - minR) / denom;
                if (v < 0.f)   v = 0.f;
                if (v > 255.f) v = 255.f;
                Rviz.data[y * width + x] = (unsigned char)(v + 0.5f);
            }
    }
    imwrite("Corner.bmp", Rviz);

    // 4) 임계값 기반 circle 시각화
    Mat resultImg;
    cvtColor(imgGray, resultImg, COLOR_GRAY2BGR);

    // 임계값: maxR의 일정 비율
    const float thresh_ratio = 0.4f;
    const float thresh = minR + (maxR - minR) * thresh_ratio;

    const int radius = 8;
    Scalar c; c.val[0] = 0; c.val[1] = 0; c.val[2] = 255; // 빨간색(BGR)

    for (y = 1; y < height - 1; ++y) {
        for (x = 1; x < width - 1; ++x) {
            if (R[y * width + x] >= thresh) {
                Point pCenter; pCenter.x = x; pCenter.y = y;
                circle(resultImg, pCenter, radius, c, 1, 8, 0);
            }
        }
    }

    imwrite("Corner_with_circles.png", resultImg);

    // 메모리 해제
    free(Ix);
    free(Iy);
    free(R);

    return 0;
}
