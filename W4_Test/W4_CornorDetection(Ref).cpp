#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;

// 입력: 8-bit 단일 채널 그레이스케일 Mat
// 출력: 해리스 코너가 원으로 표시된 컬러 이미지 저장("harris_corners.png")
// 구현 포인트:
//  - 3x3 Prewitt로 Ix, Iy 계산 (네 코드 스타일)
//  - 윈도우 합으로 lxlx, lyly, lxly 누적 → R = det - k * tr^2
//  - min/max 기반 정규화는 시각화(R맵)용 옵션
//  - 임계비율(thresh_ratio)로 코너 추출, 원 그리기
//

void HarrisCornerDetectAndDraw(Mat input)
{
    // ----- 파라미터 -----
    const int b_size = 3;               // 프리윗 커널 크기
    const int win_size = 3;             // Harris 윈도 크기(3, 5 등)
    const float k = 0.04f;              // Harris k (보통 0.04~0.06)
    const float thresh_ratio = 0.40f;   // R 임계 비율(0~1), 높일수록 코너 적게 찍힘
    const int circle_radius = 8;        // 원 반지름
    const int circle_thick = 1;         // 원 두께

    // ----- 기본 체크 -----
    CV_Assert(input.type() == CV_8UC1);
    const int height = input.rows;
    const int width = input.cols;

    // ----- 프리윗 마스크 -----
    const int size = 9;
    int mask_x[size] = { -1, 0, 1,
                         -1, 0, 1,
                         -1, 0, 1 };
    int mask_y[size] = { -1, -1, -1,
                          0,  0,  0,
                          1,  1,  1 };

    // ----- Ix, Iy 메모리 -----
    float* Ix = (float*)calloc(height * width, sizeof(float));
    float* Iy = (float*)calloc(height * width, sizeof(float));
    if (!Ix || !Iy) {
        fprintf(stderr, "메모리 할당 실패: Ix/Iy\n");
        free(Ix); free(Iy);
        return;
    }

    // ----- Gradient 계산 -----
    {
        int x, y, xx, yy;
        float conv_x, conv_y;

        for (y = 0; y < height; ++y) {
            for (x = 0; x < width; ++x) {

                conv_x = 0.f;
                conv_y = 0.f;

                for (yy = y - b_size / 2; yy <= y + b_size / 2; ++yy) {
                    for (xx = x - b_size / 2; xx <= x + b_size / 2; ++xx) {

                        if (yy >= 0 && yy < height && xx >= 0 && xx < width) {
                            // (dy,dx) → 1D 인덱스
                            int kx = (yy - (y - 1)) * b_size + (xx - (x - 1));
                            if (kx < 0 || kx >= size) continue; // 경계 방어

                            // 0~1 스케일로 읽기
                            float pix = (float)input.at<uchar>(yy, xx) / 255.0f;

                            conv_x += mask_x[kx] * pix;
                            conv_y += mask_y[kx] * pix;
                        }
                    }
                }

                Ix[y * width + x] = conv_x;
                Iy[y * width + x] = conv_y;
            }
        }
    }

    // ----- Harris R 계산 -----
    float* R = (float*)calloc(height * width, sizeof(float));
    if (!R) {
        fprintf(stderr, "메모리 할당 실패: R\n");
        free(Ix); free(Iy);
        return;
    }

    float minR = FLT_MAX;
    float maxR = -FLT_MAX;

    {
        int x, y, xx, yy;
        float lxlx, lyly, lxly;
        float det, tr;

        for (y = 0; y < height; ++y) {
            for (x = 0; x < width; ++x) {

                lxlx = 0.f;
                lyly = 0.f;
                lxly = 0.f;

                for (yy = y - win_size / 2; yy <= y + win_size / 2; ++yy) {
                    for (xx = x - win_size / 2; xx <= x + win_size / 2; ++xx) {
						if (yy < 0 || yy >= height || xx < 0 || xx >= width) continue; // 경계 방어

                        float ix = Ix[yy * width + xx];
                        float iy = Iy[yy * width + xx];

                        lxlx += ix * ix;  // Sxx
                        lyly += iy * iy;  // Syy
                        lxly += ix * iy;  // Sxy
                    }
                }

                det = lxlx * lyly - lxly * lxly;
                tr = lxlx + lyly;

                R[y * width + x] = det - k * tr * tr;

                if (R[y * width + x] < minR) minR = R[y * width + x];
                if (R[y * width + x] > maxR) maxR = R[y * width + x];
            }
        }
    }

    // ----- 코너 임계값 설정 -----
    float thresh = minR + (maxR - minR) * thresh_ratio;

    // ----- 시각화 준비 (컬러) -----
    Mat color;
    cvtColor(input, color, COLOR_GRAY2BGR);

    // ----- 코너 찍기(원) -----
    {
        int x, y;
        Scalar red(0, 0, 255);

        for (y = 1; y < height - 1; ++y) {
            for (x = 1; x < width - 1; ++x) {
                if (R[y * width + x] >= thresh) {
                    circle(color, Point(x, y), circle_radius, red, circle_thick, 8, 0);
                }
            }
        }
    }

    imwrite("harris_corners.png", color);

    // (옵션) R 맵 저장(정규화)
    {
        Mat Rviz(height, width, CV_8UC1, Scalar(0));
        float denom = (maxR - minR);
        if (denom < 1e-12f) denom = 1.f; // 안전 가드

        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                float v = 255.f * (R[y * width + x] - minR) / denom; // [0,255]
                if (v < 0.f) v = 0.f;
                if (v > 255.f) v = 255.f;
                Rviz.at<uchar>(y, x) = (uchar)(v + 0.5f);
            }
        }
        imwrite("harris_Rmap.png", Rviz);
    }

    // ----- 메모리 해제 -----
    free(Ix);
    free(Iy);
    free(R);
}

int main()
{
    // 그레이스케일 입력
    Mat img = imread("C:/Users/shinj/Desktop/3-2/AEEDS/Data/checkerboard.png", IMREAD_GRAYSCALE);
    if (img.empty()) {
        fprintf(stderr, "이미지 로드 실패. 경로 확인 필요\n");
        return -1;
    }

    HarrisCornerDetectAndDraw(img);

    printf("Done. Output files: harris_corners.png, harris_Rmap.png\n");
    return 0;
}
