#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define PI 3.14159265358979323846f

using namespace cv;

// Edge + HOG (블록 16, 이동간격 block/2, 9-bin)
void EdgeDetection_HOG(const Mat input, float** out_hog, int* out_dim,
    int block, int nbins, float eps)
{
    int x, y, xx, yy;
    int height = input.rows, width = input.cols;
    const int b_size = 3; // 3x3
    float conv_x, conv_y;

    // 마스크
    int mask_x[9] = { -1, 0, 1,  -1, 0, 1,  -1, 0, 1 };
    int mask_y[9] = { -1,-1,-1,   0, 0, 0,   1, 1, 1 };

    // gradient magnitude / orientation 메모리
    float* val = (float*)calloc(height * width, sizeof(float));
    unsigned char* bin_img = (unsigned char*)calloc(height * width, sizeof(unsigned char));


    // 1) Gradient magnitude, orientation 저장
    for (y = 0; y < height; ++y) {
        for (x = 0; x < width; ++x) {
            conv_x = conv_y = 0.0f;
            // 3x3 컨볼루션(경계는 무시)
            for (yy = y - b_size / 2; yy <= y + b_size / 2; ++yy) {
                if (yy < 0 || yy >= height) continue;
                const unsigned char* prow = input.ptr<unsigned char>(yy);
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
            if (theta < 0.0f) theta += 180.0f;         // [0,180)
            int b = (int)(theta / (180.0f / nbins));   // 9 bin (20도 간격)
            if (b >= nbins) b = nbins - 1;

            val[y * width + x] = M;
            bin_img[y * width + x] = (unsigned char)b;
        }
    }

    // 2) 블록/슬라이딩
    int nx = (width - block) / (block / 2) + 1; // x 방향 블록 개수
    int ny = (height - block) / (block / 2) + 1; // y 방향 블록 개수

    int blocks = nx * ny; // 전체 블록 개수
    int DIM = nbins * blocks; // 전체 HOG 차원
    float* whole = (float*)malloc(sizeof(float) * DIM);

    int bi = 0; // 블록 인덱스
    for (int by = 0; by <= height - block; by += block / 2) {
        for (int bx = 0; bx <= width - block; bx += block / 2) {
            float hist[9]; // 블록 단위 히스토그램
            for (int i = 0;i < nbins;++i) hist[i] = 0.0f; // 초기화

            // 2-1) 누적 (hist[bin]+=val)
            for (y = by; y < by + block; ++y) {
                for (x = bx; x < bx + block; ++x) {
                    unsigned char b = bin_img[y * width + x];
                    hist[b] += val[y * width + x];
                }
            }

            // 2-2) 블록별 L2 정규화
            float n2 = eps;
            for (int i = 0;i < nbins;++i) n2 += hist[i] * hist[i];
            n2 = 1.0f / sqrt
            (n2);

            // 2-3) WholeHist에 연결
            for (int i = 0;i < nbins;++i)
                whole[(bi * nbins) + i] = hist[i] * n2;

            bi += 1;
        }
    }

    // out
    *out_hog = whole; // HOG 벡터
    *out_dim = DIM; // HOG 차원

    // free
    free(val);
    free(bin_img);
}

// 비교(L1/L2)
void compare_hog(const float* a, const float* b, int n, double* L1, double* L2) {
    long double l1 = 0.0L, l2 = 0.0L;
    for (int i = 0;i < n;++i) {
        long double d = (long double)a[i] - (long double)b[i];
        l1 += fabsl(d);
        l2 += d * d;
    }
    *L1 = (double)l1;
    *L2 = sqrt((double)l2);
}


void main()
{
	VideoCapture capture(0);
	Mat frame;
	if (!capture.isOpened()) {
		printf("Couldn’t open the web camera...\n");
		return;
	}
	while (true) {
		capture >> frame;
        // 여기
		imshow("Video", frame);
		if (waitKey(30) >= 0) break;
	}
}