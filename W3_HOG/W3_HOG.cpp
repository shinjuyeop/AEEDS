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
	float conv_x, conv_y; // convolution 결과

    // 마스크
    int mask_x[9] = { -1, 0, 1,  -1, 0, 1,  -1, 0, 1 };
    int mask_y[9] = { -1,-1,-1,   0, 0, 0,   1, 1, 1 };

    // gradient magnitude / orientation 메모리
    float* val = (float*)calloc(height * width, sizeof(float));
    unsigned char* bin_img = (unsigned char*)calloc(height * width, sizeof(unsigned char));


    // 1) Gradient magnitude, orientation 저장
	// 전체를 순회하면서 3x3 컨볼루션
    // 
    for (y = 0; y < height; ++y) {
        for (x = 0; x < width; ++x) {
            conv_x = conv_y = 0.0f;
            // 3x3 컨볼루션(경계는 무시)
            for (yy = y - b_size / 2; yy <= y + b_size / 2; ++yy) {
                if (yy < 0 || yy >= height) continue;
                const unsigned char* prow = input.data + yy * input.step; // 행 포인터
                for (xx = x - b_size / 2; xx <= x + b_size / 2; ++xx) {
                    if (xx < 0 || xx >= width) continue;
					int ky = yy - (y - b_size / 2); // mask y 인덱스
					int kx = xx - (x - b_size / 2); // mask x 인덱스
                    float I = (float)prow[xx] / 255.0f; // xx에 대한 픽셀 값
					conv_x += mask_x[ky * b_size + kx] * I; // x 방향 컨볼루션
					conv_y += mask_y[ky * b_size + kx] * I; // y 방향 컨볼루션
                }
            }
			float M = sqrt(conv_x * conv_x + conv_y * conv_y);  // magnitude
			float theta = atan2f(conv_y, conv_x) * 180.0f / PI; // degree
            if (theta < 0.0f) theta += 180.0f;                  // [0,180)
            int b = (int)(theta / (180.0f / nbins));            // 9 bin (20도 간격)
			if (b >= nbins) b = nbins - 1;                      // 180도는 160도로 처리

			val[y * width + x] = M;                     // gradient magnitude
			bin_img[y * width + x] = (unsigned char)b;  // gradient orientation bin (0~8)
        }
    }

    // 2) 블록/슬라이딩
	int nx = (width - block) / (block / 2) + 1; // x 방향 블록 개수
	int ny = (height - block) / (block / 2) + 1; // y 방향 블록 개수

	int blocks = nx * ny; // 전체 블록 개수 (15 * 7 = 105)
	int DIM = nbins * blocks; // 전체 HOG 차원 (9 * 105 = 945)
	float* whole = (float*)malloc(sizeof(float) * DIM); // 전체 HOG 벡터

    int bi = 0; // 블록 인덱스

	// 슬라이딩 윈도우 (블록 크기 16, 이동간격 8)
    for (int by = 0; by <= height - block; by += block/2){
        for (int bx = 0; bx <= width - block; bx += block/2) {
			float hist[9]; // 블록 단위 히스토그램
			for (int i = 0;i < nbins;++i) hist[i] = 0.0f; // 초기화

            // 2-1) 누적 (hist[bin]+=val)
            for (y = by; y < by + block; ++y) {
                for (x = bx; x < bx + block; ++x) {
					unsigned char b = bin_img[y * width + x]; // bin index(0~8)
					hist[b] += val[y * width + x]; // 누적 (gradient magnitude)
                }
            }

			// 2-2) 블록별 L2 정규화
            float n2 = eps;
            for (int i = 0;i < nbins;++i) n2 += hist[i] * hist[i];
            n2 = 1.0f / sqrt
            (n2);

            // 2-3) WholeHist에 연결
            for (int i = 0;i < nbins;++i)
				whole[(bi * nbins) + i] = hist[i] * n2; // 정규화된 히스토그램

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
		l1 += fabsl(d); // 절대값
		l2 += d * d;    // 제곱값
    }
    *L1 = (double)l1;
    *L2 = sqrt((double)l2);
}

int main() {
    const char* ref_path = "C:/Users/shinj/Desktop/3-2/AEEDS/Data/LectureNote_03.bmp";
    const char* c1_path = "C:/Users/shinj/Desktop/3-2/AEEDS/Data/compare1.bmp";
    const char* c2_path = "C:/Users/shinj/Desktop/3-2/AEEDS/Data/compare2.bmp";
	const char* c3_path = "C:/Users/shinj/Desktop/3-2/AEEDS/Data/leaf_copy.bmp";
	const char* c4_path = "C:/Users/shinj/Desktop/3-2/AEEDS/Data/checkerboard_copy.bmp";
	const char* c5_path = "C:/Users/shinj/Desktop/3-2/AEEDS/Data/wave.bmp";
	const char* c6_path = "C:/Users/shinj/Desktop/3-2/AEEDS/Data/LectureNote_03_noise.bmp";


    Mat ref = imread(ref_path, IMREAD_GRAYSCALE);
    Mat c1 = imread(c1_path, IMREAD_GRAYSCALE);
    Mat c2 = imread(c2_path, IMREAD_GRAYSCALE);
    Mat c3 = imread(c3_path, IMREAD_GRAYSCALE);
    Mat c4 = imread(c4_path, IMREAD_GRAYSCALE);
    Mat c5 = imread(c5_path, IMREAD_GRAYSCALE);
    Mat c6 = imread(c6_path, IMREAD_GRAYSCALE);

	// 파라미터
	const int B = 16, NB = 9; // 블록 크기, 히스토그램 빈 개수
	const float EPS = 1e-6f; // 정규화 상수

    float* hog_ref = NULL, * hog_c1 = NULL, * hog_c2 = NULL,
        * hog_c3 = NULL, * hog_c4 = NULL, * hog_c5 = NULL, * hog_c6 = NULL;

    int dim_ref = 0, dim_c1 = 0, dim_c2 = 0, dim_c3 = 0, dim_c4 = 0, dim_c5 = 0, dim_c6 = 0;

    EdgeDetection_HOG(ref, &hog_ref, &dim_ref, B, NB, EPS);
    EdgeDetection_HOG(c1, &hog_c1, &dim_c1, B, NB, EPS);
    EdgeDetection_HOG(c2, &hog_c2, &dim_c2, B, NB, EPS);
    EdgeDetection_HOG(c3, &hog_c3, &dim_c3, B, NB, EPS);
    EdgeDetection_HOG(c4, &hog_c4, &dim_c4, B, NB, EPS);
    EdgeDetection_HOG(c5, &hog_c5, &dim_c5, B, NB, EPS);
    EdgeDetection_HOG(c6, &hog_c6, &dim_c6, B, NB, EPS);

    double L1_1, L2_1, L1_2, L2_2, L1_3, L2_3, L1_4, L2_4, L1_5, L2_5, L1_6, L2_6;
    compare_hog(hog_ref, hog_c1, dim_ref, &L1_1, &L2_1);
    compare_hog(hog_ref, hog_c2, dim_ref, &L1_2, &L2_2);
    compare_hog(hog_ref, hog_c3, dim_ref, &L1_3, &L2_3);
    compare_hog(hog_ref, hog_c4, dim_ref, &L1_4, &L2_4);
    compare_hog(hog_ref, hog_c5, dim_ref, &L1_5, &L2_5);
    compare_hog(hog_ref, hog_c6, dim_ref, &L1_6, &L2_6);

    printf("dim = %d (=%d bins × %d blocks)\n", dim_ref, NB, dim_ref / NB);
    printf("[ref vs compare1] L1=%.6f  L2=%.6f\n", L1_1, L2_1);
    printf("[ref vs compare2] L1=%.6f  L2=%.6f\n", L1_2, L2_2);
    printf("[ref vs leaf_copy] L1=%.6f  L2=%.6f\n", L1_3, L2_3);
    printf("[ref vs checkerboard_copy] L1=%.6f  L2=%.6f\n", L1_4, L2_4);
    printf("[ref vs wave] L1=%.6f  L2=%.6f\n", L1_5, L2_5);
    printf("[ref vs LectureNote_03_noise] L1=%.6f  L2=%.6f\n", L1_6, L2_6);

    free(hog_ref); free(hog_c1); free(hog_c2); free(hog_c3); free(hog_c4); free(hog_c5); free(hog_c6);
    return 0;
}