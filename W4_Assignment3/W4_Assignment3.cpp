#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#define PI 3.14159265358979323846f

using namespace cv;

typedef struct { int x; int y; } CPoint;

/*** 공통 파라미터 ***/
static const int   HARRIS_WIN = 3;        // Harris 윈도 크기(3,5…)
static const float HARRIS_K = 0.04f;      // Harris k
static const float THRESH_RATIO = 0.4f;   // R 임계 비율(0~1)
static const int   HOG_BLOCK = 16;        // 코너 주변 블록(정사각)
static const int   HOG_BINS = 9;          // 0~180도 9-bin
static const float HOG_EPS = 1e-6f;       // L2 정규화 epsilon

/*** 1) Gradient (Ix, Iy) ***/
void computePrewittIxIy(const Mat& gray, float* Ix, float* Iy)
{
    const int W = gray.cols, H = gray.rows;
    const int size = 9;
    int mask_x[size] = { -1, 0, 1,  -1, 0, 1,  -1, 0, 1 };
    int mask_y[size] = { -1,-1,-1,   0, 0, 0,   1, 1, 1 };

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float gx = 0.f, gy = 0.f;
            for (int yy = y - 1; yy <= y + 1; ++yy) {
                if (yy < 0 || yy >= H) continue;
                for (int xx = x - 1; xx <= x + 1; ++xx) {
                    if (xx < 0 || xx >= W) continue;
					int ky = yy - (y - 1); // mask y index
					int kx = xx - (x - 1); // mask x index
                    float I = (float)gray.at<uchar>(yy, xx) / 255.0f;
					gx += mask_x[ky * 3 + kx] * I; // mask * image
					gy += mask_y[ky * 3 + kx] * I; // mask * image
                }
            }
			Ix[y * W + x] = gx; // gradient x
			Iy[y * W + x] = gy; // gradient y
        }
    }
}

/*** 2) Harris R 맵 계산 ***/
void computeHarrisR(const float* Ix, const float* Iy,
    int W, int H, int win_size, float k,
    float* R, float* out_minR, float* out_maxR)
{
    float minR = FLT_MAX, maxR = -FLT_MAX;
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float Sxx = 0.f, Syy = 0.f, Sxy = 0.f;
            for (int yy = y - win_size / 2; yy <= y + win_size / 2; ++yy) {
                if (yy < 0 || yy >= H) continue;
                for (int xx = x - win_size / 2; xx <= x + win_size / 2; ++xx) {
                    if (xx < 0 || xx >= W) continue;
                    float ix = Ix[yy * W + xx];
                    float iy = Iy[yy * W + xx];
                    Sxx += ix * ix;
                    Syy += iy * iy;
                    Sxy += ix * iy;
                }
            }
            float det = Sxx * Syy - Sxy * Sxy;
            float tr = Sxx + Syy;
            float rVal = det - k * tr * tr;
            R[y * W + x] = rVal;
            if (rVal < minR) minR = rVal;
            if (rVal > maxR) maxR = rVal;
        }
    }
    *out_minR = minR; *out_maxR = maxR;
}

/*** 3) 임곗값 기반 코너 수집 ***/
int collectCornersByThreshold(const Mat& gray, const float* R,
    int W, int H, float minR, float maxR,
    float thresh_ratio, CPoint* out_pts, int cap)
{
    const float thresh = minR + (maxR - minR) * thresh_ratio;
    int n = 0;
	// 경계 1픽셀 제외
    for (int y = 1; y < H - 1; ++y) {
        for (int x = 1; x < W - 1; ++x) {
			if (R[y * W + x] >= thresh) { // 임곗값 이상
				if (n < cap) { out_pts[n].x = x; out_pts[n].y = y; ++n; } // 저장
            }
        }
    }
    return n;
}

/*** 4) 한 코너 주변 HOG 계산 (16×16, 9-bin, 0~180) ***/
int computeHOG_around_corner(const Mat& gray, int cx, int cy,
    int block, int nbins, float* hist_out)
{
    const int W = gray.cols, H = gray.rows;
    const int half = block / 2;

    if (cx - half < 1 || cy - half < 1 || cx + half > W - 2 || cy + half > H - 2) {
        for (int i = 0; i < nbins; ++i) hist_out[i] = 0.f;
        return 0;
    }
    for (int i = 0; i < nbins; ++i) hist_out[i] = 0.f;

    for (int y = cy - half; y < cy + half; ++y) {
        for (int x = cx - half; x < cx + half; ++x) {
            float gx = (float)gray.at<uchar>(y, x + 1) - (float)gray.at<uchar>(y, x - 1);
            float gy = (float)gray.at<uchar>(y + 1, x) - (float)gray.at<uchar>(y - 1, x);
            float mag = sqrtf(gx * gx + gy * gy);
            float ang = atan2f(gy, gx) * 180.0f / PI;
            if (ang < 0.f) ang += 180.f;
            int bin = (int)(ang / (180.f / (float)nbins));
            if (bin >= nbins) bin = nbins - 1;
            hist_out[bin] += mag;
        }
    }

    // L2 정규화
    float ss = HOG_EPS;
    for (int i = 0; i < nbins; ++i) ss += hist_out[i] * hist_out[i];
    ss = 1.0f / sqrtf(ss);
    for (int i = 0; i < nbins; ++i) hist_out[i] *= ss;
    return 1;
}

/*** 5) L2 거리 계산 함수 ***/
float L2dist(const float* a, const float* b, int n) {
    float s = 0.f;
    for (int i = 0; i < n; ++i) {
        float d = a[i] - b[i];
        s += d * d;
    }
    return sqrtf(s);
}

/*** 6) ref의 각 코너를 tar의 가장 가까운 HOG에 매칭 ***/
void match_nearest(const float* Href, const unsigned char* ok_ref, int n_ref,
    const float* Htar, const unsigned char* ok_tar, int n_tar,
    int nbins, int* match_idx_ref2tar)
{
    for (int i = 0; i < n_ref; ++i) {
		if (!ok_ref[i]) { match_idx_ref2tar[i] = -1; continue; } // 유효하지 않으면 -1
		const float* h1 = Href + i * nbins; // ref의 i번째 HOG
        float best = FLT_MAX; int bestj = -1;
        for (int j = 0; j < n_tar; ++j) {
            if (!ok_tar[j]) continue;
            const float* h2 = Htar + j * nbins;
            float d = L2dist(h1, h2, nbins);
            if (d < best) { best = d; bestj = j; }
        }
        match_idx_ref2tar[i] = bestj;
    }
}

/*** 7) ref/tar 좌우 합성 + 매칭 선 그리기 ***/
void draw_matches_and_save(const Mat& ref_gray, const Mat& tar_gray,
    const CPoint* pref, int nref,
    const CPoint* ptar, int ntar,
    const int* match_idx, const char* out_path)
{
    Mat ref_color, tar_color;
    cvtColor(ref_gray, ref_color, COLOR_GRAY2BGR);
    cvtColor(tar_gray, tar_color, COLOR_GRAY2BGR);

    int Wref = ref_color.cols, Href = ref_color.rows;
    int Wtar = tar_color.cols, Htar = tar_color.rows;
    int H = std::max(Href, Htar);
    int W = Wref + Wtar;

    Mat canvas(H, W, CV_8UC3, Scalar(0, 0, 0));
    ref_color.copyTo(canvas(Rect(0, 0, Wref, Href)));
    tar_color.copyTo(canvas(Rect(Wref, 0, Wtar, Htar)));

    Scalar blue(255, 0, 0), green(0, 255, 0);
    for (int i = 0; i < nref; ++i) {
        int j = match_idx[i];
        if (j < 0 || j >= ntar) continue;
        Point p1(pref[i].x, pref[i].y);
        Point p2(ptar[j].x + Wref, ptar[j].y);
        line(canvas, p1, p2, blue, 2, 8, 0);
        circle(canvas, p1, 10, green, 1, 8, 0);
        circle(canvas, p2, 10, green, 1, 8, 0);
    }
    imwrite(out_path, canvas);
}

int main()
{
    // 0) 입력
    Mat ref = imread("C:/Users/shinj/Desktop/3-2/AEEDS/Data/ref.bmp", IMREAD_GRAYSCALE);
    Mat tar = imread("C:/Users/shinj/Desktop/3-2/AEEDS/Data/tar.bmp", IMREAD_GRAYSCALE);
    if (ref.empty() || tar.empty()) {
        fprintf(stderr, "이미지 로드 실패(ref.bmp / tar.bmp 확인)\n");
        return -1;
    }

    const int Wref = ref.cols, Href = ref.rows;
    const int Wtar = tar.cols, Htar = tar.rows;

    // 1) Gradient
    float* Ix_ref = (float*)calloc(Wref * Href, sizeof(float));
    float* Iy_ref = (float*)calloc(Wref * Href, sizeof(float));
    float* Ix_tar = (float*)calloc(Wtar * Htar, sizeof(float));
    float* Iy_tar = (float*)calloc(Wtar * Htar, sizeof(float));
    if (!Ix_ref || !Iy_ref || !Ix_tar || !Iy_tar) {
        fprintf(stderr, "메모리 할당 실패(grad)\n");
        if (Ix_ref) free(Ix_ref);
        if (Iy_ref) free(Iy_ref);
        if (Ix_tar) free(Ix_tar);
        if (Iy_tar) free(Iy_tar);
        return -1;
    }
    computePrewittIxIy(ref, Ix_ref, Iy_ref);
    computePrewittIxIy(tar, Ix_tar, Iy_tar);

    // 2) Harris R
    float* Rref = (float*)calloc(Wref * Href, sizeof(float));
    float* Rtar = (float*)calloc(Wtar * Htar, sizeof(float));
    float minR_ref, maxR_ref, minR_tar, maxR_tar;
    computeHarrisR(Ix_ref, Iy_ref, Wref, Href, HARRIS_WIN, HARRIS_K, Rref, &minR_ref, &maxR_ref);
    computeHarrisR(Ix_tar, Iy_tar, Wtar, Htar, HARRIS_WIN, HARRIS_K, Rtar, &minR_tar, &maxR_tar);

    // 3) 코너 수집
    CPoint* cref = (CPoint*)malloc(sizeof(CPoint) * Wref * Href);
    CPoint* ctar = (CPoint*)malloc(sizeof(CPoint) * Wtar * Htar);
    int nref = collectCornersByThreshold(ref, Rref, Wref, Href, minR_ref, maxR_ref, THRESH_RATIO, cref, Wref * Href);
    int ntar = collectCornersByThreshold(tar, Rtar, Wtar, Htar, minR_tar, maxR_tar, THRESH_RATIO, ctar, Wtar * Htar);
    printf("corners: ref=%d, tar=%d\n", nref, ntar);

    // 4) 각 코너의 HOG(16×16, 9-bin)
    float* HrefArr = (float*)malloc(sizeof(float) * nref * HOG_BINS);
    float* HtarArr = (float*)malloc(sizeof(float) * ntar * HOG_BINS);
    unsigned char* ok_ref = (unsigned char*)malloc(nref);
    unsigned char* ok_tar = (unsigned char*)malloc(ntar);
    if (!HrefArr || !HtarArr || !ok_ref || !ok_tar) {
        fprintf(stderr, "메모리 할당 실패(HOG)\n");
        if (HrefArr) free(HrefArr); if (HtarArr) free(HtarArr);
        if (ok_ref) free(ok_ref);   if (ok_tar) free(ok_tar);
        if (Ix_ref) free(Ix_ref);   if (Iy_ref) free(Iy_ref);
        if (Ix_tar) free(Ix_tar);   if (Iy_tar) free(Iy_tar);
        if (Rref) free(Rref);       if (Rtar) free(Rtar);
        if (cref) free(cref);       if (ctar) free(ctar);
        return -1;
    }

	// 각 코너 주변 HOG 계산
    for (int i = 0; i < nref; ++i)
        ok_ref[i] = (unsigned char)computeHOG_around_corner(ref, cref[i].x, cref[i].y, HOG_BLOCK, HOG_BINS, HrefArr + i * HOG_BINS);
    for (int j = 0; j < ntar; ++j)
        ok_tar[j] = (unsigned char)computeHOG_around_corner(tar, ctar[j].x, ctar[j].y, HOG_BLOCK, HOG_BINS, HtarArr + j * HOG_BINS);

    // 5) 최근접 매칭(ref -> tar)
    int* match_idx = (int*)malloc(sizeof(int) * nref);
    match_nearest(HrefArr, ok_ref, nref, HtarArr, ok_tar, ntar, HOG_BINS, match_idx);

    // 6) 좌우 합성 & 선 그리기
    draw_matches_and_save(ref, tar, cref, nref, ctar, ntar, match_idx, "matched_lines.png");

    // 7) 코너 원 시각화
    {
        Mat refc, tarc;
        cvtColor(ref, refc, COLOR_GRAY2BGR);
        cvtColor(tar, tarc, COLOR_GRAY2BGR);
        Scalar red(0, 0, 255);
        for (int i = 0; i < nref; ++i)
            circle(refc, Point(cref[i].x, cref[i].y), 8, red, 1, 8, 0);
        for (int j = 0; j < ntar; ++j)
            circle(tarc, Point(ctar[j].x, ctar[j].y), 8, red, 1, 8, 0);
        imwrite("ref_corners.png", refc);
        imwrite("tar_corners.png", tarc);
    }

    // free
    free(Ix_ref); free(Iy_ref); free(Ix_tar); free(Iy_tar);
    free(Rref); free(Rtar); free(cref); free(ctar);
    free(HrefArr); free(HtarArr); free(ok_ref); free(ok_tar);
    free(match_idx);

    printf("Done. Output: ref_corners.png, tar_corners.png, matched_lines.png\n");
    return 0;
}