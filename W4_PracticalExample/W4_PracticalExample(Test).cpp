#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;

typedef struct { int x; int y; } CPoint;

/*** 1) Gradient ***/
void computePrewittIxIy(const Mat* gray, float* Ix, float* Iy)
{
    const int W = gray->cols;
    const int H = gray->rows;

    int xmask[3][3] = { {-1, 0, 1}, {-1, 0, 1}, {-1, 0, 1} };
    int ymask[3][3] = { {-1,-1,-1}, { 0, 0, 0}, { 1, 1, 1} };

    int x, y, dx, dy;
    // 경계 1픽셀은 0 유지
    for (y = 1; y < H - 1; ++y) {
        for (x = 1; x < W - 1; ++x) {
            int gx = 0, gy = 0;
            for (dy = -1; dy <= 1; ++dy) {
                for (dx = -1; dx <= 1; ++dx) {
                    unsigned char pix = gray->data[(y + dy) * W + (x + dx)];
                    gx += xmask[dy + 1][dx + 1] * (int)pix;
                    gy += ymask[dy + 1][dx + 1] * (int)pix;
                }
            }
            Ix[y * W + x] = (float)gx;
            Iy[y * W + x] = (float)gy;
        }
    }
}

/*** 2) Harris Corner Response R = det - k*(trace^2) ***/
void computeHarrisR(const float* Ix, const float* Iy,
    int W, int H, int win_size, float k,
    float* R, float* out_minR, float* out_maxR)
{
    const int r = win_size / 2;
    float minR = FLT_MAX;
    float maxR = -FLT_MAX;

    int x, y;
    for (y = 0; y < H; ++y) {
        for (x = 0; x < W; ++x) {
            int y0 = (y - r > 0) ? (y - r) : 0;
            int y1 = (y + r < H - 1) ? (y + r) : (H - 1);
            int x0 = (x - r > 0) ? (x - r) : 0;
            int x1 = (x + r < W - 1) ? (x + r) : (W - 1);

            float Sxx = 0.f, Syy = 0.f, Sxy = 0.f;
            int yy, xx;
            for (yy = y0; yy <= y1; ++yy) {
                int row = yy * W;
                for (xx = x0; xx <= x1; ++xx) {
                    float ix = Ix[row + xx];
                    float iy = Iy[row + xx];
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
    *out_minR = minR;
    *out_maxR = maxR;
}

/*** (옵션) R 맵 저장 (0~255 정규화) ***/
/*
void saveRmap(const float* R, int W, int H, float minR, float maxR, const char* path)
{
    Mat Rviz(H, W, CV_8UC1, Scalar(0));
    if (maxR > minR) {
        float denom = maxR - minR;
        int x, y;
        for (y = 0; y < H; ++y) {
            for (x = 0; x < W; ++x) {
                float v = 255.f * (R[y * W + x] - minR) / denom;
                if (v < 0.f)   v = 0.f;
                if (v > 255.f) v = 255.f;
                Rviz.data[y * W + x] = (unsigned char)(v + 0.5f);
            }
        }
    }
    imwrite(path, Rviz);
}
*/

/*** 3) Threshold 기반 Corner 좌표 수집 + 원(circle) 표시 (NMS 없음) ***/
/* out_points: 호출자가 cap 크기로 미리 malloc, 반환: 저장된 코너 개수 */
int collectCornersByThreshold(const Mat* gray, const float* R,
    int W, int H, float minR, float maxR,
    float thresh_ratio, int draw_circle,
    Scalar color, int radius,
    Mat* vis_out, const char* save_path,
    CPoint* out_points, int out_cap)
{
    cvtColor(*gray, *vis_out, COLOR_GRAY2BGR);
    const float thresh = minR + (maxR - minR) * thresh_ratio;

    int count = 0;
    int x, y;
    for (y = 1; y < H - 1; ++y) {
        for (x = 1; x < W - 1; ++x) {
            if (R[y * W + x] >= thresh) {
                if (count < out_cap) {
                    out_points[count].x = x;
                    out_points[count].y = y;
                    ++count;
                }
                if (draw_circle) {
                    Point p; p.x = x; p.y = y;
                    circle(*vis_out, p, radius, color, 1, 8, 0);
                }
            }
        }
    }
    if (save_path && save_path[0] != '\0') {
        imwrite(save_path, *vis_out);
    }
    return count;
}

/*** 4) HOG(9-bin, 0~180deg) 한 코너 주변에서 계산 (block=16) ***/
int computeHOG_around_point(const Mat* gray, int cx, int cy,
    int block, int nbins, float* hist_out)
{
    const int W = gray->cols;
    const int H = gray->rows;
    const int half = block / 2;

    // 블록이 영상 밖으로 나가면 0 반환(계산 건너뜀)
    if (cx - half < 1 || cy - half < 1 || cx + half >= W - 1 || cy + half >= H - 1) {
        int i; for (i = 0; i < nbins; ++i) hist_out[i] = 0.f;
        return 0;
    }

    int i;
    for (i = 0; i < nbins; ++i) hist_out[i] = 0.f;

    int x, y;
    for (y = cy - half; y < cy + half; ++y) {
        for (x = cx - half; x < cx + half; ++x) {
            // 간단한 중심차분으로 gradient (Sobel/Prewitt 대신)
            float gx = (float)gray->data[y * W + (x + 1)] - (float)gray->data[y * W + (x - 1)];
            float gy = (float)gray->data[(y + 1) * W + x] - (float)gray->data[(y - 1) * W + x];

            float mag = sqrtf(gx * gx + gy * gy);
            float angle = atan2f(gy, gx) * 180.0f / (float)CV_PI;  // [-180,180]
            if (angle < 0.f) angle += 180.f;                        // [0,180)

            // bin 할당 (9-bin → 20도 간격)
            int bin = (int)(angle / (180.f / (float)nbins));
            if (bin >= nbins) bin = nbins - 1;
            hist_out[bin] += mag;
        }
    }

    // L2 정규화
    float ss = 0.f;
    for (i = 0; i < nbins; ++i) ss += hist_out[i] * hist_out[i];
    ss = sqrtf(ss) + 1e-6f;
    for (i = 0; i < nbins; ++i) hist_out[i] /= ss;

    return 1;
}

/*** 5) HOG 전체 코너에 대해 계산 (out_hogs: corners_count x nbins) ***/
void computeHOG_for_corners(const Mat* gray, const CPoint* corners, int corner_cnt,
    int block, int nbins, float* out_hogs /*size corner_cnt*nbins*/,
    unsigned char* valid /*size corner_cnt: 1=ok,0=skip*/)
{
    int i, ok;
    for (i = 0; i < corner_cnt; ++i) {
        ok = computeHOG_around_point(gray, corners[i].x, corners[i].y, block, nbins,
            out_hogs + (i * nbins));
        valid[i] = (unsigned char)ok;
        if (!ok) {
            int k;
            for (k = 0; k < nbins; ++k) out_hogs[i * nbins + k] = 0.f;
        }
    }
}

/*** 6) L2 거리 ***/
float L2_distance(const float* a, const float* b, int n)
{
    float s = 0.f;
    int i;
    for (i = 0; i < n; ++i) {
        float d = a[i] - b[i];
        s += d * d;
    }
    return sqrtf(s);
}

/*** 7) 최근접 매칭 (ref→tar) ***/
/* match_idx_ref2tar[i] = tar의 인덱스(또는 -1) */
void match_nearest_L2(const float* H_ref, const unsigned char* valid_ref, int n_ref,
    const float* H_tar, const unsigned char* valid_tar, int n_tar,
    int nbins, int* match_idx_ref2tar)
{
    int i, j;
    for (i = 0; i < n_ref; ++i) {
        if (!valid_ref[i]) { match_idx_ref2tar[i] = -1; continue; }
        float best = FLT_MAX;
        int best_j = -1;
        const float* h1 = H_ref + i * nbins;
        for (j = 0; j < n_tar; ++j) {
            if (!valid_tar[j]) continue;
            const float* h2 = H_tar + j * nbins;
            float d = L2_distance(h1, h2, nbins);
            if (d < best) { best = d; best_j = j; }
        }
        match_idx_ref2tar[i] = best_j;
    }
}

/*** 8) ref,tar 가로 합성 & 라인 시각화 ***/
void draw_matches_and_save(const Mat* ref_color, const Mat* tar_color,
    const CPoint* corners_ref, int n_ref,
    const CPoint* corners_tar, int n_tar,
    const int* match_idx_ref2tar,
    const char* save_path_lines,
    int draw_circles)
{
    int Wref = ref_color->cols, Href = ref_color->rows;
    int Wtar = tar_color->cols, Htar = tar_color->rows;

    int H = (Href > Htar) ? Href : Htar;
    int W = Wref + Wtar;

    Mat canvas(H, W, CV_8UC3, Scalar(0, 0, 0));
    // ref 복사
    Mat roiL = canvas(Rect(0, 0, Wref, Href));
    ref_color->copyTo(roiL);
    // tar 복사 (우측)
    Mat roiR = canvas(Rect(Wref, 0, Wtar, Htar));
    tar_color->copyTo(roiR);

    Scalar blue(255, 0, 0);
    Scalar green(0, 255, 0);
    int i;
    for (i = 0; i < n_ref; ++i) {
        int j = match_idx_ref2tar[i];
        if (j < 0 || j >= n_tar) continue;

        Point p1, p2;
        p1.x = corners_ref[i].x;
        p1.y = corners_ref[i].y;
        p2.x = corners_tar[j].x + Wref; // 오른쪽으로 오프셋
        p2.y = corners_tar[j].y;

        line(canvas, p1, p2, blue, 1, 8, 0);

        if (draw_circles) {
            circle(canvas, p1, 3, green, 1, 8, 0);
            circle(canvas, p2, 3, green, 1, 8, 0);
        }
    }

    imwrite(save_path_lines, canvas);
}

/*** ===== main: ref vs tar 매칭 데모 ===== ***/
int main(void)
{
    // 0) 입력
    Mat ref_gray = imread("C:/Users/shinj/Desktop/3-2/AEEDS/Data/ref.bmp", IMREAD_GRAYSCALE);
    Mat tar_gray = imread("C:/Users/shinj/Desktop/3-2/AEEDS/Data/tar.bmp", IMREAD_GRAYSCALE);
    if (ref_gray.empty() || tar_gray.empty()) {
        fprintf(stderr, "이미지 로드 실패 (경로 확인)\n");
        return -1;
    }
    // 컬러 버전 (그리기용)
    Mat ref_color, tar_color;
    cvtColor(ref_gray, ref_color, COLOR_GRAY2BGR);
    cvtColor(tar_gray, tar_color, COLOR_GRAY2BGR);

    const int Wref = ref_gray.cols, Href = ref_gray.rows;
    const int Wtar = tar_gray.cols, Htar = tar_gray.rows;

    // 1) Gradient
    float* Ix_ref = (float*)calloc(Wref * Href, sizeof(float));
    float* Iy_ref = (float*)calloc(Wref * Href, sizeof(float));
    float* Ix_tar = (float*)calloc(Wtar * Htar, sizeof(float));
    float* Iy_tar = (float*)calloc(Wtar * Htar, sizeof(float));
    if (!Ix_ref || !Iy_ref || !Ix_tar || !Iy_tar) {
        fprintf(stderr, "메모리 할당 실패(grad)\n");
        free(Ix_ref); free(Iy_ref); free(Ix_tar); free(Iy_tar);
        return -1;
    }
    computePrewittIxIy(&ref_gray, Ix_ref, Iy_ref);
    computePrewittIxIy(&tar_gray, Ix_tar, Iy_tar);

    // 2) Harris R
    float* R_ref = (float*)calloc(Wref * Href, sizeof(float));
    float* R_tar = (float*)calloc(Wtar * Htar, sizeof(float));
    if (!R_ref || !R_tar) {
        fprintf(stderr, "메모리 할당 실패(R)\n");
        free(Ix_ref); free(Iy_ref); free(Ix_tar); free(Iy_tar);
        free(R_ref); free(R_tar);
        return -1;
    }
    float minR_ref, maxR_ref, minR_tar, maxR_tar;
    computeHarrisR(Ix_ref, Iy_ref, Wref, Href, 3, 0.04f, R_ref, &minR_ref, &maxR_ref);
    computeHarrisR(Ix_tar, Iy_tar, Wtar, Htar, 3, 0.04f, R_tar, &minR_tar, &maxR_tar);

    // (옵션) R 맵 저장
    // saveRmap(R_ref, Wref, Href, minR_ref, maxR_ref, "ref_Rmap.png");
    // saveRmap(R_tar, Wtar, Htar, minR_tar, maxR_tar, "tar_Rmap.png");

    // 3) Corner 좌표 수집
    const float thresh_ratio = 0.50f;  // 필요시 0.01~0.5 사이로 조절
    Scalar red(0, 0, 255);
    Mat ref_vis, tar_vis;

    // 최대 코너 수를 영상 전체 크기로(상한) 잡음
    CPoint* corners_ref = (CPoint*)malloc(sizeof(CPoint) * Wref * Href);
    CPoint* corners_tar = (CPoint*)malloc(sizeof(CPoint) * Wtar * Htar);
    if (!corners_ref || !corners_tar) {
        fprintf(stderr, "메모리 할당 실패(corners)\n");
        free(Ix_ref); free(Iy_ref); free(Ix_tar); free(Iy_tar);
        free(R_ref); free(R_tar);
        free(corners_ref); free(corners_tar);
        return -1;
    }

    int n_ref = collectCornersByThreshold(&ref_gray, R_ref, Wref, Href,
        minR_ref, maxR_ref, thresh_ratio,
        /*draw_circle*/1, red, 3,
        &ref_vis, "ref_corners.png",
        corners_ref, Wref * Href);
    int n_tar = collectCornersByThreshold(&tar_gray, R_tar, Wtar, Htar,
        minR_tar, maxR_tar, thresh_ratio,
        /*draw_circle*/1, red, 3,
        &tar_vis, "tar_corners.png",
        corners_tar, Wtar * Htar);
    printf("corners: ref=%d, tar=%d\n", n_ref, n_tar);

    // 4) 각 코너의 HOG(16x16, 9-bin)
    const int block = 16;
    const int nbins = 9;

    float* HOG_ref = (float*)malloc(sizeof(float) * n_ref * nbins);
    float* HOG_tar = (float*)malloc(sizeof(float) * n_tar * nbins);
    unsigned char* valid_ref = (unsigned char*)malloc(n_ref);
    unsigned char* valid_tar = (unsigned char*)malloc(n_tar);
    if (!HOG_ref || !HOG_tar || !valid_ref || !valid_tar) {
        fprintf(stderr, "메모리 할당 실패(HOG)\n");
        free(Ix_ref); free(Iy_ref); free(Ix_tar); free(Iy_tar);
        free(R_ref); free(R_tar);
        free(corners_ref); free(corners_tar);
        free(HOG_ref); free(HOG_tar); free(valid_ref); free(valid_tar);
        return -1;
    }

    computeHOG_for_corners(&ref_gray, corners_ref, n_ref, block, nbins, HOG_ref, valid_ref);
    computeHOG_for_corners(&tar_gray, corners_tar, n_tar, block, nbins, HOG_tar, valid_tar);

    // 5) 최근접 매칭 (ref -> tar)
    int* match_idx = (int*)malloc(sizeof(int) * n_ref);
    if (!match_idx) {
        fprintf(stderr, "메모리 할당 실패(match)\n");
        free(Ix_ref); free(Iy_ref); free(Ix_tar); free(Iy_tar);
        free(R_ref); free(R_tar);
        free(corners_ref); free(corners_tar);
        free(HOG_ref); free(HOG_tar); free(valid_ref); free(valid_tar);
        return -1;
    }
    match_nearest_L2(HOG_ref, valid_ref, n_ref, HOG_tar, valid_tar, n_tar, nbins, match_idx);

    // 6) 라인 그리기 (ref, tar 합성)
    draw_matches_and_save(&ref_color, &tar_color,
        corners_ref, n_ref, corners_tar, n_tar,
        match_idx,
        "Matched_lines.png",
        /*draw_circles=*/1);

    // free
    free(Ix_ref); free(Iy_ref); free(Ix_tar); free(Iy_tar);
    free(R_ref); free(R_tar);
    free(corners_ref); free(corners_tar);
    free(HOG_ref); free(HOG_tar); free(valid_ref); free(valid_tar);
    free(match_idx);

    printf("Done. Output: ref_corners.png, tar_corners.png, Matched_lines.png\n");
    return 0;
}