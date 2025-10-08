/*
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <math.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;

void main()
{
    // 0) 입력 (그레이스케일)
    Mat imgGray = imread(
        "C:/Users/shinj/Desktop/3-2/AEEDS/Data/checkerboard.png",
        IMREAD_GRAYSCALE);

    if (imgGray.empty()) {
        fprintf(stderr, "이미지 로드 실패\n");
    }


    const int width = imgGray.cols;
    const int height = imgGray.rows;

    // 1) 3x3 Prewitt gradient
    int xmask[3][3] = { {-1, 0, 1}, {-1, 0, 1}, {-1, 0, 1} };
    int ymask[3][3] = { {-1,-1,-1}, { 0, 0, 0}, { 1, 1, 1} };

    float* Ix = (float*)calloc(width * height, sizeof(float));
    float* Iy = (float*)calloc(width * height, sizeof(float));
    float* R = (float*)calloc(width * height, sizeof(float));
    if (!Ix || !Iy || !R) {
        fprintf(stderr, "메모리 할당 실패\n");
        free(Ix); free(Iy); free(R);
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

    const int radius = 3;
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

}
*/

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;

/* C-Style point */
typedef struct { int x; int y; } CPoint;

/* ========== 1) Prewitt Gradient (C-style) ========== */
void computePrewittIxIy(const Mat* gray, float* Ix, float* Iy)
{
    const int width = gray->cols;
    const int height = gray->rows;

    int xmask[3][3] = { {-1, 0, 1}, {-1, 0, 1}, {-1, 0, 1} };
    int ymask[3][3] = { {-1,-1,-1}, { 0, 0, 0}, { 1, 1, 1} };

    int x, y, dx, dy;
    for (y = 1; y < height - 1; ++y) {
        for (x = 1; x < width - 1; ++x) {
            int gx = 0, gy = 0;
            for (dy = -1; dy <= 1; ++dy) {
                for (dx = -1; dx <= 1; ++dx) {
                    unsigned char pix = gray->data[(y + dy) * width + (x + dx)];
                    gx += xmask[dy + 1][dx + 1] * (int)pix;
                    gy += ymask[dy + 1][dx + 1] * (int)pix;
                }
            }
            Ix[y * width + x] = (float)gx;
            Iy[y * width + x] = (float)gy;
        }
    }
}

/* ========== 2) Harris R (C-style) ========== */
void computeHarrisR(const float* Ix, const float* Iy,
    int width, int height, int win_size, float k,
    float* R, float* out_minR, float* out_maxR)
{
    const int r = win_size / 2;
    float minR = FLT_MAX;
    float maxR = -FLT_MAX;

    int x, y;
    for (y = 0; y < height; ++y) {
        for (x = 0; x < width; ++x) {
            int y0 = (y - r > 0) ? (y - r) : 0;
            int y1 = (y + r < height - 1) ? (y + r) : (height - 1);
            int x0 = (x - r > 0) ? (x - r) : 0;
            int x1 = (x + r < width - 1) ? (x + r) : (width - 1);

            float Sxx = 0.f, Syy = 0.f, Sxy = 0.f;
            int yy, xx;
            for (yy = y0; yy <= y1; ++yy) {
                int row = yy * width;
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

            R[y * width + x] = rVal;
            if (rVal < minR) minR = rVal;
            if (rVal > maxR) maxR = rVal;
        }
    }
    *out_minR = minR;
    *out_maxR = maxR;
}

/* ========== 3) Save R-map (0~255) ========== */
void saveRmap(const float* R, int width, int height,
    float minR, float maxR, const char* path)
{
    Mat Rviz(height, width, CV_8UC1, Scalar(0));
    if (maxR > minR) {
        float denom = maxR - minR;
        int x, y;
        for (y = 0; y < height; ++y) {
            for (x = 0; x < width; ++x) {
                float v = 255.f * (R[y * width + x] - minR) / denom;
                if (v < 0.f)   v = 0.f;
                if (v > 255.f) v = 255.f;
                Rviz.data[y * width + x] = (unsigned char)(v + 0.5f);
            }
        }
    }
    imwrite(path, Rviz);
}

/* ========== 4) Draw circles & collect corners (C-style) ========== */
/* out_points: 호출자가 미리 width*height 크기로 malloc 가능 (최대치)
   반환값: 저장된 코너 개수 */
int drawCornersByThreshold(const Mat* gray, const float* R,
    int width, int height,
    float minR, float maxR, float thresh_ratio,
    Scalar color, int radius,
    Mat* out_vis, const char* save_path,
    CPoint* out_points, int out_points_cap)
{
    cvtColor(*gray, *out_vis, COLOR_GRAY2BGR);

    const float thresh = minR + (maxR - minR) * thresh_ratio;
    int count = 0;

    int x, y;
    for (y = 1; y < height - 1; ++y) {
        for (x = 1; x < width - 1; ++x) {
            if (R[y * width + x] >= thresh) {
                if (count < out_points_cap) {
                    out_points[count].x = x;
                    out_points[count].y = y;
                    ++count;
                }
                Point p; p.x = x; p.y = y;
                circle(*out_vis, p, radius, color, 1, 8, 0);
            }
        }
    }
    if (save_path && save_path[0] != '\0') {
        imwrite(save_path, *out_vis);
    }
    return count;
}

/* ========== Example main (C-style) ========== */
int main(void)
{
    Mat gray = imread("C:/Users/shinj/Desktop/3-2/AEEDS/Data/checkerboard.png",
        IMREAD_GRAYSCALE);
    if (gray.empty()) {
        fprintf(stderr, "이미지 로드 실패\n");
        return -1;
    }

    const int W = gray.cols;
    const int H = gray.rows;

    /* buffers */
    float* Ix = (float*)calloc(W * H, sizeof(float));
    float* Iy = (float*)calloc(W * H, sizeof(float));
    float* R = (float*)calloc(W * H, sizeof(float));
    if (!Ix || !Iy || !R) {
        fprintf(stderr, "메모리 할당 실패\n");
        free(Ix); free(Iy); free(R);
        return -1;
    }

    /* 1) gradient */
    computePrewittIxIy(&gray, Ix, Iy);

    /* 2) Harris R */
    float minR, maxR;
    computeHarrisR(Ix, Iy, W, H, 3, 0.04f, R, &minR, &maxR);

    /* 3) (옵션) R-map 저장 */
    saveRmap(R, W, H, minR, maxR, "Corner.bmp");

    /* 4) circle 시각화 + 코너 좌표 수집 */
    Mat vis;
    Scalar red; red.val[0] = 0; red.val[1] = 0; red.val[2] = 255;
    CPoint* corners = (CPoint*)malloc(sizeof(CPoint) * W * H); /* 최대치 */
    int corner_cnt = drawCornersByThreshold(
        &gray, R, W, H, minR, maxR,
        /*thresh_ratio=*/0.4f, red, /*radius=*/3,
        &vis, "Corner_with_circles.png",
        corners, W * H
    );
    printf("Detected corners: %d\n", corner_cnt);

    /* (여기서 corners 배열을 이용해 HOG 계산/매칭 후 line() 그리기 가능)
       예: line(result, Point(x1,y1), Point(x2+Wref,y2), Scalar(255,0,0), 2, 8, 0); */

       /* free */
    free(Ix); free(Iy); free(R);
    free(corners);
    return 0;
}