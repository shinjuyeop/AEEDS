#include "MotionLBP.h"
#include <cmath>

using namespace cv;

#define EPS 1e-6f
#define GESTURE_WIN 64

// [Lecture Note 8, Page 13] Uniform LBP Lookup Table
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

// 에지 계산 함수
// [Lecture Note 3, Page 60 참고]
void calc_gradient_magnitude(const Mat& srcGray, Mat& dstMag) {
    // 1. 결과 영상 메모리 할당
    dstMag.create(srcGray.size(), CV_8UC1);

    dstMag = Scalar(0);

    int height = srcGray.rows;
    int width = srcGray.cols;

    // ymatrix: 수평 에지 검출 / xmatrix: 수직 에지 검출
    int ymatrix[3][3] = { {-1, -1, -1}, {0, 0, 0}, {1, 1, 1} };
    int xmatrix[3][3] = { {-1, 0, 1}, {-1, 0, 1}, {-1, 0, 1} };

    // 2. 컨볼루션 연산
    // 가장자리는 제외하고 반복문 수행
    for (int y = 1; y < height - 1; y++) {

        // 속도를 위해 행 포인터를 미리 가져옴
        uchar* dRow = dstMag.ptr<uchar>(y);

        for (int x = 1; x < width - 1; x++) {
            int fx = 0;
            int fy = 0;

            // 3x3 윈도우를 순회하며 행렬 곱의 합을 계산
            for (int yy = -1; yy <= 1; yy++) {
                // y+yy 행의 포인터를 가져옴
                const uchar* sRow = srcGray.ptr<uchar>(y + yy);

                for (int xx = -1; xx <= 1; xx++) {
                    // 행렬(가중치) * 픽셀값
                    int weightY = ymatrix[yy + 1][xx + 1];
                    int weightX = xmatrix[yy + 1][xx + 1];
                    int pixelVal = sRow[x + xx];

                    fy += weightY * pixelVal;
                    fx += weightX * pixelVal;
                }
            }

            // 3. Magnitude 계산 (L2 Norm)
            // sqrt 사용 (제공 코드와 동일)
            float val = sqrtf((float)(fx * fx + fy * fy));

            // 4. Clamping (0~255 범위 제한)
            if (val > 255.0f) val = 255.0f;

            dRow[x] = (uchar)val;
        }
    }
}

// Uniform LBP 적용 + Grid
void LBPdescriptor_Gesture(const Mat& inputROI, float* LBPhist) {
    // 1. 리사이징 (Scale Invariance)
    Mat resizedImg;
    resize(inputROI, resizedImg, Size(GESTURE_WIN, GESTURE_WIN));

    // 2. LBP 패턴 계산
    Mat lbpImg(GESTURE_WIN, GESTURE_WIN, CV_8UC1);

    for (int y = 1; y < GESTURE_WIN - 1; ++y) {
        uchar* ptr = lbpImg.ptr<uchar>(y);
        const uchar* p_prev = resizedImg.ptr<uchar>(y - 1);
        const uchar* p_curr = resizedImg.ptr<uchar>(y);
        const uchar* p_next = resizedImg.ptr<uchar>(y + 1);

        for (int x = 1; x < GESTURE_WIN - 1; ++x) {
            uchar c = p_curr[x];
            int v = 0;
            if (p_prev[x] > c)     v |= (1 << 0);
            if (p_prev[x + 1] > c) v |= (1 << 1);
            if (p_curr[x + 1] > c) v |= (1 << 2);
            if (p_next[x + 1] > c) v |= (1 << 3);
            if (p_next[x] > c)     v |= (1 << 4);
            if (p_next[x - 1] > c) v |= (1 << 5);
            if (p_curr[x - 1] > c) v |= (1 << 6);
            if (p_prev[x - 1] > c) v |= (1 << 7);

            // [중요] 여기서 lookup 테이블을 통과시켜 0~58 값으로 변환
            ptr[x] = lookup[v];
        }
    }

    // 3. Grid별 히스토그램 (59 bins)
    int blockH = GESTURE_WIN / GRID_SIZE;
    int blockW = GESTURE_WIN / GRID_SIZE;
    int histIdx = 0;

    for (int r = 0; r < GRID_SIZE; ++r) {
        for (int c = 0; c < GRID_SIZE; ++c) {
            int y_start = r * blockH;
            int x_start = c * blockW;
            int y_end = y_start + blockH;
            int x_end = x_start + blockW;
            if (y_end > GESTURE_WIN) y_end = GESTURE_WIN;
            if (x_end > GESTURE_WIN) x_end = GESTURE_WIN;

            // 59 (ULBP_BINS)
            float tempHist[ULBP_BINS] = { 0, };

            for (int yy = y_start; yy < y_end; ++yy) {
                if (yy < 1 || yy >= GESTURE_WIN - 1) continue;
                uchar* ptr = lbpImg.ptr<uchar>(yy);
                for (int xx = x_start; xx < x_end; ++xx) {
                    if (xx < 1 || xx >= GESTURE_WIN - 1) continue;
                    // lookup을 거친 값이므로 0~58 사이임이 보장됨
                    tempHist[ptr[xx]] += 1.0f;
                }
            }

            // 정규화
            float denomi = 0.f;
            for (int b = 0; b < ULBP_BINS; ++b) denomi += tempHist[b] * tempHist[b];
            denomi = sqrtf(denomi) + EPS;

            // 결과 저장
            for (int b = 0; b < ULBP_BINS; ++b) {
                LBPhist[histIdx * ULBP_BINS + b] = tempHist[b] / denomi;
            }
            histIdx++;
        }
    }
}

// Cosine similarity with exponential scaling
float mlbp_cosineSimilarityExp(const float* feat, const float* tpl, int size) {
    if (!feat || !tpl) return 0.0f;
    float dot = 0.0f, normFeat = 0.0f, normTpl = 0.0f;
    for (int i = 0; i < size; ++i) {
        dot += feat[i] * tpl[i];
        normFeat += feat[i] * feat[i];
        normTpl += tpl[i] * tpl[i];
    }
    float denom = sqrtf(normFeat) * sqrtf(normTpl);
    float cosine = (denom > EPS) ? (dot / denom) : 0.0f;
    return exp(5.0f * cosine);
}