#include "MotionLBP.h"
#include <cmath>
#include <opencv2/opencv.hpp>


using namespace cv;

#define EPS 1e-6f

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

void mlbp_uniformLBPHistogram(const Mat& src, float* hist)
{
    for (int i = 0; i < ULBP_BIN; ++i) hist[i] = 0.0f;

    Mat gray;
    if (src.channels() == 3)
        cvtColor(src, gray, COLOR_BGR2GRAY);
    else
        gray = src;

    for (int y = 1; y < gray.rows - 1; ++y) {
        const uchar* prev = gray.ptr<uchar>(y - 1);
        const uchar* curr = gray.ptr<uchar>(y);
        const uchar* next = gray.ptr<uchar>(y + 1);
        for (int x = 1; x < gray.cols - 1; ++x) {
            uchar center = curr[x];
            int code = 0;
            if (prev[x] > center) code |= (1 << 0);
            if (prev[x + 1] > center) code |= (1 << 1);
            if (curr[x + 1] > center) code |= (1 << 2);
            if (next[x + 1] > center) code |= (1 << 3);
            if (next[x] > center) code |= (1 << 4);
            if (next[x - 1] > center) code |= (1 << 5);
            if (curr[x - 1] > center) code |= (1 << 6);
            if (prev[x - 1] > center) code |= (1 << 7);
            int bin = lookup[code];
            hist[bin] += 1.0f;
        }
    }

    float norm = 0.0f;
    for (int i = 0; i < ULBP_BIN; ++i) norm += hist[i] * hist[i];
    norm = sqrtf(norm) + EPS;
    for (int i = 0; i < ULBP_BIN; ++i) hist[i] /= norm;
}

float mlbp_cosineSimilarityExp(const float* feat, const float* tpl) {
    if (!feat || !tpl) return 0.0f;
    float dot = 0.0f, normFeat = 0.0f, normTpl = 0.0f;
    for (int i = 0; i < ULBP_BIN; ++i) {
        dot += feat[i] * tpl[i];
        normFeat += feat[i] * feat[i];
        normTpl += tpl[i] * tpl[i];
    }
    float denom = sqrtf(normFeat) * sqrtf(normTpl);
    float cosine = (denom > EPS) ? (dot / denom) : 0.0f;
    return exp(5.0f * cosine);
}

void mlbp_averageTemplates(float** samples, int sampleCount, float* outTpl) {
    if (!samples || sampleCount <= 0 || !outTpl) return;
    for (int i = 0; i < ULBP_BIN; ++i) outTpl[i] = 0.0f;

    for (int s = 0; s < sampleCount; ++s) {
        const float* sample = samples[s];
        if (!sample) continue;
        for (int i = 0; i < ULBP_BIN; ++i) outTpl[i] += sample[i];
    }
    float n = (float)sampleCount;
    for (int i = 0; i < ULBP_BIN; ++i) outTpl[i] /= n;

    float norm = 0.0f;
    for (int i = 0; i < ULBP_BIN; ++i) norm += outTpl[i] * outTpl[i];
    norm = sqrtf(norm) + EPS;
    for (int i = 0; i < ULBP_BIN; ++i) outTpl[i] /= norm;
}

// [Lecture Note 3, Page 60~62 참고]
// 입력: 그레이스케일 영상 (srcGray)
// 출력: 에지 크기 영상 (dstMag) - LBP의 입력으로 사용됨
void calc_gradient_magnitude(const Mat& srcGray, Mat& dstMag) {
    // 1. 결과 영상 메모리 할당 (OpenCV Mat 사용 허용됨)
    dstMag.create(srcGray.size(), CV_8UC1);

    // 2. 소벨(Sobel) 또는 프리윗(Prewitt) 마스크 정의 (C언어 배열)
    // 제공해주신 코드의 마스크 (Prewitt 마스크와 유사)
    int ymatrix[3][3] = { {-1, -1, -1}, {0, 0, 0}, {1, 1, 1} };
    int xmatrix[3][3] = { {-1, 0, 1}, {-1, 0, 1}, {-1, 0, 1} };

    int height = srcGray.rows;
    int width = srcGray.cols;

    // 3. 컨볼루션 연산 (C언어 스타일 포인터 접근 & 최적화)
    // 가장자리(Border)는 처리하지 않음 (0으로 둠)
    dstMag = Scalar(0);

    for (int y = 1; y < height - 1; y++) {
        // 행 포인터 미리 가져오기 (속도 향상)
        const uchar* prev = srcGray.ptr<uchar>(y - 1);
        const uchar* curr = srcGray.ptr<uchar>(y);
        const uchar* next = srcGray.ptr<uchar>(y + 1);
        uchar* dRow = dstMag.ptr<uchar>(y);

        for (int x = 1; x < width - 1; x++) {
            int fx = 0;
            int fy = 0;

            // 3x3 윈도우 연산 (Loop Unrolling으로 속도 최적화 가능)
            // 제공된 코드의 이중 for문을 풀어서 씀

            // y-1 행 (prev)
            fy += ymatrix[0][0] * prev[x - 1] + ymatrix[0][1] * prev[x] + ymatrix[0][2] * prev[x + 1];
            fx += xmatrix[0][0] * prev[x - 1] + xmatrix[0][1] * prev[x] + xmatrix[0][2] * prev[x + 1];

            // y 행 (curr)
            fy += ymatrix[1][0] * curr[x - 1] + ymatrix[1][1] * curr[x] + ymatrix[1][2] * curr[x + 1];
            fx += xmatrix[1][0] * curr[x - 1] + xmatrix[1][1] * curr[x] + xmatrix[1][2] * curr[x + 1];

            // y+1 행 (next)
            fy += ymatrix[2][0] * next[x - 1] + ymatrix[2][1] * next[x] + ymatrix[2][2] * next[x + 1];
            fx += xmatrix[2][0] * next[x - 1] + xmatrix[2][1] * next[x] + xmatrix[2][2] * next[x + 1];

            // 4. Magnitude 계산
            // sqrt는 double을 반환하므로 캐스팅 필요
            float val = sqrtf((float)(fx * fx + fy * fy));

            // 5. Clamping (0~255 범위 제한)
            // LBP는 픽셀값의 대소 관계만 보므로, Min-Max 정규화 대신 255 넘는 것만 잘라내도 충분함
            if (val > 255.0f) val = 255.0f;

            dRow[x] = (uchar)val;
        }
    }
}


#define GESTURE_WIN 64   // 리사이즈할 크기 (64x64 추천)
#define GRID_SIZE 3      // 3x3 격자 사용
#define LBP_DIM 256      // 기본 LBP (256차원) 사용 시

// 입력: 다양한 크기의 손 ROI (또는 에지 영상)
// 출력: 정규화된 LBP 히스토그램 (길이: GRID_SIZE * GRID_SIZE * 256)
void LBPdescriptor_Gesture(const Mat& inputROI, float* LBPhist) {
    // 1. 고정된 크기(64x64)로 리사이징 (Scale Invariance 확보!)
    Mat resizedImg;
    resize(inputROI, resizedImg, Size(GESTURE_WIN, GESTURE_WIN));

    // 2. LBP 영상 계산 (메모리 직접 관리 대신 Mat 사용하되 포인터 접근)
    Mat lbpImg(GESTURE_WIN, GESTURE_WIN, CV_8UC1);

    // 가장자리 제외하고 루프
    for (int y = 1; y < GESTURE_WIN - 1; ++y) {
        uchar* ptr = lbpImg.ptr<uchar>(y);

        // 3x3 접근을 위한 포인터들
        const uchar* p_prev = resizedImg.ptr<uchar>(y - 1);
        const uchar* p_curr = resizedImg.ptr<uchar>(y);
        const uchar* p_next = resizedImg.ptr<uchar>(y + 1);

        for (int x = 1; x < GESTURE_WIN - 1; ++x) {
            uchar c = p_curr[x];
            int v = 0;
            // 시계 방향 비트 연산 (작성하신 코드와 동일)
            if (p_prev[x] > c) v |= (1 << 0);
            if (p_prev[x + 1] > c) v |= (1 << 1);
            if (p_curr[x + 1] > c) v |= (1 << 2);
            if (p_next[x + 1] > c) v |= (1 << 3);
            if (p_next[x] > c) v |= (1 << 4);
            if (p_next[x - 1] > c) v |= (1 << 5);
            if (p_curr[x - 1] > c) v |= (1 << 6);
            if (p_prev[x - 1] > c) v |= (1 << 7);

            ptr[x] = (uchar)v;
        }
    }

    // 3. 블록별 히스토그램 계산 (3x3 격자, Non-overlapping)
    int blockH = GESTURE_WIN / GRID_SIZE; // 예: 64 / 3 = 21
    int blockW = GESTURE_WIN / GRID_SIZE;

    int histIdx = 0; // 전체 히스토그램 인덱스

    for (int r = 0; r < GRID_SIZE; ++r) {
        for (int c = 0; c < GRID_SIZE; ++c) {

            // 현재 블록의 시작 좌표와 끝 좌표
            int y_start = r * blockH;
            int x_start = c * blockW;
            int y_end = y_start + blockH;
            int x_end = x_start + blockW;

            // 범위 보정 (이미지 크기 안 넘어가게)
            if (y_end > GESTURE_WIN) y_end = GESTURE_WIN;
            if (x_end > GESTURE_WIN) x_end = GESTURE_WIN;

            // 임시 히스토그램 (256차원)
            float tempHist[256] = { 0, };

            // 블록 내부 루프
            for (int yy = y_start; yy < y_end; ++yy) {
                // 가장자리(Border) 부분은 LBP 값이 쓰레기일 수 있으므로 제외하는 게 좋음
                if (yy < 1 || yy >= GESTURE_WIN - 1) continue;

                uchar* ptr = lbpImg.ptr<uchar>(yy);
                for (int xx = x_start; xx < x_end; ++xx) {
                    if (xx < 1 || xx >= GESTURE_WIN - 1) continue;

                    tempHist[ptr[xx]] += 1.0f;
                }
            }

            // L2 정규화 (Normalization)
            float denomi = 0.f;
            for (int b = 0; b < 256; ++b) denomi += tempHist[b] * tempHist[b];
            denomi = sqrtf(denomi) + 1e-6f; // EPS

            // 전체 히스토그램 배열에 저장
            for (int b = 0; b < 256; ++b) {
                LBPhist[histIdx * 256 + b] = tempHist[b] / denomi;
            }
            histIdx++;
        }
    }
}