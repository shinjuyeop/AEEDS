#include "BackgroundEffects.h"

using namespace cv;

static Mat gBackgroundImage;

void be_setBackgroundImage(const Mat& img) {
    if (img.empty()) {
        gBackgroundImage.release();
    } else {
        gBackgroundImage = img.clone();
    }
}

void be_makeGrayEffect(const Mat& src, Mat& dst) {
    CV_Assert(src.type() == CV_8UC3);
    dst.create(src.size(), src.type());

    for (int y = 0; y < src.rows; ++y) {
        const Vec3b* s = src.ptr<Vec3b>(y);
        Vec3b* d = dst.ptr<Vec3b>(y);
        for (int x = 0; x < src.cols; ++x) {
            uchar v = (uchar)((s[x][2] + s[x][1] + s[x][0]) / 3);
            d[x] = Vec3b(v, v, v);
        }
    }
}

/*
void be_boxBlurEffect(const Mat& src, Mat& dst, int ksize) {
    CV_Assert(src.type() == CV_8UC3);
    CV_Assert(ksize % 2 == 1);

    dst.create(src.size(), src.type());
    int radius = ksize / 2;

    for (int y = 0; y < src.rows; ++y) {
        Vec3b* drow = dst.ptr<Vec3b>(y);
        for (int x = 0; x < src.cols; ++x) {
            int sumB = 0, sumG = 0, sumR = 0, cnt = 0;
            for (int dy = -radius; dy <= radius; ++dy) {
                int yy = y + dy;
                if (yy < 0 || yy >= src.rows) continue;
                const Vec3b* srow = src.ptr<Vec3b>(yy);
                for (int dx = -radius; dx <= radius; ++dx) {
                    int xx = x + dx;
                    if (xx < 0 || xx >= src.cols) continue;
                    const Vec3b& sp = srow[xx];
                    sumB += sp[0]; sumG += sp[1]; sumR += sp[2]; ++cnt;
                }
            }
            if (cnt > 0) {
                drow[x][0] = (uchar)(sumB / cnt);
                drow[x][1] = (uchar)(sumG / cnt);
                drow[x][2] = (uchar)(sumR / cnt);
            } else {
                drow[x] = src.at<Vec3b>(y, x);
            }
        }
    }
}
*/

// [Lecture Note 3 참고]
// Sigma를 ksize에 비례하여 자동 설정하도록 수정
void be_gaussianBlurEffect(const Mat& src, Mat& dst, int ksize) {
    if (ksize % 2 == 0) ksize++;
    int radius = ksize / 2;

    double* kernel = (double*)malloc(ksize * ksize * sizeof(double));
    if (!kernel) return;

    double sum = 0.0;

    // [수정 포인트] 블러를 세게 하려면 Sigma도 같이 커져야 함
    // Lecture Note 3, 9페이지 G(x) 수식 활용
    // 보통 Sigma의 6배가 커널 크기가 되도록 설정함 (ksize = 6*sigma)
    double sigma = ksize / 6.0;
    if (sigma < 1.0) sigma = 1.0; // 최소값 안전장치

    double sigma2_2 = 2.0 * sigma * sigma;

    // ... (이하 커널 생성 및 정규화 코드는 기존과 동일) ...
    for (int y = -radius; y <= radius; y++) {
        for (int x = -radius; x <= radius; x++) {
            double dist = (double)(x * x + y * y);
            double val = exp(-dist / sigma2_2);
            kernel[(y + radius) * ksize + (x + radius)] = val;
            sum += val;
        }
    }
    for (int i = 0; i < ksize * ksize; i++) kernel[i] /= sum;

    dst = Mat::zeros(src.size(), src.type());

	// Convolution
    for (int y = radius; y < src.rows - radius; ++y) {
        Vec3b* drow = dst.ptr<Vec3b>(y);
        for (int x = radius; x < src.cols - radius; ++x) {
            double sumB = 0.0, sumG = 0.0, sumR = 0.0;
            int k_idx = 0;
            for (int ky = -radius; ky <= radius; ++ky) {
                const Vec3b* srow = src.ptr<Vec3b>(y + ky);
                for (int kx = -radius; kx <= radius; ++kx) {
                    double w = kernel[k_idx++];
                    const Vec3b& sp = srow[x + kx];
                    sumB += sp[0] * w;
                    sumG += sp[1] * w;
                    sumR += sp[2] * w;
                }
            }
            drow[x][0] = (uchar)sumB;
            drow[x][1] = (uchar)sumG;
            drow[x][2] = (uchar)sumR;
        }
    }

    free(kernel);
}

void be_mosaicEffect(const Mat& src, Mat& dst, int blockSize) {
    CV_Assert(src.type() == CV_8UC3);
    dst = src.clone();
    for (int y = 0; y < src.rows; y += blockSize) {
        for (int x = 0; x < src.cols; x += blockSize) {
            int yEnd = std::min(y + blockSize, src.rows);
            int xEnd = std::min(x + blockSize, src.cols);
            Vec3b color = src.at<Vec3b>(y, x);
            for (int yy = y; yy < yEnd; ++yy) {
                Vec3b* drow = dst.ptr<Vec3b>(yy);
                for (int xx = x; xx < xEnd; ++xx) {
                    drow[xx] = color;
                }
            }
        }
    }
}

void be_applyBackgroundEffectMask(const Mat& src, Mat& dst, const Mat& fgMask, BgMode mode) {
    dst = src.clone();
    if (mode == BG_MODE_ORIGINAL) return;

    Mat effected;
    switch (mode) {
    case BG_MODE_GRAY:
        be_makeGrayEffect(src, effected);
        break;
    case BG_MODE_BLUR:
        be_gaussianBlurEffect(src, effected, 11);
        break;
    case BG_MODE_MOSAIC:
        be_mosaicEffect(src, effected, 16);
        break;
    case BG_MODE_IMAGE: {
        if (!gBackgroundImage.empty()) {
            Mat canvas(src.size(), src.type(), Scalar(0, 0, 0));
            int imgW = gBackgroundImage.cols;
            int imgH = gBackgroundImage.rows;
            int offsetX = std::max(0, (src.cols - imgW) / 2);
            int offsetY = std::max(0, (src.rows - imgH) / 2);
            int copyW = std::min(imgW, src.cols);
            int copyH = std::min(imgH, src.rows);
            gBackgroundImage(Rect(0, 0, copyW, copyH)).copyTo(canvas(Rect(offsetX, offsetY, copyW, copyH)));
            effected = canvas;
        } else {
            effected = Mat(src.size(), src.type(), Scalar(0, 0, 0));
        }
    } break;
    default:
        return;
    }

    for (int y = 0; y < src.rows; ++y) {
        const uchar* m = fgMask.ptr<uchar>(y);
        const Vec3b* e = effected.ptr<Vec3b>(y);
        Vec3b* d = dst.ptr<Vec3b>(y);
        for (int x = 0; x < src.cols; ++x) {
            if (m[x] == 0) {
                d[x] = e[x];
            }
        }
    }
}
