#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <math.h>
#include <float.h>

#define PI 3.141592

using namespace cv;
int main()
{
    Mat imgGray = imread(
        "C:/Users/shinj/Desktop/3-2/AEEDS/Data/checkerboard.png",
        IMREAD_GRAYSCALE);

    if (imgGray.empty()) {
        fprintf(stderr, "이미지 로드 실패\n");
        return -1;
    }

    int height = imgGray.rows;
    int width = imgGray.cols;

	// X and Y gradient masks
    int ymatrix[3][3] = { {-1, -1, -1}, {0, 0, 0}, {1, 1, 1} };
    int xmatrix[3][3] = { {-1, 0, 1}, {-1, 0, 1}, {-1, 0, 1} };

    // Harris parameters
    float k = 0.04f;

    // Initialize to zeros to avoid reading uninitialized memory at borders
    Mat Ixx = Mat::zeros(height, width, CV_32FC1);
    Mat Iyy = Mat::zeros(height, width, CV_32FC1);
    Mat Ixy = Mat::zeros(height, width, CV_32FC1);
    Mat Rmat = Mat::zeros(height, width, CV_32FC1);

    // 1) gradient 계산
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int fx = 0, fy = 0;
            for (int yy = -1; yy <= 1; yy++) {
                for (int xx = -1; xx <= 1; xx++) {
					// convolution
                    uchar pix = imgGray.at<uchar>(y + yy, x + xx);
                    fy += ymatrix[yy + 1][xx + 1] * pix;
                    fx += xmatrix[yy + 1][xx + 1] * pix;
                }
            }

            float fxf = (float)fx;
            float fyf = (float)fy;
            Ixx.at<float>(y, x) = fxf * fxf;
            Iyy.at<float>(y, x) = fyf * fyf;
            Ixy.at<float>(y, x) = fxf * fyf;
        }
    }

    // 2) Harris response R = det(M) - k*(trace(M))^2, with M summed over 3x3 window
    float minR = FLT_MAX; 
    float maxR = -FLT_MAX;
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            float Sxx = 0.f, Syy = 0.f, Sxy = 0.f;
            for (int yy = -1; yy <= 1; yy++) {
                for (int xx = -1; xx <= 1; xx++) {
                    Sxx += Ixx.at<float>(y + yy, x + xx);
                    Syy += Iyy.at<float>(y + yy, x + xx);
                    Sxy += Ixy.at<float>(y + yy, x + xx);
                }
            }
			// det = ad - bc
            float det = Sxx * Syy - Sxy * Sxy;
			// trace = a + d
            float trace = Sxx + Syy;
            // Corner metric
            float R = det - k * (trace * trace);

            // 저장
            Rmat.at<float>(y, x) = R;
			// 최대, 최소값 갱신
            if (R < minR) minR = R;
            if (R > maxR) maxR = R;
        }
    }

    // 3) Normalize R to 0~255 for visualization
    Mat imgCorner(height, width, CV_8UC1);
    imgCorner.setTo(0);
    if (maxR > minR) {
        for (int y = 1; y < height - 1; y++) {
            for (int x = 1; x < width - 1; x++) {
                float R = Rmat.at<float>(y, x);
				// Normalize to 0~255
                float normVal = 255.f * (R - minR) / (maxR - minR);
                float clampedVal = (normVal < 0.f) ? 0.f : (normVal > 255.f) ? 255.f : normVal;
				// 반올림 후 대입
                imgCorner.at<uchar>(y, x) = (uchar)(clampedVal + 0.5f);
            }
        }
    }

    imwrite("Corner.bmp", imgCorner);

    return 0;
}

/*
for (y=0; y < height; y++) {
    for (x = 0; x < width; x++) {
            IxIx=0;
			IyIy=0;
            IxIy=0;

            for (yy = y - win_size/2; yy <= y + win_size; yy++) {
                for (xx = x - win_size/2; xx <= x + win_size/2; xx++) {
					if (yy == 0 && xx == 0% && yy < height && xx < width) continue;
					IxIx += Ix[yy * width + xx] * Ix[yy * width + xx];
					IyIy += Iy[yy * width + xx] * lx[yy * width + xx];
					IxIy += Iy[yy * width + xx] * lx[yy * width + xx];
                }
			}
			det = IxIx * IyIy - IxIy * IxIy;
			tr = IxIx + IyIy;
			R[y * width + x] = det - k * tr * tr;




#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <cfloat>
#include <cmath>
#include <cstdio>

using namespace cv;

int main()
{
    Mat imgGray = imread(
        "C:/Users/shinj/Desktop/3-2/AEEDS/Data/checkerboard.png",
        IMREAD_GRAYSCALE);

    if (imgGray.empty()) {
        fprintf(stderr, "이미지 로드 실패\n");
        return -1;
    }

    const int width = imgGray.cols;
    const int height = imgGray.rows;

    // Gradient masks (3x3 Prewitt-like)
    int xmask[3][3] = { {-1, 0, 1}, {-1, 0, 1}, {-1, 0, 1} };
    int ymask[3][3] = { {-1, -1, -1}, {0, 0, 0}, {1, 1, 1} };

    // Allocate Ix, Iy, and R buffers
    std::vector<float> Ix(width * height, 0.0f);
    std::vector<float> Iy(width * height, 0.0f);
    std::vector<float> R(width * height, 0.0f);

    // 1) Compute image gradients Ix, Iy (skip 1-pixel border)
    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            int gx = 0, gy = 0;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    uchar pix = imgGray.at<uchar>(y + dy, x + dx);
                    gx += xmask[dy + 1][dx + 1] * pix;
                    gy += ymask[dy + 1][dx + 1] * pix;
                }
            }
            Ix[y * width + x] = static_cast<float>(gx);
            Iy[y * width + x] = static_cast<float>(gy);
        }
    }

    // 2) Harris response using windowed sums of Ix^2, Iy^2, Ix*Iy
    const int win_size = 3; // odd number (e.g., 3, 5, 7)
    const int r = win_size / 2;
    const float k = 0.04f;

    float minR = FLT_MAX;
    float maxR = -FLT_MAX;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float Sxx = 0.f, Syy = 0.f, Sxy = 0.f;

            const int y0 = std::max(0, y - r);
            const int y1 = std::min(height - 1, y + r);
            const int x0 = std::max(0, x - r);
            const int x1 = std::min(width - 1, x + r);

            for (int yy = y0; yy <= y1; ++yy) {
                const int row = yy * width;
                for (int xx = x0; xx <= x1; ++xx) {
                    const float ix = Ix[row + xx];
                    const float iy = Iy[row + xx];
                    Sxx += ix * ix;
                    Syy += iy * iy;
                    Sxy += ix * iy;
                }
            }

            const float det = Sxx * Syy - Sxy * Sxy;
            const float tr = Sxx + Syy;
            const float rVal = det - k * tr * tr;
            R[y * width + x] = rVal;

            if (rVal < minR) minR = rVal;
            if (rVal > maxR) maxR = rVal;
        }
    }

    // 3) Normalize R to 0~255 for visualization
    Mat out(height, width, CV_8UC1, Scalar(0));
    if (maxR > minR) {
        const float denom = maxR - minR;
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                float v = 255.f * (R[y * width + x] - minR) / denom;
                v = std::max(0.f, std::min(255.f, v));
                out.at<uchar>(y, x) = static_cast<uchar>(v + 0.5f);
            }
        }
    }

    imwrite("Corner_window.bmp", out);

    // 원한다면 결과를 표시
    // imshow("R (normalized)", out);
    // waitKey();

    return 0;
}

*/