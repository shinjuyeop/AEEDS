// Edge Detection
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <math.h>

#define PI 3.141592

using namespace cv;
int main()
{
    Mat imgGray = imread(
        "C:/Users/shinj/Desktop/3-2/AEEDS/AEEDS/data/test2.jpg",
        IMREAD_GRAYSCALE);

    int height = imgGray.rows;
    int width = imgGray.cols;

    Mat imgEdge(height, width, CV_8UC1);

    int ymatrix[3][3] = { {-1, -1, -1}, {0, 0, 0}, {1, 1, 1} };
    int xmatrix[3][3] = { {-1, 0, 1}, {-1, 0, 1}, {-1, 0, 1} };

    float minVal = 100000; float maxVal = -1;
    /*
    float val;
    int *bin;
    float *pdf;
    */
    // Edge detection (grayscale)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (y == 0 || y == height - 1 || x == 0 || x == width - 1) {
                imgEdge.at<uchar>(y, x) = 0; // 경계는 0
                continue;
            }
            int fx = 0, fy = 0;
            for (int yy = -1; yy <= 1; yy++) {
                for (int xx = -1; xx <= 1; xx++) {
                    fy += ymatrix[yy + 1][xx + 1] * imgGray.at<uchar>(y + yy, x + xx);
                    fx += xmatrix[yy + 1][xx + 1] * imgGray.at<uchar>(y + yy, x + xx);
                }
            }
			// magnitude
            float val = (float)sqrt(fx * fx + fy * fy);
            
            /*
            float dir;

            dir = atan2(fy, fx);
            dir = dir * 180 / PI;
			if (dir < 0) dir += 180;

			bin[y * width + x] = (int)(dir / 20); // 0~8
			//histogram
            pdf[bin[y * width + x]] += val[y * width + x];
            */

			// 최대값, 최소값 설정
            if (y == 0 && x == 0) {
                minVal = maxVal = val;
            }
			// 최대값, 최소값 비교
            else {
                if (val < minVal) minVal = val;
                if (val > maxVal) maxVal = val;
            }
            imgEdge.at<uchar>(y, x) = (uchar)val; // 저장
        }
    }

    // 최대값, 최소값 반영
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float val = imgEdge.at<uchar>(y, x);
            float normVal = 255 * (val - minVal) / (maxVal - minVal);
            imgEdge.at<uchar>(y, x) = (uchar)normVal;
        }
    }

    //imshow("EdgeDetection.bmp", imgEdge);
    //waitKey(0);
    imwrite("EdgeDetection.bmp", imgEdge);
    return 0;
}