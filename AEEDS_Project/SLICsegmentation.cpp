#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include "SLIC.h" // SLIC 클래스 정의가 들어있는 헤더

using namespace cv;
using namespace std;

void SLICsegmentation(Mat image)
{
    SLIC slic;
    int x, y;
    int height, width;
    int numlabels;            // Generated number of superpixels
    int m_spcount = 1000;     // Desired number of superpixels
    double m_compactness = 30;// 20.0; // compactness factor (1-40)
    height = image.rows;
    width = image.cols;
    unsigned int* ubuff = (unsigned int*)calloc(height * width, sizeof(unsigned int));
    int* labels = (int*)calloc(height * width, sizeof(int));
    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x++) {
            ubuff[y * width + x] = (int)image.at<Vec3b>(y, x)[0] + ((int)image.at<Vec3b>(y, x)[1] << 8) +
                ((int)image.at<Vec3b>(y, x)[2] << 16);
        }
    }

    slic.DoSuperpixelSegmentation_ForGivenNumberOfSuperpixels(ubuff, width, height, labels,
        numlabels, m_spcount, m_compactness);

    // for drawing SLIC superpixels
    Mat result(height, width, CV_8UC3);
    slic.DrawContoursAroundSegments(ubuff, labels, width, height, 0);
    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x++) {
            result.at<Vec3b>(y, x)[0] = ubuff[y * width + x] & 0xff;
            result.at<Vec3b>(y, x)[1] = ubuff[y * width + x] >> 8 & 0xff;
            result.at<Vec3b>(y, x)[2] = ubuff[y * width + x] >> 16 & 0xff;
        }
    }
    imshow("SLIC_segmentation.bmp", result);
    imwrite("SLIC_segmentation.bmp", result);
    free(ubuff);
    free(labels);
	waitKey(0);
}
