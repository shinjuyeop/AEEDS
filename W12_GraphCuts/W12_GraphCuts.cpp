#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

void computeSaliencyMap(const cv::Mat& input, float* saliencyMap)
{
    cv::Mat gray;
    cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    for (int y = 0; y < gray.rows; ++y) {
        for (int x = 0; x < gray.cols; ++x) {
            saliencyMap[y * gray.cols + x] = gray.at<uchar>(y, x) / 255.0f;
        }
    }
}

void main()
{
    Mat input = imread("C:/Users/shinj/Desktop/3-2/AEEDS/Data/input.jpg");
    Mat mask_color = imread("C:/Users/shinj/Desktop/3-2/AEEDS/Data/input_initial.bmp");

    // grabCut용 마스크 생성: 기본값은 '배경'
    Mat mask(mask_color.size(), CV_8UC1, Scalar(GC_PR_BGD));

    // 빨간색(전경), 파란색(배경) 선을 찾아서 마스크에 할당
    for (int y = 0; y < mask_color.rows; ++y) {
        for (int x = 0; x < mask_color.cols; ++x) {
            Vec3b color = mask_color.at<Vec3b>(y, x);
            if (color[2] > 200 && color[1] < 80 && color[0] < 80) // 빨간색
                mask.at<uchar>(y, x) = GC_FGD;
            else if (color[0] > 200 && color[1] < 80 && color[2] < 80) // 파란색
                mask.at<uchar>(y, x) = GC_BGD;
        }
    }

    float* saliencyMap = (float*)calloc(input.rows * input.cols, sizeof(float));
    Mat bg, fg;

    computeSaliencyMap(input, saliencyMap);

    grabCut(input, mask, Rect(), bg, fg, 5, GC_INIT_WITH_MASK);

    Mat foregroundMask = (mask == GC_FGD) | (mask == GC_PR_FGD);

    Mat foreground(input.size(), CV_8UC3, Scalar(255, 255, 255));
    input.copyTo(foreground, foregroundMask);

    imshow("Foreground", foreground);
    imwrite("Foreground.bmp", foreground);
    waitKey(0);
}
