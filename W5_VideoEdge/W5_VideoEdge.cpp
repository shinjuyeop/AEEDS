#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <windows.h>
#include "cornerDetector.h"

#define PI 3.14159265358979323846f

using namespace cv;

void main()
{   
    LARGE_INTEGER freq, start, stop;
    double diff;

    VideoCapture capture(0);

	Mat frame;      // 카메라에서 받은 프레임
    Mat cornerMap;  // 코너 맵

    // 예외 처리
    if (!capture.isOpened()) {
        printf("Couldn't open the web camera...\n");
        return;
    }

    while (true) {
		capture >> frame;           // 카메라에서 프레임 읽기
		if (frame.empty()) break;   // 빈 프레임이면 종료

        QueryPerformanceFrequency(&freq);
        QueryPerformanceCounter(&start);

        // Higher threshold (230) + larger window (5) reduces number of corners; k slightly higher for stricter response
        HarrisCornerDetector(frame, cornerMap, 230, 4, 5, 0.04f);

        QueryPerformanceCounter(&stop);
        diff = (double)(stop.QuadPart - start.QuadPart) / freq.QuadPart;
        printf("Harris time: %f sec\n", diff);

        imshow("Video", frame);
        //imshow("CornerMap", cornerMap);
        if (waitKey(1) >= 0) break;
    }
}