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

    Mat frame;
    Mat cornerMap; // normalized cornerness map only

    if (!capture.isOpened()) {
        printf("Couldn't open the web camera...\n");
        return;
    }

    while (true) {
        capture >> frame;
        if (frame.empty()) break;

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