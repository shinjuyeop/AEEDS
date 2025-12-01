#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include "SLIC.h"                // SLIC 클래스 정의
#include "SLICsegmentation.h"    // SLICsegmentation 함수 선언

using namespace cv;
using namespace std;

int main()
{
    // 이미지 경로는 필요에 따라 수정
    string imagePath = "C:/Users/shinj/Desktop/3-2/AEEDS/Data/input.jpg";
    Mat image = imread(imagePath);

    if (image.empty()) {
        cerr << "이미지를 불러올 수 없습니다: " << imagePath << endl;
        return -1;
    }

    SLICsegmentation(image);

    cout << "SLIC segmentation 완료." << endl;
    return 0;
}