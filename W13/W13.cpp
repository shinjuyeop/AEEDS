#include <opencv2/opencv.hpp>
#include <opencv2/photo.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main()
{
    // 1. 파일 경로 (본인 경로로 수정 필수)
    string imagePath = "C:/Users/shinj/Desktop/3-2/AEEDS/Data/640x480.jpeg";
    string maskPath = "C:/Users/shinj/Desktop/3-2/AEEDS/Data/inpaint.png";

    // 2. 이미지 읽기
    Mat src = imread(imagePath, IMREAD_COLOR);
    Mat mask = imread(maskPath, IMREAD_GRAYSCALE); // 마스크는 흑백으로 읽기

    if (src.empty() || mask.empty()) {
        cout << "파일을 찾을 수 없습니다." << endl;
        return -1;
    }

    // 3. [핵심 수정] 흰색(255) 빼고 다 검은색으로 만들기
    // 설명: 픽셀값이 254보다 클 때만(즉 255일 때만) 남기고 나머지는 0으로 만듦
    threshold(mask, mask, 254, 255, THRESH_BINARY);

    // 4. Inpainting 수행
    Mat dst;
    inpaint(src, mask, dst, 3.0, INPAINT_TELEA);

    // 5. 결과 확인
    imshow("Source", src);
    imshow("Mask (Processed)", mask); // 검은색으로 잘 정리됐는지 확인 가능
    imshow("Result", dst);

    waitKey(0);
    return 0;
}