#include "Gesture.h"
#include "MotionLBP.h"

using namespace cv;

// 손 ROI 계산 함수
bool gest_computeHandROI(const Rect& faceRect, const Size& frameSize, Rect& outROI)
{
    int marginX = (int)(faceRect.width * 0.3);

    Rect roi(
        faceRect.x + faceRect.width + marginX, // 시작점 X를 오른쪽으로 밀어줌
        faceRect.y,
        (int)(faceRect.width),
        (int)(faceRect.height)
    );

    if (roi.x + roi.width > frameSize.width)
        roi.width = frameSize.width - roi.x;
    if (roi.y + roi.height > frameSize.height)
        roi.height = frameSize.height - roi.y;

    roi &= Rect(0, 0, frameSize.width, frameSize.height);

	
    if (roi.area() <= 500) return false; // 너무 작은 ROI는 무시
	outROI = roi; // 출력
    return true;
}