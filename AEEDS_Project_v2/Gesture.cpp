#include "Gesture.h"
#include "MotionLBP.h"

using namespace cv;
using namespace std;

bool gest_computeHandROI(const Rect& faceRect, const Size& frameSize, Rect& outROI)
{
    /*
    Rect roi(faceRect.x + faceRect.width,
             faceRect.y,
             (int)(faceRect.width * 1.5),
             (int)(faceRect.height * 1.5));
    */
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

    if (roi.area() <= 500) return false;
    outROI = roi;
    return true;
}