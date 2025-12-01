#include "Segmentation.h"

using namespace cv;

void seg_getForegroundMaskWithGrabCut(const Mat& src, Mat& outputMask, Rect faceRect)
{
    if (src.empty() || faceRect.area() <= 0) {
        outputMask = Mat::zeros(src.size(), CV_8UC1);
        return;
    }

    Mat smallSrc;
    double scale = 0.25;
    resize(src, smallSrc, Size(), scale, scale);

    Mat gcMask(smallSrc.size(), CV_8UC1, Scalar(GC_BGD));

    Rect sRect;
    sRect.x = (int)(faceRect.x * scale);
    sRect.y = (int)(faceRect.y * scale);
    sRect.width = (int)(faceRect.width * scale);
    sRect.height = (int)(faceRect.height * scale);

    double topMarginRatio = 0.6;
    double sideMarginRatio = 2.2;
    double bottomMarginRatio = 3.5;

    Rect bodyRect;

    int bodyWidth = (int)(sRect.width * sideMarginRatio);
    int bodyX = sRect.x + sRect.width / 2 - bodyWidth / 2;
    bodyRect.x = std::max(0, bodyX);
    bodyRect.width = std::min(smallSrc.cols - bodyRect.x, bodyWidth);

    int hairHeight = (int)(sRect.height * topMarginRatio);
    int bodyY = sRect.y - hairHeight;
    bodyRect.y = std::max(0, bodyY);

    int totalHeight = (int)(sRect.height * bottomMarginRatio) + hairHeight;
    bodyRect.height = std::min(smallSrc.rows - bodyRect.y, totalHeight);

    rectangle(gcMask, bodyRect, Scalar(GC_PR_FGD), -1);

    Rect centerFace = sRect;
    centerFace.x += sRect.width / 4;
    centerFace.y += sRect.height / 4;
    centerFace.width /= 2;
    centerFace.height /= 2;
    rectangle(gcMask, centerFace, Scalar(GC_FGD), -1);

    Rect centerBody;
    centerBody.x = sRect.x + sRect.width / 3;
    centerBody.width = sRect.width / 3;
    centerBody.y = sRect.y + sRect.height;
    centerBody.height = (int)(sRect.height * 1.5);

    if (centerBody.y < smallSrc.rows) {
        if (centerBody.y + centerBody.height > smallSrc.rows)
            centerBody.height = smallSrc.rows - centerBody.y;
        rectangle(gcMask, centerBody, Scalar(GC_FGD), -1);
    }

    Mat bgModel, fgModel;
    grabCut(smallSrc, gcMask, bodyRect, bgModel, fgModel, 1, GC_INIT_WITH_MASK);

    Mat smallResultMask;
    compare(gcMask & 1, 1, smallResultMask, CMP_EQ);
    resize(smallResultMask, outputMask, src.size(), 0, 0, INTER_NEAREST);
}
