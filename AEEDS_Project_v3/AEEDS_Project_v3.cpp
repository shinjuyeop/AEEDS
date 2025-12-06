// AEEDS_Project.cpp
// 1. Face & Body Segmentation using GrabCut with Auto-seeding.
// 2. Hand Gesture Recognition using Gradient Magnitude + Grid LBP.
// 3. Background Effects (Gaussian Blur, etc.) using manual C implementation.

#include <opencv2/opencv.hpp>
#include <cstdio>
#include <vector>
#include "MotionLBP.h"
#include "BackgroundEffects.h"
#include "Segmentation.h"
#include "Gesture.h"

using namespace cv;
using namespace std;

int main()
{
    // 1. 카메라 및 리소스 초기화
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        fprintf(stderr, "Camera open failed\n");
        return -1;
    }

	// 얼굴 검출기 로드
    CascadeClassifier faceCascade;
    string cascadePath = "C:/opencv/sources/data/lbpcascades/lbpcascade_frontalface.xml";
    if (!faceCascade.load(cascadePath)) {
        fprintf(stderr, "Failed to load cascade.\n");
        return -1;
    }

    // 배경 이미지 로드
    be_setBackgroundImage(imread("C:/Users/shinj/Desktop/3-2/AEEDS/Data/640x360.jpeg"));

    // 2. 변수 선언
    BgMode mode = BG_MODE_ORIGINAL;

    printf("Press ESC to exit.\n");
    printf("Press 'o':Original, 'g':Gray, 'b':Blur, 'm':Mosaic, 'i':Image to Enroll.\n");

    bool enrollOriginal = false, enrollGray = false, enrollBlur = false, enrollMosaic = false, enrollImage = false;

    // 템플릿 메모리 할당
    float* tplOriginal = (float*)malloc(sizeof(float) * TOTAL_LBP_DIM);
    float* tplGray = (float*)malloc(sizeof(float) * TOTAL_LBP_DIM);
    float* tplBlur = (float*)malloc(sizeof(float) * TOTAL_LBP_DIM);
    float* tplMosaic = (float*)malloc(sizeof(float) * TOTAL_LBP_DIM);
    float* tplImage = (float*)malloc(sizeof(float) * TOTAL_LBP_DIM);

    // 단일 프레임 등록 모드: 샘플 누적/평균 제거
    const float gestureThreshold = 120.0f; // 임계값
	int gestureCombo = 0;                  // 연속 인식 수
	int lastGestureIdx = -1;               // 마지막 인식된 제스처 인덱스
    const int targetCombo = 10;            // 10프레임 연속 인식 시 변경

    string persistTriggerText;
    int triggerFramesLeft = 0;              

    // 3. 메인 루프
    while (true) {
        Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        flip(frame, frame, 1);
        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        // 텍스트 변수 미리 선언
        string overlayScoreText;
        string overlayEnrollText; // 단일 등록 상태 표시에만 사용

        vector<Rect> faces;
        faceCascade.detectMultiScale(gray, faces, 1.2, 3, 0, Size(80, 80));

        Rect faceRect;
        bool haveFace = false;
        if (!faces.empty()) {
            faceRect = faces[0];
            for (size_t i = 1; i < faces.size(); ++i)
                if (faces[i].area() > faceRect.area()) faceRect = faces[i];
            haveFace = true;
        }

        // [Segmentation] GrabCut with Auto-Seed
        Mat fgMask;
        if (haveFace) {
            seg_getForegroundMaskWithGrabCut(frame, fgMask, faceRect);
        }
        else {
            fgMask = Mat::zeros(frame.size(), CV_8UC1);
        }

        // [Gesture Recognition]
        float* handFeature = nullptr;
        Rect handROI;
        bool shouldDrawHandROI = false;

        if (haveFace) {
            // 1. 손 ROI 계산
            if (gest_computeHandROI(faceRect, frame.size(), handROI)) {
                shouldDrawHandROI = true;

                Mat handGray = gray(handROI);

                // 2. 에지(Magnitude) 영상 생성
                Mat handEdge;
                calc_gradient_magnitude(handGray, handEdge);

                // [디버깅] 에지 영상 확인
                imshow("Edge", handEdge);

                // 3. 유효성 검사 (Edge Density Check)
                double edgeSum = 0.0;
                for (int y = 0; y < handEdge.rows; ++y) {
                    uchar* ptr = handEdge.ptr<uchar>(y);
                    for (int x = 0; x < handEdge.cols; ++x) {
                        edgeSum += ptr[x];
                    }
                }
                double edgeDensity = edgeSum / (double)(handEdge.rows * handEdge.cols);

                // 밀도가 일정 이상일 때만 손으로 간주
                if (edgeDensity >= 15.0) {
                    handFeature = (float*)malloc(sizeof(float) * TOTAL_LBP_DIM);
                    if (handFeature) {
                        memset(handFeature, 0, sizeof(float) * TOTAL_LBP_DIM);
                        // 4. Grid LBP 특징 추출
                        LBPdescriptor_Gesture(handEdge, handFeature);
                    }
                }
            }
        }

        // [Enrollment Logic - Single Frame]
        if (enrollOriginal || enrollGray || enrollBlur || enrollMosaic || enrollImage) {
            if (handFeature) {
                if (enrollOriginal)      memcpy(tplOriginal, handFeature, sizeof(float) * TOTAL_LBP_DIM);
                else if (enrollGray)     memcpy(tplGray, handFeature, sizeof(float) * TOTAL_LBP_DIM);
                else if (enrollBlur)     memcpy(tplBlur, handFeature, sizeof(float) * TOTAL_LBP_DIM);
                else if (enrollMosaic)   memcpy(tplMosaic, handFeature, sizeof(float) * TOTAL_LBP_DIM);
                else if (enrollImage)    memcpy(tplImage, handFeature, sizeof(float) * TOTAL_LBP_DIM);

                printf("Template Saved!\n");

                enrollOriginal = enrollGray = enrollBlur = enrollMosaic = enrollImage = false;
            }

            string status = "Enroll ";
            if (enrollOriginal)      status += "Original";
            else if (enrollGray)     status += "Gray";
            else if (enrollBlur)     status += "Blur";
            else if (enrollMosaic)   status += "Mosaic";
            else if (enrollImage)    status += "Image";

            overlayEnrollText = status;
        }

        // [Recognition Logic]
        if (!enrollOriginal && !enrollGray && !enrollBlur && !enrollMosaic && !enrollImage && handFeature) {
            float simO = mlbp_cosineSimilarityExp(handFeature, tplOriginal, TOTAL_LBP_DIM);
            float simG = mlbp_cosineSimilarityExp(handFeature, tplGray, TOTAL_LBP_DIM);
            float simB = mlbp_cosineSimilarityExp(handFeature, tplBlur, TOTAL_LBP_DIM);
            float simM = mlbp_cosineSimilarityExp(handFeature, tplMosaic, TOTAL_LBP_DIM);
            float simI = mlbp_cosineSimilarityExp(handFeature, tplImage, TOTAL_LBP_DIM);

            // 텍스트 저장
            overlayScoreText = format("O:%.0f G:%.0f B:%.0f M:%.0f I:%.0f", simO, simG, simB, simM, simI);

            float bestScore = 0.0f; int bestIdx = -1;
            if (simO > bestScore) { bestScore = simO; bestIdx = 0; }
            if (simG > bestScore) { bestScore = simG; bestIdx = 1; }
            if (simB > bestScore) { bestScore = simB; bestIdx = 2; }
            if (simM > bestScore) { bestScore = simM; bestIdx = 3; }
            if (simI > bestScore) { bestScore = simI; bestIdx = 4; }

            if (bestScore > gestureThreshold) {
                if (bestIdx == lastGestureIdx) gestureCombo++;
                else { gestureCombo = 1; lastGestureIdx = bestIdx; }
            }
            else { gestureCombo = 0; lastGestureIdx = -1; }

            if (gestureCombo == targetCombo) {
                if (bestIdx == 0) { mode = BG_MODE_ORIGINAL; persistTriggerText = "Gesture -> Original"; }
                else if (bestIdx == 1) { mode = BG_MODE_GRAY; persistTriggerText = "Gesture -> Gray"; }
                else if (bestIdx == 2) { mode = BG_MODE_BLUR; persistTriggerText = "Gesture -> Blur"; }
                else if (bestIdx == 3) { mode = BG_MODE_MOSAIC; persistTriggerText = "Gesture -> Mosaic"; }
                else if (bestIdx == 4) { mode = BG_MODE_IMAGE; persistTriggerText = "Gesture -> Photo"; }
                triggerFramesLeft = 30;
            }
        }

        if (handFeature) { free(handFeature); handFeature = nullptr; }

        // [Rendering: 배경 효과 적용]
        Mat output;
        if (haveFace) be_applyBackgroundEffectMask(frame, output, fgMask, mode);
        else output = frame.clone();

        // [Rendering: UI 및 텍스트]
        string modeStr;
        switch (mode) {
        case BG_MODE_ORIGINAL: modeStr = "Mode: Original"; break;
        case BG_MODE_GRAY:     modeStr = "Mode: Gray"; break;
        case BG_MODE_BLUR:     modeStr = "Mode: Blur"; break;
        case BG_MODE_MOSAIC:   modeStr = "Mode: Mosaic"; break;
        case BG_MODE_IMAGE:    modeStr = "Mode: Photo"; break;
        }
        
        putText(output, modeStr, Point(10, 25), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);

        if (haveFace)
            putText(output, "Show Hand Gesture", Point(10, 50), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 200, 0), 2);
        
        if (!overlayScoreText.empty()) {
            int margin = 10; int baseline = 0;
            Size scoreSize = getTextSize(overlayScoreText, FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseline);
            putText(output, overlayScoreText, Point(output.cols - scoreSize.width - margin, margin + scoreSize.height), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 255), 2);
        }
        
        if (triggerFramesLeft > 0 && !persistTriggerText.empty()) {
            int marginBR = 10; int bl = 0;
            Size tsize = getTextSize(persistTriggerText, FONT_HERSHEY_SIMPLEX, 0.8, 2, &bl);
            putText(output, persistTriggerText, Point(output.cols - tsize.width - marginBR, output.rows - marginBR), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);
            --triggerFramesLeft;
        }
        
        if (!overlayEnrollText.empty()) {
            int bottomMargin = 20;
            putText(output, overlayEnrollText, Point(10, output.rows - bottomMargin), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 2);
        }
        
        if (shouldDrawHandROI) {
            rectangle(output, handROI, Scalar(255, 0, 0), 2);
        }

        imshow("AEEDS PhotoBooth", output);

        // 키 입력 처리
        int key = waitKey(1);
		if (key == 27) break;   // ESC 키

        if (key == 'o') { enrollOriginal = !enrollOriginal; }
        if (key == 'g') { enrollGray = !enrollGray;         }
        if (key == 'b') { enrollBlur = !enrollBlur;         }
        if (key == 'm') { enrollMosaic = !enrollMosaic;     }
        if (key == 'i') { enrollImage = !enrollImage;       }
    }

    // [Cleanup]
    free(tplOriginal); free(tplGray); free(tplBlur); free(tplMosaic); free(tplImage);

    return 0;
}