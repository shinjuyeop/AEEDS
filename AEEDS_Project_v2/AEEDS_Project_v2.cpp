// AEEDS_Project.cpp : Motion-gesture triggered background change selfie booth prototype.
// Requirements: OpenCV installed and accessible via include/lib paths.
// Features: 
// 1. Face & Body Segmentation using GrabCut (Auto-seed).
// 2. Hand Gesture Recognition using LBP (Texture-based).
// 3. Background Effect (Original / Gray / Blur / Mosaic / Background Image)
//    - Effects are applied ONLY to background pixels (person is kept as original).
//    - Gray / Blur / Mosaic are implemented manually in C-style loops.

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
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        fprintf(stderr, "Camera open failed\n");
        return -1;
    }

    CascadeClassifier faceCascade;
    string cascadePath = "C:/opencv/sources/data/lbpcascades/lbpcascade_frontalface.xml";
    if (!faceCascade.load(cascadePath)) {
        fprintf(stderr, "Failed to load cascade: %s\n", cascadePath.c_str());
        return -1;
    }

    BgMode mode = BgMode::Original;
    const int cooldownFrames = 15;
    int framesSinceTrigger = cooldownFrames;

    printf("Press ESC to exit.\n");
    printf("Press 'o' to enroll Original, 'g' for Gray, 'b' for Blur, 'm' for Mosaic, 'i' for Image.\n");

    be_setBackgroundImage(imread("C:/Users/shinj/Desktop/3-2/AEEDS/Data/640x360.jpeg"));

    bool enrollOriginal = false, enrollGray = false, enrollBlur = false, enrollMosaic = false, enrollImage = false;
    // C-style storage for samples and templates
    float* tplOriginal = (float*)malloc(sizeof(float) * ULBP_BIN);
    float* tplGray = (float*)malloc(sizeof(float) * ULBP_BIN);
    float* tplBlur = (float*)malloc(sizeof(float) * ULBP_BIN);
    float* tplMosaic = (float*)malloc(sizeof(float) * ULBP_BIN);
    float* tplImage = (float*)malloc(sizeof(float) * ULBP_BIN);
    if (!tplOriginal || !tplGray || !tplBlur || !tplMosaic || !tplImage) {
        fprintf(stderr, "Failed to allocate template buffers\n");
        return -1;
    }

    // dynamic sample arrays (C-style arrays of float* with manual resize)
    float** samplesOriginal = nullptr; int samplesOriginalCount = 0; int samplesOriginalCap = 0;
    float** samplesGray = nullptr; int samplesGrayCount = 0; int samplesGrayCap = 0;
    float** samplesBlur = nullptr; int samplesBlurCount = 0; int samplesBlurCap = 0;
    float** samplesMosaic = nullptr; int samplesMosaicCount = 0; int samplesMosaicCap = 0;
    float** samplesImage = nullptr; int samplesImageCount = 0; int samplesImageCap = 0;

    auto push_sample = [](float*** arr, int* count, int* cap, float* elem) {
        if (!elem) return;
        if (*count >= *cap) {
            int newCap = (*cap == 0) ? 32 : (*cap * 2);
            float** newArr = (float**)realloc((*arr), sizeof(float*) * newCap);
            if (!newArr) return; // allocation failed, drop sample
            *arr = newArr; *cap = newCap;
        }
        (*arr)[(*count)++] = elem;
        };

    auto clear_samples = [](float** arr, int* count) {
        if (!arr) { *count = 0; return; }
        for (int i = 0; i < *count; ++i) if (arr[i]) free(arr[i]);
        *count = 0;
        };

    int enrollFrameCount = 0;
    const int enrollFramesTarget = 50;
    int startupFrames = 0;
    const float gestureThreshold = 125.0f;
    int gestureCombo = 0;
    int lastGestureIdx = -1;
    const int targetCombo = 10;

    string persistTriggerText;
    int triggerFramesLeft = 0;

    while (true) {
        Mat frame;
        cap >> frame;
        if (frame.empty()) break;
        ++startupFrames;

        flip(frame, frame, 1);
        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        string overlayScoreText;
        string overlayEnrollText;

        vector<Rect> faces;
        faceCascade.detectMultiScale(gray, faces, 1.2, 3, 0, Size(80, 80));

        Rect faceRect;
        bool haveFace = false;
        if (!faces.empty()) {
            faceRect = faces[0];
            for (size_t i = 1; i < faces.size(); ++i)
                if (faces[i].area() > faceRect.area())
                    faceRect = faces[i];
            haveFace = true;
        }

        Mat fgMask;
        if (haveFace) {
            seg_getForegroundMaskWithGrabCut(frame, fgMask, faceRect);
        }
        else {
            fgMask = Mat::zeros(frame.size(), CV_8UC1);
        }

        
        float* handFeature = nullptr;
        Rect handROI;
        bool shouldDrawHandROI = false;
        /* 
        if (haveFace) {
            if (gest_computeHandROI(faceRect, frame.size(), handROI)) {
                shouldDrawHandROI = true;
                Mat handGray = gray(handROI);
                // allocate feature and extract
                handFeature = (float*)malloc(sizeof(float) * ULBP_BIN);
                if (handFeature) {
                    memset(handFeature, 0, sizeof(float) * ULBP_BIN);
                    mlbp_uniformLBPHistogram(handGray, handFeature);
                }
            }
        }
        */

        /*
        if (haveFace) {
            // 1. 먼저 손 영역(ROI)을 계산 (이 함수가 true를 반환해야 사각형이 그려짐)
            if (gest_computeHandROI(faceRect, frame.size(), handROI)) {
                shouldDrawHandROI = true; // 여기서 true로 바뀜

                // 2. ROI 영역만 잘라내기
                Mat handGray = gray(handROI);

                // 3. [추가된 부분] 에지 크기(Magnitude) 영상 생성
                Mat handEdge;
                calc_gradient_magnitude(handGray, handEdge);
                imshow("Debug Edge", handEdge);

                // 4. [수정된 부분] 에지 영상을 LBP의 입력으로 사용
                handFeature = (float*)malloc(sizeof(float) * ULBP_BIN);
                if (handFeature) {
                    memset(handFeature, 0, sizeof(float) * ULBP_BIN);

                    // handGray 대신 handEdge를 넣음
                    mlbp_uniformLBPHistogram(handEdge, handFeature);
                }
            }
        }
        */

        // 에지 밀도 검사를 추가하여 오작동 방지
        if (haveFace) {
            // 1. 손 영역(ROI)을 계산
            if (gest_computeHandROI(faceRect, frame.size(), handROI)) {
                shouldDrawHandROI = true;

                // 2. ROI 영역만 잘라내기
                Mat handGray = gray(handROI);

                // 3. 에지 크기(Magnitude) 영상 생성
                Mat handEdge;
                calc_gradient_magnitude(handGray, handEdge);

                // [디버깅] 에지 영상 확인용 (필요 없으면 주석 처리)
                imshow("Debug Edge", handEdge);

                // [추가됨] 3-1. 유효성 검사: 에지 밀도(Density) 계산
                // ROI 안에 에지(흰색 선)가 얼마나 많은지 평균을 냅니다.
                double edgeSum = 0.0;
                for (int y = 0; y < handEdge.rows; ++y) {
                    uchar* ptr = handEdge.ptr<uchar>(y);
                    for (int x = 0; x < handEdge.cols; ++x) {
                        edgeSum += ptr[x];
                    }
                }
                double edgeDensity = edgeSum / (double)(handEdge.rows * handEdge.cols);

                // [디버깅] 콘솔창에 수치를 찍어서 확인해보세요.
                // 벽일 때: 5.0 미만 / 손일 때: 30.0 이상 나오는지 확인 필요
                //printf("Current Edge Density: %.2f\n", edgeDensity);

                // [핵심] 밀도가 임계값(예: 15.0) 이상일 때만 특징 추출 수행!
                // *주의* 벽을 비출 때 Density가 5~10 정도 나온다면, 이 값을 15~20으로 올리세요.
                if (edgeDensity >= 10.0) {
                    // 4. 에지 영상을 LBP의 입력으로 사용
                    handFeature = (float*)malloc(sizeof(float) * ULBP_BIN);

                    if (handFeature) {
                        // 3. 수정된 함수 호출 (handEdge를 넣으면 내부에서 리사이즈 후 처리함)
                        LBPdescriptor_Gesture(handEdge, handFeature);
                    }
                }
                else {
                    // 밀도가 낮으면(벽이면) handFeature를 할당하지 않음 (nullptr 유지)
                    // -> 결과적으로 유사도 계산이 스킵되어 점수가 0이 됨.
                }
            }
        }

        if (enrollOriginal || enrollGray || enrollBlur || enrollMosaic || enrollImage) {
            if (handFeature) {
                if (enrollOriginal)      push_sample(&samplesOriginal, &samplesOriginalCount, &samplesOriginalCap, handFeature);
                else if (enrollGray)     push_sample(&samplesGray, &samplesGrayCount, &samplesGrayCap, handFeature);
                else if (enrollBlur)     push_sample(&samplesBlur, &samplesBlurCount, &samplesBlurCap, handFeature);
                else if (enrollMosaic)   push_sample(&samplesMosaic, &samplesMosaicCount, &samplesMosaicCap, handFeature);
                else if (enrollImage)    push_sample(&samplesImage, &samplesImageCount, &samplesImageCap, handFeature);
                ++enrollFrameCount;
                // transfer ownership, allocate a fresh buffer for next frame
                handFeature = nullptr;
            }
            else {
                // if not used for enrollment, free
                if (handFeature) { free(handFeature); handFeature = nullptr; }
            }

            string status = "Enroll ";
            if (enrollOriginal)      status += "Original";
            else if (enrollGray)     status += "Gray";
            else if (enrollBlur)     status += "Blur";
            else if (enrollMosaic)   status += "Mosaic";
            else if (enrollImage)    status += "Image";

            overlayEnrollText = format("%s %d/%d", status.c_str(), enrollFrameCount, enrollFramesTarget);

            if (enrollFrameCount >= enrollFramesTarget) {
                if (enrollOriginal) { mlbp_averageTemplates(samplesOriginal, samplesOriginalCount, tplOriginal); printf("Original template saved!\n"); }
                if (enrollGray) { mlbp_averageTemplates(samplesGray, samplesGrayCount, tplGray);     printf("Gray template saved!\n"); }
                if (enrollBlur) { mlbp_averageTemplates(samplesBlur, samplesBlurCount, tplBlur);     printf("Blur template saved!\n"); }
                if (enrollMosaic) { mlbp_averageTemplates(samplesMosaic, samplesMosaicCount, tplMosaic);   printf("Mosaic template saved!\n"); }
                if (enrollImage) { mlbp_averageTemplates(samplesImage, samplesImageCount, tplImage);    printf("Image template saved!\n"); }
                enrollOriginal = enrollGray = enrollBlur = enrollMosaic = enrollImage = false;
                enrollFrameCount = 0;
                clear_samples(samplesOriginal, &samplesOriginalCount);
                clear_samples(samplesGray, &samplesGrayCount);
                clear_samples(samplesBlur, &samplesBlurCount);
                clear_samples(samplesMosaic, &samplesMosaicCount);
                clear_samples(samplesImage, &samplesImageCount);
            }
        }

        if (!enrollOriginal && !enrollGray && !enrollBlur && !enrollMosaic && !enrollImage && handFeature) {
            float simO = tplOriginal ? mlbp_cosineSimilarityExp(handFeature, tplOriginal) : 0.0f;
            float simG = tplGray ? mlbp_cosineSimilarityExp(handFeature, tplGray) : 0.0f;
            float simB = tplBlur ? mlbp_cosineSimilarityExp(handFeature, tplBlur) : 0.0f;
            float simM = tplMosaic ? mlbp_cosineSimilarityExp(handFeature, tplMosaic) : 0.0f;
            float simI = tplImage ? mlbp_cosineSimilarityExp(handFeature, tplImage) : 0.0f;

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
                if (bestIdx == 0) { mode = BgMode::Original; persistTriggerText = "Gesture -> Original"; }
                else if (bestIdx == 1) { mode = BgMode::Gray; persistTriggerText = "Gesture -> Gray"; }
                else if (bestIdx == 2) { mode = BgMode::Blur; persistTriggerText = "Gesture -> Blur"; }
                else if (bestIdx == 3) { mode = BgMode::Mosaic; persistTriggerText = "Gesture -> Mosaic"; }
                else if (bestIdx == 4) { mode = BgMode::Image; persistTriggerText = "Gesture -> Photo"; }
                triggerFramesLeft = 30;
            }
        }
        framesSinceTrigger++;

        // free handFeature if not consumed by enrollment
        if (handFeature) { free(handFeature); handFeature = nullptr; }

        Mat output;
        if (haveFace) be_applyBackgroundEffectMask(frame, output, fgMask, mode);
        else output = frame.clone();

        string modeStr;
        switch (mode) {
        case BgMode::Original: modeStr = "Mode: Original"; break;
        case BgMode::Gray:     modeStr = "Mode: Gray"; break;
        case BgMode::Blur:     modeStr = "Mode: Blur"; break;
        case BgMode::Mosaic:   modeStr = "Mode: Mosaic"; break;
        case BgMode::Image:    modeStr = "Mode: Photo"; break;
        }
        putText(output, modeStr, Point(10, 25), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);

        if (haveFace)
            putText(output, "Show Hand Gesture to Change Background", Point(10, 50), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 200, 0), 2);

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

        int key = waitKey(1);
        if (key == 27) break;

        if (key == 'o' && !enrollGray && !enrollBlur && !enrollMosaic && !enrollImage) {
            enrollOriginal = !enrollOriginal; enrollFrameCount = 0; clear_samples(samplesOriginal, &samplesOriginalCount);
            printf("%s\n", (enrollOriginal ? "Enroll Original start" : "Enroll Original stop"));
        }
        if (key == 'g' && !enrollOriginal && !enrollBlur && !enrollMosaic && !enrollImage) {
            enrollGray = !enrollGray; enrollFrameCount = 0; clear_samples(samplesGray, &samplesGrayCount);
            printf("%s\n", (enrollGray ? "Enroll Gray start" : "Enroll Gray stop"));
        }
        if (key == 'b' && !enrollOriginal && !enrollGray && !enrollMosaic && !enrollImage) {
            enrollBlur = !enrollBlur; enrollFrameCount = 0; clear_samples(samplesBlur, &samplesBlurCount);
            printf("%s\n", (enrollBlur ? "Enroll Blur start" : "Enroll Blur stop"));
        }
        if (key == 'm' && !enrollOriginal && !enrollGray && !enrollBlur && !enrollImage) {
            enrollMosaic = !enrollMosaic; enrollFrameCount = 0; clear_samples(samplesMosaic, &samplesMosaicCount);
            printf("%s\n", (enrollMosaic ? "Enroll Mosaic start" : "Enroll Mosaic stop"));
        }
        if (key == 'i' && !enrollOriginal && !enrollGray && !enrollBlur && !enrollMosaic) {
            enrollImage = !enrollImage; enrollFrameCount = 0; clear_samples(samplesImage, &samplesImageCount);
            printf("%s\n", (enrollImage ? "Enroll Image start" : "Enroll Image stop"));
        }
    }

    // cleanup
    free(tplOriginal); free(tplGray); free(tplBlur); free(tplMosaic); free(tplImage);

    // free any remaining sample buffers and the arrays themselves
    clear_samples(samplesOriginal, &samplesOriginalCount);
    clear_samples(samplesGray, &samplesGrayCount);
    clear_samples(samplesBlur, &samplesBlurCount);
    clear_samples(samplesMosaic, &samplesMosaicCount);
    clear_samples(samplesImage, &samplesImageCount);
    free(samplesOriginal); free(samplesGray); free(samplesBlur); free(samplesMosaic); free(samplesImage);

    return 0;
}
