#include <vector>
#include <iostream>
#include <fstream>
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "ldmarkmodel.h"
#include "LBP.h"

using namespace std;
using namespace cv;

int main()
{
	ldmarkmodel modelt;
    std::string modelFilePath = "roboman-landmark-model.bin";
    while (!load_ldmarkmodel(modelFilePath, modelt)) {
        std::cout << "파일 열기 오류, 경로 재입력." << std::endl;
        std::cin >> modelFilePath;
    }

    VideoCapture mCamera(0);
    if (!mCamera.isOpened()) {
        std::cout << "Camera opening failed..." << std::endl;
        system("pause");
        return 0;
    }
    Mat Image, gray;
    Mat current_shape;
    CascadeClassifier cascade;
    cascade.load("C:/opencv/sources/data/lbpcascades/lbpcascade_frontalface.xml");
    vector<Rect> faces;

    // 사용할 랜드마크 인덱스(eye, nose, mouth 등41개)
    vector<int> selected_indices = {
    36,37,38,39,40,41, // left eye
    42,43,44,45,46,47, // right eye
    27,28,29,30,31,32,33,34,35, // nose ridge & bottom
    48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67 // mouth
    };
     int window_size =16; //16x16 window
     vector<cv::Point> landmarks;
     vector<float> enroll_feature, test_feature;
     bool enrolled = false;
     int cnt =0;
     float th =60.0f; // threshold (exp 스케일 이후 값)

     while (true) {
            mCamera >> Image;
            if (Image.empty()) break;
            cvtColor(Image, gray, COLOR_BGR2GRAY);
            cascade.detectMultiScale(gray, faces,1.1,4,0 | CV_HAAR_SCALE_IMAGE, Size(100,100));

            // 얼굴 검출 시 rectangle 표시
            for (size_t i =0; i < faces.size(); ++i) {
                rectangle(Image, faces[i], Scalar(255,0,0),2);
            }

            // 랜드마크 추출 및 점 표시
            if (!faces.empty()) {
                modelt.track(Image, current_shape);
                int numLandmarks = current_shape.cols /2;
                landmarks.clear();
                for (int j =0; j < numLandmarks; j++) {
                    int x = current_shape.at<float>(j);
                    int y = current_shape.at<float>(j + numLandmarks);
                    landmarks.push_back(cv::Point(x, y));
                }
                // 특징점(랜드마크) 점으로 표시
                for (int idx : selected_indices) {
                    if (idx < landmarks.size())
                        cv::circle(Image, landmarks[idx],2, cv::Scalar(0,0,255), -1);
                }
            }

            // 등록(enroll): 얼굴1개 검출되고 등록 안 됐고 cnt >30
            if (faces.size() ==1 && !enrolled && cnt >30) {
                extractLandmarkLBPFeatures(Image, landmarks, selected_indices, window_size, enroll_feature);
                enrolled = true;
                std::cout << "[Enroll] Feature saved. Size: " << enroll_feature.size() << std::endl;
            }

            // 검증(verification): 등록 후 얼굴 검출 시
            if (enrolled && !faces.empty()) {
                extractLandmarkLBPFeatures(Image, landmarks, selected_indices, window_size, test_feature);
                float score = computeSimilarity(enroll_feature.data(), test_feature.data());
                // 맞으면 초록, 아니면 빨강
                for (size_t k =0; k < faces.size(); ++k) {
                    Point lb(faces[k].x + faces[k].width, faces[k].y + faces[k].height);
                    Point tr(faces[k].x, faces[k].y);
                    if (score > th)
                        rectangle(Image, lb, tr, Scalar(0,255,0),3,8,0);
                    else
                        rectangle(Image, lb, tr, Scalar(0,0,255),3,8,0);
                    putText(Image, format("score: %.2f", score), Point(faces[k].x, faces[k].y -10), FONT_HERSHEY_SIMPLEX,0.7, Scalar(255,255,0),2);
                }
            }

            imshow("Face", Image);
            if (waitKey(30) ==27) break;
            cnt++;
     }
     system("pause");
     return 0;
}

/*
// --------------- main ----------------
int main()
{
    int flag = 0;
    int cnt = 0;
    int k;
    float score;
    float th = 50.0f; // threshold (exp 스케일 이후 값)

    VideoCapture capture(0);
    Mat frame;
    if (!capture.isOpened()) {
        printf("Couldn’t open the web camera...\n");
        return -1;
    }

    CascadeClassifier cascade;
    cascade.load("C:/opencv/sources/data/lbpcascades/lbpcascade_frontalface.xml");
    vector<Rect> faces;

    float refLBPhist[DIM];
    float tarLBPhist[DIM];

    while (true) {
        capture >> frame;
        if (frame.empty()) break;

        // 1. 얼굴 검출
        cascade.detectMultiScale(frame, faces, 1.1, 4, 0 | CV_HAAR_SCALE_IMAGE, Size(100, 100));

        // 2. enrollment (ref)
        if (faces.size() == 1 && flag == 0 && cnt > 30) {
            Point lb(faces[0].x + faces[0].width,
                faces[0].y + faces[0].height);
            Point tr(faces[0].x, faces[0].y);

            rectangle(frame, lb, tr, Scalar(0, 255, 0), 3, 8, 0);

            // face verification reference feature 추출
            LBPdescriptor(frame, refLBPhist,
                faces[0].x, faces[0].y,
                faces[0].width, faces[0].height);
            flag = 1;
        }

        // 3. verification (tar)
        if (!faces.empty() && flag != 0 && cnt > 30) {
            for (k = 0; k < (int)faces.size(); ++k) {
                LBPdescriptor(frame, tarLBPhist,
                    faces[k].x, faces[k].y,
                    faces[k].width, faces[k].height);

                score = computeSimilarity(refLBPhist, tarLBPhist);
                printf("score : %f\n", score);

                Point lb(faces[k].x + faces[k].width,
                    faces[k].y + faces[k].height);
                Point tr(faces[k].x, faces[k].y);

                if (score > th)
                    rectangle(frame, lb, tr, Scalar(0, 255, 0), 3, 8, 0);
                else
                    rectangle(frame, lb, tr, Scalar(0, 0, 255), 3, 8, 0);
            }
        }

        imshow("Face", frame);
        if (waitKey(30) >= 0) break;
        cnt++;
    }

    return 0;
}
*/