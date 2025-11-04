#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>

using namespace cv;
using namespace cv::ml;
using namespace std;

void main() {
	CascadeClassifier cascade;
	cascade.load("C:/opencv/sources/data/lbpcascades/lbpcascade_frontalface.xml");
	Mat img = imread("C:/Users/shinj/Desktop/3-2/AEEDS/Data/face_tar.bmp", 1); // image read (color)
	vector<Rect> faces;

	cascade.detectMultiScale(img, faces, 1.1, 4, 0 | CV_HAAR_SCALE_IMAGE, Size(10, 10));

	int y;
	for (y = 0; y < faces.size(); y++)
	{
		Point lb(faces[y].x + faces[y].width, faces[y].y + faces[y].height);
		Point tr(faces[y].x, faces[y].y);
		rectangle(img, lb, tr, Scalar(0, 255, 0), 3, 8, 0);
	}
	imshow("Face", img);
	imwrite("Face.bmp", img);
	waitKey(50000);
}