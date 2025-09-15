#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
int main()
{
	Mat imgColor = imread(
		"C:/Users/shinj/Desktop/3-2/AEEDS/AEEDS/data/test.jpg",
		IMREAD_COLOR); // 1이면 color 2면 grey

	int height, width;
	height = imgColor.rows;
	width = imgColor.cols;

	Mat result(height, width, CV_8UC3);
	result = imgColor.clone();

	int x, y, cx, cy, BLK;
	float rVal, gVal, bVal;
	float avgVal;

	cx = width / 2;
	cy = height / 2;
	BLK = 100;

	for (y = cy; y < cy + BLK; y++) {
		for (x = cx; x < cx + BLK; x++) {
			rVal = imgColor.at<Vec3b>(y, x)[2];
			gVal = imgColor.at<Vec3b>(y, x)[1];
			bVal = imgColor.at<Vec3b>(y, x)[0];

			avgVal = (rVal + gVal + bVal) / 3;

			result.at<Vec3b>(y, x)[2] = avgVal;
			result.at<Vec3b>(y, x)[1] = avgVal;
			result.at<Vec3b>(y, x)[0] = avgVal;
		}
	}
	
	imwrite("result.bmp", result);

	return 0;
}