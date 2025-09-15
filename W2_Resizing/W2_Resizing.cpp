// Bi-linear Resizing
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
int main()
{
	Mat imgColor = imread(
		"C:/Users/shinj/Desktop/3-2/AEEDS/AEEDS/data/test.jpg",
		IMREAD_COLOR); // 1이면 color 2면 gray

	int height, width, nh, nw;
	height = imgColor.rows;
	width = imgColor.cols;

	float scale_factor = 0.3;
	nh = (int)(scale_factor * height);
	nw = (int)(scale_factor * width);

	Mat result(nh, nw, CV_8UC3);

	for (int y = 0; y < nh; y++) {
		for (int x = 0; x < nw; x++) {
			float nx = x / scale_factor;
			float ny = y / scale_factor;

			int x0 = (int)nx, x1 = x0 + 1;
			int y0 = (int)ny, y1 = y0 + 1;

			// 경계 처리
			if (x1 >= width) x1 = width - 1;
			if (y1 >= height) y1 = height - 1;

			float dx = nx - x0, dy = ny - y0;

			// color image (BGR)
			for (int c = 0; c < 3; ++c) {
				float v00 = imgColor.at<Vec3b>(y0, x0)[c];
				float v01 = imgColor.at<Vec3b>(y1, x0)[c];
				float v10 = imgColor.at<Vec3b>(y0, x1)[c];
				float v11 = imgColor.at<Vec3b>(y1, x1)[c];

				float value = v00 * (1 - dx) * (1 - dy)
					+ v10 * dx * (1 - dy)
					+ v01 * (1 - dx) * dy
					+ v11 * dx * dy;

				result.at<Vec3b>(y, x)[c] = (uchar)value;
			}
		}
	}

	imshow("Bi-linear.bmp", result);
	waitKey(0);

	imwrite("Bi-linear.bmp", result);

	return 0;
}