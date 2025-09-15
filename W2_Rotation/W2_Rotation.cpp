// Rotation with Bilinear
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
int main()
{
	Mat imgColor = imread(
		"C:/Users/shinj/Desktop/3-2/AEEDS/AEEDS/data/test.jpg",
		IMREAD_COLOR); // 1이면 color 2면 gray

	int height, width;
	height = imgColor.rows;
	width = imgColor.cols;

	float scale_factor = 90; // degree
	float s = scale_factor * 3.141592 / 180.0; // degree -> radian
	float matrix[2][2] = { {cos(s), -sin(s)}, {sin(s), cos(s)} };
	float det = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];

	Mat result(height, width, CV_8UC3);

	// 이미지의 중심 좌표
	int cx = width / 2;
	int cy = height / 2;

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			// 중심으로 이동
			float x_shifted = x - cx;
			float y_shifted = y - cy;

			// 역회전 변환
			float nx = 1 / det * (matrix[1][1] * x_shifted - matrix[0][1] * y_shifted);
			float ny = 1 / det * (-matrix[1][0] * x_shifted + matrix[0][0] * y_shifted);

			// 원래 좌표로 복원
			nx += cx;
			ny += cy;

			// 범위 벗어나면 0으로
			if (nx < 0 || nx >= width - 1 || ny < 0 || ny >= height - 1) {
				result.at<Vec3b>(y, x) = Vec3b(0, 0, 0);
				continue;
			}

			int x0 = (int)nx, x1 = x0 + 1;
			int y0 = (int)ny, y1 = y0 + 1;
			if (x1 >= width) x1 = width - 1;
			if (y1 >= height) y1 = height - 1;
			float dx = nx - x0, dy = ny - y0;

			for (int c = 0; c < 3; ++c) {
				float v00 = imgColor.at<Vec3b>(y0, x0)[c];
				float v01 = imgColor.at<Vec3b>(y1, x0)[c];
				float v10 = imgColor.at<Vec3b>(y0, x1)[c];
				float v11 = imgColor.at<Vec3b>(y1, x1)[c];

				float value = v00 * (1 - dx) * (1 - dy)
						+ v10 * dx * (1 - dy)
						+ v01 * (1 - dx) * dy
						+ v11 * dx * dy;

				result.at<Vec3b>(y, x)[c] = uchar(value);
			}
		}
	}

	imshow("Rotation.bmp", result);
	waitKey(0);

	imwrite("Rotation.bmp", result);

	return 0;
}