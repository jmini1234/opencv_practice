#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;

template <typename T>
Mat myrotate(const Mat input, float angle, const char* opt);

int main()
{
	Mat input, rotated;
	
	// Read each image
	input = imread("lena.jpg");

	// Check for invalid input
	if (!input.data) {
		std::cout << "Could not open or find the image" << std::endl;
		return -1;
	}
	
	// original image
	namedWindow("image");
	imshow("image", input);

	rotated = myrotate<Vec3b>(input, 45, "bilinear");

	// rotated image
	namedWindow("rotated");
	imshow("rotated", rotated);

	waitKey(0);

	return 0;
}

template <typename T>
Mat myrotate(const Mat input, float angle, const char* opt) {
	int row = input.rows;
	int col = input.cols;

	float radian = angle * CV_PI / 180;

	// ceil - 올림
	// floor - 내림
	float sq_row = ceil(row * sin(radian) + col * cos(radian));
	float sq_col = ceil(col * sin(radian) + row * cos(radian));

	Mat output = Mat::zeros(sq_row, sq_col, input.type());

	for (int i = 0; i < sq_row; i++) {
		for (int j = 0; j < sq_col; j++) {
			float x = (j - sq_col / 2) * cos(radian) - (i - sq_row / 2) * sin(radian) + col / 2;
			float y = (j - sq_col / 2) * sin(radian) + (i - sq_row / 2) * cos(radian) + row / 2;

			if ((y >= 0) && (y <= (row - 1)) && (x >= 0) && (x <= (col - 1))) {
				if (!strcmp(opt, "nearest")) {
					 //가장 가까운 값을 찾아야 하기 때문에 input의 반올림 값을 output으로 설정한다. 
					int x1 = floor(x + 0.5); 
					int y1 = floor(y + 0.5);

					output.at<Vec3b>(i, j) = input.at<Vec3b>(y1, x1);

				}
				else if (!strcmp(opt, "bilinear")) {
					float x1 = floor(x);
					float x2 = ceil(x);
					float y1 = floor(y);
					float y2 = ceil(y);

					float u = y - y1;
					float al = x - x1;

					Vec3b P1 = u * input.at<Vec3b>(y2, x1) + (1 - u) * input.at<Vec3b>(y1, x1);
					Vec3b P2 = u * input.at<Vec3b>(y2, x2) + (1 - u) * input.at<Vec3b>(y1, x2);
					
					output.at<Vec3b>(i, j) = al * P2 + (1 - al) * P1;
				}
			}
		}
	}

	return output;
}
