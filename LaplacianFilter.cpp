#include <iostream>
#include <opencv2/opencv.hpp>
#include <math.h>       /* exp */
#define IM_TYPE	CV_8UC3

using namespace cv;

// Image Type
// "G" for GrayScale Image, "C" for Color Image
#if (IM_TYPE == CV_8UC3)
typedef uchar G;
typedef cv::Vec3b C;
#elif (IM_TYPE == CV_16SC3)
typedef short G;
typedef Vec3s C;
#elif (IM_TYPE == CV_32SC3)
typedef int G;
typedef Vec3i C;
#elif (IM_TYPE == CV_32FC3)
typedef float G;
typedef Vec3f C;
#elif (IM_TYPE == CV_64FC3)
typedef double G;
typedef Vec3d C;
#endif

Mat laplacianfilter(const Mat input);
Mat laplacianfilter_RGB(const Mat input);
float visualization(float val);

int main() {

	Mat input = imread("lena.jpg", CV_LOAD_IMAGE_COLOR);
	Mat input_gray;
	Mat output,output_RGB;

	cvtColor(input, input_gray, CV_RGB2GRAY);

	if (!input.data)
	{
		std::cout << "Could not open" << std::endl;
		return -1;
	}

	namedWindow("RGB", WINDOW_AUTOSIZE);
	imshow("RGB", input);

	output_RGB = laplacianfilter_RGB(input); //Boundary process: mirroring
	namedWindow("Laplacian Filter RGB", WINDOW_AUTOSIZE);
	imshow("Laplacian Filter RGB", output_RGB);

	namedWindow("Grayscale", WINDOW_AUTOSIZE);
	imshow("Grayscale", input_gray);

	output = laplacianfilter(input_gray); //Boundary process: mirroring
	namedWindow("Laplacian Filter", WINDOW_AUTOSIZE);
	imshow("Laplacian Filter", output);

	waitKey(0);

	return 0;
}

float visualization(float val) {
	if (val < 0)
		val = 0;
	else
		val *= 8;

	if (val > 255)
		val = 255;

	return val;
}

Mat laplacianfilter(const Mat input) {

	int row = input.rows;
	int col = input.cols;
	int n = 1; // Sobel Filter Kernel N

	Mat O(3, 3, CV_32F);

	int arr_o[3][3] = {
		{ 0,1,0 },
		{ 1,-4,1 },
		{ 0,1,0 }
	};

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			O.at<float>(i, j) = arr_o[i][j];
		}
	}

	Mat output = Mat::zeros(row, col, input.type());

	int tempa;
	int tempb;

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			float sum1 = 0.0;
			for (int a = -n; a <= n; a++) {
				for (int b = -n; b <= n; b++) {
					// Use mirroring boundary process
					if (i + a > row - 1) {  //mirroring for the border pixels
						tempa = i - a;
					}
					else if (i + a < 0) {
						tempa = -(i + a);
					}
					else {
						tempa = i + a;
					}
					if (j + b > col - 1) {
						tempb = j - b;
					}
					else if (j + b < 0) {
						tempb = -(j + b);
					}
					else {
						tempb = j + b;
					}
					sum1 += O.at<float>(a + n, b + n)*(float)(input.at<G>(tempa, tempb));
				}
			}

			output.at<G>(i, j) = visualization(sum1);
		}
	}


	return output;
}

Mat laplacianfilter_RGB(const Mat input) {

	int row = input.rows;
	int col = input.cols;
	int n = 1; // Sobel Filter Kernel N

	Mat O(3, 3, CV_32F);

	int arr_o[3][3] = {
		{ 0,1,0 },
		{ 1,-4,1 },
		{ 0,1,0 }
	};

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			O.at<float>(i, j) = arr_o[i][j];
		}
	}

	Mat output = Mat::zeros(row, col, 0);

	int tempa;
	int tempb;

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			float sum1_r = 0.0;
			float sum1_g = 0.0;
			float sum1_b = 0.0;
			for (int a = -n; a <= n; a++) {
				for (int b = -n; b <= n; b++) {
					// Use mirroring boundary process
					if (i + a > row - 1) {  //mirroring for the border pixels
						tempa = i - a;
					}
					else if (i + a < 0) {
						tempa = -(i + a);
					}
					else {
						tempa = i + a;
					}
					if (j + b > col - 1) {
						tempb = j - b;
					}
					else if (j + b < 0) {
						tempb = -(j + b);
					}
					else {
						tempb = j + b;
					}
					sum1_r += (float)(O.at<float>(a + n, b + n))*(float)(input.at<C>(tempa, tempb)[0]);
					sum1_g += (float)(O.at<float>(a + n, b + n))*(float)(input.at<C>(tempa, tempb)[1]);
					sum1_b += (float)(O.at<float>(a + n, b + n))*(float)(input.at<C>(tempa, tempb)[2]);
				}
			}
			
			float result = (sum1_r + sum1_g + sum1_b) / 3;
			output.at<G>(i, j) = visualization(result);
		}
	}
	return output;
}
