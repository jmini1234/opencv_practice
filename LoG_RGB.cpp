// opencv_test.cpp : Defines the entry point for the console application.
//

//#include "stdafx.h"
#include <opencv2/opencv.hpp>
#include "opencv2/highgui.hpp"
#include <iostream>

using namespace cv;
using namespace std;
//ADD
// Image Type
// "G" for GrayScale Image, "C" for Color Image
#if (IM_TYPE == CV_8UC3)
typedef uchar G;
typedef cv::Vec3d C;
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

Mat get_Gaussian_Kernel(int n, double sigma_t, double sigma_s, bool normalize);
Mat get_Laplacian_Kernel();
Mat Gaussianfilter_RGB(const Mat input, int n, double sigma_t, double sigma_s);
Mat Laplacianfilter_RGB(const Mat input);
Mat Mirroring_RGB(const Mat input, int n);

float visualization(double val) {
	if (val < 0)
		val = 0.0;
	else
		val *= 10.0;

	if (val > 255)
		val = 255.0;

	return val;
}


int main() {

	Mat input = imread("lena.jpg", CV_LOAD_IMAGE_COLOR);
	Mat input_RGB;
	// check for validation
	if (!input.data) {
		printf("Could not open\n");
		return -1;
	}

	//Gaussian smoothing parameters
	int window_radius = 2;
	double sigma_t = 2.0;
	double sigma_s = 2.0;

	input.convertTo(input_RGB, CV_64FC3, 1.0 / 255);	// 8-bit unsigned char -> 64-bit floating point

	Mat h_f = Gaussianfilter_RGB(input_RGB, window_radius, sigma_t, sigma_s);	// h(x,y) * f(x,y)

	Mat Laplacian = Laplacianfilter_RGB(h_f);

	normalize(Laplacian, Laplacian, 0, 1, CV_MINMAX);

	namedWindow("RGB", WINDOW_AUTOSIZE);
	imshow("RGB", input);

	namedWindow("Gaussian blur", WINDOW_AUTOSIZE);
	imshow("Gaussian blur", h_f);

	namedWindow("Laplacian filter", WINDOW_AUTOSIZE);
	imshow("Laplacian filter", Laplacian);

	waitKey(0);

	return 0;
}



Mat Gaussianfilter_RGB(const Mat input, int n, double sigma_t, double sigma_s) {

	int row = input.rows;
	int col = input.cols;

	// generate gaussian kernel
	Mat kernel = get_Gaussian_Kernel(n, sigma_t, sigma_s, true);
	Mat output = Mat::zeros(row, col, input.type());
	int tempa;
	int tempb;

	//Intermediate data generation for mirroring
	Mat input_mirror = Mirroring_RGB(input, n);

	for (int i = n; i < row + n; i++) {
		for (int j = n; j < col + n; j++) {
			//float sum1_r = 0.0;
			//float sum1_g = 0.0;
			//float sum1_b = 0.0;
			//for (int x = -n; x <= n; x++) { // for each kernel window
			//	for (int y = -n; y <= n; y++) {
			//		if (i + x > row - 1) {  //mirroring for the border pixels
			//			tempa = i - x;
			//		}
			//		else if (i + x < 0) {
			//			tempa = -(i + x);
			//		}
			//		else {
			//			tempa = i + x;
			//		}
			//		if (j + y > col - 1) {
			//			tempb = j - y;
			//		}
			//		else if (j + y < 0) {
			//			tempb = -(j + y);
			//		}
			//		else {
			//			tempb = j + y;
			//		}
			//		sum1_r += (double)kernel.at<double>(x + n, y + n)*(double)input_mirror.at<Vec3d>(tempa, tempb)[0];
			//		sum1_g += (double)kernel.at<double>(x + n, y + n)*(double)input_mirror.at<Vec3d>(tempa, tempb)[1];
			//		sum1_g += (double)kernel.at<double>(x + n, y + n)*(double)input_mirror.at<Vec3d>(tempa, tempb)[2];
			//	}
			//}
			//output.at<Vec3d>(i - n, j - n)[0] = (double)sum1_r;
			//output.at<Vec3d>(i - n, j - n)[1] = (double)sum1_g;
			//output.at<Vec3d>(i - n, j - n)[2] = (double)sum1_g;
			double sum1_r = 0.0;
			double sum1_g = 0.0;
			double sum1_b = 0.0;
			for (int x = -n; x <= n; x++) { // for each kernel window
				for (int y = -n; y <= n; y++) {
					sum1_r += (double)kernel.at<double>(x + n, y + n)*(double)input_mirror.at<Vec3d>(i + x, j + y)[0];
					sum1_g += (double)kernel.at<double>(x + n, y + n)*(double)input_mirror.at<Vec3d>(i + x, j + y)[1];
					sum1_b += (double)kernel.at<double>(x + n, y + n)*(double)input_mirror.at<Vec3d>(i + x, j + y)[2];
				}
			}
			output.at<Vec3d>(i - n, j - n)[0] = (double)sum1_r;
			output.at<Vec3d>(i - n, j - n)[1] = (double)sum1_g;
			output.at<Vec3d>(i - n, j - n)[2] = (double)sum1_b;
		}
	}
	return output;
}

Mat Laplacianfilter_RGB(const Mat input) {

	int row = input.rows;
	int col = input.cols;

	Mat kernel = get_Laplacian_Kernel();
	Mat output = Mat::zeros(row, col, input.type());

	int n = 1;
	Mat input_mirror = Mirroring_RGB(input, n);

	for (int i = n; i < row + n; i++) {
		for (int j = n; j < col + n; j++) {
			float sum1_r = 0.0;
			float sum1_g = 0.0;
			float sum1_b = 0.0;
			for (int x = -n; x <= n; x++) {
				for (int y = -n; y <= n; y++) {
					sum1_r += (double)kernel.at<double>(x + n, y + n)*(double)input_mirror.at<Vec3d>(i + x, j + y)[0];
					sum1_g += (double)kernel.at<double>(x + n, y + n)*(double)input_mirror.at<Vec3d>(i + x, j + y)[1];
					sum1_b += (double)kernel.at<double>(x + n, y + n)*(double)input_mirror.at<Vec3d>(i + x, j + y)[2];
				}
			}
			output.at<Vec3d>(i - n, j - n)[0] = visualization((double)sum1_r);
			output.at<Vec3d>(i - n, j - n)[1] = visualization((double)sum1_g);
			output.at<Vec3d>(i - n, j - n)[2] = visualization((double)sum1_b);
		}
	}
	return output;
}

Mat Mirroring_RGB(const Mat input, int n)
{
	int row = input.rows;
	int col = input.cols;

	Mat input2 = Mat::zeros(row + 2 * n, col + 2 * n, input.type());
	int row2 = input2.rows;
	int col2 = input2.cols;
		for (int i = n; i < row + n; i++) {
			for (int j = n; j < col + n; j++) {
				input2.at<Vec3d>(i, j)[0] = input.at<Vec3d>(i - n, j - n)[0];
				input2.at<Vec3d>(i, j)[1] = input.at<Vec3d>(i - n, j - n)[1];
				input2.at<Vec3d>(i, j)[2] = input.at<Vec3d>(i - n, j - n)[2];
			}
		}
		for (int i = n; i < row + n; i++) {
			for (int j = 0; j < n; j++) {
				input2.at<Vec3d>(i, j)[0] = input2.at<Vec3d>(i, 2 * n - j)[0];
				input2.at<Vec3d>(i, j)[1] = input2.at<Vec3d>(i, 2 * n - j)[1];
				input2.at<Vec3d>(i, j)[2] = input2.at<Vec3d>(i, 2 * n - j)[2];
			}
			for (int j = col + n; j < col2; j++) {
				input2.at<Vec3d>(i, j)[0] = input2.at<Vec3d>(i, 2 * col - 2 + 2 * n - j)[0];
				input2.at<Vec3d>(i, j)[1] = input2.at<Vec3d>(i, 2 * col - 2 + 2 * n - j)[1];
				input2.at<Vec3d>(i, j)[2] = input2.at<Vec3d>(i, 2 * col - 2 + 2 * n - j)[2];
			}
		}
		for (int j = 0; j < col2; j++) {
			for (int i = 0; i < n; i++) {
				input2.at<Vec3d>(i, j)[0] = input2.at<Vec3d>(2 * n - i, j)[0];
				input2.at<Vec3d>(i, j)[1] = input2.at<Vec3d>(2 * n - i, j)[1];
				input2.at<Vec3d>(i, j)[2] = input2.at<Vec3d>(2 * n - i, j)[2];
			}
			for (int i = row + n; i < row2; i++) {
				input2.at<Vec3d>(i, j)[0] = input2.at<Vec3d>(2 * row - 2 + 2 * n - i, j)[0];
				input2.at<Vec3d>(i, j)[1] = input2.at<Vec3d>(2 * row - 2 + 2 * n - i, j)[1];
				input2.at<Vec3d>(i, j)[2] = input2.at<Vec3d>(2 * row - 2 + 2 * n - i, j)[2];
			}
		}
	return input2;
}


Mat get_Gaussian_Kernel(int n, double sigma_t, double sigma_s, bool normalize) {

	int kernel_size = (2 * n + 1);
	Mat kernel = Mat::zeros(kernel_size, kernel_size, CV_64F);
	double kernel_sum = 0.0;

	for (int i = -n; i <= n; i++) {
		for (int j = -n; j <= n; j++) {
			kernel.at<double>(i + n, j + n) = exp(-((i * i) / (2.0*sigma_t * sigma_t) + (j * j) / (2.0*sigma_s * sigma_s)));
			kernel_sum += kernel.at<double>(i + n, j + n);
		}
	}

	if (normalize) {
		for (int i = 0; i < kernel_size; i++)
			for (int j = 0; j < kernel_size; j++)
				kernel.at<double>(i, j) /= kernel_sum;		// normalize
	}

	return kernel;
}

Mat get_Laplacian_Kernel() {

	Mat kernel = Mat::zeros(3, 3, CV_64F);

	kernel.at<double>(0, 1) = 1.0;
	kernel.at<double>(2, 1) = 1.0;
	kernel.at<double>(1, 0) = 1.0;
	kernel.at<double>(1, 2) = 1.0;
	kernel.at<double>(1, 1) = -4.0;

	return kernel;
}
