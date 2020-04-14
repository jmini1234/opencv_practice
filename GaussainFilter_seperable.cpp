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

Mat gaussian_filter_seperable(const Mat input, int n, float sigmaT, float sigmaS, const char* opt);
Mat gaussian_filter_seperable_RGB(const Mat input, int n, float sigmaT, float sigmaS, const char* opt);

int main() {

	Mat input = imread("lena.jpg", CV_LOAD_IMAGE_COLOR);
	Mat input_gray;
	Mat output, output_RGB;

	cvtColor(input, input_gray, CV_RGB2GRAY);

	if (!input.data)
	{
		std::cout << "Could not open" << std::endl;
		return -1;
	}

	namedWindow("RGB", WINDOW_AUTOSIZE);
	imshow("RGB", input);
	
	namedWindow("Grayscale", WINDOW_AUTOSIZE);
	imshow("Grayscale", input_gray);

	output = gaussian_filter_seperable(input_gray, 5, 5, 5, "adjustkernel"); //Boundary process: zero-paddle, mirroring, adjustkernel
	namedWindow("Gaussian Seperable Filter", WINDOW_AUTOSIZE);
	imshow("Gaussian Seperable Filter", output);

	output_RGB = gaussian_filter_seperable_RGB(input, 5, 5, 5, "adjustkernel"); //Boundary process: zero-paddle, mirroring, adjustkernel
	namedWindow("Gaussian Seperate Filter RGB", WINDOW_AUTOSIZE);
	imshow("Gaussian Seperate Filter RGB", output_RGB);

	waitKey(0);

	return 0;
}

Mat gaussian_filter_seperable(const Mat input, int n, float sigmaT, float sigmaS, const char* opt) {
	Mat kernel_t, kernel_s;

	int row = input.rows;
	int col = input.cols;
	int kernel_size = (2 * n + 1);
	int tempa;
	int tempb;
	float denom_t, denom_s;

	// Initialiazing Kernel Matrix 
	kernel_t = Mat::zeros(1, kernel_size, CV_32F);
	kernel_s = Mat::zeros(kernel_size, 1, CV_32F);

	// W(t) 계산 
	denom_t = 0.0;
	for (int b = -n; b <= n; b++) {
		float value1 = exp(-(pow(b, 2) / (2 * pow(sigmaT, 2))));
		kernel_t.at<float>(b + n) = value1;
		denom_t += value1;
	}

	// W(s) 계산 
	denom_s = 0.0;
	for (int a = -n; a <= n; a++) {  // Denominator in m(s,t)
		float value1 = exp(-(pow(a, 2) / (2 * pow(sigmaS, 2))));
		kernel_s.at<float>(a + n) = value1;
		denom_s += value1;
	}

	for (int b = -n; b <= n; b++) {
		kernel_t.at<float>(b + n) /= denom_t;
	}

	for (int a = -n; a <= n; a++) {
		kernel_s.at<float>(a + n) /= denom_s;
	}

	Mat output = Mat::zeros(row, col, input.type());


	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			float res = 0.0;

			if (!strcmp(opt, "zero-paddle")) {
				for (int a = -n; a <= n; a++) {
					float sum1 = 0.0;
					for (int b = -n; b <= n; b++) {
						if ((i + a <= row - 1) && (i + a >= 0) && (j + b <= col - 1) && (j + b >= 0)) { //if the pixel is not a border pixel
							sum1 += float(kernel_t.at<float>(b + n))*(float)(input.at<G>(i + a, j + b));
						}
					}
					res += float(kernel_s.at<float>(a + n))*sum1;
				}
				output.at<G>(i, j) = (G)res;
			}

			else if (!strcmp(opt, "mirroring")) {
				for (int a = -n; a <= n; a++) {
					float sum1 = 0.0;
					for (int b = -n; b <= n; b++) {
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
						sum1 += float(kernel_t.at<float>(b + n))*(float)(input.at<G>(tempa, tempb));
					}
					res += float(kernel_s.at<float>(a + n))*sum1;
				}
				output.at<G>(i, j) = (G)res;
			}

			else if (!strcmp(opt, "adjustkernel")) {

				// for adjust-kernel
				float res1 = 0.0;
				float res2 = 0.0;

				for (int a = -n; a <= n; a++) {
					float sum1 = 0.0;
					float sum2 = 0.0;
					for (int b = -n; b <= n; b++) {
						if ((i + a <= row - 1) && (i + a >= 0) && (j + b <= col - 1) && (j + b >= 0)) {
							sum1 += float(kernel_t.at<float>(b + n))*(float)(input.at<G>(i + a, j + b));
							sum2 += float(kernel_t.at<float>(b + n));
						}
					}
					res1 += float(kernel_s.at<float>(a + n))*sum1;
					res2 += float(kernel_s.at<float>(a + n))*sum2;
				}
				output.at<G>(i, j) = (G)(res1 / res2);
			}
		}
	}

	return output;
}


Mat gaussian_filter_seperable_RGB(const Mat input, int n, float sigmaT, float sigmaS, const char* opt) {
	Mat kernel_t, kernel_s;

	int row = input.rows;
	int col = input.cols;
	int kernel_size = (2 * n + 1);
	int tempa;
	int tempb;
	float denom_t, denom_s;

	// Initialiazing Kernel Matrix 
	kernel_t = Mat::zeros(1, kernel_size, CV_32F);
	kernel_s = Mat::zeros(kernel_size, 1, CV_32F);


	// W(t) 계산 
	denom_t = 0.0;
	for (int b = -n; b <= n; b++) {
		float value1 = exp(-(pow(b, 2) / (2 * pow(sigmaT, 2))));
		kernel_t.at<float>(b + n) = value1;
		denom_t += value1;
	}

	// W(s) 계산 
	denom_s = 0.0;
	for (int a = -n; a <= n; a++) {  // Denominator in m(s,t)
		float value1 = exp(-(pow(a, 2) / (2 * pow(sigmaS, 2))));
		kernel_s.at<float>(a + n) = value1;
		denom_s += value1;
	}

	for (int b = -n; b <= n; b++) {
		kernel_t.at<float>(b + n) /= denom_t;
	}

	for (int a = -n; a <= n; a++) {
		kernel_s.at<float>(a + n) /= denom_s;
	}

	Mat output = Mat::zeros(row, col, input.type());


	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			float res_r = 0.0;
			float res_g = 0.0;
			float res_b = 0.0;

			if (!strcmp(opt, "zero-paddle")) {
				for (int a = -n; a <= n; a++) {
					float sum1_r = 0.0;
					float sum1_g = 0.0;
					float sum1_b = 0.0;
					for (int b = -n; b <= n; b++) {
						if ((i + a <= row - 1) && (i + a >= 0) && (j + b <= col - 1) && (j + b >= 0)) { //if the pixel is not a border pixel
							sum1_r += float(kernel_t.at<float>(b + n))*(float)(input.at<C>(i + a, j + b)[0]);
							sum1_g += float(kernel_t.at<float>(b + n))*(float)(input.at<C>(i + a, j + b)[1]);
							sum1_b += float(kernel_t.at<float>(b + n))*(float)(input.at<C>(i + a, j + b)[2]);
						}
					}
					res_r += float(kernel_s.at<float>(a + n))*sum1_r;
					res_g += float(kernel_s.at<float>(a + n))*sum1_g;
					res_b += float(kernel_s.at<float>(a + n))*sum1_b;

				}
				output.at<C>(i, j)[0] = (G)res_r;
				output.at<C>(i, j)[1] = (G)res_g;
				output.at<C>(i, j)[2] = (G)res_b;
			}

			else if (!strcmp(opt, "mirroring")) {
				for (int a = -n; a <= n; a++) {
					float sum1_r = 0.0;
					float sum1_g = 0.0;
					float sum1_b = 0.0;
					for (int b = -n; b <= n; b++) {
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
						sum1_r += float(kernel_t.at<float>(b + n))*(float)(input.at<C>(tempa, tempb)[0]);
						sum1_g += float(kernel_t.at<float>(b + n))*(float)(input.at<C>(tempa, tempb)[1]);
						sum1_b += float(kernel_t.at<float>(b + n))*(float)(input.at<C>(tempa, tempb)[2]);
					}
					res_r += float(kernel_s.at<float>(a + n))*sum1_r;
					res_g += float(kernel_s.at<float>(a + n))*sum1_g;
					res_b += float(kernel_s.at<float>(a + n))*sum1_b;
				}
				output.at<C>(i, j)[0] = (G)res_r;
				output.at<C>(i, j)[1] = (G)res_g;
				output.at<C>(i, j)[2] = (G)res_b;
			}

			else if (!strcmp(opt, "adjustkernel")) {

				// for adjust-kernel
				float res1_r = 0.0;
				float res1_g = 0.0;
				float res1_b = 0.0;
				float res2 = 0.0;

				for (int a = -n; a <= n; a++) {
					float sum1_r = 0.0;
					float sum1_g = 0.0;
					float sum1_b = 0.0;
					float sum2 = 0.0;
					for (int b = -n; b <= n; b++) {
						if ((i + a <= row - 1) && (i + a >= 0) && (j + b <= col - 1) && (j + b >= 0)) {
							sum1_r += float(kernel_t.at<float>(b + n))*(float)(input.at<C>(i + a, j + b)[0]);
							sum1_g += float(kernel_t.at<float>(b + n))*(float)(input.at<C>(i + a, j + b)[1]);
							sum1_b += float(kernel_t.at<float>(b + n))*(float)(input.at<C>(i + a, j + b)[2]);
							sum2 += float(kernel_t.at<float>(b + n));
						}
					}
					res1_r += float(kernel_s.at<float>(a + n))*sum1_r;
					res1_g += float(kernel_s.at<float>(a + n))*sum1_g;
					res1_b += float(kernel_s.at<float>(a + n))*sum1_b;
					res2 += float(kernel_s.at<float>(a + n))*sum2;
				}
				output.at<C>(i, j)[0] = (G)(res1_r / res2);
				output.at<C>(i, j)[1] = (G)(res1_g / res2);
				output.at<C>(i, j)[2] = (G)(res1_b / res2);
			}
		}
	}
	return output;
}
