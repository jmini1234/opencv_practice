#include <opencv2/opencv.hpp>
#include <stdio.h>

#define IM_TYPE	CV_64FC3

using namespace cv;

//ADD
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

Mat Add_Gaussian_noise(const Mat input, double mean, double sigma);
Mat Gaussianfilter_Gray(const Mat input, int n, double sigma_t, double sigma_s, const char *opt);
Mat Gaussianfilter_RGB(const Mat input, int n, double sigma_t, double sigma_s, const char *opt);
Mat Bilateralfilter_Gray(const Mat input, int n, double sigma_t, double sigma_s, double sigma_r, const char *opt);
Mat Bilateralfilter_RGB(const Mat input, int n, double sigma_t, double sigma_s, double sigma_r, const char *opt);

int main() {

	Mat input = imread("lena.jpg", CV_LOAD_IMAGE_COLOR);
	Mat input_gray;

	// check for validation
	if (!input.data) {
		printf("Could not open\n");
		return -1;
	}

	cvtColor(input, input_gray, CV_RGB2GRAY);	// convert RGB to Grayscale
	
	// 8-bit unsigned char -> 64-bit floating point
	input.convertTo(input, CV_64FC3, 1.0 / 255);
	input_gray.convertTo(input_gray, CV_64F, 1.0 / 255);

	// Add noise to original image
	Mat noise_Gray = Add_Gaussian_noise(input_gray, 0, 0.1);
	Mat noise_RGB = Add_Gaussian_noise(input, 0, 0.1);

	// Denoise, using gaussian filter
	Mat Denoised_Gray = Gaussianfilter_Gray(noise_Gray, 3, 10, 10, "adjustkernel");
	Mat Denoised_RGB = Gaussianfilter_RGB(noise_RGB, 3, 10, 10, "adjustkernel");

	// Denoise, using gaussian bilateral filter
	Mat Denoised_bilateral_Gray = Bilateralfilter_Gray(noise_Gray, 3, 10, 10, 7, "adjustkernel");
	Mat Denoised_bilateral_RGB = Bilateralfilter_RGB(noise_RGB, 3, 10, 10, 7,"adjustkernel");


	namedWindow("Grayscale", WINDOW_AUTOSIZE);
	imshow("Grayscale", input_gray);

	namedWindow("RGB", WINDOW_AUTOSIZE);
	imshow("RGB", input);

	namedWindow("Gaussian Noise (Grayscale)", WINDOW_AUTOSIZE);
	imshow("Gaussian Noise (Grayscale)", noise_Gray);

	namedWindow("Gaussian Noise (RGB)", WINDOW_AUTOSIZE);
	imshow("Gaussian Noise (RGB)", noise_RGB);

	namedWindow("Denoised (Grayscale)", WINDOW_AUTOSIZE);
	imshow("Denoised (Grayscale)", Denoised_Gray);

	namedWindow("Denoised (RGB)", WINDOW_AUTOSIZE);
	imshow("Denoised (RGB)", Denoised_RGB);

	namedWindow("Denoised bilateral (Grayscale)", WINDOW_AUTOSIZE);
	imshow("Denoised bilateral (Grayscale)", Denoised_bilateral_Gray);

	namedWindow("Denoised bilateral (RGB)", WINDOW_AUTOSIZE);
	imshow("Denoised bilateral (RGB)", Denoised_bilateral_RGB);

	waitKey(0);

	return 0;
}

Mat Add_Gaussian_noise(const Mat input, double mean, double sigma) {

	Mat NoiseArr = Mat::zeros(input.rows, input.cols, input.type());
	RNG rng;
	rng.fill(NoiseArr, RNG::NORMAL, mean, sigma);

	add(input, NoiseArr, NoiseArr);

	return NoiseArr;
}

Mat Gaussianfilter_Gray(const Mat input, int n, double sigma_t, double sigma_s, const char *opt) {

	int row = input.rows;
	int col = input.cols;
	int kernel_size = (2 * n + 1);
	int tempa;
	int tempb;
	float denom;

	Mat kernel = Mat::zeros(kernel_size, kernel_size, CV_32F);

	Mat output = Mat::zeros(row, col, input.type());

	denom = 0.0;
	for (int a = -n; a <= n; a++) {  // Denominator in m(s,t)
		for (int b = -n; b <= n; b++) {
			float value1 = exp(-(pow(a, 2) / (2 * pow(sigma_s, 2))) - (pow(b, 2) / (2 * pow(sigma_t, 2))));
			kernel.at<float>(a + n, b + n) = value1;
			denom += value1;
		}
	}

	for (int a = -n; a <= n; a++) {  // calculate kernel
		for (int b = -n; b <= n; b++) {
			kernel.at<float>(a + n, b + n) /= denom;
		}
	}


	// convolution
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {

			if (!strcmp(opt, "zero-padding")) {
				float sum1 = 0.0;
				for (int x = -n; x <= n; x++) { // for each kernel window
					for (int y = -n; y <= n; y++) {
						if ((i + x <= row - 1) && (i + x >= 0) && (j + y <= col - 1) && (j + y >= 0)) { //if the pixel is not a border pixel
							sum1 += float(kernel.at<float>(x + n,y + n))*(float)(input.at<G>(i + x, j + y));
						}
					}
				}
				output.at<G>(i, j) = (G)sum1;
			}

			else if (!strcmp(opt, "mirroring")) {
				float sum1 = 0.0;
				for (int x = -n; x <= n; x++) { // for each kernel window
					for (int y = -n; y <= n; y++) {

						if (i + x > row - 1) {  //mirroring for the border pixels
							tempa = i - x;
						}
						else if (i + x < 0) {
							tempa = -(i + x);
						}
						else {
							tempa = i + x;
						}
						if (j + y > col - 1) {
							tempb = j - y;
						}
						else if (j + y < 0) {
							tempb = -(j + y);
						}
						else {
							tempb = j + y;
						}
						sum1 += (float)(kernel.at<float>(x + n, y + n))*(float)(input.at<G>(tempa, tempb));
					}
				}
				output.at<G>(i, j) = (G)sum1;
			}

			else if (!strcmp(opt, "adjustkernel")) {
				float sum1 = 0.0;
				float sum2 = 0.0;
				for (int x = -n; x <= n; x++) { // for each kernel window
					for (int y = -n; y <= n; y++) {
						if ((i + x <= row - 1) && (i + x >= 0) && (j + y <= col - 1) && (j + y >= 0)) {
							sum1 += (float)(kernel.at<float>(x + n, y + n))*(float)(input.at<G>(i + x, j + y));
							sum2 += (float)(kernel.at<float>(x + n, y + n));
						}
					}
				}
				output.at<G>(i, j) = (G)(sum1 / sum2);
			}

		}
	}

	return output;
}

Mat Gaussianfilter_RGB(const Mat input, int n, double sigma_t, double sigma_s, const char *opt) {
	
	int row = input.rows;
	int col = input.cols;
	int kernel_size = (2 * n + 1);
	int tempa;
	int tempb;
	float denom;

	Mat output = Mat::zeros(row, col, input.type());

	// convolution

	//add kernel Mat for guassian filter

	Mat kernel = Mat::zeros(kernel_size, kernel_size, CV_32F);

	denom = 0.0;
	for (int a = -n; a <= n; a++) {  // Denominator in m(s,t)
		for (int b = -n; b <= n; b++) {
			float value1 = exp(-(pow(a, 2) / (2 * pow(sigma_s, 2))) - (pow(b, 2) / (2 * pow(sigma_t, 2))));
			kernel.at<float>(a + n, b + n) = value1;
			denom += value1;
		}
	}

	for (int a = -n; a <= n; a++) {  // Denominator in m(s,t)
		for (int b = -n; b <= n; b++) {
			kernel.at<float>(a + n, b + n) /= denom;
		}
	}

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {

			if (!strcmp(opt, "zero-padding")) {
				float sum1_r = 0.0;
				float sum1_g = 0.0;
				float sum1_b = 0.0;
				for (int x = -n; x <= n; x++) { // for each kernel window
					for (int y = -n; y <= n; y++) {
						if ((i + x <= row - 1) && (i + x >= 0) && (j + y <= col - 1) && (j + y >= 0)) { //if the pixel is not a border pixel
							sum1_r += float(kernel.at<float>(x + n, y + n))*(float)(input.at<C>(i + x, j + y)[0]);
							sum1_g += float(kernel.at<float>(x + n, y + n))*(float)(input.at<C>(i + x, j + y)[1]);
							sum1_b += float(kernel.at<float>(x + n, y + n))*(float)(input.at<C>(i + x, j + y)[2]);
						}
					}
				}
				output.at<C>(i, j)[0] = (G)sum1_r;
				output.at<C>(i, j)[1] = (G)sum1_g;
				output.at<C>(i, j)[2] = (G)sum1_b;
			}

			else if (!strcmp(opt, "mirroring")) {
				float sum1_r = 0.0;
				float sum1_g = 0.0;
				float sum1_b = 0.0;
				for (int x = -n; x <= n; x++) { // for each kernel window
					for (int y = -n; y <= n; y++) {
						if (i + x > row - 1) {  //mirroring for the border pixels
							tempa = i - x;
						}
						else if (i + x < 0) {
							tempa = -(i + x);
						}
						else {
							tempa = i + x;
						}
						if (j + y > col - 1) {
							tempb = j - y;
						}
						else if (j + y < 0) {
							tempb = -(j + y);
						}
						else {
							tempb = j + y;
						}
						sum1_r += (float)(kernel.at<float>(x + n, y + n))*(float)(input.at<C>(tempa, tempb)[0]);
						sum1_g += (float)(kernel.at<float>(x + n, y + n))*(float)(input.at<C>(tempa, tempb)[1]);
						sum1_b += (float)(kernel.at<float>(x + n, y + n))*(float)(input.at<C>(tempa, tempb)[2]);
					}
				}
				output.at<C>(i, j)[0] = (G)sum1_r;
				output.at<C>(i, j)[1] = (G)sum1_g;
				output.at<C>(i, j)[2] = (G)sum1_b;

			}

			else if (!strcmp(opt, "adjustkernel")) {
				float sum1_r = 0.0;
				float sum1_g = 0.0;
				float sum1_b = 0.0;
				float sum2 = 0.0;
				for (int x = -n; x <= n; x++) { // for each kernel window
					for (int y = -n; y <= n; y++) {
						if ((i + x <= row - 1) && (i + x >= 0) && (j + y <= col - 1) && (j + y >= 0)) {
							sum1_r += (float)(kernel.at<float>(x + n, y + n))*(float)(input.at<C>(i + x, j + y)[0]);
							sum1_g += (float)(kernel.at<float>(x + n, y + n))*(float)(input.at<C>(i + x, j + y)[1]);
							sum1_b += (float)(kernel.at<float>(x + n, y + n))*(float)(input.at<C>(i + x, j + y)[2]);
							sum2 += (float)(kernel.at<float>(x + n, y + n));
						}
					}
				}
				output.at<C>(i, j)[0] = (G)(sum1_r / sum2);
				output.at<C>(i, j)[1] = (G)(sum1_g / sum2);
				output.at<C>(i, j)[2] = (G)(sum1_b / sum2);
			}

		}
	}

	return output;
}

Mat Bilateralfilter_Gray(const Mat input, int n, double sigma_t, double sigma_s, double sigma_r, const char *opt) {
	
	int row = input.rows;
	int col = input.cols;
	int kernel_size = (2 * n + 1);
	int tempa;
	int tempb;
	float denom;

	Mat output = Mat::zeros(row, col, input.type());

	// for W(i,j)
	Mat W_kernel = Mat::zeros(row, col, CV_32F);

	//for distance exp
	Mat exp_kernel = Mat::zeros(kernel_size, kernel_size, CV_32F);

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			// sum for two points distance and intensity exp
			denom = 0.0;
			for (int a = -n; a <= n; a++) {  
				for (int b = -n; b <= n; b++) {
					if ((i + a <= row - 1) && (i + a >= 0) && (j + b <= col - 1) && (j + b >= 0)) {
						float value1 = exp(-(pow(a, 2) / (2 * pow(sigma_s, 2))) - (pow(b, 2) / (2 * pow(sigma_t, 2))));
						float value2 = exp(-(pow((input.at<G>(i, j) - input.at<G>(i + a, j + b)), 2)) / 2 * pow(sigma_r, 2));
						exp_kernel.at<float>(a + n, b + n) = value1;
						denom += value1*value2;
					}
				}
			}
			W_kernel.at<float>(i, j) = denom;
		}
	}

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {

			if (!strcmp(opt, "zero-padding")) {
				float sum1 = 0.0;
				for (int x = -n; x <= n; x++) { // for each kernel window
					for (int y = -n; y <= n; y++) {
						if ((i + x <= row - 1) && (i + x >= 0) && (j + y <= col - 1) && (j + y >= 0)) { //if the pixel is not a border pixel
							float ex1 = exp_kernel.at<float>(x + n, y + n);
							// for intensity exp
							float ex2 = exp(-(pow((input.at<G>(i, j) - input.at<G>(i + x, j + y)), 2)) / 2 * pow(sigma_r, 2));
							// w(s,t) - kernel
							float s = (1 / W_kernel.at<float>(i, j))*ex1*ex2;
							sum1 += s *(float)(input.at<G>(i + x, j + y));
						}
					}
				}
				output.at<G>(i, j) = (G)sum1;
			}

			else if (!strcmp(opt, "mirroring")) {
				float sum1 = 0.0;
				for (int x = -n; x <= n; x++) { // for each kernel window
					for (int y = -n; y <= n; y++) {

						if (i + x > row - 1) {  //mirroring for the border pixels
							tempa = i - x;
						}
						else if (i + x < 0) {
							tempa = -(i + x);
						}
						else {
							tempa = i + x;
						}
						if (j + y > col - 1) {
							tempb = j - y;
						}
						else if (j + y < 0) {
							tempb = -(j + y);
						}
						else {
							tempb = j + y;
						}
						float ex1 = exp_kernel.at<float>(x + n, y + n);
						float ex2 = exp(-(pow((input.at<G>(i, j) - input.at<G>(tempa, tempb)), 2)) / 2 * pow(sigma_r, 2));
						float s = (1 / W_kernel.at<float>(i, j))*ex1*ex2;
						sum1 += s *(float)(input.at<G>(tempa, tempb));
					}
				}
				output.at<G>(i, j) = (G)sum1;
			}

			else if (!strcmp(opt, "adjustkernel")) {
				float sum1 = 0.0;
				float sum2 = 0.0;
				for (int x = -n; x <= n; x++) { // for each kernel window
					for (int y = -n; y <= n; y++) {
						if ((i + x <= row - 1) && (i + x >= 0) && (j + y <= col - 1) && (j + y >= 0)) {
							float ex1 = exp_kernel.at<float>(x + n, y + n);
							float ex2 = exp(-(pow((input.at<G>(i, j) - input.at<G>(i + x, j + y)), 2)) / 2 * pow(sigma_r, 2));
							float s = (1 / W_kernel.at<float>(i, j))*ex1*ex2;

							sum1 += s*(float)(input.at<G>(i + x, j + y));
							sum2 += s;
						}
					}
				}
				output.at<G>(i, j) = (G)(sum1 / sum2);
			}

		}
	}

	return output;

}

Mat Bilateralfilter_RGB(const Mat input, int n, double sigma_t, double sigma_s, double sigma_r, const char *opt) {
	int row = input.rows;
	int col = input.cols;
	int kernel_size = (2 * n + 1);
	int tempa;
	int tempb;
	float denom;

	Mat output = Mat::zeros(row, col, input.type());

	// for W(i,j)
	Mat W_kernel = Mat::zeros(row, col, CV_32F);

	// for distance exp
	Mat exp_kernel = Mat::zeros(kernel_size, kernel_size, CV_32F);

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			denom = 0.0;
			for (int a = -n; a <= n; a++) {  
				for (int b = -n; b <= n; b++) {
					if ((i + a <= row - 1) && (i + a >= 0) && (j + b <= col - 1) && (j + b >= 0)) {
						float value1 = exp(-(pow(a, 2) / (2 * pow(sigma_s, 2))) - (pow(b, 2) / (2 * pow(sigma_t, 2))));
						float value2 = exp(-(pow((input.at<C>(i, j)[0] - input.at<C>(i + a, j + b)[0]), 2)+ pow((input.at<C>(i, j)[1] - input.at<C>(i + a, j + b)[1]), 2)+ pow((input.at<C>(i, j)[2] - input.at<C>(i + a, j + b)[2]), 2)) / 2 * pow(sigma_r, 2));
						exp_kernel.at<float>(a + n, b + n) = value1;
						denom += value1*value2;
					}
				}
			}
			W_kernel.at<float>(i, j) = denom;
		}
	}

	// convolution
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {

			if (!strcmp(opt, "zero-padding")) {
				float sum1_r = 0.0;
				float sum1_g = 0.0;
				float sum1_b = 0.0;
				for (int x = -n; x <= n; x++) { // for each kernel window
					for (int y = -n; y <= n; y++) {
						if ((i + x <= row - 1) && (i + x >= 0) && (j + y <= col - 1) && (j + y >= 0)) { 
							float ex1 = exp_kernel.at<float>(x + n, y + n);
							// sum for RGB intensity exp
							float ex2 = exp(-(pow((input.at<C>(i, j)[0] - input.at<C>(i + x, j + y)[0]), 2) + pow((input.at<C>(i, j)[1] - input.at<C>(i + x, j + y)[1]), 2) + pow((input.at<C>(i, j)[2] - input.at<C>(i + x, j + y)[2]), 2)) / 2 * pow(sigma_r, 2));
							// w(s,t)
							float s = (1 / W_kernel.at<float>(i, j))*ex1*ex2;
							// RGB caculate for each
							sum1_r += s *(float)(input.at<C>(i + x, j + y)[0]);
							sum1_g += s *(float)(input.at<C>(i + x, j + y)[1]);
							sum1_b += s *(float)(input.at<C>(i + x, j + y)[2]);
						}
					}
				}
				output.at<C>(i, j)[0] = (G)sum1_r;
				output.at<C>(i, j)[1] = (G)sum1_g;
				output.at<C>(i, j)[2] = (G)sum1_b;

			}

			else if (!strcmp(opt, "mirroring")) {
				float sum1_r = 0.0;
				float sum1_g = 0.0;
				float sum1_b = 0.0;
				for (int x = -n; x <= n; x++) { // for each kernel window
					for (int y = -n; y <= n; y++) {

						if (i + x > row - 1) {  //mirroring for the border pixels
							tempa = i - x;
						}
						else if (i + x < 0) {
							tempa = -(i + x);
						}
						else {
							tempa = i + x;
						}
						if (j + y > col - 1) {
							tempb = j - y;
						}
						else if (j + y < 0) {
							tempb = -(j + y);
						}
						else {
							tempb = j + y;
						}
						float ex1 = exp_kernel.at<float>(x + n, y + n);
						// sum for RGB intensity exp
						float ex2 = exp(-(pow((input.at<C>(i, j)[0] - input.at<C>(tempa, tempb)[0]), 2) + pow((input.at<C>(i, j)[1] - input.at<C>(tempa, tempb)[1]), 2) + pow((input.at<C>(i, j)[2] - input.at<C>(tempa, tempb)[2]), 2)) / 2 * pow(sigma_r, 2));
						float s = (1 / W_kernel.at<float>(i, j))*ex1*ex2;
						sum1_r += s *(float)(input.at<C>(tempa, tempb)[0]);
						sum1_g += s *(float)(input.at<C>(tempa, tempb)[1]);
						sum1_b += s *(float)(input.at<C>(tempa, tempb)[2]);
					}
				}
				output.at<C>(i, j)[0] = (G)sum1_r;
				output.at<C>(i, j)[1] = (G)sum1_g;
				output.at<C>(i, j)[2] = (G)sum1_b;
			}

			else if (!strcmp(opt, "adjustkernel")) {
				float sum1_r = 0.0;
				float sum1_g = 0.0;
				float sum1_b = 0.0;
				float sum2 = 0.0;
				for (int x = -n; x <= n; x++) { // for each kernel window
					for (int y = -n; y <= n; y++) {
						if ((i + x <= row - 1) && (i + x >= 0) && (j + y <= col - 1) && (j + y >= 0)) {
							float ex1 = exp_kernel.at<float>(x + n, y + n);
							float ex2 = exp(-(pow((input.at<C>(i, j)[0] - input.at<C>(i + x, j + y)[0]), 2) + pow((input.at<C>(i, j)[1] - input.at<C>(i + x, j + y)[1]), 2) + pow((input.at<C>(i, j)[2] - input.at<C>(i + x, j + y)[2]), 2)) / 2 * pow(sigma_r, 2));
							float s = (1 / W_kernel.at<float>(i, j))*ex1*ex2;

							sum1_r += s *(float)(input.at<C>(i + x, j + y)[0]);
							sum1_g += s *(float)(input.at<C>(i + x, j + y)[1]);
							sum1_b += s *(float)(input.at<C>(i + x, j + y)[2]);
							sum2 += s;
						}
					}
				}
				output.at<C>(i, j)[0] = (G)sum1_r/sum2;
				output.at<C>(i, j)[1] = (G)sum1_g/sum2;
				output.at<C>(i, j)[2] = (G)sum1_b/sum2;
			}

		}
	}

	return output;
}
