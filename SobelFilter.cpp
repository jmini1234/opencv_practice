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

Mat sobelfilter(const Mat input);
Mat sobelfilter_RGB(const Mat input);

int main() {

	Mat input = imread("lena.jpg", CV_LOAD_IMAGE_COLOR);
	Mat input_gray;
	Mat output;
	Mat output_rgb;


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
	output = sobelfilter(input_gray); //Boundary process:  mirroring

	namedWindow("Sobel Filter", WINDOW_AUTOSIZE);
	imshow("Sobel Filter", output);

	output_rgb = sobelfilter_RGB(input); //Boundary process:  mirroring

	namedWindow("Sobel Filter RGB", WINDOW_AUTOSIZE);
	imshow("Sobel Filter RGB", output_rgb);

	waitKey(0);

	return 0;
}

float visualization(float val) {
	if (val < 0)
		val = 0;

	if (val > 255)
		val = 255;

	return val;
}

Mat sobelfilter(const Mat input) {

	Mat kernel;

	int row = input.rows;
	int col = input.cols;
	int n = 1; // Sobel Filter Kernel N

	// Initialiazing 2 Kernel Matrix with 3x3 size for Sx and Sy
	//Fill code to initialize Sobel filter kernel matrix for Sx and Sy (Given in the lecture notes)

	Mat Sx(3, 3, CV_32F); // 16bit signed integer
	Mat Sy(3, 3, CV_32F);

	int arr_x[3][3] = {
		{-1,0,1},
		{-2,0,2},
		{-1,0,1}
	};
	
	int arr_y[3][3] = {
			{ -1, -2, -1},
			{ 0, 0, 0 },
			{ 1, 2, 1 }
	};

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			Sx.at<float>(i, j) = arr_x[i][j];
			Sy.at<float>(i, j) = arr_y[i][j];
		}
	}

	Mat output = Mat::zeros(row, col, input.type());

	int tempa;
	int tempb;

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			float res_x = 0.0;
			float res_y = 0.0;
			for (int a = -n; a <= n; a++) {
				for (int b = -n; b <= n; b++) {
					// Use mirroring boundary process
					// Find output M(x,y) = sqrt( input.at<G>(x, y)*Sx + input.at<G>(x, y)*Sy ) 
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
					res_x += Sx.at<float>(a+n,b+n)*(float)(input.at<G>(tempa, tempb));
					res_y += Sy.at<float>(a + n, b + n)*(float)(input.at<G>(tempa, tempb));
				}
			}

			int sobel_val = (G)sqrt(abs(res_x)*abs(res_x) + abs(res_y)*abs(res_y));

			output.at<G>(i, j) = sobel_val;
		}
	}
	return output;
}

Mat sobelfilter_RGB(const Mat input) {

	Mat kernel;

	int row = input.rows;
	int col = input.cols;
	int n = 1; // Sobel Filter Kernel N

			   // Initialiazing 2 Kernel Matrix with 3x3 size for Sx and Sy
			   //Fill code to initialize Sobel filter kernel matrix for Sx and Sy (Given in the lecture notes)

	Mat Sx(3, 3, CV_32F); // 16bit signed integer
	Mat Sy(3, 3, CV_32F);

	int arr_x[3][3] = {
		{ -1,0,1 },
		{ -2,0,2 },
		{ -1,0,1 }
	};

	int arr_y[3][3] = {
		{ -1, -2, -1 },
		{ 0, 0, 0 },
		{ 1, 2, 1 }
	};

	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			Sx.at<float>(i, j) = arr_x[i][j];
			Sy.at<float>(i, j) = arr_y[i][j];
		}
	}

	Mat output = Mat::zeros(row, col, 0);

	int tempa;
	int tempb;

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			float res_x_r = 0.0;
			float res_x_g = 0.0;
			float res_x_b = 0.0;

			float res_y_r = 0.0;
			float res_y_g = 0.0;
			float res_y_b = 0.0;

			for (int a = -n; a <= n; a++) {
				for (int b = -n; b <= n; b++) {
					// Use mirroring boundary process
					// Find output M(x,y) = sqrt( input.at<G>(x, y)*Sx + input.at<G>(x, y)*Sy ) 
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
					res_x_r += Sx.at<float>(a + n, b + n)*(float)(input.at<C>(tempa, tempb)[0]);
					res_x_g += Sx.at<float>(a + n, b + n)*(float)(input.at<C>(tempa, tempb)[1]);
					res_x_b += Sx.at<float>(a + n, b + n)*(float)(input.at<C>(tempa, tempb)[2]);

					res_y_r += Sx.at<float>(a + n, b + n)*(float)(input.at<C>(tempa, tempb)[0]);
					res_y_g += Sx.at<float>(a + n, b + n)*(float)(input.at<C>(tempa, tempb)[1]);
					res_y_b += Sx.at<float>(a + n, b + n)*(float)(input.at<C>(tempa, tempb)[2]);

				}
			}

			float sobel_val_r = (G)sqrt((res_x_r)*(res_x_r) + (res_y_r)*(res_y_r));
			float sobel_val_g = (G)sqrt((res_x_g)*(res_x_g)+(res_y_g)*(res_y_g));
			float sobel_val_b = (G)sqrt((res_x_b)*(res_x_b)+(res_y_b)*(res_y_b));

			output.at<G>(i,j) = visualization((sobel_val_r + sobel_val_g + sobel_val_b) / 3);

		}
	}
	return output;
}
