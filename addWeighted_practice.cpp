#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {

	double alpha = 0.5; double beta; double input;
	Mat src1, src2, dst;

	cout << "Simple Linear Blender " << endl;
	cout << "----------------------" << endl;
	cout << "* Enter alpha [0.0-1.0]:";
	cin >> input;

	if (input >= 0 && input <= 1) {
		alpha = input;
	}

	src1 = imread("ex-img2.png", IMREAD_COLOR);
	src2 = imread("ex-img4.png", IMREAD_COLOR);

	if (src1.empty()) { cout << "Error loading src1" << endl; return -1; }
	if (src2.empty()) { cout << "Error loading src2" << endl; return -1; }

	// alpha가 커질 수록 src1의 weight 비율이 커진다.
	beta = (1 - alpha);
	
	addWeighted(src1, alpha, src2, beta, 0.0,dst);

	imshow("Linear Blend", dst);

	waitKey(0);

	return 0;

}
