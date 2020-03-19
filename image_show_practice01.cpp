#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;

int main() {
	// IMREAD_COLOR : imaeg를 BGR color image로 change 
	// jpg는 코드와 같은 폴더에 있어야 한다.
	Mat img = imread("noname01.jpg", IMREAD_COLOR);

	if (!img.data) {
		std::cout << "Could not open or find the image" << std::endl;
		return -1;
	}

	namedWindow("lena"); //display를 위한 window를 create
	imshow("lena", img); // show image inside lena window

	waitKey(0);  // wait for a keystroke in the window

	return 0;

}
