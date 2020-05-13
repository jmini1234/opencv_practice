#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>

#define RATIO_THR 0.4

using namespace std;
using namespace cv;

template <typename T>
Mat cal_affine(double ptl_x[], double ptl_y[], double ptr_x[], double ptr_y[], int number_of_points);
double euclidDistance(Mat& vec1, Mat& vec2);
int nearestNeighbor(Mat& vec, vector<KeyPoint>& keypoints, Mat& descriptors);
double nearestNeighbor_dist(Mat& vec, vector<KeyPoint>& keypoints, Mat& descriptors);
double secondnearestNeighbor_dist(Mat& vec, vector<KeyPoint>& keypoints, Mat& descriptors);
void findPairs(vector<KeyPoint>& keypoints1, Mat& descriptors1,
	vector<KeyPoint>& keypoints2, Mat& descriptors2,
	vector<Point2f>& srcPoints, vector<Point2f>& dstPoints, bool crossCheck, bool ratio_threshold);

void blend_stitching(const Mat I1, const Mat I2, Mat &I_f, int diff_x, int diff_y, float alpha);
void cal_RANSAC(int i, vector<Point2f>& srcPoints, vector<Point2f>& dstPoints);
Mat last_affine(Mat max_AF, int count, vector<Point2f>& srcPoints, vector<Point2f>& dstPoints);
Mat last_affine_reverse(Mat max_AF, int count, vector<Point2f>& srcPoints, vector<Point2f>& dstPoints);

int random(int num) {
	return(rand() % num);
}

Mat input1, input2;
int sigma = 1.5;
int max_count = 0;


double p = 0.99;
double e = 0.3;
int k = 4;

int s = log(1 - p) / log(1 - pow(1 - e, k));
Mat *affine_trans = new Mat[s];

int *cnt = new int[s];

int main() {
	input1 = imread("input1.jpg");
	input2 = imread("input2.jpg");
	Mat input1_gray, input2_gray;

	if (!input1.data || !input2.data)
	{
		std::cout << "Could not open" << std::endl;
		return -1;
	}

	//resize(input1, input1, Size(input1.cols / 2, input1.rows / 2));
	//resize(input2, input2, Size(input2.cols / 2, input2.rows / 2));

	cvtColor(input1, input1_gray, CV_RGB2GRAY);
	cvtColor(input2, input2_gray, CV_RGB2GRAY);

	FeatureDetector* detector = new SiftFeatureDetector(
		0,		// nFeatures
		4,		// nOctaveLayers
		0.04,	// contrastThreshold
		10,		// edgeThreshold
		1.6		// sigma
	);

	DescriptorExtractor* extractor = new SiftDescriptorExtractor();

	// Create a image for displaying mathing keypoints
	Size size = input2.size();
	Size sz = Size(size.width + input1_gray.size().width, max(size.height, input1_gray.size().height));
	Mat matchingImage = Mat::zeros(sz, CV_8UC3);

	input1.copyTo(matchingImage(Rect(size.width, 0, input1_gray.size().width, input1_gray.size().height)));
	input2.copyTo(matchingImage(Rect(0, 0, size.width, size.height)));

	// Compute keypoints and descriptor from the source image in advance
	vector<KeyPoint> keypoints1;
	Mat descriptors1;

	detector->detect(input1_gray, keypoints1);
	extractor->compute(input1_gray, keypoints1, descriptors1);
	printf("input1 : %d keypoints are found.\n", (int)keypoints1.size());

	vector<KeyPoint> keypoints2;
	Mat descriptors2;

	// Detect keypoints
	detector->detect(input2_gray, keypoints2);
	extractor->compute(input2_gray, keypoints2, descriptors2);

	printf("input2 : %zd keypoints are found.\n", keypoints2.size());

	for (int i = 0; i < keypoints1.size(); i++) {
		KeyPoint kp = keypoints1[i];
		kp.pt.x += size.width;
		circle(matchingImage, kp.pt, cvRound(kp.size*0.25), Scalar(255, 255, 0), 1, 8, 0);
	}

	for (int i = 0; i < keypoints2.size(); i++) {
		KeyPoint kp = keypoints2[i];
		circle(matchingImage, kp.pt, cvRound(kp.size*0.25), Scalar(255, 255, 0), 1, 8, 0);
	}

	// Find nearest neighbor pairs
	vector<Point2f> srcPoints;
	vector<Point2f> dstPoints;
	bool crossCheck = true;
	bool ratio_threshold = true;
	//findPairs(keypoints2, descriptors2, keypoints1, descriptors1, srcPoints, dstPoints, crossCheck, ratio_threshold);
	findPairs(keypoints1, descriptors1, keypoints2, descriptors2, srcPoints, dstPoints, crossCheck, ratio_threshold);

	// corresponding pixels
	for (int i = 0; i < s; i++) {
		cal_RANSAC(i,srcPoints, dstPoints);
	}

	int max = 0;
	int max_idx = 0;

	for (int i = 0; i < s; i++) {
		if (cnt[i] > max) {
			max_idx = i;
			max = cnt[i];
		}
	}

	Mat max_AF = affine_trans[max_idx];

	const float I1_row = input1.rows;
	const float I1_col = input1.cols;
	const float I2_row = input2.rows;
	const float I2_col = input2.cols;

	// calculate affine Matrix A12, A21
	Mat A12 = last_affine(max_AF, max, srcPoints, dstPoints);

	Mat A21 = last_affine_reverse(max_AF, max, srcPoints, dstPoints);

	// compute corners (p1, p2, p3, p4)
	Point2f p1(A21.at<float>(0) * 0 + A21.at<float>(1) * 0 + A21.at<float>(2),
		A21.at<float>(3) * 0 + A21.at<float>(4) * 0 + A21.at<float>(5));

	Point2f p2(A21.at<float>(0) * 0 + A21.at<float>(1) * I2_col + A21.at<float>(2),
		A21.at<float>(3) * 0 + A21.at<float>(4) * I2_col + A21.at<float>(5));

	Point2f p3(A21.at<float>(0) * I2_row + A21.at<float>(1) * 0 + A21.at<float>(2),
		A21.at<float>(3) * I2_row + A21.at<float>(4) * 0 + A21.at<float>(5));

	Point2f p4(A21.at<float>(0) * I2_row + A21.at<float>(1) * I2_col + A21.at<float>(2),
		A21.at<float>(3) * I2_row + A21.at<float>(4) * I2_col + A21.at<float>(5));

	// for inverse warping
	Point2f p1_(A12.at<float>(0) * 0 + A12.at<float>(1) * 0 + A12.at<float>(2),
		A12.at<float>(3) * 0 + A12.at<float>(4) * 0 + A12.at<float>(5));

	Point2f p2_(A12.at<float>(0) * 0 + A12.at<float>(1) * I1_col + A12.at<float>(2),
		A12.at<float>(3) * 0 + A12.at<float>(4) * I1_col + A12.at<float>(5));

	Point2f p3_(A12.at<float>(0) * I1_row + A12.at<float>(1) * 0 + A12.at<float>(2),
		A12.at<float>(3) * I1_row + A12.at<float>(4) * 0 + A12.at<float>(5));

	Point2f p4_(A12.at<float>(0) * I1_row + A12.at<float>(1) * I1_col + A12.at<float>(2),
		A12.at<float>(3) * I1_row + A12.at<float>(4) * I1_col + A12.at<float>(5));

	// compute boundary for merged image(I_f)
	int bound_u = (int)round(min(0.0f, min(p1.x, p2.x)));
	int bound_b = (int)round(std::max(I1_row, std::max(p3.x, p4.x)));
	int bound_l = (int)round(min(0.0f, min(p1.y, p3.y)));
	int bound_r = (int)round(std::max(I1_col, std::max(p2.y, p4.y)));

	// compute boundary for inverse warping
	int bound_u_ = (int)round(min(0.0f, min(p1_.x, p2_.x)));
	int bound_b_ = (int)round(std::max(I2_row, std::max(p3_.x, p4_.x)));
	int bound_l_ = (int)round(min(0.0f, min(p1_.y, p3_.y)));
	int bound_r_ = (int)round(std::max(I2_col, std::max(p2_.y, p4_.y)));

	int diff_x = abs(bound_u);
	int diff_y = abs(bound_l);

	int diff_x_ = abs(bound_u_);
	int diff_y_ = abs(bound_l_);

	// initialize merged image
	Mat I_f(bound_b - bound_u + 1, bound_r - bound_l + 1, CV_32FC3, Scalar(0));

	input1.convertTo(input1, CV_32FC3, 1.0 / 255);
	input2.convertTo(input2, CV_32FC3, 1.0 / 255);
	
	// inverse warping with bilinear interplolation
	for (int i = -diff_x_; i < I_f.rows - diff_x_; i++) {
		for (int j = -diff_y_; j < I_f.cols - diff_y_; j++) {

			float x = A12.at<float>(0) * i + A12.at<float>(1) * j + A12.at<float>(2) + diff_x_;
			float y = A12.at<float>(3) * i + A12.at<float>(4) * j + A12.at<float>(5) + diff_y_;

			float y1 = floor(y);
			float y2 = ceil(y);
			float x1 = floor(x);
			float x2 = ceil(x);

			float mu = y - y1;
			float lambda = x - x1;

			if (x1 >= 0 && x2 < I2_row && y1 >= 0 && y2 < I2_col)
			{
				Vec3f V1 = mu * input2.at<Vec3f>(x1, y2) + (1 - mu) * input2.at<Vec3f>(x1, y1);
				Vec3f V2 = mu * input2.at<Vec3f>(x2, y2) + (1 - mu) * input2.at<Vec3f>(x2, y1);

				I_f.at<Vec3f>(i + diff_x_, j + diff_y_) = lambda * V1 + (1 - lambda) * V2;
			}
		}
	}

	// image stitching with blend
	blend_stitching(input1, input2, I_f, diff_x, diff_y, 0.5);
	
	namedWindow("Left Image");
	imshow("Left Image", input1);
	
	namedWindow("Right Image");
	imshow("Right Image", input2);
	
	namedWindow("result");
	imshow("result", I_f);

	waitKey(0);

	return 0;
}

// Do RANSAC
void cal_RANSAC(int idx, vector<Point2f>& srcPoints, vector<Point2f>& dstPoints) {
	// inlier count 
	int count = 0;
	// k=4
	double *arr = new double[k];
	// get random idx
	for (int i = 0; i < k; i++) {
		arr[i] = random((int)srcPoints.size());
	}
	double *ptl_x = new double[k];
	double *ptl_y = new double[k];
	double *ptr_x = new double[k];
	double *ptr_y = new double[k];


	for (int i = 0; i < k; ++i) {
		int idx = arr[i];
		Point2f pt1 = srcPoints[idx];
		Point2f pt2 = dstPoints[idx];
		ptl_x[i] = pt1.x;
		ptl_y[i] = pt1.y;
		ptr_x[i] = pt2.x;
		ptr_y[i] = pt2.y;
	}

	// calculate affine Matrix A12
	Mat A12 = cal_affine<float>(ptl_x, ptl_y, ptr_x, ptr_y, k);

	affine_trans[idx] = A12;
	for (int i = 0; i < (int)srcPoints.size(); i++) {
		float x = srcPoints[i].x;
		float y = srcPoints[i].y;
		float Tx = A12.at<float>(0)*x + A12.at<float>(1)*y+ A12.at<float>(2) * 1;
		float Ty = A12.at<float>(3)*x + A12.at<float>(4)*y + A12.at<float>(5) * 1;

		float val = pow(abs(Tx - dstPoints[i].x), 2) + pow(abs(Ty - dstPoints[i].y), 2);
		if (val < pow(sigma, 2)) {
			count++;
		}
	}
	cnt[idx] = count;
}

Mat last_affine(Mat max_AF,int count, vector<Point2f>& srcPoints, vector<Point2f>& dstPoints) {
	int j = 0;
	double *ptl_x = new double[count];
	double *ptl_y = new double[count];
	double *ptr_x = new double[count];
	double *ptr_y = new double[count];

	for (int i = 0; i < (int)srcPoints.size(); i++) {
		float x = srcPoints[i].x;
		float y = srcPoints[i].y;
		float Tx = max_AF.at<float>(0)*x + max_AF.at<float>(1)*y + max_AF.at<float>(2) * 1;
		float Ty = max_AF.at<float>(3)*x + max_AF.at<float>(4)*y + max_AF.at<float>(5) * 1;

		float val = pow(abs(Tx - dstPoints[i].x), 2) + pow(abs(Ty - dstPoints[i].y), 2);

		if (val < pow(sigma, 2)) {
			ptl_x[j] = x;
			ptl_y[j] = y;
			ptr_x[j] = dstPoints[i].x;
			ptr_y[j] = dstPoints[i].y;
			j++;
		}
	}
	Mat final = cal_affine<float>(ptl_x, ptl_y, ptr_x, ptr_y, 8);
	return final;
}

Mat last_affine_reverse(Mat max_AF, int count, vector<Point2f>& srcPoints, vector<Point2f>& dstPoints) {
	int j = 0;
	double *ptl_x = new double[count];
	double *ptl_y = new double[count];
	double *ptr_x = new double[count];
	double *ptr_y = new double[count];

	for (int i = 0; i < (int)srcPoints.size(); i++) {
		float x = srcPoints[i].x;
		float y = srcPoints[i].y;
		float Tx = max_AF.at<float>(0)*x + max_AF.at<float>(1)*y + max_AF.at<float>(2) * 1;
		float Ty = max_AF.at<float>(3)*x + max_AF.at<float>(4)*y + max_AF.at<float>(5) * 1;

		float val = pow(abs(Tx - dstPoints[i].x), 2) + pow(abs(Ty - dstPoints[i].y), 2);
		if (val < pow(sigma, 2)) {
			ptl_x[j] = x;
			ptl_y[j] = y;
			ptr_x[j] = dstPoints[i].x;
			ptr_y[j] = dstPoints[i].y;
			j++;
		}
	}
	Mat final = cal_affine<float>(ptr_x, ptr_y, ptl_x, ptl_y, 8);
	return final;
}


template <typename T>
Mat cal_affine(double ptl_x[], double ptl_y[], double ptr_x[], double ptr_y[], int number_of_points) {

	Mat M(2 * number_of_points, 6, CV_32F, Scalar(0));
	Mat b(2 * number_of_points, 1, CV_32F);

	Mat M_trans, temp, affineM;

	// initialize matrix
	for (int i = 0; i < number_of_points; i++) {

		double pf1[6] = { ptl_x[i],ptl_y[i],1,0,0,0 };
		double pf2[6] = { 0,0,0,ptl_x[i],ptl_y[i],1 };

		for (int j = 0; j < 6; j++) {
			M.at<float>(2 * i, j) = pf1[j];
			M.at<float>(2 * i + 1, j) = pf2[j];
		}
		b.at<float>(2 * i, 0) = ptr_x[i];
		b.at<float>(2 * i + 1, 0) = ptr_y[i];
	}

	M_trans = M.t();

	temp = (M_trans * M).inv();

	affineM = temp*M_trans*b;

	return affineM;
}

void blend_stitching(const Mat I1, const Mat I2, Mat &I_f, int diff_x, int diff_y, float alpha) {

	int bound_x = I1.rows + diff_x;
	int bound_y = I1.cols + diff_y;

	int col = I_f.cols;
	int row = I_f.rows;

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			// for check validation of I1 & I2
			bool cond1 = (i < bound_x && i > diff_x) && (j < bound_y && j > diff_y) ? true : false;
			bool cond2 = I_f.at<Vec3f>(i, j) != Vec3f(0, 0, 0) ? true : false;

			// I2 is already in I_f by inverse warping
			// So, It is not necessary to check that only I2 is valid
			// if both are valid
			if (cond1 && cond2) {
				I_f.at<Vec3f>(i, j) = alpha * I1.at<Vec3f>(i - diff_x, j - diff_y) + (1 - alpha) * I_f.at<Vec3f>(i, j);
			}
			// only I1 is valid
			else if (cond1) {
				I_f.at<Vec3f>(i, j) = I1.at<Vec3f>(i - diff_x, j - diff_y);
			}
		}
	}
}

double euclidDistance(Mat& vec1, Mat& vec2) {
	double sum = 0.0;
	int dim = vec1.cols;

	for (int i = 0; i < 1; i++)
		for (int j = 0; j < dim; j++)
		{
			uchar * a = vec1.ptr(i);
			uchar * b = vec2.ptr(i);
			sum += (a[j] - b[j]) *(a[j] - b[j]);
		}
	return sqrt(sum);
}


/**
* Find the index of nearest neighbor point from keypoints.
*/
int nearestNeighbor(Mat& vec, vector<KeyPoint>& keypoints, Mat& descriptors) {
	int neighbor = -1;
	double minDist = 1e6;

	for (int i = 0; i < descriptors.rows; i++) {
		Mat v = descriptors.row(i);		// each row of descriptor
		KeyPoint pt = keypoints[i];

		double d = euclidDistance(vec, v);

		if (d < minDist) {
			minDist = d;
			neighbor = i;
		}
	}
	return neighbor;
}

double nearestNeighbor_dist(Mat& vec, vector<KeyPoint>& keypoints, Mat& descriptors) {
	double minDist = 1e6;

	for (int i = 0; i < descriptors.rows; i++) {
		Mat v = descriptors.row(i);		// each row of descriptor
		KeyPoint pt = keypoints[i];

		double d = euclidDistance(vec, v);

		if (d < minDist) {
			minDist = d;
		}
	}
	return minDist;
}

double secondnearestNeighbor_dist(Mat& vec, vector<KeyPoint>& keypoints, Mat& descriptors) {
	double minDist = 1e6;
	double secondminDist = minDist;

	for (int i = 0; i < descriptors.rows; i++) {
		Mat v = descriptors.row(i);		// each row of descriptor
		KeyPoint pt = keypoints[i];

		double d = euclidDistance(vec, v);

		if (d < minDist) {
			secondminDist = minDist;
			minDist = d;
		}
		else if ((minDist < d && d < secondminDist) || minDist == secondminDist)
			secondminDist = d;
	}
	return secondminDist;
}

/**
* Find pairs of points with the smallest distace between them
*/
void findPairs(vector<KeyPoint>& keypoints1, Mat& descriptors1,
	vector<KeyPoint>& keypoints2, Mat& descriptors2,
	vector<Point2f>& srcPoints, vector<Point2f>& dstPoints, bool crossCheck, bool ratio_threshold) {
	for (int i = 0; i < descriptors1.rows; i++) {
		KeyPoint pt1 = keypoints1[i];
		Mat desc1 = descriptors1.row(i);

		int nn = nearestNeighbor(desc1, keypoints2, descriptors2);

		// Refine matching points using ratio_based thresholding
		if (ratio_threshold) {
			double nn_dist = nearestNeighbor_dist(desc1, keypoints2, descriptors2);
			double nn_second_dist = secondnearestNeighbor_dist(desc1, keypoints2, descriptors2);
			if (nn_dist / nn_second_dist > RATIO_THR)
				continue;
		}

		// Refine matching points using cross-checking
		if (crossCheck) {
			Mat desc2 = descriptors2.row(nn);
			int cc = nearestNeighbor(desc2, keypoints1, descriptors1);
			//keypoints1[cc].pt != pt1.pt
			if (cc != i)
				continue;
		}

		KeyPoint pt2 = keypoints2[nn];
		srcPoints.push_back(pt1.pt);
		dstPoints.push_back(pt2.pt);
	}
}
