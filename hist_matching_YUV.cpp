#include "hist_func.h"
#include <iostream>

void hist_matching(Mat &input, Mat &matching, G *matching_func);
void compute_function(G *trans_func_T, G *trans_func_G, G*matching_func, float *CDF, float *CDF_G);

int main() {

	Mat input = imread("input.jpg", CV_LOAD_IMAGE_COLOR);
	Mat matching_input;

	Mat reference = imread("reference.jpg", CV_LOAD_IMAGE_COLOR);
	Mat matching_reference;

	cvtColor(input, matching_input, CV_RGB2YUV);	// convert RGB to Grayscale
	cvtColor(reference, matching_reference, CV_RGB2YUV);

//	 split each channel(Y, U, V) - input 
	Mat channels[3];
	split(matching_input, channels);
	Mat Y = channels[0];						// U = channels[1], V = channels[2] - input

//	 split each channel(Y, U, V) - reference

	Mat channels_matching[3];
	split(matching_reference, channels_matching);
	Mat Y_matching = channels_matching[0];		// U = channels[1], V = channels[2] - reference

//	 PDF or transfer function txt files
//	 original 
	FILE *f_PDF;
//	 reference 
	FILE *f_refer_PDF;
//	 output 
	FILE *f_matched_PDF_YUV;
//	 trans 
	FILE *f_trans_func_mat_YUV ;

	fopen_s(&f_PDF, "YUV_original_PDF_matching.txt", "w+");
	fopen_s(&f_refer_PDF, "YUV_reference_PDF_matching.txt", "w+");
	fopen_s(&f_matched_PDF_YUV, "YUV_output_PDF_matching.txt", "w+");
	fopen_s(&f_trans_func_mat_YUV, "YUV_trans_func_mat.txt", "w+");

	// Y에 대해서만 CDF를 각각 계산한다.
	float **original_PDF = cal_PDF_RGB(matching_input);	
	float *original_CDF = cal_CDF(Y);

	float **reference_PDF = cal_PDF_RGB(matching_reference);	
	float *reference_CDF = cal_CDF(Y_matching);	

	G trans_func_mat_T[L] = { 0 };			// transfer function - T
	G trans_func_mat_G[L] = { 0 };			// transfer function - G
	G matching_func[L] = { 0 };				// transfer function

	// compute trans function using Y of each image(input, reference)
	compute_function(trans_func_mat_T, trans_func_mat_G, matching_func, original_CDF, reference_CDF);

	//	histogram matching of Y  using matching_func 
	hist_matching(Y, Y, matching_func);

	//	merging channels of input to matching_input  
	merge(channels, 3, matching_input);

	//	change matching_input from YUV to RGB
	cvtColor(matching_input, matching_input, CV_YUV2RGB);

	//	calculate PDF of matching input
	float **matched_PDF_YUV = cal_PDF_RGB(matching_input);

	for (int i = 0; i < L; i++) {
		//		 write original pdf 
		fprintf(f_PDF, "%d\t%f %f %f\n", i, original_PDF[i][0], original_PDF[i][1], original_PDF[i][2]);

		//		write reference pdf
		fprintf(f_refer_PDF, "%d\t%f %f %f\n", i, reference_PDF[i][0], reference_PDF[i][1], reference_PDF[i][2]);

		//		 write output pdf  
		fprintf(f_matched_PDF_YUV, "%d\t%f %f %f\n", i, matched_PDF_YUV[i][0], matched_PDF_YUV[i][1], matched_PDF_YUV[i][2]);

		//		 write transfer functions
		fprintf(f_trans_func_mat_YUV, "%d\t%d\n", i, matching_func[i]);


	}

//	 memory release
	free(original_PDF);
	free(original_CDF);
	free(reference_PDF);
	free(reference_CDF);

	fclose(f_PDF);
	fclose(f_refer_PDF);
	fclose(f_matched_PDF_YUV);
	fclose(f_trans_func_mat_YUV);


	//////////////////// Show each image ///////////////////////

	namedWindow("original", WINDOW_AUTOSIZE);
	imshow("original", input);

	namedWindow("reference", WINDOW_AUTOSIZE);
	imshow("reference", reference);

	namedWindow("Matched", WINDOW_AUTOSIZE);
	imshow("Matched", matching_input);

	////////////////////////////////////////////////////////////

	waitKey(0);

	return 0;
}


void compute_function(G *trans_func_T, G *trans_func_G, G*matching_func, float *CDF, float *CDF_G) {
	//compute transfer function - T
	for (int i = 0; i < L; i++)
		trans_func_T[i] = (G)((L - 1) * CDF[i]);

	//compute transfer function - G
	for (int i = 0; i < L; i++)
		trans_func_G[i] = (G)((L - 1) * CDF_G[i]);

	// compute transfer fucnction 
	for (int i = 0; i < L; i++) {
		int tmp = trans_func_T[i];
		for (int j = 0; j < L; j++) {
			int flag = 0;
			// 역함수를 구해야하기 때문에 G 함수에서 tmp와 맞는 j 값을 mathcing 함수에 넣어준다. 
			if (trans_func_G[j] == tmp) {
				matching_func[i] = j;
				flag = 1;
				break;
			}
			// 만약 matching되는 값이 끝까지 없으면 nearest value (1개 이전의 값)을 대입한다.
			else if (flag == 0 && j == L - 1) {
				matching_func[i] = matching_func[i-1];
			}
		}
	}
}

 //histogram matching
void hist_matching(Mat &input, Mat &matching, G *matching_func) {
	 //perform the transfer function
	for (int i = 0; i < input.rows; i++)
		for (int j = 0; j < input.cols; j++) {
			// matching 함수를 통해 matching image의 pixel intensity를 대입한다.
			matching.at<G>(i,j) = matching_func[input.at<G>(i, j)]; 
		}
}
