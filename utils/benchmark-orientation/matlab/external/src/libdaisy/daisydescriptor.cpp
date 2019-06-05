
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core/core.hpp>

#include <fstream>		// std::ofstream
// setprecision example
#include <iostream>		// std::cout, std::fixed
#include <iomanip>		// std::setprecision

#include "daisy/daisy.h"

using namespace std;
using namespace cv;

void printParams(cv::Algorithm * algorithm)
{
	std::vector < std::string > parameters;
	algorithm->getParams(parameters);

	for (int i = 0; i < (int)parameters.size(); i++) {
		std::string param = parameters[i];
		int type = algorithm->paramType(param);
		std::string helpText = algorithm->paramHelp(param);
		std::string typeText;

		switch (type) {
		case cv::Param::BOOLEAN:
			typeText = "bool";
			break;
		case cv::Param::INT:
			typeText = "int";
			break;
		case cv::Param::REAL:
			typeText = "real (double)";
			break;
		case cv::Param::STRING:
			typeText = "string";
			break;
		case cv::Param::MAT:
			typeText = "Mat";
			break;
		case cv::Param::ALGORITHM:
			typeText = "Algorithm";
			break;
		case cv::Param::MAT_VECTOR:
			typeText = "Mat vector";
			break;
		}
		std::cout << "Parameter '" << param << "' type=" << typeText << " help=" << helpText << std::endl;
	}
}

void dumpMat(const cv::Mat & matp, FILE * ofp)
{
	uchar depth = matp.type() & CV_MAT_DEPTH_MASK;
	//cout <<int(depth)<<" "<<int(CV_8U)<<endl;
	for (int j = 0; j < matp.rows; j++) {
		for (int i = 0; i < matp.cols; i++) {
			if (depth == CV_8U) {
				uchar mask = 1;
				uchar bit = matp.at < uchar > (j, i);
				for (int k = 0; k < 8; k++) {
					fprintf(ofp, "%d\t", int (bit & mask) >> k);
					mask = mask << 1;
				}
			} else
				fprintf(ofp, "%e\t", matp.at < float >(j, i));
		}
		fprintf(ofp, "\n");
	}
}

void rotate(cv::Mat & src, double angle, cv::Mat & dst)
{
	int len = std::max(src.cols, src.rows);
	cv::Point2f pt(len / 2., len / 2.);
	cv::Mat r = cv::getRotationMatrix2D(pt, angle, 1.0);

	cv::warpAffine(src, dst, r, cv::Size(len, len));
	dst = dst(Rect(Point(0, 0), src.size())).clone();
}

//Yannick's original implementation
void extractPatches(const cv::Mat & image, const std::vector < cv::KeyPoint > &kp, const float maxscale, const int sizePatch, std::vector < cv::Mat > &patches)
{
	//int M = sizePatch;
	const float biggerScaleCoeff = 4;

	// let border be the same in all directions
	const int border = max < int >(1, ceil(maxscale * biggerScaleCoeff * sqrt(2)));
	cout << "using mirroring of size " << border << endl;
	// constructs a larger image to fit both the image and the border
	Mat gray_buf(image.rows + border * 2, image.cols + border * 2, image.type());
	// form a border in-place
	copyMakeBorder(image, gray_buf, border, border, border, border, BORDER_REFLECT);
	//imshow("img padded",gray_buf);
	for (auto & it:kp) {
		cv::Mat patch(Size(sizePatch, sizePatch), image.type());
		cv::Mat buf, buf2;
		int bigHalfPatch = max < int >(3, ceil(it.size * biggerScaleCoeff));

		if (bigHalfPatch < 3)
			throw std::runtime_error("scale is too small, so we cannot get a right size for the patch");

		Point p1(round(border + it.pt.x - bigHalfPatch * sqrt(2)), round(border + it.pt.y - bigHalfPatch * sqrt(2)));
		Size s1(round(2 * bigHalfPatch * sqrt(2)), round(2 * bigHalfPatch * sqrt(2)));

		buf = gray_buf(Rect(p1, s1)).clone();
		//imshow("patch buf",buf);waitKey(0);
		//cout<<"angle is "<<it.angle<<endl;
		rotate(buf, it.angle, buf2);
		//imshow("patch buf rot",buf2);waitKey(0);
		//cout<<bigHalfPatch<<" "<<buf2.size()<<endl;
		p1 = Point(buf2.cols / 2. - bigHalfPatch, buf2.rows / 2. - bigHalfPatch);
		s1 = Size(round(2 * bigHalfPatch), round(2 * bigHalfPatch));
		//cout<<p1<<" "<<buf2.size()<<endl;
		cv::resize(buf2(Rect(p1, s1)), patch, patch.size());

		//imshow("patch",patch);waitKey(0);
		//patch.convertTo(patch, CV_32F, 1./255.);
		// patch = patch.isContinuous() ? patch : patch.clone();
		patches.push_back(patch);	// - m[0]/255.);
	}
}

// Kwang's code
void extractPatches(const cv::Mat & image, const std::vector < std::vector<float> > &kp, const int sizePatch, std::vector < cv::Mat > &patches)
{
	//int M = sizePatch;
	const float biggerScaleCoeff = 4;

	cv::Mat M(cv::Size(3,2),CV_32F);	// the transformation matrix for affine
	for (int ii=0; ii < kp.size(); ++ii){

		// Affine is as follows in kp
		// 9,  11
		// 10, 12
		// Divide here so that we get patch size
	    M.at<float>(0,0) = biggerScaleCoeff * kp[ii][9] / ((float)sizePatch * 0.5);
	    M.at<float>(1,0) = biggerScaleCoeff * kp[ii][10] / ((float)sizePatch * 0.5);
	    M.at<float>(0,1) = biggerScaleCoeff * kp[ii][11] / ((float)sizePatch * 0.5);
		M.at<float>(1,1) = biggerScaleCoeff * kp[ii][12] / ((float)sizePatch * 0.5);

		// add bias to translation so that we get from -1 to 1
		float tx = kp[ii][0] - biggerScaleCoeff * (kp[ii][9] + kp[ii][11]);
		float ty = kp[ii][1] - biggerScaleCoeff * (kp[ii][10] + kp[ii][12]);		
		M.at<float>(0,2) = tx;
		M.at<float>(1,2) = ty;

		// cout << M << endl;
		// cout << M.rows << ", " << M.cols << endl;

		// Get the patch
		cv::Mat patch(cv::Size(sizePatch, sizePatch), image.type());
	    cv::warpAffine(image, patch, M, cv::Size(sizePatch,sizePatch), CV_WARP_INVERSE_MAP | CV_INTER_CUBIC, BORDER_REFLECT101);
		
		patches.push_back(patch);

	}
}



void Tokenize(const std::string & mystring, std::vector < std::string > &tok, const std::string & sep = " ");
std::string delSpaces(std::string & str);

void Tokenize(const std::string & text, std::vector < std::string > &tok, const std::string & sep)
{
	int start = 0, end = 0;
	while ((end = text.find(sep, start)) != string::npos) {
		tok.push_back(text.substr(start, end - start));
		start = end + 1;
	}
	tok.push_back(text.substr(start));
}

std::string delSpaces(std::string & str)
{
	std::stringstream trim;
	trim << str;
	trim >> str;
	return str;
}

int main(int argc, char *argv[])
{

	if (argc != 4) {
		cout << "Usage: <InputImg> <kp name> <OutputKeypointFileWithoutExtention>" << endl;
		return 0;
		//./originalDaLIdescriptor ~/GitData/Datasets/OxfordRelease/graf_test/test/image_gray/img1.png  ~/GitDaceiceigso  Makefileefilet.cppta/Datasets/OxfordRelease/graf_test/test/features/img1_SIFT_keypoints-none-txt out.txt
	}
	// Load the image
	Mat imgColor = imread(argv[1]);
	Mat image;
	cvtColor(imgColor, image, CV_BGR2GRAY);
	//image.convertTo(image,CV_32F,1.0/255.0,0);

//load kp
	ifstream ifs_keypoints;
	ifs_keypoints.open(argv[2]);

	std::string lineread;
	std::vector < std::string > tokens;

	Tokenize(argv[1], tokens, "/");
	string nameImgAlone = tokens[tokens.size() - 1];
	cout << "name is " << nameImgAlone << endl;
	tokens.clear();

	getline(ifs_keypoints, lineread);	//skip 1
	Tokenize(lineread, tokens);
	int nbVar = stoi(delSpaces(tokens[0]));

	tokens.clear();

	getline(ifs_keypoints, lineread);	//
	Tokenize(lineread, tokens);
	int nbkp = stoi(delSpaces(tokens[0]));

	// vector < KeyPoint > keypoints(nbkp);
	vector < vector<float> > keypoints_orig(nbkp);
	for (int ii=0; ii < keypoints_orig.size(); ++ii){
		keypoints_orig[ii] = std::vector<float>(13);
	}

	//get parameters
	int count = 0;
	float maxscale = 0;

	while (getline(ifs_keypoints, lineread)) {
		tokens.clear();
		Tokenize(lineread, tokens);
		for (int ii=0; ii < 13; ++ii){
			keypoints_orig[count][ii] = stof(delSpaces(tokens[ii]));
		}

		if (maxscale < keypoints_orig[count][2])
			maxscale = keypoints_orig[count][2];
		
		count++;
	}
	
	// for (int ii=0; ii < keypoints_orig.size(); ++ii){
	// 	keypoints[ii].pt.x = keypoints_orig[ii][0];
	// 	keypoints[ii].pt.y = keypoints_orig[ii][1];
	// 	keypoints[ii].size = keypoints_orig[ii][2];
	// 	keypoints[ii].angle = keypoints_orig[ii][3];
	// 	keypoints[ii].response = keypoints_orig[ii][4];
	// 	keypoints[ii].octave = (int)(keypoints_orig[ii][5]);
	// 	keypoints[ii].class_id = keypoints[ii].octave;
	// }

	ifs_keypoints.close();

	if (count != nbkp)
		throw std::runtime_error("the number of kp does not match !");

	Mat descriptors;

	const int sizePatch = 41;	//use coeff of 4 with the scale (hard coded in extractPatches)
	// std::vector < cv::Mat > patches1;
	std::vector < cv::Mat > patches;
	// extractPatches(image, keypoints, maxscale, sizePatch, patches1);
	extractPatches(image, keypoints_orig, sizePatch, patches);

	// Validated the two patch extractions work fine
	// for (int ii=0 ; ii < patches1.size(); ++ii){
	// 	cv::imshow("PATCH_YAN", patches1[ii]);
	// 	cv::imshow("PATCH_KWANG",patches[ii]);
	// 	cv::waitKey(-1);
	// }
	

	const int rad = 15;
	const int radq = 3;
	const int thq = 8;
	const int histq = 8;

	FILE *fid = fopen(argv[3], "w");
	for (auto & p:patches) {
		daisy *desc = new daisy();

		//Mat ucharp; p.convertTo(ucharp,CV_8U, 255.0);
		desc->set_image(p.data, sizePatch, sizePatch);
		desc->verbose(0);
		desc->set_parameters(rad, radq, thq, histq);	// default values are 15,3,8,8
		desc->initialize_single_descriptor_mode();
		desc->compute_descriptors();	// precompute all the descriptors (NOT NORMALIZED!)
		// the descriptors are not normalized yet
		desc->normalize_descriptors();

		float *thor = new float[desc->descriptor_size()];
		memset(thor, 0, sizeof(float) * desc->descriptor_size());

		desc->get_descriptor(sizePatch / 2, sizePatch / 2, 0, thor);
		//desc->save_descriptors_ascii(outname);

		for (int j = 0; j < desc->descriptor_size(); j++) {
			fprintf(fid, "%f ", thor[j]);
		}
		fprintf(fid, "\n");

		delete[]thor;

		delete desc;
	}
	fclose(fid);

	exit(0);

}
