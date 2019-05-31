#include "KAZE.h"
#include "utils.h"
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>

#include <fstream>      // std::ofstream
// setprecision example
#include <iostream>     // std::cout, std::fixed
#include <iomanip>      // std::setprecision

using namespace std;
using namespace cv;

void printParams( cv::Algorithm* algorithm ) {
    std::vector<std::string> parameters;
    algorithm->getParams(parameters);

    for (int i = 0; i < (int) parameters.size(); i++) {
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

void dumpMat(const cv::Mat& matp, FILE* ofp)
{	
	uchar depth = matp.type() & CV_MAT_DEPTH_MASK;
	//cout <<int(depth)<<" "<<int(CV_8U)<<endl;
    for(int j = 0; j < matp.rows; j++ ){
        for(int i = 0; i < matp.cols; i++ ){
        	if (depth == CV_8U)
        	{
        		uchar mask = 1;
        		uchar bit = matp.at<uchar>(j,i);
        		for (int k = 0;k<8;k++)
        		{
            		fprintf(ofp,"%d\t",int(bit & mask)>>k);
            		mask = mask<<1;
        		}
            }
        	else
        		fprintf(ofp,"%e\t",matp.at<float>(j,i));
        }
        fprintf(ofp,"\n");
    }
}


// void writeMatToFile(const Mat& m, const string s)
// {
//     cv::FileStorage file(s, cv::FileStorage::WRITE);
//     file <<"m"<<m;
// }


void Tokenize(const std::string & mystring, std::vector < std::string > &tok, const std::string & sep = " ");
std::string delSpaces(std::string & str);

void Tokenize(const std::string & text, std::vector < std::string > &tok,
	      const std::string & sep)
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

int
main(int argc, char *argv[]) {

	if (argc != 4){
		cout << "Usage: <InputImg> <kp name> <OutputKeypointFileWithoutExtention>" << endl;
		return 0;
	}

	// Load the image
	Mat imgColor = imread(argv[1]);
	Mat image;
	cvtColor(imgColor, image, CV_BGR2GRAY );
	image.convertTo(image,CV_32F,1.0/255.0,0);


//load kp
	ifstream ifs_keypoints;
	ifs_keypoints.open(argv[2]);

	std::string lineread;
	std::vector < std::string > tokens;

	getline(ifs_keypoints, lineread);//skip 1
	Tokenize(lineread, tokens);
	int nbVar = stoi(delSpaces(tokens[0]));

	tokens.clear();

	getline(ifs_keypoints, lineread);//
	Tokenize(lineread, tokens);
	int nbkp = stoi(delSpaces(tokens[0]));

	vector<KeyPoint> keypoints(nbkp);

	//get parameters
	int count = 0;
	while (getline(ifs_keypoints, lineread)) 
	{
		tokens.clear();
		Tokenize(lineread, tokens);
		keypoints[count].pt.x 		 = stof(delSpaces(tokens[0]));
		keypoints[count].pt.y 		 = stof(delSpaces(tokens[1]));
		keypoints[count].size 		 = stof(delSpaces(tokens[2])) * 2.0; // we use radius in kp file
		keypoints[count].angle 		 = stof(delSpaces(tokens[3]));
		keypoints[count].response    = stof(delSpaces(tokens[4]));
		keypoints[count].octave 	 = stoi(delSpaces(tokens[5]));
		keypoints[count].class_id    = keypoints[count].octave;
		count++;
	}

	ifs_keypoints.close();

	if (count != nbkp)
		throw std::runtime_error("the number of kp does not match !");

	
	
	//compute(const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors) 
	Mat descriptors;
	
	KAZEOptions options;
	options.img_width = image.cols;
    options.img_height = image.rows;

	  // Create the KAZE object
	  KAZE evolution(options);
	  // Create the nonlinear scale space
	  evolution.Create_Nonlinear_Scale_Space(image);
	  evolution.Feature_Description(keypoints,descriptors);

	const bool resaveKp = true;
	if (resaveKp)
	{
	ofstream ofs_keypoints;
	//ofstream ofs_score;
	ofs_keypoints.open(argv[2], std::ofstream::trunc);
	//ofs_score.open(score_name, std::ofstream::trunc);
	// Save keypoints
	ofs_keypoints << 6 << endl;
	ofs_keypoints << keypoints.size() << endl;
	for(int i=0; i < keypoints.size(); ++i){
		ofs_keypoints << std::setprecision(10) << keypoints[i].pt.x << " ";
		ofs_keypoints << std::setprecision(10) << keypoints[i].pt.y << " ";
		ofs_keypoints << std::setprecision(10) << keypoints[i].size*0.5 << " "; // we radius in our kp file
		ofs_keypoints << std::setprecision(10) << keypoints[i].angle << " ";
		ofs_keypoints << std::setprecision(10) << keypoints[i].response << " ";
		ofs_keypoints << int(keypoints[i].class_id) 			 << endl;
	}
	ofs_keypoints.close();
}



    FILE* ofp = fopen(argv[3],"w");
    dumpMat(descriptors, ofp);
    fclose(ofp);

    exit(0);
}




