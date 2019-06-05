
#include 	<cstdlib>
#include	<iostream>
#include	<fstream>
#include	<string>
#include 	<iomanip>

#include "freak.h"


#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/legacy/legacy.hpp>

#include 	<cstdlib>
#include	<iostream>
#include	<fstream>
#include	<string>
#include 	<iomanip>
#include <list>


#include <bitset>

using namespace std;
using namespace cv;

void dumpMat(const cv::Mat& matp, FILE* ofp)
{	
	uchar depth = matp.type() & CV_MAT_DEPTH_MASK;
	//cout <<int(depth)<<" "<<int(CV_8U)<<endl;
    for(int j = 0; j < matp.rows; j++ ){
        for(int i = 0; i < matp.cols; i++ ){
        	if (depth == CV_8U)
        	{
        		bitset<8> bit = matp.at<uchar>(j,i);
        		for (int k = 0;k<8;k++)
            		fprintf(ofp,"%d\t",bit[k] & 0x1);
            }
        	else
        		fprintf(ofp,"%e\t",matp.at<float>(j,i));
        }
        fprintf(ofp,"\n");
    }
}



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

	const bool resaveKp = true;

	if (argc != 4){
		cout << "Usage: <InputImg> <kp name> <OutputKeypointFileWithoutExtention>" << endl;
		return 0;
	}

	// Load the image
	Mat imgColor = imread(argv[1]);
	Mat image;
	cvtColor(imgColor, image, CV_BGR2GRAY );



//load kp
	ifstream ifs_keypoints;
	ifs_keypoints.open(argv[2]);

	std::string lineread;
	std::vector < std::string > tokens;

	getline(ifs_keypoints, lineread);//skip 1
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
		//cout<<"val:"<<lineread<<endl;
		keypoints[count].pt.x 		 = stof(delSpaces(tokens[0]));
		keypoints[count].pt.y 		 = stof(delSpaces(tokens[1]));
		keypoints[count].size 		 = stof(delSpaces(tokens[2]));
		keypoints[count].angle 		 = stof(delSpaces(tokens[3]));
		keypoints[count].response    = stof(delSpaces(tokens[4]));
		keypoints[count].octave 	 = stoi(delSpaces(tokens[5]));
		//tokens.clear();
		//Tokenize(lineread, tokens);
		count++;
	}

	ifs_keypoints.close();


	//initModule_nonfree();

//SurfDescriptorExtractor
    Ptr<original::FREAK>		myDescriptor;

	myDescriptor = new original::FREAK();
	if (!myDescriptor)
		printf("OpenCV was built without keypoint\n" );
	
	//compute(const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors) 
	Mat descriptors;
	myDescriptor->compute(image,keypoints,descriptors);

	if (!resaveKp)
		if (keypoints.size() != count)
		{
			cout<<"FREAK deletes kp !!! ,this is not good"<<endl;
		}
	//cout<<"seize kp "<<keypoints.size()<<" "<<descriptors.rows<<" "<<descriptors.cols<<endl;
	//string score_name(argv[3]); score_name.append(".score");
	
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
			ofs_keypoints << std::setprecision(10) << keypoints[i].size << " ";
			ofs_keypoints << std::setprecision(10) << keypoints[i].angle << " ";
			//cout<<"octave number: "<< int(keypoints[i].octave) <<endl;
			//ofs_keypoints << endl;
			ofs_keypoints << std::setprecision(10) << keypoints[i].response << " ";
			ofs_keypoints << int(keypoints[i].octave) 					 << endl;
		}
		ofs_keypoints.close();
	}


    // char buf[100];
    // sprintf(buf,argv[4]);
    FILE* ofp = fopen(argv[3],"w");
    dumpMat(descriptors, ofp);
    fclose(ofp);
// 	cv::FileStorage fsWrite(argv[4], cv::FileStorage::WRITE );
// 	fsWrite<<"descriptors" << descriptors;
// 	fsWrite.release();

}




