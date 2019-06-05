
#include 	<cstdlib>
#include	<iostream>
#include	<fstream>
#include	<string>
#include 	<iomanip>

// #include	<cv.h>
// #include	<highgui.h>
#include	<opencv2/opencv.hpp>
#include	<opencv2/features2d/features2d.hpp>
#include	<opencv2/opencv_modules.hpp>
#include	<opencv2/nonfree/features2d.hpp>
#include	<opencv2/nonfree/nonfree.hpp>
//#include <string>
//#include <bitset>
#include	<opencv2/highgui/highgui.hpp>
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

	if (argc != 5){
		cout << "Usage: <Method> <InputImg> <kp name> <OutputKeypointFileWithoutExtention>" << endl;
		return 0;
	}

	// Load the image
	Mat imgColor = imread(argv[2]);
	Mat image;
	cvtColor(imgColor, image, CV_BGR2GRAY );
	// IplImage oldStyleImage = image;
	// cvSmooth(&oldStyleImage,&oldStyleImage,CV_GAUSSIAN,5,5,1);


//load kp
	ifstream ifs_keypoints;
	ifs_keypoints.open(argv[3]);

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

	float coeff = 1;
	// if (strcmp(argv[1],"BRISK") == 0 || strcmp(argv[1],"FREAK") == 0)
	// {
	// 	//cout<<argv[1]<<": I am changing the scale !!!!!"<<endl;
	// 	coeff = 6;
	// }

	//get parameters
	int count = 0;
	while (getline(ifs_keypoints, lineread)) 
	{
		tokens.clear();
		Tokenize(lineread, tokens);
		//cout<<"val:"<<lineread<<endl;
		keypoints[count].pt.x 		 = stof(delSpaces(tokens[0]));
		keypoints[count].pt.y 		 = stof(delSpaces(tokens[1]));
		keypoints[count].size 		 = stof(delSpaces(tokens[2]))*2.0;// 2.0 as it is diameter and we save radius
		keypoints[count].angle 		 = stof(delSpaces(tokens[3]));
		keypoints[count].response    = stof(delSpaces(tokens[4]));
		keypoints[count].octave 	 = stoi(delSpaces(tokens[5]));
		//tokens.clear();
		//Tokenize(lineread, tokens);
		count++;
	}

	ifs_keypoints.close();

	if (count != nbkp)
		throw std::runtime_error("the number of kp does not match !");

	initModule_nonfree();

//SurfDescriptorExtractor
    Ptr<DescriptorExtractor>		myDescriptor;

	myDescriptor = DescriptorExtractor::create(argv[1]);
	if (!myDescriptor)
		printf("OpenCV was built without keypoint\n" );

	if (strcmp(argv[1],"ORB")==0)
	{
		//printParams(myDescriptor);
		myDescriptor->set("nFeatures",5000);
		myDescriptor->set("nLevels",3);
		//sortMe = false;//orb does it already
	}
	
	//compute(const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors) 
	Mat descriptors;
	myDescriptor->compute(image,keypoints,descriptors);

	if (count != keypoints.size())
		cout<<"descriptor deleted kp, this is wrong !"<<endl;

	// cout<<"I am saving in "<<string(argv[3])+string(".bmp")<<endl;
	// cv::imwrite(string(argv[3])+string(".bmp"),descriptors);

 //    cv::BFMatcher matcher(cv::NORM_HAMMING);
 //  	std::vector< cv::DMatch > matches;

 //  	Mat descriptors1 = descriptors.row(0);
 //  	Mat descriptors2 = descriptors.row(1);
 //    matcher.match(descriptors1, descriptors2, matches);

	// cout<<"distance between the 2 first is "<<matches[0].distance<<endl;

	const bool resaveKp = true;
	if (resaveKp)
	{
	ofstream ofs_keypoints;
	//ofstream ofs_score;
	ofs_keypoints.open(argv[3], std::ofstream::trunc);
	//ofs_score.open(score_name, std::ofstream::trunc);
	// Save keypoints
	ofs_keypoints << 6 << endl;
	ofs_keypoints << keypoints.size() << endl;
	for(int i=0; i < keypoints.size(); ++i){
		ofs_keypoints << std::setprecision(10) << keypoints[i].pt.x << " ";
		ofs_keypoints << std::setprecision(10) << keypoints[i].pt.y << " ";
		ofs_keypoints << std::setprecision(10) << keypoints[i].size*0.5 << " ";// 0.5 as it is diameter
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
    FILE* ofp = fopen(argv[4],"w");
    dumpMat(descriptors, ofp);
    fclose(ofp);
// 	cv::FileStorage fsWrite(argv[4], cv::FileStorage::WRITE );
// 	fsWrite<<"descriptors" << descriptors;
// 	fsWrite.release();

}




