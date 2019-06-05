
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core/core.hpp>

#include <fstream>      // std::ofstream
// setprecision example
#include <iostream>     // std::cout, std::fixed
#include <iomanip>      // std::setprecision

#include "sgloh.h"

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
	image.convertTo(image,CV_64F,1/255.);  //again, it is double, so 64F

	bmp_img gray;
	gray.row = image.cols;
	gray.col = image.rows;//here less sure why we reverse row and cols, but this has been debugged with harrisz....

	gray.val =(double*)malloc(sizeof(double)*gray.row*gray.col);
	memcpy(gray.val,image.data,sizeof(double)*gray.row*gray.col);


//load kp
	ifstream ifs_keypoints;
	ifs_keypoints.open(argv[2]);

	std::string lineread;
	std::vector < std::string > tokens;

	getline(ifs_keypoints, lineread);//skip 1
	Tokenize(lineread, tokens);
	int nbVar = stoi(delSpaces(tokens[0]));

	if (nbVar <9)
		throw std::runtime_error("this method expects 9 parameters (e.g 6 parameters + the equation of the ellipse a,b,c)");

	tokens.clear();

	getline(ifs_keypoints, lineread);//
	Tokenize(lineread, tokens);
	int nbkp = stoi(delSpaces(tokens[0]));


	//vector<KeyPoint> keypoints(nbkp);
	//aff_pt rosy_from_mk_pt(mk_pt in) 

	//get parameters
	int count = 0;
	h_pt feat;feat.n = nbkp;
	feat.aff=(aff_pt *)malloc(sizeof(aff_pt)*feat.n);

	while (getline(ifs_keypoints, lineread)) 
	{
		tokens.clear();
		Tokenize(lineread, tokens);

		mk_pt m;
		//cout<<"val:"<<lineread<<endl;
		m.p[0] 		 = stof(delSpaces(tokens[0]));
		m.p[1] 		 = stof(delSpaces(tokens[1]));
		m.p[2] 		 = stof(delSpaces(tokens[6]));
		m.p[3] 		 = stof(delSpaces(tokens[7]));
		m.p[4] 		 = stof(delSpaces(tokens[8]));

		double angle_d = stof(delSpaces(tokens[3]));
		
		feat.aff[count]=rosy_from_mk_pt(m, angle_d);
		count++;
	}

	ifs_keypoints.close();

	
	if (count != nbkp)
		throw std::runtime_error("the number of kp does not match !");

	
	
	//compute(const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors) 
	Mat descriptors;
	// RIFFDescriptor riff;
	// riff.Descriptor_Generation(image,descriptors,keypoints);
	opt_struct opt;
	DEFAULT_OPTS
	opt.data.img=strdup(argv[1]);
	opt.data.in=NULL;//kp already loaded
	opt.data.out=strdup(argv[3]);

	i_table bin;
 	bin=rosy_init(opt);
 	rosy_go(opt,gray,feat,bin);
 	rosy_end(opt,gray,feat,bin);
}




