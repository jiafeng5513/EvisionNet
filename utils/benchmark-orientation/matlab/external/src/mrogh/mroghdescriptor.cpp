
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/legacy/legacy.hpp>

#include <fstream>      // std::ofstream
// setprecision example
#include <iostream>     // std::cout, std::fixed
#include <iomanip>      // std::setprecision

#include "mrogh.h"

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

	// printf("reading image %s\n",argv[1]);
	
	// Load the image
	Mat imgColor = imread(argv[1]);
	//Mat image;
	//IplImage * p_oldStyleImage;
    cv::Mat image;//(p_oldStyleImage); 
	cvtColor(imgColor, image, CV_BGR2GRAY );
	IplImage oldStyleImage = image;
	cvSmooth(&oldStyleImage,&oldStyleImage,CV_GAUSSIAN,5,5,1);
	// cvShowImage("image",&oldStyleImage);
	// cv::waitKey(100);

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
	int m_nKeys = stoi(delSpaces(tokens[0]));


	//vector<KeyPoint> keypoints(nbkp);
	//vector<OxKey> keypoints(nbkp);
	//aff_pt rosy_from_mk_pt(mk_pt in) 

	//get parameters
	int count = 0;
	// h_pt feat;feat.n = nbkp;
	// feat.aff=(aff_pt *)malloc(sizeof(aff_pt)*feat.n);
	OxKey *m_pKeys = (OxKey *)malloc(sizeof(OxKey)*m_nKeys);

	while (getline(ifs_keypoints, lineread)) 
	{
		tokens.clear();
		Tokenize(lineread, tokens);

		OxKey m;
		//cout<<"val:"<<lineread<<endl;
		m.x 		 = stof(delSpaces(tokens[0]));
		m.y 		 = stof(delSpaces(tokens[1]));
		m.a		 	 = stof(delSpaces(tokens[6]));
		m.b 		 = stof(delSpaces(tokens[7]));
		m.c 		 = stof(delSpaces(tokens[8]));
		m.angle 	 = stof(delSpaces(tokens[3]));

		// printf("ANGLE = %f\n",m.angle);
		
		// feat.aff[count]=rosy_from_mk_pt(m);
		m_pKeys[count] = (m);
		count++;
	}

	ifs_keypoints.close();


	if (count != m_nKeys)
		throw std::runtime_error("the number of kp does not match !");


	// Note that the CalcuTrans now is using VLFeat Mapping +
	// Orientation for Proper computation
	CalcuTrans(m_pKeys,m_nKeys);

	// printf(" %f %f \n %f %f \n",m_pKeys[0].trans[0], m_pKeys[0].trans[1], m_pKeys[0].trans[2], m_pKeys[0].trans[3]);
	
	int nDir = 8, nOrder = 6, nMultiRegion = 4;
	int m_Dim = nDir * nOrder * nMultiRegion;

	FILE* fid = fopen(argv[3],"w");
	for (int i = 0;i < m_nKeys;i++)
	{
		int *desc = 0;
		desc = Extract_MROGH(m_pKeys[i],&oldStyleImage,nDir,nOrder,nMultiRegion);
		if ( !desc )	continue;
		// fprintf(fid,"%f %f %f %f %f",m_pKeys[i].x,m_pKeys[i].y,m_pKeys[i].a,m_pKeys[i].b,m_pKeys[i].c);
		for (int j = 0;j < m_Dim;j++)
		{
			fprintf(fid," %d\t",desc[j]);
		}
		fprintf(fid,"\n");
		delete [] desc;
	}
	fclose(fid);

	delete [] m_pKeys;
	
	exit(0);
}




