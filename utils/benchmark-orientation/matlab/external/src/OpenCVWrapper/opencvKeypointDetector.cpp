
#include 	<cstdlib>
#include	<iostream>
#include	<fstream>
#include	<string>
#include 	<iomanip>


#include	<opencv2/opencv.hpp>
#include	<opencv2/features2d/features2d.hpp>
#include	<opencv2/opencv_modules.hpp>
#include	<opencv2/nonfree/features2d.hpp>
#include	<opencv2/nonfree/nonfree.hpp>

#include	<opencv2/highgui/highgui.hpp>


#include "sift.hpp"

// KMYI  
// The globals which were originally class varaibles...
//int nfeatures = 0;
int nOctaveLayers = 3;
//double contrastThreshold = 0.04;
//double edgeThreshold = 10;
double sigma = 1.6;

int firstOctave = -1, actualNOctaves = 0, actualNLayers = 0;


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


int
main(int argc, char *argv[]) {

	if (argc < 4){
		cout << "Usage: <Method> <InputImg> <OutputKeypointFileWithoutExtention>" << endl;
		return 0;
	}

///////////////Parameters
	//remove
	bool sortMe = true;
	const float coeffDescriptorScaleOut = 7.5f;
	bool cleanKeyPoints = false;

	if (argc == 5){
		cleanKeyPoints = atoi(argv[4]);
	}

	// Load the image
	Mat imgColor = imread(argv[2]);
	Mat image;
	cvtColor(imgColor, image, CV_BGR2GRAY );

	initModule_nonfree();

    Ptr<FeatureDetector>		myDetector;

	myDetector = FeatureDetector::create(argv[1]);
	if (!myDetector)
		printf("OpenCV was built without keypoint\n" );

	if (strcmp(argv[1],"ORB")==0)
	{
		//printParams(myDetector);
		myDetector->set("nFeatures",1000);
		myDetector->set("nLevels",3);
		sortMe = false;//orb does it already
	}

	
	float threshold;
	// string nameP = "thres";
	// if (std::string("SIFT").compare(string(argv[1])) == 0)
	// {
	// 	nameP = "contrastThreshold";
	// threshold = 0.04;
	// myDetector->set("contrastThreshold", threshold);//default 0.04
	//myDetector->set("edgeThreshold",12);//default 10
	// }else
	// {
	// 	nameP = "thres";
	// 	myDetector->set("octaves", 4);
	// }


	// Detect keypoints
	int count = 0;
	
	vector<KeyPoint> keypoints;
	//do
	//{
	myDetector->detect(image, keypoints);
	// myDetector->set(nameP, threshold);
	//  threshold*=0.9;
	//}
	//while(keypoints.size() < 1500 && count++ < 10);

	// if (keypoints.size() < 1500)
	// 	cout<<"WARNING, could not get more than 1000 initial KeyPoints..."<<endl;

	//cout<<keypoints.size()<<endl;

	// // remove small keypoints for SIFT
	// if (strcmp(argv[1],"SIFT") == 0){
	// 	vector<KeyPoint> tmp;

	// 	for(int i=0; i < keypoints.size(); ++i){
	// 		if (keypoints[i].size >= 4){
	// 			tmp.push_back(keypoints[i]);
	// 		}
	// 	}
	// 	keypoints = tmp;
	// }

	if (cleanKeyPoints)
	{
		vector<KeyPoint> tmp;
		for(int i=0; i < keypoints.size(); ++i)
		{
			bool dontTakeMe = false;
			//if (cleanKeyPoints)

			for(int j=i+1; j < keypoints.size() && !dontTakeMe; ++j)
			{
				if ( keypoints[i].pt.x ==  keypoints[j].pt.x &&  keypoints[i].pt.y == keypoints[j].pt.y && keypoints[i].size == keypoints[j].size)
				{
					dontTakeMe = true;
				}
			}


			//patch is partially out of the image
			float half_crop_size = round(keypoints[i].size*coeffDescriptorScaleOut*0.5); // 0.5 as it is diameter
			//if (cleanKeyPoints)
			if (round(keypoints[i].pt.x)-half_crop_size < 0
				|| round(keypoints[i].pt.x)+half_crop_size > image.cols-1
				|| round(keypoints[i].pt.y)-half_crop_size < 0
				|| round(keypoints[i].pt.y)+half_crop_size > image.rows-1)
			{
				dontTakeMe = true;
			}



			if (!dontTakeMe)
			{
				tmp.push_back(keypoints[i]);
			}
		}
		keypoints = tmp;
	}
	//recompute orentation

	if (cleanKeyPoints)
	{
		firstOctave = 0;
		int maxOctave = INT_MIN;
		for (size_t i = 0; i < keypoints.size(); i++) {
			int octave, layer;
			float scale;
			unpackOctave(keypoints[i], octave, layer, scale);
			firstOctave = std::min(firstOctave, octave);
			maxOctave = std::max(maxOctave, octave);
			actualNLayers = std::max(actualNLayers, layer - 2);
		}

		firstOctave = std::min(firstOctave, 0);
		CV_Assert(firstOctave >= -1 && actualNLayers <= nOctaveLayers);
		actualNOctaves = maxOctave - firstOctave + 1;
	
		Mat base = createInitialImage(image, firstOctave < 0, (float)sigma);
		vector < Mat > gpyr, dogpyr;
		int nOctaves =
		    actualNOctaves >
		    0 ? actualNOctaves : cvRound(log((double)std::min(base.cols, base.rows)) /
					log(2.) - 2) - firstOctave;

		buildGaussianPyramid(base, gpyr, nOctaves);
		buildDoGPyramid(gpyr, dogpyr);

		const bool bSingleOrientation = true;
		vector < double *>hists;
		funcRecomputeOrientation(gpyr, dogpyr, keypoints, hists, bSingleOrientation);
	}

	if (sortMe)
	{
		sort(keypoints.begin(),keypoints.end(),[](const KeyPoint & a, const KeyPoint & b) -> bool { return a.response> b.response;});
	}


	//string score_name(argv[3]); score_name.append(".score");
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
	//ofs_score.close();

}




