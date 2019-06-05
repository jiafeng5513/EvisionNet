#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/legacy/legacy.hpp>
#include "brisk/brisk.h"
#include 	<cstdlib>
#include	<iostream>
#include	<fstream>
#include	<string>
#include 	<iomanip>
#include <list>


using namespace std;
using namespace cv;

int
main(int argc, char *argv[]) {

	if (argc != 3){
		cout << "Usage: <InputImg> <OutputKeypointFileWithoutExtention>" << endl;
		return 0;
	}

///////////////Parameters
		//remove
	const bool sortMe = true;
	const float coeffDescriptorScaleOut = 4.5f;
	const bool removeScaleOut = false;


	// Load the image
	Mat imgColor = imread(argv[1]);
	Mat image;
	cvtColor(imgColor, image, CV_BGR2GRAY );

	//initModule_nonfree();

    Ptr<FeatureDetector>		myDetector;

    float threshold = 20;//40 for common_part = 1, 115 for common_part = 0 non-rescaled, 67 for common_part = 0 rescaled 
	myDetector = new cv::BriskFeatureDetector(threshold,4);
	if (!myDetector)
		printf("OpenCV was built without keypoint\n" );


	//cout<<"here"<<endl;
	// Detect keypoints
	int count = 0;
	
	vector<KeyPoint> keypoints;
	// do
	// {
	 	myDetector->detect(image, keypoints);
	// 	if (keypoints.size() < 2000)
	// 	{
	// 		threshold*=0.8;
	// 		//myDetector.delete_obj();
	// 		myDetector->getKeypoints(threshold,keypoints);
	// 	}
		 
	// }
	// while(keypoints.size() < 2000 && count++ < 10);

	// if (keypoints.size() < 2000)
	// 	cout<<"WARNING, could not get more than 2000 initial KeyPoints..."<<endl;

	//cout<<keypoints.size()<<endl;

	cv::BriskDescriptorExtractor* descriptor_extractor = new cv::BriskDescriptorExtractor();
    descriptor_extractor->computeAngles(image, keypoints);

    if (removeScaleOut)
    {
		vector<KeyPoint> tmp;
		for(int i=0; i < keypoints.size(); ++i)
		{
			bool dontTakeMe = false;
			for(int j=i+1; j < keypoints.size() && !dontTakeMe; ++j)
			{
				if ( keypoints[i].pt.x ==  keypoints[j].pt.x &&  keypoints[i].pt.y == keypoints[j].pt.y && keypoints[i].size == keypoints[j].size)
				{
					dontTakeMe = true;
				}
			}


			// //patch is partially out of the image
			// float half_crop_size = round(keypoints[i].size*coeffDescriptorScaleOut);
			// if (round(keypoints[i].pt.x)-half_crop_size < 0
			// 	|| round(keypoints[i].pt.x)+half_crop_size > image.cols-1
			// 	|| round(keypoints[i].pt.y)-half_crop_size < 0
			// 	|| round(keypoints[i].pt.y)+half_crop_size > image.rows-1)
			// {
			// 		dontTakeMe = true;
			// }



			if (!dontTakeMe)
			{
				tmp.push_back(keypoints[i]);
			}
		}
		keypoints = tmp;
	//recompute orentation
	}

	if (sortMe)
	{
		sort(keypoints.begin(),keypoints.end(),[](const KeyPoint & a, const KeyPoint & b) -> bool { return a.response> b.response;});
	}

	//cout<<keypoints.size()<<endl;

	//string score_name(argv[3]); score_name.append(".score");
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
	//ofs_score.close();

}




