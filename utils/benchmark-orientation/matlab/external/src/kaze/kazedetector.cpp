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


using namespace cv;
using namespace std;

int
main(int argc, char *argv[]) {
	if (argc < 3){
		cout << "Usage: <InputImg> <OutputKeypointFileWithoutExtention>" << endl;
		return 0;
	}

///////////////Parameters
		//remove
	const float coeffDescriptorScaleOut = 4.5f;
	bool sortMe = true;
	bool cleanKeyPoints = false;
	if (argc == 5){
		cleanKeyPoints = atoi(argv[3]);
	}

	// Load the image
	Mat imgColor = imread(argv[1]);
	Mat image;
	cvtColor(imgColor, image, CV_BGR2GRAY );
	image.convertTo(image,CV_32F,1.0/255.0,0);

	KAZEOptions options;
	options.img_width = image.cols;
    options.img_height = image.rows;



	int count = 0;
	
	vector<KeyPoint> keypoints;
	// Create the KAZE object
  	KAZE evolution(options);

  	// Create the nonlinear scale space
  	evolution.Create_Nonlinear_Scale_Space(image);
  	evolution.Feature_Detection(keypoints);
	
	//recompute orentation
	if (sortMe)
	{
		sort(keypoints.begin(),keypoints.end(),[](const KeyPoint & a, const KeyPoint & b) -> bool { return a.response> b.response;});
	}


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
		ofs_keypoints << std::setprecision(10) << keypoints[i].size*0.5 << " "; // We use radius in our kp file
		ofs_keypoints << std::setprecision(10) << keypoints[i].angle << " ";
		//cout<<"octave number: "<< int(keypoints[i].octave) <<endl;
		//ofs_keypoints << endl;
		ofs_keypoints << std::setprecision(10) << keypoints[i].response << " ";
		ofs_keypoints << int(keypoints[i].class_id) 					 << endl;//HACK: kaze descriptor need classid :(, so we save it as octave 
	}
	ofs_keypoints.close();
	//ofs_score.close();
}
