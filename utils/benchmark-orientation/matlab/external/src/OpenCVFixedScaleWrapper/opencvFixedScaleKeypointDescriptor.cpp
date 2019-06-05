
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

void dumpMat(const std::vector< cv::Mat > matp, FILE* ofp)
{	
	uchar depth = matp[0].type() & CV_MAT_DEPTH_MASK;
	//cout <<int(depth)<<" "<<int(CV_8U)<<endl;
    for(int j = 0; j < matp.size(); j++ ){
        for(int i = 0; i < matp[j].cols; i++ ){
        	if (depth == CV_8U)
        	{
        		uchar mask = 1;
        		uchar bit = matp[j].at<uchar>(0,i);
        		for (int k = 0;k<8;k++)
        		{
            		fprintf(ofp,"%d\t",int(bit & mask)>>k);
            		mask = mask<<1;
        		}
            }
        	else
        		fprintf(ofp,"%e\t",matp[j].at<float>(0,i));
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

// Kwang's code
void extractPatches(const cv::Mat & image, const std::vector < std::vector<float> > &kp, const int sizePatch, std::vector < cv::Mat > &patches, const float biggerScaleCoeff, const int interMethod = CV_INTER_CUBIC)
{
	//int M = sizePatch;
	// const float biggerScaleCoeff = 4;

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

// Patch Extraction for SURF
// WARINING: IT WOULD ONLY WORK FOR OPENCV KP!
void extractPatchesSURF(const cv::Mat & image, const std::vector < std::vector<float> > &kp,  std::vector < cv::Mat > &patches)
{

	for (int ii=0; ii < kp.size(); ++ii){

		const int PATCH_SZ = 20;

		// if my SURF scale is 9.0/1.2 / 2.0 then I should have to crop 10.5 (i.e. SURF_s == 1)
		// In this case, scale fed to this guy should be 10.5 / 7.5
		// Which means that the conversion from SURF scale to this scale is
		float fRescale = (10.5 / 7.5) / (9.0 / 1.2 / 2.0);

		// Also since we need s from SURF size we get
		float SURF_scale = 1./fRescale * kp[ii][2];
		float SURF_size = 2.0 * SURF_scale;
		float s = SURF_size * 1.2 / 9.0;
		
		int win_size = (int)((PATCH_SZ+1)*s);
		
		cv::Mat win(win_size, win_size, CV_8U);

		// float descriptor_dir = std::fmod(kp[ii][3] - 90.0 + 720.0, 360.0); // SURF does additional 90 when upright provided
		float descriptor_dir = kp[ii][3]; // SURF does additional 90 when upright provided
		descriptor_dir *= (float)(CV_PI/180);
		float sin_dir = -std::sin(descriptor_dir);
		float cos_dir =  std::cos(descriptor_dir);

		/* Subpixel interpolation version (slower). Subpixel not required since
		   the pixels will all get averaged when we scale down to 20 pixels */
		/*
		  float w[] = { cos_dir, sin_dir, center.x,
		  -sin_dir, cos_dir , center.y };
		  CvMat W = cvMat(2, 3, CV_32F, w);
		  cvGetQuadrangleSubPix( img, &win, &W );
		*/

		// Nearest neighbour version (faster)
		float win_offset = -(float)(win_size-1)/2;
		float start_x = kp[ii][0] + win_offset*cos_dir + win_offset*sin_dir;
		float start_y = kp[ii][1] - win_offset*sin_dir + win_offset*cos_dir;
		uchar* WIN = win.data;
		for(int i = 0; i < win_size; i++, start_x += sin_dir, start_y += cos_dir )
		{
			float pixel_x = start_x;
			float pixel_y = start_y;
			for(int j = 0; j < win_size; j++, pixel_x += cos_dir, pixel_y -= sin_dir )
			{
				int x = std::min(std::max(cvRound(pixel_x), 0), image.cols-1);
				int y = std::min(std::max(cvRound(pixel_y), 0), image.rows-1);
				WIN[j*win_size + i] = image.at<uchar>(y, x); // note that I transpose here...
			}
		}		

		patches.push_back(win);

	}
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

	// Keypoints to be read
	vector < vector<float> > keypoints_orig(nbkp);
	for (int ii=0; ii < keypoints_orig.size(); ++ii){
		keypoints_orig[ii] = std::vector<float>(13);
	}


	//get parameters
	int count = 0;
	while (getline(ifs_keypoints, lineread)) {
		tokens.clear();
		Tokenize(lineread, tokens);
		for (int ii=0; ii < 13; ++ii){
			keypoints_orig[count][ii] = stof(delSpaces(tokens[ii]));
		}

		count++;
	}
	ifs_keypoints.close();

	if (count != nbkp)
		throw std::runtime_error("the number of kp does not match !");

	//#--------------------------------------------------------------------------------------
	// Patch Extraction
	float scaleRatio = 7.5; // use coeff of 47.5 with the scale (hard coded in extractPatches)
    int sizePatch = -1;	// depends on the algorithm
	float sizeKp = -1;

	// std::vector< int> sizePatch_list; // size list if we need to change the patch size as well :-) FUCK SURF
	
	int interMethod = CV_INTER_CUBIC;
	bool isExact = false;

	if (strcmp(argv[1],"BRISK") == 0){
		float fRadius = std::ceil((0.85*1.0*10.8) * 2.3) + 1.0;
		sizePatch = (int)(fRadius * 2.0 + 1.0);
		sizeKp = 1.0;
	} else if (strcmp(argv[1],"FREAK") == 0){			
		float fRadius = (std::ceil((2.0/3.0 + 2.0/3.0/2.0) * 1.0 * 22.0) + 1.0) +1.0; // freak looks at one more pixel
		sizePatch = (int)(fRadius * 2.0 + 1.0);
		sizeKp = 1.0;
	} else if (strcmp(argv[1],"ORB") == 0){
		sizeKp = 1.0;	// ORB does not use scale and uses octave (which would simply be zero!
		sizePatch = (int)(31);	// This should make it work properly
	}

	// If we know exactly the code, we can simply rotation without resizing scale to have EXACT esults
	if (strcmp(argv[1],"SURF") == 0){
		isExact = true;
		// // float s = kp.size * 1.2f / 9.0f
		// // as kp.size is twice the scale, use scale * 1.2f / 9.0f * 2.0f
		// // if s == 1.0, SURF will take patch size of 21x21
		// interMethod = 0;//CV_INTER_NEAREST;
		// // float target_s = 2.;
		// // sizeKp = 9.0f / 1.2f * target_s;	// this will make SURF look at the region he is supposed to look at
		// // sizePatch = (int)(21*target_s);
		// for (int ii = 0; ii < keypoints_orig.size(); ++ii){

		// 	int curSizePatch;
		// 	const int PATCH_SZ = 20;

		// 	// if my SURF scale is 9.0/1.2 / 2.0 then I should have to crop 10.5 (i.e. SURF_s == 1)
		// 	// In this case, scale fed to this guy should be 10.5 / 7.5
		// 	// Which means that the conversion from SURF scale to this scale is
		// 	float fRescale = (10.5 / 7.5) / (9.0 / 1.2 / 2.0);

		// 	// Also since we need s from SURF size we get
		// 	float SURF_scale = 1./fRescale * keypoints_orig[ii][2];
		// 	float SURF_size = 2.0 * SURF_scale;
		// 	float SURF_s = SURF_size * 1.2 / 9.0;

		// 	// Get the stuff
		// 	curSizePatch = (int)((PATCH_SZ+1)*SURF_s);			   // this is the win_size of SURF
			
		// 	sizePatch_list.push_back(curSizePatch);
		// }
	}	

	
	// std::vector < cv::Mat > patches1;
	std::vector < cv::Mat > patches;
	// extractPatches(image, keypoints, maxscale, sizePatch, patches1);
	if (!isExact){
		extractPatches(image, keypoints_orig, sizePatch, patches, scaleRatio, interMethod);
	} else {
		extractPatchesSURF(image, keypoints_orig, patches);
	}


	//#--------------------------------------------------------------------------------------
	// Opencv Stuff

	// Initialize Opencv Descriptor
	initModule_nonfree();
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
	} else if (strcmp(argv[1],"FREAK") == 0){
		// Set upright
		myDescriptor->set("orientationNormalized",0);
	} else if (strcmp(argv[1],"SURF") == 0){
		// Set upright
		myDescriptor->set("upright",1);
	}

	// Setup the center position for kp extraction
	std::vector < cv::KeyPoint > keypoints(1); // we only need one
	keypoints[0].pt.x 		 = sizePatch/2;	   // as we put sizePatch as odd, integer division should do the job
	keypoints[0].pt.y 		 = sizePatch/2;    // as we put sizePatch as odd, integer division should do the job
	keypoints[0].size 		 = sizeKp;		   // size of one to use the smallest pattern size
	keypoints[0].angle 		 = 0;			   // angle of zero since we already rotated
	keypoints[0].response    = 0;			   // doesn't matter what you put here
	keypoints[0].octave 	 = 0;			   // octave should be zero
	
	// For each patch, loop and extract
	std::vector < cv::Mat > descriptor_list;
	for (int i = 0; i < patches.size(); ++i) {

		if (isExact){

			const int PATCH_SZ = 20;

			// From Extract patch
			float fRescale = (10.5 / 7.5) / (9.0 / 1.2 / 2.0);
			float SURF_scale = 1./fRescale * keypoints_orig[i][2];
			float SURF_size = 2.0 * SURF_scale;
			float s = SURF_size * 1.2 / 9.0;
		
			int win_size = (int)((PATCH_SZ+1)*s);
			float win_offset = -(float)(win_size-1)/2;
			
			// Assign the original keypoiont information			
			keypoints[0].pt.x = -win_offset;
			keypoints[0].pt.y = -win_offset;
			
			keypoints[0].size = SURF_size;
			
		}		

		cv::Mat descriptor;
		myDescriptor->compute(patches[i],keypoints,descriptor);		

		descriptor_list.push_back(descriptor);
	}
	

	// No need to resave Kp as we are manually doing the patch extraction
	// const bool resaveKp = true;
	// if (resaveKp)
	// {
	// 	ofstream ofs_keypoints;
	// 	//ofstream ofs_score;
	// 	ofs_keypoints.open(argv[3], std::ofstream::trunc);
	// 	//ofs_score.open(score_name, std::ofstream::trunc);
	// 	// Save keypoints
	// 	ofs_keypoints << 6 << endl;
	// 	ofs_keypoints << keypoints.size() << endl;
	// 	for(int i=0; i < keypoints.size(); ++i){
	// 		ofs_keypoints << std::setprecision(10) << keypoints[i].pt.x << " ";
	// 		ofs_keypoints << std::setprecision(10) << keypoints[i].pt.y << " ";
	// 		ofs_keypoints << std::setprecision(10) << keypoints[i].size*0.5 << " ";// 0.5 as it is diameter
	// 		ofs_keypoints << std::setprecision(10) << keypoints[i].angle << " ";
	// 		//cout<<"octave number: "<< int(keypoints[i].octave) <<endl;
	// 		//ofs_keypoints << endl;
	// 		ofs_keypoints << std::setprecision(10) << keypoints[i].response << " ";
	// 		ofs_keypoints << int(keypoints[i].octave) 					 << endl;
	// 	}
	// 	ofs_keypoints.close();
	// }
	

    // char buf[100];
    // sprintf(buf,argv[4]);
    FILE* ofp = fopen(argv[4],"w");
    dumpMat(descriptor_list, ofp);
    fclose(ofp);
// 	cv::FileStorage fsWrite(argv[4], cv::FileStorage::WRITE );
// 	fsWrite<<"descriptors" << descriptors;
// 	fsWrite.release();

}




