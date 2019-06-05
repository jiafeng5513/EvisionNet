#include <mex.h>
#include <math.h>


double computeDistance(const double * feat1,const double * feat2,const int f1,const int f2, const int size);


void mexFunction(int nlhs,       mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
	// must have single double array input.
	if (nrhs != 3) { printf("in nrhs != 3\n"); return; }
	if (mxGetClassID(prhs[0]) != mxDOUBLE_CLASS) { printf("input must be a double array\n"); return; }
	if (mxGetClassID(prhs[1]) != mxDOUBLE_CLASS) { printf("input must be a double array\n"); return; }
  
	// must have single output.
	if (nlhs != 5) { printf("out nlhs != 5\n"); return; }
  
	// get size of x.
	int const num_dims1 = mxGetNumberOfDimensions(prhs[0]);
	int const *dims1 = mxGetDimensions(prhs[0]);
	int const num_dims2 = mxGetNumberOfDimensions(prhs[1]);
	int const *dims2 = mxGetDimensions(prhs[1]);
	int const num_dims3 = mxGetNumberOfDimensions(prhs[2]);
	int const *dims3 = mxGetDimensions(prhs[2]);
	if(dims1[0]<9 || dims2[0]<9 || dims1[0]!=dims2[0] || dims3[0]!=1){ printf("dims != 9\n"); return; }
  
	// create new array of the same size as x.

	int *odim = new int[2];
	odim[0]=dims2[1];
	odim[1]=dims1[1];
	plhs[0] = mxCreateNumericArray(2, odim, mxDOUBLE_CLASS, mxREAL);
	plhs[1] = mxCreateNumericArray(2, odim, mxDOUBLE_CLASS, mxREAL);
	plhs[2] = mxCreateNumericArray(2, odim, mxDOUBLE_CLASS, mxREAL);
	plhs[3] = mxCreateNumericArray(2, odim, mxDOUBLE_CLASS, mxREAL);
	plhs[4] = mxCreateNumericArray(2, odim, mxDOUBLE_CLASS, mxREAL);
  

	// get pointers to beginning of x and y.
	double const *feat1 = (double *)mxGetData(prhs[0]);
	double const *feat2 = (double *)mxGetData(prhs[1]);
	double const *flag = (double *)mxGetData(prhs[2]);
	double       *over_out = (double *)mxGetData(plhs[0]);
	double       *mover_out = (double *)mxGetData(plhs[1]);
	double       *desc_out = (double *)mxGetData(plhs[2]);
	double       *mdesc_out = (double *)mxGetData(plhs[3]);
	double       *sdesc_out = (double *)mxGetData(plhs[4]);


	float *feat1a = new float[9];
	float *feat2a = new float[9];
	float *tdesc_out = new float[dims2[1]*dims1[1]];
	float *tover_out = new float[dims2[1]*dims1[1]];

	int common_part=(int)flag[0];

	//init the returned value with max results
	for(int j=0;j<dims1[1];j++){    
		for (int i=0; i<dims2[1]; i++){
			over_out[j*dims2[1]+i]=100.0;
			desc_out[j*dims2[1]+i]=1000000.0;
			sdesc_out[j*dims2[1]+i]=0;
		}
	} 

	// printf("%f %f\n",flag[0],flag[1]);
	// total number of elements in arrays.
	/*int total = 1;
	  for (int i=0; i<num_dims1; ++i){
	  total *= dims1[i];  
	  printf("feat1 %d  %d  \n",num_dims1, dims1[i]);
	  }
	*/

	float max_dist,fac,dist,dx,dy,bna,bua,descd,ov;
	for(int j=0,f1=0;j<dims1[1];j++,f1+=dims1[0]){    
		max_dist=sqrt(feat1[f1+5]*feat1[f1+6]);
		if(common_part)fac=30.0/max_dist;
		else fac=3;
		max_dist=max_dist*4;
		fac=1.0/(fac*fac);
		feat1a[2]=fac*feat1[f1+2];
		feat1a[3]=fac*feat1[f1+3];
		feat1a[4]=fac*feat1[f1+4];
		feat1a[7] = sqrt(feat1a[4]/(feat1a[2]*feat1a[4] - feat1a[3]*feat1a[3]));
		feat1a[8] = sqrt(feat1a[2]/(feat1a[2]*feat1a[4] - feat1a[3]*feat1a[3]));
		for (int i=0,f2=0; i<dims2[1]; i++,f2+=dims1[0]){
			//compute shift error between ellipses
			dx=feat2[f2]-feat1[f1];
			dy=feat2[f2+1]-feat1[f1+1];
			dist=sqrt(dx*dx+dy*dy);
			if(dist<max_dist){
				feat2a[2]=fac*feat2[f2+2];
				feat2a[3]=fac*feat2[f2+3];
				feat2a[4]=fac*feat2[f2+4];
				feat2a[7] = sqrt(feat2a[4]/(feat2a[2]*feat2a[4] - feat2a[3]*feat2a[3]));
				feat2a[8] = sqrt(feat2a[2]/(feat2a[2]*feat2a[4] - feat2a[3]*feat2a[3]));
				//find the largest eigenvalue
				float maxx=ceil((feat1a[7]>(dx+feat2a[7]))?feat1a[7]:(dx+feat2a[7]));
				float minx=floor((-feat1a[7]<(dx-feat2a[7]))?(-feat1a[7]):(dx-feat2a[7]));
				float maxy=ceil((feat1a[8]>(dy+feat2a[8]))?feat1a[8]:(dy+feat2a[8]));
				float miny=floor((-feat1a[8]<(dy-feat2a[8]))?(-feat1a[8]):(dy-feat2a[8]));

				float mina=(maxx-minx)<(maxy-miny)?(maxx-minx):(maxy-miny);
				float dr=mina/50.0;
				bua=0;bna=0;int t1=0,t2=0;
				//compute the area
				for(float rx=minx;rx<=maxx;rx+=dr){
					float rx2=rx-dx;t1++;
					for(float ry=miny;ry<=maxy;ry+=dr){
						float ry2=ry-dy;
						//compute the distance from the ellipse center
						float a=feat1a[2]*rx*rx+2*feat1a[3]*rx*ry+feat1a[4]*ry*ry;
						float b=feat2a[2]*rx2*rx2+2*feat2a[3]*rx2*ry2+feat2a[4]*ry2*ry2;
						//compute the area
						if(a<1 && b<1)bna++;
						if(a<1 || b<1)bua++;
					}
				}
				ov=100.0*(1-bna/bua);
				tover_out[j*dims2[1]+i]=ov;
				mover_out[j*dims2[1]+i]=ov;
				//printf("overlap %f  \n",over_out[j*dims2[1]+i]);return;
			}else {
				tover_out[j*dims2[1]+i]=100.0;//100 is the overlaps error
				mover_out[j*dims2[1]+i]=100.0;
			}
			descd = computeDistance(feat1,feat2,f1,f2,dims1[0]);
			tdesc_out[j*dims2[1]+i]=descd;  
			mdesc_out[j*dims2[1]+i]=descd;  
		}
	}

	// Re-loop over j to replace descriptor distances with the minimum for kp repeated along i
	for(int j=0,f1=0;j<dims1[1];j++,f1+=dims1[0]){
		int i_start = 0;
		int f2_start =0;
		int i_end = 0;
		int f2_end = 0;
		float min_descd = tdesc_out[j*dims2[1]+i_end];

		while (i_end < dims2[1]-1){
			// increment end
			i_end++;
			f2_end += dims1[0];
			// get current descd at end
			float end_descd = tdesc_out[j*dims2[1]+i_end];
			// check if start and end are same kp
			bool isSame = true;
			for (int k=0; k < 5; ++k){
				isSame &= (feat2[f2_end+k] == feat2[f2_start+k]);
			}
			// if they are not the same
			if (false == isSame){
				// discard all other matching results except for the start
				tdesc_out[j*dims2[1]+i_start] = min_descd;
				mdesc_out[j*dims2[1]+i_start] = min_descd;					
				for (int ii=i_start+1; ii < i_end; ++ii){
					tdesc_out[j*dims2[1]+ii] = 1000000.0;
					mdesc_out[j*dims2[1]+ii] = 1000000.0;					
					tover_out[j*dims2[1]+ii] = 100.0;
					mover_out[j*dims2[1]+ii] = 100.0;
				}
				// update start marking new start using end
				i_start = i_end;
				f2_start = f2_end;
				// reset min_descd
				min_descd = end_descd;
			// if they are the same
			} else {
				// update min_descd with new min
				if (min_descd > end_descd){
					min_descd = end_descd;
				}				
			}
		}
		// one last cleanup
		tdesc_out[j*dims2[1]+i_start] = min_descd;
		mdesc_out[j*dims2[1]+i_start] = min_descd;					
		for (int ii=i_start+1; ii <= i_end; ++ii){
			tdesc_out[j*dims2[1]+ii] = 1000000.0;
			mdesc_out[j*dims2[1]+ii] = 1000000.0;					
			tover_out[j*dims2[1]+ii] = 100.0;
			mover_out[j*dims2[1]+ii] = 100.0;
		}
	}
		
	// Re-loop over i to replace descriptor distances with the minimum for kp repeated along j
	for(int i=0,f2=0;i<dims2[1];i++,f2+=dims2[0]){
		int j_start = 0;
		int f1_start =0;
		int j_end = 0;
		int f1_end = 0;
		float min_descd = tdesc_out[j_end*dims2[1]+i];

		while (j_end < dims1[1]-1){
			// increment end
			j_end++;
			f1_end += dims1[0];
			// get current descd at end
			float end_descd = tdesc_out[j_end*dims2[1]+i];
			// check if start and end are same kp
			bool isSame = true;
			for (int k=0; k < 5; ++k){
				isSame &= (feat1[f1_end+k] == feat1[f1_start+k]);
			}
			// if they are not the same
			if (false == isSame){
				// discard all other matching results except for the start
				tdesc_out[j_start*dims2[1]+i] = min_descd;
				mdesc_out[j_start*dims2[1]+i] = min_descd;					
				for (int jj=j_start+1; jj < j_end; ++jj){
					tdesc_out[jj*dims2[1]+i] = 1000000.0;
					mdesc_out[jj*dims2[1]+i] = 1000000.0;					
					tover_out[jj*dims2[1]+i] = 100.0;
					mover_out[jj*dims2[1]+i] = 100.0;
				}
				// update start marking new start using end
				j_start = j_end;
				f1_start = f1_end;
				// reset min_descd
				min_descd = end_descd;
			// if they are the same
			} else {
				// update min_descd with new min
				if (min_descd > end_descd){
					min_descd = end_descd;
				}				
			}
		}
		// one last cleanup
		tdesc_out[j_start*dims2[1]+i] = min_descd;
		mdesc_out[j_start*dims2[1]+i] = min_descd;					
		for (int jj=j_start+1; jj <= j_end; ++jj){
			tdesc_out[jj*dims2[1]+i] = 1000000.0;
			mdesc_out[jj*dims2[1]+i] = 1000000.0;					
			tover_out[jj*dims2[1]+i] = 100.0;
			mover_out[jj*dims2[1]+i] = 100.0;
		}
	}
  
	float minr=100,ratio,minratio;
	int mini=0;
	int minj=0;
	int dnbr=0;

	//kp region overlaps
	do{
		minr=100;
		for(int j=0;j<dims1[1];j++){    
			for (int i=0; i<dims2[1]; i++){
				if(minr>tover_out[j*dims2[1]+i]){
					minr=tover_out[j*dims2[1]+i];
					mini=i;
					minj=j;
				}
			}
		}
		if(minr<100){
			for(int j=0;j<dims1[1];j++){
				tover_out[j*dims2[1]+mini]=100;
			}   
			for (int i=0; i<dims2[1]; i++){
				tover_out[minj*dims2[1]+i]=100;
			}
			// // block all other repeated kps
			// for(int j=min_i;j<dims1[1];j++){
			// 	for (int i=min_j; i<dims2[1]; i++){
			// 		// check if current is new
			// 		bool isSame = true;
			// 		for (int k=0; k < 5; ++k){
			// 			isSame &= (feat2[i*dims1[0]+k] == feat2[min_i*dims1[0]+k]);
			// 		}
			// 		// if new, then break
			// 		if (false == isSame){
			// 			break;
			// 		}
					
			// 		tover_out[j*dims2[1]+i]=100; // block
			// 	}
			// 	// check if current is new
			// 	bool isSame = true;
			// 	for (int k=0; k < 5; ++k){
			// 		isSame &= (feat1[j*dims1[0]+k] == feat1[min_j*dims1[0]+k]);
			// 	}
			// 	// if new, then break
			// 	if (false == isSame){
			// 		break;
			// 	}
			// }
			over_out[minj*dims2[1]+mini]=minr;
		}
	}while(minr<70);
  
	//descriptor  
	do{
		minr=1000000;
		for(int j=0;j<dims1[1];j++){    
			for (int i=0; i<dims2[1]; i++){
				if(minr>tdesc_out[j*dims2[1]+i]){
					minr=tdesc_out[j*dims2[1]+i];
					mini=i;
					minj=j;
				}
			}
		}
		
		if(minr<1000000){
		
            for(int j=0;j<dims1[1];j++){
				tdesc_out[j*dims2[1]+mini]=1000000;
			}   
			for (int i=0; i<dims2[1]; i++){
				tdesc_out[minj*dims2[1]+i]=1000000;
			}
			
			desc_out[minj*dims2[1]+mini]=minr;//dnbr++;	//minr
			minratio=1000000.0;
			
			for (int i=0; i<dims2[1]; i++){
				ratio=mdesc_out[minj*dims2[1]+i]/mdesc_out[minj*dims2[1]+mini];
				if(ratio<minratio && i!=mini && ratio>1)
					minratio=ratio;	
			}
			sdesc_out[minj*dims2[1]+mini]=minratio;
		}
		
	}while(minr<1000000);
  


	delete []odim;
	delete []tdesc_out;
	delete []tover_out;
	delete []feat1a;
	delete []feat2a;
}
