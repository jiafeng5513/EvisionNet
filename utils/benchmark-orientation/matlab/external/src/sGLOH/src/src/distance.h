/* 	Precision/recall computation for sGLOH (but also SIFT).
 *  Code by Fabio Bellavia (fbellavia@unipa.it),
 *  refer to: F. Bellavia, D. Tegolo, C. Valenti,
 *  "Keypoint descriptor matching with context-based orientation
 *  estimation", Image and Vision Computing 2014, and
 *  F. Bellavia, D. Tegolo, E. Trucco, "Improving SIFT-based Descriptors
 *  Stability to Rotations", ICPR 2010.
 *  Only for academic or other non-commercial purposes.
 *  Code partially adapted from K. Mikolajczyk:
 *  http://www.robots.ox.ac.uk/~vgg/research/affine/
 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <fftw3.h>
#include <sys/stat.h>

#ifdef _WIN32
#include <time.h>
#else
#include <sys/times.h>
#endif

#include <libconfig.h>
#include <unistd.h>

#define ADD_ONE 0

double my_inf;
#define INFINITY_ my_inf

#define SILENT 	0
#define VERBOSE 1

#define UPPERLOWER 1
#define NONE 	   0

#define NO_PAD 	0
#define PAD 	1

#define IMG_ROW 1024
#define IMG_COL 768

#define MEAN 	  0
#define CUBE_ROOT 1
#define LUMINANCE 2

#ifdef _WIN32
#define SLASH '\\'
#else
#define SLASH '/'
#endif

#define R(s,o) ((ceil(3*(s))>1?ceil(3*(s)):1)+(o))
#define R2D(r) (2*(r)+1)
#define D2R(d) (((d)-1)/2)

#define PATCH_SIZE	47
#define PATCH_TH_NO	 4

#define BI_RGB  0  // No compression - straight BGR data
#define BI_RLE8 1  // 8-bit run-length compression
#define BI_RLE4 2  // 4-bit run-length compression

#define NO_EDGE  		0
#define	EDGE			1
#define LEFT 			2
#define EDGE_LEFT		3
#define RIGHT 			4
#define EDGE_RIGHT		5
#define LEFT_RIGHT		6
#define LEFT_RIGHT_EDGE	7

#define START 		0
#define SPECIAL 	1
#define RLE 		2
#define OFFSET		3
#define ABS			4
#define PAD0_FIRST 	5
#define PAD0_LAST  	6
#define END			7

#define BUFFER_SIZE 1024

#define BLUE         0xFF
#define GREEN      0xFF00
#define RED	     0xFF0000
#define PURPLE   0xFF00FF
#define YELLOW   0xFFFF00

#define YES 1
#define NO  0

#define TRUE 1
#define FALSE 0

#define L1 				0
#define L2 				1
#define CHI_SQUARED 	2
#define BATACHARYYA 	3
#define KULLBACK_LEIBER 4
#define JENSEN_SHANNON	5

#define OPT_DATA_IM1 "im1.bmp"
#define OPT_DATA_IM2 "im2.bmp"
#define OPT_DATA_DATA1 "data1.txt"
#define OPT_DATA_DATA2 "data2.txt"
#define OPT_DATA_H "h.txt"
#define OPT_DATA_DRAW FALSE
#define OPT_INFO_OVERLAP 50
#define OPT_INFO_DISTANCE "L1"
#define OPT_INFO_D_SAVE FALSE
#define OPT_INFO_D_SAVE_MATRIX "ms"
#define OPT_INFO_D_LOAD FALSE
#define OPT_INFO_D_LOAD_MATRIX "ml"
#define OPT_INFO_R_SAVE FALSE
#define OPT_INFO_R_SAVE_MATRIX "rs"
#define OPT_INFO_R_LOAD FALSE
#define OPT_INFO_R_LOAD_MATRIX "rl"

#define OPT_INFO_NN_ELN 19
double opt_info_nn_el[]={0.01,0.0225,0.04,0.0625,0.09,0.1225,0.16,0.2025,
					  0.25,0.3025,0.36,0.4225,0.49,0.5625,0.64,0.7225,
					  0.8100,0.9025,1.0};
#define OPT_INFO_NN_EL opt_info_nn_el
#define OPT_INFO_NNR_ELN 18;
double opt_info_nnr_el[]={6.0,5.0,4.5,4.0,3.8,3.5,3,2.8,2.6,
					      2.4,2.2,2.0,1.8,1.6,1.4,1.2,1.1,1};
#define OPT_INFO_NNR_EL opt_info_nnr_el
#define OPT_DESC_SHIFT TRUE
#define OPT_DESC_DIR 8
#define OPT_DESC_RAD 2
#define OPT_DESC_UNIQUE_CENTER_BIN FALSE
#define OPT_DESC_HIST 8

#define INIT_DIST_VECT {						\
	opt.info.dist_vect[0]=&dist_L1;				\
	opt.info.dist_vect[1]=&dist_L2;				\
	opt.info.dist_vect[2]=&dist_chisq;			\
	opt.info.dist_vect[3]=&dist_batacharyya;	\
	opt.info.dist_vect[4]=&dist_KL;				\
	opt.info.dist_vect[5]=&dist_JS;				\
}

#define EXIT_ERROR(err_file,err_line,err_string) {	  			\
			fprintf(stderr,"Error at file %s line %d\n%s\n",	\
			err_file,err_line,err_string);    		  			\
			exit(EXIT_FAILURE);									\
}

#define ALL 0
#define NN  1

#define NO_DRAW_MATCH 0
#define DRAW_MATCH	  1

char *dists[]={"L1","L2","CHI","B","KL","JS","MIN_NEAR_DIFF"};

#define DISTS 6

char *usage="\nBad parameters format!\n*** Usage ***"			\
		    "\ndistance [<bmp image 1> <bmp image 2>"		 	\
		    "\n         [<desc data 1> <desc data 2>"			\
		    "\n         [<homography file>]]]";

#ifndef _WIN32
	extern FILE *stderr, *stdin, *stdout;
#endif

typedef struct dvect {
	int n;
	double *el;
} dvect;

typedef struct ivect {
	int n;
	int *el;
} ivect;

typedef struct data_struct {
	char *im1;
	char *im2;
	char *data1;
	char *data2;
	char *h;
	int draw;
} data_struct;

typedef struct desc_struct {
	int shift;
	int dir;
	int rad;
	int hist;
	int unique_center_bin;
	int *check_dir;
} desc_struct;

typedef struct info_struct {
	int overlap;
	char *distance;
	char *dload_matrix;
	int dload;
	char *dsave_matrix;
	int dsave;
	char *rload_matrix;
	int rload;
	char *rsave_matrix;
	int rsave;
	double (*dist_vect[DISTS])(dvect, dvect);
	int dist_n;
	double *nn_el;
	double *nnr_el;
	int nn_eln;
	int nnr_eln;
} info_struct;

typedef struct prg_opt {
	data_struct data;
	info_struct info;
	desc_struct desc;
} prg_opt;

typedef struct BITMAPFILEHEADER {
	unsigned short int bfType;
	unsigned int bfSize;
	unsigned short int bfReserved1;
	unsigned short int bfReserved2;
	unsigned int bfOffBits;
} BITMAPFILEHEADER;

typedef struct BITMAPINFOHEADER {
	unsigned int biSize;
	unsigned int biWidth;
    unsigned int biHeight;
	unsigned short int biPlanes;
	unsigned short int biBitCount;
	unsigned int biCompression;
	unsigned int biSizeImage;
	unsigned int biXPelsPerMeter;
	unsigned int biYPelsPerMeter;
	unsigned int biClrUsed;
	unsigned int biClrImportant;
} BITMAPINFOHEADER;

typedef struct RGBQUAD {
	unsigned char rgbBlue;
	unsigned char rgbGreen;
	unsigned char rgbRed;
	unsigned char gbReserved;
} RGBQUAD;

typedef struct bmp_img {
	double *val;
	int row;
	int col;
} bmp_img;

typedef struct mk_pt {
	double p[5]; // x, y and ellipse coefficents
} mk_pt;

typedef struct rgb_img {
	unsigned int *val;
	int row;
	int col;
} rgb_img;

typedef struct fts {
	int n;
	int d;
	double *f;
} fts;

typedef struct aff_pt {
	double x;
	double y;
	double si;
	double U[4];
} aff_pt;

typedef struct pt {
	double x;
	double y;
} pt;

typedef struct pt2 {
	int prec;
	int next;
} pt2;

typedef struct xy {
	int x;
	int y;
} xy;

typedef struct svect {
	int n;
	pt2 *el;
} svect;

typedef struct pt3 {
	int x;
	int y;
	double val;
	int index;
} pt3;

#ifdef _WIN32

typedef struct tictac {
	clock_t t;
} tictac;

#else

typedef struct tictac {
	clock_t t;
	struct tms c;
} tictac;

#endif

typedef struct stime {
	double realt;
	double usert;
	double syst;
} stime;

typedef struct matched_vect {
	int n;
	int c;
	int m;
	pt3 *el;
} matched_vect;

typedef struct matched_struct {
	matched_vect nn;
	matched_vect nnr;
} matched_struct;

typedef struct rpt_struct {
	matched_vect good_table;
	double overlap;
	int corr;
} rpt_struct;

typedef struct matched_pt {
	int n;
	int *d1;
	int *d2;
	double *val;
} matched_pt;

typedef struct finfo {
	FILE *in;
	int d;
	int n;
	dvect desc;
} finfo;

typedef struct dist_matrix {
	int r;
	int c;
	float *el;
} dist_matrix;

typedef struct dmatrix {
	int r;
	int c;
	double *el;
} dmatrix;

typedef struct patch_stat {
	int p; // current patch
	int c; // current patch page
	int n; // patches pages
	int pp; // patches per page
	int pr; // patches per row
	int pc; // patches per column
	char *cs; // current patch string
	int ds; // digits for cs
	char *s; // prefix string
	char *s_; // suffix string
	rgb_img im; // page
	int m; // features per patch
} patch_stat;

////////////////////////////////////////////////////////////////////////

double dist_JS(dvect a, dvect b);
double dist_KL(dvect a, dvect b);
double dist_batacharyya(dvect a, dvect b);
double dist_chisq(dvect a, dvect b);
double dist_L2(dvect a, dvect b);
double dist_L1(dvect a, dvect b);

fts mk_load(char *filename, int s);
int h_load(char *filename, double *h);

int get_affine(double x, double y, double *h, double *res);
int diagonalize2x2(double *m, double *D, double *R);
int inv2x2(double *m,double *res);
int prod2x2(double *m1, double *m2, double *res);
int transpose2x2(double *m,double *res);

double trunc(double x);
double round(double x);
double ceil(double x);
double log2(double x);
double floor(double x);

int best_swap(dmatrix m, ivect r_swap, ivect c_swap, dvect b, xy p);
dvect gauss_elimination(dmatrix m, dvect b);
int inv(double *h, double *r, int n);

finfo desc_init(char *filename);
dvect *desc_all(finfo data);
int destroy_desc_all(dvect *d, int n);
int desc_end(finfo data);

dist_matrix compute_dist(prg_opt opt);
dist_matrix compute_dist_guess(prg_opt opt);
int dist_shift(dvect in, dvect out, int step, desc_struct d);
matched_struct matching(dist_matrix md);

matched_vect matched_vect_init(int m);
matched_vect matched_vect_insert(matched_vect v, pt3 e);
matched_vect matched_vect_resize(matched_vect v);
int matched_vect_end(matched_vect v);

svect svect_init(int n);
svect svect_cancel(svect r, int k);

rpt_struct repeatability(prg_opt opt);
matched_vect rpt_table(fts ft1, fts ft2, double overlap, char *rpt_load,
	char *rpt_save);
fts project_region(fts feat, double *h);

int sort_el(pt3 *a, pt3 *b);
int matched_vect_qsort(matched_vect in, int el);
int rpt_bin_search(rpt_struct rpt, pt3 p);

int get_info(prg_opt opt);
int visual_match(prg_opt opt, matched_struct match, dist_matrix m,
				 rpt_struct rpt);

rgb_img load_bmp(char filename[]);
int save_bmp(char *filename, rgb_img im);
int gray2rgb(bmp_img im, rgb_img res, xy p, int t, unsigned int c);
bmp_img rgb2gray(rgb_img im, int mode);

bmp_img conv2(bmp_img A, bmp_img B, int pad);
bmp_img patch_trans(bmp_img im, pt l, double U[]);
bmp_img get_patch(int s, bmp_img im, aff_pt apt);
bmp_img make_disk(int r, int *a);
int patch_norm(bmp_img im);
bmp_img gauss_ker2_cov(double s, double U[]);

stime elapsed_time(tictac start, tictac stop, int mode);
tictac gimme_clock(void);

char *append(char *prefix, char *string, char *suffix);
int multiplestrcmp(char *string,char *strings[],int nstrings,int mode);
char *get_basename(char *s);
char *get_filename(char *s);
prg_opt optsdigest(char *argv[], int argc);
int config_lookup_my_float(const config_t *config, const char *path,double *value);
int config_lookup_my_int(const config_t *config, const char *path, int *value);
char *strdup(const char *s);

patch_stat save_patch_init(int n, int m, char pr[], char suff[]);
patch_stat save_patch_next(patch_stat s, bmp_img p, int v,
						   unsigned int c);
int save_patch_end(patch_stat s);

int sort_index(pt3 *a, pt3 *b);
int matched_vect_index_qsort(matched_vect in, int el);

int sort_val(pt3 *a, pt3 *b);
int matched_vect_val_qsort(matched_vect in, int el);

int main(int argc, char *argv[]);

int i_bin_search(ivect r, int p);
int sort_i(int *a, int *b);
int i_qsort(ivect in, int el);

int bin_search(matched_vect r, pt3 p);

dist_matrix load_dist_matrix(char *filename);
int save_dist_matrix(dist_matrix r, char *filename);
