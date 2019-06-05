#ifndef SGLOF
#define SGLOF
/* 	sGLOH descriptor by Fabio Bellavia (fbellavia@unipa.it),
 *  refer to: F. Bellavia, D. Tegolo, C. Valenti,
 *  "Keypoint descriptor matching with context-based orientation
 *  estimation", Image and Vision Computing 2014, and
 *  F. Bellavia, D. Tegolo, E. Trucco, "Improving SIFT-based Descriptors
 *  Stability to Rotations", ICPR 2010.
 *  Only for academic or other non-commercial purposes.
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

#include <unistd.h>
#include <libconfig.h>  //library is libconfig

#include <string.h>

#define TRUE 1
#define FALSE 0

//default options
#define DATA_IMG "im.bmp"
#define DATA_IN "feat.txt"
#define DATA_OUT "desc.txt"
#define DATA_DRAW FALSE
#define DATA_QUANTIZER 512
#define DESC_DIR 8
#define DESC_RADN 2
static int def_desc_rad[]={12,20};
#define DESC_RAD def_desc_rad
#define DESC_SCALE_FACTOR 3.0
#define DESC_UNIQUE_CENTER_BIN FALSE
#define DESC_HIST_DIR_CF 1.0
#define DIR_HIST_STD 0.7
#define DIR_WEIGHT_WIN FALSE
#define DIR_SMOOTH_STD 1.0
#define ORT_ADJUST FALSE
#define ORT_PREC 8
#define ORT_DIR 8
#define ORT_HIST_STD 0.7
#define ORT_WEIGHT_WIN FALSE
#define ORT_SMOOTH_STD 1.0

static char *rosy_usage="\nBad parameters format!\n*** Usage ***"		\
		  	     "\nsgloh [<bmp image> [<input file> [<output file>]]]";

#define EXIT_ERROR(err_file,err_line,err_string) {	  			\
			fprintf(stderr,"Error at file %s line %d\n%s\n",	\
			err_file,err_line,err_string);    		  			\
			exit(EXIT_FAILURE);   					  			\
}

#ifdef _WIN32
#define SLASH '\\'
#else
#define SLASH '/'
#endif

#define SILENT 	0
#define VERBOSE 1

#define R(s,o) ((ceil(3*(s))>1?ceil(3*(s)):1)+(o))
#define R2D(r) (2*(r)+1)
#define D2R(d) (((d)-1)/2)

#define UPPERLOWER 1
#define NONE 	   0

#define NO_PAD 0
#define PAD 1

#define NO_CENTER 0
#define CENTER 1

#define IMG_ROW 1024
#define IMG_COL 768

#define BUFFER_SIZE 1024

#define BLUE         0xFF
#define GREEN      0xFF00
#define RED	     0xFF0000
#define PURPLE   0xFF00FF
#define YELLOW   0xFFFF00

#define MEAN 	  0
#define CUBE_ROOT 1
#define LUMINANCE 2

#define BI_RGB  0  // No compression - straight BGR data
#define BI_RLE8 1  // 8-bit run-length compression
#define BI_RLE4 2  // 4-bit run-length compression

#define START 		0
#define SPECIAL 	1
#define RLE 		2
#define OFFSET		3
#define ABS			4
#define PAD0_FIRST 	5
#define PAD0_LAST  	6
#define END			7

#ifndef _WIN32
	extern FILE *stderr, *stdin, *stdout;
#endif

#define IN_LENGTH 8
#define OUT_LENGTH 9

static char in_data[]="_in.txt";
static char out_data[]="_out.txt";

#define DEFAULT_OPTS {					\
	opt.data.img=strdup(DATA_IMG);		\
	opt.data.in=strdup(DATA_IN);\
	opt.data.out=strdup(DATA_OUT);\
	opt.data.draw=DATA_DRAW;\
	opt.data.quantizer=DATA_QUANTIZER;\
	opt.desc.dir=DESC_DIR;\
	opt.desc.radn=DESC_RADN;\
	opt.desc.rad=DESC_RAD;\
	opt.desc.scale_factor=DESC_SCALE_FACTOR;\
	opt.desc.unique_center_bin=DESC_UNIQUE_CENTER_BIN;\
	opt.desc.hist_dir_cf=DESC_HIST_DIR_CF;\
	opt.dir.hist_std=DIR_HIST_STD;\
	opt.dir.weight_win=DIR_WEIGHT_WIN;\
	opt.dir.smooth_std=DIR_SMOOTH_STD;\
	opt.ort.adjust=ORT_ADJUST;\
	opt.ort.prec=ORT_PREC;\
	opt.ort.dir=ORT_DIR;\
	opt.ort.hist_std=ORT_HIST_STD;\
	opt.ort.weight_win=ORT_WEIGHT_WIN;\
	opt.ort.smooth_std=ORT_SMOOTH_STD;\
}

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

typedef struct dir_struct {
	double hist_std;
	int weight_win;
	double smooth_std;
} dir_struct;

typedef struct data_struct {
	char *img;
	char *in;
	char *out;
	int draw;
	int quantizer;
} data_struct;

typedef struct ort_struct {
	int adjust;
	int prec;
	int dir;
	double hist_std;
	int weight_win;
	double smooth_std;
} ort_struct;

typedef struct desc_struct {
	int dir;
	int radn;
	int *rad;
	double scale_factor;
	int unique_center_bin;
	int hist_dir_cf;
} desc_struct;

typedef struct mk_pt {
	double p[5]; // x, y and ellipse coefficents
} mk_pt;

typedef struct bmp_img {
	double *val;
	int row;
	int col;
} bmp_img;

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
	bmp_img im; // page
	bmp_img disk; // patch circular mask
} patch_stat;

typedef struct pt {
	double x;
	double y;
} pt;

typedef struct aff_pt {
	double x;
	double y;
	double si;
	double U[4];
} aff_pt;

typedef struct rgb_img {
	unsigned int *val;
	int row;
	int col;
} rgb_img;

typedef struct h_pt {
	int n;
	aff_pt *aff;
} h_pt;

typedef struct dvect {
	int n;
	double *el;
} dvect;

typedef struct i_table {
	int row;
	int col;
	int *val;
	int *val_;
} i_table;

typedef struct opt_struct {
	data_struct data;
	desc_struct desc;
	dir_struct dir;
	ort_struct ort;
} opt_struct;

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

/* typedef struct stime { */
/* 	double realt; */
/* 	double usert; */
/* 	double syst; */
/* } stime; */

////////////////////////////////////////////////////////////////////////

bmp_img gauss_ker2(double s);
bmp_img gauss_ker2_cov(double s, double U[]);
bmp_img conv2(bmp_img A, bmp_img B, int pad);
int rosy_derivative(bmp_img img, bmp_img *dx, bmp_img *dy, int c);
bmp_img rosy_make_disk(int r, int *a);
bmp_img rosy_make_disk_(int r, double r_);

i_table rosy_init(opt_struct opt);
int rosy_go(opt_struct opt, bmp_img im, h_pt ft, i_table bin);
int rosy_end(opt_struct opt, bmp_img im, h_pt ft, i_table bin);
FILE *rosy_mk_save_init(opt_struct opt, int m, int n);
int rosy_mk_save_next(dvect f, FILE *file_idx, int quantizer);
int rosy_mk_save_end(dvect f, FILE *file_idx);
h_pt rosy_mk_load(char *filename);

bmp_img rosy_patch(int s, double scale_factor, ort_struct ort, bmp_img im, aff_pt apt);
int rosy_desc_el(bmp_img patch, i_table mask, dir_struct opt, int hist_dir, int hist_dir_cf, dvect q);
mk_pt rosy_to_mk_pt(aff_pt in);
aff_pt rosy_from_mk_pt(mk_pt in, double angle_d);

bmp_img rosy_trans(bmp_img im, pt l, double U[]);
double rosy_angle(bmp_img im, ort_struct opt);
bmp_img rosy_norm(bmp_img im);

int prod2x2(double m1[], double m2[], double res[]);
int diagonalize2x2(double *m, double *D, double *R);
int inv2x2(double *m,double *res);
double trunc(double x);
double round(double x);
double ceil(double x);
double floor(double x);
double atan2(double y, double x);

patch_stat rosy_save_patch_init(int r, int n, char pr[]);
patch_stat rosy_save_patch_next(patch_stat s, bmp_img p);
int rosy_save_patch_end(patch_stat s);

int rosy_norm01_save(bmp_img im, char *a, char *b, char *c);
bmp_img rosy_norm01(bmp_img im);
rgb_img rosy_gray2rgb(bmp_img im);
int rosy_save_bmp(char *filename, rgb_img im);
rgb_img rosy_load_bmp(char filename[]);
bmp_img rosy_rgb2gray(rgb_img im, int mode);

char *append(char *a, char *b, char *c);
//char *strdup(const char *s);
opt_struct rosy_opts_digest(int argc, char *argv[]);
char *rosy_basename(char *s);
int config_lookup_my_float(const config_t *config, const char *path,double *value);
int config_lookup_my_int(const config_t *config, const char *path, int *value);

/* stime elapsed_time(tictac start, tictac stop, int mode); */
tictac gimme_clock(void);

//int main(int argc, char *argv[]);
#endif

