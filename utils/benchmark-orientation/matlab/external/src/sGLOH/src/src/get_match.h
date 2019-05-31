/* 	Distance matrix, nnr and nn match computation for sGLOH
 *  by Fabio Bellavia (fbellavia@unipa.it),
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
#include <sys/stat.h>

#ifdef _WIN32
#include <time.h>
#else
#include <sys/times.h>
#endif

#include <libconfig.h>
#include <unistd.h>

#define ADD_ONE 0

#define SILENT 	0
#define VERBOSE 1

#define UPPERLOWER 1
#define NONE 	   0

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

#ifdef _WIN32
#define SLASH '\\'
#else
#define SLASH '/'
#endif

double my_inf;
#define INFINITY_ my_inf;

#define OPT_DATA_OUT_NN "out_nn.txt"
#define OPT_DATA_OUT_NNR "out_nnr.txt"
#define OPT_DATA_DATA1 "data1.txt"
#define OPT_DATA_DATA2 "data2.txt"
#define OPT_INFO_DISTANCE "L1"
#define OPT_INFO_RAD_MAX 0
#define OPT_INFO_RAD_XMAX 0
#define OPT_INFO_RAD_YMAX 0
#define OPT_INFO_FAST TRUE
#define OPT_INFO_SAVE FALSE
#define OPT_INFO_SAVE_MATRIX "ms"
#define OPT_INFO_LOAD FALSE
#define OPT_INFO_LOAD_MATRIX "ml"
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

char *dists[]={"L1","L2","CHI","B","KL","JS","MIN_NEAR_DIFF"};

#define DISTS 6

char *usage="\nBad parameters format!\n*** Usage ***"			\
		    "\nget_match [<desc data 1> <desc data 2>"		 	\
		    "\n         [<out nn> <out nnr>]]";

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
	char *out_nn;
	char *out_nnr;
	char *data1;
	char *data2;
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
	int fast;
	double rad_max;
	double rad_xmax;
	double rad_ymax;
	char *distance;
	char *load_matrix;
	int load;
	char *save_matrix;
	int save;
	double (*dist_vect[DISTS])(dvect, dvect);
	int dist_n;
} info_struct;

typedef struct prg_opt {
	data_struct data;
	info_struct info;
	desc_struct desc;
} prg_opt;

typedef struct fts {
	int n;
	int d;
	double *f;
} fts;

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

////////////////////////////////////////////////////////////////////////

double dist_JS(dvect a, dvect b);
double dist_KL(dvect a, dvect b);
double dist_batacharyya(dvect a, dvect b);
double dist_chisq(dvect a, dvect b);
double dist_L2(dvect a, dvect b);
double dist_L1(dvect a, dvect b);

double trunc(double x);
double round(double x);
double ceil(double x);
double log2(double x);
double floor(double x);

finfo desc_init(char *filename);
double *desc_all(finfo data, dvect **pts, dvect **desc);
int destroy_desc_all(dvect *d, int n, double *b);
int desc_end(finfo data);

dist_matrix pos_mask(dvect *dd1, dvect *dd2, int rr, int cc,
	info_struct oinfo);
dist_matrix pos_matrix(dvect *dd1, dvect *dd2, int rr, int cc);
dist_matrix compute_dist(prg_opt opt);
dist_matrix compute_dist_guess(prg_opt opt);
int dist_shift(dvect in, dvect out, int step, desc_struct d);
matched_struct matching(dist_matrix md, info_struct oinfo);
matched_struct matching_fast(dist_matrix md);

matched_vect matched_vect_init(int m);
matched_vect matched_vect_insert(matched_vect v, pt3 e);
matched_vect matched_vect_resize(matched_vect v);
int matched_vect_end(matched_vect v);

svect svect_init(int n);
svect svect_cancel(svect r, int k);

int get_info(prg_opt opt);

stime elapsed_time(tictac start, tictac stop, int mode);
tictac gimme_clock(void);

char *append(char *prefix, char *string, char *suffix);
int multiplestrcmp(char *string,char *strings[],int nstrings,int mode);
char *get_basename(char *s);
char *get_filename(char *s);
prg_opt optsdigest(char *argv[], int argc);
int config_lookup_my_int(const config_t *config, const char *path, int *value);
int config_lookup_my_float(const config_t *config, const char *path, double *value);
char *strdup(const char *s);

int sort_val(pt3 *a, pt3 *b);
int matched_vect_val_qsort(matched_vect in, int el);

int main(int argc, char *argv[]);

dist_matrix load_dist_matrix(char *filename);
int save_dist_matrix(dist_matrix r, char *filename);
