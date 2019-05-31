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

// to compile: gcc distance.c -O3 -lconfig -lfftw3 -lm -o distance
// libconfig and fftw3 required

#include "distance.h"

// WARNING: inner direction is reversed !!!

double dist_L1(dvect a, dvect b) {
	int i;
	double res;

	res=0;
	for (i=0;i<a.n;i++)
		res+=fabs(a.el[i]-b.el[i]);
	return res;
}

double dist_L2(dvect a, dvect b) {
	int i;
	double res;

	res=0;
	for (i=0;i<a.n;i++)
		res+=(a.el[i]-b.el[i])*
			 (a.el[i]-b.el[i]);
	return sqrt(res);
}

double dist_chisq(dvect a, dvect b) {
	int i;
	double res;

	res=0;
	for (i=0;i<a.n;i++)
		res+=(a.el[i]-b.el[i])*
			 (a.el[i]-b.el[i])/
			 (a.el[i]+b.el[i]);
	return res/2.0;
}

double dist_batacharyya(dvect a, dvect b) {
	int i;
	double res, sa, sb;

	sa=0;
	sb=0;
	res=0;
	for (i=0;i<a.n;i++) {
		sa+=a.el[i];
		sb+=b.el[i];
		res+=sqrt(a.el[i]*b.el[i]);
	}
	return -log(res/(a.n*sqrt(sa*sb)));
}

double dist_KL(dvect a, dvect b) {
	int i;
	double res, sa, sb;

	sa=0;
	sb=0;
	res=0;
	for (i=0;i<a.n;i++) {
		sa+=a.el[i];
		sb+=b.el[i];
	}
	for (i=0;i<a.n;i++)
		res+=a.el[i]/sa*log2(a.el[i]*sb/(b.el[i]*sa))+
			 b.el[i]/sb*log2(b.el[i]*sa/(a.el[i]*sb));
	return res/2.0;
}

double dist_JS(dvect a, dvect b) {
	int i;
	double res, sa, sb;

	sa=0;
	sb=0;
	res=0;
	for (i=0;i<a.n;i++) {
		sa+=a.el[i];
		sb+=b.el[i];
	}
	for (i=0;i<a.n;i++)
		res+=a.el[i]/sa*log2(a.el[i]*(sa+sb)/
			 ((a.el[i]+b.el[i])*sa))+
			 b.el[i]/sb*log2(b.el[i]*(sb+sa)/
			 ((b.el[i]+a.el[i])*sb));
	return res/2.0;
}

dist_matrix compute_dist_guess(prg_opt opt) {
	finfo desc1, desc2;
	dist_matrix m, mf;
	int i, j, k, q, ti, k0, k1, k2;
	dvect aux, *dd1, *dd2;
	ivect h;
	double v, tm, v0, v1, v2;

	q=opt.desc.dir;
	desc1=desc_init(opt.data.data1);
	desc2=desc_init(opt.data.data2);
	m.r=desc2.n;
	m.c=desc1.n;
	m.el=(float *)malloc(sizeof(float)*m.r*m.c);
	mf.r=desc2.n;
	mf.c=desc1.n;
	mf.el=(float *)malloc(sizeof(float)*mf.r*mf.c*q);
	for (i=0;i<mf.r*mf.c*q;i++)
		mf.el[i]=INFINITY_;
	dd1=desc_all(desc1);
	dd2=desc_all(desc2);
	aux.n=desc1.d;
	aux.el=(double *)malloc(sizeof(double)*aux.n);
	for (i=0;i<mf.c;i++) {
		desc1.desc.el=dd1[i].el;
		for (k=0;k<q;k++) {
			dist_shift(desc1.desc,aux,k,opt.desc);
			for (j=0;j<mf.r;j++) {
				desc2.desc.el=dd2[j].el;
				v=opt.info.dist_vect[opt.info.dist_n]
					(aux,desc2.desc);
				mf.el[i*mf.r*q+j*q+k]=v;
			}
		}
	}
	h.n=q;
	h.el=(int *)calloc(h.n,sizeof(int));
	for (i=0;i<mf.c;i++) {
		tm=INFINITY_;
		ti=-1;
		for (k=0;k<q;k++)
			for (j=0;j<mf.r;j++)
				if (mf.el[i*mf.r*q+j*q+k]<tm) {
					tm=mf.el[i*mf.r*q+j*q+k];
					ti=k;
				}
		h.el[ti]++;
	}
	for (j=0;j<mf.r;j++) {
		tm=INFINITY_;
		ti=-1;
		for (k=0;k<q;k++)
			for (i=0;i<mf.c;i++)
				if (mf.el[i*mf.r*q+j*q+k]<tm) {
					tm=mf.el[i*mf.r*q+j*q+k];
					ti=k;
				}
		h.el[ti]++;
	}
	tm=0;
	ti=-1;
	printf("\nRotation distribution:\n");
	for (i=0;i<h.n;i++) {
		printf("%d ",h.el[i]);
		if (h.el[i]>tm) {
			tm=h.el[i];
			ti=i;
		}
	}
	printf("\n");
	k0=(ti-1);
	k0=(k0<0)?q-1:k0;
	k1=ti;
	k2=(ti+1);
	k2=(k2==q)?0:k2;
	printf("Selected rotation to check: %d %d %d\n\n",k0,k1,k2);
	for (j=0;j<m.r;j++)
		for (i=0;i<m.c;i++) {
			v0=mf.el[i*m.r*q+j*q+k0];
			v1=mf.el[i*m.r*q+j*q+k1];
			v2=mf.el[i*m.r*q+j*q+k2];
			if (v0>v1)
				v0=v1;
			if (v0>v2)
				v0=v2;
			m.el[i*m.r+j]=v0;
		}	
	free(aux.el);
	free(mf.el);
	free(h.el);
	desc_end(desc1);
	desc_end(desc2);
	return m;
}

dist_matrix compute_dist(prg_opt opt) {
	finfo desc1, desc2;
	dist_matrix m;
	int i, j, k;
	dvect aux, *dd1, *dd2;
	double v;

	desc1=desc_init(opt.data.data1);
	desc2=desc_init(opt.data.data2);
	m.r=desc2.n;
	m.c=desc1.n;
	m.el=(float *)malloc(sizeof(float)*m.r*m.c);
	for (i=0;i<m.r*m.c;i++)
		m.el[i]=INFINITY_;
	dd1=desc_all(desc1);
	dd2=desc_all(desc2);
	if (opt.desc.shift==FALSE) {
		for (i=0;i<m.c;i++) {
			desc1.desc.el=dd1[i].el;
			for (j=0;j<m.r;j++) {
				desc2.desc.el=dd2[j].el;
				m.el[i*m.r+j]=opt.info.dist_vect[opt.info.dist_n]
					(desc1.desc,desc2.desc);
			}
		}
	} else {
		aux.n=desc1.d;
		aux.el=(double *)malloc(sizeof(double)*aux.n);
		for (i=0;i<m.c;i++) {
			desc1.desc.el=dd1[i].el;
			for (k=0;k<opt.desc.dir;k++)
				if (opt.desc.check_dir[k])
				{
					dist_shift(desc1.desc,aux,k,opt.desc);
					for (j=0;j<m.r;j++) {
						desc2.desc.el=dd2[j].el;
						v=opt.info.dist_vect[opt.info.dist_n]
							(aux,desc2.desc);
						m.el[i*m.r+j]=(v<m.el[i*m.r+j])?v:m.el[i*m.r+j];
					}
				}
		}
		free(aux.el);
	}
	desc_end(desc1);
	desc_end(desc2);
	return m;
}

int dist_shift(dvect in, dvect out, int step, desc_struct d) {
	int i, j, n, k, m;

	k=d.hist/d.dir;
	n=in.n/d.hist;
	m=0;
	if (d.unique_center_bin==TRUE) {
		for (i=0;i<d.hist;i++)
			out.el[i]=in.el[(i+k*step)%d.hist];
		m=1;
	}
	for (i=0;i<d.rad-m;i++)
		for (k=0;k<d.dir;k++)
			for (j=0;j<d.hist;j++)
				out.el[d.hist*(m+i*d.dir+((k+step)%d.dir))+j]=
					in.el[d.hist*(m+i*d.dir+k)+j];
	return 1;
}

dvect *desc_all(finfo data) {
	dvect *res;
	int i, k;
	double dummy;

	free(data.desc.el);
	res=(dvect *)malloc(sizeof(dvect)*data.n);
	for (k=0;k<data.n;k++) {
		for (i=0;i<5;i++)
			fscanf(data.in,"%lf",&dummy);
		res[k].n=data.d;
		res[k].el=(double *)malloc(sizeof(double)*data.d);
		for (i=0;i<data.d;i++)
			fscanf(data.in,"%lf",&(res[k].el[i]));
	}
	return res;
}

int destroy_desc_all(dvect *d, int n) {
	int k;

	for (k=0;k<n;k++)
		free(d[k].el);
	free(d);
	return 1;
}

finfo desc_init(char *filename) {
	finfo res;

	res.in=fopen(filename,"r");
	fscanf(res.in,"%d %d",&res.d,&res.n);
	res.desc.el=NULL;
	res.desc.n=res.d;
	return res;
}

int desc_end(finfo data) {

	fclose(data.in);
	return 1;
}

matched_vect matched_vect_init(int m) {
	matched_vect r;

	r.m=m;
	r.c=1;
	r.n=0;
	r.el=(pt3 *)malloc(sizeof(pt3));
	return r;
}

matched_vect matched_vect_insert(matched_vect v, pt3 e) {

	if (v.n>=v.c) {
		v.c*=2;
		v.c=(v.m<v.c)?v.m:v.c;
		v.el=(pt3 *)realloc(v.el,sizeof(pt3)*v.c);
	}
	v.el[v.n]=e;
	v.n++;
	return v;
}

matched_vect matched_vect_resize(matched_vect v) {

	if (v.n)
		v.el=(pt3 *)realloc(v.el,sizeof(pt3)*v.n);
	else
		v.el=NULL;
	v.c=v.n;
	return v;
}

int matched_vect_end(matched_vect v) {

	if (v.el)
		free(v.el);
	return 1;
}

svect svect_init(int n) {
	svect r;
	int i;

	r.n=n;
	r.el=(pt2 *)malloc(sizeof(pt2)*r.n);
	for (i=0;i<r.n;i++) {
		r.el[i].prec=i-1;
		r.el[i].next=i+1;
	}
	r.el[0].prec=0;
	r.el[r.n-1].next=-1;
	return r;
}

svect svect_cancel(svect r, int k) {

	if (r.el[k].next!=-1)
		r.el[r.el[k].next].prec=r.el[k].prec;
	r.el[r.el[k].prec].next=r.el[k].next;
	if (r.el[0].prec==k)
		r.el[0].prec=r.el[k].next;
	return r;
}

matched_struct matching(dist_matrix md) {
	matched_struct res;
	int i, j, m, n;
	long int k;
	pt3 e;
	matched_vect tmp;
	double aux;
	int *r_i, *c_i;
	int index;

	index=1;
	r_i=(int *)calloc(md.r,sizeof(int));
	c_i=(int *)calloc(md.c,sizeof(int));
	tmp.n=md.r*md.c;
	tmp.el=(pt3 *)malloc(tmp.n*sizeof(pt3));
	for (i=0;i<md.c;i++)
	   for (j=0;j<md.r;j++) {
		  tmp.el[i*md.r+j].val=md.el[i*md.r+j];
          tmp.el[i*md.r+j].x=j;
          tmp.el[i*md.r+j].y=i;
    }
	matched_vect_val_qsort(tmp,tmp.n);
	m=(md.r<md.c)?md.r:md.c;
	res.nn=matched_vect_init(m);
	for (k=0,n=0;n<m;k++) {
	   if (isinf(tmp.el[k].val))
	      break;
	   if ((!(r_i[tmp.el[k].x])) && (!(c_i[tmp.el[k].y]))) {
       e.val=tmp.el[k].val;
       e.x=tmp.el[k].x;
       e.y=tmp.el[k].y;
	   e.index=index;
       res.nn=matched_vect_insert(res.nn,e);
       r_i[tmp.el[k].x]=1;
       c_i[tmp.el[k].y]=1;
       n++;
     }
   }
   res.nn=matched_vect_resize(res.nn);
   matched_vect_end(tmp);
   free(r_i);
   free(c_i);
   res.nnr=matched_vect_init(m);
   for (k=0;k<res.nn.n;k++) {
      e.val=INFINITY_;
	     for (j=0;j<md.r;j++) {
		 e.y=res.nn.el[k].y;
		 e.x=res.nn.el[k].x;
		 aux=md.el[e.y*md.r+j]/md.el[e.y*md.r+e.x];
		 if ((e.x!=j) && (aux<e.val) && (aux>1))
		    e.val=aux;
		 }
	  e.index=index;
	  e.val*=-1;
	  res.nnr=matched_vect_insert(res.nnr,e);
   }
   res.nnr=matched_vect_resize(res.nnr);
   matched_vect_val_qsort(res.nnr,res.nnr.n);
   for (i=0;i<res.nnr.n;i++) {
      res.nnr.el[i].val*=-1;
   }
   return res;
}

/*
matched_struct matching(dist_matrix md) {
	matched_struct res;
	svect r, c;
	int i, j, m, n, index;
	pt3 e;
	double aux;

	index=1;
	res.nn=matched_vect_init(md.r*md.c);
	res.nnr=matched_vect_init(md.r*md.c);
	r=svect_init(md.r);
	c=svect_init(md.c);
	m=(r.n<c.n)?r.n:c.n;
	for (n=0;n<m;n++) {
		e.val=INFINITY_;
		for (i=c.el[0].prec;i>=0;i=c.el[i].next)
			for (j=r.el[0].prec;j>=0;j=r.el[j].next)
				if (md.el[i*md.r+j]<e.val) {
					e.x=j;
					e.y=i;
					e.val=md.el[i*md.r+j];
				}
		if (isinf(e.val))
			break;
		e.index=index++;
		res.nn=matched_vect_insert(res.nn,e);
		e.val=INFINITY_;
		for (j=0;j<md.r;j++) {
			aux=md.el[e.y*md.r+j]/md.el[e.y*md.r+e.x];
			if ((e.x!=j) && (aux<e.val) && (aux>1))
				e.val=aux;
		}
		e.val=-e.val;
		res.nnr=matched_vect_insert(res.nnr,e);
		r=svect_cancel(r,e.x);
		c=svect_cancel(c,e.y);
	}
	res.nn=matched_vect_resize(res.nn);
	res.nnr=matched_vect_resize(res.nnr);
	matched_vect_val_qsort(res.nnr,res.nnr.n);
	for (i=0;i<res.nnr.n;i++) {
		res.nnr.el[i].index=i+1;
		res.nnr.el[i].val*=-1;
	}
	free(r.el);
	free(c.el);
	return res;
}
* */

rpt_struct repeatability(prg_opt opt) {
	fts feat1, feat2,  feat1_, feat2_;
	double h[9], h_[9];
	int i;
	rpt_struct r;
	rgb_img im1, im2;
	char *rpt_save, *rpt_load;

	rpt_load=NULL;
	if (opt.info.rload)
		rpt_load=opt.info.rload_matrix;
	rpt_save=NULL;
	if (opt.info.rsave)
		rpt_save=opt.info.rsave_matrix;
	im1=load_bmp(opt.data.im1);
	free(im1.val);
	im2=load_bmp(opt.data.im2);
	free(im2.val);
	feat1=mk_load(opt.data.data1,9);
	feat2=mk_load(opt.data.data2,9);
	h_load(opt.data.h,h);
	inv(h,h_,3);
	feat1_=project_region(feat1,h);
	feat2_=project_region(feat2,h_);
	r.good_table=rpt_table(feat1,feat2_,opt.info.overlap,rpt_load,
		rpt_save);
	r.corr=0;
	for (i=0;i<r.good_table.n;i++)
		r.corr+=(r.good_table.el[i].index==NN);
	r.overlap=opt.info.overlap;
	free(feat1.f);
	free(feat2.f);
	free(feat1_.f);
	free(feat2_.f);
	return r;
}

aff_pt from_mk_pt(double *in) {
	aff_pt r;
	double U[4], D[4], V[4], T[4];

	r.x=in[0];
	r.y=in[1];
	U[0]=in[2];
	U[1]=in[3];
	U[2]=in[3];
	U[3]=in[4];
	diagonalize2x2(U,D,V);
	D[0]=1/(3.0*sqrt(D[0]));
	D[3]=1/(3.0*sqrt(D[3]));
	r.si=D[0]>D[3]?D[0]:D[3];
	D[0]/=r.si;
	D[3]/=r.si;
	inv2x2(V,T);
	prod2x2(D,T,U);
	prod2x2(V,U,r.U);
	return r;
}

fts mk_load(char *filename, int s) {
	FILE *file_idx;
	int i, j;
	fts feat;
	double dummy;

	file_idx=fopen(filename,"r");
	fscanf(file_idx,"%d %d",&feat.d,&feat.n);
	feat.f=(double *)malloc(sizeof(double)*s*feat.n);
	for (i=0;i<feat.n;i++) {
		for (j=0;j<5;j++)
			fscanf(file_idx,"%lf",&(feat.f[i*s+j]));
		for (j=0;j<feat.d;j++)
			fscanf(file_idx,"%lf",&dummy);
		feat.f[i*s]+=ADD_ONE;
		feat.f[i*s+1]+=ADD_ONE;
	}
	feat.d=0;
	fclose(file_idx);
	return feat;
}

int h_load(char *filename, double *h) {
	FILE *file_idx;
	int i, j;

	file_idx=fopen(filename,"r");
	for (i=0;i<3;i++)
		for (j=0;j<3;j++)
			fscanf(file_idx,"%lf",&(h[i*3+j]));
	fclose(file_idx);
	return 1;
}

int get_affine(double x, double y, double *h, double *res) {
	double aux;

	aux=h[6]*x+h[7]*y+h[8];
	res[0]=h[0]/aux-(h[0]*x+h[1]*y+h[2])*h[6]/(aux*aux);
	res[1]=h[1]/aux-(h[0]*x+h[1]*y+h[2])*h[7]/(aux*aux);
	res[2]=h[3]/aux-(h[3]*x+h[4]*y+h[5])*h[6]/(aux*aux);
	res[3]=h[4]/aux-(h[3]*x+h[4]*y+h[5])*h[7]/(aux*aux);
	return 1;
}

int diagonalize2x2(double *m, double *D, double *R) {
	double tr, det, d;

	tr=m[0]+m[3];
	det=m[0]*m[3]-m[1]*m[2];
	D[0]=(tr-sqrt(tr*tr-4*det))/2.0;
	D[1]=0;
	D[2]=0;
	D[3]=(tr+sqrt(tr*tr-4*det))/2.0;
	if (!m[1]) {
		R[0]=1;
		R[1]=0;
		R[2]=0;
		R[3]=1;
		return 1;
	}
	d=sqrt(m[1]*m[1]+D[0]*D[0]-2*D[0]*m[0]+m[0]*m[0]);
	if (d) {
		R[0]=m[1]/d;
		R[2]=(D[0]-m[0])/d;
	} else {
		R[0]=1;
		R[2]=0;
	}
	d=sqrt(m[1]*m[1]+D[3]*D[3]-2*D[3]*m[0]+m[0]*m[0]);
	if (d) {
		R[1]=m[1]/d;
		R[3]=(D[3]-m[0])/d;
	} else {
		R[1]=0;
		R[3]=1;
	}
	return 1;
}

int transpose2x2(double *m,double *res) {

	res[0]=m[0];
	res[1]=m[2];
	res[2]=m[1];
	res[3]=m[3];
	return 1;
}

fts project_region(fts feat, double *h) {
	double D[4], V[4], M[4], M_[4], aux, aff[4], afft[4];
	int i;
	fts feat_;

	feat_.f=(double *)malloc(sizeof(double)*feat.n*9);
	for (i=0;i<feat.n;i++) {
		M[0]=feat.f[i*9+2];
		M[1]=feat.f[i*9+3];
		M[2]=M[1];
		M[3]=feat.f[i*9+4];
		diagonalize2x2(M,D,V);
		feat.f[i*9+5]=1/sqrt(D[0]);
		feat.f[i*9+6]=1/sqrt(D[3]);
		aux=feat.f[i*9+2]*feat.f[i*9+4]-
			feat.f[i*9+3]*feat.f[i*9+3];
		feat.f[i*9+7]=sqrt(feat.f[i*9+4]/aux);
		feat.f[i*9+8]=sqrt(feat.f[i*9+2]/aux);
		get_affine(feat.f[i*9+0],feat.f[i*9+1],h,aff);
		aux=h[6]*feat.f[i*9+0]+
			h[7]*feat.f[i*9+1]+h[8];
		feat_.f[i*9+0]=(h[0]*feat.f[i*9+0]+
						h[1]*feat.f[i*9+1]+h[2])/aux;
		feat_.f[i*9+1]=(h[3]*feat.f[i*9+0]+
						h[4]*feat.f[i*9+1]+h[5])/aux;
		inv2x2(M,M_);
		transpose2x2(aff,afft);
		prod2x2(aff,M_,M);
		prod2x2(M,afft,M_);
		inv2x2(M_,M);
		feat_.f[i*9+2]=M[0];
		feat_.f[i*9+3]=M[1];
		feat_.f[i*9+4]=M[3];
		diagonalize2x2(M,D,V);
		feat_.f[i*9+5]=1/sqrt(D[0]);
		feat_.f[i*9+6]=1/sqrt(D[3]);
		aux=feat_.f[i*9+2]*feat_.f[i*9+4]-
			feat_.f[i*9+3]*feat_.f[i*9+3];
		feat_.f[i*9+7]=sqrt(feat_.f[i*9+4]/aux);
		feat_.f[i*9+8]=sqrt(feat_.f[i*9+2]/aux);
	}
	feat_.n=feat.n;
	feat_.d=0;
	return feat_;
}

matched_vect rpt_table(fts ft1, fts ft2, double overlap, char *rpt_load,
	char *rpt_save) {
    double *feat1, *feat2;
    int n1, n2, desc_size1;
	int i, j, f1, f2, n, m, k;
	float feat1a[9];
	float feat2a[9];
	float *tover_out;
	float max_dist, fac1, fac2, dist, dx, dy, ov, a, b, bua, bna;
	float maxx, minx, maxy, miny, mina, dr, rx, rx2, ry, ry2;
    matched_vect res;
	svect r, c;
	pt3 e;
	dist_matrix rpt_matrix;

    feat1=ft1.f;
    feat2=ft2.f;
    n1=ft1.n;
    n2=ft2.n;
    desc_size1=ft1.d;
	if (rpt_load) {
		rpt_matrix=load_dist_matrix(rpt_load);
		tover_out=rpt_matrix.el;
	} else {
		tover_out=(float *)malloc(sizeof(float)*n1*n2);
		for (i=0;i<n1*n2;i++)
			tover_out[i]=100.0;
		for(j=0,f1=0;j<n1;j++,f1+=9+desc_size1) {
			max_dist=sqrt(feat1[f1+5]*feat1[f1+6]);
			fac1=1/9.0;
			max_dist=max_dist*4;
			feat1a[2]=fac1*feat1[f1+2];
			feat1a[3]=fac1*feat1[f1+3];
			feat1a[4]=fac1*feat1[f1+4];
			feat1a[7]=sqrt(feat1a[4]/
				(feat1a[2]*feat1a[4]-feat1a[3]*feat1a[3]));
			feat1a[8]=sqrt(feat1a[2]/
				(feat1a[2]*feat1a[4]-feat1a[3]*feat1a[3]));
			for (i=0,f2=0;i<n2;i++,f2+=9) {
				//compute shift error between ellipses
				dx=feat2[f2]-feat1[f1];
				dy=feat2[f2+1]-feat1[f1+1];
				dist=sqrt(dx*dx+dy*dy);
				if (dist<max_dist) {
					fac2=1/9.0;
					feat2a[2]=fac2*feat2[f2+2];
					feat2a[3]=fac2*feat2[f2+3];
					feat2a[4]=fac2*feat2[f2+4];
					feat2a[7]=sqrt(feat2a[4]/(feat2a[2]*feat2a[4]-
											  feat2a[3]*feat2a[3]));
					feat2a[8]=sqrt(feat2a[2]/(feat2a[2]*feat2a[4]-
											  feat2a[3]*feat2a[3]));
					//find the largest eigenvalue
					maxx=ceil((feat1a[7]>(dx+feat2a[7]))?
							  feat1a[7]:(dx+feat2a[7]));
					minx=floor((-feat1a[7]<(dx-feat2a[7]))?
							  (-feat1a[7]):(dx-feat2a[7]));
					maxy=ceil((feat1a[8]>(dy+feat2a[8]))?
							  feat1a[8]:(dy+feat2a[8]));
	  				miny=floor((-feat1a[8]<(dy-feat2a[8]))?
							  (-feat1a[8]):(dy-feat2a[8]));
					mina=((maxx-minx)<(maxy-miny))?
						  (maxx-minx):(maxy-miny);
					dr=mina/50.0;
					bua=0;
					bna=0;
					//compute the area
					for(rx=minx;rx<=maxx;rx+=dr) {
		  				rx2=rx-dx;
		  				for(ry=miny;ry<=maxy;ry+=dr) {
		    				ry2=ry-dy;
							//compute distance from ellipse center
							a=feat1a[2]*rx*rx+2*feat1a[3]*rx*ry+
							  feat1a[4]*ry*ry;
		   					b=feat2a[2]*rx2*rx2+2*feat2a[3]*rx2*ry2+
		   					  feat2a[4]*ry2*ry2;
							//compute the area
		   					if (a<1 && b<1)
								bna++;
		   					if (a<1 || b<1)
								bua++;
		 				}
					}
					ov=100.0*(1-bna/bua);
	   				tover_out[j*n2+i]=ov;
				}
			}
		}
	}
	if (rpt_save) {
		rpt_matrix.r=ft2.n;
		rpt_matrix.c=ft1.n;
		rpt_matrix.el=tover_out;
		save_dist_matrix(rpt_matrix,rpt_save);
	}
	res=matched_vect_init(n1*n2);
	for (j=0;j<n2;j++)
		for (i=0;i<n1;i++)
			if (tover_out[i*n2+j]<overlap) {
				e.x=j;
				e.y=i;
				e.index=ALL;
				e.val=tover_out[i*n2+j];
				k=bin_search(res,e);
				if (k<0)
					res=matched_vect_insert(res,e);
				else if (res.el[k].val>e.val)
					res.el[k].val=e.val;
			}
	res=matched_vect_resize(res);
	r=svect_init(n2);
	c=svect_init(n1);
	m=(n1<n2)?n1:n2;
	for (n=0;n<m;n++) {
		e.val=100;
		for (i=c.el[0].prec;i>=0;i=c.el[i].next)
			for (j=r.el[0].prec;j>=0;j=r.el[j].next)
				if (tover_out[i*n2+j]<e.val) {
					e.x=j;
					e.y=i;
					e.val=tover_out[i*n2+j];
				}
		if (e.val>=overlap)
			break;
		res.el[bin_search(res,e)].index=NN;
		r=svect_cancel(r,e.x);
		c=svect_cancel(c,e.y);
	}
	free(r.el);
	free(c.el);
	free(tover_out);
	return res;
}

int inv2x2(double *m,double *res) {
	double det;

	det=m[0]*m[3]-m[1]*m[2];
	res[0]=m[3]/det;
	res[1]=-m[1]/det;
	res[2]=-m[2]/det;
	res[3]=m[0]/det;
	return 1;
}

int prod2x2(double *m1, double *m2, double *res) {
	res[0]=m1[0]*m2[0]+m1[1]*m2[2];
	res[1]=m1[0]*m2[1]+m1[1]*m2[3];
	res[2]=m1[2]*m2[0]+m1[3]*m2[2];
	res[3]=m1[2]*m2[1]+m1[3]*m2[3];
	return 1;
}

int best_swap(dmatrix m, ivect r_swap, ivect c_swap, dvect b, xy p) {
	xy e;
	int i, j;

	e.x=p.x;
	e.y=p.y;
	for (i=p.y;i<m.c;i++)
		for (j=p.x;j<m.r;j++)
			if (fabs(m.el[c_swap.el[e.y]*m.r+r_swap.el[e.x]])<
			    fabs(m.el[c_swap.el[i]*m.r+r_swap.el[j]])) {
				e.x=j;
				e.y=i;
			}
	i=r_swap.el[e.x];
	r_swap.el[e.x]=r_swap.el[p.x];
	r_swap.el[p.x]=i;
	i=c_swap.el[e.y];
	c_swap.el[e.y]=c_swap.el[p.y];
	c_swap.el[p.y]=i;
	return 1;
}

dvect gauss_elimination(dmatrix m, dvect b) {
	dvect a;
	ivect r_swap, c_swap;
	int i, j, k;
	xy p;
	double v;

	a.n=m.r;
	a.el=(double *)malloc(sizeof(double)*a.n);
	r_swap.n=m.r;
	r_swap.el=(int *)malloc(sizeof(int)*r_swap.n);
	for (i=0;i<r_swap.n;i++)
		r_swap.el[i]=i;
	c_swap.n=m.c;
	c_swap.el=(int *)malloc(sizeof(int)*c_swap.n);
	for (i=0;i<c_swap.n;i++)
		c_swap.el[i]=i;
	for (i=0;i<m.r-1;i++) {
		p.x=i;
		p.y=i;
		best_swap(m,r_swap,c_swap,b,p);
		for (j=i+1;j<m.c;j++) {
			v=m.el[c_swap.el[i]*m.r+r_swap.el[i]]/
			  m.el[c_swap.el[j]*m.r+r_swap.el[i]];
			if (isinf(v))
				continue;
			for (k=i;k<m.r;k++)
				m.el[c_swap.el[j]*m.r+r_swap.el[k]]=v*
					m.el[c_swap.el[j]*m.r+r_swap.el[k]]-
					m.el[c_swap.el[i]*m.r+r_swap.el[k]];
			b.el[c_swap.el[j]]=v*b.el[c_swap.el[j]]-b.el[c_swap.el[i]];
		}
	}
	for (i=m.c-1;i>=0;i--) {
		a.el[r_swap.el[i]]=b.el[c_swap.el[i]];
		for (j=m.c-1;j>i;j--) {
			a.el[r_swap.el[i]]-=a.el[r_swap.el[j]]*
								m.el[c_swap.el[i]*m.r+r_swap.el[j]];
		}
		a.el[r_swap.el[i]]/=m.el[c_swap.el[i]*m.r+r_swap.el[i]];
	}
	free(r_swap.el);
	free(c_swap.el);
	return a;
}

int inv(double *h, double *r, int n) {
	dvect a, b;
	dmatrix m;
	int i, j, n2;

	n2=n*n;
	b.n=n2;
	b.el=(double *)malloc(sizeof(double)*n2);
	m.r=n2;
	m.c=n2;
	m.el=(double *)calloc(n2*n2,sizeof(double));
	for (i=0;i<n2;i++) {
		b.el[i]=((i%n)==(i/n))?1:0;
		for (j=0;j<n;j++)
			m.el[i*n2+i/n+j*n]=h[(i%n)*n+j];
		}
	a=gauss_elimination(m,b);
	for (i=0;i<n2;i++)
		r[i]=a.el[i];
	free(a.el);
	free(b.el);
	free(m.el);
	return 1;
}

int sort_el(pt3 *a, pt3 *b) {
	if (a->x==b->x) {
		if (a->y==b->y)
			return 0;
		return (a->y>b->y)?1:-1;
	}
	return (a->x>b->x)?1:-1;
}

int matched_vect_qsort(matched_vect in, int el) {
	qsort((void *)in.el,el,sizeof(pt3),
		  (int(*)(const void *, const void *))&sort_el);
	return 1;
}

rgb_img load_bmp(char filename[]) {
	FILE *bmp_file;
	rgb_img res;
	unsigned int width,
				 height,
				 type,
				 compression,
				 i,j,k,w,z,d,
				 char_read,
				 char_pad,
				 char_rle,
				 abs_rle,
				 abs_pad,
				 color_index,
				 rle_status,
				 run_length,
				 rle_char,
				*image=NULL;
	RGBQUAD *color=NULL;
	BITMAPFILEHEADER bfh;
	BITMAPINFOHEADER bih;
	unsigned char buffer[BUFFER_SIZE];

	bmp_file=fopen(filename,"rb");
	if ((!fread(&(bfh.bfType),sizeof(unsigned short),1,bmp_file)) ||
		(bfh.bfType!=19778) ||
	    (!fread(&(bfh.bfSize),sizeof(unsigned int),1,bmp_file)) ||
	    (!fread(&(bfh.bfReserved1),sizeof(unsigned short),1,
	    	bmp_file)) || (bfh.bfReserved1) ||
	    (!fread(&(bfh.bfReserved2),sizeof(unsigned short),1,
	    	bmp_file)) || (bfh.bfReserved2) ||
	    (!fread(&(bfh.bfOffBits),sizeof(unsigned int),1,
	    	bmp_file)) ||
	    (!fread(&(bih.biSize),sizeof(unsigned int),1,
	    	bmp_file)) || (bih.biSize!=40) ||
		(!fread(&(bih.biWidth),sizeof(unsigned int),1,
			bmp_file)) ||
	    (!fread(&(bih.biHeight),sizeof(unsigned int),1,
	    	bmp_file)) ||
	    (!fread(&(bih.biPlanes),sizeof(unsigned short),1,
	    	bmp_file)) || (bih.biPlanes!=1) ||
	    (!fread(&(bih.biBitCount),sizeof(unsigned short),1,
	    	bmp_file)) ||
		(!fread(&(bih.biCompression),sizeof(unsigned int),1,
			bmp_file)) ||
		(!fread(&(bih.biSizeImage),sizeof(unsigned int),1,
			bmp_file)) ||
		(!fread(&(bih.biXPelsPerMeter),sizeof(unsigned int),1,
			bmp_file)) ||
		(!fread(&(bih.biYPelsPerMeter),sizeof(unsigned int),1,
			bmp_file)) ||
    	(!fread(&(bih.biClrUsed),sizeof(unsigned int),1,
    		bmp_file)) ||
		(!fread(&(bih.biClrImportant),sizeof(unsigned int),1,
			bmp_file)))
		EXIT_ERROR(__FILE__,__LINE__,"Invalid file or image header!")

	res.row=width=bih.biWidth;
	res.col=height=bih.biHeight;
	type=bih.biBitCount;
	switch (type) {
			break;
		case 4:
		case 8:
		    compression=(((bih.biCompression==BI_RLE8) && (type==8)) ||
						 ((bih.biCompression==BI_RLE4) && (type==4)))?
						 1:0;
			if ((!compression) && (bih.biCompression!=BI_RGB))
				EXIT_ERROR(__FILE__,__LINE__,"Unknown compression!")
			break;
		case 24:
			if ((compression=bih.biCompression))
				EXIT_ERROR(__FILE__,__LINE__,"Wrong format!")
			break;
		default:
			EXIT_ERROR(__FILE__,__LINE__,"Unsupported bitmap format!")
	}
	if (!(compression^bih.biSizeImage))
		EXIT_ERROR(__FILE__,__LINE__,"Invalid data size!")
	image=(unsigned int *)malloc(sizeof(unsigned int)*width*height);
	if (type!=24) {
		color=(RGBQUAD *)malloc(sizeof(RGBQUAD)*bih.biClrUsed);
		if (!(fread(color,sizeof(RGBQUAD),bih.biClrUsed,bmp_file)==
			bih.biClrUsed))
			EXIT_ERROR(__FILE__,__LINE__,"Bad palette!")
	}
	if (((type==24) && (bfh.bfOffBits!=54)) ||
		(((type==4) || (type==8) || (type==1)) &&
		 (bfh.bfOffBits!=54+4*bih.biClrUsed)))
		EXIT_ERROR(__FILE__,__LINE__,"Wrong image offset!")
	fseek(bmp_file,bfh.bfOffBits,SEEK_SET);
	switch (type) {
		case 24:
			char_pad=(4-((width*3)%4))%4;
			i=j=k=0;
			while ((i<height) && ((char_read=
								   fread(&buffer,sizeof(unsigned char),
										 BUFFER_SIZE,bmp_file)))) {
				for (;k<char_read;k+=3) {
					if ((k>BUFFER_SIZE-3) && (k<BUFFER_SIZE)) {
						fseek(bmp_file,(long)k-BUFFER_SIZE,SEEK_CUR);
						k=0;
						break;
					}
					if (i==height)
						EXIT_ERROR(__FILE__,__LINE__,"Bad bitmap data!")
					image[(height-i-1)*width+j]=(*(unsigned int *)
												(&buffer[k])
												 & 0xFFFFFF);
					if (++j==width) {
						i++;
						j=0;
						k+=char_pad;
					}
				}
				k%=BUFFER_SIZE;
			}
			if ((char_read=fread(&buffer,sizeof(unsigned char),
								 BUFFER_SIZE,bmp_file)) || (i<height))
				EXIT_ERROR(__FILE__,__LINE__,"Bad bitmap data!")
			break;
		case 1:
			char_pad=(4-(((unsigned int)ceil((double)width/8))%4))%4;
			i=j=k=0;
			while ((i<height) && ((char_read=fread(&buffer,
					  sizeof(unsigned char),BUFFER_SIZE,bmp_file)))) {
				for (;k<char_read;k++) {
					for (w=0;w<8;w++) {
						color_index=(buffer[k] & (1<<w))?1:0;
						if (i==height)
							EXIT_ERROR(__FILE__,__LINE__,
									   "Bad bitmap data!")
						image[(height-i-1)*width+j]=(*(unsigned int *)
												(&color[color_index]));
						if (++j==width) {
							i++;
							j=0;
							k+=char_pad;
							break;
						}
					}
				}
				k%=BUFFER_SIZE;
			}
			if ((char_read=fread(&buffer,1,BUFFER_SIZE,bmp_file)) ||
			    (i<height) || j)
				EXIT_ERROR(__FILE__,__LINE__,"Bad bitmap data!")
			break;
		case 4:
		case 8:
			if (type==4) {
				z=2;
				d=1;
			} else {
				z=1;
				d=2;
			}
			if (!compression) {
				char_pad=(4-(((unsigned int)
							  ceil((double)width/z))%4))%4;
				i=j=k=0;
				while ((i<height) && ((char_read=fread(&buffer,
					sizeof(unsigned char),BUFFER_SIZE,bmp_file)))) {
					for (;k<char_read;k++) {
						for (w=0;w<(z*4);w+=4) {
							color_index=(type==4)?(buffer[k]>>(4-w)) &
							 					  0xF:buffer[k];
							if (i==height)
								EXIT_ERROR(__FILE__,__LINE__,
										   "Bad bitmap data!")
							image[(height-i-1)*width+j]=
								((unsigned int *)color)[color_index];
							if (++j==width) {
								i++;
								j=0;
								k+=char_pad;
								break;
							}
						}
					}
					k%=BUFFER_SIZE;
				}
				if ((char_read=fread(&buffer,1,BUFFER_SIZE,bmp_file))
					|| (i<height) || j)
					EXIT_ERROR(__FILE__,__LINE__,"Bad bitmap data!")
			} else {
				rle_char=0;
				i=j=k=0;
				rle_status=0;
				char_rle=0;
				while ((char_read=fread(&buffer,1,BUFFER_SIZE,
						bmp_file)) && ((char_rle+=char_read)<=
										bih.biSizeImage)) {
					for (;k<char_read;k++) {
						rle_char++;
						switch (rle_status) {
							case START:
								if (!buffer[k])
									rle_status=SPECIAL;
								else {
									rle_status=RLE;
									run_length=buffer[k];
								}
								break;
							case PAD0_FIRST:
								if (!buffer[k])
									rle_status=PAD0_LAST;
								else
									EXIT_ERROR(__FILE__,__LINE__,
											   "Bad bitmap data!");
								break;
							case END:
								EXIT_ERROR(__FILE__,__LINE__,
										   "Bad bitmap data!")
								break;
							case PAD0_LAST:
								if (!buffer[k])
									rle_status=END;
								else
									EXIT_ERROR(__FILE__,__LINE__,
											   "Bad bitmap data!");
								break;
							case RLE:
								for (w=0;(w<(run_length*d));w+=d) {
									color_index=(type==4)?
									(buffer[k]>>(4*(1-(w%2)))) &
										0xF:buffer[k];
									if ((i==height) && (j>=width) &&
										(w<run_length))
										EXIT_ERROR(__FILE__,__LINE__,
												   "Bad bitmap data!")
									image[(height-i-1)*width+j++]=
										((unsigned int *)
										color)[color_index];
								}
								rle_status=START;
								break;
							case OFFSET:
								i+=buffer[k++];
								j+=buffer[k];
								rle_char++;
								rle_status=START;
								break;
							case ABS:
								color_index=(type==4)?(buffer[k])>>4 &
									0xF:buffer[k];
								if ((i==height) && (j>=width))
									EXIT_ERROR(__FILE__,__LINE__,
											   "Bad bitmap data!")
								image[(height-i-1)*width+j++]=
									((unsigned int *)color)[color_index];
								if (type==4) {
									if ((i==height) && (j>=width))
										EXIT_ERROR(__FILE__,__LINE__,
												   "Bad bitmap data!")
									color_index=buffer[k] & 0xF;
									image[(height-i-1)*width+j++]=
										((unsigned int *)color)
										[color_index];
								}
								abs_rle-=2;
								if (abs_rle==1) {
									color_index=buffer[++k]>>4 & 0xF;
									rle_char++;
									if ((i==height) && (j>=width))
										EXIT_ERROR(__FILE__,__LINE__,
												   "Bad bitmap data!")
									image[(height-i-1)*width+j++]=
										((unsigned int *)color)
										[color_index];
								}
								if (abs_rle<2) {
									rle_status=START;
									k+=abs_pad;
									rle_char+=abs_pad;
								}
								break;
							case SPECIAL:
								switch (buffer[k]) {
									case 0:
										i++;
										j=0;
										rle_status=START;
										break;
									case 1:
										if ((rle_char%4)==2)
											rle_status=PAD0_FIRST;
										else
											rle_status=END;
										break;
									case 2:
										rle_status=OFFSET;
										break;
									default:
										rle_status=ABS;
										abs_rle=(buffer[k])*d;
										abs_pad=((unsigned int)
											(ceil(((double)buffer[k])/
																z)))%2;
										break;
								}
								break;
						}
					}
					k%=BUFFER_SIZE;
				}
				if ((rle_status!=END) ||
					(ftell(bmp_file)-bfh.bfOffBits!=bih.biSizeImage))
					EXIT_ERROR(__FILE__,__LINE__,"Bad bitmap data!")
			}
			break;
	}
	res.val=image;
	fclose(bmp_file);
	if (color)
		free(color);
	return res;
}

#ifdef _WIN32

stime elapsed_time(tictac start, tictac stop, int mode) {
	stime res;
	double clock_per_sec, h, m, s;

	clock_per_sec=(double)CLK_TCK;
	res.realt=(double)(stop.t-start.t)/clock_per_sec;
	if (mode==VERBOSE) {
		fprintf(stdout,"***** Elapsed  times *****\n");
		h=trunc(res.realt/(60*60));
		m=trunc((res.realt-h*60*60)/60);
		s=res.realt-h*60*60-m*60;
		fprintf(stdout,"  Real time: %.0f:%.0f:%0.2f\n",h,m,s);
	}
	return res;
}

tictac gimme_clock(void) {
	tictac in;

	in.t=clock();
	return in;
}

#else

stime elapsed_time(tictac start, tictac stop, int mode) {
	stime res;
	double clock_per_sec, h, m, s;

	clock_per_sec=(double)sysconf(_SC_CLK_TCK);
	res.realt=(double)(stop.t-start.t)/clock_per_sec;
	res.usert=(double)(stop.c.tms_utime-start.c.tms_utime)/
		clock_per_sec;
	res.syst=(double)(stop.c.tms_stime-start.c.tms_stime)/
		clock_per_sec;
	if (mode==VERBOSE) {
		fprintf(stdout,"***** Elapsed  time *****\n");
		h=trunc(res.realt/(60.0*60.0));
		m=trunc((res.realt-h*60.0*60.0)/60.0);
		s=res.realt-h*60.0*60.0-m*60.0;
		fprintf(stdout,"  Real time: %02.0f:%02.0f:%05.2f\n",h,m,s);
		h=trunc(res.usert/(60.0*60.0));
		m=trunc((res.usert-h*60.0*60.0)/60.0);
		s=res.usert-h*60.0*60.0-m*60.0;
		fprintf(stdout,"  User time: %02.0f:%02.0f:%05.2f\n",h,m,s);
		h=trunc(res.syst/(60.0*60.0));
		m=trunc((res.syst-h*60.0*60.0)/60.0);
		s=res.syst-h*60.0*60.0-m*60.0;
		fprintf(stdout,"System time: %02.0f:%02.0f:%05.2f\a\n",h,m,s);
	}
	return res;
}

tictac gimme_clock(void) {
	tictac in;

	in.t=times(&in.c);
	return in;
}

#endif

char *append(char *prefix, char *string, char *suffix) {
	char *res;
	int i, l;
	l=i=0;
	if (prefix)
		for (l=0;prefix[l];l++);
	if (string)
		for (i=0;string[i];i++,l++);
	if (suffix)
		for (i=0;suffix[i];i++,l++);
	res=(char *)malloc(sizeof(char)*++l);
	if (prefix)
		for (l=i=0;prefix[i];res[l++]=prefix[i++]);
	if (string)
		for (i=0;string[i];res[l++]=string[i++]);
	if (suffix)
		for (i=0;suffix[i];res[l++]=suffix[i++]);
	res[l]='\0';
	return res;
}

int multiplestrcmp(char *string,char *strings[],int nstrings,int mode) {
	int i, j;

	if (mode & UPPERLOWER) {
		for (i=0;string[i]!='\0';i++)
			if ((string[i]>='a') && (string[i]<='z'))
				string[i]-=32;
		for (i=0;i<nstrings;i++)
			for(j=0;strings[i][j]!='\0';j++)
				if ((strings[i][j]>='a') && (strings[i][j]<='z'))
					strings[i][j]-=32;
	}
	for (i=0;i<nstrings;i++) {
		for (j=0;(strings[i][j]!='\0') && (string[j]!='\0') &&
			 (strings[i][j]==string[j]);j++);
		if ((strings[i][j]=='\0') && (string[j]=='\0'))
			return i;
	}
	return -1;
}

int save_bmp(char *filename, rgb_img im) {
	FILE *bmp_file;
	unsigned short aux;
	unsigned int aux_, i, j, k, g, char_pad;
	char buffer[BUFFER_SIZE];

	char_pad=(4-((im.row*3)%4))%4;
	bmp_file=fopen(filename,"wb");
	aux=19778;
	fwrite(&aux,sizeof(unsigned short),1,bmp_file);
	aux_=54+3*im.row*im.col+char_pad*im.col;
	fwrite(&aux_,sizeof(unsigned int),1,bmp_file);
	aux_=0;
	fwrite(&aux_,sizeof(unsigned int),1,bmp_file);
	aux_=54;
	fwrite(&aux_,sizeof(unsigned int),1,bmp_file);
	aux_=40;
	fwrite(&aux_,sizeof(unsigned int),1,bmp_file);
	aux_=im.row;
	fwrite(&aux_,sizeof(unsigned int),1,bmp_file);
	aux_=im.col;
	fwrite(&aux_,sizeof(unsigned int),1,bmp_file);
	aux=1;
	fwrite(&aux,sizeof(unsigned short),1,bmp_file);
	aux=24;
	fwrite(&aux,sizeof(unsigned short),1,bmp_file);
	for (aux_=i=0;i<6;i++)
		fwrite(&aux_,sizeof(unsigned int),1,bmp_file);
	for (i=j=k=g=0;i<im.col;)	{
		while ((k<BUFFER_SIZE-3) && (i<im.col)) {
			if (j==im.row) {
				if (g++==char_pad) {
					i++;
					j=g=0;
				} else
					buffer[k++]=0;
			} else {
				*((unsigned int *)(&buffer[k]))=
					im.val[(im.col-i-1)*im.row+j++];
				k+=3;
			}
		}
		fwrite(buffer,sizeof(char),k,bmp_file);
		k=0;
	}
	fclose(bmp_file);
	return 1;
}

bmp_img conv2(bmp_img A, bmp_img B, int pad) {
	double *r_a, *r_b, *r_c, *r_x, *r_r;
	fftw_plan p_a, p_b, p_c;
	unsigned int i, j, k;
	double scale;
	bmp_img r;

	if (pad) {
		r.row=A.row+B.row-1;
		r.col=A.col+B.col-1;
	} else {
		r.row=A.row;
		r.col=A.col;
	}
	scale=1.0/(r.row*r.col);
	r_a=(double *)fftw_malloc(sizeof(double)*2*r.col*(r.row/2+1));
	r_b=(double *)fftw_malloc(sizeof(double)*2*r.col*(r.row/2+1));
    r_c=(double *)fftw_malloc(sizeof(double)*r.row*r.col);
    r_x=(double *)fftw_malloc(sizeof(double)*2*r.col*(r.row/2+1));
	for (i=0;i<A.col;i++)
		for (j=0;j<A.row;j++)
			r_a[i*2*(r.row/2+1)+j]=A.val[i*A.row+j];
	for (i=A.col;i<r.col;i++)
		for (j=0;j<r.row;j++)
			r_a[i*2*(r.row/2+1)+j]=0;
	for (i=0;i<r.col;i++)
		for (j=A.row;j<r.row;j++)
			r_a[i*2*(r.row/2+1)+j]=0;
	for (i=0;i<B.col;i++)
		for (j=0;j<B.row;j++)
			r_b[i*2*(r.row/2+1)+j]=B.val[i*B.row+j];
	for (i=B.col;i<r.col;i++)
		for (j=0;j<r.row;j++)
			r_b[i*2*(r.row/2+1)+j]=0;
	for (i=0;i<r.col;i++)
		for (j=B.row;j<r.row;j++)
			r_b[i*2*(r.row/2+1)+j]=0;
	p_a=fftw_plan_dft_r2c_2d(r.col,r.row,r_a,(fftw_complex *)r_a,FFTW_ESTIMATE);
	p_b=fftw_plan_dft_r2c_2d(r.col,r.row,r_b,(fftw_complex *)r_b,FFTW_ESTIMATE);
	fftw_execute(p_a);
	fftw_execute(p_b);
	fftw_destroy_plan(p_a);
	fftw_destroy_plan(p_b);
	for (i=0;i<r.col;i++)
		for (j=0;j<(r.row/2+1);j++)
		{
			k=(i*(r.row/2+1)+j)*2;
			r_x[k]=(r_a[k]*r_b[k]-r_a[k+1]*r_b[k+1])*scale;
			r_x[k+1]=(r_a[k]*r_b[k+1]+r_a[k+1]*r_b[k])*scale;
		}
	fftw_free(r_a);
	fftw_free(r_b);
	p_c=fftw_plan_dft_c2r_2d(r.col,r.row,(fftw_complex *)r_x,r_c,FFTW_ESTIMATE);
	fftw_execute(p_c);
	fftw_destroy_plan(p_c);
	fftw_free(r_x);
	if (pad)
	{
		r_r=(double *)malloc(sizeof(double)*A.row*A.col);
		for (i=0;i<A.col;i++)
			for (j=0;j<A.row;j++)
				r_r[i*A.row+j]=r_c[(i+((B.col-1)/2))*r.row+j+((B.row-1)/2)];
		fftw_free(r_c);
		r.val=r_r;
	}
	else
		r.val=r_c;
	r.row=A.row;
	r.col=A.col;
	return r;
}

bmp_img get_patch(int s, bmp_img im, aff_pt apt) {
    double ks, r, r_, *U, R[4], T[4];
    bmp_img ker, patch, d;
    int i, j;
    pt c, m, m_, l;

    r_=R(apt.si,0);
    ks=r_/s;
    U=apt.U;
    c.x=apt.x;
    c.y=apt.y;
    r=r_+ceil(s/r_); // +1 for derivative
    l.x=R2D(s+1); // +1 for derivative
    l.y=l.x;
    m.x=U[0]*-r+U[1]*-r;
    m.y=U[2]*-r+U[3]*-r;
    for (i=-1;i<2;i+=2)
        for (j=-1;j<2;j+=2) {
            m_.x=U[0]*i*r+U[1]*j*r;
            m_.y=U[2]*i*r+U[3]*j*r;
            if (m_.x>m.x) m.x=m_.x;
            if (m_.y>m.y) m.y=m_.y;
        }
    m.x=ceil(m.x);
    m.y=ceil(m.y);
    patch.row=R2D(m.x);
    patch.col=R2D(m.y);
    if (ks>1) { // smooth with a gaussian to avoid aliasing
        ker=gauss_ker2_cov(ks,U);
        patch.row+=ker.row-1;
        patch.col+=ker.col-1;
        m_.x=D2R(patch.row);
        m_.y=D2R(patch.col);
    } else {
        m_.x=m.x;
        m_.y=m.y;
    }
    patch.val=(double *)calloc(patch.row*patch.col,sizeof(double));
    for (i=-m_.y;i<=m_.y;i++)
        for (j=-m_.x;j<=m_.x;j++)
            patch.val[(int)((i+m_.y)*patch.row+(j+m_.x))]=
            	(round(i+c.y)>=0) && (round(i+c.y)<im.col) &&
				(round(j+c.x)>=0) && (round(j+c.x)<im.row)?
				im.val[(int)(round(i+c.y)*im.row+round(j+c.x))]:0;
    if (ks>1) {
        d=conv2(patch,ker,NO_PAD); // convolution
        free(patch.val);
        patch.val=d.val;
        free(ker.val);
    }
    R[0]=ks;
    R[1]=0;
    R[2]=0;
    R[3]=ks;
    prod2x2(U,R,T);
    d=patch_trans(patch,l,T);
    free(patch.val);
    patch_norm(d); // normalize pixel values
    return d;
}

bmp_img patch_trans(bmp_img im, pt l, double U[]) {
    int i, j, xx_, xy_, yx_, yy_;
    double n1, n2, v_xx, v_xy, v_yx, v_yy;;
    bmp_img p;
    pt r, r_, m;

    r.x=D2R(l.x);
    r.y=D2R(l.y);
    r_.x=D2R(im.row);
    r_.y=D2R(im.col);
    p.row=l.x;
    p.col=l.y;
    p.val=(double *)malloc(sizeof(double)*l.x*l.y);
    for (i=0;i<l.y;i++)
        for (j=0;j<l.x;j++) {
            m.x=U[0]*(j-r.x)+U[1]*(i-r.y)+r_.x;
            m.y=U[2]*(j-r.x)+U[3]*(i-r.y)+r_.y;
            n1=1-m.y+floor(m.y);
            n2=m.y-floor(m.y);
            xx_=floor(m.y)*im.row+floor(m.x);
            xy_=(floor(m.y)+1)*im.row+floor(m.x);
            yx_=floor(m.y)*im.row+floor(m.x)+1;
            yy_=(floor(m.y)+1)*im.row+floor(m.x)+1;
            v_xx=(floor(m.y)>=0) && (floor(m.y)<im.col) &&
            	 (floor(m.x)>=0) && (floor(m.x)<im.row)?
            	 im.val[xx_]:0;
            v_xy=(floor(m.y)+1>=0) && (floor(m.y)+1<im.col) &&
            	 (floor(m.x)>=0) && (floor(m.x)<im.row)?
            	 im.val[xy_]:0;
            v_yx=(floor(m.y)>=0) && (floor(m.y)<im.col) &&
            	 (floor(m.x)+1>=0) && (floor(m.x)+1<im.row)?
            	 im.val[yx_]:0;
            v_yy=(floor(m.y)+1>=0) && (floor(m.y)+1<im.col) &&
            	 (floor(m.x)+1>=0) && (floor(m.x)+1<im.row)?
            	 im.val[yy_]:0;
            p.val[(int)(i*l.x+j)]=(1-m.x+floor(m.x))*
                				  (n1*v_xx+n2*v_xy)+
                                  (m.x-floor(m.x))*
                				  (n1*v_yx+n2*v_yy);
        }
    return p;
}

bmp_img make_disk(int r, int *a) {
    bmp_img disk;
    int i, j;

    *a=0;
    disk.row=R2D(r);
    disk.col=disk.row;
    disk.val=(double *)malloc(sizeof(double)*disk.row*disk.col);
    for (i=-r;i<=r;i++)
        for (j=-r;j<=r;j++)
            *a+=((disk.val[(i+r)*disk.row+j+r]=
                  (sqrt((double)(i*i+j*j))<=r)));
    return disk;
}

int patch_norm(bmp_img im) {
    bmp_img disk;
    int a, i, j;
    double m, mm;

	m=im.val[0];
	mm=m;
	for (i=0;i<(im.row*im.col);i++) {
			m=(im.val[i]<m)?im.val[i]:m;
			mm=(im.val[i]>mm)?im.val[i]:mm;
	}
	if (mm-m) {
		disk=make_disk(D2R(im.row),&a);
  		for (i=0;i<im.col;i++)
        	for (j=0;j<im.row;j++)
				im.val[i*im.row+j]=(disk.val[i*im.row+j])?
								   (im.val[i*im.row+j]-m)/(mm-m):0;
		free(disk.val);
	} else
		for (i=0;i<im.row*im.col;im.val[i++]=0);
	return 1;
}

char *get_basename(char *s) {
	int i, l, r;
	char *e;

	for (r=0;s[r];r++);
	for (l=r;(l>=0) && (s[l]!=SLASH);l--);
	for (r=l+1;(s[r]) && (s[r]!='.');r++);
	e=(char *)malloc(sizeof(char)*(r-l));
	for (i=0,l++;l<r;i++,l++)
		e[i]=s[l];
	e[i]='\0';
	return e;
}

char *get_filename(char *s) {
	int i, l, r;
	char *e;

	r=-1;
	for (i=0;s[i];i++) {
		if (s[i]=='.')
			r=i;
		if (s[i]==SLASH)
			r=-1;
	}
	l=(r>0)?r:i;
	e=(char *)malloc(sizeof(char)*(l+1));
	for (i=0;i<l;i++)
		e[i]=s[i];
	e[i]='\0';
	return e;
}

bmp_img gauss_ker2_cov(double s, double U[]) {
        double *k, aux1, aux2, U_[4], U2_[4], p, cf, s2;
        int i, j, l1, l2, r, c;
        bmp_img ker;

        inv2x2(U,U_);
        U2_[0]=U_[0]*U_[0]+U_[2]*U_[2];
        U2_[1]=U_[0]*U_[1]+U_[2]*U_[3];
        U2_[2]=U_[0]*U_[1]+U_[2]*U_[3];
        U2_[3]=U_[1]*U_[1]+U_[3]*U_[3];
        s2=s*s;
        p=-9.0*s2;
        l1=ceil(sqrt(4.0*U2_[3]*p/(4.0*U2_[1]*U2_[1]-4.0*U2_[0]*U2_[3])));
        l2=ceil(sqrt(4.0*U2_[0]*p/(4.0*U2_[1]*U2_[1]-4.0*U2_[0]*U2_[3])));
        r=R2D(l1);
        c=R2D(l2);
        k=(double *)malloc(sizeof(double)*r*c);
        cf=2.0*M_PI*s/(U2_[0]*U2_[3]-U2_[1]*U2_[2]);
        for (i=0,aux1=-l2;i<c;i++,aux1++)
            for (j=0,aux2=-l1;j<r;j++,aux2++)
            {
                k[i*r+j]=exp(-(U2_[0]*aux2*aux2/s2+
                                       U2_[1]*aux1*aux2/s2+
                                       U2_[2]*aux1*aux2/s2+
                                       U2_[3]*aux1*aux1/s2)/2.0)/cf;
            }
        ker.val=k;
        ker.row=r;
        ker.col=c;
        return ker;
}

bmp_img rgb2gray(rgb_img im, int mode) {
	int i, z;
	double *gray;
	bmp_img res;

	gray=malloc(sizeof(double)*im.row*im.col);
	z=im.row*im.col;
	switch (mode) {
		case MEAN:
			for (i=0;i<z;i++)
				gray[i]=(double)((im.val[i] & 0xFF)+
								 ((im.val[i]>>8) & 0xFF)+
								 ((im.val[i]>>16) & 0xFF))/(3.0*255.0);
			break;
		case CUBE_ROOT:
			for (i=0;i<z;i++)
				gray[i]=cbrt((im.val[i] & 0xFF)*
							 ((im.val[i]>>8) & 0xFF)*
							 ((im.val[i]>>16) & 0xFF))/255.0;
			break;
		case LUMINANCE:
			for (i=0;i<z;i++)
				gray[i]=(0.11*(im.val[i] & 0xFF)+
						 0.59*((im.val[i]>>8) & 0xFF)+
						 0.3*((im.val[i]>>16) & 0xFF))/255.0;
			break;
	}
	res.row=im.row;
	res.col=im.col;
	res.val=gray;
	return res;
}

dist_matrix load_dist_matrix(char *filename) {
	dist_matrix r;
	int i;
	FILE *file_idx;

	file_idx=fopen(filename,"rb");
	fread(&r.r,sizeof(int),1,file_idx);
	fread(&r.c,sizeof(int),1,file_idx);
	r.el=(float *)malloc(sizeof(float)*r.r*r.c);
	for (i=0;i<r.r*r.c;i++)
		fread(&r.el[i],sizeof(float),1,file_idx);
	fclose(file_idx);
	return r;
}

int save_dist_matrix(dist_matrix r, char *filename) {
	int i;
	FILE *file_idx;

	file_idx=fopen(filename,"wb");
	fwrite(&r.r,sizeof(int),1,file_idx);
	fwrite(&r.c,sizeof(int),1,file_idx);
	for (i=0;i<r.r*r.c;i++)
		fwrite(&r.el[i],sizeof(float),1,file_idx);
	fclose(file_idx);
	return 1;
}

int get_info(prg_opt opt) {
	rpt_struct rpt;
	matched_struct match;
	dist_matrix m;
	dvect t, t_;
	tictac start, stop;
//	double q;
	int i, j, a, b, c_match, t_match, k, tt, kk;

	rpt=repeatability(opt);
	if (opt.info.dload==TRUE)
		m=load_dist_matrix(opt.info.dload_matrix);
	else {
		tt=0;
		for (kk=0;kk<opt.desc.dir;kk++)
			tt+=opt.desc.check_dir[kk];
		fprintf(stdout,"\n-- Computing distance matrix --\n");
		start=gimme_clock();
		if (tt)
			m=compute_dist(opt);
		else 
			m=compute_dist_guess(opt);
		stop=gimme_clock();
		elapsed_time(start,stop,VERBOSE);
		if (opt.info.dsave)
			save_dist_matrix(m,opt.info.dsave_matrix);
	}
	match=matching(m);
	matched_vect_qsort(rpt.good_table,rpt.good_table.n);
	fprintf(stdout,"\nMax overlap error: %0.2f\n",rpt.overlap/100.0);
/*
	t.n=19;
	t.el=(double *)malloc(sizeof(double)*t.n);
	for (i=0,q=1;i<t.n;i++,q+=0.5)
		t.el[i]=ceil(q*q/100*match.nn.n);
*/
	if ((opt.info.nn_eln==1)&&(!opt.info.nn_el[0])) {
			t.n=match.nn.n;
			t.el=(double *)malloc(sizeof(double)*t.n);
			for (i=0;i<t.n;i++)
				t.el[i]=i;
	} else {
		t.n=opt.info.nn_eln;
		t.el=(double *)malloc(sizeof(double)*t.n);
		for (i=0;i<t.n;i++)
			t.el[i]=round(opt.info.nn_el[i]*match.nn.n);
	}
	t_.n=t.n;
	t_.el=(double *)malloc(sizeof(double)*t_.n);
	fprintf(stdout,"\n             NN match\n"	 	  	  	\
				     "*********************************\n"	\
				     " recall - 1-prec - #match - #tot\n\n");
	j=c_match=0;
	for (i=0;i<t.n;i++) {
		t_match=t.el[i]+1;
		t_.el[i]=(i>0)?t_.el[i-1]:0;
		for (;(j<match.nn.n)&&(j<=t.el[i]);j++) {
			if (rpt_bin_search(rpt,match.nn.el[j])) {
				c_match++;
				t_.el[i]=
					m.el[match.nn.el[j].y*m.r+match.nn.el[j].x];
			} else
				match.nn.el[j].index=-1*abs(match.nn.el[j].index);
		}
		fprintf(stdout," %5.5f - %5.5f - %4d - %d\n",
			(double)c_match/rpt.corr,(t_match>0)?
			1.0-(double)c_match/t_match:0,c_match,t_match);
	}
	fprintf(stdout,"\n         Similarity match\n"	   	  	\
				     "*********************************\n"	\
				     " recall - 1-prec - #match - #tot\n\n");
	for (i=0;i<t_.n;i++) {
		c_match=t_match=0;
		for (b=0;b<m.r*m.c;b++)
			t_match+=(m.el[b]<=t_.el[i]);
		for (a=0;a<rpt.good_table.n;a++)
			c_match+=(m.el[rpt.good_table.el[a].y*m.r+
						   rpt.good_table.el[a].x]<=t_.el[i]);
		fprintf(stdout," %5.5f - %5.5f - %4d - %d\n",
			(double)c_match/rpt.good_table.n,(t_match>0)?
			1.0-(double)c_match/t_match:0,c_match,t_match);
	}
	free(t_.el);
	free(t.el);
	fprintf(stdout,"\n              NNR match\n"	   	  	\
				     "*********************************\n"	\
				     " recall - 1-prec - #match - #tot\n\n");
	if ((opt.info.nnr_eln==1)&&(!opt.info.nnr_el[0])) {
		t.n=match.nnr.n;
		t.el=(double *)malloc(sizeof(double)*t.n);
		for (i=0;i<t.n;i++)
			t.el[i]=match.nnr.el[i].val;
	} else {
		t.n=opt.info.nnr_eln;
		t.el=opt.info.nnr_el;
	}
	j=c_match=t_match=0;
	for (i=0;i<t.n;i++) {
		for (;(j<match.nnr.n)&&(match.nnr.el[j].val>=t.el[i]);j++) {
			if (rpt_bin_search(rpt,match.nnr.el[j]))
				c_match++;
			else
				match.nnr.el[j].index=-1*abs(match.nnr.el[j].index);
			t_match++;
		}
		fprintf(stdout," %5.5f - %5.5f - %4d - %d\n",
			(double)c_match/rpt.corr,(t_match>0)?
			1.0-(double)c_match/t_match:0,c_match,t_match);
	}
	fprintf(stdout,"\n");
	if (opt.data.draw==TRUE)
		visual_match(opt,match,m,rpt);
	free(rpt.good_table.el);
	free(match.nn.el);
	free(match.nnr.el);
	free(m.el);
	return 1;
}

int gray2rgb(bmp_img im, rgb_img res, xy p, int t, unsigned int c) {
	int i, j, n;

	for (i=0;i<im.col;i++)
		for (j=0;j<im.row;j++) {
			n=255.0*im.val[i*im.row+j];
			res.val[(p.y*PATCH_SIZE+i)*res.row+(p.x*PATCH_SIZE+j)]=
													   n|(n<<8)|(n<<16);
	}
	if (t & LEFT)
		for (i=0;i<im.col;i++)
			res.val[(p.y*PATCH_SIZE+i)*res.row+(p.x*PATCH_SIZE)]=c;
	if (t & RIGHT)
		for (i=0;i<im.col;i++)
			res.val[(p.y*PATCH_SIZE+i)*res.row+
					(p.x*PATCH_SIZE+im.row-1)]=c;
	if (t & EDGE) {
		for (j=0;j<im.row;j++)
			res.val[(p.y*PATCH_SIZE)*res.row+(p.x*PATCH_SIZE+j)]=c;
		for (j=0;j<im.row;j++)
			res.val[(p.y*PATCH_SIZE+im.col-1)*res.row+
					(p.x*PATCH_SIZE+j)]=c;
	}
	return 1;
}

patch_stat save_patch_init(int n, int m, char pr[], char suff[]) {
    int i, h;
    patch_stat s;

    h=PATCH_SIZE;
    s.im.row=IMG_ROW;
    s.im.col=IMG_COL;
    s.m=m;
    s.pr=s.im.row/(s.m*h);
    s.pc=s.im.col/h;
    s.pp=s.pr*s.pc*s.m;
    s.c=0;
    s.p=0;
    s.n=ceil((double)n/s.pp*s.m);
    s.ds=(s.n>0?log10(s.n):0)+1;
    for (i=0;pr[i];i++);
	s.s=strdup(pr);
	s.s_=strdup(suff);
    s.cs=(char *)malloc(sizeof(char)*(s.ds+2));
    s.cs[0]='_';
    for (i=1;i<=s.ds;i++)
		s.cs[i]='0';
	s.cs[i]='\0';
    s.im.val=(unsigned int *)calloc(s.im.row*s.im.col,
    								sizeof(unsigned int));
    return s;
}

patch_stat save_patch_next(patch_stat s, bmp_img p, int v,
						   unsigned int c) {
	int i, j, n;
	char *name;
	xy p_;

	if (s.p==s.pp) {
		s.p=0;
		name=append(s.s,s.cs,s.s_);
		save_bmp(name,s.im);
		free(name);
		free(s.im.val);
    	s.im.val=(unsigned int *)calloc(s.im.row*s.im.col,
    									sizeof(unsigned int));
		j=0;
		s.c++;
		for (i=0,n=s.c;n;i++,n/=10)
			s.cs[s.ds-i]=48+n%10;
	}
	p_.x=s.m*(s.p/s.m/s.pc)+s.p%s.m;
	p_.y=s.p/s.m%s.pc;
	gray2rgb(p,s.im,p_,v,c);
	s.p++;
	return s;
}

int save_patch_end(patch_stat s) {
	char *name;

	if (s.p>0) {
		name=append(s.s,s.cs,s.s_);
		save_bmp(name,s.im);
		free(name);
		free(s.im.val);
	}
	free(s.s);
	free(s.s_);
	free(s.cs);
	return 1;
}

int sort_index(pt3 *a, pt3 *b) {
	if (abs(a->index)==abs(b->index))
		return 0;
	return (abs(a->index)>abs(b->index))?1:-1;
}

int matched_vect_index_qsort(matched_vect in, int el) {
	qsort((void *)in.el,el,sizeof(pt3),
		  (int(*)(const void *, const void *))&sort_index);
	return 1;
}

int sort_val(pt3 *a, pt3 *b) {
	if (a->val==b->val)
		return 0;
	return (a->val>b->val)?1:-1;
}

int matched_vect_val_qsort(matched_vect in, int el) {
	qsort((void *)in.el,el,sizeof(pt3),
		  (int(*)(const void *, const void *))&sort_val);
	return 1;
}

int visual_match(prg_opt opt, matched_struct match, dist_matrix m,
				 rpt_struct rpt) {
	rgb_img img1, img2;
	bmp_img gray1, gray2, patch;
	fts feat1, feat2;
	aff_pt a_pt;
	int i, j, t;
	unsigned int c;
	char *aux_file1, *aux_file2, *filename;
	patch_stat p_stat;
	matched_vect aux;
	ivect dummy;
	int sa[PATCH_TH_NO], si, sc, st;

	aux_file1=get_filename(opt.data.data1);
	aux_file2=get_basename(opt.data.data2);
	filename=append(aux_file1,"_",aux_file2);
	free(aux_file1);
	free(aux_file2);
	feat1=mk_load(opt.data.data1,5);
	img1=load_bmp(opt.data.im1);
	gray1=rgb2gray(img1,LUMINANCE);
	free(img1.val);
	feat2=mk_load(opt.data.data2,5);
	img2=load_bmp(opt.data.im2);
	gray2=rgb2gray(img2,LUMINANCE);
	free(img2.val);
	c=BLUE;
	matched_vect_val_qsort(rpt.good_table,rpt.good_table.n);
 	p_stat=save_patch_init(rpt.good_table.n,2,filename,"_rpt.bmp");
	for (i=0;i<rpt.good_table.n;i++) {
		a_pt=from_mk_pt(&feat1.f[rpt.good_table.el[i].y*5]);
		patch=get_patch(D2R(PATCH_SIZE)-1,gray1,a_pt);
		t=EDGE_LEFT;
		p_stat=save_patch_next(p_stat,patch,t,c);
		free(patch.val);
		a_pt=from_mk_pt(&feat2.f[rpt.good_table.el[i].x*5]);
		patch=get_patch(D2R(PATCH_SIZE)-1,gray2,a_pt);
		t=EDGE_RIGHT;
		p_stat=save_patch_next(p_stat,patch,t,c);
		free(patch.val);
	}
	save_patch_end(p_stat);
	matched_vect_qsort(rpt.good_table,rpt.good_table.n);
	matched_vect_index_qsort(match.nn,match.nn.n);
 	p_stat=save_patch_init(match.nn.n,2,filename,"_nn.bmp");
	for (i=0;i<match.nn.n;i++) {
		a_pt=from_mk_pt(&feat1.f[match.nn.el[i].y*5]);
		patch=get_patch(D2R(PATCH_SIZE)-1,gray1,a_pt);
		if (match.nn.el[i].index>0) {
			t=EDGE_LEFT;
			c=BLUE;
		} else {
			t=LEFT;
			c=RED;
		}
		p_stat=save_patch_next(p_stat,patch,t,c);
		free(patch.val);
		a_pt=from_mk_pt(&feat2.f[match.nn.el[i].x*5]);
		patch=get_patch(D2R(PATCH_SIZE)-1,gray2,a_pt);
		if (match.nn.el[i].index>0) {
			t=EDGE_RIGHT;
			c=BLUE;
		} else {
			t=RIGHT;
			c=RED;
		}
		p_stat=save_patch_next(p_stat,patch,t,c);
		free(patch.val);
	}
	save_patch_end(p_stat);
	matched_vect_index_qsort(match.nnr,match.nnr.n);
 	p_stat=save_patch_init(match.nnr.n,2,filename,"_nnr.bmp");
	for (i=0;i<match.nnr.n;i++) {
		a_pt=from_mk_pt(&feat1.f[match.nnr.el[i].y*5]);
		patch=get_patch(D2R(PATCH_SIZE)-1,gray1,a_pt);
		if (match.nnr.el[i].index>0) {
			t=EDGE_LEFT;
			c=BLUE;
		} else {
			t=LEFT;
			c=RED;
		}
		p_stat=save_patch_next(p_stat,patch,t,c);
		free(patch.val);
		a_pt=from_mk_pt(&feat2.f[match.nnr.el[i].x*5]);
		patch=get_patch(D2R(PATCH_SIZE)-1,gray2,a_pt);
		if (match.nnr.el[i].index>0) {
			t=EDGE_RIGHT;
			c=BLUE;
		} else {
			t=RIGHT;
			c=RED;
		}
		p_stat=save_patch_next(p_stat,patch,t,c);
		free(patch.val);
	}
	save_patch_end(p_stat);
	dummy.n=rpt.good_table.n;
	dummy.el=(int *)malloc(sizeof(int)*dummy.n);
	for (i=0;i<rpt.good_table.n;i++)
		dummy.el[i]=rpt.good_table.el[i].y;
	i_qsort(dummy,dummy.n);
	for (si=0;si<PATCH_TH_NO;si++)
		sa[si]=0;
	p_stat=save_patch_init(m.c,PATCH_TH_NO+1,filename,"_th.bmp");
	aux.n=m.r;
	aux.el=(pt3 *)malloc(sizeof(pt3)*aux.n);
	for (i=0;i<m.c;i++) {
		for (j=0;j<m.r;j++) {
			aux.el[j].index=j;
			aux.el[j].val=m.el[i*m.r+j];
			aux.el[j].x=j;
			aux.el[j].y=i;
		}
		matched_vect_val_qsort(aux,aux.n);
		a_pt=from_mk_pt(&feat1.f[i*5]);
		patch=get_patch(D2R(PATCH_SIZE)-1,gray1,a_pt);
		c=i_bin_search(dummy,i)?YELLOW:GREEN;
		p_stat=save_patch_next(p_stat,patch,LEFT_RIGHT_EDGE,c);
		free(patch.val);
		for (j=0;j<PATCH_TH_NO;j++) {
			st=1;
			a_pt=from_mk_pt(&feat2.f[aux.el[j].index*5]);
			patch=get_patch(D2R(PATCH_SIZE)-1,gray2,a_pt);
			if (rpt_bin_search(rpt,aux.el[j])) {
				t=LEFT_RIGHT_EDGE;
				c=BLUE;
				if (st) {
					for (sc=j;sc<PATCH_TH_NO;sc++)
						sa[sc]++;
						st=0;
				}
			} else {
				t=LEFT_RIGHT_EDGE;
				c=RED;
			}
			p_stat=save_patch_next(p_stat,patch,t,c);
			free(patch.val);
		}
	}
	save_patch_end(p_stat);
	free(dummy.el);
	free(aux.el);
	free(filename);
	aux_file1=get_filename(opt.data.data2);
	aux_file2=get_basename(opt.data.data1);
	filename=append(aux_file1,"_",aux_file2);
	free(aux_file1);
	free(aux_file2);
	dummy.el=(int *)malloc(sizeof(int)*dummy.n);
	for (i=0;i<rpt.good_table.n;i++)
		dummy.el[i]=rpt.good_table.el[i].x;
	i_qsort(dummy,dummy.n);
	for (si=0;si<PATCH_TH_NO;si++)
		sa[si]=0;
	p_stat=save_patch_init(m.r,PATCH_TH_NO+1,filename,"_th.bmp");
	aux.n=m.c;
	aux.el=(pt3 *)malloc(sizeof(pt3)*aux.n);
	for (i=0;i<m.r;i++) {
		for (j=0;j<m.c;j++) {
			aux.el[j].index=j;
			aux.el[j].val=m.el[j*m.r+i];
			aux.el[j].x=i;
			aux.el[j].y=j;
		}
		matched_vect_val_qsort(aux,aux.n);
		a_pt=from_mk_pt(&feat2.f[i*5]);
		patch=get_patch(D2R(PATCH_SIZE)-1,gray2,a_pt);
		c=i_bin_search(dummy,i)?YELLOW:GREEN;
		p_stat=save_patch_next(p_stat,patch,LEFT_RIGHT_EDGE,c);
		free(patch.val);
		for (j=0;j<PATCH_TH_NO;j++) {
			st=1;
			a_pt=from_mk_pt(&feat1.f[aux.el[j].index*5]);
			patch=get_patch(D2R(PATCH_SIZE)-1,gray1,a_pt);
			if (rpt_bin_search(rpt,aux.el[j])) {
				t=LEFT_RIGHT_EDGE;
				c=BLUE;
				if (st) {
					for (sc=j;sc<PATCH_TH_NO;sc++)
						sa[sc]++;
						st=0;
				}
			} else {
				t=LEFT_RIGHT_EDGE;
				c=RED;
			}
			p_stat=save_patch_next(p_stat,patch,t,c);
			free(patch.val);
		}
	}
	save_patch_end(p_stat);
	free(gray1.val);
	free(gray2.val);
	free(aux.el);
	free(dummy.el);
	free(filename);
	free(feat1.f);
	free(feat2.f);
	return 1;
}

int bin_search(matched_vect r, pt3 p) {
	int i, j, k, t;

	i=0;
	j=r.n;
	if (!j)
		return -1;
	do {
		k=i+(j-i)/2;
		t=sort_el(&r.el[k],&p);
		if (!t)
			return k;
		if (t<0)
			i=k+1;
		else
			j=k;
	} while (i<j);
	return -1;
}

int rpt_bin_search(rpt_struct rpt, pt3 p) {
	int i, j, k, t;

	i=0;
	j=rpt.good_table.n;
	if (!j)
		return 0;
	do {
		k=i+(j-i)/2;
		t=sort_el(&rpt.good_table.el[k],&p);
		if (!t)
			return 1;
		if (t<0)
			i=k+1;
		else
			j=k;
	} while (i<j);
	return 0;
}

int i_bin_search(ivect r, int p) {
	int i, j, k, t;

	i=0;
	j=r.n;
	do {
		k=i+(j-i)/2;
		t=sort_i(&r.el[k],&p);
		if (!t)
			return 1;
		if (t<0)
			i=k+1;
		else
			j=k;
	} while (i<j);
	return 0;
}

int sort_i(int *a, int *b) {
	if (*a==*b)
		return 0;
	return (*a>*b)?1:-1;
}

int i_qsort(ivect in, int el) {
	qsort((void *)in.el,el,sizeof(int),
		  (int(*)(const void *, const void *))&sort_i);
	return 1;
}

prg_opt opts_digest(char *argv[], int argc) {
	config_t cfg;
	config_setting_t *aux_cfg;
	prg_opt opt;

#ifdef _WIN32
	char *aux, *aux_;
#else
	char *aux;
#endif

	const char *tmp;
	struct stat dummy;
	int i;

	INIT_DIST_VECT
	if (!((argc==1)||(argc==3)||(argc==5)||(argc==6)))
		EXIT_ERROR(__FILE__,__LINE__,usage)
	opt.data.im1=strdup(OPT_DATA_IM1);
	opt.data.im2=strdup(OPT_DATA_IM2);
	opt.data.data1=strdup(OPT_DATA_DATA1);
	opt.data.data2=strdup(OPT_DATA_DATA2);
	opt.data.h=strdup(OPT_DATA_H);
	opt.data.draw=OPT_DATA_DRAW;
	opt.info.overlap=OPT_INFO_OVERLAP;
	opt.info.distance=strdup(OPT_INFO_DISTANCE);
	opt.info.dsave=OPT_INFO_D_SAVE;
	opt.info.dsave_matrix=strdup(OPT_INFO_D_SAVE_MATRIX);
	opt.info.dload=OPT_INFO_D_LOAD;
	opt.info.dload_matrix=strdup(OPT_INFO_D_LOAD_MATRIX);
	opt.info.rsave=OPT_INFO_R_SAVE;
	opt.info.rsave_matrix=strdup(OPT_INFO_R_SAVE_MATRIX);
	opt.info.rload=OPT_INFO_R_LOAD;
	opt.info.rload_matrix=strdup(OPT_INFO_R_LOAD_MATRIX);
	opt.info.nn_eln=OPT_INFO_NN_ELN;
	opt.info.nnr_eln=OPT_INFO_NNR_ELN;
	opt.info.nn_el=OPT_INFO_NN_EL;
	opt.info.nnr_el=OPT_INFO_NNR_EL;
	opt.desc.shift=OPT_DESC_SHIFT;
	opt.desc.dir=OPT_DESC_DIR;
	opt.desc.rad=OPT_DESC_RAD;
	opt.desc.unique_center_bin=OPT_DESC_UNIQUE_CENTER_BIN;
	opt.desc.hist=OPT_DESC_HIST;
	opt.desc.check_dir=(int *)malloc(sizeof(int)*opt.desc.dir);
	for (i=0;i<opt.desc.dir;i++)
		opt.desc.check_dir[i]=1;				

#ifdef _WIN32
	aux_=strdup(argv[0]);
	if (aux_[strlen(aux_)-4]=='.')
		aux_[strlen(aux_)-4]='\0';
	aux=append(aux_,".ini",NULL);
	free(aux_);
#else
	aux=append(argv[0],".ini",NULL);
#endif

	config_init(&cfg);
	if (!stat(aux,&dummy)) {
		if (config_read_file(&cfg,aux)==CONFIG_TRUE) {
			if (config_lookup_string(&cfg,"data.im1",&tmp)!=CONFIG_FALSE) {
				free(opt.data.im1);
				opt.data.im1=strdup(tmp);
			}
			if (config_lookup_string(&cfg,"data.im2",&tmp)!=CONFIG_FALSE) {
				free(opt.data.im2);
				opt.data.im2=strdup(tmp);
			}
			if (config_lookup_string(&cfg,"data.data1",&tmp)!=CONFIG_FALSE) {
				free(opt.data.data1);
				opt.data.data1=strdup(tmp);
			}
			if (config_lookup_string(&cfg,"data.data2",&tmp)!=CONFIG_FALSE) {
				free(opt.data.data2);
				opt.data.data2=strdup(tmp);
			}
			if (config_lookup_string(&cfg,"data.h",&tmp)!=CONFIG_FALSE) {
				free(opt.data.h);
				opt.data.h=strdup(tmp);
			}
			config_lookup_bool(&cfg,"data.draw",&opt.data.draw);
			config_lookup_my_int(&cfg,"info.overlap",&opt.info.overlap);
			if (config_lookup_string(&cfg,"info.distance",&tmp)!=CONFIG_FALSE) {
				free(opt.info.distance);
				opt.info.distance=strdup(tmp);
			}
			config_lookup_bool(&cfg,"info.load_d",&opt.info.dload);
			if (config_lookup_string(&cfg,"info.load_d_matrix",&tmp)!=CONFIG_FALSE) {
				free(opt.info.dload_matrix);
				opt.info.dload_matrix=strdup(tmp);
			}
			config_lookup_bool(&cfg,"info.save_d",&opt.info.dsave);
			if (config_lookup_string(&cfg,"info.save_d_matrix",&tmp)!=CONFIG_FALSE) {
				free(opt.info.dsave_matrix);
				opt.info.dsave_matrix=strdup(tmp);
			}
			config_lookup_bool(&cfg,"info.load_r",&opt.info.rload);
			if (config_lookup_string(&cfg,"info.load_r_matrix",&tmp)!=CONFIG_FALSE) {
				free(opt.info.rload_matrix);
				opt.info.rload_matrix=strdup(tmp);
			}
			config_lookup_bool(&cfg,"info.save_r",&opt.info.rsave);
			if (config_lookup_string(&cfg,"info.save_r_matrix",&tmp)!=CONFIG_FALSE) {
				free(opt.info.rsave_matrix);
				opt.info.rsave_matrix=strdup(tmp);
			}
			if ((aux_cfg=config_lookup(&cfg,"info.nn_el"))!=NULL)
				if (opt.info.nn_eln=config_setting_length(aux_cfg)) {
					opt.info.nn_el=(double *)malloc(sizeof(double)*opt.info.nn_eln);
					for(i=0;i<opt.info.nn_eln;i++)
						opt.info.nn_el[i]=config_setting_get_float_elem(aux_cfg,i);
				}
			if ((aux_cfg=config_lookup(&cfg,"info.nnr_el"))!=NULL)
				if (opt.info.nnr_eln=config_setting_length(aux_cfg)) {
					opt.info.nnr_el=(double *)malloc(sizeof(double)*opt.info.nnr_eln);
					for(i=0;i<opt.info.nnr_eln;i++)
						opt.info.nnr_el[i]=config_setting_get_float_elem(aux_cfg,i);
				}
			config_lookup_bool(&cfg,"desc.unique_center_bin",&opt.desc.unique_center_bin);
			config_lookup_bool(&cfg,"desc.shift",&opt.desc.shift);
			config_lookup_my_int(&cfg,"desc.dir",&opt.desc.dir);
			
			opt.desc.check_dir=(int *)malloc(sizeof(int)*opt.desc.dir);
			for (i=0;i<opt.desc.dir;i++)
				opt.desc.check_dir[i]=1;				
			if ((aux_cfg=config_lookup(&cfg,"desc.check_dir"))!=NULL)
				if (opt.desc.dir==config_setting_length(aux_cfg)) {
					for(i=0;i<opt.desc.dir;i++)
						opt.desc.check_dir[i]=config_setting_get_int_elem(aux_cfg,i);	
				} else
					printf("\n*** desc.check_dir is not used because its dimension\n" \
					       "*** is not consistent with desc.dir\n");							
			
			config_lookup_my_int(&cfg,"desc.rad",&opt.desc.rad);
			config_lookup_my_int(&cfg,"desc.hist",&opt.desc.hist);
		}
		else {
			tmp=config_error_text(&cfg);
			i=config_error_line(&cfg);
			printf("Config file %s error: %s at line %d\n",aux,tmp,i);
			EXIT_ERROR(__FILE__,__LINE__,"Bad config file!")
		}
	} else
		printf("Config file %s not found, using default configuration\n",aux);
	free(aux);
	config_destroy(&cfg);
    if ((opt.info.dist_n=multiplestrcmp(opt.info.distance,dists,DISTS,UPPERLOWER))<0)
		EXIT_ERROR(__FILE__,__LINE__,"Bad distance!")
	if (argc>1) {
		free(opt.data.im1);
		opt.data.im1=strdup(argv[1]);
		free(opt.data.im2);
		opt.data.im2=strdup(argv[2]);
	}
	if (argc>3) {
		free(opt.data.data1);
		opt.data.data1=strdup(argv[3]);
		free(opt.data.data2);
		opt.data.data2=strdup(argv[4]);
	}
	if (argc>5) {
		free(opt.data.h);
		opt.data.h=strdup(argv[5]);
	}
	if (stat(opt.data.im1,&dummy) ||
		stat(opt.data.im2,&dummy) ||
		stat(opt.data.data1,&dummy) ||
		stat(opt.data.data2,&dummy) ||
		stat(opt.data.h,&dummy) ||
		((opt.info.dload==TRUE) && stat(opt.info.dload_matrix,&dummy)) ||
		((opt.info.rload==TRUE) && stat(opt.info.rload_matrix,&dummy)))
		EXIT_ERROR(__FILE__,__LINE__,usage)
	return opt;
}

int config_lookup_my_float(const config_t *config, const char *path,double *value) {
	int tmp;
	config_setting_t *s;
	
	s=config_lookup(config,path);
	if(!s)
		return(CONFIG_FALSE);
	switch (config_setting_type(s)) {
		case CONFIG_TYPE_FLOAT:
			*value=config_setting_get_float(s);
			return CONFIG_TRUE;
			break;
		case CONFIG_TYPE_INT:
			tmp=config_setting_get_int(s);
			*value=tmp;
			return CONFIG_TRUE;
			break;
		default:
			return CONFIG_FALSE;
	}
}

int config_lookup_my_int(const config_t *config, const char *path, int *value) {
	config_setting_t *s;
	
	s=config_lookup(config,path);
	if ((!s) || (config_setting_type(s)!=CONFIG_TYPE_INT)) 
		return(CONFIG_FALSE);	
	*value=config_setting_get_int(s);
	return CONFIG_TRUE;	
}

int main(int argc, char *argv[]) {
	tictac start, stop;
	prg_opt opt;

	INFINITY_=(pow(1,0)/0.0);
	fprintf(stdout,"Precision/recall computation for sGLOH (but also SIFT).\n"\
				   "Code by Fabio Bellavia (fbellavia@unipa.it)\n"\
				   "[refer to: F. Bellavia, D. Tegolo, E. Trucco,\n"\
				   "\"Improving SIFT-based Descriptors Stability to Rotations\",\n"\
                   "ICPR 2010].\n"\
				   "Only for academic or other non-commercial purposes.\n");
	start=gimme_clock();
	opt=opts_digest(argv,argc);
	get_info(opt);
	stop=gimme_clock();
	elapsed_time(start,stop,VERBOSE);
	exit(EXIT_SUCCESS);
}
