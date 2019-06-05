/* 	Distance matrix, nnr and nn match computation for sGLOH
 *  by Fabio Bellavia (fbellavia@unipa.it),
 *  refer to: F. Bellavia, D. Tegolo, C. Valenti,
 *  "Keypoint descriptor matching with context-based orientation
 *  estimation", Image and Vision Computing 2014, and
 *  F. Bellavia, D. Tegolo, E. Trucco, "Improving SIFT-based Descriptors
 *  Stability to Rotations", ICPR 2010.
 *  Only for academic or other non-commercial purposes.
 */

// to compile: gcc get_match.c -O3 -lconfig -lm -o get_match
// libconfig required

#include "get_match.h"

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
	dvect aux, *dd1, *dd2, *pts1, *pts2;
	ivect h;
	double v, tm, v0, v1, v2, *b1, *b2;

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
	b1=desc_all(desc1,&pts1,&dd1);
	b2=desc_all(desc2,&pts2,&dd2);
	destroy_desc_all(pts1,desc1.n,NULL);
	destroy_desc_all(pts2,desc2.n,NULL);
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
	printf("Selected rotation to check: %d %d %d\n",k0,k1,k2);
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
	destroy_desc_all(dd1,desc1.n,b1);
	destroy_desc_all(dd2,desc2.n,b2);	
	desc_end(desc1);
	desc_end(desc2);
	return m;
}

dist_matrix pos_matrix(dvect *dd1, dvect *dd2, int rr, int cc) {
	int i, j;
	dist_matrix m;
	double t1, t2;
	
	m.r=cc;
	m.c=rr;
	m.el=(float *)malloc(sizeof(float)*m.r*m.c);
	for (i=0;i<m.c;i++)
		for (j=0;j<m.r;j++) {
			t1=dd1[i].el[0]-dd2[j].el[0];
			t2=dd1[i].el[1]-dd2[j].el[1];
			m.el[i*m.r+j]=sqrt(t1*t1+t2*t2);
		}
	return m;
} 

dist_matrix pos_mask(dvect *dd1, dvect *dd2, int rr, int cc,
	info_struct oinfo) {
	int i, j;
	dist_matrix m;
	double t1, t2, t, r, rx, ry;

	if (oinfo.rad_max>0) {
		r=oinfo.rad_max;
		rx=r;
		ry=r;
	} else {
		r=0;
		rx=(oinfo.rad_xmax>0)?oinfo.rad_xmax:INFINITY_;
		ry=(oinfo.rad_ymax>0)?oinfo.rad_ymax:INFINITY_;
	}
	m.r=cc;
	m.c=rr;
	m.el=(float *)malloc(sizeof(float)*m.r*m.c);
	for (i=0;i<m.c;i++)
		for (j=0;j<m.r;j++) 
			if (((t1=fabs(dd1[i].el[0]-dd2[j].el[0]))<rx)&&
				((t2=fabs(dd1[i].el[1]-dd2[j].el[1]))<ry)) { 
				t=sqrt(t1*t1+t2*t2);	
				if (!r||(t<r))
					m.el[i*m.r+j]=t;
				else
					m.el[i*m.r+j]=INFINITY_;
			} else
				m.el[i*m.r+j]=INFINITY_;
	return m;
} 

dist_matrix compute_dist(prg_opt opt) {
	finfo desc1, desc2;
	dist_matrix m, mask;
	int i, j, k, r, p;
	dvect aux, *dd1, *dd2, *pts1, *pts2;
	double v, *b1, *b2;
	tictac start, stop;

	r=opt.info.rad_max;
	desc1=desc_init(opt.data.data1);
	desc2=desc_init(opt.data.data2);
	m.r=desc2.n;
	m.c=desc1.n;
	m.el=(float *)malloc(sizeof(float)*m.r*m.c);
	for (i=0;i<m.r*m.c;i++)
		m.el[i]=INFINITY_;
	b1=desc_all(desc1,&pts1,&dd1);
	b2=desc_all(desc2,&pts2,&dd2);
	p=opt.info.rad_max||opt.info.rad_xmax||opt.info.rad_ymax;
	if (p) {
		fprintf(stdout,"\n***** Locality mask *****\n");
		start=gimme_clock();
		mask=pos_mask(pts1,pts2,desc1.n,desc2.n,opt.info);
		stop=gimme_clock();
		elapsed_time(start,stop,VERBOSE);
	}
	if (opt.desc.shift==FALSE) {
		for (i=0;i<m.c;i++) {
			desc1.desc.el=dd1[i].el;
			for (j=0;j<m.r;j++)
				if ((!p)||
					(p&&(!isinf(mask.el[i*m.r+j])))) {
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
					for (j=0;j<m.r;j++)
						if ((!p)||
							(p&&(!isinf(mask.el[i*m.r+j])))) {
							desc2.desc.el=dd2[j].el;
							v=opt.info.dist_vect[opt.info.dist_n]
								(aux,desc2.desc);
							m.el[i*m.r+j]=(v<m.el[i*m.r+j])?v:m.el[i*m.r+j];
						}	
				}
		}
		free(aux.el);
	}
	if (opt.info.rad_max)
		free(mask.el);
	destroy_desc_all(pts1,desc1.n,NULL);
	destroy_desc_all(pts2,desc2.n,NULL);
	destroy_desc_all(dd1,desc1.n,b1);
	destroy_desc_all(dd2,desc2.n,b2);	
	desc_end(desc1);
	desc_end(desc2);
	return m;
}

int dist_shift(dvect in, dvect out, int step, desc_struct d) {
	int i, j, n, k, m;

// a shift-index lookup table can be pre-computed to be
// more efficient... but I'm lazy 8-) 

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

double *desc_all(finfo data, dvect **pts, dvect **desc) {
	int i, k;
	double *bblock;
	int off_pts, off_desc;

	free(data.desc.el);
	*desc=(dvect *)malloc(sizeof(dvect)*data.n);
	*pts=(dvect *)malloc(sizeof(dvect)*data.n);
	bblock=(double *)malloc(sizeof(double)*data.n*(5+data.d));
	off_pts=0;
	off_desc=data.n*5;
	for (k=0;k<data.n;k++) {
		(*pts)[k].n=5;
		(*pts)[k].el=&(bblock[off_pts]);
		for (i=0;i<5;i++)
			fscanf(data.in,"%lf",&((*pts)[k].el[i]));
		off_pts+=5;
		(*desc)[k].n=data.d;
		(*desc)[k].el=&(bblock[off_desc]);
		for (i=0;i<data.d;i++)
			fscanf(data.in,"%lf",&((*desc)[k].el[i]));
		off_desc+=data.d;
	}
	return bblock;
}

int destroy_desc_all(dvect *d, int n, double *b) {
	int k;

//	for (k=0;k<n;k++)
//		free(d[k].el);
	if (b)
		free(b);
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

matched_struct matching_fast(dist_matrix md) {
	matched_struct res;
	int i, j, m, n;
	long int k;
	pt3 e;
	matched_vect tmp;
	double aux;
	int *r_i, *c_i;
	double *r_m, *c_m, *r_a, *c_a, *r_q, *c_q, *r_t, *c_t, v,
		rtm, ctm;
	int bc=0;

	r_m=(double *)calloc(md.r,sizeof(double));
	c_m=(double *)calloc(md.c,sizeof(double));
/*	
	for computing the standard deviation
	r_a=(double *)calloc(md.r,sizeof(double));
	c_a=(double *)calloc(md.c,sizeof(double));
	r_q=(double *)calloc(md.r,sizeof(double));
	c_q=(double *)calloc(md.c,sizeof(double));
	r_t=(double *)calloc(md.r,sizeof(double));
	c_t=(double *)calloc(md.c,sizeof(double));
*/
	for (i=0;i<md.c;i++)
		for (j=0;j<md.r;j++) {
			v=md.el[i*md.r+j];
			r_m[j]+=v;
			c_m[i]+=v;
/*		
			r_t[j]++;
			c_t[i]++;
			rtm=r_a[j];
			ctm=c_a[i];
			r_a[j]=rtm+((v-rtm)/(double)r_t[j]);
			c_a[i]=ctm+((v-ctm)/(double)c_t[i]);

			r_q[j]+=(v-rtm)*(v-r_a[j]);
			c_q[i]+=(v-ctm)*(v-c_a[i]);
*/
	}
	for (i=0;i<md.r;i++) {
		r_m[i]/=md.r;
//		r_q[i]=sqrt(r_q[i]/(md.r*md.c-1));
	}
	for (i=0;i<md.c;i++) {
		c_m[i]/=md.c;
//		c_q[i]=sqrt(c_q[i]/(md.r*md.c-1));
	}
		
	r_i=(int *)calloc(md.r,sizeof(int));
	c_i=(int *)calloc(md.c,sizeof(int));
	tmp.n=md.r*md.c;
	tmp.el=(pt3 *)malloc(tmp.n*sizeof(pt3));
	
	k=0;
	for (i=0;i<md.c;i++)
	   for (j=0;j<md.r;j++)
		  if ((md.el[i*md.r+j]<r_m[j])&&
			(md.el[i*md.r+j]<c_m[i])) {   
			tmp.el[k].val=md.el[i*md.r+j];
			tmp.el[k].x=j;
			tmp.el[k].y=i;
			k++;
		}
	fprintf(stdout,"\nfast matching --> %0.1f%% of matches used\n",
		(double)k/(md.r*md.c)*100);
	tmp.n=k;		
	tmp.el=realloc((void *)tmp.el,tmp.n*sizeof(pt3));
	
	free(r_m);
	free(c_m);
/*	
	free(r_a);
	free(c_a);
	free(r_q);
	free(c_q);
	free(r_t);
	free(c_t);
*/	
	matched_vect_val_qsort(tmp,tmp.n);
	m=(md.r<md.c)?md.r:md.c;
	res.nn=matched_vect_init(m);
	for (k=0,n=0;n<m;k++) {
	   if ((k==tmp.n)||(isinf(tmp.el[k].val)))
	      break;
	   if ((!(r_i[tmp.el[k].x])) && (!(c_i[tmp.el[k].y]))) {
       e.val=tmp.el[k].val;
       e.x=tmp.el[k].x;
       e.y=tmp.el[k].y;
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

matched_struct matching(dist_matrix md, info_struct oinfo) {
	matched_struct res;
	int i, j, m, n;
	long int k;
	pt3 e;
	matched_vect tmp;
	double aux;
	int *r_i, *c_i;

	r_i=(int *)calloc(md.r,sizeof(int));
	c_i=(int *)calloc(md.c,sizeof(int));
	tmp.n=md.r*md.c;
	tmp.el=(pt3 *)malloc(tmp.n*sizeof(pt3));
	
	k=0;
	for (i=0;i<md.c;i++)
	   for (j=0;j<md.r;j++) 
		  if (!isinf(md.el[i*md.r+j])) {
			tmp.el[k].val=md.el[i*md.r+j];
			tmp.el[k].x=j;
			tmp.el[k].y=i;
			k++;
		  }
	if (oinfo.rad_max||oinfo.rad_xmax||oinfo.rad_ymax)
		fprintf(stdout,"\nlocal matching --> %0.1f%% of matches used\n",
		(double)k/(md.r*md.c)*100);
	tmp.n=k;		
	tmp.el=realloc((void *)tmp.el,tmp.n*sizeof(pt3));
	
	matched_vect_val_qsort(tmp,tmp.n);
	m=(md.r<md.c)?md.r:md.c;
	res.nn=matched_vect_init(m);
	for (k=0,n=0;n<m;k++) {
	   if ((k==tmp.n)||isinf(tmp.el[k].val))
	      break;
	   if ((!(r_i[tmp.el[k].x])) && (!(c_i[tmp.el[k].y]))) {
       e.val=tmp.el[k].val;
       e.x=tmp.el[k].x;
       e.y=tmp.el[k].y;
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
	int i, j, m, n;
	pt3 e;
	double aux;

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
		res.nnr.el[i].val*=-1;
	}
	free(r.el);
	free(c.el);
	return res;
}
*/

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
	matched_struct match;
	dist_matrix m;
//	double q;
	int i, k, t;
	FILE *out_file;
	tictac start, stop;

	if (opt.info.load==TRUE)
		m=load_dist_matrix(opt.info.load_matrix);
	else {
		t=0;
		for (k=0;k<opt.desc.dir;k++)
			t+=opt.desc.check_dir[k];
		start=gimme_clock();
		if (t)
			m=compute_dist(opt);
		else {
			printf("Guess rotation distance computation\n");			
			m=compute_dist_guess(opt);
		}
		stop=gimme_clock();
		printf("\nDistance matrix computation time\n");
		elapsed_time(start,stop,VERBOSE);
		if (opt.info.save)
			save_dist_matrix(m,opt.info.save_matrix);
	}
	if (opt.info.fast&&!opt.info.rad_max&&
		!opt.info.rad_xmax&&!opt.info.rad_ymax)
		match=matching_fast(m);
	else
		match=matching(m,opt.info);		
	out_file=fopen(opt.data.out_nn,"w");
	fprintf(out_file,"%d\n",match.nn.n);
	for (i=0;i<match.nn.n;i++)
		fprintf(out_file,"%d %d %f\n",match.nn.el[i].x,
			match.nn.el[i].y,match.nn.el[i].val);
	fclose(out_file);
	out_file=fopen(opt.data.out_nnr,"w");
	fprintf(out_file,"%d\n",match.nnr.n);
	for (i=0;i<match.nnr.n;i++)
		fprintf(out_file,"%d %d %f\n",match.nnr.el[i].x,
			match.nnr.el[i].y,match.nnr.el[i].val);
	fclose(out_file);
	free(match.nn.el);
	free(match.nnr.el);
	free(m.el);
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

int config_lookup_my_float(const config_t *config, const char *path, double *value) {
	int tmp;
	const config_setting_t *s;
	
	s=config_lookup(config,path);
	if (!s)
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
	if (!((argc==1)||(argc==3)||(argc==5)))
		EXIT_ERROR(__FILE__,__LINE__,usage)
	opt.data.out_nn=strdup(OPT_DATA_OUT_NN);
	opt.data.out_nnr=strdup(OPT_DATA_OUT_NNR);
	opt.data.data1=strdup(OPT_DATA_DATA1);
	opt.data.data2=strdup(OPT_DATA_DATA2);
	opt.info.distance=strdup(OPT_INFO_DISTANCE);
	opt.info.fast=OPT_INFO_FAST;
	opt.info.rad_max=OPT_INFO_RAD_MAX;
	opt.info.rad_xmax=OPT_INFO_RAD_XMAX;
	opt.info.rad_ymax=OPT_INFO_RAD_YMAX;
	opt.info.save=OPT_INFO_SAVE;
	opt.info.save_matrix=strdup(OPT_INFO_SAVE_MATRIX);
	opt.info.load=OPT_INFO_LOAD;
	opt.info.load_matrix=strdup(OPT_INFO_LOAD_MATRIX);
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
			if (config_lookup_string(&cfg,"data.out_nn",&tmp)!=CONFIG_FALSE) {
				free(opt.data.out_nn);
				opt.data.out_nn=strdup(tmp);
			}
			if (config_lookup_string(&cfg,"data.out_nnr",&tmp)!=CONFIG_FALSE) {
				free(opt.data.out_nnr);
				opt.data.out_nnr=strdup(tmp);
			}
			if (config_lookup_string(&cfg,"data.data1",&tmp)!=CONFIG_FALSE) {
				free(opt.data.data1);
				opt.data.data1=strdup(tmp);
			}
			if (config_lookup_string(&cfg,"data.data2",&tmp)!=CONFIG_FALSE) {
				free(opt.data.data2);
				opt.data.data2=strdup(tmp);
			}
			if (config_lookup_string(&cfg,"info.distance",&tmp)!=CONFIG_FALSE) {
				free(opt.info.distance);
				opt.info.distance=strdup(tmp);
			}
			config_lookup_my_float(&cfg,"info.rad_max",&opt.info.rad_max);
			config_lookup_my_float(&cfg,"info.rad_x",&opt.info.rad_xmax);
			config_lookup_my_float(&cfg,"info.rad_y",&opt.info.rad_ymax);
			config_lookup_bool(&cfg,"info.fast",&opt.info.fast);
			config_lookup_bool(&cfg,"info.load",&opt.info.load);
			if (config_lookup_string(&cfg,"info.load_matrix",&tmp)!=CONFIG_FALSE) {
				free(opt.info.load_matrix);
				opt.info.load_matrix=strdup(tmp);
			}
			config_lookup_bool(&cfg,"info.save",&opt.info.save);
			if (config_lookup_string(&cfg,"info.save_matrix",&tmp)!=CONFIG_FALSE) {
				free(opt.info.save_matrix);
				opt.info.save_matrix=strdup(tmp);
			}
			config_lookup_bool(&cfg,"desc.unique_center_bin",&opt.desc.unique_center_bin);
			config_lookup_bool(&cfg,"desc.shift",&opt.desc.shift);
			config_lookup_my_int(&cfg,"desc.dir",&opt.desc.dir);

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
	config_destroy(&cfg);
	free(aux);
    if ((opt.info.dist_n=multiplestrcmp(opt.info.distance,dists,DISTS,UPPERLOWER))<0)
		EXIT_ERROR(__FILE__,__LINE__,"Bad distance!")
	if (argc>1) {
		free(opt.data.data1);
		opt.data.data1=strdup(argv[1]);
		free(opt.data.data2);
		opt.data.data2=strdup(argv[2]);
	}
	if (argc>3) {
		free(opt.data.out_nn);
		opt.data.out_nn=strdup(argv[3]);
		free(opt.data.out_nnr);
		opt.data.out_nnr=strdup(argv[4]);
	}
	if (stat(opt.data.data1,&dummy) ||
		stat(opt.data.data2,&dummy) ||
		((opt.info.load==TRUE) && stat(opt.info.load_matrix,&dummy)))
		EXIT_ERROR(__FILE__,__LINE__,usage)
	return opt;
}

int main(int argc, char *argv[]) {
	tictac start, stop;
	prg_opt opt;

	my_inf=(pow(1,0)/0.0);
	fprintf(stdout,"Distance matrix, nn and nnr matching for sGLOH\n"\
				   "by Fabio Bellavia (fbellavia@unipa.it)\n"\
				   "[refer to: F. Bellavia, D. Tegolo, E. Trucco,\n"\
				   "\"Improving SIFT-based Descriptors Stability to Rotations\",\n"\
                   "ICPR 2010].\n"\
				   "Only for academic or other non-commercial purposes.\n");
	start=gimme_clock();
	opt=opts_digest(argv,argc);
	get_info(opt);
	stop=gimme_clock();
	printf("\nTotal running time\n");
	elapsed_time(start,stop,VERBOSE);
	exit(EXIT_SUCCESS);
}
