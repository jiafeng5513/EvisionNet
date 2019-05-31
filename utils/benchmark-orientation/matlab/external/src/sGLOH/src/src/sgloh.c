/* 	sGLOH descriptor by Fabio Bellavia (fbellavia@unipa.it),
 *  refer to: F. Bellavia, D. Tegolo, C. Valenti,
 *  "Keypoint descriptor matching with context-based orientation
 *  estimation", Image and Vision Computing 2014, and
 *  F. Bellavia, D. Tegolo, E. Trucco, "Improving SIFT-based Descriptors
 *  Stability to Rotations", ICPR 2010.
 *  Only for academic or other non-commercial purposes.
 */

// to compile: gcc sgloh.c -O3 -lconfig -lfftw3 -lm -o sgloh
// libconfig required

#include "sgloh.h"

char *strdup (const char *s) {
    char *d = (char*) malloc (strlen (s) + 1);   // Space for length plus nul
    if (d == NULL) return NULL;          // No memory
    strcpy (d,s);                        // Copy the characters
    return d;                            // Return the new string
}

// WARNING: inner direction is reversed !!!

bmp_img gauss_ker2(double s) {
    double *k, aux1, aux2, cf;
    int i,j, h, r;
    bmp_img ker;

    r=R(s,0);
    h=R2D(r);
    k=(double *)malloc(sizeof(double)*h*h);
    cf=(s*sqrt(2.0*M_PI))*(s*sqrt(2.0*M_PI));
    for (i=0,aux1=-r;i<h;i++,aux1++)
        for (j=0,aux2=-r;j<h;j++,aux2++)
            k[i*h+j]=exp(-((aux1/s)*(aux1/s))/2.0)*
                     exp(-((aux2/s)*(aux2/s))/2.0)/cf;
    ker.val=k;
    ker.row=h;
    ker.col=h;
    return ker;
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
        l1=ceil(sqrt(4*U2_[3]*p/(4*U2_[1]*U2_[1]-4*U2_[0]*U2_[3])));
        l2=ceil(sqrt(4*U2_[0]*p/(4*U2_[1]*U2_[1]-4*U2_[0]*U2_[3])));
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

i_table rosy_init(opt_struct opt) {
	i_table m;
	int k;
	double i, j, a, t, r, rd, s;

	s=2.0*M_PI/opt.desc.dir;
	rd=opt.desc.rad[opt.desc.radn-1]+1;
	m.row=R2D(rd);
	m.col=m.row;
	m.val=(int *)malloc(sizeof(double)*m.row*m.col);
	m.val_=(int *)malloc(sizeof(double)*m.row*m.col);
	for (i=-rd;i<=rd;i++)
		for (j=-rd;j<=rd;j++) {
			r=sqrt(i*i+j*j);
			if (r+1>rd) {
				m.val[(int)(i+rd)*m.row+(int)(j+rd)]=-1;
				m.val_[(int)(i+rd)*m.row+(int)(j+rd)]=-1;
				continue;
			}
			a=trunc((atan2(j,i)+M_PI)/s);
			if (a==opt.desc.dir)
				a=0;
			for (k=0;(k<opt.desc.radn)&&(opt.desc.rad[k]<r);k++);
			t=opt.desc.dir*k+a;
			if (opt.desc.unique_center_bin==TRUE)
				t=(t<opt.desc.dir)?0:t-opt.desc.dir+1.0;
			m.val[(int)(i+rd)*m.row+(int)(j+rd)]=t;
			if ((opt.desc.unique_center_bin==TRUE)&&(!t))
				a=0;
			m.val_[(int)(i+rd)*m.row+(int)(j+rd)]=a;
		}
	return m;
}

bmp_img rosy_patch(int s, double scale_factor, ort_struct ort, bmp_img im, aff_pt apt) {
    double ks, r, r_, *U, R[4], T[4], angle;
    bmp_img patch, d;
//	bmp_img ker;
    int i, j;
    pt c, m, m_, l;

	r_=ceil(scale_factor*3.0*apt.si);
	if (r_<1)
		r_=1;
    ks=r_/s;
    U=apt.U;
    c.x=apt.x;
    c.y=apt.y;
    r=r_+ceil(ks); // +1 for derivative
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
/*
    if (ks>1) { // smooth with a gaussian to avoid aliasing
        ker=gauss_ker2_cov(ks,U);
        patch.row+=ker.row-1;
        patch.col+=ker.col-1;
        m_.x=D2R(patch.row);
        m_.y=D2R(patch.col);
    } else {
*/
        m_.x=m.x;
        m_.y=m.y;
//  }
    patch.val=(double *)calloc(patch.row*patch.col,sizeof(double));
    for (i=-m_.y;i<=m_.y;i++)
        for (j=-m_.x;j<=m_.x;j++)
            patch.val[(int)((i+m_.y)*patch.row+(j+m_.x))]=
            	(round(i+c.y)>=0) && (round(i+c.y)<im.col) &&
				(round(j+c.x)>=0) && (round(j+c.x)<im.row)?
				im.val[(int)(round(i+c.y)*im.row+round(j+c.x))]:0;
/*
    if (ks>1) {
        d=conv2(patch,ker,NO_PAD); // convolution
        free(patch.val);
        patch.val=d.val;
        free(ker.val);
    }
*/
    R[0]=ks;
    R[1]=0;
    R[2]=0;
    R[3]=ks;
    prod2x2(U,R,T);
    d=rosy_trans(patch,l,T);
    if (ort.adjust==TRUE) { // adjust orientation
    	angle=rosy_angle(d,ort);
    	free(d.val);
    	R[0]=ks*cos(angle);
    	R[1]=ks*sin(angle);
   		R[2]=ks*(-sin(angle));
   		R[3]=ks*cos(angle);
   		prod2x2(U,R,T);
    	d=rosy_trans(patch,l,T);
	}
    free(patch.val);
    patch=rosy_norm(d); // normalize pixel values
    free(d.val);
    return patch;
}

bmp_img rosy_make_disk_(int r, double r_) {
    bmp_img disk;
    int i, j;

    disk.row=R2D(r);
    disk.col=disk.row;
    disk.val=(double *)malloc(sizeof(double)*disk.row*disk.col);
    for (i=-r;i<=r;i++)
        for (j=-r;j<=r;j++)
            disk.val[(i+r)*disk.row+j+r]=
				(double)(i*i+j*j)<(r_*r_);
    return disk;
}

bmp_img rosy_make_disk(int r, int *a) {
    bmp_img disk;
    int i, j;

    *a=0;
    disk.row=R2D(r);
    disk.col=disk.row;
    disk.val=(double *)malloc(sizeof(double)*disk.row*disk.col);
    for (i=-r;i<=r;i++)
        for (j=-r;j<=r;j++)
            *a+=(disk.val[(i+r)*disk.row+j+r]=
                  i*i+j*j<r*r);
    return disk;
}

bmp_img rosy_norm(bmp_img im) {
    bmp_img disk, res;
    int a, i, j;
    double m, s;

    disk=rosy_make_disk(D2R(im.row),&a);
    m=0;
    for (i=0;i<im.col;i++)
        for (j=0;j<im.row;j++)
            if (disk.val[i*im.row+j])
                m+=im.val[i*im.row+j];
    m/=a;
    s=0;
    for (i=0;i<im.col;i++)
        for (j=0;j<im.row;j++)
            if (disk.val[i*im.row+j])
                s+=(im.val[i*im.row+j]-m)*
                   (im.val[i*im.row+j]-m);
    s/=a-1;
    s=sqrt(s);
    res.val=(double *)malloc(sizeof(double)*im.row*im.col);
    res.row=im.row;
    res.col=im.col;
    for (i=0;i<res.row*res.col;i++)
        res.val[i]=s?(im.val[i]-m)/s:0;
    free(disk.val);
    return res;
}

int rosy_derivative(bmp_img img, bmp_img *dx, bmp_img *dy, int c) {
    int i,j;

    dx->val=(double *)malloc(sizeof(double)*img.row*img.col);
    dy->val=(double *)malloc(sizeof(double)*img.row*img.col);
    dx->row=img.row;
    dx->col=img.col;
    dy->row=img.row;
    dy->col=img.col;
    for (i=0;i<img.col;i++)
        for (j=c;j<img.row-1;j++)
            dx->val[i*dx->row+j]=img.val[i*img.row+j-c]-
                                 img.val[i*img.row+j+1];
    for (i=0;i<dx->col;i++)
        dx->val[(i+1)*dx->row-1]=0;
    for (i=c;i<dy->col-1;i++)
        for (j=0;j<dy->row;j++)
            dy->val[i*dy->row+j]=img.val[(i-c)*img.row+j]-
                                 img.val[(i+1)*img.row+j];
    for (j=0;j<dy->row;j++)
        dy->val[(dy->col-1)*dy->row+j]=0;
    if (c) {
        for (i=0;i<dx->col;i++)
            dx->val[i*dx->row]=0;
        for (j=0;j<dy->row;j++)
            dy->val[j]=0;
    }
    return 1;
}

double rosy_angle(bmp_img im, ort_struct opt) {
    bmp_img dx, dy, dm, dr, disk, ker, aux;
    int i, j, k, r;
    double *hist, sum_hist, angle, m_angle, v, dir_angle;

    disk=rosy_make_disk_(D2R(im.row),im.row/5.0);
    if (opt.smooth_std>0) {
    	ker=gauss_ker2(opt.smooth_std);
		aux=conv2(im,ker,PAD);
		rosy_derivative(aux,&dx,&dy,NO_CENTER);
		free(ker.val);
		free(aux.val);
	}
	else
		rosy_derivative(im,&dx,&dy,NO_CENTER);
	r=opt.prec;
    dm.val=(double *)calloc(im.row*im.col,sizeof(double));
    dm.row=im.row;
    dm.col=im.col;
    dr.val=(double *)calloc(im.row*im.col,sizeof(double));
    dr.row=im.row;
    dr.col=im.col;
    for (i=0;i<im.row*im.col;i++)
	  	if (disk.val[i]) {
        	dm.val[i]=sqrt(dx.val[i]*dx.val[i]+dy.val[i]*dy.val[i]);
        	dr.val[i]=atan2(dy.val[i],dx.val[i])+M_PI;
    	}
    free(dx.val);
    free(dy.val);
    if (opt.weight_win==TRUE) {
    	ker=gauss_ker2(D2R(im.row));
    	for (i=0;i<im.row*im.col;i++)
    		dm.val[i]*=ker.val[i];
    	free(ker.val);
    }
	angle=2.0*M_PI/(r*opt.dir);
	dir_angle=angle*r;
    if (opt.dir>0) {
		hist=(double *)calloc(r,sizeof(double));
		if (opt.hist_std>0) {
			for (i=0;i<r;i++) {
				for (j=0;j<im.row*im.col;j++) {
					v=fabs(dr.val[j]-dir_angle*
						trunc(dr.val[j]/(dir_angle))-angle*i);
					v=(v>dir_angle/2.0)?dir_angle-v:v;
					hist[i]+=dm.val[j]*exp(-v*v/
						(2.0*angle*opt.hist_std*angle*opt.hist_std));
				}
			}
		}
		else {
			for (i=0;i<im.row*im.col;i++)
					hist[(int)trunc(dr.val[i]/angle)%opt.prec]+=dm.val[i];
		}
		for (k=i=0;i<r;i++)
			if (hist[i]>=hist[k])
				k=i;
		m_angle=k*angle;
    	free(hist);
	}
	else {
		m_angle=sum_hist=0;
	    for (i=0;i<dr.row*dr.col;i++) {
	    	m_angle+=(dr.val[i]+M_PI)*dm.val[i];
    	    sum_hist+=dm.val[i];
		}
    	m_angle/=sum_hist;
  	}
    free(dr.val);
    free(dm.val);
    free(disk.val);
    return m_angle;
}

bmp_img rosy_trans(bmp_img im, pt l, double U[]) {
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

int prod2x2(double m1[], double m2[], double res[]) {
    res[0]=m1[0]*m2[0]+m1[1]*m2[2];
    res[1]=m1[0]*m2[1]+m1[1]*m2[3];
    res[2]=m1[2]*m2[0]+m1[3]*m2[2];
    res[3]=m1[2]*m2[1]+m1[3]*m2[3];
    return 1;
}

patch_stat rosy_save_patch_init(int r, int n, char pr[]) {
    int i, h, a;
    patch_stat s;
	char *aux;

	aux=rosy_basename(pr);
    h=R2D(r+1);
    s.im.row=IMG_ROW;
    s.im.col=IMG_COL;
    s.pr=s.im.row/h;
    s.pc=s.im.col/h;
    s.pp=s.pr*s.pc;
    s.c=0;
    s.p=0;
    s.n=ceil((double)n/s.pp);
    s.ds=(s.n>0?log10(s.n):0)+1;
    for (i=0;aux[i];i++);
	s.s=strdup(aux);
    s.cs=(char *)malloc(sizeof(char)*(s.ds+2));
    s.cs[0]='_';
    for (i=1;i<=s.ds;i++)
		s.cs[i]='0';
	s.cs[i]='\0';
    s.im.val=(double *)calloc(s.im.row*s.im.col,sizeof(double));
    s.disk=rosy_make_disk(D2R(h),&a);
	free(aux);
    return s;
}

patch_stat rosy_save_patch_next(patch_stat s, bmp_img p) {
	int i, j, n, offset;
	bmp_img p_;

	if (s.p==s.pp) {
		s.p=0;
		rosy_norm01_save(s.im,s.s,s.cs,"_patch.bmp");
		free(s.im.val);
		s.im.val=(double *)calloc(s.im.row*s.im.col,sizeof(double));
		j=0;
		s.c++;
		for (i=0,n=s.c;n;i++,n/=10)
			s.cs[s.ds-i]=48+n%10;
	}
	p_=rosy_norm01(p);
	for (i=0;i<p_.col;i++)
		for (j=0;j<p_.row;j++) {
			if (s.disk.val[i*p_.row+j]) {
				offset=(p_.col*(s.p/s.pr)+i)*(IMG_ROW)+
					   (p_.row*(s.p%s.pr)+j);
				s.im.val[offset]=p_.val[i*p_.row+j];
			}
		}
	free(p_.val);
	s.p++;
	return s;
}

int rosy_save_patch_end(patch_stat s) {
	if (s.p>0) {
		rosy_norm01_save(s.im,s.s,s.cs,"_patch.bmp");
		free(s.im.val);
	}
	free(s.disk.val);
	free(s.cs);
	free(s.s);
	return 1;
}

int rosy_go(opt_struct opt, bmp_img im, h_pt ft, i_table bin) {
    int i, k, r, block, ks, hist_dir;
	double sum;
	dvect f;
    bmp_img patch;
    patch_stat stat;
    mk_pt p;
	FILE *file_idx;

	hist_dir=opt.desc.hist_dir_cf*opt.desc.dir;
	r=opt.desc.rad[opt.desc.radn-1];
	if (opt.desc.unique_center_bin==TRUE)
		block=hist_dir*(opt.desc.dir*(opt.desc.radn-1)+1)+5;
	else
		block=hist_dir*opt.desc.dir*(opt.desc.radn)+5;
	f.el=(double *)malloc(sizeof(double)*block);


	f.n=block;
    //printf("size is %d %d %d %d   \n",f.n, hist_dir, opt.desc.dir, opt.desc.radn);

	if (opt.data.draw==TRUE)
    	stat=rosy_save_patch_init(r,ft.n,opt.data.img);

	file_idx=rosy_mk_save_init(opt,ft.n,block-5);

// #ifndef _WIN32
// 	printf("Current patch: ");
// 	ks=floor(log10(ft.n))+1;
// #endif


    for (k=0;k<ft.n;k++) {

// #ifndef _WIN32
// 		printf("%*d",ks,k);
// 		fflush(stdout);
// #endif
        patch=rosy_patch(r,opt.desc.scale_factor,opt.ort,im,ft.aff[k]); // get patch
		if (opt.data.draw==TRUE)
        	stat=rosy_save_patch_next(stat,patch);
        p=rosy_to_mk_pt(ft.aff[k]); // obtain points in mk format
        for (i=0;i<5;i++)
        	f.el[i]=p.p[i];
		rosy_desc_el(patch,bin,opt.dir,hist_dir,opt.desc.hist_dir_cf,f); // compute descriptor
		sum=0;
		for (i=5;i<block;i++)
			sum+=f.el[i];
		if (sum)
			for (i=5;i<block;i++)
				f.el[i]/=sum;
		if (opt.data.quantizer)
			for (i=5;i<block;i++)
				f.el[i]=trunc(f.el[i]*opt.data.quantizer);
		rosy_mk_save_next(f,file_idx,opt.data.quantizer);
		free(patch.val);

// #ifndef _WIN32
// 		printf("%c[%dD",0x1b,ks);
// #endif

    }
	printf("\n");
	if (opt.data.draw==TRUE)
    	rosy_save_patch_end(stat);
	rosy_mk_save_end(f,file_idx);
    return 1;
}

int rosy_end(opt_struct opt, bmp_img im, h_pt ft, i_table bin) {

    
	if (im.val != NULL)free(im.val);
	if (ft.aff != NULL)free(ft.aff);
	if (bin.val != NULL)free(bin.val);
	if (bin.val_ != NULL)free(bin.val_);
	if (opt.data.img != NULL)free(opt.data.img);
	if (opt.data.in != NULL)free(opt.data.in);
	if (opt.data.out != NULL)free(opt.data.out);

 //    if (opt.desc.rad != NULL)
	// if (DESC_RAD!=opt.desc.rad)
	// 	free(opt.desc.rad);
	return 1;
}

bmp_img rosy_norm01(bmp_img im) {
	bmp_img res;
	int i;
	double m, mm;

	m=im.val[0];
	mm=m;
	res.val=(double *)malloc(sizeof(double)*im.row*im.col);
	res.row=im.row;
	res.col=im.col;
	for (i=0;i<(im.row*im.col);i++) {
			m=(im.val[i]<m)?im.val[i]:m;
			mm=(im.val[i]>mm)?im.val[i]:mm;
	}
	if (mm-m)
		for (i=0;i<res.row*res.col;i++)
			res.val[i]=(im.val[i]-m)/(mm-m);
	else
		for (i=0;i<res.row*res.col;res.val[i++]=1);
	return res;
}

int rosy_norm01_save(bmp_img im, char *a, char *b, char *c) {
	bmp_img aux;
	rgb_img im_;
	char *filename;

	aux=rosy_norm01(im);
	im_=rosy_gray2rgb(aux);
	filename=append(a,b,c);
	rosy_save_bmp(filename,im_);
	free(im_.val);
	free(aux.val);
	free(filename);
	return 1;
}

char *append(char *a, char *b, char *c) {
	char *res;
	int i, l;

	l=i=0;
	if (a)
		for (l=0;a[l];l++);
	if (b)
		for (i=0;b[i];i++,l++);
	if (c)
		for (i=0;c[i];i++,l++);
	res=(char *)malloc(sizeof(char)*++l);
	if (a)
		for (l=i=0;a[i];res[l++]=a[i++]);
	if (b)
		for (i=0;b[i];res[l++]=b[i++]);
	if (c)
		for (i=0;c[i];res[l++]=c[i++]);
	res[l]='\0';
	return res;
}

rgb_img rosy_gray2rgb(bmp_img im) {
	rgb_img im_;
	int i, n;

	im_.val=(unsigned int *)malloc(sizeof(unsigned int)*im.row*im.col);
	im_.row=im.row;
	im_.col=im.col;
	for (i=0;i<(im_.row*im_.col);i++) {
		n=255.0*im.val[i];
		im_.val[i]=n|(n<<8)|(n<<16);
	}
	return im_;
}

int rosy_save_bmp(char *filename, rgb_img im) {
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

int rosy_desc_el(bmp_img patch, i_table mask, dir_struct opt, int hist_dir, int hist_dir_cf, dvect q) {
	int i, j, ii, off_bin;
	double a;
	double v;
	bmp_img dx, dy, dm, da, ker, aux;

    if (opt.smooth_std>0) {
    	ker=gauss_ker2(opt.smooth_std);
		aux=conv2(patch,ker,PAD);
		rosy_derivative(aux,&dx,&dy,NO_CENTER);
		free(ker.val);
		free(aux.val);
	}
	else
		rosy_derivative(patch,&dx,&dy,NO_CENTER);
	dm.row=dx.row;
	dm.col=dx.col;
	dm.val=(double *)calloc(dm.row*dm.col,sizeof(double));
	da.row=dx.row;
	da.col=dx.col;
	da.val=(double *)calloc(da.row*da.col,sizeof(double));
    a=2.0*M_PI/hist_dir;
	for (i=5;i<q.n;i++)
		q.el[i]=0;
	for (i=0;i<dm.row*dm.col;i++) {
		dm.val[i]=sqrt(dx.val[i]*dx.val[i]+dy.val[i]*dy.val[i]);
		da.val[i]=atan2(dy.val[i],dx.val[i])+M_PI;
	}
    if (opt.weight_win==TRUE) {
		ker=gauss_ker2(D2R(dm.row));
    	for (i=0;i<dm.row*dm.col;i++)
    		dm.val[i]*=ker.val[i];
		free(ker.val);
    }
	if (opt.hist_std>0) {
		for (j=0;j<dm.row*dm.col;j++)
			if (mask.val[j]>-1) {
				off_bin=mask.val[j]*hist_dir;
				for (i=0;i<hist_dir;i++) {
					ii=(i+hist_dir-hist_dir_cf*mask.val_[j])%hist_dir;
					v=fabs(da.val[j]-a*ii);
					v=(v>M_PI)?2.0*M_PI-v:v;
					q.el[off_bin+i+5]+=dm.val[j]*
						exp(-v*v/(2.0*a*opt.hist_std*a*opt.hist_std));
				}
			}
	}
	else
		for (i=0;i<dm.row*dm.col;i++)
			if (mask.val[i]>-1) {
				off_bin=mask.val[i]*hist_dir;
				q.el[(int)(round(da.val[i]/a)+
					hist_dir+hist_dir_cf*mask.val_[i])%hist_dir+
					off_bin+5]+=dm.val[i];
			}
	free(dx.val);
	free(dy.val);
	free(dm.val);
	free(da.val);
	return 1;
}

#ifndef MAX
#define MAX(a,b) (a > b ? a : b)
#endif

aff_pt rosy_from_mk_pt(mk_pt in, double angle_d) {
	aff_pt r;
	double U[4], D[4], V[4], T[4];
	double S[4], invS[4];

	r.x=in.p[0];
	r.y=in.p[1];
	S[0]=in.p[2];
	S[1]=in.p[3];
	S[2]=in.p[3];
	S[3]=in.p[4];

	/* Invert S */
	inv2x2(S,invS);
	
    /* from VLFeat code */
	double a = (double)sqrt(invS[0]);
	double b = invS[1] / MAX(a, 1e-18);
	double c = (double)sqrt(MAX(invS[3]-(b*b),0.));
    /* A = [a, 0; b, c] */

    /* cos and sin  */
	double rad = angle_d / 180.0 * M_PI;
	double cos_val = cos(rad);
	double sin_val = sin(rad);		
    /* R = [cos, -sin; sin, cos] */

	/* Affine = A*R */
    U[0] = a*cos_val;
	U[1] = a*(-sin_val);
	U[2] = b*cos_val + c*sin_val;
	U[3] = b*(-sin_val) + c*cos_val;

	/* Scale Space Adaptation for sGLOH */
	diagonalize2x2(S,D,V);
	D[0]=1/(3.0*sqrt(D[0]));
	D[3]=1/(3.0*sqrt(D[3]));
	r.si=D[0]>D[3]?D[0]:D[3];
	double scaler = 1.0 / (3.0 * r.si);

	/* Apply the scaler  */
	for(int ii = 0; ii < 4; ++ii)
		r.U[ii] = scaler * U[ii];

	/* ORIGINAL */
	/* r.x=in.p[0]; */
	/* r.y=in.p[1]; */
	/* U[0]=in.p[2]; */
	/* U[1]=in.p[3]; */
	/* U[2]=in.p[3]; */
	/* U[3]=in.p[4]; */
	/* diagonalize2x2(U,D,V); */
	/* D[0]=1/(3.0*sqrt(D[0])); */
	/* D[3]=1/(3.0*sqrt(D[3])); */
	/* r.si=D[0]>D[3]?D[0]:D[3]; */
	/* D[0]/=r.si; */
	/* D[3]/=r.si; */
	/* inv2x2(V,T); */
	/* prod2x2(D,T,U); */
	/* prod2x2(V,U,r.U); */
	
	return r;
}

mk_pt rosy_to_mk_pt(aff_pt in) {
	mk_pt d;
	double U[4], D[4], V[4], T[4];
	double S[4], invS[4];

	d.p[0]=in.x;
	d.p[1]=in.y;

	double scaler = 1.0 / (3.0 * in.si);
	for (int ii = 0; ii < 4; ++ii)
		U[ii] = in.U[ii] / scaler;

	/* U*U' == invS */
	invS[0] = U[0]*U[0] + U[1]*U[1];
	invS[1] = U[0]*U[2] + U[1]*U[3];
	invS[2] = U[0]*U[2] + U[1]*U[3];
	invS[3] = U[2]*U[2] + U[3]*U[3];

	/* S = inv(invS) */
	inv2x2(invS,S);

	/* S = [a, b; b, c] */
	d.p[2] = S[0];
	d.p[3] = S[1];
	d.p[4] = S[3];
	
	/* ORIGINAL */
	/* d.p[0]=in.x; */
	/* d.p[1]=in.y; */
	/* U[0]=in.U[0]; */
	/* U[1]=in.U[1]; */
	/* U[2]=in.U[2]; */
	/* U[3]=in.U[3]; */
	/* diagonalize2x2(U,D,V); */
	/* D[0]=1/((3.0*in.si*D[0])*(3.0*in.si*D[0])); */
	/* D[3]=1/((3.0*in.si*D[3])*(3.0*in.si*D[3])); */
	/* inv2x2(V,T); */
	/* prod2x2(D,T,U); */
	/* prod2x2(V,U,T); */
	/* d.p[2]=T[0]; */
	/* d.p[3]=T[1]; */
	/* d.p[4]=T[3]; */
	
	return d;
}

int diagonalize2x2(double *m, double *D, double *R) {
	double tr, det, d;

	tr=m[0]+m[3];
	det=m[0]*m[3]-m[1]*m[2];
	D[0]=(tr-sqrt(tr*tr-4.0*det))/2.0;
	D[1]=0;
	D[2]=0;
	D[3]=(tr+sqrt(tr*tr-4.0*det))/2.0;
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

int inv2x2(double *m,double *res) {
	double det;

	det=m[0]*m[3]-m[1]*m[2];
	res[0]=m[3]/det;
	res[1]=-m[1]/det;
	res[2]=-m[2]/det;
	res[3]=m[0]/det;
	return 1;
}

FILE *rosy_mk_save_init(opt_struct opt, int m, int n) {
	FILE *file_idx;

	file_idx=fopen(opt.data.out,"w");
	//fprintf(file_idx,"%d %d\n",n,m);
    //printf("saving %s %d\n",opt.data.out,file_idx);
	return file_idx;
}

int rosy_mk_save_next(dvect f, FILE *file_idx, int quantizer) {
	int i;

	char *nq="%f";
	char *q="%.0f";
	char *s=NULL;
	
	if (quantizer)
		s=q;
	else
		s=nq;


	//the kp
    if (false)
    	for (i=0;i<5;i++)
    		fprintf(file_idx,"%f ",f.el[i]);

    //the descriptor
	for (i=5;i<f.n;i++) {
        //printf(s,f.el[i]);//remove me
		fprintf(file_idx,s,f.el[i]);
		if (i==(f.n-1))
			fprintf(file_idx,"\n");
		else
			fprintf(file_idx,"\t");
	}

    //fprintf(file_idx,"yop");
	return 1;
}

int rosy_mk_save_end(dvect f, FILE *file_idx) {
    //printf("saving %d now...!\n",file_idx);
	if (f.el != NULL) free(f.el);
	fclose(file_idx);
	return 1;
}

h_pt rosy_mk_load(char *filename) {
	FILE *file_idx;
	int i, j, d;
	double dummy;
	h_pt feat;
	mk_pt m;

	file_idx=fopen(filename,"r");
	fscanf(file_idx,"%d %d",&d,&feat.n);
	feat.aff=(aff_pt *)malloc(sizeof(aff_pt)*feat.n);
	for (i=0;i<feat.n;i++) {
		for (j=0;j<5;j++)
			fscanf(file_idx,"%lf",&m.p[j]);
		for (j=0;j<d;j++)
			fscanf(file_idx,"%lf",&dummy);
		feat.aff[i]=rosy_from_mk_pt(m,0); /* Iam Ignoring the
										   * orientation for
										   * compatibility with
										   * original implementation */
	}
	fclose(file_idx);
	return feat;
}

rgb_img rosy_load_bmp(char filename[]) {
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

bmp_img rosy_rgb2gray(rgb_img im, int mode) {
	int i, z;
	double *gray;
	bmp_img res;

	gray=(double *)malloc(sizeof(double)*im.row*im.col);
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

#ifdef _WIN32

/* stime elapsed_time(tictac start, tictac stop, int mode) { */
/* 	stime res; */
/* 	double clock_per_sec, h, m, s; */

/* 	clock_per_sec=(double)CLK_TCK; */
/* 	res.realt=(double)(stop.t-start.t)/clock_per_sec; */
/* 	if (mode==VERBOSE) { */
/* 		fprintf(stdout,"***** Elapsed  times *****\n"); */
/* 		h=trunc(res.realt/(60*60)); */
/* 		m=trunc((res.realt-h*60*60)/60); */
/* 		s=res.realt-h*60*60-m*60; */
/* 		fprintf(stdout,"  Real time: %.0f:%.0f:%0.2f\n",h,m,s); */
/* 	} */
/* 	return res; */
/* } */

tictac gimme_clock(void) {
	tictac in;

	in.t=clock();
	return in;
}

#else

/* stime elapsed_time(tictac start, tictac stop, int mode) { */
/* 	stime res; */
/* 	double clock_per_sec, h, m, s; */

/* 	clock_per_sec=(double)sysconf(_SC_CLK_TCK); */
/* 	res.realt=(double)(stop.t-start.t)/clock_per_sec; */
/* 	res.usert=(double)(stop.c.tms_utime-start.c.tms_utime)/ */
/* 		clock_per_sec; */
/* 	res.syst=(double)(stop.c.tms_stime-start.c.tms_stime)/ */
/* 		clock_per_sec; */
/* 	if (mode==VERBOSE) { */
/* 		fprintf(stdout,"***** Elapsed  time *****\n"); */
/* 		h=trunc(res.realt/(60.0*60.0)); */
/* 		m=trunc((res.realt-h*60.0*60.0)/60.0); */
/* 		s=res.realt-h*60.0*60.0-m*60.0; */
/* 		fprintf(stdout,"  Real time: %02.0f:%02.0f:%05.2f\n",h,m,s); */
/* 		h=trunc(res.usert/(60.0*60.0)); */
/* 		m=trunc((res.usert-h*60.0*60.0)/60.0); */
/* 		s=res.usert-h*60.0*60.0-m*60.0; */
/* 		fprintf(stdout,"  User time: %02.0f:%02.0f:%05.2f\n",h,m,s); */
/* 		h=trunc(res.syst/(60.0*60.0)); */
/* 		m=trunc((res.syst-h*60.0*60.0)/60.0); */
/* 		s=res.syst-h*60.0*60.0-m*60.0; */
/* 		fprintf(stdout,"System time: %02.0f:%02.0f:%05.2f\a\n",h,m,s); */
/* 	} */
/* 	return res; */
/* } */

tictac gimme_clock(void) {
	tictac in;

	in.t=times(&in.c);
	return in;
}

#endif

int main_original(int argc, char *argv[]) {
	rgb_img img;
	bmp_img gray;
	h_pt aff;
	i_table bin;
	opt_struct opt;
	tictac start, stop;

	fprintf(stdout,"sGLOH descriptor by Fabio Bellavia (fbellavia@unipa.it)\n"\
				   "[refer to: F. Bellavia, D. Tegolo, E. Trucco,\n"\
				   "\"Improving SIFT-based Descriptors Stability to Rotations\",\n"\
                   "ICPR 2010].\n"\
				   "Only for academic or other non-commercial purposes.\n");
	start=gimme_clock();
	opt=rosy_opts_digest(argc,argv);
	img=rosy_load_bmp(opt.data.img);
	gray=rosy_rgb2gray(img,LUMINANCE);
	free(img.val);
	aff=rosy_mk_load(opt.data.in);
	bin=rosy_init(opt);
	rosy_go(opt,gray,aff,bin);
	rosy_end(opt,gray,aff,bin);
	stop=gimme_clock();
	/* elapsed_time(start,stop,VERBOSE); */
	exit(EXIT_SUCCESS);
}

char *rosy_basename(char *s) {
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

opt_struct rosy_opts_digest(int argc, char *argv[]) {
	config_t cfg;
	config_setting_t *aux_cfg;
	opt_struct opt;

#ifdef _WIN32
	char *aux, *aux_;
#else
	char *aux;
#endif

	const char *tmp;
	struct stat dummy;
	int i;

	if (argc>4)
		EXIT_ERROR(__FILE__,__LINE__,rosy_usage)
	DEFAULT_OPTS

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
			if (config_lookup_string(&cfg,"data.img",&tmp)!=CONFIG_FALSE) {
				free(opt.data.img);
				opt.data.img=strdup(tmp);
			}
			if (config_lookup_string(&cfg,"data.in",&tmp)!=CONFIG_FALSE) {
				free(opt.data.in);
				opt.data.in=strdup(tmp);
			}
			if (config_lookup_string(&cfg,"data.out",&tmp)!=CONFIG_FALSE) {
				free(opt.data.out);
				opt.data.out=strdup(tmp);
			}
			config_lookup_bool(&cfg,"data.draw",&opt.data.draw);
			config_lookup_my_int(&cfg,"data.quantizer",&opt.data.quantizer);
			config_lookup_my_int(&cfg,"desc.dir",&opt.desc.dir);
			if ((aux_cfg=config_lookup(&cfg,"desc.rad"))!=NULL)
				if ((opt.desc.radn=config_setting_length(aux_cfg))) {
					opt.desc.rad=(int *)malloc(sizeof(int)*opt.desc.radn);
					for(i=0;i<opt.desc.radn;i++)
						opt.desc.rad[i]=config_setting_get_int_elem(aux_cfg,i);
				}
			config_lookup_my_float(&cfg,"desc.scale_factor",&opt.desc.scale_factor);
			config_lookup_bool(&cfg,"desc.unique_center_bin",&opt.desc.unique_center_bin);
			config_lookup_my_int(&cfg,"desc.hist_dir_cf",&opt.desc.hist_dir_cf);
			config_lookup_my_float(&cfg,"dir.hist_std",&opt.dir.hist_std);
			config_lookup_bool(&cfg,"dir.weight_win",&opt.dir.weight_win);
			config_lookup_my_float(&cfg,"dir.smooth_std",&opt.dir.smooth_std);
			config_lookup_bool(&cfg,"ort.adjust",&opt.ort.adjust);
			config_lookup_my_int(&cfg,"ort.prec",&opt.ort.prec);
			config_lookup_my_int(&cfg,"ort.dir",&opt.ort.dir);
			config_lookup_my_float(&cfg,"ort.hist_std",&opt.ort.hist_std);
			config_lookup_bool(&cfg,"ort.weight_win",&opt.ort.weight_win);
			config_lookup_my_float(&cfg,"ort.smooth_std",&opt.ort.smooth_std);
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
	if (argc>1) {
		free(opt.data.img);
		opt.data.img=strdup(argv[1]);
		free(opt.data.in);
		free(opt.data.out);
		opt.data.in=NULL;
		opt.data.out=NULL;
	}
	if (argc>2) {
		if (opt.data.in)
			free(opt.data.in);
		opt.data.in=strdup(argv[2]);
		if (opt.data.out)
			free(opt.data.out);
		opt.data.out=NULL;
	}
	if (argc>3) {
		free(opt.data.out);
		opt.data.out=strdup(argv[3]);
	}
	if (stat(opt.data.img,&dummy))
		EXIT_ERROR(__FILE__,__LINE__,rosy_usage)
	if (!opt.data.in) {
		aux=rosy_basename(opt.data.img);
		opt.data.in=append(aux,in_data,NULL);
		free(aux);
	}
	if (stat(opt.data.in,&dummy))
		EXIT_ERROR(__FILE__,__LINE__,rosy_usage)
	if (!opt.data.out) {
		aux=rosy_basename(opt.data.img);
		opt.data.out=append(aux,out_data,NULL);
		free(aux);
	}
	return opt;
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

