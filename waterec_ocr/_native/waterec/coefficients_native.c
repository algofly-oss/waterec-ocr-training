/* Pillow-compatible full-image bicubic coefficient calculation.
 * Adapted mathematical algorithm: Pillow libImaging/Resample.c; PILLOW_LICENSE.
 */
#include <math.h>
#include <stdlib.h>
static double cubic(double x) {
 x=fabs(x);
 if(x<1.)return (1.5*x-2.5)*x*x+1.;
 if(x<2.)return (((x-5.)*x+8.)*x-4.)*-.5;
 return 0.;
}
int resize_coefficients(int insize,int outsize,int ksize,int* bounds,int* weights) {
 double scale=(double)insize/outsize,fs=scale>1.?scale:1.,support=2.*fs,inv=1./fs;
 double* tmp=(double*)malloc(ksize*sizeof(double));if(!tmp)return 0;
 for(int j=0;j<outsize;j++) {
  double center=(j+.5)*scale,total=0.;int lo=(int)(center-support+.5),hi=(int)(center+support+.5);
  if(lo<0)lo=0;if(hi>insize)hi=insize;
  bounds[2*j]=lo;bounds[2*j+1]=hi-lo;
  for(int i=0;i<hi-lo;i++){tmp[i]=cubic((i+lo-center+.5)*inv);total+=tmp[i];}
  for(int i=0;i<ksize;i++){
   double value=i<hi-lo?tmp[i]:0.;if(total!=0.)value/=total;
   weights[j*ksize+i]=(int)(value*4194304.+(value<0.?-.5:.5));
  }
 }
 free(tmp);return 1;
}
