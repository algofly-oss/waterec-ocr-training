// Fixed-point separable bicubic, matching Pillow's 8-bit intermediate rounding.
// Coefficient algorithm attribution: PILLOW_LICENSE and coefficients.py.
extern "C" __global__ void horizontal(const unsigned char* src,unsigned char* tmp,
 const int* bounds,const int* weights,int width,int height,int outwidth,int ksize) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i>=height*outwidth*3)return;
 int c=i%3,x=(i/3)%outwidth,y=i/(3*outwidth);
 int start=bounds[2*x],n=bounds[2*x+1],acc=1<<21;
 for(int k=0;k<n;k++)acc+=((int)src[(y*width+start+k)*3+c])*weights[x*ksize+k];
 int v=acc>>22;tmp[i]=(unsigned char)max(0,min(255,v));
}
extern "C" __global__ void vertical_pack(const unsigned char* tmp,float* patches,
 const int* bounds,const int* weights,int width,int height,int outheight,int ksize,int maxtokens) {
 int i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i>=maxtokens*48)return;
 int pw=width/4,ph=outheight/4,token=i/48;
 if(token>=pw*ph){patches[i]=0.f;return;}
 int q=i%48,c=q/16,iy=(q%16)/4,ix=q%4;
 int x=(token%pw)*4+ix,y=(token/pw)*4+iy;
 int start=bounds[2*y],n=bounds[2*y+1],acc=1<<21;
 for(int k=0;k<n;k++)acc+=((int)tmp[((start+k)*width+x)*3+c])*weights[y*ksize+k];
 int value=max(0,min(255,acc>>22));
 float v=__fdiv_rn((float)value,255.f);v=__fsub_rn(v,.5f);v=__fdiv_rn(v,.5f);patches[i]=v;
}
