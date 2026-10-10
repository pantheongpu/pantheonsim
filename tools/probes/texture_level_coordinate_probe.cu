// Round-5 card probe: which texel a point (or, with LINEAR=1, linear) fetch at a normalized x reads on a level of a mipmapped
// WxH float texture whose texels hold their column index + 1000 * level (WRAP=1: wrap addressing).
// usage: <W> <H> <in.bin: (x, lod) pairs> <out.bin>. It showed that a size that is not a power of two floors x to 21 fractional bits first.
// Card probe: which texel a point-sampled fetch at x reads. Level l of a WxH mipmapped float texture has value = column index + 1000 l.
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <cstdlib>
__global__ void k(cudaTextureObject_t t,const float* in,float* out,int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) out[i]=tex2DLod<float>(t,in[2*i],0.5f,in[2*i+1]); }
int main(int argc,char**argv){
  int W=atoi(argv[1]),H=atoi(argv[2]); const char* fin=argv[3]; const char* fout=argv[4];
  int levels=1; while((1<<levels)<=(W>H?W:H)) ++levels;
  cudaFree(0);
  cudaChannelFormatDesc d=cudaCreateChannelDesc<float>(); cudaMipmappedArray_t mip;
  cudaMallocMipmappedArray(&mip,&d,make_cudaExtent(W,H,0),levels);
  for(int l=0;l<levels;++l){ cudaArray_t a; cudaGetMipmappedArrayLevel(&a,mip,l); int lw=W>>l?W>>l:1, lh=H>>l?H>>l:1; std::vector<float> h(lw*lh); for(int y=0;y<lh;y++)for(int x=0;x<lw;x++) h[y*lw+x]=float(x)+1000.f*l; cudaMemcpy2DToArray(a,0,0,h.data(),lw*4,lw*4,lh,cudaMemcpyHostToDevice);}
  cudaResourceDesc rd={}; rd.resType=cudaResourceTypeMipmappedArray; rd.res.mipmap.mipmap=mip;
  cudaTextureDesc td={}; td.addressMode[0]=td.addressMode[1]=getenv("WRAP")?cudaAddressModeWrap:cudaAddressModeClamp; td.filterMode=getenv("LINEAR")?cudaFilterModeLinear:cudaFilterModePoint; td.mipmapFilterMode=cudaFilterModePoint; td.readMode=cudaReadModeElementType; td.normalizedCoords=1; td.maxMipmapLevelClamp=20;
  cudaTextureObject_t t; cudaCreateTextureObject(&t,&rd,&td,nullptr);
  FILE*f=fopen(fin,"rb"); fseek(f,0,SEEK_END); long sz=ftell(f); fseek(f,0,SEEK_SET); int n=sz/8; std::vector<float> in(2*n); if(fread(in.data(),4,2*n,f)!=(size_t)2*n) return 3; fclose(f);
  float *di,*dout; cudaMalloc(&di,n*8); cudaMalloc(&dout,n*4); cudaMemcpy(di,in.data(),n*8,cudaMemcpyHostToDevice);
  k<<<(n+127)/128,128>>>(t,di,dout,n); std::vector<float> out(n); cudaMemcpy(out.data(),dout,n*4,cudaMemcpyDeviceToHost);
  f=fopen(fout,"wb"); fwrite(out.data(),4,n,f); fclose(f);
}
