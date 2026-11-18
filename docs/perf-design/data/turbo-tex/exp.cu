// Native-CUDA experiments for the texture/surface card path (no simulator). Driver API for objects, runtime launch.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <vector>
#include <cstdint>
#define CK(x) do{ CUresult r_=(x); if(r_){const char*s; cuGetErrorName(r_,&s); printf("FAIL %s:%d %s -> %s\n",__FILE__,__LINE__,#x,s); } }while(0)

struct Handles { cudaTextureObject_t t; cudaSurfaceObject_t s; float* out; };
__global__ void k_tex(cudaTextureObject_t t, float* out, int w, int h) {
  int x = blockIdx.x*blockDim.x+threadIdx.x, y = blockIdx.y*blockDim.y+threadIdx.y;
  if (x<w && y<h) out[y*w+x] = tex2D<float>(t, x+0.5f, y+0.5f);
}
__global__ void k_tex_struct(Handles hs, int w, int h) {
  int x = blockIdx.x*blockDim.x+threadIdx.x, y = blockIdx.y*blockDim.y+threadIdx.y;
  if (x<w && y<h) hs.out[y*w+x] = tex2D<float>(hs.t, x+0.5f, y+0.5f);
}
__global__ void k_tex_mem(const Handles* hs, int w, int h) {   // handle read from global memory
  int x = blockIdx.x*blockDim.x+threadIdx.x, y = blockIdx.y*blockDim.y+threadIdx.y;
  if (x<w && y<h) hs->out[y*w+x] = tex2D<float>(hs->t, x+0.5f, y+0.5f);
}
__global__ void k_surf(cudaSurfaceObject_t s, int w, int h) {
  int x = blockIdx.x*blockDim.x+threadIdx.x, y = blockIdx.y*blockDim.y+threadIdx.y;
  if (x<w && y<h) surf2Dwrite((float)(y*w+x)*2.f, s, x*4, y);
}
__global__ void k_lin(cudaTextureObject_t t, float* out, int n) { int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) out[i]=tex1Dfetch<float>(t,i); }
__global__ void k_3d(cudaTextureObject_t t, float* out, int w,int h,int d) {
  int x=threadIdx.x,y=blockIdx.x,z=blockIdx.y; if(x<w) out[(z*h+y)*w+x]=tex3D<float>(t,x+0.5f,y+0.5f,z+0.5f); }
__global__ void k_layer(cudaTextureObject_t t, float* out, int w,int h,int L) {
  int x=threadIdx.x,y=blockIdx.x,l=blockIdx.y; if(x<w) out[(l*h+y)*w+x]=tex2DLayered<float>(t,x+0.5f,y+0.5f,l); }

int main() {
  cudaFree(0);
  CUcontext ctx; cuCtxGetCurrent(&ctx);
  const int W=16,H=8,D=4;
  // --- E1: handle values
  float* dlin; cudaMalloc(&dlin, W*H*4);
  std::vector<float> src(W*H*D); for (size_t i=0;i<src.size();++i) src[i]=(float)i;
  cudaMemcpy(dlin, src.data(), W*H*4, cudaMemcpyHostToDevice);
  auto mkres_lin=[&](){ CUDA_RESOURCE_DESC r{}; r.resType=CU_RESOURCE_TYPE_LINEAR; r.res.linear.devPtr=(CUdeviceptr)dlin; r.res.linear.format=CU_AD_FORMAT_FLOAT; r.res.linear.numChannels=1; r.res.linear.sizeInBytes=W*H*4; return r; };
  CUDA_TEXTURE_DESC td{}; td.addressMode[0]=td.addressMode[1]=td.addressMode[2]=CU_TR_ADDRESS_MODE_CLAMP; td.filterMode=CU_TR_FILTER_MODE_POINT; td.flags=CU_TRSF_READ_AS_INTEGER;
  CUtexObject tl[3]; for (int i=0;i<3;++i){ CUDA_RESOURCE_DESC r=mkres_lin(); CK(cuTexObjectCreate(&tl[i],&r,&td,nullptr)); }
  printf("E1 linear handles: %llx %llx %llx\n",(unsigned long long)tl[0],(unsigned long long)tl[1],(unsigned long long)tl[2]);
  CK(cuTexObjectDestroy(tl[1])); CUtexObject tn; { CUDA_RESOURCE_DESC r=mkres_lin(); CK(cuTexObjectCreate(&tn,&r,&td,nullptr)); }
  printf("E1 after destroy+create: %llx (reuse of freed %s)\n",(unsigned long long)tn, tn==tl[1]?"YES":"no");
  // 2D array
  CUDA_ARRAY3D_DESCRIPTOR ad{}; ad.Width=W; ad.Height=H; ad.Format=CU_AD_FORMAT_FLOAT; ad.NumChannels=1; ad.Flags=CUDA_ARRAY3D_SURFACE_LDST;
  CUarray arr; CK(cuArray3DCreate(&arr,&ad));
  // E3: linear (pitch=W*4) -> array via cuMemcpy2D device->array
  CUDA_MEMCPY2D m{}; m.srcMemoryType=CU_MEMORYTYPE_DEVICE; m.srcDevice=(CUdeviceptr)dlin; m.srcPitch=W*4; m.dstMemoryType=CU_MEMORYTYPE_ARRAY; m.dstArray=arr; m.WidthInBytes=W*4; m.Height=H;
  CK(cuMemcpy2D(&m));
  CUDA_RESOURCE_DESC ra{}; ra.resType=CU_RESOURCE_TYPE_ARRAY; ra.res.array.hArray=arr;
  CUtexObject ta; CUsurfObject sa; CK(cuTexObjectCreate(&ta,&ra,&td,nullptr)); CK(cuSurfObjectCreate(&sa,&ra));
  printf("E1 array tex %llx surf %llx\n",(unsigned long long)ta,(unsigned long long)sa);
  float* dout; cudaMalloc(&dout, W*H*D*4);
  dim3 b(8,8), g(2,1);
  // E2a: handle passed as param, run with real handle
  cudaMemset(dout,0,W*H*4); k_tex<<<g,b>>>(ta,dout,W,H); cudaDeviceSynchronize();
  std::vector<float> o(W*H*D); cudaMemcpy(o.data(),dout,W*H*4,cudaMemcpyDeviceToHost);
  int bad=0; for(int i=0;i<W*H;++i) bad+= o[i]!=src[i]; printf("E3 array read via device->array copy: %d bad of %d\n",bad,W*H);
  // E2b: patching: build the arg vector by hand with a wrong handle then substitute the word
  CUtexObject wrong = ta ^ 0x1234; // an invalid value
  void* args_bytes[4]; unsigned long long hv = (unsigned long long)wrong; float* po=dout; int w=W,h=H;
  // substitute by scanning the raw 8-byte param for the simulator handle, as the runtime would
  unsigned long long simh = 0x10000000001ull; unsigned long long p0 = simh; if (p0==simh) p0=(unsigned long long)ta;
  void* params[]={&p0,&po,&w,&h}; cudaMemset(dout,0,W*H*4);
  (void)args_bytes;(void)hv;
  // use runtime-API launch with substituted value (same bytes)
  k_tex<<<g,b>>>((cudaTextureObject_t)p0,dout,W,H); cudaDeviceSynchronize();
  cudaMemcpy(o.data(),dout,W*H*4,cudaMemcpyDeviceToHost); bad=0; for(int i=0;i<W*H;++i) bad+= o[i]!=src[i]; printf("E2 substituted-value launch: %d bad\n",bad);
  // E2c: struct param with handle + pointer
  Handles hs{ta,sa,dout}; cudaMemset(dout,0,W*H*4); k_tex_struct<<<g,b>>>(hs,W,H); cudaDeviceSynchronize();
  cudaMemcpy(o.data(),dout,W*H*4,cudaMemcpyDeviceToHost); bad=0; for(int i=0;i<W*H;++i) bad+= o[i]!=src[i]; printf("E2 struct-param handle: %d bad\n",bad);
  // E2d: handle only in global memory
  Handles* dh; cudaMalloc(&dh,sizeof hs); cudaMemcpy(dh,&hs,sizeof hs,cudaMemcpyHostToDevice); cudaMemset(dout,0,W*H*4);
  k_tex_mem<<<g,b>>>(dh,W,H); cudaDeviceSynchronize(); cudaMemcpy(o.data(),dout,W*H*4,cudaMemcpyDeviceToHost); bad=0; for(int i=0;i<W*H;++i) bad+= o[i]!=src[i];
  printf("E2 handle only in device memory: %d bad (native ok)\n",bad);
  // E2e: a bogus handle value
  cudaMemset(dout,0,W*H*4); k_tex<<<g,b>>>((cudaTextureObject_t)0x10000000001ull,dout,W,H); cudaError_t e=cudaDeviceSynchronize();
  cudaMemcpy(o.data(),dout,W*H*4,cudaMemcpyDeviceToHost);
  printf("E2 bogus handle 0x10000000001: %s; first outputs %g %g %g %g\n", cudaGetErrorString(e), o[0],o[1],o[2],o[3]); cudaGetLastError();
  // a destroyed handle, and a small integer that was never made
  cuTexObjectDestroy(tl[0]); cudaMemset(dout,0,W*H*4); k_tex<<<g,b>>>((cudaTextureObject_t)tl[0],dout,W,H); e=cudaDeviceSynchronize(); cudaMemcpy(o.data(),dout,W*H*4,cudaMemcpyDeviceToHost);
  printf("E2 destroyed handle %llx: %s; first outputs %g %g\n",(unsigned long long)tl[0],cudaGetErrorString(e),o[0],o[1]); cudaGetLastError();
  // E3b: surface write then array->linear copy
  k_surf<<<g,b>>>(sa,W,H); cudaDeviceSynchronize();
  float* dback; cudaMalloc(&dback,W*H*4);
  CUDA_MEMCPY2D m2{}; m2.srcMemoryType=CU_MEMORYTYPE_ARRAY; m2.srcArray=arr; m2.dstMemoryType=CU_MEMORYTYPE_DEVICE; m2.dstDevice=(CUdeviceptr)dback; m2.dstPitch=W*4; m2.WidthInBytes=W*4; m2.Height=H; CK(cuMemcpy2D(&m2));
  cudaMemcpy(o.data(),dback,W*H*4,cudaMemcpyDeviceToHost); bad=0; for(int i=0;i<W*H;++i) bad+= o[i]!=(float)i*2.f; printf("E3 surface write then array->linear: %d bad\n",bad);
  // E3c: 3D and layered arrays through cuMemcpy3D with dense linear source
  for (int kind=0; kind<3; ++kind) { // 0=3D,1=layered,2=cubemap(6 faces, W=H)
    CUDA_ARRAY3D_DESCRIPTOR a3{}; a3.Width=(kind==2?8:W); a3.Height=(kind==2?8:H); a3.Depth=(kind==2?6:D); a3.Format=CU_AD_FORMAT_FLOAT; a3.NumChannels=1;
    a3.Flags = kind==1?CUDA_ARRAY3D_LAYERED:kind==2?(CUDA_ARRAY3D_CUBEMAP):0;
    CUarray a; CUresult r=cuArray3DCreate(&a,&a3); if(r){printf("E3 kind %d create fail %d\n",kind,r);continue;}
    size_t n=a3.Width*a3.Height*a3.Depth; float* dl; cudaMalloc(&dl,n*4); std::vector<float> s2(n); for(size_t i=0;i<n;++i)s2[i]=(float)i; cudaMemcpy(dl,s2.data(),n*4,cudaMemcpyHostToDevice);
    CUDA_MEMCPY3D c{}; c.srcMemoryType=CU_MEMORYTYPE_DEVICE; c.srcDevice=(CUdeviceptr)dl; c.srcPitch=a3.Width*4; c.srcHeight=a3.Height; c.dstMemoryType=CU_MEMORYTYPE_ARRAY; c.dstArray=a; c.WidthInBytes=a3.Width*4; c.Height=a3.Height; c.Depth=a3.Depth; CK(cuMemcpy3D(&c));
    CUDA_RESOURCE_DESC r3{}; r3.resType=CU_RESOURCE_TYPE_ARRAY; r3.res.array.hArray=a; CUtexObject t3; CUDA_TEXTURE_DESC tdd=td; CK(cuTexObjectCreate(&t3,&r3,&tdd,nullptr));
    float* oo; cudaMalloc(&oo,n*4); cudaMemset(oo,0,n*4);
    if(kind==0) k_3d<<<dim3(H,D),W>>>(t3,oo,W,H,D); else if(kind==1) k_layer<<<dim3(H,D),W>>>(t3,oo,W,H,D);
    cudaError_t ee=cudaDeviceSynchronize(); cudaMemcpy(s2.data(),oo,n*4,cudaMemcpyDeviceToHost); int b3=0; if(kind<2) for(size_t i=0;i<n;++i) b3+= s2[i]!=(float)i;
    printf("E3 kind %d (%s) dense-linear->array read back by kernel: %d bad of %zu (%s)\n",kind,kind==0?"3D":kind==1?"layered":"cubemap",b3,n,cudaGetErrorString(ee));
    // reverse copy array->linear
    cudaMemset(dl,0,n*4); CUDA_MEMCPY3D c2{}; c2.srcMemoryType=CU_MEMORYTYPE_ARRAY; c2.srcArray=a; c2.dstMemoryType=CU_MEMORYTYPE_DEVICE; c2.dstDevice=(CUdeviceptr)dl; c2.dstPitch=a3.Width*4; c2.dstHeight=a3.Height; c2.WidthInBytes=a3.Width*4; c2.Height=a3.Height; c2.Depth=a3.Depth; CK(cuMemcpy3D(&c2));
    cudaMemcpy(s2.data(),dl,n*4,cudaMemcpyDeviceToHost); int b4=0; for(size_t i=0;i<n;++i) b4+= s2[i]!=(float)i; printf("E3 kind %d array->linear roundtrip: %d bad\n",kind,b4);
  }
  // E4: handle uniqueness across a second context? skip. Timing of per-launch array copy (4096^2 float)
  { const int N=4096; CUDA_ARRAY3D_DESCRIPTOR a4{}; a4.Width=N;a4.Height=N;a4.Format=CU_AD_FORMAT_FLOAT;a4.NumChannels=1; CUarray a; CK(cuArray3DCreate(&a,&a4));
    float* dl; cudaMalloc(&dl,(size_t)N*N*4); cudaMemset(dl,0,(size_t)N*N*4);
    CUDA_MEMCPY2D c{}; c.srcMemoryType=CU_MEMORYTYPE_DEVICE; c.srcDevice=(CUdeviceptr)dl; c.srcPitch=N*4; c.dstMemoryType=CU_MEMORYTYPE_ARRAY; c.dstArray=a; c.WidthInBytes=N*4; c.Height=N;
    cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1); CK(cuMemcpy2D(&c)); cudaEventRecord(e0); for(int i=0;i<10;++i) CK(cuMemcpy2D(&c)); cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms; cudaEventElapsedTime(&ms,e0,e1); printf("E4 device->array copy of 64 MiB: %.3f ms each (%.1f GB/s)\n",ms/10,64.0*1.048576/(ms/10));
  }
  return 0;
}
