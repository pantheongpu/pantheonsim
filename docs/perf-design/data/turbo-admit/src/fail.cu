// What happens on a real card when assert / __trap / __brkpt fires. mode: assert|trap|brkpt
#include <cstdio>
#include <cstring>
#include <cassert>
#include <cuda_runtime.h>
__global__ void k_assert(int* p, int fire){ p[threadIdx.x]=7; assert(!(fire && threadIdx.x==3)); p[threadIdx.x+32]=9; }
__global__ void k_trap(int* p, int fire){ p[threadIdx.x]=7; if(fire && threadIdx.x==3) __trap(); p[threadIdx.x+32]=9; }
__global__ void k_brk(int* p, int fire){ p[threadIdx.x]=7; if(fire && threadIdx.x==3) __brkpt(); p[threadIdx.x+32]=9; }
__global__ void k_ok(int* p){ p[threadIdx.x+64]=5; }
#define E(x) do{ cudaError_t e=(x); printf("%-34s -> %d (%s)\n", #x, (int)e, cudaGetErrorString(e)); fflush(stdout);}while(0)
int main(int argc,char**argv){
  const char* m=argc>1?argv[1]:"assert";
  int* d; cudaMalloc(&d,1024); cudaMemset(d,0,1024);
  int h[96]; 
  printf("mode %s\n", m);
  k_ok<<<1,32>>>(d); E(cudaDeviceSynchronize());
  E(cudaGetLastError());
  if(!strcmp(m,"assert")) k_assert<<<1,32>>>(d,1); else if(!strcmp(m,"trap")) k_trap<<<1,32>>>(d,1); else k_brk<<<1,32>>>(d,1);
  E(cudaGetLastError());
  E(cudaDeviceSynchronize());
  E(cudaGetLastError());
  E(cudaMemcpy(h,d,sizeof h,cudaMemcpyDeviceToHost));
  k_ok<<<1,32>>>(d); E(cudaGetLastError());
  E(cudaDeviceSynchronize());
  E(cudaMalloc((void**)&d,16));
  E(cudaDeviceReset());
  E(cudaMalloc(&d,1024)); k_ok<<<1,32>>>(d); E(cudaDeviceSynchronize());
  return 0;
}
