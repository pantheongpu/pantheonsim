// does an assert/trap in one warp end a kernel whose other warps wait (barrier, spin on a flag)? mode: barrier|spin
#include <cstdio>
#include <cstring>
#include <cassert>
#include <cuda_runtime.h>
__global__ void k_bar(int* p){ if (threadIdx.x==3) assert(false && "fired"); __syncthreads(); p[threadIdx.x]=1; }
__global__ void k_spin(volatile int* flag, int* p){ if (threadIdx.x==3) __trap(); if (threadIdx.x>=32) { while(*flag==0){} } p[threadIdx.x]=1; }
int main(int argc,char**argv){ int*d; cudaMalloc(&d,4096); cudaMemset(d,0,4096);
  if(!strcmp(argv[1],"barrier")) k_bar<<<1,64>>>(d); else k_spin<<<1,64>>>(d,d+512);
  cudaError_t e=cudaDeviceSynchronize(); printf("%s sync=%d (%s)\n",argv[1],(int)e,cudaGetErrorString(e)); return 0; }
