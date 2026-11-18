// a fired assert/trap with no error checking at all: what does the process do? (host code does not check anything but prints after)
#include <cstdio>
#include <cassert>
#include <cuda_runtime.h>
__global__ void k(int* p){ p[threadIdx.x]=1; assert(threadIdx.x!=3); }
int main(){ int*d; cudaMalloc(&d,256); k<<<1,32>>>(d); cudaError_t e=cudaDeviceSynchronize(); printf("sync=%d\n",(int)e); int h[32]; e=cudaMemcpy(h,d,128,cudaMemcpyDeviceToHost); printf("copy=%d h3=%d\n",(int)e,h[3]); return 0; }
