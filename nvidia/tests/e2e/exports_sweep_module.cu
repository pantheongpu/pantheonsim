__device__ int gvar[8];
__device__ __managed__ int mvar = 5;
__device__ int dfun(int x) { return x + 1; }
extern "C" __global__ void k1(int* p) { p[threadIdx.x] = dfun(gvar[0]) + mvar; }
extern "C" __global__ void k2(float* p) { p[0] = 1.f; }
extern "C" __global__ void k3() {}
