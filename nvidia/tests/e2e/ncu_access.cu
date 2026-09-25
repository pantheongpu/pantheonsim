// Kernels whose Nsight Compute memory metrics can be worked out by hand, one
// warp each: a coalesced float load (1 request, 4 sectors, every byte used), a
// stride-8 one (32 sectors, an eighth of each used), a float4 one (16 sectors,
// all used), and a shared store at stride 2 (one extra pass for bank conflicts).
#include <cstdio>
#include <cuda_runtime.h>

__global__ void coalesced(const float* a, float* b) { b[threadIdx.x] = a[threadIdx.x]; }
__global__ void strided(const float* a, float* b) { b[threadIdx.x] = a[threadIdx.x * 8]; }
__global__ void vector4(const float4* a, float4* b) { b[threadIdx.x] = a[threadIdx.x]; }
__global__ void banks(float* out) {
  __shared__ float sm[64];
  sm[threadIdx.x * 2] = threadIdx.x;       // stride 2: two lanes per bank, one extra pass
  __syncthreads();
  out[threadIdx.x] = sm[threadIdx.x];      // stride 1: conflict-free
}
int main() {
  float *a, *b; cudaMalloc(&a, 4096 * sizeof(float)); cudaMalloc(&b, 4096 * sizeof(float));
  coalesced<<<1, 32>>>(a, b);
  strided<<<1, 32>>>(a, b);
  vector4<<<1, 32>>>(reinterpret_cast<float4*>(a), reinterpret_cast<float4*>(b));
  banks<<<1, 32>>>(b);
  cudaDeviceSynchronize();
  printf("app done\n");
  return 0;
}
