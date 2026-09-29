// Decoder probe: float, half2, bfloat162 and 64-bit atomics and
// reductions, shared and generic. Every architecture.
#include <cuda_fp16.h>
#include <cuda_bf16.h>
__global__ void k(float* f, double* d, __half2* h, __nv_bfloat162* b, float* of, double* od, __half2* oh, __nv_bfloat162* ob, __half* hh, __nv_bfloat16* bb) {
  float v = threadIdx.x;
  atomicAdd(f, v); of[threadIdx.x] = atomicAdd(f + 1, v);
  atomicAdd(d, (double)v); od[threadIdx.x] = atomicAdd(d + 1, (double)v);
  atomicAdd(h, __floats2half2_rn(v, v)); oh[threadIdx.x] = atomicAdd(h + 1, __floats2half2_rn(v, v));
  atomicAdd(b, __floats2bfloat162_rn(v, v)); ob[threadIdx.x] = atomicAdd(b + 1, __floats2bfloat162_rn(v, v));
  atomicAdd(hh, __float2half(v)); atomicAdd(bb, __float2bfloat16(v));
  atomicMax((int*)f + 5, threadIdx.x); atomicMin((unsigned*)f + 6, threadIdx.x);
  atomicExch((unsigned long long*)d + 5, 7ull);
  __shared__ float s[32]; atomicAdd(&s[threadIdx.x % 32], v); of[threadIdx.x + 64] = s[0];
  float* gen = threadIdx.x > 5 ? f : s; atomicAdd(gen, v);
}
