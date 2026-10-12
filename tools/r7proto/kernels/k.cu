// Throwaway kernels for the r7-design feasibility prototype (not merged).
// Compiled to a cubin for sm_86 and run by the prototype harness through the
// real SASS engine and through the closure-compiled prototype.
#include <cstdint>

// vectoradd-like: memory bound, one load pair + add + store per thread.
extern "C" __global__ void vecadd(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

// ALU loop, f32: independent FFMA chains, uniform trip count.
extern "C" __global__ void ffma_loop(float* out, int iters, float a, float b) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  float x0 = t * 0.5f, x1 = t * 0.25f + 1.0f, x2 = t * 0.125f + 2.0f, x3 = t * 0.0625f + 3.0f;
#pragma unroll 1
  for (int i = 0; i < iters; ++i) {
    x0 = x0 * a + b;
    x1 = x1 * a + b;
    x2 = x2 * b + a;
    x3 = x3 * b + a;
    x0 = x0 * a + x1;
    x1 = x1 * b + x2;
    x2 = x2 * a + x3;
    x3 = x3 * b + x0;
  }
  out[t] = x0 + x1 + x2 + x3;
}

// ALU loop, integer: an LCG and xorshift mix (IMAD, SHF, LOP3, IADD3).
extern "C" __global__ void int_loop(uint32_t* out, int iters, uint32_t seed) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t x = seed + t, y = t * 2654435761u;
#pragma unroll 1
  for (int i = 0; i < iters; ++i) {
    x = x * 1664525u + 1013904223u;
    y ^= y << 13;
    y ^= y >> 17;
    y ^= y << 5;
    x += y;
    x = (x << 7) | (x >> 25);
    y += x ^ 0x9e3779b9u;
  }
  out[t] = x ^ y;
}

// Saxpy with a grid-stride loop: loads, FFMA, stores, loop control.
extern "C" __global__ void saxpy(float* y, const float* x, float a, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    y[i] = a * x[i] + y[i];
}
