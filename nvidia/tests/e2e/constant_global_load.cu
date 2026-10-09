// The address of a __constant__ variable, loaded through with a global load.
// A pointer to constant data that reaches a __noinline__ function, or an
// __ldg(), is read with LDG.E.CONSTANT, and ptxas forms its address from the
// base the driver puts in constant bank 0 (ULDC.64 UR6, c[0x0][0x50]) plus the
// variable's offset in bank 3. The simulator keeps that data in its parameter
// window, which only generic loads could read, so the LDG was refused as "not
// inside any device allocation (looks like a host pointer)". GooFit's Thrust
// reductions read their parameter tables this way. Every value is computed on
// the host and the program passes on an RTX 3080 Ti.
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

__constant__ double table[64];
__constant__ int steps[16];

__device__ __noinline__ double read_at(const double* p, int i) { return __ldg(p + i); }
__device__ __noinline__ int read_int(const int* p, int i) { return __ldg(p + i); }

__global__ void k(double* out, int* iout, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  out[i] = read_at(table, i % 64) * 2.0;          // a global load of a __constant__ array
  iout[i] = read_int(steps, i % 16) + read_int(steps, (i + 3) % 16);
}

int main() {
  double ht[64];
  int hs[16];
  for (int i = 0; i < 64; ++i) ht[i] = 0.25 * i + 1.0 / (i + 1);
  for (int i = 0; i < 16; ++i) hs[i] = 100 * i - 7;
  cudaMemcpyToSymbol(table, ht, sizeof ht);
  cudaMemcpyToSymbol(steps, hs, sizeof hs);
  const int n = 500;
  double* d;
  int* di;
  cudaMalloc(&d, n * sizeof(double));
  cudaMalloc(&di, n * sizeof(int));
  k<<<(n + 127) / 128, 128>>>(d, di, n);
  double go[n];
  int gi[n];
  if (cudaMemcpy(go, d, sizeof go, cudaMemcpyDeviceToHost) != cudaSuccess ||
      cudaMemcpy(gi, di, sizeof gi, cudaMemcpyDeviceToHost) != cudaSuccess) {
    std::printf("FAIL: the kernel did not run\n");
    return 1;
  }
  int bad = 0;
  for (int i = 0; i < n; ++i) {
    const double w = ht[i % 64] * 2.0;
    const int wi = hs[i % 16] + hs[(i + 3) % 16];
    if (go[i] != w || gi[i] != wi) {
      if (bad++ < 5) std::printf("FAIL: element %d got %.17g / %d want %.17g / %d\n", i, go[i], gi[i], w, wi);
    }
  }
  std::printf(bad ? "FAIL: %d of %d\n" : "PASS: %d elements\n", bad ? bad : n, n);
  return bad != 0;
}
