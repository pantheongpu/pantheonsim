// A call through a pointer from inside a divergent loop, to a function that
// diverges itself. The callee uses the same convergence barrier register as its
// caller (ptxas gives both B6), so it saves the caller's mask with
// BMOV.32.CLEAR on entry and puts it back on return:
//   BMOV.32.CLEAR R2, B6 ... BSSY B6 ... BSYNC B6 ... BMOV.32 B6, R2
// While the callee runs, the lanes the caller has already parked at its own
// BSYNC B6 are waiting on the saved mask. The simulator let the callee's BSYNC
// release them too; they ran on, and the caller's BSYNC then waited for lanes
// that had left, which it reported as "every warp ... is waiting (a barrier some
// threads never reach)". CUB's reduce kernels calling GooFit's functions
// through pointers hit it. The results are computed on the host and the program
// passes on an RTX 3080 Ti.
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

typedef double (*fn_t)(double);

// Three levels of calls through pointers, each inside a divergent branch and
// each with a branch of its own, so every level needs convergence barriers
// and ptxas shares them across the calls.
__device__ __noinline__ double leaf_a(double x) {
  double r;
  if (x > 2.0) r = x * 3.0; else r = x + 1.0;
  if (r > 5.0) r -= 2.0;
  return r;
}
__device__ __noinline__ double leaf_b(double x) {
  double r = 0.0;
  for (int i = 0; i < (int)x % 3 + 1; ++i) r += x * i;
  return r;
}
__device__ fn_t leaves[2] = {leaf_a, leaf_b};

__device__ __noinline__ double mid_a(double x) {
  double r = x;
  fn_t g = leaves[(int)x & 1];
  if (x > 1.0) r += g(x * 0.5);
  if (r > 9.0) r = g(r - 9.0);
  return r;
}
__device__ __noinline__ double mid_b(double x) {
  double r = 0.0;
  fn_t g = leaves[((int)x + 1) & 1];
  for (int i = 0; i < (int)x % 2 + 1; ++i) { if (x > 0.5) r += g(x + i); else r -= 1.0; }
  return r;
}
__device__ fn_t table[2] = {mid_a, mid_b};

__global__ void k(const int* trips, const int* which, double* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  fn_t f;
  f = table[which[i] & 1];
  double acc = 0.0;
  // the trip count differs per lane, so lanes leave the loop one by one and wait
  // at its BSYNC while the rest call through the pointer
  for (int j = 0; j < trips[i]; ++j) acc += f(acc + j + i % 5);
  out[i] = acc;
}

static double leaf_a_h(double x) {
  double r;
  if (x > 2.0) r = x * 3.0; else r = x + 1.0;
  if (r > 5.0) r -= 2.0;
  return r;
}
static double leaf_b_h(double x) {
  double r = 0.0;
  for (int i = 0; i < (int)x % 3 + 1; ++i) r += x * i;
  return r;
}
static double (*leaves_h[2])(double) = {leaf_a_h, leaf_b_h};
static double mid_a_h(double x) {
  double r = x;
  double (*g)(double) = leaves_h[(int)x & 1];
  if (x > 1.0) r += g(x * 0.5);
  if (r > 9.0) r = g(r - 9.0);
  return r;
}
static double mid_b_h(double x) {
  double r = 0.0;
  double (*g)(double) = leaves_h[((int)x + 1) & 1];
  for (int i = 0; i < (int)x % 2 + 1; ++i) { if (x > 0.5) r += g(x + i); else r -= 1.0; }
  return r;
}

int main() {
  const int n = 1000;
  int ht[n], hw[n];
  double want[n];
  for (int i = 0; i < n; ++i) {
    // lanes 0..7 of every warp loop; the other 24 do not (and a few loop a little)
    ht[i] = (i % 32 < 8) ? 3 + i % 4 : (i % 11 == 0 ? 1 : 0);
    hw[i] = (i / 32) % 3 == 0 ? 0 : i % 2;
    double acc = 0.0;
    for (int j = 0; j < ht[i]; ++j) acc += (hw[i] & 1 ? mid_b_h : mid_a_h)(acc + j + i % 5);
    want[i] = acc;
  }
  int *dt, *dw;
  double* d;
  cudaMalloc(&dt, sizeof ht);
  cudaMalloc(&dw, sizeof hw);
  cudaMalloc(&d, sizeof want);
  cudaMemcpy(dt, ht, sizeof ht, cudaMemcpyHostToDevice);
  cudaMemcpy(dw, hw, sizeof hw, cudaMemcpyHostToDevice);
  k<<<(n + 255) / 256, 256>>>(dt, dw, d, n);
  double got[n];
  if (cudaMemcpy(got, d, sizeof got, cudaMemcpyDeviceToHost) != cudaSuccess) {
    std::printf("FAIL: the kernel did not finish\n");
    return 1;
  }
  int bad = 0;
  for (int i = 0; i < n; ++i)
    if (got[i] != want[i]) {
      if (bad++ < 5) std::printf("FAIL: element %d got %.17g want %.17g\n", i, got[i], want[i]);
    }
  std::printf(bad ? "FAIL: %d of %d\n" : "PASS: %d elements\n", bad ? bad : n, n);
  return bad != 0;
}
