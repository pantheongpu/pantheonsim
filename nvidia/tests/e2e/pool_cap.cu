// A loop of small launches (the launch64 shape: 64 blocks of one warp, a few microseconds of work each) followed by launches of
// other sizes of the same kernels, and a grid with plenty of work. Which host thread runs a block, and how many threads a launch
// gets, must change nothing: the program checks every value against the arithmetic and prints a checksum, and run_pool_cap.sh
// runs it at several VGPU_THREADS under both engines with the performance counters on, and requires the same output each time.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

__global__ void bump(float* p) { p[blockIdx.x * 32 + threadIdx.x] += 1.f + blockIdx.x; }
__global__ void tally(unsigned* c, unsigned* hist) {
  unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
  atomicAdd(c, 1u);
  atomicAdd(&hist[t % 16], t);
}
__global__ void heavy(unsigned* out, int iters) {
  unsigned t = blockIdx.x * blockDim.x + threadIdx.x, x = t * 2654435761u + 1, s = 0;
  for (int i = 0; i < iters; i++) { x ^= x << 13; x ^= x >> 17; x ^= x << 5; s += (x & 1) ? x : x >> 3; }
  out[t] = s;
}

int main() {
  const int kLaunches = 1500;
  float* f; CK(cudaMalloc(&f, 64 * 32 * 4)); CK(cudaMemset(f, 0, 64 * 32 * 4));
  for (int i = 0; i < kLaunches; i++) bump<<<64, 32>>>(f);
  // Same kernel, other grid sizes (the cost of a block is known by now, the grid is not the one it was learned on).
  for (int i = 0; i < 100; i++) bump<<<7, 32>>>(f);
  for (int i = 0; i < 20; i++) bump<<<64, 32>>>(f);
  CK(cudaDeviceSynchronize());
  std::vector<float> hf(64 * 32);
  CK(cudaMemcpy(hf.data(), f, 64 * 32 * 4, cudaMemcpyDeviceToHost));
  int bad = 0; double sum = 0;
  for (int b = 0; b < 64; b++)
    for (int l = 0; l < 32; l++) {
      const float want = (kLaunches + 20 + (b < 7 ? 100 : 0)) * (1.f + b);
      if (hf[b * 32 + l] != want) bad++;
      sum += hf[b * 32 + l];
    }
  unsigned *c, *hist; CK(cudaMalloc(&c, 4)); CK(cudaMalloc(&hist, 64)); CK(cudaMemset(c, 0, 4)); CK(cudaMemset(hist, 0, 64));
  const int kTally = 600;
  for (int i = 0; i < kTally; i++) tally<<<32, 64>>>(c, hist);
  unsigned hc, hh[16];
  CK(cudaMemcpy(&hc, c, 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(hh, hist, 64, cudaMemcpyDeviceToHost));
  unsigned long long want_hist[16] = {};
  for (unsigned t = 0; t < 32 * 64; t++) want_hist[t % 16] += t;
  if (hc != (unsigned)kTally * 32 * 64) bad++;
  for (int i = 0; i < 16; i++) if (hh[i] != (unsigned)(want_hist[i] * kTally)) bad++;
  // A grid with plenty of work, then small ones again.
  const int B = 256, T = 128; unsigned* o; CK(cudaMalloc(&o, B * T * 4));
  heavy<<<B, T>>>(o, 2000);
  std::vector<unsigned> ho(B * T); CK(cudaMemcpy(ho.data(), o, B * T * 4, cudaMemcpyDeviceToHost));
  unsigned long long h = 1469598103934665603ull;
  for (unsigned v : ho) { h ^= v; h *= 1099511628211ull; }
  for (int i = 0; i < 200; i++) heavy<<<3, T>>>(o, 20);
  CK(cudaDeviceSynchronize());
  printf("bump sum %.1f tally %u heavy %016llx bad %d\n", sum, hc, h, bad);
  puts(bad ? "FAIL" : "PASS");
  return bad != 0;
}
