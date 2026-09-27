// Dynamic parallelism (CDP2, CUDA 12 and later): kernels launching kernels,
// built the way a program builds them (-rdc=true, linked with cudadevrt).
//
// 1. A parent whose threads each launch a child grid.
// 2. Nesting: each level launches the next, three deep, and a grid is only
//    complete once its children are.
// 3. Order: one thread launches several children into the same stream, and
//    they run in the order launched.
// 4. The named device streams (cudaStreamTailLaunch, cudaStreamFireAndForget).
// 5. A child whose parameters mix sizes and alignments, including a struct
//    passed by value, so the parameter buffer's layout is exercised.
//
// Every result is checked exactly. Prints PASS on the last line.
#include <cstdio>
#include <cstring>

#define CK(x)                                                                          \
  do {                                                                                 \
    cudaError_t e_ = (x);                                                              \
    if (e_ != cudaSuccess) {                                                           \
      std::printf("FAIL: %s: %s (line %d)\n", #x, cudaGetErrorString(e_), __LINE__);   \
      return false;                                                                    \
    }                                                                                  \
  } while (0)

__global__ void fill(int* out, int base) { out[base + threadIdx.x] = base * 10 + threadIdx.x; }
__global__ void fan_out(int* out) {
  if (threadIdx.x < 4) fill<<<1, 8>>>(out, threadIdx.x * 8);
}

__global__ void level(int* out, int depth) {
  atomicAdd(&out[depth], 1);
  if (depth < 3 && threadIdx.x == 0) level<<<2, 2>>>(out, depth + 1);
}

__global__ void append(int* log, int* next, int id) { log[atomicAdd(next, 1)] = id; }
__global__ void in_order(int* log, int* next) {
  for (int i = 0; i < 5; ++i) append<<<1, 1>>>(log, next, i);
}

__global__ void mark(int* out, int v) { out[v] = v + 100; }
__global__ void named_streams(int* out) {
  mark<<<1, 1, 0, cudaStreamTailLaunch>>>(out, 0);
  mark<<<1, 1, 0, cudaStreamFireAndForget>>>(out, 1);
}

struct Mixed {
  char c;
  double d;
  short s[3];
};
__global__ void takes_mixed(double* out, char a, Mixed m, int b, long long c) {
  out[0] = a;
  out[1] = m.c;
  out[2] = m.d;
  out[3] = m.s[0] + m.s[1] * 10 + m.s[2] * 100;
  out[4] = b;
  out[5] = double(c);
}
__global__ void layout(double* out) {
  Mixed m;
  m.c = 7;
  m.d = 2.5;
  m.s[0] = 1; m.s[1] = 2; m.s[2] = 3;
  takes_mixed<<<1, 1>>>(out, 'x', m, -42, 1ll << 40);
}

bool run() {
  bool ok = true;
  {
    int* d;
    CK(cudaMalloc(&d, 32 * sizeof(int)));
    fan_out<<<1, 32>>>(d);
    CK(cudaDeviceSynchronize());
    int h[32];
    CK(cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost));
    int bad = 0;
    for (int i = 0; i < 32; ++i) bad += h[i] != (i / 8) * 80 + i % 8;
    std::printf("fan out: %d of 32 wrong\n", bad);
    ok = ok && !bad;
  }
  {
    int* d;
    CK(cudaMalloc(&d, 4 * sizeof(int)));
    CK(cudaMemset(d, 0, 4 * sizeof(int)));
    level<<<1, 2>>>(d, 0);
    CK(cudaDeviceSynchronize());
    int h[4];
    CK(cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost));
    // Level 0 is one block of 2 threads. Thread 0 of every block launches a
    // 2x2 grid of the next level, so each level has twice the blocks.
    const bool good = h[0] == 2 && h[1] == 4 && h[2] == 8 && h[3] == 16;
    std::printf("nesting: %d %d %d %d%s\n", h[0], h[1], h[2], h[3], good ? "" : " (want 2 4 8 16)");
    ok = ok && good;
  }
  {
    int *log, *next;
    CK(cudaMalloc(&log, 5 * sizeof(int)));
    CK(cudaMalloc(&next, sizeof(int)));
    CK(cudaMemset(next, 0, sizeof(int)));
    in_order<<<1, 1>>>(log, next);
    CK(cudaDeviceSynchronize());
    int h[5];
    CK(cudaMemcpy(h, log, sizeof h, cudaMemcpyDeviceToHost));
    bool good = true;
    for (int i = 0; i < 5; ++i) good = good && h[i] == i;
    std::printf("order: %d %d %d %d %d%s\n", h[0], h[1], h[2], h[3], h[4], good ? "" : " (want 0 1 2 3 4)");
    ok = ok && good;
  }
  {
    int* d;
    CK(cudaMalloc(&d, 2 * sizeof(int)));
    CK(cudaMemset(d, 0, 2 * sizeof(int)));
    named_streams<<<1, 1>>>(d);
    CK(cudaDeviceSynchronize());
    int h[2];
    CK(cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost));
    const bool good = h[0] == 100 && h[1] == 101;
    std::printf("tail and fire-and-forget streams: %d %d%s\n", h[0], h[1], good ? "" : " (want 100 101)");
    ok = ok && good;
  }
  {
    double* d;
    CK(cudaMalloc(&d, 6 * sizeof(double)));
    layout<<<1, 1>>>(d);
    CK(cudaDeviceSynchronize());
    double h[6];
    CK(cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost));
    const double want[6] = {'x', 7, 2.5, 321, -42, double(1ll << 40)};
    bool good = true;
    for (int i = 0; i < 6; ++i) good = good && h[i] == want[i];
    std::printf("parameter layout: %s\n", good ? "exact" : "WRONG");
    ok = ok && good;
  }
  return ok;
}

int main() {
  const bool ok = run();
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
