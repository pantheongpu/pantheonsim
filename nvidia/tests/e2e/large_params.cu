// Kernel parameters past the old 4 KiB limit (CUDA 12.1 and later allow
// 32764 bytes on Volta and newer), as CUDA Samples' LargeKernelParameter
// passes them. ptxas then describes the parameters with KPARAM_INFO_V2
// records and moves them in constant bank 0 (on sm_86 from 0x160 to 0x1a80),
// so the last ones sit past 0x8000, where the instructions' 16-bit offset
// reads as negative: c[0x0][-0x6880] is the pointer after an 8000-int struct.
//
// 1. A sum over every int of a 32000-byte __grid_constant__ struct, and the
//    pointer passed after it (the sample's kernelLargeParam).
// 2. Elements picked by an index the kernel is passed (LDC with a register,
//    from both ends: sm_100 reaches the far end as R + -0x7f84, which only
//    wraps to the right place in the bank's 64 KiB).
// 3. A pointer to the struct, read through (the parameter window).
// 4. Small parameters of mixed sizes around a large one: each where the cubin
//    says it is.
//
// Older toolkits cannot build it; it then launches a kernel all the same and
// says SKIP. Prints PASS (or SKIP) on the last line.
#include <cstdio>

__global__ void touch(int* p) { *p = 1; }

#if CUDART_VERSION >= 12010
constexpr int kInts = 8000;
struct Big {
  int v[kInts];
};
struct Mid {   // with Big and the rest, 32440 bytes of the 32764 allowed
  char c;
  int v[100];
};

__global__ void sum_all(const __grid_constant__ Big b, int* out) {
  int s = 0;
  for (int i = 0; i < kInts; ++i) s += b.v[i];
  *out = s;
}
__global__ void pick(const __grid_constant__ Big b, int i, int* out) {
  out[0] = b.v[i];
  out[1] = b.v[kInts - 1 - i];
}
__global__ void through_pointer(const __grid_constant__ Big b, int* out, int i) {
  const int* p = &b.v[0];
  out[threadIdx.x] = p[i + threadIdx.x];
}
__global__ void mixed(char a, const Mid m, short s, const Big b, double d, int* out, long long ll) {
  out[0] = a;
  out[1] = m.c;
  out[2] = m.v[0] + m.v[99];
  out[3] = s;
  out[4] = b.v[0] + b.v[kInts - 1];
  out[5] = static_cast<int>(d * 4);
  out[6] = static_cast<int>(ll >> 33);
}

bool run() {
  static Big b;
  static Mid m;
  long long want = 0;
  for (int i = 0; i < kInts; ++i) {
    b.v[i] = (i * 7) & 0xff;
    want += b.v[i];
  }
  m.c = 'q';
  for (int i = 0; i < 100; ++i) m.v[i] = 1000 + i;
  int* d;
  if (cudaMalloc(&d, 64 * sizeof(int)) != cudaSuccess) return false;
  bool ok = true;

  sum_all<<<1, 1>>>(b, d);
  int h[64] = {};
  cudaError_t e = cudaDeviceSynchronize();
  cudaMemcpy(h, d, sizeof(int), cudaMemcpyDeviceToHost);
  std::printf("sum of 8000 ints passed by value: %d (%s)\n", h[0], e == cudaSuccess && h[0] == want ? "right" : cudaGetErrorString(e));
  ok = ok && e == cudaSuccess && h[0] == want;

  pick<<<1, 1>>>(b, 123, d);
  e = cudaDeviceSynchronize();
  cudaMemcpy(h, d, 2 * sizeof(int), cudaMemcpyDeviceToHost);
  const bool picked = e == cudaSuccess && h[0] == b.v[123] && h[1] == b.v[kInts - 1 - 123];
  std::printf("indexed: %d %d%s\n", h[0], h[1], picked ? "" : " (WRONG)");
  ok = ok && picked;

  through_pointer<<<1, 32>>>(b, d, kInts - 40);
  e = cudaDeviceSynchronize();
  cudaMemcpy(h, d, 32 * sizeof(int), cudaMemcpyDeviceToHost);
  bool via = e == cudaSuccess;
  for (int t = 0; t < 32; ++t) via = via && h[t] == b.v[kInts - 40 + t];
  std::printf("through a pointer: %s\n", via ? "right" : "WRONG");
  ok = ok && via;

  mixed<<<1, 1>>>('a', m, -7, b, 2.25, d, 5ll << 33);
  e = cudaDeviceSynchronize();
  cudaMemcpy(h, d, 7 * sizeof(int), cudaMemcpyDeviceToHost);
  const int expect[7] = {'a', 'q', 1000 + 1099, -7, b.v[0] + b.v[kInts - 1], 9, 5};
  bool mix = e == cudaSuccess;
  for (int i = 0; i < 7; ++i) mix = mix && h[i] == expect[i];
  std::printf("mixed parameters: %d %d %d %d %d %d %d%s\n", h[0], h[1], h[2], h[3], h[4], h[5], h[6], mix ? "" : " (WRONG)");
  ok = ok && mix;
  cudaFree(d);
  return ok;
}
#endif

int main() {
#if CUDART_VERSION >= 12010
  const bool ok = run();
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
#else
  int* d;
  cudaMalloc(&d, sizeof(int));
  touch<<<1, 1>>>(d);
  cudaDeviceSynchronize();
  cudaFree(d);
  std::printf("SKIP: parameters past 4 KiB need CUDA 12.1 or later (this is %d)\n", CUDART_VERSION);
  return 0;
#endif
}
