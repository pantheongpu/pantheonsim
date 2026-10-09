// gfx1250 (CDNA 5, MI455X) kernel argument preloading (preload1250.gfx1250, built twice: with the first 3 dwords of the
// arguments preloaded into SGPRs and with 16). The hardware puts those dwords in the last user SGPRs of the wave; the compiler's
// code, unlike gfx942's, has no prologue that loads them itself, so a kernel that gets them wrong gets its first arguments
// as zero (a null pointer, a zero scale). The arguments here mix ints, floats and pointers so that the dwords a preload
// covers are each kind, and more of them than 16 spill past the registers.
#include <hip/hip_runtime.h>

#include <cstdio>
#include <vector>

#define CHECK(x)                                                                        \
  do {                                                                                  \
    hipError_t e_ = (x);                                                                \
    if (e_ != hipSuccess) {                                                             \
      std::printf("FAIL %s: %s\n", #x, hipGetErrorString(e_));                          \
      return 1;                                                                         \
    }                                                                                   \
  } while (0)

__global__ void args(int a, float s, int b, int* x, int* y, float* f, int n, long long big, int c, int d, int e, int g,
                     int h, int i, int j, int k, int l, int m, int* out) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n) return;
  y[t] = a * x[t] + b + c + d + e + g + h + i + j + k + l + m + static_cast<int>(big >> 32) + static_cast<int>(big);
  f[t] = s * static_cast<float>(x[t]);
  if (t == 0) *out = a + b + c + d + e + g + h + i + j + k + l + m;
}

int main() {
  constexpr int n = 1000;
  std::vector<int> hx(n);
  for (int t = 0; t < n; ++t) hx[t] = t * 3 - 500;
  int *x, *y, *out;
  float* f;
  CHECK(hipMalloc(&x, n * 4));
  CHECK(hipMalloc(&y, n * 4));
  CHECK(hipMalloc(&f, n * 4));
  CHECK(hipMalloc(&out, 4));
  CHECK(hipMemcpy(x, hx.data(), n * 4, hipMemcpyHostToDevice));
  const long long big = (7ll << 32) | 11;
  args<<<(n + 127) / 128, 128>>>(2, 0.5f, 3, x, y, f, n, big, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, out);
  CHECK(hipDeviceSynchronize());
  std::vector<int> ry(n);
  std::vector<float> rf(n);
  int sum = 0;
  CHECK(hipMemcpy(ry.data(), y, n * 4, hipMemcpyDeviceToHost));
  CHECK(hipMemcpy(rf.data(), f, n * 4, hipMemcpyDeviceToHost));
  CHECK(hipMemcpy(&sum, out, 4, hipMemcpyDeviceToHost));
  int wrong = 0;
  for (int t = 0; t < n; ++t) {
    wrong += ry[t] != 2 * hx[t] + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11 + 12 + 13 + 7 + 11;
    wrong += rf[t] != 0.5f * static_cast<float>(hx[t]);
  }
  wrong += sum != 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11 + 12 + 13;
  std::printf("arguments of a kernel: %d of %d wrong\n", wrong, 2 * n + 1);
  std::printf("preload1250: %d failed\n", wrong != 0);
  return wrong != 0;
}
