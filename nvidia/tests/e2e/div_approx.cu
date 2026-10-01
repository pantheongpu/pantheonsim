// div.approx.f32 -- what __fdividef and -use_fast_math divisions compile to --
// against the ISA's description of it: a * (1/b), with the reciprocal
// rcp.approx gives, for |b| in [2^-126, 2^126]; 0 for 2^126 < |b| < 2^128
// (NaN when a is infinite). An RTX 3060 follows both to the bit, so the check
// is of the rule, not of a table: each lane computes the division and the
// documented formula side by side. rcp.approx itself is only within an ulp of
// the correctly rounded reciprocal on the card and exactly it here (the
// simulator's documented divergence for the approximate math), which the
// formula carries through the same way on both.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>

__global__ void k(const float* a, const float* b, uint32_t* bad, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float x = a[i], y = b[i];
  float d, dftz, r, want, wantftz;
  asm("div.approx.f32 %0, %1, %2;" : "=f"(d) : "f"(x), "f"(y));
  asm("div.approx.ftz.f32 %0, %1, %2;" : "=f"(dftz) : "f"(x), "f"(y));
  asm("rcp.approx.f32 %0, %1;" : "=f"(r) : "f"(y));
  asm("mul.rn.f32 %0, %1, %2;" : "=f"(want) : "f"(x), "f"(r));
  asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(wantftz) : "f"(x), "f"(r));
  const float ay = fabsf(y);
  if (ay > 0x1p126f && ay <= 3.402823466e38f) {
    // The documented huge-divisor rule, signed as a quotient is.
    want = isinf(x) ? __int_as_float(0x7fffffff) : copysignf(0.0f, x) * copysignf(1.0f, y);
    wantftz = want;
  }
  const float fd = __fdividef(x, y);   // the same instruction, from C++
  if (__float_as_uint(d) != __float_as_uint(want)) atomicAdd(&bad[0], 1);
  if (__float_as_uint(dftz) != __float_as_uint(wantftz)) atomicAdd(&bad[1], 1);
  if (__float_as_uint(fd) != __float_as_uint(want)) atomicAdd(&bad[2], 1);
}

static uint64_t st = 0x2545F4914F6CDD1Dull;
static uint32_t rnd() { st ^= st << 13; st ^= st >> 7; st ^= st << 17; return static_cast<uint32_t>(st >> 11); }
static float f(uint32_t u) { float x; std::memcpy(&x, &u, 4); return x; }

int main() {
  const int n = 1 << 16;
  float *a, *b;
  uint32_t* bad;
  cudaMallocManaged(&a, n * sizeof(float));
  cudaMallocManaged(&b, n * sizeof(float));
  cudaMallocManaged(&bad, 3 * sizeof(uint32_t));
  for (int i = 0; i < n; ++i) {
    // Operands whose quotient spans normal and subnormal results; one in
    // sixty-four divisors above 2^126, one dividend in 512 infinite.
    a[i] = f((rnd() & 0x807fffffu) | ((1 + rnd() % 253) << 23));
    b[i] = f((rnd() & 0x807fffffu) | ((1 + rnd() % 252) << 23));
    if (i % 64 == 0) b[i] = f((rnd() & 0x807fffffu) | ((253 + rnd() % 2) << 23));
    if (i % 512 == 0) a[i] = i % 1024 ? INFINITY : -INFINITY;
  }
  bad[0] = bad[1] = bad[2] = 0;
  k<<<n / 256, 256>>>(a, b, bad, n);
  if (cudaDeviceSynchronize() != cudaSuccess) { std::printf("launch failed\nFAIL\n"); return 1; }
  std::printf("div.approx.f32: %u of %d differ from a * rcp.approx(b)\n", bad[0], n);
  std::printf("div.approx.ftz.f32: %u of %d differ\n", bad[1], n);
  std::printf("__fdividef: %u of %d differ\n", bad[2], n);
  const bool ok = !bad[0] && !bad[1] && !bad[2];
  cudaFree(a);
  cudaFree(b);
  cudaFree(bad);
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
