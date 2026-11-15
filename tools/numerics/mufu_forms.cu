// The PTX approximate forms on a real card, on the operands where ptxas puts more than the one MUFU instruction around them
// (docs/numerics/math-functions.md, "The PTX forms"): each form is one kernel holding the instruction as inline PTX, run over a
// fixed list of operands, and the results are printed as text. The PTX interpreter of the simulator is held to the output
// (nvidia/tests/data/mufu/rtx3060.forms.txt, tests/unit/test_mufu_model.cpp).
//
//   nvcc -std=c++17 -arch=sm_<card> mufu_forms.cu -o mufu_forms && ./mufu_forms > forms.txt
//
// Output: "= <form>" starts a form; then "<operand bits> <result bits>" a line (div: "<a bits> <b bits> <result bits>").
#include <cstdint>
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

#define UNARY32(name, ptx) \
  __global__ void name(uint32_t* out, const uint32_t* in, int n) { \
    const int i = blockIdx.x * blockDim.x + threadIdx.x; \
    if (i >= n) return; \
    const float x = __uint_as_float(in[i]); float r; \
    asm(ptx " %0, %1;" : "=f"(r) : "f"(x)); \
    out[i] = __float_as_uint(r); \
  }
#define UNARY64(name, ptx) \
  __global__ void name(uint64_t* out, const uint64_t* in, int n) { \
    const int i = blockIdx.x * blockDim.x + threadIdx.x; \
    if (i >= n) return; \
    const double x = __longlong_as_double(in[i]); double r; \
    asm(ptx " %0, %1;" : "=d"(r) : "d"(x)); \
    out[i] = __double_as_longlong(r); \
  }
#define BINARY32(name, ptx) \
  __global__ void name(uint32_t* out, const uint32_t* a, const uint32_t* b, int n) { \
    const int i = blockIdx.x * blockDim.x + threadIdx.x; \
    if (i >= n) return; \
    const float x = __uint_as_float(a[i]), y = __uint_as_float(b[i]); float r; \
    asm(ptx " %0, %1, %2;" : "=f"(r) : "f"(x), "f"(y)); \
    out[i] = __float_as_uint(r); \
  }

UNARY32(k_rcp_approx_f32, "rcp.approx.f32")
UNARY32(k_rcp_approx_ftz_f32, "rcp.approx.ftz.f32")
UNARY32(k_rsqrt_approx_f32, "rsqrt.approx.f32")
UNARY32(k_rsqrt_approx_ftz_f32, "rsqrt.approx.ftz.f32")
UNARY32(k_sqrt_approx_f32, "sqrt.approx.f32")
UNARY32(k_sqrt_approx_ftz_f32, "sqrt.approx.ftz.f32")
UNARY32(k_sin_approx_f32, "sin.approx.f32")
UNARY32(k_sin_approx_ftz_f32, "sin.approx.ftz.f32")
UNARY32(k_cos_approx_f32, "cos.approx.f32")
UNARY32(k_cos_approx_ftz_f32, "cos.approx.ftz.f32")
UNARY32(k_ex2_approx_f32, "ex2.approx.f32")
UNARY32(k_ex2_approx_ftz_f32, "ex2.approx.ftz.f32")
UNARY32(k_lg2_approx_f32, "lg2.approx.f32")
UNARY32(k_lg2_approx_ftz_f32, "lg2.approx.ftz.f32")
UNARY32(k_tanh_approx_f32, "tanh.approx.f32")
UNARY64(k_rcp_approx_ftz_f64, "rcp.approx.ftz.f64")
UNARY64(k_rsqrt_approx_ftz_f64, "rsqrt.approx.ftz.f64")
UNARY64(k_rsqrt_approx_f64, "rsqrt.approx.f64")
BINARY32(k_div_approx_f32, "div.approx.f32")
BINARY32(k_div_approx_ftz_f32, "div.approx.ftz.f32")
BINARY32(k_div_full_f32, "div.full.f32")
BINARY32(k_div_full_ftz_f32, "div.full.ftz.f32")

// Operands: subnormals, the smallest normal numbers, the largest, the thresholds the expansions test (2^126, -126), zeros,
// infinities, NaNs, a few ordinary values.
static const uint32_t kIn32[] = {
    0x00000000u, 0x80000000u, 0x00000001u, 0x80000001u, 0x00000100u, 0x00400000u, 0x80400000u, 0x007fffffu, 0x807fffffu,
    0x00800000u, 0x80800000u, 0x00800001u, 0x00ffffffu, 0x01000000u, 0x0c000000u, 0x7e7fffffu, 0x7e800000u, 0x7e800001u,
    0xfe800001u, 0x7f000000u, 0x7f7fffffu, 0xff7fffffu, 0x7f800000u, 0xff800000u, 0x7fc00000u, 0x7f800001u, 0xffc12345u,
    0xc2fa0000u, 0xc2fbffffu, 0xc2fc0000u, 0xc2fc0001u, 0xc2fe0000u, 0xc3000000u, 0xc30a8000u, 0xc3150000u, 0xc3160000u,
    0xc3800000u, 0x42fe0000u, 0x43000000u, 0x3f800000u, 0xbf800000u, 0x3fc00000u, 0x40490fdbu, 0x3dcccccdu, 0x41200000u,
    0x3eaaaaabu, 0x4b800000u, 0x5f000000u, 0x00012345u, 0x00654321u, 0x3a83126fu, 0x3f000000u};
static const uint64_t kIn64[] = {
    0x0000000000000000ull, 0x8000000000000000ull, 0x0000000000000001ull, 0x000fffffffffffffull, 0x800fffffffffffffull,
    0x0010000000000000ull, 0x0010000000000001ull, 0x001fffffffffffffull, 0x7fe0000000000000ull, 0x7fefffffffffffffull,
    0x7ff0000000000000ull, 0xfff0000000000000ull, 0x7ff8000000000000ull, 0x7ff0000000000001ull, 0x3ff0000000000000ull,
    0xbff0000000000000ull, 0x3ff8000000000000ull, 0x4000000000000000ull, 0x4008000000000000ull, 0x40080000ffffffffull,
    0x400921fb54442d18ull, 0x3fb999999999999aull, 0x4024000000000000ull, 0x3fd5555555555555ull, 0x3fefffffffffffffull,
    0x3ff0000000000001ull, 0x0000000012345678ull, 0x0000123456789abcull, 0x43e0000000000000ull, 0x3c9cd2b297d889bcull,
    0x7fd0000000000000ull, 0x0020000000000000ull};
static const uint32_t kDiv[] = {   // dividends and divisors: all pairs
    0x00000000u, 0x80000000u, 0x00400000u, 0x00800000u, 0x3f800000u, 0xbfc00000u, 0x40490fdbu, 0x7e800000u, 0x7e800001u,
    0xff7fffffu, 0x7f800000u, 0x7fc00000u};

template <typename T, typename K>
static void unary(const char* form, K kernel, const T* in, int n, const char* fmt) {
  T *di = nullptr, *dr = nullptr;
  cudaMalloc(&di, n * sizeof(T)); cudaMalloc(&dr, n * sizeof(T));
  cudaMemcpy(di, in, n * sizeof(T), cudaMemcpyHostToDevice);
  kernel<<<1, 64>>>(dr, di, n);
  std::vector<T> r(n);
  const bool ok = cudaDeviceSynchronize() == cudaSuccess && cudaMemcpy(r.data(), dr, n * sizeof(T), cudaMemcpyDeviceToHost) == cudaSuccess;
  printf("= %s\n", form);
  if (!ok) printf("FAILED\n");
  else for (int i = 0; i < n; ++i) { printf(fmt, static_cast<unsigned long long>(in[i])); printf(" "); printf(fmt, static_cast<unsigned long long>(r[i])); printf("\n"); }
  cudaFree(di); cudaFree(dr);
}
template <typename K>
static void binary(const char* form, K kernel) {
  const int m = sizeof kDiv / sizeof kDiv[0], n = m * m;
  std::vector<uint32_t> a(n), b(n), r(n);
  for (int i = 0; i < n; ++i) { a[i] = kDiv[i / m]; b[i] = kDiv[i % m]; }
  uint32_t *da = nullptr, *db = nullptr, *dr = nullptr;
  cudaMalloc(&da, n * 4); cudaMalloc(&db, n * 4); cudaMalloc(&dr, n * 4);
  cudaMemcpy(da, a.data(), n * 4, cudaMemcpyHostToDevice); cudaMemcpy(db, b.data(), n * 4, cudaMemcpyHostToDevice);
  kernel<<<(n + 63) / 64, 64>>>(dr, da, db, n);
  const bool ok = cudaDeviceSynchronize() == cudaSuccess && cudaMemcpy(r.data(), dr, n * 4, cudaMemcpyDeviceToHost) == cudaSuccess;
  printf("= %s\n", form);
  if (!ok) printf("FAILED\n");
  else for (int i = 0; i < n; ++i) printf("%08x %08x %08x\n", a[i], b[i], r[i]);
  cudaFree(da); cudaFree(db); cudaFree(dr);
}

int main() {
  const int n32 = sizeof kIn32 / sizeof kIn32[0], n64 = sizeof kIn64 / sizeof kIn64[0];
  unary("rcp.approx.f32", k_rcp_approx_f32, kIn32, n32, "%08llx");
  unary("rcp.approx.ftz.f32", k_rcp_approx_ftz_f32, kIn32, n32, "%08llx");
  unary("rsqrt.approx.f32", k_rsqrt_approx_f32, kIn32, n32, "%08llx");
  unary("rsqrt.approx.ftz.f32", k_rsqrt_approx_ftz_f32, kIn32, n32, "%08llx");
  unary("sqrt.approx.f32", k_sqrt_approx_f32, kIn32, n32, "%08llx");
  unary("sqrt.approx.ftz.f32", k_sqrt_approx_ftz_f32, kIn32, n32, "%08llx");
  unary("sin.approx.f32", k_sin_approx_f32, kIn32, n32, "%08llx");
  unary("sin.approx.ftz.f32", k_sin_approx_ftz_f32, kIn32, n32, "%08llx");
  unary("cos.approx.f32", k_cos_approx_f32, kIn32, n32, "%08llx");
  unary("cos.approx.ftz.f32", k_cos_approx_ftz_f32, kIn32, n32, "%08llx");
  unary("ex2.approx.f32", k_ex2_approx_f32, kIn32, n32, "%08llx");
  unary("ex2.approx.ftz.f32", k_ex2_approx_ftz_f32, kIn32, n32, "%08llx");
  unary("lg2.approx.f32", k_lg2_approx_f32, kIn32, n32, "%08llx");
  unary("lg2.approx.ftz.f32", k_lg2_approx_ftz_f32, kIn32, n32, "%08llx");
  unary("tanh.approx.f32", k_tanh_approx_f32, kIn32, n32, "%08llx");
  unary("rcp.approx.ftz.f64", k_rcp_approx_ftz_f64, kIn64, n64, "%016llx");
  unary("rsqrt.approx.ftz.f64", k_rsqrt_approx_ftz_f64, kIn64, n64, "%016llx");
  unary("rsqrt.approx.f64", k_rsqrt_approx_f64, kIn64, n64, "%016llx");
  binary("div.approx.f32", k_div_approx_f32);
  binary("div.approx.ftz.f32", k_div_approx_ftz_f32);
  binary("div.full.f32", k_div_full_f32);
  binary("div.full.ftz.f32", k_div_full_ftz_f32);
  return 0;
}
