// gfx1250's (CDNA 5, MI455X) scaled narrowing conversions, rounded to nearest even (CDNA5 ISA, V_CVT_SCALEF32_PK8_FP8_F32 ... V_CVT_SCALEF32_PK16_BF6_BF16),
// called through the clang builtins (scalef32_narrow1250.gfx1250, wave32). Each takes 8 or 16 floats, halves or bfloat16s and the scale operand
// (the exponent field of a float) and writes 8-bit floats, nibbles or six-bit values packed from bit 0 of the result. As for gfx950's
// (scalef32_narrow.cpp) the guide leaves its helper functions undefined and the rule comes from Composable Kernel (the scale divides) and HIP's host reference
// (amd_hip_ocp_host.hpp, fcbx::from_float), which this compares every converted value with. The sweep keeps to the range of the 8-bit formats
// (HIP's host reference rounds before it judges an overflow, the guide's table does not) and has no NaN going into a 4- or 6-bit float.
#include <hip/hip_runtime.h>
#include <hip/hip_ext_ocp.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <vector>

static void check_hip(hipError_t e) {
  if (e != hipSuccess) {
    std::printf("HIP error %d (%s)\n", (int)e, hipGetErrorString(e));
    std::exit(2);
  }
}
#define hipMalloc(...) check_hip(hipMalloc(__VA_ARGS__))
#define hipMemcpy(...) check_hip(hipMemcpy(__VA_ARGS__))

using v8f = float __attribute__((ext_vector_type(8)));
using v16f = float __attribute__((ext_vector_type(16)));
using v8h = _Float16 __attribute__((ext_vector_type(8)));
using v16h = _Float16 __attribute__((ext_vector_type(16)));
using v8b = __bf16 __attribute__((ext_vector_type(8)));
using v16b = __bf16 __attribute__((ext_vector_type(16)));
using v2u = uint32_t __attribute__((ext_vector_type(2)));
using v3u = uint32_t __attribute__((ext_vector_type(3)));

static int failed = 0;
static void report(const char* what, int wrong, int of) {
  std::printf("%s: %d of %d wrong\n", what, wrong, of);
  failed += wrong != 0;
}

#define PK8_8BIT(NAME, B, V, E) \
  __global__ void NAME(const E* a, const float* s, uint32_t* o) { \
    const int i = blockIdx.x * 32 + threadIdx.x; V in; for (int k = 0; k < 8; ++k) in[k] = a[8 * i + k]; \
    v2u r = B(in, s[i]); o[2 * i] = r[0]; o[2 * i + 1] = r[1]; }
#define PK8_4BIT(NAME, B, V, E) \
  __global__ void NAME(const E* a, const float* s, uint32_t* o) { \
    const int i = blockIdx.x * 32 + threadIdx.x; V in; for (int k = 0; k < 8; ++k) in[k] = a[8 * i + k]; \
    o[i] = B(in, s[i]); }
#define PK16_6BIT(NAME, B, V, E) \
  __global__ void NAME(const E* a, const float* s, uint32_t* o) { \
    const int i = blockIdx.x * 32 + threadIdx.x; V in; for (int k = 0; k < 16; ++k) in[k] = a[16 * i + k]; \
    v3u r = B(in, s[i]); for (int k = 0; k < 3; ++k) o[3 * i + k] = r[k]; }
PK8_8BIT(k_pk8_fp8_f32, __builtin_amdgcn_cvt_scalef32_pk8_fp8_f32, v8f, float)
PK8_8BIT(k_pk8_bf8_f32, __builtin_amdgcn_cvt_scalef32_pk8_bf8_f32, v8f, float)
PK8_8BIT(k_pk8_fp8_f16, __builtin_amdgcn_cvt_scalef32_pk8_fp8_f16, v8h, _Float16)
PK8_8BIT(k_pk8_bf8_f16, __builtin_amdgcn_cvt_scalef32_pk8_bf8_f16, v8h, _Float16)
PK8_8BIT(k_pk8_fp8_bf16, __builtin_amdgcn_cvt_scalef32_pk8_fp8_bf16, v8b, __bf16)
PK8_8BIT(k_pk8_bf8_bf16, __builtin_amdgcn_cvt_scalef32_pk8_bf8_bf16, v8b, __bf16)
PK8_4BIT(k_pk8_fp4_f32, __builtin_amdgcn_cvt_scalef32_pk8_fp4_f32, v8f, float)
PK8_4BIT(k_pk8_fp4_f16, __builtin_amdgcn_cvt_scalef32_pk8_fp4_f16, v8h, _Float16)
PK8_4BIT(k_pk8_fp4_bf16, __builtin_amdgcn_cvt_scalef32_pk8_fp4_bf16, v8b, __bf16)
PK16_6BIT(k_pk16_fp6_f32, __builtin_amdgcn_cvt_scalef32_pk16_fp6_f32, v16f, float)
PK16_6BIT(k_pk16_bf6_f32, __builtin_amdgcn_cvt_scalef32_pk16_bf6_f32, v16f, float)
PK16_6BIT(k_pk16_fp6_f16, __builtin_amdgcn_cvt_scalef32_pk16_fp6_f16, v16h, _Float16)
PK16_6BIT(k_pk16_bf6_f16, __builtin_amdgcn_cvt_scalef32_pk16_bf6_f16, v16h, _Float16)
PK16_6BIT(k_pk16_fp6_bf16, __builtin_amdgcn_cvt_scalef32_pk16_fp6_bf16, v16b, __bf16)
PK16_6BIT(k_pk16_bf6_bf16, __builtin_amdgcn_cvt_scalef32_pk16_bf6_bf16, v16b, __bf16)

static uint32_t next_random() {
  static uint32_t seed = 13579246;
  return seed = seed * 1664525u + 1013904223u;
}
static float random_value(int lo, int hi, float mant_max, int cap) {
  const float m = 1.0f + (float)(next_random() >> 8) / 16777216.0f * (mant_max - 1.0f);
  if (hi > cap) hi = cap;
  if (lo > hi) lo = hi;
  const int q = lo + (int)(next_random() % (uint32_t)(hi - lo + 1));
  return std::ldexp(next_random() & 1 ? -m : m, q);
}
template <class T> static T* upload(const std::vector<T>& v) {
  T* d;
  hipMalloc(&d, v.size() * sizeof(T));
  hipMemcpy(d, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice);
  return d;
}
template <fcbx::Encoding E, class T> static uint32_t ref(T x, int k) { return fcbx::from_float<T, E, true>(x, (int8_t)k); }

constexpr int kCases = 512;   // lanes: 16 waves of 32

// N elements a lane, BITS bits each, packed from bit 0 of REGS registers.
template <class T, fcbx::Encoding E, int N, int BITS, int REGS, class K>
static void check(const char* what, K kernel, int lo_exp, int hi_exp, float mant_max) {
  std::vector<T> x(N * kCases);
  std::vector<float> s(kCases);
  std::vector<int> k(kCases);
  for (int i = 0; i < kCases; ++i) {
    k[i] = (int)(next_random() % 21) - 10;
    s[i] = std::ldexp(1.0f, k[i]);
    for (int j = 0; j < N; ++j)
      x[N * i + j] = (T)(random_value(lo_exp, hi_exp, mant_max, std::is_same_v<T, _Float16> ? 14 - k[i] : 100) * s[i]);
  }
  uint32_t* o;
  hipMalloc(&o, REGS * kCases * 4);
  T* dx = upload(x);
  float* ds = upload(s);
  kernel<<<kCases / 32, 32>>>(dx, ds, o);
  std::vector<uint32_t> got(REGS * kCases);
  hipMemcpy(got.data(), o, got.size() * 4, hipMemcpyDeviceToHost);
  int wrong = 0;
  for (int i = 0; i < kCases; ++i) {
    uint32_t want[REGS] = {};
    for (int j = 0; j < N; ++j) {
      const uint32_t code = ref<E>(x[N * i + j], k[i]);
      for (int b = 0; b < BITS; ++b) want[(BITS * j + b) / 32] |= (code >> b & 1) << ((BITS * j + b) % 32);
    }
    for (int r = 0; r < REGS; ++r)
      if (got[REGS * i + r] != want[r] && wrong++ < 3) std::printf("  %s: lane %d register %d is 0x%08x, not 0x%08x\n", what, i, r, got[REGS * i + r], want[r]);
  }
  report(what, wrong, kCases);
}

int main() {
  using fcbx::Encoding;
  check<float, Encoding::E4M3Mx, 8, 8, 2>("v_cvt_scalef32_pk8_fp8_f32", k_pk8_fp8_f32, -12, 8, 1.7f);
  check<float, Encoding::E5M2Mx, 8, 8, 2>("v_cvt_scalef32_pk8_bf8_f32", k_pk8_bf8_f32, -18, 14, 1.9f);
  check<_Float16, Encoding::E4M3Mx, 8, 8, 2>("v_cvt_scalef32_pk8_fp8_f16", k_pk8_fp8_f16, -8, 7, 1.7f);
  check<_Float16, Encoding::E5M2Mx, 8, 8, 2>("v_cvt_scalef32_pk8_bf8_f16", k_pk8_bf8_f16, -14, 13, 1.9f);
  check<__bf16, Encoding::E4M3Mx, 8, 8, 2>("v_cvt_scalef32_pk8_fp8_bf16", k_pk8_fp8_bf16, -12, 8, 1.7f);
  check<__bf16, Encoding::E5M2Mx, 8, 8, 2>("v_cvt_scalef32_pk8_bf8_bf16", k_pk8_bf8_bf16, -18, 14, 1.9f);
  check<float, Encoding::E2M1, 8, 4, 1>("v_cvt_scalef32_pk8_fp4_f32", k_pk8_fp4_f32, -4, 5, 2.0f);
  check<_Float16, Encoding::E2M1, 8, 4, 1>("v_cvt_scalef32_pk8_fp4_f16", k_pk8_fp4_f16, -4, 5, 2.0f);
  check<__bf16, Encoding::E2M1, 8, 4, 1>("v_cvt_scalef32_pk8_fp4_bf16", k_pk8_fp4_bf16, -4, 5, 2.0f);
  check<float, Encoding::E2M3, 16, 6, 3>("v_cvt_scalef32_pk16_fp6_f32", k_pk16_fp6_f32, -6, 6, 2.0f);
  check<float, Encoding::E3M2, 16, 6, 3>("v_cvt_scalef32_pk16_bf6_f32", k_pk16_bf6_f32, -6, 6, 2.0f);
  check<_Float16, Encoding::E2M3, 16, 6, 3>("v_cvt_scalef32_pk16_fp6_f16", k_pk16_fp6_f16, -6, 6, 2.0f);
  check<_Float16, Encoding::E3M2, 16, 6, 3>("v_cvt_scalef32_pk16_bf6_f16", k_pk16_bf6_f16, -6, 6, 2.0f);
  check<__bf16, Encoding::E2M3, 16, 6, 3>("v_cvt_scalef32_pk16_fp6_bf16", k_pk16_fp6_bf16, -6, 6, 2.0f);
  check<__bf16, Encoding::E3M2, 16, 6, 3>("v_cvt_scalef32_pk16_bf6_bf16", k_pk16_bf6_bf16, -6, 6, 2.0f);
  std::printf("scalef32_narrow1250: %d failed\n", failed);
  return failed != 0;
}
