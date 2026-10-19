// gfx950's scaled narrowing conversions (CDNA4 ISA, V_CVT_SCALEF32_PK_FP8_F32 ... V_CVT_SCALEF32_PK32_BF6_BF16, opcodes 565 to 603, the
// round-to-nearest-even ones) and the single-half widening V_CVT_SCALEF32_F16_FP8 / _BF8 (586, 587), called through the clang builtins
// Composable Kernel and HIP's headers call them by (scalef32_narrow.gfx950, wave64).
//
// The guide gives where every source and result goes and leaves its helper functions (f32_to_fp8_scale ...) undefined. What the simulator
// computes for them comes from other public sources, and this checks it two ways:
//  - a sweep of values, each converted on the device and by HIP's own host reference (amd_hip_ocp_host.hpp, `fcbx::from_float`, which
//    <hip/hip_ext_ocp.h> brings in), with the result laid out as the guide says: pairs of bytes or nibbles into the half or byte OPSEL names
//    (the rest of the register kept), 2xpk16's two arrays interleaved, six-bit values packed from bit 0. The sweep keeps to values whose
//    scaled magnitude is inside the 8-bit formats' range: the host reference rounds before it decides on an overflow, where the guide's table
//    (and CK's device test) have a NaN or an infinity for anything past the largest value, and its NaN into a 4- or 6-bit float
//    is not what CK's tests (the largest value) and AMD's emulator tests (zero) say, so neither is in the sweep.
//  - a table of the cases Composable Kernel's own tests (test/data_type/test_mx_fp8.cpp, test_mx_fp4.cpp, test_mx_fp6.cpp) run on a gfx950, with
//    the answers they expect of the device: scale 2.0 turns {-8, 4} into {-4, 2} (the scale divides), infinity / 2 and 256 / 0.5 are NaN in fp8
//    and the largest value in fp4 and fp6, 4.75 / 4 is 1.25 in fp6, and so on; bf8's overflow is the infinity (CK's comment in mxf8_utils.hpp and
//    the guide's FP8/BF8 table).
// The value goes into the format it is converted to, so a table row says what it must widen back to.
#include <hip/hip_runtime.h>
#include <hip/hip_ext_ocp.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <vector>

#pragma clang diagnostic ignored "-Wunused-function"

static void check_hip(hipError_t e) {
  if (e != hipSuccess) {
    std::printf("HIP error %d (%s)\n", (int)e, hipGetErrorString(e));
    std::exit(2);
  }
}
#define hipMalloc(...) check_hip(hipMalloc(__VA_ARGS__))
#define hipMemcpy(...) check_hip(hipMemcpy(__VA_ARGS__))

using v2f = float __attribute__((ext_vector_type(2)));
using v16f = float __attribute__((ext_vector_type(16)));
using v2s = short __attribute__((ext_vector_type(2)));
using v2h = _Float16 __attribute__((ext_vector_type(2)));
using v2b = __bf16 __attribute__((ext_vector_type(2)));
using v32h = _Float16 __attribute__((ext_vector_type(32)));
using v32b = __bf16 __attribute__((ext_vector_type(32)));
using v6u = uint32_t __attribute__((ext_vector_type(6)));

static int failed = 0;
static void report(const char* what, int wrong, int of) {
  std::printf("%s: %d of %d wrong\n", what, wrong, of);
  failed += wrong != 0;
}

// A scale operand: the float 2^k, whose exponent field is the E8M0 code k + 127.
static float scale_of(int k) { return std::ldexp(1.0f, k); }

// ---- the device kernels -------------------------------------------------------------------------------------------------------------------
// Pairs of values into 16 bits (8-bit formats, OPSEL[3] = HI) or a byte (fp4, OPSEL[3:2] = BYTE) of a destination that starts 0xA5A5A5A5.
#define PK8_F32(NAME, B, HI) \
  __global__ void NAME(const float* a, const float* b, const float* s, uint32_t* o) { \
    const int i = blockIdx.x * 64 + threadIdx.x; v2s old; old[0] = old[1] = (short)0xA5A5; \
    v2s r = B(old, a[i], b[i], s[i], HI); o[i] = (uint32_t)(uint16_t)r[0] | (uint32_t)(uint16_t)r[1] << 16; }
#define PK8_H(NAME, B, V, E, HI) \
  __global__ void NAME(const E* a, const float* s, uint32_t* o) { \
    const int i = blockIdx.x * 64 + threadIdx.x; v2s old; old[0] = old[1] = (short)0xA5A5; \
    v2s r = B(old, V{a[2 * i], a[2 * i + 1]}, s[i], HI); o[i] = (uint32_t)(uint16_t)r[0] | (uint32_t)(uint16_t)r[1] << 16; }
PK8_F32(k_pk_fp8_f32_lo, __builtin_amdgcn_cvt_scalef32_pk_fp8_f32, false)
PK8_F32(k_pk_fp8_f32_hi, __builtin_amdgcn_cvt_scalef32_pk_fp8_f32, true)
PK8_F32(k_pk_bf8_f32_lo, __builtin_amdgcn_cvt_scalef32_pk_bf8_f32, false)
PK8_F32(k_pk_bf8_f32_hi, __builtin_amdgcn_cvt_scalef32_pk_bf8_f32, true)
PK8_H(k_pk_fp8_f16_lo, __builtin_amdgcn_cvt_scalef32_pk_fp8_f16, v2h, _Float16, false)
PK8_H(k_pk_fp8_f16_hi, __builtin_amdgcn_cvt_scalef32_pk_fp8_f16, v2h, _Float16, true)
PK8_H(k_pk_bf8_f16_lo, __builtin_amdgcn_cvt_scalef32_pk_bf8_f16, v2h, _Float16, false)
PK8_H(k_pk_bf8_f16_hi, __builtin_amdgcn_cvt_scalef32_pk_bf8_f16, v2h, _Float16, true)
PK8_H(k_pk_fp8_bf16_lo, __builtin_amdgcn_cvt_scalef32_pk_fp8_bf16, v2b, __bf16, false)
PK8_H(k_pk_fp8_bf16_hi, __builtin_amdgcn_cvt_scalef32_pk_fp8_bf16, v2b, __bf16, true)
PK8_H(k_pk_bf8_bf16_lo, __builtin_amdgcn_cvt_scalef32_pk_bf8_bf16, v2b, __bf16, false)
PK8_H(k_pk_bf8_bf16_hi, __builtin_amdgcn_cvt_scalef32_pk_bf8_bf16, v2b, __bf16, true)

#define PK4_F32(NAME, BYTE) \
  __global__ void NAME(const float* a, const float* b, const float* s, uint32_t* o) { \
    const int i = blockIdx.x * 64 + threadIdx.x; \
    o[i] = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(0xA5A5A5A5u, a[i], b[i], s[i], BYTE); }
#define PK4_H(NAME, B, V, E, BYTE) \
  __global__ void NAME(const E* a, const float* s, uint32_t* o) { \
    const int i = blockIdx.x * 64 + threadIdx.x; o[i] = B(0xA5A5A5A5u, V{a[2 * i], a[2 * i + 1]}, s[i], BYTE); }
PK4_F32(k_pk_fp4_f32_0, 0)
PK4_F32(k_pk_fp4_f32_1, 1)
PK4_F32(k_pk_fp4_f32_2, 2)
PK4_F32(k_pk_fp4_f32_3, 3)
PK4_H(k_pk_fp4_f16_0, __builtin_amdgcn_cvt_scalef32_pk_fp4_f16, v2h, _Float16, 0)
PK4_H(k_pk_fp4_f16_2, __builtin_amdgcn_cvt_scalef32_pk_fp4_f16, v2h, _Float16, 2)
PK4_H(k_pk_fp4_bf16_1, __builtin_amdgcn_cvt_scalef32_pk_fp4_bf16, v2b, __bf16, 1)
PK4_H(k_pk_fp4_bf16_3, __builtin_amdgcn_cvt_scalef32_pk_fp4_bf16, v2b, __bf16, 3)

// 32 six-bit values into six registers: from two arrays of sixteen floats (interleaved), or from 32 halves or bfloat16s.
#define X2PK16(NAME, B) \
  __global__ void NAME(const float* a, const float* s, uint32_t* o) { \
    const int i = threadIdx.x + blockIdx.x * 64; v16f lo, hi; \
    for (int k = 0; k < 16; ++k) { lo[k] = a[32 * i + 2 * k]; hi[k] = a[32 * i + 2 * k + 1]; } \
    v6u r = B(lo, hi, s[i]); for (int k = 0; k < 6; ++k) o[6 * i + k] = r[k]; }
X2PK16(k_2xpk16_fp6_f32, __builtin_amdgcn_cvt_scalef32_2xpk16_fp6_f32)
X2PK16(k_2xpk16_bf6_f32, __builtin_amdgcn_cvt_scalef32_2xpk16_bf6_f32)
#define PK32(NAME, B, V, T) \
  __global__ void NAME(const T* a, const float* s, uint32_t* o) { \
    const int i = threadIdx.x + blockIdx.x * 64; V in; for (int k = 0; k < 32; ++k) in[k] = a[32 * i + k]; \
    v6u r = B(in, s[i]); for (int k = 0; k < 6; ++k) o[6 * i + k] = r[k]; }
PK32(k_pk32_fp6_f16, __builtin_amdgcn_cvt_scalef32_pk32_fp6_f16, v32h, _Float16)
PK32(k_pk32_bf6_f16, __builtin_amdgcn_cvt_scalef32_pk32_bf6_f16, v32h, _Float16)
PK32(k_pk32_fp6_bf16, __builtin_amdgcn_cvt_scalef32_pk32_fp6_bf16, v32b, __bf16)
PK32(k_pk32_bf6_bf16, __builtin_amdgcn_cvt_scalef32_pk32_bf6_bf16, v32b, __bf16)

// One 8-bit float to a half, into the half named, the other half kept (starts 0xA5A5A5A5): the byte of the source named too.
#define WIDEN_H(NAME, B, BYTE, HI) \
  __global__ void NAME(const uint32_t* a, const float* s, uint32_t* o) { \
    const int i = blockIdx.x * 64 + threadIdx.x; v2h old; old[0] = old[1] = __builtin_bit_cast(_Float16, (uint16_t)0xA5A5); \
    v2h r = B(old, (int)a[i], s[i], BYTE, HI); o[i] = __builtin_bit_cast(uint32_t, r); }
WIDEN_H(k_f16_fp8_b0_lo, __builtin_amdgcn_cvt_scalef32_f16_fp8, 0, false)
WIDEN_H(k_f16_fp8_b1_hi, __builtin_amdgcn_cvt_scalef32_f16_fp8, 1, true)
WIDEN_H(k_f16_fp8_b2_lo, __builtin_amdgcn_cvt_scalef32_f16_fp8, 2, false)
WIDEN_H(k_f16_fp8_b3_hi, __builtin_amdgcn_cvt_scalef32_f16_fp8, 3, true)
WIDEN_H(k_f16_bf8_b0_hi, __builtin_amdgcn_cvt_scalef32_f16_bf8, 0, true)
WIDEN_H(k_f16_bf8_b3_lo, __builtin_amdgcn_cvt_scalef32_f16_bf8, 3, false)

// ---- the host side ------------------------------------------------------------------------------------------------------------------------
// Small floats decoded without the simulator's code: sign, `eb` exponent bits (bias `bias`), `mb` mantissa bits; E4M3 (OCP) has a NaN at all-ones
// and no infinity, E5M2 an infinity and NaNs; the 4- and 6-bit ones neither.
struct Fmt { int eb, mb, bias; int kind; };   // kind 0: finite only, 1: E4M3 (NaN at S.1111.111), 2: E5M2
static const Fmt FP8{4, 3, 7, 1}, BF8{5, 2, 15, 2}, FP6{2, 3, 1, 0}, BF6{3, 2, 3, 0}, FP4{2, 1, 1, 0};
static float decode(const Fmt& f, uint32_t code) {
  const int bits = 1 + f.eb + f.mb;
  const bool neg = code >> (bits - 1) & 1;
  const int e = code >> f.mb & ((1 << f.eb) - 1), m = code & ((1 << f.mb) - 1);
  float v;
  if (f.kind == 1 && e == 15 && m == 7) return NAN;
  if (f.kind == 2 && e == 31) v = m ? NAN : INFINITY;
  else if (e == 0) v = std::ldexp((float)m, 1 - f.bias - f.mb);
  else v = std::ldexp((float)((1 << f.mb) | m), e - f.bias - f.mb);
  return neg ? -v : v;
}
static bool same(float a, float b) { return (std::isnan(a) && std::isnan(b)) || std::memcmp(&a, &b, 4) == 0; }

static uint32_t next_random() {
  static uint32_t seed = 987654321;
  return seed = seed * 1664525u + 1013904223u;
}
// A value of random sign and mantissa and a power of two between lo and hi, and a scale 2^k.
static float random_value(int lo, int hi, float mant_max = 2.0f, int cap = 100) {
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
template <class T> static std::vector<T> download(const T* d, size_t n) {
  std::vector<T> v(n);
  hipMemcpy(v.data(), d, n * sizeof(T), hipMemcpyDeviceToHost);
  return v;
}

constexpr int kCases = 1024;   // lanes: 16 waves of 64

template <fcbx::Encoding E, class T> static uint32_t ref(T x, int k) { return fcbx::from_float<T, E, true>(x, (int8_t)k); }

// Pairs: `hi` says which half or byte the pair goes to.
template <class T, fcbx::Encoding E, int BITS, class K>
static void check_pairs(const char* what, K kernel, int pos, int lo_exp, int hi_exp, float mant_max) {
  std::vector<T> x(2 * kCases);
  std::vector<float> s(kCases);
  std::vector<int> k(kCases);
  for (int i = 0; i < kCases; ++i) {
    k[i] = (int)(next_random() % 31) - 15;
    s[i] = scale_of(k[i]);
    // (A half holds up to 65504: the value is kept under 2^15 once scaled.)
    for (int j = 0; j < 2; ++j) x[2 * i + j] = (T)(random_value(lo_exp, hi_exp, mant_max, std::is_same_v<T, _Float16> ? 14 - k[i] : 100) * s[i]);
  }
  uint32_t* o;
  hipMalloc(&o, kCases * 4);
  T* dx = upload(x);
  float* ds = upload(s);
  if constexpr (std::is_same_v<T, float>) {
    std::vector<float> a(kCases), b(kCases);
    for (int i = 0; i < kCases; ++i) { a[i] = x[2 * i]; b[i] = x[2 * i + 1]; }
    float *da = upload(a), *db = upload(b);
    kernel<<<kCases / 64, 64>>>(da, db, ds, o);
  } else {
    kernel<<<kCases / 64, 64>>>(dx, ds, o);
  }
  const std::vector<uint32_t> got = download(o, kCases);
  int wrong = 0;
  for (int i = 0; i < kCases; ++i) {
    const uint32_t lo = ref<E>(x[2 * i], k[i]), hi = ref<E>(x[2 * i + 1], k[i]);
    uint32_t want;
    if (BITS == 8) {   // two bytes into the half `pos` names, the other half kept
      const uint32_t pair = lo | hi << 8;
      want = pos ? (0xA5A5A5A5u & 0xFFFFu) | pair << 16 : (0xA5A5A5A5u & 0xFFFF0000u) | pair;
    } else {           // two nibbles into the byte `pos` names
      const uint32_t byte = lo | hi << 4;
      want = (0xA5A5A5A5u & ~(0xFFu << (8 * pos))) | byte << (8 * pos);
    }
    if (got[i] != want && wrong++ < 3)
      std::printf("  %s: %a %a (scale 2^%d) gives 0x%08x, not 0x%08x\n", what, (float)x[2 * i], (float)x[2 * i + 1], k[i], got[i], want);
  }
  report(what, wrong, kCases);
}

// 32 six-bit values a lane, from f32 (as two arrays of 16), f16 or bf16.
template <class T, fcbx::Encoding E, class K>
static void check_wide(const char* what, K kernel, bool interleaved_f32) {
  std::vector<T> x(32 * kCases);
  std::vector<float> s(kCases);
  std::vector<int> k(kCases);
  for (int i = 0; i < kCases; ++i) {
    k[i] = (int)(next_random() % 21) - 10;
    s[i] = scale_of(k[i]);
    for (int j = 0; j < 32; ++j) {
      // Mostly in the range of the 6-bit formats once scaled; some far outside (they saturate).
      const int cap = std::is_same_v<T, _Float16> ? 14 - k[i] : 100;
      const float v = next_random() % 8 == 0 ? random_value(4, 9, 2.0f, cap) : random_value(-6, 3, 2.0f, cap);
      x[32 * i + j] = (T)(v * s[i]);
    }
  }
  uint32_t* o;
  hipMalloc(&o, 6 * kCases * 4);
  T* dx = upload(x);
  float* ds = upload(s);
  kernel<<<kCases / 64, 64>>>(dx, ds, o);
  const std::vector<uint32_t> got = download(o, 6 * kCases);
  int wrong = 0;
  for (int i = 0; i < kCases; ++i) {
    uint32_t want[6] = {};
    for (int j = 0; j < 32; ++j) {
      // 2xpk16 takes the even elements of a lane's 32 as its first array and the odd ones as its second, and the result puts them back
      // in order: element j of the result is element j.
      const uint32_t code = ref<E>(x[32 * i + j], k[i]);
      for (int b = 0; b < 6; ++b) want[(6 * j + b) / 32] |= (code >> b & 1) << ((6 * j + b) % 32);
    }
    (void)interleaved_f32;
    for (int r = 0; r < 6; ++r)
      if (got[6 * i + r] != want[r] && wrong++ < 3) std::printf("  %s: lane %d register %d is 0x%08x, not 0x%08x\n", what, i, r, got[6 * i + r], want[r]);
  }
  report(what, wrong, kCases);
}

// The table: the pair (x0, x1) with scale 2^k, narrowed to `fmt` and widened by the decoder.
struct Row { const char* what; const Fmt* fmt; float x0, x1; int k; float want0, want1; };
// Composable Kernel's expectations are quoted by file in the comment above; the answers are written out here from them.
static const Row kRows[] = {
    // test_mx_fp8.cpp, "f32x2 -> f8x2 RNE, expect {-4, 2}" with scale 2.0
    {"fp8 {-8, 4} / 2", &FP8, -8.0f, 4.0f, 1, -4.0f, 2.0f},
    {"bf8 {-8, 4} / 2", &BF8, -8.0f, 4.0f, 1, -4.0f, 2.0f},
    // mxf8_convert_rne<f8_ocp_t>(1024.0f, 4.0f) = 256
    {"fp8 1024 / 4", &FP8, 1024.0f, 1.0f, 2, 256.0f, 0.25f},
    // "10000 / 32 = 312.5" rounds to the fp8 nearest it, 320
    {"fp8 10000 / 32", &FP8, 10000.0f, 1.0f, 5, 320.0f, 0.03125f},
    // NaN is NaN; "Inf/2 > 448 => NaN on device", "256/0.5 > 448 => NaN on device", "-256/0.5 < -448 => NaN on device"
    {"fp8 NaN", &FP8, NAN, 0.0f, 2, NAN, 0.0f},
    {"fp8 +-Inf / 2", &FP8, INFINITY, -INFINITY, 1, NAN, NAN},
    {"fp8 256 / 0.5, -256 / 0.5", &FP8, 256.0f, -256.0f, -1, NAN, NAN},
    // the pk8 row, divided by 4 first: {2, -4, 512, -512, NaN, Inf, 2^-10, 0.04} -> {.5, -1, 128, -128, NaN, NaN, 0, 0.01 as 5 * 2^-9}
    {"fp8 {2, -4} / 4", &FP8, 2.0f, -4.0f, 2, 0.5f, -1.0f},
    {"fp8 {512, -512} / 4", &FP8, 512.0f, -512.0f, 2, 128.0f, -128.0f},
    {"fp8 {2^-10, 0.04} / 4", &FP8, 0x1p-10f, 0.04f, 2, 0.0f, 0.009765625f},
    // a negative zero stays one (the OCP formats have it)
    {"fp8 -1e-5, 1e-5", &FP8, -1e-5f, 1e-5f, 0, -0.0f, 0.0f},
    // bf8 keeps its infinities: "If fval / scale > max bf8, returns Inf"; the guide's FP8/BF8 table (FP16_OVFL = 0): +-Inf, and NaN
    {"bf8 +-1e5", &BF8, 1e5f, -1e5f, 0, INFINITY, -INFINITY},
    {"bf8 Inf, NaN", &BF8, INFINITY, NAN, 1, INFINITY, NAN},
    // test_mx_fp4.cpp: 24 / 4 = 6; "Inf/2 > 6.0 => 6.0 on device", "256/0.5 > 6.0 => 6.0", "-256/0.5 < -6.0 => -6.0"; 5.0 is halfway between 4 and 6
    {"fp4 {24, -24} / 4", &FP4, 24.0f, -24.0f, 2, 6.0f, -6.0f},
    {"fp4 +-Inf / 2", &FP4, INFINITY, -INFINITY, 1, 6.0f, -6.0f},
    {"fp4 {256, -256} / 0.5", &FP4, 256.0f, -256.0f, -1, 6.0f, -6.0f},
    {"fp4 {5, -5}: ties to even", &FP4, 5.0f, -5.0f, 0, 4.0f, -4.0f},
    // E2M1: 0.25 is halfway between 0 and 0.5 (even: 0), 0.75 between 0.5 and 1 (even: 1), 2.5 between 2 and 3 (even: 2)
    {"fp4 {0.75, 2.5}", &FP4, 0.75f, 2.5f, 0, 1.0f, 2.0f},
    {"fp4 {-0.1, 0.1}", &FP4, -0.1f, 0.1f, 0, -0.0f, 0.0f},
    // test_mx_fp6.cpp: 4.75 / 4 = 1.1875, halfway between 1.125 and 1.25; infinity, 256 / 0.5 and -256 / 0.5 are the largest values (7.5)
    {"fp6 {4.75, -4.75} / 4", &FP6, 4.75f, -4.75f, 2, 1.25f, -1.25f},
    {"fp6 +-Inf", &FP6, INFINITY, -INFINITY, 0, 7.5f, -7.5f},
    {"fp6 {256, -256} / 0.5", &FP6, 256.0f, -256.0f, -1, 7.5f, -7.5f},
    // E3M2: largest 28, smallest subnormal 0.0625 (0.2 = 3.2 of them, 0.09 = 1.44)
    {"bf6 {30, -30}", &BF6, 30.0f, -30.0f, 0, 28.0f, -28.0f},
    {"bf6 {0.2, 0.09}", &BF6, 0.2f, 0.09f, 0, 0.1875f, 0.0625f},
};

// The rows through the pk_*_f32 builtins (fp8, bf8, fp4), or through 2xpk16 (fp6, bf6): element 0 and 1 are the pair.
__global__ void k_row_fp8(const float* x, const float* s, uint32_t* o) {
  const int i = threadIdx.x; v2s old; old[0] = old[1] = 0;
  v2s r = __builtin_amdgcn_cvt_scalef32_pk_fp8_f32(old, x[2 * i], x[2 * i + 1], s[i], false);
  o[i] = (uint32_t)(uint16_t)r[0];
}
__global__ void k_row_bf8(const float* x, const float* s, uint32_t* o) {
  const int i = threadIdx.x; v2s old; old[0] = old[1] = 0;
  v2s r = __builtin_amdgcn_cvt_scalef32_pk_bf8_f32(old, x[2 * i], x[2 * i + 1], s[i], false);
  o[i] = (uint32_t)(uint16_t)r[0];
}
__global__ void k_row_fp4(const float* x, const float* s, uint32_t* o) {
  const int i = threadIdx.x;
  o[i] = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(0, x[2 * i], x[2 * i + 1], s[i], 0) & 0xFF;
}
__global__ void k_row_fp6(const float* x, const float* s, uint32_t* o) {
  const int i = threadIdx.x; v16f lo, hi;
  for (int j = 0; j < 16; ++j) lo[j] = hi[j] = 0.0f;
  lo[0] = x[2 * i]; hi[0] = x[2 * i + 1];
  v6u r = __builtin_amdgcn_cvt_scalef32_2xpk16_fp6_f32(lo, hi, s[i]);
  o[i] = r[0] & 0xFFF;
}
__global__ void k_row_bf6(const float* x, const float* s, uint32_t* o) {
  const int i = threadIdx.x; v16f lo, hi;
  for (int j = 0; j < 16; ++j) lo[j] = hi[j] = 0.0f;
  lo[0] = x[2 * i]; hi[0] = x[2 * i + 1];
  v6u r = __builtin_amdgcn_cvt_scalef32_2xpk16_bf6_f32(lo, hi, s[i]);
  o[i] = r[0] & 0xFFF;
}

static void check_rows() {
  const int n = (int)(sizeof(kRows) / sizeof(kRows[0]));
  uint32_t* o;
  hipMalloc(&o, 64 * 4);
  int wrong = 0;
  const auto run = [&](auto kernel, const Fmt& f, int bits) {
    // Only this format's rows are given values (a NaN into fp4 is refused, and 0 is harmless).
    std::vector<float> x(2 * 64, 0.0f), s(64, 1.0f);
    for (int i = 0; i < n; ++i)
      if (kRows[i].fmt == &f) { x[2 * i] = kRows[i].x0; x[2 * i + 1] = kRows[i].x1; s[i] = scale_of(kRows[i].k); }
    float *dx = upload(x), *ds = upload(s);
    kernel<<<1, 64>>>(dx, ds, o);
    const std::vector<uint32_t> got = download(o, 64);
    for (int i = 0; i < n; ++i) {
      if (kRows[i].fmt != &f) continue;
      const uint32_t m = (1u << bits) - 1;
      const float g0 = decode(f, got[i] & m), g1 = decode(f, got[i] >> bits & m);
      if (!(same(g0, kRows[i].want0) && same(g1, kRows[i].want1)) && wrong++ < 8)
        std::printf("  %s: gives {%g, %g} (codes 0x%x 0x%x), not {%g, %g}\n", kRows[i].what, g0, g1, got[i] & m, got[i] >> bits & m, kRows[i].want0, kRows[i].want1);
    }
  };
  run(k_row_fp8, FP8, 8);
  run(k_row_bf8, BF8, 8);
  run(k_row_fp4, FP4, 4);
  run(k_row_fp6, FP6, 6);
  run(k_row_bf6, BF6, 6);
  report("the cases Composable Kernel's tests run on a gfx950", wrong, n);
}

// v_cvt_scalef32_f16_fp8 / _bf8: every byte, three scales, the byte and the half chosen, the other half kept.
template <class K>
static void check_widen_half(const char* what, K kernel, const Fmt& f, int byte, bool hi, const int (&scales)[3]) {
  std::vector<uint32_t> a(256 * 3);
  std::vector<float> s(256 * 3);
  std::vector<int> ks(256 * 3);
  for (int j = 0; j < 3; ++j)
    for (int c = 0; c < 256; ++c) {
      // The wanted byte holds the code and the others hold other bytes.
      a[j * 256 + c] = (0x3C5A9Bu * (c + 1) & ~(0xFFu << (8 * byte))) | (uint32_t)c << (8 * byte);
      ks[j * 256 + c] = scales[j];
      s[j * 256 + c] = scale_of(scales[j]);
    }
  uint32_t* o;
  hipMalloc(&o, a.size() * 4);
  uint32_t* da = upload(a);
  float* ds = upload(s);
  kernel<<<(int)a.size() / 64, 64>>>(da, ds, o);
  const std::vector<uint32_t> got = download(o, a.size());
  int wrong = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    const float v = decode(f, a[i] >> (8 * byte) & 0xFF);
    const _Float16 h = (_Float16)std::ldexp(v, ks[i]);
    uint16_t hb;
    std::memcpy(&hb, &h, 2);
    const uint32_t kept = 0xA5A5u, mine = got[i] >> (hi ? 16 : 0) & 0xFFFF, other = got[i] >> (hi ? 0 : 16) & 0xFFFF;
    // (A NaN is any NaN.)
    const bool right = other == kept && (std::isnan(v) ? (mine & 0x7C00) == 0x7C00 && (mine & 0x3FF) != 0 : mine == hb);
    if (!right && wrong++ < 3) std::printf("  %s: byte 0x%02x (scale 2^%d) gives 0x%08x, not 0x%04x with 0xa5a5 kept\n", what, a[i] >> (8 * byte) & 0xFF, ks[i], got[i], hb);
  }
  report(what, wrong, (int)a.size());
}

// (Scales that keep the result inside a half: fp8 reaches 448, bf8 57344.)
static const int kScalesFp8[3] = {0, -9, 5}, kScalesBf8[3] = {0, -9, 0};

int main() {
  // The sweeps. Values go through the 8-bit formats inside their range (below fp8's 448 and bf8's 57344, with the mantissa capped where the
  // exponent is the top one), and through the 4- and 6-bit ones well past theirs (they saturate), once scaled by 2^k.
  check_pairs<float, fcbx::Encoding::E4M3Mx, 8>("v_cvt_scalef32_pk_fp8_f32 (low half)", k_pk_fp8_f32_lo, 0, -12, 8, 1.7f);
  check_pairs<float, fcbx::Encoding::E4M3Mx, 8>("v_cvt_scalef32_pk_fp8_f32 (high half)", k_pk_fp8_f32_hi, 1, -12, 8, 1.7f);
  check_pairs<float, fcbx::Encoding::E5M2Mx, 8>("v_cvt_scalef32_pk_bf8_f32 (low half)", k_pk_bf8_f32_lo, 0, -18, 14, 1.9f);
  check_pairs<float, fcbx::Encoding::E5M2Mx, 8>("v_cvt_scalef32_pk_bf8_f32 (high half)", k_pk_bf8_f32_hi, 1, -18, 14, 1.9f);
  check_pairs<_Float16, fcbx::Encoding::E4M3Mx, 8>("v_cvt_scalef32_pk_fp8_f16 (low half)", k_pk_fp8_f16_lo, 0, -8, 7, 1.7f);
  check_pairs<_Float16, fcbx::Encoding::E4M3Mx, 8>("v_cvt_scalef32_pk_fp8_f16 (high half)", k_pk_fp8_f16_hi, 1, -8, 7, 1.7f);
  check_pairs<_Float16, fcbx::Encoding::E5M2Mx, 8>("v_cvt_scalef32_pk_bf8_f16 (low half)", k_pk_bf8_f16_lo, 0, -14, 13, 1.9f);
  check_pairs<_Float16, fcbx::Encoding::E5M2Mx, 8>("v_cvt_scalef32_pk_bf8_f16 (high half)", k_pk_bf8_f16_hi, 1, -14, 13, 1.9f);
  check_pairs<__bf16, fcbx::Encoding::E4M3Mx, 8>("v_cvt_scalef32_pk_fp8_bf16 (low half)", k_pk_fp8_bf16_lo, 0, -12, 8, 1.7f);
  check_pairs<__bf16, fcbx::Encoding::E4M3Mx, 8>("v_cvt_scalef32_pk_fp8_bf16 (high half)", k_pk_fp8_bf16_hi, 1, -12, 8, 1.7f);
  check_pairs<__bf16, fcbx::Encoding::E5M2Mx, 8>("v_cvt_scalef32_pk_bf8_bf16 (low half)", k_pk_bf8_bf16_lo, 0, -18, 14, 1.9f);
  check_pairs<__bf16, fcbx::Encoding::E5M2Mx, 8>("v_cvt_scalef32_pk_bf8_bf16 (high half)", k_pk_bf8_bf16_hi, 1, -18, 14, 1.9f);
  check_pairs<float, fcbx::Encoding::E2M1, 4>("v_cvt_scalef32_pk_fp4_f32 (byte 0)", k_pk_fp4_f32_0, 0, -4, 5, 2.0f);
  check_pairs<float, fcbx::Encoding::E2M1, 4>("v_cvt_scalef32_pk_fp4_f32 (byte 1)", k_pk_fp4_f32_1, 1, -4, 5, 2.0f);
  check_pairs<float, fcbx::Encoding::E2M1, 4>("v_cvt_scalef32_pk_fp4_f32 (byte 2)", k_pk_fp4_f32_2, 2, -4, 5, 2.0f);
  check_pairs<float, fcbx::Encoding::E2M1, 4>("v_cvt_scalef32_pk_fp4_f32 (byte 3)", k_pk_fp4_f32_3, 3, -4, 5, 2.0f);
  check_pairs<_Float16, fcbx::Encoding::E2M1, 4>("v_cvt_scalef32_pk_fp4_f16 (byte 0)", k_pk_fp4_f16_0, 0, -4, 5, 2.0f);
  check_pairs<_Float16, fcbx::Encoding::E2M1, 4>("v_cvt_scalef32_pk_fp4_f16 (byte 2)", k_pk_fp4_f16_2, 2, -4, 5, 2.0f);
  check_pairs<__bf16, fcbx::Encoding::E2M1, 4>("v_cvt_scalef32_pk_fp4_bf16 (byte 1)", k_pk_fp4_bf16_1, 1, -4, 5, 2.0f);
  check_pairs<__bf16, fcbx::Encoding::E2M1, 4>("v_cvt_scalef32_pk_fp4_bf16 (byte 3)", k_pk_fp4_bf16_3, 3, -4, 5, 2.0f);
  check_wide<float, fcbx::Encoding::E2M3>("v_cvt_scalef32_2xpk16_fp6_f32", k_2xpk16_fp6_f32, true);
  check_wide<float, fcbx::Encoding::E3M2>("v_cvt_scalef32_2xpk16_bf6_f32", k_2xpk16_bf6_f32, true);
  check_wide<_Float16, fcbx::Encoding::E2M3>("v_cvt_scalef32_pk32_fp6_f16", k_pk32_fp6_f16, false);
  check_wide<_Float16, fcbx::Encoding::E3M2>("v_cvt_scalef32_pk32_bf6_f16", k_pk32_bf6_f16, false);
  check_wide<__bf16, fcbx::Encoding::E2M3>("v_cvt_scalef32_pk32_fp6_bf16", k_pk32_fp6_bf16, false);
  check_wide<__bf16, fcbx::Encoding::E3M2>("v_cvt_scalef32_pk32_bf6_bf16", k_pk32_bf6_bf16, false);
  check_rows();
  check_widen_half("v_cvt_scalef32_f16_fp8 (byte 0, low half)", k_f16_fp8_b0_lo, FP8, 0, false, kScalesFp8);
  check_widen_half("v_cvt_scalef32_f16_fp8 (byte 1, high half)", k_f16_fp8_b1_hi, FP8, 1, true, kScalesFp8);
  check_widen_half("v_cvt_scalef32_f16_fp8 (byte 2, low half)", k_f16_fp8_b2_lo, FP8, 2, false, kScalesFp8);
  check_widen_half("v_cvt_scalef32_f16_fp8 (byte 3, high half)", k_f16_fp8_b3_hi, FP8, 3, true, kScalesFp8);
  check_widen_half("v_cvt_scalef32_f16_bf8 (byte 0, high half)", k_f16_bf8_b0_hi, BF8, 0, true, kScalesBf8);
  check_widen_half("v_cvt_scalef32_f16_bf8 (byte 3, low half)", k_f16_bf8_b3_lo, BF8, 3, false, kScalesBf8);
  std::printf("scalef32_narrow: %d failed\n", failed);
  return failed != 0;
}
