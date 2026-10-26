// The arithmetic MFMA and WMMA instructions share, in the form the interpreter runs it. Kept in a header so a unit
// test can hold it against the straightforward version it replaced, bit for bit (tests/unit/test_amd_mfma_math.cpp).
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

#include "amd_simd_cap.hpp"
#if defined(__SSE2__)
#include <emmintrin.h>
#endif
#if defined(__x86_64__) && defined(__GNUC__)
#include <immintrin.h>
#endif

namespace vgpu::amd {

// The double a binary16 value (its 16 bits) converts to: exact, as `static_cast<double>(_Float16)` is. A signalling
// NaN comes out quiet, as the conversion makes it, with the payload shifted up. Integer work only: the baseline x86-64
// has no F16C, so the compiler's conversion is a call into libgcc, which was a third of an MFMA's time.
inline double half_bits_to_double(uint16_t h) {
  const uint64_t sign = uint64_t{h & 0x8000u} << 48;
  const uint32_t e = (h >> 10) & 0x1F, m = h & 0x3FF;
  uint64_t bits;
  if (e == 0x1F) {   // infinity or NaN
    bits = sign | 0x7FF0000000000000ull | (uint64_t{m} << 42) | (m ? 1ull << 51 : 0);
  } else if (e) {    // normal: rebias 15 -> 1023
    bits = sign | (uint64_t{e + 1008} << 52) | (uint64_t{m} << 42);
  } else if (!m) {   // zero
    bits = sign;
  } else {           // subnormal: m * 2^-24, normalised
    const uint32_t top = 31 - static_cast<uint32_t>(__builtin_clz(m));   // 0..9
    bits = sign | (uint64_t{top + 999} << 52) | (uint64_t{(m << (10 - top)) & 0x3FF} << 42);
  }
  double d;
  std::memcpy(&d, &bits, 8);
  return d;
}


#if defined(__SSE2__)
// Four binary16 values (each in the low 16 bits of a 32-bit lane of `u`, the rest zero) as four doubles, in `lo` (the first two)
// and `hi`: the same doubles half_bits_to_double gives, NaN payloads and the quiet bit included. Done in float, where every
// binary16 value is exact: a normal number is a rebias of its exponent and a shift of its mantissa, a subnormal is its mantissa
// times 2^-24, and an infinity or NaN keeps its payload (shifted) under the all-ones exponent; the float to double conversion
// then widens it (making a signalling NaN quiet, as half_bits_to_double does).
inline void half4_to_double(__m128i u, __m128d* lo, __m128d* hi) {
  const __m128i sign = _mm_slli_epi32(_mm_and_si128(u, _mm_set1_epi32(0x8000)), 16);
  const __m128i em = _mm_and_si128(u, _mm_set1_epi32(0x7FFF));
  const __m128i e = _mm_srli_epi32(em, 10);
  // normal (and, with the exponent corrected below, infinity and NaN): (em << 13) + (112 << 23)
  __m128i bits = _mm_add_epi32(_mm_slli_epi32(em, 13), _mm_set1_epi32(112 << 23));
  bits = _mm_add_epi32(bits, _mm_and_si128(_mm_cmpeq_epi32(e, _mm_set1_epi32(31)), _mm_set1_epi32(112 << 23)));
  // zero and subnormal: mantissa * 2^-24 (exact)
  const __m128 sub = _mm_mul_ps(_mm_cvtepi32_ps(_mm_and_si128(em, _mm_set1_epi32(0x3FF))), _mm_set1_ps(5.9604644775390625e-08f));
  const __m128i is_sub = _mm_cmpeq_epi32(e, _mm_setzero_si128());
  bits = _mm_or_si128(_mm_and_si128(is_sub, _mm_castps_si128(sub)), _mm_andnot_si128(is_sub, bits));
  const __m128 f = _mm_castsi128_ps(_mm_or_si128(bits, sign));
  *lo = _mm_cvtps_pd(f);
  *hi = _mm_cvtps_pd(_mm_movehl_ps(f, f));
}
#endif

#if defined(__x86_64__) && defined(__GNUC__)
// Eight binary16 values (each in the low 16 bits of a 32-bit lane of `u`, the rest zero) as eight doubles, by F16C: the same doubles
// half_bits_to_double gives (the conversion is exact, makes a signalling NaN quiet and shifts its payload up, whatever MXCSR says).
__attribute__((target("avx2,f16c"))) inline void half8_to_double(__m256i u, __m256d* lo, __m256d* hi) {
  // (pack the dwords to words; the two 128-bit halves' values are then in the two ends of the 256-bit result, so bring them together)
  const __m256i packed = _mm256_permute4x64_epi64(_mm256_packus_epi32(u, u), 0xD8);
  const __m256 f = _mm256_cvtph_ps(_mm256_castsi256_si128(packed));
  *lo = _mm256_cvtps_pd(_mm256_castps256_ps128(f));
  *hi = _mm256_cvtps_pd(_mm256_extractf128_ps(f, 1));
}
#endif

// A product and a sum with the NaN rule of the machine spelled out. Where both operands of an x86 SSE multiply or add are
// NaNs the result is the first one (the destination register's), and which of two commutative operands the compiler puts
// there is its choice -- one that differs between loops, and between a scalar and a vectorized form of the same loop. The
// recorded results of the instructions (test_amd_gcn_same_bits) have NaN payloads that depend on it, so the order is fixed
// here: the sum is the first operand of the add, B the first of the multiply, and the fused form takes B first as well
// (found by trying the combinations against the recorded digests; random operands with NaNs among them, which the earlier
// test data -- no NaNs, or one quiet NaN -- could not see).
#ifndef MFMA_MUL_FIRST_A
#define MFMA_MUL_FIRST_A 0
#endif
#ifndef MFMA_ADD_FIRST_SUM
#define MFMA_ADD_FIRST_SUM 1
#endif
inline double mfma_mul(double a, double b) {
#if defined(__x86_64__)
#if MFMA_MUL_FIRST_A
  __asm__("mulsd %1, %0" : "+x"(a) : "x"(b));
  return a;
#else
  __asm__("mulsd %1, %0" : "+x"(b) : "x"(a));
  return b;
#endif
#else
  return a * b;
#endif
}
inline double mfma_add(double sum, double p) {
#if defined(__x86_64__)
#if MFMA_ADD_FIRST_SUM
  __asm__("addsd %1, %0" : "+x"(sum) : "x"(p));
  return sum;
#else
  __asm__("addsd %1, %0" : "+x"(p) : "x"(sum));
  return p;
#endif
#else
  return sum + p;
#endif
}

// A value times the scale of the block-scaled forms (or 1.0). Where both are NaNs the product is the first operand's payload, which
// the compiler would choose, so the order is fixed (the reference path and the tile path both use this).
#ifndef MFMA_SCALE_FIRST_V
#define MFMA_SCALE_FIRST_V 1
#endif
inline double mfma_scale_mul(double v, double s) {
#if defined(__x86_64__)
#if MFMA_SCALE_FIRST_V
  __asm__("mulsd %1, %0" : "+x"(v) : "x"(s));
  return v;
#else
  __asm__("mulsd %1, %0" : "+x"(s) : "x"(v));
  return s;
#endif
#else
  return v * s;
#endif
}

// One output of a matrix multiply-accumulate: c plus the dot product of a row of A with a column of B, k terms taken in
// order and summed in double (so the order is the result). `bt` is that column, stored contiguously.
//
// The plain loops are what the compiler may schedule and vectorize as it likes; they give the same bits as any other order
// of the two operands of a multiply or an add, except when both are NaNs (the result then takes one operand's payload, and
// which is the compiler's choice). A NaN stays in a sum once it is there, so a sum that ends as a number never met one, and
// one that ends as a NaN is done again below with the operand order fixed above.
inline double mfma_dot_ordered(const double* a, const double* bt, uint32_t k, double c) {
  double sum = c;
  for (uint32_t i = 0; i < k; ++i) sum = mfma_add(sum, mfma_mul(a[i], bt[i]));
  return sum;
}
inline double mfma_dot(const double* a, const double* bt, uint32_t k, double c) {
  double sum = c;
  for (uint32_t i = 0; i < k; ++i) sum += a[i] * bt[i];
  return sum != sum ? mfma_dot_ordered(a, bt, k, c) : sum;
}
// Four of them at once, from four rows and four columns: four sums that nothing joins, each taken in order, so each
// is what mfma_dot gives; they overlap in the machine instead of waiting on one another.
inline void mfma_dot4(const double* const a[4], const double* const bt[4], uint32_t k, const double c[4], double out[4]) {
  double s0 = c[0], s1 = c[1], s2 = c[2], s3 = c[3];
  for (uint32_t i = 0; i < k; ++i) {
    s0 += a[0][i] * bt[0][i];
    s1 += a[1][i] * bt[1][i];
    s2 += a[2][i] * bt[2][i];
    s3 += a[3][i] * bt[3][i];
  }
  out[0] = s0 != s0 ? mfma_dot_ordered(a[0], bt[0], k, c[0]) : s0;
  out[1] = s1 != s1 ? mfma_dot_ordered(a[1], bt[1], k, c[1]) : s1;
  out[2] = s2 != s2 ? mfma_dot_ordered(a[2], bt[2], k, c[2]) : s2;
  out[3] = s3 != s3 ? mfma_dot_ordered(a[3], bt[3], k, c[3]) : s3;
}
// The fused form, for the double-precision instruction. fma(x, y, z) is the same for any order of x and y unless two of the
// operands are NaNs, when the one whose payload comes through is the hardware's choice by operand position and the compiler's
// choice of which of x and y is first -- it swaps the two freely, differently in two copies of the same loop (a unit test found
// it). So where an operand is a NaN the result is spelled out: the first NaN of B, A and the sum, made quiet; a NaN made by the
// operation itself (infinity times zero, infinities of opposite signs) is the default NaN, which the fma gives.
#ifndef MFMA_FMA_FIRST_B
#define MFMA_FMA_FIRST_B 1
#endif
inline double mfma_quiet(double x) {
  uint64_t b;
  std::memcpy(&b, &x, 8);
  b |= 1ull << 51;
  std::memcpy(&x, &b, 8);
  return x;
}
inline double mfma_fma(double b, double a, double sum) {
  const double r = std::fma(b, a, sum);
  if (r == r) return r;
#if MFMA_FMA_FIRST_B
  if (b != b) return mfma_quiet(b);
  if (a != a) return mfma_quiet(a);
#else
  if (a != a) return mfma_quiet(a);
  if (b != b) return mfma_quiet(b);
#endif
  if (sum != sum) return mfma_quiet(sum);
  return r;
}
inline double mfma_dot_fma(const double* a, const double* bt, uint32_t k, double c) {
  double sum = c;
  for (uint32_t i = 0; i < k; ++i) sum = mfma_fma(bt[i], a[i], sum);
  return sum;
}
// acc + a * b in 64 bits, wrapping as the hardware's integer adders do (done unsigned: the signed overflow is undefined, and a sanitizer says so).
inline int64_t wrap_madd(int64_t acc, int64_t a, int64_t b) {
  return static_cast<int64_t>(static_cast<uint64_t>(acc) + static_cast<uint64_t>(a) * static_cast<uint64_t>(b));
}
// Integers are summed exactly and wrap as the hardware's do.
inline int64_t mfma_dot_int(const double* a, const double* bt, uint32_t k, int64_t c) {
  int64_t sum = c;
  for (uint32_t i = 0; i < k; ++i) sum = wrap_madd(sum, static_cast<int64_t>(a[i]), static_cast<int64_t>(bt[i]));
  return sum;
}

// ---- whole tiles at once ----------------------------------------------------------------------------------------------
//
// The fast path of the matrix instructions works on a whole tile: out = c + a x b for an m by n block of outputs and k terms,
// rather than one dot product at a time. The sums are the ones above (a double per output, the terms taken in order of k,
// the product rounded in double before it is added); only the outputs that are worked out together change, so each is
// bit for bit what mfma_dot gives, except where the result is a NaN, which is done again in the fixed operand order
// (mfma_dot_ordered), as before.
//
//   a  m rows of k (row-major), bk  k rows of n (B by row of k, so that the outputs of one row of the tile are adjacent),
//   c and out  m rows of n.   m and n are multiples of 4.
typedef double mfma_v2d __attribute__((vector_size(16), aligned(8)));

inline mfma_v2d mfma_ld2(const double* p) { return *reinterpret_cast<const mfma_v2d*>(p); }
inline void mfma_st2(double* p, mfma_v2d v) { *reinterpret_cast<mfma_v2d*>(p) = v; }
inline mfma_v2d mfma_splat(double x) { return mfma_v2d{x, x}; }

// The NaN rule of a product and a sum for either choice of the first operand (see mfma_mul / mfma_add above, which are the
// choice the MFMA instructions were recorded with): bit 0 of Order says the first operand of the multiply is A rather than B,
// bit 1 that the first operand of the add is the product rather than the sum. The WMMA instructions' recorded results were
// made with another choice (found, as before, by trying them against the digests).
template <int Order>
inline double mfma_mul_o(double a, double b) {
#if defined(__x86_64__)
  if (Order & 1) {
    __asm__("mulsd %1, %0" : "+x"(a) : "x"(b));
    return a;
  }
  __asm__("mulsd %1, %0" : "+x"(b) : "x"(a));
  return b;
#else
  return a * b;
#endif
}
template <int Order>
inline double mfma_add_o(double sum, double p) {
#if defined(__x86_64__)
  if (Order & 2) {
    __asm__("addsd %1, %0" : "+x"(p) : "x"(sum));
    return p;
  }
  __asm__("addsd %1, %0" : "+x"(sum) : "x"(p));
  return sum;
#else
  return sum + p;
#endif
}
#ifndef WMMA_ORDER
#define WMMA_ORDER 2   /* the multiply takes B first, the add the product first */
#endif
constexpr int kMfmaOrder = (MFMA_MUL_FIRST_A ? 1 : 0) | (MFMA_ADD_FIRST_SUM ? 0 : 2);

// One output, in the fixed operand order, from the tile's layouts (b read down its column at stride n).
template <int Order>
inline double mfma_tile_ordered(const double* arow, const double* bcol, uint32_t n, uint32_t k, double c) {
  double sum = c;
  for (uint32_t t = 0; t < k; ++t) sum = mfma_add_o<Order>(sum, mfma_mul_o<Order>(arow[t], bcol[t * n]));
  return sum;
}
#if defined(__x86_64__) && defined(__GNUC__)
// Whether the 4-wide kernels are used: the host has AVX2 (a test turns it off to hold the 2-wide ones against the same cases).
inline bool& mfma_avx2_switch() {
  static bool on = !simd_capped_to_base() && __builtin_cpu_supports("avx2");
  return on;
}
inline bool mfma_have_avx2() { return mfma_avx2_switch(); }
// The same sums with four doubles to an instruction (AVX2, chosen when the host has it; no fused multiply-add: a product and a
// sum are two roundings, as in the 2-wide kernel, so the bits are the same). n is a multiple of 8.
typedef double mfma_v4d __attribute__((vector_size(32), aligned(8)));
__attribute__((target("avx2"))) inline void mfma_tile_f_avx2(uint32_t m, uint32_t n, uint32_t k, const double* a, const double* bk,
                                                            const double* c, double* out) {
#define MFMA_LD4(p) (*reinterpret_cast<const mfma_v4d*>(p))
#define MFMA_ST4(p, v) (*reinterpret_cast<mfma_v4d*>(p) = (v))
  for (uint32_t i = 0; i < m; i += 4) {
    const double* const a0 = a + size_t{i} * k;
    const double* const a1 = a0 + k;
    const double* const a2 = a1 + k;
    const double* const a3 = a2 + k;
    for (uint32_t j = 0; j < n; j += 8) {
      const double* const cr = c + size_t{i} * n + j;
      mfma_v4d s00 = MFMA_LD4(cr), s01 = MFMA_LD4(cr + 4), s10 = MFMA_LD4(cr + n), s11 = MFMA_LD4(cr + n + 4),
               s20 = MFMA_LD4(cr + 2 * n), s21 = MFMA_LD4(cr + 2 * n + 4), s30 = MFMA_LD4(cr + 3 * n), s31 = MFMA_LD4(cr + 3 * n + 4);
      const double* b = bk + j;
      for (uint32_t t = 0; t < k; ++t, b += n) {
        const mfma_v4d b0 = MFMA_LD4(b), b1 = MFMA_LD4(b + 4);
        const mfma_v4d x0 = {a0[t], a0[t], a0[t], a0[t]}, x1 = {a1[t], a1[t], a1[t], a1[t]}, x2 = {a2[t], a2[t], a2[t], a2[t]},
                       x3 = {a3[t], a3[t], a3[t], a3[t]};
        s00 += x0 * b0, s01 += x0 * b1;
        s10 += x1 * b0, s11 += x1 * b1;
        s20 += x2 * b0, s21 += x2 * b1;
        s30 += x3 * b0, s31 += x3 * b1;
      }
      double* const o = out + size_t{i} * n + j;
      MFMA_ST4(o, s00), MFMA_ST4(o + 4, s01), MFMA_ST4(o + n, s10), MFMA_ST4(o + n + 4, s11);
      MFMA_ST4(o + 2 * n, s20), MFMA_ST4(o + 2 * n + 4, s21), MFMA_ST4(o + 3 * n, s30), MFMA_ST4(o + 3 * n + 4, s31);
    }
  }
#undef MFMA_LD4
#undef MFMA_ST4
}
#endif
inline void mfma_tile_f_sse(uint32_t m, uint32_t n, uint32_t k, const double* a, const double* bk, const double* c, double* out) {
  for (uint32_t i = 0; i < m; i += 4) {
    const double* const a0 = a + size_t{i} * k;
    const double* const a1 = a0 + k;
    const double* const a2 = a1 + k;
    const double* const a3 = a2 + k;
    for (uint32_t j = 0; j < n; j += 4) {
      const double* const cr = c + size_t{i} * n + j;
      mfma_v2d s00 = mfma_ld2(cr), s01 = mfma_ld2(cr + 2), s10 = mfma_ld2(cr + n), s11 = mfma_ld2(cr + n + 2),
               s20 = mfma_ld2(cr + 2 * n), s21 = mfma_ld2(cr + 2 * n + 2), s30 = mfma_ld2(cr + 3 * n), s31 = mfma_ld2(cr + 3 * n + 2);
      const double* b = bk + j;
      for (uint32_t t = 0; t < k; ++t, b += n) {
        const mfma_v2d b0 = mfma_ld2(b), b1 = mfma_ld2(b + 2);
        const mfma_v2d x0 = mfma_splat(a0[t]), x1 = mfma_splat(a1[t]), x2 = mfma_splat(a2[t]), x3 = mfma_splat(a3[t]);
        s00 += x0 * b0, s01 += x0 * b1;
        s10 += x1 * b0, s11 += x1 * b1;
        s20 += x2 * b0, s21 += x2 * b1;
        s30 += x3 * b0, s31 += x3 * b1;
      }
      double* const o = out + size_t{i} * n + j;
      mfma_st2(o, s00), mfma_st2(o + 2, s01), mfma_st2(o + n, s10), mfma_st2(o + n + 2, s11);
      mfma_st2(o + 2 * n, s20), mfma_st2(o + 2 * n + 2, s21), mfma_st2(o + 3 * n, s30), mfma_st2(o + 3 * n + 2, s31);
    }
  }
}
template <int Order = kMfmaOrder>
inline void mfma_tile_f(uint32_t m, uint32_t n, uint32_t k, const double* a, const double* bk, const double* c, double* out) {
#if defined(__x86_64__) && defined(__GNUC__)
  if (n % 8 == 0 && m % 4 == 0 && mfma_have_avx2()) mfma_tile_f_avx2(m, n, k, a, bk, c, out);
  else
#endif
    mfma_tile_f_sse(m, n, k, a, bk, c, out);
  // A sum that is a NaN was formed with whichever operand order the compiler chose; do it again in the fixed one.
  unsigned any_nan = 0;
  for (size_t e = 0; e < size_t{m} * n; ++e) any_nan |= out[e] != out[e];
  if (any_nan)
    for (uint32_t i = 0; i < m; ++i)
      for (uint32_t j = 0; j < n; ++j) {
        double& x = out[size_t{i} * n + j];
        if (x != x) x = mfma_tile_ordered<Order>(a + size_t{i} * k, bk + j, n, k, c[size_t{i} * n + j]);
      }
}

// The fused form (the double-precision instruction): one fma per term, taken in order. The result of a fused multiply-add
// is defined to the bit except for which NaN a NaN operand passes on, so a NaN result is again done the way it always was.
#if defined(__x86_64__) && defined(__GNUC__)
__attribute__((target("fma"))) inline void mfma_tile_fma_hw(uint32_t m, uint32_t n, uint32_t k, const double* a, const double* bk,
                                                           const double* c, double* out) {
  if (k == 4) {
    // The double-precision matrix instructions' K: four terms, the fused multiply-adds one after another on one sum, written out so that
    // the loop has no trip count to run: its few instructions were three times as slow in some positions of the code as in others.
    for (uint32_t i = 0; i < m; ++i) {
      const double* arow = a + size_t{i} * 4;
      for (uint32_t j = 0; j < n; ++j) {
        double sum = c[size_t{i} * n + j];
        sum = __builtin_fma(bk[j], arow[0], sum);
        sum = __builtin_fma(bk[size_t{n} + j], arow[1], sum);
        sum = __builtin_fma(bk[size_t{2} * n + j], arow[2], sum);
        sum = __builtin_fma(bk[size_t{3} * n + j], arow[3], sum);
        out[size_t{i} * n + j] = sum;
      }
    }
    return;
  }
  for (uint32_t i = 0; i < m; ++i)
    for (uint32_t j = 0; j < n; ++j) {
      double sum = c[size_t{i} * n + j];
      const double* arow = a + size_t{i} * k;
      for (uint32_t t = 0; t < k; ++t) sum = __builtin_fma(bk[size_t{t} * n + j], arow[t], sum);
      out[size_t{i} * n + j] = sum;
    }
}
inline bool mfma_have_fma() {
  static const bool have = !simd_capped_to_base() && __builtin_cpu_supports("fma");
  return have;
}
#endif
inline void mfma_tile_fma(uint32_t m, uint32_t n, uint32_t k, const double* a, const double* bk, const double* c, double* out) {
#if defined(__x86_64__) && defined(__GNUC__)
  if (mfma_have_fma()) {
    mfma_tile_fma_hw(m, n, k, a, bk, c, out);
  } else
#endif
  {
    for (uint32_t i = 0; i < m; ++i)
      for (uint32_t j = 0; j < n; ++j) {
        double sum = c[size_t{i} * n + j];
        const double* arow = a + size_t{i} * k;
        for (uint32_t t = 0; t < k; ++t) sum = std::fma(bk[size_t{t} * n + j], arow[t], sum);
        out[size_t{i} * n + j] = sum;
      }
  }
  // (One pass for "is any a NaN", with no branch to place, then the fix-up pass only if so.)
  unsigned any_nan = 0;
  for (size_t e = 0; e < size_t{m} * n; ++e) any_nan |= out[e] != out[e];
  if (any_nan)
    for (uint32_t i = 0; i < m; ++i)
      for (uint32_t j = 0; j < n; ++j) {
        double& x = out[size_t{i} * n + j];
        if (x != x) {
          double sum = c[size_t{i} * n + j];
          for (uint32_t t = 0; t < k; ++t) sum = mfma_fma(bk[size_t{t} * n + j], a[size_t{i} * k + t], sum);
          x = sum;
        }
      }
}

// Integer forms: the dot product of int16 values (the inputs are 8-bit or narrower), exact, so any order will do. `ia` is m rows
// of k; `bi` holds B as pairs of k for each column, bi[((t / 2) * n + j) * 2 + t % 2] (k even), which is what a multiply-add
// of adjacent 16-bit pairs reads. `out` is m rows of n int32 sums, starting from zero (the accumulator is added by the caller,
// who knows how it saturates or wraps).
inline void mfma_tile_int_scalar(uint32_t m, uint32_t n, uint32_t k, const int16_t* ia, const int16_t* bi, int32_t* out) {
  for (uint32_t i = 0; i < m; ++i) {
    const int16_t* const arow = ia + size_t{i} * k;
    for (uint32_t j = 0; j < n; ++j) {
      uint32_t sum = 0;   // (unsigned: the sum wraps, and that is not undefined)
      for (uint32_t t = 0; t < k; t += 2) {
        const int16_t* const bp = bi + (size_t{t / 2} * n + j) * 2;
        sum += static_cast<uint32_t>(int32_t{arow[t]} * bp[0]) + static_cast<uint32_t>(int32_t{arow[t + 1]} * bp[1]);
      }
      out[size_t{i} * n + j] = static_cast<int32_t>(sum);
    }
  }
}
#if defined(__x86_64__) && defined(__GNUC__)
// (AVX2: eight columns to an instruction; n is a multiple of 8)
__attribute__((target("avx2"))) inline void mfma_tile_int_avx2(uint32_t m, uint32_t n, uint32_t k, const int16_t* ia, const int16_t* bi,
                                                              int32_t* out) {
  for (uint32_t i = 0; i < m; i += 4) {
    const int16_t* const a0 = ia + size_t{i} * k;
    for (uint32_t j = 0; j < n; j += 8) {
      __m256i s0 = _mm256_setzero_si256(), s1 = s0, s2 = s0, s3 = s0;
      const int16_t* b = bi + size_t{j} * 2;
      for (uint32_t t = 0; t < k; t += 2, b += size_t{n} * 2) {
        const __m256i bv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));
        int32_t p0, p1, p2, p3;
        std::memcpy(&p0, a0 + t, 4), std::memcpy(&p1, a0 + k + t, 4), std::memcpy(&p2, a0 + 2 * size_t{k} + t, 4), std::memcpy(&p3, a0 + 3 * size_t{k} + t, 4);
        s0 = _mm256_add_epi32(s0, _mm256_madd_epi16(_mm256_set1_epi32(p0), bv));
        s1 = _mm256_add_epi32(s1, _mm256_madd_epi16(_mm256_set1_epi32(p1), bv));
        s2 = _mm256_add_epi32(s2, _mm256_madd_epi16(_mm256_set1_epi32(p2), bv));
        s3 = _mm256_add_epi32(s3, _mm256_madd_epi16(_mm256_set1_epi32(p3), bv));
      }
      int32_t* const o = out + size_t{i} * n + j;
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(o), s0);
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(o + n), s1);
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(o + 2 * size_t{n}), s2);
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(o + 3 * size_t{n}), s3);
    }
  }
}
#endif
inline void mfma_tile_int(uint32_t m, uint32_t n, uint32_t k, const int16_t* ia, const int16_t* bi, int32_t* out) {
#if defined(__x86_64__) && defined(__GNUC__)
  if (m % 4 == 0 && n % 8 == 0 && mfma_have_avx2()) {
    mfma_tile_int_avx2(m, n, k, ia, bi, out);
    return;
  }
#endif
#if defined(__SSE2__)
  if (m % 4 == 0 && n % 4 == 0) {
    // Four rows by four columns at a time: a multiply-add of adjacent 16-bit pairs takes the pair of k of A (the same in all four
    // lanes) and the pair of k of four columns of B, and adds the two products to its 32-bit lane.
    for (uint32_t i = 0; i < m; i += 4) {
      const int16_t* const a0 = ia + size_t{i} * k;
      for (uint32_t j = 0; j < n; j += 4) {
        __m128i s0 = _mm_setzero_si128(), s1 = s0, s2 = s0, s3 = s0;
        const int16_t* b = bi + size_t{j} * 2;
        for (uint32_t t = 0; t < k; t += 2, b += size_t{n} * 2) {
          const __m128i bv = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b));
          int32_t p0, p1, p2, p3;
          std::memcpy(&p0, a0 + t, 4), std::memcpy(&p1, a0 + k + t, 4), std::memcpy(&p2, a0 + 2 * size_t{k} + t, 4), std::memcpy(&p3, a0 + 3 * size_t{k} + t, 4);
          s0 = _mm_add_epi32(s0, _mm_madd_epi16(_mm_set1_epi32(p0), bv));
          s1 = _mm_add_epi32(s1, _mm_madd_epi16(_mm_set1_epi32(p1), bv));
          s2 = _mm_add_epi32(s2, _mm_madd_epi16(_mm_set1_epi32(p2), bv));
          s3 = _mm_add_epi32(s3, _mm_madd_epi16(_mm_set1_epi32(p3), bv));
        }
        int32_t* const o = out + size_t{i} * n + j;
        _mm_storeu_si128(reinterpret_cast<__m128i*>(o), s0);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(o + n), s1);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(o + 2 * size_t{n}), s2);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(o + 3 * size_t{n}), s3);
      }
    }
    return;
  }
#endif
  mfma_tile_int_scalar(m, n, k, ia, bi, out);
}

}  // namespace vgpu::amd
