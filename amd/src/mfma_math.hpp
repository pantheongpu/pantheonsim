// The arithmetic MFMA and WMMA instructions share, in the form the interpreter runs it. Kept in a header so a unit
// test can hold it against the straightforward version it replaced, bit for bit (tests/unit/test_amd_mfma_math.cpp).
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

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

// One output of a matrix multiply-accumulate: c plus the dot product of a row of A with a column of B, k terms taken in
// order and summed in double (so the order is the result). `bt` is that column, stored contiguously.
inline double mfma_dot(const double* a, const double* bt, uint32_t k, double c) {
  double sum = c;
  for (uint32_t i = 0; i < k; ++i) sum = mfma_add(sum, mfma_mul(a[i], bt[i]));
  return sum;
}
// Four of them at once, from four rows and four columns: four sums that nothing joins, each taken in order, so each
// is what mfma_dot gives; they overlap in the machine instead of waiting on one another.
inline void mfma_dot4(const double* const a[4], const double* const bt[4], uint32_t k, const double c[4], double out[4]) {
  double s0 = c[0], s1 = c[1], s2 = c[2], s3 = c[3];
  for (uint32_t i = 0; i < k; ++i) {
    s0 = mfma_add(s0, mfma_mul(a[0][i], bt[0][i]));
    s1 = mfma_add(s1, mfma_mul(a[1][i], bt[1][i]));
    s2 = mfma_add(s2, mfma_mul(a[2][i], bt[2][i]));
    s3 = mfma_add(s3, mfma_mul(a[3][i], bt[3][i]));
  }
  out[0] = s0, out[1] = s1, out[2] = s2, out[3] = s3;
}
// The fused form, for the double-precision instruction.
inline double mfma_dot_fma(const double* a, const double* bt, uint32_t k, double c) {
  double sum = c;
  for (uint32_t i = 0; i < k; ++i) sum = std::fma(bt[i], a[i], sum);
  return sum;
}
// Integers are summed exactly and wrap as the hardware's do.
inline int64_t mfma_dot_int(const double* a, const double* bt, uint32_t k, int64_t c) {
  int64_t sum = c;
  for (uint32_t i = 0; i < k; ++i) sum += static_cast<int64_t>(a[i]) * static_cast<int64_t>(bt[i]);
  return sum;
}

}  // namespace vgpu::amd
