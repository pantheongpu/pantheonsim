// The sum an Ada (sm_89) tensor core makes of e4m3 and e5m2 products (mma.sync m16n8k32, QMMA), and the choice between the FP8 sums a profile
// can name (vgpu::Fp8Sum, include/vgpu/profile.hpp; the evidence is in docs/matrix-numerics.md).
// The same arithmetic, with the numbers of kHopperWgmmaFp8Rule (one chunk of 32, grid 14, 13 result bits), is the sum of the dense FP8 forms of wgmma
// and QGMMA on a Hopper device (vgpu::WgmmaFp8Sum; docs/numerics/fp8-vectors.md).
//
// K is cut into chunks of `chunk` products, in order. A chunk is its exact products and the accumulator, each cut toward zero at
// 2^(e_max - (grid - 1)), where e_max is the largest of the terms' exponents (a product's exponent is the sum of its factors', each
// clamped at the format's smallest normal exponent; a zero term has none), added exactly, and then
//   - an f32 accumulator: cut toward zero to `out_bits` fraction bits (13 on Ada: the result has the range of an f32 and the precision
//     of a 14-bit significand) and to the subnormal grid below 2^-126;
//   - an f16 accumulator: rounded to nearest even to binary16.
// The result is the next chunk's accumulator. A zero result is +0. NaN and infinity are IEEE on exact products, the NaN canonical.
// (Xie et al. give a zero term the exponent -132, or -20 for an f16 accumulator. With e4m3 and e5m2 operands every product's exponent is at
// least -12 and an accumulator's at least -126 (f16: -14), so that floor is below every other term and changes no result: it is left out.)
// This is the model of Xie et al. (arXiv 2511.10909) written out again; it reproduces the 5,000 vectors of each Ada and L40S file of
// Khattak and Mikaitis (arXiv 2512.07004, e4m3 and e5m2, f32 and f16 accumulators) bit for bit.
#pragma once
#include <algorithm>
#include <bit>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "vgpu/exec/tc_ampere.hpp"
#include "vgpu/profile.hpp"

namespace vgpu::exec::tc86 {

inline constexpr int kEminE4M3 = -6, kEminE5M2 = -14;

// The rule in force on a device for the e4m3 / e5m2 tensor-core forms: null for the sequential sum (and with VGPU_TC_EXACT=0).
inline const vgpu::Fp8SumRule* fp8_rule(const vgpu::DeviceProfile& p) {
  return exact_enabled() ? vgpu::fp8_sum_rule(p.numerics.tensor_sum_fp8) : nullptr;
}
// Whether the device takes its f16 rule for these forms instead (Hopper and datacenter Blackwell: ptxas turns an FP8 mma.sync into
// a conversion to f16 and HMMA, so the f16 rule's chunks of 16 are what sums them; the SASS engine runs that code already).
inline bool fp8_via_f16(const vgpu::DeviceProfile& p) {
  return exact_enabled() && p.numerics.tensor_sum_fp8 == vgpu::Fp8Sum::ViaF16 && applies(p);
}

namespace detail {

struct Fp8Term {
  int64_t sig;   // signed significand
  int q;         // value = sig * 2^q
  int e;         // the exponent the term is aligned by
};

// x finite, non zero, a normal double (every e4m3, e5m2, f16 and f32 value is).
inline Fp8Term fp8_split(double x, int emin) {
  const uint64_t u = std::bit_cast<uint64_t>(x);
  const int ex = static_cast<int>((u >> 52) & 0x7ff);
  uint64_t m = (u & ((uint64_t{1} << 52) - 1)) | (uint64_t{1} << 52);
  const int tz = std::countr_zero(m);
  m >>= tz;
  return {(u >> 63) ? -static_cast<int64_t>(m) : static_cast<int64_t>(m), ex - 1075 + tz, std::max(ex - 1023, emin)};
}

// sig * 2^g after the cut to the chunk's grid: the magnitude's bits below 2^g are dropped.
inline int64_t fp8_align(const Fp8Term& t, int g) {
  const int s = t.q - g;
  const int64_t mag = t.sig < 0 ? -t.sig : t.sig;
  const int64_t v = s >= 0 ? mag << s : (-s >= 63 ? 0 : mag >> -s);
  return t.sig < 0 ? -v : v;
}

// S * 2^g to the accumulator's format: f16 to nearest even, f32 cut toward zero to out_bits fraction bits.
inline double fp8_round(int64_t S, int g, bool f16, int out_bits) {
  if (S == 0) return 0.0;
  const bool neg = S < 0;
  const uint64_t mag = neg ? -static_cast<uint64_t>(S) : static_cast<uint64_t>(S);
  const int top = std::bit_width(mag) - 1 + g;
  const int lsb = f16 ? std::max(top, -14) - 10 : std::max(top, -126) - out_bits;
  uint64_t r = mag;
  int at = g;
  if (lsb > g) {
    const int sh = lsb - g;
    if (sh >= 64) {
      r = 0;
    } else {
      r = mag >> sh;
      if (f16) {
        const uint64_t rem = mag & ((uint64_t{1} << sh) - 1), half = uint64_t{1} << (sh - 1);
        if (rem > half || (rem == half && (r & 1))) ++r;
      }
    }
    at = lsb;
  }
  const double v = std::ldexp(static_cast<double>(r), at);
  return neg ? -v : v;
}

}  // namespace detail

// D = sum_k a[k] * b[k] + c for one output element of an e4m3 / e5m2 mma: a and b are the exact values (strides in elements), c the
// accumulator (a value of f32, or of f16 when acc_f16). Returns the result, exactly representable in the accumulator's format.
inline double fp8_dot(const vgpu::Fp8SumRule& r, bool acc_f16, int emin_a, int emin_b, const double* a, size_t astride, const double* b,
                      size_t bstride, unsigned K, double c) {
  using detail::Fp8Term;
  const int F = r.grid - 1;
  const double qnan = std::numeric_limits<double>::quiet_NaN(), inf = std::numeric_limits<double>::infinity();
  double acc = c;
  for (unsigned k0 = 0; k0 < K; k0 += r.chunk) {
    const unsigned n = std::min(r.chunk, K - k0);
    Fp8Term t[64];
    unsigned nt = 0;
    int e_max = INT_MIN;
    bool nan = false, pos = false, neg = false;
    for (unsigned i = 0; i < n; ++i) {
      const double x = a[(k0 + i) * astride], y = b[(k0 + i) * bstride];
      if (std::isnan(x) || std::isnan(y)) { nan = true; continue; }
      if (std::isinf(x) || std::isinf(y)) {
        if (x == 0 || y == 0) nan = true;
        else if (std::signbit(x) != std::signbit(y)) neg = true;
        else pos = true;
        continue;
      }
      if (x == 0 || y == 0) continue;
      const Fp8Term fa = detail::fp8_split(x, emin_a), fb = detail::fp8_split(y, emin_b);
      t[nt] = {fa.sig * fb.sig, fa.q + fb.q, fa.e + fb.e};
      e_max = std::max(e_max, t[nt++].e);
    }
    if (std::isnan(acc)) nan = true;
    else if (std::isinf(acc)) (acc > 0 ? pos : neg) = true;
    else if (acc != 0) {
      t[nt] = detail::fp8_split(acc, acc_f16 ? -14 : -126);
      e_max = std::max(e_max, t[nt++].e);
    }
    // (an infinity or a NaN is the next chunk's accumulator, so that +inf in one chunk and -inf in another meet as NaN)
    if (nan || (pos && neg)) { acc = qnan; continue; }
    if (pos || neg) { acc = pos ? inf : -inf; continue; }
    if (nt == 0) { acc = 0.0; continue; }   // nothing but zeros: +0
    const int g = e_max - F;
    int64_t S = 0;
    for (unsigned i = 0; i < nt; ++i) S += detail::fp8_align(t[i], g);
    acc = detail::fp8_round(S, g, acc_f16, r.out_bits);
    if (acc_f16 && std::fabs(acc) > 65504.0) acc = acc > 0 ? inf : -inf;   // a binary16 accumulator that overflows is an infinity from then on
  }
  return acc;
}

// The rule in force on a device for the e4m3 / e5m2 forms of wgmma and QGMMA: null for the sequential sum (and with VGPU_TC_EXACT=0). Dense
// forms only: the caller does not ask for a sparse one.
inline const vgpu::Fp8SumRule* wgmma_fp8_rule(const vgpu::DeviceProfile& p) {
  return exact_enabled() ? vgpu::wgmma_fp8_sum_rule(p.numerics.tensor_sum_wgmma_fp8) : nullptr;
}

// D = A * B + C for a rows x N tile of an FP8 wgmma: A is rows x K, B is K x N (values, row-major), C and D rows x N (the accumulator's values,
// f16 ones already widened; D is exact in the accumulator's format).
inline void fp8_tile(const vgpu::Fp8SumRule& r, bool acc_f16, int emin_a, int emin_b, uint32_t rows, uint32_t N, uint32_t K, const float* A,
                     const float* B, const float* C, float* D) {
  thread_local std::vector<double> Ad, Bd;
  Ad.assign(A, A + size_t{rows} * K);
  Bd.assign(B, B + size_t{K} * N);
  // (a NaN is the canonical 0x7fffffff, as the tile kernels write it)
  for (uint32_t i = 0; i < rows; ++i)
    for (uint32_t j = 0; j < N; ++j) {
      const float x = static_cast<float>(fp8_dot(r, acc_f16, emin_a, emin_b, &Ad[size_t{i} * K], 1, &Bd[j], N, K, C[size_t{i} * N + j]));
      D[size_t{i} * N + j] = x != x ? std::bit_cast<float>(0x7fffffffu) : x;
    }
}

}  // namespace vgpu::exec::tc86
