// cuBLAS's fixed-point emulation of double precision (CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT,
// CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH): a GEMM done on integer slices, in the
// way NVIDIA documents ("algorithms based on the Ozaki-I and Ozaki-II schemes";
// elements of a row of A or a column of B share one power-of-two scaling factor
// and become integers of a given mantissa bit count) and as an RTX 3060's
// cuBLAS 13.0 does it when the strategy is EAGER, found by experiment on that
// card (nvidia/tests/e2e/blas_emulation_paths.cu is the check):
//
//  * Each row of op(A) and column of op(B) has one exponent e, that of its
//    largest element (frexp's: |x| = m 2^e, m in [0.5, 1)), plus one when that
//    element's leading seven mantissa bits are all ones (m >= 127/128).
//  * An element becomes the integer floor(|x| 2^(bits - e)) with its sign, bits
//    being 7 + 8 (s - 1) for s slices; s = M / 8 + 1 for a mantissa bit count M
//    (M = 4 to 7 is one slice, 8 to 15 two, ...). The integer is cut into s
//    base-256 digits, the top one of 7 bits, and the lower digits are balanced
//    as digits of the signed integer would be (int8 holds -128 to 127): a
//    digit of 128 or more of a positive element, or above 128 of a negative
//    one, is made 256 less and carries 1 into the digit above.
//  * The products of the digits (i, j) with i + j <= s + 1 (1-based) are added
//    over the inner dimension in integers, one sum per i + j, and the sums are
//    scaled by their powers of two and added in double precision, the largest
//    first.
//
// Checked against the card on 41,538 outputs of random matrices of seven shapes
// (transposed and not, with and without wide exponent ranges) at every mantissa
// bit count from 4 to 64, the product is bit for bit the card's.
//
// The dynamic mantissa control of the card (an "automatic dynamic precision"
// framework that chose between 54 and 97 bits for the matrices tried) is not
// reproduced: its rule is not public. This library picks 54 bits plus the
// largest spread, over one output element's terms, between the exponents of the
// products a(i,p) b(p,j) -- enough to keep every term of the sum -- and adds
// the offset. A rule of this kind gave results as accurate as the card's
// (relative error about 3e-16).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace vgpu::fpemu {

struct Options {
  bool dynamic = true;   // CUDA_EMULATION_MANTISSA_CONTROL_DYNAMIC (default) or FIXED
  int max_bits = 0;      // the maximum mantissa bit count; 0 is the library's default
  int offset = 0;        // dynamic control: added to the recommended bit count
};

// The mantissa bit count of FIXED control when none is set (cuBLAS documents 55).
constexpr int kDefaultFixedBits = 55;
// The most bits dynamic control will emulate when no limit is set; beyond
// that the call is done in plain double precision.
constexpr int kDefaultDynamicLimit = 256;

namespace detail {

struct Scaled {
  std::vector<int> e;                    // the row/column exponent, or INT_MIN for an all-zero one
  std::vector<std::vector<int16_t>> d;   // per line: s digits of k entries each, digit-major
};

inline int line_exponent(double mx) {
  int e;
  const double m = std::frexp(mx, &e);
  return std::floor(m * 128.0) >= 127.0 ? e + 1 : e;
}

// Digit t (0 = top, s digits) of floor(|x| 2^(bits - e)), before balancing.
// |x| = f 2^q with f a 53-bit integer.
inline int raw_digit(uint64_t f, int sh, int s, int t) {
  const int lo = 8 * (s - 1 - t);   // the digit's lowest bit within the integer
  const int pos = lo - sh;          // ... within f
  if (pos >= 64) return 0;
  uint64_t v;
  if (pos >= 0) v = f >> pos;
  else if (-pos >= 8) return 0;
  else v = f << -pos;
  return t == 0 ? (int)v : (int)(v & 255);
}

// The digits of one element, balanced.
inline void digits(double x, int e, int s, int bits, int16_t* out, size_t stride) {
  int ex;
  const double fr = std::frexp(std::fabs(x), &ex);   // |x| = fr 2^ex, fr in [0.5, 1)
  const uint64_t f = (uint64_t)std::ldexp(fr, 53);   // |x| = f 2^(ex - 53)
  const int sh = (ex - 53) + (bits - e);              // floor(|x| 2^(bits-e)) = floor(f 2^sh)
  std::vector<int> d((size_t)s);
  for (int t = 0; t < s; ++t) d[(size_t)t] = x == 0.0 ? 0 : raw_digit(f, sh, s, t);
  const int carry_at = x < 0 ? 129 : 128;
  for (int t = s - 1; t >= 1; --t)
    if (d[(size_t)t] >= carry_at) {
      d[(size_t)t] -= 256;
      d[(size_t)t - 1] += 1;
    }
  for (int t = 0; t < s; ++t) out[(size_t)t * stride] = (int16_t)(x < 0 ? -d[(size_t)t] : d[(size_t)t]);
}

}  // namespace detail

// The outcome of multiply: the mantissa bit count to report, and whether the
// product was formed (P is filled in) or is left to plain double arithmetic.
// Plain arithmetic is taken for a non-finite input (the fixed-point forms
// support no special values: no bit count is reported then) and for a dynamic
// bit count above the limit (the card reports the bit count it wanted, 62 in
// the case measured, and computes in plain double precision).
struct Result {
  int bits = -1;       // -1: nothing to report
  bool formed = false;
};

// P = op(A) op(B) for column-major A and B (leading dimensions lda, ldb), P
// m x n with leading dimension m.
inline Result multiply(bool ta, bool tb, int m, int n, int k, const double* A, int lda, const double* B, int ldb, const Options& o,
                    std::vector<double>* P) {
  auto a_at = [&](int i, int p) { return ta ? A[(size_t)p + (size_t)i * lda] : A[(size_t)i + (size_t)p * lda]; };
  auto b_at = [&](int p, int j) { return tb ? B[(size_t)j + (size_t)p * ldb] : B[(size_t)p + (size_t)j * ldb]; };
  for (int i = 0; i < m; ++i)
    for (int p = 0; p < k; ++p)
      if (!std::isfinite(a_at(i, p))) return {};
  for (int j = 0; j < n; ++j)
    for (int p = 0; p < k; ++p)
      if (!std::isfinite(b_at(p, j))) return {};

  // The bit count and the number of slices.
  int bits_report;
  if (!o.dynamic) {
    bits_report = o.max_bits > 0 ? o.max_bits : kDefaultFixedBits;
  } else {
    int spread = 0;
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < m; ++i) {
        int lo = 1 << 30, hi = -(1 << 30);
        for (int p = 0; p < k; ++p) {
          const double x = a_at(i, p), y = b_at(p, j);
          if (x == 0.0 || y == 0.0) continue;
          int ex, ey;
          std::frexp(x, &ex);
          std::frexp(y, &ey);
          lo = std::min(lo, ex + ey);
          hi = std::max(hi, ex + ey);
        }
        if (hi >= lo) spread = std::max(spread, hi - lo);
      }
    bits_report = std::max(1, 54 + spread + o.offset);
    const int limit = o.max_bits > 0 ? o.max_bits : kDefaultDynamicLimit;
    if (bits_report / 8 + 1 > limit / 8 + 1) return {bits_report, false};
  }
  const int s = bits_report / 8 + 1;
  const int bits = 7 + 8 * (s - 1);   // significant bits of an element's integer

  auto scale = [&](int lines, int len, auto at) {
    detail::Scaled sc;
    sc.e.assign((size_t)lines, INT32_MIN);
    sc.d.assign((size_t)lines, std::vector<int16_t>((size_t)s * (size_t)std::max(len, 1), 0));
    for (int l = 0; l < lines; ++l) {
      double mx = 0.0;
      for (int p = 0; p < len; ++p) mx = std::max(mx, std::fabs(at(l, p)));
      if (mx == 0.0) continue;
      const int e = detail::line_exponent(mx);
      sc.e[(size_t)l] = e;
      for (int p = 0; p < len; ++p) detail::digits(at(l, p), e, s, bits, &sc.d[(size_t)l][(size_t)p], (size_t)len);
    }
    return sc;
  };
  const auto sa = scale(m, k, [&](int i, int p) { return a_at(i, p); });
  const auto sb = scale(n, k, [&](int j, int p) { return b_at(p, j); });

  P->assign((size_t)m * (size_t)n, 0.0);
  std::vector<int64_t> sums((size_t)s + 2);
  for (int j = 0; j < n; ++j) {
    if (sb.e[(size_t)j] == INT32_MIN) continue;
    for (int i = 0; i < m; ++i) {
      if (sa.e[(size_t)i] == INT32_MIN) continue;
      std::fill(sums.begin(), sums.end(), 0);
      for (int x = 0; x < s; ++x)
        for (int y = 0; x + y <= s - 1; ++y) {   // 0-based digit pair; 1-based i + j <= s + 1
          const int16_t* da = &sa.d[(size_t)i][(size_t)x * (size_t)k];
          const int16_t* db = &sb.d[(size_t)j][(size_t)y * (size_t)k];
          int64_t acc = 0;
          for (int p = 0; p < k; ++p) acc += (int64_t)da[p] * db[p];
          sums[(size_t)(x + y + 2)] += acc;
        }
      double total = 0.0;
      for (int dd = 2; dd <= s + 1; ++dd)   // the largest weight first
        total += std::ldexp((double)sums[(size_t)dd], 8 * (2 * s - dd) - 2 * bits + sa.e[(size_t)i] + sb.e[(size_t)j]);
      (*P)[(size_t)i + (size_t)j * (size_t)m] = total;
    }
  }
  return {bits_report, true};
}

}  // namespace vgpu::fpemu
