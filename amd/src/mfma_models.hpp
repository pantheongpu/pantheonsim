// The matrix-core models of Khattak, Mikaitis and Graziani, "Accurate Models of AMD Matrix Cores", arXiv 2609.14845
// (github.com/north-numerical-computing/MATLAB-tensor-core, BSD-2-Clause, read at commit 6fd05208): one sum for CDNA1 (MI100),
// CDNA2 (MI210, MI250) and CDNA3 (MI300A, MI300X), written again here from the MATLAB files (models/MI100MC.m, MI210MC.m, MI300AMC.m
// and the tools they call), not copied. Opt-in (VGPU_MFMA_MODEL=khattak); class "published": no card of ours has run it and the
// repository has no AMD validation vectors, so nothing here is checked against hardware. docs/matrix-numerics.md says where the
// three sources of an MI250X (this paper, FTTN, MMA-Sim) disagree.
//
// What a model is: the K products of an instruction are cut into blocks of `fma` (CDNA1: 4 for binary16, 2 for bfloat16; CDNA2: 4;
// CDNA3: 8 for binary16 and bfloat16, 4 for xf32, 16 for fp8), and each block with the running sum c gives the next c:
//   CDNA1  the exact sum of the block's products and c, rounded once to nearest even, subnormals kept.
//   CDNA2  inputs below the format's smallest normal number are zero, each product is rounded to binary32, the block is added as a
//          balanced binary tree of binary32 additions, then c is added; every sum below 2^-126 becomes zero.
//   CDNA3  the products are aligned to the largest exponent of the block with one extra bit and the bits shifted out cut off (so
//          the cut is toward zero for each product), summed as integers, then added to c: c is aligned to that, or the products to
//          c (eight more bits kept), and what is shifted out of the smaller one is cut toward minus infinity; one rounding to
//          nearest even. fp8 adds the even-numbered products and the odd-numbered ones as two such sums first.
// Parts of the MATLAB code that look unintended are followed as written and listed in docs/matrix-numerics.md.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>

namespace vgpu::amd::khattak {

enum class Model { Cdna1, Cdna2, Cdna3 };

// An operand's format, in the letters of the executor's tile code: 'h' binary16, 'b' bfloat16, 'F' xf32, 'e' fp8 e4m3 and 'g' fp8
// e5m2 (CDNA3's FNUZ forms). False for the forms a model does not describe.
inline bool supports(Model m, char t) {
  switch (t) {
    case 'h': case 'b': return true;
    case 'F': case 'e': case 'g': return m == Model::Cdna3;
    default: return false;
  }
}
// The exponent below which a value of the format is subnormal (the model's emin_input; the FNUZ forms are one lower).
inline int emin_in(char t) {
  switch (t) {
    case 'h': return -14;
    case 'e': return -7;
    case 'g': return -15;
    default: return -126;
  }
}
inline uint32_t block_size(Model m, char t) {
  switch (m) {
    case Model::Cdna1: return t == 'b' ? 2 : 4;
    case Model::Cdna2: return 4;
    default: return t == 'F' ? 4 : (t == 'e' || t == 'g') ? 16 : 8;
  }
}

namespace detail {

constexpr double kInf = std::numeric_limits<double>::infinity();
inline double nan_() { return std::numeric_limits<double>::quiet_NaN(); }

// +-(mag * 2^e2) rounded to binary32, to nearest even, subnormals kept, as a double. mag < 2^53, so the double is exact.
inline double round32(bool neg, uint64_t mag, int e2) {
  if (!mag) return 0.0;
  const double v = std::ldexp(static_cast<double>(mag), e2);
  const double r = v >= 0x1p128 ? kInf : static_cast<double>(static_cast<float>(v));
  return neg ? -r : r;
}

// An exact sum of doubles (each a product of two short significands or a binary32), rounded once to binary32 (nearest even). Two
// fixed-point magnitudes, one for the positive terms and one for the negative, wide enough for every exponent of a bfloat16 product.
struct Wide {
  static constexpr int kLimbs = 12, kOffset = 384;   // bit i has the weight 2^(i - kOffset)
  uint64_t v[kLimbs] = {};
  void add(double x) {   // x > 0
    int e;
    const double f = std::frexp(x, &e);
    const uint64_t m = static_cast<uint64_t>(std::ldexp(f, 53));
    const int p = e - 53 + kOffset, i = p / 64, s = p % 64;
    unsigned __int128 t = static_cast<unsigned __int128>(m) << s;
    for (int k = i; k < kLimbs && t; ++k) {
      t += v[k];
      v[k] = static_cast<uint64_t>(t);
      t >>= 64;
    }
  }
  bool bit(int p) const { return p >= 0 && ((v[p / 64] >> (p % 64)) & 1); }
  bool below(int p) const {   // any bit under position p
    for (int k = 0; k < p / 64; ++k) if (v[k]) return true;
    return p % 64 && (v[p / 64] & ((uint64_t{1} << (p % 64)) - 1));
  }
};
inline int cmp(const Wide& a, const Wide& b) {
  for (int k = Wide::kLimbs - 1; k >= 0; --k)
    if (a.v[k] != b.v[k]) return a.v[k] < b.v[k] ? -1 : 1;
  return 0;
}
inline void sub(Wide& a, const Wide& b) {   // a -= b, a >= b
  uint64_t borrow = 0;
  for (int k = 0; k < Wide::kLimbs; ++k) {
    const uint64_t x = a.v[k], y = b.v[k] + borrow;
    borrow = (b.v[k] + borrow < b.v[k]) || x < y;
    a.v[k] = x - y;
  }
}

inline double exact_sum32(const double* x, int n) {
  Wide pos, neg;
  for (int i = 0; i < n; ++i)
    if (x[i] > 0) pos.add(x[i]);
    else if (x[i] < 0) neg.add(-x[i]);
  const int c = cmp(pos, neg);
  if (!c) return 0.0;
  const bool negative = c < 0;
  Wide m = negative ? neg : pos;
  if (negative) sub(m, pos); else sub(m, neg);
  int h = Wide::kLimbs * 64 - 1;
  while (!m.bit(h)) --h;
  const int ulp = std::max(h - Wide::kOffset - 23, -149);   // the exponent of the last bit kept
  const int low = ulp + Wide::kOffset;
  uint64_t q = 0;
  for (int p = h; p >= low; --p) q = q << 1 | m.bit(p);
  const bool round = m.bit(low - 1), sticky = m.below(low - 1);
  if (round && (sticky || (q & 1))) ++q;
  const double v = std::ldexp(static_cast<double>(q), ulp);
  const double r = v >= 0x1p128 ? kInf : static_cast<double>(static_cast<float>(v));
  return negative ? -r : r;
}

// ---- CDNA2 -----------------------------------------------------------------------------------------------------------------
inline float flush32(float x) { return std::fabs(x) < 0x1p-126f ? std::copysign(0.0f, x) : x; }
inline float tree(const float* p, uint32_t n) {
  if (n == 1) return p[0];
  return flush32(tree(p, n / 2) + tree(p + n / 2, n / 2));
}

// ---- CDNA3 -----------------------------------------------------------------------------------------------------------------
struct Part {   // a sum of aligned significands: its exponent, its sign, its magnitude
  int exp;
  bool neg;
  uint64_t mag;
};
inline uint64_t shr(uint64_t x, int s) { return s >= 64 ? 0 : x >> s; }
inline uint64_t shl(uint64_t x, int s) { return s >= 64 ? 0 : x << s; }

// The model's AddTwoNonNormSums, round-down alignment. `neab` is in and out (the bits kept below the significand).
inline Part add_two(const Part& p1, const Part& p2, int eab1, int eab2, int& neab, bool exp_check, int64_t* total_out) {
  const int mx = std::max(p1.exp, p2.exp);
  int shift = std::abs(p1.exp - p2.exp);
  if (shift > 512) shift = 0;
  uint64_t s1 = p1.mag, s2 = p2.mag;
  if (p1.exp < mx) {
    neab += eab1;
    s1 = shl(s1, eab1);
    s2 = shl(s2, eab1);
    const bool lost = s1 - shl(shr(s1, shift), shift) > 0;
    s1 = shr(s1, shift) + (lost && p1.neg ? 1 : 0);
  } else if (p2.exp < mx) {
    neab += eab2;
    s1 = shl(s1, eab2);
    s2 = shl(s2, eab2);
    bool adj = s2 - shl(shr(s2, shift), shift) > 0 && p2.neg;
    if (exp_check && p2.exp < mx - 25) adj = false;
    s2 = shr(s2, shift) + (adj ? 1 : 0);
  }
  const int64_t total = (p1.neg ? -static_cast<int64_t>(s1) : static_cast<int64_t>(s1)) + (p2.neg ? -static_cast<int64_t>(s2) : static_cast<int64_t>(s2));
  *total_out = total;
  return Part{mx, total < 0, static_cast<uint64_t>(total < 0 ? -total : total)};
}

// The model's GlobalAlignmentSum: products aligned to the largest exponent with `neab` extra bits, each cut toward zero.
inline Part global_sum(const double* sig, const int* ex, const bool* neg, int n, int neab) {
  int mx = -1024;
  bool any = false;
  for (int i = 0; i < n; ++i)
    if (!any || ex[i] > mx) mx = ex[i], any = true;
  if (!any) return Part{0, false, 0};   // (an empty list: its exponent is never looked at, its sum is zero)
  int64_t sum = 0;
  for (int i = 0; i < n; ++i) {
    const uint64_t s = static_cast<uint64_t>(std::fabs(sig[i]) * 8388608.0) << neab;
    const int sh = mx - ex[i];
    const uint64_t a = sh <= 31 ? s >> sh : 0;
    sum += neg[i] ? -static_cast<int64_t>(a) : static_cast<int64_t>(a);
  }
  return Part{mx, sum < 0, static_cast<uint64_t>(sum < 0 ? -sum : sum)};
}

// c as a Part at the scale of `neab` extra bits.
inline Part c_part(double c, int neab) {
  if (c == 0) return Part{-126, false, 0};
  const int e = std::max(std::ilogb(c), -126);
  const double sig = std::fabs(c) / std::ldexp(1.0, e);
  return Part{e, c < 0, static_cast<uint64_t>(sig * std::ldexp(1.0, 23 + neab))};
}

}  // namespace detail

// One block of a model: `n` products a[i]*b[i] (n == block_size, padded with zeros) and the running sum c. Returns a binary32 value
// as a double, or a NaN. `ea` and `eb` are the operands' emin_in. The caller has handled NaN and infinity in the whole row.
inline double block(Model m, char ta, char tb, const double* a, const double* b, uint32_t n, double c) {
  using namespace detail;
  if (m == Model::Cdna1) {
    if (std::isinf(c)) return c;   // (an earlier block overflowed)
    double t[17];
    int k = 0;
    for (uint32_t i = 0; i < n; ++i)
      if (a[i] != 0 && b[i] != 0) t[k++] = a[i] * b[i];
    if (!k && c == 0) return 0.0;
    t[k++] = c;
    return exact_sum32(t, k);
  }
  const int ea_min = emin_in(ta), eb_min = emin_in(tb);
  // The products may not reach 2^128 (CDNA2, CDNA3). Both signs: NaN; one sign: infinity -- of the sign of the largest product, as the
  // MATLAB code takes it (max(prd) * Inf), which is the wrong one when a negative overflow meets a positive product.
  {
    double mx = -kInf;
    bool hi = false, lo = false;
    for (uint32_t i = 0; i < n; ++i) {
      if (m == Model::Cdna3 && (a[i] == 0 || b[i] == 0)) continue;   // (CDNA3 drops the zero products first)
      const double p = a[i] * b[i];
      mx = std::max(mx, p);
      hi = hi || p >= 0x1p128;
      lo = lo || p <= -0x1p128;
    }
    if (hi || lo) return hi && lo ? nan_() : mx * kInf;
  }
  if (std::isinf(c)) return c;   // (an earlier block overflowed)
  if (m == Model::Cdna2) {
    float p[16];
    for (uint32_t i = 0; i < n; ++i) {
      double x = a[i], y = b[i];
      if (std::fabs(x) < std::ldexp(1.0, ea_min) || std::fabs(y) < std::ldexp(1.0, eb_min)) x = y = 0;   // (zero is +0 here, whatever the sign)
      float q = static_cast<float>(x * y);
      if (std::fabs(q) < 0x1p-126f) q = 0.0f;
      p[i] = q;
    }
    return static_cast<double>(flush32(flush32(static_cast<float>(c)) + tree(p, n)));
  }
  // CDNA3
  const bool fp8 = ta == 'e' || ta == 'g';
  double sig[16];
  int ex[16];
  bool neg[16];
  uint32_t k = 0;
  for (uint32_t i = 0; i < n; ++i) {
    if (!fp8 && (a[i] == 0 || b[i] == 0)) continue;
    if (a[i] == 0 || b[i] == 0) {   // (kept, to keep the numbering: a zero product has the exponent -1024)
      sig[k] = 0, ex[k] = -1024, neg[k] = false, ++k;
      continue;
    }
    const int xa = std::max(std::ilogb(a[i]), ea_min), xb = std::max(std::ilogb(b[i]), eb_min);
    sig[k] = (a[i] / std::ldexp(1.0, xa)) * (b[i] / std::ldexp(1.0, xb));
    ex[k] = xa + xb;
    neg[k] = sig[k] < 0;
    ++k;
  }
  if (!fp8 && !k) return c == 0 ? 0.0 : c;   // (every product zero: the sum is c, as the pipeline leaves it)
  int neab = 1;
  Part prod;
  int64_t total;
  if (!fp8) {
    prod = global_sum(sig, ex, neg, static_cast<int>(k), neab);
  } else {
    bool ne[8], no[8];
    double se[8], so[8];
    int xe[8], xo[8];
    int ke = 0, ko = 0;
    for (uint32_t i = 0; i < k; ++i) {
      if (i % 2 == 0) se[ke] = sig[i], xe[ke] = ex[i], ne[ke++] = neg[i];
      else so[ko] = sig[i], xo[ko] = ex[i], no[ko++] = neg[i];
    }
    const Part pe = global_sum(se, xe, ne, ke, neab), po = global_sum(so, xo, no, ko, neab);
    prod = add_two(pe, po, 0, 0, neab, false, &total);
  }
  const Part cp = c_part(c, neab);
  const Part r = add_two(prod, cp, 8, 0, neab, fp8, &total);
  return round32(total < 0, r.mag, r.exp - 23 - neab);
}

// One output: c and k products a[t] * b[t * bstride], in blocks, each block's result the next block's c. NaN and infinity are settled
// over the whole row first, as the model's GEMM does.
inline double dot(Model m, char ta, char tb, const double* a, const double* b, size_t bstride, uint32_t k, double c) {
  using namespace detail;
  bool nan = c != c, pinf = c == kInf, ninf = c == -kInf;
  for (uint32_t t = 0; t < k; ++t) {
    const double p = a[t] * b[t * bstride];
    nan = nan || p != p;
    pinf = pinf || p == kInf;
    ninf = ninf || p == -kInf;
  }
  if (nan || (pinf && ninf)) return nan_();
  if (pinf) return kInf;
  if (ninf) return -kInf;
  const uint32_t f = block_size(m, ta);
  double d = c;
  for (uint32_t t0 = 0; t0 < k; t0 += f) {
    double x[16] = {}, y[16] = {};
    for (uint32_t i = 0; i < f && t0 + i < k; ++i) x[i] = a[t0 + i], y[i] = b[(t0 + i) * bstride];
    d = block(m, ta, tb, x, y, f, d);
    if (d != d) return d;
  }
  return d;
}

// A whole tile, in the layouts of mfma_tile_f (a: m rows of k; bk: k rows of n; c and out: m rows of n).
inline void tile(Model m, char ta, char tb, uint32_t rows, uint32_t cols, uint32_t k, const double* a, const double* bk, const double* c, double* out) {
  for (uint32_t i = 0; i < rows; ++i)
    for (uint32_t j = 0; j < cols; ++j) out[size_t{i} * cols + j] = dot(m, ta, tb, a + size_t{i} * k, bk + j, cols, k, c[size_t{i} * cols + j]);
}

}  // namespace vgpu::amd::khattak
