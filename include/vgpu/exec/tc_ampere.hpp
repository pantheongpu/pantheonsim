// The sum a GA10x (sm_86) tensor core makes of an f16, bf16 or tf32 matrix product with an f32 or f16 accumulator.
//
// An RTX 3060 does not add the products one at a time in f32. Measured on the card by black-box probing (mma.sync m16n8k8 and
// wmma 16x16x16 and 16x16x8 in f16, bf16 and tf32, random finite operands over narrow and wide exponent ranges, special values,
// element by element): 106,496 results of the dense forms, all of them reproduced by the rule below, with the f32 accumulator and
// (16,384 of the 106,496) the f16 one:
//
//   * K is cut into chunks in order: 8 products for f16 and bf16, 4 for tf32 (m16n8k16 is two chunks, m16n8k8 one).
//   * A chunk is the 8 (4) exact products and the accumulator, summed in one step: every term is aligned to the largest of
//     their frame positions and truncated toward zero at 2^(top - 25), the aligned terms are added exactly, and the sum is
//     truncated toward zero to f32 (24 significant bits) -- or rounded to nearest even to f16 when the accumulator is f16.
//     The result is the accumulator of the next chunk.
//   * A product's frame position is the top bit of the full product of the operands' significands, whatever their values:
//     exp(a) + exp(b) + 1 (exp clamped at the format's smallest normal exponent, so a subnormal half counts as 2^-14). The
//     accumulator's is its own leading bit + 1. Zero terms take no part in choosing the largest.
//   * A zero sum is +0, whatever the signs of the accumulator and the products (-0 + -0 products gives +0).
//
// The sequential f32 sum (acc += a*b, k in order) that this simulator used before agrees with the card only when the sums are
// exact, which is why every card check of the tensor core (mma_forms, wmma_types) passed on small integers and no check on
// random data ever did.
//
// Infinities and NaNs decide a chunk's result as in IEEE arithmetic on exact products: a NaN operand, 0 * inf and inf - inf give a NaN,
// otherwise the infinity there is, and the finite terms do not matter. A chunk whose finite sum overflows the format is an infinity
// for the next chunk (checked against the card, element by element, on random bf16 and tf32 data whose products reach 2^254).
//
// Not covered, and left as they were (the sequential sum): the m8n8k4 form, the sparse forms, and other architectures (Hopper and
// Blackwell tensor cores are not this unit; sm_80 and sm_89 are not measured). Not probed: results in the f32 subnormal range (here
// ldexp's), and the sign of a zero f16 accumulator (+0 by analogy with f32).
#pragma once

#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <type_traits>

#include "vgpu/simd/fp16.hpp"
#include "vgpu/simd/simd.hpp"

#if defined(__x86_64__) && defined(__GNUC__)
#include <immintrin.h>
#define VGPU_TC86_AVX2 1
#endif

#ifndef VGPU_TC86_ROWS
#define VGPU_TC86_ROWS 4
#endif
// ... and for the float-product kernel, whose rows hold fewer live vectors: eight rows measured 3 to 10% fewer cycles than four on m16n8 (a loop
// of tc86::mma, f16, bf16 and tf32 operands) and 1 to 2% on 16 columns.
#ifndef VGPU_TC86_ROWS_F
#define VGPU_TC86_ROWS_F 8
#endif
// (The loops over a few rows and a chunk's products must be unrolled for their arrays to live in registers; GCC leaves them rolled otherwise.)
#define VGPU_TC86_UNROLL _Pragma("GCC unroll 8")

namespace vgpu::exec::tc86 {

// VGPU_TC_EXACT=0 turns this model off: the tensor sums are then the sequential f32 ones of the earlier releases, bit for bit, on
// every device. Read from the environment when the process starts and again at each launch (as VGPU_FASTPATH is), so a test can
// set it between launches. (Function-local statics of inline functions: no external-linkage global of the shared code.)
inline std::atomic<bool>& exact_flag() {
  static std::atomic<bool> f{[] {
    const char* v = std::getenv("VGPU_TC_EXACT");
    return !(v && v[0] == '0');
  }()};
  return f;
}
inline void refresh_exact(const char* v) { exact_flag().store(!(v && v[0] == '0'), std::memory_order_relaxed); }
inline void refresh_exact() { refresh_exact(std::getenv("VGPU_TC_EXACT")); }
inline bool exact_enabled() { return exact_flag().load(std::memory_order_relaxed); }

// Whether a device of this compute capability sums as an RTX 3060 does (only the part that was measured), and the switch is on.
inline bool applies(int cc_major, int cc_minor) { return cc_major == 8 && cc_minor == 6 && exact_enabled(); }

// The format of the multiplicands: the smallest normal exponent, for the frame positions.
inline constexpr int kEminF16 = -14;
inline constexpr int kEminWide = -126;   // bf16 and tf32

// A finite float that holds an f16, bf16 or tf32 value (at most 11 significant bits) as sig * 2^lsb, with the frame exponent.
struct Dec {
  int32_t sig;   // signed, |sig| < 2^11
  int32_t lsb;
  int32_t ex;    // max(exponent, emin)
};

inline Dec decode_operand(float v, int emin) {
  const uint32_t b = std::bit_cast<uint32_t>(v);
  const uint32_t e = (b >> 23) & 0xff, m = b & 0x7fffff;
  Dec d;
  if (e == 0) {   // zero, or a subnormal float (a bf16 or tf32 subnormal): m's top bits, at the smallest exponent
    d.sig = static_cast<int32_t>(m >> 13);
    d.lsb = -126 - 10;
    d.ex = -126;
  } else {
    d.sig = 0x400 | static_cast<int32_t>(m >> 13);
    d.lsb = static_cast<int32_t>(e) - 127 - 10;
    d.ex = std::max(static_cast<int32_t>(e) - 127, emin);
  }
  if (b >> 31) d.sig = -d.sig;
  return d;
}

namespace detail {

// A float as the accumulator: its significand and the exponent of its least significant bit.
struct Acc {
  int32_t sig;   // signed
  int32_t lsb;
};
inline Acc decode_acc(float v) {
  const uint32_t b = std::bit_cast<uint32_t>(v);
  const uint32_t e = (b >> 23) & 0xff, m = b & 0x7fffff;
  Acc a;
  if (e == 0) {
    a.sig = static_cast<int32_t>(m);
    a.lsb = -149;
  } else {
    a.sig = static_cast<int32_t>(m | 0x800000u);
    a.lsb = static_cast<int32_t>(e) - 150;
  }
  if (b >> 31) a.sig = -a.sig;
  return a;
}

// |x| << s for s >= 0 and |x| >> -s otherwise (toward zero in magnitude), as a signed value.
inline int32_t align_term(int32_t sig, int32_t s) {
  if (sig == 0) return 0;
  const uint32_t mag = static_cast<uint32_t>(sig < 0 ? -sig : sig);
  const uint32_t v = s >= 0 ? (mag << s) : (-s >= 32 ? 0u : mag >> -s);
  return sig < 0 ? -static_cast<int32_t>(v) : static_cast<int32_t>(v);
}

// The value S * 2^g (S a signed 32-bit integer) truncated toward zero to 24 significant bits, as an f32 (a zero is +0).
inline float to_f32_rz(int32_t S, int32_t g) {
  if (S == 0) return 0.0f;
  uint32_t mag = static_cast<uint32_t>(S < 0 ? -static_cast<int64_t>(S) : static_cast<int64_t>(S));
  const int n = 32 - std::countl_zero(mag);
  int e = g;
  if (n > 24) {
    mag >>= (n - 24);
    e += n - 24;
  }
  const float f = std::ldexp(static_cast<float>(mag), e);
  return S < 0 ? -f : f;
}

// The same value rounded to nearest even to f16, as the float that holds it.
inline float to_f16_rn(int32_t S, int32_t g) {
  if (S == 0) return 0.0f;
  const double d = std::ldexp(static_cast<double>(S), g);
  return vgpu::simd::f16_to_f32(vgpu::simd::f64_to_f16_bits(d));
}

}  // namespace detail

// One chunk of one output: the accumulator and the products a[k0..k0+n), b[k0..k0+n) (decoded, all finite), summed as above. The
// accumulator is a finite float (an f16 value if acc_f16). Returns the new accumulator: an f32 (or the f16 value as a float).
inline float chunk_finite(const Dec* a, size_t astride, const Dec* b, size_t bstride, unsigned n, float acc, bool acc_f16) {
  int32_t top = INT32_MIN;
  int32_t psig[16];
  int32_t plsb[16];
  for (unsigned t = 0; t < n; ++t) {
    const Dec& x = a[t * astride];
    const Dec& y = b[t * bstride];
    psig[t] = x.sig * y.sig;
    plsb[t] = x.lsb + y.lsb;
    if (psig[t] != 0) {
      const int32_t T = x.ex + y.ex + 1;
      if (T > top) top = T;
    }
  }
  const detail::Acc c = detail::decode_acc(acc);
  if (c.sig != 0) {
    const int32_t T = (32 - std::countl_zero(static_cast<uint32_t>(c.sig < 0 ? -c.sig : c.sig))) - 1 + c.lsb + 1;
    if (T > top) top = T;
  }
  if (top == INT32_MIN) return 0.0f;   // all terms zero: +0
  const int32_t g = top - 25;
  int32_t S = detail::align_term(c.sig, c.lsb - g);
  for (unsigned t = 0; t < n; ++t) S += detail::align_term(psig[t], plsb[t] - g);
  return acc_f16 ? detail::to_f16_rn(S, g) : detail::to_f32_rz(S, g);
}

// A chunk with an infinity or a NaN among the accumulator and the operands: as IEEE arithmetic on exact products has it. A NaN
// operand, 0 * inf, and inf - inf give the NaN (canonical 0x7fffffff); otherwise the infinity there is; the finite terms do not matter
// (a finite sum is exact, however big, and an infinity plus it is that infinity). The products are formed in double, in which a bf16 or
// tf32 product is exact and finite (they reach 2^254).
inline float chunk_special(const float* a, size_t astride, const float* b, size_t bstride, unsigned n, float acc) {
  bool nan = std::isnan(acc), pinf = acc > 0 && std::isinf(acc), ninf = acc < 0 && std::isinf(acc);
  for (unsigned t = 0; t < n; ++t) {
    const double p = static_cast<double>(a[t * astride]) * static_cast<double>(b[t * bstride]);
    nan |= std::isnan(p);
    pinf |= p > 0 && std::isinf(p);
    ninf |= p < 0 && std::isinf(p);
  }
  if (nan || (pinf && ninf)) return std::bit_cast<float>(0x7fffffffu);
  return std::bit_cast<float>(pinf ? 0x7f800000u : ninf ? 0xff800000u : 0u);
}

// D = C + A x B for an M x N output of depth K, chunk by chunk as above. A is M x K (row stride lda), B is K x N (row stride ldb), C and
// D are M x N, every operand a float holding its value exactly (a tf32 operand with its low 13 bits already cleared). D may be C.
#pragma GCC push_options
#pragma GCC optimize("fp-contract=off")
inline void mma_scalar(unsigned M, unsigned N, unsigned K, unsigned chunk, int emin, bool acc_f16, const float* A, size_t lda, const float* B,
                       size_t ldb, const float* C, size_t ldc, float* D, size_t ldd) {
  constexpr unsigned kMaxM = 32, kMaxN = 32, kMaxK = 16;
  Dec da[kMaxM * kMaxK], db[kMaxK * kMaxN];
  uint8_t row_special[kMaxM], col_special[kMaxN];   // bit c: chunk c holds an infinity or a NaN
  for (unsigned i = 0; i < M; ++i) {
    uint8_t bits = 0;
    for (unsigned k = 0; k < K; ++k) {
      const float v = A[i * lda + k];
      if (!std::isfinite(v)) bits |= static_cast<uint8_t>(1u << (k / chunk));
      da[i * K + k] = decode_operand(std::isfinite(v) ? v : 0.0f, emin);
    }
    row_special[i] = bits;
  }
  for (unsigned j = 0; j < N; ++j) {
    uint8_t bits = 0;
    for (unsigned k = 0; k < K; ++k) {
      const float v = B[k * ldb + j];
      if (!std::isfinite(v)) bits |= static_cast<uint8_t>(1u << (k / chunk));
      db[k * N + j] = decode_operand(std::isfinite(v) ? v : 0.0f, emin);
    }
    col_special[j] = bits;
  }
  for (unsigned i = 0; i < M; ++i)
    for (unsigned j = 0; j < N; ++j) {
      float acc = C[i * ldc + j];
      for (unsigned k0 = 0, c = 0; k0 < K; k0 += chunk, ++c) {
        const unsigned n = (K - k0 < chunk) ? K - k0 : chunk;
        if (((row_special[i] | col_special[j]) >> c & 1) || !std::isfinite(acc))
          acc = chunk_special(&A[i * lda + k0], 1, &B[k0 * ldb + j], ldb, n, acc);
        else
          acc = chunk_finite(&da[i * K + k0], 1, &db[k0 * N + j], N, n, acc, acc_f16);
      }
      D[i * ldd + j] = acc;
    }
}
#pragma GCC pop_options

#ifdef VGPU_TC86_AVX2
// |sig| shifted left by s (s >= 0) or right by -s (toward zero), with sig's sign: one aligned term of the sum.
__attribute__((target("avx2"))) inline __m256i shift_term_avx2(__m256i sig, __m256i s) {
  const __m256i zero = _mm256_setzero_si256();
  const __m256i mag = _mm256_abs_epi32(sig);
  const __m256i left = _mm256_sllv_epi32(mag, _mm256_max_epi32(s, zero));
  const __m256i right = _mm256_srlv_epi32(mag, _mm256_max_epi32(_mm256_sub_epi32(zero, s), zero));
  return _mm256_sign_epi32(_mm256_blendv_epi8(right, left, _mm256_cmpgt_epi32(s, _mm256_set1_epi32(-1))), sig);
}

// Eight operands at once: significand (signed), least-significant-bit exponent and frame exponent, as decode_operand gives them; an
// infinity or a NaN decodes as a zero and sets its lane in the returned mask.
__attribute__((target("avx2"))) inline unsigned decode8_avx2(const float* p, int emin, int32_t* sig_out, int32_t* lsb_out, int32_t* ex_out) {
  const __m256i bits = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
  const __m256i e = _mm256_and_si256(_mm256_srli_epi32(bits, 23), _mm256_set1_epi32(0xff)), m = _mm256_and_si256(bits, _mm256_set1_epi32(0x7fffff));
  const __m256i nonfin = _mm256_cmpeq_epi32(e, _mm256_set1_epi32(255));
  const __m256i enz = _mm256_xor_si256(_mm256_cmpeq_epi32(e, _mm256_setzero_si256()), _mm256_set1_epi32(-1));
  __m256i sig = _mm256_or_si256(_mm256_srli_epi32(m, 13), _mm256_and_si256(enz, _mm256_set1_epi32(0x400)));
  const __m256i lsb = _mm256_blendv_epi8(_mm256_set1_epi32(-136), _mm256_sub_epi32(e, _mm256_set1_epi32(137)), enz);
  const __m256i ex = _mm256_blendv_epi8(_mm256_set1_epi32(-126), _mm256_max_epi32(_mm256_sub_epi32(e, _mm256_set1_epi32(127)), _mm256_set1_epi32(emin)), enz);
  sig = _mm256_andnot_si256(nonfin, sig);
  sig = _mm256_sign_epi32(sig, _mm256_or_si256(_mm256_srai_epi32(bits, 31), _mm256_set1_epi32(1)));
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(sig_out), sig);
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(lsb_out), lsb);
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(ex_out), ex);
  return static_cast<unsigned>(_mm256_movemask_ps(_mm256_castsi256_ps(nonfin)));
}

// The same rule over eight columns of the output at a time, in 32-bit lanes: every aligned term is below 2^27, so the sum of nine
// is below 2^31. For an f32 accumulator and an N that is a multiple of 8; an element with an infinity or a NaN in its chunk, or a
// result outside the f32 normal range, is finished by the scalar code, so the two agree on every input (test_tc_ampere.cpp).
__attribute__((target("avx2"))) inline void mma_avx2_generic(unsigned M, unsigned N, unsigned K, unsigned chunk, int emin, const float* A, size_t lda,
                                                              const float* B, size_t ldb, const float* C, size_t ldc, float* D, size_t ldd) {
  constexpr unsigned kMaxM = 32, kMaxN = 32, kMaxK = 16;
  alignas(32) int32_t bsig[kMaxK * kMaxN], blsb[kMaxK * kMaxN], bex[kMaxK * kMaxN];
  uint8_t colmask[4][kMaxN / 8] = {};   // [chunk][block of 8 columns]: bit l set if column l has an infinity or a NaN in the chunk
  alignas(32) int32_t asg[kMaxM * kMaxK + 8], als[kMaxM * kMaxK + 8], aex[kMaxM * kMaxK + 8];
  uint8_t row_special[kMaxM];
  for (unsigned i = 0; i < M; ++i) {
    uint8_t bits = 0;
    unsigned k = 0;
    for (; k + 8 <= K; k += 8) {
      const unsigned nf = decode8_avx2(A + i * lda + k, emin, &asg[i * K + k], &als[i * K + k], &aex[i * K + k]);
      for (unsigned l = 0; l < 8; ++l)
        if (nf >> l & 1) bits |= static_cast<uint8_t>(1u << ((k + l) / chunk));
    }
    for (; k < K; ++k) {
      const float v = A[i * lda + k];
      if (!std::isfinite(v)) bits |= static_cast<uint8_t>(1u << (k / chunk));
      const Dec d = decode_operand(std::isfinite(v) ? v : 0.0f, emin);
      asg[i * K + k] = d.sig;
      als[i * K + k] = d.lsb;
      aex[i * K + k] = d.ex;
    }
    row_special[i] = bits;
  }
  const __m256i c255 = _mm256_set1_epi32(255), c7fffff = _mm256_set1_epi32(0x7fffff), cff = _mm256_set1_epi32(0xff), zero = _mm256_setzero_si256(),
                one = _mm256_set1_epi32(1);
  const __m256i imin = _mm256_set1_epi32(INT32_MIN);
  for (unsigned k = 0; k < K; ++k)
    for (unsigned jb = 0; jb < N / 8; ++jb)
      colmask[k / chunk][jb] |= static_cast<uint8_t>(decode8_avx2(B + k * ldb + jb * 8, emin, &bsig[k * N + jb * 8], &blsb[k * N + jb * 8], &bex[k * N + jb * 8]));
  const __m256i idx_even = _mm256_setr_epi32(0, 2, 4, 6, 0, 0, 0, 0);
  for (unsigned i = 0; i < M; ++i)
    for (unsigned jb = 0; jb < N / 8; ++jb) {
      __m256i accb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(C + i * ldc + jb * 8));
      for (unsigned k0 = 0, c = 0; k0 < K; k0 += chunk, ++c) {
        const unsigned n = (K - k0 < chunk) ? K - k0 : chunk;
        __m256i top = imin;
        __m256i ps[16], pl[16];
        for (unsigned t = 0; t < n; ++t) {
          const unsigned k = k0 + t;
          const __m256i p = _mm256_mullo_epi32(_mm256_set1_epi32(asg[i * K + k]), _mm256_load_si256(reinterpret_cast<const __m256i*>(&bsig[k * N + jb * 8])));
          ps[t] = p;
          pl[t] = _mm256_add_epi32(_mm256_set1_epi32(als[i * K + k]), _mm256_load_si256(reinterpret_cast<const __m256i*>(&blsb[k * N + jb * 8])));
          const __m256i T = _mm256_add_epi32(_mm256_set1_epi32(aex[i * K + k] + 1), _mm256_load_si256(reinterpret_cast<const __m256i*>(&bex[k * N + jb * 8])));
          top = _mm256_max_epi32(top, _mm256_blendv_epi8(T, imin, _mm256_cmpeq_epi32(p, zero)));
        }
        const __m256i ae = _mm256_and_si256(_mm256_srli_epi32(accb, 23), cff), am = _mm256_and_si256(accb, c7fffff);
        const __m256i aenz = _mm256_xor_si256(_mm256_cmpeq_epi32(ae, zero), _mm256_set1_epi32(-1));
        const __m256i acc_nonfin = _mm256_cmpeq_epi32(ae, c255);
        __m256i asig = _mm256_or_si256(am, _mm256_and_si256(aenz, _mm256_set1_epi32(0x800000)));
        const __m256i alsb = _mm256_blendv_epi8(_mm256_set1_epi32(-149), _mm256_sub_epi32(ae, _mm256_set1_epi32(150)), aenz);
        const __m256i lead = _mm256_sub_epi32(_mm256_srli_epi32(_mm256_castps_si256(_mm256_cvtepi32_ps(asig)), 23), _mm256_set1_epi32(127));
        const __m256i Tacc = _mm256_add_epi32(_mm256_add_epi32(lead, alsb), one);
        top = _mm256_max_epi32(top, _mm256_blendv_epi8(Tacc, imin, _mm256_cmpeq_epi32(asig, zero)));
        asig = _mm256_sign_epi32(asig, _mm256_or_si256(_mm256_srai_epi32(accb, 31), one));
        const __m256i allz = _mm256_cmpeq_epi32(top, imin);
        const __m256i g = _mm256_andnot_si256(allz, _mm256_sub_epi32(top, _mm256_set1_epi32(25)));
        __m256i S = shift_term_avx2(asig, _mm256_sub_epi32(alsb, g));
        // A product's shift is at most 4 to the left (its frame position is at most the largest), so it is the product << 4 shifted
        // right by 4 - s: the same magnitude truncated toward zero, with one variable shift and no choice of direction.
        const __m256i g4 = _mm256_add_epi32(g, _mm256_set1_epi32(4));
        for (unsigned t = 0; t < n; ++t) {
          const __m256i v = _mm256_srlv_epi32(_mm256_slli_epi32(_mm256_abs_epi32(ps[t]), 4), _mm256_sub_epi32(g4, pl[t]));
          S = _mm256_add_epi32(S, _mm256_sign_epi32(v, ps[t]));
        }
        // The sum truncated to 24 significant bits, as an f32: its bit length from the exponent of the double it converts to exactly.
        const __m256i neg = _mm256_srai_epi32(S, 31);
        const __m256i mag = _mm256_abs_epi32(S);
        const __m256d dlo = _mm256_cvtepi32_pd(_mm256_castsi256_si128(mag)), dhi = _mm256_cvtepi32_pd(_mm256_extracti128_si256(mag, 1));
        const __m256i elo = _mm256_srli_epi64(_mm256_castpd_si256(dlo), 52), ehi = _mm256_srli_epi64(_mm256_castpd_si256(dhi), 52);
        const __m256i bl = _mm256_inserti128_si256(_mm256_permutevar8x32_epi32(elo, idx_even), _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(ehi, idx_even)), 1);
        const __m256i floorlog = _mm256_sub_epi32(bl, _mm256_set1_epi32(1023));
        const __m256i sh = _mm256_max_epi32(_mm256_sub_epi32(floorlog, _mm256_set1_epi32(23)), zero);
        const __m256i magt = _mm256_sllv_epi32(_mm256_srlv_epi32(mag, sh), sh);
        const __m256i fb = _mm256_castps_si256(_mm256_cvtepi32_ps(magt));
        const __m256i newexp = _mm256_add_epi32(_mm256_and_si256(_mm256_srli_epi32(fb, 23), cff), g);
        const __m256i snz = _mm256_xor_si256(_mm256_cmpeq_epi32(S, zero), _mm256_set1_epi32(-1));
        const __m256i inrange = _mm256_and_si256(_mm256_cmpgt_epi32(newexp, zero), _mm256_cmpgt_epi32(_mm256_set1_epi32(255), newexp));
        const __m256i valid = _mm256_and_si256(snz, inrange);
        __m256i res = _mm256_or_si256(_mm256_or_si256(_mm256_and_si256(fb, _mm256_set1_epi32(0x007fffff)), _mm256_slli_epi32(newexp, 23)),
                                      _mm256_and_si256(neg, _mm256_set1_epi32(static_cast<int>(0x80000000u))));
        res = _mm256_and_si256(res, valid);
        alignas(32) uint32_t out[8];
        _mm256_store_si256(reinterpret_cast<__m256i*>(out), res);
        // Lanes the vector code does not finish: an overflow, or a result below the normal range (the scalar conversion), and the
        // chunks with an infinity or a NaN in them.
        const unsigned redo = static_cast<unsigned>(_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_andnot_si256(inrange, snz))));
        const unsigned special = ((row_special[i] >> c) & 1 ? 0xffu : 0u) | colmask[c][jb] |
                                 static_cast<unsigned>(_mm256_movemask_ps(_mm256_castsi256_ps(acc_nonfin)));
        if (redo | special) {
          alignas(32) int32_t Sv[8], gv[8];
          alignas(32) uint32_t accv[8];
          _mm256_store_si256(reinterpret_cast<__m256i*>(Sv), S);
          _mm256_store_si256(reinterpret_cast<__m256i*>(gv), g);
          _mm256_store_si256(reinterpret_cast<__m256i*>(accv), accb);
          for (unsigned l = 0; l < 8; ++l) {
            if (special >> l & 1)
              out[l] = std::bit_cast<uint32_t>(chunk_special(&A[i * lda + k0], 1, &B[k0 * ldb + jb * 8 + l], ldb, n, std::bit_cast<float>(accv[l])));
            else if (redo >> l & 1)
              out[l] = std::bit_cast<uint32_t>(detail::to_f32_rz(Sv[l], gv[l]));
          }
        }
        accb = _mm256_load_si256(reinterpret_cast<const __m256i*>(out));
      }
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(D + i * ldd + jb * 8), accb);
    }
}

// Eight operands at once, as decode8_avx2, for the kernel below: the frame term is stored in place of the frame exponent, with an
// operand that is zero (or an infinity or a NaN, decoded as a zero) given kNoTerm so that a product with a zero factor never
// chooses the largest frame position; `bias` is added to a live term (1 for the A side: a product's position is ea + eb + 1).
inline constexpr int32_t kNoTerm = -(1 << 28);
__attribute__((target("avx2"))) inline unsigned decode8_term_avx2(const float* p, int emin, int bias, int32_t* sig_out, int32_t* lsb_out, int32_t* term_out) {
  const __m256i bits = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
  const __m256i e = _mm256_and_si256(_mm256_srli_epi32(bits, 23), _mm256_set1_epi32(0xff)), m = _mm256_and_si256(bits, _mm256_set1_epi32(0x7fffff));
  const __m256i nonfin = _mm256_cmpeq_epi32(e, _mm256_set1_epi32(255));
  const __m256i enz = _mm256_xor_si256(_mm256_cmpeq_epi32(e, _mm256_setzero_si256()), _mm256_set1_epi32(-1));
  __m256i sig = _mm256_or_si256(_mm256_srli_epi32(m, 13), _mm256_and_si256(enz, _mm256_set1_epi32(0x400)));
  const __m256i lsb = _mm256_blendv_epi8(_mm256_set1_epi32(-136), _mm256_sub_epi32(e, _mm256_set1_epi32(137)), enz);
  const __m256i ex = _mm256_blendv_epi8(_mm256_set1_epi32(-126), _mm256_max_epi32(_mm256_sub_epi32(e, _mm256_set1_epi32(127)), _mm256_set1_epi32(emin)), enz);
  sig = _mm256_andnot_si256(nonfin, sig);
  const __m256i term = _mm256_blendv_epi8(_mm256_add_epi32(ex, _mm256_set1_epi32(bias)), _mm256_set1_epi32(kNoTerm), _mm256_cmpeq_epi32(sig, _mm256_setzero_si256()));
  sig = _mm256_sign_epi32(sig, _mm256_or_si256(_mm256_srai_epi32(bits, 31), _mm256_set1_epi32(1)));
  if (bias != 0) sig = _mm256_slli_epi32(sig, 4);   // the A side keeps its significand times 16 (see mma_avx2_chunked)
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(sig_out), sig);
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(lsb_out), lsb);
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(term_out), term);
  return static_cast<unsigned>(_mm256_movemask_ps(_mm256_castsi256_ps(nonfin)));
}

// The sum S * 2^g (S a signed 32-bit integer per lane, |S| below 2^31) truncated toward zero to 24 significant bits, as an f32 (detail::to_f32_rz
// for the lanes that are normal): `res` holds it, `snz` marks the lanes with S != 0 and `inrange` those whose result is a normal f32;
// a lane with S != 0 outside that range (res is 0 there) is finished by the caller with the scalar conversion. The bit length comes from
// the float of the magnitude with its low 7 bits cleared (at most 24 significant bits for a magnitude below 2^31, so exact; a magnitude
// below 128 needs no truncation).
__attribute__((target("avx2"))) inline void sum_to_f32_avx2(__m256i S, __m256i g, __m256i cff, __m256i c7fffff, __m256i& res, __m256i& snz, __m256i& inrange) {
  const __m256i zero = _mm256_setzero_si256(), c150 = _mm256_set1_epi32(150), c255 = _mm256_set1_epi32(255);
  const __m256i neg = _mm256_srai_epi32(S, 31);
  const __m256i mag = _mm256_abs_epi32(S);
  const __m256i hi = _mm256_castps_si256(_mm256_cvtepi32_ps(_mm256_and_si256(mag, _mm256_set1_epi32(~127))));
  const __m256i sh = _mm256_max_epi32(_mm256_sub_epi32(_mm256_srli_epi32(hi, 23), c150), zero);   // floor(log2) - 23, at least 0
  const __m256i magt = _mm256_sllv_epi32(_mm256_srlv_epi32(mag, sh), sh);
  const __m256i fb = _mm256_castps_si256(_mm256_cvtepi32_ps(magt));
  const __m256i newexp = _mm256_add_epi32(_mm256_and_si256(_mm256_srli_epi32(fb, 23), cff), g);
  snz = _mm256_xor_si256(_mm256_cmpeq_epi32(S, zero), _mm256_set1_epi32(-1));
  inrange = _mm256_and_si256(_mm256_cmpgt_epi32(newexp, zero), _mm256_cmpgt_epi32(c255, newexp));
  const __m256i valid = _mm256_and_si256(snz, inrange);
  res = _mm256_or_si256(_mm256_or_si256(_mm256_and_si256(fb, c7fffff), _mm256_slli_epi32(newexp, 23)),
                        _mm256_and_si256(neg, _mm256_set1_epi32(static_cast<int>(0x80000000u))));
  res = _mm256_and_si256(res, valid);
}

// The sum S * 2^g rounded to nearest even to f16, as the float that holds it (detail::to_f16_rn): the same lanes as sum_to_f32_avx2 give
// (`snz` marks S != 0, `inrange` the lanes this code answers; a lane that is not is finished with the scalar conversion). The significand
// is S's magnitude shifted right to the f16 grid of its binade (or of the subnormal range) with the remainder deciding the rounding, the
// float is that integer times a power of two, an exponent past f16's is an infinity, a result that rounds to zero keeps the sign of S.
__attribute__((target("avx2"))) inline void sum_to_f16_avx2(__m256i S, __m256i g, __m256i& res, __m256i& snz, __m256i& inrange) {
  const __m256i zero = _mm256_setzero_si256(), one = _mm256_set1_epi32(1), ones = _mm256_set1_epi32(-1), sb = _mm256_set1_epi32(static_cast<int>(0x80000000u));
  const __m256i neg = _mm256_srai_epi32(S, 31);
  const __m256i mag = _mm256_abs_epi32(S);
  const __m256i hi = _mm256_and_si256(mag, _mm256_set1_epi32(~127));
  // floor(log2 mag) from a float that holds the magnitude, or its top 24 bits, exactly
  const __m256i x = _mm256_blendv_epi8(mag, hi, _mm256_cmpgt_epi32(hi, zero));
  const __m256i lg = _mm256_sub_epi32(_mm256_srli_epi32(_mm256_castps_si256(_mm256_cvtepi32_ps(x)), 23), _mm256_set1_epi32(127));
  const __m256i E = _mm256_add_epi32(lg, g);
  const __m256i Eq = _mm256_sub_epi32(_mm256_max_epi32(E, _mm256_set1_epi32(-14)), _mm256_set1_epi32(10));   // exponent of the grid's step
  const __m256i r = _mm256_sub_epi32(Eq, g);                                                                   // bits below the grid
  const __m256i rp = _mm256_min_epi32(_mm256_max_epi32(r, zero), _mm256_set1_epi32(32));
  const __m256i l = _mm256_max_epi32(_mm256_sub_epi32(zero, r), zero);
  snz = _mm256_xor_si256(_mm256_cmpeq_epi32(S, zero), ones);
  inrange = _mm256_cmpgt_epi32(r, _mm256_set1_epi32(-32));
  const __m256i q = _mm256_srlv_epi32(mag, rp);
  const __m256i rem = _mm256_and_si256(mag, _mm256_sub_epi32(_mm256_sllv_epi32(one, rp), one));
  const __m256i half = _mm256_sllv_epi32(one, _mm256_sub_epi32(rp, one));   // (a count of -1 is past 31: zero)
  const __m256i gt = _mm256_cmpgt_epi32(_mm256_xor_si256(rem, sb), _mm256_xor_si256(half, sb));   // unsigned
  const __m256i tie = _mm256_and_si256(_mm256_and_si256(_mm256_cmpeq_epi32(rem, half), _mm256_cmpeq_epi32(_mm256_and_si256(q, one), one)),
                                       _mm256_xor_si256(_mm256_cmpeq_epi32(half, zero), ones));
  const __m256i up = _mm256_or_si256(gt, tie);   // -1 where the significand rounds up
  const __m256i R = _mm256_sllv_epi32(_mm256_sub_epi32(q, up), l);
  const __m256i fR = _mm256_castps_si256(_mm256_cvtepi32_ps(R));
  const __m256i Rz = _mm256_cmpeq_epi32(R, zero);
  const __m256i eres = _mm256_add_epi32(_mm256_sub_epi32(_mm256_srli_epi32(fR, 23), _mm256_set1_epi32(127)), Eq);
  const __m256i ovf = _mm256_cmpgt_epi32(eres, _mm256_set1_epi32(15));
  __m256i val = _mm256_andnot_si256(Rz, _mm256_add_epi32(fR, _mm256_slli_epi32(Eq, 23)));
  val = _mm256_blendv_epi8(val, _mm256_set1_epi32(0x7f800000), ovf);
  res = _mm256_and_si256(_mm256_or_si256(val, _mm256_and_si256(neg, sb)), snz);
}

// mma_avx2_generic with the chunk length a compile-time constant (K a multiple of it) and the work per chunk cut down; the same bits
// (tests/unit/test_tc_ampere_fast.cpp holds it to the earlier form over random and adversarial operands):
//   * a product with a zero factor is kept out of the largest frame position by a term of kNoTerm on its operand, not by a compare
//     of the product, and an all-zero chunk needs no special case (every aligned term is then zero whatever the shift);
//   * the accumulator's own frame position is not blended away when it is zero (it is below every product's, so it never wins);
//   * A's significands are kept times 16, so that a product is already its magnitude << 4 (a product's shift is at most 4 to the left
//     of the largest frame position, so it is that magnitude shifted right by 4 - s: truncation toward zero with one variable shift);
//   * the bit length of the sum comes from a float conversion of its top 24 bits, not from two double conversions and permutes;
//   * the new accumulator stays in its register; the lanes the vector code does not finish are the only ones that touch memory;
//   * VGPU_TC86_ROWS rows at a time, stage by stage, so that their independent dependency chains overlap.
template <unsigned NC, bool ACC16>
__attribute__((target("avx2"))) inline void mma_avx2_chunked(unsigned M, unsigned N, unsigned K, int emin, const float* A, size_t lda, const float* B,
                                                            size_t ldb, const float* C, size_t ldc, float* D, size_t ldd) {
  constexpr unsigned kMaxM = 32, kMaxN = 32, kMaxK = 16;
  alignas(32) int32_t bsig[kMaxK * kMaxN], blsb[kMaxK * kMaxN], bterm[kMaxK * kMaxN];
  uint8_t colmask[4][kMaxN / 8] = {};
  alignas(32) int32_t asg[kMaxM * kMaxK + 8], als[kMaxM * kMaxK + 8], aterm[kMaxM * kMaxK + 8];
  uint8_t row_special[kMaxM];
  for (unsigned i = 0; i < M; ++i) {
    uint8_t bits = 0;
    unsigned k = 0;
    for (; k + 8 <= K; k += 8) {
      const unsigned nf = decode8_term_avx2(A + i * lda + k, emin, 1, &asg[i * K + k], &als[i * K + k], &aterm[i * K + k]);
      for (unsigned l = 0; l < 8; ++l)
        if (nf >> l & 1) bits |= static_cast<uint8_t>(1u << ((k + l) / NC));
    }
    for (; k < K; ++k) {
      const float v = A[i * lda + k];
      if (!std::isfinite(v)) bits |= static_cast<uint8_t>(1u << (k / NC));
      const Dec d = decode_operand(std::isfinite(v) ? v : 0.0f, emin);
      asg[i * K + k] = d.sig * 16;
      als[i * K + k] = d.lsb;
      aterm[i * K + k] = d.sig == 0 ? kNoTerm : d.ex + 1;
    }
    row_special[i] = bits;
  }
  for (unsigned k = 0; k < K; ++k)
    for (unsigned jb = 0; jb < N / 8; ++jb)
      colmask[k / NC][jb] |= static_cast<uint8_t>(
          decode8_term_avx2(B + k * ldb + jb * 8, emin, 0, &bsig[k * N + jb * 8], &blsb[k * N + jb * 8], &bterm[k * N + jb * 8]));
  const __m256i c255 = _mm256_set1_epi32(255), c7fffff = _mm256_set1_epi32(0x7fffff), cff = _mm256_set1_epi32(0xff), zero = _mm256_setzero_si256(),
                one = _mm256_set1_epi32(1), c25 = _mm256_set1_epi32(25), c4 = _mm256_set1_epi32(4);
  const __m256i c150 = _mm256_set1_epi32(150);
  // R rows at a time, stage by stage, so that their independent dependency chains sit next to each other in the instruction stream.
  const auto rows = [&]<unsigned R>(std::integral_constant<unsigned, R>, unsigned i0) __attribute__((target("avx2"))) {
    for (unsigned jb = 0; jb < N / 8; ++jb) {
      __m256i accb[R];
      for (unsigned r = 0; r < R; ++r) accb[r] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(C + (i0 + r) * ldc + jb * 8));
      for (unsigned k0 = 0, c = 0; k0 < K; k0 += NC, ++c) {
        __m256i top[R], g[R], S[R], acc_nonfin[R];
        for (unsigned r = 0; r < R; ++r) {
          const unsigned i = i0 + r;
          top[r] = _mm256_set1_epi32(kNoTerm);
          for (unsigned t = 0; t < NC; ++t) {
            const unsigned k = k0 + t;
            top[r] = _mm256_max_epi32(top[r], _mm256_add_epi32(_mm256_set1_epi32(aterm[i * K + k]), _mm256_load_si256(reinterpret_cast<const __m256i*>(&bterm[k * N + jb * 8]))));
          }
        }
        for (unsigned r = 0; r < R; ++r) {
          const __m256i ae = _mm256_and_si256(_mm256_srli_epi32(accb[r], 23), cff), am = _mm256_and_si256(accb[r], c7fffff);
          const __m256i aenz = _mm256_xor_si256(_mm256_cmpeq_epi32(ae, zero), _mm256_set1_epi32(-1));
          acc_nonfin[r] = _mm256_cmpeq_epi32(ae, c255);
          __m256i asig = _mm256_or_si256(am, _mm256_and_si256(aenz, _mm256_set1_epi32(0x800000)));
          const __m256i alsb = _mm256_blendv_epi8(_mm256_set1_epi32(-149), _mm256_sub_epi32(ae, c150), aenz);
          const __m256i lead = _mm256_sub_epi32(_mm256_srli_epi32(_mm256_castps_si256(_mm256_cvtepi32_ps(asig)), 23), _mm256_set1_epi32(127));
          top[r] = _mm256_max_epi32(top[r], _mm256_add_epi32(_mm256_add_epi32(lead, alsb), one));   // (a zero accumulator: -275 at most, below any product's)
          asig = _mm256_sign_epi32(asig, _mm256_or_si256(_mm256_srai_epi32(accb[r], 31), one));
          g[r] = _mm256_sub_epi32(top[r], c25);
          S[r] = shift_term_avx2(asig, _mm256_sub_epi32(alsb, g[r]));
        }
        // (The products are formed again here, not kept from the pass above: sixteen live vectors would spill.)
        for (unsigned r = 0; r < R; ++r) {
          const unsigned i = i0 + r;
          const __m256i g4 = _mm256_add_epi32(g[r], c4);
          for (unsigned t = 0; t < NC; ++t) {
            const unsigned k = k0 + t;
            const __m256i p = _mm256_mullo_epi32(_mm256_set1_epi32(asg[i * K + k]), _mm256_load_si256(reinterpret_cast<const __m256i*>(&bsig[k * N + jb * 8])));
            const __m256i cnt = _mm256_sub_epi32(_mm256_sub_epi32(g4, _mm256_set1_epi32(als[i * K + k])), _mm256_load_si256(reinterpret_cast<const __m256i*>(&blsb[k * N + jb * 8])));
            const __m256i v = _mm256_srlv_epi32(_mm256_abs_epi32(p), cnt);
            S[r] = _mm256_add_epi32(S[r], _mm256_sign_epi32(v, p));
          }
        }
        __m256i res[R], snz[R], inrange[R];
        for (unsigned r = 0; r < R; ++r) {
          if constexpr (ACC16)
            sum_to_f16_avx2(S[r], g[r], res[r], snz[r], inrange[r]);
          else
            sum_to_f32_avx2(S[r], g[r], cff, c7fffff, res[r], snz[r], inrange[r]);
        }
        for (unsigned r = 0; r < R; ++r) {
          const unsigned i = i0 + r;
          const unsigned redo = static_cast<unsigned>(_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_andnot_si256(inrange[r], snz[r]))));
          const unsigned special = ((row_special[i] >> c) & 1 ? 0xffu : 0u) | colmask[c][jb] |
                                   static_cast<unsigned>(_mm256_movemask_ps(_mm256_castsi256_ps(acc_nonfin[r])));
          if (__builtin_expect(redo | special, 0)) {
            alignas(32) int32_t Sv[8], gv[8];
            alignas(32) uint32_t accv[8], out[8];
            _mm256_store_si256(reinterpret_cast<__m256i*>(Sv), S[r]);
            _mm256_store_si256(reinterpret_cast<__m256i*>(gv), g[r]);
            _mm256_store_si256(reinterpret_cast<__m256i*>(accv), accb[r]);
            _mm256_store_si256(reinterpret_cast<__m256i*>(out), res[r]);
            for (unsigned l = 0; l < 8; ++l) {
              if (special >> l & 1)
                out[l] = std::bit_cast<uint32_t>(chunk_special(&A[i * lda + k0], 1, &B[k0 * ldb + jb * 8 + l], ldb, NC, std::bit_cast<float>(accv[l])));
              else if (redo >> l & 1)
                out[l] = std::bit_cast<uint32_t>(ACC16 ? detail::to_f16_rn(Sv[l], gv[l]) : detail::to_f32_rz(Sv[l], gv[l]));
            }
            res[r] = _mm256_load_si256(reinterpret_cast<const __m256i*>(out));
          }
          accb[r] = res[r];
        }
      }
      for (unsigned r = 0; r < R; ++r) _mm256_storeu_si256(reinterpret_cast<__m256i*>(D + (i0 + r) * ldd + jb * 8), accb[r]);
    }
  };
  unsigned i = 0;
  for (; i + VGPU_TC86_ROWS <= M; i += VGPU_TC86_ROWS) rows(std::integral_constant<unsigned, VGPU_TC86_ROWS>{}, i);
  for (; i < M; ++i) rows(std::integral_constant<unsigned, 1>{}, i);
}

// A finite chunk on raw floats, by the scalar rule (chunk_finite): for the lanes the vector form below does not finish.
inline float chunk_finite_raw(const float* a, size_t astride, const float* b, size_t bstride, unsigned n, float acc, bool acc_f16, int emin) {
  Dec da[16], db[16];
  for (unsigned t = 0; t < n; ++t) {
    da[t] = decode_operand(a[t * astride], emin);
    db[t] = decode_operand(b[t * bstride], emin);
  }
  return chunk_finite(da, 1, db, 1, n, acc, acc_f16);
}

#ifdef VGPU_TC86_AVX2
// The chunks of eight lanes that mma_avx2_chunked_f does not finish in vector form: the lanes with an infinity or a NaN (`special`), those whose scale
// 2^-g is out of range (`bad`) and those whose sum needs the rounding of the scalar form (`redo`); the other lanes of `res` stay. Out of line and cold,
// so that the hot loop is only its vector code (the eight rows' copies of this were a third of the function, between the hot lines).
template <unsigned NC, bool ACC16>
__attribute__((target("avx2"), noinline, cold)) __m256i mma_f_fix(__m256i S, __m256i g, __m256i accb, __m256i res, unsigned redo, unsigned special, unsigned bad,
                                                                  const float* a, const float* b, size_t ldb, int emin) {
  alignas(32) int32_t Sv[8], gv[8];
  alignas(32) uint32_t accv[8], out[8];
  _mm256_store_si256(reinterpret_cast<__m256i*>(Sv), S);
  _mm256_store_si256(reinterpret_cast<__m256i*>(gv), g);
  _mm256_store_si256(reinterpret_cast<__m256i*>(accv), accb);
  _mm256_store_si256(reinterpret_cast<__m256i*>(out), res);
  for (unsigned l = 0; l < 8; ++l) {
    if (special >> l & 1)
      out[l] = std::bit_cast<uint32_t>(chunk_special(a, 1, b + l, ldb, NC, std::bit_cast<float>(accv[l])));
    else if (bad >> l & 1)
      out[l] = std::bit_cast<uint32_t>(chunk_finite_raw(a, 1, b + l, ldb, NC, std::bit_cast<float>(accv[l]), ACC16, emin));
    else if (redo >> l & 1)
      out[l] = std::bit_cast<uint32_t>(ACC16 ? detail::to_f16_rn(Sv[l], gv[l]) : detail::to_f32_rz(Sv[l], gv[l]));
  }
  return _mm256_load_si256(reinterpret_cast<const __m256i*>(out));
}

// mma_avx2_chunked with the products formed in floating point. A tf32, bf16 or f16 operand has at most 11 significant bits, so the product of
// two is exact in an f32 -- as long as it neither overflows nor falls below the normal range, which the exponents of the operands say up front
// (the function returns false, having written nothing, when they do not promise it, or an operand is a float subnormal; the integer kernel then
// does the work). An aligned term is the product's magnitude shifted right by (g - its least bit) with the bits shifted out dropped, which is
// the product times 2^-g truncated toward zero: one multiply by that power of two (a float whose exponent field is 127 - g) and one
// truncating conversion, in place of the integer product, two subtractions, an absolute value, a variable shift and a sign. An exact term
// below 1 after the scaling truncates to 0 whatever the multiply does with it, and a term is below 2^27 (the sum of nine is below 2^31).
// What the float form cannot say stays on the old path: a chunk whose g puts 2^-g outside the float range (a subnormal accumulator and
// nothing to speak of beside it) is finished by the scalar rule, as are the chunks with an infinity or a NaN.
template <unsigned NC, bool ACC16>
__attribute__((target("avx2"), aligned(64))) inline bool mma_avx2_chunked_f(unsigned M, unsigned N, unsigned K, int emin, const float* A, size_t lda, const float* B,
                                                              size_t ldb, const float* C, size_t ldc, float* D, size_t ldd) {
  constexpr unsigned kMaxM = 32, kMaxN = 32, kMaxK = 16;
  alignas(32) float bf[kMaxK * kMaxN];
  alignas(32) int32_t bterm[kMaxK * kMaxN];
  unsigned any_nonfin = 0;   // an infinity or a NaN anywhere: the integer kernel takes the call
  alignas(32) float af[kMaxM * kMaxK + 8];
  alignas(32) int32_t aterm[kMaxM * kMaxK + 8];
  const __m256i zero = _mm256_setzero_si256(), c255 = _mm256_set1_epi32(255), cff = _mm256_set1_epi32(0xff), c7fffff = _mm256_set1_epi32(0x7fffff),
                one = _mm256_set1_epi32(1), c25 = _mm256_set1_epi32(25), c127 = _mm256_set1_epi32(127), c150 = _mm256_set1_epi32(150),
                cnoterm = _mm256_set1_epi32(kNoTerm), cemin = _mm256_set1_epi32(emin), cmask = _mm256_set1_epi32(static_cast<int>(0xffffe000u));
  // The exponent fields seen, per side: the largest (an infinity or a NaN counts as 255, which sends the call to the integer kernel when the
  // other side has anything but zeros) and, as an unsigned number, the smallest minus one (a zero is then the largest); and whether a float
  // subnormal with a significand turned up.
  __m256i amx = zero, amn = _mm256_set1_epi32(-1), bmx = zero, bmn = _mm256_set1_epi32(-1), subn = zero;
  const auto decode8 = [&](const float* p, int bias, float* fout, int32_t* tout, __m256i& mx, __m256i& mn) __attribute__((target("avx2"))) {
    const __m256i bits = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
    const __m256i masked = _mm256_and_si256(bits, cmask);
    const __m256i e = _mm256_and_si256(_mm256_srli_epi32(bits, 23), cff);
    const __m256i nonfin = _mm256_cmpeq_epi32(e, c255), ez = _mm256_cmpeq_epi32(e, zero);
    // a subnormal float whose significand (the top ten bits of its mantissa) is not zero: the mantissa bits that a zero exponent leaves
    subn = _mm256_or_si256(subn, _mm256_and_si256(_mm256_and_si256(ez, masked), c7fffff));
    const __m256i term = _mm256_add_epi32(_mm256_max_epi32(_mm256_sub_epi32(e, c127), cemin), _mm256_set1_epi32(bias));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(tout), _mm256_blendv_epi8(term, cnoterm, _mm256_or_si256(nonfin, ez)));
    _mm256_storeu_ps(fout, _mm256_castsi256_ps(masked));
    mx = _mm256_max_epi32(mx, e);
    mn = _mm256_min_epu32(mn, _mm256_sub_epi32(e, one));
    return static_cast<unsigned>(_mm256_movemask_ps(_mm256_castsi256_ps(nonfin)));
  };
  int sm_x = 0, sm_n = 255;   // the tail's exponent fields, for A
  bool tail_sub = false;
  for (unsigned i = 0; i < M; ++i) {
    unsigned k = 0;
    for (; k + 8 <= K; k += 8) {
      const unsigned nf = decode8(A + i * lda + k, 1, &af[i * K + k], &aterm[i * K + k], amx, amn);
      any_nonfin |= nf;
    }
    for (; k < K; ++k) {
      const uint32_t b = std::bit_cast<uint32_t>(A[i * lda + k]);
      const uint32_t e = (b >> 23) & 0xffu, mb = b & 0xffffe000u;
      af[i * K + k] = std::bit_cast<float>(mb);
      if (e == 255) any_nonfin = 1;
      const bool dead = e == 0 || e == 255;
      if (e == 0 && (mb & 0x7fffffu)) tail_sub = true;
      aterm[i * K + k] = dead ? kNoTerm : std::max(static_cast<int32_t>(e) - 127, emin) + 1;
      if (!dead) {
        sm_x = std::max(sm_x, static_cast<int>(e));
        sm_n = std::min(sm_n, static_cast<int>(e));
      }
    }
  }
  for (unsigned jb = 0; jb < N / 8; ++jb)
    for (unsigned k0 = 0; k0 < K; k0 += NC) {
      unsigned nf = 0;
      for (unsigned k = k0; k < k0 + NC; ++k) nf |= decode8(B + k * ldb + jb * 8, 0, &bf[k * N + jb * 8], &bterm[k * N + jb * 8], bmx, bmn);
      any_nonfin |= nf;
    }
  // The reductions of the exponent vectors, through memory.
  const auto hmax = [](__m256i v) __attribute__((target("avx2"))) {
    alignas(32) int32_t t[8];
    _mm256_store_si256(reinterpret_cast<__m256i*>(t), v);
    return *std::max_element(t, t + 8);
  };
  if (any_nonfin || tail_sub || !_mm256_testz_si256(subn, subn)) return false;
  const auto hminu = [](__m256i v) __attribute__((target("avx2"))) {
    alignas(32) uint32_t t[8];
    _mm256_store_si256(reinterpret_cast<__m256i*>(t), v);
    return *std::min_element(t, t + 8);
  };
  // (a field of 0 for a zero, as a minimum minus one it is the largest unsigned number: no nonzero operand at all reads 255 here)
  const auto field_min = [](uint32_t m) { return m >= 254 ? 255 : static_cast<int>(m) + 1; };
  const int a_max = std::max(hmax(amx), sm_x), a_min = std::min(field_min(hminu(amn)), sm_n), b_max = hmax(bmx), b_min = field_min(hminu(bmn));
  // Products between 2^-126 and 2^127, whatever pair meets: smallest exponents added (fields: the unbiased sum is a_min + b_min - 254) at
  // least -126, largest at most 125 (a product is below 2^(sum + 2)).
  if (a_min <= 254 && b_min <= 254 && a_min + b_min < 128) return false;
  if (a_max + b_max > 379) return false;
  // R rows at a time, stage by stage, so that their independent dependency chains sit next to each other in the instruction stream.
  // (Everything the lambda needs is copied into locals first: through the closure it was reloaded after every store, which may alias it.)
  const auto rows = [&]<unsigned R>(std::integral_constant<unsigned, R>, unsigned i0) __attribute__((target("avx2"))) {
    const unsigned Nl = N, Kl = K, nb = N / 8;
    const size_t ldcl = ldc, lddl = ldd, ldbl = ldb;
    const float* const Cl = C;
    float* const Dl = D;
    const float* const Al = A;
    const float* const Bl = B;
    const int32_t* const atl = aterm;
    const int32_t* const btl = bterm;
    const float* const afl = af;
    const float* const bfl = bf;
    const int emin_l = emin;
    for (unsigned jb = 0; jb < nb; ++jb) {
      __m256i accb[R];
      VGPU_TC86_UNROLL
      for (unsigned r = 0; r < R; ++r) accb[r] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(Cl + (i0 + r) * ldcl + jb * 8));
      for (unsigned k0 = 0, c = 0; k0 < Kl; k0 += NC, ++c) {
        __m256i top[R], g[R], S[R], acc_nonfin[R];
        __m256 scale[R];
        __m256i badg[R];
        const int32_t* const bt = btl + size_t{k0} * Nl + jb * 8;
        const float* const bp = bfl + size_t{k0} * Nl + jb * 8;
        VGPU_TC86_UNROLL
        for (unsigned r = 0; r < R; ++r) {
          const int32_t* const at = atl + size_t{i0 + r} * Kl + k0;
          top[r] = cnoterm;
          VGPU_TC86_UNROLL
          for (unsigned t = 0; t < NC; ++t)
            top[r] = _mm256_max_epi32(top[r], _mm256_add_epi32(_mm256_set1_epi32(at[t]), _mm256_load_si256(reinterpret_cast<const __m256i*>(bt + size_t{t} * Nl))));
        }
        VGPU_TC86_UNROLL
        for (unsigned r = 0; r < R; ++r) {
          const __m256i ae = _mm256_and_si256(_mm256_srli_epi32(accb[r], 23), cff), am = _mm256_and_si256(accb[r], c7fffff);
          const __m256i aenz = _mm256_xor_si256(_mm256_cmpeq_epi32(ae, zero), _mm256_set1_epi32(-1));
          acc_nonfin[r] = _mm256_cmpeq_epi32(ae, c255);
          __m256i asig = _mm256_or_si256(am, _mm256_and_si256(aenz, _mm256_set1_epi32(0x800000)));
          const __m256i alsb = _mm256_blendv_epi8(_mm256_set1_epi32(-149), _mm256_sub_epi32(ae, c150), aenz);
          const __m256i lead = _mm256_sub_epi32(_mm256_srli_epi32(_mm256_castps_si256(_mm256_cvtepi32_ps(asig)), 23), c127);
          top[r] = _mm256_max_epi32(top[r], _mm256_add_epi32(_mm256_add_epi32(lead, alsb), one));   // (a zero accumulator: -275 at most, below any product's)
          asig = _mm256_sign_epi32(asig, _mm256_or_si256(_mm256_srai_epi32(accb[r], 31), one));
          g[r] = _mm256_sub_epi32(top[r], c25);
          S[r] = shift_term_avx2(asig, _mm256_sub_epi32(alsb, g[r]));
          // 2^-g as a float, for a chunk with something in it (top above -200: a zero accumulator is -275 and every product's term is
          // above that; a chunk of zeros has nothing to scale), and the lanes whose g it cannot be made for.
          const __m256i live = _mm256_cmpgt_epi32(top[r], _mm256_set1_epi32(-200));
          const __m256i sexp = _mm256_and_si256(_mm256_sub_epi32(c127, g[r]), live);   // the biased exponent of 2^-g (0 for a chunk of zeros: scale 0 * anything = 0)
          badg[r] = _mm256_and_si256(live, _mm256_or_si256(_mm256_cmpgt_epi32(one, sexp), _mm256_cmpgt_epi32(sexp, _mm256_set1_epi32(254))));
          scale[r] = _mm256_castsi256_ps(_mm256_slli_epi32(sexp, 23));
        }
        // (The products are formed again here, not kept from the pass above: sixteen live vectors would spill.)
        VGPU_TC86_UNROLL
        for (unsigned r = 0; r < R; ++r) {
          const float* const ap = afl + size_t{i0 + r} * Kl + k0;
          VGPU_TC86_UNROLL
          for (unsigned t = 0; t < NC; ++t) {
            const __m256 x = _mm256_mul_ps(_mm256_broadcast_ss(ap + t), _mm256_load_ps(bp + size_t{t} * Nl));
            S[r] = _mm256_add_epi32(S[r], _mm256_cvttps_epi32(_mm256_mul_ps(x, scale[r])));
          }
        }
        __m256i res[R], snz[R], inrange[R];
        VGPU_TC86_UNROLL
        for (unsigned r = 0; r < R; ++r) {
          if constexpr (ACC16)
            sum_to_f16_avx2(S[r], g[r], res[r], snz[r], inrange[r]);
          else
            sum_to_f32_avx2(S[r], g[r], cff, c7fffff, res[r], snz[r], inrange[r]);
        }
        VGPU_TC86_UNROLL
        for (unsigned r = 0; r < R; ++r) {
          const unsigned redo = static_cast<unsigned>(_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_andnot_si256(inrange[r], snz[r]))));
          const unsigned special = static_cast<unsigned>(_mm256_movemask_ps(_mm256_castsi256_ps(acc_nonfin[r])));
          const unsigned bad = static_cast<unsigned>(_mm256_movemask_ps(_mm256_castsi256_ps(badg[r])));
          if (__builtin_expect(redo | special | bad, 0))
            res[r] = mma_f_fix<NC, ACC16>(S[r], g[r], accb[r], res[r], redo, special, bad, &Al[(i0 + r) * lda + k0], &Bl[k0 * ldbl + jb * 8], ldbl, emin_l);
          accb[r] = res[r];
        }
      }
      VGPU_TC86_UNROLL
      for (unsigned r = 0; r < R; ++r) _mm256_storeu_si256(reinterpret_cast<__m256i*>(Dl + (i0 + r) * lddl + jb * 8), accb[r]);
    }
  };
  unsigned i = 0;
  for (; i + VGPU_TC86_ROWS_F <= M; i += VGPU_TC86_ROWS_F) rows(std::integral_constant<unsigned, VGPU_TC86_ROWS_F>{}, i);
  for (; i < M; ++i) rows(std::integral_constant<unsigned, 1>{}, i);
  return true;
}
#endif

// ---- the 512-bit form of mma_avx2_chunked_f ------------------------------------------------------------------------------------------
// The same arithmetic on sixteen lanes (mask registers in place of compare vectors), for the AVX-512 level. A unit of sixteen lanes is
//   * one row and 16 columns, when N is a multiple of 16: the A side is a broadcast of one number, the B side a vector of 16 columns;
//   * two rows and 8 columns, when N == 8 (mma.sync m16n8, wmma m32n8): lane 2j + r holds row r of the pair and column j. The A side is then a
//     broadcast of a 64-bit pair (row 2p and 2p+1 at the same k), the B side the 8 columns with each doubled; the accumulators are
//     interleaved on the way in and out of the registers.
// Any other N (and an odd M with N == 8) stays with the 256-bit form. The checks that send a call to the integer kernel, and the bits of every
// lane, are those of mma_avx2_chunked_f (tests/unit/test_tc_ampere_fast.cpp holds both to the scalar rule).
#define VGPU_TC86_T512 target("avx2,avx512f,avx512bw,avx512dq,avx512vl")
#ifndef VGPU_TC86_ROWS_Z
#define VGPU_TC86_ROWS_Z 4
#endif

// shift_term_avx2 on sixteen lanes. (A count past 31, and a negative count read as unsigned, give 0: the other direction's shift is the
// answer; for s == 0 both are |sig|.)
__attribute__((VGPU_TC86_T512)) inline __m512i shift_term_avx512(__m512i sig, __m512i s) {
  const __m512i zero = _mm512_setzero_si512();
  const __m512i mag = _mm512_abs_epi32(sig);
  const __m512i v = _mm512_or_si512(_mm512_sllv_epi32(mag, s), _mm512_srlv_epi32(mag, _mm512_sub_epi32(zero, s)));
  return _mm512_mask_sub_epi32(v, _mm512_movepi32_mask(sig), zero, v);
}

// sum_to_f32_avx2 on sixteen lanes; `snz` and `inrange` are lane masks.
__attribute__((VGPU_TC86_T512)) inline void sum_to_f32_avx512(__m512i S, __m512i g, __m512i& res, __mmask16& snz, __mmask16& inrange) {
  const __m512i zero = _mm512_setzero_si512(), cff = _mm512_set1_epi32(0xff);
  const __m512i mag = _mm512_abs_epi32(S);
  const __m512i hi = _mm512_castps_si512(_mm512_cvtepi32_ps(_mm512_and_si512(mag, _mm512_set1_epi32(~127))));
  const __m512i sh = _mm512_max_epi32(_mm512_sub_epi32(_mm512_srli_epi32(hi, 23), _mm512_set1_epi32(150)), zero);
  const __m512i magt = _mm512_sllv_epi32(_mm512_srlv_epi32(mag, sh), sh);
  const __m512i fb = _mm512_castps_si512(_mm512_cvtepi32_ps(magt));
  const __m512i newexp = _mm512_add_epi32(_mm512_and_si512(_mm512_srli_epi32(fb, 23), cff), g);
  snz = _mm512_cmpneq_epi32_mask(S, zero);
  inrange = _mm512_mask_cmplt_epi32_mask(_mm512_cmpgt_epi32_mask(newexp, zero), newexp, _mm512_set1_epi32(255));
  const __m512i r = _mm512_or_si512(_mm512_or_si512(_mm512_and_si512(fb, _mm512_set1_epi32(0x7fffff)), _mm512_slli_epi32(newexp, 23)),
                                    _mm512_and_si512(S, _mm512_set1_epi32(static_cast<int>(0x80000000u))));
  res = _mm512_maskz_mov_epi32(static_cast<__mmask16>(snz & inrange), r);
}

// sum_to_f16_avx2 on sixteen lanes.
__attribute__((VGPU_TC86_T512)) inline void sum_to_f16_avx512(__m512i S, __m512i g, __m512i& res, __mmask16& snz, __mmask16& inrange) {
  const __m512i zero = _mm512_setzero_si512(), one = _mm512_set1_epi32(1), sb = _mm512_set1_epi32(static_cast<int>(0x80000000u));
  const __m512i mag = _mm512_abs_epi32(S);
  const __m512i hi = _mm512_and_si512(mag, _mm512_set1_epi32(~127));
  const __m512i x = _mm512_mask_blend_epi32(_mm512_cmpgt_epi32_mask(hi, zero), mag, hi);
  const __m512i lg = _mm512_sub_epi32(_mm512_srli_epi32(_mm512_castps_si512(_mm512_cvtepi32_ps(x)), 23), _mm512_set1_epi32(127));
  const __m512i E = _mm512_add_epi32(lg, g);
  const __m512i Eq = _mm512_sub_epi32(_mm512_max_epi32(E, _mm512_set1_epi32(-14)), _mm512_set1_epi32(10));
  const __m512i r = _mm512_sub_epi32(Eq, g);
  const __m512i rp = _mm512_min_epi32(_mm512_max_epi32(r, zero), _mm512_set1_epi32(32));
  const __m512i l = _mm512_max_epi32(_mm512_sub_epi32(zero, r), zero);
  snz = _mm512_cmpneq_epi32_mask(S, zero);
  inrange = _mm512_cmpgt_epi32_mask(r, _mm512_set1_epi32(-32));
  const __m512i q = _mm512_srlv_epi32(mag, rp);
  const __m512i rem = _mm512_and_si512(mag, _mm512_sub_epi32(_mm512_sllv_epi32(one, rp), one));
  const __m512i half = _mm512_sllv_epi32(one, _mm512_sub_epi32(rp, one));
  const __mmask16 gt = _mm512_cmpgt_epu32_mask(rem, half);
  const __mmask16 tie = _mm512_mask_cmpneq_epi32_mask(_mm512_mask_cmpeq_epi32_mask(_mm512_test_epi32_mask(q, one), rem, half), half, zero);
  const __m512i R = _mm512_sllv_epi32(_mm512_mask_add_epi32(q, static_cast<__mmask16>(gt | tie), q, one), l);
  const __m512i fR = _mm512_castps_si512(_mm512_cvtepi32_ps(R));
  const __mmask16 Rnz = _mm512_cmpneq_epi32_mask(R, zero);
  const __m512i eres = _mm512_add_epi32(_mm512_sub_epi32(_mm512_srli_epi32(fR, 23), _mm512_set1_epi32(127)), Eq);
  const __mmask16 ovf = _mm512_cmpgt_epi32_mask(eres, _mm512_set1_epi32(15));
  __m512i val = _mm512_maskz_add_epi32(Rnz, fR, _mm512_slli_epi32(Eq, 23));
  val = _mm512_mask_blend_epi32(ovf, val, _mm512_set1_epi32(0x7f800000));
  res = _mm512_maskz_mov_epi32(snz, _mm512_or_si512(val, _mm512_and_si512(S, sb)));
}

// mma_f_fix on sixteen lanes (a unit as described above): the lanes the vector form does not finish.
template <unsigned NC, bool ACC16, bool PAIR>
__attribute__((VGPU_TC86_T512, noinline, cold)) __m512i mma_f_fix512(__m512i S, __m512i g, __m512i accb, __m512i res, unsigned redo, unsigned special, unsigned bad,
                                                                     const float* a, size_t lda, const float* b, size_t ldb, int emin) {
  alignas(64) int32_t Sv[16], gv[16];
  alignas(64) uint32_t accv[16], out[16];
  _mm512_store_si512(Sv, S);
  _mm512_store_si512(gv, g);
  _mm512_store_si512(accv, accb);
  _mm512_store_si512(out, res);
  for (unsigned l = 0; l < 16; ++l) {
    const float* const al = PAIR ? a + (l & 1) * lda : a;   // (the row of the lane, at the chunk's first k)
    const float* const bl = b + (PAIR ? l >> 1 : l);        // (the column of the lane, at the chunk's first k)
    if (special >> l & 1)
      out[l] = std::bit_cast<uint32_t>(chunk_special(al, 1, bl, ldb, NC, std::bit_cast<float>(accv[l])));
    else if (bad >> l & 1)
      out[l] = std::bit_cast<uint32_t>(chunk_finite_raw(al, 1, bl, ldb, NC, std::bit_cast<float>(accv[l]), ACC16, emin));
    else if (redo >> l & 1)
      out[l] = std::bit_cast<uint32_t>(ACC16 ? detail::to_f16_rn(Sv[l], gv[l]) : detail::to_f32_rz(Sv[l], gv[l]));
  }
  return _mm512_load_si512(out);
}

// mma_avx2_chunked_f on sixteen-lane units. Returns false, having written nothing, in the cases it does (and for a shape it does not take:
// the caller checks mma_avx512_f_shape first).
inline bool mma_avx512_f_shape(unsigned M, unsigned N) { return N % 16 == 0 || (N == 8 && M % 2 == 0); }

template <unsigned NC, bool ACC16>
__attribute__((VGPU_TC86_T512, aligned(64), noinline)) inline bool mma_avx512_chunked_f(unsigned M, unsigned N, unsigned K, int emin, const float* A, size_t lda, const float* B,
                                                                              size_t ldb, const float* C, size_t ldc, float* D, size_t ldd) {
  constexpr unsigned kMaxM = 32, kMaxN = 32, kMaxK = 16;
  alignas(64) float bf[kMaxK * kMaxN];
  alignas(64) int32_t bterm[kMaxK * kMaxN];
  alignas(64) float af[kMaxM * kMaxK + 8];
  alignas(64) int32_t aterm[kMaxM * kMaxK + 8];
  alignas(64) int64_t apf[kMaxM / 2 * kMaxK + 4];   // N == 8: the pairs of rows (2p, 2p+1) at each k, as one 64-bit element of floats
  alignas(64) int64_t apt[kMaxM / 2 * kMaxK + 4];   //  ... and of terms
  alignas(64) float bf2[kMaxK * 16];                //  ... and the columns, each doubled
  alignas(64) int32_t bt2[kMaxK * 16];
  unsigned any_nonfin = 0;
  const __m256i zero = _mm256_setzero_si256(), c255 = _mm256_set1_epi32(255), cff = _mm256_set1_epi32(0xff), c7fffff = _mm256_set1_epi32(0x7fffff),
                one = _mm256_set1_epi32(1), c127 = _mm256_set1_epi32(127), cnoterm = _mm256_set1_epi32(kNoTerm), cemin = _mm256_set1_epi32(emin),
                cmask = _mm256_set1_epi32(static_cast<int>(0xffffe000u));
  // (the decoding and the range checks are those of mma_avx2_chunked_f)
  __m256i amx = zero, amn = _mm256_set1_epi32(-1), bmx = zero, bmn = _mm256_set1_epi32(-1), subn = zero;
  const auto decode8 = [&](const float* p, int bias, float* fout, int32_t* tout, __m256i& mx, __m256i& mn) __attribute__((target("avx2"))) {
    const __m256i bits = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
    const __m256i masked = _mm256_and_si256(bits, cmask);
    const __m256i e = _mm256_and_si256(_mm256_srli_epi32(bits, 23), cff);
    const __m256i nonfin = _mm256_cmpeq_epi32(e, c255), ez = _mm256_cmpeq_epi32(e, zero);
    subn = _mm256_or_si256(subn, _mm256_and_si256(_mm256_and_si256(ez, masked), c7fffff));
    const __m256i term = _mm256_add_epi32(_mm256_max_epi32(_mm256_sub_epi32(e, c127), cemin), _mm256_set1_epi32(bias));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(tout), _mm256_blendv_epi8(term, cnoterm, _mm256_or_si256(nonfin, ez)));
    _mm256_storeu_ps(fout, _mm256_castsi256_ps(masked));
    mx = _mm256_max_epi32(mx, e);
    mn = _mm256_min_epu32(mn, _mm256_sub_epi32(e, one));
    return static_cast<unsigned>(_mm256_movemask_ps(_mm256_castsi256_ps(nonfin)));
  };
  int sm_x = 0, sm_n = 255;
  bool tail_sub = false;
  for (unsigned i = 0; i < M; ++i) {
    unsigned k = 0;
    for (; k + 8 <= K; k += 8) any_nonfin |= decode8(A + i * lda + k, 1, &af[i * K + k], &aterm[i * K + k], amx, amn);
    for (; k < K; ++k) {
      const uint32_t b = std::bit_cast<uint32_t>(A[i * lda + k]);
      const uint32_t e = (b >> 23) & 0xffu, mb = b & 0xffffe000u;
      af[i * K + k] = std::bit_cast<float>(mb);
      if (e == 255) any_nonfin = 1;
      const bool dead = e == 0 || e == 255;
      if (e == 0 && (mb & 0x7fffffu)) tail_sub = true;
      aterm[i * K + k] = dead ? kNoTerm : std::max(static_cast<int32_t>(e) - 127, emin) + 1;
      if (!dead) {
        sm_x = std::max(sm_x, static_cast<int>(e));
        sm_n = std::min(sm_n, static_cast<int>(e));
      }
    }
  }
  for (unsigned jb = 0; jb < N / 8; ++jb)
    for (unsigned k0 = 0; k0 < K; k0 += NC) {
      unsigned nf = 0;
      for (unsigned k = k0; k < k0 + NC; ++k) nf |= decode8(B + k * ldb + jb * 8, 0, &bf[k * N + jb * 8], &bterm[k * N + jb * 8], bmx, bmn);
      any_nonfin |= nf;
    }
  const auto hmax = [](__m256i v) __attribute__((target("avx2"))) {
    alignas(32) int32_t t[8];
    _mm256_store_si256(reinterpret_cast<__m256i*>(t), v);
    return *std::max_element(t, t + 8);
  };
  if (any_nonfin || tail_sub || !_mm256_testz_si256(subn, subn)) return false;
  const auto hminu = [](__m256i v) __attribute__((target("avx2"))) {
    alignas(32) uint32_t t[8];
    _mm256_store_si256(reinterpret_cast<__m256i*>(t), v);
    return *std::min_element(t, t + 8);
  };
  const auto field_min = [](uint32_t m) { return m >= 254 ? 255 : static_cast<int>(m) + 1; };
  const int a_max = std::max(hmax(amx), sm_x), a_min = std::min(field_min(hminu(amn)), sm_n), b_max = hmax(bmx), b_min = field_min(hminu(bmn));
  if (a_min <= 254 && b_min <= 254 && a_min + b_min < 128) return false;
  if (a_max + b_max > 379) return false;

  const bool pair = N == 8;
  if (pair) {
    // the A side as 64-bit elements (row 2p, row 2p+1), the B side with each column doubled
    for (unsigned p = 0; p < M / 2; ++p) {
      unsigned k = 0;
      for (; k + 8 <= K; k += 8)
        for (int w = 0; w < 2; ++w) {
          const int32_t* const src = w ? aterm : reinterpret_cast<const int32_t*>(af);
          int64_t* const dst = w ? apt : apf;
          const __m256i x = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 2 * p * K + k));
          const __m256i y = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + (2 * p + 1) * K + k));
          const __m256i lo = _mm256_unpacklo_epi32(x, y), hi = _mm256_unpackhi_epi32(x, y);
          _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + p * K + k), _mm256_permute2x128_si256(lo, hi, 0x20));
          _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + p * K + k + 4), _mm256_permute2x128_si256(lo, hi, 0x31));
        }
      for (; k < K; ++k) {
        apt[p * K + k] = static_cast<int64_t>(static_cast<uint64_t>(static_cast<uint32_t>(aterm[2 * p * K + k])) | static_cast<uint64_t>(static_cast<uint32_t>(aterm[(2 * p + 1) * K + k])) << 32);
        apf[p * K + k] = static_cast<int64_t>(static_cast<uint64_t>(std::bit_cast<uint32_t>(af[2 * p * K + k])) | static_cast<uint64_t>(std::bit_cast<uint32_t>(af[(2 * p + 1) * K + k])) << 32);
      }
    }
    const __m512i dup = _mm512_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7);
    for (unsigned k = 0; k < K; ++k) {
      _mm512_store_si512(&bf2[k * 16], _mm512_permutexvar_epi32(dup, _mm512_castsi256_si512(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(&bf[k * 8])))));
      _mm512_store_si512(&bt2[k * 16], _mm512_permutexvar_epi32(dup, _mm512_castsi256_si512(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(&bterm[k * 8])))));
    }
  }

  const auto rows = [&]<bool PAIR, unsigned R>(std::bool_constant<PAIR>, std::integral_constant<unsigned, R>, unsigned u0) __attribute__((VGPU_TC86_T512)) {
    const unsigned Nl = N, Kl = K, nb = PAIR ? 1 : N / 16;
    const size_t ldcl = ldc, lddl = ldd, ldbl = ldb, ldal = lda;
    const float* const Cl = C;
    float* const Dl = D;
    const float* const Al = A;
    const float* const Bl = B;
    const int32_t* const atl = aterm;
    const float* const afl = af;
    const int64_t* const aptl = apt;
    const int64_t* const apfl = apf;
    const int32_t* const btl = PAIR ? bt2 : bterm;
    const float* const bfl = PAIR ? bf2 : bf;
    const size_t bstride = PAIR ? 16 : Nl;
    const int emin_l = emin;
    const __m512i cff = _mm512_set1_epi32(0xff), c7fffff = _mm512_set1_epi32(0x7fffff), c255 = _mm512_set1_epi32(255), c150 = _mm512_set1_epi32(150),
                  c127 = _mm512_set1_epi32(127), c25 = _mm512_set1_epi32(25), one = _mm512_set1_epi32(1), zero = _mm512_setzero_si512(),
                  idx_in = _mm512_setr_epi32(0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15),     // lane 2j + r <- r * 8 + j
                  idx_out = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15);    // r * 8 + j <- lane 2j + r
    // (the A side of unit u at k: a broadcast of one number, or of a pair)
    const auto bcast_t = [&](unsigned u, unsigned k) __attribute__((VGPU_TC86_T512)) {
      if constexpr (PAIR) return _mm512_set1_epi64(aptl[size_t{u} * Kl + k]);
      else return _mm512_set1_epi32(atl[size_t{u} * Kl + k]);
    };
    const auto bcast_f = [&](unsigned u, unsigned k) __attribute__((VGPU_TC86_T512)) {
      if constexpr (PAIR) return _mm512_castsi512_ps(_mm512_set1_epi64(apfl[size_t{u} * Kl + k]));
      else return _mm512_set1_ps(afl[size_t{u} * Kl + k]);
    };
    for (unsigned jb = 0; jb < nb; ++jb) {
      __m512i accb[R];
      VGPU_TC86_UNROLL
      for (unsigned r = 0; r < R; ++r) {
        if constexpr (PAIR) {
          const size_t i = 2 * size_t{u0 + r};
          const __m512i v = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(Cl + i * ldcl))),
                                               _mm256_loadu_si256(reinterpret_cast<const __m256i*>(Cl + (i + 1) * ldcl)), 1);
          accb[r] = _mm512_permutexvar_epi32(idx_in, v);
        } else
          accb[r] = _mm512_loadu_si512(Cl + (u0 + r) * ldcl + jb * 16);
      }
      for (unsigned k0 = 0; k0 < Kl; k0 += NC) {
        __m512i top[R], g[R], S[R];
        __m512 scale[R];
        __mmask16 acc_nonfin[R], badg[R];
        const int32_t* const bt = btl + size_t{k0} * bstride + (PAIR ? 0 : jb * 16);
        const float* const bp = bfl + size_t{k0} * bstride + (PAIR ? 0 : jb * 16);
        VGPU_TC86_UNROLL
        for (unsigned r = 0; r < R; ++r) {
          top[r] = _mm512_set1_epi32(kNoTerm);
          VGPU_TC86_UNROLL
          for (unsigned t = 0; t < NC; ++t)
            top[r] = _mm512_max_epi32(top[r], _mm512_add_epi32(bcast_t(u0 + r, k0 + t), _mm512_load_si512(bt + size_t{t} * bstride)));
        }
        VGPU_TC86_UNROLL
        for (unsigned r = 0; r < R; ++r) {
          const __m512i ae = _mm512_and_si512(_mm512_srli_epi32(accb[r], 23), cff), am = _mm512_and_si512(accb[r], c7fffff);
          const __mmask16 aenz = _mm512_cmpneq_epi32_mask(ae, zero);
          acc_nonfin[r] = _mm512_cmpeq_epi32_mask(ae, c255);
          __m512i asig = _mm512_mask_or_epi32(am, aenz, am, _mm512_set1_epi32(0x800000));
          const __m512i alsb = _mm512_mask_blend_epi32(aenz, _mm512_set1_epi32(-149), _mm512_sub_epi32(ae, c150));
          const __m512i lead = _mm512_sub_epi32(_mm512_srli_epi32(_mm512_castps_si512(_mm512_cvtepi32_ps(asig)), 23), c127);
          top[r] = _mm512_max_epi32(top[r], _mm512_add_epi32(_mm512_add_epi32(lead, alsb), one));
          asig = _mm512_mask_sub_epi32(asig, _mm512_movepi32_mask(accb[r]), zero, asig);
          g[r] = _mm512_sub_epi32(top[r], c25);
          S[r] = shift_term_avx512(asig, _mm512_sub_epi32(alsb, g[r]));
          const __mmask16 live = _mm512_cmpgt_epi32_mask(top[r], _mm512_set1_epi32(-200));
          const __m512i sexp = _mm512_maskz_sub_epi32(live, c127, g[r]);
          badg[r] = static_cast<__mmask16>(_mm512_mask_cmplt_epi32_mask(live, sexp, one) | _mm512_mask_cmpgt_epi32_mask(live, sexp, _mm512_set1_epi32(254)));
          scale[r] = _mm512_castsi512_ps(_mm512_slli_epi32(sexp, 23));
        }
        VGPU_TC86_UNROLL
        for (unsigned r = 0; r < R; ++r) {
          VGPU_TC86_UNROLL
          for (unsigned t = 0; t < NC; ++t) {
            const __m512 x = _mm512_mul_ps(bcast_f(u0 + r, k0 + t), _mm512_load_ps(bp + size_t{t} * bstride));
            S[r] = _mm512_add_epi32(S[r], _mm512_cvttps_epi32(_mm512_mul_ps(x, scale[r])));
          }
        }
        VGPU_TC86_UNROLL
        for (unsigned r = 0; r < R; ++r) {
          __m512i res;
          __mmask16 snz, inrange;
          if constexpr (ACC16)
            sum_to_f16_avx512(S[r], g[r], res, snz, inrange);
          else
            sum_to_f32_avx512(S[r], g[r], res, snz, inrange);
          const unsigned redo = static_cast<unsigned>(snz & ~inrange & 0xffffu), special = acc_nonfin[r], bad = badg[r];
          if (__builtin_expect(redo | special | bad, 0)) {
            const size_t row = PAIR ? 2 * size_t{u0 + r} : size_t{u0 + r};
            res = mma_f_fix512<NC, ACC16, PAIR>(S[r], g[r], accb[r], res, redo, special, bad, &Al[row * ldal + k0], ldal, &Bl[size_t{k0} * ldbl + (PAIR ? 0 : jb * 16)], ldbl, emin_l);
          }
          accb[r] = res;
        }
      }
      VGPU_TC86_UNROLL
      for (unsigned r = 0; r < R; ++r) {
        if constexpr (PAIR) {
          const size_t i = 2 * size_t{u0 + r};
          const __m512i v = _mm512_permutexvar_epi32(idx_out, accb[r]);
          _mm256_storeu_si256(reinterpret_cast<__m256i*>(Dl + i * lddl), _mm512_castsi512_si256(v));
          _mm256_storeu_si256(reinterpret_cast<__m256i*>(Dl + (i + 1) * lddl), _mm512_extracti64x4_epi64(v, 1));
        } else
          _mm512_storeu_si512(Dl + (u0 + r) * lddl + jb * 16, accb[r]);
      }
    }
  };
  const unsigned units = pair ? M / 2 : M;
  unsigned u = 0;
  if (pair) {
    for (; u + VGPU_TC86_ROWS_Z <= units; u += VGPU_TC86_ROWS_Z) rows(std::true_type{}, std::integral_constant<unsigned, VGPU_TC86_ROWS_Z>{}, u);
    for (; u < units; ++u) rows(std::true_type{}, std::integral_constant<unsigned, 1>{}, u);
  } else {
    for (; u + VGPU_TC86_ROWS_Z <= units; u += VGPU_TC86_ROWS_Z) rows(std::false_type{}, std::integral_constant<unsigned, VGPU_TC86_ROWS_Z>{}, u);
    for (; u < units; ++u) rows(std::false_type{}, std::integral_constant<unsigned, 1>{}, u);
  }
  return true;
}

// The vector form for an N that is a multiple of 8 (an f32 accumulator, or with acc_f16 an f16 one).
template <bool ACC16>
__attribute__((target("avx2"))) inline void mma_avx2_acc(unsigned M, unsigned N, unsigned K, unsigned chunk, int emin, const float* A, size_t lda, const float* B,
                                                         size_t ldb, const float* C, size_t ldc, float* D, size_t ldd) {
  if (chunk == 8 && K % 8 == 0) {
    if (!(simd::kernels().level == simd::Level::Avx512 && mma_avx512_f_shape(M, N) ? mma_avx512_chunked_f<8, ACC16>(M, N, K, emin, A, lda, B, ldb, C, ldc, D, ldd)
                                                                                  : mma_avx2_chunked_f<8, ACC16>(M, N, K, emin, A, lda, B, ldb, C, ldc, D, ldd)))
      mma_avx2_chunked<8, ACC16>(M, N, K, emin, A, lda, B, ldb, C, ldc, D, ldd);
  } else if (chunk == 4 && K % 4 == 0) {
    if (!(simd::kernels().level == simd::Level::Avx512 && mma_avx512_f_shape(M, N) ? mma_avx512_chunked_f<4, ACC16>(M, N, K, emin, A, lda, B, ldb, C, ldc, D, ldd)
                                                                                  : mma_avx2_chunked_f<4, ACC16>(M, N, K, emin, A, lda, B, ldb, C, ldc, D, ldd)))
      mma_avx2_chunked<4, ACC16>(M, N, K, emin, A, lda, B, ldb, C, ldc, D, ldd);
  } else
    mma_avx2_generic(M, N, K, chunk, emin, A, lda, B, ldb, C, ldc, D, ldd);   // (f32 accumulator only: mma() sends no other)
}
__attribute__((target("avx2"))) inline void mma_avx2(unsigned M, unsigned N, unsigned K, unsigned chunk, int emin, const float* A, size_t lda, const float* B,
                                                    size_t ldb, const float* C, size_t ldc, float* D, size_t ldd) {
  mma_avx2_acc<false>(M, N, K, chunk, emin, A, lda, B, ldb, C, ldc, D, ldd);
}
#endif

// D = C + A x B, by the rule above, with the vector code where there is a CPU level for it.
inline void mma(unsigned M, unsigned N, unsigned K, unsigned chunk, int emin, bool acc_f16, const float* A, size_t lda, const float* B,
                size_t ldb, const float* C, size_t ldc, float* D, size_t ldd) {
#ifdef VGPU_TC86_AVX2
  if (N % 8 == 0 && simd::enabled() && simd::kernels().level != simd::Level::Base) {
    if (!acc_f16) {
      mma_avx2(M, N, K, chunk, emin, A, lda, B, ldb, C, ldc, D, ldd);
      return;
    }
    if (K % chunk == 0 && (chunk == 8 || chunk == 4)) {   // (the f16 accumulator has the chunked form only)
      mma_avx2_acc<true>(M, N, K, chunk, emin, A, lda, B, ldb, C, ldc, D, ldd);
      return;
    }
  }
#endif
  mma_scalar(M, N, K, chunk, emin, acc_f16, A, lda, B, ldb, C, ldc, D, ldd);
}

}  // namespace vgpu::exec::tc86
