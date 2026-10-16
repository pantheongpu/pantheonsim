#pragma once
// AVX2 kernels for the integer inner loops of the video codecs (nvenc_hevc_enc.cpp and friends), chosen at run time.
//
// Everything here is integer arithmetic, so "the same bits as the scalar loop" is a matter of getting the same values in the
// same lanes: the 2-D transforms below run the very butterflies of the scalar code (nvenc_hevc_enc.cpp's fdct1d / idct1d) on
// eight lines at once, and the interpolation filters sum the same taps (in 16 bits where the sum provably fits, in 32 where it
// may not). tests/unit/test_video_simd.cpp holds the scalar loops as the reference and compares every kernel over random and
// extreme inputs.
//
// The build is baseline x86-64: the kernels are compiled under a target pragma and called only after a check of the CPU. Nothing
// in the pragma region may call a std:: template (it would share its mangled name with the baseline instantiation).
#include <cstddef>
#include <cstdint>

#if defined(__x86_64__) && defined(__GNUC__)
#define VGPU_VIDEO_AVX2 1
#include <immintrin.h>
#endif

namespace vgpu::videosimd {

inline bool avx2_available() {
#ifdef VGPU_VIDEO_AVX2
  static const bool ok = __builtin_cpu_supports("avx2");
  return ok;
#else
  return false;
#endif
}

#ifdef VGPU_VIDEO_AVX2
#pragma GCC push_options
#pragma GCC target("avx2")
namespace avx2 {

typedef __m256i V;

inline V mul(V a, int c) { return _mm256_mullo_epi32(a, _mm256_set1_epi32(c)); }

// Eight lines at a time: x[n] holds element n of eight lines, one per lane.
// y[k] = sum_n T[k][n] x[n], the N-point forward core transform by even/odd decomposition (nvenc_hevc_enc.cpp's fdct1d).
template <int N>
inline void fdct_v(const V* x, V* y, const int8_t (*M)[32]) {
  if constexpr (N == 4) {
    const V s0 = _mm256_add_epi32(x[0], x[3]), s1 = _mm256_add_epi32(x[1], x[2]);
    const V d0 = _mm256_sub_epi32(x[0], x[3]), d1 = _mm256_sub_epi32(x[1], x[2]);
    y[0] = mul(_mm256_add_epi32(s0, s1), 64);
    y[2] = mul(_mm256_sub_epi32(s0, s1), 64);
    y[1] = _mm256_add_epi32(mul(d0, 83), mul(d1, 36));
    y[3] = _mm256_sub_epi32(mul(d0, 36), mul(d1, 83));
  } else {
    constexpr int hn = N / 2, step = 32 / N;
    V E[hn], O[hn], YE[hn];
    for (int n = 0; n < hn; ++n) {
      E[n] = _mm256_add_epi32(x[n], x[N - 1 - n]);
      O[n] = _mm256_sub_epi32(x[n], x[N - 1 - n]);
    }
    fdct_v<hn>(E, YE, M);
    for (int k = 0; k < hn; ++k) y[2 * k] = YE[k];
    for (int k = 1; k < N; k += 2) {
      V s = _mm256_setzero_si256();
      for (int n = 0; n < hn; ++n) s = _mm256_add_epi32(s, mul(O[n], M[k * step][n]));
      y[k] = s;
    }
  }
}

// x[n] = sum_k T[k][n] y[k]: the inverse (nvenc_hevc_enc.cpp's idct1d).
template <int N>
inline void idct_v(const V* y, V* x, const int8_t (*M)[32]) {
  if constexpr (N == 4) {
    const V o0 = _mm256_add_epi32(mul(y[1], 83), mul(y[3], 36)), o1 = _mm256_sub_epi32(mul(y[1], 36), mul(y[3], 83));
    const V e0 = mul(_mm256_add_epi32(y[0], y[2]), 64), e1 = mul(_mm256_sub_epi32(y[0], y[2]), 64);
    x[0] = _mm256_add_epi32(e0, o0);
    x[1] = _mm256_add_epi32(e1, o1);
    x[2] = _mm256_sub_epi32(e1, o1);
    x[3] = _mm256_sub_epi32(e0, o0);
  } else {
    constexpr int hn = N / 2, step = 32 / N;
    V ye[hn], E[hn];
    for (int k = 0; k < hn; ++k) ye[k] = y[2 * k];
    idct_v<hn>(ye, E, M);
    for (int n = 0; n < hn; ++n) {
      V o = _mm256_setzero_si256();
      for (int k = 1; k < N; k += 2) o = _mm256_add_epi32(o, mul(y[k], M[k * step][n]));
      x[n] = _mm256_add_epi32(E[n], o);
      x[N - 1 - n] = _mm256_sub_epi32(E[n], o);
    }
  }
}

inline void transpose8(V r[8]) {
  const V t0 = _mm256_unpacklo_epi32(r[0], r[1]), t1 = _mm256_unpackhi_epi32(r[0], r[1]);
  const V t2 = _mm256_unpacklo_epi32(r[2], r[3]), t3 = _mm256_unpackhi_epi32(r[2], r[3]);
  const V t4 = _mm256_unpacklo_epi32(r[4], r[5]), t5 = _mm256_unpackhi_epi32(r[4], r[5]);
  const V t6 = _mm256_unpacklo_epi32(r[6], r[7]), t7 = _mm256_unpackhi_epi32(r[6], r[7]);
  const V u0 = _mm256_unpacklo_epi64(t0, t2), u1 = _mm256_unpackhi_epi64(t0, t2);
  const V u2 = _mm256_unpacklo_epi64(t1, t3), u3 = _mm256_unpackhi_epi64(t1, t3);
  const V u4 = _mm256_unpacklo_epi64(t4, t6), u5 = _mm256_unpackhi_epi64(t4, t6);
  const V u6 = _mm256_unpacklo_epi64(t5, t7), u7 = _mm256_unpackhi_epi64(t5, t7);
  r[0] = _mm256_permute2x128_si256(u0, u4, 0x20);
  r[1] = _mm256_permute2x128_si256(u1, u5, 0x20);
  r[2] = _mm256_permute2x128_si256(u2, u6, 0x20);
  r[3] = _mm256_permute2x128_si256(u3, u7, 0x20);
  r[4] = _mm256_permute2x128_si256(u0, u4, 0x31);
  r[5] = _mm256_permute2x128_si256(u1, u5, 0x31);
  r[6] = _mm256_permute2x128_si256(u2, u6, 0x31);
  r[7] = _mm256_permute2x128_si256(u3, u7, 0x31);
}

// x[n] = lane l: src[(y0 + l) * N + n], n = 0 .. N-1 (eight rows of a raster block, one per lane).
template <int N>
inline void load_rows_t(const int32_t* src, int y0, V* x) {
  for (int nb = 0; nb < N; nb += 8) {
    V r[8];
    for (int l = 0; l < 8; ++l) r[l] = _mm256_loadu_si256(reinterpret_cast<const V*>(src + static_cast<size_t>(y0 + l) * N + nb));
    transpose8(r);
    for (int i = 0; i < 8; ++i) x[nb + i] = r[i];
  }
}
// The reverse: dst[(y0 + l) * N + n] = lane l of x[n].
template <int N>
inline void store_rows_t(const V* x, int32_t* dst, int y0) {
  for (int nb = 0; nb < N; nb += 8) {
    V r[8];
    for (int i = 0; i < 8; ++i) r[i] = x[nb + i];
    transpose8(r);
    for (int l = 0; l < 8; ++l) _mm256_storeu_si256(reinterpret_cast<V*>(dst + static_cast<size_t>(y0 + l) * N + nb), r[l]);
  }
}

inline V sra(V v, int s) { return _mm256_sra_epi32(v, _mm_cvtsi32_si128(s)); }

// forward_transform of nvenc_hevc_enc.cpp for the DCT (not the 4x4 DST) at N = 8, 16, 32: rows (shift1 = log2 - 1), then
// columns (shift2 = log2 + 6), each result rounded by adding half the divisor before the shift.
template <int N>
inline void forward_dct(const int32_t* res, int32_t* coef, const int8_t (*M)[32]) {
  constexpr int log2 = N == 8 ? 3 : (N == 16 ? 4 : 5);
  constexpr int shift1 = log2 - 1, shift2 = log2 + 6;
  alignas(32) int32_t tmp[N * N];
  const V r1 = _mm256_set1_epi32(1 << (shift1 - 1)), r2 = _mm256_set1_epi32(1 << (shift2 - 1));
  for (int y0 = 0; y0 < N; y0 += 8) {   // eight rows at a time, one per lane
    V x[N], out[N];
    load_rows_t<N>(res, y0, x);
    fdct_v<N>(x, out, M);
    for (int k = 0; k < N; ++k) out[k] = sra(_mm256_add_epi32(out[k], r1), shift1);
    store_rows_t<N>(out, tmp, y0);
  }
  for (int x0 = 0; x0 < N; x0 += 8) {   // eight columns at a time
    V c[N], out[N];
    for (int y = 0; y < N; ++y) c[y] = _mm256_load_si256(reinterpret_cast<const V*>(tmp + static_cast<size_t>(y) * N + x0));
    fdct_v<N>(c, out, M);
    for (int k = 0; k < N; ++k)
      _mm256_storeu_si256(reinterpret_cast<V*>(coef + static_cast<size_t>(k) * N + x0), sra(_mm256_add_epi32(out[k], r2), shift2));
  }
}

// inverse_transform of nvenc_hevc_enc.cpp for the DCT at N = 8, 16, 32: columns first (rounded by 64, shifted by 7, clipped to 16 bits), then rows
// (rounded by half of 1 << shift2, shifted by shift2: 12 for 8-bit video, 20 - bitDepth in general). Inputs are 16-bit values, so the 32-bit sums
// cannot overflow and equal a 64-bit sum.
template <int N>
inline void inverse_dct(const int32_t* d, int32_t* res, const int8_t (*M)[32], int shift2) {
  alignas(32) int32_t g[N * N];
  const V r1 = _mm256_set1_epi32(64), r2 = _mm256_set1_epi32(1 << (shift2 - 1));
  const V lo = _mm256_set1_epi32(-32768), hi = _mm256_set1_epi32(32767);
  for (int x0 = 0; x0 < N; x0 += 8) {   // eight columns at a time
    V c[N], out[N];
    V any = _mm256_setzero_si256();
    for (int y = 0; y < N; ++y) {
      c[y] = _mm256_loadu_si256(reinterpret_cast<const V*>(d + static_cast<size_t>(y) * N + x0));
      any = _mm256_or_si256(any, c[y]);
    }
    if (_mm256_testz_si256(any, any)) {   // eight columns of zeros transform to zeros: (0 + 64) >> 7 = 0
      for (int y = 0; y < N; ++y) _mm256_store_si256(reinterpret_cast<V*>(g + static_cast<size_t>(y) * N + x0), _mm256_setzero_si256());
      continue;
    }
    idct_v<N>(c, out, M);
    for (int y = 0; y < N; ++y) {
      V v = sra(_mm256_add_epi32(out[y], r1), 7);
      v = _mm256_min_epi32(_mm256_max_epi32(v, lo), hi);
      _mm256_store_si256(reinterpret_cast<V*>(g + static_cast<size_t>(y) * N + x0), v);
    }
  }
  for (int y0 = 0; y0 < N; y0 += 8) {   // eight rows at a time, one per lane
    V x[N], out[N];
    load_rows_t<N>(g, y0, x);
    idct_v<N>(x, out, M);
    for (int n = 0; n < N; ++n) out[n] = sra(_mm256_add_epi32(out[n], r2), shift2);
    store_rows_t<N>(out, res, y0);
  }
}


// ---- interpolation filters of inter prediction (nvenc_hevc_enc.cpp's mc_luma / mc_chroma) --------------------------------------
//
// out[j * bw + i] = sum over taps t of c[t] * sample, the samples one step apart along x (h), along y (v), or in two passes (hv: the
// rows filtered along x first, kept as 16-bit values -- they are at most 255 * 112 in size -- then filtered along y in 32 bits,
// shifted down by 6 and cut to 16 bits). In 16 bits a one-pass sum of u8 samples against these coefficients cannot overflow
// (the sum of the absolute values of a filter's taps times 255 is under 32768), so each lane holds exactly the value the 32-bit sum
// had, whatever the order of the additions.

// 16 / 8 / 4 outputs at a time along a row. s: the first tap of output 0 of this row; xs: the distance between taps.
template <int TAPS>
inline void fir_row(const uint8_t* s, int xs, int16_t* out, int bw, const __m256i* cv) {
  int i = 0;
  for (; i + 16 <= bw; i += 16) {
    __m256i acc = _mm256_setzero_si256();
    for (int t = 0; t < TAPS; ++t) {
      const __m256i x = _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(s + i + t * xs)));
      acc = _mm256_add_epi16(acc, _mm256_mullo_epi16(x, cv[t]));
    }
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + i), acc);
  }
  for (; i + 8 <= bw; i += 8) {
    __m128i acc = _mm_setzero_si128();
    for (int t = 0; t < TAPS; ++t) {
      const __m128i x = _mm_cvtepu8_epi16(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(s + i + t * xs)));
      acc = _mm_add_epi16(acc, _mm_mullo_epi16(x, _mm256_castsi256_si128(cv[t])));
    }
    _mm_storeu_si128(reinterpret_cast<__m128i*>(out + i), acc);
  }
  for (; i < bw; i += 4) {   // bw is a multiple of 4
    __m128i acc = _mm_setzero_si128();
    for (int t = 0; t < TAPS; ++t) {
      int v;
      __builtin_memcpy(&v, s + i + t * xs, 4);
      const __m128i x = _mm_cvtepu8_epi16(_mm_cvtsi32_si128(v));
      acc = _mm_add_epi16(acc, _mm_mullo_epi16(x, _mm256_castsi256_si128(cv[t])));
    }
    _mm_storel_epi64(reinterpret_cast<__m128i*>(out + i), acc);
  }
}

template <int TAPS>
inline void coef_vectors(const int* c, __m256i* cv) {
  for (int t = 0; t < TAPS; ++t) cv[t] = _mm256_set1_epi16(static_cast<short>(c[t]));
}

// The filtered row along x for rows [0, rows): src is the first tap of row 0's output 0, stride st; out rows of bw 16-bit values.
template <int TAPS>
inline void fir_h(const uint8_t* src, int st, int bw, int rows, const int* cx, int16_t* out) {
  __m256i cv[TAPS];
  coef_vectors<TAPS>(cx, cv);
  for (int j = 0; j < rows; ++j) fir_row<TAPS>(src + static_cast<size_t>(j) * st, 1, out + static_cast<size_t>(j) * bw, bw, cv);
}
// Along y: src is the first tap row of output row 0.
template <int TAPS>
inline void fir_v(const uint8_t* src, int st, int bw, int bh, const int* cy, int16_t* out) {
  __m256i cv[TAPS];
  coef_vectors<TAPS>(cy, cv);
  for (int j = 0; j < bh; ++j) fir_row<TAPS>(src + static_cast<size_t>(j) * st, st, out + static_cast<size_t>(j) * bw, bw, cv);
}
// Both: tmp holds (bh + TAPS - 1) rows of bw values, then the vertical pass over them.
template <int TAPS>
inline void fir_hv(const uint8_t* src, int st, int bw, int bh, const int* cx, const int* cy, int16_t* out) {
  alignas(32) int16_t tmp[(32 + 7) * 32 + 16];
  fir_h<TAPS>(src, st, bw, bh + TAPS - 1, cx, tmp);
  __m256i cv[TAPS];
  for (int t = 0; t < TAPS; ++t) cv[t] = _mm256_set1_epi32(cy[t]);
  const __m256i mask = _mm256_set1_epi32(0xFFFF);
  for (int j = 0; j < bh; ++j)
    for (int i = 0; i < bw; i += 8) {
      __m256i acc = _mm256_setzero_si256();
      for (int t = 0; t < TAPS; ++t) {
        const __m256i x = _mm256_cvtepi16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(tmp + static_cast<size_t>(j + t) * bw + i)));
        acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(x, cv[t]));
      }
      acc = _mm256_and_si256(_mm256_srai_epi32(acc, 6), mask);   // (>> 6, then the low 16 bits)
      const __m128i packed = _mm_packus_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
      int16_t* o = out + static_cast<size_t>(j) * bw + i;
      if (i + 8 <= bw) _mm_storeu_si128(reinterpret_cast<__m128i*>(o), packed);
      else _mm_storel_epi64(reinterpret_cast<__m128i*>(o), packed);   // the last four of a block four wide
    }
}


// ---- sums of absolute differences (nvenc_hevc_enc.cpp's sad_pu) --------------------------------------------------------------------

// sum of |a - b| over bw x bh bytes (bw a multiple of 4).
inline int sad_u8(const uint8_t* a, int as, const uint8_t* b, int bs, int bw, int bh) {
  __m256i acc = _mm256_setzero_si256();
  __m128i acc128 = _mm_setzero_si128();
  int tail = 0;
  for (int j = 0; j < bh; ++j) {
    const uint8_t* pa = a + static_cast<size_t>(j) * as;
    const uint8_t* pb = b + static_cast<size_t>(j) * bs;
    int i = 0;
    for (; i + 32 <= bw; i += 32)
      acc = _mm256_add_epi64(acc, _mm256_sad_epu8(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(pa + i)), _mm256_loadu_si256(reinterpret_cast<const __m256i*>(pb + i))));
    for (; i + 16 <= bw; i += 16)
      acc128 = _mm_add_epi64(acc128, _mm_sad_epu8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(pa + i)), _mm_loadu_si128(reinterpret_cast<const __m128i*>(pb + i))));
    for (; i + 8 <= bw; i += 8)
      acc128 = _mm_add_epi64(acc128, _mm_sad_epu8(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(pa + i)), _mm_loadl_epi64(reinterpret_cast<const __m128i*>(pb + i))));
    for (; i < bw; ++i) {
      const int d = static_cast<int>(pa[i]) - static_cast<int>(pb[i]);
      tail += d < 0 ? -d : d;
    }
  }
  acc128 = _mm_add_epi64(acc128, _mm_add_epi64(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1)));
  return tail + static_cast<int>(_mm_cvtsi128_si64(acc128) + _mm_extract_epi64(acc128, 1));
}

// sum |src - clip8((p + 32) >> 6)| over bw x bh 14-bit prediction samples p (row stride bw) against the source bytes (stride ss).
inline int sad_uni(const int16_t* p, int bw, int bh, const uint8_t* src, int ss) {
  const __m256i k32 = _mm256_set1_epi16(32), zero = _mm256_setzero_si256(), c255 = _mm256_set1_epi16(255), ones = _mm256_set1_epi16(1);
  __m256i acc = _mm256_setzero_si256();
  int tail = 0;
  for (int j = 0; j < bh; ++j) {
    const int16_t* pj = p + static_cast<size_t>(j) * bw;
    const uint8_t* sj = src + static_cast<size_t>(j) * ss;
    int i = 0;
    for (; i + 16 <= bw; i += 16) {
      __m256i v = _mm256_srai_epi16(_mm256_adds_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(pj + i)), k32), 6);   // (saturating: only values past 16320 + 32 can saturate, and they clip to 255 either way)
      v = _mm256_min_epi16(_mm256_max_epi16(v, zero), c255);
      const __m256i sv = _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(sj + i)));
      acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_abs_epi16(_mm256_sub_epi16(sv, v)), ones));
    }
    for (; i + 8 <= bw; i += 8) {
      __m128i v = _mm_srai_epi16(_mm_adds_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(pj + i)), _mm256_castsi256_si128(k32)), 6);
      v = _mm_min_epi16(_mm_max_epi16(v, _mm256_castsi256_si128(zero)), _mm256_castsi256_si128(c255));
      const __m128i sv = _mm_cvtepu8_epi16(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(sj + i)));
      acc = _mm256_add_epi32(acc, _mm256_castsi128_si256(_mm_madd_epi16(_mm_abs_epi16(_mm_sub_epi16(sv, v)), _mm256_castsi256_si128(ones))));
    }
    for (; i < bw; ++i) {
      int v = (pj[i] + 32) >> 6;
      v = v < 0 ? 0 : (v > 255 ? 255 : v);
      const int d = static_cast<int>(sj[i]) - v;
      tail += d < 0 ? -d : d;
    }
  }
  const __m128i s4 = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
  const __m128i s2 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 0x4e));
  return tail + _mm_cvtsi128_si32(_mm_add_epi32(s2, _mm_shuffle_epi32(s2, 0xb1)));
}

// the same for the average of two predictions: clip8((a + b + 64) >> 7).
inline int sad_bi(const int16_t* a, const int16_t* b, int bw, int bh, const uint8_t* src, int ss) {
  const __m256i k64 = _mm256_set1_epi32(64), zero = _mm256_setzero_si256(), c255 = _mm256_set1_epi32(255);
  __m256i acc = _mm256_setzero_si256();
  int tail = 0;
  for (int j = 0; j < bh; ++j) {
    const int16_t* aj = a + static_cast<size_t>(j) * bw;
    const int16_t* bj = b + static_cast<size_t>(j) * bw;
    const uint8_t* sj = src + static_cast<size_t>(j) * ss;
    int i = 0;
    for (; i + 8 <= bw; i += 8) {
      const __m256i x = _mm256_cvtepi16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(aj + i)));
      const __m256i y = _mm256_cvtepi16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(bj + i)));
      __m256i v = _mm256_srai_epi32(_mm256_add_epi32(_mm256_add_epi32(x, y), k64), 7);
      v = _mm256_min_epi32(_mm256_max_epi32(v, zero), c255);
      const __m256i sv = _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(sj + i)));
      acc = _mm256_add_epi32(acc, _mm256_abs_epi32(_mm256_sub_epi32(sv, v)));
    }
    for (; i < bw; ++i) {
      int v = (aj[i] + bj[i] + 64) >> 7;
      v = v < 0 ? 0 : (v > 255 ? 255 : v);
      const int d = static_cast<int>(sj[i]) - v;
      tail += d < 0 ? -d : d;
    }
  }
  const __m128i s4 = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
  const __m128i s2 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 0x4e));
  return tail + _mm_cvtsi128_si32(_mm_add_epi32(s2, _mm_shuffle_epi32(s2, 0xb1)));
}


// ---- the inverse DCT of MPEG-2 (mpeg2_idct.inc) ----
//
// Two passes of eight-term sums in double, each term rounded and then added (no fused multiply-add: the target has no FMA), the sum
// starting at 0 and taking its terms in order; the outputs of a row are the four lanes of a vector, so each lane is the chain the
// scalar loop makes. The sums are rounded to the nearest integer, halves up (floor(s + 0.5)), and saturated to -256 .. 255.
// c[x][u] is the scalar code's basis. The transposed table is formed on the first call.
inline void mpeg2_idct(int16_t* blk, const double (*c)[8]) {
  alignas(32) static double ct[8][8];   // ct[u][x] = c[x][u]: the lanes (x) of one u are adjacent
  static const bool once = [&] {
    for (int u = 0; u < 8; ++u)
      for (int x = 0; x < 8; ++x) ct[u][x] = c[x][u];
    return true;
  }();
  (void)once;
  alignas(32) double tmp[64];
  for (int v = 0; v < 8; ++v) {
    __m256d lo = _mm256_setzero_pd(), hi = _mm256_setzero_pd();   // outputs x = 0..3 and 4..7 of row v
    for (int u = 0; u < 8; ++u) {
      const __m256d b = _mm256_set1_pd(static_cast<double>(blk[v * 8 + u]));
      lo = _mm256_add_pd(lo, _mm256_mul_pd(_mm256_load_pd(&ct[u][0]), b));
      hi = _mm256_add_pd(hi, _mm256_mul_pd(_mm256_load_pd(&ct[u][4]), b));
    }
    _mm256_store_pd(&tmp[v * 8], lo);
    _mm256_store_pd(&tmp[v * 8 + 4], hi);
  }
  const __m256d half = _mm256_set1_pd(0.5);
  const __m256i lo256 = _mm256_set1_epi32(-256), hi255 = _mm256_set1_epi32(255);
  for (int y = 0; y < 8; ++y) {
    __m256d lo = _mm256_setzero_pd(), hi = _mm256_setzero_pd();   // outputs x = 0..3 and 4..7 of row y
    for (int v = 0; v < 8; ++v) {
      const __m256d cv = _mm256_set1_pd(c[y][v]);
      lo = _mm256_add_pd(lo, _mm256_mul_pd(cv, _mm256_load_pd(&tmp[v * 8])));
      hi = _mm256_add_pd(hi, _mm256_mul_pd(cv, _mm256_load_pd(&tmp[v * 8 + 4])));
    }
    const __m128i rl = _mm256_cvttpd_epi32(_mm256_floor_pd(_mm256_add_pd(lo, half)));
    const __m128i rh = _mm256_cvttpd_epi32(_mm256_floor_pd(_mm256_add_pd(hi, half)));
    __m256i r = _mm256_inserti128_si256(_mm256_castsi128_si256(rl), rh, 1);
    r = _mm256_min_epi32(_mm256_max_epi32(r, lo256), hi255);
    const __m128i p = _mm_packs_epi32(_mm256_castsi256_si128(r), _mm256_extracti128_si256(r, 1));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(blk + y * 8), p);
  }
}

}  // namespace avx2
#pragma GCC pop_options
#endif

// The 2-D core transforms for log2 = 3, 4, 5, with the 32x32 matrix T (row k, column n) the scalar code uses.
// Only when avx2_available().
inline void forward_dct(const int32_t* res, int32_t* coef, int log2, const int8_t (*M)[32]) {
#ifdef VGPU_VIDEO_AVX2
  switch (log2) {
    case 3: avx2::forward_dct<8>(res, coef, M); return;
    case 4: avx2::forward_dct<16>(res, coef, M); return;
    default: avx2::forward_dct<32>(res, coef, M); return;
  }
#else
  (void)res; (void)coef; (void)log2; (void)M;
#endif
}
inline void inverse_dct(const int32_t* d, int32_t* res, int log2, const int8_t (*M)[32], int shift2 = 12) {
#ifdef VGPU_VIDEO_AVX2
  switch (log2) {
    case 3: avx2::inverse_dct<8>(d, res, M, shift2); return;
    case 4: avx2::inverse_dct<16>(d, res, M, shift2); return;
    default: avx2::inverse_dct<32>(d, res, M, shift2); return;
  }
#else
  (void)d; (void)res; (void)log2; (void)M; (void)shift2;
#endif
}

// Filtering of a block: taps = 8 (luma) or 4 (chroma); h-only, v-only or both; see avx2::fir_*. The src pointers are as in the scalar
// code: h: the first tap of (0, 0); v: the first tap row; hv: the first tap of the first row. Only when avx2_available().
inline void fir_h(const uint8_t* src, int st, int bw, int bh, int taps, const int* cx, int16_t* out) {
#ifdef VGPU_VIDEO_AVX2
  if (taps == 8) avx2::fir_h<8>(src, st, bw, bh, cx, out);
  else avx2::fir_h<4>(src, st, bw, bh, cx, out);
#else
  (void)src; (void)st; (void)bw; (void)bh; (void)taps; (void)cx; (void)out;
#endif
}
inline void fir_v(const uint8_t* src, int st, int bw, int bh, int taps, const int* cy, int16_t* out) {
#ifdef VGPU_VIDEO_AVX2
  if (taps == 8) avx2::fir_v<8>(src, st, bw, bh, cy, out);
  else avx2::fir_v<4>(src, st, bw, bh, cy, out);
#else
  (void)src; (void)st; (void)bw; (void)bh; (void)taps; (void)cy; (void)out;
#endif
}
inline void fir_hv(const uint8_t* src, int st, int bw, int bh, int taps, const int* cx, const int* cy, int16_t* out) {
#ifdef VGPU_VIDEO_AVX2
  if (taps == 8) avx2::fir_hv<8>(src, st, bw, bh, cx, cy, out);
  else avx2::fir_hv<4>(src, st, bw, bh, cx, cy, out);
#else
  (void)src; (void)st; (void)bw; (void)bh; (void)taps; (void)cx; (void)cy; (void)out;
#endif
}

// Sums of absolute differences; only when avx2_available().
inline int sad_u8(const uint8_t* a, int as, const uint8_t* b, int bs, int bw, int bh) {
#ifdef VGPU_VIDEO_AVX2
  return avx2::sad_u8(a, as, b, bs, bw, bh);
#else
  (void)a; (void)as; (void)b; (void)bs; (void)bw; (void)bh;
  return 0;
#endif
}
inline int sad_uni(const int16_t* p, int bw, int bh, const uint8_t* src, int ss) {
#ifdef VGPU_VIDEO_AVX2
  return avx2::sad_uni(p, bw, bh, src, ss);
#else
  (void)p; (void)bw; (void)bh; (void)src; (void)ss;
  return 0;
#endif
}
inline int sad_bi(const int16_t* a, const int16_t* b, int bw, int bh, const uint8_t* src, int ss) {
#ifdef VGPU_VIDEO_AVX2
  return avx2::sad_bi(a, b, bw, bh, src, ss);
#else
  (void)a; (void)b; (void)bw; (void)bh; (void)src; (void)ss;
  return 0;
#endif
}

// The MPEG-2 inverse DCT of a block of 64 coefficients in place (blk[v * 8 + u]), with the scalar code's basis c[x][u]; only when
// avx2_available().
inline void mpeg2_idct(int16_t* blk, const double (*c)[8]) {
#ifdef VGPU_VIDEO_AVX2
  avx2::mpeg2_idct(blk, c);
#else
  (void)blk; (void)c;
#endif
}

}  // namespace vgpu::videosimd
