#pragma once
// The inverse DCT of nvJPEG's host decoder, with the arithmetic of the scalar loops it replaces.
//
// The transform is two passes of eight-term sums in single precision, each term a fused multiply-add
// (std::fmaf), the sum starting at 0 and taking u = 0 .. 7 in order; the result is level-shifted, floored and
// clamped to a byte. A fused multiply-add is one rounding of an exact product plus sum, so a hardware FMA
// gives the bits fmaf gives, whichever way it is called. The AVX2/FMA path therefore does the 8 outputs of a
// row at once, one per lane, each lane the same chain of eight fused steps in the same order.
// (tests/unit/test_host_jpeg.cpp keeps the scalar loops as the reference.)
#include <cmath>
#include <cstdint>
#include <cstring>

#if defined(__x86_64__) && defined(__GNUC__)
#include <immintrin.h>
#define VGPU_HOSTJPEG_AVX2 1
#endif

namespace vgpu::hostjpeg {

// c[u][x] = (u == 0 ? sqrt(1/8) : 1/2) * cos((2x + 1) u pi / 16), rounded to float.
inline const float (*basis())[8] {
  static const struct Table {
    float c[8][8];
    Table() {
      for (int u = 0; u < 8; ++u)
        for (int x = 0; x < 8; ++x)
          c[u][x] = static_cast<float>((u == 0 ? std::sqrt(0.125) : 0.5) * std::cos((2 * x + 1) * u * M_PI / 16.0));
    }
  } t;
  return t.c;
}

// The scalar transform: `in` is the dequantised block in natural order; out is 8 rows of 8 bytes at `stride`.
inline void idct8x8_scalar(const int* in, uint8_t* out, int stride) {
  const float(*c)[8] = basis();
  float tmp[64];
  for (int y = 0; y < 8; ++y)
    for (int x = 0; x < 8; ++x) {
      float s = 0;
      for (int u = 0; u < 8; ++u) s = std::fmaf(c[u][x], static_cast<float>(in[y * 8 + u]), s);
      tmp[y * 8 + x] = s;
    }
  for (int x = 0; x < 8; ++x)
    for (int y = 0; y < 8; ++y) {
      float s = 0;
      for (int v = 0; v < 8; ++v) s = std::fmaf(c[v][y], tmp[v * 8 + x], s);
      const float r = std::floor(s + 128.5f);
      out[y * stride + x] = static_cast<uint8_t>(r < 0 ? 0 : (r > 255 ? 255 : r));
    }
}

#ifdef VGPU_HOSTJPEG_AVX2
__attribute__((target("avx2,fma"))) inline void idct8x8_avx2(const int* in, uint8_t* out, int stride) {
  const float(*c)[8] = basis();
  __m256 ct[8];
  for (int u = 0; u < 8; ++u) ct[u] = _mm256_loadu_ps(c[u]);
  __m256 tmp[8];   // tmp[y] holds row y of the first pass, lane x
  for (int y = 0; y < 8; ++y) {
    __m256 s = _mm256_setzero_ps();
    for (int u = 0; u < 8; ++u) s = _mm256_fmadd_ps(ct[u], _mm256_set1_ps(static_cast<float>(in[y * 8 + u])), s);
    tmp[y] = s;
  }
  const __m256 shift = _mm256_set1_ps(128.5f), lo = _mm256_setzero_ps(), hi = _mm256_set1_ps(255.0f);
  for (int y = 0; y < 8; ++y) {
    __m256 s = _mm256_setzero_ps();
    for (int v = 0; v < 8; ++v) s = _mm256_fmadd_ps(_mm256_set1_ps(c[v][y]), tmp[v], s);
    __m256 r = _mm256_floor_ps(_mm256_add_ps(s, shift));
    r = _mm256_min_ps(_mm256_max_ps(r, lo), hi);
    const __m256i i32 = _mm256_cvttps_epi32(r);
    const __m128i i16 = _mm_packus_epi32(_mm256_castsi256_si128(i32), _mm256_extracti128_si256(i32, 1));
    const __m128i i8 = _mm_packus_epi16(i16, i16);
    _mm_storel_epi64(reinterpret_cast<__m128i*>(out + y * stride), i8);
  }
}
#endif

inline bool wide_available() {
#ifdef VGPU_HOSTJPEG_AVX2
  static const bool ok = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
  return ok;
#else
  return false;
#endif
}

// `wide` is for tests; the library asks for whatever the host has.
inline void idct8x8(const int* in, uint8_t* out, int stride, bool wide = wide_available()) {
#ifdef VGPU_HOSTJPEG_AVX2
  if (wide) {
    idct8x8_avx2(in, out, stride);
    return;
  }
#endif
  (void)wide;
  idct8x8_scalar(in, out, stride);
}

}  // namespace vgpu::hostjpeg

// ---- colour conversion ----
//
// YCbCr to RGB as nvjpeg_api.cpp's ycbcr_to_rgb does it, in single precision with every product and sum rounded
// separately (no fused multiply-add), then rounded to nearest even and clamped:
//   R = y + 1.402f*(cr - 128), G = (y - 0.344136f*(cb - 128)) - 0.714136f*(cr - 128), B = y + 1.772f*(cb - 128).
// The wide path is compiled for AVX2 without FMA, so it cannot contract either; it does eight pixels at a time.
namespace vgpu::hostjpeg {

inline uint8_t to_byte(float v) {
  const float r = std::nearbyint(v);
  return static_cast<uint8_t>(r < 0 ? 0 : (r > 255 ? 255 : r));
}

#ifdef VGPU_HOSTJPEG_AVX2
// Eight bytes to eight floats; eight floats to bytes (to nearest even, clamped to 0..255).
__attribute__((target("avx2"))) inline __m256 load8_ps(const uint8_t* p) {
  return _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p))));
}
__attribute__((target("avx2"))) inline void store8_bytes(__m256 v, uint8_t* dst) {
  __m256i i = _mm256_cvtps_epi32(v);   // to nearest even, as nearbyint
  i = _mm256_min_epi32(_mm256_max_epi32(i, _mm256_setzero_si256()), _mm256_set1_epi32(255));
  const __m128i p16 = _mm_packus_epi32(_mm256_castsi256_si128(i), _mm256_extracti128_si256(i, 1));
  _mm_storel_epi64(reinterpret_cast<__m128i*>(dst), _mm_packus_epi16(p16, p16));
}
#endif

inline void ycc_to_rgb_scalar(const uint8_t* y, const uint8_t* cb, const uint8_t* cr, uint8_t* r, uint8_t* g, uint8_t* b,
                              int64_t n) {
  for (int64_t i = 0; i < n; ++i) {
    const float yy = y[i], bb = cb[i], rr = cr[i];
    r[i] = to_byte(yy + 1.402f * (rr - 128.0f));
    g[i] = to_byte(yy - 0.344136f * (bb - 128.0f) - 0.714136f * (rr - 128.0f));
    b[i] = to_byte(yy + 1.772f * (bb - 128.0f));
  }
}

#ifdef VGPU_HOSTJPEG_AVX2
__attribute__((target("avx2"))) inline void ycc_to_rgb_avx2(const uint8_t* y, const uint8_t* cb, const uint8_t* cr,
                                                           uint8_t* r, uint8_t* g, uint8_t* b, int64_t n) {
  const __m256 c128 = _mm256_set1_ps(128.0f), k1 = _mm256_set1_ps(1.402f), k2 = _mm256_set1_ps(0.344136f),
               k3 = _mm256_set1_ps(0.714136f), k4 = _mm256_set1_ps(1.772f);
  int64_t i = 0;
  for (; i + 8 <= n; i += 8) {
    const __m256 yy = load8_ps(y + i);
    const __m256 bb = _mm256_sub_ps(load8_ps(cb + i), c128);
    const __m256 rr = _mm256_sub_ps(load8_ps(cr + i), c128);
    store8_bytes(_mm256_add_ps(yy, _mm256_mul_ps(k1, rr)), r + i);
    store8_bytes(_mm256_sub_ps(_mm256_sub_ps(yy, _mm256_mul_ps(k2, bb)), _mm256_mul_ps(k3, rr)), g + i);
    store8_bytes(_mm256_add_ps(yy, _mm256_mul_ps(k4, bb)), b + i);
  }
  ycc_to_rgb_scalar(y + i, cb + i, cr + i, r + i, g + i, b + i, n - i);
}
#endif

inline bool avx2_available() {
#ifdef VGPU_HOSTJPEG_AVX2
  static const bool ok = __builtin_cpu_supports("avx2");
  return ok;
#else
  return false;
#endif
}

inline void ycc_to_rgb(const uint8_t* y, const uint8_t* cb, const uint8_t* cr, uint8_t* r, uint8_t* g, uint8_t* b,
                       int64_t n, bool wide = avx2_available()) {
#ifdef VGPU_HOSTJPEG_AVX2
  if (wide) {
    ycc_to_rgb_avx2(y, cb, cr, r, g, b, n);
    return;
  }
#endif
  (void)wide;
  ycc_to_rgb_scalar(y, cb, cr, r, g, b, n);
}

// 0.299*R + 0.587*G + 0.114*B in single precision, left to right with every step rounded, to a byte as above.
inline void rgb_to_luma_scalar(const uint8_t* r, const uint8_t* g, const uint8_t* b, uint8_t* out, int64_t n) {
  for (int64_t i = 0; i < n; ++i) out[i] = to_byte(0.299f * r[i] + 0.587f * g[i] + 0.114f * b[i]);
}
#ifdef VGPU_HOSTJPEG_AVX2
__attribute__((target("avx2"))) inline void rgb_to_luma_avx2(const uint8_t* r, const uint8_t* g, const uint8_t* b,
                                                            uint8_t* out, int64_t n) {
  const __m256 k1 = _mm256_set1_ps(0.299f), k2 = _mm256_set1_ps(0.587f), k3 = _mm256_set1_ps(0.114f);
  int64_t i = 0;
  for (; i + 8 <= n; i += 8)
    store8_bytes(_mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(k1, load8_ps(r + i)), _mm256_mul_ps(k2, load8_ps(g + i))),
                               _mm256_mul_ps(k3, load8_ps(b + i))),
                 out + i);
  rgb_to_luma_scalar(r + i, g + i, b + i, out + i, n - i);
}
#endif
inline void rgb_to_luma(const uint8_t* r, const uint8_t* g, const uint8_t* b, uint8_t* out, int64_t n,
                        bool wide = avx2_available()) {
#ifdef VGPU_HOSTJPEG_AVX2
  if (wide) {
    rgb_to_luma_avx2(r, g, b, out, n);
    return;
  }
#endif
  (void)wide;
  rgb_to_luma_scalar(r, g, b, out, n);
}


// ---- the forward DCT of the encoder ----
//
// Two passes of eight-term sums in double, each term c[u][x] * in rounded and then added (no fused multiply-add), the sum
// starting at 0 and taking x = 0 .. 7 in order. The wide path keeps four outputs of a row in the lanes of a vector, each lane
// the same chain in the same order.

// c[u][x] = (u == 0 ? sqrt(1/8) : 1/2) * cos((2x + 1) u pi / 16), in double.
inline const double (*basis_d())[8] {
  static const struct Table {
    double c[8][8];
    Table() {
      for (int u = 0; u < 8; ++u)
        for (int x = 0; x < 8; ++x) c[u][x] = (u == 0 ? std::sqrt(0.125) : 0.5) * std::cos((2 * x + 1) * u * M_PI / 16.0);
    }
  } t;
  return t.c;
}

inline void fdct8x8_scalar(const double* in, double* out) {
  const double(*c)[8] = basis_d();
  double tmp[64];
  for (int y = 0; y < 8; ++y)
    for (int u = 0; u < 8; ++u) {
      double s = 0;
      for (int x = 0; x < 8; ++x) s += c[u][x] * in[y * 8 + x];
      tmp[y * 8 + u] = s;
    }
  for (int u = 0; u < 8; ++u)
    for (int v = 0; v < 8; ++v) {
      double s = 0;
      for (int y = 0; y < 8; ++y) s += c[v][y] * tmp[y * 8 + u];
      out[v * 8 + u] = s;
    }
}

#ifdef VGPU_HOSTJPEG_AVX2
__attribute__((target("avx2"))) inline void fdct8x8_avx2(const double* in, double* out) {
  const double(*c)[8] = basis_d();
  // ct[x][u]: the coefficient of input x for output u, so the lanes (u) of one x are adjacent.
  alignas(32) double ct[8][8];
  for (int x = 0; x < 8; ++x)
    for (int u = 0; u < 8; ++u) ct[x][u] = c[u][x];
  alignas(32) double tmp[64];
  for (int y = 0; y < 8; ++y) {
    __m256d lo = _mm256_setzero_pd(), hi = _mm256_setzero_pd();   // outputs u = 0..3 and 4..7 of row y
    for (int x = 0; x < 8; ++x) {
      const __m256d v = _mm256_set1_pd(in[y * 8 + x]);
      lo = _mm256_add_pd(lo, _mm256_mul_pd(_mm256_load_pd(&ct[x][0]), v));
      hi = _mm256_add_pd(hi, _mm256_mul_pd(_mm256_load_pd(&ct[x][4]), v));
    }
    _mm256_store_pd(&tmp[y * 8], lo);
    _mm256_store_pd(&tmp[y * 8 + 4], hi);
  }
  for (int v = 0; v < 8; ++v) {
    __m256d lo = _mm256_setzero_pd(), hi = _mm256_setzero_pd();   // outputs u = 0..3 and 4..7 of coefficient row v
    for (int y = 0; y < 8; ++y) {
      const __m256d cv = _mm256_set1_pd(c[v][y]);
      lo = _mm256_add_pd(lo, _mm256_mul_pd(cv, _mm256_load_pd(&tmp[y * 8])));
      hi = _mm256_add_pd(hi, _mm256_mul_pd(cv, _mm256_load_pd(&tmp[y * 8 + 4])));
    }
    _mm256_storeu_pd(&out[v * 8], lo);
    _mm256_storeu_pd(&out[v * 8 + 4], hi);
  }
}
#endif

inline void fdct8x8(const double* in, double* out, bool wide = avx2_available()) {
#ifdef VGPU_HOSTJPEG_AVX2
  if (wide) {
    fdct8x8_avx2(in, out);
    return;
  }
#endif
  (void)wide;
  fdct8x8_scalar(in, out);
}

}  // namespace vgpu::hostjpeg
