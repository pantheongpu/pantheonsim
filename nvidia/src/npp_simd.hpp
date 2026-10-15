#pragma once
// AVX2 versions of the geometric samplers behind NPP's warps, rotation, remapping and resizing (npp_core.hpp's warp() and
// sample(), npp_imgproc.cpp's resize_sqr): eight destination pixels at a time, each lane computing what the scalar code
// computes for its pixel with the same single-precision operations in the same order -- the same float products, the same
// fused multiply-adds (written out with _mm256_fmadd_ps where the scalar code calls std::fma, so there is one rounding
// where it has one), no other fusion -- and the same rounding to the pixel type. npp_core.hpp's scalar code is the generic
// path and the reference of tests/unit/test_npp_simd.cpp, which compares every pixel of the results.
//
// What is covered: 8-bit, 16-bit unsigned and 32-bit float images of one, three and four channels, with nearest, linear and
// cubic sampling. Anything else (32-bit integer images, other channel counts, a CPU without AVX2 and FMA) is the generic
// path's, which the callers fall back to when avail<T>() says no.
//
// Reading a pixel is a gather of 32 bits at its first byte, so a source has to be followed by kPad readable bytes: the copy
// the callers make is (fetch_padded()).
//
// Nothing between the pragmas may call a std:: template (it would share its mangled name with the baseline instantiation), and
// the pixel-type conversions are written out here.
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "npp_core.hpp"

#if defined(__x86_64__) && defined(__GNUC__)
#define VGPU_NPP_AVX2 1
#include <immintrin.h>
#endif

namespace vgpu_npp::simd {

// Bytes that must be readable after the last pixel of a source image.
constexpr size_t kPad = 16;

inline bool cpu_ok() {
#ifdef VGPU_NPP_AVX2
  static const bool ok = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
  return ok;
#else
  return false;
#endif
}

// Whether the fast path handles this pixel type and channel count.
template <class T>
constexpr bool type_ok() {
  return std::is_same_v<T, uint8_t> || std::is_same_v<T, uint16_t> || std::is_same_v<T, float>;
}
template <class T>
inline bool avail(int ch) {
  return cpu_ok() && type_ok<T>() && (ch == 1 || ch == 3 || ch == 4);
}

#ifdef VGPU_NPP_AVX2
#pragma GCC push_options
#pragma GCC target("avx2,fma")
namespace avx2 {

typedef __m256 V;
typedef __m256i VI;

// The image a sampler reads: the whole image (taps clamp to it), `stride` elements per row.
struct Src {
  const void* p;
  int w, h, stride;
};

inline VI clampi(VI v, VI lo, VI hi) { return _mm256_min_epi32(_mm256_max_epi32(v, lo), hi); }

template <class T> struct Fmt;
template <> struct Fmt<uint8_t> { static constexpr int bytes = 1; };
template <> struct Fmt<uint16_t> { static constexpr int bytes = 2; };
template <> struct Fmt<float> { static constexpr int bytes = 4; };

// The raw value (as an integer lane: the byte, the 16-bit word, or the float's bits) of channel `c` of the pixels at the element
// offsets `off` (element offsets of the first channel; c is added here).
template <class T, int CH>
inline VI load_raw(const Src& s, VI off, int c) {
  if constexpr (std::is_same_v<T, uint8_t>) {
    const VI g = _mm256_i32gather_epi32(reinterpret_cast<const int*>(s.p), _mm256_add_epi32(off, _mm256_set1_epi32(c)), 1);
    return _mm256_and_si256(g, _mm256_set1_epi32(0xFF));
  } else if constexpr (std::is_same_v<T, uint16_t>) {
    const VI g = _mm256_i32gather_epi32(reinterpret_cast<const int*>(s.p), _mm256_add_epi32(off, _mm256_set1_epi32(c)), 2);
    return _mm256_and_si256(g, _mm256_set1_epi32(0xFFFF));
  } else {
    return _mm256_i32gather_epi32(reinterpret_cast<const int*>(s.p), _mm256_add_epi32(off, _mm256_set1_epi32(c)), 4);
  }
}
// The same, as a float (what the scalar code makes of the pixel with static_cast<float>).
template <class T, int CH>
inline V load_f(const Src& s, VI off, int c) {
  const VI raw = load_raw<T, CH>(s, off, c);
  if constexpr (std::is_same_v<T, float>) return _mm256_castsi256_ps(raw);
  else return _mm256_cvtepi32_ps(raw);
}
// All the channels of the pixels at `off`, as floats. An 8-bit pixel of up to four channels is one gather.
template <class T, int CH>
inline void load_pixels_f(const Src& s, VI off, V* out) {
  if constexpr (std::is_same_v<T, uint8_t>) {
    const VI g = _mm256_i32gather_epi32(reinterpret_cast<const int*>(s.p), off, 1);
    for (int c = 0; c < CH; ++c)
      out[c] = _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(g, 8 * c), _mm256_set1_epi32(0xFF)));
  } else {
    for (int c = 0; c < CH; ++c) out[c] = load_f<T, CH>(s, off, c);
  }
}

// Element offset of the pixel in column xi, row yi.
inline VI pixel_offset(const Src& s, VI xi, VI yi, int ch) {
  return _mm256_add_epi32(_mm256_mullo_epi32(yi, _mm256_set1_epi32(s.stride)), _mm256_mullo_epi32(xi, _mm256_set1_epi32(ch)));
}

// to_pixel<T>(double(v)) for a float v, as a float lane holding the integer result (8- and 16-bit) -- round half away from zero
// and saturate; a negative value and a NaN give 0. floor(v + 0.5) with v + 0.5 formed in double is trunc(v) plus one when the
// fraction (exact in float) is at least a half.
template <class T>
inline V round_pixel(V v) {
  constexpr float hi = std::is_same_v<T, uint8_t> ? 255.0f : 65535.0f;
  const V ge0 = _mm256_cmp_ps(v, _mm256_setzero_ps(), _CMP_GE_OQ);
  const V t = _mm256_round_ps(v, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
  const V inc = _mm256_and_ps(_mm256_cmp_ps(_mm256_sub_ps(v, t), _mm256_set1_ps(0.5f), _CMP_GE_OQ), _mm256_set1_ps(1.0f));
  V r = _mm256_and_ps(_mm256_add_ps(t, inc), ge0);
  return _mm256_min_ps(r, _mm256_set1_ps(hi));
}

// The output of eight pixels: res[c][lane], written for the lanes in okm.
template <class T, int CH>
struct Out {
  alignas(32) T res[CH][8];
};

// Float results (linear, cubic) to the pixel type.
template <class T, int CH>
inline void convert_f(const V* v, Out<T, CH>& o) {
  for (int c = 0; c < CH; ++c) {
    if constexpr (std::is_same_v<T, float>) {
      _mm256_store_ps(o.res[c], v[c]);
    } else {
      const VI ri = _mm256_cvttps_epi32(round_pixel<T>(v[c]));
      const __m128i lo = _mm256_castsi256_si128(ri), hi = _mm256_extracti128_si256(ri, 1);
      const __m128i p16 = _mm_packus_epi32(lo, hi);
      if constexpr (std::is_same_v<T, uint8_t>) {
        const __m128i p8 = _mm_packus_epi16(p16, p16);
        _mm_storel_epi64(reinterpret_cast<__m128i*>(o.res[c]), p8);
      } else {
        _mm_store_si128(reinterpret_cast<__m128i*>(o.res[c]), p16);
      }
    }
  }
}
// Raw lanes (nearest neighbour) to the pixel type.
template <class T, int CH>
inline void convert_raw(const VI* v, Out<T, CH>& o) {
  for (int c = 0; c < CH; ++c) {
    if constexpr (std::is_same_v<T, float>) {
      _mm256_store_si256(reinterpret_cast<VI*>(o.res[c]), v[c]);
    } else {
      const __m128i lo = _mm256_castsi256_si128(v[c]), hi = _mm256_extracti128_si256(v[c], 1);
      const __m128i p16 = _mm_packus_epi32(lo, hi);
      if constexpr (std::is_same_v<T, uint8_t>) {
        const __m128i p8 = _mm_packus_epi16(p16, p16);
        _mm_storel_epi64(reinterpret_cast<__m128i*>(o.res[c]), p8);
      } else {
        _mm_store_si128(reinterpret_cast<__m128i*>(o.res[c]), p16);
      }
    }
  }
}

template <class T, int CH>
inline void store_lanes(T* drow, const Out<T, CH>& o, int okm) {
  if (okm == 0xFF) {
    for (int l = 0; l < 8; ++l)
      for (int c = 0; c < CH; ++c) drow[l * CH + c] = o.res[c][l];
  } else {
    for (int l = 0; l < 8; ++l)
      if (okm & (1 << l))
        for (int c = 0; c < CH; ++c) drow[l * CH + c] = o.res[c][l];
  }
}

// ---- the three samplers ----

// Nearest: std::floor(s + 0.5f), clamped to the image.
template <class T, int CH>
inline void sample_nn(const Src& s, V sx, V sy, Out<T, CH>& o) {
  const V half = _mm256_set1_ps(0.5f);
  VI xi = _mm256_cvttps_epi32(_mm256_floor_ps(_mm256_add_ps(sx, half)));
  VI yi = _mm256_cvttps_epi32(_mm256_floor_ps(_mm256_add_ps(sy, half)));
  xi = clampi(xi, _mm256_setzero_si256(), _mm256_set1_epi32(s.w - 1));
  yi = clampi(yi, _mm256_setzero_si256(), _mm256_set1_epi32(s.h - 1));
  const VI off = pixel_offset(s, xi, yi, CH);
  VI v[CH];
  for (int c = 0; c < CH; ++c) v[c] = load_raw<T, CH>(s, off, c);
  convert_raw<T, CH>(v, o);
}

// Linear: the four taps clamp to the image; per channel top = fma(p00, 1-fx, p10*fx), bot likewise, out = fma(top, 1-fy, bot*fy).
template <class T, int CH>
inline void sample_linear(const Src& s, V sx, V sy, Out<T, CH>& o) {
  const V x0f = _mm256_floor_ps(sx), y0f = _mm256_floor_ps(sy);
  const VI x0 = _mm256_cvttps_epi32(x0f), y0 = _mm256_cvttps_epi32(y0f);
  const V fx = _mm256_sub_ps(sx, x0f), fy = _mm256_sub_ps(sy, y0f);
  const V one = _mm256_set1_ps(1.0f);
  const V gx = _mm256_sub_ps(one, fx), gy = _mm256_sub_ps(one, fy);
  const VI zero = _mm256_setzero_si256(), wm1 = _mm256_set1_epi32(s.w - 1), hm1 = _mm256_set1_epi32(s.h - 1), one_i = _mm256_set1_epi32(1);
  const VI xa = clampi(x0, zero, wm1), ya = clampi(y0, zero, hm1);
  const VI x1 = clampi(_mm256_add_epi32(x0, one_i), zero, wm1), y1 = clampi(_mm256_add_epi32(y0, one_i), zero, hm1);
  V p00[CH], p10[CH], p01[CH], p11[CH];
  load_pixels_f<T, CH>(s, pixel_offset(s, xa, ya, CH), p00);
  load_pixels_f<T, CH>(s, pixel_offset(s, x1, ya, CH), p10);
  load_pixels_f<T, CH>(s, pixel_offset(s, xa, y1, CH), p01);
  load_pixels_f<T, CH>(s, pixel_offset(s, x1, y1, CH), p11);
  V r[CH];
  for (int c = 0; c < CH; ++c) {
    const V top = _mm256_fmadd_ps(p00[c], gx, _mm256_mul_ps(p10[c], fx));
    const V bot = _mm256_fmadd_ps(p01[c], gx, _mm256_mul_ps(p11[c], fx));
    r[c] = _mm256_fmadd_ps(top, gy, _mm256_mul_ps(bot, fy));
  }
  convert_f<T, CH>(r, o);
}

// The Lagrange weights of npp_core.hpp's lagrange4, in the same operations.
inline void lagrange4v(V f, V w[4]) {
  const V one = _mm256_set1_ps(1.0f), two = _mm256_set1_ps(2.0f), six = _mm256_set1_ps(6.0f);
  const V neg = _mm256_set1_ps(-0.0f);
  const V omf = _mm256_sub_ps(one, f);                       // 1 - f
  const V fp1 = _mm256_add_ps(f, one);                       // f + 1
  // w0 = -((f * (1 - f)) * ((2 - f) / 6))
  w[0] = _mm256_xor_ps(_mm256_mul_ps(_mm256_mul_ps(f, omf), _mm256_div_ps(_mm256_sub_ps(two, f), six)), neg);
  // w1 = (f + 1) * (f - 1) * (f - 2) / 2
  w[1] = _mm256_div_ps(_mm256_mul_ps(_mm256_mul_ps(fp1, _mm256_sub_ps(f, one)), _mm256_sub_ps(f, two)), two);
  // w2 = -(f + 1) * f * (f - 2) / 2
  w[2] = _mm256_div_ps(_mm256_mul_ps(_mm256_mul_ps(_mm256_xor_ps(fp1, neg), f), _mm256_sub_ps(f, two)), two);
  // w3 = -((((f + 1) / 6) * f) * (1 - f))
  w[3] = _mm256_xor_ps(_mm256_mul_ps(_mm256_mul_ps(_mm256_div_ps(fp1, six), f), omf), neg);
}

// Cubic: four taps each way (clamped), each row a fused chain starting at the second tap, the column of row results one starting
// at the first (npp_core.hpp's row4 and col4).
template <class T, int CH>
inline void sample_cubic(const Src& s, V sx, V sy, Out<T, CH>& o) {
  const V x0f = _mm256_floor_ps(sx), y0f = _mm256_floor_ps(sy);
  const VI x0 = _mm256_cvttps_epi32(x0f), y0 = _mm256_cvttps_epi32(y0f);
  const V fx = _mm256_sub_ps(sx, x0f), fy = _mm256_sub_ps(sy, y0f);
  V wx[4], wy[4];
  lagrange4v(fx, wx);
  lagrange4v(fy, wy);
  const VI zero = _mm256_setzero_si256(), wm1 = _mm256_set1_epi32(s.w - 1), hm1 = _mm256_set1_epi32(s.h - 1);
  VI xs[4], ys[4];
  for (int i = 0; i < 4; ++i) {
    xs[i] = clampi(_mm256_add_epi32(x0, _mm256_set1_epi32(i - 1)), zero, wm1);
    ys[i] = clampi(_mm256_add_epi32(y0, _mm256_set1_epi32(i - 1)), zero, hm1);
  }
  V rows[CH][4];
  for (int j = 0; j < 4; ++j) {
    V p[4][CH];
    for (int i = 0; i < 4; ++i) load_pixels_f<T, CH>(s, pixel_offset(s, xs[i], ys[j], CH), p[i]);
    for (int c = 0; c < CH; ++c) {
      // row4: fma(w3, p3, fma(w2, p2, fma(w0, p0, w1 * p1)))
      rows[c][j] = _mm256_fmadd_ps(wx[3], p[3][c], _mm256_fmadd_ps(wx[2], p[2][c], _mm256_fmadd_ps(wx[0], p[0][c], _mm256_mul_ps(wx[1], p[1][c]))));
    }
  }
  V r[CH];
  for (int c = 0; c < CH; ++c) {
    // col4: fma(w3, r3, fma(w2, r2, fma(w1, r1, w0 * r0)))
    r[c] = _mm256_fmadd_ps(wy[3], rows[c][3], _mm256_fmadd_ps(wy[2], rows[c][2], _mm256_fmadd_ps(wy[1], rows[c][1], _mm256_mul_ps(wy[0], rows[c][0]))));
  }
  convert_f<T, CH>(r, o);
}

template <class T, int CH, int INTERP>
inline void sample8(const Src& s, V sx, V sy, Out<T, CH>& o) {
  if constexpr (INTERP == kNN) sample_nn<T, CH>(s, sx, sy, o);
  else if constexpr (INTERP == kLinear) sample_linear<T, CH>(s, sx, sy, o);
  else sample_cubic<T, CH>(s, sx, sy, o);
}

// The mask of lanes below `n` (0 .. 8) as a float mask.
inline V lane_mask(int n) {
  return _mm256_castsi256_ps(_mm256_cmpgt_epi32(_mm256_set1_epi32(n), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7)));
}

// ---- warp rows ----

// How the source point of a destination pixel is formed, and which points are taken (npp_core.hpp's warp()).
struct WarpBounds {
  float x_lo, x_hi, y_lo, y_hi;
  float roi_x, roi_y;   // the source ROI's origin, for clamp_low
  bool hi_open, clamp_low;
};

enum class MapKind { Affine, Perspective, Table };

// Coefficients for Affine (c[0..5] = c00 c01 c02 c10 c11 c12, as floats) and Perspective (c[0..8], rows of the 3x3), or the two
// tables for Table.
struct MapData {
  float c[9];
  const float* tx;   // table of x source points, row-major with `tw` entries per row, indexed [y][x] with x relative to dx0
  const float* ty;
  int tw;
};

template <MapKind K>
inline void make_point(const MapData& m, int x, int y, V& sx, V& sy, int valid) {
  const V xf = _mm256_cvtepi32_ps(_mm256_add_epi32(_mm256_set1_epi32(x), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7)));
  if constexpr (K == MapKind::Affine) {
    // affine_point: fma(c00, x, c01 * y) + c02, and likewise for y
    const V r0 = _mm256_set1_ps(m.c[1] * static_cast<float>(y)), r1 = _mm256_set1_ps(m.c[4] * static_cast<float>(y));
    sx = _mm256_add_ps(_mm256_fmadd_ps(_mm256_set1_ps(m.c[0]), xf, r0), _mm256_set1_ps(m.c[2]));
    sy = _mm256_add_ps(_mm256_fmadd_ps(_mm256_set1_ps(m.c[3]), xf, r1), _mm256_set1_ps(m.c[5]));
  } else if constexpr (K == MapKind::Perspective) {
    // row(r, x, y) = fma(r0, x, r1 * y) + r2; w = row(c[2]); sx = row(c[0]) / w; sy = row(c[1]) / w
    const float yf = static_cast<float>(y);
    const V w = _mm256_add_ps(_mm256_fmadd_ps(_mm256_set1_ps(m.c[6]), xf, _mm256_set1_ps(m.c[7] * yf)), _mm256_set1_ps(m.c[8]));
    const V nx = _mm256_add_ps(_mm256_fmadd_ps(_mm256_set1_ps(m.c[0]), xf, _mm256_set1_ps(m.c[1] * yf)), _mm256_set1_ps(m.c[2]));
    const V ny = _mm256_add_ps(_mm256_fmadd_ps(_mm256_set1_ps(m.c[3]), xf, _mm256_set1_ps(m.c[4] * yf)), _mm256_set1_ps(m.c[5]));
    sx = _mm256_div_ps(nx, w);
    sy = _mm256_div_ps(ny, w);
  } else {
    (void)xf;
    alignas(32) float bx[8], by[8];
    for (int l = 0; l < 8; ++l) {
      bx[l] = l < valid ? m.tx[l] : 0.0f;
      by[l] = l < valid ? m.ty[l] : 0.0f;
    }
    sx = _mm256_load_ps(bx);
    sy = _mm256_load_ps(by);
    (void)x;
    (void)y;
  }
}

// One destination row of a warp: destination pixels x in [x0, x1) of row y, written where the point is taken.
// drow points at the pixel x0 of the row. tx/ty (Table) point at the entries of pixel x0 of the row.
template <class T, int CH, int INTERP, MapKind K>
inline void warp_row(const Src& s, const WarpBounds& b, MapData m, int y, int x0, int x1, T* drow, const float* tx, const float* ty) {
  const V vxlo = _mm256_set1_ps(b.x_lo), vxhi = _mm256_set1_ps(b.x_hi), vylo = _mm256_set1_ps(b.y_lo), vyhi = _mm256_set1_ps(b.y_hi);
  const V vrx = _mm256_set1_ps(b.roi_x), vry = _mm256_set1_ps(b.roi_y);
  for (int x = x0; x < x1; x += 8, drow += 8 * CH) {
    const int n = x1 - x < 8 ? x1 - x : 8;
    V sx, sy;
    if constexpr (K == MapKind::Table) {
      m.tx = tx + (x - x0);
      m.ty = ty + (x - x0);
    }
    make_point<K>(m, x, y, sx, sy, n);
    const V hx = b.hi_open ? _mm256_cmp_ps(sx, vxhi, _CMP_LT_OQ) : _mm256_cmp_ps(sx, vxhi, _CMP_LE_OQ);
    const V hy = b.hi_open ? _mm256_cmp_ps(sy, vyhi, _CMP_LT_OQ) : _mm256_cmp_ps(sy, vyhi, _CMP_LE_OQ);
    V ok = _mm256_and_ps(_mm256_and_ps(_mm256_cmp_ps(sx, vxlo, _CMP_GE_OQ), _mm256_cmp_ps(sy, vylo, _CMP_GE_OQ)), _mm256_and_ps(hx, hy));
    if (n < 8) ok = _mm256_and_ps(ok, lane_mask(n));
    const int okm = _mm256_movemask_ps(ok);
    if (!okm) continue;
    if (b.clamp_low && INTERP != kNN) {
      // std::max(sx, roi) = (sx < roi) ? roi : sx
      sx = _mm256_blendv_ps(sx, vrx, _mm256_cmp_ps(sx, vrx, _CMP_LT_OQ));
      sy = _mm256_blendv_ps(sy, vry, _mm256_cmp_ps(sy, vry, _CMP_LT_OQ));
    }
    sx = _mm256_and_ps(sx, ok);
    sy = _mm256_and_ps(sy, ok);
    Out<T, CH> o;
    sample8<T, CH, INTERP>(s, sx, sy, o);
    store_lanes<T, CH>(drow, o, okm);
  }
}

// ---- resize rows ----

// One destination row of ResizeSqrPixel: the point of column x is fma(invx, x, cx), of the row sy (given); the columns for which
// `col_ok` is zero are not written.
template <class T, int CH, int INTERP>
inline void resize_row(const Src& s, float invx, float cx, float sy_row, int x0, int x1, T* drow, const uint8_t* col_ok) {
  const V vinv = _mm256_set1_ps(invx), vcx = _mm256_set1_ps(cx), vsy = _mm256_set1_ps(sy_row);
  for (int x = x0; x < x1; x += 8, drow += 8 * CH, col_ok += 8) {
    const int n = x1 - x < 8 ? x1 - x : 8;
    const V xf = _mm256_cvtepi32_ps(_mm256_add_epi32(_mm256_set1_epi32(x), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7)));
    V sx = _mm256_fmadd_ps(vinv, xf, vcx);
    alignas(32) uint32_t okb[8];
    for (int l = 0; l < 8; ++l) okb[l] = (l < n && col_ok[l]) ? 0xFFFFFFFFu : 0u;
    const V ok = _mm256_castsi256_ps(_mm256_load_si256(reinterpret_cast<const VI*>(okb)));
    const int okm = _mm256_movemask_ps(ok);
    if (!okm) continue;
    sx = _mm256_and_ps(sx, ok);
    const V sy = _mm256_and_ps(vsy, ok);
    Out<T, CH> o;
    sample8<T, CH, INTERP>(s, sx, sy, o);
    store_lanes<T, CH>(drow, o, okm);
  }
}

}  // namespace avx2
#pragma GCC pop_options
#endif  // VGPU_NPP_AVX2


// What ResizeSqrPixel's loop derives from its arguments (npp_imgproc.cpp): the float factors that form a sample point, and which
// columns and rows are written at all -- a destination pixel is written when its sample point is within [-0.25, width + 0.5) of
// the source ROI, ux = (x + 0.5 - shift) / factor, in pixel edges, and that separates into a condition on x and one on y.
struct ResizeTables {
  float invx, cx, invy, cy;
  std::vector<uint8_t> colok, rowok;
};
inline ResizeTables resize_tables(const Rect& droi, int src_w, int src_h, double fx, double fy, double shx, double shy) {
  ResizeTables t;
  t.invx = static_cast<float>(1.0 / fx);
  t.invy = static_cast<float>(1.0 / fy);
  t.cx = static_cast<float>(0.5 * t.invx - 0.5 - shx * t.invx);
  t.cy = static_cast<float>(0.5 * t.invy - 0.5 - shy * t.invy);
  t.colok.assign(static_cast<size_t>(droi.w) + 8, 0);
  t.rowok.assign(static_cast<size_t>(droi.h), 0);
  for (int i = 0; i < droi.w; ++i) {
    const double ux = (droi.x + i + 0.5 - shx) / fx;
    t.colok[i] = !(ux < -0.25 || ux >= src_w + 0.5);
  }
  for (int i = 0; i < droi.h; ++i) {
    const double uy = (droi.y + i + 0.5 - shy) / fy;
    t.rowok[i] = !(uy < -0.25 || uy >= src_h + 0.5);
  }
  return t;
}

#ifdef VGPU_NPP_AVX2
#pragma GCC push_options
#pragma GCC target("avx2")
// Without FMA in the target, so that nothing here can be contracted: this is for sums of rounded products.
namespace avx2nf {

// One row of nppiResize_8u's bilinear filter (npp_api.cpp's resize_8u), eight columns at a time. For column x the taps are the
// pixels at the byte offsets c0[x] and c1[x] (the clamped columns, times CH) of the rows at byte offsets ra and rb, weighted
//     v = p00 * (1 - ax) * (1 - ay) + p10 * ax * (1 - ay) + p01 * (1 - ax) * ay + p11 * ax * ay        (float, left to right)
// and the pixel is floor(v) clipped to a byte.
template <int CH>
inline void resize_linear_row(const uint8_t* base, int ra, int rb, float ay, const int* c0, const int* c1, const float* ax, int n, uint8_t* drow) {
  typedef __m256 V;
  typedef __m256i VI;
  const V vay = _mm256_set1_ps(ay), vomay = _mm256_set1_ps(1 - ay);
  const VI vra = _mm256_set1_epi32(ra), vrb = _mm256_set1_epi32(rb);
  for (int x = 0; x < n; x += 8, drow += 8 * CH) {
    const int m = n - x < 8 ? n - x : 8;
    alignas(32) int a0[8], a1[8];
    alignas(32) float fx[8];
    for (int l = 0; l < 8; ++l) {
      const int k = l < m ? x + l : x;   // (lanes past the end repeat the first column: valid offsets, never stored)
      a0[l] = c0[k];
      a1[l] = c1[k];
      fx[l] = ax[k];
    }
    const VI vc0 = _mm256_load_si256(reinterpret_cast<const VI*>(a0)), vc1 = _mm256_load_si256(reinterpret_cast<const VI*>(a1));
    const V vax = _mm256_load_ps(fx);
    const V vomax = _mm256_sub_ps(_mm256_set1_ps(1.0f), vax);
    const VI g00 = _mm256_i32gather_epi32(reinterpret_cast<const int*>(base), _mm256_add_epi32(vra, vc0), 1);
    const VI g10 = _mm256_i32gather_epi32(reinterpret_cast<const int*>(base), _mm256_add_epi32(vra, vc1), 1);
    const VI g01 = _mm256_i32gather_epi32(reinterpret_cast<const int*>(base), _mm256_add_epi32(vrb, vc0), 1);
    const VI g11 = _mm256_i32gather_epi32(reinterpret_cast<const int*>(base), _mm256_add_epi32(vrb, vc1), 1);
    alignas(32) uint8_t res[CH][8];
    for (int c = 0; c < CH; ++c) {
      const VI mask = _mm256_set1_epi32(0xFF);
      const V p00 = _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(g00, 8 * c), mask));
      const V p10 = _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(g10, 8 * c), mask));
      const V p01 = _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(g01, 8 * c), mask));
      const V p11 = _mm256_cvtepi32_ps(_mm256_and_si256(_mm256_srli_epi32(g11, 8 * c), mask));
      const V t0 = _mm256_mul_ps(_mm256_mul_ps(p00, vomax), vomay);
      const V t1 = _mm256_mul_ps(_mm256_mul_ps(p10, vax), vomay);
      const V t2 = _mm256_mul_ps(_mm256_mul_ps(p01, vomax), vay);
      const V t3 = _mm256_mul_ps(_mm256_mul_ps(p11, vax), vay);
      V v = _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(t0, t1), t2), t3);
      v = _mm256_min_ps(_mm256_max_ps(_mm256_floor_ps(v), _mm256_setzero_ps()), _mm256_set1_ps(255.0f));
      const VI ri = _mm256_cvttps_epi32(v);
      const __m128i p16 = _mm_packus_epi32(_mm256_castsi256_si128(ri), _mm256_extracti128_si256(ri, 1));
      _mm_storel_epi64(reinterpret_cast<__m128i*>(res[c]), _mm_packus_epi16(p16, p16));
    }
    for (int l = 0; l < m; ++l)
      for (int c = 0; c < CH; ++c) drow[l * CH + c] = res[c][l];
  }
}

}  // namespace avx2nf
#pragma GCC pop_options
#endif  // VGPU_NPP_AVX2

// ---- the drivers: the same rows as npp_core.hpp's warp() and npp_imgproc.cpp's resize_sqr, on the threads they use ----

#ifdef VGPU_NPP_AVX2
namespace detail {

// Calls f(std::integral_constant<int, CH>{}, std::integral_constant<int, INTERP>{}) for the channel count and interpolation.
template <class F>
inline bool dispatch(int ch, int interp, F&& f) {
  auto by_interp = [&](auto chc) {
    switch (interp) {
      case kNN: f(chc, std::integral_constant<int, kNN>{}); return true;
      case kLinear: f(chc, std::integral_constant<int, kLinear>{}); return true;
      case kCubic: f(chc, std::integral_constant<int, kCubic>{}); return true;
      default: return false;
    }
  };
  switch (ch) {
    case 1: return by_interp(std::integral_constant<int, 1>{});
    case 3: return by_interp(std::integral_constant<int, 3>{});
    case 4: return by_interp(std::integral_constant<int, 4>{});
    default: return false;
  }
}

// An image is in the fast path's range when the gathers' 32-bit element offsets cannot overflow.
template <class T>
inline bool offsets_fit(const Image<T>& s) {
  return static_cast<int64_t>(s.h) * s.stride + 64 < (int64_t(1) << 28) / 1;
}

inline avx2::WarpBounds bounds_of(const Rect& sroi, const Bounds& bounds) {
  avx2::WarpBounds b;
  b.x_lo = sroi.x - bounds.lo_slack;
  b.x_hi = sroi.x + sroi.w - 1 + bounds.hi_slack;
  b.y_lo = sroi.y - bounds.lo_slack;
  b.y_hi = sroi.y + sroi.h - 1 + bounds.hi_slack;
  b.roi_x = static_cast<float>(sroi.x);
  b.roi_y = static_cast<float>(sroi.y);
  b.hi_open = bounds.hi_open;
  b.clamp_low = bounds.clamp_low;
  return b;
}

template <avx2::MapKind K, class T>
inline bool warp_with(const Image<T>& src, const Rect& sroi, Image<T>& dst, const Rect& droi, int interp, const avx2::MapData& md,
                      const Bounds& bounds, const float* tx, int tstride, const float* ty) {
  if (!avail<T>(src.ch) || src.ch != dst.ch || !offsets_fit(src)) return false;
  const avx2::WarpBounds b = bounds_of(sroi, bounds);
  const avx2::Src s{src.p, src.w, src.h, src.stride};
  // (The same estimate as warp(): every destination pixel is its own, and rows go to as many threads as the work warrants.)
  const unsigned threads = vgpu::hostpar::plan(static_cast<uint64_t>(droi.w) * static_cast<uint64_t>(droi.h) * 60u, droi.h);
  return dispatch(src.ch, interp, [&](auto chc, auto ic) {
    constexpr int CH = decltype(chc)::value, INTERP = decltype(ic)::value;
    vgpu::hostpar::parallel_for(threads, droi.h, [&](int64_t row) {
      const int y = droi.y + static_cast<int>(row);
      T* drow = dst.p + static_cast<size_t>(y - dst.oy) * dst.stride + static_cast<size_t>(droi.x - dst.ox) * CH;
      const float* rtx = tx ? tx + static_cast<size_t>(row) * tstride : nullptr;
      const float* rty = ty ? ty + static_cast<size_t>(row) * tstride : nullptr;
      avx2::warp_row<T, CH, INTERP, K>(s, b, md, y, droi.x, droi.x + droi.w, drow, rtx, rty);
    });
  });
}

}  // namespace detail

// WarpAffineBack's map: the 2x3 matrix `c` takes destination points to source points (npp_core.hpp's affine_point).
template <class T>
inline bool warp_affine_back(const Image<T>& src, const Rect& sroi, Image<T>& dst, const Rect& droi, const double c[2][3], int interp,
                             const Bounds& bounds) {
  avx2::MapData md{};
  md.c[0] = static_cast<float>(c[0][0]); md.c[1] = static_cast<float>(c[0][1]); md.c[2] = static_cast<float>(c[0][2]);
  md.c[3] = static_cast<float>(c[1][0]); md.c[4] = static_cast<float>(c[1][1]); md.c[5] = static_cast<float>(c[1][2]);
  return detail::warp_with<avx2::MapKind::Affine, T>(src, sroi, dst, droi, interp, md, bounds, nullptr, 0, nullptr);
}

// The perspective map's source points (warp_perspective_back).
template <class T>
inline bool warp_perspective_back(const Image<T>& src, const Rect& sroi, Image<T>& dst, const Rect& droi, const double c[3][3], int interp) {
  avx2::MapData md{};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) md.c[i * 3 + j] = static_cast<float>(c[i][j]);
  return detail::warp_with<avx2::MapKind::Perspective, T>(src, sroi, dst, droi, interp, md, Bounds(), nullptr, 0, nullptr);
}

// Remap's source points, read from two float tables laid out like the destination (`tstride` entries a row, row 0 being the
// destination ROI's first row, entry 0 its first column).
template <class T>
inline bool warp_table(const Image<T>& src, const Rect& sroi, Image<T>& dst, const Rect& droi, const float* tx, const float* ty,
                       int tstride, int interp, const Bounds& bounds) {
  avx2::MapData md{};
  return detail::warp_with<avx2::MapKind::Table, T>(src, sroi, dst, droi, interp, md, bounds, tx, tstride, ty);
}

// ResizeSqrPixel for nearest, linear and cubic: dst(x, y), x in [droi.x, droi.x + droi.w) and y likewise, from the source point
// (fma(invx, x, cx), fma(invy, y, cy)); a column or row for which colok/rowok is zero is not written. `src` is the window the
// taps clamp to.
template <class T>
inline bool resize_sqr(const Image<T>& src, Image<T>& dst, const Rect& droi, float invx, float cx, float invy, float cy,
                       const uint8_t* colok, const uint8_t* rowok, int interp) {
  if (!avail<T>(src.ch) || src.ch != dst.ch || !detail::offsets_fit(src)) return false;
  const avx2::Src s{src.p, src.w, src.h, src.stride};
  const unsigned threads = vgpu::hostpar::plan(static_cast<uint64_t>(droi.w) * static_cast<uint64_t>(droi.h) * 80u, droi.h);
  return detail::dispatch(src.ch, interp, [&](auto chc, auto ic) {
    constexpr int CH = decltype(chc)::value, INTERP = decltype(ic)::value;
    vgpu::hostpar::parallel_for(threads, droi.h, [&](int64_t row) {
      if (!rowok[row]) return;
      const int y = droi.y + static_cast<int>(row);
      const float sy = std::fma(invy, static_cast<float>(y), cy);
      T* drow = dst.p + static_cast<size_t>(y - dst.oy) * dst.stride + static_cast<size_t>(droi.x - dst.ox) * CH;
      avx2::resize_row<T, CH, INTERP>(s, invx, cx, sy, droi.x, droi.x + droi.w, drow, colok);
    });
  });
}

// nppiResize_8u_C1R / C3R (npp_api.cpp's resize_8u): the destination rectangle `droi` of `dst` (whole destination image, `dstride`
// bytes a row) from the source rectangle `sroi` of `src` (the whole source image, `sstride` bytes a row, followed by kPad
// readable bytes), nearest or bilinear. The sample point of destination column x is (x + 0.5) * fx + sroi.x - 0.5 in double with
// fx = sroi.w / droi.w, and taps clamp to the source rectangle. Returns false (nothing written) where it does not apply.
inline bool resize_legacy_u8(int ch, const uint8_t* src, int sstride, const Rect& sroi, uint8_t* dst, int dstride, const Rect& droi, bool linear) {
  if (ch != 1 && ch != 3) return false;
  if (linear && !cpu_ok()) return false;
  if (static_cast<int64_t>(sroi.y + sroi.h) * sstride + 64 >= (int64_t(1) << 30)) return false;
  const double fx = static_cast<double>(sroi.w) / droi.w, fy = static_cast<double>(sroi.h) / droi.h;
  const int w = droi.w, h = droi.h;
  auto clampc = [&](int x) { return x < sroi.x ? sroi.x : (x > sroi.x + sroi.w - 1 ? sroi.x + sroi.w - 1 : x); };
  auto clampr = [&](int y) { return y < sroi.y ? sroi.y : (y > sroi.y + sroi.h - 1 ? sroi.y + sroi.h - 1 : y); };
  std::vector<int> c0(static_cast<size_t>(w)), c1(static_cast<size_t>(w));
  std::vector<float> ax(static_cast<size_t>(w));
  for (int x = 0; x < w; ++x) {
    const double sx = (x + 0.5) * fx + sroi.x - 0.5;
    if (!linear) {
      c0[x] = clampc(static_cast<int>(std::nearbyint(sx))) * ch;
    } else {
      const int x0 = static_cast<int>(std::floor(sx));
      ax[x] = static_cast<float>(sx - x0);
      c0[x] = clampc(x0) * ch;
      c1[x] = clampc(x0 + 1) * ch;
    }
  }
  const unsigned threads = vgpu::hostpar::plan(static_cast<uint64_t>(w) * static_cast<uint64_t>(h) * (linear ? 30u : 6u), h);
  vgpu::hostpar::parallel_for(threads, h, [&](int64_t row) {
    const int y = static_cast<int>(row);
    const double sy = (y + 0.5) * fy + sroi.y - 0.5;
    uint8_t* drow = dst + static_cast<size_t>(y + droi.y) * dstride + static_cast<size_t>(droi.x) * ch;
    if (!linear) {
      const uint8_t* srow = src + static_cast<size_t>(clampr(static_cast<int>(std::nearbyint(sy)))) * sstride;
      if (ch == 1) {
        for (int x = 0; x < w; ++x) drow[x] = srow[c0[x]];
      } else {
        for (int x = 0; x < w; ++x) {
          const uint8_t* p = srow + c0[x];
          drow[3 * x] = p[0];
          drow[3 * x + 1] = p[1];
          drow[3 * x + 2] = p[2];
        }
      }
      return;
    }
#ifdef VGPU_NPP_AVX2
    const int y0 = static_cast<int>(std::floor(sy));
    const float ay = static_cast<float>(sy - y0);
    const int ra = clampr(y0) * sstride, rb = clampr(y0 + 1) * sstride;
    if (ch == 1) avx2nf::resize_linear_row<1>(src, ra, rb, ay, c0.data(), c1.data(), ax.data(), w, drow);
    else avx2nf::resize_linear_row<3>(src, ra, rb, ay, c0.data(), c1.data(), ax.data(), w, drow);
#endif
  });
  return true;
}
#else   // no AVX2: the generic path does everything
inline bool resize_legacy_u8(int, const uint8_t*, int, const Rect&, uint8_t*, int, const Rect&, bool) { return false; }
template <class T> inline bool warp_affine_back(const Image<T>&, const Rect&, Image<T>&, const Rect&, const double[2][3], int, const Bounds&) { return false; }
template <class T> inline bool warp_perspective_back(const Image<T>&, const Rect&, Image<T>&, const Rect&, const double[3][3], int) { return false; }
template <class T> inline bool warp_table(const Image<T>&, const Rect&, Image<T>&, const Rect&, const float*, const float*, int, int, const Bounds&) { return false; }
template <class T> inline bool resize_sqr(const Image<T>&, Image<T>&, const Rect&, float, float, float, float, const uint8_t*, const uint8_t*, int) { return false; }
#endif  // VGPU_NPP_AVX2

}  // namespace vgpu_npp::simd
