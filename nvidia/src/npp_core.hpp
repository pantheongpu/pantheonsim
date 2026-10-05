// Host-side cores of VirtualGPU's NPP image-processing entry points.
//
// Pure C++ with no CUDA in it: the exported functions in npp_imgproc.cpp fetch
// the pixels from device memory, call these on host copies, and store the
// result back. Keeping the arithmetic here, apart from the transfers, is what
// lets the same code be compared with NVIDIA's NPP on a card pixel by pixel.
//
// Every convention below that the NPP documentation leaves open was measured
// on an RTX 3060 against NVIDIA's NPP 13.0; the comment at each one says what
// was observed.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

namespace vgpu_npp {

// An interleaved host image: `stride` elements per row, `ch` per pixel. A
// window onto a larger image sets (ox, oy) to the image coordinates of its
// first pixel, so callers index it in image coordinates.
template <class T>
struct Image {
  T* p = nullptr;
  int w = 0, h = 0, stride = 0, ch = 1, ox = 0, oy = 0;
  T& at(int x, int y, int c) {
    return p[static_cast<size_t>(y - oy) * stride + static_cast<size_t>(x - ox) * ch + c];
  }
  const T& at(int x, int y, int c) const {
    return p[static_cast<size_t>(y - oy) * stride + static_cast<size_t>(x - ox) * ch + c];
  }
};

struct Rect {
  int x, y, w, h;
};

// Conversion of a computed value to a pixel. Integer pixels round half away
// from zero and saturate: measured on WarpAffine/Remap, where a
// bilinear value of exactly 4.5 comes back 5 and 7.5 comes back 8 (round to
// even would give 4 and 8).
template <class T>
inline T to_pixel(double v) {
  if constexpr (std::is_floating_point_v<T>) {
    return static_cast<T>(v);
  } else {
    if (std::isnan(v)) return 0;
    // Halves go away from zero (a signed 16-bit remap gives -9037 for
    // -9036.5); for unsigned pixels that is floor(v + 0.5).
    const double r = v < 0 ? std::ceil(v - 0.5) : std::floor(v + 0.5);
    constexpr double lo = static_cast<double>(std::numeric_limits<T>::min());
    constexpr double hi = static_cast<double>(std::numeric_limits<T>::max());
    return static_cast<T>(r < lo ? lo : (r > hi ? hi : r));
  }
}

template <class T>
inline T saturate(double v) {
  if constexpr (std::is_floating_point_v<T>) {
    return static_cast<T>(v);
  } else {
    constexpr double lo = static_cast<double>(std::numeric_limits<T>::min());
    constexpr double hi = static_cast<double>(std::numeric_limits<T>::max());
    return static_cast<T>(v < lo ? lo : (v > hi ? hi : v));
  }
}

/* ---- geometric sampling ----

   Measured on an RTX 3060 (WarpAffineBack with an 8x6 ramp, a single-pixel
   impulse, then random images of every depth):
   - A destination pixel is written only when its source point (sx, sy) lies
     inside the source ROI, edges included: roi.x <= sx <= roi.x + roi.w - 1.
     Outside, the destination keeps what it held.
   - Nearest neighbour takes floor(s + 0.5): 0.5 goes to 1, 0.49 to 0.
   - Linear is bilinear over the four neighbours, the right/bottom neighbour
     clamped to the ROI at its last column/row.
   - Cubic is four-point Lagrange interpolation (the cubic through the four
     nearest samples) in each direction -- an impulse sampled a quarter pixel
     away gives -0.0547, 0.8203, 0.2734, -0.0391 -- with out-of-ROI taps
     repeating the edge pixel.
   - The arithmetic is single precision: 32-bit integer images come back
     with float-sized rounding errors in the low bits. */
enum Interp { kNN = 1, kLinear = 2, kCubic = 4 };

inline void lagrange4(float f, float w[4]) {
  w[0] = -f * (f - 1) * (f - 2) / 6;
  w[1] = (f + 1) * (f - 1) * (f - 2) / 2;
  w[2] = -(f + 1) * f * (f - 2) / 2;
  w[3] = (f + 1) * f * (f - 1) / 6;
}

// Samples every channel of `src` at (sx, sy). Taps outside the image repeat
// its edge pixels; whether a point is inside the ROI at all is the caller's
// question (see warp() and remap()), because NPP answers it differently per
// function.
template <class T>
inline void sample(const Image<T>& src, float sx, float sy, int interp, double* out) {
  auto ix_img = [&](int x) { return std::min(std::max(x, 0), src.w - 1); };
  auto iy_img = [&](int y) { return std::min(std::max(y, 0), src.h - 1); };
  if (interp == kNN) {
    const int ix = ix_img(static_cast<int>(std::floor(sx + 0.5f)));
    const int iy = iy_img(static_cast<int>(std::floor(sy + 0.5f)));
    for (int c = 0; c < src.ch; ++c) out[c] = static_cast<double>(src.at(ix, iy, c));
    return;
  }
  const int x0 = static_cast<int>(std::floor(sx)), y0 = static_cast<int>(std::floor(sy));
  const float fx = sx - x0, fy = sy - y0;
  if (interp == kLinear) {
    const int x1 = ix_img(x0 + 1), y1 = iy_img(y0 + 1);
    for (int c = 0; c < src.ch; ++c) {
      // Fused exactly like this: any other grouping is off by one count on a
      // few pixels of a 16-bit image and by tens on a 32-bit one.
      const float top = std::fma(static_cast<float>(src.at(x0, y0, c)), 1 - fx,
                                 static_cast<float>(src.at(x1, y0, c)) * fx);
      const float bot = std::fma(static_cast<float>(src.at(x0, y1, c)), 1 - fx,
                                 static_cast<float>(src.at(x1, y1, c)) * fx);
      out[c] = std::fma(top, 1 - fy, bot * fy);
    }
    return;
  }
  float wx[4], wy[4];
  int xs[4], ys[4];
  lagrange4(fx, wx);
  lagrange4(fy, wy);
  for (int i = 0; i < 4; ++i) {
    xs[i] = ix_img(x0 + i - 1);
    ys[i] = iy_img(y0 + i - 1);
  }
  // Each row first, then the column of row results, each a fused chain from
  // the first tap: the order that reproduces NVIDIA's 16-bit results exactly.
  // (32-bit integer images still differ from it by one float ulp on about one
  // pixel in ten; no grouping tried removes that.)
  auto dot4 = [](const float w[4], const float p[4]) {
    return std::fma(w[3], p[3], std::fma(w[2], p[2], std::fma(w[1], p[1], w[0] * p[0])));
  };
  for (int c = 0; c < src.ch; ++c) {
    float rows[4], p[4];
    for (int j = 0; j < 4; ++j) {
      for (int i = 0; i < 4; ++i) p[i] = static_cast<float>(src.at(xs[i], ys[j], c));
      rows[j] = dot4(wx, p);
    }
    out[c] = dot4(wy, rows);
  }
}

// Which source points a function accepts, relative to the source ROI.
//
// The warps (WarpAffine, WarpPerspective and their Back forms) take a point
// only when roi.x <= sx <= roi.x + roi.w - 1, edges included, and likewise in
// y: an 8x6 image shifted by half a pixel loses its last column (measured).
// Rotate and Remap also take points up to half a pixel before the ROI's first
// column and row, and sample them as if they were on it. Remap's far edge is a
// whole pixel later and open: it takes 32.99 for an ROI whose last column is
// 32 but not 33.0, and its nearest-neighbour sample there reads column 33 of
// the image.
struct Bounds {
  float lo_slack = 0;     // accepted distance before the first column/row
  float hi_slack = 0;     // ... after the last
  bool hi_open = false;   // the far bound itself is excluded
  bool clamp_low = false; // points before the ROI are moved onto it
};

// Writes every destination pixel in `droi` whose mapped source point `bounds`
// accepts. map(x, y, &sx, &sy) gives the source point of destination (x, y)
// in absolute image coordinates and returns false to skip the pixel.
template <class T, class Map>
inline void warp(const Image<T>& src, const Rect& sroi, Image<T>& dst, const Rect& droi, int interp,
                 Map map, const Bounds& bounds = Bounds()) {
  double v[4];
  const float x_lo = sroi.x - bounds.lo_slack, x_hi = sroi.x + sroi.w - 1 + bounds.hi_slack;
  const float y_lo = sroi.y - bounds.lo_slack, y_hi = sroi.y + sroi.h - 1 + bounds.hi_slack;
  for (int y = droi.y; y < droi.y + droi.h; ++y)
    for (int x = droi.x; x < droi.x + droi.w; ++x) {
      float sx, sy;
      if (!map(x, y, &sx, &sy)) continue;
      if (!(sx >= x_lo && sy >= y_lo)) continue;
      if (bounds.hi_open ? !(sx < x_hi && sy < y_hi) : !(sx <= x_hi && sy <= y_hi)) continue;
      if (bounds.clamp_low && interp != kNN) {
        sx = std::max(sx, static_cast<float>(sroi.x));
        sy = std::max(sy, static_cast<float>(sroi.y));
      }
      sample(src, sx, sy, interp, v);
      for (int c = 0; c < dst.ch; ++c) dst.at(x, y, c) = to_pixel<T>(v[c]);
    }
}

// The inverse of a 2x3 affine matrix (forward warps map destination pixels
// back through it). Returns false for a singular matrix.
inline bool invert_affine(const double c[2][3], double inv[2][3]) {
  const double det = c[0][0] * c[1][1] - c[0][1] * c[1][0];
  if (det == 0) return false;
  inv[0][0] = c[1][1] / det;
  inv[0][1] = -c[0][1] / det;
  inv[1][0] = -c[1][0] / det;
  inv[1][1] = c[0][0] / det;
  inv[0][2] = -(inv[0][0] * c[0][2] + inv[0][1] * c[1][2]);
  inv[1][2] = -(inv[1][0] * c[0][2] + inv[1][1] * c[1][2]);
  return true;
}

inline bool invert_3x3(const double m[3][3], double r[3][3]) {
  const double det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
                     m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
                     m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
  if (det == 0) return false;
  r[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) / det;
  r[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) / det;
  r[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) / det;
  r[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) / det;
  r[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) / det;
  r[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) / det;
  r[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) / det;
  r[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) / det;
  r[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) / det;
  return true;
}

// The source point of an affine map, in single precision and fused the way
// NPP's is: fma(c00, x, c01 * y) + c02. Computing it in double, or fused in
// any other order, puts a handful of nearest-neighbour samples on the other
// side of a .5 and so on a different pixel.
inline void affine_point(const double c[2][3], int x, int y, float* sx, float* sy) {
  *sx = std::fma(static_cast<float>(c[0][0]), static_cast<float>(x), static_cast<float>(c[0][1]) * y) +
        static_cast<float>(c[0][2]);
  *sy = std::fma(static_cast<float>(c[1][0]), static_cast<float>(x), static_cast<float>(c[1][1]) * y) +
        static_cast<float>(c[1][2]);
}

template <class T>
inline void warp_affine_back(const Image<T>& src, const Rect& sroi, Image<T>& dst, const Rect& droi,
                             const double c[2][3], int interp) {
  warp(src, sroi, dst, droi, interp, [&](int x, int y, float* sx, float* sy) {
    affine_point(c, x, y, sx, sy);
    return true;
  });
}

template <class T>
inline void warp_perspective_back(const Image<T>& src, const Rect& sroi, Image<T>& dst,
                                  const Rect& droi, const double c[3][3], int interp) {
  // Each row of the matrix the same way as affine_point, then one float
  // division (a reciprocal-multiply, or double, is a count out on some
  // 16-bit pixels).
  auto row = [](const double* r, int x, int y) {
    return std::fma(static_cast<float>(r[0]), static_cast<float>(x), static_cast<float>(r[1]) * y) +
           static_cast<float>(r[2]);
  };
  warp(src, sroi, dst, droi, interp, [&](int x, int y, float* sx, float* sy) {
    const float w = row(c[2], x, y);
    if (w == 0) return false;
    *sx = row(c[0], x, y) / w;
    *sy = row(c[1], x, y) / w;
    return true;
  });
}

}  // namespace vgpu_npp
