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
#include <utility>
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

// The four weights NPP's cubic filter gives the taps at x-1, x, x+1, x+2 for a
// fractional position f. They are the Lagrange weights, and they are rounded
// the way NPP's are: the outer two are the products f(1-f)(2-f)/6 and
// (f+1)f(1-f)/6 grouped as below, the inner two the plain products over 2.
// Fitted bit for bit against NPP 13.0 on an RTX 3060: the impulse response of
// nppiResizeSqrPixel_32f_C1R_Ctx at factors 3, 5, 6, 7, 10, 1.7 and 2.3 (an
// exact match on 559 samples) fixes each weight to the last bit, and every
// other grouping of the factors tried differs by up to an ulp on some phase.
inline void lagrange4(float f, float w[4]) {
  w[0] = -((f * (1 - f)) * ((2 - f) / 6));
  w[1] = (f + 1) * (f - 1) * (f - 2) / 2;
  w[2] = -(f + 1) * f * (f - 2) / 2;
  w[3] = -((((f + 1) / 6) * f) * (1 - f));
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
    // Both taps repeat the edge pixel where the point is past it.
    const int xa = ix_img(x0), ya = iy_img(y0);
    const int x1 = ix_img(x0 + 1), y1 = iy_img(y0 + 1);
    for (int c = 0; c < src.ch; ++c) {
      // Fused exactly like this: any other grouping is off by one count on a
      // few pixels of a 16-bit image and by tens on a 32-bit one.
      const float top = std::fma(static_cast<float>(src.at(xa, ya, c)), 1 - fx,
                                 static_cast<float>(src.at(x1, ya, c)) * fx);
      const float bot = std::fma(static_cast<float>(src.at(xa, y1, c)), 1 - fx,
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
  // Each row first, then the column of row results. A row is a fused chain
  // that starts at the second tap -- w1*p1, then w0, w2, w3 -- and the column
  // one that starts at the first -- w0*r0, then w1, w2, w3: the orders that
  // reproduce NVIDIA's float output bit for bit (nppiResizeSqrPixel_32f_C1R
  // on random data, interior pixels: all of them; the 8- and 16-bit warps and
  // resizes round it anyway). Exhaustive search over every order and fusing
  // of four terms found these two and no other.
  auto row4 = [](const float w[4], const float p[4]) {
    return std::fma(w[3], p[3], std::fma(w[2], p[2], std::fma(w[0], p[0], w[1] * p[1])));
  };
  auto col4 = [](const float w[4], const float p[4]) {
    return std::fma(w[3], p[3], std::fma(w[2], p[2], std::fma(w[1], p[1], w[0] * p[0])));
  };
  for (int c = 0; c < src.ch; ++c) {
    float rows[4], p[4];
    for (int j = 0; j < 4; ++j) {
      for (int i = 0; i < 4; ++i) p[i] = static_cast<float>(src.at(xs[i], ys[j], c));
      rows[j] = row4(wx, p);
    }
    out[c] = col4(wy, rows);
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


/* ---- watershed segmentation ----

   nppiSegmentWatershed has no published algorithm, so this is what NPP 13.0
   was measured to do on an RTX 3060 (nvidia/tests/e2e/npp_segment.cpp pins it,
   and the card's output is its expected file).

   Every pixel flows to its lowest neighbour -- the 8 neighbours for
   nppiNormInf, the 4 for nppiNormL1 -- when that neighbour is strictly lower,
   taking the first of equal ones in raster order; a pixel with no strictly
   lower neighbour is a root. The image comes back with each pixel replaced by
   its root's value (measured exactly, on random images of 2x2 to 512x512 with
   many equal values and on the CUDA Samples' teapot, skull and rocks images).
   The pixel's marker label is `neighbourhood_min` of its root: the smallest
   linear index in the root's closed neighbourhood -- the pixel above and to
   the left of it for 8-way connectivity (observed on all distinct-valued
   images, where the labels are exactly that).

   Where values are equal the labels follow rules that were fitted rather
   than documented, and they are not exact:
   - roots that are 8-neighbours and equal share one label (the smallest);
   - a pixel whose lowest neighbours are tied makes the later roots among them
     share the first one's label if it is a root too, and otherwise take its
     pixel index as a candidate label (found on teapot-image events, where it
     reproduces NVIDIA's label for a root);
   - NPP leaves some plateau groups unmerged and treats a plateau at the
     image's first pixel differently (its label is one higher): not
     reproduced. On the three CUDA Samples images 98.5% to 99.5% of the pixels
     carry NVIDIA's label (the images themselves are exact); on images
     of pure noise, every one.

   4-way connectivity (nppiNormL1) has a defect of NPP 13.0 that this reproduces:
   a pixel whose lowest neighbour is the one to its right (below) is not written, and so stays a
   root with its own value, when its column (row) is in a set that depends on the image's width (height)
   alone: with t = width - 1 - x, t is in the set for width % 112 of kUnwritten4 (a bit per t from 1 to
   11). The set came from east-flowing ramps of every width from 2 to 1099 (it repeats every 112 from
   a width of 12; below that it is the same set cut at t <= width - 1) and is symmetric in x and y.
   Random images of 8 to 300 pixels a side, with and without equal values, 8-bit and 16-bit, have the
   segmented image of the model with those pixels as roots exactly as NPP writes it, and the marker label of
   such a pixel is its own pixel index (a root's is the smallest index in its closed 4-neighbourhood).

   Boundaries are drawn on the segmented image: a pixel is a boundary pixel
   when the pixel above it or the one to its left has another segmented
   value (measured exactly, with ties). */
// The columns (rows) in which NPP 13.0's 4-way watershed leaves a pixel that flows right (down) unwritten, by
// the width (height) % 112: bit t, for t = width - 1 - x from 1 to 11, says whether column x is one.
inline constexpr uint16_t kUnwritten4[112] = {
    62, 126, 124, 248, 240, 482, 450, 902, 774, 1550, 1038, 2078, 30, 62,
    62, 126, 124, 248, 240, 480, 448, 898, 770, 1542, 1030, 2062, 14, 30,
    30, 62, 60, 120, 112, 226, 192, 384, 256, 514, 2, 6, 6, 14,
    14, 30, 28, 56, 48, 98, 66, 134, 4, 8, 0, 2, 2, 6,
    6, 14, 12, 24, 16, 34, 2, 6, 6, 14, 12, 24, 16, 34,
    2, 6, 4, 8, 0, 2, 2, 6, 6, 14, 14, 30, 28, 56,
    48, 98, 64, 128, 0, 2, 2, 6, 6, 14, 14, 30, 30, 62,
    60, 120, 112, 224, 192, 386, 258, 518, 6, 14, 14, 30, 30, 62};
inline bool ws_unwritten4(int n, int k) {
  const int t = n - 1 - k;
  return t >= 1 && t <= 11 && ((kUnwritten4[n % 112] >> t) & 1) != 0;
}

enum class WsNorm { k8, k4 };

struct Watershed {
  std::vector<int32_t> root;      // the pixel each one flows to
  std::vector<uint32_t> label;
  std::vector<uint8_t> is_root;
};

inline int ws_neighbours(int i, int w, int h, bool four, int out[8]) {
  const int y = i / w, x = i % w;
  int n = 0;
  for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx) {
      if (!dy && !dx) continue;
      if (four && dy && dx) continue;
      const int yy = y + dy, xx = x + dx;
      if (yy < 0 || yy >= h || xx < 0 || xx >= w) continue;
      out[n++] = yy * w + xx;
    }
  return n;
}

// The smallest linear index in the closed neighbourhood of pixel i.
inline int ws_nbhd_min(int i, int w, int h, bool four) {
  int nb[8];
  int m = i;
  const int n = ws_neighbours(i, w, h, four, nb);
  for (int k = 0; k < n; ++k) m = std::min(m, nb[k]);
  return m;
}

template <class T>
inline Watershed watershed(const T* v, int w, int h, bool four) {
  const int n = w * h;
  Watershed r;
  r.root.assign(n, -1);
  r.label.assign(n, 0);
  r.is_root.assign(n, 0);
  std::vector<int32_t> ptr(n, -1);
  std::vector<uint8_t> unwritten(n, 0);   // 4-way: a pixel NPP leaves as it is, a root with its own index
  for (int i = 0; i < n; ++i) {
    int nb[8];
    const int k = ws_neighbours(i, w, h, four, nb);
    int best = -1;
    for (int j = 0; j < k; ++j)
      if (v[nb[j]] < v[i] && (best < 0 || v[nb[j]] < v[best])) best = nb[j];
    if (four && best >= 0) {
      const int x = i % w, y = i / w;
      if ((best == i + 1 && x + 1 < w && ws_unwritten4(w, x)) || (best == i + w && ws_unwritten4(h, y))) {
        best = -1;
        unwritten[i] = 1;
      }
    }
    ptr[i] = best;
    r.is_root[i] = best < 0;
  }
  // Pointer chains run to strictly smaller values, so they end; resolve them
  // with path compression.
  std::vector<int32_t> stack;
  for (int i = 0; i < n; ++i) {
    if (r.root[i] >= 0) continue;
    int a = i;
    stack.clear();
    while (r.root[a] < 0 && ptr[a] >= 0) {
      stack.push_back(a);
      a = ptr[a];
    }
    const int rt = r.root[a] >= 0 ? r.root[a] : a;
    r.root[a] = rt;
    for (int b : stack) r.root[b] = rt;
  }
  // Union-find over the roots, smallest index as the representative.
  std::vector<int32_t> par(n);
  for (int i = 0; i < n; ++i) par[i] = i;
  auto find = [&](int a) {
    while (par[a] != a) {
      par[a] = par[par[a]];
      a = par[a];
    }
    return a;
  };
  auto unite = [&](int a, int b) {
    a = find(a);
    b = find(b);
    if (a != b) par[std::max(a, b)] = std::min(a, b);
  };
  std::vector<std::pair<int32_t, int32_t>> pulls;  // (root, pixel index candidate)
  for (int i = 0; i < n; ++i) {
    int nb[8];
    const int k = ws_neighbours(i, w, h, four, nb);
    if (r.is_root[i]) {
      if (unwritten[i]) continue;
      for (int j = 0; j < k; ++j)
        if (nb[j] > i && r.is_root[nb[j]] && !unwritten[nb[j]] && v[nb[j]] == v[i]) unite(i, nb[j]);
      continue;
    }
    int low[8], nl = 0;
    for (int j = 0; j < k; ++j)
      if (v[nb[j]] == v[ptr[i]]) low[nl++] = nb[j];
    for (int j = 1; j < nl; ++j) {
      if (!r.is_root[low[j]] || unwritten[low[j]]) continue;
      if (r.is_root[low[0]] && !unwritten[low[0]])
        unite(low[0], low[j]);
      else
        pulls.emplace_back(low[j], low[0]);
    }
  }
  std::vector<uint32_t> group(n, std::numeric_limits<uint32_t>::max());
  for (int i = 0; i < n; ++i)
    if (r.is_root[i]) {
      const int g = find(i);
      group[g] = std::min<uint32_t>(group[g], unwritten[i] ? static_cast<uint32_t>(i)
                                                          : static_cast<uint32_t>(ws_nbhd_min(i, w, h, four)));
    }
  for (const auto& p : pulls) {
    const int g = find(p.first);
    group[g] = std::min<uint32_t>(group[g], static_cast<uint32_t>(p.second));
  }
  for (int i = 0; i < n; ++i) r.label[i] = group[find(r.root[i])];
  return r;
}

// Marker labels renumbered: the labels below `limit` (a label is a pixel index
// of the image the markers were generated for) take their rank among those
// present plus 0, which is always counted -- so 0 stays 0 and an image with
// no 0 starts at 1; labels from `limit` up stay. Returns the number of
// distinct labels counted, 0 included. Measured on 300 random label images.
inline int compress_labels(uint32_t* labels, size_t count, long long limit) {
  std::vector<uint32_t> used;
  used.push_back(0);
  for (size_t i = 0; i < count; ++i)
    if (static_cast<long long>(labels[i]) < limit) used.push_back(labels[i]);
  std::sort(used.begin(), used.end());
  used.erase(std::unique(used.begin(), used.end()), used.end());
  for (size_t i = 0; i < count; ++i)
    if (static_cast<long long>(labels[i]) < limit)
      labels[i] = static_cast<uint32_t>(std::lower_bound(used.begin(), used.end(), labels[i]) - used.begin());
  return static_cast<int>(used.size());
}

}  // namespace vgpu_npp
