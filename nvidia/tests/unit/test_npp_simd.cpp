// The AVX2 samplers behind NPP's warps, rotation, remapping and ResizeSqrPixel (src/npp_simd.hpp), pixel for pixel against the
// scalar code they stand in for (src/npp_core.hpp's warp() and sample(), and the resize loop of npp_imgproc.cpp, kept here as
// the reference). Every pixel type (8-bit, 16-bit, float), channel count (1, 3, 4) and interpolation (nearest, linear, cubic)
// is run over many sizes (widths that are and are not a multiple of eight, windows with an origin, destinations that hold
// something already, which must stay where a pixel is not written), maps (affine and perspective, mild and extreme, the
// boundary rules of the warps, of Rotate and of Remap) and sources (random, extreme values, and for float images infinities,
// NaNs, signed zeros and subnormals). Two NaNs count as equal (which NaN comes out of an operation on two is the compiler's
// choice of operand order).
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../../src/npp_simd.hpp"
#include "vtest.hpp"

namespace {

using namespace vgpu_npp;

struct Rng {
  uint64_t s = 0x9E3779B97F4A7C15ull;
  uint64_t next() {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return s;
  }
  double unit() { return static_cast<double>(next() >> 11) * (1.0 / 9007199254740992.0); }       // [0, 1)
  double range(double a, double b) { return a + (b - a) * unit(); }
  int below(int n) { return static_cast<int>(next() % static_cast<uint64_t>(n)); }
  bool coin(unsigned n) { return next() % n == 0; }
};

template <class T>
bool same_pixel(T a, T b) {
  if constexpr (std::is_same_v<T, float>) {
    if (std::isnan(a) && std::isnan(b)) return true;
  }
  return std::memcmp(&a, &b, sizeof(T)) == 0;
}

template <class T>
T random_value(Rng& rng, bool extreme) {
  if constexpr (std::is_same_v<T, float>) {
    if (extreme) {
      switch (rng.below(9)) {
        case 6: return 0.0f;
        case 0: return std::numeric_limits<float>::infinity();
        case 1: return -std::numeric_limits<float>::infinity();
        case 2: return std::numeric_limits<float>::quiet_NaN();
        case 3: return -0.0f;
        case 4: return std::numeric_limits<float>::denorm_min() * static_cast<float>(1 + rng.below(100));
        case 5: return 3.0e38f;
        default: break;
      }
    }
    return static_cast<float>(rng.range(-300.0, 300.0));
  } else {
    if (extreme && rng.coin(3)) return rng.coin(2) ? std::numeric_limits<T>::max() : T(0);
    return static_cast<T>(rng.below(static_cast<int>(std::numeric_limits<T>::max()) + 1));
  }
}

enum class MapK { Affine, Persp, Table };

struct Case {
  int sw, sh;                 // source size
  int dw, dh;                 // destination image size
  Rect droi;                  // destination ROI
  bool window;                // the destination is a window of just the ROI (origin ox, oy)
  int interp;
  Bounds bounds;
  MapK map;
  bool extreme_src;
  int dstep_pad;              // elements added to the destination's row length
  int sstep_pad;
};

template <class T>
size_t differing(const std::vector<T>& a, const std::vector<T>& b) {
  size_t bad = a.size() == b.size() ? 0 : 1;
  for (size_t i = 0; i < a.size() && i < b.size(); ++i)
    if (!same_pixel(a[i], b[i])) ++bad;
  return bad;
}

// Runs one warp both ways. Returns the number of destination elements that differ; `ran` says whether the fast path took it.
template <class T, int CH>
size_t run_warp(Rng& rng, const Case& cs, bool* ran) {
  const int sstride = cs.sw * CH + cs.sstep_pad;
  std::vector<T> src(static_cast<size_t>(sstride) * cs.sh + simd::kPad, T(0));
  for (int y = 0; y < cs.sh; ++y)
    for (int x = 0; x < cs.sw * CH; ++x) src[static_cast<size_t>(y) * sstride + x] = random_value<T>(rng, cs.extreme_src);
  Image<T> si{src.data(), cs.sw, cs.sh, sstride, CH};
  // the source ROI: the whole image, or a part of it
  Rect sroi{0, 0, cs.sw, cs.sh};
  if (rng.coin(3) && cs.sw > 2 && cs.sh > 2) sroi = Rect{rng.below(cs.sw / 2), rng.below(cs.sh / 2), 1 + rng.below(cs.sw / 2), 1 + rng.below(cs.sh / 2)};
  // the destination (a window of the ROI alone, or the whole image), holding noise to start with
  const int dstride = (cs.window ? cs.droi.w : cs.dw) * CH + cs.dstep_pad;
  const int drows = cs.window ? cs.droi.h : cs.dh;
  std::vector<T> d0(static_cast<size_t>(dstride) * drows);
  for (auto& v : d0) v = random_value<T>(rng, false);
  std::vector<T> d1 = d0;
  Image<T> a{d0.data(), cs.window ? cs.droi.w : cs.dw, drows, dstride, CH, cs.window ? cs.droi.x : 0, cs.window ? cs.droi.y : 0};
  Image<T> b{d1.data(), a.w, a.h, dstride, CH, a.ox, a.oy};
  bool handled = false;
  if (cs.map == MapK::Affine) {
    double c[2][3];
    switch (rng.below(6)) {
      case 4: c[0][0] = 0.5; c[0][1] = 0; c[0][2] = 0; c[1][0] = 0; c[1][1] = 0.5; c[1][2] = rng.coin(2) ? 0.0 : 0.5; break;   // exactly halves
      case 5: c[0][0] = 1; c[0][1] = 0; c[0][2] = rng.below(3) - 0.5; c[1][0] = 0; c[1][1] = 1; c[1][2] = rng.below(3) - 0.5; break;   // on the pixel edges
      case 0: c[0][0] = 1; c[0][1] = 0; c[1][0] = 0; c[1][1] = 1; c[0][2] = rng.range(-3, 3); c[1][2] = rng.range(-3, 3); break;
      case 1: { const double t = rng.range(0, 6.3), sc = rng.range(0.4, 2.2);
                c[0][0] = sc * std::cos(t); c[0][1] = sc * std::sin(t); c[1][0] = -sc * std::sin(t); c[1][1] = sc * std::cos(t);
                c[0][2] = rng.range(-5, cs.sw); c[1][2] = rng.range(-5, cs.sh); break; }
      case 2: c[0][0] = rng.range(-3, 3); c[0][1] = rng.range(-3, 3); c[0][2] = rng.range(-50, 50);
              c[1][0] = rng.range(-3, 3); c[1][1] = rng.range(-3, 3); c[1][2] = rng.range(-50, 50); break;
      default: c[0][0] = 0.5; c[0][1] = 0; c[0][2] = 0.25; c[1][0] = 0; c[1][1] = 0.5; c[1][2] = 0.5; break;   // exactly on the half pixels
    }
    warp(si, sroi, a, cs.droi, cs.interp, [&](int x, int y, float* sx, float* sy) { affine_point(c, x, y, sx, sy); return true; }, cs.bounds);
    handled = simd::warp_affine_back(si, sroi, b, cs.droi, c, cs.interp, cs.bounds);
  } else if (cs.map == MapK::Persp) {
    double c[3][3];
    const double t = rng.range(-0.4, 0.4);
    c[0][0] = std::cos(t) * rng.range(0.6, 1.5); c[0][1] = std::sin(t); c[0][2] = rng.range(-5, 5);
    c[1][0] = -std::sin(t); c[1][1] = std::cos(t) * rng.range(0.6, 1.5); c[1][2] = rng.range(-5, 5);
    c[2][0] = rng.coin(5) ? 0.0 : rng.range(-0.02, 0.02); c[2][1] = rng.coin(5) ? 0.0 : rng.range(-0.02, 0.02); c[2][2] = rng.coin(9) ? 0.0 : 1.0;
    warp_perspective_back(si, sroi, a, Rect{cs.droi.x, cs.droi.y, cs.droi.w, cs.droi.h}, c, cs.interp);
    handled = simd::warp_perspective_back(si, sroi, b, cs.droi, c, cs.interp);
  } else {
    // tables of source points, around and beyond the source ROI
    const int tw = cs.droi.w + static_cast<int>(rng.below(5));
    std::vector<float> tx(static_cast<size_t>(tw) * cs.droi.h), ty(tx.size());
    for (size_t i = 0; i < tx.size(); ++i) {
      tx[i] = static_cast<float>(rng.range(-3, cs.sw + 3));
      ty[i] = static_cast<float>(rng.range(-3, cs.sh + 3));
      if (rng.coin(7)) {   // exactly on the edges the boundary rules are about
        static const float xs[] = {-0.5f, -0.25f, 0.0f, 0.5f, 1.0f};
        tx[i] = rng.coin(2) ? static_cast<float>(sroi.x) + xs[rng.below(5)] : static_cast<float>(sroi.x + sroi.w) + (rng.coin(2) ? 0.0f : -1.0f);
        ty[i] = rng.coin(2) ? static_cast<float>(sroi.y) + xs[rng.below(5)] : static_cast<float>(sroi.y + sroi.h) + (rng.coin(2) ? 0.0f : -1.0f);
      }
      if (rng.coin(40)) tx[i] = std::numeric_limits<float>::quiet_NaN();
      if (rng.coin(60)) ty[i] = std::numeric_limits<float>::infinity();
    }
    warp(si, sroi, a, cs.droi, cs.interp,
         [&](int x, int y, float* sx, float* sy) {
           const size_t i = static_cast<size_t>(y - cs.droi.y) * tw + (x - cs.droi.x);
           *sx = tx[i];
           *sy = ty[i];
           return true;
         },
         cs.bounds);
    handled = simd::warp_table(si, sroi, b, cs.droi, tx.data(), ty.data(), tw, cs.interp, cs.bounds);
  }
  *ran = handled;
  if (!handled) return 0;
  return differing(d0, d1);
}

template <class T, int CH>
size_t run_all_warps(Rng& rng, int* cases, int* fast) {
  size_t bad = 0;
  const int sizes[][2] = {{1, 1}, {2, 3}, {5, 7}, {8, 8}, {17, 13}, {33, 31}, {64, 21}};
  Bounds warp_b, rot_b, remap_b;
  rot_b.lo_slack = 0.5f;
  rot_b.clamp_low = true;
  remap_b.lo_slack = 0.5f;
  remap_b.hi_slack = 1;
  remap_b.hi_open = true;
  remap_b.clamp_low = true;
  for (const auto& sz : sizes)
    for (const int interp : {kNN, kLinear, kCubic})
      for (const MapK map : {MapK::Affine, MapK::Persp, MapK::Table})
        for (int rep = 0; rep < 4; ++rep) {
          Case cs;
          cs.sw = sz[0];
          cs.sh = sz[1];
          cs.dw = 1 + rng.below(40);
          cs.dh = 1 + rng.below(12);
          cs.droi = Rect{rng.below(cs.dw), rng.below(cs.dh), 0, 0};
          cs.droi.w = 1 + rng.below(cs.dw - cs.droi.x);
          cs.droi.h = 1 + rng.below(cs.dh - cs.droi.y);
          cs.window = rng.coin(2);
          cs.interp = interp;
          cs.bounds = map == MapK::Persp ? warp_b : (rep % 3 == 0 ? warp_b : (rep % 3 == 1 ? rot_b : remap_b));
          cs.map = map;
          cs.extreme_src = rep & 1;
          cs.dstep_pad = rng.below(3);
          cs.sstep_pad = rng.below(3);
          bool ran = false;
          const size_t d = run_warp<T, CH>(rng, cs, &ran);
          ++*cases;
          if (ran) ++*fast;
          if (d) std::printf("  warp type %zu ch %d interp %d map %d rep %d src %dx%d: %zu differ\n", sizeof(T), CH, interp, static_cast<int>(map), rep, cs.sw, cs.sh, d);
          bad += d;
        }
  return bad;
}

// The resize loop of npp_imgproc.cpp's resize_sqr for nearest, linear and cubic, as it was.
template <class T, int CH>
void ref_resize(const Image<T>& sic, Image<T>& dimg, const Rect& droi, double fx, double fy, double shx, double shy, int interp, int src_w, int src_h) {
  const float invx = static_cast<float>(1.0 / fx), invy = static_cast<float>(1.0 / fy);
  const float cx = static_cast<float>(0.5 * invx - 0.5 - shx * invx);
  const float cy = static_cast<float>(0.5 * invy - 0.5 - shy * invy);
  for (int y = droi.y; y < droi.y + droi.h; ++y) {
    double v[4];
    for (int x = droi.x; x < droi.x + droi.w; ++x) {
      const double ux = (x + 0.5 - shx) / fx, uy = (y + 0.5 - shy) / fy;
      if (ux < -0.25 || uy < -0.25 || ux >= src_w + 0.5 || uy >= src_h + 0.5) continue;
      const float sx = std::fma(invx, static_cast<float>(x), cx);
      const float sy = std::fma(invy, static_cast<float>(y), cy);
      sample(sic, sx, sy, interp, v);
      for (int c = 0; c < CH; ++c) dimg.at(x, y, c) = to_pixel<T>(v[c]);
    }
  }
}

template <class T, int CH>
size_t run_all_resizes(Rng& rng, int* cases, int* fast) {
  size_t bad = 0;
  const int sizes[][2] = {{1, 1}, {3, 2}, {7, 5}, {16, 9}, {31, 17}, {50, 40}};
  for (const auto& sz : sizes)
    for (const int interp : {kNN, kLinear, kCubic})
      for (int rep = 0; rep < 6; ++rep) {
        const int sw = sz[0], sh = sz[1];
        std::vector<T> src(static_cast<size_t>(sw) * CH * sh + simd::kPad);
        for (size_t i = 0; i + simd::kPad < src.size() + 0 && i < static_cast<size_t>(sw) * CH * sh; ++i) src[i] = random_value<T>(rng, rep & 1);
        Image<T> si{src.data(), sw, sh, sw * CH, CH};
        double fx = rep % 3 == 0 ? rng.range(0.3, 0.99) : rng.range(1.0, 3.5), fy = rep % 2 ? rng.range(0.3, 0.99) : rng.range(1.0, 3.5);
        double shx = rng.range(-2, 3), shy = rng.range(-2, 3);
        if (rep >= 4) {   // factors and shifts that put sample points exactly on halves
          fx = rep == 4 ? 2.0 : 0.5;
          fy = rep == 4 ? 2.0 : 0.5;
          shx = shy = rep == 4 ? -0.5 : 0.25;
        }
        const int dw = std::max(1, static_cast<int>(sw * fx) + 2), dh = std::max(1, static_cast<int>(sh * fy) + 2);
        const Rect droi{rng.below(3), rng.below(3), dw, dh};
        const int dstride = droi.w * CH + rng.below(3);
        std::vector<T> d0(static_cast<size_t>(dstride) * droi.h);
        for (auto& v : d0) v = random_value<T>(rng, false);
        std::vector<T> d1 = d0;
        Image<T> a{d0.data(), droi.w, droi.h, dstride, CH, droi.x, droi.y}, b{d1.data(), droi.w, droi.h, dstride, CH, droi.x, droi.y};
        ref_resize<T, CH>(si, a, droi, fx, fy, shx, shy, interp, sw, sh);
        const auto t = simd::resize_tables(droi, sw, sh, fx, fy, shx, shy);
        const bool handled = simd::resize_sqr(si, b, droi, t.invx, t.cx, t.invy, t.cy, t.colok.data(), t.rowok.data(), interp);
        ++*cases;
        if (handled) {
          ++*fast;
          const size_t d = differing(d0, d1);
          if (d) std::printf("  resize type %zu ch %d interp %d rep %d src %dx%d f %.3f %.3f: %zu differ\n", sizeof(T), CH, interp, rep, sw, sh, fx, fy, d);
          bad += d;
        }
      }
  return bad;
}

// nppiResize_8u_C1R / C3R's loop of npp_api.cpp, as it was.
inline uint8_t ref_sat8u(double v) {
  const double c = v < 0 ? 0 : (v > 256 ? 256 : v);
  const double r = (c + 6755399441055744.0) - 6755399441055744.0;
  return static_cast<uint8_t>(r < 0 ? 0 : (r > 255 ? 255 : r));
}
template <int CH>
void ref_resize_legacy(const std::vector<uint8_t>& src, int sw, int sh, const Rect& sroi, std::vector<uint8_t>& dst, int dw, const Rect& droi, bool linear) {
  (void)sh;
  const double fx = static_cast<double>(sroi.w) / droi.w;
  const double fy = static_cast<double>(sroi.h) / droi.h;
  auto at = [&](int y, int x, int c) -> double {
    x = std::min(std::max(x, sroi.x), sroi.x + sroi.w - 1);
    y = std::min(std::max(y, sroi.y), sroi.y + sroi.h - 1);
    return src[(static_cast<size_t>(y) * sw + x) * CH + c];
  };
  for (int y = 0; y < droi.h; ++y)
    for (int x = 0; x < droi.w; ++x) {
      const double sx = (x + 0.5) * fx + sroi.x - 0.5;
      const double sy = (y + 0.5) * fy + sroi.y - 0.5;
      for (int c = 0; c < CH; ++c) {
        double v;
        if (!linear) {
          v = at(static_cast<int>(std::nearbyint(sy)), static_cast<int>(std::nearbyint(sx)), c);
        } else {
          const int x0 = static_cast<int>(std::floor(sx)), y0 = static_cast<int>(std::floor(sy));
          const float ax = static_cast<float>(sx - x0), ay = static_cast<float>(sy - y0);
          v = static_cast<float>(at(y0, x0, c)) * (1 - ax) * (1 - ay) +
              static_cast<float>(at(y0, x0 + 1, c)) * ax * (1 - ay) +
              static_cast<float>(at(y0 + 1, x0, c)) * (1 - ax) * ay +
              static_cast<float>(at(y0 + 1, x0 + 1, c)) * ax * ay;
        }
        dst[(static_cast<size_t>(y + droi.y) * dw + (x + droi.x)) * CH + c] = ref_sat8u(std::floor(v));
      }
    }
}

template <int CH>
size_t run_legacy_resizes(Rng& rng, int* cases, int* fast) {
  size_t bad = 0;
  for (int rep = 0; rep < 60; ++rep) {
    const int sw = 1 + rng.below(70), sh = 1 + rng.below(40);
    std::vector<uint8_t> src(static_cast<size_t>(sw) * CH * sh + simd::kPad, 0);
    for (size_t i = 0; i < static_cast<size_t>(sw) * CH * sh; ++i) src[i] = random_value<uint8_t>(rng, rep & 1);
    Rect sroi{rng.below(sw), rng.below(sh), 0, 0};
    sroi.w = 1 + rng.below(sw - sroi.x);
    sroi.h = 1 + rng.below(sh - sroi.y);
    const int dw = 1 + rng.below(90), dh = 1 + rng.below(50);
    Rect droi{rng.below(dw), rng.below(dh), 0, 0};
    droi.w = 1 + rng.below(dw - droi.x);
    droi.h = 1 + rng.below(dh - droi.y);
    if (rep % 7 == 0) { droi = Rect{0, 0, dw, dh}; sroi = Rect{0, 0, sw, sh}; }
    std::vector<uint8_t> d0(static_cast<size_t>(dw) * CH * dh);
    for (auto& v : d0) v = random_value<uint8_t>(rng, false);
    std::vector<uint8_t> d1 = d0;
    for (const bool linear : {false, true}) {
      d1 = d0;
      std::vector<uint8_t> want = d0;
      ref_resize_legacy<CH>(src, sw, sh, sroi, want, dw, droi, linear);
      const bool handled = simd::resize_legacy_u8(CH, src.data(), sw * CH, sroi, d1.data(), dw * CH, droi, linear);
      ++*cases;
      if (handled) {
        ++*fast;
        const size_t d = differing(want, d1);
        if (d) std::printf("  legacy resize ch %d linear %d src %dx%d -> %dx%d: %zu differ\n", CH, (int)linear, sw, sh, dw, dh, d);
        bad += d;
      }
    }
  }
  return bad;
}

// Three host threads for the large images below (read once, at the first call that wants more than one): the pixels may not depend
// on how many there are.
struct SetThreads {
  SetThreads() { ::setenv("VGPU_THREADS", "3", 0); }
} set_threads;

// Dilate/Erode's loop of npp_imgproc.cpp's neighbourhood(), as it was (for a minimum or a maximum).
template <class T, bool MAX>
void ref_extreme(const std::vector<T>& src, int sw, std::vector<T>& dst, int rw, int rh, int ch, int mw, int mh, const std::vector<uint8_t>* mask, T init) {
  for (int y = 0; y < rh; ++y)
    for (int x = 0; x < rw; ++x)
      for (int c = 0; c < ch; ++c) {
        T acc = init;
        for (int j = 0; j < mh; ++j)
          for (int i = 0; i < mw; ++i) {
            if (mask && !(*mask)[static_cast<size_t>(j) * mw + i]) continue;
            const T v = src[(static_cast<size_t>(y + j) * sw + x + i) * ch + c];
            acc = MAX ? std::max(acc, v) : std::min(acc, v);
          }
        dst[(static_cast<size_t>(y) * rw + x) * ch + c] = acc;
      }
}

template <class T, bool MAX>
size_t run_extreme(Rng& rng, int ch, int* cases) {
  size_t bad = 0;
  for (int rep = 0; rep < 40; ++rep) {
    const int rw = 1 + rng.below(45), rh = 1 + rng.below(9), mw = 1 + rng.below(6), mh = 1 + rng.below(6);
    const int sw = rw + mw - 1, sh = rh + mh - 1;
    std::vector<T> src(static_cast<size_t>(sw) * sh * ch);
    for (auto& v : src) v = random_value<T>(rng, rep & 1);
    std::vector<uint8_t> mask(static_cast<size_t>(mw) * mh);
    for (auto& m : mask) m = rng.coin(3) ? 0 : 1 + rng.below(255);
    const bool use_mask = rep % 5 != 0;
    const T init = MAX ? (std::is_floating_point_v<T> ? -std::numeric_limits<T>::max() : T(0)) : (std::is_floating_point_v<T> ? std::numeric_limits<T>::max() : std::numeric_limits<T>::max());
    std::vector<T> want(static_cast<size_t>(rw) * rh * ch), got(want.size());
    ref_extreme<T, MAX>(src, sw, want, rw, rh, ch, mw, mh, use_mask ? &mask : nullptr, init);
    std::vector<std::pair<int, int>> taps;
    for (int j = 0; j < mh; ++j)
      for (int i = 0; i < mw; ++i)
        if (!use_mask || mask[static_cast<size_t>(j) * mw + i]) taps.emplace_back(j, i * ch);
    extreme_filter<T, MAX>(src.data(), sw * ch, got.data(), rw, rh, ch, taps, init);
    ++*cases;
    const size_t d = differing(want, got);
    if (d) std::printf("  extreme type %zu ch %d max %d: %zu differ\n", sizeof(T), ch, (int)MAX, d);
    bad += d;
  }
  return bad;
}

}  // namespace

VTEST(npp_simd_warps_match_the_scalar_code) {
  Rng rng;
  int cases = 0, fast = 0;
  size_t bad = 0;
  bad += run_all_warps<uint8_t, 1>(rng, &cases, &fast);
  bad += run_all_warps<uint8_t, 3>(rng, &cases, &fast);
  bad += run_all_warps<uint8_t, 4>(rng, &cases, &fast);
  bad += run_all_warps<uint16_t, 1>(rng, &cases, &fast);
  bad += run_all_warps<uint16_t, 3>(rng, &cases, &fast);
  bad += run_all_warps<uint16_t, 4>(rng, &cases, &fast);
  bad += run_all_warps<float, 1>(rng, &cases, &fast);
  bad += run_all_warps<float, 3>(rng, &cases, &fast);
  bad += run_all_warps<float, 4>(rng, &cases, &fast);
  std::printf("  warps: %d cases, %d on the fast path, %zu elements differ\n", cases, fast, bad);
  VCHECK_EQ(bad, static_cast<size_t>(0));
  if (simd::cpu_ok()) VCHECK_EQ(fast, cases);
}

VTEST(npp_simd_resizes_match_the_scalar_code) {
  Rng rng;
  rng.s ^= 0x1234567;
  int cases = 0, fast = 0;
  size_t bad = 0;
  bad += run_all_resizes<uint8_t, 1>(rng, &cases, &fast);
  bad += run_all_resizes<uint8_t, 3>(rng, &cases, &fast);
  bad += run_all_resizes<uint8_t, 4>(rng, &cases, &fast);
  bad += run_all_resizes<uint16_t, 1>(rng, &cases, &fast);
  bad += run_all_resizes<uint16_t, 3>(rng, &cases, &fast);
  bad += run_all_resizes<float, 1>(rng, &cases, &fast);
  bad += run_all_resizes<float, 3>(rng, &cases, &fast);
  bad += run_all_resizes<float, 4>(rng, &cases, &fast);
  std::printf("  resizes: %d cases, %d on the fast path, %zu elements differ\n", cases, fast, bad);
  VCHECK_EQ(bad, static_cast<size_t>(0));
  if (simd::cpu_ok()) VCHECK_EQ(fast, cases);
}

VTEST(npp_simd_large_images_on_several_threads) {
  Rng rng;
  rng.s ^= 0x5555;
  const int W = 330, H = 190;
  size_t bad = 0;
  int ran = 0;
  for (const int interp : {kNN, kLinear, kCubic}) {
    std::vector<uint8_t> src(static_cast<size_t>(W) * 3 * H + simd::kPad);
    for (size_t i = 0; i < static_cast<size_t>(W) * 3 * H; ++i) src[i] = static_cast<uint8_t>(rng.below(256));
    Image<uint8_t> si{src.data(), W, H, W * 3, 3};
    std::vector<uint8_t> d0(static_cast<size_t>(W) * 3 * H, 7), d1 = d0;
    Image<uint8_t> a{d0.data(), W, H, W * 3, 3}, b{d1.data(), W, H, W * 3, 3};
    const double c[2][3] = {{0.93, 0.31, 11.5}, {-0.29, 0.95, 17.25}};
    const Rect roi{0, 0, W, H};
    warp(si, roi, a, roi, interp, [&](int x, int y, float* sx, float* sy) { affine_point(c, x, y, sx, sy); return true; }, Bounds());
    if (simd::warp_affine_back(si, roi, b, roi, c, interp, Bounds())) ++ran;
    bad += differing(d0, d1);
    // a resize of the same image
    std::vector<uint8_t> e0(static_cast<size_t>(W) * 3 * H, 9), e1 = e0;
    Image<uint8_t> ea{e0.data(), W, H, W * 3, 3}, eb{e1.data(), W, H, W * 3, 3};
    ref_resize<uint8_t, 3>(si, ea, roi, 1.3, 0.8, 0.1, -0.2, interp, W, H);
    const auto t = simd::resize_tables(roi, W, H, 1.3, 0.8, 0.1, -0.2);
    if (simd::resize_sqr(si, eb, roi, t.invx, t.cx, t.invy, t.cy, t.colok.data(), t.rowok.data(), interp)) ++ran;
    bad += differing(e0, e1);
  }
  std::printf("  large images: %d runs on the fast path, %zu elements differ\n", ran, bad);
  VCHECK_EQ(bad, static_cast<size_t>(0));
  if (simd::cpu_ok()) VCHECK_EQ(ran, 6);
}

VTEST(npp_simd_legacy_resize_matches_the_scalar_code) {
  Rng rng;
  rng.s ^= 0xABCDEF;
  int cases = 0, fast = 0;
  size_t bad = 0;
  bad += run_legacy_resizes<1>(rng, &cases, &fast);
  bad += run_legacy_resizes<3>(rng, &cases, &fast);
  std::printf("  legacy resizes: %d cases, %d on the fast path, %zu elements differ\n", cases, fast, bad);
  VCHECK_EQ(bad, static_cast<size_t>(0));
  if (simd::cpu_ok()) VCHECK_EQ(fast, cases);
}

VTEST(npp_morphology_matches_the_loops) {
  Rng rng;
  rng.s ^= 0xFEED;
  int cases = 0;
  size_t bad = 0;
  for (const int ch : {1, 3, 4}) {
    bad += run_extreme<uint8_t, true>(rng, ch, &cases);
    bad += run_extreme<uint8_t, false>(rng, ch, &cases);
    bad += run_extreme<float, true>(rng, ch, &cases);
    bad += run_extreme<float, false>(rng, ch, &cases);
  }
  std::printf("  morphology: %d cases, %zu elements differ\n", cases, bad);
  VCHECK_EQ(bad, static_cast<size_t>(0));
}
VTEST(npp_box_filter_matches_the_window_sums) {
  Rng rng;
  rng.s ^= 0xB0C5;
  size_t bad = 0;
  int cases = 0;
  for (int rep = 0; rep < 150; ++rep) {
    const int ch = rep % 3 == 0 ? 1 : (rep % 3 == 1 ? 3 : 4);
    const int rw = 1 + rng.below(40), rh = 1 + rng.below(12), mw = 1 + rng.below(9), mh = 1 + rng.below(9);
    const int sw = rw + mw - 1, sh = rh + mh - 1;
    std::vector<uint8_t> src(static_cast<size_t>(sw) * sh * ch);
    for (auto& v : src) v = rep & 1 ? (rng.coin(2) ? 255 : 0) : static_cast<uint8_t>(rng.below(256));
    std::vector<uint8_t> want(static_cast<size_t>(rw) * rh * ch), got(want.size(), 7);
    const double n = static_cast<double>(mw) * mh;
    for (int y = 0; y < rh; ++y)
      for (int x = 0; x < rw; ++x)
        for (int c = 0; c < ch; ++c) {   // the loops of neighbourhood(): a sum over the window in double, floor(sum / n), saturated
          double acc = 0.0;
          for (int j = 0; j < mh; ++j)
            for (int i = 0; i < mw; ++i) acc = acc + src[(static_cast<size_t>(y + j) * sw + x + i) * ch + c];
          want[(static_cast<size_t>(y) * rw + x) * ch + c] = saturate<uint8_t>(std::floor(acc / n));
        }
    const bool done = box_filter_u8(src.data(), sw, ch, got.data(), rw, rh, mw, mh);
    ++cases;
    if (done) bad += differing(want, got);
    else ++bad;
  }
  std::printf("  box filter: %d cases, %zu elements differ\n", cases, bad);
  VCHECK_EQ(bad, static_cast<size_t>(0));
}

VTEST_MAIN
