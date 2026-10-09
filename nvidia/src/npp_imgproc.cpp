// VirtualGPU's NPP, second half: the image-processing entry points real
// programs call that npp_api.cpp does not cover -- geometric transforms,
// remapping, histograms, morphology and rank filters, the logical and shift
// operators with constants, alpha compositing, gamma, Bayer demosaicing,
// integral images, Canny and the rest of what OpenCV's cudaimgproc,
// cudaarithm, cudawarping and cudafilters, DALI, FFmpeg's scale_npp and the
// CUDA Samples use (nvidia/docs/libraries.md has the list, ranked by those
// users, with what is and is not here).
//
// Same boundary as npp_api.cpp: the pixels are fetched to the host, computed
// there and stored back. The arithmetic that has to match NVIDIA's to the bit
// (sampling, rounding, fused multiply-adds) lives in npp_core.hpp, where a
// card program can compare it with NVIDIA's NPP directly; the conventions are
// recorded there and next to each function below, all measured on an RTX 3060
// against NPP 13.0.
//
// Each function is exported in its NppStreamContext (_Ctx) form, which is the
// only one CUDA 13 declares, and in the plain form CUDA 12 still declares.
#include <npp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <tuple>
#include <type_traits>
#include <vector>

#include <cuda_runtime.h>

#include "npp_core.hpp"

using namespace vgpu_npp;

#define VGPU_EXPORT extern "C" __attribute__((visibility("default")))

namespace {

// The type of argument I of a function pointer type: GetBufferSize's
// out-parameter is int* on CUDA 12.0 and size_t* later, so each definition
// takes it from the header's own declaration.
template <size_t I, class F>
struct arg_of;
template <size_t I, class R, class... A>
struct arg_of<I, R (*)(A...)> {
  using type = std::tuple_element_t<I, std::tuple<A...>>;
};

/* ---- transfers ---- */

template <class T>
std::vector<T> fetch(const void* dev, int step, int elems_per_row, int rows) {
  std::vector<T> host(static_cast<size_t>(elems_per_row) * rows);
  const auto* base = static_cast<const unsigned char*>(dev);
  for (int r = 0; r < rows; ++r)
    cudaMemcpy(host.data() + static_cast<size_t>(r) * elems_per_row,
               base + static_cast<size_t>(r) * step, elems_per_row * sizeof(T),
               cudaMemcpyDeviceToHost);
  return host;
}

template <class T>
void store(void* dev, int step, int elems_per_row, int rows, const std::vector<T>& host) {
  auto* base = static_cast<unsigned char*>(dev);
  for (int r = 0; r < rows; ++r)
    cudaMemcpy(base + static_cast<size_t>(r) * step,
               host.data() + static_cast<size_t>(r) * elems_per_row, elems_per_row * sizeof(T),
               cudaMemcpyHostToDevice);
}

template <class T>
const T* offset(const T* p, int step, int x, int y, int ch) {
  return reinterpret_cast<const T*>(reinterpret_cast<const char*>(p) + static_cast<ptrdiff_t>(y) * step) +
         static_cast<ptrdiff_t>(x) * ch;
}
template <class T>
T* offset(T* p, int step, int x, int y, int ch) {
  return reinterpret_cast<T*>(reinterpret_cast<char*>(p) + static_cast<ptrdiff_t>(y) * step) +
         static_cast<ptrdiff_t>(x) * ch;
}

template <class T>
void put(void* dev, const T& v) {
  cudaMemcpy(dev, &v, sizeof(T), cudaMemcpyHostToDevice);
}
template <class T>
void put_n(void* dev, const T* v, size_t n) {
  cudaMemcpy(dev, v, n * sizeof(T), cudaMemcpyHostToDevice);
}

// A pointer the caller passed for NPP to read device-side may also be a host
// pointer on hardware with unified addressing; copy from whichever it is.
template <class T>
void get_n(T* host, const T* any, size_t n) {
  cudaPointerAttributes a{};
  if (cudaPointerGetAttributes(&a, any) == cudaSuccess &&
      (a.type == cudaMemoryTypeDevice || a.type == cudaMemoryTypeManaged)) {
    cudaMemcpy(host, any, n * sizeof(T), cudaMemcpyDeviceToHost);
  } else {
    cudaGetLastError();
    std::memcpy(host, any, n * sizeof(T));
  }
}

bool bad(NppiSize s) { return s.width <= 0 || s.height <= 0; }

// A whole-row window [x0, x0 + w) x [y0, y0 + h) of a device image, fetched.
template <class T>
struct Window {
  std::vector<T> data;
  Image<T> img;
  Window(const T* dev, int step, int x0, int y0, int w, int h, int ch) {
    data = fetch<T>(offset(dev, step, x0, y0, ch), step, w * ch, h);
    img = Image<T>{data.data(), w, h, w * ch, ch, x0, y0};
  }
  void put_back(T* dev, int step) const {
    store<T>(offset(dev, step, img.ox, img.oy, img.ch), step, img.w * img.ch, img.h, data);
  }
};

}  // namespace

// Plain (CUDA 12) form of an entry point defined in its _Ctx form.
#define VGPU_PLAIN(NAME, PARAMS, ARGS) \
  VGPU_EXPORT NppStatus NAME PARAMS { return NAME##_Ctx ARGS; }

/* ======================================================================
   Geometry (nppig): WarpAffine, WarpPerspective, Rotate, Remap, Mirror,
   ResizeSqrPixel, and the quad/bound/transform helpers.
   ====================================================================== */

namespace {

// The quadrangle a source ROI maps to: its corner pixels (x, y), (x+w-1, y),
// (x+w-1, y+h-1), (x, y+h-1), which is what nppiGetRotateQuad returns
// (measured: the second corner of a 30-wide ROI at x=3 is column 32).
void affine_quad(const NppiRect& r, const double c[2][3], double q[4][2]) {
  const double xs[4] = {double(r.x), double(r.x + r.width - 1), double(r.x + r.width - 1), double(r.x)};
  const double ys[4] = {double(r.y), double(r.y), double(r.y + r.height - 1), double(r.y + r.height - 1)};
  for (int i = 0; i < 4; ++i) {
    q[i][0] = c[0][0] * xs[i] + c[0][1] * ys[i] + c[0][2];
    q[i][1] = c[1][0] * xs[i] + c[1][1] * ys[i] + c[1][2];
  }
}
void perspective_quad(const NppiRect& r, const double c[3][3], double q[4][2]) {
  const double xs[4] = {double(r.x), double(r.x + r.width - 1), double(r.x + r.width - 1), double(r.x)};
  const double ys[4] = {double(r.y), double(r.y), double(r.y + r.height - 1), double(r.y + r.height - 1)};
  for (int i = 0; i < 4; ++i) {
    const double w = c[2][0] * xs[i] + c[2][1] * ys[i] + c[2][2];
    q[i][0] = (c[0][0] * xs[i] + c[0][1] * ys[i] + c[0][2]) / w;
    q[i][1] = (c[1][0] * xs[i] + c[1][1] * ys[i] + c[1][2]) / w;
  }
}
void quad_bound(const double q[4][2], double b[2][2]) {
  b[0][0] = b[1][0] = q[0][0];
  b[0][1] = b[1][1] = q[0][1];
  for (int i = 1; i < 4; ++i) {
    b[0][0] = std::min(b[0][0], q[i][0]);
    b[0][1] = std::min(b[0][1], q[i][1]);
    b[1][0] = std::max(b[1][0], q[i][0]);
    b[1][1] = std::max(b[1][1], q[i][1]);
  }
}
// Whether the bounding box of the mapped source ROI meets the destination ROI.
// When it does not, NPP writes nothing and returns
// NPP_WRONG_INTERSECTION_QUAD_WARNING (measured with Rotate by 137 and 200
// degrees, which turn the source away from the destination entirely).
bool meets(const double q[4][2], const NppiRect& d) {
  double b[2][2];
  quad_bound(q, b);
  return !(b[1][0] < d.x || b[0][0] > d.x + d.width - 1 || b[1][1] < d.y || b[0][1] > d.y + d.height - 1);
}

// The source ROI clipped to the image.
bool clip_src(NppiSize size, NppiRect r, Rect* out) {
  const int x0 = std::max(r.x, 0), y0 = std::max(r.y, 0);
  const int x1 = std::min(r.x + r.width, size.width), y1 = std::min(r.y + r.height, size.height);
  if (x1 <= x0 || y1 <= y0) return false;
  *out = Rect{x0, y0, x1 - x0, y1 - y0};
  return true;
}

bool known_interp(int i) { return i == NPPI_INTER_NN || i == NPPI_INTER_LINEAR || i == NPPI_INTER_CUBIC; }

NppStatus check_geometry(const void* s, NppiSize ssz, NppiRect sroi, const void* d, NppiRect droi, int interp,
                         Rect* src_roi) {
  if (!s || !d) return NPP_NULL_POINTER_ERROR;
  if (bad(ssz) || sroi.width <= 0 || sroi.height <= 0 || droi.width <= 0 || droi.height <= 0)
    return NPP_SIZE_ERROR;
  if (!known_interp(interp)) return NPP_INTERPOLATION_ERROR;
  if (!clip_src(ssz, sroi, src_roi)) return NPP_WRONG_INTERSECTION_ROI_ERROR;
  return NPP_SUCCESS;
}

// mode: 0 = forward affine (c maps source to destination), 1 = back affine
// (c maps destination to source), 2 = forward perspective, 3 = back
// perspective, 4 = rotate (c is the forward affine map).
template <class T>
NppStatus geometric(const T* pSrc, NppiSize ssz, int sstep, NppiRect sroi, T* pDst, int dstep, NppiRect droi,
                    const double* coeffs, int interp, int ch, int mode) {
  Rect sr{};
  if (NppStatus st = check_geometry(pSrc, ssz, sroi, pDst, droi, interp, &sr)) return st;
  if (!coeffs) return NPP_NULL_POINTER_ERROR;
  double q[4][2];
  double fwd2[2][3], back2[2][3], fwd3[3][3], back3[3][3];
  const bool affine = mode == 0 || mode == 1 || mode == 4;
  if (affine) {
    std::memcpy(mode == 1 ? back2 : fwd2, coeffs, sizeof fwd2);
    if (mode == 1 ? !invert_affine(back2, fwd2) : !invert_affine(fwd2, back2)) return NPP_COEFFICIENT_ERROR;
    affine_quad(NppiRect{sr.x, sr.y, sr.w, sr.h}, fwd2, q);
  } else {
    std::memcpy(mode == 3 ? back3 : fwd3, coeffs, sizeof fwd3);
    if (mode == 3 ? !invert_3x3(back3, fwd3) : !invert_3x3(fwd3, back3)) return NPP_COEFFICIENT_ERROR;
    perspective_quad(NppiRect{sr.x, sr.y, sr.w, sr.h}, fwd3, q);
  }
  if (!meets(q, droi)) return NPP_WRONG_INTERSECTION_QUAD_WARNING;
  auto src = fetch<T>(pSrc, sstep, ssz.width * ch, ssz.height);
  Image<T> si{src.data(), ssz.width, ssz.height, ssz.width * ch, ch};
  Window<T> dst(pDst, dstep, droi.x, droi.y, droi.width, droi.height, ch);
  const Rect dr{droi.x, droi.y, droi.width, droi.height};
  if (affine) {
    // Rotate accepts half a pixel more before the ROI than the warps do and
    // samples such points on its edge (see Bounds).
    Bounds b;
    if (mode == 4) {
      b.lo_slack = 0.5f;
      b.clamp_low = true;
    }
    warp(si, sr, dst.img, dr, interp,
         [&](int x, int y, float* sx, float* sy) {
           affine_point(back2, x, y, sx, sy);
           return true;
         },
         b);
  } else {
    warp_perspective_back(si, sr, dst.img, dr, back3, interp);
  }
  dst.put_back(pDst, dstep);
  return NPP_SUCCESS;
}

}  // namespace

#define VGPU_NPP_WARP(NAME, MODE, SFX, T, CH, CT)                                                            \
  VGPU_EXPORT NppStatus nppi##NAME##_##SFX##_Ctx(const T* s, NppiSize ssz, int ss, NppiRect sroi, T* d,      \
                                                 int ds, NppiRect droi, CT, int interp, NppStreamContext) { \
    return geometric<T>(s, ssz, ss, sroi, d, ds, droi, &c[0][0], interp, CH, MODE);                          \
  }                                                                                                          \
  VGPU_PLAIN(nppi##NAME##_##SFX,                                                                              \
             (const T* s, NppiSize ssz, int ss, NppiRect sroi, T* d, int ds, NppiRect droi, CT, int interp), \
             (s, ssz, ss, sroi, d, ds, droi, c, interp, NppStreamContext{}))

#define VGPU_NPP_WARP_TYPES(NAME, MODE, CT)          \
  VGPU_NPP_WARP(NAME, MODE, 8u_C1R, Npp8u, 1, CT)    \
  VGPU_NPP_WARP(NAME, MODE, 8u_C3R, Npp8u, 3, CT)    \
  VGPU_NPP_WARP(NAME, MODE, 8u_C4R, Npp8u, 4, CT)    \
  VGPU_NPP_WARP(NAME, MODE, 16u_C1R, Npp16u, 1, CT)  \
  VGPU_NPP_WARP(NAME, MODE, 16u_C3R, Npp16u, 3, CT)  \
  VGPU_NPP_WARP(NAME, MODE, 16u_C4R, Npp16u, 4, CT)  \
  VGPU_NPP_WARP(NAME, MODE, 32s_C1R, Npp32s, 1, CT)  \
  VGPU_NPP_WARP(NAME, MODE, 32s_C3R, Npp32s, 3, CT)  \
  VGPU_NPP_WARP(NAME, MODE, 32s_C4R, Npp32s, 4, CT)  \
  VGPU_NPP_WARP(NAME, MODE, 32f_C1R, Npp32f, 1, CT)  \
  VGPU_NPP_WARP(NAME, MODE, 32f_C3R, Npp32f, 3, CT)  \
  VGPU_NPP_WARP(NAME, MODE, 32f_C4R, Npp32f, 4, CT)

VGPU_NPP_WARP_TYPES(WarpAffine, 0, const double c[2][3])
VGPU_NPP_WARP_TYPES(WarpAffineBack, 1, const double c[2][3])
VGPU_NPP_WARP_TYPES(WarpPerspective, 2, const double c[3][3])
VGPU_NPP_WARP_TYPES(WarpPerspectiveBack, 3, const double c[3][3])

// Rotate by nAngle degrees about the origin, then shift: the forward map is
// x' = cos(a) x + sin(a) y + shiftX, y' = -sin(a) x + cos(a) y + shiftY
// (measured against the four sign conventions; this one matches every pixel).
namespace {
void rotate_coeffs(double angle, double sx, double sy, double c[2][3]) {
  const double a = angle * M_PI / 180.0;
  c[0][0] = std::cos(a);
  c[0][1] = std::sin(a);
  c[0][2] = sx;
  c[1][0] = -std::sin(a);
  c[1][1] = std::cos(a);
  c[1][2] = sy;
}
}  // namespace

#define VGPU_NPP_ROTATE(SFX, T, CH)                                                                        \
  VGPU_EXPORT NppStatus nppiRotate_##SFX##_Ctx(const T* s, NppiSize ssz, int ss, NppiRect sroi, T* d,      \
                                               int ds, NppiRect droi, double angle, double shx,           \
                                               double shy, int interp, NppStreamContext) {                \
    double c[2][3];                                                                                        \
    rotate_coeffs(angle, shx, shy, c);                                                                     \
    return geometric<T>(s, ssz, ss, sroi, d, ds, droi, &c[0][0], interp, CH, 4);                           \
  }                                                                                                        \
  VGPU_PLAIN(nppiRotate_##SFX,                                                                              \
             (const T* s, NppiSize ssz, int ss, NppiRect sroi, T* d, int ds, NppiRect droi, double angle,  \
              double shx, double shy, int interp),                                                         \
             (s, ssz, ss, sroi, d, ds, droi, angle, shx, shy, interp, NppStreamContext{}))
VGPU_NPP_ROTATE(8u_C1R, Npp8u, 1)
VGPU_NPP_ROTATE(8u_C3R, Npp8u, 3)
VGPU_NPP_ROTATE(8u_C4R, Npp8u, 4)
VGPU_NPP_ROTATE(16u_C1R, Npp16u, 1)
VGPU_NPP_ROTATE(16u_C3R, Npp16u, 3)
VGPU_NPP_ROTATE(16u_C4R, Npp16u, 4)
VGPU_NPP_ROTATE(32f_C1R, Npp32f, 1)
VGPU_NPP_ROTATE(32f_C3R, Npp32f, 3)
VGPU_NPP_ROTATE(32f_C4R, Npp32f, 4)

VGPU_EXPORT NppStatus nppiGetRotateQuad(NppiRect r, double q[4][2], double angle, double shx, double shy) {
  if (!q) return NPP_NULL_POINTER_ERROR;
  double c[2][3];
  rotate_coeffs(angle, shx, shy, c);
  affine_quad(r, c, q);
  return NPP_SUCCESS;
}
VGPU_EXPORT NppStatus nppiGetRotateBound(NppiRect r, double b[2][2], double angle, double shx, double shy) {
  if (!b) return NPP_NULL_POINTER_ERROR;
  double q[4][2];
  nppiGetRotateQuad(r, q, angle, shx, shy);
  quad_bound(q, b);
  return NPP_SUCCESS;
}
VGPU_EXPORT NppStatus nppiGetAffineQuad(NppiRect r, double q[4][2], const double c[2][3]) {
  if (!q || !c) return NPP_NULL_POINTER_ERROR;
  affine_quad(r, c, q);
  return NPP_SUCCESS;
}
VGPU_EXPORT NppStatus nppiGetAffineBound(NppiRect r, double b[2][2], const double c[2][3]) {
  if (!b || !c) return NPP_NULL_POINTER_ERROR;
  double q[4][2];
  affine_quad(r, c, q);
  quad_bound(q, b);
  return NPP_SUCCESS;
}
VGPU_EXPORT NppStatus nppiGetPerspectiveQuad(NppiRect r, double q[4][2], const double c[3][3]) {
  if (!q || !c) return NPP_NULL_POINTER_ERROR;
  perspective_quad(r, c, q);
  return NPP_SUCCESS;
}
VGPU_EXPORT NppStatus nppiGetPerspectiveBound(NppiRect r, double b[2][2], const double c[3][3]) {
  if (!b || !c) return NPP_NULL_POINTER_ERROR;
  double q[4][2];
  perspective_quad(r, c, q);
  quad_bound(q, b);
  return NPP_SUCCESS;
}
// The affine map taking three corners of the ROI to quad[0..2] (the fourth
// must agree with it, as a parallelogram).
VGPU_EXPORT NppStatus nppiGetAffineTransform(NppiRect r, const double q[4][2], double c[2][3]) {
  if (!q || !c) return NPP_NULL_POINTER_ERROR;
  if (r.width <= 1 || r.height <= 1) return NPP_RECTANGLE_ERROR;
  const double w = r.width - 1, h = r.height - 1;
  // x' = a x + b y + t with corners (x0,y0)->q0, (x0+w,y0)->q1, (x0+w,y0+h)->q2.
  for (int k = 0; k < 2; ++k) {
    const double a = (q[1][k] - q[0][k]) / w;
    const double b = (q[2][k] - q[1][k]) / h;
    c[k][0] = a;
    c[k][1] = b;
    c[k][2] = q[0][k] - a * r.x - b * r.y;
  }
  return NPP_SUCCESS;
}
// The projective map taking the four ROI corners to quad (a 3x3 with c22 = 1).
VGPU_EXPORT NppStatus nppiGetPerspectiveTransform(NppiRect r, const double q[4][2], double c[3][3]) {
  if (!q || !c) return NPP_NULL_POINTER_ERROR;
  if (r.width <= 1 || r.height <= 1) return NPP_RECTANGLE_ERROR;
  const double xs[4] = {double(r.x), double(r.x + r.width - 1), double(r.x + r.width - 1), double(r.x)};
  const double ys[4] = {double(r.y), double(r.y), double(r.y + r.height - 1), double(r.y + r.height - 1)};
  // Solve the 8x8 system for c00..c21.
  double m[8][9] = {};
  for (int i = 0; i < 4; ++i) {
    double* a = m[2 * i];
    double* b = m[2 * i + 1];
    a[0] = xs[i]; a[1] = ys[i]; a[2] = 1; a[6] = -xs[i] * q[i][0]; a[7] = -ys[i] * q[i][0]; a[8] = q[i][0];
    b[3] = xs[i]; b[4] = ys[i]; b[5] = 1; b[6] = -xs[i] * q[i][1]; b[7] = -ys[i] * q[i][1]; b[8] = q[i][1];
  }
  for (int col = 0; col < 8; ++col) {
    int piv = col;
    for (int r2 = col + 1; r2 < 8; ++r2)
      if (std::fabs(m[r2][col]) > std::fabs(m[piv][col])) piv = r2;
    if (std::fabs(m[piv][col]) < 1e-300) return NPP_COEFFICIENT_ERROR;
    if (piv != col)
      for (int k = 0; k < 9; ++k) std::swap(m[piv][k], m[col][k]);
    for (int r2 = 0; r2 < 8; ++r2) {
      if (r2 == col) continue;
      const double f = m[r2][col] / m[col][col];
      for (int k = col; k < 9; ++k) m[r2][k] -= f * m[col][k];
    }
  }
  double v[8];
  for (int i = 0; i < 8; ++i) v[i] = m[i][8] / m[i][i];
  c[0][0] = v[0]; c[0][1] = v[1]; c[0][2] = v[2];
  c[1][0] = v[3]; c[1][1] = v[4]; c[1][2] = v[5];
  c[2][0] = v[6]; c[2][1] = v[7]; c[2][2] = 1;
  return NPP_SUCCESS;
}

/* ---- Remap ----
   dst(x, y) = src(xmap(x, y), ymap(x, y)). Measured: a point is taken when
   roi.x - 0.5 <= sx < roi.x + roi.w (and likewise in y) -- half a pixel early,
   a whole pixel late and open there -- and linear and cubic samples before
   the ROI are taken on its edge. Outside, the destination keeps its value. */
namespace {
template <class T, class M>
NppStatus remap(const T* pSrc, NppiSize ssz, int sstep, NppiRect sroi, const M* xm, int xstep, const M* ym,
                int ystep, T* pDst, int dstep, NppiSize dsz, int interp, int ch) {
  if (!pSrc || !pDst || !xm || !ym) return NPP_NULL_POINTER_ERROR;
  if (bad(ssz) || bad(dsz) || sroi.width <= 0 || sroi.height <= 0) return NPP_SIZE_ERROR;
  if (!known_interp(interp)) return NPP_INTERPOLATION_ERROR;
  Rect sr{};
  if (!clip_src(ssz, sroi, &sr)) return NPP_WRONG_INTERSECTION_ROI_ERROR;
  auto src = fetch<T>(pSrc, sstep, ssz.width * ch, ssz.height);
  Image<T> si{src.data(), ssz.width, ssz.height, ssz.width * ch, ch};
  auto hx = fetch<M>(xm, xstep, dsz.width, dsz.height);
  auto hy = fetch<M>(ym, ystep, dsz.width, dsz.height);
  Window<T> dst(pDst, dstep, 0, 0, dsz.width, dsz.height, ch);
  Bounds b;
  b.lo_slack = 0.5f;
  b.hi_slack = 1;
  b.hi_open = true;
  b.clamp_low = true;
  warp(si, sr, dst.img, Rect{0, 0, dsz.width, dsz.height}, interp,
       [&](int x, int y, float* sx, float* sy) {
         const size_t i = static_cast<size_t>(y) * dsz.width + x;
         *sx = static_cast<float>(hx[i]);
         *sy = static_cast<float>(hy[i]);
         return true;
       },
       b);
  dst.put_back(pDst, dstep);
  return NPP_SUCCESS;
}
}  // namespace

#define VGPU_NPP_REMAP(SFX, T, M, CH)                                                                      \
  VGPU_EXPORT NppStatus nppiRemap_##SFX##_Ctx(const T* s, NppiSize ssz, int ss, NppiRect sroi, const M* xm, \
                                              int xs, const M* ym, int ys, T* d, int ds, NppiSize dsz,     \
                                              int interp, NppStreamContext) {                              \
    return remap<T, M>(s, ssz, ss, sroi, xm, xs, ym, ys, d, ds, dsz, interp, CH);                          \
  }                                                                                                        \
  VGPU_PLAIN(nppiRemap_##SFX,                                                                               \
             (const T* s, NppiSize ssz, int ss, NppiRect sroi, const M* xm, int xs, const M* ym, int ys,   \
              T* d, int ds, NppiSize dsz, int interp),                                                     \
             (s, ssz, ss, sroi, xm, xs, ym, ys, d, ds, dsz, interp, NppStreamContext{}))
VGPU_NPP_REMAP(8u_C1R, Npp8u, Npp32f, 1)
VGPU_NPP_REMAP(8u_C3R, Npp8u, Npp32f, 3)
VGPU_NPP_REMAP(8u_C4R, Npp8u, Npp32f, 4)
VGPU_NPP_REMAP(16u_C1R, Npp16u, Npp32f, 1)
VGPU_NPP_REMAP(16u_C3R, Npp16u, Npp32f, 3)
VGPU_NPP_REMAP(16u_C4R, Npp16u, Npp32f, 4)
VGPU_NPP_REMAP(16s_C1R, Npp16s, Npp32f, 1)
VGPU_NPP_REMAP(16s_C3R, Npp16s, Npp32f, 3)
VGPU_NPP_REMAP(16s_C4R, Npp16s, Npp32f, 4)
VGPU_NPP_REMAP(32f_C1R, Npp32f, Npp32f, 1)
VGPU_NPP_REMAP(32f_C3R, Npp32f, Npp32f, 3)
VGPU_NPP_REMAP(32f_C4R, Npp32f, Npp32f, 4)
VGPU_NPP_REMAP(64f_C1R, Npp64f, Npp64f, 1)
VGPU_NPP_REMAP(64f_C3R, Npp64f, Npp64f, 3)
VGPU_NPP_REMAP(64f_C4R, Npp64f, Npp64f, 4)

/* ---- Mirror ---- */
namespace {
template <class T>
NppStatus mirror(const T* s, int ss, T* d, int ds, NppiSize roi, NppiAxis flip, int ch) {
  if (!s || !d) return NPP_NULL_POINTER_ERROR;
  if (bad(roi)) return NPP_SIZE_ERROR;
  if (flip != NPP_HORIZONTAL_AXIS && flip != NPP_VERTICAL_AXIS && flip != NPP_BOTH_AXIS)
    return NPP_MIRROR_FLIP_ERROR;
  auto src = fetch<T>(s, ss, roi.width * ch, roi.height);
  std::vector<T> dst(src.size());
  const bool flip_rows = flip == NPP_HORIZONTAL_AXIS || flip == NPP_BOTH_AXIS;
  const bool flip_cols = flip == NPP_VERTICAL_AXIS || flip == NPP_BOTH_AXIS;
  for (int y = 0; y < roi.height; ++y)
    for (int x = 0; x < roi.width; ++x) {
      const int sy = flip_rows ? roi.height - 1 - y : y;
      const int sx = flip_cols ? roi.width - 1 - x : x;
      for (int c = 0; c < ch; ++c)
        dst[(static_cast<size_t>(y) * roi.width + x) * ch + c] = src[(static_cast<size_t>(sy) * roi.width + sx) * ch + c];
    }
  store<T>(d, ds, roi.width * ch, roi.height, dst);
  return NPP_SUCCESS;
}
}  // namespace

#define VGPU_NPP_MIRROR_R(SFX, T, CH)                                                                \
  VGPU_EXPORT NppStatus nppiMirror_##SFX##_Ctx(const T* s, int ss, T* d, int ds, NppiSize roi,       \
                                               NppiAxis f, NppStreamContext) {                       \
    return mirror<T>(s, ss, d, ds, roi, f, CH);                                                      \
  }                                                                                                  \
  VGPU_PLAIN(nppiMirror_##SFX, (const T* s, int ss, T* d, int ds, NppiSize roi, NppiAxis f),          \
             (s, ss, d, ds, roi, f, NppStreamContext{}))
#define VGPU_NPP_MIRROR_IR(SFX, T, CH)                                                               \
  VGPU_EXPORT NppStatus nppiMirror_##SFX##_Ctx(T* sd, int ss, NppiSize roi, NppiAxis f,              \
                                               NppStreamContext) {                                   \
    /* In place needs an even width and height whatever the axis (measured: \
       12x8 works, 12x9 and 13x8 are NPP_SIZE_ERROR). */                    \
    if (roi.width % 2 || roi.height % 2) return NPP_SIZE_ERROR;                                      \
    return mirror<T>(sd, ss, sd, ss, roi, f, CH);                                                    \
  }                                                                                                  \
  VGPU_PLAIN(nppiMirror_##SFX, (T * sd, int ss, NppiSize roi, NppiAxis f), (sd, ss, roi, f, NppStreamContext{}))
// nppiMirror_8u_C1R is in npp_api.cpp.
VGPU_NPP_MIRROR_R(8u_C3R, Npp8u, 3)
VGPU_NPP_MIRROR_R(8u_C4R, Npp8u, 4)
VGPU_NPP_MIRROR_IR(8u_C1IR, Npp8u, 1)
VGPU_NPP_MIRROR_IR(8u_C3IR, Npp8u, 3)
VGPU_NPP_MIRROR_IR(8u_C4IR, Npp8u, 4)
VGPU_NPP_MIRROR_R(16u_C1R, Npp16u, 1)
VGPU_NPP_MIRROR_R(16u_C3R, Npp16u, 3)
VGPU_NPP_MIRROR_R(16u_C4R, Npp16u, 4)
VGPU_NPP_MIRROR_IR(16u_C1IR, Npp16u, 1)
VGPU_NPP_MIRROR_IR(16u_C3IR, Npp16u, 3)
VGPU_NPP_MIRROR_IR(16u_C4IR, Npp16u, 4)
VGPU_NPP_MIRROR_R(32s_C1R, Npp32s, 1)
VGPU_NPP_MIRROR_R(32s_C3R, Npp32s, 3)
VGPU_NPP_MIRROR_R(32s_C4R, Npp32s, 4)
VGPU_NPP_MIRROR_IR(32s_C1IR, Npp32s, 1)
VGPU_NPP_MIRROR_IR(32s_C3IR, Npp32s, 3)
VGPU_NPP_MIRROR_IR(32s_C4IR, Npp32s, 4)
VGPU_NPP_MIRROR_R(32f_C1R, Npp32f, 1)
VGPU_NPP_MIRROR_R(32f_C3R, Npp32f, 3)
VGPU_NPP_MIRROR_R(32f_C4R, Npp32f, 4)
VGPU_NPP_MIRROR_IR(32f_C1IR, Npp32f, 1)
VGPU_NPP_MIRROR_IR(32f_C3IR, Npp32f, 3)
VGPU_NPP_MIRROR_IR(32f_C4IR, Npp32f, 4)

/* ---- ResizeSqrPixel ----
   dst(x, y) samples the source at ((x + 0.5 - shiftX) / xFactor - 0.5, and
   likewise in y) -- pixel centres mapped through the scale (measured: nearest
   neighbour matches every pixel at factors 0.5, 0.73, 1.37 and 2) -- with the
   sample point held inside the source ROI. A destination pixel is written
   when its sample point is within [-0.25, width + 0.5) of the ROI in pixel
   edges, ux = (x + 0.5 - shift) / factor -- so a 23-wide source at factor 2
   writes columns 0..46, at 0.73 columns 0..16 (measured at six factors, and
   at shifts from -1 to 7 at six more: the lower bound is -0.25 in source
   pixels at every factor, closed, and the upper bound open at width + 0.5),
   clipped to the destination ROI. Nearest neighbour is exact; linear is the warps' bilinear filter, a
   count away on the odd pixel; cubic is a four-point Lagrange filter whose
   weights and fused multiply-add order were fitted bit for bit to NVIDIA's
   resize (npp_core.hpp, lagrange4 and sample). Lanczos and super-sampling
   follow below, each with what was measured. */
namespace {
// Lanczos-3 (nppiResizeSqrPixel's NPPI_INTER_LANCZOS): the windowed sinc,
// sinc(d) * sinc(d / 3) for |d| < 3, taken at the taps within three pixels of
// the sample point and normalised to sum to one at every point. Not widened
// when shrinking: at factor 0.5 NPP's weights at the three half-integer
// offsets are exactly -0.135870, 0.611413 and 0.024457 (an unwidened
// kernel's), and at factors whose phases are multiples of 0.05 the impulse
// response is this kernel to within a float rounding. Between those NPP's
// values differ from the analytic kernel by up to 2e-3 relative (3e-5 in the
// weights at three and a third); that is where the card's tabulated or
// approximated sinc shows, and not reproduced here.
inline double sinc_pi(double x) {
  if (x == 0) return 1;
  const double t = 3.14159265358979323846 * x;
  return std::sin(t) / t;
}
inline double lanczos3(double d) { return std::fabs(d) < 3 ? sinc_pi(d) * sinc_pi(d / 3) : 0; }

template <class T>
NppStatus resize_sqr(const T* pSrc, NppiSize ssz, int sstep, NppiRect sroi, T* pDst, int dstep, NppiRect droi,
                     double fx, double fy, double shx, double shy, int interp, int ch) {
  const bool supported = interp == NPPI_INTER_NN || interp == NPPI_INTER_LINEAR || interp == NPPI_INTER_CUBIC ||
                         interp == NPPI_INTER_SUPER || interp == NPPI_INTER_LANCZOS;
  Rect sr{};
  if (!pSrc || !pDst) return NPP_NULL_POINTER_ERROR;
  if (bad(ssz) || sroi.width <= 0 || sroi.height <= 0 || droi.width <= 0 || droi.height <= 0) return NPP_SIZE_ERROR;
  if (!supported) return NPP_INTERPOLATION_ERROR;
  if (!clip_src(ssz, sroi, &sr)) return NPP_WRONG_INTERSECTION_ROI_ERROR;
  if (!(fx > 0) || !(fy > 0)) return NPP_RESIZE_FACTOR_ERROR;
  // Super-sampling shrinks: both factors must be below one (measured: factors
  // (1, 0.5), (0.5, 1) and (1, 1) are NPP_RESIZE_FACTOR_ERROR).
  if (interp == NPPI_INTER_SUPER && !(fx < 1 && fy < 1)) return NPP_RESIZE_FACTOR_ERROR;
  auto src = fetch<T>(pSrc, sstep, ssz.width * ch, ssz.height);
  // Taps clamp to the ROI: a window of the source holding just it.
  std::vector<T> win(static_cast<size_t>(sr.w) * sr.h * ch);
  for (int y = 0; y < sr.h; ++y)
    std::memcpy(&win[static_cast<size_t>(y) * sr.w * ch], &src[(static_cast<size_t>(y + sr.y) * ssz.width + sr.x) * ch],
                sizeof(T) * sr.w * ch);
  Image<T> si{win.data(), sr.w, sr.h, sr.w * ch, ch};
  Window<T> dst(pDst, dstep, droi.x, droi.y, droi.width, droi.height, ch);
  double v[4];
  // The sample point of destination column x is fma(1/f, x, c) in single
  // precision, with 1/f rounded to float and c = 0.5/f - 0.5 - shift/f formed
  // from it in double and rounded: the only grouping that reproduces NPP's
  // impulse responses to the last bit, at every factor and shift tried.
  const float invx = static_cast<float>(1.0 / fx), invy = static_cast<float>(1.0 / fy);
  const float cx = static_cast<float>(0.5 * invx - 0.5 - shx * invx);
  const float cy = static_cast<float>(0.5 * invy - 0.5 - shy * invy);
  for (int y = droi.y; y < droi.y + droi.height; ++y)
    for (int x = droi.x; x < droi.x + droi.width; ++x) {
      if (interp == NPPI_INTER_SUPER) {
        // Each destination pixel is the mean of the source over the interval
        // it covers, [(x - shift) / factor, (x + 1 - shift) / factor): the
        // overlap-weighted sum times fx * fy (no division by the covered area,
        // so the part of an interval beyond the ROI counts as zero). It is
        // written only when its interval starts inside the ROI (measured: a
        // pixel straddling the first column or row is left alone, one
        // straddling the last is written).
        const double x0 = (x - shx) / fx, x1 = (x + 1 - shx) / fx;
        const double y0 = (y - shy) / fy, y1 = (y + 1 - shy) / fy;
        if (x0 < 0 || y0 < 0 || x0 >= sr.w || y0 >= sr.h) continue;
        const int xa = static_cast<int>(std::floor(x0)), xb = std::min(sr.w, static_cast<int>(std::ceil(x1)));
        const int ya = static_cast<int>(std::floor(y0)), yb = std::min(sr.h, static_cast<int>(std::ceil(y1)));
        for (int c = 0; c < ch; ++c) {
          double acc = 0;
          for (int sy = ya; sy < yb; ++sy) {
            const double oy = std::min<double>(sy + 1, y1) - std::max<double>(sy, y0);
            if (oy <= 0) continue;
            for (int sx = xa; sx < xb; ++sx) {
              const double ox = std::min<double>(sx + 1, x1) - std::max<double>(sx, x0);
              if (ox <= 0) continue;
              acc += static_cast<double>(si.at(sx, sy, c)) * ox * oy;
            }
          }
          v[c] = acc * fx * fy;
        }
        for (int c = 0; c < ch; ++c) dst.img.at(x, y, c) = to_pixel<T>(v[c]);
        continue;
      }
      const double ux = (x + 0.5 - shx) / fx, uy = (y + 0.5 - shy) / fy;
      if (ux < -0.25 || uy < -0.25 || ux >= sr.w + 0.5 || uy >= sr.h + 0.5) continue;
      const float sx = std::fma(invx, static_cast<float>(x), cx);
      const float sy = std::fma(invy, static_cast<float>(y), cy);
      if (interp == NPPI_INTER_LANCZOS) {
        const int xa = static_cast<int>(std::floor(sx)) - 2, xb = xa + 6;
        const int ya = static_cast<int>(std::floor(sy)) - 2, yb = ya + 6;
        double wx[6], wy[6], sxs = 0, sys = 0;
        for (int i = 0; i < 6; ++i) {
          wx[i] = lanczos3((xa + i) - static_cast<double>(sx));
          wy[i] = lanczos3((ya + i) - static_cast<double>(sy));
          sxs += wx[i];
          sys += wy[i];
        }
        (void)xb;
        (void)yb;
        for (int c = 0; c < ch; ++c) {
          double acc = 0;
          for (int j = 0; j < 6; ++j) {
            if (wy[j] == 0) continue;
            const int yy = std::min(std::max(ya + j, 0), sr.h - 1);
            double row = 0;
            for (int i = 0; i < 6; ++i) {
              if (wx[i] == 0) continue;
              const int xx = std::min(std::max(xa + i, 0), sr.w - 1);
              row += wx[i] * static_cast<double>(si.at(xx, yy, c));
            }
            acc += wy[j] * row;
          }
          v[c] = acc / (sxs * sys);
        }
        for (int c = 0; c < ch; ++c) dst.img.at(x, y, c) = to_pixel<T>(v[c]);
        continue;
      }
      sample(si, sx, sy, interp, v);
      for (int c = 0; c < ch; ++c) dst.img.at(x, y, c) = to_pixel<T>(v[c]);
    }
  dst.put_back(pDst, dstep);
  return NPP_SUCCESS;
}
}  // namespace

#define VGPU_NPP_RESIZESQR(SFX, T, CH)                                                                     \
  VGPU_EXPORT NppStatus nppiResizeSqrPixel_##SFX##_Ctx(const T* s, NppiSize ssz, int ss, NppiRect sroi, T* d, \
                                                       int ds, NppiRect droi, double fx, double fy,          \
                                                       double shx, double shy, int interp, NppStreamContext) { \
    return resize_sqr<T>(s, ssz, ss, sroi, d, ds, droi, fx, fy, shx, shy, interp, CH);                      \
  }                                                                                                        \
  VGPU_PLAIN(nppiResizeSqrPixel_##SFX,                                                                      \
             (const T* s, NppiSize ssz, int ss, NppiRect sroi, T* d, int ds, NppiRect droi, double fx,      \
              double fy, double shx, double shy, int interp),                                              \
             (s, ssz, ss, sroi, d, ds, droi, fx, fy, shx, shy, interp, NppStreamContext{}))
VGPU_NPP_RESIZESQR(8u_C1R, Npp8u, 1)
VGPU_NPP_RESIZESQR(8u_C3R, Npp8u, 3)
VGPU_NPP_RESIZESQR(8u_C4R, Npp8u, 4)
VGPU_NPP_RESIZESQR(16u_C1R, Npp16u, 1)
VGPU_NPP_RESIZESQR(16u_C3R, Npp16u, 3)
VGPU_NPP_RESIZESQR(16u_C4R, Npp16u, 4)
VGPU_NPP_RESIZESQR(32f_C1R, Npp32f, 1)
VGPU_NPP_RESIZESQR(32f_C3R, Npp32f, 3)
VGPU_NPP_RESIZESQR(32f_C4R, Npp32f, 4)

/* ======================================================================
   Arithmetic and logical operators with constants (nppial), magnitude,
   alpha compositing.
   ====================================================================== */

namespace {
template <class T, class Op>
NppStatus per_channel(const T* s, int ss, T* d, int ds, NppiSize roi, int ch, int touched, Op op) {
  if (!s || !d) return NPP_NULL_POINTER_ERROR;
  if (bad(roi)) return NPP_SIZE_ERROR;
  auto src = fetch<T>(s, ss, roi.width * ch, roi.height);
  // AC4 forms leave the destination's alpha as it was: start from it.
  std::vector<T> dst = touched < ch ? fetch<T>(d, ds, roi.width * ch, roi.height) : src;
  for (size_t i = 0; i < src.size(); ++i) {
    const int c = static_cast<int>(i % ch);
    if (c < touched) dst[i] = op(src[i], c);
  }
  store<T>(d, ds, roi.width * ch, roi.height, dst);
  return NPP_SUCCESS;
}
}  // namespace

#define VGPU_NPP_LOGICC(NAME, OPX, SFX, T, CH)                                                          \
  VGPU_EXPORT NppStatus nppi##NAME##_##SFX##_Ctx(const T* s, int ss, const T k[CH], T* d, int ds,        \
                                                 NppiSize roi, NppStreamContext) {                      \
    if (!k) return NPP_NULL_POINTER_ERROR;                                                              \
    T kk[4] = {};                                                                                       \
    std::memcpy(kk, k, sizeof(T) * CH);                                                                 \
    return per_channel<T>(s, ss, d, ds, roi, CH, CH, [&](T v, int c) { return T(v OPX kk[c]); });        \
  }                                                                                                     \
  VGPU_PLAIN(nppi##NAME##_##SFX, (const T* s, int ss, const T k[CH], T* d, int ds, NppiSize roi),        \
             (s, ss, k, d, ds, roi, NppStreamContext{}))
#define VGPU_NPP_LOGICC1(NAME, OPX, SFX, T)                                                             \
  VGPU_EXPORT NppStatus nppi##NAME##_##SFX##_Ctx(const T* s, int ss, const T k, T* d, int ds,            \
                                                 NppiSize roi, NppStreamContext) {                      \
    return per_channel<T>(s, ss, d, ds, roi, 1, 1, [&](T v, int) { return T(v OPX k); });               \
  }                                                                                                     \
  VGPU_PLAIN(nppi##NAME##_##SFX, (const T* s, int ss, const T k, T* d, int ds, NppiSize roi),            \
             (s, ss, k, d, ds, roi, NppStreamContext{}))
#define VGPU_NPP_LOGICC_ALL(NAME, OPX)                  \
  VGPU_NPP_LOGICC1(NAME, OPX, 8u_C1R, Npp8u)            \
  VGPU_NPP_LOGICC(NAME, OPX, 8u_C3R, Npp8u, 3)          \
  VGPU_NPP_LOGICC(NAME, OPX, 8u_C4R, Npp8u, 4)          \
  VGPU_NPP_LOGICC1(NAME, OPX, 16u_C1R, Npp16u)          \
  VGPU_NPP_LOGICC(NAME, OPX, 16u_C3R, Npp16u, 3)        \
  VGPU_NPP_LOGICC(NAME, OPX, 16u_C4R, Npp16u, 4)        \
  VGPU_NPP_LOGICC1(NAME, OPX, 32s_C1R, Npp32s)          \
  VGPU_NPP_LOGICC(NAME, OPX, 32s_C3R, Npp32s, 3)        \
  VGPU_NPP_LOGICC(NAME, OPX, 32s_C4R, Npp32s, 4)
VGPU_NPP_LOGICC_ALL(AndC, &)
VGPU_NPP_LOGICC_ALL(OrC, |)
VGPU_NPP_LOGICC_ALL(XorC, ^)

// Shifts by constants: the count is Npp32u; a left shift drops the bits that
// leave the type and a right shift of a signed type is arithmetic.
#define VGPU_NPP_SHIFTC(NAME, OPX, SFX, T, CH)                                                          \
  VGPU_EXPORT NppStatus nppi##NAME##_##SFX##_Ctx(const T* s, int ss, const Npp32u k[CH], T* d, int ds,   \
                                                 NppiSize roi, NppStreamContext) {                      \
    if (!k) return NPP_NULL_POINTER_ERROR;                                                              \
    Npp32u kk[4] = {};                                                                                  \
    std::memcpy(kk, k, sizeof(Npp32u) * CH);                                                            \
    return per_channel<T>(s, ss, d, ds, roi, CH, CH, [&](T v, int c) { return shift_##OPX<T>(v, kk[c]); }); \
  }                                                                                                     \
  VGPU_PLAIN(nppi##NAME##_##SFX, (const T* s, int ss, const Npp32u k[CH], T* d, int ds, NppiSize roi),   \
             (s, ss, k, d, ds, roi, NppStreamContext{}))
#define VGPU_NPP_SHIFTC1(NAME, OPX, SFX, T)                                                             \
  VGPU_EXPORT NppStatus nppi##NAME##_##SFX##_Ctx(const T* s, int ss, const Npp32u k, T* d, int ds,       \
                                                 NppiSize roi, NppStreamContext) {                      \
    return per_channel<T>(s, ss, d, ds, roi, 1, 1, [&](T v, int) { return shift_##OPX<T>(v, k); });     \
  }                                                                                                     \
  VGPU_PLAIN(nppi##NAME##_##SFX, (const T* s, int ss, const Npp32u k, T* d, int ds, NppiSize roi),       \
             (s, ss, k, d, ds, roi, NppStreamContext{}))
namespace {
template <class T>
T shift_left(T v, Npp32u k) {
  using U = std::make_unsigned_t<T>;
  if (k >= sizeof(T) * 8) return 0;
  return static_cast<T>(static_cast<U>(static_cast<U>(v) << k));
}
template <class T>
T shift_right(T v, Npp32u k) {
  if (k >= sizeof(T) * 8) return std::is_signed_v<T> && v < 0 ? T(-1) : T(0);
  return static_cast<T>(v >> k);
}
}  // namespace
#define VGPU_NPP_SHIFT_TYPE(NAME, OPX, T, TS)      \
  VGPU_NPP_SHIFTC1(NAME, OPX, TS##_C1R, T)         \
  VGPU_NPP_SHIFTC(NAME, OPX, TS##_C3R, T, 3)       \
  VGPU_NPP_SHIFTC(NAME, OPX, TS##_C4R, T, 4)
VGPU_NPP_SHIFT_TYPE(LShiftC, left, Npp8u, 8u)
VGPU_NPP_SHIFT_TYPE(LShiftC, left, Npp16u, 16u)
VGPU_NPP_SHIFT_TYPE(LShiftC, left, Npp32s, 32s)
VGPU_NPP_SHIFT_TYPE(RShiftC, right, Npp8u, 8u)
VGPU_NPP_SHIFT_TYPE(RShiftC, right, Npp8s, 8s)
VGPU_NPP_SHIFT_TYPE(RShiftC, right, Npp16u, 16u)
VGPU_NPP_SHIFT_TYPE(RShiftC, right, Npp16s, 16s)
VGPU_NPP_SHIFT_TYPE(RShiftC, right, Npp32s, 32s)

VGPU_EXPORT NppStatus nppiMagnitude_32fc32f_C1R_Ctx(const Npp32fc* s, int ss, Npp32f* d, int ds, NppiSize roi,
                                                     NppStreamContext) {
  if (!s || !d) return NPP_NULL_POINTER_ERROR;
  if (bad(roi)) return NPP_SIZE_ERROR;
  auto src = fetch<Npp32fc>(s, ss, roi.width, roi.height);
  std::vector<Npp32f> dst(src.size());
  for (size_t i = 0; i < src.size(); ++i)
    dst[i] = std::sqrt(std::fma(src[i].re, src[i].re, src[i].im * src[i].im));
  store<Npp32f>(d, ds, roi.width, roi.height, dst);
  return NPP_SUCCESS;
}
VGPU_PLAIN(nppiMagnitude_32fc32f_C1R, (const Npp32fc* s, int ss, Npp32f* d, int ds, NppiSize roi),
           (s, ss, d, ds, roi, NppStreamContext{}))
VGPU_EXPORT NppStatus nppiMagnitudeSqr_32fc32f_C1R_Ctx(const Npp32fc* s, int ss, Npp32f* d, int ds, NppiSize roi,
                                                        NppStreamContext) {
  if (!s || !d) return NPP_NULL_POINTER_ERROR;
  if (bad(roi)) return NPP_SIZE_ERROR;
  auto src = fetch<Npp32fc>(s, ss, roi.width, roi.height);
  std::vector<Npp32f> dst(src.size());
  for (size_t i = 0; i < src.size(); ++i) dst[i] = std::fma(src[i].re, src[i].re, src[i].im * src[i].im);
  store<Npp32f>(d, ds, roi.width, roi.height, dst);
  return NPP_SUCCESS;
}
VGPU_PLAIN(nppiMagnitudeSqr_32fc32f_C1R, (const Npp32fc* s, int ss, Npp32f* d, int ds, NppiSize roi),
           (s, ss, d, ds, roi, NppStreamContext{}))

/* ---- alpha compositing ----
   8-bit, measured over every (value, alpha) pair: a product x * a / 255 is
   (x * a' + 127) >> 8 with a' = a except that 255 counts as 256, and
   1 - a is 256 - a'. Every operator's alpha, and every colour except ATOP's
   and XOR's, then matches NVIDIA's exactly; those two are within one count.
   The 4-channel forms need a step that is a whole number of pixels
   (NPP_NOT_EVEN_STEP_ERROR otherwise, as on hardware). */
namespace {
inline int a256(int a) { return a == 255 ? 256 : a; }
inline int mr(int x, int a) { return (x * a + 127) >> 8; }

// The 0..1-scale operators, for every depth but 8-bit.
template <class T>
void alpha_comp_float(const T* a, const T* b, T* o, int op, double full) {
  const double aA = a[3] / full, aB = b[3] / full;
  double ca[3], cb[3];
  for (int c = 0; c < 3; ++c) {
    ca[c] = a[c];
    cb[c] = b[c];
  }
  double fa = 0, fb = 0, alpha = 0;  // colour = A * fa + B * fb
  switch (op) {
    case NPPI_OP_ALPHA_OVER: fa = aA; fb = aB * (1 - aA); alpha = aA + aB * (1 - aA); break;
    case NPPI_OP_ALPHA_IN: fa = aA * aB; alpha = aA * aB; break;
    case NPPI_OP_ALPHA_OUT: fa = aA * (1 - aB); alpha = aA * (1 - aB); break;
    case NPPI_OP_ALPHA_ATOP: fa = aA * aB; fb = aB * (1 - aA); alpha = aB; break;
    case NPPI_OP_ALPHA_XOR: fa = aA * (1 - aB); fb = aB * (1 - aA); alpha = aA * (1 - aB) + aB * (1 - aA); break;
    case NPPI_OP_ALPHA_PLUS: fa = aA; fb = aB; alpha = aA + aB; break;
    case NPPI_OP_ALPHA_OVER_PREMUL: fa = 1; fb = 1 - aA; alpha = aA + aB * (1 - aA); break;
    case NPPI_OP_ALPHA_IN_PREMUL: fa = aB; alpha = aA * aB; break;
    case NPPI_OP_ALPHA_OUT_PREMUL: fa = 1 - aB; alpha = aA * (1 - aB); break;
    case NPPI_OP_ALPHA_ATOP_PREMUL: fa = aB; fb = 1 - aA; alpha = aB; break;
    case NPPI_OP_ALPHA_XOR_PREMUL: fa = 1 - aB; fb = 1 - aA; alpha = aA * (1 - aB) + aB * (1 - aA); break;
    case NPPI_OP_ALPHA_PLUS_PREMUL: fa = 1; fb = 1; alpha = aA + aB; break;
    default: fa = aA; alpha = aA; break;  // NPPI_OP_ALPHA_PREMUL
  }
  for (int c = 0; c < 3; ++c) o[c] = std::is_floating_point_v<T> ? T(ca[c] * fa + cb[c] * fb)
                                                                  : to_pixel<T>(ca[c] * fa + cb[c] * fb);
  // The PLUS operators' alpha saturates at 1 (measured on 32f: 0.75 + 0.5
  // gives 1); their colours do not.
  alpha = std::min(alpha, 1.0);
  o[3] = std::is_floating_point_v<T> ? T(alpha * full) : to_pixel<T>(alpha * full);
}

void alpha_comp_8u(const Npp8u* a, const Npp8u* b, Npp8u* o, int op) {
  const int pA = a256(a[3]), pB = a256(b[3]), iA = 256 - pA, iB = 256 - pB;
  int alpha = 0;
  for (int c = 0; c < 3; ++c) {
    const int A = a[c], B = b[c];
    int v = 0;
    switch (op) {
      case NPPI_OP_ALPHA_OVER: v = mr(A, pA) + mr(B, mr(pB, iA)); break;
      case NPPI_OP_ALPHA_IN: v = mr(A, mr(pA, pB)); break;
      case NPPI_OP_ALPHA_OUT: v = mr(A, mr(pA, iB)); break;
      case NPPI_OP_ALPHA_ATOP: v = mr(mr(A, pA) + mr(B, iA), pB); break;
      case NPPI_OP_ALPHA_XOR: v = mr(A, mr(pA, iB)) + mr(B, mr(pB, iA)); break;
      case NPPI_OP_ALPHA_PLUS: v = mr(A, pA) + mr(B, pB); break;
      case NPPI_OP_ALPHA_OVER_PREMUL: v = A + mr(B, iA); break;
      case NPPI_OP_ALPHA_IN_PREMUL: v = mr(A, pB); break;
      case NPPI_OP_ALPHA_OUT_PREMUL: v = mr(A, iB); break;
      case NPPI_OP_ALPHA_ATOP_PREMUL: v = mr(A, pB) + mr(B, iA); break;
      case NPPI_OP_ALPHA_XOR_PREMUL: v = mr(A, iB) + mr(B, iA); break;
      case NPPI_OP_ALPHA_PLUS_PREMUL: v = A + B; break;
      default: v = mr(A, pA); break;
    }
    o[c] = static_cast<Npp8u>(std::min(v, 255));
  }
  switch (op) {
    case NPPI_OP_ALPHA_OVER:
    case NPPI_OP_ALPHA_OVER_PREMUL: alpha = pA + mr(iA, pB); break;
    case NPPI_OP_ALPHA_IN:
    case NPPI_OP_ALPHA_IN_PREMUL: alpha = mr(pA, pB); break;
    case NPPI_OP_ALPHA_OUT:
    case NPPI_OP_ALPHA_OUT_PREMUL: alpha = mr(pA, iB); break;
    case NPPI_OP_ALPHA_ATOP:
    case NPPI_OP_ALPHA_ATOP_PREMUL: alpha = mr(pA, pB) + mr(iA, pB); break;
    case NPPI_OP_ALPHA_XOR:
    case NPPI_OP_ALPHA_XOR_PREMUL: alpha = mr(pA, iB) + mr(iA, pB); break;
    case NPPI_OP_ALPHA_PLUS:
    case NPPI_OP_ALPHA_PLUS_PREMUL: alpha = pA + pB; break;
    default: alpha = pA; break;
  }
  o[3] = static_cast<Npp8u>(std::min(alpha, 255));
}

template <class T>
NppStatus alpha_comp(const T* s1, int ss1, const T* s2, int ss2, T* d, int ds, NppiSize roi, NppiAlphaOp op) {
  if (!s1 || !s2 || !d) return NPP_NULL_POINTER_ERROR;
  if (bad(roi)) return NPP_SIZE_ERROR;
  if (op < NPPI_OP_ALPHA_OVER || op > NPPI_OP_ALPHA_PREMUL) return NPP_NOT_SUPPORTED_MODE_ERROR;
  if (ss1 % (4 * int(sizeof(T))) || ss2 % (4 * int(sizeof(T))) || ds % (4 * int(sizeof(T))))
    return NPP_NOT_EVEN_STEP_ERROR;
  auto a = fetch<T>(s1, ss1, roi.width * 4, roi.height);
  auto b = fetch<T>(s2, ss2, roi.width * 4, roi.height);
  std::vector<T> o(a.size());
  for (size_t i = 0; i < a.size(); i += 4) {
    if constexpr (std::is_same_v<T, Npp8u>)
      alpha_comp_8u(&a[i], &b[i], &o[i], op);
    else
      alpha_comp_float<T>(&a[i], &b[i], &o[i], op,
                          std::is_floating_point_v<T> ? 1.0 : double(std::numeric_limits<T>::max()));
  }
  store<T>(d, ds, roi.width * 4, roi.height, o);
  return NPP_SUCCESS;
}
}  // namespace

#define VGPU_NPP_ALPHACOMP(SFX, T)                                                                         \
  VGPU_EXPORT NppStatus nppiAlphaComp_##SFX##_Ctx(const T* s1, int ss1, const T* s2, int ss2, T* d, int ds, \
                                                  NppiSize roi, NppiAlphaOp op, NppStreamContext) {        \
    return alpha_comp<T>(s1, ss1, s2, ss2, d, ds, roi, op);                                                \
  }                                                                                                        \
  VGPU_PLAIN(nppiAlphaComp_##SFX,                                                                           \
             (const T* s1, int ss1, const T* s2, int ss2, T* d, int ds, NppiSize roi, NppiAlphaOp op),     \
             (s1, ss1, s2, ss2, d, ds, roi, op, NppStreamContext{}))
VGPU_NPP_ALPHACOMP(8u_AC4R, Npp8u)
VGPU_NPP_ALPHACOMP(16u_AC4R, Npp16u)
VGPU_NPP_ALPHACOMP(32s_AC4R, Npp32s)
VGPU_NPP_ALPHACOMP(32f_AC4R, Npp32f)

// Premultiply by the pixel's own alpha, which is copied through.
#define VGPU_NPP_PREMUL(SFX, T)                                                                            \
  VGPU_EXPORT NppStatus nppiAlphaPremul_##SFX##_Ctx(const T* s, int ss, T* d, int ds, NppiSize roi,         \
                                                    NppStreamContext) {                                    \
    if (!s || !d) return NPP_NULL_POINTER_ERROR;                                                           \
    if (bad(roi)) return NPP_SIZE_ERROR;                                                                   \
    if (ss % (4 * int(sizeof(T))) || ds % (4 * int(sizeof(T)))) return NPP_NOT_EVEN_STEP_ERROR;            \
    auto a = fetch<T>(s, ss, roi.width * 4, roi.height);                                                   \
    for (size_t i = 0; i < a.size(); i += 4) {                                                             \
      for (int c = 0; c < 3; ++c) {                                                                        \
        if constexpr (sizeof(T) == 1)                                                                      \
          a[i + c] = static_cast<T>(mr(a[i + c], a256(a[i + 3])));                                         \
        else                                                                                               \
          a[i + c] = to_pixel<T>(double(a[i + c]) * a[i + 3] / 65535.0);                                   \
      }                                                                                                    \
    }                                                                                                      \
    store<T>(d, ds, roi.width * 4, roi.height, a);                                                         \
    return NPP_SUCCESS;                                                                                    \
  }                                                                                                        \
  VGPU_PLAIN(nppiAlphaPremul_##SFX, (const T* s, int ss, T* d, int ds, NppiSize roi),                       \
             (s, ss, d, ds, roi, NppStreamContext{}))
VGPU_NPP_PREMUL(8u_AC4R, Npp8u)
VGPU_NPP_PREMUL(16u_AC4R, Npp16u)

/* ======================================================================
   Colour (nppicc): gamma, channel swaps, YCbCr 4:2:0 plane layouts, Bayer
   demosaicing, piecewise-linear lookup.
   ====================================================================== */

// Gamma: the BT.709 transfer function, in single precision with truncation
// (both measured against all 256 inputs):
//   forward v' = v < 0.018 ? 4.5 v : 1.099 v^0.45 - 0.099
//   inverse v  = v' < 0.081 ? v' * 0.222222 : ((v' + 0.099) / 1.099)^2.22
// The inverse's exponent is 2.22 rather than 1/0.45 and its slope the float
// 0.222222f rather than 1/4.5: either exact value moves 9 and 18 (and with
// them a dozen others) to the next count.
namespace {
Npp8u gamma_fwd(Npp8u x) {
  const float v = x / 255.f;
  const float r = v < 0.018f ? 4.5f * v : 1.099f * std::pow(v, 0.45f) - 0.099f;
  return static_cast<Npp8u>(std::min(255, std::max(0, static_cast<int>(r * 255.f))));
}
Npp8u gamma_inv(Npp8u x) {
  const float v = x / 255.f;
  const float r = v < 0.081f ? v * 0.222222f : std::pow((v + 0.099f) / 1.099f, 2.22f);
  return static_cast<Npp8u>(std::min(255, std::max(0, static_cast<int>(r * 255.f))));
}
}  // namespace

#define VGPU_NPP_GAMMA(DIR, FN)                                                                            \
  VGPU_EXPORT NppStatus nppiGamma##DIR##_8u_C3R_Ctx(const Npp8u* s, int ss, Npp8u* d, int ds, NppiSize roi, \
                                                    NppStreamContext) {                                    \
    return per_channel<Npp8u>(s, ss, d, ds, roi, 3, 3, [](Npp8u v, int) { return FN(v); });                \
  }                                                                                                        \
  VGPU_PLAIN(nppiGamma##DIR##_8u_C3R, (const Npp8u* s, int ss, Npp8u* d, int ds, NppiSize roi),            \
             (s, ss, d, ds, roi, NppStreamContext{}))                                                      \
  VGPU_EXPORT NppStatus nppiGamma##DIR##_8u_C3IR_Ctx(Npp8u* sd, int ss, NppiSize roi, NppStreamContext) {   \
    return per_channel<Npp8u>(sd, ss, sd, ss, roi, 3, 3, [](Npp8u v, int) { return FN(v); });              \
  }                                                                                                        \
  VGPU_PLAIN(nppiGamma##DIR##_8u_C3IR, (Npp8u * sd, int ss, NppiSize roi), (sd, ss, roi, NppStreamContext{})) \
  VGPU_EXPORT NppStatus nppiGamma##DIR##_8u_AC4R_Ctx(const Npp8u* s, int ss, Npp8u* d, int ds, NppiSize roi, \
                                                     NppStreamContext) {                                   \
    if (ss % 4 || ds % 4) return NPP_NOT_EVEN_STEP_ERROR;                                                  \
    return per_channel<Npp8u>(s, ss, d, ds, roi, 4, 3, [](Npp8u v, int) { return FN(v); });                \
  }                                                                                                        \
  VGPU_PLAIN(nppiGamma##DIR##_8u_AC4R, (const Npp8u* s, int ss, Npp8u* d, int ds, NppiSize roi),           \
             (s, ss, d, ds, roi, NppStreamContext{}))                                                      \
  VGPU_EXPORT NppStatus nppiGamma##DIR##_8u_AC4IR_Ctx(Npp8u* sd, int ss, NppiSize roi, NppStreamContext) {  \
    if (ss % 4) return NPP_NOT_EVEN_STEP_ERROR;                                                            \
    return per_channel<Npp8u>(sd, ss, sd, ss, roi, 4, 3, [](Npp8u v, int) { return FN(v); });              \
  }                                                                                                        \
  VGPU_PLAIN(nppiGamma##DIR##_8u_AC4IR, (Npp8u * sd, int ss, NppiSize roi), (sd, ss, roi, NppStreamContext{}))
VGPU_NPP_GAMMA(Fwd, gamma_fwd)
VGPU_NPP_GAMMA(Inv, gamma_inv)

VGPU_EXPORT NppStatus nppiSwapChannels_8u_C4IR_Ctx(Npp8u* sd, int ss, NppiSize roi, const int order[4],
                                                   NppStreamContext) {
  if (!sd || !order) return NPP_NULL_POINTER_ERROR;
  if (bad(roi)) return NPP_SIZE_ERROR;
  for (int c = 0; c < 4; ++c)
    if (order[c] < 0 || order[c] > 3) return NPP_CHANNEL_ORDER_ERROR;
  auto px = fetch<Npp8u>(sd, ss, roi.width * 4, roi.height);
  for (size_t i = 0; i < px.size(); i += 4) {
    const Npp8u v[4] = {px[i], px[i + 1], px[i + 2], px[i + 3]};
    for (int c = 0; c < 4; ++c) px[i + c] = v[order[c]];
  }
  store<Npp8u>(sd, ss, roi.width * 4, roi.height, px);
  return NPP_SUCCESS;
}
VGPU_PLAIN(nppiSwapChannels_8u_C4IR, (Npp8u * sd, int ss, NppiSize roi, const int order[4]),
           (sd, ss, roi, order, NppStreamContext{}))

// YCbCr 4:2:0 between three planes and NV12's luma plus interleaved CbCr.
VGPU_EXPORT NppStatus nppiYCbCr420_8u_P3P2R_Ctx(const Npp8u* const src[3], int sstep[3], Npp8u* dy, int dys,
                                                Npp8u* dc, int dcs, NppiSize roi, NppStreamContext) {
  if (!src || !sstep || !src[0] || !src[1] || !src[2] || !dy || !dc) return NPP_NULL_POINTER_ERROR;
  if (bad(roi)) return NPP_SIZE_ERROR;
  const int cw = roi.width / 2, chh = roi.height / 2;
  store<Npp8u>(dy, dys, roi.width, roi.height, fetch<Npp8u>(src[0], sstep[0], roi.width, roi.height));
  auto cb = fetch<Npp8u>(src[1], sstep[1], cw, chh);
  auto cr = fetch<Npp8u>(src[2], sstep[2], cw, chh);
  std::vector<Npp8u> uv(static_cast<size_t>(cw) * 2 * chh);
  for (size_t i = 0; i < cb.size(); ++i) {
    uv[2 * i] = cb[i];
    uv[2 * i + 1] = cr[i];
  }
  store<Npp8u>(dc, dcs, cw * 2, chh, uv);
  return NPP_SUCCESS;
}
VGPU_PLAIN(nppiYCbCr420_8u_P3P2R,
           (const Npp8u* const src[3], int sstep[3], Npp8u* dy, int dys, Npp8u* dc, int dcs, NppiSize roi),
           (src, sstep, dy, dys, dc, dcs, roi, NppStreamContext{}))
VGPU_EXPORT NppStatus nppiYCbCr420_8u_P2P3R_Ctx(const Npp8u* const sy, int sys, const Npp8u* sc, int scs,
                                                Npp8u* dst[3], int dstep[3], NppiSize roi, NppStreamContext) {
  if (!sy || !sc || !dst || !dstep || !dst[0] || !dst[1] || !dst[2]) return NPP_NULL_POINTER_ERROR;
  if (bad(roi)) return NPP_SIZE_ERROR;
  const int cw = roi.width / 2, chh = roi.height / 2;
  store<Npp8u>(dst[0], dstep[0], roi.width, roi.height, fetch<Npp8u>(sy, sys, roi.width, roi.height));
  auto uv = fetch<Npp8u>(sc, scs, cw * 2, chh);
  std::vector<Npp8u> cb(static_cast<size_t>(cw) * chh), cr(cb.size());
  for (size_t i = 0; i < cb.size(); ++i) {
    cb[i] = uv[2 * i];
    cr[i] = uv[2 * i + 1];
  }
  store<Npp8u>(dst[1], dstep[1], cw, chh, cb);
  store<Npp8u>(dst[2], dstep[2], cw, chh, cr);
  return NPP_SUCCESS;
}
VGPU_PLAIN(nppiYCbCr420_8u_P2P3R,
           (const Npp8u* const sy, int sys, const Npp8u* sc, int scs, Npp8u* dst[3], int dstep[3], NppiSize roi),
           (sy, sys, sc, scs, dst, dstep, roi, NppStreamContext{}))

// Piecewise-linear lookup through (pLevels[i], pValues[i]). Inputs below the
// first level or at/after the last pass through unchanged (measured: levels
// 20..200 leave 0..19 and 200..255 as they were). The tables may be in host
// or device memory; NVIDIA's reads either.
VGPU_EXPORT NppStatus nppiLUT_Linear_8u_C1R_Ctx(const Npp8u* s, int ss, Npp8u* d, int ds, NppiSize roi,
                                                const Npp32s* values, const Npp32s* levels, int n,
                                                NppStreamContext) {
  if (!s || !d || !values || !levels) return NPP_NULL_POINTER_ERROR;
  if (bad(roi)) return NPP_SIZE_ERROR;
  if (n < 2) return NPP_LUT_NUMBER_OF_LEVELS_ERROR;
  std::vector<Npp32s> lv(n), vv(n);
  get_n(lv.data(), levels, n);
  get_n(vv.data(), values, n);
  Npp8u table[256];
  for (int x = 0; x < 256; ++x) {
    table[x] = static_cast<Npp8u>(x);
    for (int k = 0; k + 1 < n; ++k) {
      if (x >= lv[k] && x < lv[k + 1]) {
        // Integer division, so the step truncates toward zero whichever way
        // the segment slopes (measured: 199 on a 250->5 segment is 7, not 6).
        const long long step = static_cast<long long>(x - lv[k]) * (vv[k + 1] - vv[k]) / (lv[k + 1] - lv[k]);
        table[x] = saturate<Npp8u>(static_cast<double>(vv[k] + step));
        break;
      }
    }
  }
  return per_channel<Npp8u>(s, ss, d, ds, roi, 1, 1, [&](Npp8u v, int) { return table[v]; });
}
VGPU_PLAIN(nppiLUT_Linear_8u_C1R,
           (const Npp8u* s, int ss, Npp8u* d, int ds, NppiSize roi, const Npp32s* values, const Npp32s* levels,
            int n),
           (s, ss, d, ds, roi, values, levels, n, NppStreamContext{}))

/* ======================================================================
   Statistics (nppist): masked and float mean/standard deviation,
   histograms, integral images, windowed sums.
   ====================================================================== */

namespace {
// Scratch sizes: NPP wants a device buffer it sizes; this implementation
// needs none, but a size of zero would make a caller's allocation fail.
constexpr int kScratch = 4096;

template <class T>
NppStatus mean_stddev(const T* s, int ss, const Npp8u* mask, int ms, NppiSize roi, Npp64f* mean, Npp64f* sd) {
  if (!s || !mean || !sd) return NPP_NULL_POINTER_ERROR;
  if (bad(roi)) return NPP_SIZE_ERROR;
  auto v = fetch<T>(s, ss, roi.width, roi.height);
  std::vector<Npp8u> m;
  if (mask) m = fetch<Npp8u>(mask, ms, roi.width, roi.height);
  double sum = 0, sq = 0;
  size_t n = 0;
  for (size_t i = 0; i < v.size(); ++i) {
    if (mask && !m[i]) continue;
    sum += v[i];
    sq += double(v[i]) * v[i];
    ++n;
  }
  const double mu = n ? sum / n : 0;
  // Population deviation, as npp_api.cpp's 8-bit form.
  const double var = n ? std::max(0.0, sq / n - mu * mu) : 0;
  put<Npp64f>(mean, mu);
  put<Npp64f>(sd, std::sqrt(var));
  return NPP_SUCCESS;
}
}  // namespace

VGPU_EXPORT NppStatus nppiMean_StdDev_8u_C1MR_Ctx(const Npp8u* s, int ss, const Npp8u* m, int ms, NppiSize roi,
                                                  Npp8u*, Npp64f* mean, Npp64f* sd, NppStreamContext) {
  if (!m) return NPP_NULL_POINTER_ERROR;
  return mean_stddev<Npp8u>(s, ss, m, ms, roi, mean, sd);
}
VGPU_PLAIN(nppiMean_StdDev_8u_C1MR,
           (const Npp8u* s, int ss, const Npp8u* m, int ms, NppiSize roi, Npp8u* b, Npp64f* mean, Npp64f* sd),
           (s, ss, m, ms, roi, b, mean, sd, NppStreamContext{}))
VGPU_EXPORT NppStatus nppiMean_StdDev_32f_C1MR_Ctx(const Npp32f* s, int ss, const Npp8u* m, int ms, NppiSize roi,
                                                   Npp8u*, Npp64f* mean, Npp64f* sd, NppStreamContext) {
  if (!m) return NPP_NULL_POINTER_ERROR;
  return mean_stddev<Npp32f>(s, ss, m, ms, roi, mean, sd);
}
VGPU_PLAIN(nppiMean_StdDev_32f_C1MR,
           (const Npp32f* s, int ss, const Npp8u* m, int ms, NppiSize roi, Npp8u* b, Npp64f* mean, Npp64f* sd),
           (s, ss, m, ms, roi, b, mean, sd, NppStreamContext{}))
VGPU_EXPORT NppStatus nppiMean_StdDev_32f_C1R_Ctx(const Npp32f* s, int ss, NppiSize roi, Npp8u*, Npp64f* mean,
                                                  Npp64f* sd, NppStreamContext) {
  return mean_stddev<Npp32f>(s, ss, nullptr, 0, roi, mean, sd);
}
VGPU_PLAIN(nppiMean_StdDev_32f_C1R, (const Npp32f* s, int ss, NppiSize roi, Npp8u* b, Npp64f* mean, Npp64f* sd),
           (s, ss, roi, b, mean, sd, NppStreamContext{}))

#define VGPU_NPP_HOSTSIZE(NAME)                                                                       \
  VGPU_EXPORT NppStatus NAME##_Ctx(NppiSize roi,                                                      \
                                   arg_of<1, decltype(&NAME##_Ctx)>::type size, NppStreamContext) {   \
    if (!size) return NPP_NULL_POINTER_ERROR;                                                         \
    if (bad(roi)) return NPP_SIZE_ERROR;                                                              \
    *size = kScratch;                                                                                 \
    return NPP_SUCCESS;                                                                               \
  }                                                                                                   \
  VGPU_PLAIN(NAME, (NppiSize roi, arg_of<1, decltype(&NAME##_Ctx)>::type size),                       \
             (roi, size, NppStreamContext{}))
VGPU_NPP_HOSTSIZE(nppiMeanStdDevGetBufferHostSize_8u_C1MR)
VGPU_NPP_HOSTSIZE(nppiMeanStdDevGetBufferHostSize_32f_C1R)
VGPU_NPP_HOSTSIZE(nppiMeanStdDevGetBufferHostSize_32f_C1MR)

// Even levels: nLevels boundaries from lower to upper. The span divides into
// equal integer steps with the remainder given one each to the first bins
// (measured: 7 levels over 0..255 are 0 43 86 129 171 213 255).
VGPU_EXPORT NppStatus nppiEvenLevelsHost_32s(Npp32s* levels, int n, Npp32s lower, Npp32s upper) {
  if (!levels) return NPP_NULL_POINTER_ERROR;
  if (n < 2) return NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR;
  const long long span = static_cast<long long>(upper) - lower;
  const long long step = span / (n - 1), rem = span % (n - 1);
  long long v = lower;
  for (int i = 0; i < n; ++i) {
    levels[i] = static_cast<Npp32s>(v);
    v += step + (i < rem ? 1 : 0);
  }
  return NPP_SUCCESS;
}

namespace {
// HistogramEven's own bins are not nppiEvenLevelsHost's: every bin is
// floor((upper - lower) / (nLevels - 1)) wide, so values past the last whole
// bin are not counted (measured: 17 levels over 3..250 count 15 values in each
// of 16 bins, 240 in all).
std::vector<Npp32s> even_bins(int n, Npp32s lo, Npp32s hi) {
  std::vector<Npp32s> lv(n);
  const long long step = (static_cast<long long>(hi) - lo) / (n - 1);
  for (int i = 0; i < n; ++i) lv[i] = static_cast<Npp32s>(lo + step * i);
  return lv;
}

// Counts each channel into bins [level[i], level[i+1]); values outside every
// bin are not counted.
template <class T, class L>
NppStatus histogram(const T* s, int ss, NppiSize roi, int ch, Npp32s* const* hist, const std::vector<L>* levels) {
  if (!s) return NPP_NULL_POINTER_ERROR;
  if (bad(roi)) return NPP_SIZE_ERROR;
  // Measured on the 8-bit four-channel form: a step that is not a whole
  // number of pixels is refused.
  if (ch == 4 && ss % (4 * static_cast<int>(sizeof(T)))) return NPP_NOT_EVEN_STEP_ERROR;
  auto v = fetch<T>(s, ss, roi.width * ch, roi.height);
  for (int c = 0; c < ch; ++c) {
    const auto& lv = levels[c];
    if (!hist[c]) return NPP_NULL_POINTER_ERROR;
    if (lv.size() < 2) return NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR;
    std::vector<Npp32s> bins(lv.size() - 1, 0);
    for (size_t i = c; i < v.size(); i += ch) {
      const auto x = v[i];
      // Binary search for the last level <= x.
      auto it = std::upper_bound(lv.begin(), lv.end(), static_cast<L>(x));
      if (it == lv.begin() || it == lv.end()) continue;
      if (std::is_floating_point_v<T> && std::isnan(double(x))) continue;
      ++bins[static_cast<size_t>(it - lv.begin()) - 1];
    }
    put_n<Npp32s>(hist[c], bins.data(), bins.size());
  }
  return NPP_SUCCESS;
}
}  // namespace

#define VGPU_NPP_HISTEVEN1(SFX, T)                                                                         \
  VGPU_EXPORT NppStatus nppiHistogramEven_##SFX##_Ctx(const T* s, int ss, NppiSize roi, Npp32s* hist, int n, \
                                                      Npp32s lo, Npp32s hi, Npp8u*, NppStreamContext) {     \
    if (n < 2) return NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR;                                                \
    std::vector<Npp32s> lv = even_bins(n, lo, hi);                                                         \
    return histogram<T, Npp32s>(s, ss, roi, 1, &hist, &lv);                                                \
  }                                                                                                        \
  VGPU_PLAIN(nppiHistogramEven_##SFX,                                                                       \
             (const T* s, int ss, NppiSize roi, Npp32s* hist, int n, Npp32s lo, Npp32s hi, Npp8u* b),      \
             (s, ss, roi, hist, n, lo, hi, b, NppStreamContext{}))
#define VGPU_NPP_HISTEVEN4(SFX, T)                                                                         \
  VGPU_EXPORT NppStatus nppiHistogramEven_##SFX##_Ctx(const T* s, int ss, NppiSize roi, Npp32s* hist[4],     \
                                                      int n[4], Npp32s lo[4], Npp32s hi[4], Npp8u*,         \
                                                      NppStreamContext) {                                  \
    if (!hist || !n || !lo || !hi) return NPP_NULL_POINTER_ERROR;                                          \
    std::vector<Npp32s> lv[4];                                                                             \
    for (int c = 0; c < 4; ++c) {                                                                          \
      if (n[c] < 2) return NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR;                                           \
      lv[c] = even_bins(n[c], lo[c], hi[c]);                                                               \
    }                                                                                                      \
    return histogram<T, Npp32s>(s, ss, roi, 4, hist, lv);                                                  \
  }                                                                                                        \
  VGPU_PLAIN(nppiHistogramEven_##SFX,                                                                       \
             (const T* s, int ss, NppiSize roi, Npp32s* hist[4], int n[4], Npp32s lo[4], Npp32s hi[4],     \
              Npp8u* b),                                                                                   \
             (s, ss, roi, hist, n, lo, hi, b, NppStreamContext{}))
VGPU_NPP_HISTEVEN1(8u_C1R, Npp8u)
VGPU_NPP_HISTEVEN1(16u_C1R, Npp16u)
VGPU_NPP_HISTEVEN1(16s_C1R, Npp16s)
VGPU_NPP_HISTEVEN4(8u_C4R, Npp8u)
VGPU_NPP_HISTEVEN4(16u_C4R, Npp16u)
VGPU_NPP_HISTEVEN4(16s_C4R, Npp16s)

#define VGPU_NPP_HISTRANGE1(SFX, T, L)                                                                     \
  VGPU_EXPORT NppStatus nppiHistogramRange_##SFX##_Ctx(const T* s, int ss, NppiSize roi, Npp32s* hist,        \
                                                       const L* levels, int n, Npp8u*, NppStreamContext) {   \
    if (!levels) return NPP_NULL_POINTER_ERROR;                                                            \
    if (n < 2) return NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR;                                                \
    std::vector<L> lv(n);                                                                                  \
    get_n(lv.data(), levels, n);                                                                           \
    return histogram<T, L>(s, ss, roi, 1, &hist, &lv);                                                     \
  }                                                                                                        \
  VGPU_PLAIN(nppiHistogramRange_##SFX,                                                                      \
             (const T* s, int ss, NppiSize roi, Npp32s* hist, const L* levels, int n, Npp8u* b),           \
             (s, ss, roi, hist, levels, n, b, NppStreamContext{}))
#define VGPU_NPP_HISTRANGE4(SFX, T, L)                                                                     \
  VGPU_EXPORT NppStatus nppiHistogramRange_##SFX##_Ctx(const T* s, int ss, NppiSize roi, Npp32s* hist[4],     \
                                                       const L* levels[4], int n[4], Npp8u*,                 \
                                                       NppStreamContext) {                                  \
    if (!hist || !levels || !n) return NPP_NULL_POINTER_ERROR;                                             \
    std::vector<L> lv[4];                                                                                  \
    for (int c = 0; c < 4; ++c) {                                                                          \
      if (!levels[c]) return NPP_NULL_POINTER_ERROR;                                                       \
      if (n[c] < 2) return NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR;                                           \
      lv[c].resize(n[c]);                                                                                  \
      get_n(lv[c].data(), levels[c], n[c]);                                                                \
    }                                                                                                      \
    return histogram<T, L>(s, ss, roi, 4, hist, lv);                                                       \
  }                                                                                                        \
  VGPU_PLAIN(nppiHistogramRange_##SFX,                                                                      \
             (const T* s, int ss, NppiSize roi, Npp32s* hist[4], const L* levels[4], int n[4], Npp8u* b),  \
             (s, ss, roi, hist, levels, n, b, NppStreamContext{}))
VGPU_NPP_HISTRANGE1(8u_C1R, Npp8u, Npp32s)
VGPU_NPP_HISTRANGE1(16u_C1R, Npp16u, Npp32s)
VGPU_NPP_HISTRANGE1(16s_C1R, Npp16s, Npp32s)
VGPU_NPP_HISTRANGE1(32f_C1R, Npp32f, Npp32f)
VGPU_NPP_HISTRANGE4(8u_C4R, Npp8u, Npp32s)
VGPU_NPP_HISTRANGE4(16u_C4R, Npp16u, Npp32s)
VGPU_NPP_HISTRANGE4(16s_C4R, Npp16s, Npp32s)
VGPU_NPP_HISTRANGE4(32f_C4R, Npp32f, Npp32f)

// NPP 12.0 declares these _Ctx forms without the stream context (and with an
// int* size); NPP 12.3 (CUDA 12.8) and later have both. The version macro is
// the only way to tell the two declarations apart.
#if NPP_VER_MAJOR == 12 && NPP_VER_MINOR == 0
#define VGPU_NPP_HISTSIZE_PARAMS(N) (NppiSize roi, N, int* size)
#define VGPU_NPP_HISTSIZE_CTX_PARAMS(N) (NppiSize roi, N, int* size)
#else
#define VGPU_NPP_HISTSIZE_PARAMS(N) (NppiSize roi, N, size_t* size)
#define VGPU_NPP_HISTSIZE_CTX_PARAMS(N) (NppiSize roi, N, size_t* size, NppStreamContext)
#endif
#define VGPU_NPP_HISTSIZE1(NAME)                                         \
  VGPU_EXPORT NppStatus NAME##_Ctx VGPU_NPP_HISTSIZE_CTX_PARAMS(int n) { \
    if (!size) return NPP_NULL_POINTER_ERROR;                            \
    if (bad(roi)) return NPP_SIZE_ERROR;                                 \
    if (n < 2) return NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR;              \
    *size = kScratch;                                                    \
    return NPP_SUCCESS;                                                  \
  }                                                                      \
  VGPU_EXPORT NppStatus NAME VGPU_NPP_HISTSIZE_PARAMS(int n) {           \
    if (!size) return NPP_NULL_POINTER_ERROR;                            \
    if (bad(roi)) return NPP_SIZE_ERROR;                                 \
    if (n < 2) return NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR;              \
    *size = kScratch;                                                    \
    return NPP_SUCCESS;                                                  \
  }
#define VGPU_NPP_HISTSIZE4(NAME)                                            \
  VGPU_EXPORT NppStatus NAME##_Ctx VGPU_NPP_HISTSIZE_CTX_PARAMS(int n[4]) { \
    if (!size || !n) return NPP_NULL_POINTER_ERROR;                         \
    if (bad(roi)) return NPP_SIZE_ERROR;                                    \
    *size = kScratch;                                                       \
    return NPP_SUCCESS;                                                     \
  }                                                                         \
  VGPU_EXPORT NppStatus NAME VGPU_NPP_HISTSIZE_PARAMS(int n[4]) {           \
    if (!size || !n) return NPP_NULL_POINTER_ERROR;                         \
    if (bad(roi)) return NPP_SIZE_ERROR;                                    \
    *size = kScratch;                                                       \
    return NPP_SUCCESS;                                                     \
  }
VGPU_NPP_HISTSIZE1(nppiHistogramEvenGetBufferSize_8u_C1R)
VGPU_NPP_HISTSIZE1(nppiHistogramEvenGetBufferSize_16u_C1R)
VGPU_NPP_HISTSIZE1(nppiHistogramEvenGetBufferSize_16s_C1R)
VGPU_NPP_HISTSIZE4(nppiHistogramEvenGetBufferSize_8u_C4R)
VGPU_NPP_HISTSIZE4(nppiHistogramEvenGetBufferSize_16u_C4R)
VGPU_NPP_HISTSIZE4(nppiHistogramEvenGetBufferSize_16s_C4R)
VGPU_NPP_HISTSIZE1(nppiHistogramRangeGetBufferSize_8u_C1R)
VGPU_NPP_HISTSIZE1(nppiHistogramRangeGetBufferSize_16u_C1R)
VGPU_NPP_HISTSIZE1(nppiHistogramRangeGetBufferSize_16s_C1R)
VGPU_NPP_HISTSIZE1(nppiHistogramRangeGetBufferSize_32f_C1R)
VGPU_NPP_HISTSIZE4(nppiHistogramRangeGetBufferSize_8u_C4R)
VGPU_NPP_HISTSIZE4(nppiHistogramRangeGetBufferSize_16u_C4R)
VGPU_NPP_HISTSIZE4(nppiHistogramRangeGetBufferSize_16s_C4R)
VGPU_NPP_HISTSIZE4(nppiHistogramRangeGetBufferSize_32f_C4R)

// Integral image: (w+1) x (h+1), first row and column nVal, then running sums
// on top of it.
#define VGPU_NPP_INTEGRAL(SFX, D)                                                                          \
  VGPU_EXPORT NppStatus nppiIntegral_##SFX##_Ctx(const Npp8u* s, int ss, D* d, int ds, NppiSize roi, D val,  \
                                                 NppStreamContext) {                                       \
    if (!s || !d) return NPP_NULL_POINTER_ERROR;                                                           \
    if (bad(roi)) return NPP_SIZE_ERROR;                                                                   \
    auto v = fetch<Npp8u>(s, ss, roi.width, roi.height);                                                   \
    const int W = roi.width + 1, H = roi.height + 1;                                                       \
    std::vector<D> o(static_cast<size_t>(W) * H, val);                                                     \
    for (int y = 1; y < H; ++y) {                                                                          \
      D row = 0;                                                                                           \
      for (int x = 1; x < W; ++x) {                                                                        \
        row += v[static_cast<size_t>(y - 1) * roi.width + x - 1];                                          \
        o[static_cast<size_t>(y) * W + x] = o[static_cast<size_t>(y - 1) * W + x] + row;                   \
      }                                                                                                    \
    }                                                                                                      \
    store<D>(d, ds, W, H, o);                                                                              \
    return NPP_SUCCESS;                                                                                    \
  }                                                                                                        \
  VGPU_PLAIN(nppiIntegral_##SFX, (const Npp8u* s, int ss, D* d, int ds, NppiSize roi, D val),              \
             (s, ss, d, ds, roi, val, NppStreamContext{}))
VGPU_NPP_INTEGRAL(8u32s_C1R, Npp32s)
VGPU_NPP_INTEGRAL(8u32f_C1R, Npp32f)

// Standard deviation over oRect at each ROI position, from an integral image
// and an integral of squares (as nppiSqrIntegral makes them).
VGPU_EXPORT NppStatus nppiRectStdDev_32s32f_C1R_Ctx(const Npp32s* s, int ss, const Npp64f* sq, int sqs, Npp32f* d,
                                                    int ds, NppiSize roi, NppiRect r, NppStreamContext) {
  if (!s || !sq || !d) return NPP_NULL_POINTER_ERROR;
  if (bad(roi) || r.width <= 0 || r.height <= 0) return NPP_SIZE_ERROR;
  const int W = roi.width + r.x + r.width, H = roi.height + r.y + r.height;
  auto I = fetch<Npp32s>(s, ss, W, H);
  auto Q = fetch<Npp64f>(sq, sqs, W, H);
  const double n = double(r.width) * r.height;
  std::vector<Npp32f> o(static_cast<size_t>(roi.width) * roi.height);
  auto at = [&](const auto& v, int x, int y) { return double(v[static_cast<size_t>(y) * W + x]); };
  for (int y = 0; y < roi.height; ++y)
    for (int x = 0; x < roi.width; ++x) {
      const int x0 = x + r.x, y0 = y + r.y, x1 = x0 + r.width, y1 = y0 + r.height;
      const double sum = at(I, x1, y1) - at(I, x0, y1) - at(I, x1, y0) + at(I, x0, y0);
      const double s2 = at(Q, x1, y1) - at(Q, x0, y1) - at(Q, x1, y0) + at(Q, x0, y0);
      const double var = s2 / n - (sum / n) * (sum / n);
      o[static_cast<size_t>(y) * roi.width + x] = static_cast<Npp32f>(std::sqrt(std::max(0.0, var)));
    }
  store<Npp32f>(d, ds, roi.width, roi.height, o);
  return NPP_SUCCESS;
}
VGPU_PLAIN(nppiRectStdDev_32s32f_C1R,
           (const Npp32s* s, int ss, const Npp64f* sq, int sqs, Npp32f* d, int ds, NppiSize roi, NppiRect r),
           (s, ss, sq, sqs, d, ds, roi, r, NppStreamContext{}))

// Sums over a 1-D window: the source pointer is the first ROI pixel and the
// window reads nAnchor pixels before it, as the filters do.
namespace {
NppStatus sum_window(const Npp8u* s, int ss, Npp32f* d, int ds, NppiSize roi, int mask, int anchor, bool rows) {
  if (!s || !d) return NPP_NULL_POINTER_ERROR;
  if (bad(roi)) return NPP_SIZE_ERROR;
  if (mask <= 0) return NPP_MASK_SIZE_ERROR;
  if (anchor < 0 || anchor >= mask) return NPP_ANCHOR_ERROR;
  const int w = rows ? roi.width + mask - 1 : roi.width, h = rows ? roi.height : roi.height + mask - 1;
  const Npp8u* base = rows ? offset(s, ss, -anchor, 0, 1) : offset(s, ss, 0, -anchor, 1);
  auto v = fetch<Npp8u>(base, ss, w, h);
  std::vector<Npp32f> o(static_cast<size_t>(roi.width) * roi.height);
  for (int y = 0; y < roi.height; ++y)
    for (int x = 0; x < roi.width; ++x) {
      float acc = 0;
      for (int k = 0; k < mask; ++k)
        acc += rows ? v[static_cast<size_t>(y) * w + x + k] : v[static_cast<size_t>(y + k) * w + x];
      o[static_cast<size_t>(y) * roi.width + x] = acc;
    }
  store<Npp32f>(d, ds, roi.width, roi.height, o);
  return NPP_SUCCESS;
}
}  // namespace
VGPU_EXPORT NppStatus nppiSumWindowRow_8u32f_C1R_Ctx(const Npp8u* s, Npp32s ss, Npp32f* d, Npp32s ds, NppiSize roi,
                                                     Npp32s mask, Npp32s anchor, NppStreamContext) {
  return sum_window(s, ss, d, ds, roi, mask, anchor, true);
}
VGPU_PLAIN(nppiSumWindowRow_8u32f_C1R,
           (const Npp8u* s, Npp32s ss, Npp32f* d, Npp32s ds, NppiSize roi, Npp32s mask, Npp32s anchor),
           (s, ss, d, ds, roi, mask, anchor, NppStreamContext{}))
VGPU_EXPORT NppStatus nppiSumWindowColumn_8u32f_C1R_Ctx(const Npp8u* s, Npp32s ss, Npp32f* d, Npp32s ds,
                                                        NppiSize roi, Npp32s mask, Npp32s anchor, NppStreamContext) {
  return sum_window(s, ss, d, ds, roi, mask, anchor, false);
}
VGPU_PLAIN(nppiSumWindowColumn_8u32f_C1R,
           (const Npp8u* s, Npp32s ss, Npp32f* d, Npp32s ds, NppiSize roi, Npp32s mask, Npp32s anchor),
           (s, ss, d, ds, roi, mask, anchor, NppStreamContext{}))

/* ======================================================================
   Filters (nppif) and morphology (nppim). The fixed-ROI forms read the
   pixels around the ROI (the caller's source pointer is the first ROI pixel
   and the image must extend past it by the mask); the *Border forms take
   the whole source image and synthesise what lies beyond it.
   ====================================================================== */

namespace {
// The ROI plus the margin a mask needs, fetched.
template <class T>
std::vector<T> fetch_halo(const T* s, int ss, NppiSize roi, NppiSize mask, NppiPoint anchor, int ch, int* w) {
  *w = roi.width + mask.width - 1;
  const int h = roi.height + mask.height - 1;
  return fetch<T>(offset(s, ss, -anchor.x, -anchor.y, ch), ss, *w * ch, h);
}

// Applies reduce() over every mask position (where `mask` is null or
// nonzero) for each channel of each ROI pixel.
template <class T, class Init, class Step, class Done>
NppStatus neighbourhood(const T* s, int ss, T* d, int ds, NppiSize roi, NppiSize msz, NppiPoint anchor,
                        const Npp8u* mask, int ch, Init init, Step step, Done done) {
  if (!s || !d) return NPP_NULL_POINTER_ERROR;
  if (bad(roi)) return NPP_SIZE_ERROR;
  if (msz.width <= 0 || msz.height <= 0) return NPP_MASK_SIZE_ERROR;
  if (anchor.x < 0 || anchor.y < 0 || anchor.x >= msz.width || anchor.y >= msz.height) return NPP_ANCHOR_ERROR;
  std::vector<Npp8u> m;
  if (mask) {
    m.resize(static_cast<size_t>(msz.width) * msz.height);
    get_n(m.data(), mask, m.size());
  }
  int w = 0;
  auto src = fetch_halo(s, ss, roi, msz, anchor, ch, &w);
  std::vector<T> dst(static_cast<size_t>(roi.width) * roi.height * ch);
  for (int y = 0; y < roi.height; ++y)
    for (int x = 0; x < roi.width; ++x)
      for (int c = 0; c < ch; ++c) {
        auto acc = init();
        for (int j = 0; j < msz.height; ++j)
          for (int i = 0; i < msz.width; ++i) {
            if (mask && !m[static_cast<size_t>(j) * msz.width + i]) continue;
            acc = step(acc, src[(static_cast<size_t>(y + j) * w + x + i) * ch + c]);
          }
        dst[(static_cast<size_t>(y) * roi.width + x) * ch + c] = done(acc);
      }
  store<T>(d, ds, roi.width * ch, roi.height, dst);
  return NPP_SUCCESS;
}
}  // namespace

// Box filter: 8-bit truncates the mean, as npp_api.cpp's C1 form measured.
#define VGPU_NPP_BOX(SFX, T, CH)                                                                           \
  VGPU_EXPORT NppStatus nppiFilterBox_##SFX##_Ctx(const T* s, Npp32s ss, T* d, Npp32s ds, NppiSize roi,      \
                                                  NppiSize m, NppiPoint a, NppStreamContext) {             \
    const double n = double(m.width) * m.height;                                                           \
    return neighbourhood<T>(                                                                               \
        s, ss, d, ds, roi, m, a, nullptr, CH, [] { return 0.0; }, [](double acc, T v) { return acc + v; },  \
        [n](double acc) {                                                                                  \
          if constexpr (std::is_floating_point_v<T>)                                                       \
            return static_cast<T>(acc / n);                                                                \
          else                                                                                             \
            return saturate<T>(std::floor(acc / n));                                                       \
        });                                                                                                \
  }                                                                                                        \
  VGPU_PLAIN(nppiFilterBox_##SFX,                                                                           \
             (const T* s, Npp32s ss, T* d, Npp32s ds, NppiSize roi, NppiSize m, NppiPoint a),              \
             (s, ss, d, ds, roi, m, a, NppStreamContext{}))
VGPU_NPP_BOX(8u_C4R, Npp8u, 4)
VGPU_NPP_BOX(32f_C1R, Npp32f, 1)

#define VGPU_NPP_RANK(NAME, PICK, INIT, SFX, T, CH)                                                        \
  VGPU_EXPORT NppStatus nppiFilter##NAME##_##SFX##_Ctx(const T* s, Npp32s ss, T* d, Npp32s ds, NppiSize roi, \
                                                       NppiSize m, NppiPoint a, NppStreamContext) {        \
    return neighbourhood<T>(                                                                               \
        s, ss, d, ds, roi, m, a, nullptr, CH, [] { return INIT; },                                         \
        [](T acc, T v) { return PICK(acc, v); }, [](T acc) { return acc; });                               \
  }                                                                                                        \
  VGPU_PLAIN(nppiFilter##NAME##_##SFX,                                                                      \
             (const T* s, Npp32s ss, T* d, Npp32s ds, NppiSize roi, NppiSize m, NppiPoint a),              \
             (s, ss, d, ds, roi, m, a, NppStreamContext{}))
VGPU_NPP_RANK(Max, std::max, Npp8u(0), 8u_C1R, Npp8u, 1)
VGPU_NPP_RANK(Max, std::max, Npp8u(0), 8u_C4R, Npp8u, 4)
VGPU_NPP_RANK(Min, std::min, Npp8u(255), 8u_C1R, Npp8u, 1)
VGPU_NPP_RANK(Min, std::min, Npp8u(255), 8u_C4R, Npp8u, 4)

// Dilate/Erode with an arbitrary mask: the extreme over the positions whose
// mask byte is nonzero.
#define VGPU_NPP_MORPH(NAME, PICK, INIT, SFX, T, CH)                                                       \
  VGPU_EXPORT NppStatus nppi##NAME##_##SFX##_Ctx(const T* s, int ss, T* d, int ds, NppiSize roi,             \
                                                 const Npp8u* mask, NppiSize m, NppiPoint a,                \
                                                 NppStreamContext) {                                       \
    if (!mask) return NPP_NULL_POINTER_ERROR;                                                              \
    return neighbourhood<T>(                                                                               \
        s, ss, d, ds, roi, m, a, mask, CH, [] { return INIT; }, [](T acc, T v) { return PICK(acc, v); },   \
        [](T acc) { return acc; });                                                                        \
  }                                                                                                        \
  VGPU_PLAIN(nppi##NAME##_##SFX,                                                                            \
             (const T* s, int ss, T* d, int ds, NppiSize roi, const Npp8u* mask, NppiSize m, NppiPoint a), \
             (s, ss, d, ds, roi, mask, m, a, NppStreamContext{}))
VGPU_NPP_MORPH(Dilate, std::max, Npp8u(0), 8u_C1R, Npp8u, 1)
VGPU_NPP_MORPH(Dilate, std::max, Npp8u(0), 8u_C4R, Npp8u, 4)
VGPU_NPP_MORPH(Dilate, std::max, -std::numeric_limits<Npp32f>::max(), 32f_C1R, Npp32f, 1)
VGPU_NPP_MORPH(Dilate, std::max, -std::numeric_limits<Npp32f>::max(), 32f_C4R, Npp32f, 4)
VGPU_NPP_MORPH(Erode, std::min, Npp8u(255), 8u_C1R, Npp8u, 1)
VGPU_NPP_MORPH(Erode, std::min, Npp8u(255), 8u_C4R, Npp8u, 4)
VGPU_NPP_MORPH(Erode, std::min, std::numeric_limits<Npp32f>::max(), 32f_C1R, Npp32f, 1)
VGPU_NPP_MORPH(Erode, std::min, std::numeric_limits<Npp32f>::max(), 32f_C4R, Npp32f, 4)

/* ---- border forms ---- */
namespace {
// The source image with `border` pixels synthesised on every side by
// replication, the only border mode NPP's *Border filters take for these
// functions (anything else is NPP_NOT_SUPPORTED_MODE_ERROR).
struct Bordered {
  std::vector<int> px;
  int w = 0, h = 0, b = 0;
  int at(int x, int y) const { return px[static_cast<size_t>(y + b) * w + x + b]; }
};
NppStatus bordered(const Npp8u* s, int ss, NppiSize ssz, NppiPoint off, NppiSize roi, int border,
                   NppiBorderType type, Bordered* out) {
  if (!s) return NPP_NULL_POINTER_ERROR;
  if (bad(ssz) || bad(roi)) return NPP_SIZE_ERROR;
  if (type != NPP_BORDER_REPLICATE) return NPP_NOT_SUPPORTED_MODE_ERROR;
  auto img = fetch<Npp8u>(s, ss, ssz.width, ssz.height);
  // Coordinates relative to the ROI's first pixel, which sits at `off` in the image.
  out->b = border;
  out->w = roi.width + 2 * border;
  out->h = roi.height + 2 * border;
  out->px.resize(static_cast<size_t>(out->w) * out->h);
  for (int y = 0; y < out->h; ++y)
    for (int x = 0; x < out->w; ++x) {
      const int ix = std::min(std::max(off.x + x - border, 0), ssz.width - 1);
      const int iy = std::min(std::max(off.y + y - border, 0), ssz.height - 1);
      out->px[static_cast<size_t>(y) * out->w + x] = img[static_cast<size_t>(iy) * ssz.width + ix];
    }
  return NPP_SUCCESS;
}
}  // namespace

VGPU_EXPORT NppStatus nppiFilterBoxBorder_8u_C1R_Ctx(const Npp8u* s, Npp32s ss, NppiSize ssz, NppiPoint off,
                                                     Npp8u* d, Npp32s ds, NppiSize roi, NppiSize m, NppiPoint a,
                                                     NppiBorderType bt, NppStreamContext) {
  if (!d) return NPP_NULL_POINTER_ERROR;
  if (m.width <= 0 || m.height <= 0) return NPP_MASK_SIZE_ERROR;
  if (a.x < 0 || a.y < 0 || a.x >= m.width || a.y >= m.height) return NPP_ANCHOR_ERROR;
  Bordered b;
  if (NppStatus st = bordered(s, ss, ssz, off, roi, std::max(m.width, m.height), bt, &b)) return st;
  std::vector<Npp8u> o(static_cast<size_t>(roi.width) * roi.height);
  const double n = double(m.width) * m.height;
  for (int y = 0; y < roi.height; ++y)
    for (int x = 0; x < roi.width; ++x) {
      double acc = 0;
      for (int j = 0; j < m.height; ++j)
        for (int i = 0; i < m.width; ++i) acc += b.at(x + i - a.x, y + j - a.y);
      o[static_cast<size_t>(y) * roi.width + x] = saturate<Npp8u>(std::floor(acc / n));
    }
  store<Npp8u>(d, ds, roi.width, roi.height, o);
  return NPP_SUCCESS;
}
VGPU_PLAIN(nppiFilterBoxBorder_8u_C1R,
           (const Npp8u* s, Npp32s ss, NppiSize ssz, NppiPoint off, Npp8u* d, Npp32s ds, NppiSize roi, NppiSize m,
            NppiPoint a, NppiBorderType bt),
           (s, ss, ssz, off, d, ds, roi, m, a, bt, NppStreamContext{}))

// Prewitt gradients, measured with an impulse: x is the left column minus the
// right one and y the bottom row minus the top one, each summed over the 3x3
// neighbourhood; the angle is atan2(y, x) in radians, and the magnitude the
// L1 norm or the L2 norm truncated (sqrt(200) is 14, sqrt(399) 19). The
// infinity norm is NPP_BAD_ARGUMENT_ERROR. Any output may be null.
VGPU_EXPORT NppStatus nppiGradientVectorPrewittBorder_8u16s_C1R_Ctx(
    const Npp8u* s, int ss, NppiSize ssz, NppiPoint off, Npp16s* dx, int dxs, Npp16s* dy, int dys, Npp16s* dm,
    int dms, Npp32f* da, int das, NppiSize roi, NppiMaskSize msz, NppiNorm norm, NppiBorderType bt,
    NppStreamContext) {
  if (msz != NPP_MASK_SIZE_3_X_3) return NPP_MASK_SIZE_ERROR;
  if (norm != nppiNormL1 && norm != nppiNormL2) return NPP_BAD_ARGUMENT_ERROR;
  Bordered b;
  if (NppStatus st = bordered(s, ss, ssz, off, roi, 1, bt, &b)) return st;
  const size_t n = static_cast<size_t>(roi.width) * roi.height;
  std::vector<Npp16s> gx(n), gy(n), mag(n);
  std::vector<Npp32f> ang(n);
  for (int y = 0; y < roi.height; ++y)
    for (int x = 0; x < roi.width; ++x) {
      int sx = 0, sy = 0;
      for (int k = -1; k <= 1; ++k) {
        sx += b.at(x - 1, y + k) - b.at(x + 1, y + k);
        sy += b.at(x + k, y + 1) - b.at(x + k, y - 1);
      }
      const size_t i = static_cast<size_t>(y) * roi.width + x;
      gx[i] = static_cast<Npp16s>(sx);
      gy[i] = static_cast<Npp16s>(sy);
      const double m = norm == nppiNormL1 ? std::abs(sx) + std::abs(sy)
                                          : std::floor(std::sqrt(double(sx) * sx + double(sy) * sy));
      mag[i] = saturate<Npp16s>(m);
      ang[i] = static_cast<Npp32f>(std::atan2(double(sy), double(sx)));
    }
  if (dx) store<Npp16s>(dx, dxs, roi.width, roi.height, gx);
  if (dy) store<Npp16s>(dy, dys, roi.width, roi.height, gy);
  if (dm) store<Npp16s>(dm, dms, roi.width, roi.height, mag);
  if (da) store<Npp32f>(da, das, roi.width, roi.height, ang);
  return NPP_SUCCESS;
}
VGPU_PLAIN(nppiGradientVectorPrewittBorder_8u16s_C1R,
           (const Npp8u* s, int ss, NppiSize ssz, NppiPoint off, Npp16s* dx, int dxs, Npp16s* dy, int dys,
            Npp16s* dm, int dms, Npp32f* da, int das, NppiSize roi, NppiMaskSize msz, NppiNorm norm,
            NppiBorderType bt),
           (s, ss, ssz, off, dx, dxs, dy, dys, dm, dms, da, das, roi, msz, norm, bt, NppStreamContext{}))

/* ---- Canny ----
   Sobel (or Scharr) 3x3 gradients over a replicated border, magnitude by
   the L1 or L2 norm, and non-maximum suppression along the gradient direction
   quantised to 0, 45, 90 and 135 degrees, keeping a pixel that is no smaller
   than either neighbour (so a sharp step leaves a two-pixel line, as on
   hardware), on magnitudes truncated to integers. Measured on NPP 13.0: a
   pixel is an edge when it survives
   suppression and its magnitude is at least nLowThreshold -- the high
   threshold changes nothing (an isolated step of magnitude 40 is an edge with
   thresholds 30 and 32767 alike), so there is no hysteresis to reproduce.
   Edges are 255, everything else 0. */
VGPU_EXPORT NppStatus nppiFilterCannyBorderGetBufferSize(NppiSize roi, arg_of<1, decltype(&nppiFilterCannyBorderGetBufferSize)>::type size) {
  if (!size) return NPP_NULL_POINTER_ERROR;
  if (bad(roi)) return NPP_SIZE_ERROR;
  *size = static_cast<std::remove_pointer_t<decltype(size)>>(kScratch);
  return NPP_SUCCESS;
}

VGPU_EXPORT NppStatus nppiFilterCannyBorder_8u_C1R_Ctx(const Npp8u* s, int ss, NppiSize ssz, NppiPoint off,
                                                       Npp8u* d, int ds, NppiSize roi, NppiDifferentialKernel kt,
                                                       NppiMaskSize msz, Npp16s lo, Npp16s hi, NppiNorm norm,
                                                       NppiBorderType bt, Npp8u* buffer, NppStreamContext) {
  (void)hi;
  if (!d || !buffer) return NPP_NULL_POINTER_ERROR;
  if (msz != NPP_MASK_SIZE_3_X_3) return NPP_MASK_SIZE_ERROR;
  if (kt != NPP_FILTER_SOBEL && kt != NPP_FILTER_SCHARR) return NPP_NOT_SUPPORTED_MODE_ERROR;
  if (norm != nppiNormL1 && norm != nppiNormL2) return NPP_NOT_SUPPORTED_MODE_ERROR;
  Bordered b;
  if (NppStatus st = bordered(s, ss, ssz, off, roi, 2, bt, &b)) return st;
  const int W = roi.width, H = roi.height;
  // Gradients for the ROI plus a one-pixel ring, so suppression at the ROI
  // edge sees its neighbours.
  const int GW = W + 2, GH = H + 2;
  std::vector<int> gx(static_cast<size_t>(GW) * GH), gy(gx.size());
  std::vector<double> mag(gx.size());
  const int wc = kt == NPP_FILTER_SOBEL ? 1 : 3, wm = kt == NPP_FILTER_SOBEL ? 2 : 10;
  for (int y = -1; y <= H; ++y)
    for (int x = -1; x <= W; ++x) {
      const int sx = wc * (b.at(x + 1, y - 1) - b.at(x - 1, y - 1)) + wm * (b.at(x + 1, y) - b.at(x - 1, y)) +
                     wc * (b.at(x + 1, y + 1) - b.at(x - 1, y + 1));
      const int sy = wc * (b.at(x - 1, y + 1) - b.at(x - 1, y - 1)) + wm * (b.at(x, y + 1) - b.at(x, y - 1)) +
                     wc * (b.at(x + 1, y + 1) - b.at(x + 1, y - 1));
      const size_t i = static_cast<size_t>(y + 1) * GW + x + 1;
      gx[i] = sx;
      gy[i] = sy;
      // The L2 magnitude is truncated to an integer before suppression
      // compares neighbours (measured: two neighbours of magnitude 361.0 and
      // 361.03 both survive on hardware).
      mag[i] = norm == nppiNormL1 ? std::abs(sx) + std::abs(sy) : std::floor(std::sqrt(double(sx) * sx + double(sy) * sy));
    }
  auto M = [&](int x, int y) { return mag[static_cast<size_t>(y + 1) * GW + x + 1]; };
  std::vector<Npp8u> out(static_cast<size_t>(W) * H, 0);
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
      const size_t gi = static_cast<size_t>(y + 1) * GW + x + 1;
      const double m = mag[gi];
      if (m < lo) continue;
      const double ax = std::abs(gx[gi]), ay = std::abs(gy[gi]);
      double n1, n2;
      // tan(22.5) = 0.4142: within it the gradient is horizontal, beyond
      // tan(67.5) vertical, else diagonal by the signs.
      if (ay <= ax * 0.41421356237) {
        n1 = M(x - 1, y);
        n2 = M(x + 1, y);
      } else if (ay >= ax * 2.41421356237) {
        n1 = M(x, y - 1);
        n2 = M(x, y + 1);
      } else if ((gx[gi] > 0) == (gy[gi] > 0)) {
        n1 = M(x - 1, y - 1);
        n2 = M(x + 1, y + 1);
      } else {
        n1 = M(x + 1, y - 1);
        n2 = M(x - 1, y + 1);
      }
      if (m >= n1 && m >= n2) out[static_cast<size_t>(y) * W + x] = 255;
    }
  store<Npp8u>(d, ds, W, H, out);
  return NPP_SUCCESS;
}
VGPU_PLAIN(nppiFilterCannyBorder_8u_C1R,
           (const Npp8u* s, int ss, NppiSize ssz, NppiPoint off, Npp8u* d, int ds, NppiSize roi,
            NppiDifferentialKernel kt, NppiMaskSize msz, Npp16s lo, Npp16s hi, NppiNorm norm, NppiBorderType bt,
            Npp8u* buffer),
           (s, ss, ssz, off, d, ds, roi, kt, msz, lo, hi, norm, bt, buffer, NppStreamContext{}))

/* ======================================================================
   Threshold, compare, borders, Bayer demosaicing, allocation.
   ====================================================================== */

namespace {
inline bool compare(double a, double b, NppCmpOp op) {
  switch (op) {
    case NPP_CMP_LESS: return a < b;
    case NPP_CMP_LESS_EQ: return a <= b;
    case NPP_CMP_EQ: return a == b;
    case NPP_CMP_GREATER_EQ: return a >= b;
    case NPP_CMP_GREATER: return a > b;
    default: return false;
  }
}
}  // namespace

VGPU_EXPORT NppStatus nppiThreshold_32f_C1R_Ctx(const Npp32f* s, int ss, Npp32f* d, int ds, NppiSize roi,
                                                const Npp32f t, NppCmpOp op, NppStreamContext) {
  // Only "less" and "greater" are thresholds in NPP's sense.
  if (op != NPP_CMP_LESS && op != NPP_CMP_GREATER) return NPP_NOT_SUPPORTED_MODE_ERROR;
  return per_channel<Npp32f>(s, ss, d, ds, roi, 1, 1, [&](Npp32f v, int) { return compare(v, t, op) ? t : v; });
}
VGPU_PLAIN(nppiThreshold_32f_C1R, (const Npp32f* s, int ss, Npp32f* d, int ds, NppiSize roi, const Npp32f t, NppCmpOp op),
           (s, ss, d, ds, roi, t, op, NppStreamContext{}))

VGPU_EXPORT NppStatus nppiCompareC_16s_C1R_Ctx(const Npp16s* s, int ss, const Npp16s k, Npp8u* d, int ds,
                                               NppiSize roi, NppCmpOp op, NppStreamContext) {
  if (!s || !d) return NPP_NULL_POINTER_ERROR;
  if (bad(roi)) return NPP_SIZE_ERROR;
  auto v = fetch<Npp16s>(s, ss, roi.width, roi.height);
  std::vector<Npp8u> o(v.size());
  for (size_t i = 0; i < v.size(); ++i) o[i] = compare(v[i], k, op) ? 255 : 0;
  store<Npp8u>(d, ds, roi.width, roi.height, o);
  return NPP_SUCCESS;
}
VGPU_PLAIN(nppiCompareC_16s_C1R, (const Npp16s* s, int ss, const Npp16s k, Npp8u* d, int ds, NppiSize roi, NppCmpOp op),
           (s, ss, k, d, ds, roi, op, NppStreamContext{}))

// Copies the source into the destination at (left, top) and fills the rest
// of the destination with nValue.
VGPU_EXPORT NppStatus nppiCopyConstBorder_8u_C1R_Ctx(const Npp8u* s, int ss, NppiSize ssz, Npp8u* d, int ds,
                                                     NppiSize dsz, int top, int left, Npp8u value,
                                                     NppStreamContext) {
  if (!s || !d) return NPP_NULL_POINTER_ERROR;
  if (bad(ssz) || bad(dsz)) return NPP_SIZE_ERROR;
  if (top < 0 || left < 0 || top + ssz.height > dsz.height || left + ssz.width > dsz.width) return NPP_SIZE_ERROR;
  auto v = fetch<Npp8u>(s, ss, ssz.width, ssz.height);
  std::vector<Npp8u> o(static_cast<size_t>(dsz.width) * dsz.height, value);
  for (int y = 0; y < ssz.height; ++y)
    std::memcpy(&o[static_cast<size_t>(y + top) * dsz.width + left], &v[static_cast<size_t>(y) * ssz.width], ssz.width);
  store<Npp8u>(d, ds, dsz.width, dsz.height, o);
  return NPP_SUCCESS;
}
VGPU_PLAIN(nppiCopyConstBorder_8u_C1R,
           (const Npp8u* s, int ss, NppiSize ssz, Npp8u* d, int ds, NppiSize dsz, int top, int left, Npp8u value),
           (s, ss, ssz, d, ds, dsz, top, left, value, NppStreamContext{}))

/* ---- Bayer demosaicing ----
   NPP documents "bilinear interpolation with chroma correlation of generated
   green values"; what NPP 13.0 computes, fitted on 64x64 random and smooth
   images and checked at every pixel, is:
   - green at a red or blue site: the mean of its two horizontal greens when
     the same colour's horizontal second difference |2c - c(x-2) - c(x+2)| is
     smaller than the vertical one, the two vertical greens when larger, the
     mean of those two means on a tie;
   - red or blue at a blue or red site: the mean of the means of two diagonal
     pairs -- the left and right pairs on the 2x2 cell's first row, the upper
     and lower pairs on its second;
   - red or blue at a green site: the mean of the two neighbours that carry it;
   - every mean truncates, and the image edge reflects without repeating its
     last pixel (x = -1 reads x = 1), which keeps the Bayer phase.
   The grid position names the colours of the first 2x2 cell, row by row. */
namespace {
template <class T>
NppStatus cfa_to_rgb(const T* s, int ss, NppiSize ssz, NppiRect sroi, T* d, int ds, NppiBayerGridPosition grid,
                     NppiInterpolationMode interp) {
  if (!s || !d) return NPP_NULL_POINTER_ERROR;
  if (bad(ssz) || sroi.width <= 0 || sroi.height <= 0) return NPP_SIZE_ERROR;
  if (interp != NPPI_INTER_UNDEFINED) return NPP_INTERPOLATION_ERROR;
  if (grid < NPPI_BAYER_BGGR || grid > NPPI_BAYER_GRBG) return NPP_NOT_SUPPORTED_MODE_ERROR;
  Rect r{};
  if (!clip_src(ssz, sroi, &r)) return NPP_WRONG_INTERSECTION_ROI_ERROR;
  auto img = fetch<T>(s, ss, ssz.width, ssz.height);
  // colour at (x, y): 0 red, 1 green, 2 blue
  static const int cells[4][4] = {{2, 1, 1, 0}, {0, 1, 1, 2}, {1, 2, 0, 1}, {1, 0, 2, 1}};
  auto colour = [&](int x, int y) { return cells[grid][((y - r.y) & 1) * 2 + ((x - r.x) & 1)]; };
  auto at = [&](int x, int y) -> long long {
    if (x < r.x) x = 2 * r.x - x;
    if (y < r.y) y = 2 * r.y - y;
    if (x >= r.x + r.w) x = 2 * (r.x + r.w - 1) - x;
    if (y >= r.y + r.h) y = 2 * (r.y + r.h - 1) - y;
    x = std::min(std::max(x, r.x), r.x + r.w - 1);
    y = std::min(std::max(y, r.y), r.y + r.h - 1);
    return img[static_cast<size_t>(y) * ssz.width + x];
  };
  std::vector<T> o(static_cast<size_t>(r.w) * r.h * 3);
  for (int y = r.y; y < r.y + r.h; ++y)
    for (int x = r.x; x < r.x + r.w; ++x) {
      long long rgb[3];
      const int here = colour(x, y);
      rgb[here] = at(x, y);
      if (here == 1) {
        // Green site: red and blue from the pair that carries each.
        const int c_row = colour(x - 1, y), c_col = colour(x, y - 1);
        rgb[c_row] = (at(x - 1, y) + at(x + 1, y)) / 2;
        rgb[c_col] = (at(x, y - 1) + at(x, y + 1)) / 2;
      } else {
        const long long c = at(x, y);
        const long long lap_h = std::llabs(2 * c - at(x - 2, y) - at(x + 2, y));
        const long long lap_v = std::llabs(2 * c - at(x, y - 2) - at(x, y + 2));
        const long long gl = at(x - 1, y), gr = at(x + 1, y), gu = at(x, y - 1), gd = at(x, y + 1);
        rgb[1] = lap_h < lap_v ? (gl + gr) / 2 : lap_h > lap_v ? (gu + gd) / 2 : ((gl + gr) / 2 + (gu + gd) / 2) / 2;
        const long long tl = at(x - 1, y - 1), tr = at(x + 1, y - 1), bl = at(x - 1, y + 1), br = at(x + 1, y + 1);
        // On the cell's first row the diagonal pairs are taken by column,
        // on its second by row (so BGGR's red-at-blue is by column, its
        // blue-at-red by row; RGGB the other way about).
        rgb[2 - here] = ((y - r.y) & 1) == 0 ? ((tl + bl) / 2 + (tr + br) / 2) / 2
                                             : ((tl + tr) / 2 + (bl + br) / 2) / 2;
      }
      const size_t o0 = (static_cast<size_t>(y - r.y) * r.w + (x - r.x)) * 3;
      for (int c = 0; c < 3; ++c) o[o0 + c] = static_cast<T>(rgb[c]);
    }
  store<T>(d, ds, r.w * 3, r.h, o);
  return NPP_SUCCESS;
}
}  // namespace
#define VGPU_NPP_CFA(SFX, T)                                                                               \
  VGPU_EXPORT NppStatus nppiCFAToRGB_##SFX##_Ctx(const T* s, int ss, NppiSize ssz, NppiRect sroi, T* d,      \
                                                 int ds, NppiBayerGridPosition g, NppiInterpolationMode i,  \
                                                 NppStreamContext) {                                       \
    return cfa_to_rgb<T>(s, ss, ssz, sroi, d, ds, g, i);                                                   \
  }                                                                                                        \
  VGPU_PLAIN(nppiCFAToRGB_##SFX,                                                                            \
             (const T* s, int ss, NppiSize ssz, NppiRect sroi, T* d, int ds, NppiBayerGridPosition g,      \
              NppiInterpolationMode i),                                                                    \
             (s, ss, ssz, sroi, d, ds, g, i, NppStreamContext{}))
VGPU_NPP_CFA(8u_C1C3R, Npp8u)
VGPU_NPP_CFA(16u_C1C3R, Npp16u)

/* ---- allocation ---- */
namespace {
template <class T>
T* image_alloc(int w, int h, int ch, int* step) {
  if (w <= 0 || h <= 0 || !step) return nullptr;
  const size_t row = static_cast<size_t>(w) * ch * sizeof(T);
  const size_t pitch = (row + 511) / 512 * 512;
  void* p = nullptr;
  if (cudaMalloc(&p, pitch * h) != cudaSuccess) return nullptr;
  *step = static_cast<int>(pitch);
  return static_cast<T*>(p);
}
using SignalLen = arg_of<0, decltype(&nppsMalloc_8u)>::type;
template <class N>
bool negative(N n) {
  if constexpr (std::is_signed_v<N>)
    return n < 0;
  else
    return false;
}
}  // namespace
#define VGPU_NPPI_ALLOC(SFX, T, CH) \
  VGPU_EXPORT T* nppiMalloc_##SFX(int w, int h, int* step) { return image_alloc<T>(w, h, CH, step); }
VGPU_NPPI_ALLOC(16u_C2, Npp16u, 2)
VGPU_NPPI_ALLOC(16s_C2, Npp16s, 2)
VGPU_NPPI_ALLOC(16s_C4, Npp16s, 4)
VGPU_NPPI_ALLOC(32s_C3, Npp32s, 3)
VGPU_NPPI_ALLOC(32s_C4, Npp32s, 4)
VGPU_NPPI_ALLOC(32sc_C1, Npp32sc, 1)
VGPU_NPPI_ALLOC(32fc_C1, Npp32fc, 1)
#define VGPU_NPPS_ALLOC(SFX, T)                                                  \
  VGPU_EXPORT T* nppsMalloc_##SFX(SignalLen n) {                                  \
    if (negative(n)) return nullptr;                                              \
    void* p = nullptr;                                                            \
    if (cudaMalloc(&p, static_cast<size_t>(n) * sizeof(T)) != cudaSuccess)        \
      return nullptr;                                                             \
    return static_cast<T*>(p);                                                    \
  }
VGPU_NPPS_ALLOC(16sc, Npp16sc)
VGPU_NPPS_ALLOC(32sc, Npp32sc)
VGPU_NPPS_ALLOC(64sc, Npp64sc)
VGPU_NPPS_ALLOC(32fc, Npp32fc)
VGPU_NPPS_ALLOC(64fc, Npp64fc)
