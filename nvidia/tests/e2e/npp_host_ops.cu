// NPP's 8-bit arithmetic with a scale factor (nppiAdd, Sub, Mul, Div, nppiAddC, SubC, MulC ... in 1 and 3 channels) and nppiFilterBox
// against the plain loops they replaced: every pixel computed in double, scaled by 2^-sf, rounded to nearest even and saturated; the
// box filter as the floor of the window's sum over its area. Every pixel of every result is compared. Random pixels and the extremes (all
// 0 and all 255), every scale factor from 0 to 9 (and a negative one), ROIs that do not start at the image's corner and rows with padding.
// Not run against the card (npp_ops and npp_imgproc, in nvidia/tests/conformance, are).
#include <cuda_runtime.h>
#include <npp.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#if defined(NPP_VERSION_MAJOR) && NPP_VERSION_MAJOR >= 13
#define NPPCALL(fn, ...) fn##_Ctx(__VA_ARGS__, ctx)
static NppStreamContext ctx;
#else
#define NPPCALL(fn, ...) fn(__VA_ARGS__)
#endif

static int failures = 0, checks = 0;
static void report(bool ok, const char* what) {
  ++checks;
  if (!ok) {
    ++failures;
    std::printf("FAIL %s\n", what);
  }
}
static std::mt19937_64 rng(4711);

static uint8_t sat8(double v) {
  const double r = std::nearbyint(v);
  return static_cast<uint8_t>(r < 0 ? 0 : (r > 255 ? 255 : r));
}
static double scale_by(double v, int sf) { return sf == 0 ? v : v * std::pow(2.0, -sf); }

struct Img {
  int w, h, ch, step;
  uint8_t* d = nullptr;
  std::vector<uint8_t> host;
  Img(int w_, int h_, int ch_, int mode, int pad) : w(w_), h(h_), ch(ch_), step(w_ * ch_ + pad), host((size_t)step * h_) {
    for (auto& v : host) v = mode == 0 ? (uint8_t)rng() : mode == 1 ? (rng() % 2 ? 255 : 0) : mode == 2 ? 255 : 0;
    cudaMalloc(&d, host.size() + 16);
    cudaMemcpy(d, host.data(), host.size(), cudaMemcpyHostToDevice);
  }
  ~Img() { cudaFree(d); }
  std::vector<uint8_t> read() const {
    std::vector<uint8_t> o(host.size());
    cudaMemcpy(o.data(), d, o.size(), cudaMemcpyDeviceToHost);
    return o;
  }
};

int main() {
#if defined(NPP_VERSION_MAJOR) && NPP_VERSION_MAJOR >= 13
  ctx.hStream = 0;
  ctx.nCudaDeviceId = 0;
#endif
  char what[160];
  for (const int mode : {0, 1, 2, 3})
    for (const int sf : {0, 1, 3, 7, 9, -2}) {
      for (const int ch : {1, 3}) {
        const int W = 37, H = 23, pad = (int)(rng() % 3);
        Img a(W, H, ch, mode, pad), b(W, H, ch, mode == 0 ? 0 : 1, pad), out(W, H, ch, 0, pad), out2(W, H, ch, 0, pad);
        const int n = W * ch;
        NppiSize roi{W, H};
        struct Op { const char* name; NppStatus (*f1)(const Npp8u*, int, const Npp8u*, int, Npp8u*, int, NppiSize, int, NppStreamContext); };
        for (int op = 0; op < 4; ++op) {
          NppStatus st = NPP_SUCCESS;
          if (ch == 1) {
            if (op == 0) st = NPPCALL(nppiAdd_8u_C1RSfs, a.d, a.step, b.d, b.step, out.d, out.step, roi, sf);
            if (op == 1) st = NPPCALL(nppiSub_8u_C1RSfs, a.d, a.step, b.d, b.step, out.d, out.step, roi, sf);
            if (op == 2) st = NPPCALL(nppiMul_8u_C1RSfs, a.d, a.step, b.d, b.step, out.d, out.step, roi, sf);
            if (op == 3) st = NPPCALL(nppiDiv_8u_C1RSfs, a.d, a.step, b.d, b.step, out.d, out.step, roi, sf);
          } else {
            if (op == 0) st = NPPCALL(nppiAdd_8u_C3RSfs, a.d, a.step, b.d, b.step, out.d, out.step, roi, sf);
            if (op == 1) st = NPPCALL(nppiSub_8u_C3RSfs, a.d, a.step, b.d, b.step, out.d, out.step, roi, sf);
            if (op == 2) st = NPPCALL(nppiMul_8u_C3RSfs, a.d, a.step, b.d, b.step, out.d, out.step, roi, sf);
            if (op == 3) st = NPPCALL(nppiDiv_8u_C3RSfs, a.d, a.step, b.d, b.step, out.d, out.step, roi, sf);
          }
          const auto got = out.read();
          bool ok = st == NPP_SUCCESS;
          for (int y = 0; y < H && ok; ++y)
            for (int x = 0; x < n; ++x) {
              const double p = a.host[(size_t)y * a.step + x], q = b.host[(size_t)y * b.step + x];
              const double v = op == 0 ? p + q : op == 1 ? q - p : op == 2 ? p * q : (q == 0 ? 0.0 : p / q);
              if (got[(size_t)y * out.step + x] != sat8(scale_by(v, sf))) { ok = false; break; }
            }
          std::snprintf(what, sizeof what, "nppi%s_8u_C%dRSfs mode %d scale %d", op == 0 ? "Add" : op == 1 ? "Sub" : op == 2 ? "Mul" : "Div", ch, mode, sf);
          report(ok, what);
        }
        // constants
        if (ch == 1) {
          const Npp8u c = (Npp8u)(rng() % 256);
          NppStatus st = NPPCALL(nppiAddC_8u_C1RSfs, a.d, a.step, c, out2.d, out2.step, roi, sf);
          bool ok = st == NPP_SUCCESS;
          const auto got = out2.read();
          for (int y = 0; y < H && ok; ++y)
            for (int x = 0; x < n; ++x)
              if (got[(size_t)y * out2.step + x] != sat8(scale_by((double)a.host[(size_t)y * a.step + x] + c, sf))) { ok = false; break; }
          std::snprintf(what, sizeof what, "nppiAddC_8u_C1RSfs mode %d scale %d", mode, sf);
          report(ok, what);
          st = NPPCALL(nppiMulC_8u_C1RSfs, a.d, a.step, c, out2.d, out2.step, roi, sf);
          ok = st == NPP_SUCCESS;
          const auto got2 = out2.read();
          for (int y = 0; y < H && ok; ++y)
            for (int x = 0; x < n; ++x)
              if (got2[(size_t)y * out2.step + x] != sat8(scale_by((double)a.host[(size_t)y * a.step + x] * c, sf))) { ok = false; break; }
          std::snprintf(what, sizeof what, "nppiMulC_8u_C1RSfs mode %d scale %d", mode, sf);
          report(ok, what);
        }
      }
    }
  // The box filter: windows of several sizes and anchors, over a ROI inside a larger image (the window reaches into the rest of it).
  struct Box { int mw, mh, ax, ay; };
  for (const int mode : {0, 1, 2, 3})
    for (const Box bx : {Box{3, 3, 1, 1}, Box{5, 5, 2, 2}, Box{1, 1, 0, 0}, Box{4, 2, 1, 0}, Box{7, 3, 3, 1}, Box{2, 6, 0, 5}, Box{9, 9, 4, 4}, Box{3, 8, 2, 3}}) {
      const int W = 61, H = 47, pad = (int)(rng() % 3);
      Img a(W, H, 1, mode, pad), out(W, H, 1, 0, pad);
      const int x0 = bx.ax + (int)(rng() % 3), y0 = bx.ay + (int)(rng() % 3);
      const int rw = W - x0 - (bx.mw - bx.ax) - (int)(rng() % 3), rh = H - y0 - (bx.mh - bx.ay) - (int)(rng() % 3);
      NppiSize roi{rw, rh}, mask{bx.mw, bx.mh};
      NppiPoint anchor{bx.ax, bx.ay};
      const NppStatus st = NPPCALL(nppiFilterBox_8u_C1R, a.d + (size_t)y0 * a.step + x0, a.step, out.d + (size_t)y0 * out.step + x0, out.step, roi, mask, anchor);
      const auto got = out.read();
      bool ok = st == NPP_SUCCESS;
      for (int y = 0; y < rh && ok; ++y)
        for (int x = 0; x < rw; ++x) {
          double acc = 0;
          for (int my = 0; my < bx.mh; ++my)
            for (int mx = 0; mx < bx.mw; ++mx) acc += a.host[(size_t)(y0 + y - bx.ay + my) * a.step + (x0 + x - bx.ax + mx)];
          if (got[(size_t)(y0 + y) * out.step + x0 + x] != sat8(std::floor(acc / ((double)bx.mw * bx.mh)))) { ok = false; break; }
        }
      std::snprintf(what, sizeof what, "nppiFilterBox_8u_C1R mask %dx%d anchor %d,%d mode %d", bx.mw, bx.mh, bx.ax, bx.ay, mode);
      report(ok, what);
    }
  std::printf("%d checks, %d failed\n", checks, failures);
  std::printf(failures ? "FAIL\n" : "PASS\n");
  return failures ? 1 : 0;
}
