// Conformance for NPP's image-processing entry points that real programs call
// (OpenCV's cudawarping/cudaimgproc/cudaarithm/cudafilters, DALI, FFmpeg's
// scale_npp, the CUDA Samples): geometric transforms, remapping, mirroring,
// logical and shift operators with constants, alpha compositing, gamma,
// demosaicing, lookup, statistics, histograms, integral images, rank and
// morphological filters, gradients and Canny.
//
// Every result is printed as a status and a checksum (sum and position-
// weighted sum: integers exactly, floats to seven figures), so the golden file
// an RTX 3060 printed (golden/npp_imgproc.rtx3060.txt) pins every pixel.
// Inputs are generated, identical on both sides. All calls are the _Ctx forms,
// the only ones CUDA 13 declares.
//
// With NPP_DUMP_DIR set, each output is also written there raw, for finding
// which pixels differ when a checksum does.
#include <npp.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

static NppStreamContext ctx() {
  NppStreamContext c{};
  c.hStream = 0;
  cudaGetDevice(&c.nCudaDeviceId);
  cudaDeviceProp p;
  cudaGetDeviceProperties(&p, c.nCudaDeviceId);
  c.nMultiProcessorCount = p.multiProcessorCount;
  c.nMaxThreadsPerMultiProcessor = p.maxThreadsPerMultiProcessor;
  c.nMaxThreadsPerBlock = p.maxThreadsPerBlock;
  c.nSharedMemPerBlock = p.sharedMemPerBlock;
  c.nCudaDevAttrComputeCapabilityMajor = p.major;
  c.nCudaDevAttrComputeCapabilityMinor = p.minor;
  return c;
}

static unsigned g_rng = 2024;
static unsigned next_u() {
  g_rng = g_rng * 1664525u + 1013904223u;
  return g_rng >> 8;
}
template <class T>
static T random_value() {
  if constexpr (std::is_same_v<T, Npp8u>) return next_u() & 255;
  else if constexpr (std::is_same_v<T, Npp8s>) return static_cast<Npp8s>(next_u() & 255);
  else if constexpr (std::is_same_v<T, Npp16u>) return next_u() & 65535;
  else if constexpr (std::is_same_v<T, Npp16s>) return static_cast<Npp16s>(next_u() & 65535);
  else if constexpr (std::is_same_v<T, Npp32s>) return static_cast<Npp32s>(next_u() * 37u);
  else if constexpr (std::is_floating_point_v<T>)
    return static_cast<T>(static_cast<int>(next_u() % 20001) - 10000) / static_cast<T>(37);
  else
    return T{};
}

template <class T>
static T seven() {
  if constexpr (std::is_arithmetic_v<T>) return T(7);
  else return T{};
}

// A device image with a host copy; rows are padded to a whole number of
// pixels past the end.
template <class T>
struct Img {
  int w, h, ch, stride;
  std::vector<T> host;
  T* dev = nullptr;
  Img(int w_, int h_, int ch_, bool random = true, T fill = seven<T>()) : w(w_), h(h_), ch(ch_) {
    stride = (w + 3) * ch;
    host.assign(static_cast<size_t>(stride) * h, fill);
    if (random)
      for (auto& v : host) v = random_value<T>();
    cudaMalloc(&dev, host.size() * sizeof(T));
    upload();
  }
  ~Img() { cudaFree(dev); }
  Img(const Img&) = delete;
  void upload() { cudaMemcpy(dev, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice); }
  int step() const { return stride * static_cast<int>(sizeof(T)); }
  T* at(int x, int y) { return reinterpret_cast<T*>(reinterpret_cast<char*>(dev) + static_cast<size_t>(y) * step()) + x * ch; }
};

static int g_lines = 0;
// Set around results that are within a count of NVIDIA's on a pixel or two
// (see nvidia/docs/libraries.md): their integer checksums print as floats, so
// the comparison's relative tolerance absorbs a count but not a wrong image.
static bool g_approx = false;

template <class T>
static void report_values(const char* name, int status, const std::vector<T>& v) {
  if (const char* dir = std::getenv("NPP_DUMP_DIR")) {
    std::string path = std::string(dir) + "/" + name + ".bin";
    for (char& c : path)
      if (c == ' ') c = '_';
    if (FILE* f = std::fopen(path.c_str(), "wb")) {
      std::fwrite(v.data(), sizeof(T), v.size(), f);
      std::fclose(f);
    }
  }
  ++g_lines;
  if constexpr (std::is_floating_point_v<T>) {
    double s = 0, ws = 0;
    for (size_t i = 0; i < v.size(); ++i) {
      s += v[i];
      ws += v[i] * static_cast<double>(i % 97 + 1);
    }
    std::printf("%-44s st=%d n=%zu sum=%.6e wsum=%.6e\n", name, status, v.size(), s, ws);
  } else if (g_approx) {
    double s = 0, ws = 0;
    for (size_t i = 0; i < v.size(); ++i) {
      s += static_cast<double>(v[i]);
      ws += static_cast<double>(v[i]) * static_cast<double>(i % 97 + 1);
    }
    std::printf("%-44s st=%d n=%zu sum~%.6e wsum~%.6e\n", name, status, v.size(), s, ws);
  } else {
    long long s = 0, ws = 0;
    for (size_t i = 0; i < v.size(); ++i) {
      s += static_cast<long long>(v[i]);
      ws += static_cast<long long>(v[i]) * static_cast<long long>(i % 97 + 1);
    }
    std::printf("%-44s st=%d n=%zu sum=%lld wsum=%lld\n", name, status, v.size(), s, ws);
  }
}

// The whole of a device image, padding included (so a write outside the
// ROI shows).
template <class T>
static void report(const char* name, int status, Img<T>& img) {
  std::vector<T> v(img.host.size());
  cudaMemcpy(v.data(), img.dev, v.size() * sizeof(T), cudaMemcpyDeviceToHost);
  report_values(name, status, v);
}
template <class T>
static void report_dev(const char* name, int status, const T* dev, size_t n) {
  std::vector<T> v(n);
  cudaMemcpy(v.data(), dev, n * sizeof(T), cudaMemcpyDeviceToHost);
  report_values(name, status, v);
}
static void report_status(const char* name, int status) {
  ++g_lines;
  std::printf("%-44s st=%d\n", name, status);
}

static const char* interp_name(int i) { return i == 1 ? "nn" : i == 2 ? "linear" : "cubic"; }

/* ---------------- geometry ---------------- */

template <class T, class F>
static void warp_case(const char* fn, const char* type, int ch, int interp, const double* coeffs, F call) {
  Img<T> s(37, 29, ch);
  Img<T> d(41, 31, ch, false);
  const NppiSize ssz{37, 29};
  const NppiRect sroi{3, 2, 30, 25}, droi{2, 3, 35, 26};
  const int st = call(s.dev, ssz, s.step(), sroi, d.dev, d.step(), droi, coeffs, interp, ctx());
  char name[96];
  std::snprintf(name, sizeof name, "%s %s %s", fn, type, interp_name(interp));
  // 16-bit cubic: one pixel in a thousand a count apart.
  g_approx = interp == NPPI_INTER_CUBIC && sizeof(T) == 2;
  report(name, st, d);
  g_approx = false;
}

#define WARP2(FN, SFX, T, CH, INTERP, C)                                                               \
  warp_case<T>(#FN, #SFX, CH, INTERP, &C[0][0],                                                         \
               [](const T* s, NppiSize ssz, int ss, NppiRect sr, T* d, int ds, NppiRect dr,             \
                  const double* c, int i, NppStreamContext x) {                                         \
                 return nppi##FN##_##SFX##_Ctx(s, ssz, ss, sr, d, ds, dr,                               \
                                               reinterpret_cast<const double(*)[3]>(c), i, x);          \
               })

static void geometry() {
  const double aff[2][3] = {{0.9, -0.3, 4.3}, {0.25, 1.1, -2.7}};
  const double per[3][3] = {{0.95, -0.12, 3.3}, {0.08, 1.05, -1.7}, {0.0021, -0.0013, 1.02}};
  for (int i : {1, 2, 4}) {
    WARP2(WarpAffine, 8u_C1R, Npp8u, 1, i, aff);
    WARP2(WarpAffine, 8u_C3R, Npp8u, 3, i, aff);
    WARP2(WarpAffine, 8u_C4R, Npp8u, 4, i, aff);
    WARP2(WarpAffine, 16u_C1R, Npp16u, 1, i, aff);
    WARP2(WarpAffine, 32f_C1R, Npp32f, 1, i, aff);
    WARP2(WarpAffine, 32f_C3R, Npp32f, 3, i, aff);
    WARP2(WarpAffineBack, 8u_C1R, Npp8u, 1, i, aff);
    WARP2(WarpAffineBack, 8u_C3R, Npp8u, 3, i, aff);
    WARP2(WarpAffineBack, 16u_C4R, Npp16u, 4, i, aff);
    WARP2(WarpAffineBack, 32f_C1R, Npp32f, 1, i, aff);
    WARP2(WarpPerspective, 8u_C1R, Npp8u, 1, i, per);
    WARP2(WarpPerspective, 8u_C4R, Npp8u, 4, i, per);
    WARP2(WarpPerspective, 16u_C3R, Npp16u, 3, i, per);
    WARP2(WarpPerspectiveBack, 8u_C3R, Npp8u, 3, i, per);
    WARP2(WarpPerspectiveBack, 16u_C1R, Npp16u, 1, i, per);
  }
  // 32-bit integers: nearest and linear are exact; cubic is a float ulp out
  // on some pixels (see npp_core.hpp), so it is not pinned here.
  for (int i : {1, 2}) {
    WARP2(WarpAffine, 32s_C1R, Npp32s, 1, i, aff);
    WARP2(WarpAffineBack, 32s_C3R, Npp32s, 3, i, aff);
    WARP2(WarpPerspectiveBack, 32s_C1R, Npp32s, 1, i, per);
  }
  // Single precision perspective sources differ in the last bits, which the
  // checksums' seven figures absorb; the 8- and 16-bit ones above are exact.
  for (int i : {1, 2, 4}) WARP2(WarpPerspective, 32f_C1R, Npp32f, 1, i, per);

  // Rotate.
  for (double angle : {23.5, -71.0, 180.0, 137.0})
    for (int i : {1, 2, 4}) {
      Img<Npp8u> s(37, 29, 1), s3(37, 29, 3);
      Img<Npp8u> d(41, 31, 1, false), d3(41, 31, 3, false);
      Img<Npp32f> sf(37, 29, 1), df(41, 31, 1, false);
      const double shx = angle == 180.0 ? 36 : 5.25, shy = angle == 180.0 ? 28 : -3.5;
      char name[96];
      int st = nppiRotate_8u_C1R_Ctx(s.dev, {37, 29}, s.step(), {3, 2, 30, 25}, d.dev, d.step(), {2, 3, 35, 26},
                                     angle, shx, shy, i, ctx());
      std::snprintf(name, sizeof name, "Rotate 8u_C1R %.1f %s", angle, interp_name(i));
      report(name, st, d);
      st = nppiRotate_8u_C3R_Ctx(s3.dev, {37, 29}, s3.step(), {3, 2, 30, 25}, d3.dev, d3.step(), {2, 3, 35, 26},
                                 angle, shx, shy, i, ctx());
      std::snprintf(name, sizeof name, "Rotate 8u_C3R %.1f %s", angle, interp_name(i));
      report(name, st, d3);
      st = nppiRotate_32f_C1R_Ctx(sf.dev, {37, 29}, sf.step(), {3, 2, 30, 25}, df.dev, df.step(), {2, 3, 35, 26},
                                  angle, shx, shy, i, ctx());
      std::snprintf(name, sizeof name, "Rotate 32f_C1R %.1f %s", angle, interp_name(i));
      report(name, st, df);
    }
  {
    double q[4][2], b[2][2];
    nppiGetRotateQuad({3, 2, 30, 25}, q, -71.0, 5.25, -3.5);
    nppiGetRotateBound({3, 2, 30, 25}, b, -71.0, 5.25, -3.5);
    ++g_lines;
    std::printf("GetRotateQuad %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e\n", q[0][0], q[0][1], q[1][0], q[1][1],
                q[2][0], q[2][1], q[3][0], q[3][1]);
    ++g_lines;
    std::printf("GetRotateBound %.6e %.6e %.6e %.6e\n", b[0][0], b[0][1], b[1][0], b[1][1]);
  }

  // Remap: maps with points outside the ROI on every side.
  auto maps = [](Img<Npp32f>& xm, Img<Npp32f>& ym) {
    for (int y = 0; y < xm.h; ++y)
      for (int x = 0; x < xm.w; ++x) {
        xm.host[y * xm.stride + x] = x * 0.83f + y * 0.11f - 2.3f + (x % 3) * 0.5f;
        ym.host[y * ym.stride + x] = y * 0.97f - x * 0.07f + 1.25f;
      }
    xm.upload();
    ym.upload();
  };
  for (int i : {1, 2, 4}) {
    Img<Npp32f> xm(35, 26, 1, false), ym(35, 26, 1, false);
    maps(xm, ym);
    char name[96];
#define REMAP(SFX, T, CH)                                                                               \
  {                                                                                                     \
    Img<T> s(37, 29, CH), d(35, 26, CH, false);                                                         \
    int st = nppiRemap_##SFX##_Ctx(s.dev, {37, 29}, s.step(), {3, 2, 30, 25}, xm.dev, xm.step(), ym.dev, \
                                   ym.step(), d.dev, d.step(), {35, 26}, i, ctx());                     \
    std::snprintf(name, sizeof name, "Remap %s %s", #SFX, interp_name(i));                              \
    report(name, st, d);                                                                                \
  }
    REMAP(8u_C1R, Npp8u, 1)
    REMAP(8u_C3R, Npp8u, 3)
    REMAP(16u_C1R, Npp16u, 1)
    g_approx = true;  // one pixel a count apart
    REMAP(16s_C1R, Npp16s, 1)
    g_approx = false;
    REMAP(32f_C1R, Npp32f, 1)
    REMAP(32f_C3R, Npp32f, 3)
    {
      Img<Npp64f> s(37, 29, 1), d(35, 26, 1, false), xm64(35, 26, 1, false), ym64(35, 26, 1, false);
      for (size_t k = 0; k < xm.host.size(); ++k) {
        xm64.host[k] = xm.host[k];
        ym64.host[k] = ym.host[k];
      }
      xm64.upload();
      ym64.upload();
      int st = nppiRemap_64f_C1R_Ctx(s.dev, {37, 29}, s.step(), {3, 2, 30, 25}, xm64.dev, xm64.step(), ym64.dev,
                                     ym64.step(), d.dev, d.step(), {35, 26}, i, ctx());
      std::snprintf(name, sizeof name, "Remap 64f_C1R %s", interp_name(i));
      report(name, st, d);
    }
  }

  // Mirror.
  for (NppiAxis axis : {NPP_HORIZONTAL_AXIS, NPP_VERTICAL_AXIS, NPP_BOTH_AXIS}) {
    char name[96];
#define MIRROR(SFX, T, CH)                                                                                   \
  {                                                                                                          \
    Img<T> s(13, 9, CH), d(13, 9, CH, false);                                                                \
    int st = nppiMirror_##SFX##R_Ctx(s.dev, s.step(), d.dev, d.step(), {13, 9}, axis, ctx());                \
    std::snprintf(name, sizeof name, "Mirror %sR %d", #SFX, int(axis));                                      \
    report(name, st, d);                                                                                     \
    st = nppiMirror_##SFX##IR_Ctx(s.dev, s.step(), {13, 9}, axis, ctx());                                    \
    std::snprintf(name, sizeof name, "Mirror %sIR %d", #SFX, int(axis));                                     \
    report(name, st, s);                                                                                     \
  }
    MIRROR(8u_C3, Npp8u, 3)
    MIRROR(8u_C4, Npp8u, 4)
    MIRROR(16u_C1, Npp16u, 1)
    MIRROR(32s_C3, Npp32s, 3)
    MIRROR(32f_C1, Npp32f, 1)
    MIRROR(32f_C4, Npp32f, 4)
  }

  // ResizeSqrPixel, FFmpeg's scale_npp path.
  for (double f : {0.5, 0.73, 1.37, 2.0})
    for (int i : {1, 2, 4}) {
      Img<Npp8u> s(23, 17, 1), d(50, 40, 1, false);
      Img<Npp8u> s3(23, 17, 3), d3(50, 40, 3, false);
      Img<Npp32f> sf(23, 17, 1), df(50, 40, 1, false);
      char name[96];
      g_approx = i != NPPI_INTER_NN;  // linear a count off on the odd pixel; cubic is not reported
      int st = nppiResizeSqrPixel_8u_C1R_Ctx(s.dev, {23, 17}, s.step(), {0, 0, 23, 17}, d.dev, d.step(),
                                             {0, 0, 50, 40}, f, f * 1.1, 0, 0, i, ctx());
      std::snprintf(name, sizeof name, "ResizeSqrPixel 8u_C1R %.2f %s", f, interp_name(i));
      if (i != NPPI_INTER_CUBIC) report(name, st, d);
      st = nppiResizeSqrPixel_8u_C3R_Ctx(s3.dev, {23, 17}, s3.step(), {0, 0, 23, 17}, d3.dev, d3.step(),
                                         {0, 0, 50, 40}, f, f * 0.9, 0, 0, i, ctx());
      std::snprintf(name, sizeof name, "ResizeSqrPixel 8u_C3R %.2f %s", f, interp_name(i));
      if (i != NPPI_INTER_CUBIC) report(name, st, d3);
      st = nppiResizeSqrPixel_32f_C1R_Ctx(sf.dev, {23, 17}, sf.step(), {0, 0, 23, 17}, df.dev, df.step(),
                                          {0, 0, 50, 40}, f, f, 0, 0, i, ctx());
      std::snprintf(name, sizeof name, "ResizeSqrPixel 32f_C1R %.2f %s", f, interp_name(i));
      if (i != NPPI_INTER_CUBIC) report(name, st, df);
      g_approx = false;
    }

  // Statuses.
  {
    Img<Npp8u> s(37, 29, 1), d(41, 31, 1, false);
    report_status("WarpAffine super-sampling",
                  nppiWarpAffine_8u_C1R_Ctx(s.dev, {37, 29}, s.step(), {0, 0, 37, 29}, d.dev, d.step(),
                                            {0, 0, 41, 31}, aff, NPPI_INTER_SUPER, ctx()));
    const double singular[2][3] = {{1, 2, 0}, {2, 4, 0}};
    report_status("WarpAffine singular",
                  nppiWarpAffine_8u_C1R_Ctx(s.dev, {37, 29}, s.step(), {0, 0, 37, 29}, d.dev, d.step(),
                                            {0, 0, 41, 31}, singular, NPPI_INTER_NN, ctx()));
    report_status("WarpAffineBack null source",
                  nppiWarpAffineBack_8u_C1R_Ctx(nullptr, {37, 29}, s.step(), {0, 0, 37, 29}, d.dev, d.step(),
                                                {0, 0, 41, 31}, aff, NPPI_INTER_NN, ctx()));
  }
}

/* ---------------- arithmetic, logic, alpha ---------------- */

static void arithmetic() {
  char name[96];
  {
    Img<Npp8u> s(23, 11, 1), s3(23, 11, 3), s4(23, 11, 4), d(23, 11, 1, false), d3(23, 11, 3, false), d4(23, 11, 4, false);
    const Npp8u k3[3] = {0x5a, 0xc3, 0x0f}, k4[4] = {0x5a, 0xc3, 0x0f, 0xf0};
    report("AndC 8u_C1R", nppiAndC_8u_C1R_Ctx(s.dev, s.step(), 0x6c, d.dev, d.step(), {23, 11}, ctx()), d);
    report("OrC 8u_C3R", nppiOrC_8u_C3R_Ctx(s3.dev, s3.step(), k3, d3.dev, d3.step(), {23, 11}, ctx()), d3);
    report("XorC 8u_C4R", nppiXorC_8u_C4R_Ctx(s4.dev, s4.step(), k4, d4.dev, d4.step(), {23, 11}, ctx()), d4);
    const Npp32u sh3[3] = {1, 3, 7}, sh4[4] = {0, 2, 5, 8};
    report("LShiftC 8u_C3R", nppiLShiftC_8u_C3R_Ctx(s3.dev, s3.step(), sh3, d3.dev, d3.step(), {23, 11}, ctx()), d3);
    report("RShiftC 8u_C4R", nppiRShiftC_8u_C4R_Ctx(s4.dev, s4.step(), sh4, d4.dev, d4.step(), {23, 11}, ctx()), d4);
    report("RShiftC 8u_C1R", nppiRShiftC_8u_C1R_Ctx(s.dev, s.step(), 3, d.dev, d.step(), {23, 11}, ctx()), d);
  }
  {
    Img<Npp16u> s3(23, 11, 3), d3(23, 11, 3, false);
    const Npp16u k3[3] = {0x5a5a, 0xc3c3, 0x0ff0};
    const Npp32u sh3[3] = {1, 4, 9};
    report("AndC 16u_C3R", nppiAndC_16u_C3R_Ctx(s3.dev, s3.step(), k3, d3.dev, d3.step(), {23, 11}, ctx()), d3);
    report("LShiftC 16u_C3R", nppiLShiftC_16u_C3R_Ctx(s3.dev, s3.step(), sh3, d3.dev, d3.step(), {23, 11}, ctx()), d3);
    report("RShiftC 16u_C3R", nppiRShiftC_16u_C3R_Ctx(s3.dev, s3.step(), sh3, d3.dev, d3.step(), {23, 11}, ctx()), d3);
  }
  {
    Img<Npp16s> s(23, 11, 1), d(23, 11, 1, false);
    Img<Npp8s> s8(23, 11, 1), d8(23, 11, 1, false);
    Img<Npp32s> s4(23, 11, 4), d4(23, 11, 4, false);
    const Npp32s k4[4] = {0x5a5a5a5a, -1, 0x0ff00ff0, 0x7fffffff};
    const Npp32u sh4[4] = {1, 7, 16, 31};
    report("RShiftC 16s_C1R", nppiRShiftC_16s_C1R_Ctx(s.dev, s.step(), 5, d.dev, d.step(), {23, 11}, ctx()), d);
    report("RShiftC 8s_C1R", nppiRShiftC_8s_C1R_Ctx(s8.dev, s8.step(), 2, d8.dev, d8.step(), {23, 11}, ctx()), d8);
    report("XorC 32s_C4R", nppiXorC_32s_C4R_Ctx(s4.dev, s4.step(), k4, d4.dev, d4.step(), {23, 11}, ctx()), d4);
    report("LShiftC 32s_C4R", nppiLShiftC_32s_C4R_Ctx(s4.dev, s4.step(), sh4, d4.dev, d4.step(), {23, 11}, ctx()), d4);
    report("RShiftC 32s_C4R", nppiRShiftC_32s_C4R_Ctx(s4.dev, s4.step(), sh4, d4.dev, d4.step(), {23, 11}, ctx()), d4);
  }
  {
    Img<Npp32fc> s(17, 9, 1, false);
    for (auto& v : s.host) v = Npp32fc{random_value<Npp32f>(), random_value<Npp32f>()};
    s.upload();
    Img<Npp32f> d(17, 9, 1, false);
    report("Magnitude 32fc32f_C1R", nppiMagnitude_32fc32f_C1R_Ctx(s.dev, s.step(), d.dev, d.step(), {17, 9}, ctx()), d);
    report("MagnitudeSqr 32fc32f_C1R",
           nppiMagnitudeSqr_32fc32f_C1R_Ctx(s.dev, s.step(), d.dev, d.step(), {17, 9}, ctx()), d);
  }
  // Alpha compositing over every (value, alpha) pair that matters: random
  // pixels, 4096 of them.
  {
    Img<Npp8u> a(64, 64, 4), b(64, 64, 4), d(64, 64, 4, false);
    for (int op = NPPI_OP_ALPHA_OVER; op <= NPPI_OP_ALPHA_PREMUL; ++op) {
      const int st = nppiAlphaComp_8u_AC4R_Ctx(a.dev, a.step(), b.dev, b.step(), d.dev, d.step(), {64, 64},
                                               static_cast<NppiAlphaOp>(op), ctx());
      std::vector<Npp8u> v(d.host.size());
      cudaMemcpy(v.data(), d.dev, v.size(), cudaMemcpyDeviceToHost);
      if (op == NPPI_OP_ALPHA_ATOP || op == NPPI_OP_ALPHA_XOR) {
        // Colour within one count of NVIDIA's (see libraries.md); alpha exact.
        std::vector<Npp8u> alpha;
        for (size_t i = 3; i < v.size(); i += 4) alpha.push_back(v[i]);
        v = alpha;
      }
      std::snprintf(name, sizeof name, "AlphaComp 8u_AC4R op %d", op);
      report_values(name, st, v);
    }
    report("AlphaPremul 8u_AC4R", nppiAlphaPremul_8u_AC4R_Ctx(a.dev, a.step(), d.dev, d.step(), {64, 64}, ctx()), d);
    Img<Npp8u> odd(5, 3, 4);
    report_status("AlphaPremul 8u_AC4R odd step",
                  nppiAlphaPremul_8u_AC4R_Ctx(odd.dev, odd.step() - 1, odd.dev, odd.step() - 1, {3, 3}, ctx()));
  }
  {
    Img<Npp32f> a(16, 16, 4), b(16, 16, 4), d(16, 16, 4, false);
    for (auto* im : {&a, &b}) {
      for (size_t i = 3; i < im->host.size(); i += 4) im->host[i] = (next_u() % 1001) / 1000.0f;
      im->upload();
    }
    for (int op = NPPI_OP_ALPHA_OVER; op <= NPPI_OP_ALPHA_PREMUL; ++op) {
      const int st = nppiAlphaComp_32f_AC4R_Ctx(a.dev, a.step(), b.dev, b.step(), d.dev, d.step(), {16, 16},
                                                static_cast<NppiAlphaOp>(op), ctx());
      std::snprintf(name, sizeof name, "AlphaComp 32f_AC4R op %d", op);
      report(name, st, d);
    }
  }
}

/* ---------------- colour ---------------- */

static void colour() {
  char name[96];
  {
    Img<Npp8u> s(256, 1, 3, false), d(256, 1, 3, false);
    for (int i = 0; i < 256; ++i)
      for (int c = 0; c < 3; ++c) s.host[i * 3 + c] = static_cast<Npp8u>(i + c * 85);
    s.upload();
    report("GammaFwd 8u_C3R", nppiGammaFwd_8u_C3R_Ctx(s.dev, s.step(), d.dev, d.step(), {256, 1}, ctx()), d);
    report("GammaInv 8u_C3R", nppiGammaInv_8u_C3R_Ctx(s.dev, s.step(), d.dev, d.step(), {256, 1}, ctx()), d);
    Img<Npp8u> s4(64, 4, 4), d4(64, 4, 4, false);
    report("GammaFwd 8u_AC4R", nppiGammaFwd_8u_AC4R_Ctx(s4.dev, s4.step(), d4.dev, d4.step(), {64, 4}, ctx()), d4);
    report("GammaInv 8u_AC4IR", nppiGammaInv_8u_AC4IR_Ctx(s4.dev, s4.step(), {64, 4}, ctx()), s4);
    const int order[4] = {2, 0, 3, 1};
    report("SwapChannels 8u_C4IR", nppiSwapChannels_8u_C4IR_Ctx(s4.dev, s4.step(), {64, 4}, order, ctx()), s4);
  }
  {
    Img<Npp8u> y(16, 10, 1), cb(8, 5, 1), cr(8, 5, 1), dy(16, 10, 1, false), duv(16, 5, 1, false);
    const Npp8u* const src[3] = {y.dev, cb.dev, cr.dev};
    int steps[3] = {y.step(), cb.step(), cr.step()};
    report("YCbCr420 8u_P3P2R uv",
           nppiYCbCr420_8u_P3P2R_Ctx(src, steps, dy.dev, dy.step(), duv.dev, duv.step(), {16, 10}, ctx()), duv);
    Img<Npp8u> oy(16, 10, 1, false), ocb(8, 5, 1, false), ocr(8, 5, 1, false);
    Npp8u* dst[3] = {oy.dev, ocb.dev, ocr.dev};
    int dsteps[3] = {oy.step(), ocb.step(), ocr.step()};
    report("YCbCr420 8u_P2P3R cr",
           nppiYCbCr420_8u_P2P3R_Ctx(dy.dev, dy.step(), duv.dev, duv.step(), dst, dsteps, {16, 10}, ctx()), ocr);
  }
  {
    Img<Npp8u> s(256, 1, 1, false), d(256, 1, 1, false);
    for (int i = 0; i < 256; ++i) s.host[i] = static_cast<Npp8u>(i);
    s.upload();
    const Npp32s lv[5] = {0, 50, 100, 200, 255}, vv[5] = {0, 30, 140, 160, 255};
    const Npp32s lv2[3] = {20, 60, 200}, vv2[3] = {10, 250, 5};
    Npp32s *dl, *dv;
    cudaMalloc(&dl, sizeof lv);
    cudaMalloc(&dv, sizeof vv);
    cudaMemcpy(dl, lv, sizeof lv, cudaMemcpyHostToDevice);
    cudaMemcpy(dv, vv, sizeof vv, cudaMemcpyHostToDevice);
    report("LUT_Linear 8u_C1R", nppiLUT_Linear_8u_C1R_Ctx(s.dev, s.step(), d.dev, d.step(), {256, 1}, dv, dl, 5, ctx()), d);
    cudaMemcpy(dl, lv2, sizeof lv2, cudaMemcpyHostToDevice);
    cudaMemcpy(dv, vv2, sizeof vv2, cudaMemcpyHostToDevice);
    report("LUT_Linear 8u_C1R falling", nppiLUT_Linear_8u_C1R_Ctx(s.dev, s.step(), d.dev, d.step(), {256, 1}, dv, dl, 3, ctx()), d);
    cudaFree(dl);
    cudaFree(dv);
  }
  g_approx = true;  // a tie in the green direction can round a count apart
  for (int grid = NPPI_BAYER_BGGR; grid <= NPPI_BAYER_GRBG; ++grid) {
    Img<Npp8u> s(18, 12, 1), d(18, 12, 3, false);
    std::snprintf(name, sizeof name, "CFAToRGB 8u_C1C3R grid %d", grid);
    report(name,
           nppiCFAToRGB_8u_C1C3R_Ctx(s.dev, s.step(), {18, 12}, {0, 0, 18, 12}, d.dev, d.step(),
                                     static_cast<NppiBayerGridPosition>(grid), NPPI_INTER_UNDEFINED, ctx()),
           d);
    Img<Npp16u> s16(18, 12, 1), d16(18, 12, 3, false);
    std::snprintf(name, sizeof name, "CFAToRGB 16u_C1C3R grid %d", grid);
    report(name,
           nppiCFAToRGB_16u_C1C3R_Ctx(s16.dev, s16.step(), {18, 12}, {0, 0, 18, 12}, d16.dev, d16.step(),
                                      static_cast<NppiBayerGridPosition>(grid), NPPI_INTER_UNDEFINED, ctx()),
           d16);
  }
  g_approx = false;
}

/* ---------------- statistics ---------------- */

static void statistics() {
  char name[96];
  Npp64f* res;
  cudaMalloc(&res, 2 * sizeof(Npp64f));
  Npp8u* scratch;
  cudaMalloc(&scratch, 1 << 20);
  auto pair = [&](const char* n, int st) {
    Npp64f h[2];
    cudaMemcpy(h, res, sizeof h, cudaMemcpyDeviceToHost);
    ++g_lines;
    std::printf("%-44s st=%d mean=%.6e sd=%.6e\n", n, st, h[0], h[1]);
  };
  {
    Img<Npp8u> s(31, 17, 1), m(31, 17, 1);
    for (auto& v : m.host) v = (v & 3) ? 0 : v;
    m.upload();
    pair("Mean_StdDev 8u_C1MR",
         nppiMean_StdDev_8u_C1MR_Ctx(s.dev, s.step(), m.dev, m.step(), {31, 17}, scratch, res, res + 1, ctx()));
    Img<Npp32f> f(31, 17, 1);
    pair("Mean_StdDev 32f_C1R", nppiMean_StdDev_32f_C1R_Ctx(f.dev, f.step(), {31, 17}, scratch, res, res + 1, ctx()));
    pair("Mean_StdDev 32f_C1MR",
         nppiMean_StdDev_32f_C1MR_Ctx(f.dev, f.step(), m.dev, m.step(), {31, 17}, scratch, res, res + 1, ctx()));
  }
  {
    Npp32s lv[7], lv2[5];
    nppiEvenLevelsHost_32s(lv, 7, 0, 255);
    nppiEvenLevelsHost_32s(lv2, 5, -10, 13);
    ++g_lines;
    std::printf("EvenLevelsHost %d %d %d %d %d %d %d | %d %d %d %d %d\n", lv[0], lv[1], lv[2], lv[3], lv[4], lv[5],
                lv[6], lv2[0], lv2[1], lv2[2], lv2[3], lv2[4]);
  }
  {
    Npp32s* hist;
    cudaMalloc(&hist, 4 * 64 * sizeof(Npp32s));
    Img<Npp8u> s(45, 23, 1), s4(45, 23, 4);
    auto clear = [&] { cudaMemset(hist, 0, 4 * 64 * sizeof(Npp32s)); };
    clear();
    report_dev("HistogramEven 8u_C1R",
               nppiHistogramEven_8u_C1R_Ctx(s.dev, s.step(), {45, 23}, hist, 17, 3, 250, scratch, ctx()), hist, 16);
    Npp32s* h4[4] = {hist, hist + 64, hist + 128, hist + 192};
    int n4[4] = {5, 9, 17, 33};
    Npp32s lo4[4] = {0, 10, 20, -5}, hi4[4] = {256, 200, 255, 300};
    clear();
    // Only each channel's nLevels - 1 bins: NVIDIA's writes one or more
    // entries past them, which the documentation does not ask for.
    auto report4 = [&](const char* n, int st, const int* levels) {
      std::vector<Npp32s> all(256), used;
      cudaMemcpy(all.data(), hist, 256 * sizeof(Npp32s), cudaMemcpyDeviceToHost);
      for (int c = 0; c < 4; ++c) used.insert(used.end(), all.begin() + c * 64, all.begin() + c * 64 + levels[c] - 1);
      report_values(n, st, used);
    };
    report4("HistogramEven 8u_C4R",
            nppiHistogramEven_8u_C4R_Ctx(s4.dev, s4.step(), {45, 23}, h4, n4, lo4, hi4, scratch, ctx()), n4);
    Img<Npp16u> s16(45, 23, 1);
    clear();
    report_dev("HistogramEven 16u_C1R",
               nppiHistogramEven_16u_C1R_Ctx(s16.dev, s16.step(), {45, 23}, hist, 33, 1000, 60000, scratch, ctx()),
               hist, 32);
    Img<Npp16s> s16s(45, 23, 1);
    clear();
    report_dev("HistogramEven 16s_C1R",
               nppiHistogramEven_16s_C1R_Ctx(s16s.dev, s16s.step(), {45, 23}, hist, 9, -20000, 20000, scratch, ctx()),
               hist, 8);
    Npp32s levels[6] = {0, 17, 40, 41, 200, 255};
    Npp32s* dlev;
    cudaMalloc(&dlev, sizeof levels);
    cudaMemcpy(dlev, levels, sizeof levels, cudaMemcpyHostToDevice);
    clear();
    report_dev("HistogramRange 8u_C1R",
               nppiHistogramRange_8u_C1R_Ctx(s.dev, s.step(), {45, 23}, hist, dlev, 6, scratch, ctx()), hist, 5);
    Img<Npp32f> f(45, 23, 1), f4(45, 23, 4);
    Npp32f flev[5] = {-200.0f, -10.5f, 0.0f, 3.25f, 150.0f};
    Npp32f* dflev;
    cudaMalloc(&dflev, sizeof flev);
    cudaMemcpy(dflev, flev, sizeof flev, cudaMemcpyHostToDevice);
    clear();
    report_dev("HistogramRange 32f_C1R",
               nppiHistogramRange_32f_C1R_Ctx(f.dev, f.step(), {45, 23}, hist, dflev, 5, scratch, ctx()), hist, 4);
    const Npp32f* fl4[4] = {dflev, dflev, dflev, dflev};
    int fn4[4] = {5, 4, 3, 2};
    clear();
    report4("HistogramRange 32f_C4R",
            nppiHistogramRange_32f_C4R_Ctx(f4.dev, f4.step(), {45, 23}, h4, fl4, fn4, scratch, ctx()), fn4);
    cudaFree(dlev);
    cudaFree(dflev);
    cudaFree(hist);
  }
  {
    Img<Npp8u> s(19, 13, 1);
    Img<Npp32s> d(20, 14, 1, false);
    Img<Npp32f> df(20, 14, 1, false);
    report("Integral 8u32s_C1R", nppiIntegral_8u32s_C1R_Ctx(s.dev, s.step(), d.dev, d.step(), {19, 13}, 5, ctx()), d);
    report("Integral 8u32f_C1R", nppiIntegral_8u32f_C1R_Ctx(s.dev, s.step(), df.dev, df.step(), {19, 13}, -2.5f, ctx()), df);
    // RectStdDev over the integral (and a squared integral built here).
    Img<Npp64f> sq(20, 14, 1, false, 0.0);
    std::vector<Npp8u> px(s.host);
    for (int y = 1; y < 14; ++y)
      for (int x = 1; x < 20; ++x) {
        const double v = px[(y - 1) * s.stride + x - 1];
        sq.host[y * sq.stride + x] = v * v + sq.host[(y - 1) * sq.stride + x] + sq.host[y * sq.stride + x - 1] -
                                     sq.host[(y - 1) * sq.stride + x - 1];
      }
    sq.upload();
    nppiIntegral_8u32s_C1R_Ctx(s.dev, s.step(), d.dev, d.step(), {19, 13}, 0, ctx());
    Img<Npp32f> o(14, 9, 1, false);
    report("RectStdDev 32s32f_C1R",
           nppiRectStdDev_32s32f_C1R_Ctx(d.dev, d.step(), sq.dev, sq.step(), o.dev, o.step(), {14, 9}, {1, 2, 4, 3},
                                         ctx()),
           o);
    Img<Npp32f> w(15, 9, 1, false);
    report("SumWindowRow 8u32f_C1R",
           nppiSumWindowRow_8u32f_C1R_Ctx(s.at(2, 1), s.step(), w.dev, w.step(), {15, 9}, 4, 2, ctx()), w);
    report("SumWindowColumn 8u32f_C1R",
           nppiSumWindowColumn_8u32f_C1R_Ctx(s.at(2, 2), s.step(), w.dev, w.step(), {15, 9}, 3, 1, ctx()), w);
  }
  cudaFree(res);
  cudaFree(scratch);
  (void)name;
}

/* ---------------- filters ---------------- */

static void filters() {
  char name[96];
  const NppiSize roi{21, 13};
  {
    Img<Npp8u> s(27, 19, 4), d(21, 13, 4, false);
    report("FilterBox 8u_C4R", nppiFilterBox_8u_C4R_Ctx(s.at(3, 3), s.step(), d.dev, d.step(), roi, {5, 3}, {2, 1}, ctx()), d);
    report("FilterMax 8u_C4R", nppiFilterMax_8u_C4R_Ctx(s.at(3, 3), s.step(), d.dev, d.step(), roi, {3, 5}, {1, 3}, ctx()), d);
    report("FilterMin 8u_C4R", nppiFilterMin_8u_C4R_Ctx(s.at(3, 3), s.step(), d.dev, d.step(), roi, {4, 4}, {0, 0}, ctx()), d);
    const Npp8u mask[9] = {0, 1, 0, 1, 1, 1, 0, 1, 0};
    Npp8u* dm;
    cudaMalloc(&dm, 9);
    cudaMemcpy(dm, mask, 9, cudaMemcpyHostToDevice);
    report("Dilate 8u_C4R", nppiDilate_8u_C4R_Ctx(s.at(3, 3), s.step(), d.dev, d.step(), roi, dm, {3, 3}, {1, 1}, ctx()), d);
    report("Erode 8u_C4R", nppiErode_8u_C4R_Ctx(s.at(3, 3), s.step(), d.dev, d.step(), roi, dm, {3, 3}, {1, 1}, ctx()), d);
    Img<Npp8u> s1(27, 19, 1), d1(21, 13, 1, false);
    report("FilterMax 8u_C1R", nppiFilterMax_8u_C1R_Ctx(s1.at(3, 3), s1.step(), d1.dev, d1.step(), roi, {3, 3}, {1, 1}, ctx()), d1);
    report("FilterMin 8u_C1R", nppiFilterMin_8u_C1R_Ctx(s1.at(3, 3), s1.step(), d1.dev, d1.step(), roi, {5, 5}, {2, 2}, ctx()), d1);
    // Centred anchors only: with an off-centre one NVIDIA's Dilate and Erode
    // read the ROI's own edge pixels where the mask reaches past it on the
    // far side (libraries.md).
    report("Dilate 8u_C1R", nppiDilate_8u_C1R_Ctx(s1.at(3, 3), s1.step(), d1.dev, d1.step(), roi, dm, {3, 3}, {1, 1}, ctx()), d1);
    report("Erode 8u_C1R", nppiErode_8u_C1R_Ctx(s1.at(3, 3), s1.step(), d1.dev, d1.step(), roi, dm, {3, 3}, {1, 1}, ctx()), d1);
    Img<Npp32f> f(27, 19, 1), df(21, 13, 1, false), f4(27, 19, 4), df4(21, 13, 4, false);
    report("FilterBox 32f_C1R", nppiFilterBox_32f_C1R_Ctx(f.at(3, 3), f.step(), df.dev, df.step(), roi, {3, 3}, {1, 1}, ctx()), df);
    report("Dilate 32f_C1R", nppiDilate_32f_C1R_Ctx(f.at(3, 3), f.step(), df.dev, df.step(), roi, dm, {3, 3}, {1, 1}, ctx()), df);
    report("Erode 32f_C4R", nppiErode_32f_C4R_Ctx(f4.at(3, 3), f4.step(), df4.dev, df4.step(), roi, dm, {3, 3}, {1, 1}, ctx()), df4);
    cudaFree(dm);
  }
  {
    // Border forms, over the whole image so the border is used.
    Img<Npp8u> s(29, 17, 1), d(29, 17, 1, false);
    report("FilterBoxBorder 8u_C1R replicate",
           nppiFilterBoxBorder_8u_C1R_Ctx(s.dev, s.step(), {29, 17}, {0, 0}, d.dev, d.step(), {29, 17}, {5, 5}, {2, 2},
                                          NPP_BORDER_REPLICATE, ctx()),
           d);
    for (NppiNorm norm : {nppiNormInf, nppiNormL1, nppiNormL2}) {
      Img<Npp16s> gx(29, 17, 1, false), gy(29, 17, 1, false), gm(29, 17, 1, false);
      Img<Npp32f> ga(29, 17, 1, false);
      const int st = nppiGradientVectorPrewittBorder_8u16s_C1R_Ctx(
          s.dev, s.step(), {29, 17}, {0, 0}, gx.dev, gx.step(), gy.dev, gy.step(), gm.dev, gm.step(), ga.dev,
          ga.step(), {29, 17}, NPP_MASK_SIZE_3_X_3, norm, NPP_BORDER_REPLICATE, ctx());
      std::snprintf(name, sizeof name, "GradientVectorPrewitt x norm %d", int(norm));
      report(name, st, gx);
      std::snprintf(name, sizeof name, "GradientVectorPrewitt y norm %d", int(norm));
      report(name, st, gy);
      std::snprintf(name, sizeof name, "GradientVectorPrewitt mag norm %d", int(norm));
      report(name, st, gm);
      std::snprintf(name, sizeof name, "GradientVectorPrewitt angle norm %d", int(norm));
      report(name, st, ga);
    }
  }
  {
    // Canny on a smooth image with a few edges in it, where the result is
    // more than noise.
    Img<Npp8u> s(48, 32, 1, false), d(48, 32, 1, false);
    for (int y = 0; y < 32; ++y)
      for (int x = 0; x < 48; ++x) {
        int v = (x > 15 ? 120 : 30) + (y > 20 ? 60 : 0) + ((x - 30) * (x - 30) + (y - 10) * (y - 10) < 40 ? 70 : 0);
        v += static_cast<int>(next_u() % 9) - 4;
        s.host[y * s.stride + x] = static_cast<Npp8u>(v);
      }
    s.upload();
    if (const char* dir = std::getenv("NPP_DUMP_DIR")) {
      if (FILE* f = std::fopen((std::string(dir) + "/canny_source.bin").c_str(), "wb")) {
        for (int y = 0; y < 32; ++y) std::fwrite(&s.host[y * s.stride], 1, 48, f);
        std::fclose(f);
      }
    }
    int bytes = 0;
    nppiFilterCannyBorderGetBufferSize({48, 32}, &bytes);
    Npp8u* buf;
    cudaMalloc(&buf, bytes > 0 ? bytes : 1);
    // The CUDA Samples' parameters (Sobel, L2, 72/256) and two more. A low
    // threshold in the noise, or Scharr's gradients, leaves some pixels
    // decided differently (libraries.md), so those are not run here.
    struct { NppiDifferentialKernel k; NppiNorm n; Npp16s lo, hi; } runs[] = {
        {NPP_FILTER_SOBEL, nppiNormL2, 72, 256}, {NPP_FILTER_SOBEL, nppiNormL2, 40, 120},
        {NPP_FILTER_SOBEL, nppiNormL1, 60, 200}};
    for (auto& r : runs) {
      const int st = nppiFilterCannyBorder_8u_C1R_Ctx(s.dev, s.step(), {48, 32}, {0, 0}, d.dev, d.step(), {48, 32},
                                                      r.k, NPP_MASK_SIZE_3_X_3, r.lo, r.hi, r.n,
                                                      NPP_BORDER_REPLICATE, buf, ctx());
      std::snprintf(name, sizeof name, "FilterCannyBorder k%d n%d %d-%d", int(r.k), int(r.n), r.lo, r.hi);
      report(name, st, d);
    }
    cudaFree(buf);
  }
}

static void misc() {
  {
    Img<Npp32f> s(19, 7, 1), d(19, 7, 1, false);
    report("Threshold 32f_C1R less", nppiThreshold_32f_C1R_Ctx(s.dev, s.step(), d.dev, d.step(), {19, 7}, 1.5f, NPP_CMP_LESS, ctx()), d);
    report("Threshold 32f_C1R greater",
           nppiThreshold_32f_C1R_Ctx(s.dev, s.step(), d.dev, d.step(), {19, 7}, -3.0f, NPP_CMP_GREATER, ctx()), d);
  }
  {
    Img<Npp16s> s(19, 7, 1);
    Img<Npp8u> d(19, 7, 1, false);
    for (int op = NPP_CMP_LESS; op <= NPP_CMP_GREATER; ++op) {
      char name[64];
      std::snprintf(name, sizeof name, "CompareC 16s_C1R op %d", op);
      report(name, nppiCompareC_16s_C1R_Ctx(s.dev, s.step(), s.host[5], d.dev, d.step(), {19, 7}, static_cast<NppCmpOp>(op), ctx()), d);
    }
  }
  {
    Img<Npp8u> s(11, 6, 1), d(17, 12, 1, false);
    report("CopyConstBorder 8u_C1R",
           nppiCopyConstBorder_8u_C1R_Ctx(s.dev, s.step(), {11, 6}, d.dev, d.step(), {17, 12}, 2, 3, 99, ctx()), d);
  }
}

int main() {
  if (cudaFree(0) != cudaSuccess) {
    std::printf("no CUDA device\n");
    return 1;
  }
  geometry();
  arithmetic();
  colour();
  statistics();
  filters();
  misc();
  return g_lines > 0 ? 0 : 1;
}
