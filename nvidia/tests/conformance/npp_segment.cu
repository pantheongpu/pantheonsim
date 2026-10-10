// Conformance for NPP's watershed segmentation and marker-label compression
// (nppiSegmentWatershed_8u/16u_C1IR, its buffer-size queries, and
// nppiCompressMarkerLabelsUF_32u_C1IR) -- the three functions of the CUDA
// Samples' watershedSegmentationNPP.
//
// NVIDIA does not publish the algorithm, so what is pinned here is what NPP
// 13.0 printed on an RTX 3060 (golden/npp_segment.rtx3060.txt). The program
// checks what the simulator reproduces exactly:
//   - the segmented image of any input under 8-way connectivity (every image
//     below, with and without equal values, every boundary type, a padded
//     row pitch);
//   - the marker labels of images whose values are all different;
//   - 4-way connectivity on images up to 4x4 (on larger ones NPP leaves some
//     pixels near the right and bottom edges unwritten, in a pattern the
//     simulator does not reproduce);
//   - label compression of any label image, and every status and buffer size.
// What it leaves out -- the labels of images with equal neighbouring values,
// which follow plateau rules the simulator fits to within a few percent of the
// pixels -- is listed in nvidia/docs/libraries.md.
//
// Every result is a status and checksums (sum and position-weighted sum), so
// the golden file pins every pixel. Inputs are generated and identical on both
// sides; the calls are the _Ctx forms, the only ones CUDA 13 declares.
#include <npp.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
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

static unsigned g_rng = 7;
static unsigned next_u() {
  g_rng = g_rng * 1664525u + 1013904223u;
  return g_rng >> 8;
}

template <class T>
static void digest(const std::vector<T>& v, unsigned long long* sum, unsigned long long* wsum) {
  *sum = *wsum = 0;
  for (size_t i = 0; i < v.size(); ++i) {
    *sum += v[i];
    *wsum += static_cast<unsigned long long>(v[i]) * (i % 9973 + 1);
  }
}

template <class T>
static std::vector<T> download(const void* dev, int step, int w, int h) {
  std::vector<T> out(static_cast<size_t>(w) * h);
  for (int y = 0; y < h; ++y)
    cudaMemcpy(out.data() + static_cast<size_t>(y) * w, static_cast<const char*>(dev) + static_cast<size_t>(y) * step,
               w * sizeof(T), cudaMemcpyDeviceToHost);
  return out;
}

// `count` different values in a shuffled order, from [0, range).
static std::vector<unsigned> distinct_values(int count, unsigned range) {
  std::vector<unsigned> pool(range);
  std::iota(pool.begin(), pool.end(), 0u);
  for (unsigned i = range - 1; i > 0; --i) std::swap(pool[i], pool[next_u() % (i + 1)]);
  pool.resize(count);
  return pool;
}

static std::vector<unsigned> random_values(int count, unsigned range) {
  std::vector<unsigned> v(count);
  for (auto& x : v) x = next_u() % range;
  return v;
}

// The buffer-size query's out-parameter is int on CUDA 12 and size_t from 13.
static int buffer_size(NppStatus (*f)(NppiSize, size_t*), NppiSize roi, unsigned long long* out) {
  size_t v = 0;
  const int st = f(roi, &v);
  *out = v;
  return st;
}
static int buffer_size(NppStatus (*f)(NppiSize, int*), NppiSize roi, unsigned long long* out) {
  int v = 0;
  const int st = f(roi, &v);
  *out = static_cast<unsigned>(v);
  return st;
}

struct Run {
  int status = 0;
  std::vector<unsigned> labels;
  std::vector<unsigned> pixels;  // widened
  unsigned long long guard = 0;  // the bytes between rows, which must not change
};

// Runs the watershed on `vals` (W x H) in an image whose rows are `pad` bytes
// wider than the pixels, the padding holding a canary.
template <class T>
static Run watershed(const std::vector<unsigned>& vals, int W, int H, int pad, NppiNorm norm, int boundary,
                     bool want_labels) {
  Run r;
  NppiSize roi{W, H};
  unsigned long long bs = 0;
  if constexpr (sizeof(T) == 1)
    r.status = buffer_size(nppiSegmentWatershedGetBufferSize_8u_C1R, roi, &bs);
  else
    r.status = buffer_size(nppiSegmentWatershedGetBufferSize_16u_C1R, roi, &bs);
  const int step = W * static_cast<int>(sizeof(T)) + pad;
  std::vector<unsigned char> host(static_cast<size_t>(step) * H, 0xA5);
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
      T v = static_cast<T>(vals[static_cast<size_t>(y) * W + x]);
      std::memcpy(&host[static_cast<size_t>(y) * step + x * sizeof(T)], &v, sizeof(T));
    }
  void *img = nullptr, *lab = nullptr, *buf = nullptr;
  cudaMalloc(&img, host.size());
  cudaMemcpy(img, host.data(), host.size(), cudaMemcpyHostToDevice);
  cudaMalloc(&lab, static_cast<size_t>(W) * H * 4);
  cudaMemset(lab, 0, static_cast<size_t>(W) * H * 4);
  cudaMalloc(&buf, bs ? bs : 1);
  NppStreamContext c = ctx();
  Npp32u* labels = want_labels ? static_cast<Npp32u*>(lab) : nullptr;
  const auto bt = static_cast<NppiWatershedSegmentBoundaryType>(boundary);
  if constexpr (sizeof(T) == 1)
    r.status = nppiSegmentWatershed_8u_C1IR_Ctx(static_cast<Npp8u*>(img), step, labels, W * 4, norm, bt, roi,
                                                static_cast<Npp8u*>(buf), c);
  else
    r.status = nppiSegmentWatershed_16u_C1IR_Ctx(static_cast<Npp16u*>(img), step, labels, W * 4, norm, bt, roi,
                                                 static_cast<Npp8u*>(buf), c);
  cudaDeviceSynchronize();
  std::vector<T> px = download<T>(img, step, W, H);
  r.pixels.assign(px.begin(), px.end());
  if (want_labels) r.labels = download<unsigned>(lab, W * 4, W, H);
  std::vector<unsigned char> after(host.size());
  cudaMemcpy(after.data(), img, after.size(), cudaMemcpyDeviceToHost);
  for (int y = 0; y < H; ++y)
    for (int p = W * static_cast<int>(sizeof(T)); p < step; ++p) r.guard += after[static_cast<size_t>(y) * step + p];
  cudaFree(img);
  cudaFree(lab);
  cudaFree(buf);
  return r;
}

static void print_run(const char* what, const Run& r, bool with_labels) {
  unsigned long long s, w;
  digest(r.pixels, &s, &w);
  std::printf("%-40s st=%d image sum=%llu wsum=%llu guard=%llu", what, r.status, s, w, r.guard);
  if (with_labels) {
    digest(r.labels, &s, &w);
    std::printf(" labels sum=%llu wsum=%llu", s, w);
  }
  std::printf("\n");
}

static const char* kBoundary[] = {"none", "black", "white", "contrast", "only"};

// One image through every boundary type.
template <class T>
static void all_boundaries(const char* name, const std::vector<unsigned>& vals, int W, int H, int pad, NppiNorm norm,
                           bool labels_exact) {
  for (int b = 0; b < 5; ++b) {
    char what[96];
    std::snprintf(what, sizeof what, "%s %s", name, kBoundary[b]);
    print_run(what, watershed<T>(vals, W, H, pad, norm, b, labels_exact), labels_exact);
  }
}

static void compress_case(const char* name, const std::vector<unsigned>& labels, int W, int H, int start, int step,
                          bool null_buffer = false) {
  int bs = 0;
  const int bst = nppiCompressMarkerLabelsGetBufferSize_32u_C1R(start, &bs);
  const int rows = step == 0 ? 1 : H;
  const size_t bytes = std::max<size_t>(static_cast<size_t>(rows) * std::max(step, W * 4), static_cast<size_t>(W) * H * 4);
  void* d = nullptr;
  cudaMalloc(&d, bytes);
  std::vector<unsigned> mem(bytes / 4, 0xFFFFFFF0u);
  if (step <= W * 4 || step % 4) {
    std::copy(labels.begin(), labels.end(), mem.begin());
  } else {
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) mem[static_cast<size_t>(y) * (step / 4) + x] = labels[static_cast<size_t>(y) * W + x];
  }
  cudaMemcpy(d, mem.data(), bytes, cudaMemcpyHostToDevice);
  void* buf = nullptr;
  cudaMalloc(&buf, (bs > 0 ? bs : 0) + 64);
  int nn = -777;
  NppiSize roi{W, H};
  const int st = nppiCompressMarkerLabelsUF_32u_C1IR_Ctx(static_cast<Npp32u*>(d), step, roi, start, &nn,
                                                         null_buffer ? nullptr : static_cast<Npp8u*>(buf), ctx());
  cudaDeviceSynchronize();
  std::vector<unsigned> out(mem.size());
  cudaMemcpy(out.data(), d, bytes, cudaMemcpyDeviceToHost);
  unsigned long long s, w;
  digest(out, &s, &w);
  std::printf("Compress %-31s size st=%d bytes=%d  st=%d new=%d sum=%llu wsum=%llu\n", name, bst, bs, st, nn, s, w);
  cudaFree(d);
  cudaFree(buf);
}

int main() {
  NppStreamContext c = ctx();
  (void)c;

  // ---- buffer sizes ----
  const int dims[][2] = {{1, 1}, {2, 2},   {3, 3},    {4, 4},    {5, 1},     {6, 1},     {8, 8},       {8, 6},       {16, 16}, {17, 13},
                         {33, 33}, {64, 64}, {100, 50}, {100, 1}, {1, 100}, {512, 512}, {1000, 1}, {1920, 1080}, {4096, 4096}};
  for (const auto& d : dims) {
    NppiSize roi{d[0], d[1]};
    unsigned long long a = 0, b = 0;
    const int sa = buffer_size(nppiSegmentWatershedGetBufferSize_8u_C1R, roi, &a);
    const int sb = buffer_size(nppiSegmentWatershedGetBufferSize_16u_C1R, roi, &b);
    std::printf("WatershedBufferSize %4dx%-4d                st=%d/%d 8u=%llu 16u=%llu\n", d[0], d[1], sa, sb, a, b);
  }
  {
    unsigned long long x = 77;
    const NppiSize bad[] = {{0, 5}, {5, 0}, {-1, 5}, {5, -1}};
    for (const auto& r : bad) {
      x = 77;
      std::printf("WatershedBufferSize %2dx%-2d st=%d out=%llu\n", r.width, r.height,
                  buffer_size(nppiSegmentWatershedGetBufferSize_8u_C1R, r, &x), x);
    }
    NppiSize ok{5, 5};
    std::printf("WatershedBufferSize null                  st=%d\n", nppiSegmentWatershedGetBufferSize_8u_C1R(ok, nullptr));
  }

  // ---- images whose values are all different: image and labels exact ----
  {
    struct { int w, h; } small[] = {{16, 16}, {15, 17}, {12, 21}, {5, 7}, {3, 3}, {2, 2}, {1, 12}, {12, 1}, {20, 12}};
    for (const auto& s : small) {
      char name[64];
      std::snprintf(name, sizeof name, "8u distinct %dx%d", s.w, s.h);
      all_boundaries<Npp8u>(name, distinct_values(s.w * s.h, 256), s.w, s.h, 0, nppiNormInf, true);
    }
    struct { int w, h; } big[] = {{40, 40}, {64, 64}, {100, 80}, {127, 33}, {200, 200}, {1, 500}, {500, 1}};
    for (const auto& s : big) {
      char name[64];
      std::snprintf(name, sizeof name, "16u distinct %dx%d", s.w, s.h);
      all_boundaries<Npp16u>(name, distinct_values(s.w * s.h, 65536), s.w, s.h, 0, nppiNormInf, true);
    }
    // A row pitch wider than the pixels, and no label output at all.
    all_boundaries<Npp8u>("8u distinct 10x20 pitch+13", distinct_values(200, 256), 10, 20, 13, nppiNormInf, true);
    all_boundaries<Npp16u>("16u distinct 33x31 pitch+6", distinct_values(33 * 31, 65536), 33, 31, 6, nppiNormInf, true);
    print_run("8u distinct 16x16 no labels", watershed<Npp8u>(distinct_values(256, 256), 16, 16, 0, nppiNormInf, 0, false), false);
    // 4-way connectivity, on images small enough that NPP writes every pixel.
    struct { int w, h; } four[] = {{4, 4}, {3, 4}, {4, 3}, {2, 2}, {1, 4}, {4, 1}, {3, 3}};
    for (const auto& s : four) {
      for (int rep = 0; rep < 6; ++rep) {
        char name[64];
        std::snprintf(name, sizeof name, "4-way distinct %dx%d #%d", s.w, s.h, rep);
        print_run(name, watershed<Npp8u>(distinct_values(s.w * s.h, 256), s.w, s.h, 0, nppiNormL1, 0, true), true);
      }
      char name[64];
      std::snprintf(name, sizeof name, "4-way distinct %dx%d black", s.w, s.h);
      print_run(name, watershed<Npp16u>(distinct_values(s.w * s.h, 60000), s.w, s.h, 4, nppiNormL1, 1, true), true);
    }
  }

  // ---- images with many equal values: the segmented image is exact ----
  {
    struct { int w, h; unsigned range; } tie[] = {{9, 7, 2}, {16, 16, 3}, {33, 31, 4}, {64, 64, 8}, {50, 70, 16}, {100, 90, 64}, {5, 5, 2}, {1, 30, 3}, {30, 1, 3}};
    for (const auto& s : tie) {
      char name[64];
      std::snprintf(name, sizeof name, "8u ties %dx%d range %u", s.w, s.h, s.range);
      all_boundaries<Npp8u>(name, random_values(s.w * s.h, s.range), s.w, s.h, 3, nppiNormInf, false);
    }
    for (const auto& s : tie) {
      char name[64];
      std::snprintf(name, sizeof name, "16u ties %dx%d range %u", s.w, s.h, s.range * 1000);
      all_boundaries<Npp16u>(name, random_values(s.w * s.h, s.range * 1000), s.w, s.h, 0, nppiNormInf, false);
    }
    // A smooth picture quantised to 8 bits: plateaus and shallow slopes, like a photograph.
    std::vector<unsigned> smooth(96 * 80);
    for (int y = 0; y < 80; ++y)
      for (int x = 0; x < 96; ++x)
        smooth[static_cast<size_t>(y) * 96 + x] =
            static_cast<unsigned>(127.5 + 60 * std::sin(x * 0.11) * std::cos(y * 0.09) + 40 * std::sin((x + y) * 0.05) +
                                  (next_u() % 5) - 2);
    all_boundaries<Npp8u>("8u smooth 96x80", smooth, 96, 80, 0, nppiNormInf, false);
  }

  // ---- label compression ----
  {
    compress_case("all 0", std::vector<unsigned>(12, 0), 4, 3, 12, 16);
    compress_case("all 1", std::vector<unsigned>(12, 1), 4, 3, 12, 16);
    compress_case("0 1 2 3", {0, 1, 2, 3}, 4, 1, 4, 16);
    compress_case("3 2 1 0", {3, 2, 1, 0}, 4, 1, 4, 16);
    compress_case("3 3 0 3", {3, 3, 0, 3}, 4, 1, 4, 16);
    compress_case("no 0: 2 2 3 3", {2, 2, 3, 3}, 4, 1, 4, 16);
    compress_case("beyond the limit: 5 5 9 9", {5, 5, 9, 9}, 4, 1, 4, 16);
    compress_case("mixed", {8, 8, 8, 8, 15, 15, 3, 3}, 8, 1, 8, 32);
    compress_case("max label", {0xFFFFFFFFu, 4, 0xFFFFFFFFu, 4}, 4, 1, 4, 16);
    compress_case("limit below width", {1, 2, 3, 3}, 4, 1, 3, 16);
    compress_case("limit above size", {1, 2, 3, 3}, 4, 1, 5, 16);
    compress_case("two rows", {6, 6, 1, 1, 2, 2}, 3, 2, 6, 12);
    compress_case("two rows, padded pitch", {0, 0, 0, 3, 3, 3}, 3, 2, 6, 16);
    compress_case("two rows, step 0", {0, 0, 0, 3, 3, 3}, 3, 2, 6, 0);
    for (int rep = 0; rep < 12; ++rep) {
      const int W = 1 + next_u() % 40, H = 1 + next_u() % 30, N = W * H;
      char name[64];
      std::snprintf(name, sizeof name, "random %dx%d #%d", W, H, rep);
      compress_case(name, random_values(N, rep % 3 == 0 ? N : (rep % 3 == 1 ? N + N / 2 + 2 : 1 + next_u() % 5)), W, H, N, W * 4);
    }
    // The segmentation's own labels through the compression.
    for (int rep = 0; rep < 4; ++rep) {
      const int W = 20 + 7 * rep, H = 18 + 5 * rep;
      Run r = watershed<Npp16u>(distinct_values(W * H, 65536), W, H, 0, nppiNormInf, 0, true);
      char name[64];
      std::snprintf(name, sizeof name, "watershed labels %dx%d", W, H);
      compress_case(name, r.labels, W, H, W * H, W * 4);
    }
    int sizes[] = {-5, 0, 1, 2, 3, 4, 5, 16, 100, 1000, 4096, 65535, 65536, 1048576, 16777216};
    for (int n : sizes) {
      int bs = -1;
      const int st = nppiCompressMarkerLabelsGetBufferSize_32u_C1R(n, &bs);
      std::printf("CompressBufferSize %-10d st=%d bytes=%d\n", n, st, bs);
    }
    std::printf("CompressBufferSize null st=%d\n", nppiCompressMarkerLabelsGetBufferSize_32u_C1R(5, nullptr));
  }

  // ---- statuses ----
  {
    const int W = 8, H = 8;
    NppiSize roi{W, H};
    unsigned long long bs = 0;
    buffer_size(nppiSegmentWatershedGetBufferSize_8u_C1R, roi, &bs);
    Npp8u *d, *buf;
    Npp32u* lab;
    cudaMalloc(&d, W * H);
    cudaMalloc(&lab, W * H * 4);
    cudaMalloc(&buf, bs);
    cudaMemset(d, 7, W * H);
    auto ws = [&](const char* what, Npp8u* pd, Npp32u* pl, int norm, int bd, NppiSize r, Npp8u* pb) {
      const int st = nppiSegmentWatershed_8u_C1IR_Ctx(pd, W, pl, W * 4, static_cast<NppiNorm>(norm),
                                                      static_cast<NppiWatershedSegmentBoundaryType>(bd), r, pb, ctx());
      cudaDeviceSynchronize();
      std::printf("Watershed status %-36s %d\n", what, st);
    };
    ws("ok", d, lab, nppiNormInf, 0, roi, buf);
    ws("ok, 4-way", d, lab, nppiNormL1, 0, roi, buf);
    ws("null image", nullptr, lab, nppiNormInf, 0, roi, buf);
    ws("null labels", d, nullptr, nppiNormInf, 0, roi, buf);
    ws("null buffer", d, lab, nppiNormInf, 0, roi, nullptr);
    ws("null image and bad size", nullptr, lab, nppiNormInf, 0, {0, 8}, buf);
    ws("null buffer and bad size", d, lab, nppiNormInf, 0, {0, 8}, nullptr);
    ws("null image and bad norm", nullptr, lab, nppiNormL2, 0, roi, buf);
    ws("null buffer and bad boundary", d, lab, nppiNormInf, 9, roi, nullptr);
    ws("zero width", d, lab, nppiNormInf, 0, {0, 8}, buf);
    ws("zero height", d, lab, nppiNormInf, 0, {8, 0}, buf);
    ws("negative width", d, lab, nppiNormInf, 0, {-1, 8}, buf);
    ws("bad size and bad norm", d, lab, nppiNormL2, 0, {0, 8}, buf);
    ws("bad size and bad boundary", d, lab, nppiNormInf, 9, {0, 8}, buf);
    ws("norm L2", d, lab, nppiNormL2, 0, roi, buf);
    ws("norm 99", d, lab, 99, 0, roi, buf);
    ws("bad norm and bad boundary", d, lab, nppiNormL2, 9, roi, buf);
    ws("boundary 5", d, lab, nppiNormInf, 5, roi, buf);
    ws("boundary -1", d, lab, nppiNormInf, -1, roi, buf);
    ws("null labels and bad boundary", d, nullptr, nppiNormInf, 9, roi, buf);
    ws("1x1", d, lab, nppiNormInf, 0, {1, 1}, buf);
    ws("1x8", d, lab, nppiNormInf, 0, {1, 8}, buf);
    ws("8x1", d, lab, nppiNormInf, 0, {8, 1}, buf);

    Npp32u* cb;
    cudaMalloc(&cb, 70000);
    cudaMemset(cb, 0, 70000);
    Npp8u* cbuf;
    cudaMalloc(&cbuf, 70000);
    auto cp = [&](const char* what, Npp32u* pd, int step, NppiSize r, int start, int* pn, Npp8u* pb) {
      const int st = nppiCompressMarkerLabelsUF_32u_C1IR_Ctx(pd, step, r, start, pn, pb, ctx());
      cudaDeviceSynchronize();
      std::printf("Compress status %-37s %d\n", what, st);
    };
    int nn = 0;
    cp("ok", cb, W * 4, roi, W * H, &nn, cbuf);
    cp("null labels", nullptr, W * 4, roi, W * H, &nn, cbuf);
    cp("null count", cb, W * 4, roi, W * H, nullptr, cbuf);
    cp("null buffer", cb, W * 4, roi, W * H, &nn, nullptr);
    cp("null labels and bad size", nullptr, W * 4, {0, 8}, W * H, &nn, cbuf);
    cp("null count and bad size", cb, W * 4, {0, 8}, W * H, nullptr, cbuf);
    cp("bad size and start 0", cb, W * 4, {0, 8}, 0, &nn, cbuf);
    cp("start 0 and null buffer", cb, W * 4, roi, 0, &nn, nullptr);
    cp("start 0 and null labels", nullptr, W * 4, roi, 0, &nn, cbuf);
    cp("zero height", cb, W * 4, {8, 0}, W * H, &nn, cbuf);
    cp("negative width", cb, W * 4, {-1, 8}, W * H, &nn, cbuf);
    cp("start 0", cb, W * 4, roi, 0, &nn, cbuf);
    cp("start -5", cb, W * 4, roi, -5, &nn, cbuf);
    cp("start 10 (not W*H)", cb, W * 4, roi, 10, &nn, cbuf);
    cp("step 0", cb, 0, roi, W * H, &nn, cbuf);
  }
  return 0;
}
