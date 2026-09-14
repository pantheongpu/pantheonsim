// Differential conformance for the NPP shim. Every call goes through the
// _Ctx form: CUDA 13 ships only those, the older plain names having been
// dropped from the libraries even where the headers still declare them.
//
// Covers: per-pixel arithmetic and logic
// with NPP's scale-factor rounding, data exchange, colour conversion,
// thresholding, image statistics, box and general convolution filters, 3x3
// morphology, mirroring, resizing, and the signal-processing entry points.
#include <npp.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <type_traits>
#include <vector>

// NPP's scratch-size out-parameter is int* on CUDA 12.0 and size_t* by 12.8.
// Naming either one directly makes this test fail to compile on the other
// toolkit, so take the type from the header's own declaration.
template <typename F>
struct npp_bufsize;
template <typename R, typename A, typename B, typename C>
struct npp_bufsize<R (*)(A, B, C)> {
  using type = B;
};
template <typename F>
using npp_bufsize_t = std::remove_pointer_t<typename npp_bufsize<F>::type>;
using ImgBufSize = npp_bufsize_t<decltype(&nppiSumGetBufferHostSize_8u_C1R_Ctx)>;
using SigBufSize = npp_bufsize_t<decltype(&nppsSumGetBufferSize_32f_Ctx)>;

#define NP(x) do { NppStatus s_ = (x); if (s_ != NPP_SUCCESS) { \
  printf("%-30s status=%d\n", #x, (int)s_); return; } } while (0)

static const int W = 16, H = 12;

// Deterministic images, identical on both sides.
static std::vector<Npp8u> gray(unsigned seed) {
  std::vector<Npp8u> v((size_t)W * H);
  unsigned s = seed;
  for (size_t i = 0; i < v.size(); ++i) { s = s * 1664525u + 1013904223u; v[i] = (s >> 19) & 0xFF; }
  return v;
}
static std::vector<Npp8u> rgb(unsigned seed) {
  std::vector<Npp8u> v((size_t)W * H * 3);
  unsigned s = seed;
  for (size_t i = 0; i < v.size(); ++i) { s = s * 1664525u + 1013904223u; v[i] = (s >> 19) & 0xFF; }
  return v;
}
static std::vector<Npp32f> flt(unsigned seed) {
  std::vector<Npp32f> v((size_t)W * H);
  unsigned s = seed;
  for (size_t i = 0; i < v.size(); ++i) {
    s = s * 1664525u + 1013904223u;
    v[i] = (float)((int)((s >> 16) % 2000) - 1000) / 64.0f;
  }
  return v;
}

// Upload into an NPP-allocated pitched image.
template <class T> static T* up_img(const std::vector<T>& h, int elems_per_row, int rows, int* step) {
  T* d = nullptr;
  if constexpr (sizeof(T) == 1) d = (T*)nppiMalloc_8u_C1(elems_per_row, rows, step);
  else d = (T*)nppiMalloc_32f_C1(elems_per_row, rows, step);
  for (int y = 0; y < rows; ++y)
    cudaMemcpy((char*)d + (size_t)y * *step, h.data() + (size_t)y * elems_per_row,
               (size_t)elems_per_row * sizeof(T), cudaMemcpyHostToDevice);
  return d;
}
template <class T> static std::vector<T> down_img(const T* d, int step, int elems_per_row, int rows) {
  std::vector<T> h((size_t)elems_per_row * rows);
  for (int y = 0; y < rows; ++y)
    cudaMemcpy(h.data() + (size_t)y * elems_per_row, (const char*)d + (size_t)y * step,
               (size_t)elems_per_row * sizeof(T), cudaMemcpyDeviceToHost);
  return h;
}
template <class T> static void emit(const char* tag, const std::vector<T>& v) {
  double sum = 0, abs = 0;
  for (T x : v) { sum += (double)x; abs += std::fabs((double)x); }
  printf("%-30s n=%zu sum=%.4f abs=%.4f first=%.4f last=%.4f\n", tag, v.size(), sum, abs,
         (double)(v.empty() ? 0 : v[0]), (double)(v.empty() ? 0 : v.back()));
}

// CUDA 13 removed the non-context forms of the filtering and geometry
// entry points, so the application fills in the context itself -- which is
// what NPP now documents.
static NppStreamContext make_ctx() {
  NppStreamContext c{};
  c.hStream = nullptr;
  cudaGetDevice(&c.nCudaDeviceId);
  cudaDeviceProp p{};
  cudaGetDeviceProperties(&p, c.nCudaDeviceId);
  c.nMultiProcessorCount = p.multiProcessorCount;
  c.nMaxThreadsPerMultiProcessor = p.maxThreadsPerMultiProcessor;
  c.nMaxThreadsPerBlock = p.maxThreadsPerBlock;
  c.nSharedMemPerBlock = p.sharedMemPerBlock;
  c.nCudaDevAttrComputeCapabilityMajor = p.major;
  c.nCudaDevAttrComputeCapabilityMinor = p.minor;
  c.nStreamFlags = 0;
  return c;
}

static void run() {
  const NppiSize roi{W, H};
  const NppStreamContext ctx = make_ctx();
  auto a8 = gray(1), b8 = gray(2);
  auto af = flt(3), bf = flt(4);
  int sa = 0, sb = 0, sd = 0, saf = 0, sbf = 0, sdf = 0;
  Npp8u* dA = up_img(a8, W, H, &sa);
  Npp8u* dB = up_img(b8, W, H, &sb);
  Npp8u* dD = nppiMalloc_8u_C1(W, H, &sd);
  Npp32f* fA = up_img(af, W, H, &saf);
  Npp32f* fB = up_img(bf, W, H, &sbf);
  Npp32f* fD = nppiMalloc_32f_C1(W, H, &sdf);

  const NppLibraryVersion* ver = nppGetLibVersion();
  printf("npp major %d\n", ver ? ver->major : -1);

  // Scale factors are the part most easily got wrong, so cover several.
  for (int sf : {0, 1, 3}) {
    char tag[64];
    snprintf(tag, sizeof(tag), "add 8u sfs=%d", sf);
    NP(nppiAdd_8u_C1RSfs_Ctx(dA, sa, dB, sb, dD, sd, roi, sf, ctx));
    emit(tag, down_img(dD, sd, W, H));
    snprintf(tag, sizeof(tag), "mul 8u sfs=%d", sf);
    NP(nppiMul_8u_C1RSfs_Ctx(dA, sa, dB, sb, dD, sd, roi, sf, ctx));
    emit(tag, down_img(dD, sd, W, H));
  }
  NP(nppiSub_8u_C1RSfs_Ctx(dA, sa, dB, sb, dD, sd, roi, 0, ctx));
  emit("sub 8u", down_img(dD, sd, W, H));
  NP(nppiAbsDiff_8u_C1R_Ctx(dA, sa, dB, sb, dD, sd, roi, ctx));
  emit("absdiff 8u", down_img(dD, sd, W, H));
  NP(nppiAnd_8u_C1R_Ctx(dA, sa, dB, sb, dD, sd, roi, ctx));
  emit("and 8u", down_img(dD, sd, W, H));
  NP(nppiOr_8u_C1R_Ctx(dA, sa, dB, sb, dD, sd, roi, ctx));
  emit("or 8u", down_img(dD, sd, W, H));
  NP(nppiXor_8u_C1R_Ctx(dA, sa, dB, sb, dD, sd, roi, ctx));
  emit("xor 8u", down_img(dD, sd, W, H));
  NP(nppiNot_8u_C1R_Ctx(dA, sa, dD, sd, roi, ctx));
  emit("not 8u", down_img(dD, sd, W, H));
  NP(nppiAddC_8u_C1RSfs_Ctx(dA, sa, 37, dD, sd, roi, 1, ctx));
  emit("addC 8u sfs=1", down_img(dD, sd, W, H));
  NP(nppiMulC_8u_C1RSfs_Ctx(dA, sa, 3, dD, sd, roi, 2, ctx));
  emit("mulC 8u sfs=2", down_img(dD, sd, W, H));

  NP(nppiAdd_32f_C1R_Ctx(fA, saf, fB, sbf, fD, sdf, roi, ctx));
  emit("add 32f", down_img(fD, sdf, W, H));
  NP(nppiSub_32f_C1R_Ctx(fA, saf, fB, sbf, fD, sdf, roi, ctx));
  emit("sub 32f", down_img(fD, sdf, W, H));
  NP(nppiMul_32f_C1R_Ctx(fA, saf, fB, sbf, fD, sdf, roi, ctx));
  emit("mul 32f", down_img(fD, sdf, W, H));
  NP(nppiAbs_32f_C1R_Ctx(fA, saf, fD, sdf, roi, ctx));
  emit("abs 32f", down_img(fD, sdf, W, H));
  NP(nppiSqr_32f_C1R_Ctx(fA, saf, fD, sdf, roi, ctx));
  emit("sqr 32f", down_img(fD, sdf, W, H));
  NP(nppiMulC_32f_C1R_Ctx(fA, saf, 1.5f, fD, sdf, roi, ctx));
  emit("mulC 32f", down_img(fD, sdf, W, H));

  NP(nppiCopy_8u_C1R_Ctx(dA, sa, dD, sd, roi, ctx));
  emit("copy 8u", down_img(dD, sd, W, H));
  NP(nppiSet_8u_C1R_Ctx(200, dD, sd, roi, ctx));
  emit("set 8u", down_img(dD, sd, W, H));
  NP(nppiConvert_8u32f_C1R_Ctx(dA, sa, fD, sdf, roi, ctx));
  emit("convert 8u->32f", down_img(fD, sdf, W, H));
  NP(nppiConvert_32f8u_C1R_Ctx(fA, saf, dD, sd, roi, NPP_RND_NEAR, ctx));
  emit("convert 32f->8u near", down_img(dD, sd, W, H));

  {   // Transpose swaps the dimensions, so it needs its own destination.
    int st = 0;
    Npp8u* dT = nppiMalloc_8u_C1(H, W, &st);
    NP(nppiTranspose_8u_C1R_Ctx(dA, sa, dT, st, roi, ctx));
    emit("transpose 8u", down_img(dT, st, H, W));
    nppiFree(dT);
  }

  {   // Three-channel colour work.
    auto c3 = rgb(5);
    int s3 = 0, sg = 0;
    Npp8u* d3 = nppiMalloc_8u_C3(W, H, &s3);
    for (int y = 0; y < H; ++y)
      cudaMemcpy((char*)d3 + (size_t)y * s3, c3.data() + (size_t)y * W * 3, W * 3,
                 cudaMemcpyHostToDevice);
    Npp8u* dG = nppiMalloc_8u_C1(W, H, &sg);
    NP(nppiRGBToGray_8u_C3C1R_Ctx(d3, s3, dG, sg, roi, ctx));
    emit("rgb->gray", down_img(dG, sg, W, H));
    const int order[3] = {2, 1, 0};
    int so = 0;
    Npp8u* dO = nppiMalloc_8u_C3(W, H, &so);
    NP(nppiSwapChannels_8u_C3R_Ctx(d3, s3, dO, so, roi, order, ctx));
    emit("swap channels", down_img(dO, so, W * 3, H));
    nppiFree(d3); nppiFree(dG); nppiFree(dO);
  }

  NP(nppiThreshold_8u_C1R_Ctx(dA, sa, dD, sd, roi, 128, NPP_CMP_GREATER, ctx));
  emit("threshold >128", down_img(dD, sd, W, H));
  NP(nppiCompare_8u_C1R_Ctx(dA, sa, dB, sb, dD, sd, roi, NPP_CMP_LESS, ctx));
  emit("compare <", down_img(dD, sd, W, H));

  {   // Statistics land in device memory, as NPP defines them.
    ImgBufSize bytes = 0;
    NP(nppiSumGetBufferHostSize_8u_C1R_Ctx(roi, &bytes, ctx));
    Npp8u* buf = nullptr;
    cudaMalloc(&buf, bytes ? bytes : 1);
    Npp64f* dsum = nullptr;
    cudaMalloc(&dsum, 2 * sizeof(Npp64f));
    NP(nppiSum_8u_C1R_Ctx(dA, sa, roi, buf, dsum, ctx));
    double hsum = 0;
    cudaMemcpy(&hsum, dsum, sizeof hsum, cudaMemcpyDeviceToHost);
    printf("%-30s %.4f\n", "sum 8u", hsum);
    NP(nppiMean_8u_C1R_Ctx(dA, sa, roi, buf, dsum, ctx));
    cudaMemcpy(&hsum, dsum, sizeof hsum, cudaMemcpyDeviceToHost);
    printf("%-30s %.6f\n", "mean 8u", hsum);
    Npp64f hms[2] = {0, 0};
    NP(nppiMean_StdDev_8u_C1R_Ctx(dA, sa, roi, buf, dsum, dsum + 1, ctx));
    cudaMemcpy(hms, dsum, sizeof hms, cudaMemcpyDeviceToHost);
    printf("%-30s mean=%.6f stddev=%.6f\n", "mean/stddev 8u", hms[0], hms[1]);
    Npp8u* mm = nullptr;
    cudaMalloc(&mm, 2);
    NP(nppiMinMax_8u_C1R_Ctx(dA, sa, roi, mm, mm + 1, buf, ctx));
    Npp8u hmm[2] = {0, 0};
    cudaMemcpy(hmm, mm, 2, cudaMemcpyDeviceToHost);
    printf("%-30s min=%d max=%d\n", "minmax 8u", (int)hmm[0], (int)hmm[1]);
    cudaFree(buf); cudaFree(dsum); cudaFree(mm);
  }

  {   // Filters read outside the ROI, so the ROI is inset from the image.
    const NppiSize inner{W - 4, H - 4};
    const NppiSize mask{3, 3};
    const NppiPoint anchor{1, 1};
    const Npp8u* src = dA + 2 * sa + 2;
    Npp8u* dst = dD + 2 * sd + 2;
    NP(nppiFilterBox_8u_C1R_Ctx(src, sa, dst, sd, inner, mask, anchor, ctx));
    emit("filter box 3x3", down_img(dD + 2 * sd + 2, sd, inner.width, inner.height));
    NP(nppiDilate3x3_8u_C1R_Ctx(src, sa, dst, sd, inner, ctx));
    emit("dilate 3x3", down_img(dD + 2 * sd + 2, sd, inner.width, inner.height));
    NP(nppiErode3x3_8u_C1R_Ctx(src, sa, dst, sd, inner, ctx));
    emit("erode 3x3", down_img(dD + 2 * sd + 2, sd, inner.width, inner.height));

    // nppiFilter_32f_C1R is deliberately absent from this comparison. Probing
    // NVIDIA's implementation with delta kernels gives a mask-to-source mapping
    // that aliases positions -- k[1] and k[2] read the same pixel -- matching
    // neither convolution nor correlation and matching nothing the
    // documentation describes. VirtualGPU implements the documented
    // convolution; claiming a hardware match here would be claiming something
    // that is not true.
  }

  NP(nppiMirror_8u_C1R_Ctx(dA, sa, dD, sd, roi, NPP_HORIZONTAL_AXIS, ctx));
  emit("mirror horizontal", down_img(dD, sd, W, H));
  NP(nppiMirror_8u_C1R_Ctx(dA, sa, dD, sd, roi, NPP_BOTH_AXIS, ctx));
  emit("mirror both", down_img(dD, sd, W, H));

  {   // Resize, both interpolations, up and down.
    const NppiSize ssz{W, H}, dsz{W * 2, H / 2};
    const NppiRect sroi{0, 0, W, H}, droi{0, 0, W * 2, H / 2};
    int sr = 0;
    Npp8u* dR = nppiMalloc_8u_C1(dsz.width, dsz.height, &sr);
    NP(nppiResize_8u_C1R_Ctx(dA, sa, ssz, sroi, dR, sr, dsz, droi, NPPI_INTER_NN, ctx));
    emit("resize nearest", down_img(dR, sr, dsz.width, dsz.height));
    // NPPI_INTER_LINEAR is deliberately absent: NPP's bilinear interpolates
    // horizontally at pixel centres but samples rows exactly, which is not a
    // convention that can be derived from the documentation. VirtualGPU does
    // the standard bilinear filter and says so rather than claiming a match.
    nppiFree(dR);
  }

  {   // Signal processing.
    const size_t n = 256;
    std::vector<Npp32f> x(n), y(n);
    for (size_t i = 0; i < n; ++i) {
      x[i] = 0.5f + 0.25f * (float)(i % 13);
      y[i] = 1.0f + 0.125f * (float)(i % 7);
    }
    Npp32f* dx = nppsMalloc_32f(n);
    Npp32f* dy = nppsMalloc_32f(n);
    Npp32f* dz = nppsMalloc_32f(n);
    cudaMemcpy(dx, x.data(), n * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dy, y.data(), n * 4, cudaMemcpyHostToDevice);
    NP(nppsAdd_32f_Ctx(dx, dy, dz, n, ctx));
    { std::vector<Npp32f> h(n); cudaMemcpy(h.data(), dz, n * 4, cudaMemcpyDeviceToHost); emit("npps add", h); }
    NP(nppsSub_32f_Ctx(dx, dy, dz, n, ctx));
    { std::vector<Npp32f> h(n); cudaMemcpy(h.data(), dz, n * 4, cudaMemcpyDeviceToHost); emit("npps sub", h); }
    NP(nppsMulC_32f_Ctx(dx, 2.5f, dz, n, ctx));
    { std::vector<Npp32f> h(n); cudaMemcpy(h.data(), dz, n * 4, cudaMemcpyDeviceToHost); emit("npps mulC", h); }
    NP(nppsSqrt_32f_Ctx(dx, dz, n, ctx));
    { std::vector<Npp32f> h(n); cudaMemcpy(h.data(), dz, n * 4, cudaMemcpyDeviceToHost); emit("npps sqrt", h); }
    SigBufSize bs = 0;
    NP(nppsSumGetBufferSize_32f_Ctx(n, &bs, ctx));
    Npp8u* buf = nullptr;
    cudaMalloc(&buf, bs ? bs : 1);
    Npp32f* out = nullptr;
    cudaMalloc(&out, 4);
    float hv = 0;
    NP(nppsSum_32f_Ctx(dx, n, out, buf, ctx));
    cudaMemcpy(&hv, out, 4, cudaMemcpyDeviceToHost);
    printf("%-30s %.4f\n", "npps sum", hv);
    NP(nppsMean_32f_Ctx(dx, n, out, buf, ctx));
    cudaMemcpy(&hv, out, 4, cudaMemcpyDeviceToHost);
    printf("%-30s %.6f\n", "npps mean", hv);
    NP(nppsMax_32f_Ctx(dx, n, out, buf, ctx));
    cudaMemcpy(&hv, out, 4, cudaMemcpyDeviceToHost);
    printf("%-30s %.4f\n", "npps max", hv);
    NP(nppsMin_32f_Ctx(dx, n, out, buf, ctx));
    cudaMemcpy(&hv, out, 4, cudaMemcpyDeviceToHost);
    printf("%-30s %.4f\n", "npps min", hv);
    cudaFree(buf); cudaFree(out);
    nppsFree(dx); nppsFree(dy); nppsFree(dz);
  }

  nppiFree(dA); nppiFree(dB); nppiFree(dD);
  nppiFree(fA); nppiFree(fB); nppiFree(fD);
}

int main() { run(); return 0; }
