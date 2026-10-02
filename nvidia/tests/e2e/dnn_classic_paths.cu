// The rest of cuDNN's classic API, each path checked against a host
// reference, finite differences or a measured permutation -- so the same
// program passes on a real GPU with NVIDIA's libcudnn.so.9 (it does, on an
// RTX 3060 with cuDNN 9.27) and on VirtualGPU, where hosted CI runs it:
//
//   - Vectorized layouts: NCHW_VECT_C descriptors (strides, sizes, what is
//     refused), INT8x4, UINT8x4 and INT8x32 convolution against an integer
//     reference, INT8x32 filters reordered by cudnnReorderFilterAndBias and
//     run under CUDNN_NO_REORDER (filter and bias), the fused
//     bias-activation, the algorithm lists, and cudnnTransformTensor between
//     INT8/FLOAT and INT8x4.
//   - INT8 pooling and activation, rounding to nearest even.
//   - FP8, BOOLEAN and INT64 classic descriptors are refused.
//   - Divisive normalization against its formula, and its backward pass
//     against finite differences, in double.
//   - Tensor transform descriptors: padding, folding and unfolding of
//     tensors and filters, cudnnInitTransformDest.
#include <cudnn.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <vector>

static int fails = 0;
static void expect(const char* what, bool ok, double detail = 0) {
  std::printf("%s %s", ok ? "ok  " : "FAIL", what);
  if (!ok && detail != 0) std::printf(" (%g)", detail);
  std::printf("\n");
  fails += !ok;
}
#define CK(x) do { cudnnStatus_t s_ = (x); if (s_ != CUDNN_STATUS_SUCCESS) { \
  std::printf("FAIL %s -> %d\n", #x, (int)s_); ++fails; return; } } while (0)

static cudnnHandle_t H;

static std::vector<std::function<void()>> g_owned;
template <class D, class F>
static void own(D d, F destroy) {
  g_owned.push_back([d, destroy] { destroy(d); });
}
static void destroy_owned() {
  while (!g_owned.empty()) g_owned.back()(), g_owned.pop_back();
}
static cudnnTensorDescriptor_t tensor() {
  cudnnTensorDescriptor_t d;
  (cudnnCreateTensorDescriptor(&d), own(d, cudnnDestroyTensorDescriptor));
  return d;
}
static cudnnFilterDescriptor_t filter() {
  cudnnFilterDescriptor_t d;
  (cudnnCreateFilterDescriptor(&d), own(d, cudnnDestroyFilterDescriptor));
  return d;
}

template <class T>
struct Buf {
  T* p = nullptr;
  size_t n = 0;
  explicit Buf(size_t n_) : n(n_) {
    cudaMalloc(&p, n * sizeof(T) + 64);
    cudaMemset(p, 0, n * sizeof(T) + 64);
  }
  Buf(const std::vector<T>& h) : Buf(h.size()) { put(h); }
  void put(const std::vector<T>& h) { cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice); }
  std::vector<T> get() const {
    std::vector<T> h(n);
    cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost);
    return h;
  }
  ~Buf() { cudaFree(p); }
};

// Round to nearest even and saturate to int8, as the hardware's epilogues do
// (from a float).
static int sat8(double v) {
  const double r = std::nearbyint((float)v);
  return (int)std::fmax(-128, std::fmin(127, r));
}

// Logical (n, c, s) of an [N, C, S] tensor in NCHW_VECT_C with v lanes.
static size_t voff(int v, int C, int S, int n, int c, int s) {
  return (((size_t)n * (C / v) + c / v) * S + s) * v + c % v;
}

/* ---- vectorized descriptors ---- */

static void vect_descriptors() {
  cudnnTensorDescriptor_t d = tensor();
  CK(cudnnSetTensor4dDescriptor(d, CUDNN_TENSOR_NCHW_VECT_C, CUDNN_DATA_INT8x4, 2, 8, 3, 5));
  cudnnDataType_t t;
  int n, c, h, w, sn, sc, sh, sw;
  size_t bytes = 0;
  CK(cudnnGetTensor4dDescriptor(d, &t, &n, &c, &h, &w, &sn, &sc, &sh, &sw));
  CK(cudnnGetTensorSizeInBytes(d, &bytes));
  expect("INT8x4 NCHW_VECT_C: dims logical, strides in vectors, 1 byte per element",
         t == CUDNN_DATA_INT8x4 && c == 8 && sn == 30 && sc == 15 && sh == 5 && sw == 1 && bytes == 240);
  CK(cudnnSetTensor4dDescriptor(d, CUDNN_TENSOR_NCHW_VECT_C, CUDNN_DATA_INT8x32, 2, 64, 3, 5));
  CK(cudnnGetTensorSizeInBytes(d, &bytes));
  expect("INT8x32 NCHW_VECT_C 2x64x3x5 is 1920 bytes", bytes == 1920);
  CK(cudnnSetTensor4dDescriptor(d, CUDNN_TENSOR_NCHW, CUDNN_DATA_INT8x4, 2, 8, 3, 5));
  CK(cudnnGetTensorSizeInBytes(d, &bytes));
  expect("INT8x4 outside NCHW_VECT_C is one vector per element", bytes == 960);
  expect("NCHW_VECT_C refuses channels not a multiple of the vector",
         cudnnSetTensor4dDescriptor(d, CUDNN_TENSOR_NCHW_VECT_C, CUDNN_DATA_INT8x4, 2, 6, 3, 5) == CUDNN_STATUS_BAD_PARAM);
  expect("NCHW_VECT_C refuses a scalar type",
         cudnnSetTensor4dDescriptor(d, CUDNN_TENSOR_NCHW_VECT_C, CUDNN_DATA_INT8, 2, 8, 3, 5) == CUDNN_STATUS_BAD_PARAM);
  const int dims[4] = {2, 8, 3, 5}, strides[4] = {120, 15, 5, 1};
  bool refused = true;
  for (cudnnDataType_t bad : {CUDNN_DATA_FP8_E4M3, CUDNN_DATA_FP8_E5M2, CUDNN_DATA_BOOLEAN, CUDNN_DATA_INT64}) {
    refused &= cudnnSetTensor4dDescriptor(d, CUDNN_TENSOR_NCHW, bad, 2, 8, 3, 5) == CUDNN_STATUS_BAD_PARAM;
    refused &= cudnnSetTensorNdDescriptor(d, bad, 4, dims, strides) == CUDNN_STATUS_BAD_PARAM;
  }
  expect("FP8, BOOLEAN and INT64 classic tensors are BAD_PARAM", refused);
  cudnnFilterDescriptor_t f = filter();
  CK(cudnnSetFilter4dDescriptor(f, CUDNN_DATA_INT8x32, CUDNN_TENSOR_NCHW_VECT_C, 32, 32, 3, 3));
  CK(cudnnGetFilterSizeInBytes(f, &bytes));
  expect("an INT8x32 32x32x3x3 filter is 9216 bytes", bytes == 9216);
  expect("an FP8 filter is BAD_PARAM",
         cudnnSetFilter4dDescriptor(f, CUDNN_DATA_FP8_E4M3, CUDNN_TENSOR_NCHW, 8, 8, 3, 3) == CUDNN_STATUS_BAD_PARAM);
  expect("an INT8 NCHW_VECT_C filter is BAD_PARAM",
         cudnnSetFilter4dDescriptor(f, CUDNN_DATA_INT8, CUDNN_TENSOR_NCHW_VECT_C, 8, 8, 3, 3) == CUDNN_STATUS_BAD_PARAM);
}

/* ---- vectorized convolution ---- */

// The permutation cudnnReorderFilterAndBias applies to an INT8x32 filter
// [K][L bytes] and its float bias, as measured on an RTX 3060.
static std::vector<int8_t> reorder_filter(const std::vector<int8_t>& w, int K) {
  const int64_t L = (int64_t)w.size() / K, G = (K + 7) / 8;
  std::vector<int8_t> out(w.size(), 0);
  for (size_t d = 0; d < w.size(); ++d) {
    const int64_t l = d % 32, t = d / 32, j = t % 8, g = (t / 8) % G, c = t / 8 / G;
    const int64_t row = g * 8 + j / 4 + 2 * ((l % 16) / 4), col = c * 32 + (j % 4) * 8 + (l / 16) * 4 + l % 4;
    if (row < K && col < L) out[d] = w[row * L + col];
  }
  return out;
}
static std::vector<float> reorder_bias(const std::vector<float>& b) {
  const int K = (int)b.size();
  std::vector<float> out(K, 0.0f);
  for (int j = 0; j < K; ++j) {
    const int g = (j % 32) / 4, src = j / 32 * 32 + (g % 4) * 8 + (g / 4) * 4 + j % 4;
    if (src < K) out[j] = b[src];
  }
  return out;
}

struct VConv {
  const char* name;
  cudnnDataType_t xt, wt, yt;
  cudnnTensorFormat_t yf;
  int C, K, stride;
  bool no_reorder, fused;
  float alpha, beta;
};

static void vect_conv(const VConv& c) {
  const int N = 2, Hh = 6, W = 5, R = 3, pad = 1;
  const int P = (Hh + 2 * pad - R) / c.stride + 1, Q = (W + 2 * pad - R) / c.stride + 1;
  const int vx = c.xt == CUDNN_DATA_INT8x32 ? 32 : 4;
  const int vy = c.yt == CUDNN_DATA_INT8x32 ? 32 : c.yt == CUDNN_DATA_INT8x4 ? 4 : 0;
  cudnnTensorDescriptor_t xd = tensor(), yd = tensor(), bd = tensor();
  cudnnFilterDescriptor_t wd = filter();
  cudnnConvolutionDescriptor_t cd;
  (cudnnCreateConvolutionDescriptor(&cd), own(cd, cudnnDestroyConvolutionDescriptor));
  CK(cudnnSetTensor4dDescriptor(xd, CUDNN_TENSOR_NCHW_VECT_C, c.xt, N, c.C, Hh, W));
  CK(cudnnSetFilter4dDescriptor(wd, c.wt, CUDNN_TENSOR_NCHW_VECT_C, c.K, c.C, R, R));
  CK(cudnnSetTensor4dDescriptor(yd, c.yf, c.yt, N, c.K, P, Q));
  CK(cudnnSetTensor4dDescriptor(bd, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, c.K, 1, 1));
  CK(cudnnSetConvolution2dDescriptor(cd, pad, pad, c.stride, c.stride, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_INT32));
  CK(cudnnSetConvolutionReorderType(cd, c.no_reorder ? CUDNN_NO_REORDER : CUDNN_DEFAULT_REORDER));
  // Values, logical NCHW, and their bytes in the vectorized layout.
  const bool ux = c.xt == CUDNN_DATA_UINT8x4;
  std::vector<int> lx(N * c.C * Hh * W), lw(c.K * c.C * R * R);
  std::vector<int8_t> bx(lx.size()), bw(lw.size());
  for (int n = 0; n < N; ++n)
    for (int ch = 0; ch < c.C; ++ch)
      for (int s = 0; s < Hh * W; ++s) {
        int v = (n * 7 + ch * 13 + s * 5) % 23 - 11;
        if (ux) v += 11 + (ch % 3) * 60;  // up to 142: past INT8
        lx[(n * c.C + ch) * Hh * W + s] = v;
        bx[voff(vx, c.C, Hh * W, n, ch, s)] = (int8_t)(uint8_t)v;
      }
  for (int k = 0; k < c.K; ++k)
    for (int ch = 0; ch < c.C; ++ch)
      for (int s = 0; s < R * R; ++s) {
        const int v = (k * 11 + ch * 3 + s * 7) % 9 - 4;
        lw[(k * c.C + ch) * R * R + s] = v;
        bw[voff(vx, c.C, R * R, k, ch, s)] = (int8_t)v;
      }
  std::vector<float> bias(c.K);
  for (int k = 0; k < c.K; ++k) bias[k] = 3.25f * (k % 5) - 6.0f;
  if (c.no_reorder) {
    // The filter and bias as cudnnReorderFilterAndBias leaves them.
    Buf<int8_t> w0(bw), w1(bw.size());
    Buf<float> b0(bias), b1(bias.size());
    CK(cudnnReorderFilterAndBias(H, wd, CUDNN_DEFAULT_REORDER, w0.p, w1.p, 1, b0.p, b1.p));
    const auto rw = w1.get();
    const auto rb = b1.get();
    char what[160];
    std::snprintf(what, sizeof what, "%s: cudnnReorderFilterAndBias permutes filter and bias as measured", c.name);
    expect(what, rw == reorder_filter(bw, c.K) && rb == reorder_bias(bias));
    bw = rw, bias = rb;
  }
  // The prior y (and z), logical.
  const size_t yn = (size_t)N * c.K * P * Q;
  std::vector<double> lz(yn);
  for (size_t i = 0; i < yn; ++i) lz[i] = (double)((int)(i * 5 % 31) - 15);
  const bool yflt = c.yt == CUDNN_DATA_FLOAT;
  std::vector<int8_t> by(yflt ? 0 : yn);
  std::vector<float> fy(yflt ? yn : 0);
  auto yat = [&](int n, int k, int s) -> size_t {
    if (vy) return voff(vy, c.K, P * Q, n, k, s);
    if (c.yf == CUDNN_TENSOR_NHWC) return ((size_t)n * P * Q + s) * c.K + k;
    return ((size_t)n * c.K + k) * P * Q + s;
  };
  for (int n = 0; n < N; ++n)
    for (int k = 0; k < c.K; ++k)
      for (int s = 0; s < P * Q; ++s) {
        const double v = lz[((size_t)n * c.K + k) * P * Q + s];
        if (yflt) fy[yat(n, k, s)] = (float)v;
        else by[yat(n, k, s)] = (int8_t)v;
      }
  Buf<int8_t> x(bx), w(bw), y8(by.empty() ? std::vector<int8_t>(1) : by), z8(by.empty() ? std::vector<int8_t>(1) : by);
  Buf<float> yf(fy.empty() ? std::vector<float>(1) : fy), zf(fy.empty() ? std::vector<float>(1) : fy), b(bias);
  void* yp = yflt ? (void*)yf.p : (void*)y8.p;
  void* zp = yflt ? (void*)zf.p : (void*)z8.p;
  // The algorithm list: the two implicit GEMMs, nothing else.
  int got = 0, runs = 0;
  cudnnConvolutionFwdAlgoPerf_t perf[10];
  CK(cudnnGetConvolutionForwardAlgorithm_v7(H, xd, wd, cd, yd, 10, &got, perf));
  bool listed_first = true;
  for (int i = 0; i < got; ++i) {
    const bool ok = perf[i].status == CUDNN_STATUS_SUCCESS;
    runs += ok;
    if (ok && perf[i].algo != CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM &&
        perf[i].algo != CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM)
      listed_first = false;
  }
  const int want_runs = (c.yf == CUDNN_TENSOR_NHWC) ? 1 : 2;
  char what[200];
  std::snprintf(what, sizeof what, "%s: %d implicit-GEMM algorithm(s) run it", c.name, want_runs);
  expect(what, listed_first && runs == want_runs, runs);
  const auto algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
  size_t ws = 0;
  CK(cudnnGetConvolutionForwardWorkspaceSize(H, xd, wd, cd, yd, algo, &ws));
  Buf<char> work(ws + 1);
  cudnnActivationDescriptor_t ad;
  (cudnnCreateActivationDescriptor(&ad), own(ad, cudnnDestroyActivationDescriptor));
  CK(cudnnSetActivationDescriptor(ad, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0.0));
  if (c.fused)
    CK(cudnnConvolutionBiasActivationForward(H, &c.alpha, xd, x.p, wd, w.p, cd, algo, work.p, ws, &c.beta, yd, zp, bd,
                                             b.p, ad, yd, yp));
  else
    CK(cudnnConvolutionForward(H, &c.alpha, xd, x.p, wd, w.p, cd, algo, work.p, ws, &c.beta, yd, yp));
  // The logical bias, whatever order it was handed over in.
  std::vector<float> lb(c.K);
  for (int k = 0; k < c.K; ++k) lb[k] = 3.25f * (k % 5) - 6.0f;
  const auto oy8 = y8.get();
  const auto oyf = yf.get();
  double worst = 0;
  for (int n = 0; n < N; ++n)
    for (int k = 0; k < c.K; ++k)
      for (int p = 0; p < P; ++p)
        for (int q = 0; q < Q; ++q) {
          long acc = 0;
          for (int ch = 0; ch < c.C; ++ch)
            for (int r = 0; r < R; ++r)
              for (int t = 0; t < R; ++t) {
                const int ih = p * c.stride - pad + r, iw = q * c.stride - pad + t;
                if (ih < 0 || iw < 0 || ih >= Hh || iw >= W) continue;
                acc += (long)lx[((n * c.C + ch) * Hh + ih) * W + iw] * lw[((k * c.C + ch) * R + r) * R + t];
              }
          const size_t li = ((size_t)n * c.K + k) * P * Q + p * Q + q;
          double want = (float)(c.alpha * (float)acc) + c.beta * lz[li];
          if (c.fused) want = std::fmax(0.0, (float)((float)want + lb[k]));
          const size_t o = yat(n, k, p * Q + q);
          const double gotv = yflt ? (double)oyf[o] : (double)oy8[o];
          if (!yflt) want = sat8(want);
          worst = std::fmax(worst, std::fabs(gotv - want) / (1 + std::fabs(want)));
        }
  std::snprintf(what, sizeof what, "%s matches the integer reference", c.name);
  expect(what, worst < 1e-5, worst);
}

static void vect_refusals() {
  cudnnTensorDescriptor_t xd = tensor(), yd = tensor();
  cudnnFilterDescriptor_t wd = filter();
  cudnnConvolutionDescriptor_t cd;
  (cudnnCreateConvolutionDescriptor(&cd), own(cd, cudnnDestroyConvolutionDescriptor));
  CK(cudnnSetTensor4dDescriptor(xd, CUDNN_TENSOR_NCHW_VECT_C, CUDNN_DATA_INT8x4, 1, 8, 4, 4));
  CK(cudnnSetFilter4dDescriptor(wd, CUDNN_DATA_INT8x4, CUDNN_TENSOR_NCHW_VECT_C, 8, 8, 1, 1));
  CK(cudnnSetConvolution2dDescriptor(cd, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_INT32));
  Buf<int8_t> a(4096), b(4096);
  Buf<char> work(1 << 20);
  const float one = 1, zero = 0;
  const auto algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
  auto fwd = [&]() {
    return cudnnConvolutionForward(H, &one, xd, a.p, wd, a.p, cd, algo, work.p, 1 << 20, &zero, yd, b.p);
  };
  cudnnSetTensor4dDescriptor(yd, CUDNN_TENSOR_NCHW, CUDNN_DATA_INT8, 1, 8, 4, 4);
  expect("INT8x4 to an INT8 output is NOT_SUPPORTED", fwd() == CUDNN_STATUS_NOT_SUPPORTED);
  cudnnSetTensor4dDescriptor(yd, CUDNN_TENSOR_NCHW_VECT_C, CUDNN_DATA_INT8x4, 1, 8, 4, 4);
  cudnnSetConvolution2dDescriptor(cd, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
  expect("INT8x4 with FLOAT compute is NOT_SUPPORTED", fwd() == CUDNN_STATUS_NOT_SUPPORTED);
  cudnnSetConvolution2dDescriptor(cd, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_INT32);
  expect("INT8x4 backward-data is NOT_SUPPORTED",
         cudnnConvolutionBackwardData(H, &one, wd, a.p, yd, b.p, cd, CUDNN_CONVOLUTION_BWD_DATA_ALGO_1, work.p, 1 << 20,
                                      &zero, xd, a.p) == CUDNN_STATUS_NOT_SUPPORTED);
  cudnnSetConvolution2dDescriptor(cd, 0, 0, 1, 1, 2, 2, CUDNN_CROSS_CORRELATION, CUDNN_DATA_INT32);
  cudnnSetFilter4dDescriptor(wd, CUDNN_DATA_INT8x4, CUDNN_TENSOR_NCHW_VECT_C, 8, 8, 2, 2);
  cudnnSetTensor4dDescriptor(yd, CUDNN_TENSOR_NCHW_VECT_C, CUDNN_DATA_INT8x4, 1, 8, 2, 2);
  expect("INT8x4 with dilation is NOT_SUPPORTED", fwd() == CUDNN_STATUS_NOT_SUPPORTED);
  cudnnSetConvolution2dDescriptor(cd, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_INT32);
  cudnnSetFilter4dDescriptor(wd, CUDNN_DATA_INT8x4, CUDNN_TENSOR_NCHW, 8, 8, 1, 1);
  cudnnSetTensor4dDescriptor(xd, CUDNN_TENSOR_NCHW, CUDNN_DATA_INT8x4, 1, 8, 4, 4);
  cudnnSetTensor4dDescriptor(yd, CUDNN_TENSOR_NCHW, CUDNN_DATA_INT8x4, 1, 8, 4, 4);
  expect("INT8x4 outside NCHW_VECT_C is BAD_PARAM", fwd() == CUDNN_STATUS_BAD_PARAM);
  // UINT8 NHWC input: INT8 out runs, UINT8 out does not.
  cudnnSetTensor4dDescriptor(xd, CUDNN_TENSOR_NHWC, CUDNN_DATA_UINT8, 1, 8, 4, 4);
  cudnnSetFilter4dDescriptor(wd, CUDNN_DATA_INT8, CUDNN_TENSOR_NHWC, 8, 8, 1, 1);
  cudnnSetTensor4dDescriptor(yd, CUDNN_TENSOR_NHWC, CUDNN_DATA_UINT8, 1, 8, 4, 4);
  expect("UINT8 to UINT8 is NOT_SUPPORTED", fwd() == CUDNN_STATUS_NOT_SUPPORTED);
  cudnnSetTensor4dDescriptor(yd, CUDNN_TENSOR_NHWC, CUDNN_DATA_INT8, 1, 8, 4, 4);
  expect("UINT8 to INT8 (UINT8_CONFIG) runs", fwd() == CUDNN_STATUS_SUCCESS);
  // The reorder takes INT8x32 NCHW_VECT_C filters only.
  cudnnSetFilter4dDescriptor(wd, CUDNN_DATA_INT8x4, CUDNN_TENSOR_NCHW_VECT_C, 8, 8, 1, 1);
  expect("cudnnReorderFilterAndBias refuses INT8x4",
         cudnnReorderFilterAndBias(H, wd, CUDNN_DEFAULT_REORDER, a.p, b.p, 0, nullptr, nullptr) ==
             CUDNN_STATUS_NOT_SUPPORTED);
  cudnnSetFilter4dDescriptor(wd, CUDNN_DATA_INT8x32, CUDNN_TENSOR_NCHW_VECT_C, 32, 32, 1, 1);
  expect("cudnnReorderFilterAndBias wants a bias pointer when asked to reorder one",
         cudnnReorderFilterAndBias(H, wd, CUDNN_DEFAULT_REORDER, a.p, b.p, 1, nullptr, nullptr) ==
             CUDNN_STATUS_BAD_PARAM);
  std::vector<int8_t> pattern(1024);
  for (int i = 0; i < 1024; ++i) pattern[i] = (int8_t)(i * 37);
  a.put(pattern);
  CK(cudnnReorderFilterAndBias(H, wd, CUDNN_NO_REORDER, a.p, b.p, 0, nullptr, nullptr));
  const auto copied = b.get();
  expect("CUDNN_NO_REORDER copies the filter", std::equal(pattern.begin(), pattern.end(), copied.begin()));
}

/* ---- transforms to and from the vectorized layout ---- */

static void vect_transforms() {
  const int N = 2, C = 8, S = 15;
  cudnnTensorDescriptor_t s8 = tensor(), sv = tensor(), sf = tensor();
  CK(cudnnSetTensor4dDescriptor(s8, CUDNN_TENSOR_NCHW, CUDNN_DATA_INT8, N, C, 3, 5));
  CK(cudnnSetTensor4dDescriptor(sv, CUDNN_TENSOR_NCHW_VECT_C, CUDNN_DATA_INT8x4, N, C, 3, 5));
  CK(cudnnSetTensor4dDescriptor(sf, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, 3, 5));
  std::vector<int8_t> h8(N * C * S);
  std::vector<float> hf(N * C * S);
  for (size_t i = 0; i < h8.size(); ++i) h8[i] = (int8_t)((int)(i * 7 % 41) - 20), hf[i] = 11.25f * h8[i] + 0.5f;
  Buf<int8_t> a(h8), v(h8.size()), back(h8.size());
  Buf<float> f(hf);
  const float one = 1, zero = 0, half = 0.5f;
  CK(cudnnTransformTensor(H, &one, s8, a.p, &zero, sv, v.p));
  const auto hv = v.get();
  bool placed = true;
  for (int n = 0; n < N; ++n)
    for (int c = 0; c < C; ++c)
      for (int s = 0; s < S; ++s) placed &= hv[voff(4, C, S, n, c, s)] == h8[(n * C + c) * S + s];
  expect("INT8 NCHW to INT8x4 NCHW_VECT_C places each channel in its lane", placed);
  CK(cudnnTransformTensor(H, &one, sv, v.p, &zero, s8, back.p));
  expect("and back", back.get() == h8);
  // Float to INT8x4 rounds to nearest even and saturates; beta reads the prior.
  CK(cudnnTransformTensor(H, &one, sf, f.p, &half, sv, v.p));
  const auto hv2 = v.get();
  bool rounded = true;
  for (int n = 0; n < N; ++n)
    for (int c = 0; c < C; ++c)
      for (int s = 0; s < S; ++s) {
        const size_t o = voff(4, C, S, n, c, s);
        rounded &= hv2[o] == sat8(hf[(n * C + c) * S + s] + 0.5 * hv[o]);
      }
  expect("FLOAT to INT8x4 with beta: rounded to nearest even, saturated", rounded);
  cudnnTensorDescriptor_t x32 = tensor(), xn = tensor();
  CK(cudnnSetTensor4dDescriptor(x32, CUDNN_TENSOR_NCHW_VECT_C, CUDNN_DATA_INT8x32, 1, 32, 2, 2));
  CK(cudnnSetTensor4dDescriptor(xn, CUDNN_TENSOR_NCHW, CUDNN_DATA_INT8, 1, 32, 2, 2));
  expect("INT8x32 does not transform (BAD_PARAM)",
         cudnnTransformTensor(H, &one, xn, a.p, &zero, x32, v.p) == CUDNN_STATUS_BAD_PARAM);
  cudnnTensorDescriptor_t u8 = tensor();
  CK(cudnnSetTensor4dDescriptor(u8, CUDNN_TENSOR_NCHW, CUDNN_DATA_UINT8, N, C, 3, 5));
  expect("INT8x4 to UINT8 is NOT_SUPPORTED",
         cudnnTransformTensor(H, &one, sv, v.p, &zero, u8, back.p) == CUDNN_STATUS_NOT_SUPPORTED);
}

/* ---- INT8 pooling and activation ---- */

static void int8_layers() {
  const int8_t vals[16] = {1, 2, 2, 2, 1, 1, 1, 2, 1, 0, 1, 0, -1, 0, -1, 0};
  cudnnTensorDescriptor_t xd = tensor(), yd = tensor();
  CK(cudnnSetTensor4dDescriptor(xd, CUDNN_TENSOR_NCHW, CUDNN_DATA_INT8, 1, 1, 4, 4));
  CK(cudnnSetTensor4dDescriptor(yd, CUDNN_TENSOR_NCHW, CUDNN_DATA_INT8, 1, 1, 3, 3));
  Buf<int8_t> x(std::vector<int8_t>(vals, vals + 16)), y(9);
  cudnnPoolingDescriptor_t pd;
  (cudnnCreatePoolingDescriptor(&pd), own(pd, cudnnDestroyPoolingDescriptor));
  const float one = 1, zero = 0, a25 = 2.5f;
  // Average, padding counted, 2x2 windows with padding 1, stride 2: the
  // window sums over 4 are 0.25, 1, 0.5 / 0.5, 0.75, 0.5 / -0.25, -0.25, 0.
  CK(cudnnSetPooling2dDescriptor(pd, CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING, CUDNN_NOT_PROPAGATE_NAN, 2, 2, 1, 1,
                                 2, 2));
  CK(cudnnPoolingForward(H, pd, &one, xd, x.p, &zero, yd, y.p));
  const std::vector<int8_t> want1 = {0, 1, 0, 0, 1, 0, 0, 0, 0};
  expect("INT8 average pooling rounds to nearest even", y.get() == want1);
  CK(cudnnPoolingForward(H, pd, &a25, xd, x.p, &zero, yd, y.p));
  const std::vector<int8_t> want2 = {1, 2, 1, 1, 2, 1, -1, -1, 0};
  expect("... after alpha (2.5 times)", y.get() == want2);
  CK(cudnnSetPooling2dDescriptor(pd, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN, 2, 2, 1, 1, 2, 2));
  CK(cudnnPoolingForward(H, pd, &a25, xd, x.p, &zero, yd, y.p));
  const std::vector<int8_t> want3 = {2, 5, 5, 2, 2, 5, -2, 0, 0};
  expect("INT8 max pooling", y.get() == want3);
  cudnnTensorDescriptor_t fd = tensor();
  CK(cudnnSetTensor4dDescriptor(fd, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 3, 3));
  expect("INT8 pooling into FLOAT is BAD_PARAM", cudnnPoolingForward(H, pd, &one, xd, x.p, &zero, fd, y.p) ==
                                                    CUDNN_STATUS_BAD_PARAM);
  // INT8x4: per lane.
  cudnnTensorDescriptor_t vx = tensor(), vy = tensor();
  CK(cudnnSetTensor4dDescriptor(vx, CUDNN_TENSOR_NCHW_VECT_C, CUDNN_DATA_INT8x4, 1, 4, 4, 4));
  CK(cudnnSetTensor4dDescriptor(vy, CUDNN_TENSOR_NCHW_VECT_C, CUDNN_DATA_INT8x4, 1, 4, 2, 2));
  std::vector<int8_t> hv(64);
  for (int c = 0; c < 4; ++c)
    for (int i = 0; i < 16; ++i) hv[i * 4 + c] = (int8_t)(vals[(i + c) % 16] * (c + 1) * 10 + c);
  Buf<int8_t> v(hv), w(16);
  CK(cudnnSetPooling2dDescriptor(pd, CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING, CUDNN_NOT_PROPAGATE_NAN, 2, 2, 0, 0,
                                 2, 2));
  CK(cudnnPoolingForward(H, pd, &one, vx, v.p, &zero, vy, w.p));
  const auto got = w.get();
  bool lanes = true;
  for (int c = 0; c < 4; ++c)
    for (int o = 0; o < 4; ++o) {
      const int oh = o / 2, ow = o % 2;
      double s = 0;
      for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) s += hv[((oh * 2 + i) * 4 + ow * 2 + j) * 4 + c];
      lanes &= got[o * 4 + c] == sat8(s / 4);
    }
  expect("INT8x4 average pooling, lane by lane", lanes);
  // INT8 activation: ReLU and sigmoid times 2.5, rounded to nearest even.
  cudnnTensorDescriptor_t ad16 = tensor();
  CK(cudnnSetTensor4dDescriptor(ad16, CUDNN_TENSOR_NCHW, CUDNN_DATA_INT8, 1, 1, 4, 4));
  std::vector<int8_t> hx(16);
  for (int i = 0; i < 16; ++i) hx[i] = (int8_t)(i - 8);
  hx[14] = -128, hx[15] = 120;
  Buf<int8_t> ax(hx), ay(16);
  cudnnActivationDescriptor_t act;
  (cudnnCreateActivationDescriptor(&act), own(act, cudnnDestroyActivationDescriptor));
  CK(cudnnSetActivationDescriptor(act, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0.0));
  CK(cudnnActivationForward(H, act, &a25, ad16, ax.p, &zero, ad16, ay.p));
  bool relu = true, sig = true;
  auto oy = ay.get();
  for (int i = 0; i < 16; ++i) relu &= oy[i] == sat8(2.5 * std::fmax(0, hx[i]));
  CK(cudnnSetActivationDescriptor(act, CUDNN_ACTIVATION_SIGMOID, CUDNN_NOT_PROPAGATE_NAN, 0.0));
  CK(cudnnActivationForward(H, act, &a25, ad16, ax.p, &zero, ad16, ay.p));
  oy = ay.get();
  for (int i = 0; i < 16; ++i) sig &= oy[i] == sat8(2.5 / (1 + std::exp(-(double)hx[i])));
  expect("INT8 ReLU and sigmoid, times 2.5, rounded and saturated", relu && sig);
}

/* ---- divisive normalization ---- */

// The formula, per element: x / (K + alpha / n^d * sum over the window
// (x_j - m)^2)^beta, the window clipped at the edges.
static std::vector<double> divnorm_ref(const std::vector<double>& x, const std::vector<double>& m, int NC, int Hh,
                                       int W, unsigned n, double alpha, double beta, double K) {
  const int lb = ((int)n - 1) / 2, la = (int)n - lb - 1;
  std::vector<double> y(x.size());
  for (int p = 0; p < NC; ++p)
    for (int h = 0; h < Hh; ++h)
      for (int w = 0; w < W; ++w) {
        const size_t at = ((size_t)p * Hh + h) * W + w;
        double s = 0;
        for (int i = h - lb; i <= h + la; ++i)
          for (int j = w - lb; j <= w + la; ++j)
            if (i >= 0 && i < Hh && j >= 0 && j < W) {
              const double e = x[((size_t)p * Hh + i) * W + j] - m[at];
              s += e * e;
            }
        y[at] = x[at] / std::pow(K + alpha / (n * n) * s, beta);
      }
  return y;
}

static void divisive_normalization() {
  const int N = 1, C = 2, Hh = 5, W = 6, n = N * C * Hh * W;
  cudnnTensorDescriptor_t xd = tensor(), fd = tensor();
  CK(cudnnSetTensor4dDescriptor(xd, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, N, C, Hh, W));
  CK(cudnnSetTensor4dDescriptor(fd, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, Hh, W));
  std::vector<double> x(n), m(n), dy(n);
  for (int i = 0; i < n; ++i) x[i] = 2 * std::sin(0.9 * i) + 0.3, m[i] = 0.1 * std::cos(1.3 * i), dy[i] = std::cos(0.7 * i);
  cudnnLRNDescriptor_t ld;
  (cudnnCreateLRNDescriptor(&ld), own(ld, cudnnDestroyLRNDescriptor));
  Buf<double> bx(x), bm(m), bdy(dy), by(n), t1(n), t2(n), bdx(n), bdm(n);
  const double one = 1, zero = 0;
  struct P { unsigned n; double a, b, k; } ps[] = {{3, 1.0, 0.5, 1.0}, {5, 2.0, 0.75, 2.0}, {4, 0.5, 1.0, 1.5}};
  double worst = 0;
  for (const P& p : ps) {
    CK(cudnnSetLRNDescriptor(ld, p.n, p.a, p.b, p.k));
    CK(cudnnDivisiveNormalizationForward(H, ld, CUDNN_DIVNORM_PRECOMPUTED_MEANS, &one, xd, bx.p, bm.p, t1.p, t2.p,
                                         &zero, xd, by.p));
    const auto got = by.get();
    const auto want = divnorm_ref(x, m, N * C, Hh, W, p.n, p.a, p.b, p.k);
    for (int i = 0; i < n; ++i) worst = std::fmax(worst, std::fabs(got[i] - want[i]) / (1 + std::fabs(want[i])));
  }
  expect("divisive normalization matches x / (K + alpha/n^2 sum (x_j - m)^2)^beta, windows 3, 4 and 5", worst < 1e-12,
         worst);
  // Means NULL counts as zeros; float data.
  {
    std::vector<float> xf(x.begin(), x.end());
    Buf<float> fx(xf), fy(n), f1(n), f2(n);
    const float onef = 1, zerof = 0;
    CK(cudnnSetLRNDescriptor(ld, 3, 1.0, 0.5, 1.0));
    CK(cudnnDivisiveNormalizationForward(H, ld, CUDNN_DIVNORM_PRECOMPUTED_MEANS, &onef, fd, fx.p, nullptr, f1.p, f2.p,
                                         &zerof, fd, fy.p));
    const auto want = divnorm_ref(std::vector<double>(xf.begin(), xf.end()), std::vector<double>(n, 0.0), N * C, Hh,
                                  W, 3, 1.0, 0.5, 1.0);
    const auto got = fy.get();
    double e = 0;
    for (int i = 0; i < n; ++i) e = std::fmax(e, std::fabs(got[i] - want[i]) / (1 + std::fabs(want[i])));
    expect("divisive normalization in float with NULL means", e < 1e-6, e);
  }
  // Backward against central differences of the forward formula (odd window:
  // the hardware's even-window x gradient is not the forward's exact one).
  CK(cudnnSetLRNDescriptor(ld, 3, 1.5, 0.75, 1.0));
  CK(cudnnDivisiveNormalizationBackward(H, ld, CUDNN_DIVNORM_PRECOMPUTED_MEANS, &one, xd, bx.p, bm.p, bdy.p, t1.p, t2.p,
                                        &zero, xd, bdx.p, bdm.p));
  const auto gx = bdx.get(), gm = bdm.get();
  double ex = 0, em = 0;
  auto loss = [&](const std::vector<double>& xx, const std::vector<double>& mm) {
    const auto y = divnorm_ref(xx, mm, N * C, Hh, W, 3, 1.5, 0.75, 1.0);
    double s = 0;
    for (int i = 0; i < n; ++i) s += y[i] * dy[i];
    return s;
  };
  for (int i = 0; i < n; i += 7) {
    const double h = 1e-6;
    auto xp = x, xm = x, mp = m, mm = m;
    xp[i] += h, xm[i] -= h, mp[i] += h, mm[i] -= h;
    const double fx = (loss(xp, m) - loss(xm, m)) / (2 * h), fm = (loss(x, mp) - loss(x, mm)) / (2 * h);
    ex = std::fmax(ex, std::fabs(fx - gx[i]) / (std::fabs(fx) + 1e-3));
    em = std::fmax(em, std::fabs(fm - gm[i]) / (std::fabs(fm) + 1e-3));
  }
  expect("divisive normalization backward: dx matches finite differences", ex < 1e-5, ex);
  expect("divisive normalization backward: dMeans matches finite differences", em < 1e-5, em);
  cudnnTensorDescriptor_t hd = tensor();
  CK(cudnnSetTensor4dDescriptor(hd, CUDNN_TENSOR_NHWC, CUDNN_DATA_DOUBLE, N, C, Hh, W));
  expect("divisive normalization with x and y in different layouts is NOT_SUPPORTED",
         cudnnDivisiveNormalizationForward(H, ld, CUDNN_DIVNORM_PRECOMPUTED_MEANS, &one, xd, bx.p, nullptr, t1.p, t2.p,
                                           &zero, hd, by.p) == CUDNN_STATUS_NOT_SUPPORTED);
  expect("divisive normalization without its temporaries is BAD_PARAM",
         cudnnDivisiveNormalizationForward(H, ld, CUDNN_DIVNORM_PRECOMPUTED_MEANS, &one, xd, bx.p, nullptr, nullptr,
                                           t2.p, &zero, xd, by.p) == CUDNN_STATUS_BAD_PARAM);
}

/* ---- tensor transform descriptors ---- */

// The fold the hardware does (measured): pad, cut each spatial extent into
// blocks of f, output channel padBefore[C] + (i * fw + j) * C + c.
static std::vector<float> fold_ref(const std::vector<float>& src, int N, int C, int Hh, int W, const int* pb,
                                   const int* pa, int fh, int fw, int* dims) {
  const int Hp = Hh + pb[2] + pa[2], Wp = W + pb[3] + pa[3], Ho = (Hp + fh - 1) / fh, Wo = (Wp + fw - 1) / fw;
  const int No = N + pb[0] + pa[0], Co = (C + pb[1] + pa[1]) * fh * fw;
  dims[0] = No, dims[1] = Co, dims[2] = Ho, dims[3] = Wo;
  std::vector<float> out((size_t)No * Co * Ho * Wo, 0.0f);
  for (int n = 0; n < N; ++n)
    for (int c = 0; c < C; ++c)
      for (int h = 0; h < Hh; ++h)
        for (int w = 0; w < W; ++w) {
          const int hp = h + pb[2], wp = w + pb[3];
          if (hp < 0 || wp < 0 || hp >= Hp || wp >= Wp) continue;
          const int cc = pb[1] + ((hp % fh) * fw + wp % fw) * C + c;
          out[(((size_t)(n + pb[0]) * Co + cc) * Ho + hp / fh) * Wo + wp / fw] = src[(((size_t)n * C + c) * Hh + h) * W + w];
        }
  return out;
}

static void transform_descriptors() {
  cudnnTensorTransformDescriptor_t t, u;
  (cudnnCreateTensorTransformDescriptor(&t), own(t, cudnnDestroyTensorTransformDescriptor));
  (cudnnCreateTensorTransformDescriptor(&u), own(u, cudnnDestroyTensorTransformDescriptor));
  const int N = 1, C = 2, Hh = 5, W = 6;
  const int32_t pb[4] = {0, 1, 1, 0}, pa[4] = {0, 1, 0, 1}, z[4] = {0, 0, 0, 0};
  const uint32_t f23[2] = {2, 3};
  CK(cudnnSetTensorTransformDescriptor(t, 4, CUDNN_TENSOR_NCHW, pb, pa, f23, CUDNN_TRANSFORM_FOLD));
  {
    cudnnTensorFormat_t fmt;
    int32_t gb[4], ga[4];
    uint32_t gf[2];
    cudnnFoldingDirection_t dir;
    CK(cudnnGetTensorTransformDescriptor(t, 4, &fmt, gb, ga, gf, &dir));
    expect("cudnnGetTensorTransformDescriptor returns what was set",
           fmt == CUDNN_TENSOR_NCHW && dir == CUDNN_TRANSFORM_FOLD && gb[1] == 1 && gb[2] == 1 && ga[1] == 1 &&
               ga[3] == 1 && gf[0] == 2 && gf[1] == 3);
  }
  cudnnTensorDescriptor_t sd = tensor(), dd = tensor(), ud = tensor();
  CK(cudnnSetTensor4dDescriptor(sd, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, Hh, W));
  size_t bytes = 0;
  CK(cudnnInitTransformDest(t, sd, dd, &bytes));
  int want_dims[4];
  std::vector<float> src(N * C * Hh * W);
  for (size_t i = 0; i < src.size(); ++i) src[i] = (float)(i + 1);
  const int pbi[4] = {0, 1, 1, 0}, pai[4] = {0, 1, 0, 1};
  const auto want = fold_ref(src, N, C, Hh, W, pbi, pai, 2, 3, want_dims);
  {
    cudnnDataType_t dt;
    int nb = 0, dims[8], strides[8];
    CK(cudnnGetTensorNdDescriptor(dd, 8, &dt, &nb, dims, strides));
    expect("cudnnInitTransformDest gives the folded, padded shape, packed",
           nb == 4 && dims[0] == want_dims[0] && dims[1] == want_dims[1] && dims[2] == want_dims[2] &&
               dims[3] == want_dims[3] && strides[3] == 1 && bytes == want.size() * 4);
  }
  Buf<float> s(src), d(want.size()), back(src.size());
  const float one = 1, zero = 0, two = 2, half = 0.5f;
  CK(cudnnTransformTensorEx(H, t, &two, sd, s.p, &zero, dd, d.p));
  auto got = d.get();
  bool same = true;
  for (size_t i = 0; i < want.size(); ++i) same &= got[i] == 2 * want[i];
  expect("cudnnTransformTensorEx pads and folds (channels: padBefore + offset * C + c), alpha applied", same);
  expect("folding with beta != 0 is NOT_SUPPORTED",
         cudnnTransformTensorEx(H, t, &one, sd, s.p, &half, dd, d.p) == CUDNN_STATUS_NOT_SUPPORTED);
  // Unfold a padding-free fold back.
  CK(cudnnSetTensorTransformDescriptor(t, 4, CUDNN_TENSOR_NCHW, z, z, f23, CUDNN_TRANSFORM_FOLD));
  cudnnTensorDescriptor_t s2 = tensor();
  CK(cudnnSetTensor4dDescriptor(s2, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 2, 3, 4, 6));
  std::vector<float> src2(2 * 3 * 4 * 6);
  for (size_t i = 0; i < src2.size(); ++i) src2[i] = 0.5f * i - 7;
  Buf<float> a2(src2), f2(src2.size()), b2(src2.size());
  CK(cudnnInitTransformDest(t, s2, dd, &bytes));
  CK(cudnnTransformTensorEx(H, t, &one, s2, a2.p, &zero, dd, f2.p));
  CK(cudnnSetTensorTransformDescriptor(u, 4, CUDNN_TENSOR_NCHW, z, z, f23, CUDNN_TRANSFORM_UNFOLD));
  CK(cudnnInitTransformDest(u, dd, ud, &bytes));
  {
    cudnnDataType_t dt;
    int n, c, h, w, s0, s1, s3, s4;
    CK(cudnnGetTensor4dDescriptor(ud, &dt, &n, &c, &h, &w, &s0, &s1, &s3, &s4));
    expect("unfolding restores the shape", n == 2 && c == 3 && h == 4 && w == 6);
  }
  std::vector<float> prior(src2.size(), 4.0f);
  b2.put(prior);
  CK(cudnnTransformTensorEx(H, u, &two, dd, f2.p, &half, ud, b2.p));
  got = b2.get();
  same = true;
  for (size_t i = 0; i < src2.size(); ++i) same &= got[i] == 2 * src2[i] + 2.0f;
  expect("fold then unfold is the identity; unfolding takes alpha and beta", same);
  // Filters fold the same way, K padded like a batch.
  {
    cudnnFilterDescriptor_t fs = filter(), fdst = filter();
    CK(cudnnSetFilter4dDescriptor(fs, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 2, 3, 3, 3));
    CK(cudnnSetFilter4dDescriptor(fdst, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 4, 16, 2, 2));
    const int32_t fb[4] = {0, 0, 1, 1}, fa[4] = {2, 1, 0, 0};
    const uint32_t f22[2] = {2, 2};
    CK(cudnnSetTensorTransformDescriptor(t, 4, CUDNN_TENSOR_NCHW, fb, fa, f22, CUDNN_TRANSFORM_FOLD));
    std::vector<float> wv(54);
    for (int i = 0; i < 54; ++i) wv[i] = (float)(i + 1);
    Buf<float> w0(wv), w1(256);
    CK(cudnnTransformFilter(H, t, &one, fs, w0.p, &zero, fdst, w1.p));
    int dims[4];
    const int fbi[4] = {0, 0, 1, 1}, fai[4] = {2, 1, 0, 0};
    const auto fw = fold_ref(wv, 2, 3, 3, 3, fbi, fai, 2, 2, dims);
    expect("cudnnTransformFilter pads K and C and folds the taps", w1.get() == fw && dims[1] == 16);
  }
  // A plain layout change takes beta.
  CK(cudnnSetTensorTransformDescriptor(t, 4, CUDNN_TENSOR_NHWC, z, z, nullptr, CUDNN_TRANSFORM_FOLD));
  const uint32_t f11[2] = {1, 1};
  CK(cudnnSetTensorTransformDescriptor(t, 4, CUDNN_TENSOR_NHWC, z, z, f11, CUDNN_TRANSFORM_FOLD));
  cudnnTensorDescriptor_t nh = tensor();
  CK(cudnnSetTensor4dDescriptor(nh, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, N, C, Hh, W));
  Buf<float> o(src.size());
  o.put(std::vector<float>(src.size(), -2.0f));
  CK(cudnnTransformTensorEx(H, t, &two, sd, s.p, &half, nh, o.p));
  got = o.get();
  same = true;
  for (int c = 0; c < C; ++c)
    for (int i = 0; i < Hh * W; ++i) same &= got[i * C + c] == 2 * src[c * Hh * W + i] - 1.0f;
  expect("a transform without padding or folding is a layout change with alpha and beta", same);
}

/* ---- LSTM projections, clipping and the RNN getters ---- */

// One LSTM configuration, in double so that finite differences are sharp.
struct Lstm {
  int I, Hd, P, L, D;
  bool clip;
  double lclip, rclip;
};

// A host LSTM reading the weights where cudnnGetRNNWeightParams says they
// are: gates i, f, g, o; with a projection h = Wp (o * tanh(c)); clipping
// limits the cell state where it is read (c_t = f clip(c_{t-1}) + i g,
// h from tanh(clip(c_t))) and cy is the unclipped state, as measured on an
// RTX 3060.
struct HostLstm {
  Lstm c;
  int T, B;
  std::vector<int> lens;
  std::vector<std::vector<size_t>> mat, bias;  // per pseudo-layer: offsets of lin 0..8, biases 0..7 (or -1)
  void run(const std::vector<double>& w, const std::vector<double>& x, const std::vector<double>& hx,
           const std::vector<double>& cx, std::vector<double>* y, std::vector<double>* hy, std::vector<double>* cy) const {
    const int G = 4, H = c.Hd, O = c.P, D = c.D;
    // The bounds apply in float, even to double data (measured).
    auto clip = [&](double v) {
      return c.clip ? std::fmin(std::fmax(v, (double)(float)c.lclip), (double)(float)c.rclip) : v;
    };
    auto sig = [](double v) { return 1.0 / (1.0 + std::exp(-v)); };
    std::vector<double> in = x;
    int I = c.I;
    hy->assign((size_t)c.L * D * B * O, 0.0);
    cy->assign((size_t)c.L * D * B * H, 0.0);
    for (int l = 0; l < c.L; ++l) {
      std::vector<double> out((size_t)T * B * O * D, 0.0);
      for (int dir = 0; dir < D; ++dir) {
        const int pl = l * D + dir;
        for (int b = 0; b < B; ++b) {
          std::vector<double> h(hx.begin() + ((size_t)pl * B + b) * O, hx.begin() + ((size_t)pl * B + b + 1) * O);
          std::vector<double> cc(cx.begin() + ((size_t)pl * B + b) * H, cx.begin() + ((size_t)pl * B + b + 1) * H);
          for (int s = 0; s < lens[b]; ++s) {
            const int t = dir == 0 ? s : lens[b] - 1 - s;
            const double* xt = in.data() + ((size_t)t * B + b) * I;
            std::vector<double> z((size_t)G * H), hr(H);
            for (int g = 0; g < G; ++g)
              for (int j = 0; j < H; ++j) {
                double a = w[bias[pl][g] + j] + w[bias[pl][G + g] + j];
                for (int k = 0; k < I; ++k) a += w[mat[pl][g] + (size_t)j * I + k] * xt[k];
                for (int k = 0; k < O; ++k) a += w[mat[pl][G + g] + (size_t)j * O + k] * h[k];
                z[g * H + j] = a;
              }
            for (int j = 0; j < H; ++j) {
              const double i = sig(z[j]), f = sig(z[H + j]), g = std::tanh(z[2 * H + j]), o = sig(z[3 * H + j]);
              cc[j] = f * clip(cc[j]) + i * g;
              hr[j] = o * std::tanh(clip(cc[j]));
            }
            if (O < H)
              for (int p = 0; p < O; ++p) {
                double a = 0;
                for (int j = 0; j < H; ++j) a += w[mat[pl][8] + (size_t)p * H + j] * hr[j];
                h[p] = a;
              }
            else
              h = hr;
            for (int p = 0; p < O; ++p) out[((size_t)t * B + b) * O * D + (size_t)dir * O + p] = h[p];
          }
          std::copy(h.begin(), h.end(), hy->begin() + ((size_t)pl * B + b) * O);
          std::copy(cc.begin(), cc.end(), cy->begin() + ((size_t)pl * B + b) * H);
        }
      }
      in = out;
      I = O * D;
    }
    *y = in;
  }
};

static void lstm_projection(const Lstm& c, const char* name) {
  const int T = 3, B = 2, G = 4, O = c.P, LD = c.L * c.D;
  const std::vector<int> lens = {3, 2};
  cudnnRNNDescriptor_t rd;
  (cudnnCreateRNNDescriptor(&rd), own(rd, cudnnDestroyRNNDescriptor));
  CK(cudnnSetRNNDescriptor_v8(rd, CUDNN_RNN_ALGO_STANDARD, CUDNN_LSTM, CUDNN_RNN_DOUBLE_BIAS,
                              c.D == 2 ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL, CUDNN_LINEAR_INPUT,
                              CUDNN_DATA_DOUBLE, CUDNN_DATA_DOUBLE, CUDNN_DEFAULT_MATH, c.I, c.Hd, c.P, c.L, nullptr,
                              CUDNN_RNN_PADDED_IO_ENABLED));
  if (c.clip) CK(cudnnRNNSetClip_v9(rd, CUDNN_RNN_CLIP_MINMAX, c.lclip, c.rclip));
  cudnnRNNDataDescriptor_t xd, yd;
  (cudnnCreateRNNDataDescriptor(&xd), own(xd, cudnnDestroyRNNDataDescriptor));
  (cudnnCreateRNNDataDescriptor(&yd), own(yd, cudnnDestroyRNNDataDescriptor));
  CK(cudnnSetRNNDataDescriptor(xd, CUDNN_DATA_DOUBLE, CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED, T, B, c.I, lens.data(),
                               nullptr));
  CK(cudnnSetRNNDataDescriptor(yd, CUDNN_DATA_DOUBLE, CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED, T, B, O * c.D,
                               lens.data(), nullptr));
  cudnnTensorDescriptor_t hd = tensor(), cd = tensor(), md = tensor(), bd = tensor();
  const int hdims[3] = {LD, B, O}, hstr[3] = {B * O, O, 1}, cdims[3] = {LD, B, c.Hd}, cstr[3] = {B * c.Hd, c.Hd, 1};
  CK(cudnnSetTensorNdDescriptor(hd, CUDNN_DATA_DOUBLE, 3, hdims, hstr));
  CK(cudnnSetTensorNdDescriptor(cd, CUDNN_DATA_DOUBLE, 3, cdims, cstr));
  size_t wbytes = 0, work = 0, reserve = 0;
  CK(cudnnGetRNNWeightSpaceSize(H, rd, &wbytes));
  CK(cudnnGetRNNTempSpaceSizes(H, rd, CUDNN_FWD_MODE_TRAINING, xd, &work, &reserve));
  const size_t nw = wbytes / 8;
  // The weight space: every pseudo-layer's input, recurrent and projection
  // matrices, then the biases.
  size_t want_w = 0;
  for (int pl = 0; pl < LD; ++pl) {
    const int I = pl / c.D == 0 ? c.I : O * c.D;
    want_w += (size_t)G * c.Hd * I + (size_t)G * c.Hd * O + (O < c.Hd ? (size_t)O * c.Hd : 0);
  }
  const size_t mats = want_w;
  want_w += (size_t)LD * 2 * G * c.Hd;
  char what[200];
  std::snprintf(what, sizeof what, "%s: the weight space holds the matrices, then the biases (%zu)", name, want_w);
  std::vector<double> hw(nw);
  for (size_t i = 0; i < nw; ++i) hw[i] = 0.4 * std::sin(0.37 * i + 0.1);
  Buf<double> w(hw);
  HostLstm host{c, T, B, lens, {}, {}};
  bool shapes = nw == want_w;
  for (int pl = 0; pl < LD; ++pl) {
    host.mat.emplace_back(9, 0), host.bias.emplace_back(8, 0);
    const int I = pl / c.D == 0 ? c.I : O * c.D;
    for (int lin = 0; lin <= 8; ++lin) {
      void *m = nullptr, *b = nullptr;
      CK(cudnnGetRNNWeightParams(H, rd, pl, wbytes, w.p, lin, md, &m, bd, &b));
      cudnnDataType_t t;
      int nb = -1, dims[3] = {}, str[3] = {};
      CK(cudnnGetTensorNdDescriptor(md, 3, &t, &nb, dims, str));
      if (lin == 8) {
        if (O < c.Hd) shapes &= m && nb == 3 && dims[1] == O && dims[2] == c.Hd && !b;
        else shapes &= !m && nb == 0;
        if (m) host.mat[pl][8] = (double*)m - w.p;
        continue;
      }
      const int cols = lin < G ? I : O;
      shapes &= m && b && nb == 3 && dims[1] == c.Hd && dims[2] == cols;
      shapes &= (size_t)((double*)b - w.p) >= mats;
      host.mat[pl][lin] = (double*)m - w.p, host.bias[pl][lin] = (double*)b - w.p;
    }
  }
  {
    void *m = nullptr, *b = nullptr;
    shapes &= cudnnGetRNNWeightParams(H, rd, 0, wbytes, w.p, 9, md, &m, bd, &b) == CUDNN_STATUS_BAD_PARAM;
  }
  expect(what, shapes);
  // Forward against the host model.
  std::vector<double> x((size_t)T * B * c.I), hx((size_t)LD * B * O), cx((size_t)LD * B * c.Hd);
  for (size_t i = 0; i < x.size(); ++i) x[i] = std::cos(0.5 * i);
  for (size_t i = 0; i < hx.size(); ++i) hx[i] = 0.3 * std::sin(1.1 * i);
  for (size_t i = 0; i < cx.size(); ++i) cx[i] = 0.5 * std::cos(0.9 * i);
  const size_t ny = (size_t)T * B * O * c.D;
  Buf<double> bx(x), bhx(hx), bcx(cx), by(ny), bhy(hx.size()), bcy(cx.size()), wk(work / 8 + 1), rs(reserve / 8 + 1);
  CK(cudnnRNNForward(H, rd, CUDNN_FWD_MODE_TRAINING, nullptr, xd, bx.p, yd, by.p, hd, bhx.p, bhy.p, cd, bcx.p, bcy.p,
                     wbytes, w.p, work, wk.p, reserve, rs.p));
  std::vector<double> ry, rhy, rcy;
  host.run(hw, x, hx, cx, &ry, &rhy, &rcy);
  const auto gy = by.get(), ghy = bhy.get(), gcy = bcy.get();
  double e = 0;
  for (int t = 0; t < T; ++t)
    for (int b = 0; b < B; ++b)
      if (t < lens[b])
        for (int k = 0; k < O * c.D; ++k) {
          const size_t i = ((size_t)t * B + b) * O * c.D + k;
          e = std::fmax(e, std::fabs(gy[i] - ry[i]));
        }
  for (size_t i = 0; i < rhy.size(); ++i) e = std::fmax(e, std::fabs(ghy[i] - rhy[i]));
  for (size_t i = 0; i < rcy.size(); ++i) e = std::fmax(e, std::fabs(gcy[i] - rcy[i]));
  std::snprintf(what, sizeof what, "%s: forward (y, hy, cy) matches the host model", name);
  expect(what, e < 1e-12, e);
  // Backward against finite differences of L = <y, dy> + <hy, dhy> + <cy, dcy>.
  std::vector<double> dy(ny), dhy(hx.size()), dcy(cx.size());
  for (size_t i = 0; i < ny; ++i) dy[i] = std::cos(0.7 * i + 0.2);
  for (int t = 0; t < T; ++t)
    for (int b = 0; b < B; ++b)
      if (t >= lens[b])
        for (int k = 0; k < O * c.D; ++k) dy[((size_t)t * B + b) * O * c.D + k] = 0;
  for (size_t i = 0; i < dhy.size(); ++i) dhy[i] = 0.5 * std::sin(0.3 * i);
  for (size_t i = 0; i < dcy.size(); ++i) dcy[i] = 0.25 * std::cos(0.4 * i);
  Buf<double> bdy(dy), bdhy(dhy), bdcy(dcy), bdx(x.size()), bdhx(hx.size()), bdcx(cx.size()), bdw(nw);
  CK(cudnnRNNBackwardData_v8(H, rd, nullptr, yd, by.p, bdy.p, xd, bdx.p, hd, bhx.p, bdhy.p, bdhx.p, cd, bcx.p, bdcy.p,
                             bdcx.p, wbytes, w.p, work, wk.p, reserve, rs.p));
  CK(cudnnRNNBackwardWeights_v8(H, rd, CUDNN_WGRAD_MODE_ADD, nullptr, xd, bx.p, hd, bhx.p, yd, by.p, wbytes, bdw.p,
                                work, wk.p, reserve, rs.p));
  auto loss = [&](const std::vector<double>& ww, const std::vector<double>& xx, const std::vector<double>& h0,
                  const std::vector<double>& c0) {
    std::vector<double> a, b2, c2;
    host.run(ww, xx, h0, c0, &a, &b2, &c2);
    double s = 0;
    for (size_t i = 0; i < a.size(); ++i) s += a[i] * dy[i];
    for (size_t i = 0; i < b2.size(); ++i) s += b2[i] * dhy[i];
    for (size_t i = 0; i < c2.size(); ++i) s += c2[i] * dcy[i];
    return s;
  };
  const auto gx = bdx.get(), ghx = bdhx.get(), gcx = bdcx.get(), gw = bdw.get();
  auto fd_check = [&](const char* part, std::vector<double> v, const std::vector<double>& grad, int which,
                      size_t stride) {
    double worst = 0;
    for (size_t i = 0; i < v.size(); i += stride) {
      const double h = 1e-6, keep = v[i];
      v[i] = keep + h;
      const double lp = which == 0 ? loss(v, x, hx, cx) : which == 1 ? loss(hw, v, hx, cx)
                        : which == 2 ? loss(hw, x, v, cx) : loss(hw, x, hx, v);
      v[i] = keep - h;
      const double lm = which == 0 ? loss(v, x, hx, cx) : which == 1 ? loss(hw, v, hx, cx)
                        : which == 2 ? loss(hw, x, v, cx) : loss(hw, x, hx, v);
      v[i] = keep;
      const double fd = (lp - lm) / (2 * h);
      worst = std::fmax(worst, std::fabs(fd - grad[i]) / (std::fabs(fd) + 1e-3));
    }
    char t[200];
    std::snprintf(t, sizeof t, "%s: %s matches finite differences", name, part);
    expect(t, worst < 1e-5, worst);
  };
  // Only the positions a sequence reaches have a gradient.
  std::vector<double> gx_masked = gx;
  fd_check("dx", x, gx_masked, 1, 3);
  fd_check("dhx", hx, ghx, 2, 1);
  fd_check("dcx", cx, gcx, 3, 1);
  fd_check("dw (input, recurrent, projection and biases)", hw, gw, 0, nw / 61 + 1);
}

static void rnn_getters() {
  cudnnRNNDescriptor_t rd;
  (cudnnCreateRNNDescriptor(&rd), own(rd, cudnnDestroyRNNDescriptor));
  cudnnRNNAlgo_t algo;
  cudnnRNNMode_t mode;
  cudnnRNNBiasMode_t bias;
  cudnnDirectionMode_t dir;
  cudnnRNNInputMode_t input;
  cudnnDataType_t dt, mp;
  cudnnMathType_t mt;
  int32_t in, hid, proj, layers;
  cudnnDropoutDescriptor_t drop;
  uint32_t aux;
  expect("cudnnGetRNNDescriptor_v8 before a set is NOT_INITIALIZED",
         cudnnGetRNNDescriptor_v8(rd, &algo, &mode, &bias, &dir, &input, &dt, &mp, &mt, &in, &hid, &proj, &layers,
                                  &drop, &aux) == CUDNN_STATUS_NOT_INITIALIZED);
  CK(cudnnSetRNNDescriptor_v8(rd, CUDNN_RNN_ALGO_STANDARD, CUDNN_LSTM, CUDNN_RNN_SINGLE_INP_BIAS, CUDNN_BIDIRECTIONAL,
                              CUDNN_LINEAR_INPUT, CUDNN_DATA_FLOAT, CUDNN_DATA_FLOAT, CUDNN_TENSOR_OP_MATH, 5, 6, 4, 2,
                              nullptr, CUDNN_RNN_PADDED_IO_ENABLED));
  CK(cudnnGetRNNDescriptor_v8(rd, &algo, &mode, &bias, &dir, &input, &dt, &mp, &mt, &in, &hid, &proj, &layers, &drop,
                              &aux));
  expect("cudnnGetRNNDescriptor_v8 returns what was set",
         algo == CUDNN_RNN_ALGO_STANDARD && mode == CUDNN_LSTM && bias == CUDNN_RNN_SINGLE_INP_BIAS &&
             dir == CUDNN_BIDIRECTIONAL && input == CUDNN_LINEAR_INPUT && dt == CUDNN_DATA_FLOAT &&
             mp == CUDNN_DATA_FLOAT && mt == CUDNN_TENSOR_OP_MATH && in == 5 && hid == 6 && proj == 4 && layers == 2 &&
             drop == nullptr && aux == CUDNN_RNN_PADDED_IO_ENABLED);
  expect("projSize above hiddenSize is NOT_SUPPORTED",
         cudnnSetRNNDescriptor_v8(rd, CUDNN_RNN_ALGO_STANDARD, CUDNN_LSTM, CUDNN_RNN_DOUBLE_BIAS, CUDNN_UNIDIRECTIONAL,
                                  CUDNN_LINEAR_INPUT, CUDNN_DATA_FLOAT, CUDNN_DATA_FLOAT, CUDNN_DEFAULT_MATH, 4, 6, 8, 1,
                                  nullptr, 0) == CUDNN_STATUS_NOT_SUPPORTED);
  expect("a GRU with a projection is NOT_SUPPORTED",
         cudnnSetRNNDescriptor_v8(rd, CUDNN_RNN_ALGO_STANDARD, CUDNN_GRU, CUDNN_RNN_DOUBLE_BIAS, CUDNN_UNIDIRECTIONAL,
                                  CUDNN_LINEAR_INPUT, CUDNN_DATA_FLOAT, CUDNN_DATA_FLOAT, CUDNN_DEFAULT_MATH, 4, 6, 3, 1,
                                  nullptr, 0) == CUDNN_STATUS_NOT_SUPPORTED);
  // Clip settings round-trip; lclip > rclip is refused.
  cudnnRNNClipMode_t cm;
  cudnnNanPropagation_t nan;
  double lc = 0, rc = 0;
  CK(cudnnRNNSetClip_v8(rd, CUDNN_RNN_CLIP_MINMAX, CUDNN_NOT_PROPAGATE_NAN, -0.5, 0.75));
  CK(cudnnRNNGetClip_v8(rd, &cm, &nan, &lc, &rc));
  bool ok = cm == CUDNN_RNN_CLIP_MINMAX && nan == CUDNN_NOT_PROPAGATE_NAN && lc == -0.5 && rc == 0.75;
  CK(cudnnRNNSetClip_v9(rd, CUDNN_RNN_CLIP_MINMAX, -1.0, 1.0));
  CK(cudnnRNNGetClip_v9(rd, &cm, &lc, &rc));
  ok &= cm == CUDNN_RNN_CLIP_MINMAX && lc == -1.0 && rc == 1.0;
  ok &= cudnnRNNSetClip_v9(rd, CUDNN_RNN_CLIP_MINMAX, 1.0, -1.0) == CUDNN_STATUS_BAD_PARAM;
  expect("RNN clip settings round-trip, lclip > rclip is BAD_PARAM", ok);
  CK(cudnnBuildRNNDynamic(H, rd, 4));
  // The data descriptor's getter.
  cudnnRNNDataDescriptor_t xd;
  (cudnnCreateRNNDataDescriptor(&xd), own(xd, cudnnDestroyRNNDataDescriptor));
  const int lens[3] = {4, 2, 3};
  const float fill = 2.5f;
  CK(cudnnSetRNNDataDescriptor(xd, CUDNN_DATA_FLOAT, CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED, 4, 3, 7, lens, (void*)&fill));
  cudnnRNNDataLayout_t lay;
  int T = 0, B = 0, V = 0, got[5] = {-1, -1, -1, -1, -1};
  float f = 0;
  CK(cudnnGetRNNDataDescriptor(xd, &dt, &lay, &T, &B, &V, 5, got, &f));
  expect("cudnnGetRNNDataDescriptor returns what was set, the rest of the array zeroed",
         dt == CUDNN_DATA_FLOAT && lay == CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED && T == 4 && B == 3 && V == 7 &&
             got[0] == 4 && got[1] == 2 && got[2] == 3 && got[3] == 0 && got[4] == 0 && f == 2.5f);
  expect("cudnnGetRNNDataDescriptor with an array shorter than the batch is BAD_PARAM",
         cudnnGetRNNDataDescriptor(xd, &dt, &lay, &T, &B, &V, 2, got, nullptr) == CUDNN_STATUS_BAD_PARAM);
  // Unpacked sequences of different lengths need padded I/O.
  cudnnRNNDescriptor_t r2;
  (cudnnCreateRNNDescriptor(&r2), own(r2, cudnnDestroyRNNDescriptor));
  CK(cudnnSetRNNDescriptor_v8(r2, CUDNN_RNN_ALGO_STANDARD, CUDNN_LSTM, CUDNN_RNN_DOUBLE_BIAS, CUDNN_UNIDIRECTIONAL,
                              CUDNN_LINEAR_INPUT, CUDNN_DATA_FLOAT, CUDNN_DATA_FLOAT, CUDNN_DEFAULT_MATH, 3, 4, 4, 1,
                              nullptr, 0));
  cudnnRNNDataDescriptor_t a, b;
  (cudnnCreateRNNDataDescriptor(&a), own(a, cudnnDestroyRNNDataDescriptor));
  (cudnnCreateRNNDataDescriptor(&b), own(b, cudnnDestroyRNNDataDescriptor));
  const int l2[2] = {3, 2};
  CK(cudnnSetRNNDataDescriptor(a, CUDNN_DATA_FLOAT, CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED, 3, 2, 3, l2, nullptr));
  CK(cudnnSetRNNDataDescriptor(b, CUDNN_DATA_FLOAT, CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED, 3, 2, 4, l2, nullptr));
  size_t wbytes = 0;
  CK(cudnnGetRNNWeightSpaceSize(H, r2, &wbytes));
  Buf<float> w(wbytes / 4), x(18), y(24), wk(1 << 16);
  expect("unpacked sequences of different lengths without CUDNN_RNN_PADDED_IO_ENABLED are BAD_PARAM",
         cudnnRNNForward(H, r2, CUDNN_FWD_MODE_INFERENCE, nullptr, a, x.p, b, y.p, nullptr, nullptr, nullptr, nullptr,
                         nullptr, nullptr, wbytes, w.p, 1 << 18, wk.p, 0, nullptr) == CUDNN_STATUS_BAD_PARAM);
  // Without a projection, linLayerID 8 is no matrix.
  cudnnTensorDescriptor_t md = tensor(), bd = tensor();
  void *m = (void*)1, *bb = (void*)1;
  CK(cudnnGetRNNWeightParams(H, r2, 0, wbytes, w.p, 8, md, &m, bd, &bb));
  int nb = -1, dims[3], str[3];
  CK(cudnnGetTensorNdDescriptor(md, 3, &dt, &nb, dims, str));
  expect("linLayerID 8 without a projection: NULL, no dimensions", !m && !bb && nb == 0);
}

/* ---- folded backward-data descriptors ---- */

// cudnnGetFoldedConvBackwardDataDescriptors, then the pipeline they describe
// (fold the filter, pad dy, a stride-1 backward-data, unfold dx), against a
// plain strided backward-data. The descriptors' shapes are the ones an
// RTX 3060 returns.
static void folded_dgrad(int C, int K, int Hh, int W, int R, int pad, int str) {
  cudnnFilterDescriptor_t w = filter(), fw = filter();
  cudnnTensorDescriptor_t dy = tensor(), dx = tensor(), pdy = tensor(), fdx = tensor();
  cudnnConvolutionDescriptor_t c, fc;
  (cudnnCreateConvolutionDescriptor(&c), own(c, cudnnDestroyConvolutionDescriptor));
  (cudnnCreateConvolutionDescriptor(&fc), own(fc, cudnnDestroyConvolutionDescriptor));
  cudnnTensorTransformDescriptor_t t[4];
  for (auto& e : t) (cudnnCreateTensorTransformDescriptor(&e), own(e, cudnnDestroyTensorTransformDescriptor));
  const int N = 2;
  CK(cudnnSetFilter4dDescriptor(w, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, K, C, R, R));
  CK(cudnnSetTensor4dDescriptor(dx, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, Hh, W));
  CK(cudnnSetConvolution2dDescriptor(c, pad, pad, str, str, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
  int o[4];
  CK(cudnnGetConvolution2dForwardOutputDim(c, dx, w, o, o + 1, o + 2, o + 3));
  CK(cudnnSetTensor4dDescriptor(dy, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, o[0], o[1], o[2], o[3]));
  CK(cudnnGetFoldedConvBackwardDataDescriptors(H, w, dy, c, dx, CUDNN_TENSOR_NCHW, fw, pdy, fc, fdx, t[0], t[1], t[2],
                                               t[3]));
  char what[160];
  if (C == 3 && K == 4 && Hh == 8 && str == 2 && R == 3 && pad == 1) {
    cudnnDataType_t dt;
    cudnnTensorFormat_t ff;
    int k, cc, h, ww, n1, c1, h1, w1, n2, c2, h2, w2, ss[4], pd[2], sd[2], dl[2], nsp;
    cudnnConvolutionMode_t m;
    CK(cudnnGetFilter4dDescriptor(fw, &dt, &ff, &k, &cc, &h, &ww));
    CK(cudnnGetTensor4dDescriptor(pdy, &dt, &n1, &c1, &h1, &w1, ss, ss + 1, ss + 2, ss + 3));
    CK(cudnnGetTensor4dDescriptor(fdx, &dt, &n2, &c2, &h2, &w2, ss, ss + 1, ss + 2, ss + 3));
    CK(cudnnGetConvolutionNdDescriptor(fc, 2, &nsp, pd, sd, dl, &m, &dt));
    expect("folded descriptors: filter 8x16x2x2, dy 2x8x5x5, dx 2x16x4x4, stride 1 pad 1 (as on an RTX 3060)",
           k == 8 && cc == 16 && h == 2 && ww == 2 && c1 == 8 && h1 == 5 && w1 == 5 && c2 == 16 && h2 == 4 && w2 == 4 &&
               pd[0] == 1 && sd[0] == 1 && sd[1] == 1);
  }
  const size_t nw = (size_t)K * C * R * R, ny = (size_t)N * K * o[2] * o[3], nx = (size_t)N * C * Hh * W;
  std::vector<float> hw(nw), hy(ny);
  for (size_t i = 0; i < nw; ++i) hw[i] = (float)std::sin(0.3 * i);
  for (size_t i = 0; i < ny; ++i) hy[i] = (float)std::cos(0.7 * i);
  size_t sfw = 0, spy = 0, sfx = 0;
  CK(cudnnGetFilterSizeInBytes(fw, &sfw));
  CK(cudnnGetTensorSizeInBytes(pdy, &spy));
  CK(cudnnGetTensorSizeInBytes(fdx, &sfx));
  Buf<float> bw(hw), by(hy), bx(nx), bx2(nx), bfw(sfw / 4), bpy(spy / 4), bfx(sfx / 4);
  Buf<char> work(1 << 20);
  const float one = 1, zero = 0;
  int got = 0;
  cudnnConvolutionBwdDataAlgoPerf_t perf[8];
  CK(cudnnGetConvolutionBackwardDataAlgorithm_v7(H, w, dy, c, dx, 8, &got, perf));
  CK(cudnnConvolutionBackwardData(H, &one, w, bw.p, dy, by.p, c, perf[0].algo, work.p, 1 << 20, &zero, dx, bx.p));
  CK(cudnnTransformFilter(H, t[0], &one, w, bw.p, &zero, fw, bfw.p));
  CK(cudnnTransformTensorEx(H, t[1], &one, dy, by.p, &zero, pdy, bpy.p));
  CK(cudnnGetConvolutionBackwardDataAlgorithm_v7(H, fw, pdy, fc, fdx, 8, &got, perf));
  CK(cudnnConvolutionBackwardData(H, &one, fw, bfw.p, pdy, bpy.p, fc, perf[0].algo, work.p, 1 << 20, &zero, fdx, bfx.p));
  CK(cudnnTransformTensorEx(H, t[3], &one, fdx, bfx.p, &zero, dx, bx2.p));
  const auto a = bx.get(), b = bx2.get();
  double e = 0, m = 0;
  for (size_t i = 0; i < nx; ++i) e = std::fmax(e, std::fabs(a[i] - b[i])), m = std::fmax(m, std::fabs(a[i]));
  std::snprintf(what, sizeof what, "folded backward-data (C %d, K %d, %dx%d, %dx%d filter, pad %d, stride %d) = backward-data",
                C, K, Hh, W, R, R, pad, str);
  expect(what, e < 1e-3 * (1 + m), e);
}

/* ---- fused-ops plans ---- */

static cudnnTensorDescriptor_t t4(cudnnTensorFormat_t f, cudnnDataType_t t, int n, int c, int h, int w) {
  cudnnTensorDescriptor_t d = tensor();
  cudnnSetTensor4dDescriptor(d, f, t, n, c, h, w);
  return d;
}

static void fused_ops() {
  const int N = 2, C = 8, Hh = 4, W = 4, K = 32;
  const auto NHWC = CUDNN_TENSOR_NHWC;
  const cudnnFusedOpsPointerPlaceHolder_t al = CUDNN_PTR_16B_ALIGNED;
  const cudnnBatchNormMode_t spatial = CUDNN_BATCHNORM_SPATIAL;
  auto make_const = [&](cudnnFusedOps_t op) {
    cudnnFusedOpsConstParamPack_t p;
    cudnnCreateFusedOpsConstParamPack(&p, op);
    own(p, cudnnDestroyFusedOpsConstParamPack);
    return p;
  };
  auto make_var = [&](cudnnFusedOps_t op) {
    cudnnFusedOpsVariantParamPack_t p;
    cudnnCreateFusedOpsVariantParamPack(&p, op);
    own(p, cudnnDestroyFusedOpsVariantParamPack);
    return p;
  };
  auto make_plan = [&](cudnnFusedOps_t op) {
    cudnnFusedOpsPlan_t p;
    cudnnCreateFusedOpsPlan(&p, op);
    own(p, cudnnDestroyFusedOpsPlan);
    return p;
  };
  // Scale, bias, ReLU, 3x3 convolution and the output's statistics, in half NHWC.
  const auto op0 = CUDNN_FUSED_SCALE_BIAS_ACTIVATION_CONV_BNSTATS;
  cudnnTensorDescriptor_t xd = t4(NHWC, CUDNN_DATA_HALF, N, C, Hh, W), sbd = t4(NHWC, CUDNN_DATA_HALF, 1, C, 1, 1);
  cudnnTensorDescriptor_t yd = t4(NHWC, CUDNN_DATA_HALF, N, K, Hh, W), sd = t4(NHWC, CUDNN_DATA_FLOAT, 1, K, 1, 1);
  cudnnFilterDescriptor_t wd = filter();
  CK(cudnnSetFilter4dDescriptor(wd, CUDNN_DATA_HALF, NHWC, K, C, 3, 3));
  cudnnConvolutionDescriptor_t cd;
  (cudnnCreateConvolutionDescriptor(&cd), own(cd, cudnnDestroyConvolutionDescriptor));
  CK(cudnnSetConvolution2dDescriptor(cd, 1, 1, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
  CK(cudnnSetConvolutionMathType(cd, CUDNN_TENSOR_OP_MATH));
  cudnnActivationDescriptor_t ad;
  (cudnnCreateActivationDescriptor(&ad), own(ad, cudnnDestroyActivationDescriptor));
  CK(cudnnSetActivationDescriptor(ad, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0.0));
  cudnnFusedOpsConstParamPack_t cp = make_const(op0);
  CK(cudnnSetFusedOpsConstParamPackAttribute(cp, CUDNN_PARAM_XDESC, xd));
  CK(cudnnSetFusedOpsConstParamPackAttribute(cp, CUDNN_PARAM_XDATA_PLACEHOLDER, &al));
  CK(cudnnSetFusedOpsConstParamPackAttribute(cp, CUDNN_PARAM_BN_MODE, &spatial));
  CK(cudnnSetFusedOpsConstParamPackAttribute(cp, CUDNN_PARAM_BN_EQSCALEBIAS_DESC, sbd));
  CK(cudnnSetFusedOpsConstParamPackAttribute(cp, CUDNN_PARAM_BN_EQSCALE_PLACEHOLDER, &al));
  CK(cudnnSetFusedOpsConstParamPackAttribute(cp, CUDNN_PARAM_BN_EQBIAS_PLACEHOLDER, &al));
  CK(cudnnSetFusedOpsConstParamPackAttribute(cp, CUDNN_PARAM_ACTIVATION_DESC, ad));
  CK(cudnnSetFusedOpsConstParamPackAttribute(cp, CUDNN_PARAM_CONV_DESC, cd));
  CK(cudnnSetFusedOpsConstParamPackAttribute(cp, CUDNN_PARAM_WDESC, wd));
  CK(cudnnSetFusedOpsConstParamPackAttribute(cp, CUDNN_PARAM_WDATA_PLACEHOLDER, &al));
  CK(cudnnSetFusedOpsConstParamPackAttribute(cp, CUDNN_PARAM_YDESC, yd));
  CK(cudnnSetFusedOpsConstParamPackAttribute(cp, CUDNN_PARAM_YDATA_PLACEHOLDER, &al));
  CK(cudnnSetFusedOpsConstParamPackAttribute(cp, CUDNN_PARAM_YSTATS_DESC, sd));
  CK(cudnnSetFusedOpsConstParamPackAttribute(cp, CUDNN_PARAM_YSUM_PLACEHOLDER, &al));
  CK(cudnnSetFusedOpsConstParamPackAttribute(cp, CUDNN_PARAM_YSQSUM_PLACEHOLDER, &al));
  {
    cudnnTensorDescriptor_t back = tensor();
    int is_null = -1, nb = 0, dims[4], str[4];
    cudnnDataType_t dt;
    CK(cudnnGetFusedOpsConstParamPackAttribute(cp, CUDNN_PARAM_XDESC, back, &is_null));
    CK(cudnnGetTensorNdDescriptor(back, 4, &dt, &nb, dims, str));
    cudnnFusedOpsPointerPlaceHolder_t ph = CUDNN_PTR_NULL;
    int ph_null = -1;
    CK(cudnnGetFusedOpsConstParamPackAttribute(cp, CUDNN_PARAM_WDATA_PLACEHOLDER, &ph, &ph_null));
    expect("a fused-ops pack hands back a copy of its descriptor and the placeholders",
           is_null == 0 && nb == 4 && dims[1] == C && dt == CUDNN_DATA_HALF && str[1] == 1 && ph == al && ph_null == 0);
    cudnnFusedOpsConstParamPack_t other = make_const(CUDNN_FUSED_BN_FINALIZE_STATISTICS_INFERENCE);
    expect("a label from another op's table is BAD_PARAM",
           cudnnSetFusedOpsConstParamPackAttribute(other, CUDNN_PARAM_XDESC, xd) == CUDNN_STATUS_BAD_PARAM);
  }
  cudnnFusedOpsPlan_t plan = make_plan(op0);
  cudnnFusedOpsVariantParamPack_t vp = make_var(op0);
  expect("executing a plan never made is NOT_INITIALIZED",
         cudnnFusedOpsExecute(H, plan, vp) == CUDNN_STATUS_NOT_INITIALIZED);
  size_t ws = 0;
  {
    size_t w2 = 0;
    expect("a plan made from another op's pack is BAD_PARAM",
           cudnnMakeFusedOpsPlan(H, make_plan(CUDNN_FUSED_SCALE_BIAS_ACTIVATION_WGRAD), cp, &w2) ==
               CUDNN_STATUS_BAD_PARAM);
  }
  CK(cudnnMakeFusedOpsPlan(H, plan, cp, &ws));
  std::vector<__half> hx(N * C * Hh * W), hs(C), hb(C), hw(K * C * 9);
  for (size_t i = 0; i < hx.size(); ++i) hx[i] = __float2half(0.25f * ((int)(i * 7 % 17) - 8));
  for (int i = 0; i < C; ++i) hs[i] = __float2half(0.5f + 0.125f * i), hb[i] = __float2half(0.25f * (i % 3) - 0.25f);
  for (size_t i = 0; i < hw.size(); ++i) hw[i] = __float2half(0.125f * ((int)(i * 5 % 9) - 4));
  Buf<__half> x(hx), sc(hs), bi(hb), w(hw), y((size_t)N * K * Hh * W);
  Buf<float> sum(K), sq(K);
  Buf<char> work(ws + 16);
  CK(cudnnSetFusedOpsVariantParamPackAttribute(vp, CUDNN_PTR_XDATA, x.p));
  CK(cudnnSetFusedOpsVariantParamPackAttribute(vp, CUDNN_PTR_BN_EQSCALE, sc.p));
  CK(cudnnSetFusedOpsVariantParamPackAttribute(vp, CUDNN_PTR_BN_EQBIAS, bi.p));
  CK(cudnnSetFusedOpsVariantParamPackAttribute(vp, CUDNN_PTR_WDATA, w.p));
  CK(cudnnSetFusedOpsVariantParamPackAttribute(vp, CUDNN_PTR_YDATA, y.p));
  CK(cudnnSetFusedOpsVariantParamPackAttribute(vp, CUDNN_PTR_YSUM, sum.p));
  CK(cudnnSetFusedOpsVariantParamPackAttribute(vp, CUDNN_PTR_YSQSUM, sq.p));
  CK(cudnnSetFusedOpsVariantParamPackAttribute(vp, CUDNN_PTR_WORKSPACE, work.p));
  CK(cudnnSetFusedOpsVariantParamPackAttribute(vp, CUDNN_SCALAR_SIZE_T_WORKSPACE_SIZE_IN_BYTES, &ws));
  {
    void* back = nullptr;
    CK(cudnnGetFusedOpsVariantParamPackAttribute(vp, CUDNN_PTR_YDATA, &back));
    expect("a variant pack hands back its pointers", back == y.p);
  }
  CK(cudnnFusedOpsExecute(H, plan, vp));
  const auto gy = y.get();
  const auto gs = sum.get(), gq = sq.get();
  double ey = 0, es = 0;
  for (int n = 0; n < N; ++n)
    for (int k = 0; k < K; ++k)
      for (int p = 0; p < Hh; ++p)
        for (int q = 0; q < W; ++q) {
          double acc = 0;
          for (int c = 0; c < C; ++c)
            for (int r = 0; r < 3; ++r)
              for (int t = 0; t < 3; ++t) {
                const int ih = p - 1 + r, iw = q - 1 + t;
                if (ih < 0 || iw < 0 || ih >= Hh || iw >= W) continue;
                double a = __half2float(hx[((n * Hh + ih) * W + iw) * C + c]) * __half2float(hs[c]) + __half2float(hb[c]);
                a = __half2float(__float2half((float)a));
                acc += std::fmax(a, 0.0) * __half2float(hw[((k * 3 + r) * 3 + t) * C + c]);
              }
          ey = std::fmax(ey, std::fabs(__half2float(gy[((n * Hh + p) * W + q) * K + k]) - acc));
        }
  for (int k = 0; k < K; ++k) {
    double s1 = 0, s2 = 0;
    for (int i = 0; i < N * Hh * W; ++i) {
      const double v = __half2float(gy[i * K + k]);
      s1 += v, s2 += v * v;
    }
    es = std::fmax(es, std::fmax(std::fabs(s1 - gs[k]), std::fabs(s2 - gq[k])) / (1 + std::fabs(s2)));
  }
  expect("SCALE_BIAS_ACTIVATION_CONV_BNSTATS: y = conv(relu(half(x * scale + bias)))", ey < 1e-3, ey);
  expect("SCALE_BIAS_ACTIVATION_CONV_BNSTATS: per-channel sums of y and y^2", es < 1e-5, es);
  // Float x is not one of its configurations.
  CK(cudnnSetFusedOpsConstParamPackAttribute(cp, CUDNN_PARAM_XDESC, t4(NHWC, CUDNN_DATA_FLOAT, N, C, Hh, W)));
  expect("... with float x it is NOT_SUPPORTED", cudnnMakeFusedOpsPlan(H, plan, cp, &ws) == CUDNN_STATUS_NOT_SUPPORTED);
  // Batch-norm finalization, training and inference.
  for (auto op : {CUDNN_FUSED_BN_FINALIZE_STATISTICS_TRAINING, CUDNN_FUSED_BN_FINALIZE_STATISTICS_INFERENCE}) {
    const bool train = op == CUDNN_FUSED_BN_FINALIZE_STATISTICS_TRAINING;
    cudnnFusedOpsConstParamPack_t c2 = make_const(op);
    cudnnTensorDescriptor_t mv = t4(NHWC, CUDNN_DATA_FLOAT, 1, K, 1, 1), eq = t4(NHWC, CUDNN_DATA_HALF, 1, K, 1, 1);
    CK(cudnnSetFusedOpsConstParamPackAttribute(c2, CUDNN_PARAM_BN_MODE, &spatial));
    if (train) {
      CK(cudnnSetFusedOpsConstParamPackAttribute(c2, CUDNN_PARAM_YSTATS_DESC, mv));
      CK(cudnnSetFusedOpsConstParamPackAttribute(c2, CUDNN_PARAM_YSUM_PLACEHOLDER, &al));
      CK(cudnnSetFusedOpsConstParamPackAttribute(c2, CUDNN_PARAM_YSQSUM_PLACEHOLDER, &al));
      CK(cudnnSetFusedOpsConstParamPackAttribute(c2, CUDNN_PARAM_BN_SAVED_MEAN_PLACEHOLDER, &al));
      CK(cudnnSetFusedOpsConstParamPackAttribute(c2, CUDNN_PARAM_BN_SAVED_INVSTD_PLACEHOLDER, &al));
    }
    CK(cudnnSetFusedOpsConstParamPackAttribute(c2, CUDNN_PARAM_BN_SCALEBIAS_MEANVAR_DESC, mv));
    for (auto l : {CUDNN_PARAM_BN_SCALE_PLACEHOLDER, CUDNN_PARAM_BN_BIAS_PLACEHOLDER, CUDNN_PARAM_BN_RUNNING_MEAN_PLACEHOLDER,
                   CUDNN_PARAM_BN_RUNNING_VAR_PLACEHOLDER, CUDNN_PARAM_BN_EQSCALE_PLACEHOLDER,
                   CUDNN_PARAM_BN_EQBIAS_PLACEHOLDER})
      CK(cudnnSetFusedOpsConstParamPackAttribute(c2, l, &al));
    CK(cudnnSetFusedOpsConstParamPackAttribute(c2, CUDNN_PARAM_BN_EQSCALEBIAS_DESC, eq));
    cudnnFusedOpsPlan_t p2 = make_plan(op);
    size_t w2 = 0;
    CK(cudnnMakeFusedOpsPlan(H, p2, c2, &w2));
    std::vector<float> s1(K), s2(K), scale(K), bias(K), rm(K), rv(K);
    for (int k = 0; k < K; ++k)
      s1[k] = 3.0f * k - 20, s2[k] = 50.0f + 9.0f * k, scale[k] = 1.0f + 0.1f * k, bias[k] = 0.05f * k - 0.5f,
      rm[k] = 0.1f * k, rv[k] = 1.0f + 0.2f * k;
    Buf<float> bs1(s1), bs2(s2), bsc(scale), bbi(bias), brm(rm), brv(rv), bsm(K), bsi(K);
    Buf<__half> bes(K), beb(K);
    Buf<char> wk(w2 + 16);
    cudnnFusedOpsVariantParamPack_t v2 = make_var(op);
    if (train) {
      CK(cudnnSetFusedOpsVariantParamPackAttribute(v2, CUDNN_PTR_YSUM, bs1.p));
      CK(cudnnSetFusedOpsVariantParamPackAttribute(v2, CUDNN_PTR_YSQSUM, bs2.p));
      CK(cudnnSetFusedOpsVariantParamPackAttribute(v2, CUDNN_PTR_BN_SAVED_MEAN, bsm.p));
      CK(cudnnSetFusedOpsVariantParamPackAttribute(v2, CUDNN_PTR_BN_SAVED_INVSTD, bsi.p));
    }
    CK(cudnnSetFusedOpsVariantParamPackAttribute(v2, CUDNN_PTR_BN_SCALE, bsc.p));
    CK(cudnnSetFusedOpsVariantParamPackAttribute(v2, CUDNN_PTR_BN_BIAS, bbi.p));
    CK(cudnnSetFusedOpsVariantParamPackAttribute(v2, CUDNN_PTR_BN_RUNNING_MEAN, brm.p));
    CK(cudnnSetFusedOpsVariantParamPackAttribute(v2, CUDNN_PTR_BN_RUNNING_VAR, brv.p));
    CK(cudnnSetFusedOpsVariantParamPackAttribute(v2, CUDNN_PTR_BN_EQSCALE, bes.p));
    CK(cudnnSetFusedOpsVariantParamPackAttribute(v2, CUDNN_PTR_BN_EQBIAS, beb.p));
    CK(cudnnSetFusedOpsVariantParamPackAttribute(v2, CUDNN_PTR_WORKSPACE, wk.p));
    CK(cudnnSetFusedOpsVariantParamPackAttribute(v2, CUDNN_SCALAR_SIZE_T_WORKSPACE_SIZE_IN_BYTES, &w2));
    int64_t count = 32;
    double factor = 0.25, eps = 1e-3;
    if (train) {
      CK(cudnnSetFusedOpsVariantParamPackAttribute(v2, CUDNN_SCALAR_INT64_T_BN_ACCUMULATION_COUNT, &count));
      CK(cudnnSetFusedOpsVariantParamPackAttribute(v2, CUDNN_SCALAR_DOUBLE_BN_EXP_AVG_FACTOR, &factor));
    }
    CK(cudnnSetFusedOpsVariantParamPackAttribute(v2, CUDNN_SCALAR_DOUBLE_BN_EPSILON, &eps));
    CK(cudnnFusedOpsExecute(H, p2, v2));
    const auto ges = bes.get(), geb = beb.get();
    const auto grm = brm.get(), grv = brv.get(), gsm = bsm.get(), gsi = bsi.get();
    double eh = 0, ef = 0;  // the half outputs' error, and the float outputs'
    for (int k = 0; k < K; ++k) {
      const double mean = train ? s1[k] / 32.0 : rm[k];
      const double var = train ? s2[k] / 32.0 - mean * mean : rv[k];
      const double inv = 1 / std::sqrt(var + eps), wes = scale[k] * inv, web = bias[k] - mean * wes;
      eh = std::fmax(eh, std::fabs(__half2float(ges[k]) - wes) / (1 + std::fabs(wes)));
      eh = std::fmax(eh, std::fabs(__half2float(geb[k]) - web) / (1 + std::fabs(web)));
      if (train) {
        ef = std::fmax(ef, std::fabs(grm[k] - (0.75 * rm[k] + 0.25 * mean)));
        ef = std::fmax(ef, std::fabs(grv[k] - (0.75 * rv[k] + 0.25 * var * 32.0 / 31.0)) / (1 + rv[k]));
        ef = std::fmax(ef, std::fabs(gsm[k] - mean) + std::fabs(gsi[k] - inv) / inv);
      }
    }
    expect(train ? "BN_FINALIZE_STATISTICS_TRAINING: equivalent scale and bias, running and saved statistics"
                 : "BN_FINALIZE_STATISTICS_INFERENCE: equivalent scale and bias from the running statistics",
           eh < 1e-3 && ef < 1e-5, std::fmax(eh, ef));
  }
  // What this card does not run.
  {
    cudnnFusedOpsConstParamPack_t c1 = make_const(CUDNN_FUSED_SCALE_BIAS_ACTIVATION_WGRAD);
    cudnnTensorDescriptor_t dyd = t4(NHWC, CUDNN_DATA_HALF, N, K, Hh, W);
    cudnnFilterDescriptor_t dwd = filter();
    CK(cudnnSetFilter4dDescriptor(dwd, CUDNN_DATA_FLOAT, NHWC, K, C, 3, 3));
    CK(cudnnSetFusedOpsConstParamPackAttribute(c1, CUDNN_PARAM_XDESC, xd));
    CK(cudnnSetFusedOpsConstParamPackAttribute(c1, CUDNN_PARAM_CONV_DESC, cd));
    CK(cudnnSetFusedOpsConstParamPackAttribute(c1, CUDNN_PARAM_DWDESC, dwd));
    CK(cudnnSetFusedOpsConstParamPackAttribute(c1, CUDNN_PARAM_DYDESC, dyd));
    for (auto l : {CUDNN_PARAM_XDATA_PLACEHOLDER, CUDNN_PARAM_DWDATA_PLACEHOLDER, CUDNN_PARAM_DYDATA_PLACEHOLDER})
      CK(cudnnSetFusedOpsConstParamPackAttribute(c1, l, &al));
    size_t w1 = 0;
    expect("SCALE_BIAS_ACTIVATION_WGRAD is NOT_SUPPORTED (as on an RTX 3060)",
           cudnnMakeFusedOpsPlan(H, make_plan(CUDNN_FUSED_SCALE_BIAS_ACTIVATION_WGRAD), c1, &w1) ==
               CUDNN_STATUS_NOT_SUPPORTED);
  }
}

/* ---- multi-head attention ---- */

// A host model reading every weight where cudnnGetMultiHeadAttnWeights says
// it is (address, dimensions, strides), in double.
struct HostAttn {
  int H, qS, kS, vS, qP, kP, vP, oP;
  double sm;
  struct T3 { long off = -1; int d[3] = {}, s[3] = {}; };
  T3 tw[8];
  int qe() const { return qP ? qP : qS; }
  int ve() const { return vP ? vP : vS; }
  int oS() const { return oP ? oP : H * ve(); }
  void project(const std::vector<double>& w, int kind, int h, const double* x, int S, int P, double* y) const {
    if (tw[kind].off < 0) { std::copy(x, x + S, y); return; }
    for (int p = 0; p < P; ++p) {
      double a = tw[kind + 4].off >= 0 ? w[tw[kind + 4].off + (size_t)h * tw[kind + 4].s[0] + (size_t)p * tw[kind + 4].s[1]] : 0;
      for (int c = 0; c < S; ++c) a += w[tw[kind].off + (size_t)h * tw[kind].s[0] + (size_t)p * tw[kind].s[1] + (size_t)c * tw[kind].s[2]] * x[c];
      y[p] = a;
    }
  }
  // Layout [batch][time][vect] (beam 1); outputs for steps < lq, windows lo/hi.
  std::vector<double> run(const std::vector<double>& w, const std::vector<double>& Q, const std::vector<double>& K,
                          const std::vector<double>& V, const std::vector<double>* R, int B, int Tq, int Tk,
                          const std::vector<int>& lq, const std::vector<int>& lk, const std::vector<int>& lo,
                          const std::vector<int>& hi) const {
    const int E = qe(), Ve = ve(), O = oS();
    std::vector<double> out((size_t)B * Tq * O, 0.0);
    for (int b = 0; b < B; ++b)
      for (int t = 0; t < Tq; ++t) {
        double* o = out.data() + ((size_t)b * Tq + t) * O;
        for (int r = 0; r < O && oP; ++r) o[r] = tw[7].off >= 0 ? w[tw[7].off + r] : 0;
        if (t >= lq[b]) {  // past the sequence: the output bias and the residual (measured)
          if (R)
            for (int r = 0; r < O; ++r) o[r] += (*R)[((size_t)b * Tq + t) * qS + r];
          continue;
        }
        std::vector<double> hv((size_t)H * Ve, 0.0);
        for (int h = 0; h < H; ++h) {
          std::vector<double> qb(E);
          project(w, 0, h, Q.data() + ((size_t)b * Tq + t) * qS, qS, E, qb.data());
          const int first = std::max(lo[t], 0), last = std::min(hi[t], lk[b]);
          std::vector<double> sc, vbs;
          for (int k = first; k < last; ++k) {
            std::vector<double> kb(E), vb(Ve);
            project(w, 1, h, K.data() + ((size_t)b * Tk + k) * kS, kS, E, kb.data());
            project(w, 2, h, V.data() + ((size_t)b * Tk + k) * vS, vS, Ve, vb.data());
            double d = 0;
            for (int e = 0; e < E; ++e) d += kb[e] * qb[e];
            sc.push_back(sm * d);
            vbs.insert(vbs.end(), vb.begin(), vb.end());
          }
          double mx = -1e300, z = 0;
          for (double v : sc) mx = std::fmax(mx, v);
          for (double v : sc) z += std::exp(v - mx);
          for (size_t i = 0; i < sc.size(); ++i)
            for (int c = 0; c < Ve; ++c) hv[(size_t)h * Ve + c] += std::exp(sc[i] - mx) / z * vbs[i * Ve + c];
        }
        if (oP)
          for (int r = 0; r < O; ++r)
            for (int h = 0; h < H; ++h)
              for (int c = 0; c < Ve; ++c)
                o[r] += w[tw[3].off + (size_t)h * tw[3].s[0] + (size_t)r * tw[3].s[1] + (size_t)c * tw[3].s[2]] * hv[(size_t)h * Ve + c];
        else
          std::copy(hv.begin(), hv.end(), o);
        if (R)
          for (int r = 0; r < O; ++r) o[r] += (*R)[((size_t)b * Tq + t) * qS + r];
      }
    return out;
  }
};

static void attention(bool biases, int qS, int kS, int vS, int qP, int vP, int oP, bool residual, const char* name) {
  const int NH = 2, Tq = 4, Tk = 5, B = 2;
  const double sm = 0.5;
  cudnnAttnDescriptor_t ad;
  (cudnnCreateAttnDescriptor(&ad), own(ad, cudnnDestroyAttnDescriptor));
  const unsigned mode = CUDNN_ATTN_QUERYMAP_ALL_TO_ONE | (biases ? CUDNN_ATTN_ENABLE_PROJ_BIASES : 0);
  CK(cudnnSetAttnDescriptor(ad, mode, NH, sm, CUDNN_DATA_DOUBLE, CUDNN_DATA_DOUBLE, CUDNN_DEFAULT_MATH, nullptr, nullptr, qS, kS,
                            vS, qP, qP, vP, oP, Tq, Tk, B, 1));
  size_t wb = 0, ws = 0, rs = 0, wsi = 0;
  CK(cudnnGetMultiHeadAttnBuffers(H, ad, &wb, &wsi, nullptr));  // inference's workspace
  CK(cudnnGetMultiHeadAttnBuffers(H, ad, &wb, &ws, &rs));
  const size_t nw = wb / 8;
  std::vector<double> hw(nw);
  for (size_t i = 0; i < nw; ++i) hw[i] = 0.3 * std::sin(0.37 * i + 0.2);
  Buf<double> w(hw.empty() ? std::vector<double>(1) : hw);
  HostAttn host{NH, qS, kS, vS, qP, qP, vP, oP, sm, {}};
  cudnnTensorDescriptor_t td = tensor();
  const int ve = vP ? vP : vS, os = oP ? oP : NH * ve;
  bool shapes = true;
  for (int k = 0; k < 8; ++k) {
    void* a = nullptr;
    CK(cudnnGetMultiHeadAttnWeights(H, ad, (cudnnMultiHeadAttnWeightKind_t)k, wb, w.p, td, &a));
    cudnnDataType_t dt;
    int nb = -1;
    auto& e = host.tw[k];
    CK(cudnnGetTensorNdDescriptor(td, 3, &dt, &nb, e.d, e.s));
    const bool exists = (k == 0 || k == 4) ? qP > 0 : (k == 1 || k == 5) ? qP > 0 : (k == 2 || k == 6) ? vP > 0 : oP > 0;
    const bool want = exists && (k < 4 || biases);
    shapes &= want ? (a != nullptr && nb == 3) : (a == nullptr && nb == 0);
    if (a) e.off = (long)((double*)a - w.p);
  }
  // The weights in the hardware's order and strides (measured).
  if (qP && vP && oP && biases && qS == 6 && kS == 5 && vS == 4)
    shapes &= host.tw[0].off == 0 && host.tw[0].s[0] == qP && host.tw[0].s[1] == 1 && host.tw[0].s[2] == NH * qP &&
              host.tw[1].off == 36 && host.tw[2].off == 66 && host.tw[3].off == 82 && host.tw[3].s[0] == oP * vP &&
              host.tw[3].s[2] == oP && host.tw[4].off == 102 && host.tw[7].off == 118 && host.tw[7].d[0] == 1;
  char what[200];
  std::snprintf(what, sizeof what, "%s: weights where the hardware keeps them", name);
  expect(what, shapes);
  // Sequences [batch][time][vect], the second of each shorter.
  cudnnSeqDataDescriptor_t qd, kd, vd, od;
  for (auto* d : {&qd, &kd, &vd, &od}) (cudnnCreateSeqDataDescriptor(d), own(*d, cudnnDestroySeqDataDescriptor));
  cudnnSeqDataAxis_t axes[4] = {CUDNN_SEQDATA_BATCH_DIM, CUDNN_SEQDATA_BEAM_DIM, CUDNN_SEQDATA_TIME_DIM,
                                CUDNN_SEQDATA_VECT_DIM};
  auto dims = [](int T, int Bn, int V) {
    std::vector<int> d(4);
    d[CUDNN_SEQDATA_TIME_DIM] = T, d[CUDNN_SEQDATA_BATCH_DIM] = Bn, d[CUDNN_SEQDATA_BEAM_DIM] = 1, d[CUDNN_SEQDATA_VECT_DIM] = V;
    return d;
  };
  const std::vector<int> lq = {Tq, Tq - 1}, lk = {Tk, Tk - 1};
  CK(cudnnSetSeqDataDescriptor(qd, CUDNN_DATA_DOUBLE, 4, dims(Tq, B, qS).data(), axes, 2, lq.data(), nullptr));
  CK(cudnnSetSeqDataDescriptor(kd, CUDNN_DATA_DOUBLE, 4, dims(Tk, B, kS).data(), axes, 2, lk.data(), nullptr));
  CK(cudnnSetSeqDataDescriptor(vd, CUDNN_DATA_DOUBLE, 4, dims(Tk, B, vS).data(), axes, 2, lk.data(), nullptr));
  CK(cudnnSetSeqDataDescriptor(od, CUDNN_DATA_DOUBLE, 4, dims(Tq, B, os).data(), axes, 2, lq.data(), nullptr));
  const size_t nq = (size_t)B * Tq * qS, nk = (size_t)B * Tk * kS, nv = (size_t)B * Tk * vS, no = (size_t)B * Tq * os;
  std::vector<double> hq(nq), hk(nk), hv(nv), hr(residual ? nq : 0);
  for (size_t i = 0; i < nq; ++i) hq[i] = std::cos(0.5 * i);
  for (size_t i = 0; i < nk; ++i) hk[i] = std::sin(0.7 * i + 1);
  for (size_t i = 0; i < nv; ++i) hv[i] = std::cos(0.3 * i + 2);
  for (size_t i = 0; i < hr.size(); ++i) hr[i] = 0.1 * i;
  const std::vector<int> lo = {0, 0, 1, 0}, hi = {2, 3, 4, 1000};  // sliding windows, the last everything
  Buf<int> dlq(lq), dlk(lk);
  Buf<double> bq(hq), bk(hk), bv(hv), br(residual ? hr : std::vector<double>(1)), bo(no), work(std::max(ws, wsi) / 8 + 1),
      res(rs / 8 + 1);
  CK(cudnnMultiHeadAttnForward(H, ad, -1, lo.data(), hi.data(), dlq.p, dlk.p, qd, bq.p, residual ? br.p : nullptr, kd,
                               bk.p, vd, bv.p, od, bo.p, wb, w.p, ws, work.p, rs, res.p));
  const auto want = host.run(hw, hq, hk, hv, residual ? &hr : nullptr, B, Tq, Tk, lq, lk, lo, hi);
  auto got = bo.get();
  double e = 0;
  for (size_t i = 0; i < no; ++i) e = std::fmax(e, std::fabs(got[i] - want[i]));
  std::snprintf(what, sizeof what, "%s: forward over windows, steps past a sequence = the output bias", name);
  expect(what, e < 1e-12, e);
  // Inference, one step: only that step is written.
  std::vector<double> marker(no, 7.0);
  bo.put(marker);
  CK(cudnnMultiHeadAttnForward(H, ad, 2, lo.data(), hi.data(), dlq.p, dlk.p, qd, bq.p, residual ? br.p : nullptr, kd,
                               bk.p, vd, bv.p, od, bo.p, wb, w.p, wsi, work.p, 0, nullptr));
  expect("a single step with a reserve space (training) is BAD_PARAM",
         cudnnMultiHeadAttnForward(H, ad, 2, lo.data(), hi.data(), dlq.p, dlk.p, qd, bq.p, residual ? br.p : nullptr, kd,
                                   bk.p, vd, bv.p, od, bo.p, wb, w.p, std::max(ws, wsi), work.p, rs, res.p) ==
             CUDNN_STATUS_BAD_PARAM);
  got = bo.get();
  bool step = true;
  for (int b = 0; b < B; ++b)
    for (int t = 0; t < Tq; ++t)
      for (int r = 0; r < os; ++r) {
        const size_t i = ((size_t)b * Tq + t) * os + r;
        step &= t == 2 ? std::fabs(got[i] - want[i]) < 1e-12 : got[i] == 7.0;
      }
  std::snprintf(what, sizeof what, "%s: currIdx 2 writes that step alone", name);
  expect(what, step);
  // Gradients against finite differences of L = <out, dout>.
  CK(cudnnMultiHeadAttnForward(H, ad, -1, lo.data(), hi.data(), dlq.p, dlk.p, qd, bq.p, residual ? br.p : nullptr, kd,
                               bk.p, vd, bv.p, od, bo.p, wb, w.p, ws, work.p, rs, res.p));
  std::vector<double> hdo(no);
  for (int b = 0; b < B; ++b)
    for (int t = 0; t < Tq; ++t)
      for (int r = 0; r < os; ++r) hdo[((size_t)b * Tq + t) * os + r] = t < lq[b] ? std::cos(0.9 * (b * 31 + t * 7 + r)) : 0;
  Buf<double> bdo(hdo), bdq(nq), bdk(nk), bdv(nv), bdw(nw ? nw : 1);
  CK(cudnnMultiHeadAttnBackwardData(H, ad, lo.data(), hi.data(), dlq.p, dlk.p, od, bdo.p, qd, bdq.p, bq.p, kd, bdk.p,
                                    bk.p, vd, bdv.p, bv.p, wb, w.p, ws, work.p, rs, res.p));
  if (nw)
    CK(cudnnMultiHeadAttnBackwardWeights(H, ad, CUDNN_WGRAD_MODE_SET, qd, bq.p, kd, bk.p, vd, bv.p, od, bdo.p, wb, w.p,
                                         bdw.p, ws, work.p, rs, res.p));
  auto loss = [&](const std::vector<double>& ww, const std::vector<double>& q, const std::vector<double>& k,
                  const std::vector<double>& v) {
    const auto o = host.run(ww, q, k, v, residual ? &hr : nullptr, B, Tq, Tk, lq, lk, lo, hi);
    double s = 0;
    for (size_t i = 0; i < no; ++i) s += o[i] * hdo[i];
    return s;
  };
  auto fd = [&](const char* part, std::vector<double> x, const std::vector<double>& g, int which, size_t step) {
    double worst = 0;
    for (size_t i = 0; i < x.size(); i += step) {
      const double h = 1e-6, keep = x[i];
      x[i] = keep + h;
      const double lp = which == 0 ? loss(x, hq, hk, hv) : which == 1 ? loss(hw, x, hk, hv) : which == 2 ? loss(hw, hq, x, hv) : loss(hw, hq, hk, x);
      x[i] = keep - h;
      const double lm = which == 0 ? loss(x, hq, hk, hv) : which == 1 ? loss(hw, x, hk, hv) : which == 2 ? loss(hw, hq, x, hv) : loss(hw, hq, hk, x);
      x[i] = keep;
      const double f = (lp - lm) / (2 * h);
      worst = std::fmax(worst, std::fabs(f - g[i]) / (std::fabs(f) + 1e-3));
    }
    char t[200];
    std::snprintf(t, sizeof t, "%s: %s matches finite differences", name, part);
    expect(t, worst < 1e-5, worst);
  };
  fd("dqueries", hq, bdq.get(), 1, 1);
  fd("dkeys", hk, bdk.get(), 2, 1);
  fd("dvalues", hv, bdv.get(), 3, 1);
  if (nw) fd("dweights", hw, bdw.get(), 0, nw / 53 + 1);
}

static void attention_status() {
  cudnnAttnDescriptor_t ad;
  (cudnnCreateAttnDescriptor(&ad), own(ad, cudnnDestroyAttnDescriptor));
  auto set = [&](unsigned mode, double sm, int qP, int kP, cudnnDataType_t t, int qS, int kS, int beam) {
    return cudnnSetAttnDescriptor(ad, mode, 2, sm, t, t, CUDNN_DEFAULT_MATH, nullptr, nullptr, qS, kS, 4, qP, kP, 3, 4,
                                  3, 4, 2, beam);
  };
  bool ok = set(0, 0.5, 2, 2, CUDNN_DATA_FLOAT, 4, 4, 1) == CUDNN_STATUS_SUCCESS;
  ok &= set(0, 0.5, 2, 3, CUDNN_DATA_FLOAT, 4, 4, 1) == CUDNN_STATUS_BAD_PARAM;
  ok &= set(0, -1.0, 2, 2, CUDNN_DATA_FLOAT, 4, 4, 1) == CUDNN_STATUS_BAD_PARAM;
  ok &= set(0, 0.5, 0, 0, CUDNN_DATA_FLOAT, 4, 5, 1) == CUDNN_STATUS_BAD_PARAM;
  ok &= set(0, 0.5, 2, 2, CUDNN_DATA_BFLOAT16, 4, 4, 1) == CUDNN_STATUS_BAD_PARAM;
  ok &= set(CUDNN_ATTN_QUERYMAP_ONE_TO_ONE, 0.5, 2, 2, CUDNN_DATA_FLOAT, 4, 4, 2) == CUDNN_STATUS_NOT_SUPPORTED;
  expect("cudnnSetAttnDescriptor's refusals (as on an RTX 3060)", ok);
  cudnnSeqDataDescriptor_t sd;
  (cudnnCreateSeqDataDescriptor(&sd), own(sd, cudnnDestroySeqDataDescriptor));
  int dims[4];
  dims[CUDNN_SEQDATA_TIME_DIM] = 4, dims[CUDNN_SEQDATA_BATCH_DIM] = 2, dims[CUDNN_SEQDATA_BEAM_DIM] = 1, dims[CUDNN_SEQDATA_VECT_DIM] = 6;
  cudnnSeqDataAxis_t axes[4] = {CUDNN_SEQDATA_BATCH_DIM, CUDNN_SEQDATA_BEAM_DIM, CUDNN_SEQDATA_TIME_DIM,
                                CUDNN_SEQDATA_VECT_DIM};
  cudnnSeqDataAxis_t bad[4] = {CUDNN_SEQDATA_BATCH_DIM, CUDNN_SEQDATA_BEAM_DIM, CUDNN_SEQDATA_VECT_DIM,
                               CUDNN_SEQDATA_TIME_DIM};
  const int lens[2] = {4, 3};
  ok = cudnnSetSeqDataDescriptor(sd, CUDNN_DATA_FLOAT, 3, dims, axes, 2, lens, nullptr) == CUDNN_STATUS_NOT_SUPPORTED;
  ok &= cudnnSetSeqDataDescriptor(sd, CUDNN_DATA_FLOAT, 4, dims, axes, 1, lens, nullptr) == CUDNN_STATUS_BAD_PARAM;
  ok &= cudnnSetSeqDataDescriptor(sd, CUDNN_DATA_FLOAT, 4, dims, bad, 2, lens, nullptr) == CUDNN_STATUS_BAD_PARAM;
  ok &= cudnnSetSeqDataDescriptor(sd, CUDNN_DATA_FLOAT, 4, dims, axes, 2, lens, nullptr) == CUDNN_STATUS_SUCCESS;
  cudnnDataType_t t;
  int nb = 0, gd[4] = {}, gl[4] = {};
  cudnnSeqDataAxis_t ga[4];
  size_t nl = 0;
  ok &= cudnnGetSeqDataDescriptor(sd, &t, &nb, 4, gd, ga, &nl, 4, gl, nullptr) == CUDNN_STATUS_SUCCESS && nb == 4 &&
        gd[CUDNN_SEQDATA_VECT_DIM] == 6 && ga[0] == CUDNN_SEQDATA_BATCH_DIM && nl == 2 && gl[1] == 3;
  expect("sequence data descriptors: setter refusals and the getter", ok);
}

int main() {
  if (cudnnCreate(&H) != CUDNN_STATUS_SUCCESS) {
    std::printf("FAIL cudnnCreate\n");
    return 1;
  }
  vect_descriptors();
  const VConv convs[] = {
      {"INT8x4 -> INT8x4", CUDNN_DATA_INT8x4, CUDNN_DATA_INT8x4, CUDNN_DATA_INT8x4, CUDNN_TENSOR_NCHW_VECT_C, 8, 8, 1, false, false, 0.5f, 0.0f},
      {"INT8x4 -> INT8x4, beta", CUDNN_DATA_INT8x4, CUDNN_DATA_INT8x4, CUDNN_DATA_INT8x4, CUDNN_TENSOR_NCHW_VECT_C, 8, 12, 2, false, false, 0.25f, 0.5f},
      {"INT8x4 -> FLOAT NCHW", CUDNN_DATA_INT8x4, CUDNN_DATA_INT8x4, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 8, 8, 1, false, false, 0.25f, 0.5f},
      {"INT8x4 -> FLOAT NHWC", CUDNN_DATA_INT8x4, CUDNN_DATA_INT8x4, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NHWC, 8, 8, 1, false, false, 1.0f, 0.0f},
      {"UINT8x4 -> INT8x4", CUDNN_DATA_UINT8x4, CUDNN_DATA_INT8x4, CUDNN_DATA_INT8x4, CUDNN_TENSOR_NCHW_VECT_C, 8, 8, 1, false, false, 0.125f, 0.0f},
      {"INT8x32 -> INT8x32", CUDNN_DATA_INT8x32, CUDNN_DATA_INT8x32, CUDNN_DATA_INT8x32, CUDNN_TENSOR_NCHW_VECT_C, 32, 32, 1, false, false, 0.5f, 0.0f},
      {"INT8x32 -> INT8x32, stride 2, 64 channels, NO_REORDER", CUDNN_DATA_INT8x32, CUDNN_DATA_INT8x32, CUDNN_DATA_INT8x32, CUDNN_TENSOR_NCHW_VECT_C, 64, 64, 2, true, false, 0.25f, 0.0f},
      {"INT8x4 bias + ReLU -> INT8x4", CUDNN_DATA_INT8x4, CUDNN_DATA_INT8x4, CUDNN_DATA_INT8x4, CUDNN_TENSOR_NCHW_VECT_C, 8, 8, 1, false, true, 0.25f, 0.5f},
      {"INT8x4 bias + ReLU -> FLOAT", CUDNN_DATA_INT8x4, CUDNN_DATA_INT8x4, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 8, 8, 1, false, true, 0.25f, 0.5f},
      {"INT8x32 bias + ReLU, NO_REORDER", CUDNN_DATA_INT8x32, CUDNN_DATA_INT8x32, CUDNN_DATA_INT8x32, CUDNN_TENSOR_NCHW_VECT_C, 32, 64, 1, true, true, 0.125f, 0.5f},
  };
  for (const VConv& c : convs) vect_conv(c);
  vect_refusals();
  vect_transforms();
  int8_layers();
  divisive_normalization();
  transform_descriptors();
  lstm_projection({5, 6, 4, 1, 1, false, 0, 0}, "LSTM 5->6, projection 4");
  lstm_projection({5, 6, 4, 2, 2, false, 0, 0}, "LSTM 2 layers, bidirectional, projection 4");
  lstm_projection({3, 4, 4, 1, 1, true, -0.3, 0.25}, "LSTM with cell clipping");
  lstm_projection({3, 4, 2, 2, 1, true, -0.3, 0.25}, "LSTM 2 layers, projection 2, clipping");
  rnn_getters();
  fused_ops();
  folded_dgrad(3, 4, 8, 8, 3, 1, 2);
  folded_dgrad(5, 6, 9, 7, 3, 1, 2);
  folded_dgrad(3, 4, 9, 9, 3, 1, 3);
  folded_dgrad(2, 4, 8, 8, 4, 1, 2);
  attention(true, 6, 5, 4, 3, 2, 5, false, "attention with projections and biases");
  attention(false, 6, 5, 4, 3, 2, 5, false, "attention with projections, no biases");
  attention(false, 6, 6, 4, 0, 0, 0, false, "attention without projections");
  attention(true, 5, 6, 4, 3, 2, 5, true, "attention with residuals");
  attention_status();
  destroy_owned();
  cudnnDestroy(H);
  std::printf("%s\n", fails ? "FAIL" : "PASS");
  return fails ? 1 : 0;
}
