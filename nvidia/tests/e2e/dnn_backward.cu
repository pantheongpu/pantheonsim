// cuDNN's training paths in the classic API, each checked against something
// that holds on any correct implementation -- so the same program passes on a
// real GPU with NVIDIA's libcudnn.so.9 (it does, on an RTX 3060) and on
// VirtualGPU, where hosted CI runs it:
//
//   - Convolution's three passes against each other: for y = conv(x, w),
//     <y, dy> = <x, backward_data(dy, w)> = <w, backward_filter(x, dy)>, in
//     float and double, NCHW and NHWC, with groups, stride, dilation,
//     padding, both modes, 1-D and 3-D.
//   - Activation, pooling, softmax and LRN backward against central finite
//     differences of their own forward pass, in double.
//   - The algorithm lists and status codes the hardware gives: every
//     algorithm listed once, the two cuDNN never implemented listed as
//     NOT_SUPPORTED, a shape mismatch BAD_PARAM, an algorithm out of range
//     NOT_SUPPORTED, and the last error string set.
//   - Dropout: the fraction kept, the scaling, the backward pass following
//     the forward pass's mask, a reseed repeating it; and between an RNN's
//     layers, its gradients against finite differences of a reseeded
//     training pass.
//   - LSTMs in half, bfloat16 and double agree with float.
//   - CTC loss: its gradient against finite differences of its cost.
#include <cudnn.h>
#include <cuda_bf16.h>
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

// Every descriptor made here is destroyed before the program exits, so the
// sanitizer build's leak checker finds nothing left over.
static std::vector<std::function<void()>> g_owned;
template <class D, class F>
static void own(D d, F destroy) {
  g_owned.push_back([d, destroy] { destroy(d); });
}
static void destroy_owned() {
  while (!g_owned.empty()) g_owned.back()(), g_owned.pop_back();
}

static double fill(int i, double scale) { return scale * std::sin(0.7 * i + 0.3) + 0.05 * (i % 7) - 0.1; }

template <class T>
struct Buf {
  T* p = nullptr;
  size_t n = 0;
  explicit Buf(size_t n_, double scale = 0.0, int seed = 0) : n(n_) {
    cudaMalloc(&p, n * sizeof(T) + 64);
    std::vector<T> h(n);
    for (size_t i = 0; i < n; ++i) h[i] = (T)(scale == 0.0 ? 0.0 : fill((int)i + seed, scale));
    put(h);
  }
  void put(const std::vector<T>& h) { cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice); }
  std::vector<T> get() const {
    std::vector<T> h(n);
    cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost);
    return h;
  }
  ~Buf() { cudaFree(p); }
};

template <class T>
static double dot(const std::vector<T>& a, const std::vector<T>& b) {
  double s = 0;
  for (size_t i = 0; i < a.size(); ++i) s += (double)a[i] * (double)b[i];
  return s;
}

static cudnnTensorDescriptor_t nd(cudnnDataType_t t, const std::vector<int>& dims, bool nhwc = false) {
  cudnnTensorDescriptor_t d;
  (cudnnCreateTensorDescriptor(&d), own(d, cudnnDestroyTensorDescriptor));
  cudnnSetTensorNdDescriptorEx(d, nhwc ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW, t, (int)dims.size(), dims.data());
  return d;
}
static size_t count(const std::vector<int>& d) {
  size_t n = 1;
  for (int v : d) n *= (size_t)v;
  return n;
}

/* ---- convolution: the three passes agree ---- */

struct Case {
  const char* name;
  std::vector<int> x, w;
  int pad, stride, dil, groups;
  cudnnConvolutionMode_t mode;
  bool nhwc;
};

template <class T>
static void adjoint(const Case& c, cudnnDataType_t dt) {
  const int nsp = (int)c.x.size() - 2;
  cudnnConvolutionDescriptor_t cd;
  cudnnFilterDescriptor_t wd;
  (cudnnCreateConvolutionDescriptor(&cd), own(cd, cudnnDestroyConvolutionDescriptor));
  (cudnnCreateFilterDescriptor(&wd), own(wd, cudnnDestroyFilterDescriptor));
  std::vector<int> pads(nsp, c.pad), strides(nsp, c.stride), dils(nsp, c.dil);
  CK(cudnnSetConvolutionNdDescriptor(cd, nsp, pads.data(), strides.data(), dils.data(), c.mode, dt));
  CK(cudnnSetConvolutionGroupCount(cd, c.groups));
  CK(cudnnSetFilterNdDescriptor(wd, dt, c.nhwc ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW, (int)c.w.size(), c.w.data()));
  cudnnTensorDescriptor_t xd = nd(dt, c.x, c.nhwc);
  std::vector<int> y(c.x.size());
  CK(cudnnGetConvolutionNdForwardOutputDim(cd, xd, wd, (int)y.size(), y.data()));
  cudnnTensorDescriptor_t yd = nd(dt, y, c.nhwc);
  Buf<T> x(count(c.x), 1.0, 1), w(count(c.w), 0.5, 2), dy(count(y), 0.8, 3), out(count(y)), dx(count(c.x)),
      dw(count(c.w));
  const T one = 1, zero = 0;
  // Each pass with the algorithm the library ranks first, and the workspace
  // it asks for, as a framework calls it.
  int n = 0;
  cudnnConvolutionFwdAlgoPerf_t fp[16];
  cudnnConvolutionBwdDataAlgoPerf_t dp[16];
  cudnnConvolutionBwdFilterAlgoPerf_t wp[16];
  char what[160];
  std::snprintf(what, sizeof what, "%s %s: <y,dy> = <x,dx> = <w,dw>", sizeof(T) == 8 ? "double" : "float", c.name);
  // The first one listed as runnable. (NVIDIA's library runs no backward-data
  // algorithm at all for double NHWC; this one does.)
  auto first = [](auto* p, int n) {
    for (int i = 0; i < n; ++i)
      if (p[i].status == CUDNN_STATUS_SUCCESS) return i;
    return -1;
  };
  CK(cudnnGetConvolutionForwardAlgorithm_v7(H, xd, wd, cd, yd, 16, &n, fp));
  const int f = first(fp, n);
  CK(cudnnGetConvolutionBackwardDataAlgorithm_v7(H, wd, yd, cd, xd, 16, &n, dp));
  const int dd = first(dp, n);
  CK(cudnnGetConvolutionBackwardFilterAlgorithm_v7(H, xd, yd, cd, wd, 16, &n, wp));
  const int ff = first(wp, n);
  if (f < 0 || dd < 0 || ff < 0) {
    std::printf("skip %s (no algorithm runs this configuration)\n", what);
    return;
  }
  size_t fs = 0, ds = 0, ws = 0;
  CK(cudnnGetConvolutionForwardWorkspaceSize(H, xd, wd, cd, yd, fp[f].algo, &fs));
  CK(cudnnGetConvolutionBackwardDataWorkspaceSize(H, wd, yd, cd, xd, dp[dd].algo, &ds));
  CK(cudnnGetConvolutionBackwardFilterWorkspaceSize(H, xd, yd, cd, wd, wp[ff].algo, &ws));
  Buf<char> work(std::max(fs, std::max(ds, ws)) + 1);
  CK(cudnnConvolutionForward(H, &one, xd, x.p, wd, w.p, cd, fp[f].algo, work.p, fs, &zero, yd, out.p));
  CK(cudnnConvolutionBackwardData(H, &one, wd, w.p, yd, dy.p, cd, dp[dd].algo, work.p, ds, &zero, xd, dx.p));
  CK(cudnnConvolutionBackwardFilter(H, &one, xd, x.p, yd, dy.p, cd, wp[ff].algo, work.p, ws, &zero, wd, dw.p));
  const double a = dot(out.get(), dy.get()), b = dot(x.get(), dx.get()), e = dot(w.get(), dw.get());
  const double scale = std::fabs(a) + 1.0, tol = sizeof(T) == 8 ? 1e-10 : 1e-4;
  const double err = std::fmax(std::fabs(a - b), std::fabs(a - e)) / scale;
  expect(what, err < tol, err);
}

static void convolutions() {
  const Case cases[] = {
      {"3x3 pad 1", {2, 4, 7, 6}, {5, 4, 3, 3}, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, false},
      {"stride 2 dilation 2 pad 2", {1, 3, 11, 9}, {4, 3, 3, 2}, 2, 2, 2, 1, CUDNN_CROSS_CORRELATION, false},
      {"groups 2, convolution mode", {2, 4, 6, 6}, {6, 2, 3, 3}, 1, 1, 1, 2, CUDNN_CONVOLUTION, false},
      {"depthwise stride 2", {2, 6, 7, 7}, {6, 1, 3, 3}, 1, 2, 1, 6, CUDNN_CROSS_CORRELATION, false},
      {"NHWC 3x3", {2, 3, 5, 6}, {4, 3, 3, 3}, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, true},
      {"NHWC groups 3", {1, 6, 5, 5}, {3, 2, 2, 2}, 0, 1, 1, 3, CUDNN_CROSS_CORRELATION, true},
      {"3-D", {1, 2, 4, 5, 5}, {3, 2, 2, 3, 3}, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, false},
  };
  for (const Case& c : cases) adjoint<float>(c, CUDNN_DATA_FLOAT);
  for (const Case& c : cases) adjoint<double>(c, CUDNN_DATA_DOUBLE);
}

/* ---- backward passes against finite differences of the forward pass ---- */

// Central differences of L(x) = <forward(x), dy> against <backward(dy), e_i>,
// at a handful of positions i.
static void check_gradient(const char* what, size_t n, const std::vector<double>& x0,
                           const std::function<std::vector<double>(const std::vector<double>&)>& forward,
                           const std::vector<double>& dy, const std::vector<double>& dx) {
  double worst = 0;
  for (size_t i = 0; i < n; i += (n / 9 + 1)) {
    std::vector<double> xp = x0, xm = x0;
    const double h = 1e-6;
    xp[i] += h, xm[i] -= h;
    const double fd = (dot(forward(xp), dy) - dot(forward(xm), dy)) / (2 * h);
    worst = std::fmax(worst, std::fabs(fd - dx[i]) / (std::fabs(fd) + 1e-3));
  }
  expect(what, worst < 1e-5, worst);
}

static void gradients() {
  const cudnnDataType_t D = CUDNN_DATA_DOUBLE;
  const double one = 1, zero = 0;
  const std::vector<int> dims = {2, 5, 3, 4};
  const size_t n = count(dims);
  cudnnTensorDescriptor_t d = nd(D, dims);
  Buf<double> x(n, 1.3, 4), y(n), dy(n, 1.0, 5), dx(n);
  const std::vector<double> x0 = x.get(), g = dy.get();

  cudnnActivationDescriptor_t ad;
  (cudnnCreateActivationDescriptor(&ad), own(ad, cudnnDestroyActivationDescriptor));
  struct A { const char* name; cudnnActivationMode_t m; double coef; };
  for (const A& a : {A{"sigmoid", CUDNN_ACTIVATION_SIGMOID, 0}, A{"tanh", CUDNN_ACTIVATION_TANH, 0},
                     A{"elu", CUDNN_ACTIVATION_ELU, 0.7}, A{"swish", CUDNN_ACTIVATION_SWISH, 0},
                     A{"relu", CUDNN_ACTIVATION_RELU, 0}, A{"clipped relu", CUDNN_ACTIVATION_CLIPPED_RELU, 0.9}}) {
    CK(cudnnSetActivationDescriptor(ad, a.m, CUDNN_NOT_PROPAGATE_NAN, a.coef));
    CK(cudnnSetActivationDescriptorSwishBeta(ad, 1.3));
    auto fwd = [&](const std::vector<double>& in) {
      Buf<double> xi(n), yi(n);
      xi.put(in);
      cudnnActivationForward(H, ad, &one, d, xi.p, &zero, d, yi.p);
      return yi.get();
    };
    CK(cudnnActivationForward(H, ad, &one, d, x.p, &zero, d, y.p));
    CK(cudnnActivationBackward(H, ad, &one, d, y.p, d, dy.p, d, x.p, &zero, d, dx.p));
    char what[96];
    std::snprintf(what, sizeof what, "activation backward (%s) matches finite differences", a.name);
    check_gradient(what, n, x0, fwd, g, dx.get());
  }

  cudnnPoolingDescriptor_t pd;
  (cudnnCreatePoolingDescriptor(&pd), own(pd, cudnnDestroyPoolingDescriptor));
  for (auto m : {CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING, CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING,
                 CUDNN_POOLING_MAX}) {
    CK(cudnnSetPooling2dDescriptor(pd, m, CUDNN_NOT_PROPAGATE_NAN, 3, 2, 1, 1, 2, 1));
    int on, oc, oh, ow;
    CK(cudnnGetPooling2dForwardOutputDim(pd, d, &on, &oc, &oh, &ow));
    const std::vector<int> yd = {on, oc, oh, ow};
    cudnnTensorDescriptor_t pyd = nd(D, yd);
    Buf<double> py(count(yd)), pdy(count(yd), 1.0, 6), pdx(n);
    auto fwd = [&](const std::vector<double>& in) {
      Buf<double> xi(n), yi(count(yd));
      xi.put(in);
      cudnnPoolingForward(H, pd, &one, d, xi.p, &zero, pyd, yi.p);
      return yi.get();
    };
    CK(cudnnPoolingForward(H, pd, &one, d, x.p, &zero, pyd, py.p));
    CK(cudnnPoolingBackward(H, pd, &one, pyd, py.p, pyd, pdy.p, d, x.p, &zero, d, pdx.p));
    char what[96];
    std::snprintf(what, sizeof what, "pooling backward (mode %d) matches finite differences", (int)m);
    check_gradient(what, n, x0, fwd, pdy.get(), pdx.get());
  }

  for (auto algo : {CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_LOG})
    for (auto mode : {CUDNN_SOFTMAX_MODE_INSTANCE, CUDNN_SOFTMAX_MODE_CHANNEL}) {
      auto fwd = [&](const std::vector<double>& in) {
        Buf<double> xi(n), yi(n);
        xi.put(in);
        cudnnSoftmaxForward(H, algo, mode, &one, d, xi.p, &zero, d, yi.p);
        return yi.get();
      };
      CK(cudnnSoftmaxForward(H, algo, mode, &one, d, x.p, &zero, d, y.p));
      CK(cudnnSoftmaxBackward(H, algo, mode, &one, d, y.p, d, dy.p, &zero, d, dx.p));
      char what[96];
      std::snprintf(what, sizeof what, "softmax backward (algorithm %d, mode %d) matches finite differences", (int)algo,
                    (int)mode);
      check_gradient(what, n, x0, fwd, g, dx.get());
    }

  cudnnLRNDescriptor_t ld;
  (cudnnCreateLRNDescriptor(&ld), own(ld, cudnnDestroyLRNDescriptor));
  for (unsigned w : {3u, 4u}) {
    CK(cudnnSetLRNDescriptor(ld, w, 0.4, 0.75, 1.2));
    auto fwd = [&](const std::vector<double>& in) {
      Buf<double> xi(n), yi(n);
      xi.put(in);
      cudnnLRNCrossChannelForward(H, ld, CUDNN_LRN_CROSS_CHANNEL_DIM1, &one, d, xi.p, &zero, d, yi.p);
      return yi.get();
    };
    CK(cudnnLRNCrossChannelForward(H, ld, CUDNN_LRN_CROSS_CHANNEL_DIM1, &one, d, x.p, &zero, d, y.p));
    CK(cudnnLRNCrossChannelBackward(H, ld, CUDNN_LRN_CROSS_CHANNEL_DIM1, &one, d, y.p, d, dy.p, d, x.p, &zero, d,
                                    dx.p));
    char what[96];
    std::snprintf(what, sizeof what, "LRN backward (window %u) matches finite differences", w);
    check_gradient(what, n, x0, fwd, g, dx.get());
  }

  // The bias gradient is the sum of dy over everything but the channel.
  cudnnTensorDescriptor_t bd = nd(D, {1, 5, 1, 1});
  Buf<double> db(5);
  CK(cudnnConvolutionBackwardBias(H, &one, d, dy.p, &zero, bd, db.p));
  const auto got = db.get();
  double worst = 0;
  for (int c = 0; c < 5; ++c) {
    double s = 0;
    for (int b = 0; b < 2; ++b)
      for (int i = 0; i < 12; ++i) s += g[(b * 5 + c) * 12 + i];
    worst = std::fmax(worst, std::fabs(s - got[c]));
  }
  expect("convolution backward-bias sums dy per channel", worst < 1e-12, worst);
}

/* ---- algorithms and status codes ---- */

static void algorithms() {
  cudnnTensorDescriptor_t xd = nd(CUDNN_DATA_FLOAT, {2, 4, 8, 8}), yd = nd(CUDNN_DATA_FLOAT, {2, 6, 8, 8}),
                          bad = nd(CUDNN_DATA_FLOAT, {2, 6, 7, 8});
  cudnnFilterDescriptor_t wd;
  (cudnnCreateFilterDescriptor(&wd), own(wd, cudnnDestroyFilterDescriptor));
  CK(cudnnSetFilter4dDescriptor(wd, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 6, 4, 3, 3));
  cudnnConvolutionDescriptor_t cd;
  (cudnnCreateConvolutionDescriptor(&cd), own(cd, cudnnDestroyConvolutionDescriptor));
  CK(cudnnSetConvolution2dDescriptor(cd, 1, 1, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

  int fmax = 0, dmax = 0, wmax = 0;
  CK(cudnnGetConvolutionForwardAlgorithmMaxCount(H, &fmax));
  CK(cudnnGetConvolutionBackwardDataAlgorithmMaxCount(H, &dmax));
  CK(cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(H, &wmax));
  expect("the algorithm max counts cover every algorithm",
         fmax >= CUDNN_CONVOLUTION_FWD_ALGO_COUNT && dmax >= CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT &&
             wmax >= CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT);

  cudnnConvolutionFwdAlgoPerf_t fp[16];
  cudnnConvolutionBwdDataAlgoPerf_t dp[16];
  cudnnConvolutionBwdFilterAlgoPerf_t wp[16];
  int n = 0;
  auto each_once = [](auto* p, int n, int total) {
    std::vector<int> seen(total, 0);
    for (int i = 0; i < n; ++i)
      if ((int)p[i].algo >= 0 && (int)p[i].algo < total) ++seen[p[i].algo];
    for (int s : seen)
      if (s != 1) return false;
    return n == total;
  };
  auto status_of = [](auto* p, int n, int algo) {
    for (int i = 0; i < n; ++i)
      if ((int)p[i].algo == algo) return (int)p[i].status;
    return -1;
  };
  CK(cudnnGetConvolutionForwardAlgorithm_v7(H, xd, wd, cd, yd, 16, &n, fp));
  expect("forward _v7 lists every algorithm once, a runnable one first",
         each_once(fp, n, CUDNN_CONVOLUTION_FWD_ALGO_COUNT) && fp[0].status == CUDNN_STATUS_SUCCESS);
  expect("forward DIRECT is listed as NOT_SUPPORTED",
         status_of(fp, n, CUDNN_CONVOLUTION_FWD_ALGO_DIRECT) == CUDNN_STATUS_NOT_SUPPORTED);
  CK(cudnnFindConvolutionForwardAlgorithm(H, xd, wd, cd, yd, 16, &n, fp));
  expect("forward Find lists every algorithm once, timed when it ran",
         each_once(fp, n, CUDNN_CONVOLUTION_FWD_ALGO_COUNT) && fp[0].status == CUDNN_STATUS_SUCCESS && fp[0].time > 0);
  CK(cudnnGetConvolutionBackwardDataAlgorithm_v7(H, wd, yd, cd, xd, 16, &n, dp));
  expect("backward-data _v7 lists every algorithm once, a runnable one first",
         each_once(dp, n, CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT) && dp[0].status == CUDNN_STATUS_SUCCESS);
  CK(cudnnFindConvolutionBackwardDataAlgorithm(H, wd, yd, cd, xd, 16, &n, dp));
  expect("backward-data Find lists every algorithm once",
         each_once(dp, n, CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT) && dp[0].status == CUDNN_STATUS_SUCCESS);
  CK(cudnnGetConvolutionBackwardFilterAlgorithm_v7(H, xd, yd, cd, wd, 16, &n, wp));
  expect("backward-filter _v7 lists every algorithm once, a runnable one first",
         each_once(wp, n, CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT) && wp[0].status == CUDNN_STATUS_SUCCESS);
  expect("backward-filter WINOGRAD is listed as NOT_SUPPORTED",
         status_of(wp, n, CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD) == CUDNN_STATUS_NOT_SUPPORTED);
  CK(cudnnFindConvolutionBackwardFilterAlgorithm(H, xd, yd, cd, wd, 16, &n, wp));
  expect("backward-filter Find lists every algorithm once",
         each_once(wp, n, CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT) && wp[0].status == CUDNN_STATUS_SUCCESS);
  n = 0;
  CK(cudnnGetConvolutionForwardAlgorithm_v7(H, xd, wd, cd, yd, 2, &n, fp));
  expect("_v7 returns no more than requested", n == 2);

  // The algorithm _v7 put first runs, with the workspace it asks for.
  size_t ws = 1;
  CK(cudnnGetConvolutionBackwardDataAlgorithm_v7(H, wd, yd, cd, xd, 16, &n, dp));
  CK(cudnnGetConvolutionBackwardDataWorkspaceSize(H, wd, yd, cd, xd, dp[0].algo, &ws));
  Buf<float> x(512), w(216, 0.5), dy(768, 1.0), work(ws / 4 + 1);
  const float one = 1, zero = 0;
  expect("the first backward-data algorithm runs",
         cudnnConvolutionBackwardData(H, &one, wd, w.p, yd, dy.p, cd, dp[0].algo, work.p, ws, &zero, xd, x.p) ==
             CUDNN_STATUS_SUCCESS);

  size_t b = 7;
  expect("a forward workspace query with the wrong output shape is BAD_PARAM",
         cudnnGetConvolutionForwardWorkspaceSize(H, xd, wd, cd, bad, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, &b) ==
             CUDNN_STATUS_BAD_PARAM);
  expect("a backward-data call with the wrong dx shape is BAD_PARAM",
         cudnnConvolutionBackwardData(H, &one, wd, w.p, yd, dy.p, cd, CUDNN_CONVOLUTION_BWD_DATA_ALGO_1, nullptr, 0,
                                      &zero, bad, x.p) == CUDNN_STATUS_BAD_PARAM);
  char msg[512] = "";
  cudnnGetLastErrorString(msg, sizeof msg);
  expect("the last error string says something", std::strlen(msg) > 0);
  expect("a backward-filter call with the wrong dy shape is BAD_PARAM",
         cudnnConvolutionBackwardFilter(H, &one, xd, x.p, bad, dy.p, cd, CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1, nullptr,
                                        0, &zero, wd, w.p) == CUDNN_STATUS_BAD_PARAM);
  expect("a backward-data algorithm out of range is NOT_SUPPORTED",
         cudnnConvolutionBackwardData(H, &one, wd, w.p, yd, dy.p, cd, (cudnnConvolutionBwdDataAlgo_t)99, nullptr, 0,
                                      &zero, xd, x.p) == CUDNN_STATUS_NOT_SUPPORTED);
  expect("a workspace query for forward DIRECT is NOT_SUPPORTED",
         cudnnGetConvolutionForwardWorkspaceSize(H, xd, wd, cd, yd, CUDNN_CONVOLUTION_FWD_ALGO_DIRECT, &b) ==
             CUDNN_STATUS_NOT_SUPPORTED);
  expect("a null handle is BAD_PARAM",
         cudnnConvolutionBackwardData(nullptr, &one, wd, w.p, yd, dy.p, cd, CUDNN_CONVOLUTION_BWD_DATA_ALGO_1, nullptr,
                                      0, &zero, xd, x.p) == CUDNN_STATUS_BAD_PARAM);
  cudnnTensorDescriptor_t bias = nd(CUDNN_DATA_FLOAT, {1, 5, 1, 1});
  expect("a bias gradient with the wrong channel count is BAD_PARAM",
         cudnnConvolutionBackwardBias(H, &one, yd, dy.p, &zero, bias, w.p) == CUDNN_STATUS_BAD_PARAM);
  cudnnActivationDescriptor_t ad;
  (cudnnCreateActivationDescriptor(&ad), own(ad, cudnnDestroyActivationDescriptor));
  cudnnSetActivationDescriptor(ad, CUDNN_ACTIVATION_IDENTITY, CUDNN_NOT_PROPAGATE_NAN, 0);
  expect("IDENTITY activation backward is BAD_PARAM, as forward is",
         cudnnActivationBackward(H, ad, &one, xd, x.p, xd, x.p, xd, x.p, &zero, xd, x.p) == CUDNN_STATUS_BAD_PARAM);
  int dims[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1}, strides[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  cudnnTensorDescriptor_t t;
  (cudnnCreateTensorDescriptor(&t), own(t, cudnnDestroyTensorDescriptor));
  expect("a 9-dimensional tensor is NOT_SUPPORTED",
         cudnnSetTensorNdDescriptor(t, CUDNN_DATA_FLOAT, 9, dims, strides) == CUDNN_STATUS_NOT_SUPPORTED);
  dims[1] = 0;
  expect("a zero extent is BAD_PARAM",
         cudnnSetTensorNdDescriptor(t, CUDNN_DATA_FLOAT, 4, dims, strides) == CUDNN_STATUS_BAD_PARAM);
  expect("a negative padding is BAD_PARAM",
         cudnnSetConvolution2dDescriptor(cd, -1, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT) ==
             CUDNN_STATUS_BAD_PARAM);
  expect("a group count of 0 is BAD_PARAM", cudnnSetConvolutionGroupCount(cd, 0) == CUDNN_STATUS_BAD_PARAM);
}

/* ---- dropout ---- */

static void dropout() {
  size_t states = 0, rs = 0;
  CK(cudnnDropoutGetStatesSize(H, &states));
  const int n = 10000;
  cudnnTensorDescriptor_t d = nd(CUDNN_DATA_FLOAT, {1, 1, 100, 100});
  CK(cudnnDropoutGetReserveSpaceSize(d, &rs));
  expect("the reserve space holds a bit per element", rs * 8 >= (size_t)n && rs % 4 == 0);
  void* st = nullptr;
  cudaMalloc(&st, states);
  cudnnDropoutDescriptor_t dd;
  (cudnnCreateDropoutDescriptor(&dd), own(dd, cudnnDestroyDropoutDescriptor));
  const float p = 0.3f;
  CK(cudnnSetDropoutDescriptor(dd, H, p, st, states, 2024));
  Buf<float> x(n), y(n), dy(n), dx(n);
  x.put(std::vector<float>(n, 2.0f));
  dy.put(std::vector<float>(n, 1.0f));
  Buf<uint8_t> r1(rs), r2(rs);
  CK(cudnnDropoutForward(H, dd, d, x.p, d, y.p, r1.p, rs));
  const auto y1 = y.get();
  int kept = 0, wrong = 0;
  for (float v : y1) {
    if (v != 0.0f) ++kept;
    if (v != 0.0f && std::fabs(v - 2.0f / (1.0f - p)) > 1e-6f) ++wrong;
  }
  const double frac = (double)kept / n;
  expect("dropout keeps about 1 - p of the elements", std::fabs(frac - 0.7) < 0.03, frac);
  expect("dropout scales the kept ones by 1 / (1 - p)", wrong == 0, wrong);
  CK(cudnnDropoutBackward(H, dd, d, dy.p, d, dx.p, r1.p, rs));
  const auto g = dx.get();
  int mismatch = 0;
  for (int i = 0; i < n; ++i) mismatch += (y1[i] != 0.0f) != (g[i] != 0.0f);
  expect("dropout backward follows the forward pass's mask", mismatch == 0, mismatch);
  CK(cudnnDropoutForward(H, dd, d, x.p, d, y.p, r2.p, rs));
  int same = 0;
  const auto y2 = y.get();
  for (int i = 0; i < n; ++i) same += y2[i] == y1[i];
  expect("a second forward pass draws a new mask", same < n);
  CK(cudnnSetDropoutDescriptor(dd, H, p, st, states, 2024));
  CK(cudnnDropoutForward(H, dd, d, x.p, d, y.p, r2.p, rs));
  same = 0;
  const auto y3 = y.get();
  for (int i = 0; i < n; ++i) same += y3[i] == y1[i];
  expect("setting the descriptor again repeats the first mask", same == n, n - same);
  CK(cudnnSetDropoutDescriptor(dd, H, 0.0f, st, states, 1));
  CK(cudnnDropoutForward(H, dd, d, x.p, d, y.p, r2.p, rs));
  const auto y4 = y.get();
  expect("p = 0 passes everything through", y4[0] == 2.0f && y4[n - 1] == 2.0f);
  expect("a short reserve space is BAD_PARAM",
         cudnnDropoutForward(H, dd, d, x.p, d, y.p, r2.p, rs / 2) == CUDNN_STATUS_BAD_PARAM);
  cudaFree(st);
}

/* ---- RNN dropout between layers ---- */

// A two-layer LSTM with dropout 0.5 between its layers. Training draws a
// mask (the output differs from inference), a reseeded descriptor draws the
// same one again, and the gradients follow the mask: they match finite
// differences of the training forward pass, reseeded before each call.
static void rnn_dropout() {
  const int T = 4, B = 2, I = 3, Hd = 4, L = 2;
  size_t ss = 0;
  CK(cudnnDropoutGetStatesSize(H, &ss));
  void* st = nullptr;
  cudaMalloc(&st, ss);
  cudnnDropoutDescriptor_t drop;
  CK(cudnnCreateDropoutDescriptor(&drop));
  own(drop, cudnnDestroyDropoutDescriptor);
  CK(cudnnSetDropoutDescriptor(drop, H, 0.5f, st, ss, 77));
  cudnnRNNDescriptor_t rd;
  CK(cudnnCreateRNNDescriptor(&rd));
  own(rd, cudnnDestroyRNNDescriptor);
  CK(cudnnSetRNNDescriptor_v8(rd, CUDNN_RNN_ALGO_STANDARD, CUDNN_LSTM, CUDNN_RNN_DOUBLE_BIAS, CUDNN_UNIDIRECTIONAL,
                              CUDNN_LINEAR_INPUT, CUDNN_DATA_FLOAT, CUDNN_DATA_FLOAT, CUDNN_DEFAULT_MATH, I, Hd, Hd, L,
                              drop, 0));
  cudnnRNNDataDescriptor_t xd, yd;
  CK(cudnnCreateRNNDataDescriptor(&xd));
  own(xd, cudnnDestroyRNNDataDescriptor);
  CK(cudnnCreateRNNDataDescriptor(&yd));
  own(yd, cudnnDestroyRNNDataDescriptor);
  const int lens[B] = {T, T};
  CK(cudnnSetRNNDataDescriptor(xd, CUDNN_DATA_FLOAT, CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED, T, B, I, lens, nullptr));
  CK(cudnnSetRNNDataDescriptor(yd, CUDNN_DATA_FLOAT, CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED, T, B, Hd, lens, nullptr));
  cudnnTensorDescriptor_t hd = nd(CUDNN_DATA_FLOAT, {L, B, Hd});
  size_t wbytes = 0, work = 0, reserve = 0;
  CK(cudnnGetRNNWeightSpaceSize(H, rd, &wbytes));
  CK(cudnnGetRNNTempSpaceSizes(H, rd, CUDNN_FWD_MODE_TRAINING, xd, &work, &reserve));
  const size_t nw = wbytes / 4, nx = (size_t)T * B * I, ny = (size_t)T * B * Hd;
  Buf<float> w(nw, 0.4, 1), x(nx, 1.0, 2), y(ny), dy(ny, 1.0, 3), dx(nx), dw(nw), wk(work / 4 + 1), rs(reserve / 4 + 1);
  auto forward = [&](const float* xp, const float* wp, float* yp, cudnnForwardMode_t mode) {
    CK(cudnnSetDropoutDescriptor(drop, H, 0.5f, st, ss, 77));  // the same mask each time
    CK(cudnnRNNForward(H, rd, mode, nullptr, xd, xp, yd, yp, hd, nullptr, nullptr, hd, nullptr, nullptr, wbytes, wp,
                       work, wk.p, mode == CUDNN_FWD_MODE_TRAINING ? reserve : 0,
                       mode == CUDNN_FWD_MODE_TRAINING ? rs.p : nullptr));
  };
  Buf<float> yi(ny), y2(ny);
  forward(x.p, w.p, yi.p, CUDNN_FWD_MODE_INFERENCE);
  forward(x.p, w.p, y2.p, CUDNN_FWD_MODE_TRAINING);
  forward(x.p, w.p, y.p, CUDNN_FWD_MODE_TRAINING);
  const auto vi = yi.get(), v2 = y2.get(), v = y.get();
  int differ = 0, same = 0;
  for (size_t i = 0; i < ny; ++i) differ += vi[i] != v[i], same += v2[i] == v[i];
  expect("RNN training with dropout differs from inference", differ > 0);
  expect("a reseeded dropout descriptor repeats the RNN's mask", same == (int)ny, (int)ny - same);
  CK(cudnnRNNBackwardData_v8(H, rd, nullptr, yd, y.p, dy.p, xd, dx.p, hd, nullptr, nullptr, nullptr, hd, nullptr,
                             nullptr, nullptr, wbytes, w.p, work, wk.p, reserve, rs.p));
  CK(cudnnRNNBackwardWeights_v8(H, rd, CUDNN_WGRAD_MODE_ADD, nullptr, xd, x.p, hd, nullptr, yd, y.p, wbytes, dw.p, work,
                                wk.p, reserve, rs.p));
  const auto g = dy.get(), gx = dx.get(), gw = dw.get();
  auto loss = [&](const std::vector<float>& xs, const std::vector<float>& ws) {
    Buf<float> x2(nx), w2(nw), yy(ny), r2(reserve / 4 + 1);
    x2.put(xs), w2.put(ws);
    cudnnSetDropoutDescriptor(drop, H, 0.5f, st, ss, 77);
    cudnnRNNForward(H, rd, CUDNN_FWD_MODE_TRAINING, nullptr, xd, x2.p, yd, yy.p, hd, nullptr, nullptr, hd, nullptr,
                    nullptr, wbytes, w2.p, work, wk.p, reserve, r2.p);
    return dot(yy.get(), g);
  };
  const auto x0 = x.get(), w0 = w.get();
  const float e = 1e-2f;
  double worst = 0;
  for (size_t i : {size_t(0), nx / 2, nx - 1}) {
    auto a = x0, b = x0;
    a[i] += e, b[i] -= e;
    worst = std::fmax(worst, std::fabs((loss(a, w0) - loss(b, w0)) / (2 * e) - gx[i]));
  }
  expect("RNN backward-data through dropout matches finite differences", worst < 5e-3, worst);
  worst = 0;
  for (size_t i : {size_t(1), nw / 3, nw / 2, nw - 2}) {
    auto a = w0, b = w0;
    a[i] += e, b[i] -= e;
    worst = std::fmax(worst, std::fabs((loss(x0, a) - loss(x0, b)) / (2 * e) - gw[i]));
  }
  expect("RNN backward-weights through dropout matches finite differences", worst < 5e-3, worst);
  cudaFree(st);
}

/* ---- RNNs in half, bfloat16 and double ---- */

// One LSTM's output in the given type, from the same values (rounded to the
// type): the weights where cudnnGetRNNWeightSpaceSize says, the input dense.
template <class T>
static std::vector<double> lstm_output(cudnnDataType_t type, cudnnDataType_t math, int* status) {
  const int Tn = 3, B = 2, I = 3, Hd = 4, L = 2;
  *status = 0;
  cudnnRNNDescriptor_t rd;
  (cudnnCreateRNNDescriptor(&rd), own(rd, cudnnDestroyRNNDescriptor));
  cudnnStatus_t s = cudnnSetRNNDescriptor_v8(rd, CUDNN_RNN_ALGO_STANDARD, CUDNN_LSTM, CUDNN_RNN_DOUBLE_BIAS,
                                             CUDNN_UNIDIRECTIONAL, CUDNN_LINEAR_INPUT, type, math, CUDNN_DEFAULT_MATH,
                                             I, Hd, Hd, L, nullptr, 0);
  if (s) { *status = s; return {}; }
  cudnnRNNDataDescriptor_t xd, yd;
  (cudnnCreateRNNDataDescriptor(&xd), own(xd, cudnnDestroyRNNDataDescriptor)), (cudnnCreateRNNDataDescriptor(&yd), own(yd, cudnnDestroyRNNDataDescriptor));
  const int lens[B] = {Tn, Tn};
  cudnnSetRNNDataDescriptor(xd, type, CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED, Tn, B, I, lens, nullptr);
  cudnnSetRNNDataDescriptor(yd, type, CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED, Tn, B, Hd, lens, nullptr);
  cudnnTensorDescriptor_t hd = nd(type, {L, B, Hd});
  size_t wbytes = 0, work = 0, reserve = 0;
  cudnnGetRNNWeightSpaceSize(H, rd, &wbytes);
  cudnnGetRNNTempSpaceSizes(H, rd, CUDNN_FWD_MODE_INFERENCE, xd, &work, &reserve);
  const size_t nw = wbytes / sizeof(T), nx = (size_t)Tn * B * I, ny = (size_t)Tn * B * Hd;
  std::vector<T> w(nw), x(nx);
  for (size_t i = 0; i < nw; ++i) w[i] = (T)(float)fill((int)i, 0.4);
  for (size_t i = 0; i < nx; ++i) x[i] = (T)(float)fill((int)i + 5, 1.0);
  Buf<T> dw(nw), dx(nx), dy(ny);
  Buf<char> wk(work + 16);
  dw.put(w), dx.put(x);
  s = cudnnRNNForward(H, rd, CUDNN_FWD_MODE_INFERENCE, nullptr, xd, dx.p, yd, dy.p, hd, nullptr, nullptr, hd, nullptr,
                      nullptr, wbytes, dw.p, work, wk.p, 0, nullptr);
  *status = s;
  std::vector<double> out;
  for (T v : dy.get()) out.push_back((double)(float)v);
  return out;
}

static void rnn_types() {
  int sf, sh, sb, sd;
  const auto f = lstm_output<float>(CUDNN_DATA_FLOAT, CUDNN_DATA_FLOAT, &sf);
  const auto h = lstm_output<__half>(CUDNN_DATA_HALF, CUDNN_DATA_FLOAT, &sh);
  const auto b = lstm_output<__nv_bfloat16>(CUDNN_DATA_BFLOAT16, CUDNN_DATA_FLOAT, &sb);
  const auto d = lstm_output<double>(CUDNN_DATA_DOUBLE, CUDNN_DATA_DOUBLE, &sd);
  auto worst = [&](const std::vector<double>& v) {
    double m = 0;
    for (size_t i = 0; i < v.size() && i < f.size(); ++i) m = std::fmax(m, std::fabs(v[i] - f[i]));
    return v.size() == f.size() ? m : 1e30;
  };
  expect("a float LSTM runs", sf == 0, sf);
  expect("a half LSTM (float math) agrees with float to half precision", sh == 0 && worst(h) < 1e-2, sh ? sh : worst(h));
  expect("a bfloat16 LSTM agrees with float to bfloat16 precision", sb == 0 && worst(b) < 5e-2, sb ? sb : worst(b));
  expect("a double LSTM agrees with float", sd == 0 && worst(d) < 1e-5, sd ? sd : worst(d));
}

/* ---- CTC loss ---- */

// CTC's gradient with respect to the activations (SOFTMAX mode) against
// finite differences of its own cost, and the cost of a one-step,
// one-symbol label against -log of that symbol's probability.
static void ctc() {
  const int T = 5, N = 2, A = 4;
  cudnnTensorDescriptor_t pd = nd(CUDNN_DATA_FLOAT, {T, N, A});
  cudnnCTCLossDescriptor_t cd;
  CK(cudnnCreateCTCLossDescriptor(&cd));
  own(cd, cudnnDestroyCTCLossDescriptor);
  CK(cudnnSetCTCLossDescriptor_v9(cd, CUDNN_DATA_FLOAT, CUDNN_LOSS_NORMALIZATION_SOFTMAX, CUDNN_CTC_ZERO_OOB_GRADIENTS,
                                  4));
  const int labels[] = {1, 2, 3}, llen[N] = {2, 1}, ilen[N] = {5, 4};
  size_t ws = 0;
  CK(cudnnGetCTCLossWorkspaceSize(H, pd, pd, labels, llen, ilen, CUDNN_CTC_LOSS_ALGO_DETERMINISTIC, cd, &ws));
  Buf<char> work(ws + 16);
  Buf<float> x(T * N * A, 1.2, 4), costs(N), grad(T * N * A);
  CK(cudnnCTCLoss(H, pd, x.p, labels, llen, ilen, costs.p, pd, grad.p, CUDNN_CTC_LOSS_ALGO_DETERMINISTIC, cd, work.p,
                  ws));
  const auto x0 = x.get(), g = grad.get();
  auto cost = [&](const std::vector<float>& xs) {
    Buf<float> xi(T * N * A), c(N);
    xi.put(xs);
    cudnnCTCLoss(H, pd, xi.p, labels, llen, ilen, c.p, pd, nullptr, CUDNN_CTC_LOSS_ALGO_DETERMINISTIC, cd, work.p, ws);
    const auto v = c.get();
    return (double)v[0] + v[1];
  };
  double worst = 0;
  for (int t = 0; t < 4; ++t)  // both sequences run 4 steps or more
    for (int i : {0, 1, 2, 3, 5, 6}) {
      const size_t k = (size_t)t * N * A + i;
      auto a = x0, b = x0;
      a[k] += 1e-2f, b[k] -= 1e-2f;
      worst = std::fmax(worst, std::fabs((cost(a) - cost(b)) / 2e-2 - g[k]));
    }
  expect("CTC gradient matches finite differences of its cost", worst < 2e-3, worst);
  // One step, one symbol: the only path is that symbol.
  cudnnTensorDescriptor_t p1 = nd(CUDNN_DATA_FLOAT, {1, 1, A});
  const int one_label[] = {2}, one[] = {1};
  Buf<float> x1(A, 1.0, 9), c1(1);
  CK(cudnnCTCLoss(H, p1, x1.p, one_label, one, one, c1.p, p1, nullptr, CUDNN_CTC_LOSS_ALGO_DETERMINISTIC, cd, work.p,
                  ws));
  const auto v = x1.get();
  double sum = 0;
  for (float e : v) sum += std::exp((double)e);
  const double want = -(v[2] - std::log(sum));
  expect("a one-step CTC cost is -log of the symbol's probability", std::fabs(c1.get()[0] - want) < 1e-5,
         c1.get()[0] - want);
}

int main() {
  if (cudnnCreate(&H) != CUDNN_STATUS_SUCCESS) {
    std::printf("FAIL cudnnCreate\n");
    return 1;
  }
  convolutions();
  gradients();
  algorithms();
  dropout();
  rnn_dropout();
  rnn_types();
  ctc();
  destroy_owned();
  cudnnDestroy(H);
  std::printf("%s\n", fails ? "FAIL" : "PASS");
  return fails ? 1 : 0;
}
