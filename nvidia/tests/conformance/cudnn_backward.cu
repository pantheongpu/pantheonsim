// Differential conformance for the cuDNN shim's training paths, float and
// double: convolution backward-data, backward-filter and backward-bias (groups,
// dilation, stride, both modes, NHWC, strided and 3-D tensors), the fused
// convolution-bias-activation, and the backward passes of activation,
// pooling, softmax and LRN; reductions with indices, op-tensor broadcasts,
// transforms, dropout's backward pass from a known mask, and batch
// normalization's backward pass in NHWC, with and without a fused add and
// activation, and through the cuDNN 8 normalization API; the spatial
// transformer's grid and sampler, both ways; and CTC loss. The same binary runs against
// NVIDIA's libcudnn.so.9 and against VirtualGPU's; the printed values must
// agree (nvidia/tests/conformance/golden/cudnn_backward.rtx3060.txt holds what
// an RTX 3060 printed).
//
// Each result is printed as aggregates that do not cancel -- the sum of
// absolute values, of squares, and a position-weighted absolute sum -- so a
// summation order that differs between the hardware and the host moves them
// by rounding only, and a wrong element moves them by far more.
#include <cudnn.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#define CK(x) do { cudnnStatus_t s_ = (x); if (s_ != CUDNN_STATUS_SUCCESS) { \
  std::printf("%s -> %d\n", #x, (int)s_); return; } } while (0)

static cudnnHandle_t H;

static float fill(int i, float scale) { return scale * std::sin(0.7f * i + 0.3f) + 0.1f * (i % 5) - 0.2f; }

template <class T>
static void dump(const char* tag, const std::vector<T>& v) {
  double a = 0, q = 0, w = 0;
  for (size_t i = 0; i < v.size(); ++i) {
    const double x = (double)v[i];
    a += std::fabs(x), q += x * x, w += std::fabs(x) * (1 + i % 7);
  }
  std::printf("%-40s n=%zu abs=%.6e sq=%.6e wabs=%.6e\n", tag, v.size(), a, q, w);
}

template <class T>
struct Buf {
  T* p = nullptr;
  size_t n = 0;
  Buf(size_t n_, float scale, int seed = 0) : n(n_) {
    cudaMalloc(&p, n * sizeof(T) + 256);
    std::vector<T> h(n);
    for (size_t i = 0; i < n; ++i) h[i] = scale == 0.0f ? T(0) : (T)fill((int)i + seed, scale);
    cudaMemcpy(p, h.data(), n * sizeof(T), cudaMemcpyHostToDevice);
  }
  std::vector<T> get() const {
    std::vector<T> h(n);
    cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost);
    return h;
  }
  ~Buf() { cudaFree(p); }
};

static cudnnTensorDescriptor_t tensor(cudnnDataType_t t, std::vector<int> dims, std::vector<int> strides = {}) {
  cudnnTensorDescriptor_t d;
  cudnnCreateTensorDescriptor(&d);
  if (strides.empty()) {
    strides.assign(dims.size(), 1);
    for (int i = (int)dims.size() - 2; i >= 0; --i) strides[i] = strides[i + 1] * dims[i + 1];
  }
  cudnnStatus_t s = cudnnSetTensorNdDescriptor(d, t, (int)dims.size(), dims.data(), strides.data());
  if (s) std::printf("SetTensorNd -> %d\n", (int)s);
  return d;
}
static cudnnTensorDescriptor_t tensor4(cudnnTensorFormat_t f, cudnnDataType_t t, int n, int c, int h, int w) {
  cudnnTensorDescriptor_t d;
  cudnnCreateTensorDescriptor(&d);
  cudnnStatus_t s = cudnnSetTensor4dDescriptor(d, f, t, n, c, h, w);
  if (s) std::printf("SetTensor4d -> %d\n", (int)s);
  return d;
}
static size_t span_of(cudnnTensorDescriptor_t d) {
  size_t b = 0;
  cudnnGetTensorSizeInBytes(d, &b);
  return b;
}

/* ---- convolution ---- */

struct ConvCase {
  const char* tag;
  std::vector<int> x, w;  // N C spatial..., K C/g spatial...
  int pad, stride, dil, groups;
  cudnnConvolutionMode_t mode;
  cudnnTensorFormat_t fmt;  // NHWC: x, y and w channels-last
  bool strided;             // x and y with gaps between rows
};

template <class T>
static void conv_case(const ConvCase& cc, cudnnDataType_t dt) {
  const int nsp = (int)cc.x.size() - 2;
  cudnnConvolutionDescriptor_t cd;
  cudnnFilterDescriptor_t wd;
  cudnnCreateConvolutionDescriptor(&cd);
  cudnnCreateFilterDescriptor(&wd);
  std::vector<int> pads(nsp, cc.pad), strs(nsp, cc.stride), dils(nsp, cc.dil);
  CK(cudnnSetConvolutionNdDescriptor(cd, nsp, pads.data(), strs.data(), dils.data(), cc.mode, dt));
  CK(cudnnSetConvolutionGroupCount(cd, cc.groups));
  CK(cudnnSetFilterNdDescriptor(wd, dt, cc.fmt, (int)cc.w.size(), cc.w.data()));
  cudnnTensorDescriptor_t xd, yd;
  cudnnCreateTensorDescriptor(&xd);
  cudnnCreateTensorDescriptor(&yd);
  if (cc.fmt == CUDNN_TENSOR_NHWC) {
    CK(cudnnSetTensorNdDescriptorEx(xd, CUDNN_TENSOR_NHWC, dt, (int)cc.x.size(), cc.x.data()));
  } else if (cc.strided) {
    // Rows padded to twice their width.
    std::vector<int> s(cc.x.size(), 1);
    s[cc.x.size() - 1] = 1;
    s[cc.x.size() - 2] = 2 * cc.x.back();
    for (int i = (int)cc.x.size() - 3; i >= 0; --i) s[i] = s[i + 1] * cc.x[i + 1];
    CK(cudnnSetTensorNdDescriptor(xd, dt, (int)cc.x.size(), cc.x.data(), s.data()));
  } else {
    xd = tensor(dt, cc.x);
  }
  std::vector<int> y(cc.x.size());
  CK(cudnnGetConvolutionNdForwardOutputDim(cd, xd, wd, (int)y.size(), y.data()));
  if (cc.fmt == CUDNN_TENSOR_NHWC) {
    CK(cudnnSetTensorNdDescriptorEx(yd, CUDNN_TENSOR_NHWC, dt, (int)y.size(), y.data()));
  } else if (cc.strided) {
    std::vector<int> s(y.size(), 1);
    s[y.size() - 2] = y.back() + 3;
    for (int i = (int)y.size() - 3; i >= 0; --i) s[i] = s[i + 1] * y[i + 1];
    CK(cudnnSetTensorNdDescriptor(yd, dt, (int)y.size(), y.data(), s.data()));
  } else {
    yd = tensor(dt, y);
  }
  char tag[96];
  std::snprintf(tag, sizeof tag, "%s y=", cc.tag);
  std::printf("%s", tag);
  for (int v : y) std::printf("%d ", v);
  std::printf("\n");
  const size_t xn = span_of(xd) / sizeof(T), yn = span_of(yd) / sizeof(T);
  size_t wb = 0;
  cudnnGetFilterSizeInBytes(wd, &wb);
  const size_t wn = wb / sizeof(T);
  Buf<T> x(xn, 1.0f), w(wn, 0.5f, 3), dy(yn, 0.8f, 7), out_y(yn, 0.3f, 11), out_x(xn, 0.3f, 13), out_w(wn, 0.3f, 17);
  const T one = 1, zero = 0, a = 0.5, b = 0.25;
  // Forward, beta 0; backward-data with alpha and beta; backward-filter beta 0.
  CK(cudnnConvolutionForward(H, &one, xd, x.p, wd, w.p, cd, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, nullptr, 0,
                             &zero, yd, out_y.p));
  std::snprintf(tag, sizeof tag, "%s fwd", cc.tag);
  dump(tag, out_y.get());
  // The hardware takes strided rows in dx but not in dy for backward-data,
  // and in dy but not in x for backward-filter: the strided case gives each
  // the packed form of the other operand.
  cudnnTensorDescriptor_t xpk = xd, ypk = yd;
  if (cc.strided) xpk = tensor(dt, cc.x), ypk = tensor(dt, y);
  Buf<T> dyp(yn, 0.8f, 7), xp(xn, 1.0f);
  CK(cudnnConvolutionBackwardData(H, &a, wd, w.p, ypk, dyp.p, cd, CUDNN_CONVOLUTION_BWD_DATA_ALGO_0, nullptr, 0, &b,
                                  xd, out_x.p));
  std::snprintf(tag, sizeof tag, "%s bwd-data a.5 b.25", cc.tag);
  dump(tag, out_x.get());
  CK(cudnnConvolutionBackwardFilter(H, &one, xpk, xp.p, yd, dy.p, cd, CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0, nullptr, 0,
                                    &zero, wd, out_w.p));
  std::snprintf(tag, sizeof tag, "%s bwd-filter", cc.tag);
  dump(tag, out_w.get());
  // Bias gradient: [1, K, 1, ...].
  std::vector<int> bdim(y.size(), 1);
  bdim[1] = y[1];
  cudnnTensorDescriptor_t bd = tensor(dt, bdim);
  Buf<T> db(y[1], 0.5f);
  CK(cudnnConvolutionBackwardBias(H, &a, yd, dy.p, &b, bd, db.p));
  std::snprintf(tag, sizeof tag, "%s bwd-bias a.5 b.25", cc.tag);
  dump(tag, db.get());
  for (auto d : {xd, yd, bd}) cudnnDestroyTensorDescriptor(d);
  if (cc.strided) cudnnDestroyTensorDescriptor(xpk), cudnnDestroyTensorDescriptor(ypk);
  cudnnDestroyFilterDescriptor(wd);
  cudnnDestroyConvolutionDescriptor(cd);
}

static void convolutions() {
  const ConvCase f32[] = {
      {"conv 3x3 pad1", {2, 4, 8, 8}, {6, 4, 3, 3}, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_TENSOR_NCHW, false},
      {"conv 3x2 pad2 s2 d2", {1, 3, 9, 8}, {4, 3, 3, 2}, 2, 2, 2, 1, CUDNN_CROSS_CORRELATION, CUDNN_TENSOR_NCHW, false},
      {"conv flipped g2", {2, 4, 7, 7}, {6, 2, 3, 3}, 1, 1, 1, 2, CUDNN_CONVOLUTION, CUDNN_TENSOR_NCHW, false},
      {"conv depthwise s2", {1, 4, 6, 6}, {4, 1, 3, 3}, 1, 2, 1, 4, CUDNN_CROSS_CORRELATION, CUDNN_TENSOR_NCHW, false},
      {"conv nhwc 3x3", {2, 3, 5, 6}, {4, 3, 3, 3}, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_TENSOR_NHWC, false},
      {"conv strided rows", {2, 3, 6, 5}, {2, 3, 3, 3}, 0, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_TENSOR_NCHW, true},
      {"conv 3-D", {1, 2, 4, 5, 5}, {3, 2, 2, 3, 3}, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_TENSOR_NCHW, false},
  };
  for (const auto& c : f32) conv_case<float>(c, CUDNN_DATA_FLOAT);
  const ConvCase f64 = {"conv double s2", {2, 3, 6, 6}, {4, 3, 3, 3}, 1, 2, 1, 1, CUDNN_CROSS_CORRELATION,
                        CUDNN_TENSOR_NCHW, false};
  conv_case<double>(f64, CUDNN_DATA_DOUBLE);
}

// y = act(alpha1 * conv(x) + alpha2 * z + bias), RELU and IDENTITY.
static void conv_bias_act() {
  cudnnTensorDescriptor_t xd = tensor(CUDNN_DATA_FLOAT, {2, 3, 5, 5}), yd = tensor(CUDNN_DATA_FLOAT, {2, 4, 5, 5}),
                          bd = tensor(CUDNN_DATA_FLOAT, {1, 4, 1, 1});
  cudnnFilterDescriptor_t wd;
  cudnnCreateFilterDescriptor(&wd);
  CK(cudnnSetFilter4dDescriptor(wd, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 4, 3, 3, 3));
  cudnnConvolutionDescriptor_t cd;
  cudnnCreateConvolutionDescriptor(&cd);
  CK(cudnnSetConvolution2dDescriptor(cd, 1, 1, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
  cudnnActivationDescriptor_t ad;
  cudnnCreateActivationDescriptor(&ad);
  Buf<float> x(150, 1.0f), w(108, 0.5f, 3), z(200, 0.7f, 5), bias(4, 0.4f, 9), y(200, 0.0f);
  const float a1 = 1.0f, a2 = 0.5f;
  for (auto m : {CUDNN_ACTIVATION_RELU, CUDNN_ACTIVATION_IDENTITY}) {
    CK(cudnnSetActivationDescriptor(ad, m, CUDNN_NOT_PROPAGATE_NAN, 0.0));
    CK(cudnnConvolutionBiasActivationForward(H, &a1, xd, x.p, wd, w.p, cd,
                                             CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM, nullptr, 0, &a2, yd,
                                             z.p, bd, bias.p, ad, yd, y.p));
    dump(m == CUDNN_ACTIVATION_RELU ? "conv-bias-relu" : "conv-bias-identity", y.get());
  }
  CK(cudnnSetActivationDescriptor(ad, CUDNN_ACTIVATION_SIGMOID, CUDNN_NOT_PROPAGATE_NAN, 0.0));
  std::printf("conv-bias-sigmoid status %d\n",
              (int)cudnnConvolutionBiasActivationForward(H, &a1, xd, x.p, wd, w.p, cd,
                                                         CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM, nullptr, 0,
                                                         &a2, yd, z.p, bd, bias.p, ad, yd, y.p));
}

/* ---- activation, pooling, softmax, LRN ---- */

static void activations() {
  cudnnTensorDescriptor_t d = tensor(CUDNN_DATA_FLOAT, {2, 3, 4, 5});
  cudnnTensorDescriptor_t dn = tensor4(CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, 2, 3, 4, 5);
  cudnnActivationDescriptor_t ad;
  cudnnCreateActivationDescriptor(&ad);
  struct M { const char* tag; cudnnActivationMode_t m; double coef; };
  const M modes[] = {{"relu", CUDNN_ACTIVATION_RELU, 0}, {"sigmoid", CUDNN_ACTIVATION_SIGMOID, 0},
                     {"tanh", CUDNN_ACTIVATION_TANH, 0}, {"clipped relu 0.8", CUDNN_ACTIVATION_CLIPPED_RELU, 0.8},
                     {"elu 0.6", CUDNN_ACTIVATION_ELU, 0.6}, {"swish", CUDNN_ACTIVATION_SWISH, 0}};
  const float two = 2.0f, half = 0.5f, one = 1.0f, zero = 0.0f;
  for (const M& m : modes) {
    CK(cudnnSetActivationDescriptor(ad, m.m, CUDNN_NOT_PROPAGATE_NAN, m.coef));
    CK(cudnnSetActivationDescriptorSwishBeta(ad, 1.25));
    Buf<float> x(120, 1.5f), y(120, 0.0f), dy(120, 1.0f, 4), dx(120, 0.5f, 8);
    CK(cudnnActivationForward(H, ad, &one, d, x.p, &zero, d, y.p));
    CK(cudnnActivationBackward(H, ad, &two, d, y.p, d, dy.p, d, x.p, &half, d, dx.p));
    char tag[64];
    std::snprintf(tag, sizeof tag, "act bwd %s a2 b.5", m.tag);
    dump(tag, dx.get());
    // The same data seen as NHWC: the same element-wise answer.
    Buf<float> dxn(120, 0.0f);
    CK(cudnnActivationBackward(H, ad, &one, dn, y.p, dn, dy.p, dn, x.p, &zero, dn, dxn.p));
    std::snprintf(tag, sizeof tag, "act bwd %s nhwc", m.tag);
    dump(tag, dxn.get());
  }
}

static void pooling() {
  struct P { const char* tag; cudnnPoolingMode_t m; int win, pad, str; };
  const P cases[] = {{"max 2x2 s2", CUDNN_POOLING_MAX, 2, 0, 2},
                     {"max 3x3 p1 s1", CUDNN_POOLING_MAX, 3, 1, 1},
                     {"max-det 3x3 p1 s2", CUDNN_POOLING_MAX_DETERMINISTIC, 3, 1, 2},
                     {"avg incl 3x3 p1 s2", CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING, 3, 1, 2},
                     {"avg excl 3x3 p1 s2", CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING, 3, 1, 2}};
  cudnnPoolingDescriptor_t pd;
  cudnnCreatePoolingDescriptor(&pd);
  const float one = 1.0f, zero = 0.0f, a = 1.5f, b = 0.5f;
  cudnnTensorDescriptor_t xd = tensor(CUDNN_DATA_FLOAT, {2, 3, 7, 6});
  for (const P& p : cases) {
    CK(cudnnSetPooling2dDescriptor(pd, p.m, CUDNN_NOT_PROPAGATE_NAN, p.win, p.win, p.pad, p.pad, p.str, p.str));
    int n, c, h, w;
    CK(cudnnGetPooling2dForwardOutputDim(pd, xd, &n, &c, &h, &w));
    cudnnTensorDescriptor_t yd = tensor(CUDNN_DATA_FLOAT, {n, c, h, w});
    Buf<float> x(252, 2.0f), y(n * c * h * w, 0.0f), dy(n * c * h * w, 1.0f, 5), dx(252, 0.4f, 2);
    CK(cudnnPoolingForward(H, pd, &one, xd, x.p, &zero, yd, y.p));
    CK(cudnnPoolingBackward(H, pd, &a, yd, y.p, yd, dy.p, xd, x.p, &b, xd, dx.p));
    char tag[64];
    std::snprintf(tag, sizeof tag, "pool bwd %s", p.tag);
    dump(tag, dx.get());
    cudnnDestroyTensorDescriptor(yd);
  }
  // 3-D max pooling.
  const int win[3] = {2, 2, 2}, pad[3] = {0, 1, 0}, str[3] = {1, 2, 2};
  CK(cudnnSetPoolingNdDescriptor(pd, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN, 3, win, pad, str));
  cudnnTensorDescriptor_t x3 = tensor(CUDNN_DATA_FLOAT, {1, 2, 3, 4, 5});
  int od[5];
  CK(cudnnGetPoolingNdForwardOutputDim(pd, x3, 5, od));
  cudnnTensorDescriptor_t y3 = tensor(CUDNN_DATA_FLOAT, {od[0], od[1], od[2], od[3], od[4]});
  const int yn = od[0] * od[1] * od[2] * od[3] * od[4];
  Buf<float> x(120, 2.0f), y(yn, 0.0f), dy(yn, 1.0f, 3), dx(120, 0.0f);
  CK(cudnnPoolingForward(H, pd, &one, x3, x.p, &zero, y3, y.p));
  CK(cudnnPoolingBackward(H, pd, &one, y3, y.p, y3, dy.p, x3, x.p, &zero, x3, dx.p));
  dump("pool 3-D max fwd", y.get());
  dump("pool 3-D max bwd", dx.get());
}

static void softmax() {
  cudnnTensorDescriptor_t d = tensor(CUDNN_DATA_FLOAT, {2, 5, 3, 2});
  const float one = 1.0f, zero = 0.0f, a = 2.0f, b = 0.5f;
  for (auto algo : {CUDNN_SOFTMAX_FAST, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_LOG})
    for (auto mode : {CUDNN_SOFTMAX_MODE_INSTANCE, CUDNN_SOFTMAX_MODE_CHANNEL}) {
      Buf<float> x(60, 2.0f), y(60, 0.0f), dy(60, 1.0f, 9), dx(60, 0.3f, 1);
      CK(cudnnSoftmaxForward(H, algo, mode, &one, d, x.p, &zero, d, y.p));
      CK(cudnnSoftmaxBackward(H, algo, mode, &a, d, y.p, d, dy.p, &b, d, dx.p));
      char tag[64];
      std::snprintf(tag, sizeof tag, "softmax bwd algo %d mode %d", (int)algo, (int)mode);
      dump(tag, dx.get());
    }
}

static void lrn() {
  cudnnTensorDescriptor_t d = tensor(CUDNN_DATA_FLOAT, {2, 7, 3, 2});
  cudnnLRNDescriptor_t ld;
  cudnnCreateLRNDescriptor(&ld);
  const float one = 1.0f, zero = 0.0f;
  for (unsigned n : {5u, 4u, 1u}) {
    CK(cudnnSetLRNDescriptor(ld, n, 0.3, 0.75, 1.5));
    Buf<float> x(84, 1.5f), y(84, 0.0f), dy(84, 1.0f, 6), dx(84, 0.0f);
    CK(cudnnLRNCrossChannelForward(H, ld, CUDNN_LRN_CROSS_CHANNEL_DIM1, &one, d, x.p, &zero, d, y.p));
    CK(cudnnLRNCrossChannelBackward(H, ld, CUDNN_LRN_CROSS_CHANNEL_DIM1, &one, d, y.p, d, dy.p, d, x.p, &zero, d,
                                    dx.p));
    char tag[64];
    std::snprintf(tag, sizeof tag, "lrn n=%u fwd", n);
    dump(tag, y.get());
    std::snprintf(tag, sizeof tag, "lrn n=%u bwd", n);
    dump(tag, dx.get());
  }
}

/* ---- tensor arithmetic ---- */

static void reductions() {
  cudnnTensorDescriptor_t a = tensor(CUDNN_DATA_FLOAT, {2, 3, 4, 5});
  const std::vector<std::vector<int>> outs = {{1, 3, 1, 1}, {2, 1, 4, 1}, {1, 1, 1, 1}, {2, 3, 4, 1}};
  cudnnReduceTensorDescriptor_t rd;
  cudnnCreateReduceTensorDescriptor(&rd);
  const float al = 1.5f, be = 0.5f;
  Buf<float> A(120, 2.0f);
  for (int op = CUDNN_REDUCE_TENSOR_ADD; op <= CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS; ++op)
    for (const auto& o : outs) {
      if (op == CUDNN_REDUCE_TENSOR_MUL || op == CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS) {
        if (o[0] * o[1] * o[2] * o[3] < 24) continue;  // products of many terms underflow differently
      }
      const bool idx = op == CUDNN_REDUCE_TENSOR_MIN || op == CUDNN_REDUCE_TENSOR_MAX || op == CUDNN_REDUCE_TENSOR_AMAX;
      CK(cudnnSetReduceTensorDescriptor(rd, (cudnnReduceTensorOp_t)op, CUDNN_DATA_FLOAT, CUDNN_NOT_PROPAGATE_NAN,
                                        idx ? CUDNN_REDUCE_TENSOR_FLATTENED_INDICES : CUDNN_REDUCE_TENSOR_NO_INDICES,
                                        CUDNN_32BIT_INDICES));
      cudnnTensorDescriptor_t c = tensor(CUDNN_DATA_FLOAT, o);
      const int nc = o[0] * o[1] * o[2] * o[3];
      size_t ws = 0, is = 0;
      CK(cudnnGetReductionWorkspaceSize(H, rd, a, c, &ws));
      CK(cudnnGetReductionIndicesSize(H, rd, a, c, &is));
      void* wsp = nullptr;
      if (ws) cudaMalloc(&wsp, ws);
      Buf<uint32_t> ind(nc + 1, 0.0f);
      Buf<float> C(nc, 1.0f, 2);
      CK(cudnnReduceTensor(H, rd, ind.p, is, wsp, ws, &al, a, A.p, &be, c, C.p));
      char tag[64];
      std::snprintf(tag, sizeof tag, "reduce op %d to %dx%dx%dx%d", op, o[0], o[1], o[2], o[3]);
      dump(tag, C.get());
      if (idx) {
        std::printf("  indices size %zu:", is);
        const std::vector<uint32_t> iv = ind.get();
        for (int i = 0; i < nc; ++i) std::printf(" %u", iv[i]);
        std::printf("\n");
      }
      if (wsp) cudaFree(wsp);
      cudnnDestroyTensorDescriptor(c);
    }
  cudnnTensorDescriptor_t same = tensor(CUDNN_DATA_FLOAT, {2, 3, 4, 5});
  Buf<float> C(120, 1.0f);
  CK(cudnnSetReduceTensorDescriptor(rd, CUDNN_REDUCE_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_NOT_PROPAGATE_NAN,
                                    CUDNN_REDUCE_TENSOR_NO_INDICES, CUDNN_32BIT_INDICES));
  std::printf("reduce to the same shape: status %d\n",
              (int)cudnnReduceTensor(H, rd, nullptr, 0, nullptr, 0, &al, a, A.p, &be, same, C.p));
}

static void arithmetic() {
  cudnnTensorDescriptor_t c = tensor4(CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, 2, 3, 4, 5);
  cudnnTensorDescriptor_t bc = tensor(CUDNN_DATA_FLOAT, {1, 3, 1, 1}), bn = tensor(CUDNN_DATA_FLOAT, {2, 1, 4, 5});
  cudnnOpTensorDescriptor_t od;
  cudnnCreateOpTensorDescriptor(&od);
  const float a1 = 1.5f, a2 = -0.5f, b = 0.25f;
  for (int op = CUDNN_OP_TENSOR_ADD; op <= CUDNN_OP_TENSOR_NOT; ++op) {
    CK(cudnnSetOpTensorDescriptor(od, (cudnnOpTensorOp_t)op, CUDNN_DATA_FLOAT, CUDNN_NOT_PROPAGATE_NAN));
    Buf<float> A(120, 1.0f), B(40, 1.0f, 3), C(120, 0.5f, 7);
    if (op == CUDNN_OP_TENSOR_SQRT) {  // a positive A
      std::vector<float> h(120);
      for (int i = 0; i < 120; ++i) h[i] = 0.1f + 0.05f * i;
      cudaMemcpy(A.p, h.data(), 480, cudaMemcpyHostToDevice);
    }
    CK(cudnnOpTensor(H, od, &a1, c, A.p, &a2, bn, B.p, &b, c, C.p));
    char tag[64];
    std::snprintf(tag, sizeof tag, "optensor %d nhwc, B [2,1,4,5]", op);
    dump(tag, C.get());
  }
  Buf<float> A(3, 1.0f), C(120, 0.5f, 7);
  CK(cudnnAddTensor(H, &a1, bc, A.p, &b, c, C.p));
  dump("addtensor bias onto nhwc", C.get());
  // NCHW float -> NHWC double, scaled.
  // Scaling factors are doubles for a double output.
  cudnnTensorDescriptor_t x = tensor(CUDNN_DATA_FLOAT, {2, 3, 4, 5});
  cudnnTensorDescriptor_t yd = tensor4(CUDNN_TENSOR_NHWC, CUDNN_DATA_DOUBLE, 2, 3, 4, 5);
  Buf<float> X(120, 1.0f);
  Buf<double> Y(120, 0.5f, 3);
  const double da = 1.5, db = 0.25;
  CK(cudnnTransformTensor(H, &da, x, X.p, &db, yd, Y.p));
  dump("transform nchw float -> nhwc double", Y.get());
  std::vector<double> yh = Y.get();
  std::printf("  y[0..3] %.6e %.6e %.6e %.6e\n", yh[0], yh[1], yh[2], yh[3]);
}

// Dropout's backward pass from a mask written here: deterministic, so the
// hardware's answer can be compared even though its generator cannot.
static void dropout() {
  size_t states = 0;
  CK(cudnnDropoutGetStatesSize(H, &states));
  cudnnTensorDescriptor_t d = tensor(CUDNN_DATA_FLOAT, {1, 1, 10, 100}), small = tensor(CUDNN_DATA_FLOAT, {1, 1, 1, 7});
  size_t rs = 0, rs7 = 0;
  CK(cudnnDropoutGetReserveSpaceSize(d, &rs));
  CK(cudnnDropoutGetReserveSpaceSize(small, &rs7));
  std::printf("dropout reserve: 1000 elements %zu bytes, 7 elements %zu bytes\n", rs, rs7);
  void* st = nullptr;
  cudaMalloc(&st, states);
  cudnnDropoutDescriptor_t dd;
  cudnnCreateDropoutDescriptor(&dd);
  CK(cudnnSetDropoutDescriptor(dd, H, 0.3f, st, states, 42));
  // Forward on a constant: the kept elements must all be x / (1 - p).
  Buf<float> x(1000, 0.0f), y(1000, 0.0f), dy(1000, 1.0f), dx(1000, 0.0f);
  std::vector<float> ones(1000, 2.0f);
  cudaMemcpy(x.p, ones.data(), 4000, cudaMemcpyHostToDevice);
  Buf<uint8_t> mask(rs, 0.0f);
  CK(cudnnDropoutForward(H, dd, d, x.p, d, y.p, mask.p, rs));
  int other = 0;
  for (float v : y.get()) other += v != 0.0f && v != 2.0f / 0.7f;
  std::printf("dropout forward: %d values neither 0 nor x/(1-p)\n", other);
  std::vector<uint8_t> m(rs);
  for (size_t i = 0; i < rs; ++i) m[i] = (uint8_t)(0x5a ^ (i * 37));
  cudaMemcpy(mask.p, m.data(), rs, cudaMemcpyHostToDevice);
  CK(cudnnDropoutBackward(H, dd, d, dy.p, d, dx.p, mask.p, rs));
  dump("dropout bwd from a known mask", dx.get());
  std::printf("dropout bwd with a short reserve: status %d\n",
              (int)cudnnDropoutBackward(H, dd, d, dy.p, d, dx.p, mask.p, rs - 4));
  float p = 0;
  void* got = nullptr;
  unsigned long long seed = 0;
  CK(cudnnGetDropoutDescriptor(dd, H, &p, &got, &seed));
  std::printf("dropout descriptor p %.6e seed %llu states %s\n", p, seed, got == st ? "kept" : "lost");
  cudaFree(st);
}

static void batchnorm_nhwc() {
  cudnnTensorDescriptor_t x = tensor4(CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, 3, 4, 5, 2), bnd;
  cudnnCreateTensorDescriptor(&bnd);
  const float one = 1.0f, zero = 0.0f;
  for (auto mode : {CUDNN_BATCHNORM_SPATIAL, CUDNN_BATCHNORM_PER_ACTIVATION}) {
    CK(cudnnDeriveBNTensorDescriptor(bnd, x, mode));
    const size_t np = span_of(bnd) / 4;
    Buf<float> X(120, 1.5f), Y(120, 0.0f), dy(120, 1.0f, 4), dx(120, 0.0f), scale(np, 0.5f, 3), bias(np, 0.2f, 5),
        mean(np, 0.0f), inv(np, 0.0f), dscale(np, 0.0f), dbias(np, 0.0f);
    std::vector<float> s(np);
    for (size_t i = 0; i < np; ++i) s[i] = 0.75f + 0.1f * i;
    cudaMemcpy(scale.p, s.data(), np * 4, cudaMemcpyHostToDevice);
    CK(cudnnBatchNormalizationForwardTraining(H, mode, &one, &zero, x, X.p, x, Y.p, bnd, scale.p, bias.p, 0.1, nullptr,
                                              nullptr, 1e-5, mean.p, inv.p));
    CK(cudnnBatchNormalizationBackward(H, mode, &one, &zero, &one, &zero, x, X.p, x, dy.p, x, dx.p, bnd, scale.p,
                                       dscale.p, dbias.p, 1e-5, mean.p, inv.p));
    const char* m = mode == CUDNN_BATCHNORM_SPATIAL ? "spatial" : "per-activation";
    char tag[64];
    std::snprintf(tag, sizeof tag, "bn nhwc %s fwd", m);
    dump(tag, Y.get());
    std::snprintf(tag, sizeof tag, "bn nhwc %s dx", m);
    dump(tag, dx.get());
    std::snprintf(tag, sizeof tag, "bn nhwc %s dscale", m);
    dump(tag, dscale.get());
    std::snprintf(tag, sizeof tag, "bn nhwc %s dbias", m);
    dump(tag, dbias.get());
  }
}

// Batch normalization with a fused add and activation, in the one layout the
// hardware fuses (NHWC, SPATIAL_PERSISTENT): forward and backward through
// the reserve space the library asks for.
static void batchnorm_fused() {
  cudnnTensorDescriptor_t x = tensor4(CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, 2, 4, 3, 3), bnd;
  cudnnCreateTensorDescriptor(&bnd);
  CK(cudnnDeriveBNTensorDescriptor(bnd, x, CUDNN_BATCHNORM_SPATIAL_PERSISTENT));
  cudnnActivationDescriptor_t ad;
  cudnnCreateActivationDescriptor(&ad);
  const float one = 1.0f, zero = 0.0f;
  struct F { const char* tag; cudnnBatchNormOps_t ops; cudnnActivationMode_t act; };
  for (const F& f : {F{"bn+relu", CUDNN_BATCHNORM_OPS_BN_ACTIVATION, CUDNN_ACTIVATION_RELU},
                     F{"bn+add+relu", CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION, CUDNN_ACTIVATION_RELU},
                     F{"bn+swish", CUDNN_BATCHNORM_OPS_BN_ACTIVATION, CUDNN_ACTIVATION_SWISH}}) {
    CK(cudnnSetActivationDescriptor(ad, f.act, CUDNN_NOT_PROPAGATE_NAN, 0.0));
    const bool add = f.ops == CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION;
    size_t ws = 0, bws = 0, rs = 0;
    CK(cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(H, CUDNN_BATCHNORM_SPATIAL_PERSISTENT, f.ops, x,
                                                                add ? x : nullptr, x, bnd, ad, &ws));
    // The fused backward pass takes RELU only.
    const bool backward = f.act == CUDNN_ACTIVATION_RELU;
    if (backward)
      CK(cudnnGetBatchNormalizationBackwardExWorkspaceSize(H, CUDNN_BATCHNORM_SPATIAL_PERSISTENT, f.ops, x, x, x,
                                                           add ? x : nullptr, x, bnd, ad, &bws));
    CK(cudnnGetBatchNormalizationTrainingExReserveSpaceSize(H, CUDNN_BATCHNORM_SPATIAL_PERSISTENT, f.ops, ad, x, &rs));
    Buf<float> X(72, 1.5f), Z(72, 0.7f, 2), Y(72, 0.0f), dy(72, 1.0f, 4), dx(72, 0.0f), dz(72, 0.0f),
        scale(4, 0.5f, 3), bias(4, 0.2f, 5), mean(4, 0.0f), inv(4, 0.0f), dscale(4, 0.0f), dbias(4, 0.0f),
        work((ws > bws ? ws : bws) / 4 + 1, 0.0f), reserve(rs / 4 + 1, 0.0f);
    std::vector<float> s(4);
    for (int i = 0; i < 4; ++i) s[i] = 0.75f + 0.1f * i;
    cudaMemcpy(scale.p, s.data(), 16, cudaMemcpyHostToDevice);
    CK(cudnnBatchNormalizationForwardTrainingEx(H, CUDNN_BATCHNORM_SPATIAL_PERSISTENT, f.ops, &one, &zero, x, X.p,
                                                add ? x : nullptr, add ? Z.p : nullptr, x, Y.p, bnd, scale.p, bias.p,
                                                0.1, nullptr, nullptr, 1e-5, mean.p, inv.p, ad, work.p, ws,
                                                reserve.p, rs));
    char tag[64];
    std::snprintf(tag, sizeof tag, "%s fwd", f.tag);
    dump(tag, Y.get());
    if (!backward) {
      std::printf("%s backward: status %d\n", f.tag,
                  (int)cudnnBatchNormalizationBackwardEx(H, CUDNN_BATCHNORM_SPATIAL_PERSISTENT, f.ops, &one, &zero,
                                                         &one, &zero, x, X.p, x, Y.p, x, dy.p, nullptr, nullptr, x,
                                                         dx.p, bnd, scale.p, bias.p, dscale.p, dbias.p, 1e-5, mean.p,
                                                         inv.p, ad, work.p, bws, reserve.p, rs));
      continue;
    }
    CK(cudnnBatchNormalizationBackwardEx(H, CUDNN_BATCHNORM_SPATIAL_PERSISTENT, f.ops, &one, &zero, &one, &zero, x,
                                         X.p, x, Y.p, x, dy.p, add ? x : nullptr, add ? dz.p : nullptr, x, dx.p, bnd,
                                         scale.p, bias.p, dscale.p, dbias.p, 1e-5, mean.p, inv.p, ad, work.p, bws,
                                         reserve.p, rs));
    std::snprintf(tag, sizeof tag, "%s dx", f.tag);
    dump(tag, dx.get());
    if (add) {
      std::snprintf(tag, sizeof tag, "%s dz", f.tag);
      dump(tag, dz.get());
    }
    std::snprintf(tag, sizeof tag, "%s dscale", f.tag);
    dump(tag, dscale.get());
    std::snprintf(tag, sizeof tag, "%s dbias", f.tag);
    dump(tag, dbias.get());
  }
}

// The spatial transformer: an affine grid per image, bilinear sampling
// (some of it off the image), and both backward passes.
static void spatial_transformer() {
  cudnnSpatialTransformerDescriptor_t st;
  CK(cudnnCreateSpatialTransformerDescriptor(&st));
  const int dims[4] = {2, 2, 4, 5};
  CK(cudnnSetSpatialTransformerNdDescriptor(st, CUDNN_SAMPLER_BILINEAR, CUDNN_DATA_FLOAT, 4, dims));
  const float th[12] = {0.83f, 0.21f, 0.07f, -0.17f, 0.91f, -0.05f, 1.13f, -0.31f, 0.12f, 0.27f, 0.77f, 0.19f};
  Buf<float> theta(12, 0.0f), grid(80, 0.0f), x(168, 1.0f), y(80, 0.0f), dy(80, 1.0f, 3), dx(168, 0.2f, 5),
      dgrid(80, 0.0f), dtheta(12, 0.0f);
  cudaMemcpy(theta.p, th, sizeof th, cudaMemcpyHostToDevice);
  cudnnTensorDescriptor_t xd = tensor(CUDNN_DATA_FLOAT, {2, 2, 6, 7}), yd = tensor(CUDNN_DATA_FLOAT, {2, 2, 4, 5});
  const float one = 1.0f, zero = 0.0f, half = 0.5f;
  CK(cudnnSpatialTfGridGeneratorForward(H, st, theta.p, grid.p));
  dump("stn grid", grid.get());
  CK(cudnnSpatialTfSamplerForward(H, st, &one, xd, x.p, grid.p, &zero, yd, y.p));
  dump("stn sampler fwd", y.get());
  CK(cudnnSpatialTfSamplerBackward(H, st, &one, xd, x.p, &half, xd, dx.p, &one, yd, dy.p, grid.p, &zero, dgrid.p));
  dump("stn sampler dx b.5", dx.get());
  dump("stn sampler dgrid", dgrid.get());
  CK(cudnnSpatialTfGridGeneratorBackward(H, st, dgrid.p, dtheta.p));
  dump("stn dtheta", dtheta.get());
  cudnnDestroySpatialTransformerDescriptor(st);
}

// CTC loss: costs and gradients from activations (SOFTMAX) and from
// probabilities (NONE), labels in host and in device memory, and a label too
// long for its input, whose gradient is zeroed or left alone.
static void ctc_loss() {
  const int T = 6, N = 3, A = 5;
  cudnnTensorDescriptor_t pd = tensor(CUDNN_DATA_FLOAT, {T, N, A});
  cudnnCTCLossDescriptor_t cd;
  CK(cudnnCreateCTCLossDescriptor(&cd));
  const int labels[] = {1, 2, 2, 4, 3, 1, 4}, llen[N] = {3, 1, 3}, ilen[N] = {6, 5, 4};
  Buf<float> x(T * N * A, 1.5f), costs(N, 0.0f);
  // Probabilities for NONE mode: a softmax of x over A.
  std::vector<float> hx = x.get(), probs(hx.size());
  for (int r = 0; r < T * N; ++r) {
    float sum = 0;
    for (int a = 0; a < A; ++a) sum += std::exp(hx[r * A + a]);
    for (int a = 0; a < A; ++a) probs[r * A + a] = std::exp(hx[r * A + a]) / sum;
  }
  Buf<float> p(T * N * A, 0.0f);
  cudaMemcpy(p.p, probs.data(), probs.size() * 4, cudaMemcpyHostToDevice);
  for (auto norm : {CUDNN_LOSS_NORMALIZATION_SOFTMAX, CUDNN_LOSS_NORMALIZATION_NONE}) {
    CK(cudnnSetCTCLossDescriptor_v9(cd, CUDNN_DATA_FLOAT, norm, CUDNN_CTC_ZERO_OOB_GRADIENTS, 8));
    size_t ws = 0;
    CK(cudnnGetCTCLossWorkspaceSize(H, pd, pd, labels, llen, ilen, CUDNN_CTC_LOSS_ALGO_DETERMINISTIC, cd, &ws));
    Buf<char> work(ws + 16, 0.0f);
    Buf<float> grad(T * N * A, 0.25f, 3);  // steps past a sequence keep these
    CK(cudnnCTCLoss(H, pd, norm == CUDNN_LOSS_NORMALIZATION_SOFTMAX ? x.p : p.p, labels, llen, ilen, costs.p, pd,
                    grad.p, CUDNN_CTC_LOSS_ALGO_DETERMINISTIC, cd, work.p, ws));
    char tag[64];
    std::snprintf(tag, sizeof tag, "ctc norm %d costs", (int)norm);
    dump(tag, costs.get());
    std::snprintf(tag, sizeof tag, "ctc norm %d gradients", (int)norm);
    dump(tag, grad.get());
  }
  // Device-memory labels, the other algorithm.
  CK(cudnnSetCTCLossDescriptor_v9(cd, CUDNN_DATA_FLOAT, CUDNN_LOSS_NORMALIZATION_SOFTMAX, CUDNN_CTC_ZERO_OOB_GRADIENTS, 8));
  int *dl, *dll, *dil;
  cudaMalloc(&dl, sizeof labels), cudaMalloc(&dll, sizeof llen), cudaMalloc(&dil, sizeof ilen);
  cudaMemcpy(dl, labels, sizeof labels, cudaMemcpyHostToDevice);
  cudaMemcpy(dll, llen, sizeof llen, cudaMemcpyHostToDevice);
  cudaMemcpy(dil, ilen, sizeof ilen, cudaMemcpyHostToDevice);
  size_t ws = 0;
  CK(cudnnGetCTCLossWorkspaceSize_v8(H, CUDNN_CTC_LOSS_ALGO_NON_DETERMINISTIC, cd, pd, pd, &ws));
  Buf<char> work(ws + 16, 0.0f);
  Buf<float> grad(T * N * A, 0.25f, 3);
  CK(cudnnCTCLoss_v8(H, CUDNN_CTC_LOSS_ALGO_NON_DETERMINISTIC, cd, pd, x.p, dl, dll, dil, costs.p, pd, grad.p, ws,
                     work.p));
  dump("ctc v8 costs", costs.get());
  dump("ctc v8 gradients", grad.get());
  // A label of 3 with a repeat needs 4 steps; the first sequence has 3.
  const int bad_labels[] = {2, 2, 3, 4, 1}, bad_llen[N] = {3, 1, 1}, bad_ilen[N] = {3, 5, 4};
  for (auto gm : {CUDNN_CTC_ZERO_OOB_GRADIENTS, CUDNN_CTC_SKIP_OOB_GRADIENTS}) {
    CK(cudnnSetCTCLossDescriptor_v9(cd, CUDNN_DATA_FLOAT, CUDNN_LOSS_NORMALIZATION_SOFTMAX, gm, 8));
    size_t ws2 = 0;
    CK(cudnnGetCTCLossWorkspaceSize(H, pd, pd, bad_labels, bad_llen, bad_ilen, CUDNN_CTC_LOSS_ALGO_DETERMINISTIC, cd,
                                    &ws2));
    Buf<char> work2(ws2 + 16, 0.0f);
    Buf<float> g2(T * N * A, 0.25f, 3);
    CK(cudnnCTCLoss(H, pd, x.p, bad_labels, bad_llen, bad_ilen, costs.p, pd, g2.p, CUDNN_CTC_LOSS_ALGO_DETERMINISTIC,
                    cd, work2.p, ws2));
    char tag[64];
    std::snprintf(tag, sizeof tag, "ctc infeasible grad mode %d costs", (int)gm);
    dump(tag, costs.get());
    std::snprintf(tag, sizeof tag, "ctc infeasible grad mode %d gradients", (int)gm);
    dump(tag, g2.get());
  }
  cudaFree(dl), cudaFree(dll), cudaFree(dil);
  cudnnDestroyCTCLossDescriptor(cd);
}

// The cuDNN 8 normalization API: per-channel training forward and backward,
// per-activation inference, and the one group it allows.
static void normalization_api() {
  cudnnTensorDescriptor_t x = tensor(CUDNN_DATA_FLOAT, {3, 4, 5, 2}), sb, mv;
  cudnnCreateTensorDescriptor(&sb);
  cudnnCreateTensorDescriptor(&mv);
  const float one = 1.0f, zero = 0.0f;
  CK(cudnnDeriveNormTensorDescriptor(sb, mv, x, CUDNN_NORM_PER_CHANNEL, 1));
  Buf<float> X(120, 1.5f), Y(120, 0.0f), dy(120, 1.0f, 4), dx(120, 0.0f), scale(4, 0.5f, 3), bias(4, 0.2f, 5),
      rmean(4, 0.1f, 6), rvar(4, 0.0f), mean(4, 0.0f), inv(4, 0.0f), dscale(4, 0.0f), dbias(4, 0.0f);
  std::vector<float> ones(4, 1.0f);
  cudaMemcpy(rvar.p, ones.data(), 16, cudaMemcpyHostToDevice);
  size_t ws = 0, rs = 0;
  CK(cudnnGetNormalizationForwardTrainingWorkspaceSize(H, CUDNN_NORM_PER_CHANNEL, CUDNN_NORM_OPS_NORM,
                                                       CUDNN_NORM_ALGO_STANDARD, x, nullptr, x, sb, nullptr, mv, &ws, 1));
  CK(cudnnGetNormalizationTrainingReserveSpaceSize(H, CUDNN_NORM_PER_CHANNEL, CUDNN_NORM_OPS_NORM,
                                                   CUDNN_NORM_ALGO_STANDARD, nullptr, x, &rs, 1));
  Buf<char> work(ws + 16, 0.0f), reserve(rs + 16, 0.0f);
  CK(cudnnNormalizationForwardTraining(H, CUDNN_NORM_PER_CHANNEL, CUDNN_NORM_OPS_NORM, CUDNN_NORM_ALGO_STANDARD, &one,
                                       &zero, x, X.p, sb, scale.p, bias.p, 0.25, mv, rmean.p, rvar.p, 1e-5, mean.p,
                                       inv.p, nullptr, nullptr, nullptr, x, Y.p, work.p, ws, reserve.p, rs, 1));
  dump("norm api training fwd", Y.get());
  dump("norm api running variance", rvar.get());
  size_t bws = 0;
  CK(cudnnGetNormalizationBackwardWorkspaceSize(H, CUDNN_NORM_PER_CHANNEL, CUDNN_NORM_OPS_NORM,
                                                CUDNN_NORM_ALGO_STANDARD, x, x, x, nullptr, x, sb, nullptr, mv, &bws,
                                                1));
  Buf<char> bwork(bws + 16, 0.0f);
  CK(cudnnNormalizationBackward(H, CUDNN_NORM_PER_CHANNEL, CUDNN_NORM_OPS_NORM, CUDNN_NORM_ALGO_STANDARD, &one, &zero,
                                &one, &zero, x, X.p, x, Y.p, x, dy.p, nullptr, nullptr, x, dx.p, sb, scale.p, bias.p,
                                dscale.p, dbias.p, 1e-5, mv, mean.p, inv.p, nullptr, bwork.p, bws, reserve.p, rs, 1));
  dump("norm api backward dx", dx.get());
  dump("norm api backward dscale", dscale.get());
  dump("norm api backward dbias", dbias.get());
  // Per activation, inference from given statistics.
  CK(cudnnDeriveNormTensorDescriptor(sb, mv, x, CUDNN_NORM_PER_ACTIVATION, 1));
  Buf<float> s2(40, 0.5f, 7), b2(40, 0.2f, 8), m2(40, 0.3f, 9), v2(40, 0.0f);
  std::vector<float> var(40);
  for (int i = 0; i < 40; ++i) var[i] = 0.5f + 0.05f * i;
  cudaMemcpy(v2.p, var.data(), 160, cudaMemcpyHostToDevice);
  CK(cudnnNormalizationForwardInference(H, CUDNN_NORM_PER_ACTIVATION, CUDNN_NORM_OPS_NORM, CUDNN_NORM_ALGO_STANDARD,
                                        &one, &zero, x, X.p, sb, s2.p, b2.p, mv, m2.p, v2.p, nullptr, nullptr, nullptr,
                                        x, Y.p, 1e-5, 1));
  dump("norm api inference per activation", Y.get());
  std::printf("norm api with 2 groups: status %d\n",
              (int)cudnnDeriveNormTensorDescriptor(sb, mv, x, CUDNN_NORM_PER_CHANNEL, 2));
}

int main() {
  if (cudnnCreate(&H) != CUDNN_STATUS_SUCCESS) {
    std::printf("cudnnCreate failed\n");
    return 1;
  }
  convolutions();
  conv_bias_act();
  activations();
  pooling();
  softmax();
  lrn();
  reductions();
  arithmetic();
  dropout();
  batchnorm_nhwc();
  batchnorm_fused();
  spatial_transformer();
  ctc_loss();
  normalization_api();
  cudnnDestroy(H);
  return 0;
}
