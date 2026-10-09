// cuDNN's classic API inside a captured CUDA graph (see graph_capture_common.h):
// convolution, activation, pooling, softmax, normalization, tensor arithmetic,
// the spatial transformer, CTC, dropout, RNNs and multi-head attention. Each is
// captured, its descriptors destroyed, and launched with new inputs.
// run_graph_capture_dnn.sh --card runs the same program on NVIDIA's cuDNN 9.
#include <cudnn.h>

#include <algorithm>

#include "graph_capture_common.h"

using namespace gc;

/* ---- descriptors that live exactly as long as the call that builds them ---------- */

#define OK(x) do { cudnnStatus_t s_ = (x); if (s_ != CUDNN_STATUS_SUCCESS) { \
  std::printf("     %s -> %d (%s)\n", #x, (int)s_, cudnnGetErrorString(s_)); ok = false; } } while (0)

struct Tensor {
  cudnnTensorDescriptor_t d = nullptr;
  Tensor(std::vector<int> dims, cudnnDataType_t t = CUDNN_DATA_FLOAT, cudnnTensorFormat_t f = CUDNN_TENSOR_NCHW) {
    cudnnCreateTensorDescriptor(&d);
    if (dims.size() == 4) {
      cudnnSetTensor4dDescriptor(d, f, t, dims[0], dims[1], dims[2], dims[3]);
    } else {
      std::vector<int> strides(dims.size(), 1);
      for (size_t i = dims.size() - 1; i-- > 0;) strides[i] = strides[i + 1] * dims[i + 1];
      cudnnSetTensorNdDescriptor(d, t, static_cast<int>(dims.size()), dims.data(), strides.data());
    }
  }
  ~Tensor() { cudnnDestroyTensorDescriptor(d); }
  operator cudnnTensorDescriptor_t() const { return d; }
};
struct Filter {
  cudnnFilterDescriptor_t d = nullptr;
  Filter(std::vector<int> dims, cudnnDataType_t t = CUDNN_DATA_FLOAT) {
    cudnnCreateFilterDescriptor(&d);
    cudnnSetFilterNdDescriptor(d, t, CUDNN_TENSOR_NCHW, static_cast<int>(dims.size()), dims.data());
  }
  ~Filter() { cudnnDestroyFilterDescriptor(d); }
  operator cudnnFilterDescriptor_t() const { return d; }
};
struct Conv {
  cudnnConvolutionDescriptor_t d = nullptr;
  Conv(int pad = 1, int stride = 1, cudnnDataType_t t = CUDNN_DATA_FLOAT) {
    cudnnCreateConvolutionDescriptor(&d);
    cudnnSetConvolution2dDescriptor(d, pad, pad, stride, stride, 1, 1, CUDNN_CROSS_CORRELATION, t);
  }
  ~Conv() { cudnnDestroyConvolutionDescriptor(d); }
  operator cudnnConvolutionDescriptor_t() const { return d; }
};
struct Act {
  cudnnActivationDescriptor_t d = nullptr;
  Act(cudnnActivationMode_t m = CUDNN_ACTIVATION_RELU, double coef = 0.0) {
    cudnnCreateActivationDescriptor(&d);
    cudnnSetActivationDescriptor(d, m, CUDNN_PROPAGATE_NAN, coef);
  }
  ~Act() { cudnnDestroyActivationDescriptor(d); }
  operator cudnnActivationDescriptor_t() const { return d; }
};

static cudnnHandle_t make_handle(cudaStream_t st) {
  cudnnHandle_t h = nullptr;
  cudnnCreate(&h);
  cudnnSetStream(h, st);
  return h;
}

/* ---- the cases ----------------------------------------------------------------------- */

static void convolution(Runner& r, cudnnHandle_t h) {
  const int N = 2, C = 3, H = 6, W = 6, K = 4;
  float *x = r.alloc<float>(N * C * H * W), *w = r.alloc<float>(K * C * 3 * 3), *y = r.alloc<float>(N * K * H * W);
  float *dy = r.alloc<float>(N * K * H * W), *dx = r.alloc<float>(N * C * H * W), *dw = r.alloc<float>(K * C * 9);
  float *db = r.alloc<float>(K);
  // Algorithms and workspaces are chosen once, before any capture (planning allocates).
  cudnnConvolutionFwdAlgo_t fa = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
  cudnnConvolutionBwdDataAlgo_t da = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
  cudnnConvolutionBwdFilterAlgo_t wa = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
  size_t fws = 0, dws = 0, wws = 0, pws = 0;
  {
    Tensor xd({N, C, H, W}), yd({N, K, H, W});
    Filter wd({K, C, 3, 3});
    Conv cd;
    cudnnConvolutionFwdAlgoPerf_t fp[8];
    cudnnConvolutionBwdDataAlgoPerf_t dp[8];
    cudnnConvolutionBwdFilterAlgoPerf_t wp[8];
    int n = 0;
    if (cudnnGetConvolutionForwardAlgorithm_v7(h, xd, wd, cd, yd, 8, &n, fp) == CUDNN_STATUS_SUCCESS && n > 0) fa = fp[0].algo;
    if (cudnnGetConvolutionBackwardDataAlgorithm_v7(h, wd, yd, cd, xd, 8, &n, dp) == CUDNN_STATUS_SUCCESS && n > 0) da = dp[0].algo;
    if (cudnnGetConvolutionBackwardFilterAlgorithm_v7(h, xd, yd, cd, wd, 8, &n, wp) == CUDNN_STATUS_SUCCESS && n > 0) wa = wp[0].algo;
    cudnnGetConvolutionForwardWorkspaceSize(h, xd, wd, cd, yd, fa, &fws);
    cudnnGetConvolutionBackwardDataWorkspaceSize(h, wd, yd, cd, xd, da, &dws);
    cudnnGetConvolutionBackwardFilterWorkspaceSize(h, xd, yd, cd, wd, wa, &wws);
    cudnnGetConvolutionForwardWorkspaceSize(h, xd, wd, cd, yd, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM, &pws);
  }
  void* work = r.alloc<char>(std::max(std::max(fws, pws), std::max(dws, wws)) + 16);
  const std::vector<Out> outs_f{{y, (size_t)N * K * H * W, Dt::F32}};
  r.run("cudnnConvolutionForward", [&] {
    bool ok = true;
    fill(r.st, r.counter, x, N * C * H * W);
    fill(r.st, r.counter, w, K * C * 9, 0.125f, 0, 1);
    float one = 1.0f, zero = 0.0f;
    Tensor xd({N, C, H, W}), yd({N, K, H, W});
    Filter wd({K, C, 3, 3});
    Conv cd;
    OK(cudnnConvolutionForward(h, &one, xd, x, wd, w, cd, fa, work, fws, &zero, yd, y));
    return ok;
  }, outs_f);
  r.run("cudnnConvolutionBackwardData", [&] {
    bool ok = true;
    fill(r.st, r.counter, dy, N * K * H * W);
    fill(r.st, r.counter, w, K * C * 9, 0.125f, 0, 1);
    float one = 1.0f, zero = 0.0f;
    Tensor dxd({N, C, H, W}), dyd({N, K, H, W});
    Filter wd({K, C, 3, 3});
    Conv cd;
    OK(cudnnConvolutionBackwardData(h, &one, wd, w, dyd, dy, cd, da, work, dws, &zero, dxd, dx));
    return ok;
  }, {{dx, (size_t)N * C * H * W, Dt::F32}}, 1e-4);
  r.run("cudnnConvolutionBackwardFilter", [&] {
    bool ok = true;
    fill(r.st, r.counter, x, N * C * H * W);
    fill(r.st, r.counter, dy, N * K * H * W, 0.25f, 0, 2);
    float one = 1.0f, zero = 0.0f;
    Tensor xd({N, C, H, W}), dyd({N, K, H, W});
    Filter dwd({K, C, 3, 3});
    Conv cd;
    OK(cudnnConvolutionBackwardFilter(h, &one, xd, x, dyd, dy, cd, wa, work, wws, &zero, dwd, dw));
    return ok;
  }, {{dw, (size_t)K * C * 9, Dt::F32}}, 1e-4);
  r.run("cudnnConvolutionBackwardBias", [&] {
    bool ok = true;
    fill(r.st, r.counter, dy, N * K * H * W);
    float one = 1.0f, zero = 0.0f;
    Tensor dyd({N, K, H, W}), dbd({1, K, 1, 1});
    OK(cudnnConvolutionBackwardBias(h, &one, dyd, dy, &zero, dbd, db));
    return ok;
  }, {{db, (size_t)K, Dt::F32}}, 1e-4);
  float *z = r.alloc<float>(N * K * H * W), *bias = r.alloc<float>(K), *yb = r.alloc<float>(N * K * H * W);
  r.run("cudnnConvolutionBiasActivationForward", [&] {
    bool ok = true;
    fill(r.st, r.counter, x, N * C * H * W);
    fill(r.st, r.counter, w, K * C * 9, 0.125f, 0, 1);
    fill(r.st, r.counter, z, N * K * H * W, 0.25f, 0, 3);
    fill(r.st, r.counter, bias, K, 0.5f, 0, 4);
    float one = 1.0f, a2 = 1.0f;
    Tensor xd({N, C, H, W}), yd({N, K, H, W}), bd({1, K, 1, 1});
    Filter wd({K, C, 3, 3});
    Conv cd;
    Act ad(CUDNN_ACTIVATION_RELU);
    OK(cudnnConvolutionBiasActivationForward(h, &one, xd, x, wd, w, cd, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
                                             work, pws, &a2, yd, z, bd, bias, ad, yd, yb));
    return ok;
  }, {{yb, (size_t)N * K * H * W, Dt::F32}}, 1e-4);
}

static void pointwise(Runner& r, cudnnHandle_t h) {
  const int N = 2, C = 3, H = 6, W = 6, n = N * C * H * W;
  float *x = r.alloc<float>(n), *y = r.alloc<float>(n), *dy = r.alloc<float>(n), *dx = r.alloc<float>(n);
  r.run("cudnnActivationForward", [&] {
    bool ok = true;
    fill(r.st, r.counter, x, n);
    float a = 1.0f, b = 0.0f;
    Tensor d({N, C, H, W});
    Act ad(CUDNN_ACTIVATION_ELU, 1.0);
    OK(cudnnActivationForward(h, ad, &a, d, x, &b, d, y));
    return ok;
  }, {{y, (size_t)n, Dt::F32}});
  r.run("cudnnActivationBackward", [&] {
    bool ok = true;
    fill(r.st, r.counter, x, n);
    fill(r.st, r.counter, dy, n, 0.25f, 0, 1);
    float a = 1.0f, b = 0.0f;
    Tensor d({N, C, H, W});
    Act ad(CUDNN_ACTIVATION_SIGMOID);
    OK(cudnnActivationForward(h, ad, &a, d, x, &b, d, y));
    OK(cudnnActivationBackward(h, ad, &a, d, y, d, dy, d, x, &b, d, dx));
    return ok;
  }, {{y, (size_t)n, Dt::F32}, {dx, (size_t)n, Dt::F32}});

  // Pooling 2x2, stride 2.
  float *py = r.alloc<float>(N * C * 3 * 3), *pdx = r.alloc<float>(n), *pdy = r.alloc<float>(N * C * 9);
  struct Pool {
    cudnnPoolingDescriptor_t d = nullptr;
    Pool(cudnnPoolingMode_t m) {
      cudnnCreatePoolingDescriptor(&d);
      cudnnSetPooling2dDescriptor(d, m, CUDNN_PROPAGATE_NAN, 2, 2, 0, 0, 2, 2);
    }
    ~Pool() { cudnnDestroyPoolingDescriptor(d); }
    operator cudnnPoolingDescriptor_t() const { return d; }
  };
  r.run("cudnnPoolingForward", [&] {
    bool ok = true;
    fill(r.st, r.counter, x, n);
    float a = 1.0f, b = 0.0f;
    Tensor xd({N, C, H, W}), yd({N, C, 3, 3});
    Pool pd(CUDNN_POOLING_MAX);
    OK(cudnnPoolingForward(h, pd, &a, xd, x, &b, yd, py));
    return ok;
  }, {{py, (size_t)N * C * 9, Dt::F32}});
  r.run("cudnnPoolingBackward", [&] {
    bool ok = true;
    fill(r.st, r.counter, x, n);
    fill(r.st, r.counter, pdy, N * C * 9, 0.25f, 0, 1);
    float a = 1.0f, b = 0.0f;
    Tensor xd({N, C, H, W}), yd({N, C, 3, 3});
    Pool pd(CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING);
    OK(cudnnPoolingForward(h, pd, &a, xd, x, &b, yd, py));
    OK(cudnnPoolingBackward(h, pd, &a, yd, py, yd, pdy, xd, x, &b, xd, pdx));
    return ok;
  }, {{pdx, (size_t)n, Dt::F32}});
  r.run("cudnnSoftmaxForward + Backward", [&] {
    bool ok = true;
    fill(r.st, r.counter, x, n);
    fill(r.st, r.counter, dy, n, 0.25f, 0, 1);
    float a = 1.0f, b = 0.0f;
    Tensor d({N, C, H, W});
    OK(cudnnSoftmaxForward(h, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, &a, d, x, &b, d, y));
    OK(cudnnSoftmaxBackward(h, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, &a, d, y, d, dy, &b, d, dx));
    return ok;
  }, {{y, (size_t)n, Dt::F32}, {dx, (size_t)n, Dt::F32}}, 1e-4);

  struct Lrn {
    cudnnLRNDescriptor_t d = nullptr;
    Lrn() {
      cudnnCreateLRNDescriptor(&d);
      cudnnSetLRNDescriptor(d, 3, 1e-4, 0.75, 2.0);
    }
    ~Lrn() { cudnnDestroyLRNDescriptor(d); }
    operator cudnnLRNDescriptor_t() const { return d; }
  };
  r.run("cudnnLRNCrossChannelForward + Backward", [&] {
    bool ok = true;
    fill(r.st, r.counter, x, n);
    fill(r.st, r.counter, dy, n, 0.25f, 0, 1);
    float a = 1.0f, b = 0.0f;
    Tensor d({N, C, H, W});
    Lrn ld;
    OK(cudnnLRNCrossChannelForward(h, ld, CUDNN_LRN_CROSS_CHANNEL_DIM1, &a, d, x, &b, d, y));
    OK(cudnnLRNCrossChannelBackward(h, ld, CUDNN_LRN_CROSS_CHANNEL_DIM1, &a, d, y, d, dy, d, x, &b, d, dx));
    return ok;
  }, {{y, (size_t)n, Dt::F32}, {dx, (size_t)n, Dt::F32}}, 1e-4);

  float *t1 = r.alloc<float>(n), *t2 = r.alloc<float>(n), *means = r.alloc<float>(n);
  r.run("cudnnDivisiveNormalizationForward", [&] {
    bool ok = true;
    fill(r.st, r.counter, x, n);
    float a = 1.0f, b = 0.0f;
    Tensor d({N, C, H, W});
    Lrn ld;
    OK(cudnnDivisiveNormalizationForward(h, ld, CUDNN_DIVNORM_PRECOMPUTED_MEANS, &a, d, x, nullptr, t1, t2, &b, d, y));
    return ok;
  }, {{y, (size_t)n, Dt::F32}}, 1e-4);
  (void)means;
}

static void tensor_ops(Runner& r, cudnnHandle_t h) {
  const int N = 2, C = 3, H = 4, W = 5, n = N * C * H * W;
  float *a = r.alloc<float>(n), *b = r.alloc<float>(C), *c = r.alloc<float>(n), *t = r.alloc<float>(n);
  r.run("cudnnAddTensor", [&] {
    bool ok = true;
    fill(r.st, r.counter, a, n);
    fill(r.st, r.counter, b, C, 0.5f, 0, 1);
    fill(r.st, r.counter, c, n, 0.25f, 0, 2);
    float one = 1.0f, beta = 0.5f;
    Tensor bd({1, C, 1, 1}), cd({N, C, H, W});
    OK(cudnnAddTensor(h, &one, bd, b, &beta, cd, c));
    return ok;
  }, {{c, (size_t)n, Dt::F32}});
  struct OpT {
    cudnnOpTensorDescriptor_t d = nullptr;
    OpT(cudnnOpTensorOp_t op) {
      cudnnCreateOpTensorDescriptor(&d);
      cudnnSetOpTensorDescriptor(d, op, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN);
    }
    ~OpT() { cudnnDestroyOpTensorDescriptor(d); }
    operator cudnnOpTensorDescriptor_t() const { return d; }
  };
  r.run("cudnnOpTensor", [&] {
    bool ok = true;
    fill(r.st, r.counter, a, n);
    fill(r.st, r.counter, b, C, 0.5f, 0, 1);
    float a1 = 1.0f, a2 = 2.0f, beta = 0.0f;
    Tensor ad({N, C, H, W}), bd({1, C, 1, 1});
    OpT od(CUDNN_OP_TENSOR_MUL);
    OK(cudnnOpTensor(h, od, &a1, ad, a, &a2, bd, b, &beta, ad, c));
    return ok;
  }, {{c, (size_t)n, Dt::F32}});
  struct Red {
    cudnnReduceTensorDescriptor_t d = nullptr;
    Red() {
      cudnnCreateReduceTensorDescriptor(&d);
      cudnnSetReduceTensorDescriptor(d, CUDNN_REDUCE_TENSOR_ADD, CUDNN_DATA_FLOAT, CUDNN_PROPAGATE_NAN,
                                     CUDNN_REDUCE_TENSOR_NO_INDICES, CUDNN_32BIT_INDICES);
    }
    ~Red() { cudnnDestroyReduceTensorDescriptor(d); }
    operator cudnnReduceTensorDescriptor_t() const { return d; }
  };
  size_t rws = 0;
  {
    Tensor ad({N, C, H, W}), bd({1, C, 1, 1});
    Red rd;
    cudnnGetReductionWorkspaceSize(h, rd, ad, bd, &rws);
  }
  void* rwork = r.alloc<char>(rws + 16);
  r.run("cudnnReduceTensor", [&] {
    bool ok = true;
    fill(r.st, r.counter, a, n);
    float one = 1.0f, zero = 0.0f;
    Tensor ad({N, C, H, W}), bd({1, C, 1, 1});
    Red rd;
    OK(cudnnReduceTensor(h, rd, nullptr, 0, rwork, rws, &one, ad, a, &zero, bd, b));
    return ok;
  }, {{b, (size_t)C, Dt::F32}}, 1e-4);
  r.run("cudnnTransformTensor (NCHW to NHWC)", [&] {
    bool ok = true;
    fill(r.st, r.counter, a, n);
    float one = 1.0f, zero = 0.0f;
    Tensor ad({N, C, H, W}), td({N, C, H, W}, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NHWC);
    OK(cudnnTransformTensor(h, &one, ad, a, &zero, td, t));
    return ok;
  }, {{t, (size_t)n, Dt::F32}});
  r.run("cudnnSetTensor + cudnnScaleTensor", [&] {
    bool ok = true;
    fill(r.st, r.counter, a, n);
    float v = 0.75f, s = 2.0f;
    Tensor ad({N, C, H, W});
    OK(cudnnScaleTensor(h, ad, a, &s));
    OK(cudnnSetTensor(h, ad, t, &v));
    return ok;
  }, {{a, (size_t)n, Dt::F32}, {t, (size_t)n, Dt::F32}});
}

static void normalization(Runner& r, cudnnHandle_t h) {
  const int N = 4, C = 8, H = 3, W = 3, n = N * C * H * W;
  float *x = r.alloc<float>(n), *y = r.alloc<float>(n), *dy = r.alloc<float>(n), *dx = r.alloc<float>(n);
  float *scale = r.alloc<float>(C), *bias = r.alloc<float>(C), *rmean = r.alloc<float>(C), *rvar = r.alloc<float>(C);
  float *smean = r.alloc<float>(C), *sinv = r.alloc<float>(C), *dscale = r.alloc<float>(C), *dbias = r.alloc<float>(C);
  auto init = [&] {
    fill(r.st, r.counter, x, n);
    fill(r.st, r.counter, dy, n, 0.25f, 0, 3);
    fill(r.st, r.counter, scale, C, 0.25f, 1.0f, 1);
    fill(r.st, r.counter, bias, C, 0.25f, 0, 2);
    fill(r.st, r.counter, rmean, C, 0.1f, 0, 4);
    fill(r.st, r.counter, rvar, C, 0.1f, 0.5f, 5, 1);
  };
  r.run("cudnnBatchNormalizationForwardTraining", [&] {
    bool ok = true;
    init();
    float one = 1.0f, zero = 0.0f;
    Tensor xd({N, C, H, W}), bd({1, C, 1, 1});
    OK(cudnnBatchNormalizationForwardTraining(h, CUDNN_BATCHNORM_SPATIAL, &one, &zero, xd, x, xd, y, bd, scale, bias,
                                              0.1, rmean, rvar, 1e-5, smean, sinv));
    return ok;
  }, {{y, (size_t)n, Dt::F32}, {rmean, (size_t)C, Dt::F32}, {rvar, (size_t)C, Dt::F32}, {smean, (size_t)C, Dt::F32}, {sinv, (size_t)C, Dt::F32}}, 1e-4);
  r.run("cudnnBatchNormalizationForwardInference", [&] {
    bool ok = true;
    init();
    float one = 1.0f, zero = 0.0f;
    Tensor xd({N, C, H, W}), bd({1, C, 1, 1});
    OK(cudnnBatchNormalizationForwardInference(h, CUDNN_BATCHNORM_SPATIAL, &one, &zero, xd, x, xd, y, bd, scale, bias,
                                               rmean, rvar, 1e-5));
    return ok;
  }, {{y, (size_t)n, Dt::F32}}, 1e-4);
  r.run("cudnnBatchNormalizationBackward", [&] {
    bool ok = true;
    init();
    float one = 1.0f, zero = 0.0f;
    Tensor xd({N, C, H, W}), bd({1, C, 1, 1});
    OK(cudnnBatchNormalizationForwardTraining(h, CUDNN_BATCHNORM_SPATIAL, &one, &zero, xd, x, xd, y, bd, scale, bias,
                                              0.1, nullptr, nullptr, 1e-5, smean, sinv));
    OK(cudnnBatchNormalizationBackward(h, CUDNN_BATCHNORM_SPATIAL, &one, &zero, &one, &zero, xd, x, xd, dy, xd, dx, bd,
                                       scale, dscale, dbias, 1e-5, smean, sinv));
    return ok;
  }, {{dx, (size_t)n, Dt::F32}, {dscale, (size_t)C, Dt::F32}, {dbias, (size_t)C, Dt::F32}}, 1e-4);

  // The Ex forms, with a fused add and activation (PyTorch's channels-last path uses these):
  // half data in NHWC, the persistent mode, as the hardware fuses them.
  const auto H16 = CUDNN_DATA_HALF;
  const auto NHWC = CUDNN_TENSOR_NHWC;
  const auto PERSIST = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
  size_t ws = 0, res = 0, bws = 0;
  {
    Tensor xd({N, C, H, W}, H16, NHWC), bd({1, C, 1, 1});
    Act ad(CUDNN_ACTIVATION_RELU);
    cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(h, PERSIST, CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION, xd, xd, xd,
                                                             bd, ad, &ws);
    cudnnGetBatchNormalizationTrainingExReserveSpaceSize(h, PERSIST, CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION, ad, xd, &res);
    cudnnGetBatchNormalizationBackwardExWorkspaceSize(h, PERSIST, CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION, xd, xd, xd, xd, xd,
                                                      bd, ad, &bws);
  }
  void *work = r.alloc<char>(std::max(ws, bws) + 16), *reserve = r.alloc<char>(res + 16);
  __half *hx = r.alloc<__half>(n), *hz = r.alloc<__half>(n), *hy = r.alloc<__half>(n), *hdy = r.alloc<__half>(n);
  __half *hdx = r.alloc<__half>(n), *hdz = r.alloc<__half>(n);
  r.run("cudnnBatchNormalizationForwardTrainingEx + BackwardEx", [&] {
    bool ok = true;
    init();
    fill(r.st, r.counter, hx, n);
    fill(r.st, r.counter, hz, n, 0.25f, 0, 6);
    fill(r.st, r.counter, hdy, n, 0.25f, 0, 3);
    float one = 1.0f, zero = 0.0f;
    Tensor xd({N, C, H, W}, H16, NHWC), bd({1, C, 1, 1});
    Act ad(CUDNN_ACTIVATION_RELU);
    OK(cudnnBatchNormalizationForwardTrainingEx(h, PERSIST, CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION, &one, &zero, xd, hx, xd,
                                                hz, xd, hy, bd, scale, bias, 0.1, rmean, rvar, 1e-5, smean, sinv, ad, work,
                                                ws, reserve, res));
    OK(cudnnBatchNormalizationBackwardEx(h, PERSIST, CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION, &one, &zero, &one, &zero, xd,
                                         hx, xd, hy, xd, hdy, xd, hdz, xd, hdx, bd, scale, bias, dscale, dbias, 1e-5, smean,
                                         sinv, ad, work, bws, reserve, res));
    return ok;
  }, {{hy, (size_t)n, Dt::F16}, {hdx, (size_t)n, Dt::F16}, {hdz, (size_t)n, Dt::F16}, {dscale, (size_t)C, Dt::F32},
      {dbias, (size_t)C, Dt::F32}, {rmean, (size_t)C, Dt::F32}}, 2e-3);

  // The cuDNN 8 normalization API.
  struct Norm { size_t ws = 0, res = 0, bws = 0; } nm;
  {
    Tensor xd({N, C, H, W}), bd({1, C, 1, 1});
    cudnnGetNormalizationForwardTrainingWorkspaceSize(h, CUDNN_NORM_PER_CHANNEL, CUDNN_NORM_OPS_NORM, CUDNN_NORM_ALGO_STANDARD,
                                                      xd, nullptr, xd, bd, nullptr, bd, &nm.ws, 1);
    cudnnGetNormalizationTrainingReserveSpaceSize(h, CUDNN_NORM_PER_CHANNEL, CUDNN_NORM_OPS_NORM, CUDNN_NORM_ALGO_STANDARD,
                                                  nullptr, xd, &nm.res, 1);
    cudnnGetNormalizationBackwardWorkspaceSize(h, CUDNN_NORM_PER_CHANNEL, CUDNN_NORM_OPS_NORM, CUDNN_NORM_ALGO_STANDARD,
                                               xd, xd, xd, nullptr, xd, bd, nullptr, bd, &nm.bws, 1);
  }
  void *nwork = r.alloc<char>(std::max(nm.ws, nm.bws) + 16), *nres = r.alloc<char>(nm.res + 16);
  r.run("cudnnNormalizationForwardTraining + Backward", [&] {
    bool ok = true;
    init();
    float one = 1.0f, zero = 0.0f;
    Tensor xd({N, C, H, W}), bd({1, C, 1, 1});
    OK(cudnnNormalizationForwardTraining(h, CUDNN_NORM_PER_CHANNEL, CUDNN_NORM_OPS_NORM, CUDNN_NORM_ALGO_STANDARD, &one, &zero,
                                         xd, x, bd, scale, bias, 0.1, bd, rmean, rvar, 1e-5, smean, sinv, nullptr, nullptr,
                                         nullptr, xd, y, nwork, nm.ws, nres, nm.res, 1));
    OK(cudnnNormalizationBackward(h, CUDNN_NORM_PER_CHANNEL, CUDNN_NORM_OPS_NORM, CUDNN_NORM_ALGO_STANDARD, &one, &zero,
                                  &one, &zero, xd, x, xd, y, xd, dy, nullptr, nullptr, xd, dx, bd, scale, bias, dscale,
                                  dbias, 1e-5, bd, smean, sinv, nullptr, nwork, nm.bws, nres, nm.res, 1));
    return ok;
  }, {{y, (size_t)n, Dt::F32}, {dx, (size_t)n, Dt::F32}, {dscale, (size_t)C, Dt::F32}, {dbias, (size_t)C, Dt::F32},
      {rmean, (size_t)C, Dt::F32}}, 1e-4);
  r.run("cudnnNormalizationForwardInference", [&] {
    bool ok = true;
    init();
    float one = 1.0f, zero = 0.0f;
    Tensor xd({N, C, H, W}), bd({1, C, 1, 1});
    OK(cudnnNormalizationForwardInference(h, CUDNN_NORM_PER_CHANNEL, CUDNN_NORM_OPS_NORM, CUDNN_NORM_ALGO_STANDARD, &one,
                                          &zero, xd, x, bd, scale, bias, bd, rmean, rvar, nullptr, nullptr, nullptr, xd, y,
                                          1e-5, 1));
    return ok;
  }, {{y, (size_t)n, Dt::F32}}, 1e-4);
}


/* ---- the rest of the classic API ------------------------------------------------------- */

__global__ void labels_kernel(int* p, int n, const int* counter) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = 1 + (i * 3 + *counter) % 4;   // 1..4: never the blank
}

static void spatial_transformer(Runner& r, cudnnHandle_t h) {
  const int N = 2, C = 2, H = 5, W = 5, n = N * C * H * W;
  float *x = r.alloc<float>(n), *y = r.alloc<float>(n), *theta = r.alloc<float>(N * 6), *grid = r.alloc<float>(N * H * W * 2);
  float *dy = r.alloc<float>(n), *dx = r.alloc<float>(n), *dgrid = r.alloc<float>(N * H * W * 2), *dtheta = r.alloc<float>(N * 6);
  struct St {
    cudnnSpatialTransformerDescriptor_t d = nullptr;
    St(int N, int C, int H, int W) {
      cudnnCreateSpatialTransformerDescriptor(&d);
      const int dims[4] = {N, C, H, W};
      cudnnSetSpatialTransformerNdDescriptor(d, CUDNN_SAMPLER_BILINEAR, CUDNN_DATA_FLOAT, 4, dims);
    }
    ~St() { cudnnDestroySpatialTransformerDescriptor(d); }
    operator cudnnSpatialTransformerDescriptor_t() const { return d; }
  };
  r.run("cudnnSpatialTfGridGeneratorForward + SamplerForward", [&] {
    bool ok = true;
    fill(r.st, r.counter, x, n);
    fill(r.st, r.counter, theta, N * 6, 0.05f, 0.0f, 1);
    float one = 1.0f, zero = 0.0f;
    St st(N, C, H, W);
    Tensor xd({N, C, H, W}), yd({N, C, H, W});
    OK(cudnnSpatialTfGridGeneratorForward(h, st, theta, grid));
    OK(cudnnSpatialTfSamplerForward(h, st, &one, xd, x, grid, &zero, yd, y));
    return ok;
  }, {{grid, (size_t)N * H * W * 2, Dt::F32}, {y, (size_t)n, Dt::F32}}, 1e-4);
  r.run("cudnnSpatialTfSamplerBackward + GridGeneratorBackward", [&] {
    bool ok = true;
    fill(r.st, r.counter, x, n);
    fill(r.st, r.counter, dy, n, 0.25f, 0, 3);
    fill(r.st, r.counter, theta, N * 6, 0.05f, 0.0f, 1);
    float one = 1.0f, zero = 0.0f;
    St st(N, C, H, W);
    Tensor xd({N, C, H, W}), dxd({N, C, H, W}), dyd({N, C, H, W});
    OK(cudnnSpatialTfGridGeneratorForward(h, st, theta, grid));
    OK(cudnnSpatialTfSamplerBackward(h, st, &one, xd, x, &zero, dxd, dx, &one, dyd, dy, grid, &zero, dgrid));
    OK(cudnnSpatialTfGridGeneratorBackward(h, st, dgrid, dtheta));
    return ok;
  }, {{dx, (size_t)n, Dt::F32}, {dgrid, (size_t)N * H * W * 2, Dt::F32}, {dtheta, (size_t)N * 6, Dt::F32}}, 1e-3);
}

static void transforms(Runner& r, cudnnHandle_t h) {
  const int N = 2, C = 3, H = 4, W = 4, n = N * C * H * W;
  float *x = r.alloc<float>(n), *y = r.alloc<float>(n);
  struct Td {
    cudnnTensorTransformDescriptor_t d = nullptr;
    Td() {
      cudnnCreateTensorTransformDescriptor(&d);
      const int32_t before[4] = {0, 0, 0, 0}, after[4] = {0, 0, 0, 0};
      const uint32_t fold[4] = {0, 0, 0, 0};
      cudnnSetTensorTransformDescriptor(d, 4, CUDNN_TENSOR_NHWC, before, after, fold, CUDNN_TRANSFORM_FOLD);
    }
    ~Td() { cudnnDestroyTensorTransformDescriptor(d); }
    operator cudnnTensorTransformDescriptor_t() const { return d; }
  };
  r.run("cudnnTransformTensorEx", [&] {
    bool ok = true;
    fill(r.st, r.counter, x, n);
    float one = 1.0f, zero = 0.0f;
    Tensor xd({N, C, H, W}), yd({N, C, H, W}, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NHWC);
    Td td;
    OK(cudnnTransformTensorEx(h, td, &one, xd, x, &zero, yd, y));
    return ok;
  }, {{y, (size_t)n, Dt::F32}});
  // im2col: one row per (channel, filter tap), one column per output position.
  const int K = 2;
  float *w = r.alloc<float>(K * C * 9), *col = r.alloc<float>(C * 9 * N * H * W);
  r.run("cudnnIm2Col", [&] {
    bool ok = true;
    fill(r.st, r.counter, x, n);
    Tensor xd({N, C, H, W});
    Filter wd({K, C, 3, 3});
    Conv cd;
    OK(cudnnIm2Col(h, xd, x, wd, cd, col));
    return ok;
  }, {{col, (size_t)C * 9 * N * H * W, Dt::F32}});
}

static void reorder(Runner& r, cudnnHandle_t h) {
  const int K = 32, C = 32;
  unsigned char *w = r.alloc<unsigned char>(K * C * 9), *rw = r.alloc<unsigned char>(K * C * 9);
  r.run("cudnnReorderFilterAndBias (INT8x32)", [&] {
    bool ok = true;
    fill(r.st, r.counter, w, K * C * 9, 1.0f, 5.0f);
    cudnnFilterDescriptor_t fd;
    cudnnCreateFilterDescriptor(&fd);
    OK(cudnnSetFilter4dDescriptor(fd, CUDNN_DATA_INT8x32, CUDNN_TENSOR_NCHW_VECT_C, K, C, 3, 3));
    OK(cudnnReorderFilterAndBias(h, fd, CUDNN_DEFAULT_REORDER, w, rw, 0, nullptr, nullptr));
    cudnnDestroyFilterDescriptor(fd);
    return ok;
  }, {{rw, (size_t)K * C * 9, Dt::Bytes}});
}

static void dropout(Runner& r, cudnnHandle_t h) {
  const int n = 4000;
  float *x = r.alloc<float>(n), *y = r.alloc<float>(n), *dy = r.alloc<float>(n), *dx = r.alloc<float>(n);
  size_t states_bytes = 0, reserve_bytes = 0;
  cudnnDropoutGetStatesSize(h, &states_bytes);
  {
    Tensor d({1, 1, 1, n});
    cudnnDropoutGetReserveSpaceSize(d, &reserve_bytes);
  }
  void *states = r.alloc<char>(states_bytes), *reserve = r.alloc<char>(reserve_bytes);
  const float p = 0.5f;
  {   // the generators are seeded once, outside any capture
    cudnnDropoutDescriptor_t dd;
    cudnnCreateDropoutDescriptor(&dd);
    cudnnSetDropoutDescriptor(dd, h, p, states, states_bytes, 1234);
    cudnnDestroyDropoutDescriptor(dd);
    cudaStreamSynchronize(r.st);
  }
  std::vector<float> last_mask;
  r.run("cudnnDropoutForward + Backward", [&] {
    bool ok = true;
    fill(r.st, r.counter, x, n, 0.25f, 3.0f);   // 1.75 .. 4.25: never zero
    fill(r.st, r.counter, dy, n, 0.25f, 3.0f, 2);
    // PyTorch's pattern: the descriptor is rebuilt over the same states at every call.
    cudnnDropoutDescriptor_t dd;
    cudnnCreateDropoutDescriptor(&dd);
    OK(cudnnRestoreDropoutDescriptor(dd, h, p, states, states_bytes, 1234));
    Tensor d({1, 1, 1, n});
    OK(cudnnDropoutForward(h, dd, d, x, d, y, reserve, reserve_bytes));
    OK(cudnnDropoutBackward(h, dd, d, dy, d, dx, reserve, reserve_bytes));
    cudnnDestroyDropoutDescriptor(dd);
    return ok;
  }, {{y, (size_t)n, Dt::F32}, {dx, (size_t)n, Dt::F32}}, 0,
  [&](int c, int launch) {
    // Each element is dropped, or scaled by 1 / (1 - p); about half are kept; the backward
    // pass keeps the same ones; and the generators have moved on since the last launch, so the
    // mask is not the previous launch's.
    std::vector<float> hy(n), hdx(n);
    cudaMemcpy(hy.data(), y, n * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(hdx.data(), dx, n * 4, cudaMemcpyDeviceToHost);
    int kept = 0;
    bool scaled = true, same = true;
    std::vector<float> mask(n);
    for (int i = 0; i < n; ++i) {
      const float xv = 3.0f + 0.25f * pattern(i, c, 0, 0), dyv = 3.0f + 0.25f * pattern(i, c, 2, 0);
      mask[i] = hy[i] != 0.0f;
      kept += hy[i] != 0.0f;
      scaled = scaled && (hy[i] == 0.0f || std::fabs(hy[i] - xv * 2.0f) < 1e-5f);
      same = same && ((hy[i] != 0.0f) == (hdx[i] != 0.0f)) && (hdx[i] == 0.0f || std::fabs(hdx[i] - dyv * 2.0f) < 1e-5f);
    }
    const bool fresh = launch == 1 || mask != last_mask;
    last_mask = mask;
    expect("  dropout: the elements kept are scaled by 1 / (1 - p)", scaled);
    expect("  dropout: about half are kept", kept > n * 4 / 10 && kept < n * 6 / 10, kept);
    expect("  dropout: the backward pass keeps the same elements", same);
    expect("  dropout: the next launch draws a new mask", fresh);
    return scaled && same;
  });
}

static void ctc(Runner& r, cudnnHandle_t h) {
  const int T = 7, N = 2, A = 5, maxL = 3;
  float *probs = r.alloc<float>(T * N * A), *grads = r.alloc<float>(T * N * A), *costs = r.alloc<float>(N);
  int *labels = r.alloc<int>(N * maxL), *label_len = r.alloc<int>(N), *input_len = r.alloc<int>(N);
  const int hl[2] = {3, 2}, hi[2] = {7, 6};
  cudaMemcpy(label_len, hl, sizeof hl, cudaMemcpyHostToDevice);
  cudaMemcpy(input_len, hi, sizeof hi, cudaMemcpyHostToDevice);
  size_t ws = 0;
  {
    cudnnCTCLossDescriptor_t d;
    cudnnCreateCTCLossDescriptor(&d);
    cudnnSetCTCLossDescriptor_v8(d, CUDNN_DATA_FLOAT, CUDNN_LOSS_NORMALIZATION_SOFTMAX, CUDNN_PROPAGATE_NAN, maxL);
    Tensor pd({T, N, A}), gd({T, N, A});
    cudnnGetCTCLossWorkspaceSize_v8(h, CUDNN_CTC_LOSS_ALGO_DETERMINISTIC, d, pd, gd, &ws);
    cudnnDestroyCTCLossDescriptor(d);
  }
  void* work = r.alloc<char>(ws + 16);
  size_t ws7 = 0;
  {
    cudnnCTCLossDescriptor_t d;
    cudnnCreateCTCLossDescriptor(&d);
    cudnnSetCTCLossDescriptor(d, CUDNN_DATA_FLOAT);
    Tensor pd({T, N, A}), gd({T, N, A});
    const int hl_[2] = {3, 2}, hi_[2] = {7, 6}, lab[5] = {1, 2, 3, 4, 1};
    cudnnGetCTCLossWorkspaceSize(h, pd, gd, lab, hl_, hi_, CUDNN_CTC_LOSS_ALGO_DETERMINISTIC, d, &ws7);
    cudnnDestroyCTCLossDescriptor(d);
  }
  void* work7 = r.alloc<char>(ws7 + 16);
  r.run("cudnnCTCLoss_v8", [&] {
    bool ok = true;
    fill(r.st, r.counter, probs, T * N * A, 0.5f);
    labels_kernel<<<1, 32, 0, r.st>>>(labels, N * maxL, r.counter);
    cudnnCTCLossDescriptor_t d;
    cudnnCreateCTCLossDescriptor(&d);
    OK(cudnnSetCTCLossDescriptor_v8(d, CUDNN_DATA_FLOAT, CUDNN_LOSS_NORMALIZATION_SOFTMAX, CUDNN_PROPAGATE_NAN, maxL));
    Tensor pd({T, N, A}), gd({T, N, A});
    OK(cudnnCTCLoss_v8(h, CUDNN_CTC_LOSS_ALGO_DETERMINISTIC, d, pd, probs, labels, label_len, input_len, costs, gd, grads,
                       ws, work));
    cudnnDestroyCTCLossDescriptor(d);
    return ok;
  }, {{costs, (size_t)N, Dt::F32}, {grads, (size_t)T * N * A, Dt::F32}}, 1e-4);

  // The older entry point takes the labels and lengths from host memory (read at the call).
  r.run("cudnnCTCLoss (host labels and lengths)", [&] {
    bool ok = true;
    fill(r.st, r.counter, probs, T * N * A, 0.05f, 0.0f, 0, 1);   // the old descriptor takes probabilities
    // NVIDIA's cudnnCTCLoss keeps pointing at these arrays in the graph (a launch after they are
    // gone faults: measured on the RTX 3060), so they outlive it here; the shim copies them.
    static int hl_[2] = {3, 2}, hi_[2] = {7, 6}, lab[5] = {1, 2, 3, 4, 1};
    cudnnCTCLossDescriptor_t d;
    cudnnCreateCTCLossDescriptor(&d);
    OK(cudnnSetCTCLossDescriptor(d, CUDNN_DATA_FLOAT));
    Tensor pd({T, N, A}), gd({T, N, A});
    OK(cudnnCTCLoss(h, pd, probs, lab, hl_, hi_, costs, gd, grads, CUDNN_CTC_LOSS_ALGO_DETERMINISTIC, d, work7, ws7));
    cudnnDestroyCTCLossDescriptor(d);
    return ok;
  }, {{costs, (size_t)N, Dt::F32}, {grads, (size_t)T * N * A, Dt::F32}}, 1e-4);
}

struct RnnSetup {
  bool lstm;
  int layers, hid, in, T, B;
  bool dirs2;
};
static void rnn(Runner& r, cudnnHandle_t h, RnnSetup s, const char* name) {
  const int D = s.dirs2 ? 2 : 1, H = s.hid;
  const int states = s.layers * D * s.B * H;
  std::vector<int> lens(s.B, s.T);
  for (int b = 1; b < s.B; ++b) lens[b] = std::max(1, s.T - b);   // longest first
  int *dlens = r.alloc<int>(s.B);
  cudaMemcpy(dlens, lens.data(), s.B * sizeof(int), cudaMemcpyHostToDevice);
  size_t wbytes = 0, ws = 0, res = 0;
  {
    cudnnRNNDescriptor_t rd;
    cudnnCreateRNNDescriptor(&rd);
    cudnnSetRNNDescriptor_v8(rd, CUDNN_RNN_ALGO_STANDARD, s.lstm ? CUDNN_LSTM : CUDNN_GRU, CUDNN_RNN_DOUBLE_BIAS,
                             s.dirs2 ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL, CUDNN_LINEAR_INPUT, CUDNN_DATA_FLOAT,
                             CUDNN_DATA_FLOAT, CUDNN_DEFAULT_MATH, s.in, H, H, s.layers, nullptr, 0);
    cudnnRNNDataDescriptor_t xd;
    cudnnCreateRNNDataDescriptor(&xd);
    cudnnSetRNNDataDescriptor(xd, CUDNN_DATA_FLOAT, CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED, s.T, s.B, s.in, lens.data(),
                              nullptr);
    cudnnGetRNNWeightSpaceSize(h, rd, &wbytes);
    cudnnGetRNNTempSpaceSizes(h, rd, CUDNN_FWD_MODE_TRAINING, xd, &ws, &res);
    cudnnDestroyRNNDataDescriptor(xd);
    cudnnDestroyRNNDescriptor(rd);
  }
  size_t steps = 0;   // packed: only the steps each sequence has
  for (int l : lens) steps += l;
  const size_t nw = wbytes / 4, nx = steps * s.in, ny = steps * H * D;
  float *w = r.alloc<float>(nw), *dw = r.alloc<float>(nw), *x = r.alloc<float>(nx), *dx = r.alloc<float>(nx);
  float *y = r.alloc<float>(ny), *dy = r.alloc<float>(ny), *hx = r.alloc<float>(states), *cx = r.alloc<float>(states);
  float *hy = r.alloc<float>(states), *cy = r.alloc<float>(states), *dhx = r.alloc<float>(states), *dcx = r.alloc<float>(states);
  void *work = r.alloc<char>(ws + 16), *reserve = r.alloc<char>(res + 16);
  auto inputs = [&] {
    fill(r.st, r.counter, w, nw, 0.05f);
    fill(r.st, r.counter, x, nx, 0.25f, 0, 1);
    fill(r.st, r.counter, hx, states, 0.1f, 0, 2);
    fill(r.st, r.counter, cx, states, 0.1f, 0, 3);
    fill(r.st, r.counter, dy, ny, 0.25f, 0, 4);
  };
  // Built in the call, destroyed before it returns.
  auto call = [&](bool backward) {
    bool ok = true;
    inputs();
    cudnnRNNDescriptor_t rd;
    cudnnRNNDataDescriptor_t xd, yd;
    cudnnTensorDescriptor_t hd;
    cudnnCreateRNNDescriptor(&rd);
    cudnnCreateRNNDataDescriptor(&xd);
    cudnnCreateRNNDataDescriptor(&yd);
    cudnnCreateTensorDescriptor(&hd);
    OK(cudnnSetRNNDescriptor_v8(rd, CUDNN_RNN_ALGO_STANDARD, s.lstm ? CUDNN_LSTM : CUDNN_GRU, CUDNN_RNN_DOUBLE_BIAS,
                                s.dirs2 ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL, CUDNN_LINEAR_INPUT, CUDNN_DATA_FLOAT,
                                CUDNN_DATA_FLOAT, CUDNN_DEFAULT_MATH, s.in, H, H, s.layers, nullptr, 0));
    OK(cudnnSetRNNDataDescriptor(xd, CUDNN_DATA_FLOAT, CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED, s.T, s.B, s.in,
                                 lens.data(), nullptr));
    OK(cudnnSetRNNDataDescriptor(yd, CUDNN_DATA_FLOAT, CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED, s.T, s.B, H * D,
                                 lens.data(), nullptr));
    const int hdims[3] = {s.layers * D, s.B, H}, hstr[3] = {s.B * H, H, 1};
    OK(cudnnSetTensorNdDescriptor(hd, CUDNN_DATA_FLOAT, 3, hdims, hstr));
    float *cxp = s.lstm ? cx : nullptr, *cyp = s.lstm ? cy : nullptr, *dcxp = s.lstm ? dcx : nullptr;
    OK(cudnnRNNForward(h, rd, CUDNN_FWD_MODE_TRAINING, dlens, xd, x, yd, y, hd, hx, hy, hd, cxp, cyp, wbytes, w, ws, work,
                       res, reserve));
    if (backward) {
      OK(cudnnRNNBackwardData_v8(h, rd, dlens, yd, y, dy, xd, dx, hd, hx, nullptr, dhx, hd, cxp, nullptr, dcxp, wbytes, w,
                                 ws, work, res, reserve));
      OK(cudnnRNNBackwardWeights_v8(h, rd, CUDNN_WGRAD_MODE_ADD, dlens, xd, x, hd, hx, yd, y, wbytes, dw, ws, work, res,
                                    reserve));
    }
    cudnnDestroyTensorDescriptor(hd);
    cudnnDestroyRNNDataDescriptor(yd);
    cudnnDestroyRNNDataDescriptor(xd);
    cudnnDestroyRNNDescriptor(rd);
    return ok;
  };
  std::vector<Out> outs{{y, ny, Dt::F32}, {hy, (size_t)states, Dt::F32}};
  r.run(std::string(name) + ", forward", [&] { return call(false); }, outs, 1e-4);
  outs.push_back({dx, nx, Dt::F32});
  outs.push_back({dhx, (size_t)states, Dt::F32});
  outs.push_back({dw, nw, Dt::F32});
  r.run(std::string(name) + ", forward + backward data + backward weights", [&] { return call(true); }, outs, 1e-3);
}

static void attention(Runner& r, cudnnHandle_t h) {
  const int NH = 2, Tq = 4, Tk = 5, B = 2, qS = 6, kS = 6, vS = 6, qP = 4, vP = 4, oP = 6;
  const unsigned mode = CUDNN_ATTN_QUERYMAP_ALL_TO_ONE | CUDNN_ATTN_ENABLE_PROJ_BIASES;
  auto set_attn = [&](cudnnAttnDescriptor_t ad) {
    return cudnnSetAttnDescriptor(ad, mode, NH, 0.5, CUDNN_DATA_FLOAT, CUDNN_DATA_FLOAT, CUDNN_DEFAULT_MATH, nullptr, nullptr,
                                  qS, kS, vS, qP, qP, vP, oP, Tq, Tk, B, 1);
  };
  size_t wb = 0, ws = 0, res = 0;
  {
    cudnnAttnDescriptor_t ad;
    cudnnCreateAttnDescriptor(&ad);
    set_attn(ad);
    cudnnGetMultiHeadAttnBuffers(h, ad, &wb, &ws, &res);
    cudnnDestroyAttnDescriptor(ad);
  }
  const size_t nw = wb / 4, nq = (size_t)B * Tq * qS, nk = (size_t)B * Tk * kS, nv = (size_t)B * Tk * vS, no = (size_t)B * Tq * oP;
  float *w = r.alloc<float>(nw), *dw = r.alloc<float>(nw), *q = r.alloc<float>(nq), *k = r.alloc<float>(nk);
  float *v = r.alloc<float>(nv), *o = r.alloc<float>(no), *dout = r.alloc<float>(no), *dq = r.alloc<float>(nq);
  float *dk = r.alloc<float>(nk), *dv = r.alloc<float>(nv);
  void *work = r.alloc<char>(ws + 16), *reserve = r.alloc<char>(res + 16);
  const int lq[2] = {Tq, Tq - 1}, lk[2] = {Tk, Tk - 1};
  int *dlq = r.alloc<int>(2), *dlk = r.alloc<int>(2);
  cudaMemcpy(dlq, lq, sizeof lq, cudaMemcpyHostToDevice);
  cudaMemcpy(dlk, lk, sizeof lk, cudaMemcpyHostToDevice);
  auto call = [&](bool backward) {
    bool ok = true;
    fill(r.st, r.counter, w, nw, 0.1f);
    fill(r.st, r.counter, q, nq, 0.25f, 0, 1);
    fill(r.st, r.counter, k, nk, 0.25f, 0, 2);
    fill(r.st, r.counter, v, nv, 0.25f, 0, 3);
    fill(r.st, r.counter, dout, no, 0.25f, 0, 4);
    cudnnAttnDescriptor_t ad;
    cudnnCreateAttnDescriptor(&ad);
    OK(set_attn(ad));
    cudnnSeqDataDescriptor_t qd, kd, vd, od;
    for (auto* d : {&qd, &kd, &vd, &od}) cudnnCreateSeqDataDescriptor(d);
    const cudnnSeqDataAxis_t axes[4] = {CUDNN_SEQDATA_BATCH_DIM, CUDNN_SEQDATA_BEAM_DIM, CUDNN_SEQDATA_TIME_DIM,
                                        CUDNN_SEQDATA_VECT_DIM};
    auto dims = [](int T, int Bn, int V) {
      std::vector<int> d(4);
      d[CUDNN_SEQDATA_TIME_DIM] = T, d[CUDNN_SEQDATA_BATCH_DIM] = Bn, d[CUDNN_SEQDATA_BEAM_DIM] = 1, d[CUDNN_SEQDATA_VECT_DIM] = V;
      return d;
    };
    OK(cudnnSetSeqDataDescriptor(qd, CUDNN_DATA_FLOAT, 4, dims(Tq, B, qS).data(), axes, 2, lq, nullptr));
    OK(cudnnSetSeqDataDescriptor(kd, CUDNN_DATA_FLOAT, 4, dims(Tk, B, kS).data(), axes, 2, lk, nullptr));
    OK(cudnnSetSeqDataDescriptor(vd, CUDNN_DATA_FLOAT, 4, dims(Tk, B, vS).data(), axes, 2, lk, nullptr));
    OK(cudnnSetSeqDataDescriptor(od, CUDNN_DATA_FLOAT, 4, dims(Tq, B, oP).data(), axes, 2, lq, nullptr));
    int lo[4] = {0, 0, 1, 0}, hi[4] = {2, 3, 4, 1000};   // host windows, gone with the frame
    OK(cudnnMultiHeadAttnForward(h, ad, -1, lo, hi, dlq, dlk, qd, q, nullptr, kd, k, vd, v, od, o, wb, w, ws, work, res, reserve));
    if (backward) {
      OK(cudnnMultiHeadAttnBackwardData(h, ad, lo, hi, dlq, dlk, od, dout, qd, dq, q, kd, dk, k, vd, dv, v, wb, w, ws, work,
                                        res, reserve));
      OK(cudnnMultiHeadAttnBackwardWeights(h, ad, CUDNN_WGRAD_MODE_SET, qd, q, kd, k, vd, v, od, dout, wb, w, dw, ws, work,
                                           res, reserve));
    }
    for (auto* d : {&qd, &kd, &vd, &od}) cudnnDestroySeqDataDescriptor(*d);
    cudnnDestroyAttnDescriptor(ad);
    return ok;
  };
  r.run("cudnnMultiHeadAttnForward", [&] { return call(false); }, {{o, no, Dt::F32}}, 1e-4);
  r.run("cudnnMultiHeadAttnBackwardData + BackwardWeights", [&] { return call(true); },
        {{o, no, Dt::F32}, {dq, nq, Dt::F32}, {dk, nk, Dt::F32}, {dv, nv, Dt::F32}, {dw, nw, Dt::F32}}, 1e-3);
}


// The Find calls run and time kernels, which a capture cannot have. Measured on an RTX 3060 (cuDNN 9.15): the Ex forms
// answer success and invalidate the capture, in every capture mode; the others do the same in the relaxed mode, and in
// the global one are refused for the memory they allocate: CUDNN_STATUS_INTERNAL_ERROR_DEVICE_ALLOCATION_FAILED
// (4004), the capture invalidated. The Get..._v7 calls (heuristics, no kernels) leave a capture as it was.
static void find_calls(Runner& r, cudnnHandle_t h) {
  Tensor x({1, 3, 8, 8}), y({1, 4, 8, 8});
  Filter w({4, 3, 3, 3});
  Conv c;
  float *xp = r.alloc<float>(1 << 12), *wp = r.alloc<float>(1 << 12), *yp = r.alloc<float>(1 << 12);
  void* work = r.alloc<char>(1 << 24);
  cudnnConvolutionFwdAlgoPerf_t perf[8];
  cudnnConvolutionBwdDataAlgoPerf_t perf_d[8];
  cudnnConvolutionBwdFilterAlgoPerf_t perf_f[8];
  int count = 0;
  struct Find {
    const char* name;
    bool ex;
    std::function<int()> call;
  };
  const std::vector<Find> finds = {
      {"cudnnFindConvolutionForwardAlgorithm", false, [&] { return (int)cudnnFindConvolutionForwardAlgorithm(h, x, w, c, y, 8, &count, perf); }},
      {"cudnnFindConvolutionForwardAlgorithmEx", true, [&] { return (int)cudnnFindConvolutionForwardAlgorithmEx(h, x, xp, w, wp, c, y, yp, 8, &count, perf, work, 1 << 24); }},
      {"cudnnFindConvolutionBackwardDataAlgorithm", false, [&] { return (int)cudnnFindConvolutionBackwardDataAlgorithm(h, w, y, c, x, 8, &count, perf_d); }},
      {"cudnnFindConvolutionBackwardDataAlgorithmEx", true, [&] { return (int)cudnnFindConvolutionBackwardDataAlgorithmEx(h, w, wp, y, yp, c, x, xp, 8, &count, perf_d, work, 1 << 24); }},
      {"cudnnFindConvolutionBackwardFilterAlgorithm", false, [&] { return (int)cudnnFindConvolutionBackwardFilterAlgorithm(h, x, y, c, w, 8, &count, perf_f); }},
      {"cudnnFindConvolutionBackwardFilterAlgorithmEx", true, [&] { return (int)cudnnFindConvolutionBackwardFilterAlgorithmEx(h, x, xp, y, yp, c, w, wp, 8, &count, perf_f, work, 1 << 24); }},
  };
  for (const Find& f : finds) {
    for (const cudaStreamCaptureMode mode : {cudaStreamCaptureModeGlobal, cudaStreamCaptureModeRelaxed}) {
      count = 0;
      const bool global = mode == cudaStreamCaptureModeGlobal;
      cudaStreamSynchronize(r.st);
      cudaGraph_t g = nullptr;
      const cudaError_t bc = cudaStreamBeginCapture(r.st, mode);
      const int rc = f.call();
      cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
      cudaStreamIsCapturing(r.st, &status);
      const cudaError_t e = cudaStreamEndCapture(r.st, &g);
      if (g) cudaGraphDestroy(g);
      cudaGetLastError();
      cudaStreamSynchronize(r.st);
      const int want = !f.ex && global ? 4004 : 0;
      expect(std::string(f.name) + (global ? " in a global capture" : " in a relaxed capture") + ": " +
                 (want ? "refused (4004)" : "answers") + ", and the capture is invalidated",
             bc == cudaSuccess && rc == want && (want || count > 0) && status == cudaStreamCaptureStatusInvalidated &&
                 e == cudaErrorStreamCaptureInvalidated,
             rc);
    }
  }
  {
    cudaGraph_t g = nullptr;
    cudaStreamBeginCapture(r.st, cudaStreamCaptureModeGlobal);
    const int rc = (int)cudnnGetConvolutionForwardAlgorithm_v7(h, x, w, c, y, 8, &count, perf);
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(r.st, &status);
    const cudaError_t e = cudaStreamEndCapture(r.st, &g);
    if (g) cudaGraphDestroy(g);
    expect("cudnnGetConvolutionForwardAlgorithm_v7 in a capture: answers, and leaves it valid",
           rc == 0 && count > 0 && status == cudaStreamCaptureStatusActive && e == cudaSuccess, rc);
  }
}

int main() {
  Runner r;
  cudnnHandle_t h = make_handle(r.st);
  convolution(r, h);
  pointwise(r, h);
  tensor_ops(r, h);
  normalization(r, h);
  spatial_transformer(r, h);
  transforms(r, h);
  reorder(r, h);
  dropout(r, h);
  ctc(r, h);
  rnn(r, h, {true, 2, 5, 4, 5, 3, false}, "cudnnRNNForward (LSTM, 2 layers)");
  rnn(r, h, {false, 1, 4, 3, 4, 2, true}, "cudnnRNNForward (bidirectional GRU)");
  attention(r, h);
  find_calls(r, h);
  cudnnDestroy(h);
  return finish();
}
