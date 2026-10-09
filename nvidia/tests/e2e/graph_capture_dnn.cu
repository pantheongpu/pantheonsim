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

int main() {
  Runner r;
  cudnnHandle_t h = make_handle(r.st);
  convolution(r, h);
  pointwise(r, h);
  tensor_ops(r, h);
  normalization(r, h);
  cudnnDestroy(h);
  return finish();
}
