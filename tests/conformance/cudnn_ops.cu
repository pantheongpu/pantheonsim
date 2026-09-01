// Differential conformance for the cuDNN shim: convolution, activation,
// pooling, softmax, batch normalization and tensor arithmetic. The same binary
// runs against NVIDIA's libcudnn.so.9 and against VirtualGPU's; the printed
// values must agree.
#include <cudnn.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>

#define CK(x) do { cudnnStatus_t s_ = (x); if (s_ != CUDNN_STATUS_SUCCESS) { \
  printf("%s -> %d\n", #x, (int)s_); return; } } while (0)

// A fixed, reproducible filler: no RNG, so both sides see identical inputs.
static float fill(int i, float scale) { return scale * std::sin(0.7f * i + 0.3f) + 0.1f * (i % 5); }

static void dump(const char* tag, const std::vector<float>& v) {
  double sum = 0, asum = 0;
  for (float x : v) { sum += x; asum += std::fabs(x); }
  printf("%-28s n=%zu sum=%.4f abs=%.4f first=%.4f %.4f %.4f last=%.4f\n", tag, v.size(), sum,
         asum, v.size() > 0 ? v[0] : 0.f, v.size() > 1 ? v[1] : 0.f, v.size() > 2 ? v[2] : 0.f,
         v.size() ? v[v.size() - 1] : 0.f);
}

struct Dev {
  float* p = nullptr;
  explicit Dev(size_t n, float scale = 0.0f) {
    cudaMalloc(&p, n * sizeof(float));
    std::vector<float> h(n);
    for (size_t i = 0; i < n; ++i) h[i] = scale == 0.0f ? 0.0f : fill((int)i, scale);
    cudaMemcpy(p, h.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  }
  std::vector<float> get(size_t n) const {
    std::vector<float> h(n);
    cudaMemcpy(h.data(), p, n * sizeof(float), cudaMemcpyDeviceToHost);
    return h;
  }
  ~Dev() { cudaFree(p); }
};

static void conv_case(cudnnHandle_t h, const char* tag, int N, int C, int H, int W, int K, int R,
                      int S, int pad, int stride, int dil, int groups, cudnnConvolutionMode_t mode) {
  cudnnTensorDescriptor_t xd, yd;
  cudnnFilterDescriptor_t wd;
  cudnnConvolutionDescriptor_t cd;
  CK(cudnnCreateTensorDescriptor(&xd));
  CK(cudnnCreateTensorDescriptor(&yd));
  CK(cudnnCreateFilterDescriptor(&wd));
  CK(cudnnCreateConvolutionDescriptor(&cd));
  CK(cudnnSetTensor4dDescriptor(xd, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W));
  CK(cudnnSetFilter4dDescriptor(wd, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, K, C / groups, R, S));
  CK(cudnnSetConvolution2dDescriptor(cd, pad, pad, stride, stride, dil, dil, mode,
                                     CUDNN_DATA_FLOAT));
  CK(cudnnSetConvolutionGroupCount(cd, groups));
  int on, oc, oh, ow;
  CK(cudnnGetConvolution2dForwardOutputDim(cd, xd, wd, &on, &oc, &oh, &ow));
  CK(cudnnSetTensor4dDescriptor(yd, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, on, oc, oh, ow));
  printf("%-28s outdim=%dx%dx%dx%d\n", tag, on, oc, oh, ow);

  Dev x((size_t)N * C * H * W, 1.0f), w((size_t)K * (C / groups) * R * S, 0.5f);
  Dev y((size_t)on * oc * oh * ow);
  size_t ws = 0;
  CK(cudnnGetConvolutionForwardWorkspaceSize(h, xd, wd, cd, yd,
                                             CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, &ws));
  void* wsp = nullptr;
  if (ws) cudaMalloc(&wsp, ws);
  const float a = 1.0f, b = 0.0f;
  CK(cudnnConvolutionForward(h, &a, xd, x.p, wd, w.p, cd,
                             CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, wsp, ws, &b, yd, y.p));
  dump(tag, y.get((size_t)on * oc * oh * ow));
  if (wsp) cudaFree(wsp);
  cudnnDestroyConvolutionDescriptor(cd);
  cudnnDestroyFilterDescriptor(wd);
  cudnnDestroyTensorDescriptor(yd);
  cudnnDestroyTensorDescriptor(xd);
}

static void run() {
  cudnnHandle_t h;
  CK(cudnnCreate(&h));

  conv_case(h, "conv 3x3 pad1", 2, 3, 8, 8, 4, 3, 3, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION);
  conv_case(h, "conv 3x3 stride2", 1, 2, 9, 9, 3, 3, 3, 0, 2, 1, 1, CUDNN_CROSS_CORRELATION);
  conv_case(h, "conv 3x3 dilate2", 1, 2, 9, 9, 3, 3, 3, 2, 1, 2, 1, CUDNN_CROSS_CORRELATION);
  conv_case(h, "conv flipped kernel", 1, 2, 7, 7, 3, 3, 3, 1, 1, 1, 1, CUDNN_CONVOLUTION);
  conv_case(h, "conv grouped g=2", 1, 4, 7, 7, 4, 3, 3, 1, 1, 1, 2, CUDNN_CROSS_CORRELATION);
  conv_case(h, "conv 1x1", 2, 5, 6, 6, 7, 1, 1, 0, 1, 1, 1, CUDNN_CROSS_CORRELATION);

  // Shared 2x3x4x5 tensor for the pointwise operations.
  const int N = 2, C = 3, H = 4, W = 5;
  const size_t n = (size_t)N * C * H * W;
  cudnnTensorDescriptor_t td, bd;
  CK(cudnnCreateTensorDescriptor(&td));
  CK(cudnnCreateTensorDescriptor(&bd));
  CK(cudnnSetTensor4dDescriptor(td, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W));
  CK(cudnnSetTensor4dDescriptor(bd, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, C, 1, 1));
  Dev x(n, 1.5f);

  struct { const char* name; cudnnActivationMode_t mode; double coef; } acts[] = {
      {"act relu", CUDNN_ACTIVATION_RELU, 0.0},
      {"act sigmoid", CUDNN_ACTIVATION_SIGMOID, 0.0},
      {"act tanh", CUDNN_ACTIVATION_TANH, 0.0},
      {"act clipped relu 1.0", CUDNN_ACTIVATION_CLIPPED_RELU, 1.0},
      {"act elu 0.5", CUDNN_ACTIVATION_ELU, 0.5},
      {"act identity", CUDNN_ACTIVATION_IDENTITY, 0.0},
  };
  for (auto& t : acts) {
    cudnnActivationDescriptor_t ad;
    CK(cudnnCreateActivationDescriptor(&ad));
    CK(cudnnSetActivationDescriptor(ad, t.mode, CUDNN_NOT_PROPAGATE_NAN, t.coef));
    Dev y(n);
    const float a = 1.0f, b = 0.0f;
    cudnnStatus_t s = cudnnActivationForward(h, ad, &a, td, x.p, &b, td, y.p);
    if (s == CUDNN_STATUS_SUCCESS) dump(t.name, y.get(n));
    else printf("%-28s status=%d\n", t.name, (int)s);
    cudnnDestroyActivationDescriptor(ad);
  }

  // alpha/beta blending must accumulate into the existing output.
  {
    cudnnActivationDescriptor_t ad;
    CK(cudnnCreateActivationDescriptor(&ad));
    CK(cudnnSetActivationDescriptor(ad, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0.0));
    Dev y(n, 2.0f);
    const float a = 2.0f, b = 0.5f;
    CK(cudnnActivationForward(h, ad, &a, td, x.p, &b, td, y.p));
    dump("act relu alpha2 beta0.5", y.get(n));
    cudnnDestroyActivationDescriptor(ad);
  }

  // Bias add: [1,C,1,1] broadcast over [N,C,H,W].
  {
    Dev bias(C, 3.0f), y(n, 2.0f);
    const float a = 1.0f, b = 1.0f;
    CK(cudnnAddTensor(h, &a, bd, bias.p, &b, td, y.p));
    dump("addtensor bias broadcast", y.get(n));
  }

  struct { const char* name; cudnnPoolingMode_t mode; int win; int pad; int stride; } pools[] = {
      {"pool max 2x2 s2", CUDNN_POOLING_MAX, 2, 0, 2},
      {"pool max 3x3 p1 s1", CUDNN_POOLING_MAX, 3, 1, 1},
      {"pool avg excl pad", CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING, 3, 1, 1},
      {"pool avg incl pad", CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING, 3, 1, 1},
  };
  for (auto& t : pools) {
    cudnnPoolingDescriptor_t pd;
    CK(cudnnCreatePoolingDescriptor(&pd));
    CK(cudnnSetPooling2dDescriptor(pd, t.mode, CUDNN_NOT_PROPAGATE_NAN, t.win, t.win, t.pad, t.pad,
                                   t.stride, t.stride));
    int on, oc, oh, ow;
    CK(cudnnGetPooling2dForwardOutputDim(pd, td, &on, &oc, &oh, &ow));
    cudnnTensorDescriptor_t yd;
    CK(cudnnCreateTensorDescriptor(&yd));
    CK(cudnnSetTensor4dDescriptor(yd, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, on, oc, oh, ow));
    Dev y((size_t)on * oc * oh * ow);
    const float a = 1.0f, b = 0.0f;
    CK(cudnnPoolingForward(h, pd, &a, td, x.p, &b, yd, y.p));
    printf("%-28s outdim=%dx%dx%dx%d\n", t.name, on, oc, oh, ow);
    dump(t.name, y.get((size_t)on * oc * oh * ow));
    cudnnDestroyTensorDescriptor(yd);
    cudnnDestroyPoolingDescriptor(pd);
  }

  struct { const char* name; cudnnSoftmaxAlgorithm_t algo; cudnnSoftmaxMode_t mode; } sms[] = {
      {"softmax accurate channel", CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL},
      {"softmax accurate instance", CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE},
      {"softmax log channel", CUDNN_SOFTMAX_LOG, CUDNN_SOFTMAX_MODE_CHANNEL},
      {"softmax fast instance", CUDNN_SOFTMAX_FAST, CUDNN_SOFTMAX_MODE_INSTANCE},
  };
  for (auto& t : sms) {
    Dev y(n);
    const float a = 1.0f, b = 0.0f;
    CK(cudnnSoftmaxForward(h, t.algo, t.mode, &a, td, x.p, &b, td, y.p));
    dump(t.name, y.get(n));
  }

  // Batch normalization, inference and training, spatial and per-activation.
  for (int m = 0; m < 2; ++m) {
    const cudnnBatchNormMode_t mode = m ? CUDNN_BATCHNORM_PER_ACTIVATION : CUDNN_BATCHNORM_SPATIAL;
    const char* mn = m ? "per-activation" : "spatial";
    cudnnTensorDescriptor_t bnd;
    CK(cudnnCreateTensorDescriptor(&bnd));
    CK(cudnnDeriveBNTensorDescriptor(bnd, td, mode));
    int dn, dc, dh, dw, s0, s1, s2, s3;
    cudnnDataType_t dt;
    CK(cudnnGetTensor4dDescriptor(bnd, &dt, &dn, &dc, &dh, &dw, &s0, &s1, &s2, &s3));
    printf("bn %-25s params=%dx%dx%dx%d\n", mn, dn, dc, dh, dw);
    const size_t np = (size_t)dn * dc * dh * dw;
    Dev scale(np, 0.8f), bias(np, 0.4f), mean(np, 0.2f), var(np);
    {  // variance must be positive
      std::vector<float> v(np);
      for (size_t i = 0; i < np; ++i) v[i] = 0.5f + 0.1f * (i % 7);
      cudaMemcpy(var.p, v.data(), np * sizeof(float), cudaMemcpyHostToDevice);
    }
    Dev y(n);
    const float a = 1.0f, b = 0.0f;
    char tag[64];
    snprintf(tag, sizeof(tag), "bn infer %s", mn);
    CK(cudnnBatchNormalizationForwardInference(h, mode, &a, &b, td, x.p, td, y.p, bnd, scale.p,
                                               bias.p, mean.p, var.p, 1e-5));
    dump(tag, y.get(n));

    Dev rm(np, 0.3f), rv(np, 1.0f), sm(np), siv(np), y2(n);
    snprintf(tag, sizeof(tag), "bn train %s", mn);
    CK(cudnnBatchNormalizationForwardTraining(h, mode, &a, &b, td, x.p, td, y2.p, bnd, scale.p,
                                              bias.p, 0.1, rm.p, rv.p, 1e-5, sm.p, siv.p));
    dump(tag, y2.get(n));
    snprintf(tag, sizeof(tag), "bn train %s savedmean", mn);
    dump(tag, sm.get(np));
    snprintf(tag, sizeof(tag), "bn train %s running", mn);
    dump(tag, rm.get(np));
    cudnnDestroyTensorDescriptor(bnd);
  }

  struct { const char* name; cudnnOpTensorOp_t op; } ops[] = {
      {"optensor add", CUDNN_OP_TENSOR_ADD},
      {"optensor mul", CUDNN_OP_TENSOR_MUL},
      {"optensor min", CUDNN_OP_TENSOR_MIN},
      {"optensor max", CUDNN_OP_TENSOR_MAX},
  };
  for (auto& t : ops) {
    cudnnOpTensorDescriptor_t od;
    CK(cudnnCreateOpTensorDescriptor(&od));
    CK(cudnnSetOpTensorDescriptor(od, t.op, CUDNN_DATA_FLOAT, CUDNN_NOT_PROPAGATE_NAN));
    Dev bias(C, 3.0f), y(n);
    const float a1 = 1.0f, a2 = 2.0f, b = 0.0f;
    CK(cudnnOpTensor(h, od, &a1, td, x.p, &a2, bd, bias.p, &b, td, y.p));
    dump(t.name, y.get(n));
    cudnnDestroyOpTensorDescriptor(od);
  }

  {
    Dev y(n);
    const float a = 3.0f, b = 0.0f;
    CK(cudnnTransformTensor(h, &a, td, x.p, &b, td, y.p));
    dump("transform scale3", y.get(n));
    const float s = 0.5f;
    CK(cudnnScaleTensor(h, td, y.p, &s));
    dump("scaletensor 0.5", y.get(n));
    const float v = 1.25f;
    CK(cudnnSetTensor(h, td, y.p, &v));
    dump("settensor 1.25", y.get(n));
  }

  cudnnDestroyTensorDescriptor(bd);
  cudnnDestroyTensorDescriptor(td);
  cudnnDestroy(h);
}

int main() {
  // Major only: the reference library on the machine is rarely the exact
  // patch release whose headers are vendored here, and these ops are stable
  // across all of cuDNN 9.x. A version difference is not a semantics difference.
  printf("cudnn major %zu\n", cudnnGetVersion() / 10000);
  run();
  return 0;
}
