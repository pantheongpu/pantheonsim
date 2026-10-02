// Differential conformance for the cuDNN shim's reduced-precision types:
// half (with float and with half compute), bfloat16 and INT8 convolution
// forward and backward in NCHW and NHWC, and half through activation,
// pooling, softmax, batch normalization and op-tensor, plus the exact bits of
// float-to-half and float-to-bfloat16 conversion. The same binary runs
// against NVIDIA's libcudnn.so.9 and against VirtualGPU's
// (nvidia/tests/conformance/golden/cudnn_types.rtx3060.txt holds what an RTX
// 3060 printed). Half results are compared to half precision: the summation
// order inside a reduction is the hardware's own, and in half it shows.
#include <cudnn.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
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

static float to_f(float v) { return v; }
static float to_f(__half v) { return __half2float(v); }
static float to_f(__nv_bfloat16 v) { return __bfloat162float(v); }
template <class T> static T from_f(float v);
template <> float from_f<float>(float v) { return v; }
template <> __half from_f<__half>(float v) { return __float2half(v); }
template <> __nv_bfloat16 from_f<__nv_bfloat16>(float v) { return __float2bfloat16(v); }
template <> int8_t from_f<int8_t>(float v) { return (int8_t)std::lrint(v); }

template <class T>
static void dump(const char* tag, const std::vector<T>& v) {
  double a = 0, q = 0, w = 0;
  for (size_t i = 0; i < v.size(); ++i) {
    const double x = to_f(v[i]);
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
    for (size_t i = 0; i < n; ++i) h[i] = from_f<T>(scale == 0.0f ? 0.0f : fill((int)i + seed, scale));
    cudaMemcpy(p, h.data(), n * sizeof(T), cudaMemcpyHostToDevice);
  }
  std::vector<T> get() const {
    std::vector<T> h(n);
    cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost);
    return h;
  }
  ~Buf() { cudaFree(p); }
};

static cudnnTensorDescriptor_t t4(cudnnTensorFormat_t f, cudnnDataType_t t, int n, int c, int h, int w) {
  cudnnTensorDescriptor_t d;
  cudnnCreateTensorDescriptor(&d);
  cudnnStatus_t s = cudnnSetTensor4dDescriptor(d, f, t, n, c, h, w);
  if (s) std::printf("SetTensor4d -> %d\n", (int)s);
  return d;
}

template <class T>
static void conv(const char* tag, cudnnDataType_t dt, cudnnDataType_t ct, cudnnTensorFormat_t f, int N, int C, int H_,
                 int W, int K, int R, int pad, int stride) {
  cudnnTensorDescriptor_t xd = t4(f, dt, N, C, H_, W);
  cudnnFilterDescriptor_t wd;
  cudnnCreateFilterDescriptor(&wd);
  CK(cudnnSetFilter4dDescriptor(wd, dt, f, K, C, R, R));
  cudnnConvolutionDescriptor_t cd;
  cudnnCreateConvolutionDescriptor(&cd);
  CK(cudnnSetConvolution2dDescriptor(cd, pad, pad, stride, stride, 1, 1, CUDNN_CROSS_CORRELATION, ct));
  int n, c, h, w;
  CK(cudnnGetConvolution2dForwardOutputDim(cd, xd, wd, &n, &c, &h, &w));
  cudnnTensorDescriptor_t yd = t4(f, dt, n, c, h, w);
  const size_t xn = (size_t)N * C * H_ * W, wn = (size_t)K * C * R * R, yn = (size_t)n * c * h * w;
  Buf<T> x(xn, 1.0f), wt(wn, 0.5f, 3), dy(yn, 0.8f, 7), y(yn, 0.0f), dx(xn, 0.0f), dw(wn, 0.0f);
  const float one = 1.0f, zero = 0.0f;
  char t[96];
  CK(cudnnConvolutionForward(H, &one, xd, x.p, wd, wt.p, cd, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, nullptr, 0, &zero,
                             yd, y.p));
  std::snprintf(t, sizeof t, "%s fwd", tag);
  dump(t, y.get());
  CK(cudnnConvolutionBackwardData(H, &one, wd, wt.p, yd, dy.p, cd, CUDNN_CONVOLUTION_BWD_DATA_ALGO_0, nullptr, 0,
                                  &zero, xd, dx.p));
  std::snprintf(t, sizeof t, "%s bwd-data", tag);
  dump(t, dx.get());
  // With half compute the hardware runs backward-filter only as ALGO_1.
  const auto falgo = ct == CUDNN_DATA_HALF ? CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1 : CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
  CK(cudnnConvolutionBackwardFilter(H, &one, xd, x.p, yd, dy.p, cd, falgo, nullptr, 0, &zero, wd, dw.p));
  std::snprintf(t, sizeof t, "%s bwd-filter", tag);
  dump(t, dw.get());
}

// INT8: NHWC only, INT32 compute; INT8 output rounds to nearest even and
// saturates, FLOAT output is exact.
static void int8_conv() {
  cudnnTensorDescriptor_t xd = t4(CUDNN_TENSOR_NHWC, CUDNN_DATA_INT8, 2, 8, 5, 5);
  cudnnTensorDescriptor_t y8 = t4(CUDNN_TENSOR_NHWC, CUDNN_DATA_INT8, 2, 4, 5, 5);
  cudnnTensorDescriptor_t yf = t4(CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT, 2, 4, 5, 5);
  cudnnFilterDescriptor_t wd;
  cudnnCreateFilterDescriptor(&wd);
  CK(cudnnSetFilter4dDescriptor(wd, CUDNN_DATA_INT8, CUDNN_TENSOR_NHWC, 4, 8, 3, 3));
  cudnnConvolutionDescriptor_t cd;
  cudnnCreateConvolutionDescriptor(&cd);
  CK(cudnnSetConvolution2dDescriptor(cd, 1, 1, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_INT32));
  Buf<int8_t> x(400, 20.0f), w(288, 3.0f, 5), y(200, 0.0f);
  Buf<float> f(200, 0.0f);
  const float zero = 0.0f;
  for (float alpha : {1.0f, 0.05f, 0.0123f}) {
    CK(cudnnConvolutionForward(H, &alpha, xd, x.p, wd, w.p, cd, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
                               nullptr, 0, &zero, y8, y.p));
    std::printf("int8 conv alpha %.4f int8 out:", alpha);
    const auto v = y.get();
    long sum = 0;
    for (int8_t e : v) sum += e;
    std::printf(" sum %ld first", sum);
    for (int i = 0; i < 12; ++i) std::printf(" %d", v[i]);
    std::printf("\n");
    CK(cudnnConvolutionForward(H, &alpha, xd, x.p, wd, w.p, cd, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
                               nullptr, 0, &zero, yf, f.p));
    char t[64];
    std::snprintf(t, sizeof t, "int8 conv alpha %.4f float out", alpha);
    dump(t, f.get());
  }
  cudnnTensorDescriptor_t xn = t4(CUDNN_TENSOR_NCHW, CUDNN_DATA_INT8, 2, 8, 5, 5);
  const float one = 1.0f;
  std::printf("int8 conv in NCHW: status %d\n",
              (int)cudnnConvolutionForward(H, &one, xn, x.p, wd, w.p, cd, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
                                           nullptr, 0, &zero, y8, y.p));
}

static void half_layers() {
  const float one = 1.0f, zero = 0.0f, two = 2.0f, half = 0.5f;
  cudnnTensorDescriptor_t d = t4(CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, 2, 3, 4, 5);
  cudnnActivationDescriptor_t ad;
  cudnnCreateActivationDescriptor(&ad);
  for (auto m : {CUDNN_ACTIVATION_RELU, CUDNN_ACTIVATION_SIGMOID, CUDNN_ACTIVATION_TANH, CUDNN_ACTIVATION_ELU}) {
    CK(cudnnSetActivationDescriptor(ad, m, CUDNN_NOT_PROPAGATE_NAN, 0.7));
    Buf<__half> x(120, 1.5f), y(120, 0.0f), dy(120, 1.0f, 4), dx(120, 0.5f, 8);
    CK(cudnnActivationForward(H, ad, &one, d, x.p, &zero, d, y.p));
    CK(cudnnActivationBackward(H, ad, &two, d, y.p, d, dy.p, d, x.p, &half, d, dx.p));
    char t[64];
    std::snprintf(t, sizeof t, "half act %d fwd", (int)m);
    dump(t, y.get());
    std::snprintf(t, sizeof t, "half act %d bwd", (int)m);
    dump(t, dx.get());
  }
  cudnnPoolingDescriptor_t pd;
  cudnnCreatePoolingDescriptor(&pd);
  cudnnTensorDescriptor_t px = t4(CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, 2, 3, 6, 6);
  cudnnTensorDescriptor_t py = t4(CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF, 2, 3, 3, 3);
  for (auto m : {CUDNN_POOLING_MAX, CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING}) {
    CK(cudnnSetPooling2dDescriptor(pd, m, CUDNN_NOT_PROPAGATE_NAN, 3, 3, 1, 1, 2, 2));
    Buf<__half> x(216, 2.0f), y(54, 0.0f), dy(54, 1.0f, 3), dx(216, 0.0f);
    CK(cudnnPoolingForward(H, pd, &one, px, x.p, &zero, py, y.p));
    CK(cudnnPoolingBackward(H, pd, &one, py, y.p, py, dy.p, px, x.p, &zero, px, dx.p));
    char t[64];
    std::snprintf(t, sizeof t, "half pool %d fwd", (int)m);
    dump(t, y.get());
    std::snprintf(t, sizeof t, "half pool %d bwd", (int)m);
    dump(t, dx.get());
  }
  for (auto mode : {CUDNN_SOFTMAX_MODE_INSTANCE, CUDNN_SOFTMAX_MODE_CHANNEL}) {
    Buf<__half> x(120, 2.0f), y(120, 0.0f), dy(120, 1.0f, 9), dx(120, 0.0f);
    CK(cudnnSoftmaxForward(H, CUDNN_SOFTMAX_ACCURATE, mode, &one, d, x.p, &zero, d, y.p));
    CK(cudnnSoftmaxBackward(H, CUDNN_SOFTMAX_ACCURATE, mode, &one, d, y.p, d, dy.p, &zero, d, dx.p));
    char t[64];
    std::snprintf(t, sizeof t, "half softmax mode %d fwd", (int)mode);
    dump(t, y.get());
    std::snprintf(t, sizeof t, "half softmax mode %d bwd", (int)mode);
    dump(t, dx.get());
  }
  // Batch normalization: half data, float parameters.
  cudnnTensorDescriptor_t bnd;
  cudnnCreateTensorDescriptor(&bnd);
  CK(cudnnDeriveBNTensorDescriptor(bnd, d, CUDNN_BATCHNORM_SPATIAL));
  cudnnDataType_t pt;
  int n, c, h, w, s0, s1, s2, s3;
  CK(cudnnGetTensor4dDescriptor(bnd, &pt, &n, &c, &h, &w, &s0, &s1, &s2, &s3));
  std::printf("half bn parameters: type %d dims %d %d %d %d\n", (int)pt, n, c, h, w);
  Buf<__half> x(120, 1.5f), y(120, 0.0f), dy(120, 1.0f, 4), dx(120, 0.0f);
  Buf<float> scale(3, 0.5f, 3), bias(3, 0.2f, 5), mean(3, 0.0f), inv(3, 0.0f), dscale(3, 0.0f), dbias(3, 0.0f);
  CK(cudnnBatchNormalizationForwardTraining(H, CUDNN_BATCHNORM_SPATIAL, &one, &zero, d, x.p, d, y.p, bnd, scale.p,
                                            bias.p, 0.1, nullptr, nullptr, 1e-5, mean.p, inv.p));
  CK(cudnnBatchNormalizationBackward(H, CUDNN_BATCHNORM_SPATIAL, &one, &zero, &one, &zero, d, x.p, d, dy.p, d, dx.p,
                                     bnd, scale.p, dscale.p, dbias.p, 1e-5, mean.p, inv.p));
  dump("half bn fwd", y.get());
  dump("half bn saved mean", mean.get());
  dump("half bn dx", dx.get());
  dump("half bn dscale", dscale.get());
  dump("half bn dbias", dbias.get());
  // Op tensor: half data, float compute.
  cudnnOpTensorDescriptor_t od;
  cudnnCreateOpTensorDescriptor(&od);
  CK(cudnnSetOpTensorDescriptor(od, CUDNN_OP_TENSOR_MUL, CUDNN_DATA_FLOAT, CUDNN_NOT_PROPAGATE_NAN));
  cudnnTensorDescriptor_t bc = t4(CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, 1, 3, 1, 1);
  Buf<__half> A(120, 1.5f), B(3, 1.0f, 2), C(120, 1.0f, 6);
  CK(cudnnOpTensor(H, od, &two, d, A.p, &one, bc, B.p, &half, d, C.p));
  dump("half optensor mul broadcast", C.get());
  CK(cudnnSetOpTensorDescriptor(od, CUDNN_OP_TENSOR_MUL, CUDNN_DATA_HALF, CUDNN_NOT_PROPAGATE_NAN));
  std::printf("half optensor with half compute: status %d\n",
              (int)cudnnOpTensor(H, od, &two, d, A.p, &one, bc, B.p, &half, d, C.p));
}

// Exact conversion bits: float to half and to bfloat16, round to nearest even.
static void conversions() {
  const float vals[] = {1.0f, 2.5f, -3.25f, 1e-3f, 65519.0f, 65520.0f, 1e-8f, 3.14159265f, 1.00048828125f,
                        1.0009765625f + 0.00048828125f, -0.0f, 7e-5f};
  const int n = sizeof vals / sizeof vals[0];
  cudnnTensorDescriptor_t f = t4(CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 1, n);
  cudnnTensorDescriptor_t h = t4(CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, 1, 1, 1, n);
  cudnnTensorDescriptor_t b = t4(CUDNN_TENSOR_NCHW, CUDNN_DATA_BFLOAT16, 1, 1, 1, n);
  float* src;
  uint16_t *dh, *db;
  cudaMalloc(&src, sizeof vals);
  cudaMalloc(&dh, 2 * n);
  cudaMalloc(&db, 2 * n);
  cudaMemcpy(src, vals, sizeof vals, cudaMemcpyHostToDevice);
  const float one = 1.0f, zero = 0.0f;
  CK(cudnnTransformTensor(H, &one, f, src, &zero, h, dh));
  CK(cudnnTransformTensor(H, &one, f, src, &zero, b, db));
  std::vector<uint16_t> rh(n), rb(n);
  cudaMemcpy(rh.data(), dh, 2 * n, cudaMemcpyDeviceToHost);
  cudaMemcpy(rb.data(), db, 2 * n, cudaMemcpyDeviceToHost);
  std::printf("float->half bits:");
  for (uint16_t v : rh) std::printf(" %04x", v);
  std::printf("\nfloat->bf16 bits:");
  for (uint16_t v : rb) std::printf(" %04x", v);
  std::printf("\n");
  // And the value SetTensor takes for half (a float).
  const float v = 1.5f;
  CK(cudnnSetTensor(H, h, dh, &v));
  cudaMemcpy(rh.data(), dh, 2 * n, cudaMemcpyDeviceToHost);
  std::printf("settensor half 1.5: %04x\n", rh[0]);
}

int main() {
  if (cudnnCreate(&H) != CUDNN_STATUS_SUCCESS) {
    std::printf("cudnnCreate failed\n");
    return 1;
  }
  conv<__half>("half nchw pseudo", CUDNN_DATA_HALF, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 2, 4, 6, 6, 3, 3, 1, 1);
  conv<__half>("half nhwc pseudo s2", CUDNN_DATA_HALF, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NHWC, 2, 4, 7, 7, 4, 3, 1, 2);
  conv<__half>("half nchw true-half 1x1", CUDNN_DATA_HALF, CUDNN_DATA_HALF, CUDNN_TENSOR_NCHW, 1, 2, 4, 4, 2, 1, 0, 1);
  conv<__nv_bfloat16>("bf16 nhwc", CUDNN_DATA_BFLOAT16, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NHWC, 2, 4, 6, 6, 3, 3, 1, 1);
  int8_conv();
  half_layers();
  conversions();
  cudnnDestroy(H);
  return 0;
}
