// libvgpucudnn — VirtualGPU's implementation of the classic cuDNN API.
//
// Same boundary as cuBLAS: cuDNN is a library, so its math runs on the host
// and reads/writes virtual device memory rather than being interpreted. That
// keeps convolution and the pointwise layers off the ~10^8 lane-ops/s path.
//
// Scope is the legacy (descriptor) API that frameworks still use for
// convolution, activation, pooling, softmax, batch normalization and tensor
// arithmetic, in NCHW float. The graph/backend API of cuDNN 8+ is not
// implemented. Anything unimplemented returns CUDNN_STATUS_NOT_SUPPORTED so a
// caller can fall back, rather than receiving a plausible wrong answer.
//
// Compiled against the real cuDNN headers (nvidia/third_party/cudnn_include) so the
// ABI is the vendor's, not a guess.
#include <cudnn.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>
#include <vector>

#include <cuda_runtime.h>

namespace {

bool quiet() {
  const char* q = std::getenv("VGPU_QUIET");
  return q && q[0] == '1';
}
bool trace() {
  const char* t = std::getenv("VGPU_TRACE");
  return t && t[0] == '1';
}

struct Handle { cudaStream_t stream = nullptr; };

struct TensorDesc {
  cudnnDataType_t type = CUDNN_DATA_FLOAT;
  int n = 0, c = 0, h = 0, w = 0;
  int sn = 0, sc = 0, sh = 0, sw = 0;  // strides, in elements
  // As cudnnSetTensorNdDescriptor was given them, for GetTensorNdDescriptor;
  // nb == 0 for a descriptor set as 4-D.
  int nb = 0;
  int dims[CUDNN_DIM_MAX] = {}, strides[CUDNN_DIM_MAX] = {};
};

struct FilterDesc {
  cudnnDataType_t type = CUDNN_DATA_FLOAT;
  cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW;
  int k = 0, c = 0, h = 0, w = 0;
};

struct ConvDesc {
  int pad_h = 0, pad_w = 0, stride_h = 1, stride_w = 1, dil_h = 1, dil_w = 1;
  cudnnConvolutionMode_t mode = CUDNN_CROSS_CORRELATION;
  int groups = 1;
};

struct ActDesc {
  cudnnActivationMode_t mode = CUDNN_ACTIVATION_RELU;
  double coef = 0.0;
  float swish_beta = 1.0f;  // separate setter in cuDNN, so it survives SetActivationDescriptor
};

struct PoolDesc {
  cudnnPoolingMode_t mode = CUDNN_POOLING_MAX;
  int wh = 1, ww = 1, ph = 0, pw = 0, sh = 1, sw = 1;
};

std::mutex g_mu;
std::set<void*> g_live;
template <class T> T* track(T* p) { std::lock_guard<std::mutex> l(g_mu); g_live.insert(p); return p; }
bool known(const void* p) { std::lock_guard<std::mutex> l(g_mu); return p && g_live.count(const_cast<void*>(p)); }
void untrack(void* p) { std::lock_guard<std::mutex> l(g_mu); g_live.erase(p); }

std::vector<float> fetch(const void* dev, size_t n) {
  std::vector<float> h(n);
  if (n) cudaMemcpy(h.data(), dev, n * sizeof(float), cudaMemcpyDeviceToHost);
  return h;
}
void store(void* dev, const std::vector<float>& h) {
  if (!h.empty()) cudaMemcpy(dev, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice);
}
size_t elems(const TensorDesc& t) { return (size_t)t.n * t.c * t.h * t.w; }
// Index honoring the descriptor's strides, so non-packed tensors work.
inline size_t at(const TensorDesc& t, int n, int c, int h, int w) {
  return (size_t)n * t.sn + (size_t)c * t.sc + (size_t)h * t.sh + (size_t)w * t.sw;
}

float alpha_of(const void* p) { return p ? *static_cast<const float*>(p) : 1.0f; }

}  // namespace

#define VGPU_EXPORT extern "C" __attribute__((visibility("default")))

/* ---- lifecycle ---- */

VGPU_EXPORT cudnnStatus_t cudnnCreate(cudnnHandle_t* h) {
  if (!h) return CUDNN_STATUS_BAD_PARAM;
  *h = reinterpret_cast<cudnnHandle_t>(track(new Handle()));
  if (!quiet())
    std::fprintf(stderr, "[vgpu] cuDNN handle created (host-computed, legacy API only; "
                         "see nvidia/docs/libraries.md)\n");
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnDestroy(cudnnHandle_t h) {
  if (!known(h)) return CUDNN_STATUS_NOT_INITIALIZED;
  untrack(h); delete reinterpret_cast<Handle*>(h);
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnSetStream(cudnnHandle_t h, cudaStream_t s) {
  if (!known(h)) return CUDNN_STATUS_NOT_INITIALIZED;
  reinterpret_cast<Handle*>(h)->stream = s;
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnGetStream(cudnnHandle_t h, cudaStream_t* s) {
  if (!known(h)) return CUDNN_STATUS_NOT_INITIALIZED;
  if (s) *s = reinterpret_cast<Handle*>(h)->stream;
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnGetProperty(libraryPropertyType type, int* value) {
  if (!value) return CUDNN_STATUS_BAD_PARAM;
  switch (type) {
    // The release whose headers this is built against, as cudnnGetVersion
    // reports it: PyTorch refuses a runtime older than the one it was built
    // with, and compares these.
    case MAJOR_VERSION: *value = CUDNN_MAJOR; break;
    case MINOR_VERSION: *value = CUDNN_MINOR; break;
    case PATCH_LEVEL: *value = CUDNN_PATCHLEVEL; break;
    default: return CUDNN_STATUS_BAD_PARAM;
  }
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT size_t cudnnGetVersion(void) { return CUDNN_VERSION; }
VGPU_EXPORT size_t cudnnGetCudartVersion(void) { return 13000; }
VGPU_EXPORT const char* cudnnGetErrorString(cudnnStatus_t s) {
  switch (s) {
    case CUDNN_STATUS_SUCCESS: return "CUDNN_STATUS_SUCCESS";
    case CUDNN_STATUS_NOT_INITIALIZED: return "CUDNN_STATUS_NOT_INITIALIZED";
    case CUDNN_STATUS_BAD_PARAM: return "CUDNN_STATUS_BAD_PARAM";
    case CUDNN_STATUS_NOT_SUPPORTED: return "CUDNN_STATUS_NOT_SUPPORTED";
    default: return "CUDNN_STATUS_INTERNAL_ERROR";
  }
}

/* ---- descriptors ---- */

VGPU_EXPORT cudnnStatus_t cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t* d) {
  if (!d) return CUDNN_STATUS_BAD_PARAM;
  *d = reinterpret_cast<cudnnTensorDescriptor_t>(track(new TensorDesc()));
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t d) {
  if (!known(d)) return CUDNN_STATUS_BAD_PARAM;
  untrack(d); delete reinterpret_cast<TensorDesc*>(d);
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnSetTensor4dDescriptor(cudnnTensorDescriptor_t d,
                                                     cudnnTensorFormat_t fmt, cudnnDataType_t type,
                                                     int n, int c, int h, int w) {
  if (!known(d)) return CUDNN_STATUS_BAD_PARAM;
  if (type != CUDNN_DATA_FLOAT) return CUDNN_STATUS_NOT_SUPPORTED;
  if (fmt != CUDNN_TENSOR_NCHW) return CUDNN_STATUS_NOT_SUPPORTED;
  auto* t = reinterpret_cast<TensorDesc*>(d);
  *t = TensorDesc{type, n, c, h, w, c * h * w, h * w, w, 1};
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnSetTensor4dDescriptorEx(cudnnTensorDescriptor_t d,
                                                       cudnnDataType_t type, int n, int c, int h,
                                                       int w, int sn, int sc, int sh, int sw) {
  if (!known(d)) return CUDNN_STATUS_BAD_PARAM;
  if (type != CUDNN_DATA_FLOAT) return CUDNN_STATUS_NOT_SUPPORTED;
  *reinterpret_cast<TensorDesc*>(d) = TensorDesc{type, n, c, h, w, sn, sc, sh, sw};
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnGetTensor4dDescriptor(cudnnTensorDescriptor_t d,
                                                     cudnnDataType_t* type, int* n, int* c, int* h,
                                                     int* w, int* sn, int* sc, int* sh, int* sw) {
  if (!known(d)) return CUDNN_STATUS_BAD_PARAM;
  auto* t = reinterpret_cast<TensorDesc*>(d);
  if (type) *type = t->type;
  if (n) *n = t->n; if (c) *c = t->c; if (h) *h = t->h; if (w) *w = t->w;
  if (sn) *sn = t->sn; if (sc) *sc = t->sc; if (sh) *sh = t->sh; if (sw) *sw = t->sw;
  return CUDNN_STATUS_SUCCESS;
}

VGPU_EXPORT cudnnStatus_t cudnnCreateFilterDescriptor(cudnnFilterDescriptor_t* d) {
  if (!d) return CUDNN_STATUS_BAD_PARAM;
  *d = reinterpret_cast<cudnnFilterDescriptor_t>(track(new FilterDesc()));
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnDestroyFilterDescriptor(cudnnFilterDescriptor_t d) {
  if (!known(d)) return CUDNN_STATUS_BAD_PARAM;
  untrack(d); delete reinterpret_cast<FilterDesc*>(d);
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnSetFilter4dDescriptor(cudnnFilterDescriptor_t d,
                                                     cudnnDataType_t type, cudnnTensorFormat_t fmt,
                                                     int k, int c, int h, int w) {
  if (!known(d)) return CUDNN_STATUS_BAD_PARAM;
  if (type != CUDNN_DATA_FLOAT || fmt != CUDNN_TENSOR_NCHW) return CUDNN_STATUS_NOT_SUPPORTED;
  *reinterpret_cast<FilterDesc*>(d) = FilterDesc{type, fmt, k, c, h, w};
  return CUDNN_STATUS_SUCCESS;
}

VGPU_EXPORT cudnnStatus_t cudnnCreateConvolutionDescriptor(cudnnConvolutionDescriptor_t* d) {
  if (!d) return CUDNN_STATUS_BAD_PARAM;
  *d = reinterpret_cast<cudnnConvolutionDescriptor_t>(track(new ConvDesc()));
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnDestroyConvolutionDescriptor(cudnnConvolutionDescriptor_t d) {
  if (!known(d)) return CUDNN_STATUS_BAD_PARAM;
  untrack(d); delete reinterpret_cast<ConvDesc*>(d);
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnSetConvolution2dDescriptor(cudnnConvolutionDescriptor_t d, int ph,
                                                          int pw, int sh, int sw, int dh, int dw,
                                                          cudnnConvolutionMode_t mode,
                                                          cudnnDataType_t type) {
  if (!known(d)) return CUDNN_STATUS_BAD_PARAM;
  if (type != CUDNN_DATA_FLOAT) return CUDNN_STATUS_NOT_SUPPORTED;
  auto* c = reinterpret_cast<ConvDesc*>(d);
  *c = ConvDesc{ph, pw, sh, sw, dh, dw, mode, c->groups};
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnSetConvolutionGroupCount(cudnnConvolutionDescriptor_t d, int g) {
  if (!known(d)) return CUDNN_STATUS_BAD_PARAM;
  reinterpret_cast<ConvDesc*>(d)->groups = g;
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnSetConvolutionMathType(cudnnConvolutionDescriptor_t d, cudnnMathType_t) {
  return known(d) ? CUDNN_STATUS_SUCCESS : CUDNN_STATUS_BAD_PARAM;
}

VGPU_EXPORT cudnnStatus_t cudnnGetConvolution2dForwardOutputDim(
    const cudnnConvolutionDescriptor_t cd, const cudnnTensorDescriptor_t xd,
    const cudnnFilterDescriptor_t fd, int* n, int* c, int* h, int* w) {
  if (!known(cd) || !known(xd) || !known(fd)) return CUDNN_STATUS_BAD_PARAM;
  auto* cv = reinterpret_cast<ConvDesc*>(cd);
  auto* x = reinterpret_cast<TensorDesc*>(xd);
  auto* f = reinterpret_cast<FilterDesc*>(fd);
  if (n) *n = x->n;
  if (c) *c = f->k;
  if (h) *h = 1 + (x->h + 2 * cv->pad_h - ((f->h - 1) * cv->dil_h + 1)) / cv->stride_h;
  if (w) *w = 1 + (x->w + 2 * cv->pad_w - ((f->w - 1) * cv->dil_w + 1)) / cv->stride_w;
  return CUDNN_STATUS_SUCCESS;
}

/* ---- algorithm selection: there is one, and it needs no workspace ---- */

VGPU_EXPORT cudnnStatus_t cudnnGetConvolutionForwardWorkspaceSize(
    cudnnHandle_t, const cudnnTensorDescriptor_t, const cudnnFilterDescriptor_t,
    const cudnnConvolutionDescriptor_t, const cudnnTensorDescriptor_t, cudnnConvolutionFwdAlgo_t,
    size_t* bytes) {
  if (bytes) *bytes = 0;
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnGetConvolutionForwardAlgorithm_v7(
    cudnnHandle_t, const cudnnTensorDescriptor_t, const cudnnFilterDescriptor_t,
    const cudnnConvolutionDescriptor_t, const cudnnTensorDescriptor_t, int requested, int* returned,
    cudnnConvolutionFwdAlgoPerf_t* perf) {
  if (!perf || !returned || requested < 1) return CUDNN_STATUS_BAD_PARAM;
  std::memset(&perf[0], 0, sizeof(perf[0]));
  perf[0].algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
  perf[0].status = CUDNN_STATUS_SUCCESS;
  perf[0].memory = 0;
  perf[0].mathType = CUDNN_DEFAULT_MATH;
  *returned = 1;
  return CUDNN_STATUS_SUCCESS;
}

/* ---- the operations ---- */

VGPU_EXPORT cudnnStatus_t cudnnConvolutionForward(
    cudnnHandle_t h, const void* alpha, const cudnnTensorDescriptor_t xd, const void* x,
    const cudnnFilterDescriptor_t wd, const void* w, const cudnnConvolutionDescriptor_t cd,
    cudnnConvolutionFwdAlgo_t, void*, size_t, const void* beta,
    const cudnnTensorDescriptor_t yd, void* y) {
  if (!known(h) || !known(xd) || !known(wd) || !known(cd) || !known(yd))
    return CUDNN_STATUS_NOT_INITIALIZED;
  auto* X = reinterpret_cast<TensorDesc*>(xd);
  auto* W = reinterpret_cast<FilterDesc*>(wd);
  auto* C = reinterpret_cast<ConvDesc*>(cd);
  auto* Y = reinterpret_cast<TensorDesc*>(yd);
  const float a = alpha_of(alpha), b = alpha_of(beta);
  if (C->groups < 1 || W->c * C->groups != X->c) return CUDNN_STATUS_BAD_PARAM;

  auto hx = fetch(x, elems(*X));
  auto hw = fetch(w, (size_t)W->k * W->c * W->h * W->w);
  auto hy = (b != 0.0f) ? fetch(y, elems(*Y)) : std::vector<float>(elems(*Y), 0.0f);

  const int per_group_k = W->k / std::max(C->groups, 1);
  for (int n = 0; n < Y->n; ++n)
    for (int k = 0; k < Y->c; ++k) {
      const int g = per_group_k ? k / per_group_k : 0;
      for (int oh = 0; oh < Y->h; ++oh)
        for (int ow = 0; ow < Y->w; ++ow) {
          float acc = 0.0f;
          for (int ci = 0; ci < W->c; ++ci)
            for (int fh = 0; fh < W->h; ++fh)
              for (int fw = 0; fw < W->w; ++fw) {
                // CUDNN_CONVOLUTION flips the kernel; cross-correlation does not.
                const int kh = C->mode == CUDNN_CONVOLUTION ? W->h - 1 - fh : fh;
                const int kw = C->mode == CUDNN_CONVOLUTION ? W->w - 1 - fw : fw;
                const int ih = oh * C->stride_h - C->pad_h + kh * C->dil_h;
                const int iw = ow * C->stride_w - C->pad_w + kw * C->dil_w;
                if (ih < 0 || ih >= X->h || iw < 0 || iw >= X->w) continue;  // zero padding
                const int in_c = g * W->c + ci;
                acc += hx[at(*X, n, in_c, ih, iw)] *
                       hw[(((size_t)k * W->c + ci) * W->h + fh) * W->w + fw];
              }
          float& out = hy[at(*Y, n, k, oh, ow)];
          out = b == 0.0f ? a * acc : a * acc + b * out;
        }
    }
  store(y, hy);
  return CUDNN_STATUS_SUCCESS;
}

VGPU_EXPORT cudnnStatus_t cudnnAddTensor(cudnnHandle_t h, const void* alpha,
                                         const cudnnTensorDescriptor_t ad, const void* A,
                                         const void* beta, const cudnnTensorDescriptor_t cd,
                                         void* C) {
  if (!known(h) || !known(ad) || !known(cd)) return CUDNN_STATUS_NOT_INITIALIZED;
  auto* Ad = reinterpret_cast<TensorDesc*>(ad);
  auto* Cd = reinterpret_cast<TensorDesc*>(cd);
  const float a = alpha_of(alpha), b = alpha_of(beta);
  auto hA = fetch(A, elems(*Ad));
  auto hC = fetch(C, elems(*Cd));
  // A is broadcast over any dimension where it has extent 1 -- the usual bias
  // add has A shaped [1, C, 1, 1].
  for (int n = 0; n < Cd->n; ++n)
    for (int c = 0; c < Cd->c; ++c)
      for (int i = 0; i < Cd->h; ++i)
        for (int j = 0; j < Cd->w; ++j) {
          float av = hA[at(*Ad, Ad->n == 1 ? 0 : n, Ad->c == 1 ? 0 : c, Ad->h == 1 ? 0 : i,
                           Ad->w == 1 ? 0 : j)];
          float& out = hC[at(*Cd, n, c, i, j)];
          out = a * av + b * out;
        }
  store(C, hC);
  return CUDNN_STATUS_SUCCESS;
}

VGPU_EXPORT cudnnStatus_t cudnnCreateActivationDescriptor(cudnnActivationDescriptor_t* d) {
  if (!d) return CUDNN_STATUS_BAD_PARAM;
  *d = reinterpret_cast<cudnnActivationDescriptor_t>(track(new ActDesc()));
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnDestroyActivationDescriptor(cudnnActivationDescriptor_t d) {
  if (!known(d)) return CUDNN_STATUS_BAD_PARAM;
  untrack(d); delete reinterpret_cast<ActDesc*>(d);
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnSetActivationDescriptor(cudnnActivationDescriptor_t d,
                                                       cudnnActivationMode_t mode,
                                                       cudnnNanPropagation_t, double coef) {
  if (!known(d)) return CUDNN_STATUS_BAD_PARAM;
  auto* a = reinterpret_cast<ActDesc*>(d);
  *a = ActDesc{mode, coef, a->swish_beta};
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnGetActivationDescriptor(const cudnnActivationDescriptor_t d,
                                                       cudnnActivationMode_t* mode,
                                                       cudnnNanPropagation_t* nan, double* coef) {
  if (!known(d)) return CUDNN_STATUS_BAD_PARAM;
  auto* a = reinterpret_cast<ActDesc*>(d);
  if (mode) *mode = a->mode;
  if (nan) *nan = CUDNN_NOT_PROPAGATE_NAN;
  if (coef) *coef = a->coef;
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnSetActivationDescriptorSwishBeta(cudnnActivationDescriptor_t d,
                                                                double beta) {
  if (!known(d)) return CUDNN_STATUS_BAD_PARAM;
  reinterpret_cast<ActDesc*>(d)->swish_beta = static_cast<float>(beta);
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnGetActivationDescriptorSwishBeta(cudnnActivationDescriptor_t d,
                                                                double* beta) {
  if (!known(d)) return CUDNN_STATUS_BAD_PARAM;
  if (beta) *beta = reinterpret_cast<ActDesc*>(d)->swish_beta;
  return CUDNN_STATUS_SUCCESS;
}

VGPU_EXPORT cudnnStatus_t cudnnActivationForward(cudnnHandle_t h,
                                                 cudnnActivationDescriptor_t ad, const void* alpha,
                                                 const cudnnTensorDescriptor_t xd, const void* x,
                                                 const void* beta,
                                                 const cudnnTensorDescriptor_t yd, void* y) {
  if (!known(h) || !known(ad) || !known(xd) || !known(yd)) return CUDNN_STATUS_NOT_INITIALIZED;
  auto* A = reinterpret_cast<ActDesc*>(ad);
  auto* X = reinterpret_cast<TensorDesc*>(xd);
  auto* Y = reinterpret_cast<TensorDesc*>(yd);
  const float a = alpha_of(alpha), b = alpha_of(beta);
  auto hx = fetch(x, elems(*X));
  auto hy = (b != 0.0f) ? fetch(y, elems(*Y)) : std::vector<float>(elems(*Y), 0.0f);
  for (size_t i = 0; i < hx.size() && i < hy.size(); ++i) {
    float v = hx[i], r;
    switch (A->mode) {
      case CUDNN_ACTIVATION_SIGMOID: r = 1.0f / (1.0f + std::exp(-v)); break;
      // coef is the ELU alpha and the CLIPPED_RELU ceiling; plain RELU ignores it.
      case CUDNN_ACTIVATION_RELU: r = v > 0.0f ? v : 0.0f; break;
      case CUDNN_ACTIVATION_TANH: r = std::tanh(v); break;
      case CUDNN_ACTIVATION_CLIPPED_RELU:
        r = std::min(std::max(v, 0.0f), static_cast<float>(A->coef));
        break;
      case CUDNN_ACTIVATION_ELU: r = v >= 0 ? v : static_cast<float>(A->coef) * (std::exp(v) - 1.0f); break;
      // Hardware rejects IDENTITY here: real cuDNN only accepts it through
      // cudnnConvolutionBiasActivationForward, and returns BAD_PARAM (2000)
      // from this entry point. Matching that keeps callers' fallbacks intact.
      case CUDNN_ACTIVATION_IDENTITY: return CUDNN_STATUS_BAD_PARAM;
      case CUDNN_ACTIVATION_SWISH: r = v / (1.0f + std::exp(-A->swish_beta * v)); break;
      default:
        if (trace()) std::fprintf(stderr, "[vgpu] cudnnActivationForward: mode %d\n", (int)A->mode);
        return CUDNN_STATUS_NOT_SUPPORTED;
    }
    hy[i] = b == 0.0f ? a * r : a * r + b * hy[i];
  }
  store(y, hy);
  return CUDNN_STATUS_SUCCESS;
}

VGPU_EXPORT cudnnStatus_t cudnnCreatePoolingDescriptor(cudnnPoolingDescriptor_t* d) {
  if (!d) return CUDNN_STATUS_BAD_PARAM;
  *d = reinterpret_cast<cudnnPoolingDescriptor_t>(track(new PoolDesc()));
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnDestroyPoolingDescriptor(cudnnPoolingDescriptor_t d) {
  if (!known(d)) return CUDNN_STATUS_BAD_PARAM;
  untrack(d); delete reinterpret_cast<PoolDesc*>(d);
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnSetPooling2dDescriptor(cudnnPoolingDescriptor_t d,
                                                      cudnnPoolingMode_t mode,
                                                      cudnnNanPropagation_t, int wh, int ww,
                                                      int ph, int pw, int sh, int sw) {
  if (!known(d)) return CUDNN_STATUS_BAD_PARAM;
  *reinterpret_cast<PoolDesc*>(d) = PoolDesc{mode, wh, ww, ph, pw, sh, sw};
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnGetPooling2dForwardOutputDim(const cudnnPoolingDescriptor_t pd,
                                                            const cudnnTensorDescriptor_t xd,
                                                            int* n, int* c, int* h, int* w) {
  if (!known(pd) || !known(xd)) return CUDNN_STATUS_BAD_PARAM;
  auto* p = reinterpret_cast<PoolDesc*>(pd);
  auto* x = reinterpret_cast<TensorDesc*>(xd);
  if (n) *n = x->n;
  if (c) *c = x->c;
  if (h) *h = 1 + (x->h + 2 * p->ph - p->wh) / p->sh;
  if (w) *w = 1 + (x->w + 2 * p->pw - p->ww) / p->sw;
  return CUDNN_STATUS_SUCCESS;
}

VGPU_EXPORT cudnnStatus_t cudnnPoolingForward(cudnnHandle_t h, const cudnnPoolingDescriptor_t pd,
                                              const void* alpha, const cudnnTensorDescriptor_t xd,
                                              const void* x, const void* beta,
                                              const cudnnTensorDescriptor_t yd, void* y) {
  if (!known(h) || !known(pd) || !known(xd) || !known(yd)) return CUDNN_STATUS_NOT_INITIALIZED;
  auto* P = reinterpret_cast<PoolDesc*>(pd);
  auto* X = reinterpret_cast<TensorDesc*>(xd);
  auto* Y = reinterpret_cast<TensorDesc*>(yd);
  const float a = alpha_of(alpha), b = alpha_of(beta);
  auto hx = fetch(x, elems(*X));
  auto hy = (b != 0.0f) ? fetch(y, elems(*Y)) : std::vector<float>(elems(*Y), 0.0f);
  const bool count_pad = P->mode == CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
  for (int n = 0; n < Y->n; ++n)
    for (int c = 0; c < Y->c; ++c)
      for (int oh = 0; oh < Y->h; ++oh)
        for (int ow = 0; ow < Y->w; ++ow) {
          float best = -INFINITY, sum = 0.0f;
          int count = 0;
          for (int i = 0; i < P->wh; ++i)
            for (int j = 0; j < P->ww; ++j) {
              int ih = oh * P->sh - P->ph + i, iw = ow * P->sw - P->pw + j;
              if (ih < 0 || ih >= X->h || iw < 0 || iw >= X->w) {
                if (count_pad) ++count;
                continue;
              }
              float v = hx[at(*X, n, c, ih, iw)];
              best = std::max(best, v);
              sum += v;
              ++count;
            }
          float r = P->mode == CUDNN_POOLING_MAX ? best : (count ? sum / count : 0.0f);
          float& out = hy[at(*Y, n, c, oh, ow)];
          out = b == 0.0f ? a * r : a * r + b * out;
        }
  store(y, hy);
  return CUDNN_STATUS_SUCCESS;
}

VGPU_EXPORT cudnnStatus_t cudnnSoftmaxForward(cudnnHandle_t h, cudnnSoftmaxAlgorithm_t algo,
                                              cudnnSoftmaxMode_t mode, const void* alpha,
                                              const cudnnTensorDescriptor_t xd, const void* x,
                                              const void* beta, const cudnnTensorDescriptor_t yd,
                                              void* y) {
  if (!known(h) || !known(xd) || !known(yd)) return CUDNN_STATUS_NOT_INITIALIZED;
  auto* X = reinterpret_cast<TensorDesc*>(xd);
  auto* Y = reinterpret_cast<TensorDesc*>(yd);
  const float a = alpha_of(alpha), b = alpha_of(beta);
  auto hx = fetch(x, elems(*X));
  auto hy = (b != 0.0f) ? fetch(y, elems(*Y)) : std::vector<float>(elems(*Y), 0.0f);
  // CHANNEL mode reduces over C for each (n, h, w); INSTANCE over C*H*W per n.
  auto softmax_over = [&](const std::vector<size_t>& idxs) {
    float mx = -INFINITY;
    for (size_t i : idxs) mx = std::max(mx, hx[i]);
    float sum = 0.0f;
    for (size_t i : idxs) sum += std::exp(hx[i] - mx);
    for (size_t i : idxs) {
      float r = algo == CUDNN_SOFTMAX_LOG ? (hx[i] - mx - std::log(sum))
                                          : std::exp(hx[i] - mx) / sum;
      hy[i] = b == 0.0f ? a * r : a * r + b * hy[i];
    }
  };
  if (mode == CUDNN_SOFTMAX_MODE_CHANNEL) {
    for (int n = 0; n < X->n; ++n)
      for (int i = 0; i < X->h; ++i)
        for (int j = 0; j < X->w; ++j) {
          std::vector<size_t> idxs;
          for (int c = 0; c < X->c; ++c) idxs.push_back(at(*X, n, c, i, j));
          softmax_over(idxs);
        }
  } else {
    for (int n = 0; n < X->n; ++n) {
      std::vector<size_t> idxs;
      for (int c = 0; c < X->c; ++c)
        for (int i = 0; i < X->h; ++i)
          for (int j = 0; j < X->w; ++j) idxs.push_back(at(*X, n, c, i, j));
      softmax_over(idxs);
    }
  }
  store(y, hy);
  return CUDNN_STATUS_SUCCESS;
}

VGPU_EXPORT cudnnStatus_t cudnnScaleTensor(cudnnHandle_t h, const cudnnTensorDescriptor_t yd,
                                           void* y, const void* alpha) {
  if (!known(h) || !known(yd)) return CUDNN_STATUS_NOT_INITIALIZED;
  auto* Y = reinterpret_cast<TensorDesc*>(yd);
  const float a = alpha_of(alpha);
  auto hy = fetch(y, elems(*Y));
  for (auto& v : hy) v *= a;
  store(y, hy);
  return CUDNN_STATUS_SUCCESS;
}

VGPU_EXPORT cudnnStatus_t cudnnSetTensor(cudnnHandle_t h, const cudnnTensorDescriptor_t yd, void* y,
                                         const void* value) {
  if (!known(h) || !known(yd)) return CUDNN_STATUS_NOT_INITIALIZED;
  auto* Y = reinterpret_cast<TensorDesc*>(yd);
  std::vector<float> hy(elems(*Y), alpha_of(value));
  store(y, hy);
  return CUDNN_STATUS_SUCCESS;
}

/* ---- batch normalization ---- */

VGPU_EXPORT cudnnStatus_t cudnnDeriveBNTensorDescriptor(cudnnTensorDescriptor_t d,
                                                        const cudnnTensorDescriptor_t xd,
                                                        cudnnBatchNormMode_t mode) {
  if (!known(d) || !known(xd)) return CUDNN_STATUS_BAD_PARAM;
  auto* x = reinterpret_cast<TensorDesc*>(xd);
  // SPATIAL keeps one parameter per channel; PER_ACTIVATION one per (c, h, w).
  const int h = mode == CUDNN_BATCHNORM_SPATIAL ? 1 : x->h;
  const int w = mode == CUDNN_BATCHNORM_SPATIAL ? 1 : x->w;
  *reinterpret_cast<TensorDesc*>(d) =
      TensorDesc{x->type, 1, x->c, h, w, x->c * h * w, h * w, w, 1};
  return CUDNN_STATUS_SUCCESS;
}

VGPU_EXPORT cudnnStatus_t cudnnBatchNormalizationForwardInference(
    cudnnHandle_t h, cudnnBatchNormMode_t mode, const void* alpha, const void* beta,
    const cudnnTensorDescriptor_t xd, const void* x, const cudnnTensorDescriptor_t yd, void* y,
    const cudnnTensorDescriptor_t bnd, const void* scale, const void* bias, const void* mean,
    const void* var, double eps) {
  if (!known(h) || !known(xd) || !known(yd) || !known(bnd)) return CUDNN_STATUS_NOT_INITIALIZED;
  if (eps < CUDNN_BN_MIN_EPSILON) return CUDNN_STATUS_BAD_PARAM;
  auto* X = reinterpret_cast<TensorDesc*>(xd);
  auto* Y = reinterpret_cast<TensorDesc*>(yd);
  auto* B = reinterpret_cast<TensorDesc*>(bnd);
  const float a = alpha_of(alpha), b = alpha_of(beta);
  const size_t np = elems(*B);
  auto hx = fetch(x, elems(*X));
  auto hy = (b != 0.0f) ? fetch(y, elems(*Y)) : std::vector<float>(elems(*Y), 0.0f);
  auto hs = fetch(scale, np), hb = fetch(bias, np), hm = fetch(mean, np), hv = fetch(var, np);
  const bool spatial = mode == CUDNN_BATCHNORM_SPATIAL ||
                       mode == CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
  for (int n = 0; n < X->n; ++n)
    for (int c = 0; c < X->c; ++c)
      for (int i = 0; i < X->h; ++i)
        for (int j = 0; j < X->w; ++j) {
          const size_t p = spatial ? (size_t)c : at(*B, 0, c, i, j);
          const float r = hs[p] * (hx[at(*X, n, c, i, j)] - hm[p]) /
                              std::sqrt(static_cast<float>(eps) + hv[p]) + hb[p];
          float& out = hy[at(*Y, n, c, i, j)];
          out = b == 0.0f ? a * r : a * r + b * out;
        }
  store(y, hy);
  return CUDNN_STATUS_SUCCESS;
}

VGPU_EXPORT cudnnStatus_t cudnnBatchNormalizationForwardTraining(
    cudnnHandle_t h, cudnnBatchNormMode_t mode, const void* alpha, const void* beta,
    const cudnnTensorDescriptor_t xd, const void* x, const cudnnTensorDescriptor_t yd, void* y,
    const cudnnTensorDescriptor_t bnd, const void* scale, const void* bias,
    double exp_avg_factor, void* running_mean, void* running_var, double eps,
    void* save_mean, void* save_inv_var) {
  if (!known(h) || !known(xd) || !known(yd) || !known(bnd)) return CUDNN_STATUS_NOT_INITIALIZED;
  if (eps < CUDNN_BN_MIN_EPSILON) return CUDNN_STATUS_BAD_PARAM;
  auto* X = reinterpret_cast<TensorDesc*>(xd);
  auto* Y = reinterpret_cast<TensorDesc*>(yd);
  auto* B = reinterpret_cast<TensorDesc*>(bnd);
  const float a = alpha_of(alpha), b = alpha_of(beta);
  const bool spatial = mode == CUDNN_BATCHNORM_SPATIAL ||
                       mode == CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
  const size_t np = elems(*B);
  auto hx = fetch(x, elems(*X));
  auto hy = (b != 0.0f) ? fetch(y, elems(*Y)) : std::vector<float>(elems(*Y), 0.0f);
  auto hs = fetch(scale, np), hb = fetch(bias, np);

  std::vector<double> sum(np, 0.0), sq(np, 0.0);
  std::vector<size_t> cnt(np, 0);
  for (int n = 0; n < X->n; ++n)
    for (int c = 0; c < X->c; ++c)
      for (int i = 0; i < X->h; ++i)
        for (int j = 0; j < X->w; ++j) {
          const size_t p = spatial ? (size_t)c : at(*B, 0, c, i, j);
          const double v = hx[at(*X, n, c, i, j)];
          sum[p] += v; sq[p] += v * v; ++cnt[p];
        }
  std::vector<float> mean(np), var(np), inv(np);
  for (size_t p = 0; p < np; ++p) {
    const double m = cnt[p] ? sum[p] / cnt[p] : 0.0;
    const double v = cnt[p] ? sq[p] / cnt[p] - m * m : 0.0;
    mean[p] = static_cast<float>(m);
    var[p] = static_cast<float>(v);
    inv[p] = static_cast<float>(1.0 / std::sqrt(v + eps));
  }
  for (int n = 0; n < X->n; ++n)
    for (int c = 0; c < X->c; ++c)
      for (int i = 0; i < X->h; ++i)
        for (int j = 0; j < X->w; ++j) {
          const size_t p = spatial ? (size_t)c : at(*B, 0, c, i, j);
          const float r = hs[p] * (hx[at(*X, n, c, i, j)] - mean[p]) * inv[p] + hb[p];
          float& out = hy[at(*Y, n, c, i, j)];
          out = b == 0.0f ? a * r : a * r + b * out;
        }
  store(y, hy);
  if (save_mean) store(save_mean, mean);
  if (save_inv_var) store(save_inv_var, inv);
  if (running_mean && running_var) {
    auto rm = fetch(running_mean, np), rv = fetch(running_var, np);
    for (size_t p = 0; p < np; ++p) {
      // Running variance tracks the *unbiased* estimate, unlike the biased one
      // used to normalize this batch.
      const double n_ = static_cast<double>(cnt[p]);
      const double unbiased = n_ > 1 ? var[p] * n_ / (n_ - 1) : var[p];
      rm[p] = static_cast<float>(exp_avg_factor * mean[p] + (1.0 - exp_avg_factor) * rm[p]);
      rv[p] = static_cast<float>(exp_avg_factor * unbiased + (1.0 - exp_avg_factor) * rv[p]);
    }
    store(running_mean, rm);
    store(running_var, rv);
  }
  return CUDNN_STATUS_SUCCESS;
}

/* ---- op tensor: pointwise add/mul/min/max/sqrt over broadcast inputs ---- */

namespace {
struct OpDesc { cudnnOpTensorOp_t op = CUDNN_OP_TENSOR_ADD; };
}  // namespace

VGPU_EXPORT cudnnStatus_t cudnnCreateOpTensorDescriptor(cudnnOpTensorDescriptor_t* d) {
  if (!d) return CUDNN_STATUS_BAD_PARAM;
  *d = reinterpret_cast<cudnnOpTensorDescriptor_t>(track(new OpDesc()));
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnDestroyOpTensorDescriptor(cudnnOpTensorDescriptor_t d) {
  if (!known(d)) return CUDNN_STATUS_BAD_PARAM;
  untrack(d); delete reinterpret_cast<OpDesc*>(d);
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnSetOpTensorDescriptor(cudnnOpTensorDescriptor_t d,
                                                     cudnnOpTensorOp_t op, cudnnDataType_t type,
                                                     cudnnNanPropagation_t) {
  if (!known(d)) return CUDNN_STATUS_BAD_PARAM;
  if (type != CUDNN_DATA_FLOAT) return CUDNN_STATUS_NOT_SUPPORTED;
  reinterpret_cast<OpDesc*>(d)->op = op;
  return CUDNN_STATUS_SUCCESS;
}

VGPU_EXPORT cudnnStatus_t cudnnOpTensor(cudnnHandle_t h, const cudnnOpTensorDescriptor_t od,
                                        const void* alpha1, const cudnnTensorDescriptor_t ad,
                                        const void* A, const void* alpha2,
                                        const cudnnTensorDescriptor_t bd, const void* Bp,
                                        const void* beta, const cudnnTensorDescriptor_t cd,
                                        void* C) {
  if (!known(h) || !known(od) || !known(ad) || !known(bd) || !known(cd))
    return CUDNN_STATUS_NOT_INITIALIZED;
  auto* O = reinterpret_cast<OpDesc*>(od);
  auto* Ad = reinterpret_cast<TensorDesc*>(ad);
  auto* Bd = reinterpret_cast<TensorDesc*>(bd);
  auto* Cd = reinterpret_cast<TensorDesc*>(cd);
  const float a1 = alpha_of(alpha1), a2 = alpha_of(alpha2), b = alpha_of(beta);
  auto hA = fetch(A, elems(*Ad));
  auto hB = fetch(Bp, elems(*Bd));
  auto hC = fetch(C, elems(*Cd));
  auto pick = [](const TensorDesc& t, int n, int c, int i, int j) {
    return at(t, t.n == 1 ? 0 : n, t.c == 1 ? 0 : c, t.h == 1 ? 0 : i, t.w == 1 ? 0 : j);
  };
  for (int n = 0; n < Cd->n; ++n)
    for (int c = 0; c < Cd->c; ++c)
      for (int i = 0; i < Cd->h; ++i)
        for (int j = 0; j < Cd->w; ++j) {
          const float av = a1 * hA[pick(*Ad, n, c, i, j)];
          const float bv = a2 * hB[pick(*Bd, n, c, i, j)];
          float r;
          switch (O->op) {
            case CUDNN_OP_TENSOR_ADD: r = av + bv; break;
            case CUDNN_OP_TENSOR_MUL: r = av * bv; break;
            case CUDNN_OP_TENSOR_MIN: r = std::min(av, bv); break;
            case CUDNN_OP_TENSOR_MAX: r = std::max(av, bv); break;
            case CUDNN_OP_TENSOR_SQRT: r = std::sqrt(av); break;  // unary: B is ignored
            case CUDNN_OP_TENSOR_NOT: r = 1.0f - av; break;
            default: return CUDNN_STATUS_NOT_SUPPORTED;
          }
          float& out = hC[at(*Cd, n, c, i, j)];
          out = b == 0.0f ? r : r + b * out;
        }
  store(C, hC);
  return CUDNN_STATUS_SUCCESS;
}

/* ---- transform: strided copy with scaling, used for layout conversions ---- */

VGPU_EXPORT cudnnStatus_t cudnnTransformTensor(cudnnHandle_t h, const void* alpha,
                                               const cudnnTensorDescriptor_t xd, const void* x,
                                               const void* beta, const cudnnTensorDescriptor_t yd,
                                               void* y) {
  if (!known(h) || !known(xd) || !known(yd)) return CUDNN_STATUS_NOT_INITIALIZED;
  auto* X = reinterpret_cast<TensorDesc*>(xd);
  auto* Y = reinterpret_cast<TensorDesc*>(yd);
  if (X->n != Y->n || X->c != Y->c || X->h != Y->h || X->w != Y->w)
    return CUDNN_STATUS_BAD_PARAM;
  const float a = alpha_of(alpha), b = alpha_of(beta);
  auto hx = fetch(x, elems(*X));
  auto hy = fetch(y, elems(*Y));
  for (int n = 0; n < Y->n; ++n)
    for (int c = 0; c < Y->c; ++c)
      for (int i = 0; i < Y->h; ++i)
        for (int j = 0; j < Y->w; ++j) {
          float& out = hy[at(*Y, n, c, i, j)];
          const float v = a * hx[at(*X, n, c, i, j)];
          out = b == 0.0f ? v : v + b * out;
        }
  store(y, hy);
  return CUDNN_STATUS_SUCCESS;
}

/* ---- N-dimensional tensors and batch normalization's training forms ---- */

// An N-D descriptor (3 to 5 dimensions, NC first) is kept in the 4-D form the
// rest of this file computes on: a 3-D tensor as N, C, L, 1; a 5-D one with
// its depth and height folded together, which needs the height packed inside
// the depth -- anything else is refused rather than walked wrongly.
VGPU_EXPORT cudnnStatus_t cudnnSetTensorNdDescriptor(cudnnTensorDescriptor_t d, cudnnDataType_t type,
                                                     int nb, const int dims[], const int strides[]) {
  if (!known(d) || !dims || !strides || nb < 3 || nb > CUDNN_DIM_MAX) return CUDNN_STATUS_BAD_PARAM;
  if (type != CUDNN_DATA_FLOAT) return CUDNN_STATUS_NOT_SUPPORTED;
  TensorDesc t;
  t.type = type;
  t.n = dims[0], t.c = dims[1], t.sn = strides[0], t.sc = strides[1];
  if (nb == 3) {
    t.h = dims[2], t.sh = strides[2], t.w = 1, t.sw = 1;
  } else if (nb == 4) {
    t.h = dims[2], t.w = dims[3], t.sh = strides[2], t.sw = strides[3];
  } else if (nb == 5) {
    if (strides[2] != dims[3] * strides[3]) return CUDNN_STATUS_NOT_SUPPORTED;
    t.h = dims[2] * dims[3], t.sh = strides[3], t.w = dims[4], t.sw = strides[4];
  } else {
    return CUDNN_STATUS_NOT_SUPPORTED;
  }
  t.nb = nb;
  for (int i = 0; i < nb; ++i) t.dims[i] = dims[i], t.strides[i] = strides[i];
  *reinterpret_cast<TensorDesc*>(d) = t;
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnGetTensorNdDescriptor(const cudnnTensorDescriptor_t d, int requested,
                                                     cudnnDataType_t* type, int* nb, int dims[], int strides[]) {
  if (!known(d)) return CUDNN_STATUS_BAD_PARAM;
  const auto* t = reinterpret_cast<const TensorDesc*>(d);
  const int four_d[4] = {t->n, t->c, t->h, t->w}, four_s[4] = {t->sn, t->sc, t->sh, t->sw};
  const int count = t->nb ? t->nb : 4;
  if (type) *type = t->type;
  if (nb) *nb = count;
  for (int i = 0; i < std::min(requested, count); ++i) {
    if (dims) dims[i] = t->nb ? t->dims[i] : four_d[i];
    if (strides) strides[i] = t->nb ? t->strides[i] : four_s[i];
  }
  return CUDNN_STATUS_SUCCESS;
}

namespace {
// The elements from a tensor's first to one past its last, as its strides lay
// them out: what has to be copied to reach every one.
size_t span(const TensorDesc& t) {
  if (!t.n || !t.c || !t.h || !t.w) return 0;
  return (size_t)(t.n - 1) * t.sn + (size_t)(t.c - 1) * t.sc + (size_t)(t.h - 1) * t.sh + (size_t)(t.w - 1) * t.sw + 1;
}
}  // namespace

// Workspace and reserve space: none, since the work is done on the host.
VGPU_EXPORT cudnnStatus_t cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(
    cudnnHandle_t, cudnnBatchNormMode_t, cudnnBatchNormOps_t, const cudnnTensorDescriptor_t,
    const cudnnTensorDescriptor_t, const cudnnTensorDescriptor_t, const cudnnTensorDescriptor_t,
    const cudnnActivationDescriptor_t, size_t* size) {
  if (size) *size = 0;
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnGetBatchNormalizationBackwardExWorkspaceSize(
    cudnnHandle_t, cudnnBatchNormMode_t, cudnnBatchNormOps_t, const cudnnTensorDescriptor_t,
    const cudnnTensorDescriptor_t, const cudnnTensorDescriptor_t, const cudnnTensorDescriptor_t,
    const cudnnTensorDescriptor_t, const cudnnTensorDescriptor_t, const cudnnActivationDescriptor_t, size_t* size) {
  if (size) *size = 0;
  return CUDNN_STATUS_SUCCESS;
}
VGPU_EXPORT cudnnStatus_t cudnnGetBatchNormalizationTrainingExReserveSpaceSize(
    cudnnHandle_t, cudnnBatchNormMode_t, cudnnBatchNormOps_t, const cudnnActivationDescriptor_t,
    const cudnnTensorDescriptor_t, size_t* size) {
  if (size) *size = 0;
  return CUDNN_STATUS_SUCCESS;
}

// The Ex form without its fusions (an added tensor, an activation) is the
// plain one; with them it is refused by name.
VGPU_EXPORT cudnnStatus_t cudnnBatchNormalizationForwardTrainingEx(
    cudnnHandle_t h, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t ops, const void* alpha, const void* beta,
    const cudnnTensorDescriptor_t xd, const void* x, const cudnnTensorDescriptor_t, const void*,
    const cudnnTensorDescriptor_t yd, void* y, const cudnnTensorDescriptor_t bnd, const void* scale,
    const void* bias, double factor, void* running_mean, void* running_var, double eps, void* save_mean,
    void* save_inv_var, cudnnActivationDescriptor_t, void*, size_t, void*, size_t) {
  if (ops != CUDNN_BATCHNORM_OPS_BN) {
    if (!quiet()) std::fprintf(stderr, "[vgpu] cudnnBatchNormalizationForwardTrainingEx: only CUDNN_BATCHNORM_OPS_BN (no fused add or activation) is supported\n");
    return CUDNN_STATUS_NOT_SUPPORTED;
  }
  return cudnnBatchNormalizationForwardTraining(h, mode, alpha, beta, xd, x, yd, y, bnd, scale, bias, factor,
                                                running_mean, running_var, eps, save_mean, save_inv_var);
}

// The gradient of y = scale * (x - mean) * inv + bias, per channel (spatial)
// or per activation:
//   dbias  = sum dy
//   dscale = sum dy * xhat
//   dx     = scale * inv / m * (m * dy - dbias - xhat * dscale)
// with the saved mean and inverse deviation when given, else recomputed.
VGPU_EXPORT cudnnStatus_t cudnnBatchNormalizationBackward(
    cudnnHandle_t h, cudnnBatchNormMode_t mode, const void* alpha_data, const void* beta_data,
    const void* alpha_param, const void* beta_param, const cudnnTensorDescriptor_t xd, const void* x,
    const cudnnTensorDescriptor_t dyd, const void* dy, const cudnnTensorDescriptor_t dxd, void* dx,
    const cudnnTensorDescriptor_t bnd, const void* scale, void* dscale_out, void* dbias_out, double eps,
    const void* saved_mean, const void* saved_inv) {
  if (!known(h) || !known(xd) || !known(dyd) || !known(dxd) || !known(bnd)) return CUDNN_STATUS_NOT_INITIALIZED;
  auto* X = reinterpret_cast<TensorDesc*>(xd);
  auto* DY = reinterpret_cast<TensorDesc*>(dyd);
  auto* DX = reinterpret_cast<TensorDesc*>(dxd);
  auto* B = reinterpret_cast<TensorDesc*>(bnd);
  const float ad = alpha_of(alpha_data), bd = alpha_of(beta_data);
  const float ap = alpha_of(alpha_param), bp = alpha_of(beta_param);
  const bool spatial = mode == CUDNN_BATCHNORM_SPATIAL || mode == CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
  cudaStream_t stream = reinterpret_cast<Handle*>(h)->stream;
  cudaStreamSynchronize(stream);
  const size_t np = elems(*B);
  auto hx = fetch(x, span(*X)), hdy = fetch(dy, span(*DY));
  auto hdx = fetch(dx, span(*DX));
  auto hs = fetch(scale, np);
  auto slot = [&](int c, int i, int j) { return spatial ? (size_t)c : at(*B, 0, c, i, j); };

  std::vector<double> mean(np, 0.0), inv(np, 0.0), cnt(np, 0.0);
  if (saved_mean && saved_inv) {
    auto m = fetch(saved_mean, np), v = fetch(saved_inv, np);
    for (size_t p = 0; p < np; ++p) mean[p] = m[p], inv[p] = v[p];
    for (int n = 0; n < X->n; ++n) for (int c = 0; c < X->c; ++c)
      for (int i = 0; i < X->h; ++i) for (int j = 0; j < X->w; ++j) cnt[slot(c, i, j)] += 1;
  } else {
    std::vector<double> sum(np, 0.0), sq(np, 0.0);
    for (int n = 0; n < X->n; ++n) for (int c = 0; c < X->c; ++c)
      for (int i = 0; i < X->h; ++i) for (int j = 0; j < X->w; ++j) {
        const size_t p = slot(c, i, j);
        const double v = hx[at(*X, n, c, i, j)];
        sum[p] += v, sq[p] += v * v, cnt[p] += 1;
      }
    for (size_t p = 0; p < np; ++p) {
      mean[p] = cnt[p] ? sum[p] / cnt[p] : 0.0;
      const double var = cnt[p] ? sq[p] / cnt[p] - mean[p] * mean[p] : 0.0;
      inv[p] = 1.0 / std::sqrt(var + eps);
    }
  }
  std::vector<double> dbias(np, 0.0), dscale(np, 0.0);
  for (int n = 0; n < X->n; ++n) for (int c = 0; c < X->c; ++c)
    for (int i = 0; i < X->h; ++i) for (int j = 0; j < X->w; ++j) {
      const size_t p = slot(c, i, j);
      const double g = hdy[at(*DY, n, c, i, j)];
      const double xhat = (hx[at(*X, n, c, i, j)] - mean[p]) * inv[p];
      dbias[p] += g, dscale[p] += g * xhat;
    }
  for (int n = 0; n < X->n; ++n) for (int c = 0; c < X->c; ++c)
    for (int i = 0; i < X->h; ++i) for (int j = 0; j < X->w; ++j) {
      const size_t p = slot(c, i, j);
      const double xhat = (hx[at(*X, n, c, i, j)] - mean[p]) * inv[p];
      const double g = hdy[at(*DY, n, c, i, j)];
      const double r = hs[p] * inv[p] / cnt[p] * (cnt[p] * g - dbias[p] - xhat * dscale[p]);
      float& out = hdx[at(*DX, n, c, i, j)];
      out = static_cast<float>(bd == 0.0f ? ad * r : ad * r + bd * out);
    }
  store(dx, hdx);
  auto blend = [&](void* dst, const std::vector<double>& v) {
    if (!dst) return;
    auto cur = bp != 0.0f ? fetch(dst, np) : std::vector<float>(np, 0.0f);
    for (size_t p = 0; p < np; ++p) cur[p] = static_cast<float>(ap * v[p] + (bp != 0.0f ? bp * cur[p] : 0.0));
    store(dst, cur);
  };
  blend(dscale_out, dscale);
  blend(dbias_out, dbias);
  return CUDNN_STATUS_SUCCESS;
}

VGPU_EXPORT cudnnStatus_t cudnnBatchNormalizationBackwardEx(
    cudnnHandle_t h, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t ops, const void* alpha_data,
    const void* beta_data, const void* alpha_param, const void* beta_param, const cudnnTensorDescriptor_t xd,
    const void* x, const cudnnTensorDescriptor_t, const void*, const cudnnTensorDescriptor_t dyd, const void* dy,
    const cudnnTensorDescriptor_t, void*, const cudnnTensorDescriptor_t dxd, void* dx,
    const cudnnTensorDescriptor_t bnd, const void* scale, const void*, void* dscale, void* dbias, double eps,
    const void* saved_mean, const void* saved_inv, cudnnActivationDescriptor_t, void*, size_t, void*, size_t) {
  if (ops != CUDNN_BATCHNORM_OPS_BN) {
    if (!quiet()) std::fprintf(stderr, "[vgpu] cudnnBatchNormalizationBackwardEx: only CUDNN_BATCHNORM_OPS_BN (no fused add or activation) is supported\n");
    return CUDNN_STATUS_NOT_SUPPORTED;
  }
  return cudnnBatchNormalizationBackward(h, mode, alpha_data, beta_data, alpha_param, beta_param, xd, x, dyd, dy,
                                         dxd, dx, bnd, scale, dscale, dbias, eps, saved_mean, saved_inv);
}
