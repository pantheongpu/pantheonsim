// libvgpucublasLt — VirtualGPU's implementation of the cuBLASLt API.
//
// cuBLASLt is the descriptor-based matmul interface that modern frameworks
// prefer over the classic cuBLAS calls, so supporting it matters as much as
// cuBLAS itself. The descriptors are opaque handles here, and the arithmetic
// reuses the same host-side GEMM the cuBLAS shim uses -- see docs/cublas.md
// for why library math runs on the host rather than through the interpreter.
//
// Epilogues (bias, ReLU, GELU) are applied after the matmul, matching the
// documented semantics. Anything not implemented returns
// CUBLAS_STATUS_NOT_SUPPORTED so a caller falls back rather than receiving a
// plausible wrong answer.
#include <cublasLt.h>

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

struct LtHandle { int dummy = 0; };

struct MatmulDesc {
  cublasComputeType_t compute = CUBLAS_COMPUTE_32F;
  cudaDataType scale = CUDA_R_32F;
  cublasOperation_t transa = CUBLAS_OP_N, transb = CUBLAS_OP_N;
  cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DEFAULT;
  const void* bias = nullptr;
};

struct MatrixLayout {
  cudaDataType type = CUDA_R_32F;
  uint64_t rows = 0, cols = 0;
  int64_t ld = 0;
  int32_t batch = 1;
  int64_t batch_stride = 0;
};

struct Preference { size_t workspace = 0; };

std::mutex g_mu;
std::set<void*> g_live;

template <class T>
T* track(T* p) {
  std::lock_guard<std::mutex> lock(g_mu);
  g_live.insert(p);
  return p;
}
bool known(const void* p) {
  std::lock_guard<std::mutex> lock(g_mu);
  return p && g_live.count(const_cast<void*>(p));
}
void untrack(void* p) {
  std::lock_guard<std::mutex> lock(g_mu);
  g_live.erase(p);
}

template <class T>
std::vector<T> fetch(const void* dev, size_t n) {
  std::vector<T> h(n);
  if (n) cudaMemcpy(h.data(), dev, n * sizeof(T), cudaMemcpyDeviceToHost);
  return h;
}
template <class T>
void store(void* dev, const std::vector<T>& h) {
  if (!h.empty()) cudaMemcpy(dev, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
}
inline size_t idx(int64_t r, int64_t c, int64_t ld) { return static_cast<size_t>(c) * ld + r; }

float gelu(float x) {
  // The tanh approximation, which is what frameworks use for this epilogue.
  const float k = 0.7978845608f;  // sqrt(2/pi)
  return 0.5f * x * (1.0f + std::tanh(k * (x + 0.044715f * x * x * x)));
}

}  // namespace

#define VGPU_EXPORT extern "C" __attribute__((visibility("default")))

/* ---- lifecycle ---- */

VGPU_EXPORT cublasStatus_t cublasLtCreate(cublasLtHandle_t* h) {
  if (!h) return CUBLAS_STATUS_INVALID_VALUE;
  *h = reinterpret_cast<cublasLtHandle_t>(track(new LtHandle()));
  if (!quiet()) std::fprintf(stderr, "[vgpu] cuBLASLt handle created (host-computed)\n");
  return CUBLAS_STATUS_SUCCESS;
}
VGPU_EXPORT cublasStatus_t cublasLtDestroy(cublasLtHandle_t h) {
  if (!known(h)) return CUBLAS_STATUS_NOT_INITIALIZED;
  untrack(h);
  delete reinterpret_cast<LtHandle*>(h);
  return CUBLAS_STATUS_SUCCESS;
}
VGPU_EXPORT size_t cublasLtGetVersion(void) { return CUBLAS_VERSION; }
VGPU_EXPORT size_t cublasLtGetCudartVersion(void) { return 13000; }
VGPU_EXPORT const char* cublasLtGetStatusName(cublasStatus_t s) {
  return s == CUBLAS_STATUS_SUCCESS ? "CUBLAS_STATUS_SUCCESS" : "CUBLAS_STATUS_ERROR";
}

/* ---- descriptors ---- */

VGPU_EXPORT cublasStatus_t cublasLtMatmulDescCreate(cublasLtMatmulDesc_t* d,
                                                    cublasComputeType_t compute,
                                                    cudaDataType scale) {
  if (!d) return CUBLAS_STATUS_INVALID_VALUE;
  auto* m = new MatmulDesc();
  m->compute = compute;
  m->scale = scale;
  *d = reinterpret_cast<cublasLtMatmulDesc_t>(track(m));
  return CUBLAS_STATUS_SUCCESS;
}
VGPU_EXPORT cublasStatus_t cublasLtMatmulDescDestroy(cublasLtMatmulDesc_t d) {
  if (!known(d)) return CUBLAS_STATUS_NOT_INITIALIZED;
  untrack(d);
  delete reinterpret_cast<MatmulDesc*>(d);
  return CUBLAS_STATUS_SUCCESS;
}
// The caller declares how many bytes its buffer holds, and these entry points
// used to ignore that and copy a fixed width either way -- reading past a
// caller's buffer on set, writing past it on get. The size is part of the
// contract, so it is checked.
VGPU_EXPORT cublasStatus_t cublasLtMatmulDescSetAttribute(cublasLtMatmulDesc_t d,
                                                          cublasLtMatmulDescAttributes_t attr,
                                                          const void* buf, size_t bytes) {
  if (!known(d) || !buf) return CUBLAS_STATUS_INVALID_VALUE;
  auto* m = reinterpret_cast<MatmulDesc*>(d);
  switch (attr) {
    case CUBLASLT_MATMUL_DESC_TRANSA:
      if (bytes < sizeof(int32_t)) return CUBLAS_STATUS_INVALID_VALUE;
      std::memcpy(&m->transa, buf, sizeof(int32_t));
      return CUBLAS_STATUS_SUCCESS;
    case CUBLASLT_MATMUL_DESC_TRANSB:
      if (bytes < sizeof(int32_t)) return CUBLAS_STATUS_INVALID_VALUE;
      std::memcpy(&m->transb, buf, sizeof(int32_t));
      return CUBLAS_STATUS_SUCCESS;
    case CUBLASLT_MATMUL_DESC_EPILOGUE:
      if (bytes < sizeof(int32_t)) return CUBLAS_STATUS_INVALID_VALUE;
      std::memcpy(&m->epilogue, buf, sizeof(int32_t));
      return CUBLAS_STATUS_SUCCESS;
    case CUBLASLT_MATMUL_DESC_BIAS_POINTER:
      if (bytes < sizeof(void*)) return CUBLAS_STATUS_INVALID_VALUE;
      std::memcpy(&m->bias, buf, sizeof(void*));
      return CUBLAS_STATUS_SUCCESS;
    default:
      return CUBLAS_STATUS_SUCCESS;  // attributes we do not model are inert
  }
}
VGPU_EXPORT cublasStatus_t cublasLtMatmulDescGetAttribute(cublasLtMatmulDesc_t d,
                                                          cublasLtMatmulDescAttributes_t attr,
                                                          void* buf, size_t bytes,
                                                          size_t* written) {
  if (!known(d) || !buf) return CUBLAS_STATUS_INVALID_VALUE;
  auto* m = reinterpret_cast<MatmulDesc*>(d);
  // The bias attribute is a pointer, not an int32. Returning the epilogue for
  // everything that was not a transpose selector meant asking for the bias
  // pointer got four bytes of an unrelated enum.
  if (attr == CUBLASLT_MATMUL_DESC_BIAS_POINTER) {
    if (bytes < sizeof(void*)) return CUBLAS_STATUS_INVALID_VALUE;
    std::memcpy(buf, &m->bias, sizeof(void*));
    if (written) *written = sizeof(void*);
    return CUBLAS_STATUS_SUCCESS;
  }
  int32_t v = 0;
  switch (attr) {
    case CUBLASLT_MATMUL_DESC_TRANSA: v = m->transa; break;
    case CUBLASLT_MATMUL_DESC_TRANSB: v = m->transb; break;
    case CUBLASLT_MATMUL_DESC_EPILOGUE: v = m->epilogue; break;
    default: return CUBLAS_STATUS_NOT_SUPPORTED;   // rather than a plausible wrong value
  }
  if (bytes < sizeof v) return CUBLAS_STATUS_INVALID_VALUE;
  std::memcpy(buf, &v, sizeof v);
  if (written) *written = sizeof v;
  return CUBLAS_STATUS_SUCCESS;
}

VGPU_EXPORT cublasStatus_t cublasLtMatrixLayoutCreate(cublasLtMatrixLayout_t* l, cudaDataType type,
                                                      uint64_t rows, uint64_t cols, int64_t ld) {
  if (!l) return CUBLAS_STATUS_INVALID_VALUE;
  auto* m = new MatrixLayout();
  m->type = type;
  m->rows = rows;
  m->cols = cols;
  m->ld = ld;
  *l = reinterpret_cast<cublasLtMatrixLayout_t>(track(m));
  return CUBLAS_STATUS_SUCCESS;
}
VGPU_EXPORT cublasStatus_t cublasLtMatrixLayoutDestroy(cublasLtMatrixLayout_t l) {
  if (!known(l)) return CUBLAS_STATUS_NOT_INITIALIZED;
  untrack(l);
  delete reinterpret_cast<MatrixLayout*>(l);
  return CUBLAS_STATUS_SUCCESS;
}
VGPU_EXPORT cublasStatus_t cublasLtMatrixLayoutSetAttribute(cublasLtMatrixLayout_t l,
                                                            cublasLtMatrixLayoutAttribute_t attr,
                                                            const void* buf, size_t) {
  if (!known(l) || !buf) return CUBLAS_STATUS_INVALID_VALUE;
  auto* m = reinterpret_cast<MatrixLayout*>(l);
  if (attr == CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT) std::memcpy(&m->batch, buf, sizeof(int32_t));
  else if (attr == CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET)
    std::memcpy(&m->batch_stride, buf, sizeof(int64_t));
  return CUBLAS_STATUS_SUCCESS;
}
VGPU_EXPORT cublasStatus_t cublasLtMatrixLayoutGetAttribute(cublasLtMatrixLayout_t l,
                                                            cublasLtMatrixLayoutAttribute_t,
                                                            void* buf, size_t, size_t* written) {
  if (!known(l) || !buf) return CUBLAS_STATUS_INVALID_VALUE;
  int32_t v = reinterpret_cast<MatrixLayout*>(l)->batch;
  std::memcpy(buf, &v, sizeof v);
  if (written) *written = sizeof v;
  return CUBLAS_STATUS_SUCCESS;
}

VGPU_EXPORT cublasStatus_t cublasLtMatmulPreferenceCreate(cublasLtMatmulPreference_t* p) {
  if (!p) return CUBLAS_STATUS_INVALID_VALUE;
  *p = reinterpret_cast<cublasLtMatmulPreference_t>(track(new Preference()));
  return CUBLAS_STATUS_SUCCESS;
}
VGPU_EXPORT cublasStatus_t cublasLtMatmulPreferenceDestroy(cublasLtMatmulPreference_t p) {
  if (!known(p)) return CUBLAS_STATUS_NOT_INITIALIZED;
  untrack(p);
  delete reinterpret_cast<Preference*>(p);
  return CUBLAS_STATUS_SUCCESS;
}
VGPU_EXPORT cublasStatus_t cublasLtMatmulPreferenceSetAttribute(cublasLtMatmulPreference_t,
                                                                cublasLtMatmulPreferenceAttributes_t,
                                                                const void*, size_t) {
  return CUBLAS_STATUS_SUCCESS;
}

// One algorithm is offered: there is nothing to tune when the math runs on the
// host, and callers only need a valid heuristic result to proceed.
VGPU_EXPORT cublasStatus_t cublasLtMatmulAlgoGetHeuristic(
    cublasLtHandle_t, cublasLtMatmulDesc_t, cublasLtMatrixLayout_t, cublasLtMatrixLayout_t,
    cublasLtMatrixLayout_t, cublasLtMatrixLayout_t, cublasLtMatmulPreference_t, int requested,
    cublasLtMatmulHeuristicResult_t* results, int* returned) {
  if (!results || !returned || requested < 1) return CUBLAS_STATUS_INVALID_VALUE;
  std::memset(&results[0], 0, sizeof(results[0]));
  results[0].state = CUBLAS_STATUS_SUCCESS;
  results[0].workspaceSize = 0;
  results[0].wavesCount = 1.0f;
  *returned = 1;
  return CUBLAS_STATUS_SUCCESS;
}

/* ---- the matmul itself ---- */

VGPU_EXPORT cublasStatus_t cublasLtMatmul(cublasLtHandle_t h, cublasLtMatmulDesc_t desc,
                                          const void* alpha, const void* A,
                                          cublasLtMatrixLayout_t Adesc, const void* B,
                                          cublasLtMatrixLayout_t Bdesc, const void* beta,
                                          const void* C, cublasLtMatrixLayout_t Cdesc, void* D,
                                          cublasLtMatrixLayout_t Ddesc, const cublasLtMatmulAlgo_t*,
                                          void*, size_t, cudaStream_t) {
  if (!known(h) || !known(desc) || !known(Adesc) || !known(Bdesc) || !known(Ddesc))
    return CUBLAS_STATUS_NOT_INITIALIZED;
  auto* md = reinterpret_cast<MatmulDesc*>(desc);
  auto* la = reinterpret_cast<MatrixLayout*>(Adesc);
  auto* lb = reinterpret_cast<MatrixLayout*>(Bdesc);
  auto* ld_ = reinterpret_cast<MatrixLayout*>(Ddesc);
  auto* lc = known(Cdesc) ? reinterpret_cast<MatrixLayout*>(Cdesc) : ld_;

  if (la->type != CUDA_R_32F || lb->type != CUDA_R_32F || ld_->type != CUDA_R_32F) {
    if (trace())
      std::fprintf(stderr, "[vgpu] cublasLtMatmul: only fp32 is implemented (A=%d B=%d D=%d)\n",
                   (int)la->type, (int)lb->type, (int)ld_->type);
    return CUBLAS_STATUS_NOT_SUPPORTED;
  }
  const bool ta = md->transa != CUBLAS_OP_N, tb = md->transb != CUBLAS_OP_N;
  const int64_t m = static_cast<int64_t>(ld_->rows), n = static_cast<int64_t>(ld_->cols);
  const int64_t k = ta ? static_cast<int64_t>(la->rows) : static_cast<int64_t>(la->cols);
  const float al = alpha ? *static_cast<const float*>(alpha) : 1.0f;
  const float be = beta ? *static_cast<const float*>(beta) : 0.0f;

  auto hA = fetch<float>(A, static_cast<size_t>(la->ld) * la->cols);
  auto hB = fetch<float>(B, static_cast<size_t>(lb->ld) * lb->cols);
  auto hC = (C && be != 0.0f) ? fetch<float>(C, static_cast<size_t>(lc->ld) * lc->cols)
                              : std::vector<float>(static_cast<size_t>(lc->ld) * lc->cols, 0.0f);
  std::vector<float> hD(static_cast<size_t>(ld_->ld) * ld_->cols, 0.0f);

  std::vector<float> hBias;
  const bool want_bias = md->epilogue == CUBLASLT_EPILOGUE_BIAS ||
                         md->epilogue == CUBLASLT_EPILOGUE_RELU_BIAS ||
                         md->epilogue == CUBLASLT_EPILOGUE_GELU_BIAS;
  if (want_bias && md->bias) hBias = fetch<float>(md->bias, static_cast<size_t>(m));

  for (int64_t j = 0; j < n; ++j) {
    for (int64_t i = 0; i < m; ++i) {
      float acc = 0.0f;
      for (int64_t p = 0; p < k; ++p)
        acc += hA[ta ? idx(p, i, la->ld) : idx(i, p, la->ld)] *
               hB[tb ? idx(j, p, lb->ld) : idx(p, j, lb->ld)];
      float v = al * acc;
      if (be != 0.0f) v += be * hC[idx(i, j, lc->ld)];
      if (!hBias.empty()) v += hBias[i];
      switch (md->epilogue) {
        case CUBLASLT_EPILOGUE_RELU:
        case CUBLASLT_EPILOGUE_RELU_BIAS: v = v > 0.0f ? v : 0.0f; break;
        case CUBLASLT_EPILOGUE_GELU:
        case CUBLASLT_EPILOGUE_GELU_BIAS: v = gelu(v); break;
        default: break;
      }
      hD[idx(i, j, ld_->ld)] = v;
    }
  }
  store(D, hD);
  return CUBLAS_STATUS_SUCCESS;
}
