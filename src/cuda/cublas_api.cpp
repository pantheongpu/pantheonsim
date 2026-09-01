// libvgpucublas — VirtualGPU's implementation of the cuBLAS API.
//
// WHY THIS IS NOT INTERPRETED
// ---------------------------
// cuBLAS is a *library*, not user code. Nothing requires its internals to run
// on the simulated device, and running a GEMM through the SIMT interpreter
// would be both pointless and ruinously slow (~10^8 lane-ops/s). So these
// entry points read their operands out of virtual device memory, do the
// arithmetic on the host CPU at native speed (~10^10 FLOP/s), and write the
// result back.
//
// That is a deliberate boundary, and it is the same one a real system draws:
// application kernels are simulated, vendor library calls are implemented.
// The practical effect is that a model whose matmuls go through cuBLAS spends
// almost no time in the interpreter, and only its custom kernels do.
//
// Semantics follow the documented API: column-major storage, C = alpha *
// op(A) * op(B) + beta * C, with the leading dimensions and transpose modes
// respected. Results are computed in the requested precision so they can be
// compared against real cuBLAS -- which is how these were validated.
#include <cublas_v2.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>
#include <vector>

#include "vgpu/error.hpp"
#include "vgpu/memory.hpp"

// Operands are read from and written to the same virtual device memory the
// kernels see, through the runtime shim's own copy path (declared by the CUDA
// headers this file already includes).
namespace {

constexpr cudaMemcpyKind kD2H = cudaMemcpyDeviceToHost;
constexpr cudaMemcpyKind kH2D = cudaMemcpyHostToDevice;

bool quiet() {
  const char* q = std::getenv("VGPU_QUIET");
  return q && q[0] == '1';
}
bool trace() {
  const char* t = std::getenv("VGPU_TRACE");
  return t && t[0] == '1';
}

struct Handle {
  cudaStream_t stream = nullptr;
  cublasPointerMode_t pointer_mode = CUBLAS_POINTER_MODE_HOST;
  cublasMath_t math_mode = CUBLAS_DEFAULT_MATH;
};

std::mutex g_mu;
std::set<Handle*> g_handles;

bool valid(cublasHandle_t h) {
  std::lock_guard<std::mutex> lock(g_mu);
  return h && g_handles.count(reinterpret_cast<Handle*>(h));
}

// Pulls `count` elements of T out of device memory.
template <class T>
std::vector<T> fetch(const void* dev, size_t count) {
  std::vector<T> host(count);
  if (count) cudaMemcpy(host.data(), dev, count * sizeof(T), kD2H);
  return host;
}

template <class T>
void store(void* dev, const std::vector<T>& host) {
  if (!host.empty()) cudaMemcpy(dev, host.data(), host.size() * sizeof(T), kH2D);
}

// A scalar argument is either a host pointer or a device pointer, depending on
// the handle's pointer mode. Getting this wrong silently corrupts results.
template <class T>
T scalar(cublasHandle_t h, const T* p) {
  if (!p) return T(0);
  Handle* hh = reinterpret_cast<Handle*>(h);
  if (hh && hh->pointer_mode == CUBLAS_POINTER_MODE_DEVICE) return fetch<T>(p, 1)[0];
  return *p;
}

// Column-major element access with a leading dimension.
inline size_t idx(int row, int col, int ld) { return static_cast<size_t>(col) * ld + row; }

// C = alpha * op(A) * op(B) + beta * C, computed in Acc and stored as T.
template <class T, class Acc>
void gemm_host(cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, Acc alpha,
               const std::vector<T>& A, int lda, const std::vector<T>& B, int ldb, Acc beta,
               std::vector<T>& C, int ldc) {
  const bool ta = transa != CUBLAS_OP_N, tb = transb != CUBLAS_OP_N;
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < m; ++i) {
      Acc acc = Acc(0);
      for (int p = 0; p < k; ++p) {
        Acc a = static_cast<Acc>(A[ta ? idx(p, i, lda) : idx(i, p, lda)]);
        Acc b = static_cast<Acc>(B[tb ? idx(j, p, ldb) : idx(p, j, ldb)]);
        acc += a * b;
      }
      T& c = C[idx(i, j, ldc)];
      // beta == 0 means C is write-only, so its prior contents (which may be
      // uninitialized) must not be read -- multiplying by zero would poison
      // the result with NaN.
      c = static_cast<T>(beta == Acc(0) ? alpha * acc
                                        : alpha * acc + beta * static_cast<Acc>(c));
    }
  }
}

// Elements a column-major matrix occupies given its leading dimension.
size_t extent(int ld, int cols) { return static_cast<size_t>(ld) * cols; }

template <class T, class Acc>
cublasStatus_t do_gemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
                       int m, int n, int k, const Acc* alpha_p, const void* A, int lda,
                       const void* B, int ldb, const Acc* beta_p, void* C, int ldc) {
  if (!valid(handle)) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (m < 0 || n < 0 || k < 0) return CUBLAS_STATUS_INVALID_VALUE;
  if (lda < 1 || ldb < 1 || ldc < m) return CUBLAS_STATUS_INVALID_VALUE;
  if (!alpha_p || !beta_p || (!A && k) || (!B && k) || !C) return CUBLAS_STATUS_INVALID_VALUE;
  Acc alpha = scalar(handle, alpha_p), beta = scalar(handle, beta_p);
  if (m == 0 || n == 0) return CUBLAS_STATUS_SUCCESS;

  auto hA = fetch<T>(A, extent(lda, transa == CUBLAS_OP_N ? k : m));
  auto hB = fetch<T>(B, extent(ldb, transb == CUBLAS_OP_N ? n : k));
  auto hC = fetch<T>(C, extent(ldc, n));
  gemm_host<T, Acc>(transa, transb, m, n, k, alpha, hA, lda, hB, ldb, beta, hC, ldc);
  store(C, hC);
  return CUBLAS_STATUS_SUCCESS;
}

}  // namespace

#define VGPU_EXPORT extern "C" __attribute__((visibility("default")))

/* ---- handles and configuration ---- */

VGPU_EXPORT cublasStatus_t cublasCreate_v2(cublasHandle_t* handle) {
  if (!handle) return CUBLAS_STATUS_INVALID_VALUE;
  auto* h = new Handle();
  {
    std::lock_guard<std::mutex> lock(g_mu);
    g_handles.insert(h);
  }
  *handle = reinterpret_cast<cublasHandle_t>(h);
  if (!quiet())
    std::fprintf(stderr, "[vgpu] cuBLAS handle created (host-computed; see docs/cublas.md)\n");
  return CUBLAS_STATUS_SUCCESS;
}

VGPU_EXPORT cublasStatus_t cublasDestroy_v2(cublasHandle_t handle) {
  std::lock_guard<std::mutex> lock(g_mu);
  auto* h = reinterpret_cast<Handle*>(handle);
  if (!h || !g_handles.erase(h)) return CUBLAS_STATUS_NOT_INITIALIZED;
  delete h;
  return CUBLAS_STATUS_SUCCESS;
}

VGPU_EXPORT cublasStatus_t cublasSetStream_v2(cublasHandle_t handle, cudaStream_t stream) {
  if (!valid(handle)) return CUBLAS_STATUS_NOT_INITIALIZED;
  reinterpret_cast<Handle*>(handle)->stream = stream;  // execution is synchronous
  return CUBLAS_STATUS_SUCCESS;
}
VGPU_EXPORT cublasStatus_t cublasGetStream_v2(cublasHandle_t handle, cudaStream_t* stream) {
  if (!valid(handle)) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (stream) *stream = reinterpret_cast<Handle*>(handle)->stream;
  return CUBLAS_STATUS_SUCCESS;
}
VGPU_EXPORT cublasStatus_t cublasSetPointerMode_v2(cublasHandle_t handle, cublasPointerMode_t m) {
  if (!valid(handle)) return CUBLAS_STATUS_NOT_INITIALIZED;
  reinterpret_cast<Handle*>(handle)->pointer_mode = m;
  return CUBLAS_STATUS_SUCCESS;
}
VGPU_EXPORT cublasStatus_t cublasGetPointerMode_v2(cublasHandle_t handle, cublasPointerMode_t* m) {
  if (!valid(handle)) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (m) *m = reinterpret_cast<Handle*>(handle)->pointer_mode;
  return CUBLAS_STATUS_SUCCESS;
}
VGPU_EXPORT cublasStatus_t cublasSetMathMode(cublasHandle_t handle, cublasMath_t mode) {
  if (!valid(handle)) return CUBLAS_STATUS_NOT_INITIALIZED;
  reinterpret_cast<Handle*>(handle)->math_mode = mode;
  return CUBLAS_STATUS_SUCCESS;
}
VGPU_EXPORT cublasStatus_t cublasGetMathMode(cublasHandle_t handle, cublasMath_t* mode) {
  if (!valid(handle)) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (mode) *mode = reinterpret_cast<Handle*>(handle)->math_mode;
  return CUBLAS_STATUS_SUCCESS;
}
VGPU_EXPORT cublasStatus_t cublasGetVersion_v2(cublasHandle_t, int* version) {
  if (version) *version = CUBLAS_VERSION;
  return CUBLAS_STATUS_SUCCESS;
}
VGPU_EXPORT cublasStatus_t cublasGetProperty(libraryPropertyType type, int* value) {
  if (!value) return CUBLAS_STATUS_INVALID_VALUE;
  *value = type == MAJOR_VERSION ? CUBLAS_VER_MAJOR
           : type == MINOR_VERSION ? CUBLAS_VER_MINOR
                                   : CUBLAS_VER_PATCH;
  return CUBLAS_STATUS_SUCCESS;
}
VGPU_EXPORT const char* cublasGetStatusName(cublasStatus_t s) {
  switch (s) {
    case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
    default: return "CUBLAS_STATUS_INTERNAL_ERROR";
  }
}
VGPU_EXPORT const char* cublasGetStatusString(cublasStatus_t s) { return cublasGetStatusName(s); }

/* ---- GEMM ---- */

VGPU_EXPORT cublasStatus_t cublasSgemm_v2(cublasHandle_t h, cublasOperation_t ta,
                                          cublasOperation_t tb, int m, int n, int k,
                                          const float* alpha, const float* A, int lda,
                                          const float* B, int ldb, const float* beta, float* C,
                                          int ldc) {
  return do_gemm<float, float>(h, ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

VGPU_EXPORT cublasStatus_t cublasDgemm_v2(cublasHandle_t h, cublasOperation_t ta,
                                          cublasOperation_t tb, int m, int n, int k,
                                          const double* alpha, const double* A, int lda,
                                          const double* B, int ldb, const double* beta, double* C,
                                          int ldc) {
  return do_gemm<double, double>(h, ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

VGPU_EXPORT cublasStatus_t cublasSgemmStridedBatched(
    cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k,
    const float* alpha, const float* A, int lda, long long strideA, const float* B, int ldb,
    long long strideB, const float* beta, float* C, int ldc, long long strideC, int batchCount) {
  for (int i = 0; i < batchCount; ++i) {
    cublasStatus_t s = cublasSgemm_v2(h, ta, tb, m, n, k, alpha, A + i * strideA, lda,
                                      B + i * strideB, ldb, beta, C + i * strideC, ldc);
    if (s != CUBLAS_STATUS_SUCCESS) return s;
  }
  return CUBLAS_STATUS_SUCCESS;
}

/* ---- level 2 and level 1 ---- */

VGPU_EXPORT cublasStatus_t cublasSgemv_v2(cublasHandle_t h, cublasOperation_t trans, int m, int n,
                                          const float* alpha, const float* A, int lda,
                                          const float* x, int incx, const float* beta, float* y,
                                          int incy) {
  if (!valid(h)) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (m < 0 || n < 0 || lda < 1 || incx == 0 || incy == 0) return CUBLAS_STATUS_INVALID_VALUE;
  float a = scalar(h, alpha), b = scalar(h, beta);
  const int xlen = trans == CUBLAS_OP_N ? n : m;
  const int ylen = trans == CUBLAS_OP_N ? m : n;
  if (!m || !n) return CUBLAS_STATUS_SUCCESS;
  auto hA = fetch<float>(A, extent(lda, n));
  auto hx = fetch<float>(x, static_cast<size_t>(std::abs(incx)) * (xlen - 1) + 1);
  auto hy = fetch<float>(y, static_cast<size_t>(std::abs(incy)) * (ylen - 1) + 1);
  for (int i = 0; i < ylen; ++i) {
    float acc = 0.0f;
    for (int j = 0; j < xlen; ++j)
      acc += hA[trans == CUBLAS_OP_N ? idx(i, j, lda) : idx(j, i, lda)] * hx[j * incx];
    float& yi = hy[i * incy];
    yi = b == 0.0f ? a * acc : a * acc + b * yi;
  }
  store(y, hy);
  return CUBLAS_STATUS_SUCCESS;
}

VGPU_EXPORT cublasStatus_t cublasSaxpy_v2(cublasHandle_t h, int n, const float* alpha,
                                          const float* x, int incx, float* y, int incy) {
  if (!valid(h)) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (n <= 0) return CUBLAS_STATUS_SUCCESS;
  float a = scalar(h, alpha);
  auto hx = fetch<float>(x, static_cast<size_t>(std::abs(incx)) * (n - 1) + 1);
  auto hy = fetch<float>(y, static_cast<size_t>(std::abs(incy)) * (n - 1) + 1);
  for (int i = 0; i < n; ++i) hy[i * incy] += a * hx[i * incx];
  store(y, hy);
  return CUBLAS_STATUS_SUCCESS;
}

VGPU_EXPORT cublasStatus_t cublasSscal_v2(cublasHandle_t h, int n, const float* alpha, float* x,
                                          int incx) {
  if (!valid(h)) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (n <= 0) return CUBLAS_STATUS_SUCCESS;
  float a = scalar(h, alpha);
  auto hx = fetch<float>(x, static_cast<size_t>(std::abs(incx)) * (n - 1) + 1);
  for (int i = 0; i < n; ++i) hx[i * incx] *= a;
  store(x, hx);
  return CUBLAS_STATUS_SUCCESS;
}

VGPU_EXPORT cublasStatus_t cublasSdot_v2(cublasHandle_t h, int n, const float* x, int incx,
                                         const float* y, int incy, float* result) {
  if (!valid(h)) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (!result) return CUBLAS_STATUS_INVALID_VALUE;
  double acc = 0.0;
  if (n > 0) {
    auto hx = fetch<float>(x, static_cast<size_t>(std::abs(incx)) * (n - 1) + 1);
    auto hy = fetch<float>(y, static_cast<size_t>(std::abs(incy)) * (n - 1) + 1);
    for (int i = 0; i < n; ++i) acc += static_cast<double>(hx[i * incx]) * hy[i * incy];
  }
  float out = static_cast<float>(acc);
  if (reinterpret_cast<Handle*>(h)->pointer_mode == CUBLAS_POINTER_MODE_DEVICE)
    cudaMemcpy(result, &out, sizeof out, kH2D);
  else
    *result = out;
  return CUBLAS_STATUS_SUCCESS;
}

VGPU_EXPORT cublasStatus_t cublasSnrm2_v2(cublasHandle_t h, int n, const float* x, int incx,
                                          float* result) {
  if (!valid(h)) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (!result) return CUBLAS_STATUS_INVALID_VALUE;
  double acc = 0.0;
  if (n > 0) {
    auto hx = fetch<float>(x, static_cast<size_t>(std::abs(incx)) * (n - 1) + 1);
    for (int i = 0; i < n; ++i) {
      double v = hx[i * incx];
      acc += v * v;
    }
  }
  float out = static_cast<float>(std::sqrt(acc));
  if (reinterpret_cast<Handle*>(h)->pointer_mode == CUBLAS_POINTER_MODE_DEVICE)
    cudaMemcpy(result, &out, sizeof out, kH2D);
  else
    *result = out;
  return CUBLAS_STATUS_SUCCESS;
}

/* ---- anything not implemented says so, rather than returning wrong numbers ---- */

VGPU_EXPORT cublasStatus_t cublasGemmEx(cublasHandle_t h, cublasOperation_t ta,
                                        cublasOperation_t tb, int m, int n, int k,
                                        const void* alpha, const void* A, cudaDataType Atype,
                                        int lda, const void* B, cudaDataType Btype, int ldb,
                                        const void* beta, void* C, cudaDataType Ctype, int ldc,
                                        cublasComputeType_t computeType, cublasGemmAlgo_t) {
  // Only the all-fp32 and all-fp64 forms are implemented; mixed precision
  // needs the f16/bf16 conversion paths and is not silently approximated.
  if (Atype == CUDA_R_32F && Btype == CUDA_R_32F && Ctype == CUDA_R_32F)
    return do_gemm<float, float>(h, ta, tb, m, n, k, static_cast<const float*>(alpha), A, lda, B,
                                 ldb, static_cast<const float*>(beta), C, ldc);
  if (Atype == CUDA_R_64F && Btype == CUDA_R_64F && Ctype == CUDA_R_64F)
    return do_gemm<double, double>(h, ta, tb, m, n, k, static_cast<const double*>(alpha), A, lda, B,
                                   ldb, static_cast<const double*>(beta), C, ldc);
  if (trace())
    std::fprintf(stderr,
                 "[vgpu] cublasGemmEx: unsupported type combination (A=%d B=%d C=%d compute=%d)\n",
                 (int)Atype, (int)Btype, (int)Ctype, (int)computeType);
  return CUBLAS_STATUS_NOT_SUPPORTED;
}
