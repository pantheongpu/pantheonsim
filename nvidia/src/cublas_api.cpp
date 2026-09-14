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
#include <cuda_bf16.h>
#include <cuda_fp16.h>
// cublas_v2.h pulls in the runtime API transitively under CUDA 13 but not
// under CUDA 12, and this file copies operands with cudaMemcpy.
#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <set>
#include <vector>

#include "vgpu/error.hpp"
#include "vgpu/memory.hpp"
#include "vgpu/runtime/capture.hpp"

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

// ---- CUDA graph capture ----
//
// On hardware a cuBLAS call inside a captured region is recorded, not run: it
// executes when the graph is launched, over whatever the graph's kernels have
// by then produced. Computing on the host at call time would instead read
// operands that do not exist yet -- llama.cpp fills its batched-GEMM pointer
// arrays with a kernel immediately before the GEMM, so an eager read gets
// uninitialized pool memory and dereferences it as device pointers.
//
// Deferring the whole call as a closure gets both halves right: nothing is read
// during capture, and the work is there on every replay.

// alpha and beta are consumed when the call is made, not when it runs, so a
// host-mode scalar has to be copied -- the caller's variable is typically a
// local that is long gone by the time a graph replays. A device-mode scalar is
// left alone, since that memory is still there and is meant to be read late.
struct HeldScalar {
  std::shared_ptr<std::vector<uint8_t>> owned;
  const void* dev = nullptr;
  const void* get() const { return owned ? static_cast<const void*>(owned->data()) : dev; }
};

inline HeldScalar hold(cublasHandle_t h, const void* p, size_t bytes) {
  HeldScalar s;
  Handle* hh = reinterpret_cast<Handle*>(h);
  if (!p || bytes == 0 || (hh && hh->pointer_mode == CUBLAS_POINTER_MODE_DEVICE)) {
    s.dev = p;
    return s;
  }
  const auto* b = static_cast<const uint8_t*>(p);
  s.owned = std::make_shared<std::vector<uint8_t>>(b, b + bytes);
  return s;
}

// GemmEx types alpha and beta by the compute type rather than by the operand
// types, so a snapshot has to know how many bytes to keep.
inline size_t compute_scalar_bytes(cublasComputeType_t ct) {
  switch (ct) {
    case CUBLAS_COMPUTE_16F:
    case CUBLAS_COMPUTE_16F_PEDANTIC: return sizeof(__half);
    case CUBLAS_COMPUTE_64F:
    case CUBLAS_COMPUTE_64F_PEDANTIC: return sizeof(double);
    default: return sizeof(float);  // the 32F and 32I families are both 4 bytes
  }
}

// Hands the call to the graph when the handle's stream is capturing. Returns
// true if it was recorded, in which case the caller must return success now
// without touching device memory.
template <class Fn>
bool deferred_to_graph(cublasHandle_t h, Fn&& fn) {
  Handle* hh = reinterpret_cast<Handle*>(h);
  if (!hh) return false;
  return vgpu_record_host_op_if_capturing(hh->stream, std::function<void()>(std::forward<Fn>(fn)));
}

// A reduction that writes its answer through a host pointer cannot be recorded:
// the graph runs later, and by then that pointer means nothing. Real cuBLAS
// rejects this too, so saying so is the honest answer rather than a guess.
inline bool host_result_under_capture(cublasHandle_t h) {
  Handle* hh = reinterpret_cast<Handle*>(h);
  if (!hh || hh->pointer_mode != CUBLAS_POINTER_MODE_HOST) return false;
  cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
  if (cudaStreamIsCapturing(hh->stream, &st) != cudaSuccess) return false;
  return st == cudaStreamCaptureStatusActive;
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

// Elements a column-major matrix occupies, given its leading dimension.
//
// The last column holds `rows` elements, not `ld` of them: the padding a
// leading dimension implies exists *between* columns, and there is none after
// the last. Reading ld*cols therefore ran past the end of any matrix allocated
// to exactly the size it needs -- which ggml does, and which the bounds check
// caught as a read of 1544 bytes past a 1792-byte allocation.
size_t extent(int ld, int cols, int rows) {
  if (cols <= 0 || rows <= 0) return 0;
  return static_cast<size_t>(ld) * (cols - 1) + rows;
}

/* ---- mixed precision ----
   GemmEx lets every operand carry its own type. Rather than instantiate the
   product of all of them, narrow operands are widened to float on the way in
   and narrowed again on the way out; the conversions come from the toolkit's
   own host-callable intrinsics rather than hand-written bit twiddling.
   Accumulation is in float, which is what a tensor core does under
   CUBLAS_COMPUTE_32F. */

bool load_as_float(const void* dev, size_t n, cudaDataType t, std::vector<float>* out) {
  out->assign(n, 0.0f);
  if (!n) return true;
  switch (t) {
    case CUDA_R_32F:
      return cudaMemcpy(out->data(), dev, n * sizeof(float), kD2H) == cudaSuccess;
    case CUDA_R_16F: {
      auto raw = fetch<__half>(dev, n);
      for (size_t i = 0; i < n; ++i) (*out)[i] = __half2float(raw[i]);
      return true;
    }
    case CUDA_R_16BF: {
      auto raw = fetch<__nv_bfloat16>(dev, n);
      for (size_t i = 0; i < n; ++i) (*out)[i] = __bfloat162float(raw[i]);
      return true;
    }
    default: return false;
  }
}

bool store_from_float(void* dev, const std::vector<float>& host, cudaDataType t) {
  if (host.empty()) return true;
  switch (t) {
    case CUDA_R_32F:
      return cudaMemcpy(dev, host.data(), host.size() * sizeof(float), kH2D) == cudaSuccess;
    case CUDA_R_16F: {
      std::vector<__half> raw(host.size());
      for (size_t i = 0; i < host.size(); ++i) raw[i] = __float2half(host[i]);
      store(dev, raw);
      return true;
    }
    case CUDA_R_16BF: {
      std::vector<__nv_bfloat16> raw(host.size());
      for (size_t i = 0; i < host.size(); ++i) raw[i] = __float2bfloat16(host[i]);
      store(dev, raw);
      return true;
    }
    default: return false;
  }
}

size_t type_bytes(cudaDataType t) {
  switch (t) {
    case CUDA_R_8I: return 1;
    case CUDA_R_16F: case CUDA_R_16BF: return 2;
    case CUDA_R_32F: case CUDA_R_32I: return 4;
    case CUDA_R_64F: return 8;
    default: return 0;
  }
}

bool is_narrow_float(cudaDataType t) {
  return t == CUDA_R_32F || t == CUDA_R_16F || t == CUDA_R_16BF;
}

// C = alpha * op(A) * op(B) + beta * C over already-widened host operands.
void gemm_float(cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
                float alpha, const std::vector<float>& A, int lda, const std::vector<float>& B,
                int ldb, float beta, std::vector<float>& C, int ldc) {
  const bool ta = transa != CUBLAS_OP_N, tb = transb != CUBLAS_OP_N;
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < m; ++i) {
      float acc = 0.0f;
      for (int p = 0; p < k; ++p)
        acc += A[ta ? idx(p, i, lda) : idx(i, p, lda)] * B[tb ? idx(j, p, ldb) : idx(p, j, ldb)];
      float& c = C[idx(i, j, ldc)];
      c = beta == 0.0f ? alpha * acc : alpha * acc + beta * c;
    }
}

// The int8 path: operands are int8, the product accumulates in int32, and C is
// int32. Nothing is widened to float, because rounding would change the answer.
void gemm_int8(cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
               int32_t alpha, const std::vector<int8_t>& A, int lda, const std::vector<int8_t>& B,
               int ldb, int32_t beta, std::vector<int32_t>& C, int ldc) {
  const bool ta = transa != CUBLAS_OP_N, tb = transb != CUBLAS_OP_N;
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < m; ++i) {
      int32_t acc = 0;
      for (int p = 0; p < k; ++p)
        acc += static_cast<int32_t>(A[ta ? idx(p, i, lda) : idx(i, p, lda)]) *
               static_cast<int32_t>(B[tb ? idx(j, p, ldb) : idx(p, j, ldb)]);
      int32_t& c = C[idx(i, j, ldc)];
      c = beta == 0 ? alpha * acc : alpha * acc + beta * c;
    }
}

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

  auto hA = fetch<T>(A, transa == CUBLAS_OP_N ? extent(lda, k, m) : extent(lda, m, k));
  auto hB = fetch<T>(B, transb == CUBLAS_OP_N ? extent(ldb, n, k) : extent(ldb, k, n));
  auto hC = fetch<T>(C, extent(ldc, n, m));
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
    std::fprintf(stderr, "[vgpu] cuBLAS handle created (host-computed; see nvidia/docs/cublas.md)\n");
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
  if (deferred_to_graph(h, [=, a = hold(h, alpha, sizeof(float)),
                            b = hold(h, beta, sizeof(float))] {
        cublasSgemm_v2(h, ta, tb, m, n, k, static_cast<const float*>(a.get()), A, lda, B, ldb,
                       static_cast<const float*>(b.get()), C, ldc);
      }))
    return CUBLAS_STATUS_SUCCESS;
  return do_gemm<float, float>(h, ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

VGPU_EXPORT cublasStatus_t cublasDgemm_v2(cublasHandle_t h, cublasOperation_t ta,
                                          cublasOperation_t tb, int m, int n, int k,
                                          const double* alpha, const double* A, int lda,
                                          const double* B, int ldb, const double* beta, double* C,
                                          int ldc) {
  if (deferred_to_graph(h, [=, a = hold(h, alpha, sizeof(double)),
                            b = hold(h, beta, sizeof(double))] {
        cublasDgemm_v2(h, ta, tb, m, n, k, static_cast<const double*>(a.get()), A, lda, B, ldb,
                       static_cast<const double*>(b.get()), C, ldc);
      }))
    return CUBLAS_STATUS_SUCCESS;
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

// The pointer-array form: each batch entry is an independent matrix rather
// than a fixed stride apart. The pointers live in device memory, so the array
// itself has to be read back before it can be walked.
VGPU_EXPORT cublasStatus_t cublasSgemmBatched(cublasHandle_t h, cublasOperation_t ta,
                                              cublasOperation_t tb, int m, int n, int k,
                                              const float* alpha, const float* const Aarray[],
                                              int lda, const float* const Barray[], int ldb,
                                              const float* beta, float* const Carray[], int ldc,
                                              int batchCount) {
  if (!valid(h)) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (batchCount < 0) return CUBLAS_STATUS_INVALID_VALUE;
  if (batchCount == 0) return CUBLAS_STATUS_SUCCESS;
  if (!Aarray || !Barray || !Carray) return CUBLAS_STATUS_INVALID_VALUE;
  if (deferred_to_graph(h, [=, al = hold(h, alpha, sizeof(float)),
                            be = hold(h, beta, sizeof(float))] {
        cublasSgemmBatched(h, ta, tb, m, n, k, static_cast<const float*>(al.get()), Aarray, lda,
                           Barray, ldb, static_cast<const float*>(be.get()), Carray, ldc,
                           batchCount);
      }))
    return CUBLAS_STATUS_SUCCESS;
  // The three arrays are themselves in device memory; pull the pointers back
  // before dereferencing them.
  const auto a = fetch<const float*>(Aarray, static_cast<size_t>(batchCount));
  const auto b = fetch<const float*>(Barray, static_cast<size_t>(batchCount));
  const auto c = fetch<float*>(Carray, static_cast<size_t>(batchCount));
  for (int i = 0; i < batchCount; ++i) {
    cublasStatus_t st =
        cublasSgemm_v2(h, ta, tb, m, n, k, alpha, a[i], lda, b[i], ldb, beta, c[i], ldc);
    if (st != CUBLAS_STATUS_SUCCESS) return st;
  }
  return CUBLAS_STATUS_SUCCESS;
}

// A workspace is a scratch buffer the library would use for its own tiling.
// Nothing here needs one, so accepting it is honest: the caller's buffer simply
// goes unused, and refusing would stop a program that is doing nothing wrong.
VGPU_EXPORT cublasStatus_t cublasSetWorkspace_v2(cublasHandle_t h, void*, size_t) {
  return valid(h) ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_NOT_INITIALIZED;
}

/* ---- level 2 and level 1 ---- */

VGPU_EXPORT cublasStatus_t cublasSgemv_v2(cublasHandle_t h, cublasOperation_t trans, int m, int n,
                                          const float* alpha, const float* A, int lda,
                                          const float* x, int incx, const float* beta, float* y,
                                          int incy) {
  if (!valid(h)) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (m < 0 || n < 0 || lda < 1 || incx == 0 || incy == 0) return CUBLAS_STATUS_INVALID_VALUE;
  if (deferred_to_graph(h, [=, al = hold(h, alpha, sizeof(float)),
                            be = hold(h, beta, sizeof(float))] {
        cublasSgemv_v2(h, trans, m, n, static_cast<const float*>(al.get()), A, lda, x, incx,
                       static_cast<const float*>(be.get()), y, incy);
      }))
    return CUBLAS_STATUS_SUCCESS;
  float a = scalar(h, alpha), b = scalar(h, beta);
  const int xlen = trans == CUBLAS_OP_N ? n : m;
  const int ylen = trans == CUBLAS_OP_N ? m : n;
  if (!m || !n) return CUBLAS_STATUS_SUCCESS;
  auto hA = fetch<float>(A, extent(lda, n, m));
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
  if (deferred_to_graph(h, [=, al = hold(h, alpha, sizeof(float))] {
        cublasSaxpy_v2(h, n, static_cast<const float*>(al.get()), x, incx, y, incy);
      }))
    return CUBLAS_STATUS_SUCCESS;
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
  if (deferred_to_graph(h, [=, al = hold(h, alpha, sizeof(float))] {
        cublasSscal_v2(h, n, static_cast<const float*>(al.get()), x, incx);
      }))
    return CUBLAS_STATUS_SUCCESS;
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
  if (host_result_under_capture(h)) return CUBLAS_STATUS_NOT_SUPPORTED;
  if (deferred_to_graph(h, [=] { cublasSdot_v2(h, n, x, incx, y, incy, result); }))
    return CUBLAS_STATUS_SUCCESS;
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
  if (host_result_under_capture(h)) return CUBLAS_STATUS_NOT_SUPPORTED;
  if (deferred_to_graph(h, [=] { cublasSnrm2_v2(h, n, x, incx, result); }))
    return CUBLAS_STATUS_SUCCESS;
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
                                        cublasComputeType_t computeType, cublasGemmAlgo_t algo) {
  if (deferred_to_graph(h, [=, al = hold(h, alpha, compute_scalar_bytes(computeType)),
                            be = hold(h, beta, compute_scalar_bytes(computeType))] {
        cublasGemmEx(h, ta, tb, m, n, k, al.get(), A, Atype, lda, B, Btype, ldb, be.get(), C,
                     Ctype, ldc, computeType, algo);
      }))
    return CUBLAS_STATUS_SUCCESS;
  // Only the all-fp32 and all-fp64 forms are implemented; mixed precision
  // needs the f16/bf16 conversion paths and is not silently approximated.
  if (Atype == CUDA_R_32F && Btype == CUDA_R_32F && Ctype == CUDA_R_32F)
    return do_gemm<float, float>(h, ta, tb, m, n, k, static_cast<const float*>(alpha), A, lda, B,
                                 ldb, static_cast<const float*>(beta), C, ldc);
  if (Atype == CUDA_R_64F && Btype == CUDA_R_64F && Ctype == CUDA_R_64F)
    return do_gemm<double, double>(h, ta, tb, m, n, k, static_cast<const double*>(alpha), A, lda, B,
                                   ldb, static_cast<const double*>(beta), C, ldc);
  // Mixed precision: half or bfloat16 operands accumulated in float, which is
  // what a tensor core does under CUBLAS_COMPUTE_32F. C may be narrower than
  // the accumulator, so it is rounded once on the way out.
  if (is_narrow_float(Atype) && is_narrow_float(Btype) && is_narrow_float(Ctype) &&
      computeType != CUBLAS_COMPUTE_32I && computeType != CUBLAS_COMPUTE_64F) {
    if (!valid(h)) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (m < 0 || n < 0 || k < 0 || lda < 1 || ldb < 1 || ldc < m)
      return CUBLAS_STATUS_INVALID_VALUE;
    if (!alpha || !beta || !C) return CUBLAS_STATUS_INVALID_VALUE;
    if (m == 0 || n == 0) return CUBLAS_STATUS_SUCCESS;
    // alpha and beta are float here whatever the operand types, except under
    // CUBLAS_COMPUTE_16F where the API says they are half.
    float a, b;
    if (computeType == CUBLAS_COMPUTE_16F || computeType == CUBLAS_COMPUTE_16F_PEDANTIC) {
      auto widen = [&](const void* p) {
        __half v;
        if (reinterpret_cast<Handle*>(h)->pointer_mode == CUBLAS_POINTER_MODE_DEVICE)
          v = fetch<__half>(p, 1)[0];
        else
          v = *static_cast<const __half*>(p);
        return __half2float(v);
      };
      a = widen(alpha);
      b = widen(beta);
    } else {
      a = scalar(h, static_cast<const float*>(alpha));
      b = scalar(h, static_cast<const float*>(beta));
    }
    std::vector<float> hA, hB, hC;
    if (!load_as_float(A, ta == CUBLAS_OP_N ? extent(lda, k, m) : extent(lda, m, k), Atype, &hA) ||
        !load_as_float(B, tb == CUBLAS_OP_N ? extent(ldb, n, k) : extent(ldb, k, n), Btype, &hB) ||
        !load_as_float(C, extent(ldc, n, m), Ctype, &hC))
      return CUBLAS_STATUS_NOT_SUPPORTED;
    gemm_float(ta, tb, m, n, k, a, hA, lda, hB, ldb, b, hC, ldc);
    return store_from_float(C, hC, Ctype) ? CUBLAS_STATUS_SUCCESS
                                          : CUBLAS_STATUS_NOT_SUPPORTED;
  }

  // The quantized path: int8 operands, int32 accumulator and output.
  if (Atype == CUDA_R_8I && Btype == CUDA_R_8I && Ctype == CUDA_R_32I) {
    if (!valid(h)) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (m < 0 || n < 0 || k < 0 || lda < 1 || ldb < 1 || ldc < m)
      return CUBLAS_STATUS_INVALID_VALUE;
    if (!alpha || !beta || !C) return CUBLAS_STATUS_INVALID_VALUE;
    if (m == 0 || n == 0) return CUBLAS_STATUS_SUCCESS;
    const int32_t a = scalar(h, static_cast<const int32_t*>(alpha));
    const int32_t b = scalar(h, static_cast<const int32_t*>(beta));
    auto hA = fetch<int8_t>(A, ta == CUBLAS_OP_N ? extent(lda, k, m) : extent(lda, m, k));
    auto hB = fetch<int8_t>(B, tb == CUBLAS_OP_N ? extent(ldb, n, k) : extent(ldb, k, n));
    auto hC = fetch<int32_t>(C, extent(ldc, n, m));
    gemm_int8(ta, tb, m, n, k, a, hA, lda, hB, ldb, b, hC, ldc);
    store(C, hC);
    return CUBLAS_STATUS_SUCCESS;
  }

  if (trace())
    std::fprintf(stderr,
                 "[vgpu] cublasGemmEx: unsupported type combination (A=%d B=%d C=%d compute=%d)\n",
                 (int)Atype, (int)Btype, (int)Ctype, (int)computeType);
  return CUBLAS_STATUS_NOT_SUPPORTED;
}

VGPU_EXPORT cublasStatus_t cublasGemmStridedBatchedEx(
    cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k,
    const void* alpha, const void* A, cudaDataType Atype, int lda, long long strideA,
    const void* B, cudaDataType Btype, int ldb, long long strideB, const void* beta, void* C,
    cudaDataType Ctype, int ldc, long long strideC, int batchCount,
    cublasComputeType_t computeType, cublasGemmAlgo_t algo) {
  const size_t ea = type_bytes(Atype), eb = type_bytes(Btype), ec = type_bytes(Ctype);
  if (!ea || !eb || !ec) return CUBLAS_STATUS_NOT_SUPPORTED;
  for (int i = 0; i < batchCount; ++i) {
    const cublasStatus_t s = cublasGemmEx(
        h, ta, tb, m, n, k, alpha, static_cast<const char*>(A) + (size_t)i * strideA * ea, Atype,
        lda, static_cast<const char*>(B) + (size_t)i * strideB * eb, Btype, ldb, beta,
        static_cast<char*>(C) + (size_t)i * strideC * ec, Ctype, ldc, computeType, algo);
    if (s != CUBLAS_STATUS_SUCCESS) return s;
  }
  return CUBLAS_STATUS_SUCCESS;
}

// The pointer-array form of GemmEx: each batch entry is an independent matrix
// rather than a fixed stride apart, and the three arrays live in device memory.
VGPU_EXPORT cublasStatus_t cublasGemmBatchedEx(
    cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k,
    const void* alpha, const void* const Aarray[], cudaDataType Atype, int lda,
    const void* const Barray[], cudaDataType Btype, int ldb, const void* beta, void* const Carray[],
    cudaDataType Ctype, int ldc, int batchCount, cublasComputeType_t computeType,
    cublasGemmAlgo_t algo) {
  if (!valid(h)) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (batchCount < 0) return CUBLAS_STATUS_INVALID_VALUE;
  if (batchCount == 0) return CUBLAS_STATUS_SUCCESS;
  if (!Aarray || !Barray || !Carray) return CUBLAS_STATUS_INVALID_VALUE;
  if (deferred_to_graph(h, [=, al = hold(h, alpha, compute_scalar_bytes(computeType)),
                            be = hold(h, beta, compute_scalar_bytes(computeType))] {
        cublasGemmBatchedEx(h, ta, tb, m, n, k, al.get(), Aarray, Atype, lda, Barray, Btype, ldb,
                            be.get(), Carray, Ctype, ldc, batchCount, computeType, algo);
      }))
    return CUBLAS_STATUS_SUCCESS;
  const auto a = fetch<const void*>(Aarray, static_cast<size_t>(batchCount));
  const auto b = fetch<const void*>(Barray, static_cast<size_t>(batchCount));
  const auto c = fetch<void*>(Carray, static_cast<size_t>(batchCount));
  for (int i = 0; i < batchCount; ++i) {
    const cublasStatus_t st = cublasGemmEx(h, ta, tb, m, n, k, alpha, a[i], Atype, lda, b[i], Btype,
                                           ldb, beta, c[i], Ctype, ldc, computeType, algo);
    if (st != CUBLAS_STATUS_SUCCESS) return st;
  }
  return CUBLAS_STATUS_SUCCESS;
}

// cublasHgemm accumulates in half on hardware only when the math mode asks for
// it; the default path uses a float accumulator, which is what this does.
VGPU_EXPORT cublasStatus_t cublasHgemm(cublasHandle_t h, cublasOperation_t ta,
                                       cublasOperation_t tb, int m, int n, int k,
                                       const __half* alpha, const __half* A, int lda,
                                       const __half* B, int ldb, const __half* beta, __half* C,
                                       int ldc) {
  return cublasGemmEx(h, ta, tb, m, n, k, alpha, A, CUDA_R_16F, lda, B, CUDA_R_16F, ldb, beta, C,
                      CUDA_R_16F, ldc, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT);
}
