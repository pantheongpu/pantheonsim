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

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <set>
#include <vector>

#include "vgpu/error.hpp"
#include "vgpu/memory.hpp"
#include "vgpu/runtime/capture.hpp"
#include "enum_value.hpp"

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
  // Settings the host-computed routines have no use for, kept so that what
  // a program sets it reads back (see "handle settings" below).
  cublasAtomicsMode_t atomics = CUBLAS_ATOMICS_NOT_ALLOWED;
  int sm_count_target = 0;
  int emulation_strategy = 0, emulation_special_values = 0xFFFF, mantissa_control = 0;
  int max_mantissa_bits = 0, mantissa_bit_offset = 0;
  int* mantissa_bit_count = nullptr;
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
  if (!p) return T{};
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
    case CUDA_R_8I: {
      auto raw = fetch<int8_t>(dev, n);
      for (size_t i = 0; i < n; ++i) (*out)[i] = raw[i];
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
    case CUDA_R_16F: case CUDA_R_16BF: case CUDA_C_8I: return 2;
    case CUDA_R_32F: case CUDA_R_32I: return 4;
    case CUDA_R_64F: case CUDA_C_32F: return 8;
    case CUDA_C_64F: return 16;
    default: return 0;
  }
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

// The arguments a GEMM refuses, as an RTX 3060's cuBLAS does: an unknown
// operation, a negative size, or a leading dimension shorter than its
// matrix's rows (and never below 1, even for an empty matrix).
bool known_op(cublasOperation_t t) { return t == CUBLAS_OP_N || t == CUBLAS_OP_T || t == CUBLAS_OP_C; }
bool gemm_args_ok(cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k, int lda, int ldb, int ldc) {
  return known_op(ta) && known_op(tb) && m >= 0 && n >= 0 && k >= 0 && lda >= std::max(1, ta == CUBLAS_OP_N ? m : k) &&
         ldb >= std::max(1, tb == CUBLAS_OP_N ? k : n) && ldc >= std::max(1, m);
}

// A matrix whose pointer is not aligned to its element is refused by the
// card's real GEMMs as NOT_SUPPORTED (an int8 C of GemmEx too).
bool gemm_aligned(const void* A, const void* B, const void* C, size_t ab, size_t c) {
  auto ok = [](const void* p, size_t e) { return reinterpret_cast<uintptr_t>(p) % e == 0; };
  return ok(A, ab) && ok(B, ab) && ok(C, c);
}

bool known_fill(cublasFillMode_t u) { return u == CUBLAS_FILL_MODE_LOWER || u == CUBLAS_FILL_MODE_UPPER; }
bool known_side(cublasSideMode_t s) { return s == CUBLAS_SIDE_LEFT || s == CUBLAS_SIDE_RIGHT; }
bool known_diag(cublasDiagType_t d) { return d == CUBLAS_DIAG_NON_UNIT || d == CUBLAS_DIAG_UNIT; }
// trsm and trmm: A is m x m from the left, n x n from the right; B is m x n.
// The card refuses FILL_MODE_FULL here too.
bool tri3_args_ok(cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t t, cublasDiagType_t diag, int m,
                  int n, int lda, int ldb) {
  return known_side(side) && known_fill(uplo) && known_op(t) && known_diag(diag) && m >= 0 && n >= 0 &&
         lda >= std::max(1, side == CUBLAS_SIDE_LEFT ? m : n) && ldb >= std::max(1, m);
}

template <class T, class Acc>
cublasStatus_t do_gemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
                       int m, int n, int k, const Acc* alpha_p, const void* A, int lda,
                       const void* B, int ldb, const Acc* beta_p, void* C, int ldc) {
  if (!valid(handle)) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (!gemm_args_ok(transa, transb, m, n, k, lda, ldb, ldc)) return CUBLAS_STATUS_INVALID_VALUE;
  if (!gemm_aligned(A, B, C, sizeof(T), sizeof(T))) return CUBLAS_STATUS_NOT_SUPPORTED;
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
// ---- host <-> device copies: cublasSetVector, cublasGetMatrix and the rest ----
//
// The legacy helpers that move a strided vector or a column-major matrix
// between host and device. Each is one cudaMemcpy2D: a vector's elements are
// rows one element wide, a matrix's columns are rows `rows` elements wide.
// qulacs moves its state vectors with them.
namespace {
cublasStatus_t copy_2d(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width,
                       size_t height, cudaMemcpyKind kind, cudaStream_t stream, bool async) {
  if (!width || !height) return CUBLAS_STATUS_SUCCESS;
  const cudaError_t e = async ? cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, kind, stream)
                              : cudaMemcpy2D(dst, dpitch, src, spitch, width, height, kind);
  return e == cudaSuccess ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_MAPPING_ERROR;
}
cublasStatus_t copy_vector(int n, int elem, const void* x, int incx, void* y, int incy,
                           cudaMemcpyKind kind, cudaStream_t stream = nullptr, bool async = false) {
  if (incx <= 0 || incy <= 0 || elem <= 0) return CUBLAS_STATUS_INVALID_VALUE;
  if (n <= 0) return CUBLAS_STATUS_SUCCESS;
  return copy_2d(y, size_t(incy) * elem, x, size_t(incx) * elem, size_t(elem), size_t(n), kind, stream,
                 async);
}
cublasStatus_t copy_matrix(int rows, int cols, int elem, const void* A, int lda, void* B, int ldb,
                           cudaMemcpyKind kind, cudaStream_t stream = nullptr, bool async = false) {
  if (rows < 0 || cols < 0 || elem <= 0 || lda <= 0 || ldb <= 0) return CUBLAS_STATUS_INVALID_VALUE;
  // A leading dimension below `rows` is a pitch narrower than a row, which
  // cudaMemcpy2D refuses; cuBLAS reports that as a mapping error, as here.
  return copy_2d(B, size_t(ldb) * elem, A, size_t(lda) * elem, size_t(rows) * elem, size_t(cols), kind,
                 stream, async);
}
}  // namespace

VGPU_EXPORT cublasStatus_t cublasSetVector(int n, int elemSize, const void* x, int incx, void* y, int incy) {
  return copy_vector(n, elemSize, x, incx, y, incy, cudaMemcpyHostToDevice);
}
VGPU_EXPORT cublasStatus_t cublasGetVector(int n, int elemSize, const void* x, int incx, void* y, int incy) {
  return copy_vector(n, elemSize, x, incx, y, incy, cudaMemcpyDeviceToHost);
}
VGPU_EXPORT cublasStatus_t cublasSetMatrix(int rows, int cols, int elemSize, const void* A, int lda, void* B,
                                           int ldb) {
  return copy_matrix(rows, cols, elemSize, A, lda, B, ldb, cudaMemcpyHostToDevice);
}
VGPU_EXPORT cublasStatus_t cublasGetMatrix(int rows, int cols, int elemSize, const void* A, int lda, void* B,
                                           int ldb) {
  return copy_matrix(rows, cols, elemSize, A, lda, B, ldb, cudaMemcpyDeviceToHost);
}
VGPU_EXPORT cublasStatus_t cublasSetVectorAsync(int n, int elemSize, const void* x, int incx, void* y,
                                                int incy, cudaStream_t stream) {
  return copy_vector(n, elemSize, x, incx, y, incy, cudaMemcpyHostToDevice, stream, true);
}
VGPU_EXPORT cublasStatus_t cublasGetVectorAsync(int n, int elemSize, const void* x, int incx, void* y,
                                                int incy, cudaStream_t stream) {
  return copy_vector(n, elemSize, x, incx, y, incy, cudaMemcpyDeviceToHost, stream, true);
}
VGPU_EXPORT cublasStatus_t cublasSetMatrixAsync(int rows, int cols, int elemSize, const void* A, int lda,
                                                void* B, int ldb, cudaStream_t stream) {
  return copy_matrix(rows, cols, elemSize, A, lda, B, ldb, cudaMemcpyHostToDevice, stream, true);
}
VGPU_EXPORT cublasStatus_t cublasGetMatrixAsync(int rows, int cols, int elemSize, const void* A, int lda,
                                                void* B, int ldb, cudaStream_t stream) {
  return copy_matrix(rows, cols, elemSize, A, lda, B, ldb, cudaMemcpyDeviceToHost, stream, true);
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

/* ---- handle settings ----
   The atomics mode, the SM count target and CUDA 13's floating-point
   emulation controls (which apply to Blackwell's emulated GEMMs). Nothing
   here runs on SMs or emulates anything, so each is kept and read back,
   with the defaults and the refusals an RTX 3060's cuBLAS 13.0 gives
   (measured): atomics not allowed, SM target 0, emulation strategy DEFAULT,
   every special value supported (0xFFFF), mantissa control DYNAMIC, a max
   mantissa bit count and offset of 0 and no bit-count pointer; an unknown
   atomics mode, strategy or mantissa control, an SM target below 0 or above
   the device's SM count, or a negative max mantissa bit count is
   INVALID_VALUE, as is a NULL pointer to a getter. */
#if CUBLAS_VER_MAJOR >= 13
using vgpu_emu_strategy = cublasEmulationStrategy_t;
using vgpu_emu_special = cudaEmulationSpecialValuesSupport;
using vgpu_emu_mantissa = cudaEmulationMantissaControl;
#else   // the CUDA 12 headers declare none of these
using vgpu_emu_strategy = int;
using vgpu_emu_special = int;
using vgpu_emu_mantissa = int;
#endif
namespace {
template <class V, class F>
cublasStatus_t get_setting(cublasHandle_t h, V* out, F field) {
  if (!valid(h)) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (!out) return CUBLAS_STATUS_INVALID_VALUE;
  *out = (V)field(reinterpret_cast<Handle*>(h));
  return CUBLAS_STATUS_SUCCESS;
}
template <class F>
cublasStatus_t set_setting(cublasHandle_t h, bool ok, F apply) {
  if (!valid(h)) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (!ok) return CUBLAS_STATUS_INVALID_VALUE;
  apply(reinterpret_cast<Handle*>(h));
  return CUBLAS_STATUS_SUCCESS;
}
int device_sm_count() {
  int dev = 0, sms = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  return sms;
}
}  // namespace

VGPU_EXPORT cublasStatus_t cublasGetAtomicsMode(cublasHandle_t h, cublasAtomicsMode_t* mode) {
  return get_setting(h, mode, [](Handle* x) { return x->atomics; });
}
// Each setter reads its argument as an integer (enum_value.hpp) and checks
// that before the value is used as the enum.
VGPU_EXPORT cublasStatus_t cublasSetAtomicsMode(cublasHandle_t h, cublasAtomicsMode_t mode) {
  const int v = enum_value(mode);
  return set_setting(h, v == CUBLAS_ATOMICS_NOT_ALLOWED || v == CUBLAS_ATOMICS_ALLOWED,
                     [v](Handle* x) { x->atomics = (cublasAtomicsMode_t)v; });
}
VGPU_EXPORT cublasStatus_t cublasGetSmCountTarget(cublasHandle_t h, int* target) {
  return get_setting(h, target, [](Handle* x) { return x->sm_count_target; });
}
VGPU_EXPORT cublasStatus_t cublasSetSmCountTarget(cublasHandle_t h, int target) {
  return set_setting(h, valid(h) && target >= 0 && target <= device_sm_count(),
                     [target](Handle* x) { x->sm_count_target = target; });
}
VGPU_EXPORT cublasStatus_t cublasGetEmulationStrategy(cublasHandle_t h, vgpu_emu_strategy* s) {
  return get_setting(h, s, [](Handle* x) { return x->emulation_strategy; });
}
VGPU_EXPORT cublasStatus_t cublasSetEmulationStrategy(cublasHandle_t h, vgpu_emu_strategy s) {
  const int v = enum_value(s);
  return set_setting(h, v >= 0 && v <= 2, [v](Handle* x) { x->emulation_strategy = v; });
}
VGPU_EXPORT cublasStatus_t cublasGetEmulationSpecialValuesSupport(cublasHandle_t h, vgpu_emu_special* mask) {
  return get_setting(h, mask, [](Handle* x) { return x->emulation_special_values; });
}
VGPU_EXPORT cublasStatus_t cublasSetEmulationSpecialValuesSupport(cublasHandle_t h, vgpu_emu_special mask) {
  const int v = enum_value(mask);   // a bit mask: any combination is taken
  return set_setting(h, true, [v](Handle* x) { x->emulation_special_values = v; });
}
VGPU_EXPORT cublasStatus_t cublasGetFixedPointEmulationMantissaControl(cublasHandle_t h, vgpu_emu_mantissa* c) {
  return get_setting(h, c, [](Handle* x) { return x->mantissa_control; });
}
VGPU_EXPORT cublasStatus_t cublasSetFixedPointEmulationMantissaControl(cublasHandle_t h, vgpu_emu_mantissa c) {
  const int v = enum_value(c);
  return set_setting(h, v == 0 || v == 1, [v](Handle* x) { x->mantissa_control = v; });
}
VGPU_EXPORT cublasStatus_t cublasGetFixedPointEmulationMaxMantissaBitCount(cublasHandle_t h, int* bits) {
  return get_setting(h, bits, [](Handle* x) { return x->max_mantissa_bits; });
}
VGPU_EXPORT cublasStatus_t cublasSetFixedPointEmulationMaxMantissaBitCount(cublasHandle_t h, int bits) {
  return set_setting(h, bits >= 0, [bits](Handle* x) { x->max_mantissa_bits = bits; });
}
VGPU_EXPORT cublasStatus_t cublasGetFixedPointEmulationMantissaBitOffset(cublasHandle_t h, int* offset) {
  return get_setting(h, offset, [](Handle* x) { return x->mantissa_bit_offset; });
}
VGPU_EXPORT cublasStatus_t cublasSetFixedPointEmulationMantissaBitOffset(cublasHandle_t h, int offset) {
  return set_setting(h, true, [offset](Handle* x) { x->mantissa_bit_offset = offset; });
}
VGPU_EXPORT cublasStatus_t cublasGetFixedPointEmulationMantissaBitCountPointer(cublasHandle_t h, int** p) {
  return get_setting(h, p, [](Handle* x) { return x->mantissa_bit_count; });
}
VGPU_EXPORT cublasStatus_t cublasSetFixedPointEmulationMantissaBitCountPointer(cublasHandle_t h, int* p) {
  return set_setting(h, true, [p](Handle* x) { x->mantissa_bit_count = p; });
}
VGPU_EXPORT size_t cublasGetCudartVersion(void) { return CUDART_VERSION; }

// The API logger: there is nothing to log, so the settings are taken and the
// callback kept for cublasGetLoggerCallback.
namespace {
std::mutex g_log_mu;
cublasLogCallback g_log_callback = nullptr;
}  // namespace
VGPU_EXPORT cublasStatus_t cublasLoggerConfigure(int, int, int, const char*) { return CUBLAS_STATUS_SUCCESS; }
VGPU_EXPORT cublasStatus_t cublasSetLoggerCallback(cublasLogCallback cb) {
  std::lock_guard<std::mutex> lock(g_log_mu);
  g_log_callback = cb;
  return CUBLAS_STATUS_SUCCESS;
}
VGPU_EXPORT cublasStatus_t cublasGetLoggerCallback(cublasLogCallback* cb) {
  if (!cb) return CUBLAS_STATUS_INVALID_VALUE;
  std::lock_guard<std::mutex> lock(g_log_mu);
  *cb = g_log_callback;
  return CUBLAS_STATUS_SUCCESS;
}
// BLAS's error report, in the card's words, on standard output.
VGPU_EXPORT void cublasXerbla(const char* srName, int info) {
  std::printf(" ** On entry to %s parameter number %d had an illegal value\n", srName ? srName : "", info);
}

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

// The batched forms: every matrix's arguments are checked once, before any
// work and even for an empty batch, as the card does.
namespace {

template <class T>
cublasStatus_t gemm_one(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k,
                        const T* alpha, const T* A, int lda, const T* B, int ldb, const T* beta, T* C, int ldc) {
  if constexpr (std::is_same_v<T, float>) return cublasSgemm_v2(h, ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  else return cublasDgemm_v2(h, ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <class T>
cublasStatus_t gemm_strided(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k,
                            const T* alpha, const T* A, int lda, long long strideA, const T* B, int ldb,
                            long long strideB, const T* beta, T* C, int ldc, long long strideC, int batchCount) {
  if (!valid(h)) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (!gemm_args_ok(ta, tb, m, n, k, lda, ldb, ldc) || batchCount < 0) return CUBLAS_STATUS_INVALID_VALUE;
  for (int i = 0; i < batchCount; ++i) {
    cublasStatus_t s = gemm_one<T>(h, ta, tb, m, n, k, alpha, A + i * strideA, lda, B + i * strideB, ldb, beta,
                                   C + i * strideC, ldc);
    if (s != CUBLAS_STATUS_SUCCESS) return s;
  }
  return CUBLAS_STATUS_SUCCESS;
}

// The pointer-array form: each batch entry is an independent matrix rather
// than a fixed stride apart. The pointers live in device memory, so the array
// itself has to be read back before it can be walked.
template <class T>
cublasStatus_t gemm_batched(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k,
                            const T* alpha, const T* const Aarray[], int lda, const T* const Barray[], int ldb,
                            const T* beta, T* const Carray[], int ldc, int batchCount) {
  if (!valid(h)) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (!gemm_args_ok(ta, tb, m, n, k, lda, ldb, ldc) || batchCount < 0) return CUBLAS_STATUS_INVALID_VALUE;
  if (batchCount == 0) return CUBLAS_STATUS_SUCCESS;
  if (!Aarray || !Barray || !Carray) return CUBLAS_STATUS_INVALID_VALUE;
  if (deferred_to_graph(h, [=, al = hold(h, alpha, sizeof(T)), be = hold(h, beta, sizeof(T))] {
        gemm_batched<T>(h, ta, tb, m, n, k, static_cast<const T*>(al.get()), Aarray, lda, Barray, ldb,
                        static_cast<const T*>(be.get()), Carray, ldc, batchCount);
      }))
    return CUBLAS_STATUS_SUCCESS;
  const auto a = fetch<const T*>(Aarray, static_cast<size_t>(batchCount));
  const auto b = fetch<const T*>(Barray, static_cast<size_t>(batchCount));
  const auto c = fetch<T*>(Carray, static_cast<size_t>(batchCount));
  for (int i = 0; i < batchCount; ++i) {
    cublasStatus_t st = gemm_one<T>(h, ta, tb, m, n, k, alpha, a[i], lda, b[i], ldb, beta, c[i], ldc);
    if (st != CUBLAS_STATUS_SUCCESS) return st;
  }
  return CUBLAS_STATUS_SUCCESS;
}

}  // namespace

VGPU_EXPORT cublasStatus_t cublasSgemmStridedBatched(
    cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k,
    const float* alpha, const float* A, int lda, long long strideA, const float* B, int ldb,
    long long strideB, const float* beta, float* C, int ldc, long long strideC, int batchCount) {
  return gemm_strided<float>(h, ta, tb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC,
                             batchCount);
}
VGPU_EXPORT cublasStatus_t cublasDgemmStridedBatched(
    cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k,
    const double* alpha, const double* A, int lda, long long strideA, const double* B, int ldb,
    long long strideB, const double* beta, double* C, int ldc, long long strideC, int batchCount) {
  return gemm_strided<double>(h, ta, tb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC,
                              batchCount);
}
VGPU_EXPORT cublasStatus_t cublasSgemmBatched(cublasHandle_t h, cublasOperation_t ta,
                                              cublasOperation_t tb, int m, int n, int k,
                                              const float* alpha, const float* const Aarray[],
                                              int lda, const float* const Barray[], int ldb,
                                              const float* beta, float* const Carray[], int ldc,
                                              int batchCount) {
  return gemm_batched<float>(h, ta, tb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
}
VGPU_EXPORT cublasStatus_t cublasDgemmBatched(cublasHandle_t h, cublasOperation_t ta,
                                              cublasOperation_t tb, int m, int n, int k,
                                              const double* alpha, const double* const Aarray[],
                                              int lda, const double* const Barray[], int ldb,
                                              const double* beta, double* const Carray[], int ldc,
                                              int batchCount) {
  return gemm_batched<double>(h, ta, tb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount);
}

/* ---- batched LU: getrfBatched and getrsBatched ----
   What torch.linalg.inv and batched solves call. Each matrix is factored in
   double on the host (LAPACK's partial pivoting, 1-based pivots) and rounded
   to the caller's type. getrsBatched's info is a host pointer, as cuBLAS
   documents it; getrfBatched's is device memory, one per matrix. */

namespace {

// LU with partial pivoting of a column-major n x n matrix; 0 or the 1-based
// index of the first zero pivot.
int lu_factor(std::vector<double>& a, int n, int lda, std::vector<int>* ipiv) {
  auto A = [&](int r, int c) -> double& { return a[(size_t)c * lda + r]; };
  int info = 0;
  if (ipiv) ipiv->assign(n, 0);
  for (int j = 0; j < n; ++j) {
    int p = j;
    if (ipiv) {
      for (int i = j + 1; i < n; ++i)
        if (std::fabs(A(i, j)) > std::fabs(A(p, j))) p = i;
      (*ipiv)[j] = p + 1;
      if (p != j)
        for (int c = 0; c < n; ++c) std::swap(A(j, c), A(p, c));
    }
    if (A(j, j) == 0.0) { if (!info) info = j + 1; continue; }
    for (int i = j + 1; i < n; ++i) {
      A(i, j) /= A(j, j);
      for (int c = j + 1; c < n; ++c) A(i, c) -= A(i, j) * A(j, c);
    }
  }
  return info;
}

// Solves op(A) X = B with A's LU factors and pivots, X over B.
void lu_solve(const std::vector<double>& a, int n, int lda, const int* ipiv, bool trans, std::vector<double>& b,
              int nrhs, int ldb) {
  auto A = [&](int r, int c) { return a[(size_t)c * lda + r]; };
  auto B = [&](int r, int c) -> double& { return b[(size_t)c * ldb + r]; };
  for (int k = 0; k < nrhs; ++k) {
    if (!trans) {
      if (ipiv) for (int i = 0; i < n; ++i) if (ipiv[i] - 1 != i) std::swap(B(i, k), B(ipiv[i] - 1, k));
      for (int i = 0; i < n; ++i) for (int j = 0; j < i; ++j) B(i, k) -= A(i, j) * B(j, k);      // L, unit
      for (int i = n - 1; i >= 0; --i) {                                                          // U
        for (int j = i + 1; j < n; ++j) B(i, k) -= A(i, j) * B(j, k);
        B(i, k) /= A(i, i);
      }
    } else {
      for (int i = 0; i < n; ++i) {                                                               // U^T
        for (int j = 0; j < i; ++j) B(i, k) -= A(j, i) * B(j, k);
        B(i, k) /= A(i, i);
      }
      for (int i = n - 1; i >= 0; --i) for (int j = i + 1; j < n; ++j) B(i, k) -= A(j, i) * B(j, k);  // L^T
      if (ipiv) for (int i = n - 1; i >= 0; --i) if (ipiv[i] - 1 != i) std::swap(B(i, k), B(ipiv[i] - 1, k));
    }
  }
}

template <class T>
cublasStatus_t getrf_batched(cublasHandle_t h, int n, T* const Aarray[], int lda, int* pivots, int* infos, int batch) {
  if (!valid(h)) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (n < 0 || lda < std::max(1, n) || batch < 0) return CUBLAS_STATUS_INVALID_VALUE;
  if (deferred_to_graph(h, [=] { getrf_batched<T>(h, n, Aarray, lda, pivots, infos, batch); }))
    return CUBLAS_STATUS_SUCCESS;
  const auto ptrs = fetch<T*>(Aarray, (size_t)batch);
  std::vector<int> all_info(batch, 0);
  for (int b = 0; b < batch; ++b) {
    const auto hv = fetch<T>(ptrs[b], extent(lda, n, n));
    std::vector<double> a(hv.begin(), hv.end());
    std::vector<int> ipiv;
    all_info[b] = lu_factor(a, n, lda, pivots ? &ipiv : nullptr);
    store(ptrs[b], std::vector<T>(a.begin(), a.end()));
    if (pivots && n) cudaMemcpy(pivots + (size_t)b * n, ipiv.data(), n * sizeof(int), kH2D);
  }
  if (infos) store(infos, all_info);
  return CUBLAS_STATUS_SUCCESS;
}

template <class T>
cublasStatus_t getrs_batched(cublasHandle_t h, cublasOperation_t trans, int n, int nrhs, const T* const Aarray[],
                             int lda, const int* pivots, T* const Barray[], int ldb, int* info, int batch) {
  if (!valid(h)) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (info) *info = 0;
  if (n < 0 || nrhs < 0 || lda < std::max(1, n) || ldb < std::max(1, n) || batch < 0) {
    if (info) *info = -1;
    return CUBLAS_STATUS_INVALID_VALUE;
  }
  if (trans == CUBLAS_OP_C) return CUBLAS_STATUS_NOT_SUPPORTED;  // real types: C is T
  if (deferred_to_graph(h, [=] { int ignored; getrs_batched<T>(h, trans, n, nrhs, Aarray, lda, pivots, Barray, ldb, &ignored, batch); }))
    return CUBLAS_STATUS_SUCCESS;
  const auto pa = fetch<const T*>(Aarray, (size_t)batch);
  const auto pb = fetch<T*>(Barray, (size_t)batch);
  for (int b = 0; b < batch; ++b) {
    const auto av = fetch<T>(pa[b], extent(lda, n, n));
    const auto bv = fetch<T>(pb[b], extent(ldb, nrhs, n));
    std::vector<double> a(av.begin(), av.end()), x(bv.begin(), bv.end());
    std::vector<int> ipiv;
    if (pivots) ipiv = fetch<int>(pivots + (size_t)b * n, (size_t)n);
    lu_solve(a, n, lda, pivots ? ipiv.data() : nullptr, trans != CUBLAS_OP_N, x, nrhs, ldb);
    store(pb[b], std::vector<T>(x.begin(), x.end()));
  }
  return CUBLAS_STATUS_SUCCESS;
}

}  // namespace

VGPU_EXPORT cublasStatus_t cublasSgetrfBatched(cublasHandle_t h, int n, float* const A[], int lda, int* P, int* info,
                                               int batch) {
  return getrf_batched<float>(h, n, A, lda, P, info, batch);
}
VGPU_EXPORT cublasStatus_t cublasDgetrfBatched(cublasHandle_t h, int n, double* const A[], int lda, int* P, int* info,
                                               int batch) {
  return getrf_batched<double>(h, n, A, lda, P, info, batch);
}
VGPU_EXPORT cublasStatus_t cublasSgetrsBatched(cublasHandle_t h, cublasOperation_t t, int n, int nrhs,
                                               const float* const A[], int lda, const int* P, float* const B[],
                                               int ldb, int* info, int batch) {
  return getrs_batched<float>(h, t, n, nrhs, A, lda, P, B, ldb, info, batch);
}
VGPU_EXPORT cublasStatus_t cublasDgetrsBatched(cublasHandle_t h, cublasOperation_t t, int n, int nrhs,
                                               const double* const A[], int lda, const int* P, double* const B[],
                                               int ldb, int* info, int batch) {
  return getrs_batched<double>(h, t, n, nrhs, A, lda, P, B, ldb, info, batch);
}

/* ---- triangular solves: trsm and trsmBatched ----
   op(A) X = alpha B (side left) or X op(A) = alpha B (side right), X over B.
   What torch.linalg.solve_triangular and lstsq call. Only A's `uplo` triangle
   is read, and with a unit diagonal not even its diagonal. */

namespace {

// Solves M x = x in place, M = A or A^T (`trans`) of the k x k triangle `lower`.
void tri_solve(const std::vector<double>& a, int k, int lda, bool lower, bool trans, bool unit, double* x,
               size_t step) {
  auto M = [&](int i, int j) { return trans ? a[(size_t)i * lda + j] : a[(size_t)j * lda + i]; };
  auto row = [&](int i, int from, int to) {
    double s = x[(size_t)i * step];
    for (int j = from; j < to; ++j) s -= M(i, j) * x[(size_t)j * step];
    x[(size_t)i * step] = unit ? s : s / M(i, i);
  };
  if (lower != trans) for (int i = 0; i < k; ++i) row(i, 0, i);   // M is lower: forward
  else for (int i = k - 1; i >= 0; --i) row(i, i + 1, k);          // M is upper: back
}

template <class T>
cublasStatus_t trsm_one(cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag,
                        int m, int n, T alpha, const T* A, int lda, T* B, int ldb) {
  const int k = side == CUBLAS_SIDE_LEFT ? m : n;
  const auto av = fetch<T>(A, extent(lda, k, k));
  const auto bv = fetch<T>(B, extent(ldb, n, m));
  const std::vector<double> a(av.begin(), av.end());
  std::vector<double> b(bv.begin(), bv.end());
  const bool lower = uplo == CUBLAS_FILL_MODE_LOWER, t = trans != CUBLAS_OP_N, unit = diag == CUBLAS_DIAG_UNIT;
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < m; ++i) b[(size_t)j * ldb + i] *= (double)alpha;
  if (side == CUBLAS_SIDE_LEFT)  // each column of B: op(A) x = b
    for (int j = 0; j < n; ++j) tri_solve(a, k, lda, lower, t, unit, &b[(size_t)j * ldb], 1);
  else  // each row of B: x op(A) = b, which is op(A)^T x^T = b^T
    for (int i = 0; i < m; ++i) tri_solve(a, k, lda, lower, !t, unit, &b[(size_t)i], (size_t)ldb);
  store(B, std::vector<T>(b.begin(), b.end()));
  return CUBLAS_STATUS_SUCCESS;
}

template <class T>
cublasStatus_t trsm_batched(cublasHandle_t h, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans,
                            cublasDiagType_t diag, int m, int n, const T* alpha, const T* const Aarray[], int lda,
                            T* const Barray[], int ldb, int batch) {
  if (!valid(h)) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (!tri3_args_ok(side, uplo, trans, diag, m, n, lda, ldb) || batch < 0) return CUBLAS_STATUS_INVALID_VALUE;
  if (trans == CUBLAS_OP_C) trans = CUBLAS_OP_T;  // real types
  const T al = scalar(h, alpha);
  if (deferred_to_graph(h, [=] {
        const auto pa = fetch<const T*>(Aarray, (size_t)batch);
        const auto pb = fetch<T*>(Barray, (size_t)batch);
        for (int b = 0; b < batch; ++b) trsm_one<T>(side, uplo, trans, diag, m, n, al, pa[b], lda, pb[b], ldb);
      }))
    return CUBLAS_STATUS_SUCCESS;
  const auto pa = fetch<const T*>(Aarray, (size_t)batch);
  const auto pb = fetch<T*>(Barray, (size_t)batch);
  for (int b = 0; b < batch; ++b) trsm_one<T>(side, uplo, trans, diag, m, n, al, pa[b], lda, pb[b], ldb);
  return CUBLAS_STATUS_SUCCESS;
}

template <class T>
cublasStatus_t trsm(cublasHandle_t h, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans,
                    cublasDiagType_t diag, int m, int n, const T* alpha, const T* A, int lda, T* B, int ldb) {
  if (!valid(h)) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (!tri3_args_ok(side, uplo, trans, diag, m, n, lda, ldb)) return CUBLAS_STATUS_INVALID_VALUE;
  if (trans == CUBLAS_OP_C) trans = CUBLAS_OP_T;
  const T al = scalar(h, alpha);
  if (deferred_to_graph(h, [=] { trsm_one<T>(side, uplo, trans, diag, m, n, al, A, lda, B, ldb); }))
    return CUBLAS_STATUS_SUCCESS;
  return trsm_one<T>(side, uplo, trans, diag, m, n, al, A, lda, B, ldb);
}

}  // namespace

VGPU_EXPORT cublasStatus_t cublasStrsm_v2(cublasHandle_t h, cublasSideMode_t side, cublasFillMode_t uplo,
                                          cublasOperation_t trans, cublasDiagType_t diag, int m, int n,
                                          const float* alpha, const float* A, int lda, float* B, int ldb) {
  return trsm<float>(h, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
}
VGPU_EXPORT cublasStatus_t cublasDtrsm_v2(cublasHandle_t h, cublasSideMode_t side, cublasFillMode_t uplo,
                                          cublasOperation_t trans, cublasDiagType_t diag, int m, int n,
                                          const double* alpha, const double* A, int lda, double* B, int ldb) {
  return trsm<double>(h, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
}
VGPU_EXPORT cublasStatus_t cublasStrsmBatched(cublasHandle_t h, cublasSideMode_t side, cublasFillMode_t uplo,
                                              cublasOperation_t trans, cublasDiagType_t diag, int m, int n,
                                              const float* alpha, const float* const A[], int lda,
                                              float* const B[], int ldb, int batch) {
  return trsm_batched<float>(h, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batch);
}
VGPU_EXPORT cublasStatus_t cublasDtrsmBatched(cublasHandle_t h, cublasSideMode_t side, cublasFillMode_t uplo,
                                              cublasOperation_t trans, cublasDiagType_t diag, int m, int n,
                                              const double* alpha, const double* const A[], int lda,
                                              double* const B[], int ldb, int batch) {
  return trsm_batched<double>(h, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb, batch);
}

// A workspace is a scratch buffer the library would use for its own tiling.
// Nothing here needs one, so accepting it is honest: the caller's buffer simply
// goes unused, and refusing would stop a program that is doing nothing wrong.
VGPU_EXPORT cublasStatus_t cublasSetWorkspace_v2(cublasHandle_t h, void*, size_t) {
  return valid(h) ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_NOT_INITIALIZED;
}

/* ---- level 2 and level 1 ---- */

namespace {

// Where element i of an n-element vector with increment inc sits, by BLAS's
// rule: a negative increment walks the vector from its far end, so element 0
// is at (n-1)*|inc|. Indexing i*inc directly would read before the buffer.
inline size_t elem(int i, int n, int inc) {
  return inc > 0 ? static_cast<size_t>(i) * inc : static_cast<size_t>(n - 1 - i) * -inc;
}
inline size_t span(int n, int inc) { return static_cast<size_t>(std::abs(inc)) * (n - 1) + 1; }

// y = alpha op(A) x + beta y. Increments follow BLAS's rule (a negative one
// walks its vector from the far end), and a leading dimension shorter than a
// column is refused, as it is by cuBLAS. beta 0 overwrites y without reading
// it, so NaN in y stays out of the answer.
template <class T>
cublasStatus_t gemv(cublasHandle_t h, cublasOperation_t trans, int m, int n, const T* alpha,
                    const T* A, int lda, const T* x, int incx, const T* beta, T* y, int incy) {
  if (!valid(h)) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (m < 0 || n < 0 || lda < std::max(1, m) || incx == 0 || incy == 0) return CUBLAS_STATUS_INVALID_VALUE;
  if (trans == CUBLAS_OP_C) trans = CUBLAS_OP_T;  // real types
  if (deferred_to_graph(h, [=, al = hold(h, alpha, sizeof(T)), be = hold(h, beta, sizeof(T))] {
        gemv<T>(h, trans, m, n, static_cast<const T*>(al.get()), A, lda, x, incx,
                static_cast<const T*>(be.get()), y, incy);
      }))
    return CUBLAS_STATUS_SUCCESS;
  const T a = scalar(h, alpha), b = scalar(h, beta);
  const int xlen = trans == CUBLAS_OP_N ? n : m;
  const int ylen = trans == CUBLAS_OP_N ? m : n;
  if (!m || !n) return CUBLAS_STATUS_SUCCESS;
  auto hA = fetch<T>(A, extent(lda, n, m));
  auto hx = fetch<T>(x, span(xlen, incx));
  auto hy = fetch<T>(y, span(ylen, incy));
  for (int i = 0; i < ylen; ++i) {
    T acc = 0;
    for (int j = 0; j < xlen; ++j)
      acc += hA[trans == CUBLAS_OP_N ? idx(i, j, lda) : idx(j, i, lda)] * hx[elem(j, xlen, incx)];
    T& yi = hy[elem(i, ylen, incy)];
    yi = b == T(0) ? a * acc : a * acc + b * yi;
  }
  store(y, hy);
  return CUBLAS_STATUS_SUCCESS;
}

// The reductions hand their answer back through a host or device pointer,
// depending on the handle's pointer mode.
template <class R>
void put_result(cublasHandle_t h, R* result, R value) {
  if (reinterpret_cast<Handle*>(h)->pointer_mode == CUBLAS_POINTER_MODE_DEVICE)
    cudaMemcpy(result, &value, sizeof value, kH2D);
  else
    *result = value;
}

template <class T>
cublasStatus_t axpy(cublasHandle_t h, int n, const T* alpha, const T* x, int incx, T* y,
                    int incy) {
  if (!valid(h)) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (n <= 0) return CUBLAS_STATUS_SUCCESS;
  if (deferred_to_graph(h, [=, al = hold(h, alpha, sizeof(T))] {
        axpy<T>(h, n, static_cast<const T*>(al.get()), x, incx, y, incy);
      }))
    return CUBLAS_STATUS_SUCCESS;
  T a = scalar(h, alpha);
  auto hx = fetch<T>(x, span(n, incx));
  auto hy = fetch<T>(y, span(n, incy));
  for (int i = 0; i < n; ++i) hy[elem(i, n, incy)] += a * hx[elem(i, n, incx)];
  store(y, hy);
  return CUBLAS_STATUS_SUCCESS;
}

// Reference BLAS scales nothing when incx is not positive.
template <class T>
cublasStatus_t scal(cublasHandle_t h, int n, const T* alpha, T* x, int incx) {
  if (!valid(h)) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (n <= 0 || incx <= 0) return CUBLAS_STATUS_SUCCESS;
  if (deferred_to_graph(h, [=, al = hold(h, alpha, sizeof(T))] {
        scal<T>(h, n, static_cast<const T*>(al.get()), x, incx);
      }))
    return CUBLAS_STATUS_SUCCESS;
  T a = scalar(h, alpha);
  auto hx = fetch<T>(x, span(n, incx));
  for (int i = 0; i < n; ++i) hx[static_cast<size_t>(i) * incx] *= a;
  store(x, hx);
  return CUBLAS_STATUS_SUCCESS;
}

// Accumulated in long double, then rounded once: a GPU sums in a different
// order, so the last bits can differ, but never by more than that ordering.
template <class T>
cublasStatus_t dot(cublasHandle_t h, int n, const T* x, int incx, const T* y, int incy,
                   T* result) {
  if (!valid(h)) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (!result) return CUBLAS_STATUS_INVALID_VALUE;
  if (host_result_under_capture(h)) return CUBLAS_STATUS_NOT_SUPPORTED;
  if (deferred_to_graph(h, [=] { dot<T>(h, n, x, incx, y, incy, result); }))
    return CUBLAS_STATUS_SUCCESS;
  long double acc = 0.0L;
  if (n > 0) {
    auto hx = fetch<T>(x, span(n, incx));
    auto hy = fetch<T>(y, span(n, incy));
    for (int i = 0; i < n; ++i)
      acc += static_cast<long double>(hx[elem(i, n, incx)]) * hy[elem(i, n, incy)];
  }
  put_result(h, result, static_cast<T>(acc));
  return CUBLAS_STATUS_SUCCESS;
}

template <class T>
cublasStatus_t nrm2(cublasHandle_t h, int n, const T* x, int incx, T* result) {
  if (!valid(h)) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (!result) return CUBLAS_STATUS_INVALID_VALUE;
  if (host_result_under_capture(h)) return CUBLAS_STATUS_NOT_SUPPORTED;
  if (deferred_to_graph(h, [=] { nrm2<T>(h, n, x, incx, result); }))
    return CUBLAS_STATUS_SUCCESS;
  long double acc = 0.0L;
  if (n > 0 && incx > 0) {
    auto hx = fetch<T>(x, span(n, incx));
    for (int i = 0; i < n; ++i) {
      long double v = hx[static_cast<size_t>(i) * incx];
      acc += v * v;
    }
  }
  put_result(h, result, static_cast<T>(std::sqrt(acc)));
  return CUBLAS_STATUS_SUCCESS;
}

// The sum of magnitudes; 0 for an empty vector or a non-positive stride, as
// reference BLAS returns.
template <class T>
cublasStatus_t asum(cublasHandle_t h, int n, const T* x, int incx, T* result) {
  if (!valid(h)) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (!result) return CUBLAS_STATUS_INVALID_VALUE;
  if (host_result_under_capture(h)) return CUBLAS_STATUS_NOT_SUPPORTED;
  if (deferred_to_graph(h, [=] { asum<T>(h, n, x, incx, result); }))
    return CUBLAS_STATUS_SUCCESS;
  long double acc = 0.0L;
  if (n > 0 && incx > 0) {
    auto hx = fetch<T>(x, span(n, incx));
    for (int i = 0; i < n; ++i) acc += std::fabs(static_cast<long double>(hx[static_cast<size_t>(i) * incx]));
  }
  put_result(h, result, static_cast<T>(acc));
  return CUBLAS_STATUS_SUCCESS;
}

// The 1-based index of the first element of largest magnitude; 0 when there is
// nothing to search, as reference BLAS returns.
template <class T, class R = int>
cublasStatus_t iamax(cublasHandle_t h, int n, const T* x, int incx, R* result) {
  if (!valid(h)) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (!result) return CUBLAS_STATUS_INVALID_VALUE;
  if (host_result_under_capture(h)) return CUBLAS_STATUS_NOT_SUPPORTED;
  if (deferred_to_graph(h, [=] { iamax<T, R>(h, n, x, incx, result); }))
    return CUBLAS_STATUS_SUCCESS;
  R best = 0;
  if (n > 0 && incx > 0) {
    auto hx = fetch<T>(x, span(n, incx));
    T top = std::abs(hx[0]);
    best = 1;
    for (int i = 1; i < n; ++i) {
      T v = std::abs(hx[static_cast<size_t>(i) * incx]);
      if (v > top) { top = v; best = i + 1; }
    }
  }
  put_result(h, result, best);
  return CUBLAS_STATUS_SUCCESS;
}

// C = A diag(x) (CUBLAS_SIDE_RIGHT) or diag(x) A (CUBLAS_SIDE_LEFT): each
// column, or each row, of A scaled by one element of x. x is walked by BLAS's
// rule for its increment, so a zero increment scales by x[0] throughout. C may
// be A itself, which is why A is read whole before C is written.
template <class T>
cublasStatus_t dgmm(cublasHandle_t h, cublasSideMode_t mode, int m, int n, const T* A, int lda,
                    const T* x, int incx, T* C, int ldc) {
  if (!valid(h)) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (mode != CUBLAS_SIDE_LEFT && mode != CUBLAS_SIDE_RIGHT) return CUBLAS_STATUS_INVALID_VALUE;
  if (m < 0 || n < 0 || lda < std::max(1, m) || ldc < std::max(1, m)) return CUBLAS_STATUS_INVALID_VALUE;
  if (!m || !n) return CUBLAS_STATUS_SUCCESS;
  if (deferred_to_graph(h, [=] { dgmm<T>(h, mode, m, n, A, lda, x, incx, C, ldc); }))
    return CUBLAS_STATUS_SUCCESS;
  const int len = mode == CUBLAS_SIDE_LEFT ? m : n;
  auto hA = fetch<T>(A, extent(lda, n, m));
  auto hx = fetch<T>(x, incx == 0 ? 1 : span(len, incx));
  auto hC = fetch<T>(C, extent(ldc, n, m));
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < m; ++i)
      hC[idx(i, j, ldc)] = hA[idx(i, j, lda)] * hx[elem(mode == CUBLAS_SIDE_LEFT ? i : j, len, incx)];
  store(C, hC);
  return CUBLAS_STATUS_SUCCESS;
}

}  // namespace

VGPU_EXPORT cublasStatus_t cublasSdgmm(cublasHandle_t h, cublasSideMode_t mode, int m, int n,
                                       const float* A, int lda, const float* x, int incx, float* C,
                                       int ldc) {
  return dgmm(h, mode, m, n, A, lda, x, incx, C, ldc);
}
VGPU_EXPORT cublasStatus_t cublasDdgmm(cublasHandle_t h, cublasSideMode_t mode, int m, int n,
                                       const double* A, int lda, const double* x, int incx, double* C,
                                       int ldc) {
  return dgmm(h, mode, m, n, A, lda, x, incx, C, ldc);
}
VGPU_EXPORT cublasStatus_t cublasSaxpy_v2(cublasHandle_t h, int n, const float* alpha,
                                          const float* x, int incx, float* y, int incy) {
  return axpy(h, n, alpha, x, incx, y, incy);
}
VGPU_EXPORT cublasStatus_t cublasDaxpy_v2(cublasHandle_t h, int n, const double* alpha,
                                          const double* x, int incx, double* y, int incy) {
  return axpy(h, n, alpha, x, incx, y, incy);
}
VGPU_EXPORT cublasStatus_t cublasSscal_v2(cublasHandle_t h, int n, const float* alpha, float* x,
                                          int incx) {
  return scal(h, n, alpha, x, incx);
}
VGPU_EXPORT cublasStatus_t cublasDscal_v2(cublasHandle_t h, int n, const double* alpha, double* x,
                                          int incx) {
  return scal(h, n, alpha, x, incx);
}
VGPU_EXPORT cublasStatus_t cublasSgemv_v2(cublasHandle_t h, cublasOperation_t trans, int m, int n,
                                          const float* alpha, const float* A, int lda,
                                          const float* x, int incx, const float* beta, float* y,
                                          int incy) {
  return gemv(h, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}
VGPU_EXPORT cublasStatus_t cublasDgemv_v2(cublasHandle_t h, cublasOperation_t trans, int m, int n,
                                          const double* alpha, const double* A, int lda,
                                          const double* x, int incx, const double* beta, double* y,
                                          int incy) {
  return gemv(h, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}
VGPU_EXPORT cublasStatus_t cublasSdot_v2(cublasHandle_t h, int n, const float* x, int incx,
                                         const float* y, int incy, float* result) {
  return dot(h, n, x, incx, y, incy, result);
}
VGPU_EXPORT cublasStatus_t cublasDdot_v2(cublasHandle_t h, int n, const double* x, int incx,
                                         const double* y, int incy, double* result) {
  return dot(h, n, x, incx, y, incy, result);
}
VGPU_EXPORT cublasStatus_t cublasSnrm2_v2(cublasHandle_t h, int n, const float* x, int incx,
                                          float* result) {
  return nrm2(h, n, x, incx, result);
}
VGPU_EXPORT cublasStatus_t cublasDnrm2_v2(cublasHandle_t h, int n, const double* x, int incx,
                                          double* result) {
  return nrm2(h, n, x, incx, result);
}
VGPU_EXPORT cublasStatus_t cublasSasum_v2(cublasHandle_t h, int n, const float* x, int incx,
                                          float* result) {
  return asum(h, n, x, incx, result);
}
VGPU_EXPORT cublasStatus_t cublasDasum_v2(cublasHandle_t h, int n, const double* x, int incx,
                                          double* result) {
  return asum(h, n, x, incx, result);
}
VGPU_EXPORT cublasStatus_t cublasIsamax_v2(cublasHandle_t h, int n, const float* x, int incx,
                                           int* result) {
  return iamax(h, n, x, incx, result);
}
VGPU_EXPORT cublasStatus_t cublasIdamax_v2(cublasHandle_t h, int n, const double* x, int incx,
                                           int* result) {
  return iamax(h, n, x, incx, result);
}
// The 64-bit-index forms CUDA 12 added (cublasDnrm2_64 and so on; cuPDLPx
// calls that one). Sizes and strides are int64_t, and i?amax answers in an
// int64_t. Host-computed vectors that long would not fit in memory anyway, so
// a size or stride beyond int is refused rather than truncated.
namespace {
bool fits_int(int64_t v) { return v >= INT32_MIN && v <= INT32_MAX; }
}  // namespace
VGPU_EXPORT cublasStatus_t cublasSaxpy_v2_64(cublasHandle_t h, int64_t n, const float* alpha,
                                             const float* x, int64_t incx, float* y, int64_t incy) {
  if (!fits_int(n) || !fits_int(incx) || !fits_int(incy)) return CUBLAS_STATUS_NOT_SUPPORTED;
  return axpy(h, (int)n, alpha, x, (int)incx, y, (int)incy);
}
VGPU_EXPORT cublasStatus_t cublasDaxpy_v2_64(cublasHandle_t h, int64_t n, const double* alpha,
                                             const double* x, int64_t incx, double* y, int64_t incy) {
  if (!fits_int(n) || !fits_int(incx) || !fits_int(incy)) return CUBLAS_STATUS_NOT_SUPPORTED;
  return axpy(h, (int)n, alpha, x, (int)incx, y, (int)incy);
}
VGPU_EXPORT cublasStatus_t cublasSscal_v2_64(cublasHandle_t h, int64_t n, const float* alpha, float* x,
                                             int64_t incx) {
  if (!fits_int(n) || !fits_int(incx)) return CUBLAS_STATUS_NOT_SUPPORTED;
  return scal(h, (int)n, alpha, x, (int)incx);
}
VGPU_EXPORT cublasStatus_t cublasDscal_v2_64(cublasHandle_t h, int64_t n, const double* alpha, double* x,
                                             int64_t incx) {
  if (!fits_int(n) || !fits_int(incx)) return CUBLAS_STATUS_NOT_SUPPORTED;
  return scal(h, (int)n, alpha, x, (int)incx);
}
VGPU_EXPORT cublasStatus_t cublasSdot_v2_64(cublasHandle_t h, int64_t n, const float* x, int64_t incx,
                                            const float* y, int64_t incy, float* result) {
  if (!fits_int(n) || !fits_int(incx) || !fits_int(incy)) return CUBLAS_STATUS_NOT_SUPPORTED;
  return dot(h, (int)n, x, (int)incx, y, (int)incy, result);
}
VGPU_EXPORT cublasStatus_t cublasDdot_v2_64(cublasHandle_t h, int64_t n, const double* x, int64_t incx,
                                            const double* y, int64_t incy, double* result) {
  if (!fits_int(n) || !fits_int(incx) || !fits_int(incy)) return CUBLAS_STATUS_NOT_SUPPORTED;
  return dot(h, (int)n, x, (int)incx, y, (int)incy, result);
}
VGPU_EXPORT cublasStatus_t cublasSnrm2_v2_64(cublasHandle_t h, int64_t n, const float* x, int64_t incx,
                                             float* result) {
  if (!fits_int(n) || !fits_int(incx)) return CUBLAS_STATUS_NOT_SUPPORTED;
  return nrm2(h, (int)n, x, (int)incx, result);
}
VGPU_EXPORT cublasStatus_t cublasDnrm2_v2_64(cublasHandle_t h, int64_t n, const double* x, int64_t incx,
                                             double* result) {
  if (!fits_int(n) || !fits_int(incx)) return CUBLAS_STATUS_NOT_SUPPORTED;
  return nrm2(h, (int)n, x, (int)incx, result);
}
VGPU_EXPORT cublasStatus_t cublasSasum_v2_64(cublasHandle_t h, int64_t n, const float* x, int64_t incx,
                                             float* result) {
  if (!fits_int(n) || !fits_int(incx)) return CUBLAS_STATUS_NOT_SUPPORTED;
  return asum(h, (int)n, x, (int)incx, result);
}
VGPU_EXPORT cublasStatus_t cublasDasum_v2_64(cublasHandle_t h, int64_t n, const double* x, int64_t incx,
                                             double* result) {
  if (!fits_int(n) || !fits_int(incx)) return CUBLAS_STATUS_NOT_SUPPORTED;
  return asum(h, (int)n, x, (int)incx, result);
}
VGPU_EXPORT cublasStatus_t cublasIsamax_v2_64(cublasHandle_t h, int64_t n, const float* x, int64_t incx,
                                              int64_t* result) {
  if (!fits_int(n) || !fits_int(incx)) return CUBLAS_STATUS_NOT_SUPPORTED;
  return iamax(h, (int)n, x, (int)incx, result);
}
VGPU_EXPORT cublasStatus_t cublasIdamax_v2_64(cublasHandle_t h, int64_t n, const double* x, int64_t incx,
                                              int64_t* result) {
  if (!fits_int(n) || !fits_int(incx)) return CUBLAS_STATUS_NOT_SUPPORTED;
  return iamax(h, (int)n, x, (int)incx, result);
}

/* ---- GemmEx ----
   The type combinations an RTX 3060's cuBLAS takes, measured over every
   combination of A, B and C type and compute type (anything else is
   NOT_SUPPORTED; A and B are always one type):

     A = B     C             compute
     R_16F     R_16F         32F family, 16F, 16F_PEDANTIC
     R_16F     R_32F         32F family
     R_16BF    R_16BF, R_32F 32F family
     R_32F     R_32F         32F family
     R_64F     R_64F         64F, 64F_PEDANTIC
     C_32F     C_32F         32F family
     C_64F     C_64F         64F, 64F_PEDANTIC
     R_8I      R_32F         32F family
     R_8I      R_32I         32I, 32I_PEDANTIC
     C_8I      C_32F         32F family

   (the 32F family: 32F, 32F_PEDANTIC and the FAST_16F, FAST_16BF and
   FAST_TF32 forms, all computed here in single precision). The int8 path with
   an int32 result takes only what cuBLAS documents for it: lda and ldb
   multiples of 4, A and B 4-byte aligned and op(B) = N; anything else is
   NOT_SUPPORTED, as is a real matrix not aligned to its element. A bad handle is reported before a bad type, and a bad type
   before a bad argument, in that order on the card too. */
namespace {

bool compute_32f(cublasComputeType_t ct) {
  return ct == CUBLAS_COMPUTE_32F || ct == CUBLAS_COMPUTE_32F_PEDANTIC || ct == CUBLAS_COMPUTE_32F_FAST_16F ||
         ct == CUBLAS_COMPUTE_32F_FAST_16BF || ct == CUBLAS_COMPUTE_32F_FAST_TF32;
}
bool compute_64f(cublasComputeType_t ct) { return ct == CUBLAS_COMPUTE_64F || ct == CUBLAS_COMPUTE_64F_PEDANTIC; }
bool compute_16f(cublasComputeType_t ct) { return ct == CUBLAS_COMPUTE_16F || ct == CUBLAS_COMPUTE_16F_PEDANTIC; }
bool compute_32i(cublasComputeType_t ct) { return ct == CUBLAS_COMPUTE_32I || ct == CUBLAS_COMPUTE_32I_PEDANTIC; }

bool gemm_ex_supported(cudaDataType a, cudaDataType b, cudaDataType c, cublasComputeType_t ct) {
  if (a != b) return false;
  switch (a) {
    case CUDA_R_16F: return (c == CUDA_R_16F && (compute_32f(ct) || compute_16f(ct))) || (c == CUDA_R_32F && compute_32f(ct));
    case CUDA_R_16BF: return (c == CUDA_R_16BF || c == CUDA_R_32F) && compute_32f(ct);
    case CUDA_R_32F: return c == CUDA_R_32F && compute_32f(ct);
    case CUDA_R_64F: return c == CUDA_R_64F && compute_64f(ct);
    case CUDA_C_32F: return c == CUDA_C_32F && compute_32f(ct);
    case CUDA_C_64F: return c == CUDA_C_64F && compute_64f(ct);
    case CUDA_R_8I: return (c == CUDA_R_32F && compute_32f(ct)) || (c == CUDA_R_32I && compute_32i(ct));
    case CUDA_C_8I: return c == CUDA_C_32F && compute_32f(ct);
    default: return false;
  }
}

// alpha and beta are typed by the compute type, and are complex when C is.
size_t gemm_scalar_bytes(cublasComputeType_t ct, cudaDataType c) {
  return compute_scalar_bytes(ct) * (c == CUDA_C_32F || c == CUDA_C_64F ? 2 : 1);
}

// The complex forms, written with the rest of the complex routines (after
// cublas_complex.inc) whose machinery they use.
cublasStatus_t gemm_ex_complex(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k,
                               const void* alpha, const void* A, cudaDataType Atype, int lda, const void* B, int ldb,
                               const void* beta, void* C, cudaDataType Ctype, int ldc);

}  // namespace

VGPU_EXPORT cublasStatus_t cublasGemmEx(cublasHandle_t h, cublasOperation_t ta,
                                        cublasOperation_t tb, int m, int n, int k,
                                        const void* alpha, const void* A, cudaDataType Atype,
                                        int lda, const void* B, cudaDataType Btype, int ldb,
                                        const void* beta, void* C, cudaDataType Ctype, int ldc,
                                        cublasComputeType_t computeType, cublasGemmAlgo_t algo) {
  if (!valid(h)) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (!gemm_ex_supported(Atype, Btype, Ctype, computeType)) {
    if (trace())
      std::fprintf(stderr,
                   "[vgpu] cublasGemmEx: unsupported type combination (A=%d B=%d C=%d compute=%d)\n",
                   (int)Atype, (int)Btype, (int)Ctype, (int)computeType);
    return CUBLAS_STATUS_NOT_SUPPORTED;
  }
  if (!gemm_args_ok(ta, tb, m, n, k, lda, ldb, ldc)) return CUBLAS_STATUS_INVALID_VALUE;
  if (Ctype != CUDA_C_32F && Ctype != CUDA_C_64F && !gemm_aligned(A, B, C, type_bytes(Atype), type_bytes(Ctype)))
    return CUBLAS_STATUS_NOT_SUPPORTED;
  if (compute_32i(computeType) &&
      (lda % 4 || ldb % 4 || reinterpret_cast<uintptr_t>(A) % 4 || reinterpret_cast<uintptr_t>(B) % 4 ||
       tb != CUBLAS_OP_N))
    return CUBLAS_STATUS_NOT_SUPPORTED;
  if (deferred_to_graph(h, [=, al = hold(h, alpha, gemm_scalar_bytes(computeType, Ctype)),
                            be = hold(h, beta, gemm_scalar_bytes(computeType, Ctype))] {
        cublasGemmEx(h, ta, tb, m, n, k, al.get(), A, Atype, lda, B, Btype, ldb, be.get(), C,
                     Ctype, ldc, computeType, algo);
      }))
    return CUBLAS_STATUS_SUCCESS;
  if (Atype == CUDA_R_32F)
    return do_gemm<float, float>(h, ta, tb, m, n, k, static_cast<const float*>(alpha), A, lda, B,
                                 ldb, static_cast<const float*>(beta), C, ldc);
  if (Atype == CUDA_R_64F)
    return do_gemm<double, double>(h, ta, tb, m, n, k, static_cast<const double*>(alpha), A, lda, B,
                                   ldb, static_cast<const double*>(beta), C, ldc);
  if (Ctype == CUDA_C_32F || Ctype == CUDA_C_64F)
    return gemm_ex_complex(h, ta, tb, m, n, k, alpha, A, Atype, lda, B, ldb, beta, C, Ctype, ldc);
  if (!alpha || !beta || !C) return CUBLAS_STATUS_INVALID_VALUE;
  if (m == 0 || n == 0) return CUBLAS_STATUS_SUCCESS;

  // The quantized path: int8 operands, int32 accumulator and output.
  if (Ctype == CUDA_R_32I) {
    const int32_t a = scalar(h, static_cast<const int32_t*>(alpha));
    const int32_t b = scalar(h, static_cast<const int32_t*>(beta));
    auto hA = fetch<int8_t>(A, ta == CUBLAS_OP_N ? extent(lda, k, m) : extent(lda, m, k));
    auto hB = fetch<int8_t>(B, tb == CUBLAS_OP_N ? extent(ldb, n, k) : extent(ldb, k, n));
    auto hC = fetch<int32_t>(C, extent(ldc, n, m));
    gemm_int8(ta, tb, m, n, k, a, hA, lda, hB, ldb, b, hC, ldc);
    store(C, hC);
    return CUBLAS_STATUS_SUCCESS;
  }

  // Mixed precision: half, bfloat16 or int8 operands accumulated in float,
  // which is what a tensor core does under CUBLAS_COMPUTE_32F. C may be
  // narrower than the accumulator, so it is rounded once on the way out.
  // alpha and beta are float here whatever the operand types, except under
  // CUBLAS_COMPUTE_16F where the API says they are half.
  float a, b;
  if (compute_16f(computeType)) {
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

#include "cublas_complex.inc"

/* ---- the typed level-1 routines (cublas*Ex) ----
   One routine for every element type, over std::complex<double> like the
   complex half of this file: each element is read in its own type and the
   answer rounded to the execution type and then to its own. Which
   combinations of types are accepted is the table an RTX 3060's cuBLAS keeps
   (any other is NOT_SUPPORTED):

     x (and y)        Dot/Dotc result, exec   Nrm2 result   Asum result, exec   Axpy/Scal alpha = exec
     R_16F, R_16BF    same as x, R_32F        same as x     R_32F, R_32F        R_32F
     R_32F            R_32F, R_32F            R_32F         R_32F, R_32F        R_32F
     R_64F            R_64F, R_64F            R_64F         R_64F, R_64F        R_64F
     C_32F            C_32F, C_32F            R_32F         R_32F, C_32F        C_32F
     C_64F            C_64F, C_64F            R_64F         R_64F, C_64F        C_64F

   Nrm2's execution type is the real one (R_32F or R_64F). Iamax, Iamin, Copy
   and Swap take any of the six, x and y alike. Increments follow BLAS: a
   negative one walks its vector from the far end, and the reductions and Scal
   do nothing for one that is not positive. */
namespace {

bool ex_known(cudaDataType t) {
  return t == CUDA_R_16F || t == CUDA_R_16BF || t == CUDA_R_32F || t == CUDA_R_64F || t == CUDA_C_32F ||
         t == CUDA_C_64F;
}
size_t ex_bytes(cudaDataType t) {
  switch (t) {
    case CUDA_R_16F: case CUDA_R_16BF: return 2;
    case CUDA_R_32F: return 4;
    case CUDA_R_64F: case CUDA_C_32F: return 8;
    case CUDA_C_64F: return 16;
    default: return 0;
  }
}
bool ex_complex(cudaDataType t) { return t == CUDA_C_32F || t == CUDA_C_64F; }
// The type arithmetic runs in: R_32F for everything narrower than double.
cudaDataType ex_exec(cudaDataType t) {
  return t == CUDA_R_64F ? CUDA_R_64F : t == CUDA_C_32F || t == CUDA_C_64F ? t : CUDA_R_32F;
}
// The real type of the same width, for norms and sums of magnitudes.
cudaDataType ex_real(cudaDataType t) {
  return t == CUDA_R_64F || t == CUDA_C_64F ? CUDA_R_64F : CUDA_R_32F;
}

cd ex_get(cudaDataType t, const uint8_t* p) {
  switch (t) {
    case CUDA_R_16F: { __half v; std::memcpy(&v, p, 2); return __half2float(v); }
    case CUDA_R_16BF: { __nv_bfloat16 v; std::memcpy(&v, p, 2); return __bfloat162float(v); }
    case CUDA_R_32F: { float v; std::memcpy(&v, p, 4); return v; }
    case CUDA_R_64F: { double v; std::memcpy(&v, p, 8); return v; }
    case CUDA_C_32F: { float v[2]; std::memcpy(v, p, 8); return {v[0], v[1]}; }
    case CUDA_C_64F: { double v[2]; std::memcpy(v, p, 16); return {v[0], v[1]}; }
    default: return 0;
  }
}
// Rounded to the execution type first (single precision for the narrow
// types), then to the element's own.
void ex_put(cudaDataType t, uint8_t* p, cd v, cudaDataType exec) {
  if (exec == CUDA_R_32F || exec == CUDA_C_32F) v = cd((float)v.real(), (float)v.imag());
  switch (t) {
    case CUDA_R_16F: { const __half h = __float2half((float)v.real()); std::memcpy(p, &h, 2); break; }
    case CUDA_R_16BF: { const __nv_bfloat16 h = __float2bfloat16((float)v.real()); std::memcpy(p, &h, 2); break; }
    case CUDA_R_32F: { const float f = (float)v.real(); std::memcpy(p, &f, 4); break; }
    case CUDA_R_64F: { const double d = v.real(); std::memcpy(p, &d, 8); break; }
    case CUDA_C_32F: { const float f[2] = {(float)v.real(), (float)v.imag()}; std::memcpy(p, f, 8); break; }
    case CUDA_C_64F: { const double d[2] = {v.real(), v.imag()}; std::memcpy(p, d, 16); break; }
    default: break;
  }
}

// n elements of a strided vector, in BLAS order.
std::vector<cd> ex_load(const void* x, cudaDataType t, int n, int inc) {
  const size_t e = ex_bytes(t), step = (size_t)std::abs(inc);
  const auto raw = fetch<uint8_t>(x, n > 0 ? ((n - 1) * step + 1) * e : 0);
  std::vector<cd> out(std::max(n, 0));
  for (int i = 0; i < n; ++i) out[i] = ex_get(t, raw.data() + (inc > 0 ? i : n - 1 - i) * step * e);
  return out;
}
void ex_store(void* x, cudaDataType t, int n, int inc, const std::vector<cd>& v, cudaDataType exec) {
  const size_t e = ex_bytes(t), step = (size_t)std::abs(inc);
  auto raw = fetch<uint8_t>(x, n > 0 ? ((n - 1) * step + 1) * e : 0);   // keep what lies between
  for (int i = 0; i < n; ++i) ex_put(t, raw.data() + (inc > 0 ? i : n - 1 - i) * step * e, v[i], exec);
  store(x, raw);
}

cd ex_scalar(cublasHandle_t h, const void* p, cudaDataType t) {
  if (!p) return 0;
  std::vector<uint8_t> b(ex_bytes(t));
  if (reinterpret_cast<Handle*>(h)->pointer_mode == CUBLAS_POINTER_MODE_DEVICE) b = fetch<uint8_t>(p, b.size());
  else std::memcpy(b.data(), p, b.size());
  return ex_get(t, b.data());
}
void ex_result(cublasHandle_t h, void* result, cudaDataType t, cd v, cudaDataType exec) {
  std::vector<uint8_t> b(ex_bytes(t));
  ex_put(t, b.data(), v, exec);
  if (reinterpret_cast<Handle*>(h)->pointer_mode == CUBLAS_POINTER_MODE_DEVICE) store(result, b);
  else std::memcpy(result, b.data(), b.size());
}

// A reduction's checks and its graph capture, then `body`.
template <class Body>
cublasStatus_t ex_reduce(cublasHandle_t h, void* result, Body body) {
  if (!valid(h)) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (!result) return CUBLAS_STATUS_INVALID_VALUE;
  if (host_result_under_capture(h)) return CUBLAS_STATUS_NOT_SUPPORTED;
  if (deferred_to_graph(h, [=] { body(); })) return CUBLAS_STATUS_SUCCESS;
  body();
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t dot_ex(cublasHandle_t h, int n, const void* x, cudaDataType xt, int incx, const void* y,
                      cudaDataType yt, int incy, void* result, cudaDataType rt, cudaDataType et, bool conj) {
  if (!ex_known(xt) || yt != xt || rt != xt || et != ex_exec(xt)) return CUBLAS_STATUS_NOT_SUPPORTED;
  return ex_reduce(h, result, [=] {
    cd acc = 0;
    if (n > 0) {
      const auto xv = ex_load(x, xt, n, incx), yv = ex_load(y, yt, n, incy);
      for (int i = 0; i < n; ++i) acc += (conj ? std::conj(xv[i]) : xv[i]) * yv[i];
    }
    ex_result(h, result, rt, acc, et);
  });
}

cublasStatus_t nrm2_ex(cublasHandle_t h, int n, const void* x, cudaDataType xt, int incx, void* result,
                       cudaDataType rt, cudaDataType et) {
  if (!ex_known(xt) || rt != (ex_complex(xt) ? ex_real(xt) : xt) || et != ex_real(xt))
    return CUBLAS_STATUS_NOT_SUPPORTED;
  return ex_reduce(h, result, [=] {
    double acc = 0;
    if (n > 0 && incx > 0)
      for (const cd& v : ex_load(x, xt, n, incx)) acc += std::norm(v);
    ex_result(h, result, rt, std::sqrt(acc), et);
  });
}

// The sum of |re| + |im|, as BLAS's scasum and dzasum define it.
cublasStatus_t asum_ex(cublasHandle_t h, int n, const void* x, cudaDataType xt, int incx, void* result,
                       cudaDataType rt, cudaDataType et) {
  if (!ex_known(xt) || rt != ex_real(xt) || et != (ex_complex(xt) ? xt : ex_real(xt)))
    return CUBLAS_STATUS_NOT_SUPPORTED;
  return ex_reduce(h, result, [=] {
    double acc = 0;
    if (n > 0 && incx > 0)
      for (const cd& v : ex_load(x, xt, n, incx)) acc += std::fabs(v.real()) + std::fabs(v.imag());
    ex_result(h, result, rt, acc, ex_real(xt));
  });
}

// The 1-based index of the first element of largest (or smallest) |re| + |im|;
// 0 when there is nothing to search.
template <class R>
cublasStatus_t iamax_ex(cublasHandle_t h, int n, const void* x, cudaDataType xt, int incx, R* result, bool min) {
  if (!ex_known(xt)) return CUBLAS_STATUS_NOT_SUPPORTED;
  if (!valid(h)) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (!result) return CUBLAS_STATUS_INVALID_VALUE;
  if (host_result_under_capture(h)) return CUBLAS_STATUS_NOT_SUPPORTED;
  if (deferred_to_graph(h, [=] { iamax_ex<R>(h, n, x, xt, incx, result, min); })) return CUBLAS_STATUS_SUCCESS;
  R best = 0;
  if (n > 0 && incx > 0) {
    const auto xv = ex_load(x, xt, n, incx);
    double top = 0;
    for (int i = 0; i < n; ++i) {
      const double v = std::fabs(xv[i].real()) + std::fabs(xv[i].imag());
      if (i == 0 || (min ? v < top : v > top)) {
        top = v;
        best = i + 1;
      }
    }
  }
  put_result(h, result, best);
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t axpy_ex(cublasHandle_t h, int n, const void* alpha, cudaDataType at, const void* x, cudaDataType xt,
                       int incx, void* y, cudaDataType yt, int incy, cudaDataType et) {
  if (!ex_known(xt) || yt != xt || at != ex_exec(xt) || et != at) return CUBLAS_STATUS_NOT_SUPPORTED;
  if (!valid(h)) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (n <= 0) return CUBLAS_STATUS_SUCCESS;
  if (deferred_to_graph(h, [=, al = hold(h, alpha, ex_bytes(at))] {
        axpy_ex(h, n, al.get(), at, x, xt, incx, y, yt, incy, et);
      }))
    return CUBLAS_STATUS_SUCCESS;
  const cd a = ex_scalar(h, alpha, at);
  const auto xv = ex_load(x, xt, n, incx);
  auto yv = ex_load(y, yt, n, incy);
  for (int i = 0; i < n; ++i) yv[i] += a * xv[i];
  ex_store(y, yt, n, incy, yv, et);
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t scal_ex(cublasHandle_t h, int n, const void* alpha, cudaDataType at, void* x, cudaDataType xt,
                       int incx, cudaDataType et) {
  if (!ex_known(xt) || at != ex_exec(xt) || et != at) return CUBLAS_STATUS_NOT_SUPPORTED;
  if (!valid(h)) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (n <= 0 || incx <= 0) return CUBLAS_STATUS_SUCCESS;
  if (deferred_to_graph(h, [=, al = hold(h, alpha, ex_bytes(at))] { scal_ex(h, n, al.get(), at, x, xt, incx, et); }))
    return CUBLAS_STATUS_SUCCESS;
  const cd a = ex_scalar(h, alpha, at);
  auto xv = ex_load(x, xt, n, incx);
  for (cd& v : xv) v *= a;
  ex_store(x, xt, n, incx, xv, et);
  return CUBLAS_STATUS_SUCCESS;
}

// Copy and swap move elements without arithmetic; x and y are one type.
cublasStatus_t copy_ex(cublasHandle_t h, int n, const void* x, cudaDataType xt, int incx, void* y, cudaDataType yt,
                       int incy, bool swap) {
  if (!ex_known(xt) || yt != xt) return CUBLAS_STATUS_NOT_SUPPORTED;
  if (!valid(h)) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (n <= 0) return CUBLAS_STATUS_SUCCESS;
  if (deferred_to_graph(h, [=] { copy_ex(h, n, x, xt, incx, y, yt, incy, swap); })) return CUBLAS_STATUS_SUCCESS;
  const auto xv = ex_load(x, xt, n, incx);
  if (swap) ex_store(const_cast<void*>(x), xt, n, incx, ex_load(y, yt, n, incy), ex_complex(xt) ? xt : CUDA_R_64F);
  ex_store(y, yt, n, incy, xv, ex_complex(xt) ? xt : CUDA_R_64F);
  return CUBLAS_STATUS_SUCCESS;
}

}  // namespace

VGPU_EXPORT cublasStatus_t cublasDotEx(cublasHandle_t h, int n, const void* x, cudaDataType xType, int incx,
                                       const void* y, cudaDataType yType, int incy, void* result,
                                       cudaDataType resultType, cudaDataType executionType) {
  return dot_ex(h, n, x, xType, incx, y, yType, incy, result, resultType, executionType, false);
}
VGPU_EXPORT cublasStatus_t cublasDotcEx(cublasHandle_t h, int n, const void* x, cudaDataType xType, int incx,
                                        const void* y, cudaDataType yType, int incy, void* result,
                                        cudaDataType resultType, cudaDataType executionType) {
  return dot_ex(h, n, x, xType, incx, y, yType, incy, result, resultType, executionType, true);
}
VGPU_EXPORT cublasStatus_t cublasNrm2Ex(cublasHandle_t h, int n, const void* x, cudaDataType xType, int incx,
                                        void* result, cudaDataType resultType, cudaDataType executionType) {
  return nrm2_ex(h, n, x, xType, incx, result, resultType, executionType);
}
VGPU_EXPORT cublasStatus_t cublasAsumEx(cublasHandle_t h, int n, const void* x, cudaDataType xType, int incx,
                                        void* result, cudaDataType resultType, cudaDataType executionType) {
  return asum_ex(h, n, x, xType, incx, result, resultType, executionType);
}
VGPU_EXPORT cublasStatus_t cublasIamaxEx(cublasHandle_t h, int n, const void* x, cudaDataType xType, int incx,
                                         int* result) {
  return iamax_ex(h, n, x, xType, incx, result, false);
}
VGPU_EXPORT cublasStatus_t cublasIaminEx(cublasHandle_t h, int n, const void* x, cudaDataType xType, int incx,
                                         int* result) {
  return iamax_ex(h, n, x, xType, incx, result, true);
}
VGPU_EXPORT cublasStatus_t cublasAxpyEx(cublasHandle_t h, int n, const void* alpha, cudaDataType alphaType,
                                        const void* x, cudaDataType xType, int incx, void* y, cudaDataType yType,
                                        int incy, cudaDataType executiontype) {
  return axpy_ex(h, n, alpha, alphaType, x, xType, incx, y, yType, incy, executiontype);
}
VGPU_EXPORT cublasStatus_t cublasScalEx(cublasHandle_t h, int n, const void* alpha, cudaDataType alphaType, void* x,
                                        cudaDataType xType, int incx, cudaDataType executionType) {
  return scal_ex(h, n, alpha, alphaType, x, xType, incx, executionType);
}
VGPU_EXPORT cublasStatus_t cublasCopyEx(cublasHandle_t h, int n, const void* x, cudaDataType xType, int incx,
                                        void* y, cudaDataType yType, int incy) {
  return copy_ex(h, n, x, xType, incx, y, yType, incy, false);
}
VGPU_EXPORT cublasStatus_t cublasSwapEx(cublasHandle_t h, int n, void* x, cudaDataType xType, int incx, void* y,
                                        cudaDataType yType, int incy) {
  return copy_ex(h, n, x, xType, incx, y, yType, incy, true);
}

// The 64-bit forms (HeCBench's f16sp calls cublasDotEx_64), refused past int
// as the other _64 forms here are.
VGPU_EXPORT cublasStatus_t cublasDotEx_64(cublasHandle_t h, int64_t n, const void* x, cudaDataType xType,
                                          int64_t incx, const void* y, cudaDataType yType, int64_t incy,
                                          void* result, cudaDataType resultType, cudaDataType executionType) {
  if (!fits_int(n) || !fits_int(incx) || !fits_int(incy)) return CUBLAS_STATUS_NOT_SUPPORTED;
  return dot_ex(h, (int)n, x, xType, (int)incx, y, yType, (int)incy, result, resultType, executionType, false);
}
VGPU_EXPORT cublasStatus_t cublasDotcEx_64(cublasHandle_t h, int64_t n, const void* x, cudaDataType xType,
                                           int64_t incx, const void* y, cudaDataType yType, int64_t incy,
                                           void* result, cudaDataType resultType, cudaDataType executionType) {
  if (!fits_int(n) || !fits_int(incx) || !fits_int(incy)) return CUBLAS_STATUS_NOT_SUPPORTED;
  return dot_ex(h, (int)n, x, xType, (int)incx, y, yType, (int)incy, result, resultType, executionType, true);
}
VGPU_EXPORT cublasStatus_t cublasNrm2Ex_64(cublasHandle_t h, int64_t n, const void* x, cudaDataType xType,
                                           int64_t incx, void* result, cudaDataType resultType,
                                           cudaDataType executionType) {
  if (!fits_int(n) || !fits_int(incx)) return CUBLAS_STATUS_NOT_SUPPORTED;
  return nrm2_ex(h, (int)n, x, xType, (int)incx, result, resultType, executionType);
}
VGPU_EXPORT cublasStatus_t cublasAsumEx_64(cublasHandle_t h, int64_t n, const void* x, cudaDataType xType,
                                           int64_t incx, void* result, cudaDataType resultType,
                                           cudaDataType executionType) {
  if (!fits_int(n) || !fits_int(incx)) return CUBLAS_STATUS_NOT_SUPPORTED;
  return asum_ex(h, (int)n, x, xType, (int)incx, result, resultType, executionType);
}
VGPU_EXPORT cublasStatus_t cublasIamaxEx_64(cublasHandle_t h, int64_t n, const void* x, cudaDataType xType,
                                            int64_t incx, int64_t* result) {
  if (!fits_int(n) || !fits_int(incx)) return CUBLAS_STATUS_NOT_SUPPORTED;
  return iamax_ex(h, (int)n, x, xType, (int)incx, result, false);
}
VGPU_EXPORT cublasStatus_t cublasIaminEx_64(cublasHandle_t h, int64_t n, const void* x, cudaDataType xType,
                                            int64_t incx, int64_t* result) {
  if (!fits_int(n) || !fits_int(incx)) return CUBLAS_STATUS_NOT_SUPPORTED;
  return iamax_ex(h, (int)n, x, xType, (int)incx, result, true);
}
VGPU_EXPORT cublasStatus_t cublasAxpyEx_64(cublasHandle_t h, int64_t n, const void* alpha, cudaDataType alphaType,
                                           const void* x, cudaDataType xType, int64_t incx, void* y,
                                           cudaDataType yType, int64_t incy, cudaDataType executiontype) {
  if (!fits_int(n) || !fits_int(incx) || !fits_int(incy)) return CUBLAS_STATUS_NOT_SUPPORTED;
  return axpy_ex(h, (int)n, alpha, alphaType, x, xType, (int)incx, y, yType, (int)incy, executiontype);
}
VGPU_EXPORT cublasStatus_t cublasScalEx_64(cublasHandle_t h, int64_t n, const void* alpha, cudaDataType alphaType,
                                           void* x, cudaDataType xType, int64_t incx, cudaDataType executionType) {
  if (!fits_int(n) || !fits_int(incx)) return CUBLAS_STATUS_NOT_SUPPORTED;
  return scal_ex(h, (int)n, alpha, alphaType, x, xType, (int)incx, executionType);
}
VGPU_EXPORT cublasStatus_t cublasCopyEx_64(cublasHandle_t h, int64_t n, const void* x, cudaDataType xType,
                                           int64_t incx, void* y, cudaDataType yType, int64_t incy) {
  if (!fits_int(n) || !fits_int(incx) || !fits_int(incy)) return CUBLAS_STATUS_NOT_SUPPORTED;
  return copy_ex(h, (int)n, x, xType, (int)incx, y, yType, (int)incy, false);
}
VGPU_EXPORT cublasStatus_t cublasSwapEx_64(cublasHandle_t h, int64_t n, void* x, cudaDataType xType, int64_t incx,
                                           void* y, cudaDataType yType, int64_t incy) {
  if (!fits_int(n) || !fits_int(incx) || !fits_int(incy)) return CUBLAS_STATUS_NOT_SUPPORTED;
  return copy_ex(h, (int)n, x, xType, (int)incx, y, yType, (int)incy, true);
}

// The 64-bit GEMM forms, onto the 32-bit ones.
VGPU_EXPORT cublasStatus_t cublasGemmEx_64(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int64_t m,
                                           int64_t n, int64_t k, const void* alpha, const void* A, cudaDataType Atype,
                                           int64_t lda, const void* B, cudaDataType Btype, int64_t ldb,
                                           const void* beta, void* C, cudaDataType Ctype, int64_t ldc,
                                           cublasComputeType_t computeType, cublasGemmAlgo_t algo) {
  for (int64_t v : {m, n, k, lda, ldb, ldc})
    if (!fits_int(v)) return CUBLAS_STATUS_NOT_SUPPORTED;
  return cublasGemmEx(h, ta, tb, (int)m, (int)n, (int)k, alpha, A, Atype, (int)lda, B, Btype, (int)ldb, beta, C,
                      Ctype, (int)ldc, computeType, algo);
}
VGPU_EXPORT cublasStatus_t cublasGemmBatchedEx_64(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb,
                                                  int64_t m, int64_t n, int64_t k, const void* alpha,
                                                  const void* const Aarray[], cudaDataType Atype, int64_t lda,
                                                  const void* const Barray[], cudaDataType Btype, int64_t ldb,
                                                  const void* beta, void* const Carray[], cudaDataType Ctype,
                                                  int64_t ldc, int64_t batchCount, cublasComputeType_t computeType,
                                                  cublasGemmAlgo_t algo) {
  for (int64_t v : {m, n, k, lda, ldb, ldc, batchCount})
    if (!fits_int(v)) return CUBLAS_STATUS_NOT_SUPPORTED;
  return cublasGemmBatchedEx(h, ta, tb, (int)m, (int)n, (int)k, alpha, Aarray, Atype, (int)lda, Barray, Btype,
                             (int)ldb, beta, Carray, Ctype, (int)ldc, (int)batchCount, computeType, algo);
}

/* ---- grouped batched GEMM ----
   Group g applies its own transposes, sizes, leading dimensions, alpha[g] and
   beta[g] to the next group_size[g] problems of the pointer arrays. What an
   RTX 3060's cuBLAS does, measured:
     - every group is checked before any runs: a bad last group leaves the
       first group's C untouched (INVALID_VALUE for a negative count, size or
       dimension, or a leading dimension too small);
     - alpha and beta come from the host only: device pointer mode is
       NOT_SUPPORTED;
     - the types are 32F with COMPUTE_32F, _32F_PEDANTIC or _32F_FAST_TF32;
       16F or 16BF throughout with COMPUTE_32F; 64F with COMPUTE_64F. Any
       other combination is NOT_SUPPORTED, mixed C types included. */
namespace {
bool grouped_types_ok(cudaDataType at, cudaDataType bt, cudaDataType ct, cublasComputeType_t cp) {
  if (at != bt || bt != ct) return false;
  switch (at) {
    case CUDA_R_32F:
      return cp == CUBLAS_COMPUTE_32F || cp == CUBLAS_COMPUTE_32F_PEDANTIC || cp == CUBLAS_COMPUTE_32F_FAST_TF32;
    case CUDA_R_16F:
    case CUDA_R_16BF: return cp == CUBLAS_COMPUTE_32F;
    case CUDA_R_64F: return cp == CUBLAS_COMPUTE_64F;
    default: return false;
  }
}

bool op_ok(cublasOperation_t t) { return t == CUBLAS_OP_N || t == CUBLAS_OP_T || t == CUBLAS_OP_C; }

template <typename I>
cublasStatus_t gemm_grouped(cublasHandle_t h, const cublasOperation_t* ta, const cublasOperation_t* tb, const I* m,
                            const I* n, const I* k, const void* alpha, const void* const* Aarray, cudaDataType at,
                            const I* lda, const void* const* Barray, cudaDataType bt, const I* ldb, const void* beta,
                            void* const* Carray, cudaDataType ct, const I* ldc, I group_count, const I* group_size,
                            cublasComputeType_t cp) {
  if (!valid(h)) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (group_count < 0) return CUBLAS_STATUS_INVALID_VALUE;
  if (group_count == 0) return CUBLAS_STATUS_SUCCESS;
  if (!grouped_types_ok(at, bt, ct, cp)) return CUBLAS_STATUS_NOT_SUPPORTED;
  if (reinterpret_cast<Handle*>(h)->pointer_mode != CUBLAS_POINTER_MODE_HOST) return CUBLAS_STATUS_NOT_SUPPORTED;
  if (!ta || !tb || !m || !n || !k || !lda || !ldb || !ldc || !group_size || !alpha || !beta)
    return CUBLAS_STATUS_INVALID_VALUE;
  int64_t total = 0;
  for (I g = 0; g < group_count; ++g) {
    if (group_size[g] < 0 || !op_ok(ta[g]) || !op_ok(tb[g]) || m[g] < 0 || n[g] < 0 || k[g] < 0)
      return CUBLAS_STATUS_INVALID_VALUE;
    const int64_t a_rows = ta[g] == CUBLAS_OP_N ? m[g] : k[g];
    const int64_t b_rows = tb[g] == CUBLAS_OP_N ? k[g] : n[g];
    if (lda[g] < std::max<int64_t>(1, a_rows) || ldb[g] < std::max<int64_t>(1, b_rows) ||
        ldc[g] < std::max<int64_t>(1, m[g]))
      return CUBLAS_STATUS_INVALID_VALUE;
    total += group_size[g];
  }
  if (total == 0) return CUBLAS_STATUS_SUCCESS;
  const size_t sb = compute_scalar_bytes(cp);
  const size_t gc = static_cast<size_t>(group_count);
  // Under stream capture the call replays at launch: the host arrays are
  // copied now, the device pointer arrays read then.
  if (deferred_to_graph(h, [=, vta = std::vector<cublasOperation_t>(ta, ta + gc),
                            vtb = std::vector<cublasOperation_t>(tb, tb + gc), vm = std::vector<I>(m, m + gc),
                            vn = std::vector<I>(n, n + gc), vk = std::vector<I>(k, k + gc),
                            vla = std::vector<I>(lda, lda + gc), vlb = std::vector<I>(ldb, ldb + gc),
                            vlc = std::vector<I>(ldc, ldc + gc), vgs = std::vector<I>(group_size, group_size + gc),
                            al = std::vector<uint8_t>(static_cast<const uint8_t*>(alpha),
                                                      static_cast<const uint8_t*>(alpha) + gc * sb),
                            be = std::vector<uint8_t>(static_cast<const uint8_t*>(beta),
                                                      static_cast<const uint8_t*>(beta) + gc * sb)] {
        gemm_grouped<I>(h, vta.data(), vtb.data(), vm.data(), vn.data(), vk.data(), al.data(), Aarray, at,
                        vla.data(), Barray, bt, vlb.data(), be.data(), Carray, ct, vlc.data(), group_count,
                        vgs.data(), cp);
      }))
    return CUBLAS_STATUS_SUCCESS;
  const auto a = fetch<const void*>(Aarray, static_cast<size_t>(total));
  const auto b = fetch<const void*>(Barray, static_cast<size_t>(total));
  const auto c = fetch<void*>(Carray, static_cast<size_t>(total));
  size_t idx = 0;
  for (I g = 0; g < group_count; ++g) {
    const void* al = static_cast<const uint8_t*>(alpha) + static_cast<size_t>(g) * sb;
    const void* be = static_cast<const uint8_t*>(beta) + static_cast<size_t>(g) * sb;
    for (I s = 0; s < group_size[g]; ++s, ++idx) {
      const cublasStatus_t st = cublasGemmEx_64(h, ta[g], tb[g], m[g], n[g], k[g], al, a[idx], at, lda[g], b[idx], bt,
                                                ldb[g], be, c[idx], ct, ldc[g], cp, CUBLAS_GEMM_DEFAULT);
      if (st != CUBLAS_STATUS_SUCCESS) return st;
    }
  }
  return CUBLAS_STATUS_SUCCESS;
}
}  // namespace

VGPU_EXPORT cublasStatus_t cublasGemmGroupedBatchedEx(
    cublasHandle_t h, const cublasOperation_t transa_array[], const cublasOperation_t transb_array[],
    const int m_array[], const int n_array[], const int k_array[], const void* alpha_array, const void* const Aarray[],
    cudaDataType_t Atype, const int lda_array[], const void* const Barray[], cudaDataType_t Btype,
    const int ldb_array[], const void* beta_array, void* const Carray[], cudaDataType_t Ctype, const int ldc_array[],
    int group_count, const int group_size[], cublasComputeType_t computeType) {
  return gemm_grouped<int>(h, transa_array, transb_array, m_array, n_array, k_array, alpha_array, Aarray, Atype,
                           lda_array, Barray, Btype, ldb_array, beta_array, Carray, Ctype, ldc_array, group_count,
                           group_size, computeType);
}

VGPU_EXPORT cublasStatus_t cublasGemmGroupedBatchedEx_64(
    cublasHandle_t h, const cublasOperation_t transa_array[], const cublasOperation_t transb_array[],
    const int64_t m_array[], const int64_t n_array[], const int64_t k_array[], const void* alpha_array,
    const void* const Aarray[], cudaDataType_t Atype, const int64_t lda_array[], const void* const Barray[],
    cudaDataType_t Btype, const int64_t ldb_array[], const void* beta_array, void* const Carray[],
    cudaDataType_t Ctype, const int64_t ldc_array[], int64_t group_count, const int64_t group_size[],
    cublasComputeType_t computeType) {
  return gemm_grouped<int64_t>(h, transa_array, transb_array, m_array, n_array, k_array, alpha_array, Aarray, Atype,
                               lda_array, Barray, Btype, ldb_array, beta_array, Carray, Ctype, ldc_array,
                               group_count, group_size, computeType);
}

// The typed forms are the Ex form at the type's own compute type.
#define VGPU_GROUPED(NAME, T, I, DT, CT)                                                                          \
  VGPU_EXPORT cublasStatus_t NAME(cublasHandle_t h, const cublasOperation_t ta[], const cublasOperation_t tb[],  \
                                  const I m[], const I n[], const I k[], const T alpha[], const T* const A[],    \
                                  const I lda[], const T* const B[], const I ldb[], const T beta[],              \
                                  T* const C[], const I ldc[], I group_count, const I group_size[]) {            \
    return gemm_grouped<I>(h, ta, tb, m, n, k, alpha, reinterpret_cast<const void* const*>(A), DT, lda,         \
                           reinterpret_cast<const void* const*>(B), DT, ldb, beta,                               \
                           reinterpret_cast<void* const*>(C), DT, ldc, group_count, group_size, CT);             \
  }
VGPU_GROUPED(cublasSgemmGroupedBatched, float, int, CUDA_R_32F, CUBLAS_COMPUTE_32F)
VGPU_GROUPED(cublasSgemmGroupedBatched_64, float, int64_t, CUDA_R_32F, CUBLAS_COMPUTE_32F)
VGPU_GROUPED(cublasDgemmGroupedBatched, double, int, CUDA_R_64F, CUBLAS_COMPUTE_64F)
VGPU_GROUPED(cublasDgemmGroupedBatched_64, double, int64_t, CUDA_R_64F, CUBLAS_COMPUTE_64F)
#undef VGPU_GROUPED
VGPU_EXPORT cublasStatus_t cublasGemmStridedBatchedEx_64(
    cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int64_t m, int64_t n, int64_t k, const void* alpha,
    const void* A, cudaDataType Atype, int64_t lda, long long strideA, const void* B, cudaDataType Btype, int64_t ldb,
    long long strideB, const void* beta, void* C, cudaDataType Ctype, int64_t ldc, long long strideC,
    int64_t batchCount, cublasComputeType_t computeType, cublasGemmAlgo_t algo) {
  for (int64_t v : {m, n, k, lda, ldb, ldc, batchCount})
    if (!fits_int(v)) return CUBLAS_STATUS_NOT_SUPPORTED;
  return cublasGemmStridedBatchedEx(h, ta, tb, (int)m, (int)n, (int)k, alpha, A, Atype, (int)lda, strideA, B, Btype,
                                    (int)ldb, strideB, beta, C, Ctype, (int)ldc, strideC, (int)batchCount,
                                    computeType, algo);
}
VGPU_EXPORT cublasStatus_t cublasSgemmEx_64(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int64_t m,
                                            int64_t n, int64_t k, const float* alpha, const void* A, cudaDataType Atype,
                                            int64_t lda, const void* B, cudaDataType Btype, int64_t ldb,
                                            const float* beta, void* C, cudaDataType Ctype, int64_t ldc) {
  for (int64_t v : {m, n, k, lda, ldb, ldc})
    if (!fits_int(v)) return CUBLAS_STATUS_NOT_SUPPORTED;
  return cublasSgemmEx(h, ta, tb, (int)m, (int)n, (int)k, alpha, A, Atype, (int)lda, B, Btype, (int)ldb, beta, C,
                       Ctype, (int)ldc);
}

/* ---- the rot family: rot, rotg, rotm and rotmg, typed and Ex ----
   Plane rotations, after reference BLAS, with the arithmetic in the order an
   RTX 3060's cuBLAS does it, so that real results match it bit for bit:

     rot    x' = fma(c, x, s y), y' = fma(c, y, -(s x)); a complex rotation
            takes c real (the imaginary part of a complex c is ignored, as the
            card ignores it), y' = c y - conj(s) x, and is fused the same way
            part by part (rot_cplx).
     rotg   r = sigma (scl sqrt((a/scl)^2 + (b/scl)^2)), scl = max(|a|, |b|),
            sigma the sign of the larger. The larger of c and s is a/r or b/r
            and the smaller is that times the ratio of the two (s = (b/a) c),
            and z as BLAS defines it. No shortcut for a zero a or b, so rotg(0,
            -3) gives c = -0 as the card does; only a and b both zero do. The
            complex rotg is reference BLAS's crotg, which leaves b alone,
            computed in double: within a few ulps of the card, whose single-
            precision order of operations this does not reproduce.
     rotm   by the flag in param[0]: -1 x' = fma(h11, x, h12 y), y' = fma(h21,
            x, h22 y); 0 x' = fma(h12, y, x), y' = fma(h21, x, y); 1 x' =
            fma(h11, x, y), y' = fma(h22, y, -x); -2, or any other value, is
            the identity.
     rotmg  reference BLAS's srotmg/drotmg, step for step.

   The Ex forms take the types an RTX 3060 accepts (anything else is
   NOT_SUPPORTED): x, y, c and s alike in R_16F, R_16BF, R_32F or R_64F, with
   execution in R_32F (R_64F for doubles); for rot also C_32F and C_64F with c
   and s that type or its real type. rotg, rotm and rotmg are real only, and
   rotmg's five operands all one type. Half and bfloat16 values are rotated in
   single precision and rounded back. A non-positive n does nothing. The
   scalars, and every operand of rotg and rotmg, are host or device memory by
   the pointer mode. */
namespace {

template <class R>
void rot_real(std::vector<cd>& x, std::vector<cd>& y, R c, R s) {
  for (size_t i = 0; i < x.size(); ++i) {
    const R xi = (R)x[i].real(), yi = (R)y[i].real();
    x[i] = std::fma(c, xi, s * yi);
    y[i] = std::fma(c, yi, -(s * xi));
  }
}

// The complex rotation, c real: s y and conj(s) x are each one complex
// product, (pr qr - pi qi, pi qr + pr qi) with the first product of each part
// fused, and c times x (or y) is fused onto it.
template <class R>
void rot_cplx(std::vector<cd>& x, std::vector<cd>& y, R c, R sr, R si) {
  for (size_t i = 0; i < x.size(); ++i) {
    const R xr = (R)x[i].real(), xi = (R)x[i].imag(), yr = (R)y[i].real(), yi = (R)y[i].imag();
    const R tr = std::fma(sr, yr, -(si * yi)), ti = std::fma(si, yr, sr * yi);   // s y
    const R ur = std::fma(sr, xr, si * xi), ui = std::fma(-si, xr, sr * xi);     // conj(s) x
    x[i] = {std::fma(c, xr, tr), std::fma(c, xi, ti)};
    y[i] = {std::fma(c, yr, -ur), std::fma(c, yi, -ui)};
  }
}

// c is read as `ct` and s as `st`; only c's real part is used.
cublasStatus_t rot_core(cublasHandle_t h, int n, void* x, cudaDataType xt, int incx, void* y, int incy,
                        const void* c, cudaDataType ct, const void* s, cudaDataType st, cudaDataType et) {
  if (!valid(h)) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (n <= 0) return CUBLAS_STATUS_SUCCESS;
  if (deferred_to_graph(h, [=, cv = hold(h, c, ex_bytes(ct)), sv = hold(h, s, ex_bytes(st))] {
        rot_core(h, n, x, xt, incx, y, incy, cv.get(), ct, sv.get(), st, et);
      }))
    return CUBLAS_STATUS_SUCCESS;
  const double cr = ex_scalar(h, c, ct).real();
  const cd sv = ex_scalar(h, s, st);
  auto xv = ex_load(x, xt, n, incx), yv = ex_load(y, xt, n, incy);
  if (et == CUDA_R_32F) {
    rot_real<float>(xv, yv, (float)cr, (float)sv.real());
  } else if (et == CUDA_R_64F) {
    rot_real<double>(xv, yv, cr, sv.real());
  } else if (et == CUDA_C_32F) {
    rot_cplx<float>(xv, yv, (float)cr, (float)sv.real(), (float)sv.imag());
  } else {
    rot_cplx<double>(xv, yv, cr, sv.real(), sv.imag());
  }
  ex_store(x, xt, n, incx, xv, et);
  ex_store(y, xt, n, incy, yv, et);
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t rot_ex(cublasHandle_t h, int n, void* x, cudaDataType xt, int incx, void* y, cudaDataType yt,
                      int incy, const void* c, const void* s, cudaDataType cst, cudaDataType et) {
  if (!ex_known(xt) || yt != xt || et != ex_exec(xt) || (cst != xt && !(ex_complex(xt) && cst == ex_real(xt))))
    return CUBLAS_STATUS_NOT_SUPPORTED;
  return rot_core(h, n, x, xt, incx, y, incy, c, cst, s, cst, et);
}

// The real types rotg, rotm and rotmg take: R_16F, R_16BF, R_32F, R_64F.
bool rot_real_type(cudaDataType t) { return ex_known(t) && !ex_complex(t); }

template <class R>
void rotg_real(R& a, R& b, R& c, R& s) {
  const R an = std::fabs(a), bn = std::fabs(b);
  if (an == 0 && bn == 0) {
    c = 1;
    s = 0;
    return;
  }
  const R safmin = std::numeric_limits<R>::min(), safmax = 1 / safmin;
  const R scl = std::min(safmax, std::max(safmin, std::max(an, bn)));
  const R sigma = an > bn ? std::copysign(R(1), a) : std::copysign(R(1), b);
  const R p = a / scl, q = b / scl;
  const R r = sigma * (scl * std::sqrt(p * p + q * q));
  if (an > bn) {
    c = a / r;
    s = (b / a) * c;
  } else {
    s = b / r;
    c = (a / b) * s;
  }
  b = an > bn ? s : c != 0 ? 1 / c : R(1);
  a = r;
}

// The operands of rotg and rotmg are all host or all device memory.
cd rot_get(cublasHandle_t h, const void* p, cudaDataType t) { return ex_scalar(h, p, t); }
void rot_put(cublasHandle_t h, void* p, cudaDataType t, cd v) { ex_result(h, p, t, v, ex_exec(t)); }

cublasStatus_t rotg_ex(cublasHandle_t h, void* a, void* b, cudaDataType abt, void* c, void* s, cudaDataType cst,
                       cudaDataType et) {
  if (!rot_real_type(abt) || cst != abt || et != ex_exec(abt)) return CUBLAS_STATUS_NOT_SUPPORTED;
  if (!valid(h)) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (host_result_under_capture(h)) return CUBLAS_STATUS_NOT_SUPPORTED;
  if (deferred_to_graph(h, [=] { rotg_ex(h, a, b, abt, c, s, cst, et); })) return CUBLAS_STATUS_SUCCESS;
  if (et == CUDA_R_64F) {
    double av = rot_get(h, a, abt).real(), bv = rot_get(h, b, abt).real(), cv, sv;
    rotg_real(av, bv, cv, sv);
    rot_put(h, a, abt, av), rot_put(h, b, abt, bv), rot_put(h, c, cst, cv), rot_put(h, s, cst, sv);
  } else {
    float av = (float)rot_get(h, a, abt).real(), bv = (float)rot_get(h, b, abt).real(), cv, sv;
    rotg_real(av, bv, cv, sv);
    rot_put(h, a, abt, av), rot_put(h, b, abt, bv), rot_put(h, c, cst, cv), rot_put(h, s, cst, sv);
  }
  return CUBLAS_STATUS_SUCCESS;
}

// Reference BLAS's crotg/zrotg: c real, s complex, a overwritten by r and b
// left as it was.
cublasStatus_t rotg_complex(cublasHandle_t h, void* a, void* b, void* c, void* s, cudaDataType t) {
  if (!valid(h)) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (host_result_under_capture(h)) return CUBLAS_STATUS_NOT_SUPPORTED;
  if (deferred_to_graph(h, [=] { rotg_complex(h, a, b, c, s, t); })) return CUBLAS_STATUS_SUCCESS;
  const cudaDataType rt = ex_real(t);
  const cd ca = rot_get(h, a, t), cb = rot_get(h, b, t);
  if (std::abs(ca) == 0) {
    rot_put(h, c, rt, 0), rot_put(h, s, t, 1), rot_put(h, a, t, cb);
    return CUBLAS_STATUS_SUCCESS;
  }
  const double scale = std::abs(ca) + std::abs(cb);
  const double norm = scale * std::sqrt(std::norm(ca / scale) + std::norm(cb / scale));
  const cd alpha = ca / std::abs(ca);
  rot_put(h, c, rt, std::abs(ca) / norm), rot_put(h, s, t, alpha * std::conj(cb) / norm), rot_put(h, a, t, alpha * norm);
  return CUBLAS_STATUS_SUCCESS;
}

template <class R>
void rotm_real(std::vector<cd>& x, std::vector<cd>& y, const R* p) {
  const R flag = p[0], h11 = p[1], h21 = p[2], h12 = p[3], h22 = p[4];
  for (size_t i = 0; i < x.size(); ++i) {
    const R xi = (R)x[i].real(), yi = (R)y[i].real();
    if (flag == -1) {
      x[i] = std::fma(h11, xi, h12 * yi);
      y[i] = std::fma(h21, xi, h22 * yi);
    } else if (flag == 0) {
      x[i] = std::fma(h12, yi, xi);
      y[i] = std::fma(h21, xi, yi);
    } else if (flag == 1) {
      x[i] = std::fma(h11, xi, yi);
      y[i] = std::fma(h22, yi, -xi);
    }
  }
}

cublasStatus_t rotm_ex(cublasHandle_t h, int n, void* x, cudaDataType xt, int incx, void* y, cudaDataType yt,
                       int incy, const void* param, cudaDataType pt, cudaDataType et) {
  if (!rot_real_type(xt) || yt != xt || pt != xt || et != ex_exec(xt)) return CUBLAS_STATUS_NOT_SUPPORTED;
  if (!valid(h)) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (n <= 0) return CUBLAS_STATUS_SUCCESS;
  if (deferred_to_graph(h, [=, pv = hold(h, param, 5 * ex_bytes(pt))] {
        rotm_ex(h, n, x, xt, incx, y, yt, incy, pv.get(), pt, et);
      }))
    return CUBLAS_STATUS_SUCCESS;
  std::vector<uint8_t> raw(5 * ex_bytes(pt));
  if (reinterpret_cast<Handle*>(h)->pointer_mode == CUBLAS_POINTER_MODE_DEVICE) raw = fetch<uint8_t>(param, raw.size());
  else std::memcpy(raw.data(), param, raw.size());
  double p[5];
  for (int i = 0; i < 5; ++i) p[i] = ex_get(pt, raw.data() + i * ex_bytes(pt)).real();
  if (p[0] != -1 && p[0] != 0 && p[0] != 1) return CUBLAS_STATUS_SUCCESS;
  auto xv = ex_load(x, xt, n, incx), yv = ex_load(y, yt, n, incy);
  if (et == CUDA_R_64F) {
    rotm_real<double>(xv, yv, p);
  } else {
    const float pf[5] = {(float)p[0], (float)p[1], (float)p[2], (float)p[3], (float)p[4]};
    rotm_real<float>(xv, yv, pf);
  }
  ex_store(x, xt, n, incx, xv, et);
  ex_store(y, yt, n, incy, yv, et);
  return CUBLAS_STATUS_SUCCESS;
}

// Reference BLAS's srotmg, step for step; param[] entries the flag does not
// name are left as they were.
template <class R>
void rotmg_real(R& d1, R& d2, R& x1, R y1, R* param, bool* written) {
  const R gam = 4096, gamsq = gam * gam, rgamsq = 1 / gamsq;
  R flag, h11 = 0, h12 = 0, h21 = 0, h22 = 0;
  auto zero_all = [&] {
    flag = -1;
    h11 = h12 = h21 = h22 = 0;
    d1 = d2 = x1 = 0;
  };
  for (int i = 0; i < 5; ++i) written[i] = false;
  if (d1 < 0) {
    zero_all();
  } else {
    const R p2 = d2 * y1;
    if (p2 == 0) {
      param[0] = -2;
      written[0] = true;
      return;
    }
    const R p1 = d1 * x1, q2 = p2 * y1, q1 = p1 * x1;
    if (std::fabs(q1) > std::fabs(q2)) {
      h21 = -y1 / x1;
      h12 = p2 / p1;
      const R u = 1 - h12 * h21;
      if (u > 0) {
        flag = 0;
        d1 /= u;
        d2 /= u;
        x1 *= u;
      } else {
        zero_all();
      }
    } else if (q2 < 0) {
      zero_all();
    } else {
      flag = 1;
      h11 = p1 / p2;
      h22 = x1 / y1;
      const R u = 1 + h11 * h22, t = d2 / u;
      d2 = d1 / u;
      d1 = t;
      x1 = y1 * u;
    }
    // Rescale d1 and d2 into (1/gam^2, gam^2), the matrix becoming a full one.
    auto full = [&] {
      if (flag == 0) h11 = h22 = 1;
      else if (flag == 1) h21 = -1, h12 = 1;
      flag = -1;
    };
    if (d1 != 0)
      while (d1 <= rgamsq || d1 >= gamsq) {
        full();
        if (d1 <= rgamsq) d1 *= gam * gam, x1 /= gam, h11 /= gam, h12 /= gam;
        else d1 /= gam * gam, x1 *= gam, h11 *= gam, h12 *= gam;
      }
    if (d2 != 0)
      while (std::fabs(d2) <= rgamsq || std::fabs(d2) >= gamsq) {
        full();
        if (std::fabs(d2) <= rgamsq) d2 *= gam * gam, h21 /= gam, h22 /= gam;
        else d2 /= gam * gam, h21 *= gam, h22 *= gam;
      }
  }
  if (flag < 0) param[1] = h11, param[2] = h21, param[3] = h12, param[4] = h22;
  else if (flag == 0) param[2] = h21, param[3] = h12;
  else param[1] = h11, param[4] = h22;
  param[0] = flag;
  written[0] = true;
  for (int i = 1; i < 5; ++i) written[i] = flag < 0 || (flag == 0 ? i == 2 || i == 3 : i == 1 || i == 4);
}

cublasStatus_t rotmg_ex(cublasHandle_t h, void* d1, cudaDataType d1t, void* d2, cudaDataType d2t, void* x1,
                        cudaDataType x1t, const void* y1, cudaDataType y1t, void* param, cudaDataType pt,
                        cudaDataType et) {
  if (!rot_real_type(d1t) || d2t != d1t || x1t != d1t || y1t != d1t || pt != d1t || et != ex_exec(d1t))
    return CUBLAS_STATUS_NOT_SUPPORTED;
  if (!valid(h)) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (host_result_under_capture(h)) return CUBLAS_STATUS_NOT_SUPPORTED;
  if (deferred_to_graph(h, [=] { rotmg_ex(h, d1, d1t, d2, d2t, x1, x1t, y1, y1t, param, pt, et); }))
    return CUBLAS_STATUS_SUCCESS;
  const size_t e = ex_bytes(pt);
  auto run = [&](auto zero) {
    using R = decltype(zero);
    R a = (R)rot_get(h, d1, d1t).real(), b = (R)rot_get(h, d2, d2t).real(), x = (R)rot_get(h, x1, x1t).real();
    const R y = (R)rot_get(h, y1, y1t).real();
    R p[5] = {};
    bool written[5];
    rotmg_real(a, b, x, y, p, written);
    rot_put(h, d1, d1t, a), rot_put(h, d2, d2t, b), rot_put(h, x1, x1t, x);
    for (int i = 0; i < 5; ++i)
      if (written[i]) rot_put(h, static_cast<uint8_t*>(param) + i * e, pt, p[i]);
  };
  if (et == CUDA_R_64F) run(0.0);
  else run(0.0f);
  return CUBLAS_STATUS_SUCCESS;
}

}  // namespace

VGPU_EXPORT cublasStatus_t cublasRotEx(cublasHandle_t h, int n, void* x, cudaDataType xType, int incx, void* y,
                                       cudaDataType yType, int incy, const void* c, const void* s,
                                       cudaDataType csType, cudaDataType executiontype) {
  return rot_ex(h, n, x, xType, incx, y, yType, incy, c, s, csType, executiontype);
}
VGPU_EXPORT cublasStatus_t cublasRotEx_64(cublasHandle_t h, int64_t n, void* x, cudaDataType xType, int64_t incx,
                                          void* y, cudaDataType yType, int64_t incy, const void* c, const void* s,
                                          cudaDataType csType, cudaDataType executiontype) {
  if (!fits_int(n) || !fits_int(incx) || !fits_int(incy)) return CUBLAS_STATUS_NOT_SUPPORTED;
  return rot_ex(h, (int)n, x, xType, (int)incx, y, yType, (int)incy, c, s, csType, executiontype);
}
VGPU_EXPORT cublasStatus_t cublasRotgEx(cublasHandle_t h, void* a, void* b, cudaDataType abType, void* c, void* s,
                                        cudaDataType csType, cudaDataType executiontype) {
  return rotg_ex(h, a, b, abType, c, s, csType, executiontype);
}
VGPU_EXPORT cublasStatus_t cublasRotmEx(cublasHandle_t h, int n, void* x, cudaDataType xType, int incx, void* y,
                                        cudaDataType yType, int incy, const void* param, cudaDataType paramType,
                                        cudaDataType executiontype) {
  return rotm_ex(h, n, x, xType, incx, y, yType, incy, param, paramType, executiontype);
}
VGPU_EXPORT cublasStatus_t cublasRotmEx_64(cublasHandle_t h, int64_t n, void* x, cudaDataType xType, int64_t incx,
                                           void* y, cudaDataType yType, int64_t incy, const void* param,
                                           cudaDataType paramType, cudaDataType executiontype) {
  if (!fits_int(n) || !fits_int(incx) || !fits_int(incy)) return CUBLAS_STATUS_NOT_SUPPORTED;
  return rotm_ex(h, (int)n, x, xType, (int)incx, y, yType, (int)incy, param, paramType, executiontype);
}
VGPU_EXPORT cublasStatus_t cublasRotmgEx(cublasHandle_t h, void* d1, cudaDataType d1Type, void* d2,
                                         cudaDataType d2Type, void* x1, cudaDataType x1Type, const void* y1,
                                         cudaDataType y1Type, void* param, cudaDataType paramType,
                                         cudaDataType executiontype) {
  return rotmg_ex(h, d1, d1Type, d2, d2Type, x1, x1Type, y1, y1Type, param, paramType, executiontype);
}

// The typed forms, onto the Ex ones (crot's c is real and its s complex).
#define VGPU_ROT(P, T, CT, ST, XT, CST, SST, ET)                                                                  \
  VGPU_EXPORT cublasStatus_t cublas##P##rot_v2(cublasHandle_t h, int n, T* x, int incx, T* y, int incy,           \
                                               const CT* c, const ST* s) {                                        \
    return rot_core(h, n, x, XT, incx, y, incy, c, CST, s, SST, ET);                                              \
  }                                                                                                               \
  VGPU_EXPORT cublasStatus_t cublas##P##rot_v2_64(cublasHandle_t h, int64_t n, T* x, int64_t incx, T* y,          \
                                                  int64_t incy, const CT* c, const ST* s) {                       \
    if (!fits_int(n) || !fits_int(incx) || !fits_int(incy)) return CUBLAS_STATUS_NOT_SUPPORTED;                   \
    return rot_core(h, (int)n, x, XT, (int)incx, y, (int)incy, c, CST, s, SST, ET);                               \
  }
VGPU_ROT(S, float, float, float, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F)
VGPU_ROT(D, double, double, double, CUDA_R_64F, CUDA_R_64F, CUDA_R_64F, CUDA_R_64F)
VGPU_ROT(C, cuComplex, float, cuComplex, CUDA_C_32F, CUDA_R_32F, CUDA_C_32F, CUDA_C_32F)
VGPU_ROT(Cs, cuComplex, float, float, CUDA_C_32F, CUDA_R_32F, CUDA_R_32F, CUDA_C_32F)
VGPU_ROT(Z, cuDoubleComplex, double, cuDoubleComplex, CUDA_C_64F, CUDA_R_64F, CUDA_C_64F, CUDA_C_64F)
VGPU_ROT(Zd, cuDoubleComplex, double, double, CUDA_C_64F, CUDA_R_64F, CUDA_R_64F, CUDA_C_64F)
#undef VGPU_ROT

VGPU_EXPORT cublasStatus_t cublasSrotg_v2(cublasHandle_t h, float* a, float* b, float* c, float* s) {
  return rotg_ex(h, a, b, CUDA_R_32F, c, s, CUDA_R_32F, CUDA_R_32F);
}
VGPU_EXPORT cublasStatus_t cublasDrotg_v2(cublasHandle_t h, double* a, double* b, double* c, double* s) {
  return rotg_ex(h, a, b, CUDA_R_64F, c, s, CUDA_R_64F, CUDA_R_64F);
}
VGPU_EXPORT cublasStatus_t cublasCrotg_v2(cublasHandle_t h, cuComplex* a, cuComplex* b, float* c, cuComplex* s) {
  return rotg_complex(h, a, b, c, s, CUDA_C_32F);
}
VGPU_EXPORT cublasStatus_t cublasZrotg_v2(cublasHandle_t h, cuDoubleComplex* a, cuDoubleComplex* b, double* c,
                                          cuDoubleComplex* s) {
  return rotg_complex(h, a, b, c, s, CUDA_C_64F);
}
VGPU_EXPORT cublasStatus_t cublasSrotm_v2(cublasHandle_t h, int n, float* x, int incx, float* y, int incy,
                                          const float* param) {
  return rotm_ex(h, n, x, CUDA_R_32F, incx, y, CUDA_R_32F, incy, param, CUDA_R_32F, CUDA_R_32F);
}
VGPU_EXPORT cublasStatus_t cublasDrotm_v2(cublasHandle_t h, int n, double* x, int incx, double* y, int incy,
                                          const double* param) {
  return rotm_ex(h, n, x, CUDA_R_64F, incx, y, CUDA_R_64F, incy, param, CUDA_R_64F, CUDA_R_64F);
}
VGPU_EXPORT cublasStatus_t cublasSrotm_v2_64(cublasHandle_t h, int64_t n, float* x, int64_t incx, float* y,
                                             int64_t incy, const float* param) {
  return cublasRotmEx_64(h, n, x, CUDA_R_32F, incx, y, CUDA_R_32F, incy, param, CUDA_R_32F, CUDA_R_32F);
}
VGPU_EXPORT cublasStatus_t cublasDrotm_v2_64(cublasHandle_t h, int64_t n, double* x, int64_t incx, double* y,
                                             int64_t incy, const double* param) {
  return cublasRotmEx_64(h, n, x, CUDA_R_64F, incx, y, CUDA_R_64F, incy, param, CUDA_R_64F, CUDA_R_64F);
}
VGPU_EXPORT cublasStatus_t cublasSrotmg_v2(cublasHandle_t h, float* d1, float* d2, float* x1, const float* y1,
                                           float* param) {
  return rotmg_ex(h, d1, CUDA_R_32F, d2, CUDA_R_32F, x1, CUDA_R_32F, y1, CUDA_R_32F, param, CUDA_R_32F, CUDA_R_32F);
}
VGPU_EXPORT cublasStatus_t cublasDrotmg_v2(cublasHandle_t h, double* d1, double* d2, double* x1, const double* y1,
                                           double* param) {
  return rotmg_ex(h, d1, CUDA_R_64F, d2, CUDA_R_64F, x1, CUDA_R_64F, y1, CUDA_R_64F, param, CUDA_R_64F, CUDA_R_64F);
}

/* ---- the complex GEMM-family Ex routines, and SgemmEx ----
   CgemmEx and Cgemm3mEx take A and B both C_32F or both C_8I (pairs of
   int8), and C C_32F; CherkEx, Cherk3mEx, CsyrkEx and Csyrk3mEx the same A
   and C C_32F. SgemmEx takes the real rows of GemmEx's table under
   COMPUTE_32F. Those are the combinations an RTX 3060 accepts; anything else
   is NOT_SUPPORTED. The 3m forms (Gauss's three-multiplication product) give
   the same product as the others here, computed directly: on the card they
   differ from it in the last bits, and are no more exact. The arithmetic is
   the complex routines', in double and rounded to single at the end. */
namespace {

// A C_32F or C_8I matrix, as cd.
CMat cex_mat(const void* p, cudaDataType t, int ld, int cols, int rows) {
  if (t == CUDA_C_32F) return load_mat(static_cast<const cuComplex*>(p), ld, cols, rows);
  const auto raw = fetch<int8_t>(p, 2 * extent(ld, cols, rows));
  CMat m{std::vector<cd>((size_t)ld * std::max(cols, 0)), ld};
  for (size_t i = 0; 2 * i < raw.size(); ++i) m.v[i] = cd(raw[2 * i], raw[2 * i + 1]);
  return m;
}
bool cex_type(cudaDataType a, cudaDataType c) { return (a == CUDA_C_32F || a == CUDA_C_8I) && c == CUDA_C_32F; }

cublasStatus_t gemm_ex_complex(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k,
                               const void* alpha, const void* A, cudaDataType Atype, int lda, const void* B, int ldb,
                               const void* beta, void* C, cudaDataType Ctype, int ldc) {
  if (Ctype == CUDA_C_64F)
    return cgemm<cuDoubleComplex>(h, ta, tb, m, n, k, static_cast<const cuDoubleComplex*>(alpha),
                                  static_cast<const cuDoubleComplex*>(A), lda, static_cast<const cuDoubleComplex*>(B),
                                  ldb, static_cast<const cuDoubleComplex*>(beta), static_cast<cuDoubleComplex*>(C), ldc);
  if (Atype == CUDA_C_32F)
    return cgemm<cuComplex>(h, ta, tb, m, n, k, static_cast<const cuComplex*>(alpha), static_cast<const cuComplex*>(A),
                            lda, static_cast<const cuComplex*>(B), ldb, static_cast<const cuComplex*>(beta),
                            static_cast<cuComplex*>(C), ldc);
  if (!valid(h)) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (!gemm_args_ok(ta, tb, m, n, k, lda, ldb, ldc)) return CUBLAS_STATUS_INVALID_VALUE;
  if (m == 0 || n == 0) return CUBLAS_STATUS_SUCCESS;
  return run_c(h, alpha, beta, sizeof(cuComplex), [=](const void* al, const void* be) {
    const auto oa = to_op(ta), ob = to_op(tb);
    const CMat a = cex_mat(A, Atype, lda, oa == vgpu_la::Op::N ? k : m, oa == vgpu_la::Op::N ? m : k);
    const CMat b = cex_mat(B, Atype, ldb, ob == vgpu_la::Op::N ? n : k, ob == vgpu_la::Op::N ? k : n);
    CMat c = load_mat(static_cast<cuComplex*>(C), ldc, n, m);
    cgemm_mat(oa, ob, m, n, k, cscalar(h, static_cast<const cuComplex*>(al)), a, b,
              cscalar(h, static_cast<const cuComplex*>(be)), c);
    store_mat(static_cast<cuComplex*>(C), c, n, m);
    return CUBLAS_STATUS_SUCCESS;
  });
}

cublasStatus_t cgemm_ex(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k,
                        const cuComplex* alpha, const void* A, cudaDataType Atype, int lda, const void* B,
                        cudaDataType Btype, int ldb, const cuComplex* beta, void* C, cudaDataType Ctype, int ldc) {
  if (!valid(h)) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (!cex_type(Atype, Ctype) || Btype != Atype) return CUBLAS_STATUS_NOT_SUPPORTED;
  return cublasGemmEx(h, ta, tb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc,
                      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

// herk (alpha, beta float) and syrk (cuComplex) over a C_32F or C_8I A.
template <class S>
cublasStatus_t cherk_ex(cublasHandle_t h, bool hermitian, cublasFillMode_t uplo, cublasOperation_t t, int n, int k,
                        const S* alpha, const void* A, cudaDataType Atype, int lda, const S* beta, void* C,
                        cudaDataType Ctype, int ldc) {
  if (!valid(h)) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (!cex_type(Atype, Ctype)) return CUBLAS_STATUS_NOT_SUPPORTED;
  if (Atype == CUDA_C_32F)
    return cherk<cuComplex, S>(h, hermitian, uplo, t, n, k, alpha, static_cast<const cuComplex*>(A), lda, beta,
                               static_cast<cuComplex*>(C), ldc);
  if (!herk_args_ok(uplo, t, n, k, lda, ldc)) return CUBLAS_STATUS_INVALID_VALUE;
  if (n == 0) return CUBLAS_STATUS_SUCCESS;
  return run_c(h, alpha, beta, sizeof(S), [=](const void* al, const void* be) {
    auto sc = [&](const void* p) -> cd {
      if constexpr (std::is_same_v<S, cuComplex>) return cscalar(h, static_cast<const cuComplex*>(p));
      else return p ? cd(scalar(h, static_cast<const float*>(p))) : cd(0);
    };
    const bool trans = t != CUBLAS_OP_N;
    const CMat a = cex_mat(A, Atype, lda, trans ? n : k, trans ? k : n);
    CMat c = load_mat(static_cast<cuComplex*>(C), ldc, n, n);
    cherk_mat(hermitian, uplo == CUBLAS_FILL_MODE_LOWER, trans, n, k, sc(al), a, sc(be), c);
    store_mat(static_cast<cuComplex*>(C), c, n, n);
    return CUBLAS_STATUS_SUCCESS;
  });
}

}  // namespace

VGPU_EXPORT cublasStatus_t cublasSgemmEx(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m, int n,
                                         int k, const float* alpha, const void* A, cudaDataType Atype, int lda,
                                         const void* B, cudaDataType Btype, int ldb, const float* beta, void* C,
                                         cudaDataType Ctype, int ldc) {
  if (!valid(h)) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (Ctype == CUDA_C_32F || Ctype == CUDA_C_64F || Ctype == CUDA_R_32I) return CUBLAS_STATUS_NOT_SUPPORTED;
  return cublasGemmEx(h, ta, tb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc,
                      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

#define VGPU_CGEMM_EX(NAME)                                                                                        \
  VGPU_EXPORT cublasStatus_t NAME(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k, \
                                  const cuComplex* alpha, const void* A, cudaDataType Atype, int lda, const void* B, \
                                  cudaDataType Btype, int ldb, const cuComplex* beta, void* C, cudaDataType Ctype,   \
                                  int ldc) {                                                                         \
    return cgemm_ex(h, ta, tb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc);                   \
  }                                                                                                                  \
  VGPU_EXPORT cublasStatus_t NAME##_64(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int64_t m,      \
                                       int64_t n, int64_t k, const cuComplex* alpha, const void* A,                  \
                                       cudaDataType Atype, int64_t lda, const void* B, cudaDataType Btype,           \
                                       int64_t ldb, const cuComplex* beta, void* C, cudaDataType Ctype,              \
                                       int64_t ldc) {                                                                \
    for (int64_t v : {m, n, k, lda, ldb, ldc})                                                                       \
      if (!fits_int(v)) return CUBLAS_STATUS_NOT_SUPPORTED;                                                          \
    return cgemm_ex(h, ta, tb, (int)m, (int)n, (int)k, alpha, A, Atype, (int)lda, B, Btype, (int)ldb, beta, C,       \
                    Ctype, (int)ldc);                                                                                \
  }
VGPU_CGEMM_EX(cublasCgemmEx)
VGPU_CGEMM_EX(cublasCgemm3mEx)
#undef VGPU_CGEMM_EX

#define VGPU_CHERK_EX(NAME, S, HERM)                                                                               \
  VGPU_EXPORT cublasStatus_t NAME(cublasHandle_t h, cublasFillMode_t uplo, cublasOperation_t t, int n, int k,        \
                                  const S* alpha, const void* A, cudaDataType Atype, int lda, const S* beta, void* C, \
                                  cudaDataType Ctype, int ldc) {                                                     \
    return cherk_ex<S>(h, HERM, uplo, t, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc);                           \
  }                                                                                                                  \
  VGPU_EXPORT cublasStatus_t NAME##_64(cublasHandle_t h, cublasFillMode_t uplo, cublasOperation_t t, int64_t n,      \
                                       int64_t k, const S* alpha, const void* A, cudaDataType Atype, int64_t lda,    \
                                       const S* beta, void* C, cudaDataType Ctype, int64_t ldc) {                    \
    for (int64_t v : {n, k, lda, ldc})                                                                               \
      if (!fits_int(v)) return CUBLAS_STATUS_NOT_SUPPORTED;                                                          \
    return cherk_ex<S>(h, HERM, uplo, t, (int)n, (int)k, alpha, A, Atype, (int)lda, beta, C, Ctype, (int)ldc);       \
  }
VGPU_CHERK_EX(cublasCherkEx, float, true)
VGPU_CHERK_EX(cublasCherk3mEx, float, true)
VGPU_CHERK_EX(cublasCsyrkEx, cuComplex, false)
VGPU_CHERK_EX(cublasCsyrk3mEx, cuComplex, false)
#undef VGPU_CHERK_EX

/* ---- the remaining typed level-1 routines ----
   copy and swap in single and double precision, i?amin, the complex i?amax
   and i?amin, and the complex sums of magnitudes and norms (scasum, dzasum,
   scnrm2, dznrm2), with their _64 forms: each is the Ex routine above at its
   type, so the conventions are the same (BLAS's walk for a negative
   increment; nothing for an empty vector, or for a non-positive increment in
   a reduction). As on the card, none of them refuses a size or an increment. */
#define VGPU_LEVEL1(P, U, T, XT)                                                                                   \
  VGPU_EXPORT cublasStatus_t cublasI##P##amin_v2(cublasHandle_t h, int n, const T* x, int incx, int* result) {    \
    return iamax_ex(h, n, x, XT, incx, result, true);                                                             \
  }                                                                                                               \
  VGPU_EXPORT cublasStatus_t cublasI##P##amin_v2_64(cublasHandle_t h, int64_t n, const T* x, int64_t incx,        \
                                                    int64_t* result) {                                            \
    if (!fits_int(n) || !fits_int(incx)) return CUBLAS_STATUS_NOT_SUPPORTED;                                      \
    return iamax_ex(h, (int)n, x, XT, (int)incx, result, true);                                                   \
  }                                                                                                               \
  VGPU_EXPORT cublasStatus_t cublas##U##copy_v2_64(cublasHandle_t h, int64_t n, const T* x, int64_t incx, T* y,   \
                                                   int64_t incy) {                                                \
    if (!fits_int(n) || !fits_int(incx) || !fits_int(incy)) return CUBLAS_STATUS_NOT_SUPPORTED;                   \
    return copy_ex(h, (int)n, x, XT, (int)incx, y, XT, (int)incy, false);                                         \
  }                                                                                                               \
  VGPU_EXPORT cublasStatus_t cublas##U##swap_v2_64(cublasHandle_t h, int64_t n, T* x, int64_t incx, T* y,         \
                                                   int64_t incy) {                                                \
    if (!fits_int(n) || !fits_int(incx) || !fits_int(incy)) return CUBLAS_STATUS_NOT_SUPPORTED;                   \
    return copy_ex(h, (int)n, x, XT, (int)incx, y, XT, (int)incy, true);                                          \
  }
VGPU_LEVEL1(s, S, float, CUDA_R_32F)
VGPU_LEVEL1(d, D, double, CUDA_R_64F)
VGPU_LEVEL1(c, C, cuComplex, CUDA_C_32F)
VGPU_LEVEL1(z, Z, cuDoubleComplex, CUDA_C_64F)
#undef VGPU_LEVEL1

#define VGPU_REAL_COPY(P, T, XT)                                                                                  \
  VGPU_EXPORT cublasStatus_t cublas##P##copy_v2(cublasHandle_t h, int n, const T* x, int incx, T* y, int incy) {  \
    return copy_ex(h, n, x, XT, incx, y, XT, incy, false);                                                        \
  }                                                                                                               \
  VGPU_EXPORT cublasStatus_t cublas##P##swap_v2(cublasHandle_t h, int n, T* x, int incx, T* y, int incy) {        \
    return copy_ex(h, n, x, XT, incx, y, XT, incy, true);                                                         \
  }
VGPU_REAL_COPY(S, float, CUDA_R_32F)
VGPU_REAL_COPY(D, double, CUDA_R_64F)
#undef VGPU_REAL_COPY

#define VGPU_COMPLEX_LEVEL1(P, PR, T, XT, R, RT)                                                                  \
  VGPU_EXPORT cublasStatus_t cublasI##P##amax_v2(cublasHandle_t h, int n, const T* x, int incx, int* result) {    \
    return iamax_ex(h, n, x, XT, incx, result, false);                                                            \
  }                                                                                                               \
  VGPU_EXPORT cublasStatus_t cublasI##P##amax_v2_64(cublasHandle_t h, int64_t n, const T* x, int64_t incx,        \
                                                    int64_t* result) {                                            \
    if (!fits_int(n) || !fits_int(incx)) return CUBLAS_STATUS_NOT_SUPPORTED;                                      \
    return iamax_ex(h, (int)n, x, XT, (int)incx, result, false);                                                  \
  }                                                                                                               \
  VGPU_EXPORT cublasStatus_t cublas##PR##asum_v2(cublasHandle_t h, int n, const T* x, int incx, R* result) {      \
    return asum_ex(h, n, x, XT, incx, result, RT, XT);                                                            \
  }                                                                                                               \
  VGPU_EXPORT cublasStatus_t cublas##PR##asum_v2_64(cublasHandle_t h, int64_t n, const T* x, int64_t incx,        \
                                                    R* result) {                                                  \
    if (!fits_int(n) || !fits_int(incx)) return CUBLAS_STATUS_NOT_SUPPORTED;                                      \
    return asum_ex(h, (int)n, x, XT, (int)incx, result, RT, XT);                                                  \
  }                                                                                                               \
  VGPU_EXPORT cublasStatus_t cublas##PR##nrm2_v2(cublasHandle_t h, int n, const T* x, int incx, R* result) {      \
    return nrm2_ex(h, n, x, XT, incx, result, RT, RT);                                                            \
  }                                                                                                               \
  VGPU_EXPORT cublasStatus_t cublas##PR##nrm2_v2_64(cublasHandle_t h, int64_t n, const T* x, int64_t incx,        \
                                                    R* result) {                                                  \
    if (!fits_int(n) || !fits_int(incx)) return CUBLAS_STATUS_NOT_SUPPORTED;                                      \
    return nrm2_ex(h, (int)n, x, XT, (int)incx, result, RT, RT);                                                  \
  }
VGPU_COMPLEX_LEVEL1(c, Sc, cuComplex, CUDA_C_32F, float, CUDA_R_32F)
VGPU_COMPLEX_LEVEL1(z, Dz, cuDoubleComplex, CUDA_C_64F, double, CUDA_R_64F)
#undef VGPU_COMPLEX_LEVEL1

// The batched GEMVs, getri/matinv, syrkx/herkx, gemm3m, the batched Hgemms and
// the complex dgmm.
#include "cublas_batched.inc"
