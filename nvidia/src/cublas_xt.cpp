// cuBLASXt: the multi-GPU, host-memory level-3 interface NVIDIA ships inside
// libcublas. Its operands may be in host or device memory (any memory
// cudaMemcpyDefault can read), its scalars on the host.
//
// How the work is spread: each call stages its operands on the host, then
//   - gemm cuts C into blockDim x blockDim tiles and hands them round-robin to
//     the selected devices, each tile one cublas<t>gemm on its device with
//     that tile's rows of op(A) and columns of op(B) -- run one after another,
//     since the simulator's devices share this process's host;
//   - every other routine runs whole, as one cuBLAS call, on the first
//     selected device: a correct answer, without the tiling across devices.
// The arithmetic is cuBLAS's own (cublas_api.cpp), so the answers are the
// ones the single-GPU routines give. GEMM's CPU share is real: with a CPU
// routine set (cublasXtSetCpuRoutine) and a ratio above 0 (cublasXtSetCpuRatio)
// the routine is called, Fortran-style (all arguments pointers), on the last
// floor(ratio * d) rows of C if m > n, else columns, d being the longer of
// m and n (ties go to n; the product is in float, so 0.57 * 1000 is 570).
// Measured on an RTX 3060, cuBLAS 13.0: the pointers handed over are the
// caller's own, offset into A, B and C. The routine is called even for 0 rows,
// not at all for ratio 0 or without a routine, and no other routine ever calls
// it. The card divides by zero for a ratio of 1 or more; here the GPUs get
// the remainder, which may be nothing.
//
// The answers to bad arguments are an RTX 3060's (cuBLAS 13.0), measured.
// Its getters (cublasXtGetBlockDim, cublasXtGetPinningMemMode,
// cublasXtMaxBoards, cublasXtGetNumBoards) were measured to return SUCCESS
// and either write nothing or the value from before the latest set; these
// write the current value, as documented.
#include <cublasXt.h>
#include <cuda_runtime_api.h>

#include "enum_value.hpp"

#include <algorithm>
#include <complex>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <set>
#include <type_traits>
#include <vector>

#define VGPU_EXPORT extern "C" __attribute__((visibility("default")))

namespace {

struct XtHandle {
  std::vector<int> devices;
  std::vector<cublasHandle_t> handles;   // one per selected device
  bool select_tried = false;             // a handle takes one cublasXtDeviceSelect
  int block_dim = 1024;
  cublasXtPinnedMemMode_t pinning = CUBLASXT_PINNING_DISABLED;
  float cpu_ratio[CUBLASXT_ROUTINE_MAX][4] = {};
  void* cpu_routine[CUBLASXT_ROUTINE_MAX][4] = {};
};

std::mutex g_mu;
std::set<XtHandle*> g_live;

XtHandle* get(cublasXtHandle_t h) {
  std::lock_guard<std::mutex> lock(g_mu);
  auto* x = reinterpret_cast<XtHandle*>(h);
  return h && g_live.count(x) ? x : nullptr;
}

void release(XtHandle* x) {
  int cur = 0;
  cudaGetDevice(&cur);
  for (size_t i = 0; i < x->handles.size(); ++i) {
    cudaSetDevice(x->devices[i]);
    cublasDestroy(x->handles[i]);
  }
  cudaSetDevice(cur);
  x->handles.clear();
  x->devices.clear();
}

bool fits(size_t v) { return v <= (size_t)INT32_MAX; }

// Elements a column-major matrix of `rows` x `cols` occupies with leading
// dimension ld: no padding after its last column.
size_t extent(size_t ld, size_t cols, size_t rows) { return rows && cols ? ld * (cols - 1) + rows : 0; }

// A host copy of an operand wherever it lives, and the copy back.
template <class T>
std::vector<T> stage(const T* p, size_t n) {
  std::vector<T> h(n);
  if (n) cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDefault);
  return h;
}
template <class T>
void unstage(T* p, const std::vector<T>& h) {
  if (!h.empty()) cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyDefault);
}

// Device memory on the current device holding `h`.
template <class T>
struct DevBuf {
  T* p = nullptr;
  explicit DevBuf(const std::vector<T>& h) {
    cudaMalloc(reinterpret_cast<void**>(&p), std::max<size_t>(1, h.size()) * sizeof(T));
    if (!h.empty()) cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
  }
  ~DevBuf() { cudaFree(p); }
  std::vector<T> read(size_t n) const {
    std::vector<T> h(n);
    if (n) cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost);
    return h;
  }
};

// Runs `fn(handle)` on device `i` of the selection, the caller's device kept.
template <class Fn>
cublasStatus_t on_device(XtHandle* x, size_t i, Fn fn) {
  int cur = 0;
  cudaGetDevice(&cur);
  cudaSetDevice(x->devices[i]);
  const cublasStatus_t st = fn(x->handles[i]);
  cudaSetDevice(cur);
  return st;
}

// The enum arguments are read as integers first (enum_value.hpp): a value
// the enum does not declare is refused before it is ever used as the enum.
bool known_op(int t) { return t == CUBLAS_OP_N || t == CUBLAS_OP_T || t == CUBLAS_OP_C; }
// Declared values of each enum (the routines refuse some of these themselves).
bool fill_v(int v) { return v == CUBLAS_FILL_MODE_LOWER || v == CUBLAS_FILL_MODE_UPPER || v == CUBLAS_FILL_MODE_FULL; }
bool op_v(int v) { return v >= CUBLAS_OP_N && v <= CUBLAS_OP_CONJG; }
bool side_v(int v) { return v == CUBLAS_SIDE_LEFT || v == CUBLAS_SIDE_RIGHT; }
bool diag_v(int v) { return v == CUBLAS_DIAG_NON_UNIT || v == CUBLAS_DIAG_UNIT; }
// A selected handle, then enum arguments that are values of their enums
// (INVALID_VALUE otherwise, as the card answers a bad fill mode or side).
cublasStatus_t precheck(cublasXtHandle_t h, bool enums_ok) {
  XtHandle* x = get(h);
  if (!x || x->devices.empty()) return CUBLAS_STATUS_NOT_INITIALIZED;
  return enums_ok ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_INVALID_VALUE;
}

// The routines by element type, onto the single-GPU API.
template <class T> struct Blas;
#define VGPU_XT_BLAS(T, R, P)                                                                                     \
  template <> struct Blas<T> {                                                                                    \
    static constexpr auto gemm = cublas##P##gemm_v2;                                                             \
    static constexpr auto syrk = cublas##P##syrk_v2;                                                             \
    static constexpr auto syr2k = cublas##P##syr2k_v2;                                                           \
    static constexpr auto syrkx = cublas##P##syrkx;                                                              \
    static constexpr auto symm = cublas##P##symm_v2;                                                             \
    static constexpr auto trsm = cublas##P##trsm_v2;                                                             \
    static constexpr auto trmm = cublas##P##trmm_v2;                                                             \
  };
VGPU_XT_BLAS(float, float, S)
VGPU_XT_BLAS(double, double, D)
VGPU_XT_BLAS(cuComplex, float, C)
VGPU_XT_BLAS(cuDoubleComplex, double, Z)
#undef VGPU_XT_BLAS

template <class T> constexpr int xt_type_index() {
  return std::is_same_v<T, float> ? 0 : std::is_same_v<T, double> ? 1 : std::is_same_v<T, cuComplex> ? 2 : 3;
}

// C = alpha op(A) op(B) + beta C, tile by tile over the selected devices.
template <class T>
cublasStatus_t xt_gemm(cublasXtHandle_t h, cublasOperation_t ta, cublasOperation_t tb, size_t m, size_t n, size_t k,
                       const T* alpha, const T* A, size_t lda, const T* B, size_t ldb, const T* beta, T* C,
                       size_t ldc) {
  XtHandle* x = get(h);
  if (!x || x->devices.empty()) return CUBLAS_STATUS_NOT_INITIALIZED;
  for (size_t v : {m, n, k, lda, ldb, ldc})
    if (!fits(v)) return CUBLAS_STATUS_NOT_SUPPORTED;
  const int tav = enum_value(ta), tbv = enum_value(tb);
  if (!known_op(tav) || !known_op(tbv)) return CUBLAS_STATUS_INVALID_VALUE;
  const size_t arows = tav == CUBLAS_OP_N ? m : k, brows = tbv == CUBLAS_OP_N ? k : n;
  if (lda < std::max<size_t>(1, arows) || ldb < std::max<size_t>(1, brows) || ldc < std::max<size_t>(1, m))
    return CUBLAS_STATUS_INVALID_VALUE;
  if (!alpha || !beta) return CUBLAS_STATUS_INVALID_VALUE;
  if (!m || !n) return CUBLAS_STATUS_SUCCESS;
  // The CPU's share: the tail of the longer dimension of C goes to the
  // caller's routine, on the caller's memory; the devices take the head.
  if (const int ty = xt_type_index<T>(); x->cpu_routine[CUBLASXT_GEMM][ty] && x->cpu_ratio[CUBLASXT_GEMM][ty] > 0.0f) {
    using Fn = void (*)(const char*, const char*, const int*, const int*, const int*, const T*, const T*, const int*,
                        const T*, const int*, const T*, T*, const int*);
    const float ratio = std::min(x->cpu_ratio[CUBLASXT_GEMM][ty], 1.0f);
    const bool split_n = n >= m;
    const size_t whole = split_n ? n : m;
    const size_t cpu = std::min(whole, (size_t)(ratio * (float)whole));
    const size_t head = whole - cpu;
    const char opc[3] = {'N', 'T', 'C'};
    const char ca = opc[tav], cb = opc[tbv];
    int im = (int)(split_n ? m : cpu), in = (int)(split_n ? cpu : n), ik = (int)k, ilda = (int)lda, ildb = (int)ldb,
        ildc = (int)ldc;
    const T* a_cpu = split_n ? A : (tav == CUBLAS_OP_N ? A + head : A + head * lda);
    const T* b_cpu = split_n ? (tbv == CUBLAS_OP_N ? B + head * ldb : B + head) : B;
    T* c_cpu = C + (split_n ? head * ldc : head);
    reinterpret_cast<Fn>(x->cpu_routine[CUBLASXT_GEMM][ty])(&ca, &cb, &im, &in, &ik, alpha, a_cpu, &ilda, b_cpu, &ildb,
                                                            beta, c_cpu, &ildc);
    if (split_n) n = head; else m = head;
    if (!m || !n) return CUBLAS_STATUS_SUCCESS;
  }
  const auto hA = stage(A, ta == CUBLAS_OP_N ? extent(lda, k, m) : extent(lda, m, k));
  const auto hB = stage(B, tb == CUBLAS_OP_N ? extent(ldb, n, k) : extent(ldb, k, n));
  auto hC = stage(C, extent(ldc, n, m));
  const size_t bd = (size_t)x->block_dim;
  size_t tile = 0;
  for (size_t j0 = 0; j0 < n; j0 += bd)
    for (size_t i0 = 0; i0 < m; i0 += bd, ++tile) {
      const size_t mb = std::min(bd, m - i0), nb = std::min(bd, n - j0);
      // op(A)'s rows i0.. and op(B)'s columns j0.., in their stored layout.
      const size_t ar = ta == CUBLAS_OP_N ? mb : k, ac = ta == CUBLAS_OP_N ? k : mb;
      const size_t br = tb == CUBLAS_OP_N ? k : nb, bc = tb == CUBLAS_OP_N ? nb : k;
      std::vector<T> ta_(std::max<size_t>(1, ar * ac)), tb_(std::max<size_t>(1, br * bc)), tc(mb * nb);
      for (size_t c = 0; c < ac; ++c)
        for (size_t r = 0; r < ar; ++r)
          ta_[c * ar + r] = ta == CUBLAS_OP_N ? hA[c * lda + i0 + r] : hA[(i0 + c) * lda + r];
      for (size_t c = 0; c < bc; ++c)
        for (size_t r = 0; r < br; ++r)
          tb_[c * br + r] = tb == CUBLAS_OP_N ? hB[(j0 + c) * ldb + r] : hB[c * ldb + j0 + r];
      for (size_t c = 0; c < nb; ++c)
        for (size_t r = 0; r < mb; ++r) tc[c * mb + r] = hC[(j0 + c) * ldc + i0 + r];
      const cublasStatus_t st = on_device(x, tile % x->devices.size(), [&](cublasHandle_t bh) {
        DevBuf<T> da(ta_), db(tb_), dc(tc);
        const cublasStatus_t s =
            Blas<T>::gemm(bh, ta, tb, (int)mb, (int)nb, (int)k, alpha, da.p, (int)std::max<size_t>(1, ar), db.p,
                          (int)std::max<size_t>(1, br), beta, dc.p, (int)mb);
        if (s == CUBLAS_STATUS_SUCCESS) tc = dc.read(tc.size());
        return s;
      });
      if (st != CUBLAS_STATUS_SUCCESS) return st;
      for (size_t c = 0; c < nb; ++c)
        for (size_t r = 0; r < mb; ++r) hC[(j0 + c) * ldc + i0 + r] = tc[c * mb + r];
    }
  unstage(C, hC);
  return CUBLAS_STATUS_SUCCESS;
}

// One whole-matrix call on the first selected device: every operand staged
// up by its extent, the output read back, the arguments the cuBLAS routine's
// to check.
struct Operand {
  const void* src;   // where the caller keeps it
  void* dst;         // where the result goes back to, or null for an input
  size_t bytes;
};

template <class Fn>
cublasStatus_t xt_whole(cublasXtHandle_t h, std::initializer_list<size_t> dims, std::vector<Operand> ops, Fn fn) {
  XtHandle* x = get(h);
  if (!x || x->devices.empty()) return CUBLAS_STATUS_NOT_INITIALIZED;
  for (size_t v : dims)
    if (!fits(v)) return CUBLAS_STATUS_NOT_SUPPORTED;
  return on_device(x, 0, [&](cublasHandle_t bh) {
    std::vector<void*> dev(ops.size(), nullptr);
    std::vector<uint8_t> tmp;
    for (size_t i = 0; i < ops.size(); ++i) {
      cudaMalloc(&dev[i], std::max<size_t>(1, ops[i].bytes));
      tmp.resize(ops[i].bytes);
      if (ops[i].bytes && ops[i].src) {
        cudaMemcpy(tmp.data(), ops[i].src, ops[i].bytes, cudaMemcpyDefault);
        cudaMemcpy(dev[i], tmp.data(), ops[i].bytes, cudaMemcpyHostToDevice);
      }
    }
    const cublasStatus_t st = fn(bh, dev);
    for (size_t i = 0; i < ops.size(); ++i) {
      if (st == CUBLAS_STATUS_SUCCESS && ops[i].dst && ops[i].bytes) {
        tmp.resize(ops[i].bytes);
        cudaMemcpy(tmp.data(), dev[i], ops[i].bytes, cudaMemcpyDeviceToHost);
        cudaMemcpy(ops[i].dst, tmp.data(), ops[i].bytes, cudaMemcpyDefault);
      }
      cudaFree(dev[i]);
    }
    return st;
  });
}

// The bytes an operand of `rows` x `cols` (column-major, leading dimension
// ld) spans; nothing for a leading dimension too small, which the cuBLAS
// routine then refuses before reading anything.
template <class T>
size_t span_bytes(size_t ld, size_t cols, size_t rows) {
  return ld < rows ? 0 : extent(ld, cols, rows) * sizeof(T);
}

}  // namespace

/* ---- the handle ---- */

VGPU_EXPORT cublasStatus_t cublasXtCreate(cublasXtHandle_t* handle) {
  if (!handle) return CUBLAS_STATUS_INVALID_VALUE;
  auto* x = new XtHandle();
  {
    std::lock_guard<std::mutex> lock(g_mu);
    g_live.insert(x);
  }
  *handle = reinterpret_cast<cublasXtHandle_t>(x);
  return CUBLAS_STATUS_SUCCESS;
}
VGPU_EXPORT cublasStatus_t cublasXtDestroy(cublasXtHandle_t handle) {
  XtHandle* x = get(handle);
  if (!x) return CUBLAS_STATUS_NOT_INITIALIZED;
  release(x);
  {
    std::lock_guard<std::mutex> lock(g_mu);
    g_live.erase(x);
  }
  delete x;
  return CUBLAS_STATUS_SUCCESS;
}
// Every simulated device is its own board.
VGPU_EXPORT cublasStatus_t cublasXtGetNumBoards(int nbDevices, int deviceId[], int* nbBoards) {
  if (!nbBoards || nbDevices < 0 || (nbDevices && !deviceId)) return CUBLAS_STATUS_INVALID_VALUE;
  std::set<int> boards(deviceId, deviceId + nbDevices);
  *nbBoards = (int)boards.size();
  return CUBLAS_STATUS_SUCCESS;
}
VGPU_EXPORT cublasStatus_t cublasXtMaxBoards(int* nbGpuBoards) {
  if (!nbGpuBoards) return CUBLAS_STATUS_INVALID_VALUE;
  *nbGpuBoards = 16;
  return CUBLAS_STATUS_SUCCESS;
}
// A handle takes one selection: the card refuses (INVALID_VALUE) a second
// cublasXtDeviceSelect, even after a first that failed. An empty list is
// INVALID_VALUE, a device that does not exist INTERNAL_ERROR.
VGPU_EXPORT cublasStatus_t cublasXtDeviceSelect(cublasXtHandle_t handle, int nbDevices, int deviceId[]) {
  XtHandle* x = get(handle);
  if (!x) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (x->select_tried) return CUBLAS_STATUS_INVALID_VALUE;
  x->select_tried = true;
  if (nbDevices <= 0 || !deviceId) return CUBLAS_STATUS_INVALID_VALUE;
  int count = 0;
  cudaGetDeviceCount(&count);
  for (int i = 0; i < nbDevices; ++i)
    if (deviceId[i] < 0 || deviceId[i] >= count) return CUBLAS_STATUS_INTERNAL_ERROR;
  int cur = 0;
  cudaGetDevice(&cur);
  for (int i = 0; i < nbDevices; ++i) {
    cudaSetDevice(deviceId[i]);
    cublasHandle_t bh = nullptr;
    if (cublasCreate(&bh) != CUBLAS_STATUS_SUCCESS) {
      cudaSetDevice(cur);
      release(x);
      return CUBLAS_STATUS_ALLOC_FAILED;
    }
    x->devices.push_back(deviceId[i]);
    x->handles.push_back(bh);
  }
  cudaSetDevice(cur);
  return CUBLAS_STATUS_SUCCESS;
}
VGPU_EXPORT cublasStatus_t cublasXtSetBlockDim(cublasXtHandle_t handle, int blockDim) {
  XtHandle* x = get(handle);
  if (!x) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (blockDim <= 0) return CUBLAS_STATUS_INVALID_VALUE;
  x->block_dim = blockDim;
  return CUBLAS_STATUS_SUCCESS;
}
VGPU_EXPORT cublasStatus_t cublasXtGetBlockDim(cublasXtHandle_t handle, int* blockDim) {
  XtHandle* x = get(handle);
  if (!x) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (!blockDim) return CUBLAS_STATUS_INVALID_VALUE;
  *blockDim = x->block_dim;
  return CUBLAS_STATUS_SUCCESS;
}
VGPU_EXPORT cublasStatus_t cublasXtGetPinningMemMode(cublasXtHandle_t handle, cublasXtPinnedMemMode_t* mode) {
  XtHandle* x = get(handle);
  if (!x) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (!mode) return CUBLAS_STATUS_INVALID_VALUE;
  *mode = x->pinning;
  return CUBLAS_STATUS_SUCCESS;
}
VGPU_EXPORT cublasStatus_t cublasXtSetPinningMemMode(cublasXtHandle_t handle, cublasXtPinnedMemMode_t mode) {
  XtHandle* x = get(handle);
  if (!x) return CUBLAS_STATUS_NOT_INITIALIZED;
  const int v = enum_value(mode);
  if (v != CUBLASXT_PINNING_DISABLED && v != CUBLASXT_PINNING_ENABLED) return CUBLAS_STATUS_INVALID_VALUE;
  x->pinning = (cublasXtPinnedMemMode_t)v;
  return CUBLAS_STATUS_SUCCESS;
}
// The card takes a CPU routine and ratio for every type of GEMM, and for the
// complex types of herk, hemm, her2k and herkx (which then never call it);
// every other routine and type answers NOT_SUPPORTED (measured).
static bool cpu_share_known(int op, int ty) {
  if (op == CUBLASXT_GEMM) return true;
  return (op == CUBLASXT_HERK || op == CUBLASXT_HEMM || op == CUBLASXT_HER2K || op == CUBLASXT_HERKX) && ty >= 2;
}
VGPU_EXPORT cublasStatus_t cublasXtSetCpuRoutine(cublasXtHandle_t handle, cublasXtBlasOp_t blasOp,
                                                 cublasXtOpType_t type, void* blasFunctor) {
  XtHandle* x = get(handle);
  if (!x) return CUBLAS_STATUS_NOT_INITIALIZED;
  const int op = enum_value(blasOp), ty = enum_value(type);
  if (op < 0 || op >= CUBLASXT_ROUTINE_MAX || ty < 0 || ty > 3) return CUBLAS_STATUS_INVALID_VALUE;
  if (!cpu_share_known(op, ty)) return CUBLAS_STATUS_NOT_SUPPORTED;
  x->cpu_routine[op][ty] = blasFunctor;
  return CUBLAS_STATUS_SUCCESS;
}
VGPU_EXPORT cublasStatus_t cublasXtSetCpuRatio(cublasXtHandle_t handle, cublasXtBlasOp_t blasOp,
                                               cublasXtOpType_t type, float ratio) {
  XtHandle* x = get(handle);
  if (!x) return CUBLAS_STATUS_NOT_INITIALIZED;
  // The card checks the routine and type, not the ratio (2 and -1 are taken).
  const int op = enum_value(blasOp), ty = enum_value(type);
  if (op < 0 || op >= CUBLASXT_ROUTINE_MAX || ty < 0 || ty > 3) return CUBLAS_STATUS_INVALID_VALUE;
  if (!cpu_share_known(op, ty)) return CUBLAS_STATUS_NOT_SUPPORTED;
  x->cpu_ratio[op][ty] = ratio;
  return CUBLAS_STATUS_SUCCESS;
}

/* ---- the routines ---- */

#define VGPU_XT(P, T, R)                                                                                          \
  VGPU_EXPORT cublasStatus_t cublasXt##P##gemm(cublasXtHandle_t h, cublasOperation_t ta, cublasOperation_t tb,    \
                                               size_t m, size_t n, size_t k, const T* alpha, const T* A,          \
                                               size_t lda, const T* B, size_t ldb, const T* beta, T* C,           \
                                               size_t ldc) {                                                      \
    return xt_gemm<T>(h, ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);                                   \
  }                                                                                                               \
  VGPU_EXPORT cublasStatus_t cublasXt##P##syrk(cublasXtHandle_t h, cublasFillMode_t u, cublasOperation_t t,       \
                                               size_t n, size_t k, const T* alpha, const T* A, size_t lda,        \
                                               const T* beta, T* C, size_t ldc) {                                 \
    if (const cublasStatus_t st_ = precheck(h, fill_v(enum_value(u)) && op_v(enum_value(t)));                     \
        st_ != CUBLAS_STATUS_SUCCESS)                                                                             \
      return st_;                                                                                                 \
    const size_t ar = t == CUBLAS_OP_N ? n : k, ac = t == CUBLAS_OP_N ? k : n;                                    \
    return xt_whole(h, {n, k, lda, ldc},                                                                          \
                    {{A, nullptr, span_bytes<T>(lda, ac, ar)}, {C, C, span_bytes<T>(ldc, n, n)}},                 \
                    [&](cublasHandle_t bh, const std::vector<void*>& d) {                                         \
                      return Blas<T>::syrk(bh, u, t, (int)n, (int)k, alpha, (const T*)d[0], (int)lda, beta,       \
                                           (T*)d[1], (int)ldc);                                                   \
                    });                                                                                           \
  }                                                                                                               \
  VGPU_EXPORT cublasStatus_t cublasXt##P##syr2k(cublasXtHandle_t h, cublasFillMode_t u, cublasOperation_t t,      \
                                                size_t n, size_t k, const T* alpha, const T* A, size_t lda,       \
                                                const T* B, size_t ldb, const T* beta, T* C, size_t ldc) {        \
    if (const cublasStatus_t st_ = precheck(h, fill_v(enum_value(u)) && op_v(enum_value(t)));                     \
        st_ != CUBLAS_STATUS_SUCCESS)                                                                             \
      return st_;                                                                                                 \
    const size_t ar = t == CUBLAS_OP_N ? n : k, ac = t == CUBLAS_OP_N ? k : n;                                    \
    return xt_whole(h, {n, k, lda, ldb, ldc},                                                                     \
                    {{A, nullptr, span_bytes<T>(lda, ac, ar)}, {B, nullptr, span_bytes<T>(ldb, ac, ar)},          \
                     {C, C, span_bytes<T>(ldc, n, n)}},                                                           \
                    [&](cublasHandle_t bh, const std::vector<void*>& d) {                                         \
                      return Blas<T>::syr2k(bh, u, t, (int)n, (int)k, alpha, (const T*)d[0], (int)lda,            \
                                            (const T*)d[1], (int)ldb, beta, (T*)d[2], (int)ldc);                  \
                    });                                                                                           \
  }                                                                                                               \
  VGPU_EXPORT cublasStatus_t cublasXt##P##syrkx(cublasXtHandle_t h, cublasFillMode_t u, cublasOperation_t t,      \
                                                size_t n, size_t k, const T* alpha, const T* A, size_t lda,       \
                                                const T* B, size_t ldb, const T* beta, T* C, size_t ldc) {        \
    if (const cublasStatus_t st_ = precheck(h, fill_v(enum_value(u)) && op_v(enum_value(t)));                     \
        st_ != CUBLAS_STATUS_SUCCESS)                                                                             \
      return st_;                                                                                                 \
    const size_t ar = t == CUBLAS_OP_N ? n : k, ac = t == CUBLAS_OP_N ? k : n;                                    \
    return xt_whole(h, {n, k, lda, ldb, ldc},                                                                     \
                    {{A, nullptr, span_bytes<T>(lda, ac, ar)}, {B, nullptr, span_bytes<T>(ldb, ac, ar)},          \
                     {C, C, span_bytes<T>(ldc, n, n)}},                                                           \
                    [&](cublasHandle_t bh, const std::vector<void*>& d) {                                         \
                      return Blas<T>::syrkx(bh, u, t, (int)n, (int)k, alpha, (const T*)d[0], (int)lda,            \
                                            (const T*)d[1], (int)ldb, beta, (T*)d[2], (int)ldc);                  \
                    });                                                                                           \
  }                                                                                                               \
  VGPU_EXPORT cublasStatus_t cublasXt##P##symm(cublasXtHandle_t h, cublasSideMode_t s, cublasFillMode_t u,        \
                                               size_t m, size_t n, const T* alpha, const T* A, size_t lda,        \
                                               const T* B, size_t ldb, const T* beta, T* C, size_t ldc) {         \
    if (const cublasStatus_t st_ = precheck(h, side_v(enum_value(s)) && fill_v(enum_value(u)));                   \
        st_ != CUBLAS_STATUS_SUCCESS)                                                                             \
      return st_;                                                                                                 \
    const size_t ka = s == CUBLAS_SIDE_LEFT ? m : n;                                                              \
    return xt_whole(h, {m, n, lda, ldb, ldc},                                                                     \
                    {{A, nullptr, span_bytes<T>(lda, ka, ka)}, {B, nullptr, span_bytes<T>(ldb, n, m)},            \
                     {C, C, span_bytes<T>(ldc, n, m)}},                                                           \
                    [&](cublasHandle_t bh, const std::vector<void*>& d) {                                         \
                      return Blas<T>::symm(bh, s, u, (int)m, (int)n, alpha, (const T*)d[0], (int)lda,             \
                                           (const T*)d[1], (int)ldb, beta, (T*)d[2], (int)ldc);                   \
                    });                                                                                           \
  }                                                                                                               \
  VGPU_EXPORT cublasStatus_t cublasXt##P##trsm(cublasXtHandle_t h, cublasSideMode_t s, cublasFillMode_t u,        \
                                               cublasOperation_t t, cublasDiagType_t dg, size_t m, size_t n,      \
                                               const T* alpha, const T* A, size_t lda, T* B, size_t ldb) {        \
    if (const cublasStatus_t st_ = precheck(h, side_v(enum_value(s)) && fill_v(enum_value(u)) &&                  \
                                                 op_v(enum_value(t)) && diag_v(enum_value(dg)));                  \
        st_ != CUBLAS_STATUS_SUCCESS)                                                                             \
      return st_;                                                                                                 \
    const size_t ka = s == CUBLAS_SIDE_LEFT ? m : n;                                                              \
    return xt_whole(h, {m, n, lda, ldb},                                                                          \
                    {{A, nullptr, span_bytes<T>(lda, ka, ka)}, {B, B, span_bytes<T>(ldb, n, m)}},                 \
                    [&](cublasHandle_t bh, const std::vector<void*>& d) {                                         \
                      return Blas<T>::trsm(bh, s, u, t, dg, (int)m, (int)n, alpha, (const T*)d[0], (int)lda,      \
                                           (T*)d[1], (int)ldb);                                                   \
                    });                                                                                           \
  }                                                                                                               \
  VGPU_EXPORT cublasStatus_t cublasXt##P##trmm(cublasXtHandle_t h, cublasSideMode_t s, cublasFillMode_t u,        \
                                               cublasOperation_t t, cublasDiagType_t dg, size_t m, size_t n,      \
                                               const T* alpha, const T* A, size_t lda, const T* B, size_t ldb,    \
                                               T* C, size_t ldc) {                                                \
    if (const cublasStatus_t st_ = precheck(h, side_v(enum_value(s)) && fill_v(enum_value(u)) &&                  \
                                                 op_v(enum_value(t)) && diag_v(enum_value(dg)));                  \
        st_ != CUBLAS_STATUS_SUCCESS)                                                                             \
      return st_;                                                                                                 \
    const size_t ka = s == CUBLAS_SIDE_LEFT ? m : n;                                                              \
    return xt_whole(h, {m, n, lda, ldb, ldc},                                                                     \
                    {{A, nullptr, span_bytes<T>(lda, ka, ka)}, {B, nullptr, span_bytes<T>(ldb, n, m)},            \
                     {C, C, span_bytes<T>(ldc, n, m)}},                                                           \
                    [&](cublasHandle_t bh, const std::vector<void*>& d) {                                         \
                      return Blas<T>::trmm(bh, s, u, t, dg, (int)m, (int)n, alpha, (const T*)d[0], (int)lda,      \
                                           (const T*)d[1], (int)ldb, (T*)d[2], (int)ldc);                         \
                    });                                                                                           \
  }                                                                                                               \
  /* spmm: C = alpha A B + beta C (left) or alpha B A + beta C (right), A                                         \
     symmetric in packed storage: unpacked into its `uplo` triangle, then symm. */                                \
  VGPU_EXPORT cublasStatus_t cublasXt##P##spmm(cublasXtHandle_t h, cublasSideMode_t s, cublasFillMode_t u,        \
                                               size_t m, size_t n, const T* alpha, const T* AP, const T* B,       \
                                               size_t ldb, const T* beta, T* C, size_t ldc) {                     \
    if (const cublasStatus_t st_ = precheck(h, side_v(enum_value(s)) && fill_v(enum_value(u)));                   \
        st_ != CUBLAS_STATUS_SUCCESS)                                                                             \
      return st_;                                                                                                 \
    const size_t ka = s == CUBLAS_SIDE_LEFT ? m : n;                                                              \
    if (!get(h)) return CUBLAS_STATUS_NOT_INITIALIZED;                                                            \
    if (!fits(ka)) return CUBLAS_STATUS_NOT_SUPPORTED;                                                            \
    std::vector<T> full;                                                                                          \
    if (u == CUBLAS_FILL_MODE_LOWER || u == CUBLAS_FILL_MODE_UPPER) {                                             \
      const auto ap = stage(AP, ka * (ka + 1) / 2);                                                               \
      full.assign(ka * ka, T{});                                                                                  \
      const bool lower = u == CUBLAS_FILL_MODE_LOWER;                                                             \
      for (size_t j = 0; j < ka; ++j)                                                                             \
        for (size_t i = lower ? j : 0; i < (lower ? ka : j + 1); ++i)                                             \
          full[j * ka + i] = ap[lower ? i + (2 * ka - j - 1) * j / 2 : i + j * (j + 1) / 2];                      \
    }                                                                                                             \
    const size_t lda = std::max<size_t>(1, ka);                                                                   \
    return xt_whole(h, {m, n, ldb, ldc},                                                                          \
                    {{full.data(), nullptr, full.size() * sizeof(T)}, {B, nullptr, span_bytes<T>(ldb, n, m)},     \
                     {C, C, span_bytes<T>(ldc, n, m)}},                                                           \
                    [&](cublasHandle_t bh, const std::vector<void*>& d) {                                         \
                      return Blas<T>::symm(bh, s, u, (int)m, (int)n, alpha, (const T*)d[0], (int)lda,             \
                                           (const T*)d[1], (int)ldb, beta, (T*)d[2], (int)ldc);                   \
                    });                                                                                           \
  }
VGPU_XT(S, float, float)
VGPU_XT(D, double, double)
VGPU_XT(C, cuComplex, float)
VGPU_XT(Z, cuDoubleComplex, double)
#undef VGPU_XT

// The Hermitian routines, complex only.
#define VGPU_XT_HERM(P, T, R)                                                                                     \
  VGPU_EXPORT cublasStatus_t cublasXt##P##hemm(cublasXtHandle_t h, cublasSideMode_t s, cublasFillMode_t u,        \
                                               size_t m, size_t n, const T* alpha, const T* A, size_t lda,        \
                                               const T* B, size_t ldb, const T* beta, T* C, size_t ldc) {         \
    if (const cublasStatus_t st_ = precheck(h, side_v(enum_value(s)) && fill_v(enum_value(u)));                   \
        st_ != CUBLAS_STATUS_SUCCESS)                                                                             \
      return st_;                                                                                                 \
    const size_t ka = s == CUBLAS_SIDE_LEFT ? m : n;                                                              \
    return xt_whole(h, {m, n, lda, ldb, ldc},                                                                     \
                    {{A, nullptr, span_bytes<T>(lda, ka, ka)}, {B, nullptr, span_bytes<T>(ldb, n, m)},            \
                     {C, C, span_bytes<T>(ldc, n, m)}},                                                           \
                    [&](cublasHandle_t bh, const std::vector<void*>& d) {                                         \
                      return cublas##P##hemm_v2(bh, s, u, (int)m, (int)n, alpha, (const T*)d[0], (int)lda,       \
                                                 (const T*)d[1], (int)ldb, beta, (T*)d[2], (int)ldc);             \
                    });                                                                                           \
  }                                                                                                               \
  VGPU_EXPORT cublasStatus_t cublasXt##P##herk(cublasXtHandle_t h, cublasFillMode_t u, cublasOperation_t t,       \
                                               size_t n, size_t k, const R* alpha, const T* A, size_t lda,        \
                                               const R* beta, T* C, size_t ldc) {                                 \
    if (const cublasStatus_t st_ = precheck(h, fill_v(enum_value(u)) && op_v(enum_value(t)));                     \
        st_ != CUBLAS_STATUS_SUCCESS)                                                                             \
      return st_;                                                                                                 \
    const size_t ar = t == CUBLAS_OP_N ? n : k, ac = t == CUBLAS_OP_N ? k : n;                                    \
    return xt_whole(h, {n, k, lda, ldc},                                                                          \
                    {{A, nullptr, span_bytes<T>(lda, ac, ar)}, {C, C, span_bytes<T>(ldc, n, n)}},                 \
                    [&](cublasHandle_t bh, const std::vector<void*>& d) {                                         \
                      return cublas##P##herk_v2(bh, u, t, (int)n, (int)k, alpha, (const T*)d[0], (int)lda, beta,  \
                                                (T*)d[1], (int)ldc);                                              \
                    });                                                                                           \
  }                                                                                                               \
  VGPU_EXPORT cublasStatus_t cublasXt##P##her2k(cublasXtHandle_t h, cublasFillMode_t u, cublasOperation_t t,      \
                                                size_t n, size_t k, const T* alpha, const T* A, size_t lda,       \
                                                const T* B, size_t ldb, const R* beta, T* C, size_t ldc) {        \
    if (const cublasStatus_t st_ = precheck(h, fill_v(enum_value(u)) && op_v(enum_value(t)));                     \
        st_ != CUBLAS_STATUS_SUCCESS)                                                                             \
      return st_;                                                                                                 \
    const size_t ar = t == CUBLAS_OP_N ? n : k, ac = t == CUBLAS_OP_N ? k : n;                                    \
    return xt_whole(h, {n, k, lda, ldb, ldc},                                                                     \
                    {{A, nullptr, span_bytes<T>(lda, ac, ar)}, {B, nullptr, span_bytes<T>(ldb, ac, ar)},          \
                     {C, C, span_bytes<T>(ldc, n, n)}},                                                           \
                    [&](cublasHandle_t bh, const std::vector<void*>& d) {                                         \
                      return cublas##P##her2k_v2(bh, u, t, (int)n, (int)k, alpha, (const T*)d[0], (int)lda,      \
                                                 (const T*)d[1], (int)ldb, beta, (T*)d[2], (int)ldc);             \
                    });                                                                                           \
  }                                                                                                               \
  VGPU_EXPORT cublasStatus_t cublasXt##P##herkx(cublasXtHandle_t h, cublasFillMode_t u, cublasOperation_t t,      \
                                                size_t n, size_t k, const T* alpha, const T* A, size_t lda,       \
                                                const T* B, size_t ldb, const R* beta, T* C, size_t ldc) {        \
    if (const cublasStatus_t st_ = precheck(h, fill_v(enum_value(u)) && op_v(enum_value(t)));                     \
        st_ != CUBLAS_STATUS_SUCCESS)                                                                             \
      return st_;                                                                                                 \
    const size_t ar = t == CUBLAS_OP_N ? n : k, ac = t == CUBLAS_OP_N ? k : n;                                    \
    return xt_whole(h, {n, k, lda, ldb, ldc},                                                                     \
                    {{A, nullptr, span_bytes<T>(lda, ac, ar)}, {B, nullptr, span_bytes<T>(ldb, ac, ar)},          \
                     {C, C, span_bytes<T>(ldc, n, n)}},                                                           \
                    [&](cublasHandle_t bh, const std::vector<void*>& d) {                                         \
                      return cublas##P##herkx(bh, u, t, (int)n, (int)k, alpha, (const T*)d[0], (int)lda,          \
                                              (const T*)d[1], (int)ldb, beta, (T*)d[2], (int)ldc);                \
                    });                                                                                           \
  }
VGPU_XT_HERM(C, cuComplex, float)
VGPU_XT_HERM(Z, cuDoubleComplex, double)
#undef VGPU_XT_HERM
