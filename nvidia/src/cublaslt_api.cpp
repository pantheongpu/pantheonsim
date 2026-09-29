// libvgpucublasLt — VirtualGPU's implementation of the cuBLASLt API.
//
// cuBLASLt is the descriptor-based matmul interface that modern frameworks
// prefer over the classic cuBLAS calls, so supporting it matters as much as
// cuBLAS itself. The descriptors are opaque handles here, and the arithmetic
// reuses the same host-side GEMM the cuBLAS shim uses -- see nvidia/docs/cublas.md
// for why library math runs on the host rather than through the interpreter.
//
// Types: fp32, fp64, fp16, bf16 and both fp8 formats, in any mix cuBLASLt
// defines, accumulated in double and rounded once into D. FP8 follows the
// documented scaling: D = scaleD * (alpha * scaleA * scaleB * op(A) op(B) +
// beta * scaleC * C), with the absolute maximum of D before scaleD written to
// AMAX_D, and scaleA/scaleB either scalars or, in the outer-vector mode
// torch._scaled_mm uses for rowwise scaling, one per row of op(A) and one per
// column of op(B).
//
// Epilogues (bias, ReLU, GELU) are applied after the matmul, matching the
// documented semantics. Anything not implemented -- another epilogue, a
// block-scaled mode -- returns CUBLAS_STATUS_NOT_SUPPORTED so a caller falls
// back rather than receiving a plausible wrong answer.
#include <cublasLt.h>

#include <cmath>
#include <cstdint>
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

// The scale modes arrived in CUDA 12.8's header; hosted CI builds against
// 12.0, so they are named by value here.
constexpr int kAScaleMode = 31, kBScaleMode = 32, kCScaleMode = 33, kDScaleMode = 34;
constexpr int32_t kScaleScalar = 0, kScaleOuterVec = 3;

struct MatmulDesc {
  cublasComputeType_t compute = CUBLAS_COMPUTE_32F;
  cudaDataType scale = CUDA_R_32F;
  cublasOperation_t transa = CUBLAS_OP_N, transb = CUBLAS_OP_N;
  cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DEFAULT;
  const void* bias = nullptr;
  int32_t bias_type = -1;  // -1: the default, which depends on D's type
  int32_t pointer_mode = CUBLASLT_POINTER_MODE_HOST;
  const void *a_scale = nullptr, *b_scale = nullptr, *c_scale = nullptr, *d_scale = nullptr;
  void* amax_d = nullptr;
  int32_t a_scale_mode = kScaleScalar, b_scale_mode = kScaleScalar;
  int32_t c_scale_mode = kScaleScalar, d_scale_mode = kScaleScalar;
  int8_t fast_accum = 0;
};

struct MatrixLayout {
  cudaDataType type = CUDA_R_32F;
  uint64_t rows = 0, cols = 0;
  int64_t ld = 0;
  int32_t batch = 1;
  int64_t batch_stride = 0;
  int32_t order = CUBLASLT_ORDER_COL;
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

/* ---- logging ----
 * The library's API-trace logger. The simulator's cuBLASLt has no trace of its
 * own to write, so these take and keep the settings, refusing what an RTX
 * 3060's cuBLASLt refuses (a level outside 0 to 6); nothing is ever logged. */

namespace {
struct LtLogger {
  std::mutex mu;
  int level = 0;
  int mask = 0;
  bool disabled = false;
  cublasLtLoggerCallback_t callback = nullptr;
  FILE* file = nullptr;
  bool owns_file = false;
};
LtLogger& lt_logger() {
  static LtLogger l;
  return l;
}
void set_log_file(LtLogger& l, FILE* f, bool owns) {
  if (l.owns_file && l.file) std::fclose(l.file);
  l.file = f;
  l.owns_file = owns;
}
}  // namespace

VGPU_EXPORT cublasStatus_t cublasLtLoggerSetCallback(cublasLtLoggerCallback_t callback) {
  auto& l = lt_logger();
  std::lock_guard<std::mutex> g(l.mu);
  l.callback = callback;
  return CUBLAS_STATUS_SUCCESS;
}
VGPU_EXPORT cublasStatus_t cublasLtLoggerSetFile(FILE* file) {
  auto& l = lt_logger();
  std::lock_guard<std::mutex> g(l.mu);
  set_log_file(l, file, false);
  return CUBLAS_STATUS_SUCCESS;
}
VGPU_EXPORT cublasStatus_t cublasLtLoggerOpenFile(const char* path) {
  if (!path) return CUBLAS_STATUS_SUCCESS;  // as the card's library answers
  FILE* f = std::fopen(path, "w");
  if (!f) return CUBLAS_STATUS_INVALID_VALUE;
  auto& l = lt_logger();
  std::lock_guard<std::mutex> g(l.mu);
  set_log_file(l, f, true);
  return CUBLAS_STATUS_SUCCESS;
}
VGPU_EXPORT cublasStatus_t cublasLtLoggerSetLevel(int level) {
  if (level < 0 || level > 6) return CUBLAS_STATUS_INVALID_VALUE;
  auto& l = lt_logger();
  std::lock_guard<std::mutex> g(l.mu);
  l.level = level;
  return CUBLAS_STATUS_SUCCESS;
}
VGPU_EXPORT cublasStatus_t cublasLtLoggerSetMask(int mask) {
  auto& l = lt_logger();
  std::lock_guard<std::mutex> g(l.mu);
  l.mask = mask;
  return CUBLAS_STATUS_SUCCESS;
}
VGPU_EXPORT cublasStatus_t cublasLtLoggerForceDisable(void) {
  auto& l = lt_logger();
  std::lock_guard<std::mutex> g(l.mu);
  l.disabled = true;
  return CUBLAS_STATUS_SUCCESS;
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
    case CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE:
      if (bytes < sizeof(int32_t)) return CUBLAS_STATUS_INVALID_VALUE;
      std::memcpy(&m->bias_type, buf, sizeof(int32_t));
      return CUBLAS_STATUS_SUCCESS;
    case CUBLASLT_MATMUL_DESC_POINTER_MODE:
      if (bytes < sizeof(int32_t)) return CUBLAS_STATUS_INVALID_VALUE;
      std::memcpy(&m->pointer_mode, buf, sizeof(int32_t));
      return CUBLAS_STATUS_SUCCESS;
    case CUBLASLT_MATMUL_DESC_FAST_ACCUM:
      if (bytes < sizeof(int8_t)) return CUBLAS_STATUS_INVALID_VALUE;
      std::memcpy(&m->fast_accum, buf, sizeof(int8_t));  // accumulation is exact here either way
      return CUBLAS_STATUS_SUCCESS;
    case CUBLASLT_MATMUL_DESC_A_SCALE_POINTER:
    case CUBLASLT_MATMUL_DESC_B_SCALE_POINTER:
    case CUBLASLT_MATMUL_DESC_C_SCALE_POINTER:
    case CUBLASLT_MATMUL_DESC_D_SCALE_POINTER:
    case CUBLASLT_MATMUL_DESC_AMAX_D_POINTER: {
      if (bytes < sizeof(void*)) return CUBLAS_STATUS_INVALID_VALUE;
      const void** slot = attr == CUBLASLT_MATMUL_DESC_A_SCALE_POINTER   ? &m->a_scale
                          : attr == CUBLASLT_MATMUL_DESC_B_SCALE_POINTER ? &m->b_scale
                          : attr == CUBLASLT_MATMUL_DESC_C_SCALE_POINTER ? &m->c_scale
                          : attr == CUBLASLT_MATMUL_DESC_D_SCALE_POINTER ? &m->d_scale
                                                                         : const_cast<const void**>(&m->amax_d);
      std::memcpy(slot, buf, sizeof(void*));
      return CUBLAS_STATUS_SUCCESS;
    }
    default:
      break;
  }
  if ((int)attr >= kAScaleMode && (int)attr <= kDScaleMode) {
    if (bytes < sizeof(int32_t)) return CUBLAS_STATUS_INVALID_VALUE;
    int32_t* slot = (int)attr == kAScaleMode   ? &m->a_scale_mode
                    : (int)attr == kBScaleMode ? &m->b_scale_mode
                    : (int)attr == kCScaleMode ? &m->c_scale_mode
                                               : &m->d_scale_mode;
    std::memcpy(slot, buf, sizeof(int32_t));
  }
  return CUBLAS_STATUS_SUCCESS;  // attributes we do not model are inert
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
  else if (attr == CUBLASLT_MATRIX_LAYOUT_ORDER) {
    std::memcpy(&m->order, buf, sizeof(int32_t));
    if (m->order != CUBLASLT_ORDER_COL && m->order != CUBLASLT_ORDER_ROW) return CUBLAS_STATUS_NOT_SUPPORTED;
  } else if (attr == CUBLASLT_MATRIX_LAYOUT_TYPE) std::memcpy(&m->type, buf, sizeof(int32_t));
  else if (attr == CUBLASLT_MATRIX_LAYOUT_ROWS) std::memcpy(&m->rows, buf, sizeof(uint64_t));
  else if (attr == CUBLASLT_MATRIX_LAYOUT_COLS) std::memcpy(&m->cols, buf, sizeof(uint64_t));
  else if (attr == CUBLASLT_MATRIX_LAYOUT_LD) std::memcpy(&m->ld, buf, sizeof(int64_t));
  return CUBLAS_STATUS_SUCCESS;
}
// Every attribute answers for itself; this used to return the batch count
// whatever was asked.
VGPU_EXPORT cublasStatus_t cublasLtMatrixLayoutGetAttribute(cublasLtMatrixLayout_t l,
                                                            cublasLtMatrixLayoutAttribute_t attr,
                                                            void* buf, size_t bytes, size_t* written) {
  if (!known(l)) return CUBLAS_STATUS_INVALID_VALUE;
  const auto* m = reinterpret_cast<MatrixLayout*>(l);
  auto give = [&](const void* v, size_t n) {
    if (written) *written = n;
    if (!buf) return bytes == 0 ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_INVALID_VALUE;
    if (bytes < n) return CUBLAS_STATUS_INVALID_VALUE;
    std::memcpy(buf, v, n);
    return CUBLAS_STATUS_SUCCESS;
  };
  const int32_t type = m->type;
  switch (attr) {
    case CUBLASLT_MATRIX_LAYOUT_TYPE: return give(&type, sizeof type);
    case CUBLASLT_MATRIX_LAYOUT_ORDER: return give(&m->order, sizeof m->order);
    case CUBLASLT_MATRIX_LAYOUT_ROWS: return give(&m->rows, sizeof m->rows);
    case CUBLASLT_MATRIX_LAYOUT_COLS: return give(&m->cols, sizeof m->cols);
    case CUBLASLT_MATRIX_LAYOUT_LD: return give(&m->ld, sizeof m->ld);
    case CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT: return give(&m->batch, sizeof m->batch);
    case CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET: return give(&m->batch_stride, sizeof m->batch_stride);
    default: return CUBLAS_STATUS_NOT_SUPPORTED;
  }
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

namespace {

// Element conversions for every type a cuBLASLt matmul takes, by hand so the
// library needs no CUDA half or fp8 header on the host side.
double from_bits_fp8(uint8_t b, bool e4m3) {
  const int sign = b >> 7;
  const int ebits = e4m3 ? 4 : 5, mbits = e4m3 ? 3 : 2, bias = e4m3 ? 7 : 15;
  const int e = (b >> mbits) & ((1 << ebits) - 1), m = b & ((1 << mbits) - 1);
  double v;
  if (e4m3 && e == 15 && m == 7) v = NAN;  // E4M3 has no infinity, one NaN
  else if (!e4m3 && e == 31) v = m ? NAN : INFINITY;
  else if (e == 0) v = std::ldexp((double)m, 1 - bias - mbits);
  else v = std::ldexp((double)(m | (1 << mbits)), e - bias - mbits);
  return sign ? -v : v;
}
// Round to nearest even, saturating to the largest finite value as cuBLASLt's
// FP8 outputs do (448 for E4M3, 57344 for E5M2).
uint8_t to_bits_fp8(double v, bool e4m3) {
  const int mbits = e4m3 ? 3 : 2, bias = e4m3 ? 7 : 15;
  const double maxv = e4m3 ? 448.0 : 57344.0;
  if (std::isnan(v)) return 0x7f;  // the canonical NaN of both formats
  const uint8_t sign = std::signbit(v) ? 0x80 : 0;
  double a = std::fabs(v);
  if (a >= maxv) return sign | (e4m3 ? 0x7e : 0x7b);
  int e;
  std::frexp(a, &e);  // a = f * 2^e, f in [0.5, 1)
  int exp = e - 1;    // a = 1.x * 2^exp
  const int emin = 1 - bias;
  if (exp < emin) exp = emin;  // subnormal range: fixed quantum
  const double quantum = std::ldexp(1.0, exp - mbits);
  double q = std::nearbyint(a / quantum);  // default rounding: to nearest even
  double r = q * quantum;
  if (r >= maxv) return sign | (e4m3 ? 0x7e : 0x7b);
  if (r == 0) return sign;
  int re;
  std::frexp(r, &re);
  int rexp = re - 1;
  if (rexp < emin) {  // subnormal
    return sign | (uint8_t)(int)(r / std::ldexp(1.0, emin - mbits));
  }
  const int mant = (int)(r / std::ldexp(1.0, rexp - mbits)) - (1 << mbits);
  return sign | (uint8_t)(((rexp + bias) << mbits) | mant);
}
double from_half(uint16_t h) {
  const int e = (h >> 10) & 0x1f, m = h & 0x3ff;
  double v = e == 0 ? std::ldexp((double)m, -24)
             : e == 31 ? (m ? NAN : INFINITY)
                       : std::ldexp((double)(m | 0x400), e - 25);
  return (h & 0x8000) ? -v : v;
}
uint16_t to_half(double d) {
  const float f = (float)d;
  uint32_t x;
  std::memcpy(&x, &f, 4);
  const uint32_t sign = (x >> 16) & 0x8000;
  const int e = (int)((x >> 23) & 0xff) - 112;
  uint32_t m = x & 0x7fffff;
  if (((x >> 23) & 0xff) == 0xff) return (uint16_t)(sign | 0x7c00 | (m ? 0x200 : 0));
  if (e >= 31) return (uint16_t)(sign | 0x7c00);
  if (e <= 0) {
    if (e < -10) return (uint16_t)sign;
    m |= 0x800000;
    const int shift = 14 - e;
    uint32_t h = m >> shift;
    const uint32_t rem = m & ((1u << shift) - 1), half = 1u << (shift - 1);
    if (rem > half || (rem == half && (h & 1))) ++h;
    return (uint16_t)(sign | h);
  }
  uint32_t h = ((uint32_t)e << 10) | (m >> 13);
  const uint32_t rem = m & 0x1fff;
  if (rem > 0x1000 || (rem == 0x1000 && (h & 1))) ++h;
  return (uint16_t)(sign | h);
}
double from_bf16(uint16_t b) {
  const uint32_t x = (uint32_t)b << 16;
  float f;
  std::memcpy(&f, &x, 4);
  return f;
}
uint16_t to_bf16(double d) {
  const float f = (float)d;
  uint32_t x;
  std::memcpy(&x, &f, 4);
  if ((x & 0x7fffffff) > 0x7f800000) return (uint16_t)((x >> 16) | 0x40);
  x += 0x7fff + ((x >> 16) & 1);
  return (uint16_t)(x >> 16);
}

size_t elem_bytes(cudaDataType t) {
  switch (t) {
    case CUDA_R_8F_E4M3: case CUDA_R_8F_E5M2: return 1;
    case CUDA_R_16F: case CUDA_R_16BF: return 2;
    case CUDA_R_32F: return 4;
    case CUDA_R_64F: return 8;
    default: return 0;
  }
}
bool is_fp8(cudaDataType t) { return t == CUDA_R_8F_E4M3 || t == CUDA_R_8F_E5M2; }

// A device buffer of `n` elements of type `t`, as doubles.
std::vector<double> read_as_double(const void* dev, size_t n, cudaDataType t) {
  std::vector<double> out(n);
  if (!n) return out;
  std::vector<uint8_t> raw(n * elem_bytes(t));
  cudaMemcpy(raw.data(), dev, raw.size(), cudaMemcpyDeviceToHost);
  for (size_t i = 0; i < n; ++i) {
    const uint8_t* p = raw.data() + i * elem_bytes(t);
    switch (t) {
      case CUDA_R_8F_E4M3: out[i] = from_bits_fp8(*p, true); break;
      case CUDA_R_8F_E5M2: out[i] = from_bits_fp8(*p, false); break;
      case CUDA_R_16F: { uint16_t h; std::memcpy(&h, p, 2); out[i] = from_half(h); break; }
      case CUDA_R_16BF: { uint16_t h; std::memcpy(&h, p, 2); out[i] = from_bf16(h); break; }
      case CUDA_R_32F: { float f; std::memcpy(&f, p, 4); out[i] = f; break; }
      default: std::memcpy(&out[i], p, 8); break;
    }
  }
  return out;
}
void encode(double v, cudaDataType t, uint8_t* p) {
  switch (t) {
    case CUDA_R_8F_E4M3: *p = to_bits_fp8(v, true); break;
    case CUDA_R_8F_E5M2: *p = to_bits_fp8(v, false); break;
    case CUDA_R_16F: { const uint16_t h = to_half(v); std::memcpy(p, &h, 2); break; }
    case CUDA_R_16BF: { const uint16_t h = to_bf16(v); std::memcpy(p, &h, 2); break; }
    case CUDA_R_32F: { const float f = (float)v; std::memcpy(p, &f, 4); break; }
    default: std::memcpy(p, &v, 8); break;
  }
}

// The span, in elements, a layout's one matrix occupies.
size_t span(const MatrixLayout& l) {
  if (!l.rows || !l.cols) return 0;
  return l.order == CUBLASLT_ORDER_ROW ? (size_t)(l.rows - 1) * l.ld + l.cols
                                       : (size_t)(l.cols - 1) * l.ld + l.rows;
}
size_t at(const MatrixLayout& l, int64_t r, int64_t c) {
  return l.order == CUBLASLT_ORDER_ROW ? (size_t)r * l.ld + c : (size_t)c * l.ld + r;
}

// alpha and beta are of the scale type; a scale is always fp32.
double read_scalar(const void* p, cudaDataType t, bool device) {
  if (!p) return 0.0;
  if (device) return read_as_double(p, 1, t)[0];
  switch (t) {
    case CUDA_R_64F: return *static_cast<const double*>(p);
    case CUDA_R_16F: return from_half(*static_cast<const uint16_t*>(p));
    case CUDA_R_16BF: return from_bf16(*static_cast<const uint16_t*>(p));
    default: return *static_cast<const float*>(p);
  }
}

}  // namespace

VGPU_EXPORT cublasStatus_t cublasLtMatmul(cublasLtHandle_t h, cublasLtMatmulDesc_t desc,
                                          const void* alpha, const void* A,
                                          cublasLtMatrixLayout_t Adesc, const void* B,
                                          cublasLtMatrixLayout_t Bdesc, const void* beta,
                                          const void* C, cublasLtMatrixLayout_t Cdesc, void* D,
                                          cublasLtMatrixLayout_t Ddesc, const cublasLtMatmulAlgo_t*,
                                          void*, size_t, cudaStream_t stream) {
  // Any handle will do, not only one cublasLtCreate made: a cuBLAS handle is
  // a valid cuBLASLt handle, and PyTorch passes its cuBLAS handle here. This
  // library keeps nothing in a handle.
  if (!h) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (!known(desc) || !known(Adesc) || !known(Bdesc) || !known(Ddesc)) return CUBLAS_STATUS_INVALID_VALUE;
  const auto& md = *reinterpret_cast<MatmulDesc*>(desc);
  const auto& la = *reinterpret_cast<MatrixLayout*>(Adesc);
  const auto& lb = *reinterpret_cast<MatrixLayout*>(Bdesc);
  const auto& ld = *reinterpret_cast<MatrixLayout*>(Ddesc);
  const auto& lc = known(Cdesc) ? *reinterpret_cast<MatrixLayout*>(Cdesc) : ld;

  auto refuse = [&](const char* why) {
    if (trace() || !quiet())
      std::fprintf(stderr, "[vgpu] cublasLtMatmul: %s (A=%d B=%d C=%d D=%d)\n", why, (int)la.type,
                   (int)lb.type, (int)lc.type, (int)ld.type);
    return CUBLAS_STATUS_NOT_SUPPORTED;
  };
  if (!elem_bytes(la.type) || !elem_bytes(lb.type) || !elem_bytes(lc.type) || !elem_bytes(ld.type))
    return refuse("a matrix type other than fp64, fp32, fp16, bf16 or fp8");
  const bool ta = md.transa != CUBLAS_OP_N, tb = md.transb != CUBLAS_OP_N;
  const int64_t m = (int64_t)ld.rows, n = (int64_t)ld.cols;
  const int64_t k = ta ? (int64_t)la.rows : (int64_t)la.cols;
  if ((ta ? (int64_t)la.cols : (int64_t)la.rows) != m || (tb ? (int64_t)lb.cols : (int64_t)lb.rows) != k ||
      (tb ? (int64_t)lb.rows : (int64_t)lb.cols) != n || (int64_t)lc.rows != m || (int64_t)lc.cols != n)
    return CUBLAS_STATUS_INVALID_VALUE;

  bool relu = false, gelu_ = false, bias = false;
  switch (md.epilogue) {
    case CUBLASLT_EPILOGUE_DEFAULT: break;
    case CUBLASLT_EPILOGUE_RELU: relu = true; break;
    case CUBLASLT_EPILOGUE_BIAS: bias = true; break;
    case CUBLASLT_EPILOGUE_RELU_BIAS: relu = bias = true; break;
    case CUBLASLT_EPILOGUE_GELU: gelu_ = true; break;
    case CUBLASLT_EPILOGUE_GELU_BIAS: gelu_ = bias = true; break;
    default: return refuse("an epilogue other than bias, ReLU and GELU");
  }
  const bool vec_a = md.a_scale_mode == kScaleOuterVec, vec_b = md.b_scale_mode == kScaleOuterVec;
  if ((md.a_scale_mode != kScaleScalar && !vec_a) || (md.b_scale_mode != kScaleScalar && !vec_b) ||
      md.c_scale_mode != kScaleScalar || md.d_scale_mode != kScaleScalar)
    return refuse("a block-scaled mode");
  const int pm = md.pointer_mode;
  if (pm < CUBLASLT_POINTER_MODE_HOST || pm > CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST)
    return CUBLAS_STATUS_INVALID_VALUE;

  // The inputs are complete once the stream has reached this call.
  cudaStreamSynchronize(stream);

  // alpha and beta: scalars on the host or device, or per-row vectors on the device.
  const bool alpha_vec = pm >= CUBLASLT_POINTER_MODE_DEVICE_VECTOR;
  const bool beta_vec = pm == CUBLASLT_POINTER_MODE_DEVICE_VECTOR;
  const bool beta_zero = pm == CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_ZERO;
  const std::vector<double> alphas = alpha_vec ? read_as_double(alpha, (size_t)m, md.scale)
                                               : std::vector<double>{alpha ? read_scalar(alpha, md.scale, pm == CUBLASLT_POINTER_MODE_DEVICE) : 1.0};
  const std::vector<double> betas =
      beta_zero ? std::vector<double>{0.0}
      : beta_vec ? read_as_double(beta, (size_t)m, md.scale)
                 : std::vector<double>{beta ? read_scalar(beta, md.scale, pm == CUBLASLT_POINTER_MODE_DEVICE) : 0.0};
  bool any_beta = false;
  for (double b : betas) any_beta = any_beta || b != 0.0;

  // FP8 scales (fp32 on the device); a scale left unset is 1. The C and D
  // scales apply only to fp8 C and D.
  auto scales = [](const void* p, size_t count) {
    return p ? read_as_double(p, count, CUDA_R_32F) : std::vector<double>(count, 1.0);
  };
  const std::vector<double> sa = scales(md.a_scale, vec_a ? (size_t)m : 1);
  const std::vector<double> sb = scales(md.b_scale, vec_b ? (size_t)n : 1);
  const double sc = is_fp8(lc.type) ? scales(md.c_scale, 1)[0] : 1.0;
  const double sd = is_fp8(ld.type) ? scales(md.d_scale, 1)[0] : 1.0;

  cudaDataType bias_type = (cudaDataType)md.bias_type;
  if (md.bias_type < 0) bias_type = is_fp8(ld.type) ? CUDA_R_16BF : ld.type;
  std::vector<double> hbias;
  if (bias && md.bias) {
    if (!elem_bytes(bias_type)) return refuse("a bias type it cannot read");
    hbias = read_as_double(md.bias, (size_t)m, bias_type);
  }

  const int batch = ld.batch;
  if ((la.batch != 1 && la.batch != batch) || (lb.batch != 1 && lb.batch != batch)) return CUBLAS_STATUS_INVALID_VALUE;
  double amax = 0.0;
  for (int bi = 0; bi < batch; ++bi) {
    auto base = [&](const void* p, const MatrixLayout& l) {
      return static_cast<const uint8_t*>(p) + (size_t)bi * (size_t)l.batch_stride * elem_bytes(l.type);
    };
    const auto ha = read_as_double(base(A, la), span(la), la.type);
    const auto hb = read_as_double(base(B, lb), span(lb), lb.type);
    const auto hc = (C && any_beta) ? read_as_double(base(C, lc), span(lc), lc.type) : std::vector<double>();
    // D is read too, so the gaps a leading dimension leaves come back untouched.
    uint8_t* dptr = static_cast<uint8_t*>(D) + (size_t)bi * (size_t)ld.batch_stride * elem_bytes(ld.type);
    std::vector<uint8_t> hd(span(ld) * elem_bytes(ld.type));
    if (!hd.empty()) cudaMemcpy(hd.data(), dptr, hd.size(), cudaMemcpyDeviceToHost);
    for (int64_t j = 0; j < n; ++j)
      for (int64_t i = 0; i < m; ++i) {
        double acc = 0.0;
        for (int64_t p = 0; p < k; ++p)
          acc += ha[ta ? at(la, p, i) : at(la, i, p)] * hb[tb ? at(lb, j, p) : at(lb, p, j)];
        const double al = alphas[alpha_vec ? (size_t)i : 0], be = betas[beta_vec ? (size_t)i : 0];
        double v = al * sa[vec_a ? (size_t)i : 0] * sb[vec_b ? (size_t)j : 0] * acc;
        if (be != 0.0 && !hc.empty()) v += be * sc * hc[at(lc, i, j)];
        if (!hbias.empty()) v += hbias[(size_t)i];
        if (relu) v = v > 0.0 ? v : 0.0;
        if (gelu_) v = gelu((float)v);
        amax = std::fmax(amax, std::fabs(v));
        encode(v * sd, ld.type, hd.data() + at(ld, i, j) * elem_bytes(ld.type));
      }
    if (!hd.empty()) cudaMemcpy(dptr, hd.data(), hd.size(), cudaMemcpyHostToDevice);
  }
  if (md.amax_d) {
    const float a = (float)amax;
    cudaMemcpy(md.amax_d, &a, sizeof a, cudaMemcpyHostToDevice);
  }
  return CUBLAS_STATUS_SUCCESS;
}
