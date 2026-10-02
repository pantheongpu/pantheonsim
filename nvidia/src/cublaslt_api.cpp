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
// Epilogues -- bias, ReLU and GELU, their auxiliary outputs, and the
// backward ones (DRELU, DGELU, the bias gradients) -- follow the documented
// semantics and an RTX 3060's measured behaviour (see "epilogues" below).
// The block-scaled FP8 and FP4 modes (VEC16_UE4M3, VEC32_UE8M0, VEC128_32F,
// BLK128x128_32F) are implemented from NVIDIA's documentation alone: no card
// here can run them, so they are documentation-derived, not card-verified.
// Anything not implemented returns CUBLAS_STATUS_NOT_SUPPORTED so a caller
// falls back rather than receiving a plausible wrong answer.
#include <cublasLt.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
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

// Attribute numbers newer than the oldest header this builds against (CUDA
// 12.0's), named by value.
constexpr int kDOutScalePointer = 36, kDOutScaleMode = 37;
// The block-scaled modes (cublasLtMatmulMatrixScale_t, CUDA 12.8) and the
// narrow types they come with (library_types.h, CUDA 12.8).
constexpr int32_t kScaleVec16UE4M3 = 1, kScaleVec32UE8M0 = 2, kScaleVec128 = 4, kScaleBlk128x128 = 5;
constexpr int kTypeUE8M0 = 30, kTypeFP4 = 33;

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
  // The epilogue's auxiliary buffer: ReLU's bit mask or GELU's input.
  void* aux = nullptr;
  int64_t aux_ld = 0, aux_batch_stride = 0, bias_batch_stride = 0;
  int32_t aux_type = -1;   // -1: D's type
  // Block-scaled output: where the scales D's quantization computes go.
  void* d_out_scale = nullptr;
  int32_t d_out_scale_mode = kScaleScalar;
  // Every attribute as last set, for cublasLtMatmulDescGetAttribute.
  std::map<int, std::vector<uint8_t>> raw;
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
namespace {
// The width of each attribute this library reads, and its default (what
// cublasLtMatmulDescGetAttribute answers before a set).
struct AttrInfo {
  size_t bytes;
  int64_t def;
};
bool attr_info(int attr, AttrInfo* out) {
  switch (attr) {
    case CUBLASLT_MATMUL_DESC_POINTER_MODE: case CUBLASLT_MATMUL_DESC_TRANSA: case CUBLASLT_MATMUL_DESC_TRANSB:
    case CUBLASLT_MATMUL_DESC_TRANSC: case CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET:
    case kAScaleMode: case kBScaleMode: case kCScaleMode: case kDScaleMode: case 35: case kDOutScaleMode:
      *out = {sizeof(int32_t), 0}; return true;
    case CUBLASLT_MATMUL_DESC_FILL_MODE: *out = {sizeof(int32_t), CUBLAS_FILL_MODE_FULL}; return true;
    case CUBLASLT_MATMUL_DESC_EPILOGUE: *out = {sizeof(uint32_t), CUBLASLT_EPILOGUE_DEFAULT}; return true;
    case CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE: case CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE:
      *out = {sizeof(int32_t), -1}; return true;
    case CUBLASLT_MATMUL_DESC_FAST_ACCUM: *out = {sizeof(int8_t), 0}; return true;
    case CUBLASLT_MATMUL_DESC_BIAS_BATCH_STRIDE: case CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD:
    case CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE: case CUBLASLT_MATMUL_DESC_ALPHA_VECTOR_BATCH_STRIDE:
      *out = {sizeof(int64_t), 0}; return true;
    case CUBLASLT_MATMUL_DESC_BIAS_POINTER: case CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER:
    case CUBLASLT_MATMUL_DESC_A_SCALE_POINTER: case CUBLASLT_MATMUL_DESC_B_SCALE_POINTER:
    case CUBLASLT_MATMUL_DESC_C_SCALE_POINTER: case CUBLASLT_MATMUL_DESC_D_SCALE_POINTER:
    case CUBLASLT_MATMUL_DESC_AMAX_D_POINTER: case CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER:
    case CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_AMAX_POINTER: case kDOutScalePointer:
      *out = {sizeof(void*), 0}; return true;
    default: return false;
  }
}
}  // namespace

VGPU_EXPORT cublasStatus_t cublasLtMatmulDescSetAttribute(cublasLtMatmulDesc_t d,
                                                          cublasLtMatmulDescAttributes_t attr,
                                                          const void* buf, size_t bytes) {
  if (!known(d) || !buf) return CUBLAS_STATUS_INVALID_VALUE;
  auto* m = reinterpret_cast<MatmulDesc*>(d);
  AttrInfo info;
  if (!attr_info((int)attr, &info)) return CUBLAS_STATUS_SUCCESS;   // attributes not modelled are inert
  if (bytes < info.bytes) return CUBLAS_STATUS_INVALID_VALUE;
  auto take = [&](void* field) { std::memcpy(field, buf, info.bytes); };
  switch ((int)attr) {
    case CUBLASLT_MATMUL_DESC_TRANSA: take(&m->transa); break;
    case CUBLASLT_MATMUL_DESC_TRANSB: take(&m->transb); break;
    case CUBLASLT_MATMUL_DESC_EPILOGUE: take(&m->epilogue); break;
    case CUBLASLT_MATMUL_DESC_BIAS_POINTER: take(&m->bias); break;
    case CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE: take(&m->bias_type); break;
    case CUBLASLT_MATMUL_DESC_POINTER_MODE: take(&m->pointer_mode); break;
    case CUBLASLT_MATMUL_DESC_FAST_ACCUM: take(&m->fast_accum); break;   // accumulation is exact here either way
    case CUBLASLT_MATMUL_DESC_A_SCALE_POINTER: take(&m->a_scale); break;
    case CUBLASLT_MATMUL_DESC_B_SCALE_POINTER: take(&m->b_scale); break;
    case CUBLASLT_MATMUL_DESC_C_SCALE_POINTER: take(&m->c_scale); break;
    case CUBLASLT_MATMUL_DESC_D_SCALE_POINTER: take(&m->d_scale); break;
    case CUBLASLT_MATMUL_DESC_AMAX_D_POINTER: take(&m->amax_d); break;
    case CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER: take(&m->aux); break;
    case CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD: take(&m->aux_ld); break;
    case CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE: take(&m->aux_batch_stride); break;
    case CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE: take(&m->aux_type); break;
    case CUBLASLT_MATMUL_DESC_BIAS_BATCH_STRIDE: take(&m->bias_batch_stride); break;
    case kAScaleMode: take(&m->a_scale_mode); break;
    case kBScaleMode: take(&m->b_scale_mode); break;
    case kCScaleMode: take(&m->c_scale_mode); break;
    case kDScaleMode: take(&m->d_scale_mode); break;
    case kDOutScalePointer: take(&m->d_out_scale); break;
    case kDOutScaleMode: take(&m->d_out_scale_mode); break;
    default: break;
  }
  const auto* b = static_cast<const uint8_t*>(buf);
  m->raw[(int)attr].assign(b, b + info.bytes);
  return CUBLAS_STATUS_SUCCESS;
}
// What was set, or the attribute's default; the compute and scale types are
// the descriptor's own. With a NULL buffer and zero size it answers the size
// in sizeWritten, as the API documents.
VGPU_EXPORT cublasStatus_t cublasLtMatmulDescGetAttribute(cublasLtMatmulDesc_t d,
                                                          cublasLtMatmulDescAttributes_t attr,
                                                          void* buf, size_t bytes,
                                                          size_t* written) {
  if (!known(d)) return CUBLAS_STATUS_INVALID_VALUE;
  auto* m = reinterpret_cast<MatmulDesc*>(d);
  std::vector<uint8_t> v;
  AttrInfo info;
  if (attr == CUBLASLT_MATMUL_DESC_COMPUTE_TYPE || attr == CUBLASLT_MATMUL_DESC_SCALE_TYPE) {
    const int32_t t = attr == CUBLASLT_MATMUL_DESC_COMPUTE_TYPE ? (int32_t)m->compute : (int32_t)m->scale;
    v.assign((const uint8_t*)&t, (const uint8_t*)&t + sizeof t);
  } else if (m->raw.count((int)attr)) {
    v = m->raw[(int)attr];
  } else if (attr_info((int)attr, &info)) {
    v.assign(info.bytes, 0);
    std::memcpy(v.data(), &info.def, info.bytes);   // little-endian: the low bytes hold the value
  } else {
    return CUBLAS_STATUS_NOT_SUPPORTED;   // rather than a plausible wrong value
  }
  if (written) *written = v.size();
  if (!buf) return bytes == 0 && written ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_INVALID_VALUE;
  if (bytes < v.size()) return CUBLAS_STATUS_INVALID_VALUE;
  std::memcpy(buf, v.data(), v.size());
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

/* ---- FP4 and the block scales' element types ----
   Documentation-derived, not card-verified: an RTX 3060 (sm_86) runs no
   FP4 or block-scaled matmul. E2M1 holds 0, 0.5, 1, 1.5, 2, 3, 4 and 6 and
   a sign, two to a byte, the lower-addressed element in the low nibble;
   UE8M0 is a bare power of two, 2^(e - 127), 255 its NaN; UE4M3 is E4M3
   with the sign bit ignored. */
const double kFp4[8] = {0, 0.5, 1, 1.5, 2, 3, 4, 6};
double from_fp4(uint8_t nib) { return (nib & 8) ? -kFp4[nib & 7] : kFp4[nib & 7]; }
// Round to nearest even (the even code is the one with a zero mantissa bit),
// saturating to 6; NaN becomes 6 as no E2M1 value is NaN.
uint8_t to_fp4(double v) {
  const uint8_t sign = std::signbit(v) ? 8 : 0;
  const double a = std::isnan(v) ? 6.0 : std::fabs(v);
  for (int i = 0; i < 7; ++i) {
    const double mid = (kFp4[i] + kFp4[i + 1]) / 2;
    if (a < mid || (a == mid && i % 2 == 0)) return sign | (uint8_t)i;
  }
  return sign | 7;
}
double from_ue8m0(uint8_t b) { return b == 255 ? NAN : std::ldexp(1.0, (int)b - 127); }
double from_ue4m3(uint8_t b) { return from_bits_fp8(b & 0x7f, true); }

bool is_fp4(cudaDataType t) { return (int)t == kTypeFP4; }
bool narrow(cudaDataType t) { return is_fp8(t) || is_fp4(t); }
bool known_type(cudaDataType t) { return elem_bytes(t) || is_fp4(t); }
size_t elem_bits(cudaDataType t) { return is_fp4(t) ? 4 : elem_bytes(t) * 8; }

// One matrix's elements, read whole from device memory, decoded and encoded
// in place: FP4 included.
struct Elems {
  std::vector<uint8_t> raw;
  cudaDataType t = CUDA_R_32F;
  void load(const void* dev, size_t n, cudaDataType type) {
    t = type;
    raw.assign((n * elem_bits(t) + 7) / 8, 0);
    if (!raw.empty()) cudaMemcpy(raw.data(), dev, raw.size(), cudaMemcpyDeviceToHost);
  }
  void save(void* dev) const {
    if (!raw.empty()) cudaMemcpy(dev, raw.data(), raw.size(), cudaMemcpyHostToDevice);
  }
  double get(size_t i) const {
    if (is_fp4(t)) return from_fp4((raw[i / 2] >> (4 * (i & 1))) & 0xF);
    const uint8_t* p = raw.data() + i * elem_bytes(t);
    switch (t) {
      case CUDA_R_8F_E4M3: return from_bits_fp8(*p, true);
      case CUDA_R_8F_E5M2: return from_bits_fp8(*p, false);
      case CUDA_R_16F: { uint16_t h; std::memcpy(&h, p, 2); return from_half(h); }
      case CUDA_R_16BF: { uint16_t h; std::memcpy(&h, p, 2); return from_bf16(h); }
      case CUDA_R_32F: { float f; std::memcpy(&f, p, 4); return f; }
      default: { double d; std::memcpy(&d, p, 8); return d; }
    }
  }
  void set(size_t i, double v) {
    if (is_fp4(t)) {
      uint8_t& b = raw[i / 2];
      const int sh = 4 * (int)(i & 1);
      b = (uint8_t)((b & ~(0xF << sh)) | (to_fp4(v) << sh));
      return;
    }
    encode(v, t, raw.data() + i * elem_bytes(t));
  }
};
size_t batch_offset_bytes(const MatrixLayout& l, int bi) {
  return (size_t)bi * (size_t)l.batch_stride * elem_bits(l.type) / 8;
}

/* ---- the block scales' layouts (cuBLAS documentation, "1D Block Scaling
   Factors Layout" and "Scaling factors layouts"); documentation-derived, not
   card-verified ----
   VEC16_UE4M3 and VEC32_UE8M0: one byte per 16 or 32 elements along the
   inner dimension (K for A and B, M for C and D), in 128 x 4 tiles. Within a
   tile, the scale of outer index o and inner block b sits at
   (o % 32) * 16 + (o % 128 / 32) * 4 + b % 4; tiles are 512 bytes, row-major
   by (outer / 128, inner block / 4), with the inner blocks padded to a
   multiple of 4 and the outer dimension to 128. The layout does not follow a
   transpose: A's outer index is always M, B's N.
   VEC128_32F: one float per 128 elements of K, M-major for A (M x L), N-major
   for B. BLK128x128_32F: one float per 128 x 128 block, K-major with the
   block columns padded to a multiple of 4 (L4 x ceil(rows / 128)). */
struct Scales {
  int mode = kScaleScalar;
  std::vector<uint8_t> bytes;    // the tiled forms
  std::vector<double> values;    // the 32F forms (one value for a scalar)
  size_t inner_blocks = 0, outer = 0;
  int block = 1;
  // The multiplier for the element at (outer index o, inner index k).
  double at(size_t o, size_t k) const {
    switch (mode) {
      case kScaleScalar: return values.empty() ? 1.0 : values[0];
      case kScaleOuterVec: return values[o];
      case kScaleVec16UE4M3:
      case kScaleVec32UE8M0: {
        const size_t b = k / block, dim = (inner_blocks + 3) / 4 * 4;
        const size_t off = ((b / 4) * 4 + (o / 128) * dim) * 128 + (o % 32) * 16 + (o % 128 / 32) * 4 + b % 4;
        return mode == kScaleVec16UE4M3 ? from_ue4m3(bytes[off]) : from_ue8m0(bytes[off]);
      }
      case kScaleVec128: return values[o + (k / 128) * outer];
      default: return values[(k / 128) + (o / 128) * ((inner_blocks + 3) / 4 * 4)];   // kScaleBlk128x128
    }
  }
};
size_t tiled_bytes(size_t outer, size_t inner_blocks) { return (outer + 127) / 128 * 128 * ((inner_blocks + 3) / 4 * 4); }

Scales load_scales(const void* p, int mode, size_t outer, size_t inner) {
  Scales s;
  s.mode = mode;
  s.outer = outer;
  switch (mode) {
    case kScaleScalar: s.values = p ? read_as_double(p, 1, CUDA_R_32F) : std::vector<double>{1.0}; break;
    case kScaleOuterVec: s.values = p ? read_as_double(p, outer, CUDA_R_32F) : std::vector<double>(outer, 1.0); break;
    case kScaleVec16UE4M3:
    case kScaleVec32UE8M0:
      s.block = mode == kScaleVec16UE4M3 ? 16 : 32;
      s.inner_blocks = (inner + s.block - 1) / s.block;
      s.bytes.assign(tiled_bytes(outer, s.inner_blocks), 0);
      if (p) cudaMemcpy(s.bytes.data(), p, s.bytes.size(), cudaMemcpyDeviceToHost);
      break;
    case kScaleVec128:
      s.inner_blocks = (inner + 127) / 128;
      s.values = read_as_double(p, outer * s.inner_blocks, CUDA_R_32F);
      break;
    default:   // kScaleBlk128x128
      s.inner_blocks = (inner + 127) / 128;
      s.values = read_as_double(p, (s.inner_blocks + 3) / 4 * 4 * ((outer + 127) / 128), CUDA_R_32F);
      break;
  }
  return s;
}

// D's block quantization (documentation's "1D Block Quantization"): each
// run of `block` elements down a column of D shares a scale computed from
// its absolute maximum, written to the D_OUT scale tensor in the tiled
// layout; a partial block is quantized as if the missing values were 0.
//   FP8 with UE8M0 (VEC32): S = amax / max(DType); its exponent field E, and
//     E + 1 if S is normal with a nonzero mantissa and E < 254, or denormal
//     with a mantissa above one half; the scale is 2^(E - 127), stored as E,
//     and each value is multiplied by its reciprocal and rounded.
//   FP4 with UE4M3 (VEC16): the values are first multiplied by D's input
//     scale (D_SCALE_POINTER); S = e4m3(amax / 6), and each value is
//     multiplied by 1 / S and rounded to E2M1.
void quantize_column(Elems& d, const MatrixLayout& l, int64_t j, const std::vector<double>& col, int mode,
                     std::vector<uint8_t>& out_scales, size_t m) {
  const int block = mode == kScaleVec16UE4M3 ? 16 : 32;
  const size_t blocks = (m + block - 1) / block, dim = (blocks + 3) / 4 * 4;
  const double dmax = is_fp4(d.t) ? 6.0 : d.t == CUDA_R_8F_E4M3 ? 448.0 : 57344.0;
  for (size_t b = 0; b < blocks; ++b) {
    double amax = 0;
    for (size_t i = b * block; i < std::min(m, (b + 1) * block); ++i) amax = std::fmax(amax, std::fabs(col[i]));
    const float s32 = (float)(amax / dmax);
    uint8_t code;
    double scale;
    if (mode == kScaleVec32UE8M0) {
      uint32_t bits;
      std::memcpy(&bits, &s32, 4);
      int e = (int)((bits >> 23) & 0xff);
      const uint32_t mant = bits & 0x7fffff;
      if (e > 0 && e < 254 && mant > 0) ++e;
      else if (e == 0 && mant > 0x400000) ++e;
      code = (uint8_t)e;
      scale = from_ue8m0(code);
    } else {
      code = to_bits_fp8(s32, true) & 0x7f;
      scale = from_ue4m3(code);
    }
    const size_t off = ((b / 4) * 4 + ((size_t)j / 128) * dim) * 128 + ((size_t)j % 32) * 16 + ((size_t)j % 128 / 32) * 4 +
                       b % 4;
    if (off < out_scales.size()) out_scales[off] = code;
    const double r = scale > 0 && std::isfinite(scale) ? 1.0 / (float)scale : 0.0;
    for (size_t i = b * block; i < std::min(m, (b + 1) * block); ++i) d.set(at(l, (int64_t)i, j), col[i] * r);
  }
}

/* ---- epilogues ----
   Forward: BIAS adds a vector down D's rows; RELU and GELU (the tanh
   approximation) apply pointwise; the _AUX forms also write the epilogue's
   input to the auxiliary buffer -- for ReLU a bit mask, one bit per element
   of D (set where the input is not negative: an input of 0 sets it, as on
   the card), column j starting at bit j * ld;
   for GELU the value itself (bias included), in D's type, column j at
   element j * ld.
   Backward: DRELU multiplies the matmul result by the mask an earlier
   RELU_AUX wrote; DGELU by GELU's derivative at the input an earlier
   GELU_AUX wrote; their _BGRAD forms also write each row's sum of the result
   to the bias pointer. BGRADA writes each row's sum over K of op(A), BGRADB
   each column's sum over K of op(B), to the bias pointer, alongside an
   ordinary matmul.
   What an RTX 3060's cuBLASLt (13.0) does, measured in fp16, bf16, fp32 and
   fp64, and done the same here: the ReLU mask's leading dimension must be a
   multiple of 128 and at least D's rows, GELU's at least D's rows (8
   divides neither requirement, whatever the header says), else
   INVALID_VALUE; GELU's auxiliary type must be D's; BGRADA takes op(A) = N
   only and BGRADB op(B) = T only, anything else NOT_SUPPORTED; a forward
   epilogue with no auxiliary pointer is NOT_SUPPORTED, a backward one with
   none, or a bias gradient with no bias pointer, INVALID_VALUE. The mask's
   bits past D's rows are left as they were. beta C is added before the mask
   or derivative is applied, and the bias gradient is not scaled by alpha.
   GELU and its derivative are computed exactly (std::tanh); the card's in
   fp32 differ from that by up to about 5e-5 relative, an approximate tanh,
   so those values agree to a tolerance rather than bit for bit. */
struct Epilogue {
  bool known = true, bias = false, relu = false, gelu = false, aux_out = false;
  bool drelu = false, dgelu = false, bgrad = false, bgrada = false, bgradb = false;
  bool relu_aux() const { return (relu && aux_out) || drelu; }
  bool gelu_aux() const { return (gelu && aux_out) || dgelu; }
};
Epilogue epilogue_of(cublasLtEpilogue_t e) {
  Epilogue x;
  switch ((int)e) {
    case CUBLASLT_EPILOGUE_DEFAULT: break;
    case CUBLASLT_EPILOGUE_RELU: x.relu = true; break;
    case CUBLASLT_EPILOGUE_BIAS: x.bias = true; break;
    case CUBLASLT_EPILOGUE_RELU_BIAS: x.relu = x.bias = true; break;
    case CUBLASLT_EPILOGUE_RELU_AUX: x.relu = x.aux_out = true; break;
    case CUBLASLT_EPILOGUE_RELU_AUX_BIAS: x.relu = x.aux_out = x.bias = true; break;
    case CUBLASLT_EPILOGUE_GELU: x.gelu = true; break;
    case CUBLASLT_EPILOGUE_GELU_BIAS: x.gelu = x.bias = true; break;
    case CUBLASLT_EPILOGUE_GELU_AUX: x.gelu = x.aux_out = true; break;
    case CUBLASLT_EPILOGUE_GELU_AUX_BIAS: x.gelu = x.aux_out = x.bias = true; break;
    case CUBLASLT_EPILOGUE_DRELU: x.drelu = true; break;
    case CUBLASLT_EPILOGUE_DRELU_BGRAD: x.drelu = x.bgrad = true; break;
    case CUBLASLT_EPILOGUE_DGELU: x.dgelu = true; break;
    case CUBLASLT_EPILOGUE_DGELU_BGRAD: x.dgelu = x.bgrad = true; break;
    case CUBLASLT_EPILOGUE_BGRADA: x.bgrada = true; break;
    case CUBLASLT_EPILOGUE_BGRADB: x.bgradb = true; break;
    default: x.known = false; break;
  }
  return x;
}

float gelu_grad(float x) {
  const float k = 0.7978845608f, c = 0.044715f;
  const float t = std::tanh(k * (x + c * x * x * x));
  return 0.5f * (1.0f + t) + 0.5f * x * (1.0f - t * t) * k * (1.0f + 3.0f * c * x * x);
}

bool block_mode(int m) { return m == kScaleVec16UE4M3 || m == kScaleVec32UE8M0; }
bool hopper_mode(int m) { return m == kScaleVec128 || m == kScaleBlk128x128; }

// Everything that can be refused before the matmul runs, shared with
// cublasLtMatmulAlgoGetHeuristic (which on the card refuses the same
// descriptors); `at_matmul` adds the pointer checks the card makes only
// there.
cublasStatus_t validate(const MatmulDesc& md, const MatrixLayout& la, const MatrixLayout& lb, const MatrixLayout& lc,
                        const MatrixLayout& ld, bool at_matmul) {
  if (!known_type(la.type) || !known_type(lb.type) || !known_type(lc.type) || !known_type(ld.type))
    return CUBLAS_STATUS_NOT_SUPPORTED;
  const bool ta = md.transa != CUBLAS_OP_N, tb = md.transb != CUBLAS_OP_N;
  const int64_t m = (int64_t)ld.rows, n = (int64_t)ld.cols, k = ta ? (int64_t)la.rows : (int64_t)la.cols;
  if ((ta ? (int64_t)la.cols : (int64_t)la.rows) != m || (tb ? (int64_t)lb.cols : (int64_t)lb.rows) != k ||
      (tb ? (int64_t)lb.rows : (int64_t)lb.cols) != n || (int64_t)lc.rows != m || (int64_t)lc.cols != n)
    return CUBLAS_STATUS_INVALID_VALUE;
  const Epilogue ep = epilogue_of(md.epilogue);
  if (!ep.known) return CUBLAS_STATUS_NOT_SUPPORTED;
  if ((ep.bgrada && ta) || (ep.bgradb && !tb)) return CUBLAS_STATUS_NOT_SUPPORTED;
  // The auxiliary buffer's shape is checked only once there is a buffer: with
  // none, the card's heuristic finds an algorithm and the matmul refuses.
  if (md.aux && ep.relu_aux() && (md.aux_ld <= 0 || md.aux_ld % 128 || md.aux_ld < m))
    return CUBLAS_STATUS_INVALID_VALUE;
  if (md.aux && ep.gelu_aux() &&
      (md.aux_ld <= 0 || md.aux_ld < m || (md.aux_type >= 0 && md.aux_type != (int32_t)ld.type)))
    return CUBLAS_STATUS_INVALID_VALUE;
  // The scaling modes (documentation-derived for the block forms).
  const int am = md.a_scale_mode, bm = md.b_scale_mode;
  auto mode_ok = [](int x) { return x == kScaleScalar || x == kScaleOuterVec || block_mode(x) || hopper_mode(x); };
  if (!mode_ok(am) || !mode_ok(bm) || !(md.c_scale_mode == kScaleScalar || block_mode(md.c_scale_mode)) ||
      md.d_scale_mode != kScaleScalar || !(md.d_out_scale_mode == kScaleScalar || block_mode(md.d_out_scale_mode)))
    return CUBLAS_STATUS_NOT_SUPPORTED;
  if (block_mode(am) || block_mode(bm)) {
    // FP4 with UE4M3 per 16, or FP8 with UE8M0 per 32; A and B alike.
    if (am != bm) return CUBLAS_STATUS_NOT_SUPPORTED;
    const bool fp4 = am == kScaleVec16UE4M3;
    if (fp4 ? !(is_fp4(la.type) && is_fp4(lb.type)) : !(is_fp8(la.type) && is_fp8(lb.type)))
      return CUBLAS_STATUS_NOT_SUPPORTED;
    if (ld.batch > 1) return CUBLAS_STATUS_NOT_SUPPORTED;   // per-batch scale tensors are not modelled
  } else if (hopper_mode(am) || hopper_mode(bm)) {
    if (!hopper_mode(am) || !hopper_mode(bm) || (am == kScaleBlk128x128 && bm == kScaleBlk128x128) ||
        !is_fp8(la.type) || !is_fp8(lb.type) || narrow(ld.type) || ld.batch > 1)
      return CUBLAS_STATUS_NOT_SUPPORTED;
    if (m % 4 || n % 4) return CUBLAS_STATUS_INVALID_VALUE;
  } else if (is_fp4(la.type) || is_fp4(lb.type)) {
    return CUBLAS_STATUS_NOT_SUPPORTED;   // FP4 operands come only with their block scales
  }
  if (block_mode(md.c_scale_mode) &&
      (md.c_scale_mode == kScaleVec16UE4M3 ? !is_fp4(lc.type) : !is_fp8(lc.type)))
    return CUBLAS_STATUS_NOT_SUPPORTED;
  if (block_mode(md.d_out_scale_mode)
          ? (md.d_out_scale_mode == kScaleVec16UE4M3 ? !is_fp4(ld.type) : !is_fp8(ld.type)) || ld.batch > 1
          : is_fp4(ld.type))
    return CUBLAS_STATUS_NOT_SUPPORTED;
  const int pm = md.pointer_mode;
  if (pm < CUBLASLT_POINTER_MODE_HOST || pm > CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST)
    return CUBLAS_STATUS_INVALID_VALUE;
  if (at_matmul) {
    if (ep.aux_out && !md.aux) return CUBLAS_STATUS_NOT_SUPPORTED;
    if ((ep.drelu || ep.dgelu) && !md.aux) return CUBLAS_STATUS_INVALID_VALUE;
    if ((ep.bgrad || ep.bgrada || ep.bgradb) && !md.bias) return CUBLAS_STATUS_INVALID_VALUE;
    if (block_mode(md.d_out_scale_mode) && !md.d_out_scale) return CUBLAS_STATUS_INVALID_VALUE;
    if ((block_mode(am) || hopper_mode(am)) && (!md.a_scale || !md.b_scale)) return CUBLAS_STATUS_INVALID_VALUE;
  }
  return CUBLAS_STATUS_SUCCESS;
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

  const cublasStatus_t verdict = validate(md, la, lb, lc, ld, true);
  if (verdict != CUBLAS_STATUS_SUCCESS) {
    if (trace() || (!quiet() && verdict == CUBLAS_STATUS_NOT_SUPPORTED))
      std::fprintf(stderr, "[vgpu] cublasLtMatmul: refused, status %d (A=%d B=%d C=%d D=%d, epilogue %d)\n",
                   (int)verdict, (int)la.type, (int)lb.type, (int)lc.type, (int)ld.type, (int)md.epilogue);
    return verdict;
  }
  const bool ta = md.transa != CUBLAS_OP_N, tb = md.transb != CUBLAS_OP_N;
  const int64_t m = (int64_t)ld.rows, n = (int64_t)ld.cols;
  const int64_t k = ta ? (int64_t)la.rows : (int64_t)la.cols;
  const Epilogue ep = epilogue_of(md.epilogue);
  const int pm = md.pointer_mode;

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

  // A's and B's scales (row-wise, column-wise or by block along K; an unset
  // scale is 1), C's (scalar, or by block down C's columns, for a narrow C),
  // and D's: its tensor-wide scale for an FP8 D, or the input scale an FP4 D
  // is quantized with.
  const Scales sa = load_scales(md.a_scale, md.a_scale_mode, (size_t)m, (size_t)k);
  const Scales sb = load_scales(md.b_scale, md.b_scale_mode, (size_t)n, (size_t)k);
  const bool c_blocks = block_mode(md.c_scale_mode);
  const Scales sc = c_blocks ? load_scales(md.c_scale, md.c_scale_mode, (size_t)n, (size_t)m)
                             : load_scales(narrow(lc.type) ? md.c_scale : nullptr, kScaleScalar, 1, 1);
  const bool d_blocks = block_mode(md.d_out_scale_mode);
  const double sd = (is_fp8(ld.type) && !d_blocks) || (is_fp4(ld.type) && d_blocks)
                        ? load_scales(md.d_scale, kScaleScalar, 1, 1).values[0]
                        : 1.0;

  cudaDataType bias_type = (cudaDataType)md.bias_type;
  if (md.bias_type < 0) bias_type = narrow(ld.type) ? CUDA_R_16BF : ld.type;
  if ((ep.bias || ep.bgrad || ep.bgrada || ep.bgradb) && !elem_bytes(bias_type)) return CUBLAS_STATUS_NOT_SUPPORTED;

  const int batch = ld.batch;
  if ((la.batch != 1 && la.batch != batch) || (lb.batch != 1 && lb.batch != batch)) return CUBLAS_STATUS_INVALID_VALUE;
  double amax = 0.0;
  std::vector<uint8_t> out_scales;
  if (d_blocks)
    out_scales.assign(tiled_bytes((size_t)n, ((size_t)m + (md.d_out_scale_mode == kScaleVec16UE4M3 ? 15 : 31)) /
                                                 (md.d_out_scale_mode == kScaleVec16UE4M3 ? 16 : 32)),
                      0);
  for (int bi = 0; bi < batch; ++bi) {
    auto base = [&](const void* p, const MatrixLayout& l) { return static_cast<const uint8_t*>(p) + batch_offset_bytes(l, bi); };
    Elems ea, eb, ec, ed;
    ea.load(base(A, la), span(la), la.type);
    eb.load(base(B, lb), span(lb), lb.type);
    if (C && any_beta) ec.load(base(C, lc), span(lc), lc.type);
    // D is read too, so the gaps a leading dimension leaves come back untouched.
    uint8_t* dptr = static_cast<uint8_t*>(D) + batch_offset_bytes(ld, bi);
    ed.load(dptr, span(ld), ld.type);

    // The auxiliary buffer and the bias (or its gradient) of this batch.
    const bool relu_aux = ep.relu_aux(), gelu_aux = ep.gelu_aux();
    const int64_t aux_ld = md.aux_ld;
    std::vector<uint8_t> mask;
    Elems aux;
    uint8_t* aux_base = nullptr;
    if (relu_aux) {
      aux_base = static_cast<uint8_t*>(md.aux) + (size_t)bi * (size_t)md.aux_batch_stride / 8;
      mask.assign((size_t)((n - 1) * aux_ld + m + 7) / 8, 0);
      cudaMemcpy(mask.data(), aux_base, mask.size(), cudaMemcpyDeviceToHost);
    } else if (gelu_aux) {
      aux_base = static_cast<uint8_t*>(md.aux) + (size_t)bi * (size_t)md.aux_batch_stride * elem_bytes(ld.type);
      aux.load(aux_base, (size_t)((n - 1) * aux_ld + m), ld.type);
    }
    std::vector<double> hbias;
    if (ep.bias && md.bias)
      hbias = read_as_double(static_cast<const uint8_t*>(md.bias) + (size_t)bi * md.bias_batch_stride * elem_bytes(bias_type),
                             (size_t)m, bias_type);
    std::vector<double> bgrad((size_t)(ep.bgradb ? n : m), 0.0);

    std::vector<double> col((size_t)m);
    for (int64_t j = 0; j < n; ++j) {
      for (int64_t i = 0; i < m; ++i) {
        double acc = 0.0;
        for (int64_t p = 0; p < k; ++p)
          acc += ea.get(ta ? at(la, p, i) : at(la, i, p)) * sa.at((size_t)i, (size_t)p) *
                 eb.get(tb ? at(lb, j, p) : at(lb, p, j)) * sb.at((size_t)j, (size_t)p);
        const double al = alphas[alpha_vec ? (size_t)i : 0], be = betas[beta_vec ? (size_t)i : 0];
        double v = al * acc;
        if (be != 0.0 && !ec.raw.empty()) v += be * sc.at((size_t)j, (size_t)i) * ec.get(at(lc, i, j));
        if (!hbias.empty()) v += hbias[(size_t)i];
        const size_t ax = (size_t)(j * aux_ld + i);
        if (ep.relu) {
          if (relu_aux) {
            uint8_t& byte = mask[ax / 8];
            byte = (uint8_t)((byte & ~(1u << (ax % 8))) | ((v < 0.0 ? 0u : 1u) << (ax % 8)));
          }
          v = v > 0.0 ? v : 0.0;
        } else if (ep.gelu) {
          if (gelu_aux) aux.set(ax, v);
          v = gelu((float)v);
        } else if (ep.drelu) {
          if (!((mask[ax / 8] >> (ax % 8)) & 1)) v = 0.0;
        } else if (ep.dgelu) {
          v *= gelu_grad((float)aux.get(ax));
        }
        if (ep.bgrad) bgrad[(size_t)i] += v;
        amax = std::fmax(amax, std::fabs(v));
        col[(size_t)i] = v;
      }
      if (d_blocks) {   // an FP4 D's input scale applies before the quantization
        if (sd != 1.0)
          for (auto& v : col) v *= sd;
        quantize_column(ed, ld, j, col, md.d_out_scale_mode, out_scales, (size_t)m);
      } else {
        for (int64_t i = 0; i < m; ++i) ed.set(at(ld, i, j), col[(size_t)i] * sd);
      }
    }
    ed.save(dptr);
    if (relu_aux && ep.relu) cudaMemcpy(aux_base, mask.data(), mask.size(), cudaMemcpyHostToDevice);
    if (gelu_aux && ep.gelu) aux.save(aux_base);
    // The bias gradients: of the epilogue's output, or of op(A) or op(B) over K.
    if (ep.bgrada)
      for (int64_t i = 0; i < m; ++i)
        for (int64_t p = 0; p < k; ++p) bgrad[(size_t)i] += ea.get(ta ? at(la, p, i) : at(la, i, p));
    if (ep.bgradb)
      for (int64_t j = 0; j < n; ++j)
        for (int64_t p = 0; p < k; ++p) bgrad[(size_t)j] += eb.get(tb ? at(lb, j, p) : at(lb, p, j));
    if (ep.bgrad || ep.bgrada || ep.bgradb) {
      Elems g;
      g.t = bias_type;
      g.raw.assign(bgrad.size() * elem_bytes(bias_type), 0);
      for (size_t i = 0; i < bgrad.size(); ++i) g.set(i, bgrad[i]);
      g.save(static_cast<uint8_t*>(const_cast<void*>(md.bias)) + (size_t)bi * md.bias_batch_stride * elem_bytes(bias_type));
    }
  }
  if (d_blocks) cudaMemcpy(md.d_out_scale, out_scales.data(), out_scales.size(), cudaMemcpyHostToDevice);
  if (md.amax_d) {
    const float a = (float)amax;
    cudaMemcpy(md.amax_d, &a, sizeof a, cudaMemcpyHostToDevice);
  }
  return CUBLAS_STATUS_SUCCESS;
}

// One algorithm is offered: there is nothing to tune when the math runs on the
// host, and callers only need a valid heuristic result to proceed. A
// descriptor the matmul would refuse gets no algorithm and the matmul's
// answer, as on the card.
VGPU_EXPORT cublasStatus_t cublasLtMatmulAlgoGetHeuristic(
    cublasLtHandle_t, cublasLtMatmulDesc_t desc, cublasLtMatrixLayout_t Adesc, cublasLtMatrixLayout_t Bdesc,
    cublasLtMatrixLayout_t Cdesc, cublasLtMatrixLayout_t Ddesc, cublasLtMatmulPreference_t, int requested,
    cublasLtMatmulHeuristicResult_t* results, int* returned) {
  if (!results || !returned || requested < 1) return CUBLAS_STATUS_INVALID_VALUE;
  *returned = 0;
  if (known(desc) && known(Adesc) && known(Bdesc) && known(Ddesc)) {
    const auto& ld = *reinterpret_cast<MatrixLayout*>(Ddesc);
    const cublasStatus_t st =
        validate(*reinterpret_cast<MatmulDesc*>(desc), *reinterpret_cast<MatrixLayout*>(Adesc),
                 *reinterpret_cast<MatrixLayout*>(Bdesc), known(Cdesc) ? *reinterpret_cast<MatrixLayout*>(Cdesc) : ld,
                 ld, false);
    if (st != CUBLAS_STATUS_SUCCESS) return st;
  }
  std::memset(&results[0], 0, sizeof(results[0]));
  results[0].state = CUBLAS_STATUS_SUCCESS;
  results[0].workspaceSize = 0;
  results[0].wavesCount = 1.0f;
  *returned = 1;
  return CUBLAS_STATUS_SUCCESS;
}
