// libvgpucusparse -- VirtualGPU's cuSPARSE, presented as libcusparse.so.12.
//
// The generic (descriptor) API: CSR, CSC, COO and BSR sparse matrices, dense
// vectors and matrices, strided batches of them, SpMV, SpMM, SpGEMM, SDDMM,
// triangular solves (SpSV, SpSM) and conversion in both directions, in real and
// complex values. Of the legacy API, what CUDA 12 and 13 still ship: matrix
// descriptors, coo2csr, the COO and CSR sorts, csrgeam2 (C = alpha A + beta B),
// the CSR transpose csr2cscEx2 (which SCS calls), the blocked (BSR) routines --
// bsrmv, bsrxmv, bsrmm, bsrsv2, bsrsm2, bsric02, bsrilu02 and the CSR <-> BSR
// conversions -- and their CSR counterparts csric02 and csrilu02.
//
// Like the other vendor libraries the arithmetic runs on the host, in double
// (std::complex<double> for complex values), and is rounded to the requested
// type on the way out. Anything unimplemented returns
// CUSPARSE_STATUS_NOT_SUPPORTED so a caller can fall back rather than receive a
// plausible wrong answer.
#include <cusparse.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <functional>
#include <mutex>
#include <map>
#include <memory>
#include <set>
#include <type_traits>
#include <vector>

#include <cuda_runtime.h>

#include "vgpu/runtime/capture.hpp"

namespace {

bool quiet() { const char* q = std::getenv("VGPU_QUIET"); return q && q[0] == '1'; }

using cd = std::complex<double>;

// The generic API's BSR format, CUSPARSE_FORMAT_BSR. CUDA 12.0's header has no
// name for it (BSR arrived in the generic API in 12.1), and the value is ABI.
constexpr cusparseFormat_t kFormatBsr = static_cast<cusparseFormat_t>(6);

struct SpMat {
  cusparseFormat_t format = CUSPARSE_FORMAT_CSR;
  int64_t rows = 0, cols = 0, nnz = 0;  // BSR: in elements, nnz = blocks * bdim^2
  void* rows_ptr = nullptr;   // CSR/BSR row offsets, CSC column offsets, or COO row indices
  void* cols_ptr = nullptr;
  void* values = nullptr;
  cusparseIndexType_t row_type = CUSPARSE_INDEX_32I;
  cusparseIndexType_t col_type = CUSPARSE_INDEX_32I;
  cusparseIndexBase_t base = CUSPARSE_INDEX_BASE_ZERO;
  cudaDataType value_type = CUDA_R_32F;
  cusparseFillMode_t fill = CUSPARSE_FILL_MODE_LOWER;     // read by SpSV/SpSM only
  cusparseDiagType_t diag = CUSPARSE_DIAG_TYPE_NON_UNIT;
  // A strided batch: matrix b's offsets (CSR, BSR) start b * off_stride
  // elements in, its indices b * col_stride and its values b * val_stride.
  // COO steps all three by one stride.
  int batch = 1;
  int64_t off_stride = 0, val_stride = 0, col_stride = 0;
  // BSR: square blocks of bdim, each stored row- or column-major.
  int64_t bdim = 1;
  cusparseOrder_t block_order = CUSPARSE_ORDER_ROW;
};

struct DnVec {
  int64_t size = 0;
  void* values = nullptr;
  cudaDataType type = CUDA_R_32F;
};

struct DnMat {
  int64_t rows = 0, cols = 0, ld = 0;
  void* values = nullptr;
  cudaDataType type = CUDA_R_32F;
  cusparseOrder_t order = CUSPARSE_ORDER_COL;
  int batch = 1;
  int64_t batch_stride = 0;  // in elements
};

struct Handle {
  cudaStream_t stream = nullptr;
  cusparsePointerMode_t mode = CUSPARSE_POINTER_MODE_HOST;
};

// The live-descriptor registry is never destroyed: a graph a program leaks
// holds copies of descriptors (see Held), and it is released by libcudart's
// destructors at exit, which may run after this library's.
std::mutex& g_mu = *new std::mutex;
std::set<const void*>& g_live = *new std::set<const void*>;
template <class T> T* track(T* p) { std::lock_guard<std::mutex> l(g_mu); g_live.insert(p); return p; }
bool known(const void* p) { std::lock_guard<std::mutex> l(g_mu); return p && g_live.count(p); }
void untrack(const void* p) { std::lock_guard<std::mutex> l(g_mu); g_live.erase(p); }

bool is_complex(cudaDataType t) {
  return t == CUDA_C_16F || t == CUDA_C_16BF || t == CUDA_C_32F || t == CUDA_C_64F;
}
size_t type_bytes(cudaDataType t) {
  switch (t) {
    case CUDA_R_16F: case CUDA_R_16BF: return 2;
    case CUDA_R_32F: case CUDA_C_16F: case CUDA_C_16BF: return 4;
    case CUDA_R_64F: case CUDA_C_32F: return 8;
    case CUDA_C_64F: return 16;
    default: return 0;
  }
}
size_t index_bytes(cusparseIndexType_t t) {
  switch (t) {
    case CUSPARSE_INDEX_32I: return 4;
    case CUSPARSE_INDEX_64I: return 8;
    default: return 0;
  }
}

// Half and bfloat16 values, converted by hand so no CUDA half header is needed.
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
  if ((x & 0x7fffffff) > 0x7f800000) return (uint16_t)((x >> 16) | 0x40);  // NaN stays NaN
  x += 0x7fff + ((x >> 16) & 1);  // round to nearest even
  return (uint16_t)(x >> 16);
}

// One real component (a real value, or half of a complex one) of type t.
double get_part(const uint8_t* p, cudaDataType t) {
  switch (t) {
    case CUDA_R_16F: case CUDA_C_16F: { uint16_t h; std::memcpy(&h, p, 2); return from_half(h); }
    case CUDA_R_16BF: case CUDA_C_16BF: { uint16_t h; std::memcpy(&h, p, 2); return from_bf16(h); }
    case CUDA_R_32F: case CUDA_C_32F: { float f; std::memcpy(&f, p, 4); return f; }
    default: { double d; std::memcpy(&d, p, 8); return d; }
  }
}
void put_part(uint8_t* p, double v, cudaDataType t) {
  switch (t) {
    case CUDA_R_16F: case CUDA_C_16F: { const uint16_t h = to_half(v); std::memcpy(p, &h, 2); return; }
    case CUDA_R_16BF: case CUDA_C_16BF: { const uint16_t h = to_bf16(v); std::memcpy(p, &h, 2); return; }
    case CUDA_R_32F: case CUDA_C_32F: { const float f = (float)v; std::memcpy(p, &f, 4); return; }
    default: std::memcpy(p, &v, 8); return;
  }
}

// The arithmetic runs in V: double for real values, cd for complex ones. A
// real operand read as cd has a zero imaginary part; a cd stored into a real
// type keeps its real part.
inline double re(double v) { return v; }
inline double re(cd v) { return v.real(); }
inline double im(double) { return 0.0; }
inline double im(cd v) { return v.imag(); }
template <class V> V make_value(double r, double i);
template <> inline double make_value<double>(double r, double) { return r; }
template <> inline cd make_value<cd>(double r, double i) { return {r, i}; }
// op(A)'s element: conjugated under CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE.
inline double conj_op(double v, cusparseOperation_t) { return v; }
inline cd conj_op(cd v, cusparseOperation_t op) {
  return op == CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE ? std::conj(v) : v;
}
inline double conj_of(double v) { return v; }
inline cd conj_of(cd v) { return std::conj(v); }

template <class V> bool decode(const uint8_t* raw, size_t n, cudaDataType t, std::vector<V>* out) {
  const size_t eb = type_bytes(t);
  if (!eb) return false;
  const bool cplx = is_complex(t);
  if (cplx && std::is_same_v<V, double>) return false;  // a complex value has no real-only form
  out->resize(n);
  for (size_t i = 0; i < n; ++i) {
    const uint8_t* p = raw + i * eb;
    (*out)[i] = make_value<V>(get_part(p, t), cplx ? get_part(p + eb / 2, t) : 0.0);
  }
  return true;
}
template <class V> void encode(const std::vector<V>& in, cudaDataType t, uint8_t* raw) {
  const size_t eb = type_bytes(t);
  const bool cplx = is_complex(t);
  for (size_t i = 0; i < in.size(); ++i) {
    put_part(raw + i * eb, re(in[i]), t);
    if (cplx) put_part(raw + i * eb + eb / 2, im(in[i]), t);
  }
}

template <class V> bool fetch_values(const void* dev, size_t n, cudaDataType t, std::vector<V>* out) {
  out->assign(n, V{});
  if (!n) return true;
  const size_t eb = type_bytes(t);
  if (!eb) return false;
  std::vector<uint8_t> raw(n * eb);
  if (cudaMemcpy(raw.data(), dev, raw.size(), cudaMemcpyDeviceToHost) != cudaSuccess) return false;
  return decode(raw.data(), n, t, out);
}

template <class V> bool store_values(void* dev, const std::vector<V>& in, cudaDataType t) {
  if (in.empty()) return true;
  const size_t eb = type_bytes(t);
  if (!eb) return false;
  std::vector<uint8_t> raw(in.size() * eb);
  encode(in, t, raw.data());
  return cudaMemcpy(dev, raw.data(), raw.size(), cudaMemcpyHostToDevice) == cudaSuccess;
}

bool fetch_indices(const void* dev, size_t n, cusparseIndexType_t t, std::vector<int64_t>* out) {
  out->assign(n, 0);
  if (!n) return true;
  if (t == CUSPARSE_INDEX_32I) {
    std::vector<int32_t> tmp(n);
    if (cudaMemcpy(tmp.data(), dev, n * 4, cudaMemcpyDeviceToHost) != cudaSuccess) return false;
    for (size_t i = 0; i < n; ++i) (*out)[i] = tmp[i];
    return true;
  }
  if (t == CUSPARSE_INDEX_64I)
    return cudaMemcpy(out->data(), dev, n * 8, cudaMemcpyDeviceToHost) == cudaSuccess;
  return false;
}

// alpha and beta, read from host or device memory as the handle's pointer
// mode says.
template <class V> V scalar(cusparseHandle_t h, const void* p, cudaDataType t) {
  if (!p) return V(1.0);
  std::vector<V> one;
  if (reinterpret_cast<const Handle*>(h)->mode == CUSPARSE_POINTER_MODE_DEVICE)
    return fetch_values(p, 1, t, &one) ? one[0] : V(NAN);
  return decode(static_cast<const uint8_t*>(p), 1, t, &one) ? one[0] : V(NAN);
}

// Every sparse format is expanded to a coordinate list once, and both SpMV and
// SpMM work from that. It costs an extra pass over nnz and removes a whole
// class of format-specific indexing bugs. Element k of the list is element k
// of the values array, so a result computed per entry (SDDMM) is stored back
// as it is.
template <class V> struct Triplets {
  std::vector<int64_t> row, col;
  std::vector<V> val;
};

template <class V> bool expand(const SpMat& a, Triplets<V>* t) {
  std::vector<V> vals;
  if (!fetch_values(a.values, (size_t)a.nnz, a.value_type, &vals)) return false;
  const int64_t off = a.base == CUSPARSE_INDEX_BASE_ONE ? 1 : 0;
  t->val = std::move(vals);
  t->col.resize((size_t)a.nnz);
  t->row.resize((size_t)a.nnz);

  if (a.format == kFormatBsr) {
    // Square blocks, each stored whole in the block order; every element of a
    // stored block is an entry, zero or not.
    const int64_t bd = a.bdim, bs = bd * bd, brows = a.rows / bd, bnnz = a.nnz / bs;
    std::vector<int64_t> offs, cols;
    if (!fetch_indices(a.rows_ptr, (size_t)brows + 1, a.row_type, &offs) ||
        !fetch_indices(a.cols_ptr, (size_t)bnnz, a.col_type, &cols))
      return false;
    const bool row_major = a.block_order == CUSPARSE_ORDER_ROW;
    for (int64_t br = 0; br < brows; ++br)
      for (int64_t k = offs[(size_t)br] - off; k < offs[(size_t)br + 1] - off; ++k) {
        if (k < 0 || k >= bnnz) return false;
        const int64_t bc = cols[(size_t)k] - off;
        for (int64_t s = 0; s < bs; ++s) {
          const int64_t r = row_major ? s / bd : s % bd, c = row_major ? s % bd : s / bd;
          t->row[(size_t)(k * bs + s)] = br * bd + r;
          t->col[(size_t)(k * bs + s)] = bc * bd + c;
        }
      }
    return true;
  }

  std::vector<int64_t> cols;
  if (!fetch_indices(a.cols_ptr, (size_t)a.nnz, a.col_type, &cols)) return false;
  for (int64_t i = 0; i < a.nnz; ++i) t->col[(size_t)i] = cols[(size_t)i] - off;

  if (a.format == CUSPARSE_FORMAT_COO) {
    std::vector<int64_t> rows;
    if (!fetch_indices(a.rows_ptr, (size_t)a.nnz, a.row_type, &rows)) return false;
    for (int64_t i = 0; i < a.nnz; ++i) t->row[(size_t)i] = rows[(size_t)i] - off;
    return true;
  }
  if (a.format == CUSPARSE_FORMAT_CSC) {
    // CSC stores column offsets and row indices: the roles are swapped.
    std::vector<int64_t> offs;
    if (!fetch_indices(a.rows_ptr, (size_t)a.cols + 1, a.row_type, &offs)) return false;
    for (int64_t c = 0; c < a.cols; ++c)
      for (int64_t k = offs[(size_t)c] - off; k < offs[(size_t)c + 1] - off; ++k) {
        if (k < 0 || k >= a.nnz) return false;
        t->row[(size_t)k] = t->col[(size_t)k];
        t->col[(size_t)k] = c;
      }
    return true;
  }
  std::vector<int64_t> offs;
  if (!fetch_indices(a.rows_ptr, (size_t)a.rows + 1, a.row_type, &offs)) return false;
  for (int64_t r = 0; r < a.rows; ++r)
    for (int64_t k = offs[(size_t)r] - off; k < offs[(size_t)r + 1] - off; ++k) {
      if (k < 0 || k >= a.nnz) return false;
      t->row[(size_t)k] = r;
    }
  return true;
}

bool transposed(cusparseOperation_t op) {
  return op == CUSPARSE_OPERATION_TRANSPOSE || op == CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE;
}

// Dense element addressing that honours both storage orders.
inline size_t dn_index(const DnMat& m, int64_t r, int64_t c) {
  return m.order == CUSPARSE_ORDER_COL ? (size_t)c * (size_t)m.ld + (size_t)r
                                       : (size_t)r * (size_t)m.ld + (size_t)c;
}
size_t dn_elems(const DnMat& m) {
  const int64_t lead = m.order == CUSPARSE_ORDER_COL ? m.cols : m.rows;
  return (size_t)lead * (size_t)m.ld;
}

// ---- CUDA graph capture ----
//
// On hardware a cuSPARSE call made while its stream is capturing is recorded
// into the graph and runs at each launch, over whatever the graph's kernels
// have produced by then. Computing it at call time instead read inputs that
// did not exist yet and left the call out of every replay: HiGHS's HiPDLP
// solver captures its iterations, SpMV included, and went to NaN here while
// converging on an RTX 3060.
//
// So such a call is recorded as a closure over copies of what it was given:
// the descriptors as they are now (a later cusparseDnVecSetValues does not
// reach into a graph on hardware either), and host-mode scalars, whose
// variables are usually gone by the time the graph runs. Device memory, and
// device-mode scalars, are read when the graph runs.
class Held {
 public:
  template <class T> T* copy(const void* p) {
    std::shared_ptr<T> c(track(new T(*static_cast<const T*>(p))), [](T* q) {
      untrack(q);
      delete q;
    });
    keep_.push_back(c);
    return c.get();
  }
  const void* scalar(const Handle& h, const void* p, size_t bytes) {
    if (!p || bytes == 0 || h.mode == CUSPARSE_POINTER_MODE_DEVICE) return p;
    auto v = std::make_shared<std::vector<uint8_t>>(static_cast<const uint8_t*>(p),
                                                    static_cast<const uint8_t*>(p) + bytes);
    keep_.push_back(v);
    return v->data();
  }
  // The handle the recorded call runs with: the caller's pointer mode, and no
  // stream, so running it is not taken for another capture.
  cusparseHandle_t handle(cusparseHandle_t h) {
    Handle* c = copy<Handle>(h);
    c->stream = nullptr;
    return reinterpret_cast<cusparseHandle_t>(c);
  }

 private:
  std::vector<std::shared_ptr<void>> keep_;
};

bool capturing(cusparseHandle_t h) {
  cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
  return cudaStreamIsCapturing(reinterpret_cast<const Handle*>(h)->stream, &st) == cudaSuccess &&
         st == cudaStreamCaptureStatusActive;
}

// The closure owns `held`, so the copies it points into live as long as the
// graph does.
cusparseStatus_t record(cusparseHandle_t h, std::shared_ptr<Held> held, std::function<void()> call) {
  std::function<void()> op = [held = std::move(held), call = std::move(call)] { call(); };
  return vgpu_record_host_op_if_capturing(
             reinterpret_cast<CUstream_st*>(reinterpret_cast<const Handle*>(h)->stream), std::move(op))
             ? CUSPARSE_STATUS_SUCCESS
             : CUSPARSE_STATUS_EXECUTION_FAILED;
}

}  // namespace

#define VGPU_EXPORT extern "C" __attribute__((visibility("default")))

/* ---- handle ---- */

VGPU_EXPORT cusparseStatus_t cusparseCreate(cusparseHandle_t* h) {
  if (!h) return CUSPARSE_STATUS_INVALID_VALUE;
  *h = reinterpret_cast<cusparseHandle_t>(track(new Handle()));
  if (!quiet())
    std::fprintf(stderr, "[vgpu] cuSPARSE handle created (host-computed)\n");
  return CUSPARSE_STATUS_SUCCESS;
}
VGPU_EXPORT cusparseStatus_t cusparseDestroy(cusparseHandle_t h) {
  if (!known(h)) return CUSPARSE_STATUS_NOT_INITIALIZED;
  untrack(h); delete reinterpret_cast<Handle*>(h);
  return CUSPARSE_STATUS_SUCCESS;
}
VGPU_EXPORT cusparseStatus_t cusparseSetStream(cusparseHandle_t h, cudaStream_t s) {
  if (!known(h)) return CUSPARSE_STATUS_NOT_INITIALIZED;
  reinterpret_cast<Handle*>(h)->stream = s;
  return CUSPARSE_STATUS_SUCCESS;
}
VGPU_EXPORT cusparseStatus_t cusparseGetStream(cusparseHandle_t h, cudaStream_t* s) {
  if (!known(h)) return CUSPARSE_STATUS_NOT_INITIALIZED;
  if (s) *s = reinterpret_cast<Handle*>(h)->stream;
  return CUSPARSE_STATUS_SUCCESS;
}
VGPU_EXPORT cusparseStatus_t cusparseGetVersion(cusparseHandle_t h, int* v) {
  if (!known(h)) return CUSPARSE_STATUS_NOT_INITIALIZED;
  if (v) *v = CUSPARSE_VERSION;
  return CUSPARSE_STATUS_SUCCESS;
}
VGPU_EXPORT cusparseStatus_t cusparseGetProperty(libraryPropertyType type, int* v) {
  if (!v) return CUSPARSE_STATUS_INVALID_VALUE;
  switch (type) {
    case MAJOR_VERSION: *v = CUSPARSE_VER_MAJOR; break;
    case MINOR_VERSION: *v = CUSPARSE_VER_MINOR; break;
    case PATCH_LEVEL: *v = CUSPARSE_VER_PATCH; break;
    default: return CUSPARSE_STATUS_INVALID_VALUE;
  }
  return CUSPARSE_STATUS_SUCCESS;
}
VGPU_EXPORT const char* cusparseGetErrorString(cusparseStatus_t s) {
  switch (s) {
    case CUSPARSE_STATUS_SUCCESS: return "CUSPARSE_STATUS_SUCCESS";
    case CUSPARSE_STATUS_NOT_INITIALIZED: return "CUSPARSE_STATUS_NOT_INITIALIZED";
    case CUSPARSE_STATUS_INVALID_VALUE: return "CUSPARSE_STATUS_INVALID_VALUE";
    case CUSPARSE_STATUS_NOT_SUPPORTED: return "CUSPARSE_STATUS_NOT_SUPPORTED";
    default: return "CUSPARSE_STATUS_INTERNAL_ERROR";
  }
}
VGPU_EXPORT const char* cusparseGetErrorName(cusparseStatus_t s) { return cusparseGetErrorString(s); }

/* ---- sparse matrix descriptors ---- */

static cusparseStatus_t make_sparse(cusparseSpMatDescr_t* out, cusparseFormat_t fmt, int64_t rows,
                                    int64_t cols, int64_t nnz, void* offsets, void* indices,
                                    void* values, cusparseIndexType_t off_type,
                                    cusparseIndexType_t idx_type, cusparseIndexBase_t base,
                                    cudaDataType vt) {
  if (!out || rows < 0 || cols < 0 || nnz < 0) return CUSPARSE_STATUS_INVALID_VALUE;
  if (!type_bytes(vt) || !index_bytes(off_type) || !index_bytes(idx_type)) {
    std::fprintf(stderr, "[vgpu] cusparse: only half, bfloat16, float and double values, real or "
                         "complex, with 32- or 64-bit indices are implemented\n");
    return CUSPARSE_STATUS_NOT_SUPPORTED;
  }
  auto* m = new SpMat();
  *m = SpMat{fmt, rows, cols, nnz, offsets, indices, values, off_type, idx_type, base, vt};
  *out = reinterpret_cast<cusparseSpMatDescr_t>(track(m));
  return CUSPARSE_STATUS_SUCCESS;
}

// BSR, as CUDA 12.1 added it to the generic API: square blocks only (an RTX
// 3060's cuSPARSE 13.0 refuses rowBlockSize != colBlockSize with
// INVALID_VALUE), sizes counted in elements once created (rows = brows *
// blockSize, nnz = bnnz * blockSize^2).
static cusparseStatus_t make_bsr(cusparseSpMatDescr_t* out, int64_t brows, int64_t bcols, int64_t bnnz,
                                 int64_t rb, int64_t cb, void* offsets, void* cols, void* values,
                                 cusparseIndexType_t ot, cusparseIndexType_t ct, cusparseIndexBase_t base,
                                 cudaDataType vt, cusparseOrder_t order) {
  if (!out || brows < 0 || bcols < 0 || bnnz < 0 || rb <= 0 || cb <= 0 || rb != cb ||
      (order != CUSPARSE_ORDER_ROW && order != CUSPARSE_ORDER_COL))
    return CUSPARSE_STATUS_INVALID_VALUE;
  const cusparseStatus_t rc =
      make_sparse(out, kFormatBsr, brows * rb, bcols * rb, bnnz * rb * rb, offsets, cols, values, ot, ct, base, vt);
  if (rc == CUSPARSE_STATUS_SUCCESS) {
    auto* m = reinterpret_cast<SpMat*>(*out);
    m->bdim = rb;
    m->block_order = order;
  }
  return rc;
}

VGPU_EXPORT cusparseStatus_t cusparseCreateCsr(cusparseSpMatDescr_t* d, int64_t rows, int64_t cols,
                                               int64_t nnz, void* off, void* col, void* val,
                                               cusparseIndexType_t ot, cusparseIndexType_t ct,
                                               cusparseIndexBase_t base, cudaDataType vt) {
  return make_sparse(d, CUSPARSE_FORMAT_CSR, rows, cols, nnz, off, col, val, ot, ct, base, vt);
}
VGPU_EXPORT cusparseStatus_t cusparseCreateCsc(cusparseSpMatDescr_t* d, int64_t rows, int64_t cols,
                                               int64_t nnz, void* off, void* row, void* val,
                                               cusparseIndexType_t ot, cusparseIndexType_t rt,
                                               cusparseIndexBase_t base, cudaDataType vt) {
  return make_sparse(d, CUSPARSE_FORMAT_CSC, rows, cols, nnz, off, row, val, ot, rt, base, vt);
}
VGPU_EXPORT cusparseStatus_t cusparseCreateCoo(cusparseSpMatDescr_t* d, int64_t rows, int64_t cols,
                                               int64_t nnz, void* row, void* col, void* val,
                                               cusparseIndexType_t it, cusparseIndexBase_t base,
                                               cudaDataType vt) {
  return make_sparse(d, CUSPARSE_FORMAT_COO, rows, cols, nnz, row, col, val, it, it, base, vt);
}
VGPU_EXPORT cusparseStatus_t cusparseCreateConstCsr(cusparseConstSpMatDescr_t* d, int64_t rows,
                                                    int64_t cols, int64_t nnz, const void* off,
                                                    const void* col, const void* val,
                                                    cusparseIndexType_t ot, cusparseIndexType_t ct,
                                                    cusparseIndexBase_t base, cudaDataType vt) {
  // The Const* descriptor types are the same object seen through a const
  // pointer, so build one and let the assignment add the qualifier.
  cusparseSpMatDescr_t tmp = nullptr;
  const cusparseStatus_t rc = make_sparse(&tmp, CUSPARSE_FORMAT_CSR, rows, cols, nnz,
                                          const_cast<void*>(off), const_cast<void*>(col),
                                          const_cast<void*>(val), ot, ct, base, vt);
  if (rc == CUSPARSE_STATUS_SUCCESS && d) *d = tmp;
  return rc;
}
VGPU_EXPORT cusparseStatus_t cusparseCreateConstCoo(cusparseConstSpMatDescr_t* d, int64_t rows,
                                                    int64_t cols, int64_t nnz, const void* row,
                                                    const void* col, const void* val,
                                                    cusparseIndexType_t it,
                                                    cusparseIndexBase_t base, cudaDataType vt) {
  cusparseSpMatDescr_t tmp = nullptr;
  const cusparseStatus_t rc = make_sparse(&tmp, CUSPARSE_FORMAT_COO, rows, cols, nnz,
                                          const_cast<void*>(row), const_cast<void*>(col),
                                          const_cast<void*>(val), it, it, base, vt);
  if (rc == CUSPARSE_STATUS_SUCCESS && d) *d = tmp;
  return rc;
}

VGPU_EXPORT cusparseStatus_t cusparseCreateConstCsc(cusparseConstSpMatDescr_t* d, int64_t rows,
                                                    int64_t cols, int64_t nnz, const void* off,
                                                    const void* row, const void* val,
                                                    cusparseIndexType_t ot, cusparseIndexType_t rt,
                                                    cusparseIndexBase_t base, cudaDataType vt) {
  cusparseSpMatDescr_t tmp = nullptr;
  const cusparseStatus_t rc = make_sparse(&tmp, CUSPARSE_FORMAT_CSC, rows, cols, nnz,
                                          const_cast<void*>(off), const_cast<void*>(row),
                                          const_cast<void*>(val), ot, rt, base, vt);
  if (rc == CUSPARSE_STATUS_SUCCESS && d) *d = tmp;
  return rc;
}
// Declared here rather than taken from the header: CUDA 12.0's has neither.
VGPU_EXPORT cusparseStatus_t cusparseCreateBsr(cusparseSpMatDescr_t* d, int64_t brows, int64_t bcols, int64_t bnnz,
                                               int64_t rb, int64_t cb, void* off, void* col, void* val,
                                               cusparseIndexType_t ot, cusparseIndexType_t ct,
                                               cusparseIndexBase_t base, cudaDataType vt, cusparseOrder_t order) {
  return make_bsr(d, brows, bcols, bnnz, rb, cb, off, col, val, ot, ct, base, vt, order);
}
VGPU_EXPORT cusparseStatus_t cusparseCreateConstBsr(cusparseConstSpMatDescr_t* d, int64_t brows, int64_t bcols,
                                                    int64_t bnnz, int64_t rb, int64_t cb, const void* off,
                                                    const void* col, const void* val, cusparseIndexType_t ot,
                                                    cusparseIndexType_t ct, cusparseIndexBase_t base,
                                                    cudaDataType vt, cusparseOrder_t order) {
  cusparseSpMatDescr_t tmp = nullptr;
  const cusparseStatus_t rc = make_bsr(&tmp, brows, bcols, bnnz, rb, cb, const_cast<void*>(off),
                                       const_cast<void*>(col), const_cast<void*>(val), ot, ct, base, vt, order);
  if (rc == CUSPARSE_STATUS_SUCCESS && d) *d = tmp;
  return rc;
}

VGPU_EXPORT cusparseStatus_t cusparseDestroySpMat(cusparseConstSpMatDescr_t d) {
  if (!known(d)) return CUSPARSE_STATUS_INVALID_VALUE;
  untrack(d); delete reinterpret_cast<const SpMat*>(d);
  return CUSPARSE_STATUS_SUCCESS;
}
VGPU_EXPORT cusparseStatus_t cusparseSpMatGetFormat(cusparseConstSpMatDescr_t d,
                                                    cusparseFormat_t* f) {
  if (!known(d)) return CUSPARSE_STATUS_INVALID_VALUE;
  if (f) *f = reinterpret_cast<const SpMat*>(d)->format;
  return CUSPARSE_STATUS_SUCCESS;
}
VGPU_EXPORT cusparseStatus_t cusparseSpMatGetIndexBase(cusparseConstSpMatDescr_t d,
                                                       cusparseIndexBase_t* b) {
  if (!known(d)) return CUSPARSE_STATUS_INVALID_VALUE;
  if (b) *b = reinterpret_cast<const SpMat*>(d)->base;
  return CUSPARSE_STATUS_SUCCESS;
}
VGPU_EXPORT cusparseStatus_t cusparseSpMatGetValues(cusparseSpMatDescr_t d, void** v) {
  if (!known(d)) return CUSPARSE_STATUS_INVALID_VALUE;
  if (v) *v = reinterpret_cast<SpMat*>(d)->values;
  return CUSPARSE_STATUS_SUCCESS;
}
VGPU_EXPORT cusparseStatus_t cusparseSpMatSetValues(cusparseSpMatDescr_t d, void* v) {
  if (!known(d)) return CUSPARSE_STATUS_INVALID_VALUE;
  reinterpret_cast<SpMat*>(d)->values = v;
  return CUSPARSE_STATUS_SUCCESS;
}
VGPU_EXPORT cusparseStatus_t cusparseSpMatGetSize(cusparseConstSpMatDescr_t d, int64_t* rows,
                                                  int64_t* cols, int64_t* nnz) {
  if (!known(d)) return CUSPARSE_STATUS_INVALID_VALUE;
  const auto* m = reinterpret_cast<const SpMat*>(d);
  if (rows) *rows = m->rows;
  if (cols) *cols = m->cols;
  if (nnz) *nnz = m->nnz;
  return CUSPARSE_STATUS_SUCCESS;
}
VGPU_EXPORT cusparseStatus_t cusparseCsrGet(cusparseSpMatDescr_t d, int64_t* rows, int64_t* cols,
                                            int64_t* nnz, void** off, void** col, void** val,
                                            cusparseIndexType_t* ot, cusparseIndexType_t* ct,
                                            cusparseIndexBase_t* base, cudaDataType* vt) {
  if (!known(d)) return CUSPARSE_STATUS_INVALID_VALUE;
  auto* m = reinterpret_cast<SpMat*>(d);
  if (rows) *rows = m->rows;
  if (cols) *cols = m->cols;
  if (nnz) *nnz = m->nnz;
  if (off) *off = m->rows_ptr;
  if (col) *col = m->cols_ptr;
  if (val) *val = m->values;
  if (ot) *ot = m->row_type;
  if (ct) *ct = m->col_type;
  if (base) *base = m->base;
  if (vt) *vt = m->value_type;
  return CUSPARSE_STATUS_SUCCESS;
}
VGPU_EXPORT cusparseStatus_t cusparseCsrSetPointers(cusparseSpMatDescr_t d, void* off, void* col,
                                                    void* val) {
  if (!known(d)) return CUSPARSE_STATUS_INVALID_VALUE;
  auto* m = reinterpret_cast<SpMat*>(d);
  m->rows_ptr = off; m->cols_ptr = col; m->values = val;
  return CUSPARSE_STATUS_SUCCESS;
}
VGPU_EXPORT cusparseStatus_t cusparseCooSetPointers(cusparseSpMatDescr_t d, void* row, void* col,
                                                    void* val) {
  return cusparseCsrSetPointers(d, row, col, val);
}
VGPU_EXPORT cusparseStatus_t cusparseCscSetPointers(cusparseSpMatDescr_t d, void* off, void* row,
                                                    void* val) {
  return cusparseCsrSetPointers(d, off, row, val);
}
VGPU_EXPORT cusparseStatus_t cusparseCscGet(cusparseSpMatDescr_t d, int64_t* rows, int64_t* cols,
                                            int64_t* nnz, void** off, void** row, void** val,
                                            cusparseIndexType_t* ot, cusparseIndexType_t* rt,
                                            cusparseIndexBase_t* base, cudaDataType* vt) {
  return cusparseCsrGet(d, rows, cols, nnz, off, row, val, ot, rt, base, vt);
}
VGPU_EXPORT cusparseStatus_t cusparseCooGet(cusparseSpMatDescr_t d, int64_t* rows, int64_t* cols,
                                            int64_t* nnz, void** row, void** col, void** val,
                                            cusparseIndexType_t* it, cusparseIndexBase_t* base,
                                            cudaDataType* vt) {
  return cusparseCsrGet(d, rows, cols, nnz, row, col, val, it, nullptr, base, vt);
}

/* ---- dense descriptors ---- */

VGPU_EXPORT cusparseStatus_t cusparseCreateDnVec(cusparseDnVecDescr_t* d, int64_t size, void* v,
                                                 cudaDataType t) {
  if (!d || size < 0) return CUSPARSE_STATUS_INVALID_VALUE;
  if (!type_bytes(t)) return CUSPARSE_STATUS_NOT_SUPPORTED;
  auto* x = new DnVec{size, v, t};
  *d = reinterpret_cast<cusparseDnVecDescr_t>(track(x));
  return CUSPARSE_STATUS_SUCCESS;
}
VGPU_EXPORT cusparseStatus_t cusparseCreateConstDnVec(cusparseConstDnVecDescr_t* d, int64_t size,
                                                      const void* v, cudaDataType t) {
  cusparseDnVecDescr_t tmp = nullptr;
  const cusparseStatus_t rc = cusparseCreateDnVec(&tmp, size, const_cast<void*>(v), t);
  if (rc == CUSPARSE_STATUS_SUCCESS && d) *d = tmp;
  return rc;
}
VGPU_EXPORT cusparseStatus_t cusparseDestroyDnVec(cusparseConstDnVecDescr_t d) {
  if (!known(d)) return CUSPARSE_STATUS_INVALID_VALUE;
  untrack(d); delete reinterpret_cast<const DnVec*>(d);
  return CUSPARSE_STATUS_SUCCESS;
}
VGPU_EXPORT cusparseStatus_t cusparseDnVecGet(cusparseDnVecDescr_t d, int64_t* size, void** v,
                                              cudaDataType* t) {
  if (!known(d)) return CUSPARSE_STATUS_INVALID_VALUE;
  auto* x = reinterpret_cast<DnVec*>(d);
  if (size) *size = x->size;
  if (v) *v = x->values;
  if (t) *t = x->type;
  return CUSPARSE_STATUS_SUCCESS;
}
VGPU_EXPORT cusparseStatus_t cusparseDnVecGetValues(cusparseDnVecDescr_t d, void** v) {
  if (!known(d)) return CUSPARSE_STATUS_INVALID_VALUE;
  if (v) *v = reinterpret_cast<DnVec*>(d)->values;
  return CUSPARSE_STATUS_SUCCESS;
}
VGPU_EXPORT cusparseStatus_t cusparseDnVecSetValues(cusparseDnVecDescr_t d, void* v) {
  if (!known(d)) return CUSPARSE_STATUS_INVALID_VALUE;
  reinterpret_cast<DnVec*>(d)->values = v;
  return CUSPARSE_STATUS_SUCCESS;
}

VGPU_EXPORT cusparseStatus_t cusparseCreateDnMat(cusparseDnMatDescr_t* d, int64_t rows,
                                                 int64_t cols, int64_t ld, void* v, cudaDataType t,
                                                 cusparseOrder_t order) {
  if (!d || rows < 0 || cols < 0) return CUSPARSE_STATUS_INVALID_VALUE;
  if (!type_bytes(t)) return CUSPARSE_STATUS_NOT_SUPPORTED;
  auto* x = new DnMat{rows, cols, ld, v, t, order};
  *d = reinterpret_cast<cusparseDnMatDescr_t>(track(x));
  return CUSPARSE_STATUS_SUCCESS;
}
VGPU_EXPORT cusparseStatus_t cusparseCreateConstDnMat(cusparseConstDnMatDescr_t* d, int64_t rows,
                                                      int64_t cols, int64_t ld, const void* v,
                                                      cudaDataType t, cusparseOrder_t order) {
  cusparseDnMatDescr_t tmp = nullptr;
  const cusparseStatus_t rc =
      cusparseCreateDnMat(&tmp, rows, cols, ld, const_cast<void*>(v), t, order);
  if (rc == CUSPARSE_STATUS_SUCCESS && d) *d = tmp;
  return rc;
}
VGPU_EXPORT cusparseStatus_t cusparseDestroyDnMat(cusparseConstDnMatDescr_t d) {
  if (!known(d)) return CUSPARSE_STATUS_INVALID_VALUE;
  untrack(d); delete reinterpret_cast<const DnMat*>(d);
  return CUSPARSE_STATUS_SUCCESS;
}
VGPU_EXPORT cusparseStatus_t cusparseDnMatGet(cusparseDnMatDescr_t d, int64_t* rows, int64_t* cols,
                                              int64_t* ld, void** v, cudaDataType* t,
                                              cusparseOrder_t* order) {
  if (!known(d)) return CUSPARSE_STATUS_INVALID_VALUE;
  auto* x = reinterpret_cast<DnMat*>(d);
  if (rows) *rows = x->rows;
  if (cols) *cols = x->cols;
  if (ld) *ld = x->ld;
  if (v) *v = x->values;
  if (t) *t = x->type;
  if (order) *order = x->order;
  return CUSPARSE_STATUS_SUCCESS;
}
VGPU_EXPORT cusparseStatus_t cusparseDnMatGetValues(cusparseDnMatDescr_t d, void** v) {
  if (!known(d)) return CUSPARSE_STATUS_INVALID_VALUE;
  if (v) *v = reinterpret_cast<DnMat*>(d)->values;
  return CUSPARSE_STATUS_SUCCESS;
}
VGPU_EXPORT cusparseStatus_t cusparseDnMatSetValues(cusparseDnMatDescr_t d, void* v) {
  if (!known(d)) return CUSPARSE_STATUS_INVALID_VALUE;
  reinterpret_cast<DnMat*>(d)->values = v;
  return CUSPARSE_STATUS_SUCCESS;
}

/* ---- SpMV: y = alpha * op(A) * x + beta * y ---- */


namespace {
// What an RTX 3060's cuSPARSE 13.0 refuses before doing any arithmetic, so a
// program sees the same status here (nvidia/tests/e2e/sparse_complex_paths.cu):
//
//   * a conjugate transpose of a real operand: INVALID_VALUE;
//   * a complex type combination other than one complex type throughout,
//     half-precision complex computed in C_32F (SpMV and SpMM only), or a real
//     A of the compute type's precision with complex vectors (SpMV and SpMM
//     only): NOT_SUPPORTED. Real combinations are not policed here.
bool bad_conj(cusparseOperation_t op, cudaDataType t) {
  return op == CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE && !is_complex(t);
}
bool complex_combo_ok(cudaDataType a, cudaDataType x, cudaDataType y, cudaDataType ct, bool wide) {
  if (!is_complex(a) && !is_complex(x) && !is_complex(y) && !is_complex(ct)) return true;
  if (x != y) return false;
  if (wide && ct == CUDA_C_32F && (x == CUDA_C_16F || x == CUDA_C_16BF)) return a == x;
  if (x != ct || (ct != CUDA_C_32F && ct != CUDA_C_64F)) return false;
  if (a == x) return true;
  return wide && a == (ct == CUDA_C_32F ? CUDA_R_32F : CUDA_R_64F);
}
bool any_complex(std::initializer_list<cudaDataType> ts) {
  for (cudaDataType t : ts)
    if (is_complex(t)) return true;
  return false;
}

template <class V>
cusparseStatus_t spmv(cusparseHandle_t h, cusparseOperation_t op, const void* alpha, const SpMat& A,
                      const DnVec& X, const void* beta, DnVec& Y, cudaDataType ct) {
  const int64_t m = transposed(op) ? A.cols : A.rows;
  const int64_t n = transposed(op) ? A.rows : A.cols;
  if (X.size != n || Y.size != m) return CUSPARSE_STATUS_INVALID_VALUE;

  Triplets<V> t;
  if (!expand(A, &t)) return CUSPARSE_STATUS_INTERNAL_ERROR;
  std::vector<V> x, y;
  if (!fetch_values(X.values, (size_t)n, X.type, &x)) return CUSPARSE_STATUS_INTERNAL_ERROR;
  const V a = scalar<V>(h, alpha, ct), b = scalar<V>(h, beta, ct);
  if (b != V(0.0)) {
    if (!fetch_values(Y.values, (size_t)m, Y.type, &y)) return CUSPARSE_STATUS_INTERNAL_ERROR;
    for (auto& v : y) v *= b;
  } else {
    y.assign((size_t)m, V(0.0));   // beta == 0 means write-only: never read Y
  }
  for (size_t k = 0; k < t.val.size(); ++k) {
    const int64_t r = transposed(op) ? t.col[k] : t.row[k];
    const int64_t c = transposed(op) ? t.row[k] : t.col[k];
    if (r < 0 || r >= m || c < 0 || c >= n) return CUSPARSE_STATUS_INVALID_VALUE;
    y[(size_t)r] += a * conj_op(t.val[k], op) * x[(size_t)c];
  }
  return store_values(Y.values, y, Y.type) ? CUSPARSE_STATUS_SUCCESS
                                           : CUSPARSE_STATUS_INTERNAL_ERROR;
}

cusparseStatus_t spmv_check(cusparseOperation_t op, const SpMat& A, const DnVec& X, const DnVec& Y,
                            cudaDataType ct) {
  if (!type_bytes(ct)) return CUSPARSE_STATUS_NOT_SUPPORTED;
  if (bad_conj(op, A.value_type)) return CUSPARSE_STATUS_INVALID_VALUE;
  if (!complex_combo_ok(A.value_type, X.type, Y.type, ct, true)) return CUSPARSE_STATUS_NOT_SUPPORTED;
  // BSR: only A itself, as NVIDIA's (CUSPARSE_SPMV_BSR_ALG1) takes it.
  if (A.format == kFormatBsr && op != CUSPARSE_OPERATION_NON_TRANSPOSE) return CUSPARSE_STATUS_NOT_SUPPORTED;
  return CUSPARSE_STATUS_SUCCESS;
}
}  // namespace

VGPU_EXPORT cusparseStatus_t cusparseSpMV_bufferSize(cusparseHandle_t, cusparseOperation_t op,
                                                     const void*, cusparseConstSpMatDescr_t matA,
                                                     cusparseConstDnVecDescr_t vecX, const void*,
                                                     cusparseDnVecDescr_t vecY, cudaDataType ct,
                                                     cusparseSpMVAlg_t, size_t* bytes) {
  if (bytes) *bytes = 0;   // host-computed: no device workspace to size
  if (known(matA) && known(vecX) && known(vecY))
    return spmv_check(op, *reinterpret_cast<const SpMat*>(matA), *reinterpret_cast<const DnVec*>(vecX),
                      *reinterpret_cast<const DnVec*>(vecY), ct);
  return CUSPARSE_STATUS_SUCCESS;
}

VGPU_EXPORT cusparseStatus_t cusparseSpMV(cusparseHandle_t h, cusparseOperation_t op,
                                          const void* alpha, cusparseConstSpMatDescr_t matA,
                                          cusparseConstDnVecDescr_t vecX, const void* beta,
                                          cusparseDnVecDescr_t vecY, cudaDataType ct,
                                          cusparseSpMVAlg_t alg, void* buffer) {
  if (!known(h) || !known(matA) || !known(vecX) || !known(vecY))
    return CUSPARSE_STATUS_NOT_INITIALIZED;
  if (capturing(h)) {
    auto held = std::make_shared<Held>();
    const cusparseHandle_t hc = held->handle(h);
    const void* a = held->scalar(*reinterpret_cast<Handle*>(hc), alpha, type_bytes(ct));
    const void* b = held->scalar(*reinterpret_cast<Handle*>(hc), beta, type_bytes(ct));
    auto* A = reinterpret_cast<cusparseConstSpMatDescr_t>(held->copy<SpMat>(matA));
    auto* X = reinterpret_cast<cusparseConstDnVecDescr_t>(held->copy<DnVec>(vecX));
    auto* Y = reinterpret_cast<cusparseDnVecDescr_t>(held->copy<DnVec>(vecY));
    return record(h, held, [=] { cusparseSpMV(hc, op, a, A, X, b, Y, ct, alg, buffer); });
  }
  const auto& A = *reinterpret_cast<const SpMat*>(matA);
  const auto& X = *reinterpret_cast<const DnVec*>(vecX);
  auto& Y = *reinterpret_cast<DnVec*>(vecY);
  if (const cusparseStatus_t st = spmv_check(op, A, X, Y, ct); st != CUSPARSE_STATUS_SUCCESS) return st;
  return any_complex({A.value_type, X.type, Y.type, ct}) ? spmv<cd>(h, op, alpha, A, X, beta, Y, ct)
                                                         : spmv<double>(h, op, alpha, A, X, beta, Y, ct);
}

VGPU_EXPORT cusparseStatus_t cusparseSpMV_preprocess(cusparseHandle_t, cusparseOperation_t,
                                                     const void*, cusparseConstSpMatDescr_t,
                                                     cusparseConstDnVecDescr_t, const void*,
                                                     cusparseDnVecDescr_t, cudaDataType,
                                                     cusparseSpMVAlg_t, void*) {
  return CUSPARSE_STATUS_SUCCESS;   // nothing to precompute
}

/* ---- SpMM: C = alpha * op(A) * op(B) + beta * C ---- */

namespace {
// Matrix `i` of a strided batch, as a plain descriptor. A descriptor that is
// not batched is every member.
SpMat member(const SpMat& a, int i) {
  if (a.batch == 1) return a;
  SpMat m = a;
  m.batch = 1;
  auto step = [](void* p, int64_t elems, size_t bytes) {
    return p ? static_cast<void*>(static_cast<char*>(p) + (size_t)elems * bytes) : p;
  };
  m.rows_ptr = step(a.rows_ptr, (int64_t)i * a.off_stride, index_bytes(a.row_type));
  m.cols_ptr = step(a.cols_ptr, (int64_t)i * a.col_stride, index_bytes(a.col_type));
  m.values = step(a.values, (int64_t)i * a.val_stride, type_bytes(a.value_type));
  return m;
}
DnMat member(const DnMat& a, int i) {
  if (a.batch == 1) return a;
  DnMat m = a;
  m.batch = 1;
  m.values = static_cast<char*>(a.values) + (size_t)i * (size_t)a.batch_stride * type_bytes(a.type);
  return m;
}

// C = alpha * op(A) * op(B) + beta * C for one member of a batch.
template <class V>
cusparseStatus_t spmm_one(cusparseOperation_t opA, cusparseOperation_t opB, V al,
                          const SpMat& A, const DnMat& B, V be, const DnMat& C) {

  const int64_t m = transposed(opA) ? A.cols : A.rows;
  const int64_t k = transposed(opA) ? A.rows : A.cols;
  const int64_t bk = transposed(opB) ? B.cols : B.rows;
  const int64_t n = transposed(opB) ? B.rows : B.cols;
  if (bk != k || C.rows != m || C.cols != n) return CUSPARSE_STATUS_INVALID_VALUE;

  Triplets<V> t;
  if (!expand(A, &t)) return CUSPARSE_STATUS_INTERNAL_ERROR;
  std::vector<V> b, c;
  if (!fetch_values(B.values, dn_elems(B), B.type, &b)) return CUSPARSE_STATUS_INTERNAL_ERROR;
  if (be != V(0.0)) {
    if (!fetch_values(C.values, dn_elems(C), C.type, &c)) return CUSPARSE_STATUS_INTERNAL_ERROR;
    for (auto& v : c) v *= be;
  } else {
    c.assign(dn_elems(C), V(0.0));
  }
  for (size_t e = 0; e < t.val.size(); ++e) {
    const int64_t r = transposed(opA) ? t.col[e] : t.row[e];
    const int64_t cc = transposed(opA) ? t.row[e] : t.col[e];
    if (r < 0 || r >= m || cc < 0 || cc >= k) return CUSPARSE_STATUS_INVALID_VALUE;
    const V av = al * conj_op(t.val[e], opA);
    for (int64_t j = 0; j < n; ++j) {
      const size_t bi = transposed(opB) ? dn_index(B, j, cc) : dn_index(B, cc, j);
      c[dn_index(C, r, j)] += av * conj_op(b[bi], opB);
    }
  }
  return store_values(C.values, c, C.type) ? CUSPARSE_STATUS_SUCCESS
                                           : CUSPARSE_STATUS_INTERNAL_ERROR;
}

template <class V>
cusparseStatus_t spmm(cusparseHandle_t h, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha,
                      const SpMat& A, const DnMat& B, const void* beta, const DnMat& C, cudaDataType ct) {
  const V al = scalar<V>(h, alpha, ct), be = scalar<V>(h, beta, ct);
  for (int i = 0; i < C.batch; ++i) {
    const cusparseStatus_t st = spmm_one<V>(opA, opB, al, member(A, i), member(B, i), be, member(C, i));
    if (st != CUSPARSE_STATUS_SUCCESS) return st;
  }
  return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t spmm_check(cusparseOperation_t opA, cusparseOperation_t opB, const SpMat& A, const DnMat& B,
                            const DnMat& C, cudaDataType ct) {
  if (!type_bytes(ct)) return CUSPARSE_STATUS_NOT_SUPPORTED;
  if (bad_conj(opA, A.value_type) || bad_conj(opB, B.type)) return CUSPARSE_STATUS_INVALID_VALUE;
  if (!complex_combo_ok(A.value_type, B.type, C.type, ct, true)) return CUSPARSE_STATUS_NOT_SUPPORTED;
  // BSR (CUSPARSE_SPMM_BSR_ALG1): A itself, its blocks stored row-major.
  if (A.format == kFormatBsr &&
      (opA != CUSPARSE_OPERATION_NON_TRANSPOSE || A.block_order != CUSPARSE_ORDER_ROW))
    return CUSPARSE_STATUS_NOT_SUPPORTED;
  // A strided batch: C holds every product, and an operand with one matrix is
  // shared by all of them.
  if ((A.batch != 1 && A.batch != C.batch) || (B.batch != 1 && B.batch != C.batch))
    return CUSPARSE_STATUS_INVALID_VALUE;
  return CUSPARSE_STATUS_SUCCESS;
}

}  // namespace


VGPU_EXPORT cusparseStatus_t cusparseSpMM_bufferSize(cusparseHandle_t, cusparseOperation_t opA,
                                                     cusparseOperation_t opB, const void*,
                                                     cusparseConstSpMatDescr_t matA,
                                                     cusparseConstDnMatDescr_t matB, const void*,
                                                     cusparseDnMatDescr_t matC, cudaDataType ct,
                                                     cusparseSpMMAlg_t, size_t* bytes) {
  if (bytes) *bytes = 0;
  if (known(matA) && known(matB) && known(matC))
    return spmm_check(opA, opB, *reinterpret_cast<const SpMat*>(matA), *reinterpret_cast<const DnMat*>(matB),
                      *reinterpret_cast<const DnMat*>(matC), ct);
  return CUSPARSE_STATUS_SUCCESS;
}

VGPU_EXPORT cusparseStatus_t cusparseSpMM(cusparseHandle_t h, cusparseOperation_t opA,
                                          cusparseOperation_t opB, const void* alpha,
                                          cusparseConstSpMatDescr_t matA,
                                          cusparseConstDnMatDescr_t matB, const void* beta,
                                          cusparseDnMatDescr_t matC, cudaDataType ct,
                                          cusparseSpMMAlg_t alg, void* buffer) {
  if (!known(h) || !known(matA) || !known(matB) || !known(matC))
    return CUSPARSE_STATUS_NOT_INITIALIZED;
  if (capturing(h)) {
    auto held = std::make_shared<Held>();
    const cusparseHandle_t hc = held->handle(h);
    const void* a = held->scalar(*reinterpret_cast<Handle*>(hc), alpha, type_bytes(ct));
    const void* b = held->scalar(*reinterpret_cast<Handle*>(hc), beta, type_bytes(ct));
    auto* A = reinterpret_cast<cusparseConstSpMatDescr_t>(held->copy<SpMat>(matA));
    auto* B = reinterpret_cast<cusparseConstDnMatDescr_t>(held->copy<DnMat>(matB));
    auto* C = reinterpret_cast<cusparseDnMatDescr_t>(held->copy<DnMat>(matC));
    return record(h, held, [=] { cusparseSpMM(hc, opA, opB, a, A, B, b, C, ct, alg, buffer); });
  }
  const auto& A = *reinterpret_cast<const SpMat*>(matA);
  const auto& B = *reinterpret_cast<const DnMat*>(matB);
  const auto& C = *reinterpret_cast<const DnMat*>(matC);
  if (const cusparseStatus_t st = spmm_check(opA, opB, A, B, C, ct); st != CUSPARSE_STATUS_SUCCESS) return st;
  return any_complex({A.value_type, B.type, C.type, ct}) ? spmm<cd>(h, opA, opB, alpha, A, B, beta, C, ct)
                                                         : spmm<double>(h, opA, opB, alpha, A, B, beta, C, ct);
}

VGPU_EXPORT cusparseStatus_t cusparseSpMM_preprocess(cusparseHandle_t, cusparseOperation_t,
                                                     cusparseOperation_t, const void*,
                                                     cusparseConstSpMatDescr_t,
                                                     cusparseConstDnMatDescr_t, const void*,
                                                     cusparseDnMatDescr_t, cudaDataType,
                                                     cusparseSpMMAlg_t, void*) {
  return CUSPARSE_STATUS_SUCCESS;
}

/* ---- conversion ----
   Neither direction takes a BSR matrix: NVIDIA's cuSPARSE 13.0 refuses one
   (NOT_SUPPORTED) at _bufferSize, so this does too. */

namespace {
cusparseStatus_t convert_check(const SpMat& A) {
  return A.format == kFormatBsr ? CUSPARSE_STATUS_NOT_SUPPORTED : CUSPARSE_STATUS_SUCCESS;
}

template <class V> cusparseStatus_t sparse_to_dense(const SpMat& A, DnMat& B) {
  if (A.rows != B.rows || A.cols != B.cols) return CUSPARSE_STATUS_INVALID_VALUE;
  Triplets<V> t;
  if (!expand(A, &t)) return CUSPARSE_STATUS_INTERNAL_ERROR;
  std::vector<V> d(dn_elems(B), V(0.0));
  for (size_t e = 0; e < t.val.size(); ++e) {
    if (t.row[e] < 0 || t.row[e] >= A.rows || t.col[e] < 0 || t.col[e] >= A.cols)
      return CUSPARSE_STATUS_INVALID_VALUE;
    d[dn_index(B, t.row[e], t.col[e])] += t.val[e];  // duplicate entries accumulate
  }
  return store_values(B.values, d, B.type) ? CUSPARSE_STATUS_SUCCESS
                                           : CUSPARSE_STATUS_INTERNAL_ERROR;
}

template <class V> cusparseStatus_t dense_nnz(const DnMat& A, int64_t* nnz) {
  std::vector<V> d;
  if (!fetch_values(A.values, dn_elems(A), A.type, &d)) return CUSPARSE_STATUS_INTERNAL_ERROR;
  *nnz = 0;
  for (int64_t r = 0; r < A.rows; ++r)
    for (int64_t c = 0; c < A.cols; ++c)
      if (d[dn_index(A, r, c)] != V(0.0)) ++*nnz;
  return CUSPARSE_STATUS_SUCCESS;
}

template <class V> cusparseStatus_t dense_to_sparse(const DnMat& A, SpMat& B) {
  std::vector<V> d;
  if (!fetch_values(A.values, dn_elems(A), A.type, &d)) return CUSPARSE_STATUS_INTERNAL_ERROR;
  const int64_t off = B.base == CUSPARSE_INDEX_BASE_ONE ? 1 : 0;

  std::vector<int64_t> rows, cols;
  std::vector<V> vals;
  std::vector<int64_t> offsets{off};
  if (B.format == CUSPARSE_FORMAT_CSC) {
    for (int64_t c = 0; c < A.cols; ++c) {
      for (int64_t r = 0; r < A.rows; ++r) {
        const V v = d[dn_index(A, r, c)];
        if (v == V(0.0)) continue;
        rows.push_back(r + off);
        vals.push_back(v);
      }
      offsets.push_back((int64_t)vals.size() + off);
    }
  } else {
    for (int64_t r = 0; r < A.rows; ++r) {
      for (int64_t c = 0; c < A.cols; ++c) {
        const V v = d[dn_index(A, r, c)];
        if (v == V(0.0)) continue;
        rows.push_back(r + off);
        cols.push_back(c + off);
        vals.push_back(v);
      }
      offsets.push_back((int64_t)vals.size() + off);
    }
  }
  B.nnz = (int64_t)vals.size();

  auto put_indices = [](void* dev, const std::vector<int64_t>& v, cusparseIndexType_t t) {
    if (v.empty()) return true;
    if (t == CUSPARSE_INDEX_32I) {
      std::vector<int32_t> tmp(v.begin(), v.end());
      return cudaMemcpy(dev, tmp.data(), tmp.size() * 4, cudaMemcpyHostToDevice) == cudaSuccess;
    }
    return cudaMemcpy(dev, v.data(), v.size() * 8, cudaMemcpyHostToDevice) == cudaSuccess;
  };
  const bool ok =
      B.format == CUSPARSE_FORMAT_CSC
          ? put_indices(B.rows_ptr, offsets, B.row_type) && put_indices(B.cols_ptr, rows, B.col_type)
          : put_indices(B.rows_ptr, B.format == CUSPARSE_FORMAT_CSR ? offsets : rows, B.row_type) &&
                put_indices(B.cols_ptr, cols, B.col_type);
  return ok && store_values(B.values, vals, B.value_type) ? CUSPARSE_STATUS_SUCCESS
                                                          : CUSPARSE_STATUS_INTERNAL_ERROR;
}
}  // namespace

VGPU_EXPORT cusparseStatus_t cusparseSparseToDense_bufferSize(cusparseHandle_t,
                                                              cusparseConstSpMatDescr_t matA,
                                                              cusparseDnMatDescr_t,
                                                              cusparseSparseToDenseAlg_t,
                                                              size_t* bytes) {
  if (bytes) *bytes = 0;
  return known(matA) ? convert_check(*reinterpret_cast<const SpMat*>(matA)) : CUSPARSE_STATUS_SUCCESS;
}

VGPU_EXPORT cusparseStatus_t cusparseSparseToDense(cusparseHandle_t h,
                                                   cusparseConstSpMatDescr_t matA,
                                                   cusparseDnMatDescr_t matB,
                                                   cusparseSparseToDenseAlg_t alg, void* buffer) {
  if (!known(h) || !known(matA) || !known(matB)) return CUSPARSE_STATUS_NOT_INITIALIZED;
  if (capturing(h)) {
    auto held = std::make_shared<Held>();
    const cusparseHandle_t hc = held->handle(h);
    auto* A = reinterpret_cast<cusparseConstSpMatDescr_t>(held->copy<SpMat>(matA));
    auto* B = reinterpret_cast<cusparseDnMatDescr_t>(held->copy<DnMat>(matB));
    return record(h, held, [=] { cusparseSparseToDense(hc, A, B, alg, buffer); });
  }
  const auto& A = *reinterpret_cast<const SpMat*>(matA);
  auto& B = *reinterpret_cast<DnMat*>(matB);
  if (const cusparseStatus_t st = convert_check(A); st != CUSPARSE_STATUS_SUCCESS) return st;
  return any_complex({A.value_type, B.type}) ? sparse_to_dense<cd>(A, B) : sparse_to_dense<double>(A, B);
}

VGPU_EXPORT cusparseStatus_t cusparseDenseToSparse_bufferSize(cusparseHandle_t,
                                                              cusparseConstDnMatDescr_t,
                                                              cusparseSpMatDescr_t matB,
                                                              cusparseDenseToSparseAlg_t,
                                                              size_t* bytes) {
  if (bytes) *bytes = 0;
  return known(matB) ? convert_check(*reinterpret_cast<const SpMat*>(matB)) : CUSPARSE_STATUS_SUCCESS;
}

// Two-call protocol: _analysis fills in nnz, the caller allocates, _convert
// writes the arrays.
VGPU_EXPORT cusparseStatus_t cusparseDenseToSparse_analysis(cusparseHandle_t h,
                                                            cusparseConstDnMatDescr_t matA,
                                                            cusparseSpMatDescr_t matB,
                                                            cusparseDenseToSparseAlg_t, void*) {
  if (!known(h) || !known(matA) || !known(matB)) return CUSPARSE_STATUS_NOT_INITIALIZED;
  const auto& A = *reinterpret_cast<const DnMat*>(matA);
  auto& B = *reinterpret_cast<SpMat*>(matB);
  if (const cusparseStatus_t st = convert_check(B); st != CUSPARSE_STATUS_SUCCESS) return st;
  return is_complex(A.type) ? dense_nnz<cd>(A, &B.nnz) : dense_nnz<double>(A, &B.nnz);
}

VGPU_EXPORT cusparseStatus_t cusparseDenseToSparse_convert(cusparseHandle_t h,
                                                           cusparseConstDnMatDescr_t matA,
                                                           cusparseSpMatDescr_t matB,
                                                           cusparseDenseToSparseAlg_t, void*) {
  if (!known(h) || !known(matA) || !known(matB)) return CUSPARSE_STATUS_NOT_INITIALIZED;
  const auto& A = *reinterpret_cast<const DnMat*>(matA);
  auto& B = *reinterpret_cast<SpMat*>(matB);
  if (A.rows != B.rows || A.cols != B.cols) return CUSPARSE_STATUS_INVALID_VALUE;
  if (B.format != CUSPARSE_FORMAT_CSR && B.format != CUSPARSE_FORMAT_CSC && B.format != CUSPARSE_FORMAT_COO)
    return CUSPARSE_STATUS_NOT_SUPPORTED;
  return any_complex({A.type, B.value_type}) ? dense_to_sparse<cd>(A, B) : dense_to_sparse<double>(A, B);
}

/* ---- CSR to CSC ----
   The transpose of the index structure: CSC of A is CSR of A's transpose. The
   values are moved, never computed on, so every value type works and moves bit
   for bit; only the element size matters. Rows within each column come out in
   ascending order, which is what a stable pass over the rows gives. */

namespace {

size_t element_bytes(cudaDataType t) {
  switch (t) {
    case CUDA_R_8I: case CUDA_R_8U: return 1;
    case CUDA_R_16F: case CUDA_R_16BF: return 2;
    case CUDA_R_32F: case CUDA_R_32I: case CUDA_C_16F: case CUDA_C_16BF: return 4;
    case CUDA_R_64F: case CUDA_C_32F: return 8;
    case CUDA_C_64F: return 16;
    default: return 0;
  }
}

}  // namespace

VGPU_EXPORT cusparseStatus_t cusparseCsr2cscEx2_bufferSize(
    cusparseHandle_t h, int, int, int, const void*, const int*, const int*, void*, int*, int*,
    cudaDataType valType, cusparseAction_t copyValues, cusparseIndexBase_t,
    cusparseCsr2CscAlg_t, size_t* bufferSize) {
  if (!known(h)) return CUSPARSE_STATUS_NOT_INITIALIZED;
  if (!bufferSize) return CUSPARSE_STATUS_INVALID_VALUE;
  if (copyValues == CUSPARSE_ACTION_NUMERIC && !element_bytes(valType))
    return CUSPARSE_STATUS_NOT_SUPPORTED;
  *bufferSize = 0;  // the work happens on the host
  return CUSPARSE_STATUS_SUCCESS;
}

VGPU_EXPORT cusparseStatus_t cusparseCsr2cscEx2(
    cusparseHandle_t h, int m, int n, int nnz, const void* csrVal, const int* csrRowPtr,
    const int* csrColInd, void* cscVal, int* cscColPtr, int* cscRowInd, cudaDataType valType,
    cusparseAction_t copyValues, cusparseIndexBase_t idxBase, cusparseCsr2CscAlg_t alg, void* buffer) {
  if (!known(h)) return CUSPARSE_STATUS_NOT_INITIALIZED;
  if (capturing(h)) {
    auto held = std::make_shared<Held>();
    const cusparseHandle_t hc = held->handle(h);
    return record(h, held, [=] {
      cusparseCsr2cscEx2(hc, m, n, nnz, csrVal, csrRowPtr, csrColInd, cscVal, cscColPtr, cscRowInd,
                         valType, copyValues, idxBase, alg, buffer);
    });
  }
  if (m < 0 || n < 0 || nnz < 0) return CUSPARSE_STATUS_INVALID_VALUE;
  if (idxBase != CUSPARSE_INDEX_BASE_ZERO && idxBase != CUSPARSE_INDEX_BASE_ONE)
    return CUSPARSE_STATUS_INVALID_VALUE;
  const bool numeric = copyValues == CUSPARSE_ACTION_NUMERIC;
  if (!numeric && copyValues != CUSPARSE_ACTION_SYMBOLIC) return CUSPARSE_STATUS_INVALID_VALUE;
  const size_t vb = element_bytes(valType);
  if (numeric && !vb) return CUSPARSE_STATUS_NOT_SUPPORTED;
  if (n == 0) return CUSPARSE_STATUS_SUCCESS;
  if (!cscColPtr) return CUSPARSE_STATUS_INVALID_VALUE;
  const int base = idxBase == CUSPARSE_INDEX_BASE_ONE ? 1 : 0;
  std::vector<int> colptr((size_t)n + 1, base);
  if (m > 0 && nnz > 0) {
    if (!csrRowPtr || !csrColInd || !cscRowInd) return CUSPARSE_STATUS_INVALID_VALUE;
    if (numeric && (!csrVal || !cscVal)) return CUSPARSE_STATUS_INVALID_VALUE;
    std::vector<int> rowptr((size_t)m + 1), colind((size_t)nnz);
    if (cudaMemcpy(rowptr.data(), csrRowPtr, rowptr.size() * 4, cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(colind.data(), csrColInd, colind.size() * 4, cudaMemcpyDeviceToHost) != cudaSuccess)
      return CUSPARSE_STATUS_INTERNAL_ERROR;
    // A malformed matrix is refused rather than transposed into nonsense.
    if (rowptr[0] != base || rowptr[m] != nnz + base) return CUSPARSE_STATUS_INVALID_VALUE;
    for (int r = 0; r < m; ++r)
      if (rowptr[r + 1] < rowptr[r]) return CUSPARSE_STATUS_INVALID_VALUE;
    std::vector<int> count((size_t)n, 0);
    for (int c : colind) {
      if (c - base < 0 || c - base >= n) return CUSPARSE_STATUS_INVALID_VALUE;
      ++count[(size_t)(c - base)];
    }
    for (int c = 0; c < n; ++c) colptr[(size_t)c + 1] = colptr[(size_t)c] + count[(size_t)c];
    std::vector<unsigned char> in, out;
    if (numeric) {
      in.resize((size_t)nnz * vb);
      out.resize(in.size());
      if (cudaMemcpy(in.data(), csrVal, in.size(), cudaMemcpyDeviceToHost) != cudaSuccess)
        return CUSPARSE_STATUS_INTERNAL_ERROR;
    }
    std::vector<int> next(colptr.begin(), colptr.end() - 1), rowind((size_t)nnz);
    for (int r = 0; r < m; ++r)
      for (int e = rowptr[r] - base; e < rowptr[r + 1] - base; ++e) {
        const int dst = next[(size_t)(colind[(size_t)e] - base)]++ - base;
        rowind[(size_t)dst] = r + base;
        if (numeric) std::memcpy(&out[(size_t)dst * vb], &in[(size_t)e * vb], vb);
      }
    if (cudaMemcpy(cscRowInd, rowind.data(), rowind.size() * 4, cudaMemcpyHostToDevice) != cudaSuccess ||
        (numeric && cudaMemcpy(cscVal, out.data(), out.size(), cudaMemcpyHostToDevice) != cudaSuccess))
      return CUSPARSE_STATUS_INTERNAL_ERROR;
  }
  return cudaMemcpy(cscColPtr, colptr.data(), colptr.size() * 4, cudaMemcpyHostToDevice) == cudaSuccess
             ? CUSPARSE_STATUS_SUCCESS
             : CUSPARSE_STATUS_INTERNAL_ERROR;
}

/* ---- pointer mode, and the attributes a descriptor carries beyond its arrays ---- */

VGPU_EXPORT cusparseStatus_t cusparseSetPointerMode(cusparseHandle_t h, cusparsePointerMode_t mode) {
  if (!known(h)) return CUSPARSE_STATUS_NOT_INITIALIZED;
  if (mode != CUSPARSE_POINTER_MODE_HOST && mode != CUSPARSE_POINTER_MODE_DEVICE)
    return CUSPARSE_STATUS_INVALID_VALUE;
  reinterpret_cast<Handle*>(h)->mode = mode;
  return CUSPARSE_STATUS_SUCCESS;
}
VGPU_EXPORT cusparseStatus_t cusparseGetPointerMode(cusparseHandle_t h, cusparsePointerMode_t* mode) {
  if (!known(h)) return CUSPARSE_STATUS_NOT_INITIALIZED;
  if (mode) *mode = reinterpret_cast<Handle*>(h)->mode;
  return CUSPARSE_STATUS_SUCCESS;
}

VGPU_EXPORT cusparseStatus_t cusparseSpMatSetAttribute(cusparseSpMatDescr_t d, cusparseSpMatAttribute_t attr,
                                                       void* data, size_t size) {
  if (!known(d) || !data) return CUSPARSE_STATUS_INVALID_VALUE;
  auto* m = reinterpret_cast<SpMat*>(d);
  if (attr == CUSPARSE_SPMAT_FILL_MODE && size == sizeof(cusparseFillMode_t))
    std::memcpy(&m->fill, data, size);
  else if (attr == CUSPARSE_SPMAT_DIAG_TYPE && size == sizeof(cusparseDiagType_t))
    std::memcpy(&m->diag, data, size);
  else
    return CUSPARSE_STATUS_INVALID_VALUE;
  return CUSPARSE_STATUS_SUCCESS;
}
VGPU_EXPORT cusparseStatus_t cusparseSpMatGetAttribute(cusparseConstSpMatDescr_t d,
                                                       cusparseSpMatAttribute_t attr, void* data,
                                                       size_t size) {
  if (!known(d) || !data) return CUSPARSE_STATUS_INVALID_VALUE;
  const auto* m = reinterpret_cast<const SpMat*>(d);
  if (attr == CUSPARSE_SPMAT_FILL_MODE && size == sizeof(cusparseFillMode_t))
    std::memcpy(data, &m->fill, size);
  else if (attr == CUSPARSE_SPMAT_DIAG_TYPE && size == sizeof(cusparseDiagType_t))
    std::memcpy(data, &m->diag, size);
  else
    return CUSPARSE_STATUS_INVALID_VALUE;
  return CUSPARSE_STATUS_SUCCESS;
}

VGPU_EXPORT cusparseStatus_t cusparseCsrSetStridedBatch(cusparseSpMatDescr_t d, int count,
                                                        int64_t off_stride, int64_t val_stride) {
  if (!known(d) || count < 1) return CUSPARSE_STATUS_INVALID_VALUE;
  auto* m = reinterpret_cast<SpMat*>(d);
  m->batch = count;
  m->off_stride = off_stride;
  m->val_stride = m->col_stride = val_stride;
  return CUSPARSE_STATUS_SUCCESS;
}
VGPU_EXPORT cusparseStatus_t cusparseCooSetStridedBatch(cusparseSpMatDescr_t d, int count, int64_t stride) {
  if (!known(d) || count < 1) return CUSPARSE_STATUS_INVALID_VALUE;
  auto* m = reinterpret_cast<SpMat*>(d);
  m->batch = count;
  m->off_stride = m->col_stride = m->val_stride = stride;
  return CUSPARSE_STATUS_SUCCESS;
}
// Strides shorter than one member's arrays are refused, as NVIDIA's cuSPARSE
// 13.0 refuses them (INVALID_VALUE, "batchStride < nnz").
VGPU_EXPORT cusparseStatus_t cusparseBsrSetStridedBatch(cusparseSpMatDescr_t d, int count, int64_t off_stride,
                                                        int64_t col_stride, int64_t val_stride) {
  if (!known(d) || count < 1) return CUSPARSE_STATUS_INVALID_VALUE;
  auto* m = reinterpret_cast<SpMat*>(d);
  if (m->format != kFormatBsr) return CUSPARSE_STATUS_INVALID_VALUE;
  const int64_t bs = m->bdim * m->bdim;
  if (count > 1 && (off_stride < m->rows / m->bdim + 1 || col_stride < m->nnz / bs || val_stride < m->nnz))
    return CUSPARSE_STATUS_INVALID_VALUE;
  m->batch = count;
  m->off_stride = off_stride;
  m->col_stride = col_stride;
  m->val_stride = val_stride;
  return CUSPARSE_STATUS_SUCCESS;
}
VGPU_EXPORT cusparseStatus_t cusparseSpMatGetStridedBatch(cusparseConstSpMatDescr_t d, int* count) {
  if (!known(d)) return CUSPARSE_STATUS_INVALID_VALUE;
  if (count) *count = reinterpret_cast<const SpMat*>(d)->batch;
  return CUSPARSE_STATUS_SUCCESS;
}
VGPU_EXPORT cusparseStatus_t cusparseDnMatSetStridedBatch(cusparseDnMatDescr_t d, int count, int64_t stride) {
  if (!known(d) || count < 1) return CUSPARSE_STATUS_INVALID_VALUE;
  auto* m = reinterpret_cast<DnMat*>(d);
  m->batch = count;
  m->batch_stride = stride;
  return CUSPARSE_STATUS_SUCCESS;
}
VGPU_EXPORT cusparseStatus_t cusparseDnMatGetStridedBatch(cusparseConstDnMatDescr_t d, int* count,
                                                          int64_t* stride) {
  if (!known(d)) return CUSPARSE_STATUS_INVALID_VALUE;
  const auto* m = reinterpret_cast<const DnMat*>(d);
  if (count) *count = m->batch;
  if (stride) *stride = m->batch_stride;
  return CUSPARSE_STATUS_SUCCESS;
}

/* ---- legacy matrix descriptors ---- */

namespace {
struct MatDescr {
  cusparseMatrixType_t type = CUSPARSE_MATRIX_TYPE_GENERAL;
  cusparseFillMode_t fill = CUSPARSE_FILL_MODE_LOWER;
  cusparseDiagType_t diag = CUSPARSE_DIAG_TYPE_NON_UNIT;
  cusparseIndexBase_t base = CUSPARSE_INDEX_BASE_ZERO;
};
MatDescr* descr(cusparseMatDescr_t d) { return known(d) ? reinterpret_cast<MatDescr*>(d) : nullptr; }
int base_of(cusparseMatDescr_t d) {
  const MatDescr* m = descr(d);
  return m && m->base == CUSPARSE_INDEX_BASE_ONE ? 1 : 0;
}
}  // namespace

VGPU_EXPORT cusparseStatus_t cusparseCreateMatDescr(cusparseMatDescr_t* d) {
  if (!d) return CUSPARSE_STATUS_INVALID_VALUE;
  *d = reinterpret_cast<cusparseMatDescr_t>(track(new MatDescr()));
  return CUSPARSE_STATUS_SUCCESS;
}
VGPU_EXPORT cusparseStatus_t cusparseDestroyMatDescr(cusparseMatDescr_t d) {
  if (!descr(d)) return CUSPARSE_STATUS_INVALID_VALUE;
  untrack(d);
  delete reinterpret_cast<MatDescr*>(d);
  return CUSPARSE_STATUS_SUCCESS;
}
#define VGPU_DESCR_FIELD(Name, Type, field)                                                    \
  VGPU_EXPORT cusparseStatus_t cusparseSetMat##Name(cusparseMatDescr_t d, Type v) {            \
    if (!descr(d)) return CUSPARSE_STATUS_INVALID_VALUE;                                       \
    descr(d)->field = v;                                                                       \
    return CUSPARSE_STATUS_SUCCESS;                                                            \
  }                                                                                            \
  VGPU_EXPORT Type cusparseGetMat##Name(const cusparseMatDescr_t d) {                          \
    return descr(d) ? descr(d)->field : Type{};                                                \
  }
VGPU_DESCR_FIELD(Type, cusparseMatrixType_t, type)
VGPU_DESCR_FIELD(FillMode, cusparseFillMode_t, fill)
VGPU_DESCR_FIELD(DiagType, cusparseDiagType_t, diag)
VGPU_DESCR_FIELD(IndexBase, cusparseIndexBase_t, base)
#undef VGPU_DESCR_FIELD

/* ---- legacy index conversions and sorts (32-bit indices, as the API defines them) ---- */

namespace {
std::vector<int> ints(const int* dev, size_t n) {
  std::vector<int> h(n);
  if (n) cudaMemcpy(h.data(), dev, n * sizeof(int), cudaMemcpyDeviceToHost);
  return h;
}
bool put_ints(int* dev, const std::vector<int>& h) {
  return h.empty() || cudaMemcpy(dev, h.data(), h.size() * sizeof(int), cudaMemcpyHostToDevice) == cudaSuccess;
}
// Reorders `keys` (and P alongside) by a stable sort, so P ends as P[i] = the
// original position of the element now at i, composed with P's input.
template <class Less> cusparseStatus_t sort_with(int nnz, int* P, Less less, const std::function<void(const std::vector<int>&)>& apply) {
  std::vector<int> order((size_t)nnz);
  for (int i = 0; i < nnz; ++i) order[i] = i;
  std::stable_sort(order.begin(), order.end(), less);
  apply(order);
  if (P) {
    const std::vector<int> p = ints(P, (size_t)nnz);
    std::vector<int> out((size_t)nnz);
    for (int i = 0; i < nnz; ++i) out[i] = p[order[i]];
    if (!put_ints(P, out)) return CUSPARSE_STATUS_EXECUTION_FAILED;
  }
  return CUSPARSE_STATUS_SUCCESS;
}
}  // namespace

VGPU_EXPORT cusparseStatus_t cusparseXcoo2csr(cusparseHandle_t h, const int* rows, int nnz, int m,
                                              int* offsets, cusparseIndexBase_t base) {
  if (!known(h)) return CUSPARSE_STATUS_NOT_INITIALIZED;
  if (nnz < 0 || m < 0) return CUSPARSE_STATUS_INVALID_VALUE;
  const int b = base == CUSPARSE_INDEX_BASE_ONE ? 1 : 0;
  const std::vector<int> r = ints(rows, (size_t)nnz);
  std::vector<int> off((size_t)m + 1, 0);
  for (int v : r) {
    if (v - b < 0 || v - b >= m) return CUSPARSE_STATUS_INVALID_VALUE;
    ++off[(size_t)(v - b) + 1];
  }
  for (int i = 0; i < m; ++i) off[i + 1] += off[i];
  for (int& v : off) v += b;
  return put_ints(offsets, off) ? CUSPARSE_STATUS_SUCCESS : CUSPARSE_STATUS_EXECUTION_FAILED;
}
VGPU_EXPORT cusparseStatus_t cusparseXcsr2coo(cusparseHandle_t h, const int* offsets, int nnz, int m,
                                              int* rows, cusparseIndexBase_t base) {
  if (!known(h)) return CUSPARSE_STATUS_NOT_INITIALIZED;
  if (nnz < 0 || m < 0) return CUSPARSE_STATUS_INVALID_VALUE;
  const int b = base == CUSPARSE_INDEX_BASE_ONE ? 1 : 0;
  const std::vector<int> off = ints(offsets, (size_t)m + 1);
  std::vector<int> r((size_t)nnz);
  for (int i = 0; i < m; ++i)
    for (int k = off[i] - b; k < off[i + 1] - b; ++k) {
      if (k < 0 || k >= nnz) return CUSPARSE_STATUS_INVALID_VALUE;
      r[k] = i + b;
    }
  return put_ints(rows, r) ? CUSPARSE_STATUS_SUCCESS : CUSPARSE_STATUS_EXECUTION_FAILED;
}
VGPU_EXPORT cusparseStatus_t cusparseCreateIdentityPermutation(cusparseHandle_t h, int n, int* p) {
  if (!known(h)) return CUSPARSE_STATUS_NOT_INITIALIZED;
  if (n < 0) return CUSPARSE_STATUS_INVALID_VALUE;
  std::vector<int> v((size_t)n);
  for (int i = 0; i < n; ++i) v[i] = i;
  return put_ints(p, v) ? CUSPARSE_STATUS_SUCCESS : CUSPARSE_STATUS_EXECUTION_FAILED;
}
// The sorts work on the host, so the scratch buffer they ask for is only a token.
VGPU_EXPORT cusparseStatus_t cusparseXcoosort_bufferSizeExt(cusparseHandle_t h, int, int, int, const int*,
                                                            const int*, size_t* bytes) {
  if (!known(h)) return CUSPARSE_STATUS_NOT_INITIALIZED;
  if (bytes) *bytes = 16;
  return CUSPARSE_STATUS_SUCCESS;
}
static cusparseStatus_t coosort(cusparseHandle_t h, int nnz, int* rows, int* cols, int* P, bool by_row) {
  if (!known(h)) return CUSPARSE_STATUS_NOT_INITIALIZED;
  if (nnz < 0) return CUSPARSE_STATUS_INVALID_VALUE;
  std::vector<int> r = ints(rows, (size_t)nnz), c = ints(cols, (size_t)nnz);
  const std::vector<int>& first = by_row ? r : c;
  const std::vector<int>& second = by_row ? c : r;
  bool ok = true;
  const cusparseStatus_t st = sort_with(
      nnz, P,
      [&](int a, int b) { return first[a] != first[b] ? first[a] < first[b] : second[a] < second[b]; },
      [&](const std::vector<int>& order) {
        std::vector<int> r2((size_t)nnz), c2((size_t)nnz);
        for (int i = 0; i < nnz; ++i) { r2[i] = r[order[i]]; c2[i] = c[order[i]]; }
        ok = put_ints(rows, r2) && put_ints(cols, c2);
      });
  return !ok ? CUSPARSE_STATUS_EXECUTION_FAILED : st;
}
VGPU_EXPORT cusparseStatus_t cusparseXcoosortByRow(cusparseHandle_t h, int, int, int nnz, int* rows, int* cols,
                                                   int* P, void*) {
  return coosort(h, nnz, rows, cols, P, true);
}
VGPU_EXPORT cusparseStatus_t cusparseXcoosortByColumn(cusparseHandle_t h, int, int, int nnz, int* rows, int* cols,
                                                      int* P, void*) {
  return coosort(h, nnz, rows, cols, P, false);
}
VGPU_EXPORT cusparseStatus_t cusparseXcsrsort_bufferSizeExt(cusparseHandle_t h, int, int, int, const int*,
                                                            const int*, size_t* bytes) {
  if (!known(h)) return CUSPARSE_STATUS_NOT_INITIALIZED;
  if (bytes) *bytes = 16;
  return CUSPARSE_STATUS_SUCCESS;
}
// Sorts the column indices within each row.
VGPU_EXPORT cusparseStatus_t cusparseXcsrsort(cusparseHandle_t h, int m, int, int nnz, const cusparseMatDescr_t d,
                                              const int* offsets, int* cols, int* P, void*) {
  if (!known(h)) return CUSPARSE_STATUS_NOT_INITIALIZED;
  if (m < 0 || nnz < 0) return CUSPARSE_STATUS_INVALID_VALUE;
  const int b = base_of(d);
  const std::vector<int> off = ints(offsets, (size_t)m + 1);
  std::vector<int> c = ints(cols, (size_t)nnz), row((size_t)nnz, 0);
  for (int i = 0; i < m; ++i)
    for (int k = off[i] - b; k < off[i + 1] - b; ++k) {
      if (k < 0 || k >= nnz) return CUSPARSE_STATUS_INVALID_VALUE;
      row[k] = i;
    }
  bool ok = true;
  const cusparseStatus_t st = sort_with(
      nnz, P, [&](int x, int y) { return row[x] != row[y] ? row[x] < row[y] : c[x] < c[y]; },
      [&](const std::vector<int>& order) {
        std::vector<int> c2((size_t)nnz);
        for (int i = 0; i < nnz; ++i) c2[i] = c[order[i]];
        ok = put_ints(cols, c2);
      });
  return !ok ? CUSPARSE_STATUS_EXECUTION_FAILED : st;
}

/* ---- csrgeam2: C = alpha * A + beta * B, all CSR with sorted columns ---- */

namespace {
template <class V = double> struct Csr {
  std::vector<int> off, col;
  std::vector<V> val;
};
// One matrix's structure (and values, when `vals` is given) from the device,
// rebased to zero.
template <class V>
bool read_csr(int m, int nnz, cusparseMatDescr_t d, const int* off, const int* col, const void* vals,
              cudaDataType t, Csr<V>* out) {
  const int b = base_of(d);
  out->off = ints(off, (size_t)m + 1);
  out->col = ints(col, (size_t)nnz);
  for (int& v : out->off) v -= b;
  for (int& v : out->col) v -= b;
  if (out->off[0] != 0 || out->off[m] != nnz) return false;
  return !vals || fetch_values(vals, (size_t)nnz, t, &out->val);
}
// The union of the two sparsity patterns, row by row, each row's columns ascending.
template <class V> std::vector<std::vector<int>> union_pattern(int m, const Csr<V>& a, const Csr<V>& b) {
  std::vector<std::vector<int>> rows((size_t)m);
  for (int i = 0; i < m; ++i) {
    std::vector<int>& r = rows[i];
    r.assign(a.col.begin() + a.off[i], a.col.begin() + a.off[i + 1]);
    r.insert(r.end(), b.col.begin() + b.off[i], b.col.begin() + b.off[i + 1]);
    std::sort(r.begin(), r.end());
    r.erase(std::unique(r.begin(), r.end()), r.end());
  }
  return rows;
}
template <class V> V legacy_scalar(cusparseHandle_t h, const void* p, cudaDataType t) {
  return p ? scalar<V>(h, p, t) : V(0.0);
}

// The legacy API's element types: the host type a pointer names, the value
// arithmetic runs in, and the cudaDataType that describes it.
template <class T> struct Legacy;
template <> struct Legacy<float> { using V = double; static constexpr cudaDataType type = CUDA_R_32F; };
template <> struct Legacy<double> { using V = double; static constexpr cudaDataType type = CUDA_R_64F; };
template <> struct Legacy<cuComplex> { using V = cd; static constexpr cudaDataType type = CUDA_C_32F; };
template <> struct Legacy<cuDoubleComplex> { using V = cd; static constexpr cudaDataType type = CUDA_C_64F; };

template <class T>
cusparseStatus_t csrgeam2(cusparseHandle_t h, int m, int n, const T* alpha, cusparseMatDescr_t dA, int nnzA,
                          const T* valA, const int* offA, const int* colA, const T* beta, cusparseMatDescr_t dB,
                          int nnzB, const T* valB, const int* offB, const int* colB, cusparseMatDescr_t dC,
                          T* valC, int* offC, int* colC) {
  using V = typename Legacy<T>::V;
  if (!known(h)) return CUSPARSE_STATUS_NOT_INITIALIZED;
  if (m < 0 || n < 0 || nnzA < 0 || nnzB < 0) return CUSPARSE_STATUS_INVALID_VALUE;
  constexpr cudaDataType t = Legacy<T>::type;
  Csr<V> a, b;
  if (!read_csr(m, nnzA, dA, offA, colA, valA, t, &a) || !read_csr(m, nnzB, dB, offB, colB, valB, t, &b))
    return CUSPARSE_STATUS_INVALID_VALUE;
  const V al = legacy_scalar<V>(h, alpha, t), be = legacy_scalar<V>(h, beta, t);
  const auto rows = union_pattern(m, a, b);
  const int base = base_of(dC);
  std::vector<int> off{base}, col;
  std::vector<V> val;
  for (int i = 0; i < m; ++i) {
    std::map<int, V> acc;
    for (int c : rows[i]) acc[c] = V(0.0);
    for (int k = a.off[i]; k < a.off[i + 1]; ++k) acc[a.col[k]] += al * a.val[k];
    for (int k = b.off[i]; k < b.off[i + 1]; ++k) acc[b.col[k]] += be * b.val[k];
    for (const auto& [c, v] : acc) { col.push_back(c + base); val.push_back(v); }
    off.push_back((int)col.size() + base);
  }
  const bool ok = put_ints(offC, off) && put_ints(colC, col) && store_values(valC, val, t);
  return ok ? CUSPARSE_STATUS_SUCCESS : CUSPARSE_STATUS_EXECUTION_FAILED;
}
}  // namespace

VGPU_EXPORT cusparseStatus_t cusparseXcsrgeam2Nnz(cusparseHandle_t h, int m, int, cusparseMatDescr_t dA, int nnzA,
                                                  const int* offA, const int* colA, cusparseMatDescr_t dB,
                                                  int nnzB, const int* offB, const int* colB,
                                                  cusparseMatDescr_t dC, int* offC, int* nnz_total, void*) {
  if (!known(h)) return CUSPARSE_STATUS_NOT_INITIALIZED;
  if (m < 0 || nnzA < 0 || nnzB < 0) return CUSPARSE_STATUS_INVALID_VALUE;
  Csr<> a, b;
  if (!read_csr(m, nnzA, dA, offA, colA, nullptr, CUDA_R_32F, &a) ||
      !read_csr(m, nnzB, dB, offB, colB, nullptr, CUDA_R_32F, &b))
    return CUSPARSE_STATUS_INVALID_VALUE;
  const auto rows = union_pattern(m, a, b);
  const int base = base_of(dC);
  std::vector<int> off{base};
  for (const auto& r : rows) off.push_back(off.back() + (int)r.size());
  if (!put_ints(offC, off)) return CUSPARSE_STATUS_EXECUTION_FAILED;
  const int total = off.back() - base;
  if (nnz_total) {
    if (reinterpret_cast<Handle*>(h)->mode == CUSPARSE_POINTER_MODE_DEVICE)
      cudaMemcpy(nnz_total, &total, sizeof(int), cudaMemcpyHostToDevice);
    else
      *nnz_total = total;
  }
  return CUSPARSE_STATUS_SUCCESS;
}

#define VGPU_GEAM(P, T)                                                                                          \
  VGPU_EXPORT cusparseStatus_t cusparse##P##csrgeam2_bufferSizeExt(                                              \
      cusparseHandle_t h, int, int, const T*, const cusparseMatDescr_t, int, const T*, const int*, const int*,   \
      const T*, const cusparseMatDescr_t, int, const T*, const int*, const int*, const cusparseMatDescr_t,       \
      const T*, const int*, const int*, size_t* bytes) {                                                         \
    if (!known(h)) return CUSPARSE_STATUS_NOT_INITIALIZED;                                                       \
    if (bytes) *bytes = 16;                                                                                      \
    return CUSPARSE_STATUS_SUCCESS;                                                                              \
  }                                                                                                              \
  VGPU_EXPORT cusparseStatus_t cusparse##P##csrgeam2(                                                            \
      cusparseHandle_t h, int m, int n, const T* alpha, const cusparseMatDescr_t dA, int nnzA, const T* valA,    \
      const int* offA, const int* colA, const T* beta, const cusparseMatDescr_t dB, int nnzB, const T* valB,     \
      const int* offB, const int* colB, const cusparseMatDescr_t dC, T* valC, int* offC, int* colC, void*) {     \
    return csrgeam2<T>(h, m, n, alpha, dA, nnzA, valA, offA, colA, beta, dB, nnzB, valB, offB, colB, dC, valC,   \
                       offC, colC);                                                                              \
  }
VGPU_GEAM(S, float)
VGPU_GEAM(D, double)
VGPU_GEAM(C, cuComplex)
VGPU_GEAM(Z, cuDoubleComplex)
#undef VGPU_GEAM

/* ---- SpGEMM: C = alpha * A * B (+ beta * C), sparse times sparse ---- */

namespace {
// The product is computed by _compute, which is where cuSPARSE learns C's nnz,
// and held here until _copy writes it into the arrays the caller then allocates.
struct SpGEMMDescr {
  std::vector<int64_t> off, col;
  std::vector<cd> val;
};
bool put_index_array(void* dev, const std::vector<int64_t>& v, cusparseIndexType_t t, int64_t base) {
  if (v.empty()) return true;
  if (t == CUSPARSE_INDEX_32I) {
    std::vector<int32_t> tmp(v.size());
    for (size_t i = 0; i < v.size(); ++i) tmp[i] = (int32_t)(v[i] + base);
    return cudaMemcpy(dev, tmp.data(), tmp.size() * 4, cudaMemcpyHostToDevice) == cudaSuccess;
  }
  std::vector<int64_t> tmp(v);
  for (auto& x : tmp) x += base;
  return cudaMemcpy(dev, tmp.data(), tmp.size() * 8, cudaMemcpyHostToDevice) == cudaSuccess;
}
// C = alpha A B (+ beta C over C's own pattern), into the descriptor until _copy.
template <class V>
cusparseStatus_t spgemm(cusparseHandle_t h, const void* alpha, const SpMat& A, const SpMat& B, const void* beta,
                        SpMat& C, cudaDataType ct, SpGEMMDescr* g) {
  Triplets<V> ta, tb;
  if (!expand(A, &ta) || !expand(B, &tb)) return CUSPARSE_STATUS_INTERNAL_ERROR;
  const V al = scalar<V>(h, alpha, ct), be = beta ? scalar<V>(h, beta, ct) : V(0.0);
  std::vector<std::vector<std::pair<int64_t, V>>> brows((size_t)B.rows);
  for (size_t k = 0; k < tb.val.size(); ++k) {
    if (tb.row[k] < 0 || tb.row[k] >= B.rows) return CUSPARSE_STATUS_INVALID_VALUE;
    brows[(size_t)tb.row[k]].push_back({tb.col[k], tb.val[k]});
  }
  std::vector<std::map<int64_t, V>> crows((size_t)A.rows);
  for (size_t k = 0; k < ta.val.size(); ++k) {
    if (ta.row[k] < 0 || ta.row[k] >= A.rows || ta.col[k] < 0 || ta.col[k] >= B.rows)
      return CUSPARSE_STATUS_INVALID_VALUE;
    for (const auto& [c, v] : brows[(size_t)ta.col[k]]) crows[(size_t)ta.row[k]][c] += al * ta.val[k] * v;
  }
  if (be != V(0.0) && C.nnz > 0) {  // an existing C contributes its own pattern
    Triplets<V> tc;
    if (!expand(C, &tc)) return CUSPARSE_STATUS_INTERNAL_ERROR;
    for (size_t k = 0; k < tc.val.size(); ++k) crows[(size_t)tc.row[k]][tc.col[k]] += be * tc.val[k];
  }
  g->off.assign(1, 0);
  g->col.clear();
  g->val.clear();
  for (const auto& r : crows) {
    for (const auto& [c, v] : r) { g->col.push_back(c); g->val.push_back(cd(re(v), im(v))); }
    g->off.push_back((int64_t)g->col.size());
  }
  C.nnz = (int64_t)g->col.size();
  return CUSPARSE_STATUS_SUCCESS;
}
}  // namespace

VGPU_EXPORT cusparseStatus_t cusparseSpGEMM_createDescr(cusparseSpGEMMDescr_t* d) {
  if (!d) return CUSPARSE_STATUS_INVALID_VALUE;
  *d = reinterpret_cast<cusparseSpGEMMDescr_t>(track(new SpGEMMDescr()));
  return CUSPARSE_STATUS_SUCCESS;
}
VGPU_EXPORT cusparseStatus_t cusparseSpGEMM_destroyDescr(cusparseSpGEMMDescr_t d) {
  if (!known(d)) return CUSPARSE_STATUS_INVALID_VALUE;
  untrack(d);
  delete reinterpret_cast<SpGEMMDescr*>(d);
  return CUSPARSE_STATUS_SUCCESS;
}
// Both phases follow the two-call protocol: a NULL buffer asks for its size.
// The size is a token, never zero, so the second call is told apart by its
// non-NULL buffer.
VGPU_EXPORT cusparseStatus_t cusparseSpGEMM_workEstimation(cusparseHandle_t h, cusparseOperation_t,
                                                           cusparseOperation_t, const void*,
                                                           cusparseConstSpMatDescr_t, cusparseConstSpMatDescr_t,
                                                           const void*, cusparseSpMatDescr_t, cudaDataType,
                                                           cusparseSpGEMMAlg_t, cusparseSpGEMMDescr_t d,
                                                           size_t* bytes, void* buffer) {
  if (!known(h) || !known(d)) return CUSPARSE_STATUS_NOT_INITIALIZED;
  if (!buffer && bytes) *bytes = 16;
  return CUSPARSE_STATUS_SUCCESS;
}
VGPU_EXPORT cusparseStatus_t cusparseSpGEMM_compute(cusparseHandle_t h, cusparseOperation_t opA,
                                                    cusparseOperation_t opB, const void* alpha,
                                                    cusparseConstSpMatDescr_t matA, cusparseConstSpMatDescr_t matB,
                                                    const void* beta, cusparseSpMatDescr_t matC, cudaDataType ct,
                                                    cusparseSpGEMMAlg_t, cusparseSpGEMMDescr_t d, size_t* bytes,
                                                    void* buffer) {
  if (!known(h) || !known(d) || !known(matA) || !known(matB) || !known(matC))
    return CUSPARSE_STATUS_NOT_INITIALIZED;
  if (!buffer) {
    if (bytes) *bytes = 16;
    return CUSPARSE_STATUS_SUCCESS;
  }
  // cuSPARSE itself takes only non-transposed operands here, and only CSR.
  if (opA != CUSPARSE_OPERATION_NON_TRANSPOSE || opB != CUSPARSE_OPERATION_NON_TRANSPOSE)
    return CUSPARSE_STATUS_NOT_SUPPORTED;
  const auto& A = *reinterpret_cast<const SpMat*>(matA);
  const auto& B = *reinterpret_cast<const SpMat*>(matB);
  auto& C = *reinterpret_cast<SpMat*>(matC);
  if (A.cols != B.rows || C.rows != A.rows || C.cols != B.cols) return CUSPARSE_STATUS_INVALID_VALUE;
  if (C.format != CUSPARSE_FORMAT_CSR || A.format == kFormatBsr || B.format == kFormatBsr)
    return CUSPARSE_STATUS_NOT_SUPPORTED;
  auto& g = *reinterpret_cast<SpGEMMDescr*>(d);
  return any_complex({A.value_type, B.value_type, C.value_type, ct})
             ? spgemm<cd>(h, alpha, A, B, beta, C, ct, &g)
             : spgemm<double>(h, alpha, A, B, beta, C, ct, &g);
}
VGPU_EXPORT cusparseStatus_t cusparseSpGEMM_copy(cusparseHandle_t h, cusparseOperation_t, cusparseOperation_t,
                                                 const void*, cusparseConstSpMatDescr_t, cusparseConstSpMatDescr_t,
                                                 const void*, cusparseSpMatDescr_t matC, cudaDataType,
                                                 cusparseSpGEMMAlg_t, cusparseSpGEMMDescr_t d) {
  if (!known(h) || !known(d) || !known(matC)) return CUSPARSE_STATUS_NOT_INITIALIZED;
  auto& C = *reinterpret_cast<SpMat*>(matC);
  const auto& g = *reinterpret_cast<const SpGEMMDescr*>(d);
  if (g.off.empty()) return CUSPARSE_STATUS_INVALID_VALUE;  // _compute never ran
  const int64_t base = C.base == CUSPARSE_INDEX_BASE_ONE ? 1 : 0;
  const bool ok = put_index_array(C.rows_ptr, g.off, C.row_type, base) &&
                  put_index_array(C.cols_ptr, g.col, C.col_type, base) &&
                  store_values(C.values, g.val, C.value_type);
  return ok ? CUSPARSE_STATUS_SUCCESS : CUSPARSE_STATUS_INTERNAL_ERROR;
}

/* ---- SDDMM: C = alpha * (op(A) * op(B)) restricted to C's pattern + beta * C ---- */

namespace {
cusparseStatus_t sddmm_check(cusparseOperation_t opA, cusparseOperation_t opB, const DnMat& A, const DnMat& B,
                             const SpMat& C, cudaDataType ct) {
  if (!type_bytes(ct)) return CUSPARSE_STATUS_NOT_SUPPORTED;
  if (bad_conj(opA, A.type) || bad_conj(opB, B.type)) return CUSPARSE_STATUS_INVALID_VALUE;
  // A conjugate transpose of a complex operand is refused. NVIDIA documents
  // only A and A^T here; its 13.0 takes A^H without complaint and computes
  // something else (for op(A) = A^H, one term of each inner product, measured
  // on an RTX 3060), which no caller can mean.
  if (opA == CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE || opB == CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE)
    return CUSPARSE_STATUS_NOT_SUPPORTED;
  if (!complex_combo_ok(C.value_type, A.type, B.type, ct, false)) return CUSPARSE_STATUS_NOT_SUPPORTED;
  // BSR: NVIDIA's takes it when the inner dimension is a whole number of blocks.
  const int64_t k = transposed(opA) ? A.rows : A.cols;
  if (C.format == kFormatBsr && k % C.bdim != 0) return CUSPARSE_STATUS_NOT_SUPPORTED;
  return CUSPARSE_STATUS_SUCCESS;
}

template <class V>
cusparseStatus_t sddmm(cusparseHandle_t h, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha,
                       const DnMat& A, const DnMat& B, const void* beta, const SpMat& C, cudaDataType ct) {
  const int64_t m = transposed(opA) ? A.cols : A.rows, k = transposed(opA) ? A.rows : A.cols;
  const int64_t bk = transposed(opB) ? B.cols : B.rows, n = transposed(opB) ? B.rows : B.cols;
  if (k != bk || C.rows != m || C.cols != n) return CUSPARSE_STATUS_INVALID_VALUE;
  if ((A.batch != 1 && A.batch != C.batch) || (B.batch != 1 && B.batch != C.batch))
    return CUSPARSE_STATUS_INVALID_VALUE;
  const V al = scalar<V>(h, alpha, ct), be = scalar<V>(h, beta, ct);
  for (int i = 0; i < C.batch; ++i) {
    const DnMat a = member(A, i), b = member(B, i);
    const SpMat c = member(C, i);
    std::vector<V> av, bv;
    Triplets<V> t;
    if (!fetch_values(a.values, dn_elems(a), a.type, &av) || !fetch_values(b.values, dn_elems(b), b.type, &bv) ||
        !expand(c, &t))
      return CUSPARSE_STATUS_INTERNAL_ERROR;
    for (size_t e = 0; e < t.val.size(); ++e) {
      const int64_t r = t.row[e], col = t.col[e];
      if (r < 0 || r >= m || col < 0 || col >= n) return CUSPARSE_STATUS_INVALID_VALUE;
      V dot(0.0);
      for (int64_t x = 0; x < k; ++x)
        dot += av[transposed(opA) ? dn_index(a, x, r) : dn_index(a, r, x)] *
               bv[transposed(opB) ? dn_index(b, col, x) : dn_index(b, x, col)];
      t.val[e] = al * dot + (be != V(0.0) ? be * t.val[e] : V(0.0));
    }
    if (!store_values(c.values, t.val, c.value_type)) return CUSPARSE_STATUS_INTERNAL_ERROR;
  }
  return CUSPARSE_STATUS_SUCCESS;
}
}  // namespace

VGPU_EXPORT cusparseStatus_t cusparseSDDMM_bufferSize(cusparseHandle_t h, cusparseOperation_t opA,
                                                      cusparseOperation_t opB, const void*,
                                                      cusparseConstDnMatDescr_t matA, cusparseConstDnMatDescr_t matB,
                                                      const void*, cusparseSpMatDescr_t matC, cudaDataType ct,
                                                      cusparseSDDMMAlg_t, size_t* bytes) {
  if (!known(h)) return CUSPARSE_STATUS_NOT_INITIALIZED;
  if (bytes) *bytes = 0;
  if (known(matA) && known(matB) && known(matC))
    return sddmm_check(opA, opB, *reinterpret_cast<const DnMat*>(matA), *reinterpret_cast<const DnMat*>(matB),
                       *reinterpret_cast<const SpMat*>(matC), ct);
  return CUSPARSE_STATUS_SUCCESS;
}
VGPU_EXPORT cusparseStatus_t cusparseSDDMM_preprocess(cusparseHandle_t h, cusparseOperation_t, cusparseOperation_t,
                                                      const void*, cusparseConstDnMatDescr_t,
                                                      cusparseConstDnMatDescr_t, const void*, cusparseSpMatDescr_t,
                                                      cudaDataType, cusparseSDDMMAlg_t, void*) {
  return known(h) ? CUSPARSE_STATUS_SUCCESS : CUSPARSE_STATUS_NOT_INITIALIZED;
}
VGPU_EXPORT cusparseStatus_t cusparseSDDMM(cusparseHandle_t h, cusparseOperation_t opA, cusparseOperation_t opB,
                                           const void* alpha, cusparseConstDnMatDescr_t matA,
                                           cusparseConstDnMatDescr_t matB, const void* beta,
                                           cusparseSpMatDescr_t matC, cudaDataType ct, cusparseSDDMMAlg_t alg,
                                           void* buffer) {
  if (!known(h) || !known(matA) || !known(matB) || !known(matC)) return CUSPARSE_STATUS_NOT_INITIALIZED;
  if (capturing(h)) {
    auto held = std::make_shared<Held>();
    const cusparseHandle_t hc = held->handle(h);
    const void* a = held->scalar(*reinterpret_cast<Handle*>(hc), alpha, type_bytes(ct));
    const void* b = held->scalar(*reinterpret_cast<Handle*>(hc), beta, type_bytes(ct));
    auto* A = reinterpret_cast<cusparseConstDnMatDescr_t>(held->copy<DnMat>(matA));
    auto* B = reinterpret_cast<cusparseConstDnMatDescr_t>(held->copy<DnMat>(matB));
    auto* C = reinterpret_cast<cusparseSpMatDescr_t>(held->copy<SpMat>(matC));
    return record(h, held, [=] { cusparseSDDMM(hc, opA, opB, a, A, B, b, C, ct, alg, buffer); });
  }
  const auto& A = *reinterpret_cast<const DnMat*>(matA);
  const auto& B = *reinterpret_cast<const DnMat*>(matB);
  const auto& C = *reinterpret_cast<const SpMat*>(matC);
  if (const cusparseStatus_t st = sddmm_check(opA, opB, A, B, C, ct); st != CUSPARSE_STATUS_SUCCESS) return st;
  return any_complex({A.type, B.type, C.value_type, ct}) ? sddmm<cd>(h, opA, opB, alpha, A, B, beta, C, ct)
                                                         : sddmm<double>(h, opA, opB, alpha, A, B, beta, C, ct);
}

/* ---- triangular solves: op(A) y = alpha x (SpSV), op(A) C = alpha op(B) (SpSM) ---- */

namespace {
struct TriDescr {};

// op(A) restricted to the triangle A's fill mode names, as rows of (column,
// value), with its diagonal apart. Entries outside the triangle are ignored,
// as cuSPARSE ignores them.
template <class V> struct Triangle {
  int64_t n = 0;
  bool lower = true;  // of op(A), which a transpose flips
  std::vector<std::vector<std::pair<int64_t, V>>> rows;
  std::vector<V> diag;
  std::vector<char> has_diag;  // whether A stores its diagonal element at all
};
template <class V>
cusparseStatus_t triangle_from(const Triplets<V>& tr, int64_t n, bool lower, bool unit, cusparseOperation_t op,
                               Triangle<V>* t) {
  t->n = n;
  t->lower = transposed(op) ? !lower : lower;
  t->rows.assign((size_t)n, {});
  t->diag.assign((size_t)n, V(unit ? 1.0 : 0.0));
  t->has_diag.assign((size_t)n, 0);
  for (size_t e = 0; e < tr.val.size(); ++e) {
    const int64_t r = tr.row[e], c = tr.col[e];
    if (r < 0 || r >= n || c < 0 || c >= n) return CUSPARSE_STATUS_INVALID_VALUE;
    if (lower ? c > r : c < r) continue;
    const V v = conj_op(tr.val[e], op);
    if (r == c) {
      t->has_diag[(size_t)r] = 1;
      if (!unit) t->diag[(size_t)r] += v;
      continue;
    }
    if (transposed(op)) t->rows[(size_t)c].push_back({r, v});
    else t->rows[(size_t)r].push_back({c, v});
  }
  return CUSPARSE_STATUS_SUCCESS;
}
template <class V> cusparseStatus_t triangle(const SpMat& A, cusparseOperation_t op, Triangle<V>* t) {
  if (A.rows != A.cols) return CUSPARSE_STATUS_INVALID_VALUE;
  Triplets<V> tr;
  if (!expand(A, &tr)) return CUSPARSE_STATUS_INTERNAL_ERROR;
  return triangle_from(tr, A.rows, A.fill == CUSPARSE_FILL_MODE_LOWER, A.diag == CUSPARSE_DIAG_TYPE_UNIT, op, t);
}
// Substitution in place: `x` holds the right-hand side and ends as the solution.
// A zero on the diagonal gives inf/nan where hardware reports a structural zero.
template <class V> void substitute(const Triangle<V>& t, V* x) {
  auto row = [&](int64_t i) {
    V s = x[i];
    for (const auto& [c, v] : t.rows[(size_t)i]) s -= v * x[c];
    x[i] = s / t.diag[(size_t)i];
  };
  if (t.lower) for (int64_t i = 0; i < t.n; ++i) row(i);
  else for (int64_t i = t.n; i-- > 0;) row(i);
}

cusparseStatus_t spsv_check(cusparseOperation_t op, const SpMat& A, cudaDataType x, cudaDataType y, cudaDataType ct) {
  if (!type_bytes(ct)) return CUSPARSE_STATUS_NOT_SUPPORTED;
  if (bad_conj(op, A.value_type)) return CUSPARSE_STATUS_INVALID_VALUE;
  if (!complex_combo_ok(A.value_type, x, y, ct, false)) return CUSPARSE_STATUS_NOT_SUPPORTED;
  if (A.format == kFormatBsr) return CUSPARSE_STATUS_NOT_SUPPORTED;  // CSR, COO (and sliced ELL) only
  return CUSPARSE_STATUS_SUCCESS;
}
cusparseStatus_t spsm_check(cusparseOperation_t opA, cusparseOperation_t opB, const SpMat& A, const DnMat& B,
                            const DnMat& C, cudaDataType ct) {
  if (bad_conj(opB, B.type)) return CUSPARSE_STATUS_INVALID_VALUE;
  const int64_t brows = transposed(opB) ? B.cols : B.rows, ncols = transposed(opB) ? B.rows : B.cols;
  if (brows != A.rows || C.rows != A.rows || C.cols != ncols) return CUSPARSE_STATUS_INVALID_VALUE;
  // NVIDIA's takes B or its transpose, never its conjugate transpose.
  if (opB == CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE) return CUSPARSE_STATUS_NOT_SUPPORTED;
  return spsv_check(opA, A, B.type, C.type, ct);
}

template <class V>
cusparseStatus_t spsv(cusparseHandle_t h, cusparseOperation_t op, const void* alpha, const SpMat& A, const DnVec& X,
                      const DnVec& Y, cudaDataType ct) {
  if (X.size != A.rows || Y.size != A.rows) return CUSPARSE_STATUS_INVALID_VALUE;
  Triangle<V> t;
  if (const cusparseStatus_t st = triangle(A, op, &t); st != CUSPARSE_STATUS_SUCCESS) return st;
  std::vector<V> x;
  if (!fetch_values(X.values, (size_t)X.size, X.type, &x)) return CUSPARSE_STATUS_INTERNAL_ERROR;
  const V al = scalar<V>(h, alpha, ct);
  for (auto& v : x) v *= al;
  substitute(t, x.data());
  return store_values(Y.values, x, Y.type) ? CUSPARSE_STATUS_SUCCESS : CUSPARSE_STATUS_INTERNAL_ERROR;
}

template <class V>
cusparseStatus_t spsm(cusparseHandle_t h, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha,
                      const SpMat& A, const DnMat& B, const DnMat& C, cudaDataType ct) {
  const int64_t brows = transposed(opB) ? B.cols : B.rows, ncols = transposed(opB) ? B.rows : B.cols;
  if (brows != A.rows || C.rows != A.rows || C.cols != ncols) return CUSPARSE_STATUS_INVALID_VALUE;
  Triangle<V> t;
  if (const cusparseStatus_t st = triangle(A, opA, &t); st != CUSPARSE_STATUS_SUCCESS) return st;
  std::vector<V> b, c;
  if (!fetch_values(B.values, dn_elems(B), B.type, &b) || !fetch_values(C.values, dn_elems(C), C.type, &c))
    return CUSPARSE_STATUS_INTERNAL_ERROR;
  const V al = scalar<V>(h, alpha, ct);
  std::vector<V> x((size_t)A.rows);
  for (int64_t j = 0; j < ncols; ++j) {
    for (int64_t i = 0; i < A.rows; ++i)
      x[(size_t)i] = al * b[transposed(opB) ? dn_index(B, j, i) : dn_index(B, i, j)];
    substitute(t, x.data());
    for (int64_t i = 0; i < A.rows; ++i) c[dn_index(C, i, j)] = x[(size_t)i];
  }
  return store_values(C.values, c, C.type) ? CUSPARSE_STATUS_SUCCESS : CUSPARSE_STATUS_INTERNAL_ERROR;
}
}  // namespace

#define VGPU_TRI_DESCR(Kind)                                                                  \
  VGPU_EXPORT cusparseStatus_t cusparse##Kind##_createDescr(cusparse##Kind##Descr_t* d) {     \
    if (!d) return CUSPARSE_STATUS_INVALID_VALUE;                                             \
    *d = reinterpret_cast<cusparse##Kind##Descr_t>(track(new TriDescr()));                    \
    return CUSPARSE_STATUS_SUCCESS;                                                           \
  }                                                                                           \
  VGPU_EXPORT cusparseStatus_t cusparse##Kind##_destroyDescr(cusparse##Kind##Descr_t d) {     \
    if (!known(d)) return CUSPARSE_STATUS_INVALID_VALUE;                                      \
    untrack(d);                                                                               \
    delete reinterpret_cast<TriDescr*>(d);                                                    \
    return CUSPARSE_STATUS_SUCCESS;                                                           \
  }
VGPU_TRI_DESCR(SpSV)
VGPU_TRI_DESCR(SpSM)
#undef VGPU_TRI_DESCR

VGPU_EXPORT cusparseStatus_t cusparseSpSV_bufferSize(cusparseHandle_t h, cusparseOperation_t op, const void*,
                                                     cusparseConstSpMatDescr_t matA, cusparseConstDnVecDescr_t vecX,
                                                     cusparseDnVecDescr_t vecY, cudaDataType ct, cusparseSpSVAlg_t,
                                                     cusparseSpSVDescr_t, size_t* bytes) {
  if (!known(h)) return CUSPARSE_STATUS_NOT_INITIALIZED;
  if (bytes) *bytes = 16;
  if (known(matA) && known(vecX) && known(vecY))
    return spsv_check(op, *reinterpret_cast<const SpMat*>(matA), reinterpret_cast<const DnVec*>(vecX)->type,
                      reinterpret_cast<const DnVec*>(vecY)->type, ct);
  return CUSPARSE_STATUS_SUCCESS;
}
VGPU_EXPORT cusparseStatus_t cusparseSpSV_analysis(cusparseHandle_t h, cusparseOperation_t op, const void*,
                                                   cusparseConstSpMatDescr_t matA, cusparseConstDnVecDescr_t vecX,
                                                   cusparseDnVecDescr_t vecY, cudaDataType ct, cusparseSpSVAlg_t,
                                                   cusparseSpSVDescr_t d, void*) {
  if (!known(h) || !known(d)) return CUSPARSE_STATUS_NOT_INITIALIZED;
  if (known(matA) && known(vecX) && known(vecY))
    return spsv_check(op, *reinterpret_cast<const SpMat*>(matA), reinterpret_cast<const DnVec*>(vecX)->type,
                      reinterpret_cast<const DnVec*>(vecY)->type, ct);
  return CUSPARSE_STATUS_SUCCESS;
}
VGPU_EXPORT cusparseStatus_t cusparseSpSV_solve(cusparseHandle_t h, cusparseOperation_t op, const void* alpha,
                                                cusparseConstSpMatDescr_t matA, cusparseConstDnVecDescr_t vecX,
                                                cusparseDnVecDescr_t vecY, cudaDataType ct, cusparseSpSVAlg_t alg,
                                                cusparseSpSVDescr_t d) {
  if (!known(h) || !known(d) || !known(matA) || !known(vecX) || !known(vecY))
    return CUSPARSE_STATUS_NOT_INITIALIZED;
  // The analysis descriptor is used as it is when the graph runs, and must
  // outlive the graph, as on hardware.
  if (capturing(h)) {
    auto held = std::make_shared<Held>();
    const cusparseHandle_t hc = held->handle(h);
    const void* a = held->scalar(*reinterpret_cast<Handle*>(hc), alpha, type_bytes(ct));
    auto* A = reinterpret_cast<cusparseConstSpMatDescr_t>(held->copy<SpMat>(matA));
    auto* X = reinterpret_cast<cusparseConstDnVecDescr_t>(held->copy<DnVec>(vecX));
    auto* Y = reinterpret_cast<cusparseDnVecDescr_t>(held->copy<DnVec>(vecY));
    return record(h, held, [=] { cusparseSpSV_solve(hc, op, a, A, X, Y, ct, alg, d); });
  }
  const auto& A = *reinterpret_cast<const SpMat*>(matA);
  const auto& X = *reinterpret_cast<const DnVec*>(vecX);
  const auto& Y = *reinterpret_cast<const DnVec*>(vecY);
  if (const cusparseStatus_t st = spsv_check(op, A, X.type, Y.type, ct); st != CUSPARSE_STATUS_SUCCESS) return st;
  return any_complex({A.value_type, X.type, Y.type, ct}) ? spsv<cd>(h, op, alpha, A, X, Y, ct)
                                                         : spsv<double>(h, op, alpha, A, X, Y, ct);
}

VGPU_EXPORT cusparseStatus_t cusparseSpSM_bufferSize(cusparseHandle_t h, cusparseOperation_t opA,
                                                     cusparseOperation_t opB, const void*,
                                                     cusparseConstSpMatDescr_t matA, cusparseConstDnMatDescr_t matB,
                                                     cusparseDnMatDescr_t matC, cudaDataType ct, cusparseSpSMAlg_t,
                                                     cusparseSpSMDescr_t, size_t* bytes) {
  if (!known(h)) return CUSPARSE_STATUS_NOT_INITIALIZED;
  if (bytes) *bytes = 16;
  if (known(matA) && known(matB) && known(matC))
    return spsm_check(opA, opB, *reinterpret_cast<const SpMat*>(matA), *reinterpret_cast<const DnMat*>(matB),
                      *reinterpret_cast<const DnMat*>(matC), ct);
  return CUSPARSE_STATUS_SUCCESS;
}
VGPU_EXPORT cusparseStatus_t cusparseSpSM_analysis(cusparseHandle_t h, cusparseOperation_t opA,
                                                   cusparseOperation_t opB, const void*,
                                                   cusparseConstSpMatDescr_t matA, cusparseConstDnMatDescr_t matB,
                                                   cusparseDnMatDescr_t matC, cudaDataType ct, cusparseSpSMAlg_t,
                                                   cusparseSpSMDescr_t d, void*) {
  if (!known(h) || !known(d)) return CUSPARSE_STATUS_NOT_INITIALIZED;
  if (known(matA) && known(matB) && known(matC))
    return spsm_check(opA, opB, *reinterpret_cast<const SpMat*>(matA), *reinterpret_cast<const DnMat*>(matB),
                      *reinterpret_cast<const DnMat*>(matC), ct);
  return CUSPARSE_STATUS_SUCCESS;
}
VGPU_EXPORT cusparseStatus_t cusparseSpSM_solve(cusparseHandle_t h, cusparseOperation_t opA, cusparseOperation_t opB,
                                                const void* alpha, cusparseConstSpMatDescr_t matA,
                                                cusparseConstDnMatDescr_t matB, cusparseDnMatDescr_t matC,
                                                cudaDataType ct, cusparseSpSMAlg_t alg, cusparseSpSMDescr_t d) {
  if (!known(h) || !known(d) || !known(matA) || !known(matB) || !known(matC))
    return CUSPARSE_STATUS_NOT_INITIALIZED;
  if (capturing(h)) {
    auto held = std::make_shared<Held>();
    const cusparseHandle_t hc = held->handle(h);
    const void* a = held->scalar(*reinterpret_cast<Handle*>(hc), alpha, type_bytes(ct));
    auto* A = reinterpret_cast<cusparseConstSpMatDescr_t>(held->copy<SpMat>(matA));
    auto* B = reinterpret_cast<cusparseConstDnMatDescr_t>(held->copy<DnMat>(matB));
    auto* C = reinterpret_cast<cusparseDnMatDescr_t>(held->copy<DnMat>(matC));
    return record(h, held, [=] { cusparseSpSM_solve(hc, opA, opB, a, A, B, C, ct, alg, d); });
  }
  const auto& A = *reinterpret_cast<const SpMat*>(matA);
  const auto& B = *reinterpret_cast<const DnMat*>(matB);
  const auto& C = *reinterpret_cast<const DnMat*>(matC);
  if (const cusparseStatus_t st = spsm_check(opA, opB, A, B, C, ct); st != CUSPARSE_STATUS_SUCCESS) return st;
  return any_complex({A.value_type, B.type, C.type, ct}) ? spsm<cd>(h, opA, opB, alpha, A, B, C, ct)
                                                         : spsm<double>(h, opA, opB, alpha, A, B, C, ct);
}

/* ---- the blocked (BSR) legacy API, csric02 and csrilu02 ----
   CUDA 12 and 13 still ship these (deprecated). The arithmetic is the generic
   API's: a BSR matrix is read as the matrix its blocks make, every element of a
   stored block an entry, zero or not. What an RTX 3060 showed with cuSPARSE
   13.0 (nvidia/tests/e2e/sparse_bsr_paths.cu):

   - bsrmv and bsrxmv take only A itself (INVALID_VALUE otherwise), blocks of
     at least 2 (INVALID_VALUE for 1), and a general matrix
     (MATRIX_TYPE_NOT_SUPPORTED otherwise); bsrxmv leaves y's unmasked block
     rows untouched;
   - bsrsv2 and bsrsm2 solve with the triangle of the element matrix the fill
     mode names: the other half of each diagonal block is ignored, as are the
     blocks on the other side;
   - bsric02 and bsrilu02 are IC(0) and ILU(0) of that element matrix, done in
     place: IC(0) leaves everything above the diagonal (the upper half of each
     diagonal block too) as it was;
   - a zero pivot is reported by block row: a missing diagonal block at
     analysis (unless the diagonal is unit), a zero on the diagonal at solve or
     factorization time, and for IC(0) a pivot that is not positive, after which
     the factorization carries on with the square root of its magnitude;
     ILU(0)'s numeric boost replaces a pivot whose magnitude is at most tol.
   The buffers these ask for are tokens: the work happens on the host. */

namespace {

template <class V> struct HostBsr {
  int mb = 0, nb = 0, rbd = 1, cbd = 1;
  bool row_major = true;  // CUSPARSE_DIRECTION_ROW: each block stored row by row
  std::vector<int> off, col;  // zero-based
  std::vector<V> val;
  size_t at(int k, int r, int c) const {
    return (size_t)k * rbd * cbd + (row_major ? (size_t)r * cbd + c : (size_t)c * rbd + r);
  }
};

template <class V>
cusparseStatus_t read_bsr(cusparseDirection_t dir, int mb, int nb, int nnzb, cusparseMatDescr_t d, const void* val,
                          const int* off, const int* col, int rbd, int cbd, cudaDataType t, HostBsr<V>* out) {
  if (mb < 0 || nb < 0 || nnzb < 0 || rbd <= 0 || cbd <= 0) return CUSPARSE_STATUS_INVALID_VALUE;
  if (dir != CUSPARSE_DIRECTION_ROW && dir != CUSPARSE_DIRECTION_COLUMN) return CUSPARSE_STATUS_INVALID_VALUE;
  if (!descr(d)) return CUSPARSE_STATUS_INVALID_VALUE;
  out->mb = mb; out->nb = nb; out->rbd = rbd; out->cbd = cbd;
  out->row_major = dir == CUSPARSE_DIRECTION_ROW;
  const int b = base_of(d);
  out->off = ints(off, (size_t)mb + 1);
  out->col = ints(col, (size_t)nnzb);
  for (int& v : out->off) v -= b;
  for (int& v : out->col) v -= b;
  if (out->off[0] != 0 || out->off[(size_t)mb] != nnzb) return CUSPARSE_STATUS_INVALID_VALUE;
  for (int i = 0; i < mb; ++i)
    if (out->off[(size_t)i + 1] < out->off[(size_t)i]) return CUSPARSE_STATUS_INVALID_VALUE;
  for (int c : out->col)
    if (c < 0 || c >= nb) return CUSPARSE_STATUS_INVALID_VALUE;
  if (val && !fetch_values(val, (size_t)nnzb * rbd * cbd, t, &out->val)) return CUSPARSE_STATUS_EXECUTION_FAILED;
  return CUSPARSE_STATUS_SUCCESS;
}

// The element matrix as coordinates, entry e being val[e].
template <class V> Triplets<V> bsr_triplets(const HostBsr<V>& a) {
  Triplets<V> t;
  const size_t bs = (size_t)a.rbd * a.cbd;
  t.row.resize(a.col.size() * bs);
  t.col.resize(t.row.size());
  t.val = a.val;
  for (int i = 0; i < a.mb; ++i)
    for (int k = a.off[(size_t)i]; k < a.off[(size_t)i + 1]; ++k)
      for (int r = 0; r < a.rbd; ++r)
        for (int c = 0; c < a.cbd; ++c) {
          const size_t e = a.at(k, r, c);
          t.row[e] = (int64_t)i * a.rbd + r;
          t.col[e] = (int64_t)a.col[(size_t)k] * a.cbd + c;
        }
  return t;
}

// Each element row as (column, index into val), columns ascending.
template <class V> std::vector<std::vector<std::pair<int, size_t>>> element_rows(const HostBsr<V>& a) {
  std::vector<std::vector<std::pair<int, size_t>>> rows((size_t)a.mb * a.rbd);
  for (int i = 0; i < a.mb; ++i)
    for (int k = a.off[(size_t)i]; k < a.off[(size_t)i + 1]; ++k)
      for (int r = 0; r < a.rbd; ++r)
        for (int c = 0; c < a.cbd; ++c)
          rows[(size_t)i * a.rbd + r].push_back({a.col[(size_t)k] * a.cbd + c, a.at(k, r, c)});
  for (auto& r : rows) std::sort(r.begin(), r.end());
  return rows;
}

// The first block row whose diagonal block is not stored, or -1.
template <class V> int missing_diagonal(const HostBsr<V>& a) {
  for (int i = 0; i < a.mb; ++i) {
    bool found = false;
    for (int k = a.off[(size_t)i]; k < a.off[(size_t)i + 1] && !found; ++k) found = a.col[(size_t)k] == i;
    if (!found) return i;
  }
  return -1;
}

// What the analysis and the factorizations learn, and a numeric boost.
struct LegacyInfo {
  int zero = -1;  // zero-based block row of the first zero pivot
  int base = 0;
  bool boost = false;
  double tol = 0.0;
  cd boost_val{};
};
LegacyInfo* legacy_info(const void* p) { return known(p) ? static_cast<LegacyInfo*>(const_cast<void*>(p)) : nullptr; }
void note_zero(LegacyInfo* info, int block_row) {
  if (block_row >= 0 && (info->zero < 0 || block_row < info->zero)) info->zero = block_row;
}
cusparseStatus_t zero_pivot(cusparseHandle_t h, const void* p, int* position) {
  if (!known(h)) return CUSPARSE_STATUS_NOT_INITIALIZED;
  LegacyInfo* info = legacy_info(p);
  if (!info || !position) return CUSPARSE_STATUS_INVALID_VALUE;
  const int v = info->zero < 0 ? -1 : info->zero + info->base;
  if (reinterpret_cast<Handle*>(h)->mode == CUSPARSE_POINTER_MODE_DEVICE)
    cudaMemcpy(position, &v, sizeof(int), cudaMemcpyHostToDevice);
  else
    *position = v;
  return info->zero < 0 ? CUSPARSE_STATUS_SUCCESS : CUSPARSE_STATUS_ZERO_PIVOT;
}
cusparseStatus_t token_size(cusparseHandle_t h, int* bytes) {
  if (!known(h)) return CUSPARSE_STATUS_NOT_INITIALIZED;
  if (!bytes) return CUSPARSE_STATUS_INVALID_VALUE;
  *bytes = 256;
  return CUSPARSE_STATUS_SUCCESS;
}
cusparseStatus_t token_size(cusparseHandle_t h, size_t* bytes) {
  int b = 0;
  const cusparseStatus_t st = token_size(h, &b);
  if (st == CUSPARSE_STATUS_SUCCESS) *bytes = (size_t)b;
  return st;
}

// y = alpha A x + beta y over the block rows `rows` (all of them, unmasked);
// a block row's blocks are [start[i], end[i]).
template <class T>
cusparseStatus_t bsrmv(cusparseHandle_t h, cusparseDirection_t dir, cusparseOperation_t trans, int mask_size,
                       const int* mask, const int* end, int mb, int nb, int nnzb, const T* alpha,
                       cusparseMatDescr_t d, const T* val, const int* off, const int* col, int bd, const T* x,
                       const T* beta, T* y) {
  using V = typename Legacy<T>::V;
  constexpr cudaDataType t = Legacy<T>::type;
  if (!known(h)) return CUSPARSE_STATUS_NOT_INITIALIZED;
  if (trans != CUSPARSE_OPERATION_NON_TRANSPOSE || bd < 2) return CUSPARSE_STATUS_INVALID_VALUE;
  if (!descr(d)) return CUSPARSE_STATUS_INVALID_VALUE;
  if (descr(d)->type != CUSPARSE_MATRIX_TYPE_GENERAL) return CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
  if (mb < 0 || nb < 0 || nnzb < 0 || mask_size < 0) return CUSPARSE_STATUS_INVALID_VALUE;
  if (mb == 0) return CUSPARSE_STATUS_SUCCESS;
  const int b = base_of(d);
  std::vector<int> start = ints(off, (size_t)mb + (mask ? 0 : 1)), stop, rows;
  std::vector<int> cols = ints(col, (size_t)nnzb);
  std::vector<V> av, xv, yv;
  if (!fetch_values(val, (size_t)nnzb * bd * bd, t, &av) || !fetch_values(x, (size_t)nb * bd, t, &xv) ||
      !fetch_values(y, (size_t)mb * bd, t, &yv))
    return CUSPARSE_STATUS_EXECUTION_FAILED;
  if (mask) {
    stop = ints(end, (size_t)mb);
    rows = ints(mask, (size_t)mask_size);
    for (int& r : rows) r -= b;
  } else {
    stop.assign(start.begin() + 1, start.end());
    rows.resize((size_t)mb);
    for (int i = 0; i < mb; ++i) rows[(size_t)i] = i;
  }
  const V al = legacy_scalar<V>(h, alpha, t), be = legacy_scalar<V>(h, beta, t);
  const bool row_major = dir == CUSPARSE_DIRECTION_ROW;
  for (int i : rows) {
    if (i < 0 || i >= mb) return CUSPARSE_STATUS_INVALID_VALUE;
    for (int r = 0; r < bd; ++r) {
      V s(0.0);
      for (int k = start[(size_t)i] - b; k < stop[(size_t)i] - b; ++k) {
        if (k < 0 || k >= nnzb || cols[(size_t)k] - b < 0 || cols[(size_t)k] - b >= nb)
          return CUSPARSE_STATUS_INVALID_VALUE;
        const size_t blk = (size_t)k * bd * bd, xc = (size_t)(cols[(size_t)k] - b) * bd;
        for (int c = 0; c < bd; ++c)
          s += av[blk + (row_major ? (size_t)r * bd + c : (size_t)c * bd + r)] * xv[xc + c];
      }
      V& out = yv[(size_t)i * bd + r];
      out = al * s + (be != V(0.0) ? be * out : V(0.0));
    }
  }
  return store_values(y, yv, t) ? CUSPARSE_STATUS_SUCCESS : CUSPARSE_STATUS_EXECUTION_FAILED;
}

// C = alpha op(A) op(B) + beta C, C (mb*bd) x n column-major, op(A) = A.
template <class T>
cusparseStatus_t bsrmm(cusparseHandle_t h, cusparseDirection_t dir, cusparseOperation_t transA,
                       cusparseOperation_t transB, int mb, int n, int kb, int nnzb, const T* alpha,
                       cusparseMatDescr_t d, const T* val, const int* off, const int* col, int bd, const T* B,
                       int ldb, const T* beta, T* C, int ldc) {
  using V = typename Legacy<T>::V;
  constexpr cudaDataType t = Legacy<T>::type;
  if (!known(h)) return CUSPARSE_STATUS_NOT_INITIALIZED;
  if (transA != CUSPARSE_OPERATION_NON_TRANSPOSE || transB == CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE)
    return CUSPARSE_STATUS_INVALID_VALUE;
  if (descr(d) && descr(d)->type != CUSPARSE_MATRIX_TYPE_GENERAL) return CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
  const bool tb = transB == CUSPARSE_OPERATION_TRANSPOSE;
  const int m = mb * bd, k = kb * bd;
  if (n < 0 || ldc < std::max(1, m) || ldb < std::max(1, tb ? n : k)) return CUSPARSE_STATUS_INVALID_VALUE;
  HostBsr<V> a;
  if (const cusparseStatus_t st = read_bsr(dir, mb, kb, nnzb, d, val, off, col, bd, bd, t, &a);
      st != CUSPARSE_STATUS_SUCCESS)
    return st;
  if (m == 0 || n == 0) return CUSPARSE_STATUS_SUCCESS;
  std::vector<V> bv, cv;
  if (!fetch_values(B, (size_t)ldb * ((tb ? k : n) - 1) + (tb ? n : k), t, &bv) ||
      !fetch_values(C, (size_t)ldc * (n - 1) + m, t, &cv))
    return CUSPARSE_STATUS_EXECUTION_FAILED;
  const V al = legacy_scalar<V>(h, alpha, t), be = legacy_scalar<V>(h, beta, t);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < m; ++i) {
      V& c = cv[(size_t)j * ldc + i];
      c = be != V(0.0) ? be * c : V(0.0);
    }
  const Triplets<V> tr = bsr_triplets(a);
  for (size_t e = 0; e < tr.val.size(); ++e) {
    const V av = al * tr.val[e];
    for (int j = 0; j < n; ++j)
      cv[(size_t)j * ldc + tr.row[e]] += av * bv[tb ? (size_t)tr.col[e] * ldb + j : (size_t)j * ldb + tr.col[e]];
  }
  return store_values(C, cv, t) ? CUSPARSE_STATUS_SUCCESS : CUSPARSE_STATUS_EXECUTION_FAILED;
}

template <class V> cusparseStatus_t legacy_triangle(const HostBsr<V>& a, cusparseMatDescr_t d,
                                                    cusparseOperation_t op, Triangle<V>* t, LegacyInfo* info) {
  const bool unit = descr(d)->diag == CUSPARSE_DIAG_TYPE_UNIT;
  if (const cusparseStatus_t st = triangle_from(bsr_triplets(a), (int64_t)a.mb * a.rbd,
                                                descr(d)->fill == CUSPARSE_FILL_MODE_LOWER, unit, op, t);
      st != CUSPARSE_STATUS_SUCCESS)
    return st;
  if (!unit)
    for (int64_t i = 0; i < t->n; ++i)
      if (t->has_diag[(size_t)i] && t->diag[(size_t)i] == V(0.0)) note_zero(info, (int)(i / a.rbd));
  return CUSPARSE_STATUS_SUCCESS;
}

template <class T>
cusparseStatus_t bsrsv2_analysis(cusparseHandle_t h, cusparseDirection_t dir, int mb, int nnzb,
                                 cusparseMatDescr_t d, const T* val, const int* off, const int* col, int bd,
                                 void* info_p) {
  using V = typename Legacy<T>::V;
  if (!known(h)) return CUSPARSE_STATUS_NOT_INITIALIZED;
  LegacyInfo* info = legacy_info(info_p);
  if (!info) return CUSPARSE_STATUS_INVALID_VALUE;
  HostBsr<V> a;
  if (const cusparseStatus_t st = read_bsr(dir, mb, mb, nnzb, d, nullptr, off, col, bd, bd, Legacy<T>::type, &a);
      st != CUSPARSE_STATUS_SUCCESS)
    return st;
  (void)val;
  info->zero = -1;
  info->base = base_of(d);
  if (descr(d)->diag != CUSPARSE_DIAG_TYPE_UNIT) note_zero(info, missing_diagonal(a));
  return CUSPARSE_STATUS_SUCCESS;
}

// op(A) X = alpha B for n right-hand sides; with transXY the stored B and X
// are their transposes (n x m).
template <class T>
cusparseStatus_t bsrsm(cusparseHandle_t h, cusparseDirection_t dir, cusparseOperation_t transA, bool transXY, int mb,
                       int n, int nnzb, const T* alpha, cusparseMatDescr_t d, const T* val, const int* off,
                       const int* col, int bd, void* info_p, const T* B, int ldb, T* X, int ldx) {
  using V = typename Legacy<T>::V;
  constexpr cudaDataType t = Legacy<T>::type;
  if (!known(h)) return CUSPARSE_STATUS_NOT_INITIALIZED;
  LegacyInfo* info = legacy_info(info_p);
  if (!info) return CUSPARSE_STATUS_INVALID_VALUE;
  const int m = mb * bd;
  if (n < 0 || ldb < std::max(1, transXY ? n : m) || ldx < std::max(1, transXY ? n : m))
    return CUSPARSE_STATUS_INVALID_VALUE;
  HostBsr<V> a;
  if (const cusparseStatus_t st = read_bsr(dir, mb, mb, nnzb, d, val, off, col, bd, bd, t, &a);
      st != CUSPARSE_STATUS_SUCCESS)
    return st;
  Triangle<V> tri;
  if (const cusparseStatus_t st = legacy_triangle(a, d, transA, &tri, info); st != CUSPARSE_STATUS_SUCCESS)
    return st;
  if (m == 0 || n == 0) return CUSPARSE_STATUS_SUCCESS;
  auto at = [&](int ld, int i, int j) { return transXY ? (size_t)i * ld + j : (size_t)j * ld + i; };
  const size_t bn = transXY ? (size_t)ldb * (m - 1) + n : (size_t)ldb * (n - 1) + m;
  const size_t xn = transXY ? (size_t)ldx * (m - 1) + n : (size_t)ldx * (n - 1) + m;
  std::vector<V> bv, xv;
  if (!fetch_values(B, bn, t, &bv) || !fetch_values(X, xn, t, &xv)) return CUSPARSE_STATUS_EXECUTION_FAILED;
  const V al = legacy_scalar<V>(h, alpha, t);
  std::vector<V> w((size_t)m);
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < m; ++i) w[(size_t)i] = al * bv[at(ldb, i, j)];
    substitute(tri, w.data());
    for (int i = 0; i < m; ++i) xv[at(ldx, i, j)] = w[(size_t)i];
  }
  return store_values(X, xv, t) ? CUSPARSE_STATUS_SUCCESS : CUSPARSE_STATUS_EXECUTION_FAILED;
}

// IC(0) (lower: A = L L^H on A's pattern) or ILU(0), in place on the values.
template <class V>
void factor0(HostBsr<V>& a, bool cholesky, LegacyInfo* info) {
  const auto rows = element_rows(a);
  const int n = a.mb * a.rbd;
  std::vector<long long> diag((size_t)n, -1);
  for (int i = 0; i < n; ++i)
    for (const auto& [c, p] : rows[(size_t)i])
      if (c == i) diag[(size_t)i] = (long long)p;
  std::vector<long long> where((size_t)a.nb * a.cbd, -1);  // row i's column -> index into val
  for (int i = 0; i < n; ++i) {
    for (const auto& [c, p] : rows[(size_t)i]) where[(size_t)c] = (long long)p;
    for (const auto& [k, pk] : rows[(size_t)i]) {
      if (k >= i) break;
      if (diag[(size_t)k] < 0) continue;  // a structural zero, reported at analysis
      if (cholesky) {
        V s = a.val[pk];
        for (const auto& [j, pkj] : rows[(size_t)k]) {
          if (j >= k) break;
          if (where[(size_t)j] >= 0) s -= a.val[(size_t)where[(size_t)j]] * conj_of(a.val[pkj]);
        }
        a.val[pk] = s / a.val[(size_t)diag[(size_t)k]];
      } else {
        a.val[pk] /= a.val[(size_t)diag[(size_t)k]];
        const V l = a.val[pk];
        for (const auto& [j, pkj] : rows[(size_t)k])
          if (j > k && where[(size_t)j] >= 0) a.val[(size_t)where[(size_t)j]] -= l * a.val[pkj];
      }
    }
    if (diag[(size_t)i] >= 0) {
      V& piv = a.val[(size_t)diag[(size_t)i]];
      if (cholesky) {
        double dsq = re(piv);
        for (const auto& [j, pj] : rows[(size_t)i]) {
          if (j >= i) break;
          dsq -= std::norm(cd(re(a.val[pj]), im(a.val[pj])));
        }
        if (!(dsq > 0.0)) note_zero(info, i / a.rbd);
        piv = V(std::sqrt(std::fabs(dsq)));
      } else {
        if (info->boost && std::abs(cd(re(piv), im(piv))) <= info->tol) piv = make_value<V>(info->boost_val.real(), info->boost_val.imag());
        if (piv == V(0.0)) note_zero(info, i / a.rbd);
      }
    }
    for (const auto& [c, p] : rows[(size_t)i]) where[(size_t)c] = -1;
  }
}

}  // namespace

#define VGPU_LEGACY_INFO(Cap, Name)                                                              \
  VGPU_EXPORT cusparseStatus_t cusparseCreate##Cap##Info(Name##Info_t* info) {                   \
    if (!info) return CUSPARSE_STATUS_INVALID_VALUE;                                             \
    *info = reinterpret_cast<Name##Info_t>(track(new LegacyInfo()));                             \
    return CUSPARSE_STATUS_SUCCESS;                                                              \
  }                                                                                              \
  VGPU_EXPORT cusparseStatus_t cusparseDestroy##Cap##Info(Name##Info_t info) {                   \
    if (!legacy_info(info)) return CUSPARSE_STATUS_INVALID_VALUE;                                \
    untrack(info);                                                                               \
    delete reinterpret_cast<LegacyInfo*>(info);                                                  \
    return CUSPARSE_STATUS_SUCCESS;                                                              \
  }                                                                                              \
  VGPU_EXPORT cusparseStatus_t cusparseX##Name##_zeroPivot(cusparseHandle_t h, Name##Info_t info, \
                                                           int* position) {                      \
    return zero_pivot(h, info, position);                                                        \
  }
VGPU_LEGACY_INFO(Bsrsv2, bsrsv2)
VGPU_LEGACY_INFO(Bsrsm2, bsrsm2)
VGPU_LEGACY_INFO(Bsric02, bsric02)
VGPU_LEGACY_INFO(Bsrilu02, bsrilu02)
VGPU_LEGACY_INFO(Csric02, csric02)
VGPU_LEGACY_INFO(Csrilu02, csrilu02)
#undef VGPU_LEGACY_INFO

namespace {
template <class T>
cusparseStatus_t factor_analysis(cusparseHandle_t h, cusparseDirection_t dir, int mb, int nnzb, cusparseMatDescr_t d,
                                 const int* off, const int* col, int bd, void* info_p) {
  using V = typename Legacy<T>::V;
  if (!known(h)) return CUSPARSE_STATUS_NOT_INITIALIZED;
  LegacyInfo* info = legacy_info(info_p);
  if (!info) return CUSPARSE_STATUS_INVALID_VALUE;
  HostBsr<V> a;
  if (const cusparseStatus_t st = read_bsr(dir, mb, mb, nnzb, d, nullptr, off, col, bd, bd, Legacy<T>::type, &a);
      st != CUSPARSE_STATUS_SUCCESS)
    return st;
  info->zero = -1;
  info->base = base_of(d);
  note_zero(info, missing_diagonal(a));
  return CUSPARSE_STATUS_SUCCESS;
}
template <class T>
cusparseStatus_t factor(cusparseHandle_t h, cusparseDirection_t dir, int mb, int nnzb, cusparseMatDescr_t d, T* val,
                        const int* off, const int* col, int bd, void* info_p, bool cholesky) {
  using V = typename Legacy<T>::V;
  if (!known(h)) return CUSPARSE_STATUS_NOT_INITIALIZED;
  LegacyInfo* info = legacy_info(info_p);
  if (!info) return CUSPARSE_STATUS_INVALID_VALUE;
  HostBsr<V> a;
  if (const cusparseStatus_t st = read_bsr(dir, mb, mb, nnzb, d, val, off, col, bd, bd, Legacy<T>::type, &a);
      st != CUSPARSE_STATUS_SUCCESS)
    return st;
  info->base = base_of(d);
  factor0(a, cholesky, info);
  return store_values(val, a.val, Legacy<T>::type) ? CUSPARSE_STATUS_SUCCESS : CUSPARSE_STATUS_EXECUTION_FAILED;
}
template <class T> cusparseStatus_t numeric_boost(cusparseHandle_t h, void* info_p, int enable, const double* tol,
                                                  const T* boost) {
  using V = typename Legacy<T>::V;
  if (!known(h)) return CUSPARSE_STATUS_NOT_INITIALIZED;
  LegacyInfo* info = legacy_info(info_p);
  if (!info) return CUSPARSE_STATUS_INVALID_VALUE;
  info->boost = enable != 0;
  if (!info->boost) return CUSPARSE_STATUS_SUCCESS;
  if (!tol || !boost) return CUSPARSE_STATUS_INVALID_VALUE;
  std::vector<double> t1;
  if (reinterpret_cast<Handle*>(h)->mode == CUSPARSE_POINTER_MODE_DEVICE) {
    if (!fetch_values(tol, 1, CUDA_R_64F, &t1)) return CUSPARSE_STATUS_EXECUTION_FAILED;
    info->tol = t1[0];
  } else {
    info->tol = *tol;
  }
  const V b = scalar<V>(h, boost, Legacy<T>::type);
  info->boost_val = cd(re(b), im(b));
  return CUSPARSE_STATUS_SUCCESS;
}

// CSR to a general BSR of rbd x cbd blocks: the pattern of nonzero blocks.
std::vector<std::vector<int>> block_pattern(int m, int n, const std::vector<int>& off, const std::vector<int>& col,
                                            int rbd, int cbd) {
  const int mb = (m + rbd - 1) / rbd;
  std::vector<std::vector<int>> rows((size_t)mb);
  for (int i = 0; i < m; ++i)
    for (int k = off[(size_t)i]; k < off[(size_t)i + 1]; ++k)
      if (col[(size_t)k] >= 0 && col[(size_t)k] < n) rows[(size_t)(i / rbd)].push_back(col[(size_t)k] / cbd);
  for (auto& r : rows) {
    std::sort(r.begin(), r.end());
    r.erase(std::unique(r.begin(), r.end()), r.end());
  }
  return rows;
}
bool read_legacy_csr(int m, int nnz_hint, cusparseMatDescr_t d, const int* off, const int* col, std::vector<int>* o,
                     std::vector<int>* c) {
  const int b = base_of(d);
  *o = ints(off, (size_t)m + 1);
  for (int& v : *o) v -= b;
  if ((*o)[0] != 0) return false;
  const int nnz = nnz_hint >= 0 ? nnz_hint : (*o)[(size_t)m];
  if (nnz < 0) return false;
  *c = ints(col, (size_t)nnz);
  for (int& v : *c) v -= b;
  return true;
}
cusparseStatus_t csr2gebsr_nnz(cusparseHandle_t h, int m, int n, cusparseMatDescr_t dA, const int* off, const int* col,
                               cusparseMatDescr_t dC, int* offC, int rbd, int cbd, int* nnz_total) {
  if (!known(h)) return CUSPARSE_STATUS_NOT_INITIALIZED;
  if (m < 0 || n < 0 || rbd <= 0 || cbd <= 0 || !descr(dA) || !descr(dC)) return CUSPARSE_STATUS_INVALID_VALUE;
  std::vector<int> o, c;
  if (!read_legacy_csr(m, -1, dA, off, col, &o, &c)) return CUSPARSE_STATUS_INVALID_VALUE;
  const auto rows = block_pattern(m, n, o, c, rbd, cbd);
  const int base = base_of(dC);
  std::vector<int> out{base};
  for (const auto& r : rows) out.push_back(out.back() + (int)r.size());
  if (!put_ints(offC, out)) return CUSPARSE_STATUS_EXECUTION_FAILED;
  const int total = out.back() - base;
  if (nnz_total) {
    if (reinterpret_cast<Handle*>(h)->mode == CUSPARSE_POINTER_MODE_DEVICE)
      cudaMemcpy(nnz_total, &total, sizeof(int), cudaMemcpyHostToDevice);
    else
      *nnz_total = total;
  }
  return CUSPARSE_STATUS_SUCCESS;
}
// Blocks padded with zeros where the CSR matrix has no entry, block columns ascending.
template <class T>
cusparseStatus_t csr2gebsr(cusparseHandle_t h, cusparseDirection_t dir, int m, int n, cusparseMatDescr_t dA,
                           const T* val, const int* off, const int* col, cusparseMatDescr_t dC, T* valC, int* offC,
                           int* colC, int rbd, int cbd) {
  using V = typename Legacy<T>::V;
  constexpr cudaDataType t = Legacy<T>::type;
  if (!known(h)) return CUSPARSE_STATUS_NOT_INITIALIZED;
  if (m < 0 || n < 0 || rbd <= 0 || cbd <= 0 || !descr(dA) || !descr(dC)) return CUSPARSE_STATUS_INVALID_VALUE;
  if (dir != CUSPARSE_DIRECTION_ROW && dir != CUSPARSE_DIRECTION_COLUMN) return CUSPARSE_STATUS_INVALID_VALUE;
  std::vector<int> o, c;
  if (!read_legacy_csr(m, -1, dA, off, col, &o, &c)) return CUSPARSE_STATUS_INVALID_VALUE;
  std::vector<V> v;
  if (!fetch_values(val, c.size(), t, &v)) return CUSPARSE_STATUS_EXECUTION_FAILED;
  const auto rows = block_pattern(m, n, o, c, rbd, cbd);
  const int base = base_of(dC);
  HostBsr<V> out;
  out.mb = (int)rows.size();
  out.rbd = rbd;
  out.cbd = cbd;
  out.row_major = dir == CUSPARSE_DIRECTION_ROW;
  out.off.push_back(0);
  for (const auto& r : rows) {
    out.col.insert(out.col.end(), r.begin(), r.end());
    out.off.push_back((int)out.col.size());
  }
  out.val.assign(out.col.size() * rbd * cbd, V(0.0));
  for (int i = 0; i < m; ++i) {
    const int bi = i / rbd;
    for (int k = o[(size_t)i]; k < o[(size_t)i + 1]; ++k) {
      const int j = c[(size_t)k];
      if (j < 0 || j >= n) return CUSPARSE_STATUS_INVALID_VALUE;
      const auto first = out.col.begin() + out.off[(size_t)bi], last = out.col.begin() + out.off[(size_t)bi + 1];
      const int blk = (int)(std::lower_bound(first, last, j / cbd) - out.col.begin());
      out.val[out.at(blk, i % rbd, j % cbd)] += v[(size_t)k];
    }
  }
  std::vector<int> offs(out.off), cols(out.col);
  for (int& x : offs) x += base;
  for (int& x : cols) x += base;
  const bool ok = put_ints(offC, offs) && put_ints(colC, cols) && store_values(valC, out.val, t);
  return ok ? CUSPARSE_STATUS_SUCCESS : CUSPARSE_STATUS_EXECUTION_FAILED;
}
// Every element of every block becomes a CSR entry, block by block along a row.
template <class T>
cusparseStatus_t gebsr2csr(cusparseHandle_t h, cusparseDirection_t dir, int mb, int nb, cusparseMatDescr_t dA,
                           const T* val, const int* off, const int* col, int rbd, int cbd, cusparseMatDescr_t dC,
                           T* valC, int* offC, int* colC) {
  using V = typename Legacy<T>::V;
  constexpr cudaDataType t = Legacy<T>::type;
  if (!known(h)) return CUSPARSE_STATUS_NOT_INITIALIZED;
  if (!descr(dA) || !descr(dC)) return CUSPARSE_STATUS_INVALID_VALUE;
  const std::vector<int> o0 = ints(off, (size_t)std::max(mb, 0) + 1);
  const int nnzb = mb >= 0 ? o0[(size_t)mb] - o0[0] : -1;
  HostBsr<V> a;
  if (const cusparseStatus_t st = read_bsr(dir, mb, nb, nnzb, dA, val, off, col, rbd, cbd, t, &a);
      st != CUSPARSE_STATUS_SUCCESS)
    return st;
  const int base = base_of(dC);
  std::vector<int> offs{base}, cols;
  std::vector<V> vals;
  for (int i = 0; i < mb; ++i)
    for (int r = 0; r < rbd; ++r) {
      for (int k = a.off[(size_t)i]; k < a.off[(size_t)i + 1]; ++k)
        for (int c = 0; c < cbd; ++c) {
          cols.push_back(a.col[(size_t)k] * cbd + c + base);
          if (val) vals.push_back(a.val[a.at(k, r, c)]);
        }
      offs.push_back((int)cols.size() + base);
    }
  const bool ok = put_ints(offC, offs) && put_ints(colC, cols) && (!val || store_values(valC, vals, t));
  return ok ? CUSPARSE_STATUS_SUCCESS : CUSPARSE_STATUS_EXECUTION_FAILED;
}
}  // namespace

VGPU_EXPORT cusparseStatus_t cusparseXcsr2bsrNnz(cusparseHandle_t h, cusparseDirection_t, int m, int n,
                                                 const cusparseMatDescr_t dA, const int* off, const int* col, int bd,
                                                 const cusparseMatDescr_t dC, int* offC, int* nnz_total) {
  return csr2gebsr_nnz(h, m, n, dA, off, col, dC, offC, bd, bd, nnz_total);
}
VGPU_EXPORT cusparseStatus_t cusparseXcsr2gebsrNnz(cusparseHandle_t h, cusparseDirection_t, int m, int n,
                                                   const cusparseMatDescr_t dA, const int* off, const int* col,
                                                   const cusparseMatDescr_t dC, int* offC, int rbd, int cbd,
                                                   int* nnz_total, void*) {
  return csr2gebsr_nnz(h, m, n, dA, off, col, dC, offC, rbd, cbd, nnz_total);
}
VGPU_EXPORT cusparseStatus_t cusparseXgebsr2csr(cusparseHandle_t h, cusparseDirection_t dir, int mb, int nb,
                                                const cusparseMatDescr_t dA, const int* off, const int* col, int rbd,
                                                int cbd, const cusparseMatDescr_t dC, int* offC, int* colC) {
  return gebsr2csr<float>(h, dir, mb, nb, dA, nullptr, off, col, rbd, cbd, dC, nullptr, offC, colC);
}

#define VGPU_LEGACY_BSR(P, T)                                                                                       \
  VGPU_EXPORT cusparseStatus_t cusparse##P##bsrmv(                                                                  \
      cusparseHandle_t h, cusparseDirection_t dir, cusparseOperation_t trans, int mb, int nb, int nnzb,            \
      const T* alpha, const cusparseMatDescr_t d, const T* val, const int* off, const int* col, int bd, const T* x, \
      const T* beta, T* y) {                                                                                        \
    return bsrmv<T>(h, dir, trans, 0, nullptr, nullptr, mb, nb, nnzb, alpha, d, val, off, col, bd, x, beta, y);     \
  }                                                                                                                 \
  VGPU_EXPORT cusparseStatus_t cusparse##P##bsrxmv(                                                                 \
      cusparseHandle_t h, cusparseDirection_t dir, cusparseOperation_t trans, int mask_size, int mb, int nb,       \
      int nnzb, const T* alpha, const cusparseMatDescr_t d, const T* val, const int* mask, const int* off,          \
      const int* end, const int* col, int bd, const T* x, const T* beta, T* y) {                                    \
    if (!mask || !end) return CUSPARSE_STATUS_INVALID_VALUE;                                                        \
    return bsrmv<T>(h, dir, trans, mask_size, mask, end, mb, nb, nnzb, alpha, d, val, off, col, bd, x, beta, y);    \
  }                                                                                                                 \
  VGPU_EXPORT cusparseStatus_t cusparse##P##bsrmm(                                                                  \
      cusparseHandle_t h, cusparseDirection_t dir, cusparseOperation_t transA, cusparseOperation_t transB, int mb, \
      int n, int kb, int nnzb, const T* alpha, const cusparseMatDescr_t d, const T* val, const int* off,            \
      const int* col, const int bd, const T* B, const int ldb, const T* beta, T* C, int ldc) {                      \
    return bsrmm<T>(h, dir, transA, transB, mb, n, kb, nnzb, alpha, d, val, off, col, bd, B, ldb, beta, C, ldc);    \
  }                                                                                                                 \
  VGPU_EXPORT cusparseStatus_t cusparse##P##bsrsv2_bufferSize(cusparseHandle_t h, cusparseDirection_t,             \
                                                              cusparseOperation_t, int, int,                       \
                                                              const cusparseMatDescr_t, T*, const int*,            \
                                                              const int*, int, bsrsv2Info_t, int* bytes) {         \
    return token_size(h, bytes);                                                                                    \
  }                                                                                                                 \
  VGPU_EXPORT cusparseStatus_t cusparse##P##bsrsv2_bufferSizeExt(cusparseHandle_t h, cusparseDirection_t,          \
                                                                 cusparseOperation_t, int, int,                    \
                                                                 const cusparseMatDescr_t, T*, const int*,         \
                                                                 const int*, int, bsrsv2Info_t, size_t* bytes) {   \
    return token_size(h, bytes);                                                                                    \
  }                                                                                                                 \
  VGPU_EXPORT cusparseStatus_t cusparse##P##bsrsv2_analysis(                                                        \
      cusparseHandle_t h, cusparseDirection_t dir, cusparseOperation_t, int mb, int nnzb,                          \
      const cusparseMatDescr_t d, const T* val, const int* off, const int* col, int bd, bsrsv2Info_t info,         \
      cusparseSolvePolicy_t, void*) {                                                                               \
    return bsrsv2_analysis<T>(h, dir, mb, nnzb, d, val, off, col, bd, info);                                        \
  }                                                                                                                 \
  VGPU_EXPORT cusparseStatus_t cusparse##P##bsrsv2_solve(                                                           \
      cusparseHandle_t h, cusparseDirection_t dir, cusparseOperation_t trans, int mb, int nnzb, const T* alpha,    \
      const cusparseMatDescr_t d, const T* val, const int* off, const int* col, int bd, bsrsv2Info_t info,         \
      const T* f, T* x, cusparseSolvePolicy_t, void*) {                                                             \
    return bsrsm<T>(h, dir, trans, false, mb, 1, nnzb, alpha, d, val, off, col, bd, info, f, std::max(1, mb * bd),  \
                    x, std::max(1, mb * bd));                                                                       \
  }                                                                                                                 \
  VGPU_EXPORT cusparseStatus_t cusparse##P##bsrsm2_bufferSize(cusparseHandle_t h, cusparseDirection_t,             \
                                                              cusparseOperation_t, cusparseOperation_t, int, int,  \
                                                              int, const cusparseMatDescr_t, T*, const int*,       \
                                                              const int*, int, bsrsm2Info_t, int* bytes) {         \
    return token_size(h, bytes);                                                                                    \
  }                                                                                                                 \
  VGPU_EXPORT cusparseStatus_t cusparse##P##bsrsm2_bufferSizeExt(cusparseHandle_t h, cusparseDirection_t,          \
                                                                 cusparseOperation_t, cusparseOperation_t, int,    \
                                                                 int, int, const cusparseMatDescr_t, T*,           \
                                                                 const int*, const int*, int, bsrsm2Info_t,        \
                                                                 size_t* bytes) {                                  \
    return token_size(h, bytes);                                                                                    \
  }                                                                                                                 \
  VGPU_EXPORT cusparseStatus_t cusparse##P##bsrsm2_analysis(                                                        \
      cusparseHandle_t h, cusparseDirection_t dir, cusparseOperation_t, cusparseOperation_t, int mb, int,          \
      int nnzb, const cusparseMatDescr_t d, const T* val, const int* off, const int* col, int bd,                  \
      bsrsm2Info_t info, cusparseSolvePolicy_t, void*) {                                                            \
    return bsrsv2_analysis<T>(h, dir, mb, nnzb, d, val, off, col, bd, info);                                        \
  }                                                                                                                 \
  VGPU_EXPORT cusparseStatus_t cusparse##P##bsrsm2_solve(                                                           \
      cusparseHandle_t h, cusparseDirection_t dir, cusparseOperation_t transA, cusparseOperation_t transXY,        \
      int mb, int n, int nnzb, const T* alpha, const cusparseMatDescr_t d, const T* val, const int* off,           \
      const int* col, int bd, bsrsm2Info_t info, const T* B, int ldb, T* X, int ldx, cusparseSolvePolicy_t,        \
      void*) {                                                                                                      \
    if (transXY == CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE) return CUSPARSE_STATUS_INVALID_VALUE;                    \
    return bsrsm<T>(h, dir, transA, transXY == CUSPARSE_OPERATION_TRANSPOSE, mb, n, nnzb, alpha, d, val, off, col, \
                    bd, info, B, ldb, X, ldx);                                                                      \
  }                                                                                                                 \
  VGPU_EXPORT cusparseStatus_t cusparse##P##bsric02_bufferSize(cusparseHandle_t h, cusparseDirection_t, int, int,  \
                                                               const cusparseMatDescr_t, T*, const int*,           \
                                                               const int*, int, bsric02Info_t, int* bytes) {       \
    return token_size(h, bytes);                                                                                    \
  }                                                                                                                 \
  VGPU_EXPORT cusparseStatus_t cusparse##P##bsric02_bufferSizeExt(cusparseHandle_t h, cusparseDirection_t, int,    \
                                                                  int, const cusparseMatDescr_t, T*, const int*,   \
                                                                  const int*, int, bsric02Info_t, size_t* bytes) { \
    return token_size(h, bytes);                                                                                    \
  }                                                                                                                 \
  VGPU_EXPORT cusparseStatus_t cusparse##P##bsric02_analysis(                                                       \
      cusparseHandle_t h, cusparseDirection_t dir, int mb, int nnzb, const cusparseMatDescr_t d, const T*,         \
      const int* off, const int* col, int bd, bsric02Info_t info, cusparseSolvePolicy_t, void*) {                  \
    return factor_analysis<T>(h, dir, mb, nnzb, d, off, col, bd, info);                                            \
  }                                                                                                                 \
  VGPU_EXPORT cusparseStatus_t cusparse##P##bsric02(cusparseHandle_t h, cusparseDirection_t dir, int mb, int nnzb, \
                                                    const cusparseMatDescr_t d, T* val, const int* off,            \
                                                    const int* col, int bd, bsric02Info_t info,                    \
                                                    cusparseSolvePolicy_t, void*) {                                \
    return factor<T>(h, dir, mb, nnzb, d, val, off, col, bd, info, true);                                           \
  }                                                                                                                 \
  VGPU_EXPORT cusparseStatus_t cusparse##P##bsrilu02_numericBoost(cusparseHandle_t h, bsrilu02Info_t info,         \
                                                                  int enable, double* tol, T* boost) {             \
    return numeric_boost<T>(h, info, enable, tol, boost);                                                           \
  }                                                                                                                 \
  VGPU_EXPORT cusparseStatus_t cusparse##P##bsrilu02_bufferSize(cusparseHandle_t h, cusparseDirection_t, int, int, \
                                                                const cusparseMatDescr_t, T*, const int*,          \
                                                                const int*, int, bsrilu02Info_t, int* bytes) {     \
    return token_size(h, bytes);                                                                                    \
  }                                                                                                                 \
  VGPU_EXPORT cusparseStatus_t cusparse##P##bsrilu02_bufferSizeExt(                                                 \
      cusparseHandle_t h, cusparseDirection_t, int, int, const cusparseMatDescr_t, T*, const int*, const int*, int, \
      bsrilu02Info_t, size_t* bytes) {                                                                              \
    return token_size(h, bytes);                                                                                    \
  }                                                                                                                 \
  VGPU_EXPORT cusparseStatus_t cusparse##P##bsrilu02_analysis(                                                      \
      cusparseHandle_t h, cusparseDirection_t dir, int mb, int nnzb, const cusparseMatDescr_t d, T*,               \
      const int* off, const int* col, int bd, bsrilu02Info_t info, cusparseSolvePolicy_t, void*) {                 \
    return factor_analysis<T>(h, dir, mb, nnzb, d, off, col, bd, info);                                            \
  }                                                                                                                 \
  VGPU_EXPORT cusparseStatus_t cusparse##P##bsrilu02(cusparseHandle_t h, cusparseDirection_t dir, int mb,          \
                                                     int nnzb, const cusparseMatDescr_t d, T* val, const int* off, \
                                                     const int* col, int bd, bsrilu02Info_t info,                  \
                                                     cusparseSolvePolicy_t, void*) {                               \
    return factor<T>(h, dir, mb, nnzb, d, val, off, col, bd, info, false);                                          \
  }                                                                                                                 \
  VGPU_EXPORT cusparseStatus_t cusparse##P##csric02_bufferSize(cusparseHandle_t h, int, int,                       \
                                                               const cusparseMatDescr_t, T*, const int*,           \
                                                               const int*, csric02Info_t, int* bytes) {            \
    return token_size(h, bytes);                                                                                    \
  }                                                                                                                 \
  VGPU_EXPORT cusparseStatus_t cusparse##P##csric02_bufferSizeExt(cusparseHandle_t h, int, int,                    \
                                                                  const cusparseMatDescr_t, T*, const int*,        \
                                                                  const int*, csric02Info_t, size_t* bytes) {      \
    return token_size(h, bytes);                                                                                    \
  }                                                                                                                 \
  VGPU_EXPORT cusparseStatus_t cusparse##P##csric02_analysis(cusparseHandle_t h, int m, int nnz,                   \
                                                             const cusparseMatDescr_t d, const T*, const int* off, \
                                                             const int* col, csric02Info_t info,                   \
                                                             cusparseSolvePolicy_t, void*) {                       \
    return factor_analysis<T>(h, CUSPARSE_DIRECTION_ROW, m, nnz, d, off, col, 1, info);                            \
  }                                                                                                                 \
  VGPU_EXPORT cusparseStatus_t cusparse##P##csric02(cusparseHandle_t h, int m, int nnz, const cusparseMatDescr_t d, \
                                                    T* val, const int* off, const int* col, csric02Info_t info,    \
                                                    cusparseSolvePolicy_t, void*) {                                \
    return factor<T>(h, CUSPARSE_DIRECTION_ROW, m, nnz, d, val, off, col, 1, info, true);                           \
  }                                                                                                                 \
  VGPU_EXPORT cusparseStatus_t cusparse##P##csrilu02_numericBoost(cusparseHandle_t h, csrilu02Info_t info,         \
                                                                  int enable, double* tol, T* boost) {             \
    return numeric_boost<T>(h, info, enable, tol, boost);                                                           \
  }                                                                                                                 \
  VGPU_EXPORT cusparseStatus_t cusparse##P##csrilu02_bufferSize(cusparseHandle_t h, int, int,                      \
                                                                const cusparseMatDescr_t, T*, const int*,          \
                                                                const int*, csrilu02Info_t, int* bytes) {          \
    return token_size(h, bytes);                                                                                    \
  }                                                                                                                 \
  VGPU_EXPORT cusparseStatus_t cusparse##P##csrilu02_bufferSizeExt(cusparseHandle_t h, int, int,                   \
                                                                   const cusparseMatDescr_t, T*, const int*,       \
                                                                   const int*, csrilu02Info_t, size_t* bytes) {    \
    return token_size(h, bytes);                                                                                    \
  }                                                                                                                 \
  VGPU_EXPORT cusparseStatus_t cusparse##P##csrilu02_analysis(cusparseHandle_t h, int m, int nnz,                  \
                                                              const cusparseMatDescr_t d, const T*,                \
                                                              const int* off, const int* col,                      \
                                                              csrilu02Info_t info, cusparseSolvePolicy_t, void*) { \
    return factor_analysis<T>(h, CUSPARSE_DIRECTION_ROW, m, nnz, d, off, col, 1, info);                            \
  }                                                                                                                 \
  VGPU_EXPORT cusparseStatus_t cusparse##P##csrilu02(cusparseHandle_t h, int m, int nnz,                           \
                                                     const cusparseMatDescr_t d, T* val, const int* off,           \
                                                     const int* col, csrilu02Info_t info, cusparseSolvePolicy_t,   \
                                                     void*) {                                                      \
    return factor<T>(h, CUSPARSE_DIRECTION_ROW, m, nnz, d, val, off, col, 1, info, false);                          \
  }                                                                                                                 \
  VGPU_EXPORT cusparseStatus_t cusparse##P##csr2bsr(                                                                \
      cusparseHandle_t h, cusparseDirection_t dir, int m, int n, const cusparseMatDescr_t dA, const T* val,        \
      const int* off, const int* col, int bd, const cusparseMatDescr_t dC, T* valC, int* offC, int* colC) {        \
    return csr2gebsr<T>(h, dir, m, n, dA, val, off, col, dC, valC, offC, colC, bd, bd);                            \
  }                                                                                                                 \
  VGPU_EXPORT cusparseStatus_t cusparse##P##bsr2csr(                                                                \
      cusparseHandle_t h, cusparseDirection_t dir, int mb, int nb, const cusparseMatDescr_t dA, const T* val,      \
      const int* off, const int* col, int bd, const cusparseMatDescr_t dC, T* valC, int* offC, int* colC) {        \
    return gebsr2csr<T>(h, dir, mb, nb, dA, val, off, col, bd, bd, dC, valC, offC, colC);                          \
  }                                                                                                                 \
  VGPU_EXPORT cusparseStatus_t cusparse##P##csr2gebsr_bufferSize(cusparseHandle_t h, cusparseDirection_t, int,     \
                                                                 int, const cusparseMatDescr_t, const T*,          \
                                                                 const int*, const int*, int, int, int* bytes) {   \
    return token_size(h, bytes);                                                                                    \
  }                                                                                                                 \
  VGPU_EXPORT cusparseStatus_t cusparse##P##csr2gebsr_bufferSizeExt(cusparseHandle_t h, cusparseDirection_t, int,  \
                                                                    int, const cusparseMatDescr_t, const T*,       \
                                                                    const int*, const int*, int, int,              \
                                                                    size_t* bytes) {                               \
    return token_size(h, bytes);                                                                                    \
  }                                                                                                                 \
  VGPU_EXPORT cusparseStatus_t cusparse##P##csr2gebsr(                                                              \
      cusparseHandle_t h, cusparseDirection_t dir, int m, int n, const cusparseMatDescr_t dA, const T* val,        \
      const int* off, const int* col, const cusparseMatDescr_t dC, T* valC, int* offC, int* colC, int rbd,         \
      int cbd, void*) {                                                                                             \
    return csr2gebsr<T>(h, dir, m, n, dA, val, off, col, dC, valC, offC, colC, rbd, cbd);                          \
  }                                                                                                                 \
  VGPU_EXPORT cusparseStatus_t cusparse##P##gebsr2csr(                                                              \
      cusparseHandle_t h, cusparseDirection_t dir, int mb, int nb, const cusparseMatDescr_t dA, const T* val,      \
      const int* off, const int* col, int rbd, int cbd, const cusparseMatDescr_t dC, T* valC, int* offC,           \
      int* colC) {                                                                                                  \
    return gebsr2csr<T>(h, dir, mb, nb, dA, val, off, col, rbd, cbd, dC, valC, offC, colC);                        \
  }
VGPU_LEGACY_BSR(S, float)
VGPU_LEGACY_BSR(D, double)
VGPU_LEGACY_BSR(C, cuComplex)
VGPU_LEGACY_BSR(Z, cuDoubleComplex)
#undef VGPU_LEGACY_BSR
