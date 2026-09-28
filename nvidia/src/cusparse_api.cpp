// libvgpucusparse -- VirtualGPU's cuSPARSE, presented as libcusparse.so.12.
//
// The generic (descriptor) API: CSR, CSC and COO sparse matrices, dense vectors
// and matrices, strided batches of them, SpMV, SpMM, SpGEMM, SDDMM, triangular
// solves (SpSV, SpSM) and conversion in both directions. Of the legacy API, what
// CUDA 12 still ships and PyTorch calls: matrix descriptors, coo2csr, the COO
// and CSR sorts, and csrgeam2 (C = alpha A + beta B). The BSR routines are not
// implemented.
//
// Like the other vendor libraries the arithmetic runs on the host, in double,
// and is rounded to the requested type on the way out. Anything unimplemented
// returns CUSPARSE_STATUS_NOT_SUPPORTED so a caller can fall back rather than
// receive a plausible wrong answer.
#include <cusparse.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <functional>
#include <mutex>
#include <map>
#include <set>
#include <vector>

#include <cuda_runtime.h>

namespace {

bool quiet() { const char* q = std::getenv("VGPU_QUIET"); return q && q[0] == '1'; }

struct SpMat {
  cusparseFormat_t format = CUSPARSE_FORMAT_CSR;
  int64_t rows = 0, cols = 0, nnz = 0;
  void* rows_ptr = nullptr;   // CSR row offsets, or COO row indices
  void* cols_ptr = nullptr;
  void* values = nullptr;
  cusparseIndexType_t row_type = CUSPARSE_INDEX_32I;
  cusparseIndexType_t col_type = CUSPARSE_INDEX_32I;
  cusparseIndexBase_t base = CUSPARSE_INDEX_BASE_ZERO;
  cudaDataType value_type = CUDA_R_32F;
  cusparseFillMode_t fill = CUSPARSE_FILL_MODE_LOWER;     // read by SpSV/SpSM only
  cusparseDiagType_t diag = CUSPARSE_DIAG_TYPE_NON_UNIT;
  // A strided batch: matrix b's offsets (CSR) start b * off_stride elements
  // in, its indices and values b * val_stride. COO uses val_stride for all three.
  int batch = 1;
  int64_t off_stride = 0, val_stride = 0;
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

std::mutex g_mu;
std::set<const void*> g_live;
template <class T> T* track(T* p) { std::lock_guard<std::mutex> l(g_mu); g_live.insert(p); return p; }
bool known(const void* p) { std::lock_guard<std::mutex> l(g_mu); return p && g_live.count(p); }
void untrack(const void* p) { std::lock_guard<std::mutex> l(g_mu); g_live.erase(p); }

size_t type_bytes(cudaDataType t) {
  switch (t) {
    case CUDA_R_16F: case CUDA_R_16BF: return 2;
    case CUDA_R_32F: return 4;
    case CUDA_R_64F: return 8;
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

bool fetch_values(const void* dev, size_t n, cudaDataType t, std::vector<double>* out) {
  out->assign(n, 0.0);
  if (!n) return true;
  if (t == CUDA_R_32F) {
    std::vector<float> tmp(n);
    if (cudaMemcpy(tmp.data(), dev, n * 4, cudaMemcpyDeviceToHost) != cudaSuccess) return false;
    for (size_t i = 0; i < n; ++i) (*out)[i] = tmp[i];
    return true;
  }
  if (t == CUDA_R_64F)
    return cudaMemcpy(out->data(), dev, n * 8, cudaMemcpyDeviceToHost) == cudaSuccess;
  if (t == CUDA_R_16F || t == CUDA_R_16BF) {
    std::vector<uint16_t> tmp(n);
    if (cudaMemcpy(tmp.data(), dev, n * 2, cudaMemcpyDeviceToHost) != cudaSuccess) return false;
    for (size_t i = 0; i < n; ++i) (*out)[i] = t == CUDA_R_16F ? from_half(tmp[i]) : from_bf16(tmp[i]);
    return true;
  }
  return false;
}

bool store_values(void* dev, const std::vector<double>& in, cudaDataType t) {
  if (in.empty()) return true;
  if (t == CUDA_R_32F) {
    std::vector<float> tmp(in.size());
    for (size_t i = 0; i < in.size(); ++i) tmp[i] = (float)in[i];
    return cudaMemcpy(dev, tmp.data(), tmp.size() * 4, cudaMemcpyHostToDevice) == cudaSuccess;
  }
  if (t == CUDA_R_64F)
    return cudaMemcpy(dev, in.data(), in.size() * 8, cudaMemcpyHostToDevice) == cudaSuccess;
  if (t == CUDA_R_16F || t == CUDA_R_16BF) {
    std::vector<uint16_t> tmp(in.size());
    for (size_t i = 0; i < in.size(); ++i) tmp[i] = t == CUDA_R_16F ? to_half(in[i]) : to_bf16(in[i]);
    return cudaMemcpy(dev, tmp.data(), tmp.size() * 2, cudaMemcpyHostToDevice) == cudaSuccess;
  }
  return false;
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
double scalar(cusparseHandle_t h, const void* p, cudaDataType t) {
  if (!p) return 1.0;
  double v = 0;
  if (reinterpret_cast<const Handle*>(h)->mode == CUSPARSE_POINTER_MODE_DEVICE) {
    std::vector<double> one;
    return fetch_values(p, 1, t, &one) ? one[0] : NAN;
  }
  if (t == CUDA_R_64F) v = *static_cast<const double*>(p);
  else if (t == CUDA_R_32F) v = *static_cast<const float*>(p);
  else if (t == CUDA_R_16F) v = from_half(*static_cast<const uint16_t*>(p));
  else if (t == CUDA_R_16BF) v = from_bf16(*static_cast<const uint16_t*>(p));
  return v;
}

// Every sparse format is expanded to a coordinate list once, and both SpMV and
// SpMM work from that. It costs an extra pass over nnz and removes a whole
// class of format-specific indexing bugs.
struct Triplets {
  std::vector<int64_t> row, col;
  std::vector<double> val;
};

bool expand(const SpMat& a, Triplets* t) {
  std::vector<double> vals;
  if (!fetch_values(a.values, (size_t)a.nnz, a.value_type, &vals)) return false;
  std::vector<int64_t> cols;
  if (!fetch_indices(a.cols_ptr, (size_t)a.nnz, a.col_type, &cols)) return false;
  const int64_t off = a.base == CUSPARSE_INDEX_BASE_ONE ? 1 : 0;

  t->val = std::move(vals);
  t->col.resize((size_t)a.nnz);
  t->row.resize((size_t)a.nnz);
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
    std::fprintf(stderr, "[vgpu] cusparse: only 32-bit float / 64-bit float values with "
                         "32- or 64-bit indices are implemented\n");
    return CUSPARSE_STATUS_NOT_SUPPORTED;
  }
  auto* m = new SpMat();
  *m = SpMat{fmt, rows, cols, nnz, offsets, indices, values, off_type, idx_type, base, vt};
  *out = reinterpret_cast<cusparseSpMatDescr_t>(track(m));
  return CUSPARSE_STATUS_SUCCESS;
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

VGPU_EXPORT cusparseStatus_t cusparseSpMV_bufferSize(cusparseHandle_t, cusparseOperation_t,
                                                     const void*, cusparseConstSpMatDescr_t,
                                                     cusparseConstDnVecDescr_t, const void*,
                                                     cusparseDnVecDescr_t, cudaDataType,
                                                     cusparseSpMVAlg_t, size_t* bytes) {
  if (bytes) *bytes = 0;   // host-computed: no device workspace to size
  return CUSPARSE_STATUS_SUCCESS;
}

VGPU_EXPORT cusparseStatus_t cusparseSpMV(cusparseHandle_t h, cusparseOperation_t op,
                                          const void* alpha, cusparseConstSpMatDescr_t matA,
                                          cusparseConstDnVecDescr_t vecX, const void* beta,
                                          cusparseDnVecDescr_t vecY, cudaDataType ct,
                                          cusparseSpMVAlg_t, void*) {
  if (!known(h) || !known(matA) || !known(vecX) || !known(vecY))
    return CUSPARSE_STATUS_NOT_INITIALIZED;
  const auto& A = *reinterpret_cast<const SpMat*>(matA);
  const auto& X = *reinterpret_cast<const DnVec*>(vecX);
  auto& Y = *reinterpret_cast<DnVec*>(vecY);
  if (!type_bytes(ct)) return CUSPARSE_STATUS_NOT_SUPPORTED;
  const int64_t m = transposed(op) ? A.cols : A.rows;
  const int64_t n = transposed(op) ? A.rows : A.cols;
  if (X.size != n || Y.size != m) return CUSPARSE_STATUS_INVALID_VALUE;

  Triplets t;
  if (!expand(A, &t)) return CUSPARSE_STATUS_INTERNAL_ERROR;
  std::vector<double> x, y;
  if (!fetch_values(X.values, (size_t)n, X.type, &x)) return CUSPARSE_STATUS_INTERNAL_ERROR;
  const double a = scalar(h, alpha, ct), b = scalar(h, beta, ct);
  if (b != 0.0) {
    if (!fetch_values(Y.values, (size_t)m, Y.type, &y)) return CUSPARSE_STATUS_INTERNAL_ERROR;
    for (auto& v : y) v *= b;
  } else {
    y.assign((size_t)m, 0.0);   // beta == 0 means write-only: never read Y
  }
  for (size_t k = 0; k < t.val.size(); ++k) {
    const int64_t r = transposed(op) ? t.col[k] : t.row[k];
    const int64_t c = transposed(op) ? t.row[k] : t.col[k];
    if (r < 0 || r >= m || c < 0 || c >= n) return CUSPARSE_STATUS_INVALID_VALUE;
    y[(size_t)r] += a * t.val[k] * x[(size_t)c];
  }
  return store_values(Y.values, y, Y.type) ? CUSPARSE_STATUS_SUCCESS
                                           : CUSPARSE_STATUS_INTERNAL_ERROR;
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
  const bool csr = a.format != CUSPARSE_FORMAT_COO;
  m.rows_ptr = step(a.rows_ptr, (int64_t)i * (csr ? a.off_stride : a.val_stride), index_bytes(a.row_type));
  m.cols_ptr = step(a.cols_ptr, (int64_t)i * a.val_stride, index_bytes(a.col_type));
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
cusparseStatus_t spmm_one(cusparseOperation_t opA, cusparseOperation_t opB, double al,
                          const SpMat& A, const DnMat& B, double be, const DnMat& C) {

  const int64_t m = transposed(opA) ? A.cols : A.rows;
  const int64_t k = transposed(opA) ? A.rows : A.cols;
  const int64_t bk = transposed(opB) ? B.cols : B.rows;
  const int64_t n = transposed(opB) ? B.rows : B.cols;
  if (bk != k || C.rows != m || C.cols != n) return CUSPARSE_STATUS_INVALID_VALUE;

  Triplets t;
  if (!expand(A, &t)) return CUSPARSE_STATUS_INTERNAL_ERROR;
  std::vector<double> b, c;
  if (!fetch_values(B.values, dn_elems(B), B.type, &b)) return CUSPARSE_STATUS_INTERNAL_ERROR;
  if (be != 0.0) {
    if (!fetch_values(C.values, dn_elems(C), C.type, &c)) return CUSPARSE_STATUS_INTERNAL_ERROR;
    for (auto& v : c) v *= be;
  } else {
    c.assign(dn_elems(C), 0.0);
  }
  for (size_t e = 0; e < t.val.size(); ++e) {
    const int64_t r = transposed(opA) ? t.col[e] : t.row[e];
    const int64_t cc = transposed(opA) ? t.row[e] : t.col[e];
    if (r < 0 || r >= m || cc < 0 || cc >= k) return CUSPARSE_STATUS_INVALID_VALUE;
    for (int64_t j = 0; j < n; ++j) {
      const size_t bi = transposed(opB) ? dn_index(B, j, cc) : dn_index(B, cc, j);
      c[dn_index(C, r, j)] += al * t.val[e] * b[bi];
    }
  }
  return store_values(C.values, c, C.type) ? CUSPARSE_STATUS_SUCCESS
                                           : CUSPARSE_STATUS_INTERNAL_ERROR;
}


}  // namespace


VGPU_EXPORT cusparseStatus_t cusparseSpMM_bufferSize(cusparseHandle_t, cusparseOperation_t,
                                                     cusparseOperation_t, const void*,
                                                     cusparseConstSpMatDescr_t,
                                                     cusparseConstDnMatDescr_t, const void*,
                                                     cusparseDnMatDescr_t, cudaDataType,
                                                     cusparseSpMMAlg_t, size_t* bytes) {
  if (bytes) *bytes = 0;
  return CUSPARSE_STATUS_SUCCESS;
}

VGPU_EXPORT cusparseStatus_t cusparseSpMM(cusparseHandle_t h, cusparseOperation_t opA,
                                          cusparseOperation_t opB, const void* alpha,
                                          cusparseConstSpMatDescr_t matA,
                                          cusparseConstDnMatDescr_t matB, const void* beta,
                                          cusparseDnMatDescr_t matC, cudaDataType ct,
                                          cusparseSpMMAlg_t, void*) {
  if (!known(h) || !known(matA) || !known(matB) || !known(matC))
    return CUSPARSE_STATUS_NOT_INITIALIZED;
  const auto& A = *reinterpret_cast<const SpMat*>(matA);
  const auto& B = *reinterpret_cast<const DnMat*>(matB);
  const auto& C = *reinterpret_cast<const DnMat*>(matC);
  if (!type_bytes(ct)) return CUSPARSE_STATUS_NOT_SUPPORTED;
  // A strided batch: C holds every product, and an operand with one matrix is
  // shared by all of them.
  if ((A.batch != 1 && A.batch != C.batch) || (B.batch != 1 && B.batch != C.batch))
    return CUSPARSE_STATUS_INVALID_VALUE;
  const double al = scalar(h, alpha, ct), be = scalar(h, beta, ct);
  for (int i = 0; i < C.batch; ++i) {
    const cusparseStatus_t st = spmm_one(opA, opB, al, member(A, i), member(B, i), be, member(C, i));
    if (st != CUSPARSE_STATUS_SUCCESS) return st;
  }
  return CUSPARSE_STATUS_SUCCESS;
}

VGPU_EXPORT cusparseStatus_t cusparseSpMM_preprocess(cusparseHandle_t, cusparseOperation_t,
                                                     cusparseOperation_t, const void*,
                                                     cusparseConstSpMatDescr_t,
                                                     cusparseConstDnMatDescr_t, const void*,
                                                     cusparseDnMatDescr_t, cudaDataType,
                                                     cusparseSpMMAlg_t, void*) {
  return CUSPARSE_STATUS_SUCCESS;
}

/* ---- conversion ---- */

VGPU_EXPORT cusparseStatus_t cusparseSparseToDense_bufferSize(cusparseHandle_t,
                                                              cusparseConstSpMatDescr_t,
                                                              cusparseDnMatDescr_t,
                                                              cusparseSparseToDenseAlg_t,
                                                              size_t* bytes) {
  if (bytes) *bytes = 0;
  return CUSPARSE_STATUS_SUCCESS;
}

VGPU_EXPORT cusparseStatus_t cusparseSparseToDense(cusparseHandle_t h,
                                                   cusparseConstSpMatDescr_t matA,
                                                   cusparseDnMatDescr_t matB,
                                                   cusparseSparseToDenseAlg_t, void*) {
  if (!known(h) || !known(matA) || !known(matB)) return CUSPARSE_STATUS_NOT_INITIALIZED;
  const auto& A = *reinterpret_cast<const SpMat*>(matA);
  auto& B = *reinterpret_cast<DnMat*>(matB);
  if (A.rows != B.rows || A.cols != B.cols) return CUSPARSE_STATUS_INVALID_VALUE;
  Triplets t;
  if (!expand(A, &t)) return CUSPARSE_STATUS_INTERNAL_ERROR;
  std::vector<double> d(dn_elems(B), 0.0);
  for (size_t e = 0; e < t.val.size(); ++e) {
    if (t.row[e] < 0 || t.row[e] >= A.rows || t.col[e] < 0 || t.col[e] >= A.cols)
      return CUSPARSE_STATUS_INVALID_VALUE;
    d[dn_index(B, t.row[e], t.col[e])] += t.val[e];  // duplicate entries accumulate
  }
  return store_values(B.values, d, B.type) ? CUSPARSE_STATUS_SUCCESS
                                           : CUSPARSE_STATUS_INTERNAL_ERROR;
}

VGPU_EXPORT cusparseStatus_t cusparseDenseToSparse_bufferSize(cusparseHandle_t,
                                                              cusparseConstDnMatDescr_t,
                                                              cusparseSpMatDescr_t,
                                                              cusparseDenseToSparseAlg_t,
                                                              size_t* bytes) {
  if (bytes) *bytes = 0;
  return CUSPARSE_STATUS_SUCCESS;
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
  std::vector<double> d;
  if (!fetch_values(A.values, dn_elems(A), A.type, &d)) return CUSPARSE_STATUS_INTERNAL_ERROR;
  int64_t nnz = 0;
  for (int64_t r = 0; r < A.rows; ++r)
    for (int64_t c = 0; c < A.cols; ++c)
      if (d[dn_index(A, r, c)] != 0.0) ++nnz;
  B.nnz = nnz;
  return CUSPARSE_STATUS_SUCCESS;
}

VGPU_EXPORT cusparseStatus_t cusparseDenseToSparse_convert(cusparseHandle_t h,
                                                           cusparseConstDnMatDescr_t matA,
                                                           cusparseSpMatDescr_t matB,
                                                           cusparseDenseToSparseAlg_t, void*) {
  if (!known(h) || !known(matA) || !known(matB)) return CUSPARSE_STATUS_NOT_INITIALIZED;
  const auto& A = *reinterpret_cast<const DnMat*>(matA);
  auto& B = *reinterpret_cast<SpMat*>(matB);
  if (A.rows != B.rows || A.cols != B.cols) return CUSPARSE_STATUS_INVALID_VALUE;
  if (B.format != CUSPARSE_FORMAT_CSR && B.format != CUSPARSE_FORMAT_COO)
    return CUSPARSE_STATUS_NOT_SUPPORTED;
  std::vector<double> d;
  if (!fetch_values(A.values, dn_elems(A), A.type, &d)) return CUSPARSE_STATUS_INTERNAL_ERROR;
  const int64_t off = B.base == CUSPARSE_INDEX_BASE_ONE ? 1 : 0;

  std::vector<int64_t> rows, cols;
  std::vector<double> vals;
  std::vector<int64_t> offsets{off};
  for (int64_t r = 0; r < A.rows; ++r) {
    for (int64_t c = 0; c < A.cols; ++c) {
      const double v = d[dn_index(A, r, c)];
      if (v == 0.0) continue;
      rows.push_back(r + off);
      cols.push_back(c + off);
      vals.push_back(v);
    }
    offsets.push_back((int64_t)vals.size() + off);
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
      put_indices(B.rows_ptr, B.format == CUSPARSE_FORMAT_CSR ? offsets : rows, B.row_type) &&
      put_indices(B.cols_ptr, cols, B.col_type) && store_values(B.values, vals, B.value_type);
  return ok ? CUSPARSE_STATUS_SUCCESS : CUSPARSE_STATUS_INTERNAL_ERROR;
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
  m->val_stride = val_stride;
  return CUSPARSE_STATUS_SUCCESS;
}
VGPU_EXPORT cusparseStatus_t cusparseCooSetStridedBatch(cusparseSpMatDescr_t d, int count, int64_t stride) {
  if (!known(d) || count < 1) return CUSPARSE_STATUS_INVALID_VALUE;
  auto* m = reinterpret_cast<SpMat*>(d);
  m->batch = count;
  m->val_stride = stride;
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
struct Csr {
  std::vector<int> off, col;
  std::vector<double> val;
};
// One matrix's structure (and values, when `vals` is given) from the device,
// rebased to zero.
bool read_csr(int m, int nnz, cusparseMatDescr_t d, const int* off, const int* col, const void* vals,
              cudaDataType t, Csr* out) {
  const int b = base_of(d);
  out->off = ints(off, (size_t)m + 1);
  out->col = ints(col, (size_t)nnz);
  for (int& v : out->off) v -= b;
  for (int& v : out->col) v -= b;
  if (out->off[0] != 0 || out->off[m] != nnz) return false;
  return !vals || fetch_values(vals, (size_t)nnz, t, &out->val);
}
// The union of the two sparsity patterns, row by row, each row's columns ascending.
std::vector<std::vector<int>> union_pattern(int m, const Csr& a, const Csr& b) {
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
double legacy_scalar(cusparseHandle_t h, const void* p, cudaDataType t) {
  return p ? scalar(h, p, t) : 0.0;
}

template <class T> constexpr cudaDataType value_type_of() {
  return sizeof(T) == 8 ? CUDA_R_64F : CUDA_R_32F;
}

template <class T>
cusparseStatus_t csrgeam2(cusparseHandle_t h, int m, int n, const T* alpha, cusparseMatDescr_t dA, int nnzA,
                          const T* valA, const int* offA, const int* colA, const T* beta, cusparseMatDescr_t dB,
                          int nnzB, const T* valB, const int* offB, const int* colB, cusparseMatDescr_t dC,
                          T* valC, int* offC, int* colC) {
  if (!known(h)) return CUSPARSE_STATUS_NOT_INITIALIZED;
  if (m < 0 || n < 0 || nnzA < 0 || nnzB < 0) return CUSPARSE_STATUS_INVALID_VALUE;
  constexpr cudaDataType t = value_type_of<T>();
  Csr a, b;
  if (!read_csr(m, nnzA, dA, offA, colA, valA, t, &a) || !read_csr(m, nnzB, dB, offB, colB, valB, t, &b))
    return CUSPARSE_STATUS_INVALID_VALUE;
  const double al = legacy_scalar(h, alpha, t), be = legacy_scalar(h, beta, t);
  const auto rows = union_pattern(m, a, b);
  const int base = base_of(dC);
  std::vector<int> off{base}, col;
  std::vector<double> val;
  for (int i = 0; i < m; ++i) {
    std::map<int, double> acc;
    for (int c : rows[i]) acc[c] = 0.0;
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
  Csr a, b;
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
#undef VGPU_GEAM

/* ---- SpGEMM: C = alpha * A * B (+ beta * C), sparse times sparse ---- */

namespace {
// The product is computed by _compute, which is where cuSPARSE learns C's nnz,
// and held here until _copy writes it into the arrays the caller then allocates.
struct SpGEMMDescr {
  std::vector<int64_t> off, col;
  std::vector<double> val;
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
  // cuSPARSE itself takes only non-transposed operands here.
  if (opA != CUSPARSE_OPERATION_NON_TRANSPOSE || opB != CUSPARSE_OPERATION_NON_TRANSPOSE)
    return CUSPARSE_STATUS_NOT_SUPPORTED;
  const auto& A = *reinterpret_cast<const SpMat*>(matA);
  const auto& B = *reinterpret_cast<const SpMat*>(matB);
  auto& C = *reinterpret_cast<SpMat*>(matC);
  if (A.cols != B.rows || C.rows != A.rows || C.cols != B.cols) return CUSPARSE_STATUS_INVALID_VALUE;
  if (C.format != CUSPARSE_FORMAT_CSR) return CUSPARSE_STATUS_NOT_SUPPORTED;
  Triplets ta, tb;
  if (!expand(A, &ta) || !expand(B, &tb)) return CUSPARSE_STATUS_INTERNAL_ERROR;
  const double al = scalar(h, alpha, ct), be = beta ? scalar(h, beta, ct) : 0.0;
  std::vector<std::vector<std::pair<int64_t, double>>> brows((size_t)B.rows);
  for (size_t k = 0; k < tb.val.size(); ++k) {
    if (tb.row[k] < 0 || tb.row[k] >= B.rows) return CUSPARSE_STATUS_INVALID_VALUE;
    brows[(size_t)tb.row[k]].push_back({tb.col[k], tb.val[k]});
  }
  std::vector<std::map<int64_t, double>> crows((size_t)A.rows);
  for (size_t k = 0; k < ta.val.size(); ++k) {
    if (ta.row[k] < 0 || ta.row[k] >= A.rows || ta.col[k] < 0 || ta.col[k] >= B.rows)
      return CUSPARSE_STATUS_INVALID_VALUE;
    for (const auto& [c, v] : brows[(size_t)ta.col[k]]) crows[(size_t)ta.row[k]][c] += al * ta.val[k] * v;
  }
  if (be != 0.0 && C.nnz > 0) {  // an existing C contributes its own pattern
    Triplets tc;
    if (!expand(C, &tc)) return CUSPARSE_STATUS_INTERNAL_ERROR;
    for (size_t k = 0; k < tc.val.size(); ++k) crows[(size_t)tc.row[k]][tc.col[k]] += be * tc.val[k];
  }
  auto& g = *reinterpret_cast<SpGEMMDescr*>(d);
  g.off.assign(1, 0);
  g.col.clear();
  g.val.clear();
  for (const auto& r : crows) {
    for (const auto& [c, v] : r) { g.col.push_back(c); g.val.push_back(v); }
    g.off.push_back((int64_t)g.col.size());
  }
  C.nnz = (int64_t)g.col.size();
  return CUSPARSE_STATUS_SUCCESS;
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

VGPU_EXPORT cusparseStatus_t cusparseSDDMM_bufferSize(cusparseHandle_t h, cusparseOperation_t, cusparseOperation_t,
                                                      const void*, cusparseConstDnMatDescr_t,
                                                      cusparseConstDnMatDescr_t, const void*, cusparseSpMatDescr_t,
                                                      cudaDataType, cusparseSDDMMAlg_t, size_t* bytes) {
  if (!known(h)) return CUSPARSE_STATUS_NOT_INITIALIZED;
  if (bytes) *bytes = 0;
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
                                           cusparseSpMatDescr_t matC, cudaDataType ct, cusparseSDDMMAlg_t, void*) {
  if (!known(h) || !known(matA) || !known(matB) || !known(matC)) return CUSPARSE_STATUS_NOT_INITIALIZED;
  const auto& A = *reinterpret_cast<const DnMat*>(matA);
  const auto& B = *reinterpret_cast<const DnMat*>(matB);
  const auto& C = *reinterpret_cast<const SpMat*>(matC);
  if (!type_bytes(ct)) return CUSPARSE_STATUS_NOT_SUPPORTED;
  const int64_t m = transposed(opA) ? A.cols : A.rows, k = transposed(opA) ? A.rows : A.cols;
  const int64_t bk = transposed(opB) ? B.cols : B.rows, n = transposed(opB) ? B.rows : B.cols;
  if (k != bk || C.rows != m || C.cols != n) return CUSPARSE_STATUS_INVALID_VALUE;
  if ((A.batch != 1 && A.batch != C.batch) || (B.batch != 1 && B.batch != C.batch))
    return CUSPARSE_STATUS_INVALID_VALUE;
  const double al = scalar(h, alpha, ct), be = scalar(h, beta, ct);
  for (int i = 0; i < C.batch; ++i) {
    const DnMat a = member(A, i), b = member(B, i);
    const SpMat c = member(C, i);
    std::vector<double> av, bv;
    Triplets t;
    if (!fetch_values(a.values, dn_elems(a), a.type, &av) || !fetch_values(b.values, dn_elems(b), b.type, &bv) ||
        !expand(c, &t))
      return CUSPARSE_STATUS_INTERNAL_ERROR;
    for (size_t e = 0; e < t.val.size(); ++e) {
      const int64_t r = t.row[e], col = t.col[e];
      if (r < 0 || r >= m || col < 0 || col >= n) return CUSPARSE_STATUS_INVALID_VALUE;
      double dot = 0;
      for (int64_t x = 0; x < k; ++x)
        dot += av[transposed(opA) ? dn_index(a, x, r) : dn_index(a, r, x)] *
               bv[transposed(opB) ? dn_index(b, col, x) : dn_index(b, x, col)];
      t.val[e] = al * dot + (be != 0.0 ? be * t.val[e] : 0.0);
    }
    if (!store_values(c.values, t.val, c.value_type)) return CUSPARSE_STATUS_INTERNAL_ERROR;
  }
  return CUSPARSE_STATUS_SUCCESS;
}

/* ---- triangular solves: op(A) y = alpha x (SpSV), op(A) C = alpha op(B) (SpSM) ---- */

namespace {
struct TriDescr {};

// op(A) restricted to the triangle A's fill mode names, as rows of (column,
// value), with its diagonal apart. Entries outside the triangle are ignored,
// as cuSPARSE ignores them.
struct Triangle {
  int64_t n = 0;
  bool lower = true;  // of op(A), which a transpose flips
  std::vector<std::vector<std::pair<int64_t, double>>> rows;
  std::vector<double> diag;
};
cusparseStatus_t triangle(const SpMat& A, cusparseOperation_t op, Triangle* t) {
  if (A.rows != A.cols) return CUSPARSE_STATUS_INVALID_VALUE;
  Triplets tr;
  if (!expand(A, &tr)) return CUSPARSE_STATUS_INTERNAL_ERROR;
  const bool lower = A.fill == CUSPARSE_FILL_MODE_LOWER, unit = A.diag == CUSPARSE_DIAG_TYPE_UNIT;
  t->n = A.rows;
  t->lower = transposed(op) ? !lower : lower;
  t->rows.assign((size_t)A.rows, {});
  t->diag.assign((size_t)A.rows, unit ? 1.0 : 0.0);
  for (size_t e = 0; e < tr.val.size(); ++e) {
    const int64_t r = tr.row[e], c = tr.col[e];
    if (r < 0 || r >= A.rows || c < 0 || c >= A.cols) return CUSPARSE_STATUS_INVALID_VALUE;
    if (lower ? c > r : c < r) continue;
    if (r == c) {
      if (!unit) t->diag[(size_t)r] += tr.val[e];
      continue;
    }
    if (transposed(op)) t->rows[(size_t)c].push_back({r, tr.val[e]});
    else t->rows[(size_t)r].push_back({c, tr.val[e]});
  }
  return CUSPARSE_STATUS_SUCCESS;
}
// Substitution in place: `x` holds the right-hand side and ends as the solution.
// A zero on the diagonal gives inf/nan where hardware reports a structural zero.
void substitute(const Triangle& t, double* x) {
  auto row = [&](int64_t i) {
    double s = x[i];
    for (const auto& [c, v] : t.rows[(size_t)i]) s -= v * x[c];
    x[i] = s / t.diag[(size_t)i];
  };
  if (t.lower) for (int64_t i = 0; i < t.n; ++i) row(i);
  else for (int64_t i = t.n; i-- > 0;) row(i);
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

VGPU_EXPORT cusparseStatus_t cusparseSpSV_bufferSize(cusparseHandle_t h, cusparseOperation_t, const void*,
                                                     cusparseConstSpMatDescr_t, cusparseConstDnVecDescr_t,
                                                     cusparseDnVecDescr_t, cudaDataType, cusparseSpSVAlg_t,
                                                     cusparseSpSVDescr_t, size_t* bytes) {
  if (!known(h)) return CUSPARSE_STATUS_NOT_INITIALIZED;
  if (bytes) *bytes = 16;
  return CUSPARSE_STATUS_SUCCESS;
}
VGPU_EXPORT cusparseStatus_t cusparseSpSV_analysis(cusparseHandle_t h, cusparseOperation_t, const void*,
                                                   cusparseConstSpMatDescr_t, cusparseConstDnVecDescr_t,
                                                   cusparseDnVecDescr_t, cudaDataType, cusparseSpSVAlg_t,
                                                   cusparseSpSVDescr_t d, void*) {
  return known(h) && known(d) ? CUSPARSE_STATUS_SUCCESS : CUSPARSE_STATUS_NOT_INITIALIZED;
}
VGPU_EXPORT cusparseStatus_t cusparseSpSV_solve(cusparseHandle_t h, cusparseOperation_t op, const void* alpha,
                                                cusparseConstSpMatDescr_t matA, cusparseConstDnVecDescr_t vecX,
                                                cusparseDnVecDescr_t vecY, cudaDataType ct, cusparseSpSVAlg_t,
                                                cusparseSpSVDescr_t d) {
  if (!known(h) || !known(d) || !known(matA) || !known(vecX) || !known(vecY))
    return CUSPARSE_STATUS_NOT_INITIALIZED;
  const auto& A = *reinterpret_cast<const SpMat*>(matA);
  const auto& X = *reinterpret_cast<const DnVec*>(vecX);
  const auto& Y = *reinterpret_cast<const DnVec*>(vecY);
  if (X.size != A.rows || Y.size != A.rows) return CUSPARSE_STATUS_INVALID_VALUE;
  Triangle t;
  if (const cusparseStatus_t st = triangle(A, op, &t); st != CUSPARSE_STATUS_SUCCESS) return st;
  std::vector<double> x;
  if (!fetch_values(X.values, (size_t)X.size, X.type, &x)) return CUSPARSE_STATUS_INTERNAL_ERROR;
  const double al = scalar(h, alpha, ct);
  for (auto& v : x) v *= al;
  substitute(t, x.data());
  return store_values(Y.values, x, Y.type) ? CUSPARSE_STATUS_SUCCESS : CUSPARSE_STATUS_INTERNAL_ERROR;
}

VGPU_EXPORT cusparseStatus_t cusparseSpSM_bufferSize(cusparseHandle_t h, cusparseOperation_t, cusparseOperation_t,
                                                     const void*, cusparseConstSpMatDescr_t,
                                                     cusparseConstDnMatDescr_t, cusparseDnMatDescr_t, cudaDataType,
                                                     cusparseSpSMAlg_t, cusparseSpSMDescr_t, size_t* bytes) {
  if (!known(h)) return CUSPARSE_STATUS_NOT_INITIALIZED;
  if (bytes) *bytes = 16;
  return CUSPARSE_STATUS_SUCCESS;
}
VGPU_EXPORT cusparseStatus_t cusparseSpSM_analysis(cusparseHandle_t h, cusparseOperation_t, cusparseOperation_t,
                                                   const void*, cusparseConstSpMatDescr_t, cusparseConstDnMatDescr_t,
                                                   cusparseDnMatDescr_t, cudaDataType, cusparseSpSMAlg_t,
                                                   cusparseSpSMDescr_t d, void*) {
  return known(h) && known(d) ? CUSPARSE_STATUS_SUCCESS : CUSPARSE_STATUS_NOT_INITIALIZED;
}
VGPU_EXPORT cusparseStatus_t cusparseSpSM_solve(cusparseHandle_t h, cusparseOperation_t opA, cusparseOperation_t opB,
                                                const void* alpha, cusparseConstSpMatDescr_t matA,
                                                cusparseConstDnMatDescr_t matB, cusparseDnMatDescr_t matC,
                                                cudaDataType ct, cusparseSpSMAlg_t, cusparseSpSMDescr_t d) {
  if (!known(h) || !known(d) || !known(matA) || !known(matB) || !known(matC))
    return CUSPARSE_STATUS_NOT_INITIALIZED;
  const auto& A = *reinterpret_cast<const SpMat*>(matA);
  const auto& B = *reinterpret_cast<const DnMat*>(matB);
  const auto& C = *reinterpret_cast<const DnMat*>(matC);
  const int64_t brows = transposed(opB) ? B.cols : B.rows, ncols = transposed(opB) ? B.rows : B.cols;
  if (brows != A.rows || C.rows != A.rows || C.cols != ncols) return CUSPARSE_STATUS_INVALID_VALUE;
  Triangle t;
  if (const cusparseStatus_t st = triangle(A, opA, &t); st != CUSPARSE_STATUS_SUCCESS) return st;
  std::vector<double> b, c;
  if (!fetch_values(B.values, dn_elems(B), B.type, &b) || !fetch_values(C.values, dn_elems(C), C.type, &c))
    return CUSPARSE_STATUS_INTERNAL_ERROR;
  const double al = scalar(h, alpha, ct);
  std::vector<double> x((size_t)A.rows);
  for (int64_t j = 0; j < ncols; ++j) {
    for (int64_t i = 0; i < A.rows; ++i)
      x[(size_t)i] = al * b[transposed(opB) ? dn_index(B, j, i) : dn_index(B, i, j)];
    substitute(t, x.data());
    for (int64_t i = 0; i < A.rows; ++i) c[dn_index(C, i, j)] = x[(size_t)i];
  }
  return store_values(C.values, c, C.type) ? CUSPARSE_STATUS_SUCCESS : CUSPARSE_STATUS_INTERNAL_ERROR;
}
