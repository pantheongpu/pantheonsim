// libvgpucusparse -- VirtualGPU's cuSPARSE, presented as libcusparse.so.12.
//
// The generic (descriptor) API only: CSR and COO sparse matrices, dense vectors
// and matrices, SpMV, SpMM, and conversion in both directions. The legacy
// cusparse<t>csrmv-style entry points were removed by NVIDIA in CUDA 12 and are
// not resurrected here.
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
#include <cstring>
#include <mutex>
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
};

struct Handle { cudaStream_t stream = nullptr; };

std::mutex g_mu;
std::set<const void*> g_live;
template <class T> T* track(T* p) { std::lock_guard<std::mutex> l(g_mu); g_live.insert(p); return p; }
bool known(const void* p) { std::lock_guard<std::mutex> l(g_mu); return p && g_live.count(p); }
void untrack(const void* p) { std::lock_guard<std::mutex> l(g_mu); g_live.erase(p); }

size_t type_bytes(cudaDataType t) {
  switch (t) {
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

double scalar(const void* p, cudaDataType t) {
  if (!p) return 1.0;
  return t == CUDA_R_64F ? *static_cast<const double*>(p) : *static_cast<const float*>(p);
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
    std::fprintf(stderr, "[vgpu] cuSPARSE handle created (host-computed; generic API only)\n");
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
  const double a = scalar(alpha, ct), b = scalar(beta, ct);
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
  auto& C = *reinterpret_cast<DnMat*>(matC);
  if (!type_bytes(ct)) return CUSPARSE_STATUS_NOT_SUPPORTED;

  const int64_t m = transposed(opA) ? A.cols : A.rows;
  const int64_t k = transposed(opA) ? A.rows : A.cols;
  const int64_t bk = transposed(opB) ? B.cols : B.rows;
  const int64_t n = transposed(opB) ? B.rows : B.cols;
  if (bk != k || C.rows != m || C.cols != n) return CUSPARSE_STATUS_INVALID_VALUE;

  Triplets t;
  if (!expand(A, &t)) return CUSPARSE_STATUS_INTERNAL_ERROR;
  std::vector<double> b, c;
  if (!fetch_values(B.values, dn_elems(B), B.type, &b)) return CUSPARSE_STATUS_INTERNAL_ERROR;
  const double al = scalar(alpha, ct), be = scalar(beta, ct);
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
