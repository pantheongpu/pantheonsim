// cuSPARSE's sparse vectors (cusparseSpVecDescr_t) and what takes them: SpVV,
// Axpby, Gather, Scatter and Rot, and the legacy gemvi (a dense matrix times a
// sparse vector). Included by cusparse_api.cpp.
//
// cusparseSpMV takes a dense vector only: the API has no sparse-matrix times
// sparse-vector product, so there is nothing for an SpVec to be given to
// beyond these.
//
// What an RTX 3060 showed with cuSPARSE 13.0 (CUDA 13.0), which this
// reproduces -- nvidia/tests/e2e/sparse_vector_paths.cu checks it:
//
//   * cusparseCreateSpVec refuses size < 0, nnz < 0, nnz > size, an index
//     base other than zero or one and a NULL descriptor (INVALID_VALUE); NULL
//     arrays are taken. The getters refuse a NULL output (INVALID_VALUE);
//     SetValues takes a NULL.
//   * SpVV computes sum op(x_i) y[idx_i], op = A or A^H (conjugating x); A^T,
//     and A^H of real vectors, are INVALID_VALUE. x and y must be the same
//     type (NOT_SUPPORTED otherwise) and the same size (INVALID_VALUE); half
//     precision computes in single, int8 in int32 or single; any other
//     compute type is NOT_SUPPORTED. The result goes where the pointer mode
//     says.
//   * Axpby, Gather, Scatter and Rot refuse vectors of different types
//     (NOT_SUPPORTED) or sizes (INVALID_VALUE). Their scalars are of the
//     vectors' compute type (single for half precision); Rot's c and s are
//     complex for complex vectors and multiply as given: x_i = c x_i + s y_i,
//     y_i = c y_i - s x_i.
//   * Where an index repeats, the first entry with it is the one that lands
//     (Axpby, Scatter): nothing accumulates.
//   * gemvi takes A and A^T of a real matrix (A^H is INVALID_VALUE) and does
//     not check lda; NVIDIA's dies of SIGFPE on m < 0, which this refuses.

namespace {

struct SpVec {
  int64_t size = 0, nnz = 0;
  void* indices = nullptr;
  void* values = nullptr;
  cusparseIndexType_t idx_type = CUSPARSE_INDEX_32I;
  cusparseIndexBase_t base = CUSPARSE_INDEX_BASE_ZERO;
  cudaDataType type = CUDA_R_32F;
};

// The scalars' type for a vector of type t where the call names none: half
// precision computes in single.
cudaDataType scalar_type_of(cudaDataType t) {
  switch (t) {
    case CUDA_R_16F: case CUDA_R_16BF: return CUDA_R_32F;
    case CUDA_C_16F: case CUDA_C_16BF: return CUDA_C_32F;
    default: return t;
  }
}

// x's indices, rebased to zero.
bool spvec_indices(const SpVec& x, std::vector<int64_t>* out) {
  if (!fetch_indices(x.indices, (size_t)x.nnz, x.idx_type, out)) return false;
  const int64_t off = x.base == CUSPARSE_INDEX_BASE_ONE ? 1 : 0;
  for (auto& v : *out) v -= off;
  return true;
}

// Writes `v` (in compute type ct) where `result` points, host or device as the
// handle says.
template <class V> bool put_scalar(cusparseHandle_t h, void* result, V v, cudaDataType ct) {
  std::vector<V> one{v};
  if (reinterpret_cast<const Handle*>(h)->mode == CUSPARSE_POINTER_MODE_DEVICE) return store_values(result, one, ct);
  encode(one, ct, static_cast<uint8_t*>(result));
  return true;
}

const SpVec* spvec(const void* p) { return known(p) ? static_cast<const SpVec*>(p) : nullptr; }

}  // namespace

/* ---- descriptors ---- */

VGPU_EXPORT cusparseStatus_t cusparseCreateSpVec(cusparseSpVecDescr_t* d, int64_t size, int64_t nnz, void* indices,
                                                 void* values, cusparseIndexType_t it, cusparseIndexBase_t base,
                                                 cudaDataType vt) {
  if (!d || size < 0 || nnz < 0 || nnz > size) return CUSPARSE_STATUS_INVALID_VALUE;
  if (base != CUSPARSE_INDEX_BASE_ZERO && base != CUSPARSE_INDEX_BASE_ONE) return CUSPARSE_STATUS_INVALID_VALUE;
  if (it != CUSPARSE_INDEX_16U && it != CUSPARSE_INDEX_32I && it != CUSPARSE_INDEX_64I) return CUSPARSE_STATUS_INVALID_VALUE;
  if (!element_bytes(vt)) return CUSPARSE_STATUS_NOT_SUPPORTED;
  *d = reinterpret_cast<cusparseSpVecDescr_t>(track(new SpVec{size, nnz, indices, values, it, base, vt}));
  return CUSPARSE_STATUS_SUCCESS;
}
VGPU_EXPORT cusparseStatus_t cusparseCreateConstSpVec(cusparseConstSpVecDescr_t* d, int64_t size, int64_t nnz,
                                                      const void* indices, const void* values,
                                                      cusparseIndexType_t it, cusparseIndexBase_t base,
                                                      cudaDataType vt) {
  cusparseSpVecDescr_t tmp = nullptr;
  const cusparseStatus_t rc =
      cusparseCreateSpVec(d ? &tmp : nullptr, size, nnz, const_cast<void*>(indices), const_cast<void*>(values), it, base, vt);
  if (rc == CUSPARSE_STATUS_SUCCESS) *d = tmp;
  return rc;
}
VGPU_EXPORT cusparseStatus_t cusparseDestroySpVec(cusparseConstSpVecDescr_t d) {
  if (!spvec(d)) return CUSPARSE_STATUS_INVALID_VALUE;
  untrack(d);
  delete static_cast<const SpVec*>(static_cast<const void*>(d));
  return CUSPARSE_STATUS_SUCCESS;
}
VGPU_EXPORT cusparseStatus_t cusparseSpVecGet(cusparseSpVecDescr_t d, int64_t* size, int64_t* nnz, void** indices,
                                              void** values, cusparseIndexType_t* it, cusparseIndexBase_t* base,
                                              cudaDataType* vt) {
  const SpVec* x = spvec(d);
  if (!x) return CUSPARSE_STATUS_INVALID_VALUE;
  if (!size || !nnz || !indices || !values || !it || !base || !vt) return CUSPARSE_STATUS_INVALID_VALUE;
  if (size) *size = x->size;
  if (nnz) *nnz = x->nnz;
  if (indices) *indices = x->indices;
  if (values) *values = x->values;
  if (it) *it = x->idx_type;
  if (base) *base = x->base;
  if (vt) *vt = x->type;
  return CUSPARSE_STATUS_SUCCESS;
}
VGPU_EXPORT cusparseStatus_t cusparseConstSpVecGet(cusparseConstSpVecDescr_t d, int64_t* size, int64_t* nnz,
                                                   const void** indices, const void** values,
                                                   cusparseIndexType_t* it, cusparseIndexBase_t* base,
                                                   cudaDataType* vt) {
  return cusparseSpVecGet(const_cast<cusparseSpVecDescr_t>(d), size, nnz, const_cast<void**>(indices),
                          const_cast<void**>(values), it, base, vt);
}
VGPU_EXPORT cusparseStatus_t cusparseSpVecGetIndexBase(cusparseConstSpVecDescr_t d, cusparseIndexBase_t* base) {
  const SpVec* x = spvec(d);
  if (!x || !base) return CUSPARSE_STATUS_INVALID_VALUE;
  *base = x->base;
  return CUSPARSE_STATUS_SUCCESS;
}
VGPU_EXPORT cusparseStatus_t cusparseSpVecGetValues(cusparseSpVecDescr_t d, void** values) {
  const SpVec* x = spvec(d);
  if (!x || !values) return CUSPARSE_STATUS_INVALID_VALUE;
  *values = x->values;
  return CUSPARSE_STATUS_SUCCESS;
}
VGPU_EXPORT cusparseStatus_t cusparseConstSpVecGetValues(cusparseConstSpVecDescr_t d, const void** values) {
  return cusparseSpVecGetValues(const_cast<cusparseSpVecDescr_t>(d), const_cast<void**>(values));
}
VGPU_EXPORT cusparseStatus_t cusparseSpVecSetValues(cusparseSpVecDescr_t d, void* values) {
  SpVec* x = const_cast<SpVec*>(spvec(d));
  if (!x || !values) return CUSPARSE_STATUS_INVALID_VALUE;
  x->values = values;
  return CUSPARSE_STATUS_SUCCESS;
}

/* ---- SpVV: result = op(x) . y ---- */

namespace {
cusparseStatus_t spvv_check(cusparseOperation_t op, const SpVec& x, const DnVec& y, cudaDataType ct) {
  if (op == CUSPARSE_OPERATION_TRANSPOSE || bad_conj(op, x.type)) return CUSPARSE_STATUS_INVALID_VALUE;
  if (x.type != y.type) return CUSPARSE_STATUS_NOT_SUPPORTED;
  if (x.size != y.size) return CUSPARSE_STATUS_INVALID_VALUE;
  const bool ok = x.type == CUDA_R_8I ? (ct == CUDA_R_32I || ct == CUDA_R_32F) : ct == scalar_type_of(x.type);
  if (!ok || !type_bytes(ct)) return CUSPARSE_STATUS_NOT_SUPPORTED;
  return CUSPARSE_STATUS_SUCCESS;
}
template <class V>
cusparseStatus_t spvv(cusparseHandle_t h, cusparseOperation_t op, const SpVec& x, const DnVec& y, void* result,
                      cudaDataType ct) {
  std::vector<int64_t> idx;
  std::vector<V> xv, yv;
  if (!spvec_indices(x, &idx) || !fetch_values(x.values, (size_t)x.nnz, x.type, &xv) ||
      !fetch_values(y.values, (size_t)y.size, y.type, &yv))
    return CUSPARSE_STATUS_EXECUTION_FAILED;
  V sum(0.0);
  for (size_t i = 0; i < idx.size(); ++i) {
    if (idx[i] < 0 || idx[i] >= y.size) return CUSPARSE_STATUS_INVALID_VALUE;
    sum += conj_op(xv[i], op) * yv[(size_t)idx[i]];
  }
  if (ct == CUDA_R_32I) sum = V(std::nearbyint(re(sum)));
  return put_scalar(h, result, sum, ct) ? CUSPARSE_STATUS_SUCCESS : CUSPARSE_STATUS_EXECUTION_FAILED;
}
}  // namespace

VGPU_EXPORT cusparseStatus_t cusparseSpVV_bufferSize(cusparseHandle_t h, cusparseOperation_t op,
                                                     cusparseConstSpVecDescr_t vecX, cusparseConstDnVecDescr_t vecY,
                                                     const void*, cudaDataType ct, size_t* bytes) {
  if (!known(h)) return CUSPARSE_STATUS_NOT_INITIALIZED;
  if (!spvec(vecX) || !known(vecY) || !bytes) return CUSPARSE_STATUS_INVALID_VALUE;
  *bytes = 0;
  return spvv_check(op, *spvec(vecX), *reinterpret_cast<const DnVec*>(vecY), ct);
}
VGPU_EXPORT cusparseStatus_t cusparseSpVV(cusparseHandle_t h, cusparseOperation_t op, cusparseConstSpVecDescr_t vecX,
                                          cusparseConstDnVecDescr_t vecY, void* result, cudaDataType ct, void* unused6) {
  VGPU_DEFER_CALL(g_reg, stream_of(h), cusparseSpVV, h, op, vecX, vecY, result, ct, unused6);
  if (!known(h)) return CUSPARSE_STATUS_NOT_INITIALIZED;
  if (!spvec(vecX) || !known(vecY) || !result) return CUSPARSE_STATUS_INVALID_VALUE;
  const SpVec& x = *spvec(vecX);
  const DnVec& y = *reinterpret_cast<const DnVec*>(vecY);
  if (const cusparseStatus_t st = spvv_check(op, x, y, ct); st != CUSPARSE_STATUS_SUCCESS) return st;
  return any_complex({x.type, y.type, ct}) ? spvv<cd>(h, op, x, y, result, ct) : spvv<double>(h, op, x, y, result, ct);
}

/* ---- Axpby, Gather, Scatter, Rot ---- */

namespace {
cusparseStatus_t pair_check(const SpVec& x, const DnVec& y) {
  if (x.type != y.type) return CUSPARSE_STATUS_NOT_SUPPORTED;
  if (x.size != y.size) return CUSPARSE_STATUS_INVALID_VALUE;
  return CUSPARSE_STATUS_SUCCESS;
}

// Runs `body` over x's values (as V), its indices and y's values, then writes
// back whichever of the two `body` says it changed.
template <class V, class F>
cusparseStatus_t with_pair(const SpVec& x, const DnVec& y, bool write_x, bool write_y, F body) {
  std::vector<int64_t> idx;
  std::vector<V> xv, yv;
  if (!spvec_indices(x, &idx) || !fetch_values(x.values, (size_t)x.nnz, x.type, &xv) ||
      !fetch_values(y.values, (size_t)y.size, y.type, &yv))
    return CUSPARSE_STATUS_EXECUTION_FAILED;
  for (int64_t i : idx)
    if (i < 0 || i >= y.size) return CUSPARSE_STATUS_INVALID_VALUE;
  body(idx, xv, yv);
  const bool ok = (!write_x || store_values(x.values, xv, x.type)) && (!write_y || store_values(y.values, yv, y.type));
  return ok ? CUSPARSE_STATUS_SUCCESS : CUSPARSE_STATUS_EXECUTION_FAILED;
}

template <class V>
cusparseStatus_t axpby(cusparseHandle_t h, const void* alpha, const SpVec& x, const void* beta, const DnVec& y) {
  const cudaDataType st = scalar_type_of(y.type);
  const V a = scalar<V>(h, alpha, st), b = scalar<V>(h, beta, st);
  return with_pair<V>(x, y, false, true, [&](const std::vector<int64_t>& idx, std::vector<V>& xv, std::vector<V>& yv) {
    const std::vector<V> y0 = yv;
    for (auto& v : yv) v *= b;
    // Backwards, so the first entry with a repeated index is the one that lands.
    for (size_t i = idx.size(); i-- > 0;) yv[(size_t)idx[i]] = b * y0[(size_t)idx[i]] + a * xv[i];
  });
}

template <class V>
cusparseStatus_t rot(cusparseHandle_t h, const void* c_p, const void* s_p, const SpVec& x, const DnVec& y) {
  const cudaDataType st = scalar_type_of(y.type);
  const V c = scalar<V>(h, c_p, st), s = scalar<V>(h, s_p, st);
  return with_pair<V>(x, y, true, true, [&](const std::vector<int64_t>& idx, std::vector<V>& xv, std::vector<V>& yv) {
    for (size_t i = 0; i < idx.size(); ++i) {
      const V xi = xv[i], yi = yv[(size_t)idx[i]];
      xv[i] = c * xi + s * yi;
      yv[(size_t)idx[i]] = c * yi - s * xi;
    }
  });
}
}  // namespace

VGPU_EXPORT cusparseStatus_t cusparseAxpby(cusparseHandle_t h, const void* alpha, cusparseConstSpVecDescr_t vecX,
                                           const void* beta, cusparseDnVecDescr_t vecY) {
  VGPU_DEFER_CALL(g_reg, stream_of(h), cusparseAxpby, h, alpha, vecX, beta, vecY);
  if (!known(h)) return CUSPARSE_STATUS_NOT_INITIALIZED;
  if (!spvec(vecX) || !known(vecY) || !alpha || !beta) return CUSPARSE_STATUS_INVALID_VALUE;
  const SpVec& x = *spvec(vecX);
  const DnVec& y = *reinterpret_cast<const DnVec*>(vecY);
  if (const cusparseStatus_t st = pair_check(x, y); st != CUSPARSE_STATUS_SUCCESS) return st;
  return is_complex(y.type) ? axpby<cd>(h, alpha, x, beta, y) : axpby<double>(h, alpha, x, beta, y);
}

// Gather and Scatter move values without arithmetic, so they move bits: any
// element type of the right size works and comes through unchanged.
VGPU_EXPORT cusparseStatus_t cusparseGather(cusparseHandle_t h, cusparseConstDnVecDescr_t vecY,
                                            cusparseSpVecDescr_t vecX) {
  VGPU_DEFER_CALL(g_reg, stream_of(h), cusparseGather, h, vecY, vecX);
  if (!known(h)) return CUSPARSE_STATUS_NOT_INITIALIZED;
  if (!spvec(vecX) || !known(vecY)) return CUSPARSE_STATUS_INVALID_VALUE;
  const SpVec& x = *spvec(vecX);
  const DnVec& y = *reinterpret_cast<const DnVec*>(vecY);
  if (const cusparseStatus_t st = pair_check(x, y); st != CUSPARSE_STATUS_SUCCESS) return st;
  const size_t eb = element_bytes(x.type);
  std::vector<int64_t> idx;
  std::vector<uint8_t> yv((size_t)y.size * eb), xv((size_t)x.nnz * eb);
  if (!spvec_indices(x, &idx) ||
      (!yv.empty() && cudaMemcpy(yv.data(), y.values, yv.size(), cudaMemcpyDeviceToHost) != cudaSuccess))
    return CUSPARSE_STATUS_EXECUTION_FAILED;
  for (size_t i = 0; i < idx.size(); ++i) {
    if (idx[i] < 0 || idx[i] >= y.size) return CUSPARSE_STATUS_INVALID_VALUE;
    std::memcpy(&xv[i * eb], &yv[(size_t)idx[i] * eb], eb);
  }
  return xv.empty() || cudaMemcpy(x.values, xv.data(), xv.size(), cudaMemcpyHostToDevice) == cudaSuccess
             ? CUSPARSE_STATUS_SUCCESS
             : CUSPARSE_STATUS_EXECUTION_FAILED;
}
VGPU_EXPORT cusparseStatus_t cusparseScatter(cusparseHandle_t h, cusparseConstSpVecDescr_t vecX,
                                             cusparseDnVecDescr_t vecY) {
  VGPU_DEFER_CALL(g_reg, stream_of(h), cusparseScatter, h, vecX, vecY);
  if (!known(h)) return CUSPARSE_STATUS_NOT_INITIALIZED;
  if (!spvec(vecX) || !known(vecY)) return CUSPARSE_STATUS_INVALID_VALUE;
  const SpVec& x = *spvec(vecX);
  const DnVec& y = *reinterpret_cast<const DnVec*>(vecY);
  if (const cusparseStatus_t st = pair_check(x, y); st != CUSPARSE_STATUS_SUCCESS) return st;
  const size_t eb = element_bytes(x.type);
  std::vector<int64_t> idx;
  std::vector<uint8_t> yv((size_t)y.size * eb), xv((size_t)x.nnz * eb);
  if (!spvec_indices(x, &idx) ||
      (!yv.empty() && cudaMemcpy(yv.data(), y.values, yv.size(), cudaMemcpyDeviceToHost) != cudaSuccess) ||
      (!xv.empty() && cudaMemcpy(xv.data(), x.values, xv.size(), cudaMemcpyDeviceToHost) != cudaSuccess))
    return CUSPARSE_STATUS_EXECUTION_FAILED;
  for (int64_t i : idx)
    if (i < 0 || i >= y.size) return CUSPARSE_STATUS_INVALID_VALUE;
  for (size_t i = idx.size(); i-- > 0;) std::memcpy(&yv[(size_t)idx[i] * eb], &xv[i * eb], eb);
  return yv.empty() || cudaMemcpy(y.values, yv.data(), yv.size(), cudaMemcpyHostToDevice) == cudaSuccess
             ? CUSPARSE_STATUS_SUCCESS
             : CUSPARSE_STATUS_EXECUTION_FAILED;
}
VGPU_EXPORT cusparseStatus_t cusparseRot(cusparseHandle_t h, const void* c, const void* s, cusparseSpVecDescr_t vecX,
                                         cusparseDnVecDescr_t vecY) {
  VGPU_DEFER_CALL(g_reg, stream_of(h), cusparseRot, h, c, s, vecX, vecY);
  if (!known(h)) return CUSPARSE_STATUS_NOT_INITIALIZED;
  if (!spvec(vecX) || !known(vecY) || !c || !s) return CUSPARSE_STATUS_INVALID_VALUE;
  const SpVec& x = *spvec(vecX);
  const DnVec& y = *reinterpret_cast<const DnVec*>(vecY);
  if (const cusparseStatus_t st = pair_check(x, y); st != CUSPARSE_STATUS_SUCCESS) return st;
  return is_complex(y.type) ? rot<cd>(h, c, s, x, y) : rot<double>(h, c, s, x, y);
}

/* ---- gemvi: y = alpha op(A) x + beta y, A dense, x sparse ---- */

namespace {
template <class T>
cusparseStatus_t gemvi(cusparseHandle_t h, cusparseOperation_t op, int m, int n, const T* alpha, const T* A, int lda,
                       int nnz, const T* xv, const int* xi, const T* beta, T* y, cusparseIndexBase_t base) {
  using V = typename Legacy<T>::V;
  constexpr cudaDataType t = Legacy<T>::type;
  if (!known(h)) return CUSPARSE_STATUS_NOT_INITIALIZED;
  if (op == CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE && !is_complex(t)) return CUSPARSE_STATUS_INVALID_VALUE;
  if (op != CUSPARSE_OPERATION_NON_TRANSPOSE && op != CUSPARSE_OPERATION_TRANSPOSE &&
      op != CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE)
    return CUSPARSE_STATUS_INVALID_VALUE;
  if (m < 0 || n < 0 || nnz < 0 || (base != CUSPARSE_INDEX_BASE_ZERO && base != CUSPARSE_INDEX_BASE_ONE))
    return CUSPARSE_STATUS_INVALID_VALUE;
  if (m == 0 || n == 0) return CUSPARSE_STATUS_SUCCESS;
  const bool tr = op != CUSPARSE_OPERATION_NON_TRANSPOSE;
  const int ylen = tr ? n : m, xlen = tr ? m : n;
  if (ylen == 0) return CUSPARSE_STATUS_SUCCESS;
  std::vector<V> av, x, yv;
  std::vector<int> idx = ints(xi, (size_t)nnz);
  if (!fetch_values(A, (size_t)lda * (n - 1) + m, t, &av) || !fetch_values(xv, (size_t)nnz, t, &x) ||
      !fetch_values(y, (size_t)ylen, t, &yv))
    return CUSPARSE_STATUS_EXECUTION_FAILED;
  const V al = legacy_scalar<V>(h, alpha, t), be = legacy_scalar<V>(h, beta, t);
  for (auto& v : yv) v = be != V(0.0) ? be * v : V(0.0);
  const int b = base == CUSPARSE_INDEX_BASE_ONE ? 1 : 0;
  for (int k = 0; k < nnz; ++k) {
    const int j = idx[(size_t)k] - b;
    if (j < 0 || j >= xlen) return CUSPARSE_STATUS_INVALID_VALUE;
    const V s = al * x[(size_t)k];
    if (!tr) {
      for (int i = 0; i < m; ++i) yv[(size_t)i] += av[(size_t)j * lda + i] * s;
    } else {
      for (int i = 0; i < n; ++i) yv[(size_t)i] += conj_op(av[(size_t)i * lda + j], op) * s;
    }
  }
  return store_values(y, yv, t) ? CUSPARSE_STATUS_SUCCESS : CUSPARSE_STATUS_EXECUTION_FAILED;
}
}  // namespace

#define VGPU_GEMVI(P, T)                                                                                            \
  VGPU_EXPORT cusparseStatus_t cusparse##P##gemvi_bufferSize(cusparseHandle_t h, cusparseOperation_t op, int m,    \
                                                             int n, int nnz, int* bytes) {                          \
    if (!known(h)) return CUSPARSE_STATUS_NOT_INITIALIZED;                                                          \
    if (m < 0 || n < 0 || nnz < 0 || !bytes) return CUSPARSE_STATUS_INVALID_VALUE;                                 \
    (void)op;                                                                                                       \
    *bytes = 0; /* measured: NVIDIA's asks for none */                                                              \
    return CUSPARSE_STATUS_SUCCESS;                                                                                 \
  }                                                                                                                 \
  VGPU_EXPORT cusparseStatus_t cusparse##P##gemvi(cusparseHandle_t h, cusparseOperation_t op, int m, int n,        \
                                                  const T* alpha, const T* A, int lda, int nnz, const T* xv,       \
                                                  const int* xi, const T* beta, T* y, cusparseIndexBase_t base,    \
                                                  void* unused13) { \
  VGPU_DEFER_CALL(g_reg, stream_of(h), cusparse##P##gemvi, h, op, m, n, alpha, A, lda, nnz, xv, xi, beta, y, base, unused13); \
    return gemvi<T>(h, op, m, n, alpha, A, lda, nnz, xv, xi, beta, y, base);                                        \
  }
VGPU_GEMVI(S, float)
VGPU_GEMVI(D, double)
VGPU_GEMVI(C, cuComplex)
VGPU_GEMVI(Z, cuDoubleComplex)
#undef VGPU_GEMVI
