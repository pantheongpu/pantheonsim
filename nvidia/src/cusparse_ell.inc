// cuSPARSE's two ELL storage formats in the generic API, included by
// cusparse_api.cpp. expand_ell() there reads them; this creates and describes
// them.
//
//   Blocked-ELL  cusparseCreateBlockedEll: SpMM (opA = A) and DenseToSparse
//   Sliced ELL   cusparseCreateSlicedEll: SpMV (A, A^T, A^H) and SpSV
//
// Everything else NVIDIA's refuses them in is refused here with its status
// (SpMV with Blocked-ELL, SpMM, SDDMM and both conversions with sliced ELL,
// SparseToDense with either: NOT_SUPPORTED). What an RTX 3060's cuSPARSE
// 13.0 checks when one is created:
//
//   * Blocked-ELL: rows, cols and ellCols whole multiples of the block size,
//     ellCols <= cols, a block size above zero, a valid index base
//     (INVALID_VALUE otherwise). cusparseSpMatGetSize reports nnz as
//     rows * cols * blockSize, whatever is stored.
//   * Sliced ELL: sellValuesSize >= nnz (INVALID_VALUE), offsets and column
//     indices of one type, 32- or 64-bit (NOT_SUPPORTED otherwise). A slice
//     size of zero is taken; SpMV with it then faults on the card, and is
//     refused here.

namespace {
bool valid_base(cusparseIndexBase_t b) { return b == CUSPARSE_INDEX_BASE_ZERO || b == CUSPARSE_INDEX_BASE_ONE; }
}  // namespace

static cusparseStatus_t make_bell(cusparseSpMatDescr_t* out, int64_t rows, int64_t cols, int64_t bs, int64_t ell_cols,
                                  void* col_ind, void* values, cusparseIndexType_t it, cusparseIndexBase_t base,
                                  cudaDataType vt) {
  if (!out || rows < 0 || cols < 0 || ell_cols < 0 || bs < 0 || !valid_base(base)) return CUSPARSE_STATUS_INVALID_VALUE;
  if (bs == 0) {
    if (ell_cols > 0) return CUSPARSE_STATUS_INVALID_VALUE;
  } else if (rows % bs || cols % bs || ell_cols % bs) {
    return CUSPARSE_STATUS_INVALID_VALUE;
  }
  if (ell_cols > cols) return CUSPARSE_STATUS_INVALID_VALUE;
  if (it != CUSPARSE_INDEX_16U && it != CUSPARSE_INDEX_32I && it != CUSPARSE_INDEX_64I)
    return CUSPARSE_STATUS_INVALID_VALUE;
  if (!type_bytes(vt)) return CUSPARSE_STATUS_NOT_SUPPORTED;
  auto* m = new SpMat();
  m->format = kFormatBlockedEll;
  m->rows = rows;
  m->cols = cols;
  m->nnz = rows * cols * bs;  // as NVIDIA's reports it
  m->cols_ptr = col_ind;
  m->values = values;
  m->row_type = m->col_type = it;
  m->base = base;
  m->value_type = vt;
  m->bdim = bs;
  m->ell_cols = ell_cols;
  *out = reinterpret_cast<cusparseSpMatDescr_t>(track(m));
  return CUSPARSE_STATUS_SUCCESS;
}

VGPU_EXPORT cusparseStatus_t cusparseCreateBlockedEll(cusparseSpMatDescr_t* d, int64_t rows, int64_t cols,
                                                      int64_t bs, int64_t ell_cols, void* col_ind, void* values,
                                                      cusparseIndexType_t it, cusparseIndexBase_t base,
                                                      cudaDataType vt) {
  return make_bell(d, rows, cols, bs, ell_cols, col_ind, values, it, base, vt);
}
VGPU_EXPORT cusparseStatus_t cusparseCreateConstBlockedEll(cusparseConstSpMatDescr_t* d, int64_t rows, int64_t cols,
                                                           int64_t bs, int64_t ell_cols, const void* col_ind,
                                                           const void* values, cusparseIndexType_t it,
                                                           cusparseIndexBase_t base, cudaDataType vt) {
  cusparseSpMatDescr_t tmp = nullptr;
  const cusparseStatus_t rc = make_bell(d ? &tmp : nullptr, rows, cols, bs, ell_cols, const_cast<void*>(col_ind),
                                        const_cast<void*>(values), it, base, vt);
  if (rc == CUSPARSE_STATUS_SUCCESS) *d = tmp;
  return rc;
}
VGPU_EXPORT cusparseStatus_t cusparseBlockedEllGet(cusparseSpMatDescr_t d, int64_t* rows, int64_t* cols, int64_t* bs,
                                                   int64_t* ell_cols, void** col_ind, void** values,
                                                   cusparseIndexType_t* it, cusparseIndexBase_t* base,
                                                   cudaDataType* vt) {
  if (!known(d)) return CUSPARSE_STATUS_INVALID_VALUE;
  const auto* m = reinterpret_cast<const SpMat*>(d);
  if (m->format != kFormatBlockedEll) return CUSPARSE_STATUS_INVALID_VALUE;
  if (rows) *rows = m->rows;
  if (cols) *cols = m->cols;
  if (bs) *bs = m->bdim;
  if (ell_cols) *ell_cols = m->ell_cols;
  if (col_ind) *col_ind = m->cols_ptr;
  if (values) *values = m->values;
  if (it) *it = m->col_type;
  if (base) *base = m->base;
  if (vt) *vt = m->value_type;
  return CUSPARSE_STATUS_SUCCESS;
}
VGPU_EXPORT cusparseStatus_t cusparseConstBlockedEllGet(cusparseConstSpMatDescr_t d, int64_t* rows, int64_t* cols,
                                                        int64_t* bs, int64_t* ell_cols, const void** col_ind,
                                                        const void** values, cusparseIndexType_t* it,
                                                        cusparseIndexBase_t* base, cudaDataType* vt) {
  return cusparseBlockedEllGet(const_cast<cusparseSpMatDescr_t>(d), rows, cols, bs, ell_cols,
                               const_cast<void**>(col_ind), const_cast<void**>(values), it, base, vt);
}

static cusparseStatus_t make_sell(cusparseSpMatDescr_t* out, int64_t rows, int64_t cols, int64_t nnz,
                                  int64_t values_size, int64_t slice, void* offsets, void* col_ind, void* values,
                                  cusparseIndexType_t ot, cusparseIndexType_t ct, cusparseIndexBase_t base,
                                  cudaDataType vt) {
  if (!out || rows < 0 || cols < 0 || nnz < 0 || slice < 0 || !valid_base(base)) return CUSPARSE_STATUS_INVALID_VALUE;
  if (values_size < nnz) return CUSPARSE_STATUS_INVALID_VALUE;
  if (ot != ct || (ot != CUSPARSE_INDEX_32I && ot != CUSPARSE_INDEX_64I)) return CUSPARSE_STATUS_NOT_SUPPORTED;
  if (!type_bytes(vt)) return CUSPARSE_STATUS_NOT_SUPPORTED;
  auto* m = new SpMat();
  m->format = kFormatSlicedEll;
  m->rows = rows;
  m->cols = cols;
  m->nnz = nnz;
  m->rows_ptr = offsets;
  m->cols_ptr = col_ind;
  m->values = values;
  m->row_type = ot;
  m->col_type = ct;
  m->base = base;
  m->value_type = vt;
  m->slice = slice;
  m->sell_size = values_size;
  *out = reinterpret_cast<cusparseSpMatDescr_t>(track(m));
  return CUSPARSE_STATUS_SUCCESS;
}

// Declared here rather than taken from the header: CUDA 12.0's has neither.
VGPU_EXPORT cusparseStatus_t cusparseCreateSlicedEll(cusparseSpMatDescr_t* d, int64_t rows, int64_t cols, int64_t nnz,
                                                     int64_t values_size, int64_t slice, void* offsets,
                                                     void* col_ind, void* values, cusparseIndexType_t ot,
                                                     cusparseIndexType_t ct, cusparseIndexBase_t base,
                                                     cudaDataType vt) {
  return make_sell(d, rows, cols, nnz, values_size, slice, offsets, col_ind, values, ot, ct, base, vt);
}
VGPU_EXPORT cusparseStatus_t cusparseCreateConstSlicedEll(cusparseConstSpMatDescr_t* d, int64_t rows, int64_t cols,
                                                          int64_t nnz, int64_t values_size, int64_t slice,
                                                          const void* offsets, const void* col_ind,
                                                          const void* values, cusparseIndexType_t ot,
                                                          cusparseIndexType_t ct, cusparseIndexBase_t base,
                                                          cudaDataType vt) {
  cusparseSpMatDescr_t tmp = nullptr;
  const cusparseStatus_t rc =
      make_sell(d ? &tmp : nullptr, rows, cols, nnz, values_size, slice, const_cast<void*>(offsets),
                const_cast<void*>(col_ind), const_cast<void*>(values), ot, ct, base, vt);
  if (rc == CUSPARSE_STATUS_SUCCESS) *d = tmp;
  return rc;
}
