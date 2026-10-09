// The rest of cuSPARSE's generic API, included by cusparse_api.cpp: the
// Const* getters, the logger, SpGEMM's product count and memory estimate,
// SpGEMMreuse, SpMMOp, SpSV/SpSM's updateMatrix and the CSC sort. What an
// RTX 3060 showed with cuSPARSE 13.0 is noted at each.

/* ---- Const getters: the same descriptors, seen through const pointers ---- */

VGPU_EXPORT cusparseStatus_t cusparseConstSpMatGetValues(cusparseConstSpMatDescr_t d, const void** v) {
  return cusparseSpMatGetValues(const_cast<cusparseSpMatDescr_t>(d), const_cast<void**>(v));
}
VGPU_EXPORT cusparseStatus_t cusparseConstCsrGet(cusparseConstSpMatDescr_t d, int64_t* rows, int64_t* cols,
                                                 int64_t* nnz, const void** off, const void** col, const void** val,
                                                 cusparseIndexType_t* ot, cusparseIndexType_t* ct,
                                                 cusparseIndexBase_t* base, cudaDataType* vt) {
  return cusparseCsrGet(const_cast<cusparseSpMatDescr_t>(d), rows, cols, nnz, const_cast<void**>(off),
                        const_cast<void**>(col), const_cast<void**>(val), ot, ct, base, vt);
}
VGPU_EXPORT cusparseStatus_t cusparseConstCscGet(cusparseConstSpMatDescr_t d, int64_t* rows, int64_t* cols,
                                                 int64_t* nnz, const void** off, const void** row, const void** val,
                                                 cusparseIndexType_t* ot, cusparseIndexType_t* rt,
                                                 cusparseIndexBase_t* base, cudaDataType* vt) {
  return cusparseCscGet(const_cast<cusparseSpMatDescr_t>(d), rows, cols, nnz, const_cast<void**>(off),
                        const_cast<void**>(row), const_cast<void**>(val), ot, rt, base, vt);
}
VGPU_EXPORT cusparseStatus_t cusparseConstCooGet(cusparseConstSpMatDescr_t d, int64_t* rows, int64_t* cols,
                                                 int64_t* nnz, const void** row, const void** col, const void** val,
                                                 cusparseIndexType_t* it, cusparseIndexBase_t* base, cudaDataType* vt) {
  return cusparseCooGet(const_cast<cusparseSpMatDescr_t>(d), rows, cols, nnz, const_cast<void**>(row),
                        const_cast<void**>(col), const_cast<void**>(val), it, base, vt);
}
VGPU_EXPORT cusparseStatus_t cusparseConstDnVecGet(cusparseConstDnVecDescr_t d, int64_t* size, const void** v,
                                                   cudaDataType* t) {
  return cusparseDnVecGet(const_cast<cusparseDnVecDescr_t>(d), size, const_cast<void**>(v), t);
}
VGPU_EXPORT cusparseStatus_t cusparseConstDnVecGetValues(cusparseConstDnVecDescr_t d, const void** v) {
  return cusparseDnVecGetValues(const_cast<cusparseDnVecDescr_t>(d), const_cast<void**>(v));
}
VGPU_EXPORT cusparseStatus_t cusparseConstDnMatGet(cusparseConstDnMatDescr_t d, int64_t* rows, int64_t* cols,
                                                   int64_t* ld, const void** v, cudaDataType* t,
                                                   cusparseOrder_t* order) {
  return cusparseDnMatGet(const_cast<cusparseDnMatDescr_t>(d), rows, cols, ld, const_cast<void**>(v), t, order);
}
VGPU_EXPORT cusparseStatus_t cusparseConstDnMatGetValues(cusparseConstDnMatDescr_t d, const void** v) {
  return cusparseDnMatGetValues(const_cast<cusparseDnMatDescr_t>(d), const_cast<void**>(v));
}

/* ---- the logger ----
   Measured: levels 0 to 5 are taken, and so is any higher one; a negative
   level is INVALID_VALUE. Any mask is taken. A log file that cannot be
   opened is INVALID_VALUE; NULL for a file, a path or a callback is taken.
   This library logs nothing, so the settings are only kept. */

namespace {
struct LoggerState {
  std::mutex mu;
  int level = 0, mask = 0;
  cusparseLoggerCallback_t callback = nullptr;
  FILE* file = nullptr;
  bool owns_file = false, disabled = false;
};
LoggerState& logger() {
  static LoggerState& s = *new LoggerState;
  return s;
}
}  // namespace

VGPU_EXPORT cusparseStatus_t cusparseLoggerSetCallback(cusparseLoggerCallback_t callback) {
  std::lock_guard<std::mutex> l(logger().mu);
  logger().callback = callback;
  return CUSPARSE_STATUS_SUCCESS;
}
VGPU_EXPORT cusparseStatus_t cusparseLoggerSetFile(FILE* file) {
  std::lock_guard<std::mutex> l(logger().mu);
  if (logger().owns_file && logger().file) std::fclose(logger().file);
  logger().file = file;
  logger().owns_file = false;
  return CUSPARSE_STATUS_SUCCESS;
}
VGPU_EXPORT cusparseStatus_t cusparseLoggerOpenFile(const char* path) {
  if (!path) return CUSPARSE_STATUS_SUCCESS;
  FILE* f = std::fopen(path, "w");
  if (!f) return CUSPARSE_STATUS_INVALID_VALUE;
  std::lock_guard<std::mutex> l(logger().mu);
  if (logger().owns_file && logger().file) std::fclose(logger().file);
  logger().file = f;
  logger().owns_file = true;
  return CUSPARSE_STATUS_SUCCESS;
}
VGPU_EXPORT cusparseStatus_t cusparseLoggerSetLevel(int level) {
  if (level < 0) return CUSPARSE_STATUS_INVALID_VALUE;
  std::lock_guard<std::mutex> l(logger().mu);
  logger().level = level;
  return CUSPARSE_STATUS_SUCCESS;
}
VGPU_EXPORT cusparseStatus_t cusparseLoggerSetMask(int mask) {
  if (mask < 0) return CUSPARSE_STATUS_INVALID_VALUE;
  std::lock_guard<std::mutex> l(logger().mu);
  logger().mask = mask;
  return CUSPARSE_STATUS_SUCCESS;
}
VGPU_EXPORT cusparseStatus_t cusparseLoggerForceDisable(void) {
  std::lock_guard<std::mutex> l(logger().mu);
  logger().disabled = true;
  return CUSPARSE_STATUS_SUCCESS;
}

/* ---- SpGEMM: the product count and ALG2/ALG3's memory estimate ----
   Measured: cusparseSpGEMM_getNumProducts answers 0 until the second
   (buffer-carrying) workEstimation call, then the number of scalar products
   A * B takes; NULL for either argument is INVALID_VALUE.
   cusparseSpGEMM_estimateMemory takes only CUSPARSE_SPGEMM_ALG2 and ALG3
   (INVALID_VALUE otherwise) and follows the two-call protocol on buffer 3. */

namespace {
// The number of scalar products op(A) * op(B) takes, or -1 if a matrix
// cannot be read.
int64_t count_products(const SpMat& A, const SpMat& B) {
  Triplets<cd> ta, tb;
  if (!expand(A, &ta) || !expand(B, &tb)) return -1;
  std::vector<int64_t> per_row((size_t)std::max<int64_t>(B.rows, 0), 0);
  for (int64_t r : tb.row)
    if (r >= 0 && r < B.rows) ++per_row[(size_t)r];
  int64_t total = 0;
  for (int64_t c : ta.col)
    if (c >= 0 && c < B.rows) total += per_row[(size_t)c];
  return total;
}
}  // namespace

VGPU_EXPORT cusparseStatus_t cusparseSpGEMM_getNumProducts(cusparseSpGEMMDescr_t d, int64_t* num) {
  if (!known(d) || !num) return CUSPARSE_STATUS_INVALID_VALUE;
  *num = reinterpret_cast<const SpGEMMDescr*>(d)->products;
  return CUSPARSE_STATUS_SUCCESS;
}
VGPU_EXPORT cusparseStatus_t cusparseSpGEMM_estimateMemory(cusparseHandle_t h, cusparseOperation_t unused1,
                                                           cusparseOperation_t unused2, const void* unused3,
                                                           cusparseConstSpMatDescr_t unused4, cusparseConstSpMatDescr_t unused5,
                                                           const void* unused6, cusparseSpMatDescr_t unused7, cudaDataType unused8,
                                                           cusparseSpGEMMAlg_t alg, cusparseSpGEMMDescr_t d, float unused11,
                                                           size_t* bytes3, void* buffer3, size_t* bytes2) {
  VGPU_DEFER_CALL(g_reg, stream_of(h), cusparseSpGEMM_estimateMemory, h, unused1, unused2, unused3, unused4, unused5, unused6, unused7, unused8, alg, d, unused11, bytes3, buffer3, bytes2);
  if (!known(h) || !known(d)) return CUSPARSE_STATUS_NOT_INITIALIZED;
  if (alg != CUSPARSE_SPGEMM_ALG2 && alg != CUSPARSE_SPGEMM_ALG3) return CUSPARSE_STATUS_INVALID_VALUE;
  if (!bytes3) return CUSPARSE_STATUS_INVALID_VALUE;
  if (!buffer3) {
    *bytes3 = 16;
    return CUSPARSE_STATUS_SUCCESS;
  }
  if (bytes2) *bytes2 = 16;
  return CUSPARSE_STATUS_SUCCESS;
}

/* ---- SpGEMMreuse: C = alpha op(A) op(B) + beta C on a structure computed once ----
   workEstimation, nnz (which sets C's nnz), copy (which writes C's offsets
   and column indices and leaves its values alone) and compute (values only:
   alpha A B over C's structure plus beta times C's values, so a product that
   cancels stays an explicit zero). Measured: op(A) and op(B) other than A
   and B, or a compute type other than the values', are NOT_SUPPORTED; each
   phase but compute follows the two-call protocol. */

namespace {
cusparseStatus_t reuse_check(cusparseOperation_t opA, cusparseOperation_t opB, const SpMat& A, const SpMat& B,
                             const SpMat& C) {
  if (opA != CUSPARSE_OPERATION_NON_TRANSPOSE || opB != CUSPARSE_OPERATION_NON_TRANSPOSE)
    return CUSPARSE_STATUS_NOT_SUPPORTED;
  if (A.format != CUSPARSE_FORMAT_CSR || B.format != CUSPARSE_FORMAT_CSR || C.format != CUSPARSE_FORMAT_CSR)
    return CUSPARSE_STATUS_NOT_SUPPORTED;
  if (A.cols != B.rows || C.rows != A.rows || C.cols != B.cols) return CUSPARSE_STATUS_INVALID_VALUE;
  return CUSPARSE_STATUS_SUCCESS;
}
// C's structure: every (i, j) some A(i, k) B(k, j) reaches, columns ascending.
cusparseStatus_t reuse_structure(const SpMat& A, const SpMat& B, SpGEMMDescr* g) {
  Triplets<cd> ta, tb;
  if (!expand(A, &ta) || !expand(B, &tb)) return CUSPARSE_STATUS_INTERNAL_ERROR;
  std::vector<std::vector<int64_t>> brows((size_t)B.rows);
  for (size_t k = 0; k < tb.row.size(); ++k) {
    if (tb.row[k] < 0 || tb.row[k] >= B.rows) return CUSPARSE_STATUS_INVALID_VALUE;
    brows[(size_t)tb.row[k]].push_back(tb.col[k]);
  }
  std::vector<std::set<int64_t>> rows((size_t)A.rows);
  for (size_t k = 0; k < ta.row.size(); ++k) {
    if (ta.row[k] < 0 || ta.row[k] >= A.rows || ta.col[k] < 0 || ta.col[k] >= B.rows)
      return CUSPARSE_STATUS_INVALID_VALUE;
    for (int64_t c : brows[(size_t)ta.col[k]]) rows[(size_t)ta.row[k]].insert(c);
  }
  g->off.assign(1, 0);
  g->col.clear();
  for (const auto& r : rows) {
    g->col.insert(g->col.end(), r.begin(), r.end());
    g->off.push_back((int64_t)g->col.size());
  }
  g->structured = true;
  return CUSPARSE_STATUS_SUCCESS;
}
template <class V>
cusparseStatus_t reuse_compute(cusparseHandle_t h, const void* alpha, const SpMat& A, const SpMat& B,
                               const void* beta, const SpMat& C, cudaDataType ct, const SpGEMMDescr& g) {
  Triplets<V> ta, tb;
  if (!expand(A, &ta) || !expand(B, &tb)) return CUSPARSE_STATUS_INTERNAL_ERROR;
  const V al = scalar<V>(h, alpha, ct), be = beta ? scalar<V>(h, beta, ct) : V(0.0);
  std::vector<std::vector<std::pair<int64_t, V>>> brows((size_t)B.rows);
  for (size_t k = 0; k < tb.val.size(); ++k) brows[(size_t)tb.row[k]].push_back({tb.col[k], tb.val[k]});
  const size_t nnz = g.col.size();
  std::vector<V> out(nnz, V(0.0));
  if (be != V(0.0)) {
    if (!fetch_values(C.values, nnz, C.value_type, &out)) return CUSPARSE_STATUS_INTERNAL_ERROR;
    for (auto& v : out) v *= be;
  }
  for (size_t k = 0; k < ta.val.size(); ++k) {
    const int64_t i = ta.row[k];
    const auto first = g.col.begin() + g.off[(size_t)i], last = g.col.begin() + g.off[(size_t)i + 1];
    for (const auto& [j, v] : brows[(size_t)ta.col[k]]) {
      const auto it = std::lower_bound(first, last, j);
      if (it == last || *it != j) return CUSPARSE_STATUS_INVALID_VALUE;  // B's structure changed
      out[(size_t)(it - g.col.begin())] += al * ta.val[k] * v;
    }
  }
  return store_values(C.values, out, C.value_type) ? CUSPARSE_STATUS_SUCCESS : CUSPARSE_STATUS_INTERNAL_ERROR;
}
#define VGPU_REUSE_ARGS                                                                                         \
  if (!known(h) || !known(d) || !known(matA) || !known(matB) || !known(matC)) return CUSPARSE_STATUS_NOT_INITIALIZED; \
  const auto& A = *reinterpret_cast<const SpMat*>(matA);                                                       \
  const auto& B = *reinterpret_cast<const SpMat*>(matB);                                                       \
  auto& C = *reinterpret_cast<SpMat*>(matC);                                                                   \
  auto& g = *reinterpret_cast<SpGEMMDescr*>(d);                                                                \
  if (const cusparseStatus_t st = reuse_check(opA, opB, A, B, C); st != CUSPARSE_STATUS_SUCCESS) return st
}  // namespace

VGPU_EXPORT cusparseStatus_t cusparseSpGEMMreuse_workEstimation(cusparseHandle_t h, cusparseOperation_t opA,
                                                                cusparseOperation_t opB,
                                                                cusparseConstSpMatDescr_t matA,
                                                                cusparseConstSpMatDescr_t matB,
                                                                cusparseSpMatDescr_t matC, cusparseSpGEMMAlg_t unused6,
                                                                cusparseSpGEMMDescr_t d, size_t* bytes1,
                                                                void* buffer1) {
  VGPU_DEFER_CALL(g_reg, stream_of(h), cusparseSpGEMMreuse_workEstimation, h, opA, opB, matA, matB, matC, unused6, d, bytes1, buffer1);
  VGPU_REUSE_ARGS;
  (void)g;
  if (!bytes1) return CUSPARSE_STATUS_INVALID_VALUE;
  if (!buffer1) *bytes1 = 16;
  else g.products = count_products(A, B);
  return CUSPARSE_STATUS_SUCCESS;
}
VGPU_EXPORT cusparseStatus_t cusparseSpGEMMreuse_nnz(cusparseHandle_t h, cusparseOperation_t opA,
                                                     cusparseOperation_t opB, cusparseConstSpMatDescr_t matA,
                                                     cusparseConstSpMatDescr_t matB, cusparseSpMatDescr_t matC,
                                                     cusparseSpGEMMAlg_t unused6, cusparseSpGEMMDescr_t d, size_t* bytes2,
                                                     void* buffer2, size_t* bytes3, void* buffer3, size_t* bytes4,
                                                     void* buffer4) {
  VGPU_DEFER_CALL(g_reg, stream_of(h), cusparseSpGEMMreuse_nnz, h, opA, opB, matA, matB, matC, unused6, d, bytes2, buffer2, bytes3, buffer3, bytes4, buffer4);
  VGPU_REUSE_ARGS;
  if (!bytes2 || !bytes3 || !bytes4) return CUSPARSE_STATUS_INVALID_VALUE;
  if (!buffer2 || !buffer3 || !buffer4) {
    *bytes2 = *bytes3 = *bytes4 = 16;
    return CUSPARSE_STATUS_SUCCESS;
  }
  if (const cusparseStatus_t st = reuse_structure(A, B, &g); st != CUSPARSE_STATUS_SUCCESS) return st;
  C.nnz = (int64_t)g.col.size();
  return CUSPARSE_STATUS_SUCCESS;
}
VGPU_EXPORT cusparseStatus_t cusparseSpGEMMreuse_copy(cusparseHandle_t h, cusparseOperation_t opA,
                                                      cusparseOperation_t opB, cusparseConstSpMatDescr_t matA,
                                                      cusparseConstSpMatDescr_t matB, cusparseSpMatDescr_t matC,
                                                      cusparseSpGEMMAlg_t unused6, cusparseSpGEMMDescr_t d, size_t* bytes5,
                                                      void* buffer5) {
  VGPU_DEFER_CALL(g_reg, stream_of(h), cusparseSpGEMMreuse_copy, h, opA, opB, matA, matB, matC, unused6, d, bytes5, buffer5);
  VGPU_REUSE_ARGS;
  if (!bytes5) return CUSPARSE_STATUS_INVALID_VALUE;
  if (!buffer5) {
    *bytes5 = 16;
    return CUSPARSE_STATUS_SUCCESS;
  }
  if (!g.structured) return CUSPARSE_STATUS_SUCCESS;  // measured: copy before nnz is a quiet no-op
  const int64_t base = C.base == CUSPARSE_INDEX_BASE_ONE ? 1 : 0;
  const bool ok = put_index_array(C.rows_ptr, g.off, C.row_type, base) &&
                  put_index_array(C.cols_ptr, g.col, C.col_type, base);
  return ok ? CUSPARSE_STATUS_SUCCESS : CUSPARSE_STATUS_INTERNAL_ERROR;
}
VGPU_EXPORT cusparseStatus_t cusparseSpGEMMreuse_compute(cusparseHandle_t h, cusparseOperation_t opA,
                                                         cusparseOperation_t opB, const void* alpha,
                                                         cusparseConstSpMatDescr_t matA,
                                                         cusparseConstSpMatDescr_t matB, const void* beta,
                                                         cusparseSpMatDescr_t matC, cudaDataType ct,
                                                         cusparseSpGEMMAlg_t unused9, cusparseSpGEMMDescr_t d) {
  VGPU_DEFER_CALL(g_reg, stream_of(h), cusparseSpGEMMreuse_compute, h, opA, opB, alpha, matA, matB, beta, matC, ct, unused9, d);
  VGPU_REUSE_ARGS;
  if (A.value_type != B.value_type || A.value_type != C.value_type || ct != scalar_type_of(C.value_type))
    return CUSPARSE_STATUS_NOT_SUPPORTED;
  if (!g.structured) return CUSPARSE_STATUS_SUCCESS;  // measured: a quiet no-op, as copy
  return any_complex({A.value_type, ct}) ? reuse_compute<cd>(h, alpha, A, B, beta, C, ct, g)
                                         : reuse_compute<double>(h, alpha, A, B, beta, C, ct, g);
}
#undef VGPU_REUSE_ARGS

/* ---- SpMMOp ----
   NVIDIA's takes the add, multiply and epilogue operators as LTO-IR (NVVM
   bitcode) and links them into a kernel with nvJitLink. VirtualGPU cannot
   compile LTO-IR (its nvJitLink refuses it by name), so a plan cannot be
   made. Measured on an RTX 3060: NULL operator buffers are INVALID_VALUE
   ("addOperationLtoirBuffer had an illegal value: NULL pointer"), and
   buffers nvJitLink does not take are INTERNAL_ERROR ("nvJitLinkAddData
   failed"), which is what this returns, with its reason, for any. */

VGPU_EXPORT cusparseStatus_t cusparseSpMMOp_createPlan(cusparseHandle_t h, cusparseSpMMOpPlan_t* plan,
                                                       cusparseOperation_t unused2, cusparseOperation_t unused3,
                                                       cusparseConstSpMatDescr_t matA, cusparseConstDnMatDescr_t matB,
                                                       cusparseDnMatDescr_t matC, cudaDataType unused7, cusparseSpMMOpAlg_t unused8,
                                                       const void* add, size_t unused10, const void* mul, size_t unused12,
                                                       const void* epilogue, size_t unused14, size_t* workspace) {
  VGPU_DEFER_CALL(g_reg, stream_of(h), cusparseSpMMOp_createPlan, h, plan, unused2, unused3, matA, matB, matC, unused7, unused8, add, unused10, mul, unused12, epilogue, unused14, workspace);
  if (!known(h)) return CUSPARSE_STATUS_NOT_INITIALIZED;
  if (!plan || !known(matA) || !known(matB) || !known(matC) || !add || !mul || !epilogue || !workspace)
    return CUSPARSE_STATUS_INVALID_VALUE;
  *plan = nullptr;
  if (!quiet())
    std::fprintf(stderr,
                 "[vgpu] cusparseSpMMOp_createPlan: the operators are LTO-IR (NVVM bitcode), which VirtualGPU "
                 "cannot compile; returning CUSPARSE_STATUS_INTERNAL_ERROR as NVIDIA's does when nvJitLink "
                 "refuses them\n");
  return CUSPARSE_STATUS_INTERNAL_ERROR;
}
VGPU_EXPORT cusparseStatus_t cusparseSpMMOp(cusparseSpMMOpPlan_t plan, void*) {
  // No plan can be made (above), so any plan given here is not one.
  (void)plan;
  return CUSPARSE_STATUS_INVALID_VALUE;
}
VGPU_EXPORT cusparseStatus_t cusparseSpMMOp_destroyPlan(cusparseSpMMOpPlan_t plan) {
  (void)plan;
  return CUSPARSE_STATUS_INVALID_VALUE;
}

/* ---- SpSV and SpSM: replacing the analysed matrix's values ---- */

namespace {
cusparseStatus_t update_matrix(cusparseHandle_t h, void* d_p, void* values, int part) {
  if (!known(h)) return CUSPARSE_STATUS_NOT_INITIALIZED;
  if (!known(d_p) || !values || (part != 0 && part != 1)) return CUSPARSE_STATUS_INVALID_VALUE;
  auto* d = static_cast<TriDescr*>(d_p);
  if (!d->analysed) return CUSPARSE_STATUS_SUCCESS;  // measured: taken, and nothing to update yet
  const SpMat& A = d->mat;
  const size_t bytes = (size_t)(part == 0 ? A.nnz : A.rows) * type_bytes(A.value_type);
  void*& slot = part == 0 ? d->values : d->diag;
  if (part == 0 && d->diag) {  // a general update replaces the diagonal too
    cudaFree(d->diag);
    d->diag = nullptr;
  }
  if (!slot && bytes && cudaMalloc(&slot, bytes) != cudaSuccess) return CUSPARSE_STATUS_ALLOC_FAILED;
  if (bytes && cudaMemcpy(slot, values, bytes, cudaMemcpyDeviceToDevice) != cudaSuccess)
    return CUSPARSE_STATUS_EXECUTION_FAILED;
  return CUSPARSE_STATUS_SUCCESS;
}
}  // namespace

// CUDA 12.0's header has neither these nor their enums (12.1 added them).
#if CUSPARSE_VERSION >= 12100
using vgpu_spsv_update_t = cusparseSpSVUpdate_t;
using vgpu_spsm_update_t = cusparseSpSMUpdate_t;
#else
using vgpu_spsv_update_t = int;
using vgpu_spsm_update_t = int;
#endif
VGPU_EXPORT cusparseStatus_t cusparseSpSV_updateMatrix(cusparseHandle_t h, cusparseSpSVDescr_t d, void* values,
                                                       vgpu_spsv_update_t part) {
  VGPU_DEFER_CALL(g_reg, stream_of(h), cusparseSpSV_updateMatrix, h, d, values, part);
  return update_matrix(h, d, values, (int)part);
}
VGPU_EXPORT cusparseStatus_t cusparseSpSM_updateMatrix(cusparseHandle_t h, cusparseSpSMDescr_t d, void* values,
                                                       vgpu_spsm_update_t part) {
  VGPU_DEFER_CALL(g_reg, stream_of(h), cusparseSpSM_updateMatrix, h, d, values, part);
  return update_matrix(h, d, values, (int)part);
}

/* ---- the CSC sort: row indices ascending within each column ---- */

VGPU_EXPORT cusparseStatus_t cusparseXcscsort_bufferSizeExt(cusparseHandle_t h, int, int, int, const int*,
                                                            const int*, size_t* bytes) {
  if (!known(h)) return CUSPARSE_STATUS_NOT_INITIALIZED;
  if (bytes) *bytes = 16;
  return CUSPARSE_STATUS_SUCCESS;
}
// The CSR sort on the transpose: CSC's column offsets are CSR's row offsets.
VGPU_EXPORT cusparseStatus_t cusparseXcscsort(cusparseHandle_t h, int m, int n, int nnz, const cusparseMatDescr_t d,
                                              const int* offsets, int* rows, int* P, void* buffer) {
  VGPU_DEFER_CALL(g_reg, stream_of(h), cusparseXcscsort, h, m, n, nnz, d, offsets, rows, P, buffer);
  return cusparseXcsrsort(h, n, m, nnz, d, offsets, rows, P, buffer);
}
