// The legacy API's helpers, included by cusparse_api.cpp: pruning, coloring,
// nnz counting and compression, unsorted CSR, and the general BSR
// conversions gebsr2gebsr and gebsr2gebsc. What an RTX 3060 showed with
// cuSPARSE 13.0 (nvidia/tests/e2e/sparse_helper_paths.cu checks it):
//
//   * prune{Dense2csr,Csr2csr} keep the entries with |a| > threshold (read
//     host or device as the pointer mode says), columns ascending, in the
//     output descriptor's base. The ByPercentage forms prune the smallest
//     percentage: with the magnitudes sorted ascending into key[0..N-1] (N =
//     m * n, or nnzA), threshold = key[clamp(ceil(N * percentage / 100) - 1,
//     0, N - 1)], and a percentage outside [0, 100] is INVALID_VALUE (at the
//     Nnz step; the _bufferSizeExt calls check nothing). lda < m is
//     INVALID_VALUE. A negative threshold keeps every entry here; NVIDIA's
//     returns 32 entries a row with column indices past n, which nothing can
//     use.
//   * nnz counts the entries != 0 (NaN counts, -0 does not) by row or column;
//     lda < m and a bad direction are INVALID_VALUE, a matrix type other than
//     general MATRIX_TYPE_NOT_SUPPORTED.
//   * nnz_compress counts |a| > tol per row, |a| in the value type's precision
//     (cuCabsf for complex); only tol's real part is read, and a negative one
//     is INVALID_VALUE. csr2csr_compress keeps the same entries. NVIDIA's
//     keeps a > tol instead (negative entries dropped, for real and complex
//     values alike), leaving the slots nnz_compress counted for them unwritten;
//     this keeps what the count and the documentation say.
//   * csru2csr needs csru2csr_bufferSizeExt first, with the same nnz, and
//     csr2csru a csru2csr before it (INVALID_VALUE otherwise): the sort's
//     permutation lives in the info.
//   * gebsr2gebsr is gebsr2csr followed by csr2gebsr: C has a block wherever
//     A stores an element (zero or not), rows and columns past A's padded
//     with zeros, both in the one block direction given.
//   * gebsr2gebsc moves whole blocks (no block is transposed); its index base
//     is both the input's and the output's.
//   * csrcolor colors the adjacency graph: neighbours never share a color,
//     and reordering lists the nodes by color. NVIDIA's coloring algorithm is
//     undocumented and randomized (a 6-node path gets 11 colors); this one is
//     greedy, so the colors differ but have the same properties.

namespace {

template <> struct Legacy<__half> { using V = double; static constexpr cudaDataType type = CUDA_R_16F; };

// |a| in the value type's precision, as NVIDIA's compares it.
template <class T> double magnitude(typename Legacy<T>::V v) {
  if constexpr (std::is_same_v<T, cuComplex>) {
    const float x = std::fabs((float)re(v)), y = std::fabs((float)im(v));
    const float big = std::max(x, y), small = std::min(x, y);
    if (big == 0.0f) return 0.0;
    const float r = small / big;
    return (double)(big * std::sqrt(1.0f + r * r));
  } else if constexpr (std::is_same_v<T, cuDoubleComplex>) {
    return std::abs(cd(re(v), im(v)));
  } else {
    return std::fabs(re(v));
  }
}

// Writes an int count where the pointer mode says.
void put_count(cusparseHandle_t h, int* where, int v) {
  if (!where) return;
  if (reinterpret_cast<const Handle*>(h)->mode == CUSPARSE_POINTER_MODE_DEVICE)
    cudaMemcpy(where, &v, sizeof(int), cudaMemcpyHostToDevice);
  else
    *where = v;
}

/* ---- pruning ---- */

struct PruneInfo {
  double threshold = 0.0;
};

// The threshold that prunes `percentage` percent of `mags`.
double percentile_threshold(std::vector<double> mags, float percentage) {
  if (mags.empty()) return 0.0;
  std::sort(mags.begin(), mags.end());
  const double n = (double)mags.size();
  long long pos = (long long)std::ceil(n * (double)percentage / 100.0) - 1;
  pos = std::max(0LL, std::min(pos, (long long)mags.size() - 1));
  return mags[(size_t)pos];
}

// The kept entries as CSR (zero-based), from a dense or a CSR source.
template <class V> struct Kept {
  std::vector<int> off, col;
  std::vector<V> val;
};
template <class T>
cusparseStatus_t dense_source(int m, int n, const T* A, int lda, std::vector<typename Legacy<T>::V>* out) {
  if (m < 0 || n < 0 || lda < std::max(1, m)) return CUSPARSE_STATUS_INVALID_VALUE;
  if (m == 0 || n == 0) {
    out->clear();
    return CUSPARSE_STATUS_SUCCESS;
  }
  return fetch_values(A, (size_t)lda * (n - 1) + m, Legacy<T>::type, out) ? CUSPARSE_STATUS_SUCCESS
                                                                          : CUSPARSE_STATUS_EXECUTION_FAILED;
}
template <class T>
Kept<typename Legacy<T>::V> keep_dense(int m, int n, int lda, const std::vector<typename Legacy<T>::V>& a, double th) {
  Kept<typename Legacy<T>::V> k;
  k.off.assign(1, 0);
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      const auto v = a[(size_t)j * lda + i];
      if (magnitude<T>(v) > th || th < 0.0) {
        k.col.push_back(j);
        k.val.push_back(v);
      }
    }
    k.off.push_back((int)k.col.size());
  }
  return k;
}
template <class T>
cusparseStatus_t csr_source(int m, int n, int nnz, cusparseMatDescr_t d, const T* val, const int* off, const int* col,
                            Kept<typename Legacy<T>::V>* out) {
  if (m < 0 || n < 0 || nnz < 0 || !descr(d)) return CUSPARSE_STATUS_INVALID_VALUE;
  if (!read_legacy_csr(m, nnz, d, off, col, &out->off, &out->col)) return CUSPARSE_STATUS_INVALID_VALUE;
  if (out->off[(size_t)m] != nnz) return CUSPARSE_STATUS_INVALID_VALUE;
  return fetch_values(val, (size_t)nnz, Legacy<T>::type, &out->val) ? CUSPARSE_STATUS_SUCCESS
                                                                     : CUSPARSE_STATUS_EXECUTION_FAILED;
}
template <class T>
Kept<typename Legacy<T>::V> keep_csr(int m, const Kept<typename Legacy<T>::V>& a, double th) {
  Kept<typename Legacy<T>::V> k;
  k.off.assign(1, 0);
  for (int i = 0; i < m; ++i) {
    // Columns ascending, whatever order A stores them in.
    std::vector<std::pair<int, size_t>> row;
    for (int e = a.off[(size_t)i]; e < a.off[(size_t)i + 1]; ++e)
      if (magnitude<T>(a.val[(size_t)e]) > th || th < 0.0) row.push_back({a.col[(size_t)e], (size_t)e});
    std::stable_sort(row.begin(), row.end(), [](auto& x, auto& y) { return x.first < y.first; });
    for (const auto& [c, e] : row) {
      k.col.push_back(c);
      k.val.push_back(a.val[e]);
    }
    k.off.push_back((int)k.col.size());
  }
  return k;
}
template <class T> double read_threshold(cusparseHandle_t h, const T* th) {
  using V = typename Legacy<T>::V;
  return re(scalar<V>(h, th, Legacy<T>::type));
}
template <class V>
cusparseStatus_t write_offsets(cusparseHandle_t h, const Kept<V>& k, cusparseMatDescr_t dC, int* offC, int* total) {
  const int b = base_of(dC);
  std::vector<int> o(k.off);
  for (int& x : o) x += b;
  if (!put_ints(offC, o)) return CUSPARSE_STATUS_EXECUTION_FAILED;
  put_count(h, total, (int)k.col.size());
  return CUSPARSE_STATUS_SUCCESS;
}
template <class T>
cusparseStatus_t write_entries(const Kept<typename Legacy<T>::V>& k, cusparseMatDescr_t dC, T* valC, int* colC) {
  const int b = base_of(dC);
  std::vector<int> c(k.col);
  for (int& x : c) x += b;
  return put_ints(colC, c) && store_values(valC, k.val, Legacy<T>::type) ? CUSPARSE_STATUS_SUCCESS
                                                                          : CUSPARSE_STATUS_EXECUTION_FAILED;
}

template <class T>
cusparseStatus_t prune_dense(cusparseHandle_t h, int m, int n, const T* A, int lda, double th, bool by_pct, float pct,
                             cusparseMatDescr_t dC, T* valC, int* offC, int* colC, int* total, void* info) {
  if (!known(h)) return CUSPARSE_STATUS_NOT_INITIALIZED;
  if (!descr(dC)) return CUSPARSE_STATUS_INVALID_VALUE;
  if (by_pct && (pct < 0.0f || pct > 100.0f)) return CUSPARSE_STATUS_INVALID_VALUE;
  std::vector<typename Legacy<T>::V> a;
  if (const cusparseStatus_t st = dense_source(m, n, A, lda, &a); st != CUSPARSE_STATUS_SUCCESS) return st;
  if (by_pct) {
    std::vector<double> mags;
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < m; ++i) mags.push_back(magnitude<T>(a[(size_t)j * lda + i]));
    th = percentile_threshold(std::move(mags), pct);
    if (known(info)) static_cast<PruneInfo*>(info)->threshold = th;
  }
  const auto k = keep_dense<T>(m, n, lda, a, th);
  return valC ? write_entries<T>(k, dC, valC, colC) : write_offsets(h, k, dC, offC, total);
}

template <class T>
cusparseStatus_t prune_csr(cusparseHandle_t h, int m, int n, int nnzA, cusparseMatDescr_t dA, const T* valA,
                           const int* offA, const int* colA, double th, bool by_pct, float pct, cusparseMatDescr_t dC,
                           T* valC, int* offC, int* colC, int* total, void* info) {
  if (!known(h)) return CUSPARSE_STATUS_NOT_INITIALIZED;
  if (!descr(dC)) return CUSPARSE_STATUS_INVALID_VALUE;
  if (by_pct && (pct < 0.0f || pct > 100.0f)) return CUSPARSE_STATUS_INVALID_VALUE;
  Kept<typename Legacy<T>::V> a;
  if (const cusparseStatus_t st = csr_source<T>(m, n, nnzA, dA, valA, offA, colA, &a); st != CUSPARSE_STATUS_SUCCESS)
    return st;
  if (by_pct) {
    std::vector<double> mags;
    for (const auto& v : a.val) mags.push_back(magnitude<T>(v));
    th = percentile_threshold(std::move(mags), pct);
    if (known(info)) static_cast<PruneInfo*>(info)->threshold = th;
  }
  const auto k = keep_csr<T>(m, a, th);
  return valC ? write_entries<T>(k, dC, valC, colC) : write_offsets(h, k, dC, offC, total);
}

}  // namespace

VGPU_EXPORT cusparseStatus_t cusparseCreatePruneInfo(pruneInfo_t* info) {
  if (!info) return CUSPARSE_STATUS_INVALID_VALUE;
  *info = reinterpret_cast<pruneInfo_t>(track(new PruneInfo()));
  return CUSPARSE_STATUS_SUCCESS;
}
VGPU_EXPORT cusparseStatus_t cusparseDestroyPruneInfo(pruneInfo_t info) {
  if (!known(info)) return CUSPARSE_STATUS_INVALID_VALUE;
  untrack(info);
  delete reinterpret_cast<PruneInfo*>(info);
  return CUSPARSE_STATUS_SUCCESS;
}

#define VGPU_PRUNE(P, T)                                                                                            \
  VGPU_EXPORT cusparseStatus_t cusparse##P##pruneDense2csr_bufferSizeExt(                                          \
      cusparseHandle_t h, int, int, const T*, int, const T*, const cusparseMatDescr_t, const T*, const int*,       \
      const int*, size_t* bytes) {                                                                                  \
    return size_out(h, bytes, 512);                                                                                 \
  }                                                                                                                 \
  VGPU_EXPORT cusparseStatus_t cusparse##P##pruneDense2csrNnz(cusparseHandle_t h, int m, int n, const T* A,        \
                                                              int lda, const T* th, const cusparseMatDescr_t dC,   \
                                                              int* offC, int* total, void*) {                      \
    if (!known(h)) return CUSPARSE_STATUS_NOT_INITIALIZED;                                                          \
    return prune_dense<T>(h, m, n, A, lda, read_threshold(h, th), false, 0.0f, dC, nullptr, offC, nullptr, total,  \
                          nullptr);                                                                                 \
  }                                                                                                                 \
  VGPU_EXPORT cusparseStatus_t cusparse##P##pruneDense2csr(cusparseHandle_t h, int m, int n, const T* A, int lda,  \
                                                           const T* th, const cusparseMatDescr_t dC, T* valC,      \
                                                           const int*, int* colC, void*) {                         \
    if (!known(h)) return CUSPARSE_STATUS_NOT_INITIALIZED;                                                          \
    return prune_dense<T>(h, m, n, A, lda, read_threshold(h, th), false, 0.0f, dC, valC, nullptr, colC, nullptr,   \
                          nullptr);                                                                                 \
  }                                                                                                                 \
  VGPU_EXPORT cusparseStatus_t cusparse##P##pruneCsr2csr_bufferSizeExt(                                            \
      cusparseHandle_t h, int, int, int, const cusparseMatDescr_t, const T*, const int*, const int*, const T*,     \
      const cusparseMatDescr_t, const T*, const int*, const int*, size_t* bytes) {                                 \
    return size_out(h, bytes, 512);                                                                                 \
  }                                                                                                                 \
  VGPU_EXPORT cusparseStatus_t cusparse##P##pruneCsr2csrNnz(                                                        \
      cusparseHandle_t h, int m, int n, int nnzA, const cusparseMatDescr_t dA, const T* valA, const int* offA,     \
      const int* colA, const T* th, const cusparseMatDescr_t dC, int* offC, int* total, void*) {                   \
    if (!known(h)) return CUSPARSE_STATUS_NOT_INITIALIZED;                                                          \
    return prune_csr<T>(h, m, n, nnzA, dA, valA, offA, colA, read_threshold(h, th), false, 0.0f, dC, nullptr,      \
                        offC, nullptr, total, nullptr);                                                             \
  }                                                                                                                 \
  VGPU_EXPORT cusparseStatus_t cusparse##P##pruneCsr2csr(                                                           \
      cusparseHandle_t h, int m, int n, int nnzA, const cusparseMatDescr_t dA, const T* valA, const int* offA,     \
      const int* colA, const T* th, const cusparseMatDescr_t dC, T* valC, const int*, int* colC, void*) {          \
    if (!known(h)) return CUSPARSE_STATUS_NOT_INITIALIZED;                                                          \
    return prune_csr<T>(h, m, n, nnzA, dA, valA, offA, colA, read_threshold(h, th), false, 0.0f, dC, valC,         \
                        nullptr, colC, nullptr, nullptr);                                                           \
  }                                                                                                                 \
  VGPU_EXPORT cusparseStatus_t cusparse##P##pruneDense2csrByPercentage_bufferSizeExt(                              \
      cusparseHandle_t h, int, int, const T*, int, float, const cusparseMatDescr_t, const T*, const int*,          \
      const int*, pruneInfo_t, size_t* bytes) {                                                                     \
    return size_out(h, bytes, 1024);                                                                                \
  }                                                                                                                 \
  VGPU_EXPORT cusparseStatus_t cusparse##P##pruneDense2csrNnzByPercentage(                                          \
      cusparseHandle_t h, int m, int n, const T* A, int lda, float pct, const cusparseMatDescr_t dC, int* offC,    \
      int* total, pruneInfo_t info, void*) {                                                                        \
    return prune_dense<T>(h, m, n, A, lda, 0.0, true, pct, dC, nullptr, offC, nullptr, total, info);               \
  }                                                                                                                 \
  VGPU_EXPORT cusparseStatus_t cusparse##P##pruneDense2csrByPercentage(                                             \
      cusparseHandle_t h, int m, int n, const T* A, int lda, float pct, const cusparseMatDescr_t dC, T* valC,      \
      const int*, int* colC, pruneInfo_t info, void*) {                                                             \
    return prune_dense<T>(h, m, n, A, lda, 0.0, true, pct, dC, valC, nullptr, colC, nullptr, info);                \
  }                                                                                                                 \
  VGPU_EXPORT cusparseStatus_t cusparse##P##pruneCsr2csrByPercentage_bufferSizeExt(                                \
      cusparseHandle_t h, int, int, int, const cusparseMatDescr_t, const T*, const int*, const int*, float,        \
      const cusparseMatDescr_t, const T*, const int*, const int*, pruneInfo_t, size_t* bytes) {                    \
    return size_out(h, bytes, 1024);                                                                                \
  }                                                                                                                 \
  VGPU_EXPORT cusparseStatus_t cusparse##P##pruneCsr2csrNnzByPercentage(                                            \
      cusparseHandle_t h, int m, int n, int nnzA, const cusparseMatDescr_t dA, const T* valA, const int* offA,     \
      const int* colA, float pct, const cusparseMatDescr_t dC, int* offC, int* total, pruneInfo_t info, void*) {   \
    return prune_csr<T>(h, m, n, nnzA, dA, valA, offA, colA, 0.0, true, pct, dC, nullptr, offC, nullptr, total,    \
                        info);                                                                                      \
  }                                                                                                                 \
  VGPU_EXPORT cusparseStatus_t cusparse##P##pruneCsr2csrByPercentage(                                               \
      cusparseHandle_t h, int m, int n, int nnzA, const cusparseMatDescr_t dA, const T* valA, const int* offA,     \
      const int* colA, float pct, const cusparseMatDescr_t dC, T* valC, const int*, int* colC, pruneInfo_t info,   \
      void*) {                                                                                                      \
    return prune_csr<T>(h, m, n, nnzA, dA, valA, offA, colA, 0.0, true, pct, dC, valC, nullptr, colC, nullptr,     \
                        info);                                                                                      \
  }
VGPU_PRUNE(H, __half)
VGPU_PRUNE(S, float)
VGPU_PRUNE(D, double)
#undef VGPU_PRUNE

/* ---- nnz, nnz_compress, csr2csr_compress ---- */

namespace {
template <class T>
cusparseStatus_t nnz_dense(cusparseHandle_t h, cusparseDirection_t dir, int m, int n, cusparseMatDescr_t d, const T* A,
                           int lda, int* per, int* total) {
  using V = typename Legacy<T>::V;
  if (!known(h)) return CUSPARSE_STATUS_NOT_INITIALIZED;
  if (!descr(d) || (dir != CUSPARSE_DIRECTION_ROW && dir != CUSPARSE_DIRECTION_COLUMN))
    return CUSPARSE_STATUS_INVALID_VALUE;
  if (descr(d)->type != CUSPARSE_MATRIX_TYPE_GENERAL) return CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
  std::vector<V> a;
  if (const cusparseStatus_t st = dense_source(m, n, A, lda, &a); st != CUSPARSE_STATUS_SUCCESS) return st;
  const bool by_row = dir == CUSPARSE_DIRECTION_ROW;
  std::vector<int> counts((size_t)(by_row ? m : n), 0);
  int sum = 0;
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < m; ++i)
      if (a[(size_t)j * lda + i] != V(0.0) || std::isnan(re(a[(size_t)j * lda + i])) ||
          std::isnan(im(a[(size_t)j * lda + i]))) {
        ++counts[(size_t)(by_row ? i : j)];
        ++sum;
      }
  if (!put_ints(per, counts)) return CUSPARSE_STATUS_EXECUTION_FAILED;
  put_count(h, total, sum);
  return CUSPARSE_STATUS_SUCCESS;
}

template <class T>
cusparseStatus_t compress_rows(int m, cusparseMatDescr_t d, const T* val, const int* off, double tol,
                               std::vector<std::vector<size_t>>* keep, std::vector<int>* cols_out, const int* col,
                               int nnz_hint) {
  using V = typename Legacy<T>::V;
  if (m < 0 || !descr(d)) return CUSPARSE_STATUS_INVALID_VALUE;
  if (tol < 0.0) return CUSPARSE_STATUS_INVALID_VALUE;
  std::vector<int> o, c;
  if (!read_legacy_csr(m, col ? nnz_hint : 0, d, off, col, &o, &c)) return CUSPARSE_STATUS_INVALID_VALUE;
  const int nnz = o[(size_t)m];
  std::vector<V> v;
  if (!fetch_values(val, (size_t)std::max(nnz, 0), Legacy<T>::type, &v)) return CUSPARSE_STATUS_EXECUTION_FAILED;
  keep->assign((size_t)m, {});
  for (int i = 0; i < m; ++i)
    for (int e = o[(size_t)i]; e < o[(size_t)i + 1]; ++e)
      if (e >= 0 && e < nnz && magnitude<T>(v[(size_t)e]) > tol) (*keep)[(size_t)i].push_back((size_t)e);
  if (cols_out) *cols_out = c;
  return CUSPARSE_STATUS_SUCCESS;
}

template <class T>
cusparseStatus_t nnz_compress(cusparseHandle_t h, int m, cusparseMatDescr_t d, const T* val, const int* off, int* per,
                              int* total, T tol) {
  if (!known(h)) return CUSPARSE_STATUS_NOT_INITIALIZED;
  std::vector<std::vector<size_t>> keep;
  std::vector<typename Legacy<T>::V> one;
  decode(reinterpret_cast<const uint8_t*>(&tol), 1, Legacy<T>::type, &one);  // only the real part counts
  if (const cusparseStatus_t st = compress_rows<T>(m, d, val, off, re(one[0]), &keep, nullptr, nullptr, 0);
      st != CUSPARSE_STATUS_SUCCESS)
    return st;
  std::vector<int> counts;
  int sum = 0;
  for (const auto& r : keep) {
    counts.push_back((int)r.size());
    sum += (int)r.size();
  }
  if (!put_ints(per, counts)) return CUSPARSE_STATUS_EXECUTION_FAILED;
  put_count(h, total, sum);
  return CUSPARSE_STATUS_SUCCESS;
}

template <class T>
cusparseStatus_t csr2csr_compress(cusparseHandle_t h, int m, int n, cusparseMatDescr_t d, const T* valA,
                                  const int* colA, const int* offA, int nnzA, const int* per, T* valC, int* colC,
                                  int* offC, T tol) {
  using V = typename Legacy<T>::V;
  if (!known(h)) return CUSPARSE_STATUS_NOT_INITIALIZED;
  if (n < 0) return CUSPARSE_STATUS_INVALID_VALUE;
  std::vector<V> one;
  decode(reinterpret_cast<const uint8_t*>(&tol), 1, Legacy<T>::type, &one);
  std::vector<std::vector<size_t>> keep;
  std::vector<int> cols;
  if (const cusparseStatus_t st = compress_rows<T>(m, d, valA, offA, re(one[0]), &keep, &cols, colA, -1);
      st != CUSPARSE_STATUS_SUCCESS)
    return st;
  (void)nnzA;
  (void)per;  // the counts nnz_compress gave are the ones recomputed here
  std::vector<V> v;
  const int nnz = (int)cols.size();
  if (!fetch_values(valA, (size_t)nnz, Legacy<T>::type, &v)) return CUSPARSE_STATUS_EXECUTION_FAILED;
  const int b = base_of(d);
  std::vector<int> o{b}, c;
  std::vector<V> out;
  for (const auto& r : keep) {
    for (size_t e : r) {
      c.push_back(cols[e] + b);
      out.push_back(v[e]);
    }
    o.push_back((int)c.size() + b);
  }
  return put_ints(offC, o) && put_ints(colC, c) && store_values(valC, out, Legacy<T>::type)
             ? CUSPARSE_STATUS_SUCCESS
             : CUSPARSE_STATUS_EXECUTION_FAILED;
}
}  // namespace

#define VGPU_NNZ(P, T)                                                                                              \
  VGPU_EXPORT cusparseStatus_t cusparse##P##nnz(cusparseHandle_t h, cusparseDirection_t dir, int m, int n,         \
                                                const cusparseMatDescr_t d, const T* A, int lda, int* per,         \
                                                int* total) {                                                       \
    return nnz_dense<T>(h, dir, m, n, d, A, lda, per, total);                                                       \
  }                                                                                                                 \
  VGPU_EXPORT cusparseStatus_t cusparse##P##nnz_compress(cusparseHandle_t h, int m, const cusparseMatDescr_t d,    \
                                                         const T* val, const int* off, int* per, int* total,       \
                                                         T tol) {                                                   \
    return nnz_compress<T>(h, m, d, val, off, per, total, tol);                                                     \
  }                                                                                                                 \
  VGPU_EXPORT cusparseStatus_t cusparse##P##csr2csr_compress(                                                       \
      cusparseHandle_t h, int m, int n, const cusparseMatDescr_t d, const T* valA, const int* colA,                \
      const int* offA, int nnzA, const int* per, T* valC, int* colC, int* offC, T tol) {                           \
    return csr2csr_compress<T>(h, m, n, d, valA, colA, offA, nnzA, per, valC, colC, offC, tol);                     \
  }
VGPU_NNZ(S, float)
VGPU_NNZ(D, double)
VGPU_NNZ(C, cuComplex)
VGPU_NNZ(Z, cuDoubleComplex)
#undef VGPU_NNZ

/* ---- unsorted CSR: csru2csr sorts and remembers how, csr2csru undoes it ---- */

namespace {
struct Csru2csrInfo {
  int prepared_nnz = -1;   // set by csru2csr_bufferSizeExt
  std::vector<int> perm;   // sorted position k came from perm[k]; empty until csru2csr
};
Csru2csrInfo* csru_info(const void* p) { return known(p) ? static_cast<Csru2csrInfo*>(const_cast<void*>(p)) : nullptr; }

template <class T>
cusparseStatus_t csru(cusparseHandle_t h, int m, int n, int nnz, cusparseMatDescr_t d, T* val, const int* off,
                      int* col, void* info_p, bool to_sorted) {
  using V = typename Legacy<T>::V;
  if (!known(h)) return CUSPARSE_STATUS_NOT_INITIALIZED;
  Csru2csrInfo* info = csru_info(info_p);
  if (!info || !descr(d) || m < 0 || n < 0 || nnz < 0) return CUSPARSE_STATUS_INVALID_VALUE;
  if (to_sorted ? info->prepared_nnz != nnz : info->perm.size() != (size_t)nnz) return CUSPARSE_STATUS_INVALID_VALUE;
  std::vector<int> o, c;
  if (!read_legacy_csr(m, nnz, d, off, col, &o, &c) || o[(size_t)m] != nnz) return CUSPARSE_STATUS_INVALID_VALUE;
  std::vector<V> v;
  if (!fetch_values(val, (size_t)nnz, Legacy<T>::type, &v)) return CUSPARSE_STATUS_EXECUTION_FAILED;
  std::vector<int> c2((size_t)nnz);
  std::vector<V> v2((size_t)nnz);
  if (to_sorted) {
    info->perm.resize((size_t)nnz);
    for (int i = 0; i < m; ++i) {
      std::vector<int> idx;
      for (int e = o[(size_t)i]; e < o[(size_t)i + 1]; ++e) idx.push_back(e);
      std::stable_sort(idx.begin(), idx.end(), [&](int x, int y) { return c[(size_t)x] < c[(size_t)y]; });
      for (size_t k = 0; k < idx.size(); ++k) info->perm[(size_t)o[(size_t)i] + k] = idx[k];
    }
    for (int k = 0; k < nnz; ++k) {
      c2[(size_t)k] = c[(size_t)info->perm[(size_t)k]];
      v2[(size_t)k] = v[(size_t)info->perm[(size_t)k]];
    }
  } else {
    for (int k = 0; k < nnz; ++k) {
      c2[(size_t)info->perm[(size_t)k]] = c[(size_t)k];
      v2[(size_t)info->perm[(size_t)k]] = v[(size_t)k];
    }
  }
  const int b = base_of(d);
  for (int& x : c2) x += b;
  return put_ints(col, c2) && store_values(val, v2, Legacy<T>::type) ? CUSPARSE_STATUS_SUCCESS
                                                                     : CUSPARSE_STATUS_EXECUTION_FAILED;
}
}  // namespace

VGPU_EXPORT cusparseStatus_t cusparseCreateCsru2csrInfo(csru2csrInfo_t* info) {
  if (!info) return CUSPARSE_STATUS_INVALID_VALUE;
  *info = reinterpret_cast<csru2csrInfo_t>(track(new Csru2csrInfo()));
  return CUSPARSE_STATUS_SUCCESS;
}
VGPU_EXPORT cusparseStatus_t cusparseDestroyCsru2csrInfo(csru2csrInfo_t info) {
  if (!csru_info(info)) return CUSPARSE_STATUS_INVALID_VALUE;
  untrack(info);
  delete reinterpret_cast<Csru2csrInfo*>(info);
  return CUSPARSE_STATUS_SUCCESS;
}

#define VGPU_CSRU(P, T)                                                                                             \
  VGPU_EXPORT cusparseStatus_t cusparse##P##csru2csr_bufferSizeExt(cusparseHandle_t h, int m, int n, int nnz, T*,  \
                                                                   const int*, int*, csru2csrInfo_t info,           \
                                                                   size_t* bytes) {                                 \
    if (!known(h)) return CUSPARSE_STATUS_NOT_INITIALIZED;                                                          \
    if (!csru_info(info) || !bytes || m < 0 || n < 0 || nnz < 0) return CUSPARSE_STATUS_INVALID_VALUE;              \
    csru_info(info)->prepared_nnz = nnz;                                                                            \
    *bytes = 512;                                                                                                   \
    return CUSPARSE_STATUS_SUCCESS;                                                                                 \
  }                                                                                                                 \
  VGPU_EXPORT cusparseStatus_t cusparse##P##csru2csr(cusparseHandle_t h, int m, int n, int nnz,                    \
                                                     const cusparseMatDescr_t d, T* val, const int* off, int* col, \
                                                     csru2csrInfo_t info, void*) {                                  \
    return csru<T>(h, m, n, nnz, d, val, off, col, info, true);                                                     \
  }                                                                                                                 \
  VGPU_EXPORT cusparseStatus_t cusparse##P##csr2csru(cusparseHandle_t h, int m, int n, int nnz,                    \
                                                     const cusparseMatDescr_t d, T* val, const int* off, int* col, \
                                                     csru2csrInfo_t info, void*) {                                  \
    return csru<T>(h, m, n, nnz, d, val, off, col, info, false);                                                    \
  }
VGPU_CSRU(S, float)
VGPU_CSRU(D, double)
VGPU_CSRU(C, cuComplex)
VGPU_CSRU(Z, cuDoubleComplex)
#undef VGPU_CSRU

/* ---- gebsr2gebsr and gebsr2gebsc ---- */

namespace {
// A general BSR matrix's stored elements as zero-based CSR of the element
// matrix, each row's columns ascending.
template <class V> void bsr_to_csr_host(const HostBsr<V>& a, std::vector<int>* off, std::vector<int>* col,
                                        std::vector<V>* val) {
  off->assign(1, 0);
  col->clear();
  val->clear();
  for (int i = 0; i < a.mb; ++i)
    for (int r = 0; r < a.rbd; ++r) {
      std::vector<std::pair<int, V>> row;
      for (int k = a.off[(size_t)i]; k < a.off[(size_t)i + 1]; ++k)
        for (int c = 0; c < a.cbd; ++c)
          row.push_back({a.col[(size_t)k] * a.cbd + c, a.val.empty() ? V(0.0) : a.val[a.at(k, r, c)]});
      std::stable_sort(row.begin(), row.end(), [](auto& x, auto& y) { return x.first < y.first; });
      for (const auto& [cc, v] : row) {
        col->push_back(cc);
        val->push_back(v);
      }
      off->push_back((int)col->size());
    }
}

template <class T>
cusparseStatus_t gebsr2gebsr(cusparseHandle_t h, cusparseDirection_t dir, int mb, int nb, int nnzb,
                             cusparseMatDescr_t dA, const T* valA, const int* offA, const int* colA, int rbdA,
                             int cbdA, cusparseMatDescr_t dC, T* valC, int* offC, int* colC, int rbdC, int cbdC,
                             int* total) {
  using V = typename Legacy<T>::V;
  constexpr cudaDataType t = Legacy<T>::type;
  if (!known(h)) return CUSPARSE_STATUS_NOT_INITIALIZED;
  if (rbdC <= 0 || cbdC <= 0 || !descr(dC)) return CUSPARSE_STATUS_INVALID_VALUE;
  HostBsr<V> a;
  if (const cusparseStatus_t st = read_bsr(dir, mb, nb, nnzb, dA, valC ? valA : nullptr, offA, colA, rbdA, cbdA, t, &a);
      st != CUSPARSE_STATUS_SUCCESS)
    return st;
  std::vector<int> o, c;
  std::vector<V> v;
  bsr_to_csr_host(a, &o, &c, &v);
  const int m = mb * rbdA, n = nb * cbdA;
  const auto rows = block_pattern(m, n, o, c, rbdC, cbdC);
  const int base = base_of(dC);
  if (!valC) {
    std::vector<int> out{base};
    for (const auto& r : rows) out.push_back(out.back() + (int)r.size());
    if (!put_ints(offC, out)) return CUSPARSE_STATUS_EXECUTION_FAILED;
    put_count(h, total, out.back() - base);
    return CUSPARSE_STATUS_SUCCESS;
  }
  HostBsr<V> out;
  out.mb = (int)rows.size();
  out.rbd = rbdC;
  out.cbd = cbdC;
  out.row_major = dir == CUSPARSE_DIRECTION_ROW;
  out.off.push_back(0);
  for (const auto& r : rows) {
    out.col.insert(out.col.end(), r.begin(), r.end());
    out.off.push_back((int)out.col.size());
  }
  out.val.assign(out.col.size() * rbdC * cbdC, V(0.0));
  for (int i = 0; i < m; ++i) {
    const int bi = i / rbdC;
    for (int k = o[(size_t)i]; k < o[(size_t)i + 1]; ++k) {
      const int j = c[(size_t)k];
      const auto first = out.col.begin() + out.off[(size_t)bi], last = out.col.begin() + out.off[(size_t)bi + 1];
      const int blk = (int)(std::lower_bound(first, last, j / cbdC) - out.col.begin());
      out.val[out.at(blk, i % rbdC, j % cbdC)] += v[(size_t)k];
    }
  }
  std::vector<int> offs(out.off), cols(out.col);
  for (int& x : offs) x += base;
  for (int& x : cols) x += base;
  const bool ok = put_ints(offC, offs) && put_ints(colC, cols) && store_values(valC, out.val, t);
  return ok ? CUSPARSE_STATUS_SUCCESS : CUSPARSE_STATUS_EXECUTION_FAILED;
}

template <class T>
cusparseStatus_t gebsr2gebsc(cusparseHandle_t h, int mb, int nb, int nnzb, const T* val, const int* off, const int* col,
                             int rbd, int cbd, T* bscVal, int* bscRow, int* bscColPtr, cusparseAction_t action,
                             cusparseIndexBase_t base) {
  if (!known(h)) return CUSPARSE_STATUS_NOT_INITIALIZED;
  if (mb < 0 || nb < 0 || nnzb < 0 || rbd <= 0 || cbd <= 0) return CUSPARSE_STATUS_INVALID_VALUE;
  if (action != CUSPARSE_ACTION_SYMBOLIC && action != CUSPARSE_ACTION_NUMERIC) return CUSPARSE_STATUS_INVALID_VALUE;
  if (base != CUSPARSE_INDEX_BASE_ZERO && base != CUSPARSE_INDEX_BASE_ONE) return CUSPARSE_STATUS_INVALID_VALUE;
  const int b = base == CUSPARSE_INDEX_BASE_ONE ? 1 : 0;
  std::vector<int> o = ints(off, (size_t)mb + 1), c = ints(col, (size_t)nnzb);
  for (int& x : o) x -= b;
  for (int& x : c) x -= b;
  if (o[0] != 0 || o[(size_t)mb] != nnzb) return CUSPARSE_STATUS_INVALID_VALUE;
  std::vector<int> count((size_t)nb, 0);
  for (int x : c) {
    if (x < 0 || x >= nb) return CUSPARSE_STATUS_INVALID_VALUE;
    ++count[(size_t)x];
  }
  std::vector<int> colptr((size_t)nb + 1, 0);
  for (int j = 0; j < nb; ++j) colptr[(size_t)j + 1] = colptr[(size_t)j] + count[(size_t)j];
  // Whole blocks move: the values are bytes here.
  const size_t bs = (size_t)rbd * cbd * sizeof(T);
  std::vector<uint8_t> in(action == CUSPARSE_ACTION_NUMERIC ? (size_t)nnzb * bs : 0), out(in.size());
  if (!in.empty() && cudaMemcpy(in.data(), val, in.size(), cudaMemcpyDeviceToHost) != cudaSuccess)
    return CUSPARSE_STATUS_EXECUTION_FAILED;
  std::vector<int> next(colptr.begin(), colptr.end() - 1), rows((size_t)nnzb);
  for (int i = 0; i < mb; ++i)
    for (int k = o[(size_t)i]; k < o[(size_t)i + 1]; ++k) {
      const int dst = next[(size_t)c[(size_t)k]]++;
      rows[(size_t)dst] = i + b;
      if (!in.empty()) std::memcpy(&out[(size_t)dst * bs], &in[(size_t)k * bs], bs);
    }
  for (int& x : colptr) x += b;
  const bool ok = put_ints(bscColPtr, colptr) && put_ints(bscRow, rows) &&
                  (out.empty() || cudaMemcpy(bscVal, out.data(), out.size(), cudaMemcpyHostToDevice) == cudaSuccess);
  return ok ? CUSPARSE_STATUS_SUCCESS : CUSPARSE_STATUS_EXECUTION_FAILED;
}
}  // namespace

VGPU_EXPORT cusparseStatus_t cusparseXgebsr2gebsrNnz(cusparseHandle_t h, cusparseDirection_t dir, int mb, int nb,
                                                     int nnzb, const cusparseMatDescr_t dA, const int* offA,
                                                     const int* colA, int rbdA, int cbdA, const cusparseMatDescr_t dC,
                                                     int* offC, int rbdC, int cbdC, int* total, void*) {
  return gebsr2gebsr<float>(h, dir, mb, nb, nnzb, dA, nullptr, offA, colA, rbdA, cbdA, dC, nullptr, offC, nullptr,
                            rbdC, cbdC, total);
}

#define VGPU_GEBSR(P, T)                                                                                            \
  VGPU_EXPORT cusparseStatus_t cusparse##P##gebsr2gebsr_bufferSize(                                                 \
      cusparseHandle_t h, cusparseDirection_t, int, int, int, const cusparseMatDescr_t, const T*, const int*,       \
      const int*, int, int, int, int, int* bytes) {                                                                 \
    return token_size(h, bytes);                                                                                    \
  }                                                                                                                 \
  VGPU_EXPORT cusparseStatus_t cusparse##P##gebsr2gebsr_bufferSizeExt(                                              \
      cusparseHandle_t h, cusparseDirection_t, int, int, int, const cusparseMatDescr_t, const T*, const int*,       \
      const int*, int, int, int, int, size_t* bytes) {                                                              \
    return token_size(h, bytes);                                                                                    \
  }                                                                                                                 \
  VGPU_EXPORT cusparseStatus_t cusparse##P##gebsr2gebsr(                                                            \
      cusparseHandle_t h, cusparseDirection_t dir, int mb, int nb, int nnzb, const cusparseMatDescr_t dA,          \
      const T* valA, const int* offA, const int* colA, int rbdA, int cbdA, const cusparseMatDescr_t dC, T* valC,   \
      int* offC, int* colC, int rbdC, int cbdC, void*) {                                                            \
    if (!valC) return CUSPARSE_STATUS_INVALID_VALUE;                                                                \
    return gebsr2gebsr<T>(h, dir, mb, nb, nnzb, dA, valA, offA, colA, rbdA, cbdA, dC, valC, offC, colC, rbdC,      \
                          cbdC, nullptr);                                                                           \
  }                                                                                                                 \
  VGPU_EXPORT cusparseStatus_t cusparse##P##gebsr2gebsc_bufferSize(cusparseHandle_t h, int, int, int, const T*,    \
                                                                   const int*, const int*, int, int, int* bytes) { \
    return token_size(h, bytes);                                                                                    \
  }                                                                                                                 \
  VGPU_EXPORT cusparseStatus_t cusparse##P##gebsr2gebsc_bufferSizeExt(cusparseHandle_t h, int, int, int, const T*, \
                                                                      const int*, const int*, int, int,            \
                                                                      size_t* bytes) {                             \
    return token_size(h, bytes);                                                                                    \
  }                                                                                                                 \
  VGPU_EXPORT cusparseStatus_t cusparse##P##gebsr2gebsc(                                                            \
      cusparseHandle_t h, int mb, int nb, int nnzb, const T* val, const int* off, const int* col, int rbd,         \
      int cbd, T* bscVal, int* bscRow, int* bscColPtr, cusparseAction_t action, cusparseIndexBase_t base, void*) { \
    return gebsr2gebsc<T>(h, mb, nb, nnzb, val, off, col, rbd, cbd, bscVal, bscRow, bscColPtr, action, base);       \
  }
VGPU_GEBSR(S, float)
VGPU_GEBSR(D, double)
VGPU_GEBSR(C, cuComplex)
VGPU_GEBSR(Z, cuDoubleComplex)
#undef VGPU_GEBSR

/* ---- csrcolor ---- */

namespace {
struct ColorInfo {};

template <class T>
cusparseStatus_t csrcolor(cusparseHandle_t h, int m, int nnz, cusparseMatDescr_t d, const int* off, const int* col,
                          const void* fraction, int* ncolors, int* coloring, int* reordering) {
  if (!known(h)) return CUSPARSE_STATUS_NOT_INITIALIZED;
  if (m < 0 || nnz < 0 || !descr(d) || !fraction || !ncolors || !coloring) return CUSPARSE_STATUS_INVALID_VALUE;
  std::vector<int> o, c;
  if (!read_legacy_csr(m, nnz, d, off, col, &o, &c)) return CUSPARSE_STATUS_INVALID_VALUE;
  // The adjacency graph, made symmetric, without self-loops.
  std::vector<std::vector<int>> adj((size_t)m);
  for (int i = 0; i < m; ++i)
    for (int e = o[(size_t)i]; e < o[(size_t)i + 1] && e < nnz; ++e) {
      const int j = c[(size_t)e];
      if (j < 0 || j >= m || j == i) continue;
      adj[(size_t)i].push_back(j);
      adj[(size_t)j].push_back(i);
    }
  // Greedy, in node order: the smallest color no colored neighbour has. It
  // colors every node, which is at least the fraction asked for.
  std::vector<int> color((size_t)m, -1);
  int used = 0;
  for (int i = 0; i < m; ++i) {
    std::vector<char> taken((size_t)used + 1, 0);
    for (int j : adj[(size_t)i])
      if (color[(size_t)j] >= 0 && color[(size_t)j] <= used) taken[(size_t)color[(size_t)j]] = 1;
    int k = 0;
    while (taken[(size_t)k]) ++k;
    color[(size_t)i] = k;
    used = std::max(used, k + 1);
  }
  if (!put_ints(coloring, color)) return CUSPARSE_STATUS_EXECUTION_FAILED;
  if (reordering) {
    std::vector<int> order((size_t)m);
    for (int i = 0; i < m; ++i) order[(size_t)i] = i;
    std::stable_sort(order.begin(), order.end(), [&](int x, int y) { return color[(size_t)x] < color[(size_t)y]; });
    if (!put_ints(reordering, order)) return CUSPARSE_STATUS_EXECUTION_FAILED;
  }
  put_count(h, ncolors, used);
  return CUSPARSE_STATUS_SUCCESS;
}
}  // namespace

VGPU_EXPORT cusparseStatus_t cusparseCreateColorInfo(cusparseColorInfo_t* info) {
  if (!info) return CUSPARSE_STATUS_INVALID_VALUE;
  *info = reinterpret_cast<cusparseColorInfo_t>(track(new ColorInfo()));
  return CUSPARSE_STATUS_SUCCESS;
}
VGPU_EXPORT cusparseStatus_t cusparseDestroyColorInfo(cusparseColorInfo_t info) {
  if (!known(info)) return CUSPARSE_STATUS_INVALID_VALUE;
  untrack(info);
  delete reinterpret_cast<ColorInfo*>(info);
  return CUSPARSE_STATUS_SUCCESS;
}

// The fraction is single precision for S and C, double for D and Z.
#define VGPU_COLOR(P, T, F)                                                                                         \
  VGPU_EXPORT cusparseStatus_t cusparse##P##csrcolor(cusparseHandle_t h, int m, int nnz, const cusparseMatDescr_t d,\
                                                     const T*, const int* off, const int* col,                     \
                                                     const F* fraction, int* ncolors, int* coloring,               \
                                                     int* reordering, const cusparseColorInfo_t) {                 \
    return csrcolor<T>(h, m, nnz, d, off, col, fraction, ncolors, coloring, reordering);                            \
  }
VGPU_COLOR(S, float, float)
VGPU_COLOR(D, double, double)
VGPU_COLOR(C, cuComplex, float)
VGPU_COLOR(Z, cuDoubleComplex, double)
#undef VGPU_COLOR
