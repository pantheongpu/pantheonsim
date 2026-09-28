// The cuSPARSE entry points torch.sparse reaches, checked against dense
// arithmetic done here on the host.
//
//   coalesce, to_sparse_csr   Xcoo2csr, XcoosortByRow, Xcsrsort, Xcsr2coo,
//                             CreateIdentityPermutation
//   sparse @ dense, bmm       SpMM, over a strided batch, and in half precision
//   sparse @ sparse           SpGEMM (workEstimation, compute, copy)
//   sparse + sparse           csrgeam2, with the device pointer mode
//   sampled_addmm             SDDMM
//   triangular_solve          SpSV and SpSM, lower and upper, unit diagonal,
//                             transposed
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cusparse.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

static int failures = 0;

static void check(bool ok, const char* what, double err) {
  std::printf("%-4s %s (%.2e)\n", ok ? "ok" : "FAIL", what, err);
  if (!ok) ++failures;
}

#define CK(x)                                                                    \
  do {                                                                           \
    const int r_ = (int)(x);                                                     \
    if (r_ != 0) {                                                               \
      std::printf("FAIL %s returned %d\n", #x, r_);                             \
      ++failures;                                                                \
      return;                                                                    \
    }                                                                            \
  } while (0)

template <class T> T* upload(const std::vector<T>& h) {
  T* d = nullptr;
  cudaMalloc(&d, std::max<size_t>(1, h.size()) * sizeof(T));
  cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
  return d;
}
template <class T> std::vector<T> download(const T* d, size_t n) {
  std::vector<T> h(n);
  cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost);
  return h;
}

// A small sparse matrix, row-major dense on the host, with a CSR copy.
struct Sparse {
  int m, n;
  std::vector<double> dense;  // row-major
  std::vector<int> off, col;
  std::vector<float> val;
};
static Sparse make_sparse(int m, int n, int seed, double density = 0.4) {
  Sparse s{m, n, std::vector<double>((size_t)m * n, 0.0), {0}, {}, {}};
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      const double h = std::fmod(std::fabs(std::sin(12.9898 * (i + 1) + 78.233 * (j + 1) + seed)) * 43758.5453, 1.0);
      if (h < density) {
        const float v = (float)(std::cos(0.7 * i - 0.3 * j + seed) * 2.0);
        s.dense[(size_t)i * n + j] = v;
        s.col.push_back(j);
        s.val.push_back(v);
      }
    }
    s.off.push_back((int)s.col.size());
  }
  return s;
}
static double dense_val(int i, int j, int seed) { return std::sin(0.31 * i + 0.17 * j + seed); }

static double max_rel(const std::vector<double>& ref, const std::vector<double>& got) {
  double err = 0, scale = 1e-30;
  for (size_t i = 0; i < ref.size(); ++i) {
    err = std::fmax(err, std::fabs(ref[i] - got[i]));
    scale = std::fmax(scale, std::fabs(ref[i]));
  }
  return err / scale;
}

static cusparseHandle_t h;

// Unsorted COO -> sorted COO -> CSR -> COO, with the permutation carried along.
static void conversions() {
  const std::vector<int> rows = {2, 0, 3, 0, 2, 1, 3}, cols = {1, 3, 0, 1, 0, 2, 2};
  const int nnz = (int)rows.size(), m = 4, n = 4;
  int* dr = upload(rows);
  int* dc = upload(cols);
  int* dp = upload(std::vector<int>(nnz, -1));
  CK(cusparseCreateIdentityPermutation(h, nnz, dp));
  size_t bytes = 0;
  CK(cusparseXcoosort_bufferSizeExt(h, m, n, nnz, dr, dc, &bytes));
  void* buf = nullptr;
  cudaMalloc(&buf, std::max<size_t>(bytes, 1));
  CK(cusparseXcoosortByRow(h, m, n, nnz, dr, dc, dp, buf));
  const auto sr = download(dr, nnz), sc = download(dc, nnz), p = download(dp, nnz);
  bool sorted = true, perm = true;
  for (int i = 0; i < nnz; ++i) {
    if (i && (sr[i - 1] > sr[i] || (sr[i - 1] == sr[i] && sc[i - 1] >= sc[i]))) sorted = false;
    if (rows[p[i]] != sr[i] || cols[p[i]] != sc[i]) perm = false;
  }
  check(sorted && perm, "XcoosortByRow sorts by (row, column) and its permutation maps back", 0);

  int* doff = upload(std::vector<int>(m + 1, -1));
  CK(cusparseXcoo2csr(h, dr, nnz, m, doff, CUSPARSE_INDEX_BASE_ZERO));
  const auto off = download(doff, m + 1);
  check(off == std::vector<int>({0, 2, 3, 5, 7}), "Xcoo2csr counts each row", 0);

  int* dback = upload(std::vector<int>(nnz, -1));
  CK(cusparseXcsr2coo(h, doff, nnz, m, dback, CUSPARSE_INDEX_BASE_ZERO));
  check(download(dback, nnz) == sr, "Xcsr2coo inverts it", 0);

  // Unsorted columns within rows, one-based: Xcsrsort orders each row.
  const std::vector<int> off1 = {1, 3, 4, 7}, col1 = {3, 1, 2, 3, 1, 2};
  int* do1 = upload(off1);
  int* dc1 = upload(col1);
  int* dp1 = upload(std::vector<int>{0, 1, 2, 3, 4, 5});
  cusparseMatDescr_t descr;
  CK(cusparseCreateMatDescr(&descr));
  CK(cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ONE));
  CK(cusparseXcsrsort_bufferSizeExt(h, 3, 3, 6, do1, dc1, &bytes));
  void* buf2 = nullptr;
  cudaMalloc(&buf2, std::max<size_t>(bytes, 1));
  CK(cusparseXcsrsort(h, 3, 3, 6, descr, do1, dc1, dp1, buf2));
  cudaFree(buf2);
  check(download(dc1, 6) == std::vector<int>({1, 3, 2, 1, 2, 3}) &&
            download(dp1, 6) == std::vector<int>({1, 0, 2, 4, 5, 3}),
        "Xcsrsort orders the columns in each one-based row, with the permutation", 0);
  cusparseDestroyMatDescr(descr);
  for (void* x : {(void*)dr, (void*)dc, (void*)dp, buf, (void*)doff, (void*)dback, (void*)do1, (void*)dc1, (void*)dp1})
    cudaFree(x);
}

// C[b] = A[b] * B[b] over a strided batch of CSR matrices sharing one nnz,
// then the same product in half precision. Each member has its own row
// offsets: NVIDIA's cuSPARSE 13.0 used member 0's for all of them, which 13.2
// fixed; this checks the documented behavior, which 13.2 has.
static void spmm_batched() {
  const int m = 5, k = 6, n = 3, batch = 3;
  std::vector<Sparse> as;
  int nnz = 0;
  for (int b = 0; b < batch; ++b) {
    as.push_back(make_sparse(m, k, b));
    nnz = std::max(nnz, (int)as[b].val.size());
  }
  // Pad each member to the same nnz with explicit zeros in its last row, as a
  // batched tensor's members must all hold the same count.
  std::vector<int> off, col;
  std::vector<float> val;
  for (auto& a : as) {
    while ((int)a.val.size() < nnz) { a.col.push_back(k - 1); a.val.push_back(0.0f); a.off.back()++; }
    // Columns in the last row must stay sorted: re-sort that row.
    std::vector<std::pair<int, float>> last;
    for (int e = a.off[m - 1]; e < a.off[m]; ++e) last.push_back({a.col[e], a.val[e]});
    std::stable_sort(last.begin(), last.end(), [](auto x, auto y) { return x.first < y.first; });
    for (int e = a.off[m - 1], i = 0; e < a.off[m]; ++e, ++i) { a.col[e] = last[i].first; a.val[e] = last[i].second; }
    off.insert(off.end(), a.off.begin(), a.off.end());
    col.insert(col.end(), a.col.begin(), a.col.end());
    val.insert(val.end(), a.val.begin(), a.val.end());
  }
  std::vector<float> bh((size_t)k * n * batch);  // column-major, ld = k
  for (int b = 0; b < batch; ++b)
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < k; ++i) bh[(size_t)b * k * n + (size_t)j * k + i] = (float)dense_val(i, j, b);
  int* doff = upload(off);
  int* dcol = upload(col);
  float* dval = upload(val);
  float* dB = upload(bh);
  float* dC = upload(std::vector<float>((size_t)m * n * batch, 0.0f));
  cusparseSpMatDescr_t A;
  cusparseDnMatDescr_t B, C;
  CK(cusparseCreateCsr(&A, m, k, nnz, doff, dcol, dval, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                       CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
  CK(cusparseCsrSetStridedBatch(A, batch, m + 1, nnz));
  CK(cusparseCreateDnMat(&B, k, n, k, dB, CUDA_R_32F, CUSPARSE_ORDER_COL));
  CK(cusparseDnMatSetStridedBatch(B, batch, (int64_t)k * n));
  CK(cusparseCreateDnMat(&C, m, n, m, dC, CUDA_R_32F, CUSPARSE_ORDER_COL));
  CK(cusparseDnMatSetStridedBatch(C, batch, (int64_t)m * n));
  const float one = 1.0f, zero = 0.0f;
  size_t bytes = 0;
  CK(cusparseSpMM_bufferSize(h, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, A, B,
                             &zero, C, CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG2, &bytes));
  void* work = nullptr;
  cudaMalloc(&work, std::max<size_t>(bytes, 16));
  CK(cusparseSpMM(h, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, A, B, &zero, C,
                  CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG2, work));
  cudaFree(work);
  const auto ch = download(dC, (size_t)m * n * batch);
  std::vector<double> ref, got(ch.begin(), ch.end());
  for (int b = 0; b < batch; ++b)
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < m; ++i) {
        double s = 0;
        for (int x = 0; x < k; ++x) s += as[b].dense[(size_t)i * k + x] * bh[(size_t)b * k * n + (size_t)j * k + x];
        ref.push_back(s);
      }
  double e = max_rel(ref, got);
  check(e < 1e-6, "SpMM over a strided batch of CSR matrices matches the dense product", e);
  cusparseDestroySpMat(A);
  cusparseDestroyDnMat(B);
  cusparseDestroyDnMat(C);

  {  // Member 0 again, with half values and operands and single-precision compute.
    std::vector<__half> hv, hb, hc((size_t)m * n);
    for (int e2 = 0; e2 < as[0].off[m]; ++e2) hv.push_back(__float2half(as[0].val[e2]));
    for (int i = 0; i < k * n; ++i) hb.push_back(__float2half(bh[i]));
    __half* dhv = upload(hv);
    __half* dhb = upload(hb);
    __half* dhc = upload(hc);
    CK(cusparseCreateCsr(&A, m, k, as[0].off[m], doff, dcol, dhv, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                         CUSPARSE_INDEX_BASE_ZERO, CUDA_R_16F));
    CK(cusparseCreateDnMat(&B, k, n, k, dhb, CUDA_R_16F, CUSPARSE_ORDER_COL));
    CK(cusparseCreateDnMat(&C, m, n, m, dhc, CUDA_R_16F, CUSPARSE_ORDER_COL));
    CK(cusparseSpMM_bufferSize(h, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, A, B,
                               &zero, C, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &bytes));
    void* work = nullptr;
    cudaMalloc(&work, std::max<size_t>(bytes, 16));
    CK(cusparseSpMM(h, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, A, B, &zero, C,
                    CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, work));
    cudaFree(work);
    const auto out = download(dhc, (size_t)m * n);
    std::vector<double> r0(ref.begin(), ref.begin() + m * n), g0;
    for (const __half& x : out) g0.push_back(__half2float(x));
    e = max_rel(r0, g0);
    check(e < 5e-3, "SpMM with half values and single-precision compute", e);
    cusparseDestroySpMat(A);
    cusparseDestroyDnMat(B);
    cusparseDestroyDnMat(C);
    cudaFree(dhv);
    cudaFree(dhb);
    cudaFree(dhc);
  }
  for (void* x : {(void*)doff, (void*)dcol, (void*)dval, (void*)dB, (void*)dC}) cudaFree(x);
}

// C = alpha * A * B for two sparse matrices: C's size is learned from
// _compute, the caller allocates, and _copy fills it in.
static void spgemm() {
  const Sparse a = make_sparse(5, 7, 11), b = make_sparse(7, 4, 12);
  int* dao = upload(a.off);
  int* dac = upload(a.col);
  float* dav = upload(a.val);
  int* dbo = upload(b.off);
  int* dbc = upload(b.col);
  float* dbv = upload(b.val);
  int* dco = upload(std::vector<int>(a.m + 1, -1));
  cusparseSpMatDescr_t A, B, C;
  CK(cusparseCreateCsr(&A, a.m, a.n, (int64_t)a.val.size(), dao, dac, dav, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                       CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
  CK(cusparseCreateCsr(&B, b.m, b.n, (int64_t)b.val.size(), dbo, dbc, dbv, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                       CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
  CK(cusparseCreateCsr(&C, a.m, b.n, 0, dco, nullptr, nullptr, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                       CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
  cusparseSpGEMMDescr_t g;
  CK(cusparseSpGEMM_createDescr(&g));
  const float alpha = 1.5f, beta = 0.0f;
  const auto op = CUSPARSE_OPERATION_NON_TRANSPOSE;
  size_t s1 = 0, s2 = 0;
  CK(cusparseSpGEMM_workEstimation(h, op, op, &alpha, A, B, &beta, C, CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, g, &s1,
                                   nullptr));
  void *b1 = nullptr, *b2 = nullptr;
  cudaMalloc(&b1, s1);
  CK(cusparseSpGEMM_workEstimation(h, op, op, &alpha, A, B, &beta, C, CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, g, &s1,
                                   b1));
  CK(cusparseSpGEMM_compute(h, op, op, &alpha, A, B, &beta, C, CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, g, &s2, nullptr));
  cudaMalloc(&b2, s2);
  CK(cusparseSpGEMM_compute(h, op, op, &alpha, A, B, &beta, C, CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, g, &s2, b2));
  int64_t rows, cols, nnz;
  CK(cusparseSpMatGetSize(C, &rows, &cols, &nnz));
  int* dcc = upload(std::vector<int>((size_t)nnz));
  float* dcv = upload(std::vector<float>((size_t)nnz));
  CK(cusparseCsrSetPointers(C, dco, dcc, dcv));
  CK(cusparseSpGEMM_copy(h, op, op, &alpha, A, B, &beta, C, CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, g));
  const auto co = download(dco, a.m + 1), cc = download(dcc, nnz);
  const auto cv = download(dcv, nnz);
  std::vector<double> ref((size_t)a.m * b.n, 0.0), got((size_t)a.m * b.n, 0.0);
  for (int i = 0; i < a.m; ++i)
    for (int j = 0; j < b.n; ++j)
      for (int x = 0; x < a.n; ++x) ref[(size_t)i * b.n + j] += alpha * a.dense[(size_t)i * a.n + x] * b.dense[(size_t)x * b.n + j];
  bool sorted = co[0] == 0 && co[a.m] == nnz;
  for (int i = 0; i < a.m; ++i)
    for (int e = co[i]; e < co[i + 1]; ++e) {
      got[(size_t)i * b.n + cc[e]] += cv[e];
      if (e > co[i] && cc[e - 1] >= cc[e]) sorted = false;
    }
  const double err = max_rel(ref, got);
  check(err < 1e-6 && sorted && rows == a.m && cols == b.n, "SpGEMM matches the dense product, as sorted CSR", err);
  cusparseSpGEMM_destroyDescr(g);
  cusparseDestroySpMat(A);
  cusparseDestroySpMat(B);
  cusparseDestroySpMat(C);
  for (void* x : {(void*)dao, (void*)dac, (void*)dav, (void*)dbo, (void*)dbc, (void*)dbv, (void*)dco, (void*)dcc,
                  (void*)dcv, b1, b2})
    cudaFree(x);
}

// C = alpha * A + beta * B, with alpha, beta and the nnz count in device memory.
static void geam() {
  const Sparse a = make_sparse(6, 5, 21), b = make_sparse(6, 5, 22);
  const int m = 6, n = 5, na = (int)a.val.size(), nb = (int)b.val.size();
  int* dao = upload(a.off);
  int* dac = upload(a.col);
  float* dav = upload(a.val);
  int* dbo = upload(b.off);
  int* dbc = upload(b.col);
  float* dbv = upload(b.val);
  int* dco = upload(std::vector<int>(m + 1));
  float* dab = upload(std::vector<float>{2.0f, -0.5f});
  int* dnnz = upload(std::vector<int>{-1});
  cusparseMatDescr_t d;
  CK(cusparseCreateMatDescr(&d));
  CK(cusparseSetPointerMode(h, CUSPARSE_POINTER_MODE_DEVICE));
  size_t bytes = 0;
  CK(cusparseScsrgeam2_bufferSizeExt(h, m, n, dab, d, na, dav, dao, dac, dab + 1, d, nb, dbv, dbo, dbc, d, nullptr,
                                     dco, nullptr, &bytes));
  void* buf = nullptr;
  cudaMalloc(&buf, bytes);
  CK(cusparseXcsrgeam2Nnz(h, m, n, d, na, dao, dac, d, nb, dbo, dbc, d, dco, dnnz, buf));
  const int nnz = download(dnnz, 1)[0];
  int* dcc = upload(std::vector<int>((size_t)std::max(nnz, 0)));
  float* dcv = upload(std::vector<float>((size_t)std::max(nnz, 0)));
  CK(cusparseScsrgeam2(h, m, n, dab, d, na, dav, dao, dac, dab + 1, d, nb, dbv, dbo, dbc, d, dcv, dco, dcc, buf));
  CK(cusparseSetPointerMode(h, CUSPARSE_POINTER_MODE_HOST));
  const auto co = download(dco, m + 1), cc = download(dcc, nnz);
  const auto cv = download(dcv, nnz);
  std::vector<double> ref((size_t)m * n), got((size_t)m * n, 0.0);
  int pattern = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    ref[i] = 2.0 * a.dense[i] - 0.5 * b.dense[i];
    pattern += a.dense[i] != 0 || b.dense[i] != 0;
  }
  for (int i = 0; i < m; ++i)
    for (int e = co[i]; e < co[i + 1]; ++e) got[(size_t)i * n + cc[e]] += cv[e];
  const double err = max_rel(ref, got);
  check(err < 1e-6 && nnz == pattern && co[m] == nnz, "csrgeam2 (device pointer mode) gives alpha A + beta B on the union pattern", err);
  cusparseDestroyMatDescr(d);
  for (void* x : {(void*)dao, (void*)dac, (void*)dav, (void*)dbo, (void*)dbc, (void*)dbv, (void*)dco, (void*)dab,
                  (void*)dnnz, buf, (void*)dcc, (void*)dcv})
    cudaFree(x);
}

// C = alpha * (A B^T) on C's pattern + beta * C, A and B dense row-major.
static void sddmm() {
  const int m = 4, n = 5, k = 3;
  const Sparse c = make_sparse(m, n, 31, 0.5);
  std::vector<float> ah((size_t)m * k), bh((size_t)n * k);
  for (int i = 0; i < m * k; ++i) ah[i] = (float)dense_val(i / k, i % k, 1);
  for (int i = 0; i < n * k; ++i) bh[i] = (float)dense_val(i / k, i % k, 2);
  int* dco = upload(c.off);
  int* dcc = upload(c.col);
  float* dcv = upload(c.val);
  float* dA = upload(ah);
  float* dB = upload(bh);
  cusparseDnMatDescr_t A, B;
  cusparseSpMatDescr_t C;
  CK(cusparseCreateDnMat(&A, m, k, k, dA, CUDA_R_32F, CUSPARSE_ORDER_ROW));
  CK(cusparseCreateDnMat(&B, n, k, k, dB, CUDA_R_32F, CUSPARSE_ORDER_ROW));
  CK(cusparseCreateCsr(&C, m, n, (int64_t)c.val.size(), dco, dcc, dcv, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                       CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
  const float alpha = 0.75f, beta = 2.0f;
  size_t bytes = 0;
  const auto N = CUSPARSE_OPERATION_NON_TRANSPOSE, T = CUSPARSE_OPERATION_TRANSPOSE;
  CK(cusparseSDDMM_bufferSize(h, N, T, &alpha, A, B, &beta, C, CUDA_R_32F, CUSPARSE_SDDMM_ALG_DEFAULT, &bytes));
  void* work = nullptr;
  cudaMalloc(&work, std::max<size_t>(bytes, 16));  // NVIDIA's refuses a NULL buffer even when it asked for 0
  CK(cusparseSDDMM_preprocess(h, N, T, &alpha, A, B, &beta, C, CUDA_R_32F, CUSPARSE_SDDMM_ALG_DEFAULT, work));
  CK(cusparseSDDMM(h, N, T, &alpha, A, B, &beta, C, CUDA_R_32F, CUSPARSE_SDDMM_ALG_DEFAULT, work));
  cudaFree(work);
  const auto out = download(dcv, c.val.size());
  std::vector<double> ref, got(out.begin(), out.end());
  for (int i = 0; i < m; ++i)
    for (int e = c.off[i]; e < c.off[i + 1]; ++e) {
      double dot = 0;
      for (int x = 0; x < k; ++x) dot += ah[(size_t)i * k + x] * bh[(size_t)c.col[e] * k + x];
      ref.push_back(alpha * dot + beta * c.val[e]);
    }
  const double err = max_rel(ref, got);
  check(err < 1e-6, "SDDMM gives alpha (A B^T) on C's pattern + beta C", err);
  cusparseDestroyDnMat(A);
  cusparseDestroyDnMat(B);
  cusparseDestroySpMat(C);
  for (void* x : {(void*)dco, (void*)dcc, (void*)dcv, (void*)dA, (void*)dB}) cudaFree(x);
}

// A full square sparse matrix with a strong diagonal, so either triangle is
// well conditioned; the solves must read only the triangle they are told to.
static void triangular() {
  const int n = 6, nrhs = 2;
  Sparse a = make_sparse(n, n, 41, 0.6);
  a.off = {0};
  a.col.clear();
  a.val.clear();
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      double& v = a.dense[(size_t)i * n + j];
      if (i == j) v = 4.0 + i;
      if (v != 0.0) { a.col.push_back(j); a.val.push_back((float)v); }
    }
    a.off.push_back((int)a.col.size());
  }
  int* dao = upload(a.off);
  int* dac = upload(a.col);
  float* dav = upload(a.val);
  cusparseSpMatDescr_t A;
  CK(cusparseCreateCsr(&A, n, n, (int64_t)a.val.size(), dao, dac, dav, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                       CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
  std::vector<float> xh(n);
  for (int i = 0; i < n; ++i) xh[i] = (float)dense_val(i, 0, 5);
  float* dx = upload(xh);
  float* dy = upload(std::vector<float>(n));
  cusparseDnVecDescr_t X, Y;
  CK(cusparseCreateDnVec(&X, n, dx, CUDA_R_32F));
  CK(cusparseCreateDnVec(&Y, n, dy, CUDA_R_32F));
  const float alpha = 2.0f;

  struct Case { cusparseFillMode_t fill; cusparseDiagType_t diag; cusparseOperation_t op; const char* what; };
  const Case cases[] = {
      {CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, CUSPARSE_OPERATION_NON_TRANSPOSE,
       "SpSV lower: op(A) y = alpha x holds on A's lower triangle"},
      {CUSPARSE_FILL_MODE_UPPER, CUSPARSE_DIAG_TYPE_UNIT, CUSPARSE_OPERATION_NON_TRANSPOSE,
       "SpSV upper, unit diagonal: the stored diagonal is ignored"},
      {CUSPARSE_FILL_MODE_LOWER, CUSPARSE_DIAG_TYPE_NON_UNIT, CUSPARSE_OPERATION_TRANSPOSE,
       "SpSV lower, transposed: solves with the upper triangle A^T"},
  };
  for (const Case& c : cases) {
    CK(cusparseSpMatSetAttribute(A, CUSPARSE_SPMAT_FILL_MODE, (void*)&c.fill, sizeof(c.fill)));
    CK(cusparseSpMatSetAttribute(A, CUSPARSE_SPMAT_DIAG_TYPE, (void*)&c.diag, sizeof(c.diag)));
    cusparseSpSVDescr_t d;
    CK(cusparseSpSV_createDescr(&d));
    size_t bytes = 0;
    CK(cusparseSpSV_bufferSize(h, c.op, &alpha, A, X, Y, CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT, d, &bytes));
    void* buf = nullptr;
    cudaMalloc(&buf, std::max<size_t>(bytes, 1));
    CK(cusparseSpSV_analysis(h, c.op, &alpha, A, X, Y, CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT, d, buf));
    CK(cusparseSpSV_solve(h, c.op, &alpha, A, X, Y, CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT, d));
    const auto y = download(dy, n);
    // Residual of T y = alpha x, T built here from the same rules.
    std::vector<double> ref, got;
    for (int i = 0; i < n; ++i) {
      double s = 0;
      for (int j = 0; j < n; ++j) {
        const int r = c.op == CUSPARSE_OPERATION_TRANSPOSE ? j : i, col = c.op == CUSPARSE_OPERATION_TRANSPOSE ? i : j;
        const bool in = c.fill == CUSPARSE_FILL_MODE_LOWER ? col <= r : col >= r;
        double v = in ? a.dense[(size_t)r * n + col] : 0.0;
        if (r == col && c.diag == CUSPARSE_DIAG_TYPE_UNIT) v = 1.0;
        s += v * y[j];
      }
      ref.push_back(alpha * xh[i]);
      got.push_back(s);
    }
    const double err = max_rel(ref, got);
    check(err < 1e-5, c.what, err);
    cusparseSpSV_destroyDescr(d);
    cudaFree(buf);
  }

  {  // SpSM: two right-hand sides at once, upper, non-unit.
    const cusparseFillMode_t fill = CUSPARSE_FILL_MODE_UPPER;
    const cusparseDiagType_t diag = CUSPARSE_DIAG_TYPE_NON_UNIT;
    CK(cusparseSpMatSetAttribute(A, CUSPARSE_SPMAT_FILL_MODE, (void*)&fill, sizeof(fill)));
    CK(cusparseSpMatSetAttribute(A, CUSPARSE_SPMAT_DIAG_TYPE, (void*)&diag, sizeof(diag)));
    std::vector<float> bh((size_t)n * nrhs);
    for (int i = 0; i < n * nrhs; ++i) bh[i] = (float)dense_val(i % n, i / n, 7);
    float* dB = upload(bh);
    float* dC = upload(std::vector<float>((size_t)n * nrhs));
    cusparseDnMatDescr_t B, C;
    CK(cusparseCreateDnMat(&B, n, nrhs, n, dB, CUDA_R_32F, CUSPARSE_ORDER_COL));
    CK(cusparseCreateDnMat(&C, n, nrhs, n, dC, CUDA_R_32F, CUSPARSE_ORDER_COL));
    cusparseSpSMDescr_t d;
    CK(cusparseSpSM_createDescr(&d));
    const auto N = CUSPARSE_OPERATION_NON_TRANSPOSE;
    size_t bytes = 0;
    CK(cusparseSpSM_bufferSize(h, N, N, &alpha, A, B, C, CUDA_R_32F, CUSPARSE_SPSM_ALG_DEFAULT, d, &bytes));
    void* buf = nullptr;
    cudaMalloc(&buf, std::max<size_t>(bytes, 1));
    CK(cusparseSpSM_analysis(h, N, N, &alpha, A, B, C, CUDA_R_32F, CUSPARSE_SPSM_ALG_DEFAULT, d, buf));
    CK(cusparseSpSM_solve(h, N, N, &alpha, A, B, C, CUDA_R_32F, CUSPARSE_SPSM_ALG_DEFAULT, d));
    const auto ch = download(dC, (size_t)n * nrhs);
    std::vector<double> ref, got;
    for (int r = 0; r < nrhs; ++r)
      for (int i = 0; i < n; ++i) {
        double s = 0;
        for (int j = i; j < n; ++j) s += a.dense[(size_t)i * n + j] * ch[(size_t)r * n + j];
        ref.push_back(alpha * bh[(size_t)r * n + i]);
        got.push_back(s);
      }
    const double err = max_rel(ref, got);
    check(err < 1e-5, "SpSM upper: op(A) C = alpha B for every column", err);
    cusparseSpSM_destroyDescr(d);
    cusparseDestroyDnMat(B);
    cusparseDestroyDnMat(C);
    cudaFree(buf);
    cudaFree(dB);
    cudaFree(dC);
  }
  cusparseDestroyDnVec(X);
  cusparseDestroyDnVec(Y);
  cusparseDestroySpMat(A);
  for (void* x : {(void*)dao, (void*)dac, (void*)dav, (void*)dx, (void*)dy}) cudaFree(x);
}

int main() {
  if (cusparseCreate(&h)) {
    std::printf("FAIL: cusparseCreate\n");
    return 1;
  }
  conversions();
  spmm_batched();
  spgemm();
  geam();
  sddmm();
  triangular();
  cusparseDestroy(h);
  std::printf(failures ? "FAIL: %d sparse checks\n" : "PASS: every sparse check\n", failures);
  return failures ? 1 : 0;
}
