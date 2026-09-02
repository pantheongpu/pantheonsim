// Differential conformance for the cuSPARSE shim: SpMV and SpMM over CSR and
// COO, both index bases, transposed and not, both storage orders for the dense
// operands, alpha/beta blending, and conversion in both directions.
#include <cusparse.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>

#define CS(x) do { cusparseStatus_t s_ = (x); if (s_ != CUSPARSE_STATUS_SUCCESS) { \
  printf("%s -> %d\n", #x, (int)s_); return; } } while (0)

// A fixed 5x6 sparse matrix, given once as coordinates and fed to the library
// in whichever format each case wants.
static const int kRows = 5, kCols = 6, kNnz = 11;
static const int kR[kNnz] = {0, 0, 1, 1, 2, 2, 2, 3, 4, 4, 4};
static const int kC[kNnz] = {0, 3, 1, 5, 0, 2, 4, 3, 1, 2, 5};
static const float kV[kNnz] = {1.5f, -2.0f, 3.25f, 0.5f, -1.25f, 4.0f,
                               2.5f, -3.5f, 1.75f, -0.75f, 6.0f};

template <class T> static T* up(const std::vector<T>& h) {
  T* d = nullptr;
  cudaMalloc(&d, h.size() * sizeof(T));
  cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
  return d;
}
static std::vector<float> down(const float* d, size_t n) {
  std::vector<float> h(n);
  cudaMemcpy(h.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost);
  return h;
}
static void dump(const char* tag, const std::vector<float>& v) {
  double sum = 0, abs = 0;
  for (float x : v) { sum += x; abs += std::fabs(x); }
  printf("%-34s n=%zu sum=%.4f abs=%.4f first=%.4f last=%.4f\n", tag, v.size(), sum, abs,
         v.empty() ? 0.f : v[0], v.empty() ? 0.f : v.back());
}

// CSR row offsets for the fixed matrix, in the requested index base.
static std::vector<int> csr_offsets(int base) {
  std::vector<int> off(kRows + 1, base);
  for (int i = 0; i < kNnz; ++i) off[kR[i] + 1] = i + 1 + base;
  for (int r = 1; r <= kRows; ++r) off[r] = off[r] < off[r - 1] ? off[r - 1] : off[r];
  return off;
}

static void spmv_case(cusparseHandle_t h, const char* tag, bool coo, int base,
                      cusparseOperation_t op, float alpha, float beta) {
  const int m = op == CUSPARSE_OPERATION_NON_TRANSPOSE ? kRows : kCols;
  const int n = op == CUSPARSE_OPERATION_NON_TRANSPOSE ? kCols : kRows;
  std::vector<int> rowsv(kR, kR + kNnz), colsv(kC, kC + kNnz);
  for (auto& v : rowsv) v += base;
  for (auto& v : colsv) v += base;
  std::vector<float> valsv(kV, kV + kNnz);
  std::vector<int> off = csr_offsets(base);

  int* d_off = up(off);
  int* d_row = up(rowsv);
  int* d_col = up(colsv);
  float* d_val = up(valsv);
  std::vector<float> xh(n), yh(m);
  for (int i = 0; i < n; ++i) xh[i] = 1.0f + 0.5f * i;
  for (int i = 0; i < m; ++i) yh[i] = 10.0f - i;
  float* d_x = up(xh);
  float* d_y = up(yh);

  cusparseSpMatDescr_t A;
  const cusparseIndexBase_t ib = base ? CUSPARSE_INDEX_BASE_ONE : CUSPARSE_INDEX_BASE_ZERO;
  if (coo)
    CS(cusparseCreateCoo(&A, kRows, kCols, kNnz, d_row, d_col, d_val, CUSPARSE_INDEX_32I, ib,
                         CUDA_R_32F));
  else
    CS(cusparseCreateCsr(&A, kRows, kCols, kNnz, d_off, d_col, d_val, CUSPARSE_INDEX_32I,
                         CUSPARSE_INDEX_32I, ib, CUDA_R_32F));
  cusparseDnVecDescr_t X, Y;
  CS(cusparseCreateDnVec(&X, n, d_x, CUDA_R_32F));
  CS(cusparseCreateDnVec(&Y, m, d_y, CUDA_R_32F));
  size_t bytes = 0;
  CS(cusparseSpMV_bufferSize(h, op, &alpha, A, X, &beta, Y, CUDA_R_32F,
                             CUSPARSE_SPMV_ALG_DEFAULT, &bytes));
  void* buf = nullptr;
  if (bytes) cudaMalloc(&buf, bytes);
  CS(cusparseSpMV(h, op, &alpha, A, X, &beta, Y, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, buf));
  dump(tag, down(d_y, m));
  if (buf) cudaFree(buf);
  cusparseDestroyDnVec(X); cusparseDestroyDnVec(Y); cusparseDestroySpMat(A);
  cudaFree(d_off); cudaFree(d_row); cudaFree(d_col); cudaFree(d_val);
  cudaFree(d_x); cudaFree(d_y);
}

static void spmm_case(cusparseHandle_t h, const char* tag, cusparseOperation_t opA,
                      cusparseOrder_t order, int ncol, float alpha, float beta) {
  const int m = opA == CUSPARSE_OPERATION_NON_TRANSPOSE ? kRows : kCols;
  const int k = opA == CUSPARSE_OPERATION_NON_TRANSPOSE ? kCols : kRows;
  std::vector<int> colsv(kC, kC + kNnz), off = csr_offsets(0);
  std::vector<float> valsv(kV, kV + kNnz);
  int* d_off = up(off);
  int* d_col = up(colsv);
  float* d_val = up(valsv);

  const int ldb = order == CUSPARSE_ORDER_COL ? k : ncol;
  const int ldc = order == CUSPARSE_ORDER_COL ? m : ncol;
  std::vector<float> bh((size_t)k * ncol), ch((size_t)m * ncol);
  for (size_t i = 0; i < bh.size(); ++i) bh[i] = 0.5f + 0.25f * (float)(i % 7);
  for (size_t i = 0; i < ch.size(); ++i) ch[i] = 1.0f + (float)(i % 3);
  float* d_b = up(bh);
  float* d_c = up(ch);

  cusparseSpMatDescr_t A;
  CS(cusparseCreateCsr(&A, kRows, kCols, kNnz, d_off, d_col, d_val, CUSPARSE_INDEX_32I,
                       CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
  cusparseDnMatDescr_t B, C;
  CS(cusparseCreateDnMat(&B, k, ncol, ldb, d_b, CUDA_R_32F, order));
  CS(cusparseCreateDnMat(&C, m, ncol, ldc, d_c, CUDA_R_32F, order));
  size_t bytes = 0;
  CS(cusparseSpMM_bufferSize(h, opA, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, A, B, &beta, C,
                             CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &bytes));
  void* buf = nullptr;
  if (bytes) cudaMalloc(&buf, bytes);
  CS(cusparseSpMM(h, opA, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, A, B, &beta, C, CUDA_R_32F,
                  CUSPARSE_SPMM_ALG_DEFAULT, buf));
  dump(tag, down(d_c, ch.size()));
  if (buf) cudaFree(buf);
  cusparseDestroyDnMat(B); cusparseDestroyDnMat(C); cusparseDestroySpMat(A);
  cudaFree(d_off); cudaFree(d_col); cudaFree(d_val); cudaFree(d_b); cudaFree(d_c);
}

static void run() {
  cusparseHandle_t h;
  CS(cusparseCreate(&h));

  spmv_case(h, "spmv csr base0", false, 0, CUSPARSE_OPERATION_NON_TRANSPOSE, 1.0f, 0.0f);
  spmv_case(h, "spmv csr base1", false, 1, CUSPARSE_OPERATION_NON_TRANSPOSE, 1.0f, 0.0f);
  spmv_case(h, "spmv coo base0", true, 0, CUSPARSE_OPERATION_NON_TRANSPOSE, 1.0f, 0.0f);
  spmv_case(h, "spmv coo base1", true, 1, CUSPARSE_OPERATION_NON_TRANSPOSE, 1.0f, 0.0f);
  spmv_case(h, "spmv csr alpha2 beta0.5", false, 0, CUSPARSE_OPERATION_NON_TRANSPOSE, 2.0f, 0.5f);
  spmv_case(h, "spmv csr transpose", false, 0, CUSPARSE_OPERATION_TRANSPOSE, 1.0f, 0.0f);
  spmv_case(h, "spmv coo transpose beta1", true, 0, CUSPARSE_OPERATION_TRANSPOSE, 1.0f, 1.0f);

  spmm_case(h, "spmm col-major k=3", CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_ORDER_COL, 3,
            1.0f, 0.0f);
  spmm_case(h, "spmm row-major k=3", CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_ORDER_ROW, 3,
            1.0f, 0.0f);
  spmm_case(h, "spmm col-major beta0.5", CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_ORDER_COL, 4,
            2.0f, 0.5f);
  spmm_case(h, "spmm transposeA col", CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_ORDER_COL, 2, 1.0f,
            0.0f);

  {  // Sparse -> dense, then dense -> sparse, and the values must survive.
    std::vector<int> off = csr_offsets(0), colsv(kC, kC + kNnz);
    std::vector<float> valsv(kV, kV + kNnz);
    int* d_off = up(off);
    int* d_col = up(colsv);
    float* d_val = up(valsv);
    std::vector<float> zeros((size_t)kRows * kCols, 0.0f);
    float* d_dense = up(zeros);
    cusparseSpMatDescr_t A;
    CS(cusparseCreateCsr(&A, kRows, kCols, kNnz, d_off, d_col, d_val, CUSPARSE_INDEX_32I,
                         CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    cusparseDnMatDescr_t D;
    CS(cusparseCreateDnMat(&D, kRows, kCols, kRows, d_dense, CUDA_R_32F, CUSPARSE_ORDER_COL));
    size_t bytes = 0;
    CS(cusparseSparseToDense_bufferSize(h, A, D, CUSPARSE_SPARSETODENSE_ALG_DEFAULT, &bytes));
    void* buf = nullptr;
    if (bytes) cudaMalloc(&buf, bytes);
    CS(cusparseSparseToDense(h, A, D, CUSPARSE_SPARSETODENSE_ALG_DEFAULT, buf));
    dump("sparse->dense", down(d_dense, zeros.size()));

    std::vector<int> off2(kRows + 1, 0), col2(kNnz, 0);
    std::vector<float> val2(kNnz, 0.0f);
    int* d_off2 = up(off2);
    int* d_col2 = up(col2);
    float* d_val2 = up(val2);
    cusparseSpMatDescr_t B;
    CS(cusparseCreateCsr(&B, kRows, kCols, 0, d_off2, d_col2, d_val2, CUSPARSE_INDEX_32I,
                         CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    size_t b2 = 0;
    CS(cusparseDenseToSparse_bufferSize(h, D, B, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, &b2));
    void* buf2 = nullptr;
    if (b2) cudaMalloc(&buf2, b2);
    CS(cusparseDenseToSparse_analysis(h, D, B, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, buf2));
    int64_t rr = 0, cc = 0, nn = 0;
    CS(cusparseSpMatGetSize(B, &rr, &cc, &nn));
    printf("%-34s rows=%lld cols=%lld nnz=%lld\n", "dense->sparse analysis", (long long)rr,
           (long long)cc, (long long)nn);
    CS(cusparseDenseToSparse_convert(h, D, B, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, buf2));
    dump("dense->sparse values", down(d_val2, kNnz));

    if (buf) cudaFree(buf);
    if (buf2) cudaFree(buf2);
    cusparseDestroyDnMat(D); cusparseDestroySpMat(A); cusparseDestroySpMat(B);
    cudaFree(d_off); cudaFree(d_col); cudaFree(d_val); cudaFree(d_dense);
    cudaFree(d_off2); cudaFree(d_col2); cudaFree(d_val2);
  }

  int major = 0;
  cusparseGetProperty(MAJOR_VERSION, &major);
  printf("cusparse major %d\n", major);
  cusparseDestroy(h);
}

int main() { run(); return 0; }
