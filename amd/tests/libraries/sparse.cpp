// AMD's hipSPARSE, unmodified (over rocSPARSE), on a simulated MI300X, by its
// generic API (what PyTorch's torch.sparse calls). A 300 x 200 matrix with
// about one entry in twenty, each product checked against the same sums done
// on the host:
//
//   SpMV, CSR: y = 2 A x + 0.5 y, single precision, and A^T x (with atomics)
//   SpMV, COO, the same y
//   SpMV, CSR, double precision
//   SpMM, CSR times a dense 200 x 8, column-major
//   SparseToDense: the CSR matrix written out whole
//
// Built ahead of time by build.sh, from hipSPARSE's documented API.
#include <hip/hip_runtime.h>
#include <hipsparse/hipsparse.h>

#include <cmath>
#include <cstdio>
#include <vector>

namespace {

constexpr int kRows = 300, kCols = 200;
int ok = 0, total = 0;
void check(const char* what, bool good) {
  ++total;
  ok += good;
  if (!good) std::printf("wrong: %s\n", what);
}
bool ran(hipsparseStatus_t s, const char* what) {
  if (s != HIPSPARSE_STATUS_SUCCESS) std::printf("%s: hipSPARSE error %d\n", what, int(s));
  return s == HIPSPARSE_STATUS_SUCCESS;
}
double value(size_t i) { return double((i * 2654435761u) % 2001) / 1000.0 - 1.0; }

// The work buffer a call asked for, none where it asked for none (rocSPARSE
// refuses a buffer of no size).
bool scratch(void** buf, size_t size) {
  *buf = nullptr;
  return size == 0 || hipMalloc(buf, size) == hipSuccess;
}

template <class T>
T* device(const std::vector<T>& h) {
  T* p = nullptr;
  if (hipMalloc(&p, std::max<size_t>(h.size(), 1) * sizeof(T)) != hipSuccess ||
      hipMemcpy(p, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice) != hipSuccess)
    return nullptr;
  return p;
}
template <class T>
std::vector<T> host(const T* p, size_t n) {
  std::vector<T> h(n);
  if (hipMemcpy(h.data(), p, n * sizeof(T), hipMemcpyDeviceToHost) != hipSuccess) h.assign(n, T(NAN));
  return h;
}
template <class T>
bool close(const std::vector<T>& got, const std::vector<double>& want, double tol) {
  for (size_t i = 0; i < want.size(); ++i)
    if (!(std::abs(double(got[i]) - want[i]) <= tol * std::max(1.0, std::abs(want[i])))) return false;
  return true;
}

}  // namespace

int main() {
  // The matrix: entry (r, c) where a hash of the two says so, about 5%.
  std::vector<int> row_ptr{0}, col, coo_row;
  std::vector<float> val;
  std::vector<double> dense(size_t(kRows) * kCols, 0.0);
  for (int r = 0; r < kRows; ++r) {
    for (int c = 0; c < kCols; ++c)
      if ((unsigned(r * 7919 + c * 104729) * 2654435761u) % 20 == 0) {
        col.push_back(c);
        coo_row.push_back(r);
        val.push_back(float(value(col.size())));
        dense[size_t(r) * kCols + c] = val.back();
      }
    row_ptr.push_back(int(col.size()));
  }
  const int nnz = int(col.size());
  std::vector<float> x(kCols), y(kRows), xt(kRows);
  for (int i = 0; i < kCols; ++i) x[i] = float(value(50000 + i));
  for (int i = 0; i < kRows; ++i) y[i] = float(value(60000 + i)), xt[i] = float(value(70000 + i));

  hipsparseHandle_t h;
  if (!ran(hipsparseCreate(&h), "create")) return 1;
  int *d_ptr = device(row_ptr), *d_col = device(col), *d_row = device(coo_row);
  float* d_val = device(val);
  const float alpha = 2.0f, beta = 0.5f;

  // y = 2 A x + 0.5 y, through a CSR and through a COO description.
  for (const bool coo : {false, true}) {
    hipsparseSpMatDescr_t a;
    hipsparseDnVecDescr_t vx, vy;
    float *d_x = device(x), *d_y = device(y);
    size_t size = 0;
    void* buf = nullptr;
    bool good =
        (coo ? ran(hipsparseCreateCoo(&a, kRows, kCols, nnz, d_row, d_col, d_val, HIPSPARSE_INDEX_32I,
                                      HIPSPARSE_INDEX_BASE_ZERO, HIP_R_32F),
                   "COO")
             : ran(hipsparseCreateCsr(&a, kRows, kCols, nnz, d_ptr, d_col, d_val, HIPSPARSE_INDEX_32I,
                                      HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_BASE_ZERO, HIP_R_32F),
                   "CSR")) &&
        ran(hipsparseCreateDnVec(&vx, kCols, d_x, HIP_R_32F), "x") && ran(hipsparseCreateDnVec(&vy, kRows, d_y, HIP_R_32F), "y") &&
        ran(hipsparseSpMV_bufferSize(h, HIPSPARSE_OPERATION_NON_TRANSPOSE, &alpha, a, vx, &beta, vy, HIP_R_32F,
                                     HIPSPARSE_SPMV_ALG_DEFAULT, &size),
            "SpMV size") &&
        scratch(&buf, size) &&
        ran(hipsparseSpMV(h, HIPSPARSE_OPERATION_NON_TRANSPOSE, &alpha, a, vx, &beta, vy, HIP_R_32F,
                          HIPSPARSE_SPMV_ALG_DEFAULT, buf),
            "SpMV");
    std::vector<double> want(kRows);
    for (int r = 0; r < kRows; ++r) {
      double s = 0;
      for (int c = 0; c < kCols; ++c) s += dense[size_t(r) * kCols + c] * x[c];
      want[r] = alpha * s + beta * y[r];
    }
    check(coo ? "SpMV, COO" : "SpMV, CSR", good && close(host(d_y, kRows), want, 1e-5));
    hipsparseDestroySpMat(a);
    hipsparseDestroyDnVec(vx);
    hipsparseDestroyDnVec(vy);
    (void)hipFree(buf), (void)hipFree(d_x), (void)hipFree(d_y);
  }

  {  // A^T x, from the CSR description, by rows split among threads
     // (CSR_ALG2): rocSPARSE's default, adaptive, has no transpose.
    hipsparseSpMatDescr_t a;
    hipsparseDnVecDescr_t vx, vy;
    float* d_x = device(xt);
    float* d_y = device(std::vector<float>(kCols, 0.0f));
    const float one = 1.0f, zero = 0.0f;
    size_t size = 0;
    void* buf = nullptr;
    const bool good =
        ran(hipsparseCreateCsr(&a, kRows, kCols, nnz, d_ptr, d_col, d_val, HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I,
                               HIPSPARSE_INDEX_BASE_ZERO, HIP_R_32F),
            "CSR") &&
        ran(hipsparseCreateDnVec(&vx, kRows, d_x, HIP_R_32F), "x") && ran(hipsparseCreateDnVec(&vy, kCols, d_y, HIP_R_32F), "y") &&
        ran(hipsparseSpMV_bufferSize(h, HIPSPARSE_OPERATION_TRANSPOSE, &one, a, vx, &zero, vy, HIP_R_32F,
                                     HIPSPARSE_SPMV_CSR_ALG2, &size),
            "SpMV^T size") &&
        scratch(&buf, size) &&
        ran(hipsparseSpMV(h, HIPSPARSE_OPERATION_TRANSPOSE, &one, a, vx, &zero, vy, HIP_R_32F, HIPSPARSE_SPMV_CSR_ALG2,
                          buf),
            "SpMV^T");
    std::vector<double> want(kCols, 0.0);
    for (int r = 0; r < kRows; ++r)
      for (int c = 0; c < kCols; ++c) want[c] += dense[size_t(r) * kCols + c] * xt[r];
    check("SpMV, A^T x", good && close(host(d_y, kCols), want, 1e-5));
    hipsparseDestroySpMat(a);
    hipsparseDestroyDnVec(vx);
    hipsparseDestroyDnVec(vy);
    (void)hipFree(buf), (void)hipFree(d_x), (void)hipFree(d_y);
  }

  {  // The same product in double precision.
    std::vector<double> vd(val.begin(), val.end()), xd(x.begin(), x.end());
    double *d_v = device(vd), *d_x = device(xd), *d_y = device(std::vector<double>(kRows, 0.0));
    hipsparseSpMatDescr_t a;
    hipsparseDnVecDescr_t vx, vy;
    const double one = 1.0, zero = 0.0;
    size_t size = 0;
    void* buf = nullptr;
    const bool good =
        ran(hipsparseCreateCsr(&a, kRows, kCols, nnz, d_ptr, d_col, d_v, HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I,
                               HIPSPARSE_INDEX_BASE_ZERO, HIP_R_64F),
            "CSR double") &&
        ran(hipsparseCreateDnVec(&vx, kCols, d_x, HIP_R_64F), "x") && ran(hipsparseCreateDnVec(&vy, kRows, d_y, HIP_R_64F), "y") &&
        ran(hipsparseSpMV_bufferSize(h, HIPSPARSE_OPERATION_NON_TRANSPOSE, &one, a, vx, &zero, vy, HIP_R_64F,
                                     HIPSPARSE_SPMV_ALG_DEFAULT, &size),
            "SpMV double size") &&
        scratch(&buf, size) &&
        ran(hipsparseSpMV(h, HIPSPARSE_OPERATION_NON_TRANSPOSE, &one, a, vx, &zero, vy, HIP_R_64F,
                          HIPSPARSE_SPMV_ALG_DEFAULT, buf),
            "SpMV double");
    std::vector<double> want(kRows, 0.0);
    for (int r = 0; r < kRows; ++r)
      for (int c = 0; c < kCols; ++c) want[r] += dense[size_t(r) * kCols + c] * xd[c];
    check("SpMV, CSR, double", good && close(host(d_y, kRows), want, 1e-12));
    hipsparseDestroySpMat(a);
    hipsparseDestroyDnVec(vx);
    hipsparseDestroyDnVec(vy);
    (void)hipFree(buf), (void)hipFree(d_v), (void)hipFree(d_x), (void)hipFree(d_y);
  }

  {  // C = A B, B dense 200 x 8, column-major.
    constexpr int kN = 8;
    std::vector<float> b(size_t(kCols) * kN);
    for (size_t i = 0; i < b.size(); ++i) b[i] = float(value(80000 + i));
    float *d_b = device(b), *d_c = device(std::vector<float>(size_t(kRows) * kN, 0.0f));
    hipsparseSpMatDescr_t a;
    hipsparseDnMatDescr_t mb, mc;
    const float one = 1.0f, zero = 0.0f;
    size_t size = 0;
    void* buf = nullptr;
    const bool good =
        ran(hipsparseCreateCsr(&a, kRows, kCols, nnz, d_ptr, d_col, d_val, HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I,
                               HIPSPARSE_INDEX_BASE_ZERO, HIP_R_32F),
            "CSR") &&
        ran(hipsparseCreateDnMat(&mb, kCols, kN, kCols, d_b, HIP_R_32F, HIPSPARSE_ORDER_COL), "B") &&
        ran(hipsparseCreateDnMat(&mc, kRows, kN, kRows, d_c, HIP_R_32F, HIPSPARSE_ORDER_COL), "C") &&
        ran(hipsparseSpMM_bufferSize(h, HIPSPARSE_OPERATION_NON_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE, &one, a, mb,
                                     &zero, mc, HIP_R_32F, HIPSPARSE_SPMM_ALG_DEFAULT, &size),
            "SpMM size") &&
        scratch(&buf, size) &&
        ran(hipsparseSpMM(h, HIPSPARSE_OPERATION_NON_TRANSPOSE, HIPSPARSE_OPERATION_NON_TRANSPOSE, &one, a, mb, &zero, mc,
                          HIP_R_32F, HIPSPARSE_SPMM_ALG_DEFAULT, buf),
            "SpMM");
    std::vector<double> want(size_t(kRows) * kN, 0.0);
    for (int j = 0; j < kN; ++j)
      for (int r = 0; r < kRows; ++r)
        for (int c = 0; c < kCols; ++c) want[r + size_t(j) * kRows] += dense[size_t(r) * kCols + c] * b[c + size_t(j) * kCols];
    check("SpMM, CSR times dense", good && close(host(d_c, want.size()), want, 1e-5));
    hipsparseDestroySpMat(a);
    hipsparseDestroyDnMat(mb);
    hipsparseDestroyDnMat(mc);
    (void)hipFree(buf), (void)hipFree(d_b), (void)hipFree(d_c);
  }

  {  // The matrix written out whole, row-major.
    float* d_d = device(std::vector<float>(size_t(kRows) * kCols, -1.0f));
    hipsparseSpMatDescr_t a;
    hipsparseDnMatDescr_t md;
    size_t size = 0;
    void* buf = nullptr;
    const bool good =
        ran(hipsparseCreateCsr(&a, kRows, kCols, nnz, d_ptr, d_col, d_val, HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I,
                               HIPSPARSE_INDEX_BASE_ZERO, HIP_R_32F),
            "CSR") &&
        ran(hipsparseCreateDnMat(&md, kRows, kCols, kCols, d_d, HIP_R_32F, HIPSPARSE_ORDER_ROW), "dense") &&
        ran(hipsparseSparseToDense_bufferSize(h, a, md, HIPSPARSE_SPARSETODENSE_ALG_DEFAULT, &size), "to dense size") &&
        scratch(&buf, size) &&
        ran(hipsparseSparseToDense(h, a, md, HIPSPARSE_SPARSETODENSE_ALG_DEFAULT, buf), "to dense");
    check("SparseToDense", good && close(host(d_d, dense.size()), dense, 0));
    hipsparseDestroySpMat(a);
    hipsparseDestroyDnMat(md);
    (void)hipFree(buf), (void)hipFree(d_d);
  }

  hipsparseDestroy(h);
  std::printf("hipSPARSE: %d of %d products match the host's (%d entries)\n", ok, total, nnz);
  return ok == total ? 0 : 1;
}
