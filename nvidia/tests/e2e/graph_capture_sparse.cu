// cuSPARSE beyond SpMV and SpMM (graph_cusparse covers those) inside a captured CUDA graph (see
// graph_capture_common.h): sparse vector operations, conversions and the triangular solve, each
// recorded with the descriptors destroyed right after and run at each launch of the graph.
// run_graph_capture.sh sparse cusparse --card runs the same program on NVIDIA's cuSPARSE.
#include <cusparse.h>

#include <algorithm>

#include "graph_capture_common.h"

using namespace gc;

#define OK(x) do { cusparseStatus_t s_ = (x); if (s_ != CUSPARSE_STATUS_SUCCESS) { \
  std::printf("     %s -> %d\n", #x, (int)s_); ok = false; } } while (0)

int main() {
  Runner r;
  cusparseHandle_t h;
  cusparseCreate(&h);
  cusparseSetStream(h, r.st);
  const int n = 16;
  // A lower-triangular CSR matrix with a full diagonal: row i has columns i-1 and i.
  std::vector<int> rp(n + 1), ci;
  std::vector<float> va;
  for (int i = 0; i < n; ++i) {
    rp[i] = static_cast<int>(ci.size());
    if (i > 0) ci.push_back(i - 1), va.push_back(-0.5f);
    ci.push_back(i), va.push_back(2.0f);
  }
  rp[n] = static_cast<int>(ci.size());
  const int nnz = static_cast<int>(ci.size());
  int *drp = r.alloc<int>(n + 1), *dci = r.alloc<int>(nnz), *rowidx = r.alloc<int>(nnz), *back = r.alloc<int>(n + 1);
  float* dva = r.alloc<float>(nnz);
  cudaMemcpy(drp, rp.data(), 4 * (n + 1), cudaMemcpyHostToDevice);
  cudaMemcpy(dci, ci.data(), 4 * nnz, cudaMemcpyHostToDevice);
  cudaMemcpy(dva, va.data(), 4 * nnz, cudaMemcpyHostToDevice);
  float *x = r.alloc<float>(n), *y = r.alloc<float>(n), *z = r.alloc<float>(n);
  float* dot = r.alloc<float>(1);
  const float one = 1.0f, two = 2.0f;

  r.run("cusparseXcsr2coo + cusparseXcoo2csr", [&] {
    bool ok = true;
    OK(cusparseXcsr2coo(h, drp, nnz, n, rowidx, CUSPARSE_INDEX_BASE_ZERO));
    OK(cusparseXcoo2csr(h, rowidx, nnz, n, back, CUSPARSE_INDEX_BASE_ZERO));
    return ok;
  }, {{rowidx, (size_t)nnz, Dt::I32}, {back, (size_t)n + 1, Dt::I32}});

  {
    // Sparse vector operations: the vector has 4 of its 16 entries.
    int* svi = r.alloc<int>(4);
    float* svv = r.alloc<float>(4);
    const int hi[4] = {1, 5, 9, 14};
    cudaMemcpy(svi, hi, sizeof hi, cudaMemcpyHostToDevice);
    r.run("cusparseGather + cusparseScatter", [&] {
      bool ok = true;
      fill(r.st, r.counter, x, n);
      cudaMemsetAsync(z, 0, n * 4, r.st);
      cusparseDnVecDescr_t X;
      cusparseSpVecDescr_t SV;
      cusparseDnVecDescr_t Z;
      OK(cusparseCreateDnVec(&X, n, x, CUDA_R_32F));
      OK(cusparseCreateDnVec(&Z, n, z, CUDA_R_32F));
      OK(cusparseCreateSpVec(&SV, n, 4, svi, svv, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
      OK(cusparseGather(h, X, SV));
      OK(cusparseScatter(h, SV, Z));
      cusparseDestroyDnVec(X);
      cusparseDestroyDnVec(Z);
      cusparseDestroySpVec(SV);
      return ok;
    }, {{svv, 4, Dt::F32}, {z, (size_t)n, Dt::F32}});
    r.run("cusparseAxpby + cusparseSpVV (a device scalar)", [&] {
      bool ok = true;
      fill(r.st, r.counter, svv, 4, 0.5f, 1.0f);
      fill(r.st, r.counter, y, n, 0.25f, 0.0f, 3);
      fill(r.st, r.counter, x, n, 0.25f, 0.0f, 4);
      cusparseDnVecDescr_t Y, X;
      cusparseSpVecDescr_t SV;
      OK(cusparseCreateDnVec(&Y, n, y, CUDA_R_32F));
      OK(cusparseCreateDnVec(&X, n, x, CUDA_R_32F));
      OK(cusparseCreateSpVec(&SV, n, 4, svi, svv, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
      OK(cusparseAxpby(h, &two, SV, &one, Y));
      cusparsePointerMode_t mode;
      cusparseGetPointerMode(h, &mode);
      cusparseSetPointerMode(h, CUSPARSE_POINTER_MODE_DEVICE);
      OK(cusparseSpVV(h, CUSPARSE_OPERATION_NON_TRANSPOSE, SV, X, dot, CUDA_R_32F, nullptr));
      cusparseSetPointerMode(h, mode);
      cusparseDestroyDnVec(Y);
      cusparseDestroyDnVec(X);
      cusparseDestroySpVec(SV);
      return ok;
    }, {{y, (size_t)n, Dt::F32}, {dot, 1, Dt::F32}}, 1e-5);
  }
  {
    // Triangular solve: analysis, then solve, both in the capture.
    size_t bytes = 0;
    cusparseSpSVDescr_t d;
    cusparseSpSV_createDescr(&d);
    cusparseSpMatDescr_t A;
    cusparseCreateCsr(&A, n, n, nnz, drp, dci, dva, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    cusparseFillMode_t fm = CUSPARSE_FILL_MODE_LOWER;
    cusparseDiagType_t dt = CUSPARSE_DIAG_TYPE_NON_UNIT;
    cusparseSpMatSetAttribute(A, CUSPARSE_SPMAT_FILL_MODE, &fm, sizeof fm);
    cusparseSpMatSetAttribute(A, CUSPARSE_SPMAT_DIAG_TYPE, &dt, sizeof dt);
    cusparseDnVecDescr_t X, Y;
    cusparseCreateDnVec(&X, n, x, CUDA_R_32F);
    cusparseCreateDnVec(&Y, n, y, CUDA_R_32F);
    cusparseSpSV_bufferSize(h, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, A, X, Y, CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT, d, &bytes);
    void* work = r.alloc<char>(bytes + 16);
    r.run("cusparseSpSV_analysis + cusparseSpSV_solve", [&] {
      bool ok = true;
      fill(r.st, r.counter, x, n);
      OK(cusparseSpSV_analysis(h, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, A, X, Y, CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT, d, work));
      OK(cusparseSpSV_solve(h, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, A, X, Y, CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT, d));
      return ok;
    }, {{y, (size_t)n, Dt::F32}}, 1e-4);
    cusparseDestroyDnVec(X);
    cusparseDestroyDnVec(Y);
    cusparseDestroySpMat(A);
    cusparseSpSV_destroyDescr(d);
  }
  cusparseDestroy(h);
  return finish();
}
