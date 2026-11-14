// cuSPARSE's argument checks and status texts, against an RTX 3060's cuSPARSE
// 13.0. The simulator accepted every one of these refused calls.
//
//   - cusparseCreateCsr/Csc/Coo: an index base other than 0 or 1, and more
//     entries than the matrix has places, are invalid values.
//   - cusparseCreateDnMat: an unknown order, and a leading dimension below the
//     rows (COL) or columns (ROW), are invalid values.
//   - cusparseSpMV: an unknown operation and a null alpha or beta are invalid
//     values; so is an algorithm that is not CSR's (COO_ALG1, COO_ALG2,
//     SELL_ALG1) for a CSR matrix. A, x and y of one real type compute in that
//     type (half and bfloat16 in single); any other compute type is
//     CUSPARSE_STATUS_NOT_SUPPORTED, from cusparseSpMV_bufferSize too.
//   - cusparseSetMat{Type,IndexBase,FillMode,DiagType} refuse values their
//     enums do not have.
//   - Destroying a null handle or descriptor is a success.
//   - cusparseGetErrorName/String: the card's names and sentences.
#include <cuda_runtime.h>
#include <cusparse.h>

#include <cstdio>
#include <cstring>

static int fails = 0;
#define WANT(x, want) do { const int got_ = static_cast<int>(x); if (got_ != static_cast<int>(want)) { \
  std::printf("FAIL line %d: %s -> %d, expected %d\n", __LINE__, #x, got_, static_cast<int>(want)); ++fails; } } while (0)
#define WANT_STR(x, want) do { const char* got_ = (x); if (std::strcmp(got_, want) != 0) { \
  std::printf("FAIL %s -> \"%s\", expected \"%s\"\n", #x, got_, want); ++fails; } } while (0)

int main() {
  const auto ok = CUSPARSE_STATUS_SUCCESS;
  const auto bad = CUSPARSE_STATUS_INVALID_VALUE;
  const auto I32 = CUSPARSE_INDEX_32I;
  const auto Z = CUSPARSE_INDEX_BASE_ZERO;
  cusparseHandle_t h;
  WANT(cusparseCreate(&h), ok);
  int* mem = nullptr;
  WANT(cudaMalloc(&mem, 4096), cudaSuccess);
  // [1 0; 0 1]
  const int rp[] = {0, 1, 2}, ci[] = {0, 1};
  const float val[] = {1, 1}, xv[] = {3, 4};
  int* drp = mem;
  int* dci = mem + 16;
  float* dval = reinterpret_cast<float*>(mem + 32);
  float* dx = reinterpret_cast<float*>(mem + 48);
  float* dy = reinterpret_cast<float*>(mem + 64);
  cudaMemcpy(drp, rp, sizeof rp, cudaMemcpyHostToDevice);
  cudaMemcpy(dci, ci, sizeof ci, cudaMemcpyHostToDevice);
  cudaMemcpy(dval, val, sizeof val, cudaMemcpyHostToDevice);
  cudaMemcpy(dx, xv, sizeof xv, cudaMemcpyHostToDevice);

  // Descriptors.
  cusparseSpMatDescr_t B = nullptr;
  WANT(cusparseCreateCsr(&B, 3, 3, 10, drp, dci, dval, I32, I32, Z, CUDA_R_32F), bad);
  WANT(cusparseCreateCsc(&B, 3, 3, 10, drp, dci, dval, I32, I32, Z, CUDA_R_32F), bad);
  WANT(cusparseCreateCoo(&B, 3, 3, 10, drp, dci, dval, I32, Z, CUDA_R_32F), bad);
  WANT(cusparseCreateCsr(&B, 0, 0, 1, drp, dci, dval, I32, I32, Z, CUDA_R_32F), bad);
  WANT(cusparseCreateCsr(&B, 3, 3, 4, drp, dci, dval, I32, I32, static_cast<cusparseIndexBase_t>(3), CUDA_R_32F), bad);
  WANT(cusparseCreateCoo(&B, 3, 3, 4, drp, dci, dval, I32, static_cast<cusparseIndexBase_t>(2), CUDA_R_32F), bad);
  WANT(cusparseCreateCsr(&B, 3, 3, 9, drp, dci, dval, I32, I32, Z, CUDA_R_32F), ok);
  WANT(cusparseDestroySpMat(B), ok);
  cusparseDnMatDescr_t M = nullptr;
  WANT(cusparseCreateDnMat(&M, 3, 2, 2, dx, CUDA_R_32F, CUSPARSE_ORDER_COL), bad);
  WANT(cusparseCreateDnMat(&M, 3, 4, 3, dx, CUDA_R_32F, CUSPARSE_ORDER_ROW), bad);
  WANT(cusparseCreateDnMat(&M, 3, 2, 3, dx, CUDA_R_32F, static_cast<cusparseOrder_t>(5)), bad);
  WANT(cusparseCreateDnMat(&M, 3, 4, 4, dx, CUDA_R_32F, CUSPARSE_ORDER_ROW), ok);
  WANT(cusparseDestroyDnMat(M), ok);
  WANT(cusparseCreateDnMat(&M, 0, 4, 0, dx, CUDA_R_32F, CUSPARSE_ORDER_COL), ok);
  WANT(cusparseDestroyDnMat(M), ok);

  // SpMV.
  cusparseSpMatDescr_t A;
  cusparseDnVecDescr_t X, Y;
  WANT(cusparseCreateCsr(&A, 2, 2, 2, drp, dci, dval, I32, I32, Z, CUDA_R_32F), ok);
  WANT(cusparseCreateDnVec(&X, 2, dx, CUDA_R_32F), ok);
  WANT(cusparseCreateDnVec(&Y, 2, dy, CUDA_R_32F), ok);
  const float one = 1, zero = 0;
  size_t bytes = 0;
  void* buf = mem + 512;
  const auto NT = CUSPARSE_OPERATION_NON_TRANSPOSE;
  WANT(cusparseSpMV(h, static_cast<cusparseOperation_t>(9), &one, A, X, &zero, Y, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, buf), bad);
  WANT(cusparseSpMV(h, NT, nullptr, A, X, &zero, Y, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, buf), bad);
  WANT(cusparseSpMV(h, NT, &one, A, X, nullptr, Y, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, buf), bad);
  for (int alg = 0; alg < 8; ++alg) {
    const bool csr = alg == 0 || alg == 2 || alg == 3;
    WANT(cusparseSpMV_bufferSize(h, NT, &one, A, X, &zero, Y, CUDA_R_32F, static_cast<cusparseSpMVAlg_t>(alg), &bytes),
         csr ? ok : bad);
    WANT(cusparseSpMV(h, NT, &one, A, X, &zero, Y, CUDA_R_32F, static_cast<cusparseSpMVAlg_t>(alg), buf), csr ? ok : bad);
  }
  for (cudaDataType ct : {CUDA_R_16F, CUDA_R_16BF, CUDA_R_64F, CUDA_C_32F}) {
    WANT(cusparseSpMV_bufferSize(h, NT, &one, A, X, &zero, Y, ct, CUSPARSE_SPMV_ALG_DEFAULT, &bytes), CUSPARSE_STATUS_NOT_SUPPORTED);
    WANT(cusparseSpMV(h, NT, &one, A, X, &zero, Y, ct, CUSPARSE_SPMV_ALG_DEFAULT, buf), CUSPARSE_STATUS_NOT_SUPPORTED);
  }
  WANT(cusparseSpMV(h, NT, &one, A, X, &zero, Y, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, buf), ok);
  float y[2] = {};
  cudaMemcpy(y, dy, sizeof y, cudaMemcpyDeviceToHost);
  WANT(y[0] == 3 && y[1] == 4, true);
  // Half data computes in single, and only in single.
  {
    cusparseSpMatDescr_t Ah;
    cusparseDnVecDescr_t Xh, Yh;
    WANT(cusparseCreateCsr(&Ah, 2, 2, 2, drp, dci, mem + 128, I32, I32, Z, CUDA_R_16F), ok);
    WANT(cusparseCreateDnVec(&Xh, 2, mem + 160, CUDA_R_16F), ok);
    WANT(cusparseCreateDnVec(&Yh, 2, mem + 192, CUDA_R_16F), ok);
    WANT(cusparseSpMV(h, NT, &one, Ah, Xh, &zero, Yh, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, buf), ok);
    WANT(cusparseSpMV(h, NT, &one, Ah, Xh, &zero, Yh, CUDA_R_16F, CUSPARSE_SPMV_ALG_DEFAULT, buf), CUSPARSE_STATUS_NOT_SUPPORTED);
    WANT(cusparseDestroySpMat(Ah), ok);
    WANT(cusparseDestroyDnVec(Xh), ok);
    WANT(cusparseDestroyDnVec(Yh), ok);
  }

  // Legacy descriptor fields.
  cusparseMatDescr_t md;
  WANT(cusparseCreateMatDescr(&md), ok);
  for (int t = 0; t < 6; ++t) WANT(cusparseSetMatType(md, static_cast<cusparseMatrixType_t>(t)), t <= 3 ? ok : bad);
  WANT(cusparseGetMatType(md), CUSPARSE_MATRIX_TYPE_TRIANGULAR);
  for (int t = 0; t < 4; ++t) WANT(cusparseSetMatIndexBase(md, static_cast<cusparseIndexBase_t>(t)), t <= 1 ? ok : bad);
  WANT(cusparseGetMatIndexBase(md), CUSPARSE_INDEX_BASE_ONE);
  for (int t = 0; t < 4; ++t) WANT(cusparseSetMatFillMode(md, static_cast<cusparseFillMode_t>(t)), t <= 1 ? ok : bad);
  for (int t = 0; t < 4; ++t) WANT(cusparseSetMatDiagType(md, static_cast<cusparseDiagType_t>(t)), t <= 1 ? ok : bad);
  WANT(cusparseDestroyMatDescr(md), ok);

  // Destroying nothing.
  WANT(cusparseDestroySpMat(nullptr), ok);
  WANT(cusparseDestroyDnVec(nullptr), ok);
  WANT(cusparseDestroyDnMat(nullptr), ok);
  WANT(cusparseDestroySpVec(nullptr), ok);
  WANT(cusparseDestroyMatDescr(nullptr), ok);
  WANT(cusparseDestroy(nullptr), ok);

  // Names and sentences.
  const char* names[] = {"CUSPARSE_STATUS_SUCCESS", "CUSPARSE_STATUS_NOT_INITIALIZED", "CUSPARSE_STATUS_ALLOC_FAILED",
                         "CUSPARSE_STATUS_INVALID_VALUE", "CUSPARSE_STATUS_ARCH_MISMATCH", "CUSPARSE_STATUS_MAPPING_ERROR",
                         "CUSPARSE_STATUS_EXECUTION_FAILED", "CUSPARSE_STATUS_INTERNAL_ERROR",
                         "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED", "CUSPARSE_STATUS_ZERO_PIVOT",
                         "CUSPARSE_STATUS_NOT_SUPPORTED", "CUSPARSE_STATUS_INSUFFICIENT_RESOURCES", "unrecognized error code"};
  const char* texts[] = {"success", "initialization error", "out of memory", "invalid value", "architecture mismatch",
                         "texture memory mapping error", "kernel launch failure", "internal error",
                         "matrix type not supported", "zero pivot", "operation not supported", "insufficient resources",
                         "unrecognized error code"};
  for (int i = 0; i <= 12; ++i) {
    WANT_STR(cusparseGetErrorName(static_cast<cusparseStatus_t>(i)), names[i]);
    WANT_STR(cusparseGetErrorString(static_cast<cusparseStatus_t>(i)), texts[i]);
  }

  WANT(cusparseDestroySpMat(A), ok);
  WANT(cusparseDestroyDnVec(X), ok);
  WANT(cusparseDestroyDnVec(Y), ok);
  WANT(cudaFree(mem), cudaSuccess);
  WANT(cusparseDestroy(h), ok);
  std::printf("%s\n", fails ? "FAILED" : "PASS");
  return fails ? 1 : 0;
}
