// cuBLAS's grouped batched GEMM (HeCBench's megablocks): each group applies its
// own transposes, sizes, leading dimensions, alpha and beta to the next
// group_size problems. Every expectation here is what an RTX 3060's cuBLAS
// gives: the values below, the types it accepts, host-only alpha/beta, and
// that every group is checked before any of them runs.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>

static int failures = 0;
static void check(bool ok, const char* what) {
  std::printf("%-4s %s\n", ok ? "ok" : "FAIL", what);
  if (!ok) ++failures;
}
#define IS(call, want) check((int)(call) == (int)(want), #call " -> " #want)

#if CUBLAS_VERSION < 120500
// Grouped batched GEMM arrived in cuBLAS 12.5: an older toolkit has nothing to
// compile against, which is not a failure of the simulator.
int main() {
  std::printf("SKIP: cuBLAS %d has no grouped batched GEMM (12.5 and later)\nPASS\n", CUBLAS_VERSION);
  return 0;
}
#else
int main() {
  cublasHandle_t h;
  if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) { std::printf("FAIL: no handle\nFAIL\n"); return 1; }
  float *A, *B, *C;
  cudaMallocManaged(&A, 64 * 4);
  cudaMallocManaged(&B, 64 * 4);
  cudaMallocManaged(&C, 3 * 64 * 4);
  for (int i = 0; i < 64; ++i) { A[i] = i % 5 + 1; B[i] = i % 3 - 1; }
  float** P;
  cudaMallocManaged(&P, 9 * sizeof(float*));
  for (int i = 0; i < 3; ++i) { P[i] = A; P[3 + i] = B; P[6 + i] = C + 64 * i; }
  const auto Ap = (const float* const*)P;
  const auto Bp = (const float* const*)(P + 3);
  const auto Cp = (float* const*)(P + 6);
  cublasOperation_t ta[2] = {CUBLAS_OP_N, CUBLAS_OP_T}, tb[2] = {CUBLAS_OP_N, CUBLAS_OP_T};
  int m[2] = {4, 3}, n[2] = {4, 2}, k[2] = {4, 3}, ld[2] = {8, 8}, gs[2] = {2, 1};
  float al[2] = {1, 2}, be[2] = {0, 0.5f};

  // Two problems in group 0 (4x4x4, N/N, alpha 1, beta 0), one in group 1
  // (3x2x3, T/T, alpha 2, beta 0.5 over C = 7). The card's values:
  const float want01[16] = {-4, 0, -1, -2, 2, -2, 4, 5, 2, 2, -3, -3, -4, 0, -1, -2};
  const float want2[16] = {5.5f, 5.5f, 5.5f, 7, 5.5f, -4.5f, 5.5f, 7, 7, 7, 7, 7, 7, 7, 7, 7};
  for (int i = 0; i < 192; ++i) C[i] = 7;
  IS(cublasSgemmGroupedBatched(h, ta, tb, m, n, k, al, Ap, ld, Bp, ld, be, Cp, ld, 2, gs), CUBLAS_STATUS_SUCCESS);
  cudaDeviceSynchronize();
  bool same = true;
  for (int p = 0; p < 3; ++p)
    for (int j = 0; j < 4; ++j)
      for (int i = 0; i < 4; ++i)
        same &= C[64 * p + j * 8 + i] == (p < 2 ? want01 : want2)[j * 4 + i];
  check(same, "Sgemm grouped: each group's own sizes, transposes, alpha and beta (the card's values)");

  // A bad second group: nothing runs, the first group's C included.
  for (int i = 0; i < 192; ++i) C[i] = 7;
  int mbad[2] = {4, -1};
  IS(cublasSgemmGroupedBatched(h, ta, tb, mbad, n, k, al, Ap, ld, Bp, ld, be, Cp, ld, 2, gs),
     CUBLAS_STATUS_INVALID_VALUE);
  cudaDeviceSynchronize();
  check(C[0] == 7, "a bad last group leaves the first group's C untouched");
  int gneg[2] = {-1, 1}, gzero[2] = {0, 0}, ldbad[2] = {2, 8};
  IS(cublasSgemmGroupedBatched(h, ta, tb, m, n, k, al, Ap, ld, Bp, ld, be, Cp, ld, -1, gs), CUBLAS_STATUS_INVALID_VALUE);
  IS(cublasSgemmGroupedBatched(h, ta, tb, m, n, k, al, Ap, ld, Bp, ld, be, Cp, ld, 0, gs), CUBLAS_STATUS_SUCCESS);
  IS(cublasSgemmGroupedBatched(h, ta, tb, m, n, k, al, Ap, ld, Bp, ld, be, Cp, ld, 2, gneg), CUBLAS_STATUS_INVALID_VALUE);
  IS(cublasSgemmGroupedBatched(h, ta, tb, m, n, k, al, Ap, ld, Bp, ld, be, Cp, ld, 2, gzero), CUBLAS_STATUS_SUCCESS);
  IS(cublasSgemmGroupedBatched(h, ta, tb, m, n, k, al, Ap, ldbad, Bp, ld, be, Cp, ld, 2, gs), CUBLAS_STATUS_INVALID_VALUE);

  // The Ex form's types: what the card accepts, and some it refuses.
  struct T { cudaDataType a, c; cublasComputeType_t cp; int want; const char* what; };
  const T types[] = {
      {CUDA_R_32F, CUDA_R_32F, CUBLAS_COMPUTE_32F, 0, "32F, COMPUTE_32F"},
      {CUDA_R_32F, CUDA_R_32F, CUBLAS_COMPUTE_32F_PEDANTIC, 0, "32F, COMPUTE_32F_PEDANTIC"},
      {CUDA_R_32F, CUDA_R_32F, CUBLAS_COMPUTE_32F_FAST_TF32, 0, "32F, COMPUTE_32F_FAST_TF32"},
      {CUDA_R_32F, CUDA_R_32F, CUBLAS_COMPUTE_32F_FAST_16F, 15, "32F, COMPUTE_32F_FAST_16F refused"},
      {CUDA_R_32F, CUDA_R_32F, CUBLAS_COMPUTE_32F_FAST_16BF, 15, "32F, COMPUTE_32F_FAST_16BF refused"},
      {CUDA_R_16F, CUDA_R_16F, CUBLAS_COMPUTE_32F, 0, "16F, COMPUTE_32F"},
      {CUDA_R_16F, CUDA_R_32F, CUBLAS_COMPUTE_32F, 15, "16F into a 32F C refused"},
      {CUDA_R_16F, CUDA_R_16F, CUBLAS_COMPUTE_16F, 15, "16F, COMPUTE_16F refused"},
      {CUDA_R_16BF, CUDA_R_16BF, CUBLAS_COMPUTE_32F, 0, "16BF, COMPUTE_32F"},
      {CUDA_R_16BF, CUDA_R_32F, CUBLAS_COMPUTE_32F, 15, "16BF into a 32F C refused"},
      {CUDA_R_8I, CUDA_R_32I, CUBLAS_COMPUTE_32I, 15, "8I, COMPUTE_32I refused"},
      {CUDA_C_32F, CUDA_C_32F, CUBLAS_COMPUTE_32F, 15, "C32F refused"},
  };
  int ai[2] = {1, 1}, bi[2] = {0, 0};
  for (const T& t : types) {
    const void* a = t.cp == CUBLAS_COMPUTE_32I ? (const void*)ai : (const void*)al;
    const void* b = t.cp == CUBLAS_COMPUTE_32I ? (const void*)bi : (const void*)be;
    const int got = (int)cublasGemmGroupedBatchedEx(h, ta, tb, m, n, k, a, (const void* const*)P, t.a, ld,
                                                    (const void* const*)(P + 3), t.a, ld, b, (void* const*)(P + 6),
                                                    t.c, ld, 2, gs, t.cp);
    cudaDeviceSynchronize();
    char what[96];
    std::snprintf(what, sizeof what, "Ex types: %s (%d, want %d)", t.what, got, t.want);
    check(got == t.want, what);
  }

  // alpha and beta from the device: NOT_SUPPORTED.
  float *dal, *dbe;
  cudaMalloc(&dal, 8);
  cudaMalloc(&dbe, 8);
  cudaMemcpy(dal, al, 8, cudaMemcpyHostToDevice);
  cudaMemcpy(dbe, be, 8, cudaMemcpyHostToDevice);
  cublasSetPointerMode(h, CUBLAS_POINTER_MODE_DEVICE);
  IS(cublasSgemmGroupedBatched(h, ta, tb, m, n, k, dal, Ap, ld, Bp, ld, dbe, Cp, ld, 2, gs), CUBLAS_STATUS_NOT_SUPPORTED);
  cublasSetPointerMode(h, CUBLAS_POINTER_MODE_HOST);

  // The double and 64-bit forms against a host reference: group g scales by g + 1.
  double *dA, *dB, *dC;
  cudaMallocManaged(&dA, 4 * 25 * 8);
  cudaMallocManaged(&dB, 4 * 25 * 8);
  cudaMallocManaged(&dC, 4 * 25 * 8);
  double** Q;
  cudaMallocManaged(&Q, 12 * sizeof(double*));
  for (int p = 0; p < 4; ++p) {
    for (int i = 0; i < 25; ++i) { dA[25 * p + i] = std::sin(p + i * 0.3); dB[25 * p + i] = std::cos(p * 2 + i * 0.7); dC[25 * p + i] = 0; }
    Q[p] = dA + 25 * p; Q[4 + p] = dB + 25 * p; Q[8 + p] = dC + 25 * p;
  }
  cublasOperation_t nn[2] = {CUBLAS_OP_N, CUBLAS_OP_N};
  int64_t m6[2] = {5, 2}, n6[2] = {5, 3}, k6[2] = {5, 4}, ld6[2] = {5, 5}, gs6[2] = {3, 1};
  double dal2[2] = {1, 2}, dbe2[2] = {0, 0};
  IS(cublasDgemmGroupedBatched_64(h, nn, nn, m6, n6, k6, dal2, (const double* const*)Q, ld6, (const double* const*)(Q + 4),
                                  ld6, dbe2, (double* const*)(Q + 8), ld6, 2, gs6),
     CUBLAS_STATUS_SUCCESS);
  cudaDeviceSynchronize();
  double err = 0;
  for (int p = 0; p < 4; ++p) {
    const int g = p < 3 ? 0 : 1;
    for (int j = 0; j < n6[g]; ++j)
      for (int i = 0; i < m6[g]; ++i) {
        double s = 0;
        for (int l = 0; l < k6[g]; ++l) s += dA[25 * p + l * 5 + i] * dB[25 * p + j * 5 + l];
        err = std::fmax(err, std::fabs(dal2[g] * s - dC[25 * p + j * 5 + i]));
      }
  }
  check(err < 1e-12, "Dgemm grouped _64: every problem matches the host reference");

  cublasDestroy(h);
  std::printf("%s\n", failures ? "FAIL" : "PASS");
  return failures ? 1 : 0;
}
#endif
