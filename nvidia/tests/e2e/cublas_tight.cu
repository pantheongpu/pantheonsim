// cuBLAS operands allocated to exactly the size they need.
//
// A leading dimension implies padding *between* columns, not after the last
// one: a column-major matrix of `rows` x `cols` with leading dimension `ld`
// occupies ld*(cols-1) + rows elements, not ld*cols. Reading the larger figure
// runs past the end of any matrix allocated to fit, which ggml does -- it
// showed up as a read 1544 bytes past a 1792-byte allocation, from inside a
// batched GEMM.
//
// Every buffer here is allocated to the exact extent, so an over-read has
// nowhere to land and the bounds check reports it.
#include <cmath>
#include <cstdio>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CK(x)                                                                      \
  do {                                                                             \
    cudaError_t e_ = (x);                                                          \
    if (e_) {                                                                      \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_));  \
      return 1;                                                                    \
    }                                                                              \
  } while (0)
#define CB(x)                                                                          \
  do {                                                                                 \
    cublasStatus_t s_ = (x);                                                           \
    if (s_) {                                                                          \
      std::printf("FAIL %s:%d: cuBLAS status %d\n", __FILE__, __LINE__, (int)s_);       \
      return 1;                                                                        \
    }                                                                                  \
  } while (0)

int main() {
  // Leading dimensions deliberately larger than the row counts, so the tail of
  // the last column is padding that does not exist.
  const int m = 5, n = 4, k = 3;
  const int lda = 8, ldb = 7, ldc = 6;
  const size_t na = (size_t)lda * (k - 1) + m;   // A is m x k
  const size_t nb = (size_t)ldb * (n - 1) + k;   // B is k x n
  const size_t nc = (size_t)ldc * (n - 1) + m;   // C is m x n

  float hA[64], hB[64], hC[64];
  for (size_t i = 0; i < na; ++i) hA[i] = (float)(i % 7) - 3.0f;
  for (size_t i = 0; i < nb; ++i) hB[i] = (float)(i % 5) - 2.0f;
  for (size_t i = 0; i < nc; ++i) hC[i] = 1.0f;

  cublasHandle_t h;
  CB(cublasCreate(&h));
  float *dA, *dB, *dC;
  CK(cudaMalloc(&dA, na * sizeof(float)));
  CK(cudaMalloc(&dB, nb * sizeof(float)));
  CK(cudaMalloc(&dC, nc * sizeof(float)));
  CK(cudaMemcpy(dA, hA, na * sizeof(float), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dB, hB, nb * sizeof(float), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dC, hC, nc * sizeof(float), cudaMemcpyHostToDevice));

  const float alpha = 2.0f, beta = 3.0f;
  CB(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, dA, lda, dB, ldb, &beta, dC, ldc));
  CK(cudaDeviceSynchronize());

  float got[64];
  CK(cudaMemcpy(got, dC, nc * sizeof(float), cudaMemcpyDeviceToHost));

  int wrong = 0;
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < m; ++i) {
      double acc = 0.0;
      for (int p = 0; p < k; ++p) acc += (double)hA[(size_t)p * lda + i] * hB[(size_t)j * ldb + p];
      const double want = alpha * acc + beta * hC[(size_t)j * ldc + i];
      const double have = got[(size_t)j * ldc + i];
      if (std::fabs(have - want) > 1e-4) {
        if (wrong < 4) std::printf("C[%d,%d] = %g want %g\n", i, j, have, want);
        ++wrong;
      }
    }

  CK(cudaFree(dA));
  CK(cudaFree(dB));
  CK(cudaFree(dC));
  CB(cublasDestroy(h));
  if (wrong) {
    std::printf("FAIL: %d entries wrong\n", wrong);
    return 1;
  }
  std::printf("PASS\n");
  return 0;
}
