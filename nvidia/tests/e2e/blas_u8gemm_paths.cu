// cublasUint8gemmBias, the deprecated 8-bit GEMM with offsets. Its header
// gives the signature and no formula; what follows is what an RTX 3060's
// cuBLAS 13.0 does, found by experiment, and every check here passes on that
// library too:
//
//   acc    = sum_p (op(A)[i,p] - A_bias) * (op(B)[p,j] - B_bias)
//   C[i,j] = clamp(round_half_up((acc + C_bias) * C_mult / 2^C_shift), 0, 255)
//
// C is not read; C_shift counts modulo 32 and 31 gives 0; a conjugate
// transpose is the transpose; a transposed C is stored n x m; bad sizes,
// leading dimensions and operations are INVALID_VALUE.
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

static int failures = 0;
static void check(bool ok, const char* what) {
  std::printf("%-4s %s\n", ok ? "ok" : "FAIL", what);
  if (!ok) ++failures;
}

static unsigned char model(int64_t acc, int c_bias, int c_mult, int c_shift) {
  const __int128 p = (__int128)(acc + c_bias) * c_mult;
  const int s = c_shift & 31;
  __int128 r;
  if (s == 0) r = p;
  else if (s == 31) r = 0;
  else r = (p + ((__int128)1 << (s - 1))) >> s;
  return (unsigned char)(r < 0 ? 0 : r > 255 ? 255 : r);
}

struct Rng {
  uint32_t s = 12345;
  int next(int n) {
    s = s * 1664525u + 1013904223u;
    return (int)((s >> 8) % (uint32_t)n);
  }
};

int main() {
  cublasHandle_t h;
  if (cublasCreate(&h)) { std::printf("FAIL cublasCreate\n"); return 1; }
  Rng rng;

  // Random shapes, operations, offsets, multipliers and shifts against the formula.
  long checked = 0, bad = 0;
  int status_failures = 0;
  for (int it = 0; it < 400; ++it) {
    const int ta = rng.next(3), tb = rng.next(3), tc = rng.next(2);
    const int m = 1 + rng.next(9), n = 1 + rng.next(9), k = 1 + rng.next(9);
    const int Ab = rng.next(40) - 5, Bb = rng.next(40) - 5, Cb = rng.next(200) - 50, Cm = 1 + rng.next(5), Cs = rng.next(6);
    const cublasOperation_t ops[3] = {CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C};
    const bool tA = ta != 0, tB = tb != 0, tC = tc != 0;
    const int ar = tA ? k : m, ac = tA ? m : k, br = tB ? n : k, bc = tB ? k : n, cr = tC ? n : m, cc = tC ? m : n;
    const int lda = ar + rng.next(3), ldb = br + rng.next(3), ldc = cr + rng.next(3);
    std::vector<unsigned char> A((size_t)lda * ac), B((size_t)ldb * bc), C((size_t)ldc * cc, 77);
    for (auto& x : A) x = (unsigned char)rng.next(64);
    for (auto& x : B) x = (unsigned char)rng.next(64);
    unsigned char *dA, *dB, *dC;
    cudaMalloc(&dA, A.size());
    cudaMalloc(&dB, B.size());
    cudaMalloc(&dC, C.size());
    cudaMemcpy(dA, A.data(), A.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B.data(), B.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dC, C.data(), C.size(), cudaMemcpyHostToDevice);
    const cublasStatus_t st = cublasUint8gemmBias(h, ops[ta], ops[tb], tC ? CUBLAS_OP_T : CUBLAS_OP_N, m, n, k, dA, Ab, lda,
                                                  dB, Bb, ldb, dC, Cb, ldc, Cm, Cs);
    cudaDeviceSynchronize();
    if (st) ++status_failures;
    cudaMemcpy(C.data(), dC, C.size(), cudaMemcpyDeviceToHost);
    for (int i = 0; i < m; ++i)
      for (int j = 0; j < n; ++j) {
        int64_t acc = 0;
        for (int p = 0; p < k; ++p) {
          const int a = tA ? A[(size_t)p + (size_t)i * lda] : A[(size_t)i + (size_t)p * lda];
          const int b = tB ? B[(size_t)j + (size_t)p * ldb] : B[(size_t)p + (size_t)j * ldb];
          acc += (int64_t)(a - Ab) * (b - Bb);
        }
        const unsigned char want = model(acc, Cb, Cm, Cs);
        const unsigned char got = tC ? C[(size_t)j + (size_t)i * ldc] : C[(size_t)i + (size_t)j * ldc];
        ++checked;
        if (want != got) ++bad;
      }
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
  }
  {
    char what[160];
    std::snprintf(what, sizeof what, "400 random calls, %ld outputs, all equal to the formula (%ld differ, %d bad statuses)", checked, bad,
                  status_failures);
    check(bad == 0 && status_failures == 0, what);
  }

  // Corners of the arithmetic: rounding, saturation, the shift of 31, k = 0.
  {
    unsigned char hA[4] = {10, 20, 30, 40}, hB[4] = {1, 3, 2, 4}, hC[4];
    unsigned char *dA, *dB, *dC;
    cudaMalloc(&dA, 4);
    cudaMalloc(&dB, 4);
    cudaMalloc(&dC, 4);
    cudaMemcpy(dA, hA, 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, 4, cudaMemcpyHostToDevice);
    auto run = [&](int k, int Ab, int Bb, int Cb, int Cm, int Cs) {
      cudaMemset(dC, 0xEE, 4);
      cublasUint8gemmBias(h, CUBLAS_OP_N, CUBLAS_OP_N, CUBLAS_OP_N, 2, 2, k, dA, Ab, 2, dB, Bb, 2, dC, Cb, 2, Cm, Cs);
      cudaDeviceSynchronize();
      cudaMemcpy(hC, dC, 4, cudaMemcpyDeviceToHost);
    };
    // Row 0 of A is (10, 30), column 0 of B is (1, 3): entry (0,0) is 10 + 90 = 100.
    run(2, 0, 0, 0, 1, 1);
    check(hC[0] == 50, "100 >> 1 = 50");
    run(2, 0, 0, 0, 1, 3);
    check(hC[0] == 13, "100 / 8 = 12.5 rounds up to 13");
    run(2, 0, 0, 0, 3, 0);
    check(hC[0] == 255, "300 saturates at 255");
    run(2, 0, 0, 0, -1, 0);
    check(hC[0] == 0, "a negative result saturates at 0");
    run(2, 0, 0, 5, 3, 3);
    check(hC[0] == 39, "C_bias is added before the multiplier and the shift: (100 + 5) * 3 / 8 = 39");
    run(2, 0, 0, 0, 1, 31);
    check(hC[0] == 0, "a shift of 31 gives 0");
    run(2, 0, 0, 0, 1, 32);
    check(hC[0] == 100, "a shift of 32 is a shift of 0");
    run(2, 0, 0, 0, 1, 33);
    check(hC[0] == 50, "a shift of 33 is a shift of 1");
    run(0, 0, 0, 10, 3, 2);
    check(hC[0] == 8 && hC[3] == 8, "k = 0: the output is (C_bias * C_mult) >> C_shift, rounded: 30 / 4 = 7.5 -> 8");
    run(2, 2, 1, 0, 1, 0);
    check(hC[0] == 8 * 0 + (10 - 2) * (1 - 1) + (30 - 2) * (3 - 1), "A_bias and B_bias are subtracted from the operands");
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
  }

  // Arguments: INVALID_VALUE for a bad operation, size or leading dimension; empty outputs are fine.
  {
    unsigned char* d;
    cudaMalloc(&d, 4096);
    auto call = [&](int ta, int tb, int tc, int m, int n, int k, int lda, int ldb, int ldc) {
      return (int)cublasUint8gemmBias(h, (cublasOperation_t)ta, (cublasOperation_t)tb, (cublasOperation_t)tc, m, n, k, d, 0, lda, d, 0, ldb,
                                      d + 2048, 0, ldc, 1, 0);
    };
    std::fflush(stdout);
    check(call(9, 0, 0, 2, 2, 2, 2, 2, 2) == CUBLAS_STATUS_INVALID_VALUE, "an unknown operation is INVALID_VALUE");
    check(call(0, 0, 0, -1, 2, 2, 2, 2, 2) == CUBLAS_STATUS_INVALID_VALUE, "a negative m is INVALID_VALUE");
    check(call(0, 0, 0, 4, 2, 2, 2, 2, 4) == CUBLAS_STATUS_INVALID_VALUE, "lda below m is INVALID_VALUE");
    check(call(1, 0, 0, 2, 2, 4, 2, 4, 2) == CUBLAS_STATUS_INVALID_VALUE, "lda below k for a transposed A is INVALID_VALUE");
    check(call(0, 0, 0, 2, 2, 4, 2, 2, 2) == CUBLAS_STATUS_INVALID_VALUE, "ldb below k is INVALID_VALUE");
    check(call(0, 0, 0, 4, 2, 2, 4, 2, 2) == CUBLAS_STATUS_INVALID_VALUE, "ldc below m is INVALID_VALUE");
    check(call(0, 0, 1, 2, 4, 2, 2, 2, 2) == CUBLAS_STATUS_INVALID_VALUE, "ldc below n for a transposed C is INVALID_VALUE");
    check(call(0, 0, 0, 0, 2, 2, 0, 2, 1) == CUBLAS_STATUS_INVALID_VALUE, "lda 0 is INVALID_VALUE even for m = 0 (it must be at least 1)");
    check(call(0, 0, 0, 0, 3, 3, 1, 3, 1) == CUBLAS_STATUS_SUCCESS, "m = 0 with leading dimensions of 1 succeeds");
    cudaFree(d);
  }
  cublasDestroy(h);
  std::printf("%s\n", failures ? "FAIL" : "PASS");
  return failures ? 1 : 0;
}
