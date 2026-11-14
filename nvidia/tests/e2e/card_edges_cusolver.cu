// cuSOLVER's dense argument checks and gesvd's 'O' jobs, against an RTX 3060's
// cuSOLVER 13.0.
//
//   - getrs refuses an unknown operation, negative sizes and short leading
//     dimensions (CUSOLVER_STATUS_INVALID_VALUE, info untouched). The simulator
//     solved with them: wrong answers, and n = -1 aborted the program with
//     std::length_error.
//   - getrf with lda < m succeeds with info = -4.
//   - potrf, potrs, syevd and gesvd refuse a bad argument with info = -(its
//     position), LAPACK's convention: an unknown fill mode or job, a negative
//     size, a short leading dimension. potrf with CUBLAS_FILL_MODE_FULL is
//     CUSOLVER_STATUS_EXECUTION_FAILED with info -1.
//   - gesvd with m < n is an invalid value with info 0 (the simulator said
//     NOT_SUPPORTED). jobu = 'O' or jobvt = 'O' (not both) is supported: the
//     left vectors overwrite A's columns, or V^T A's first n rows. The
//     simulator refused both with NOT_SUPPORTED.
//   - cusolverDnSetAdvOptions takes GETRF and POTRF with ALG_0..ALG_2 only.
//   - potrf of a matrix that is not positive definite (info = j, the failing leading minor) leaves the columns before
//     j factored and NaN at A(j,j). From the lower triangle the factorization goes on with column j taken as 0 (the
//     entries below the diagonal are 0); from the upper, rows j.. of the triangle are all NaN. The simulator left A
//     as it was. (Measured for n up to 32; larger matrices are factored in blocks on the card, which changes the
//     part after the failing column.)
//   - getrf without pivoting (devIpiv null) multiplies the column below a zero pivot by 0 and goes on; info is the
//     first zero pivot. The simulator skipped the column, leaving it as it was.
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <cmath>
#include <cstdio>
#include <vector>

static int fails = 0;
static int* info = nullptr;
#define WANT(x, want) do { const int got_ = static_cast<int>(x); if (got_ != static_cast<int>(want)) { \
  std::printf("FAIL line %d: %s -> %d, expected %d\n", __LINE__, #x, got_, static_cast<int>(want)); ++fails; } } while (0)
static int geti() { int v = 0; cudaMemcpy(&v, info, sizeof v, cudaMemcpyDeviceToHost); return v; }
static void seti(int v) { cudaMemcpy(info, &v, sizeof v, cudaMemcpyHostToDevice); }
// The call's status, and info afterwards (info is -77 before it).
#define CALL(x, want_status, want_info) do { seti(-77); WANT(x, want_status); WANT(geti(), want_info); } while (0)

static std::vector<float> get(const float* d, int n) {
  std::vector<float> h(n);
  cudaMemcpy(h.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost);
  return h;
}

int main() {
  cusolverDnHandle_t h;
  WANT(cusolverDnCreate(&h), CUSOLVER_STATUS_SUCCESS);
  WANT(cudaMalloc(&info, sizeof(int)), cudaSuccess);
  float *ws, *A, *B, *U, *V, *s;
  const int lw = 1 << 16;
  WANT(cudaMalloc(&ws, lw * sizeof(float)), cudaSuccess);
  for (float** p : {&A, &B, &U, &V, &s}) WANT(cudaMalloc(p, 64 * sizeof(float)), cudaSuccess);
  int* piv;
  WANT(cudaMalloc(&piv, 16 * sizeof(int)), cudaSuccess);
  const std::vector<float> spd = {4, 2, 0, 2, 3, 0, 0, 0, 5};
  auto reset = [&] {
    cudaMemcpy(A, spd.data(), spd.size() * sizeof(float), cudaMemcpyHostToDevice);
    const std::vector<float> b(9, 1);
    cudaMemcpy(B, b.data(), b.size() * sizeof(float), cudaMemcpyHostToDevice);
  };
  const auto N = CUBLAS_OP_N;
  const auto L = CUBLAS_FILL_MODE_LOWER;
  const auto bad = CUSOLVER_STATUS_INVALID_VALUE;
  const auto ok = CUSOLVER_STATUS_SUCCESS;

  // getrs
  reset();
  CALL(cusolverDnSgetrf(h, 3, 3, A, 3, ws, piv, info), ok, 0);
  CALL(cusolverDnSgetrs(h, N, -1, 1, A, 3, piv, B, 3, info), bad, -77);
  CALL(cusolverDnSgetrs(h, N, 3, -1, A, 3, piv, B, 3, info), bad, -77);
  CALL(cusolverDnSgetrs(h, N, 3, 1, A, 2, piv, B, 3, info), bad, -77);
  CALL(cusolverDnSgetrs(h, N, 3, 1, A, 3, piv, B, 2, info), bad, -77);
  CALL(cusolverDnSgetrs(h, static_cast<cublasOperation_t>(3), 3, 1, A, 3, piv, B, 3, info), bad, -77);
  CALL(cusolverDnSgetrs(h, N, 0, 1, A, 1, piv, B, 1, info), ok, -77);
  CALL(cusolverDnSgetrs(h, N, 3, 0, A, 3, piv, B, 3, info), ok, -77);
  CALL(cusolverDnSgetrs(h, N, 3, 1, A, 3, piv, B, 3, info), ok, 0);
  {
    const std::vector<float> x = get(B, 3);   // A x = 1: x = (0.125, 0.25, 0.2)
    WANT(std::fabs(x[0] - 0.125f) < 1e-5f && std::fabs(x[1] - 0.25f) < 1e-5f && std::fabs(x[2] - 0.2f) < 1e-5f, true);
  }
  // getrf
  reset();
  CALL(cusolverDnSgetrf(h, 3, 3, A, 2, ws, piv, info), ok, -4);
  CALL(cusolverDnSgetrf(h, 0, 3, A, 1, ws, piv, info), ok, 0);
  // potrf
  reset();
  CALL(cusolverDnSpotrf(h, CUBLAS_FILL_MODE_FULL, 3, A, 3, ws, lw, info), CUSOLVER_STATUS_EXECUTION_FAILED, -1);
  CALL(cusolverDnSpotrf(h, static_cast<cublasFillMode_t>(3), 3, A, 3, ws, lw, info), bad, -1);
  CALL(cusolverDnSpotrf(h, L, -1, A, 3, ws, lw, info), bad, -2);
  CALL(cusolverDnSpotrf(h, L, 3, A, 2, ws, lw, info), bad, -4);
  // potrs
  CALL(cusolverDnSpotrf(h, L, 3, A, 3, ws, lw, info), ok, 0);
  CALL(cusolverDnSpotrs(h, CUBLAS_FILL_MODE_FULL, 3, 1, A, 3, B, 3, info), bad, -1);
  CALL(cusolverDnSpotrs(h, L, -1, 1, A, 3, B, 3, info), bad, -2);
  CALL(cusolverDnSpotrs(h, L, 3, -1, A, 3, B, 3, info), bad, -3);
  CALL(cusolverDnSpotrs(h, L, 3, 1, A, 2, B, 3, info), bad, -5);
  CALL(cusolverDnSpotrs(h, L, 3, 1, A, 3, B, 2, info), bad, -7);
  CALL(cusolverDnSpotrs(h, L, 0, 1, A, 1, B, 1, info), ok, 0);
  // syevd
  reset();
  CALL(cusolverDnSsyevd(h, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_FULL, 3, A, 3, s, ws, lw, info), bad, -2);
  CALL(cusolverDnSsyevd(h, static_cast<cusolverEigMode_t>(2), L, 3, A, 3, s, ws, lw, info), bad, -1);
  CALL(cusolverDnSsyevd(h, CUSOLVER_EIG_MODE_VECTOR, L, 3, A, 2, s, ws, lw, info), bad, -5);
  CALL(cusolverDnSsyevd(h, CUSOLVER_EIG_MODE_VECTOR, L, -1, A, 3, s, ws, lw, info), bad, -3);
  // gesvd
  reset();
  CALL(cusolverDnSgesvd(h, 'a', 'N', 3, 3, A, 3, s, U, 3, V, 3, ws, lw, nullptr, info), bad, -1);
  CALL(cusolverDnSgesvd(h, 'N', 'x', 3, 3, A, 3, s, U, 3, V, 3, ws, lw, nullptr, info), bad, -2);
  CALL(cusolverDnSgesvd(h, 'O', 'O', 3, 3, A, 3, s, U, 3, V, 3, ws, lw, nullptr, info), bad, -2);
  CALL(cusolverDnSgesvd(h, 'N', 'N', 2, 3, A, 2, s, U, 1, V, 1, ws, lw, nullptr, info), bad, 0);
  CALL(cusolverDnSgesvd(h, 'N', 'N', 3, 3, A, 2, s, U, 1, V, 1, ws, lw, nullptr, info), bad, -6);
  CALL(cusolverDnSgesvd(h, 'N', 'N', 0, 0, A, 1, s, U, 1, V, 1, ws, lw, nullptr, info), ok, 0);
  {
    // 'O': the same vectors as 'S' / 'A', written over A.
    const std::vector<float> g = {1, 2, 3, 4, 5, 7};   // 3 x 2
    auto load = [&] { cudaMemcpy(A, g.data(), g.size() * sizeof(float), cudaMemcpyHostToDevice); };
    auto close = [](const std::vector<float>& x, const std::vector<float>& y, int rows, int ldx, int ldy, int cols) {
      for (int c = 0; c < cols; ++c)
        for (int r = 0; r < rows; ++r)
          if (std::fabs(x[c * ldx + r] - y[c * ldy + r]) > 1e-5f) return false;
      return true;
    };
    load();
    CALL(cusolverDnSgesvd(h, 'S', 'A', 3, 2, A, 3, s, U, 3, V, 2, ws, lw, nullptr, info), ok, 0);
    const std::vector<float> u = get(U, 6), vt = get(V, 4), sv = get(s, 2);
    WANT(std::fabs(sv[0] - 10.1815f) < 1e-3f && std::fabs(sv[1] - 0.5811f) < 1e-3f, true);
    load();
    CALL(cusolverDnSgesvd(h, 'O', 'A', 3, 2, A, 3, s, U, 3, V, 2, ws, lw, nullptr, info), ok, 0);
    WANT(close(get(A, 6), u, 3, 3, 3, 2), true);
    WANT(close(get(V, 4), vt, 2, 2, 2, 2), true);
    load();
    CALL(cusolverDnSgesvd(h, 'O', 'N', 3, 2, A, 3, s, U, 3, V, 2, ws, lw, nullptr, info), ok, 0);
    WANT(close(get(A, 6), u, 3, 3, 3, 2), true);
    load();
    CALL(cusolverDnSgesvd(h, 'S', 'O', 3, 2, A, 3, s, U, 3, V, 2, ws, lw, nullptr, info), ok, 0);
    WANT(close(get(U, 6), u, 3, 3, 3, 2), true);
    WANT(close(get(A, 6), vt, 2, 3, 2, 2), true);   // V^T in A's first two rows
    load();
    CALL(cusolverDnSgesvd(h, 'N', 'O', 3, 2, A, 3, s, U, 3, V, 2, ws, lw, nullptr, info), ok, 0);
    WANT(close(get(A, 6), vt, 2, 3, 2, 2), true);
    WANT(close(get(s, 2), sv, 2, 2, 2, 1), true);
  }
  // potrf of a matrix that is not positive definite: A(i,i) = n + 1 and 1 elsewhere, with A(f,f) = -5; the other
  // triangle holds 77, which potrf does not touch.
  {
    const double nan = std::nan("");
    auto same = [](double got, double want) {
      if (std::isnan(want)) return std::isnan(got);
      return std::fabs(got - want) <= 1e-5 * std::fabs(want) + 1e-6;
    };
    for (int x = 0; x < 2; ++x)   // cusolverDnDpotrf, cusolverDnXpotrf
      for (int n : {4, 8, 32})
        for (int f : {0, 1, n / 2, n - 1})
          for (int lower = 0; lower < 2; ++lower) {
            std::vector<double> a((size_t)n * n);
            for (int j = 0; j < n; ++j)
              for (int i = 0; i < n; ++i)
                a[i + (size_t)j * n] = (lower ? i < j : i > j) ? 77 : i == j ? n + 1 : 1;
            a[f + (size_t)f * n] = -5;
            // The expected factor: left-looking Cholesky, a failing column taken as 0 (lower) or NaN from there (upper).
            std::vector<double> w = a;
            auto el = [&](int i, int j) -> double& { return lower ? w[i + (size_t)j * n] : w[j + (size_t)i * n]; };
            for (int j = 0; j < n; ++j) {
              double d = el(j, j);
              for (int k = 0; k < j; ++k) d -= el(j, k) * el(j, k);
              if (!(d > 0)) {
                if (!lower) {
                  for (int r = j; r < n; ++r)
                    for (int c = r; c < n; ++c) el(c, r) = nan;
                  break;
                }
                el(j, j) = nan;
                for (int i = j + 1; i < n; ++i) el(i, j) = 0;
                continue;
              }
              d = std::sqrt(d);
              el(j, j) = d;
              for (int i = j + 1; i < n; ++i) {
                double t = el(i, j);
                for (int k = 0; k < j; ++k) t -= el(i, k) * el(j, k);
                el(i, j) = t / d;
              }
            }
            double* dA = nullptr;
            WANT(cudaMalloc(&dA, a.size() * sizeof(double)), cudaSuccess);
            WANT(cudaMemcpy(dA, a.data(), a.size() * sizeof(double), cudaMemcpyHostToDevice), cudaSuccess);
            const auto uplo = lower ? L : CUBLAS_FILL_MODE_UPPER;
            seti(-77);
            if (x == 0) {
              int dlw = 0;
              WANT(cusolverDnDpotrf_bufferSize(h, uplo, n, dA, n, &dlw), ok);
              double* dws = nullptr;
              WANT(cudaMalloc(&dws, (dlw + 1) * sizeof(double)), cudaSuccess);
              WANT(cusolverDnDpotrf(h, uplo, n, dA, n, dws, dlw, info), ok);
              WANT(cudaFree(dws), cudaSuccess);
            } else {
              cusolverDnParams_t xp;
              WANT(cusolverDnCreateParams(&xp), ok);
              size_t dsz = 0, hsz = 0;
              WANT(cusolverDnXpotrf_bufferSize(h, xp, uplo, n, CUDA_R_64F, dA, n, CUDA_R_64F, &dsz, &hsz), ok);
              void* dws = nullptr;
              WANT(cudaMalloc(&dws, dsz + 8), cudaSuccess);
              std::vector<char> hws(hsz + 8);
              WANT(cusolverDnXpotrf(h, xp, uplo, n, CUDA_R_64F, dA, n, CUDA_R_64F, dws, dsz, hws.data(), hsz, info), ok);
              WANT(cudaFree(dws), cudaSuccess);
              WANT(cusolverDnDestroyParams(xp), ok);
            }
            WANT(geti(), f + 1);
            std::vector<double> got(a.size());
            WANT(cudaMemcpy(got.data(), dA, got.size() * sizeof(double), cudaMemcpyDeviceToHost), cudaSuccess);
            int wrong = 0;
            for (size_t i = 0; i < got.size(); ++i) wrong += !same(got[i], w[i]);
            if (wrong) {
              std::printf("FAIL %s n=%d failing at %d %s: %d entries differ from the card's\n", x ? "Xpotrf" : "Dpotrf",
                          n, f + 1, lower ? "lower" : "upper", wrong);
              ++fails;
            }
            WANT(cudaFree(dA), cudaSuccess);
          }
    // Float, a small case exactly: 4 x 4, failing at 3.
    for (int lower = 0; lower < 2; ++lower) {
      std::vector<float> a(16);
      for (int j = 0; j < 4; ++j)
        for (int i = 0; i < 4; ++i) a[i + j * 4] = (lower ? i < j : i > j) ? 77 : i == j ? 5 : 1;
      a[10] = -5;
      cudaMemcpy(A, a.data(), 16 * sizeof(float), cudaMemcpyHostToDevice);
      CALL(cusolverDnSpotrf(h, lower ? L : CUBLAS_FILL_MODE_UPPER, 4, A, 4, ws, lw, info), ok, 3);
      const std::vector<float> g = get(A, 16);
      const bool col_or_row_ok = lower ? std::isnan(g[10]) && g[11] == 0 && std::fabs(g[15] - 2.1602468f) < 1e-5f
                                       : std::isnan(g[10]) && std::isnan(g[14]) && std::isnan(g[15]);
      WANT(col_or_row_ok && std::fabs(g[0] - 2.2360680f) < 1e-5f && g[1 + (lower ? 0 : 3)] != 77, true);
    }
  }

  // getrf without pivoting, on zero pivots.
  {
    auto run = [&](int n, const std::vector<float>& a, int want_info, const std::vector<float>& want) {
      cudaMemcpy(A, a.data(), a.size() * sizeof(float), cudaMemcpyHostToDevice);
      CALL(cusolverDnSgetrf(h, n, n, A, n, ws, nullptr, info), ok, want_info);
      const std::vector<float> g = get(A, n * n);
      for (int i = 0; i < n * n; ++i)
        if (g[i] != want[i] || std::signbit(g[i]) != std::signbit(want[i])) {
          std::printf("FAIL getrf without pivoting, zero pivot: A[%d] = %g, expected %g\n", i, g[i], want[i]);
          ++fails;
        }
    };
    run(3, {0, 1, 2, 1, 1, 1, 2, 1, 3}, 1, {0, 0, 0, 1, 1, 1, 2, 1, 2});
    run(3, {1, 2, 3, 2, 4, 5, 3, 7, 9}, 2, {1, 2, 3, 2, 0, -0.0f, 3, 1, 0});
    run(2, {0, 1, 1, 1}, 1, {0, 0, 1, 1});
    // Pivoted, a zero column: the same as before.
    cudaMemcpy(A, std::vector<float>{1, 2, 2, 4}.data(), 4 * sizeof(float), cudaMemcpyHostToDevice);
    CALL(cusolverDnSgetrf(h, 2, 2, A, 2, ws, piv, info), ok, 2);
  }

  // SetAdvOptions
  cusolverDnParams_t prm;
  WANT(cusolverDnCreateParams(&prm), ok);
  for (int fn = 0; fn < 4; ++fn)
    for (int alg = 0; alg < 4; ++alg)
      WANT(cusolverDnSetAdvOptions(prm, static_cast<cusolverDnFunction_t>(fn), static_cast<cusolverAlgMode_t>(alg)),
           fn <= 1 && alg <= 2 ? ok : bad);
  WANT(cusolverDnDestroyParams(prm), ok);

  for (float* p : {ws, A, B, U, V, s}) WANT(cudaFree(p), cudaSuccess);
  WANT(cudaFree(piv), cudaSuccess);
  WANT(cudaFree(info), cudaSuccess);
  WANT(cusolverDnDestroy(h), ok);
  std::printf("%s\n", fails ? "FAILED" : "PASS");
  return fails ? 1 : 0;
}
