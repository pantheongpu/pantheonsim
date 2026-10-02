// The cuBLAS level-1, -2 and -3 routines that were stubs until the rest of
// their families were in: real copy, swap and i?amin; the complex i?amax,
// i?amin, sums of magnitudes and norms; ger, symv, trmv and trsv in real
// precision; the rank updates syr, syr2, her and her2 and their rank-k forms
// syr2k and her2k; real symm, syrk and trmm; and the double-precision batched
// GEMMs. Each is checked against a host reference over small integers, which
// every order of summation gets exactly right, so the comparisons are exact;
// and the arguments each refuses, as an RTX 3060 answers. Every check passes
// on the card too.
#include <cublas_v2.h>
#include <cuComplex.h>
#include <cuda_runtime.h>

#include <cmath>
#include <complex>
#include <cstdio>
#include <vector>

using cd = std::complex<double>;
static int failures = 0;

static void check(bool ok, const char* what) {
  std::printf("%-4s %s\n", ok ? "ok" : "FAIL", what);
  if (!ok) ++failures;
}
#define IS(call, want) check((int)(call) == (int)(want), #call " -> " #want)

template <class T> static T* dev(const std::vector<T>& h) {
  T* d = nullptr;
  cudaMalloc(&d, h.size() * sizeof(T) + 16);
  cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
  return d;
}
template <class T> static std::vector<T> host(const T* d, size_t n) {
  std::vector<T> h(n);
  cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost);
  return h;
}

// Every element type through std::complex<double>.
static cd C(float v) { return v; }
static cd C(double v) { return v; }
static cd C(cuComplex v) { return {v.x, v.y}; }
static cd C(cuDoubleComplex v) { return {v.x, v.y}; }
template <class T> static T to(cd v);
template <> float to<float>(cd v) { return (float)v.real(); }
template <> double to<double>(cd v) { return v.real(); }
template <> cuComplex to<cuComplex>(cd v) { return make_cuComplex((float)v.real(), (float)v.imag()); }
template <> cuDoubleComplex to<cuDoubleComplex>(cd v) { return make_cuDoubleComplex(v.real(), v.imag()); }
template <class T> static constexpr bool is_complex = sizeof(T) == 2 * sizeof(decltype(C(T{}).real()));
// Small integers (and complex ones), a different pattern for each seed.
template <class T> static std::vector<T> ints(size_t n, int seed) {
  std::vector<T> v(n);
  for (size_t i = 0; i < n; ++i) v[i] = to<T>(cd((int)((i * 7 + seed * 3) % 9) - 4, (int)((i * 5 + seed) % 7) - 3));
  return v;
}
template <class T> static std::vector<cd> wide(const std::vector<T>& v) {
  std::vector<cd> w(v.size());
  for (size_t i = 0; i < v.size(); ++i) w[i] = C(v[i]);
  return w;
}
template <class T> static bool equal(const std::vector<T>& got, const std::vector<cd>& want) {
  for (size_t i = 0; i < got.size(); ++i)
    if (C(got[i]) != C(to<T>(want[i]))) return false;
  return true;
}
// BLAS's walk of a strided vector: element i of n, a negative increment from
// the far end.
static size_t at(int i, int n, int inc) { return inc > 0 ? (size_t)i * inc : (size_t)(n - 1 - i) * -inc; }

// ---- level 1 ----
static void level1(cublasHandle_t h) {
  const int n = 6;
  std::vector<float> x = {3, -1, 4, -1, 5, -9, 2, 6, 5, 3, 5, -8}, y(12, 0);
  float* dx = dev(x);
  float* dy = dev(y);
  IS(cublasScopy(h, n, dx, -2, dy, 1), CUBLAS_STATUS_SUCCESS);
  std::vector<float> want(12, 0);
  for (int i = 0; i < n; ++i) want[i] = x[at(i, n, -2)];
  check(host(dy, 12) == want, "scopy with incx -2 walks x from its end");
  IS(cublasScopy(h, n, dx, 0, dy, 1), CUBLAS_STATUS_SUCCESS);
  check(host(dy, n) == std::vector<float>(n, 3), "scopy with incx 0 copies x[0] everywhere");
  IS(cublasScopy(h, -1, dx, 1, dy, 1), CUBLAS_STATUS_SUCCESS);
  IS(cublasSswap(h, n, dx, 1, dy, 2), CUBLAS_STATUS_SUCCESS);
  const auto sx = host(dx, 12), sy = host(dy, 12);
  check(sx[0] == 3 && sx[1] == 3 && sy[0] == 3 && sy[2] == -1 && sy[10] == -9, "sswap");
  cudaMemcpy(dx, x.data(), 48, cudaMemcpyHostToDevice);
  int r = -1;
  IS(cublasIsamin(h, 12, dx, 1, &r), CUBLAS_STATUS_SUCCESS);
  check(r == 2, "isamin: the first smallest magnitude, 1-based");
  IS(cublasIsamin(h, 6, dx, 2, &r), CUBLAS_STATUS_SUCCESS);
  check(r == 4, "isamin with incx 2");
  IS(cublasIsamin(h, 6, dx, 0, &r), CUBLAS_STATUS_SUCCESS);
  check(r == 0, "isamin with incx 0 is 0");
  int64_t r64 = -1;
  IS(cublasIsamin_v2_64(h, 12, dx, 1, &r64), CUBLAS_STATUS_SUCCESS);
  check(r64 == 2, "cublasIsamin_v2_64");
  std::vector<double> xd(x.begin(), x.end());
  double* dxd = dev(xd);
  double* dyd = dev(std::vector<double>(12, 0));
  IS(cublasIdamin(h, 12, dxd, 1, &r), CUBLAS_STATUS_SUCCESS);
  check(r == 2, "idamin");
  IS(cublasDcopy(h, 12, dxd, 1, dyd, 1), CUBLAS_STATUS_SUCCESS);
  IS(cublasDswap_v2_64(h, 12, dxd, 1, dyd, -1), CUBLAS_STATUS_SUCCESS);
  check(host(dxd, 1)[0] == -8 && host(dyd, 1)[0] == -8, "dcopy, and dswap_64 with incy -1");

  // Complex: |re| + |im| ranks the elements, as BLAS's icamax does.
  std::vector<cuComplex> xc = {{1, -1}, {0, 3}, {-2, -2}, {3, 0}, {0, -1}, {-4, 0}};
  cuComplex* dxc = dev(xc);
  IS(cublasIcamax(h, n, dxc, 1, &r), CUBLAS_STATUS_SUCCESS);
  check(r == 3, "icamax by |re| + |im|: (-2, -2) before (3, 0)");
  IS(cublasIcamin(h, n, dxc, 1, &r), CUBLAS_STATUS_SUCCESS);
  check(r == 5, "icamin");
  float rf = -1;
  IS(cublasScasum(h, n, dxc, 1, &rf), CUBLAS_STATUS_SUCCESS);
  check(rf == 17, "scasum: the sum of |re| + |im|");
  IS(cublasScnrm2(h, n, dxc, 1, &rf), CUBLAS_STATUS_SUCCESS);
  check(std::fabs(rf - std::sqrt(45.0f)) < 1e-5f, "scnrm2");
  IS(cublasScnrm2(h, n, dxc, -1, &rf), CUBLAS_STATUS_SUCCESS);
  check(rf == 0, "scnrm2 with a negative increment is 0");
  std::vector<cuDoubleComplex> xz = {{1, -1}, {0, 3}, {-2, -2}, {3, 0}, {0, -1}, {-4, 0}};
  cuDoubleComplex* dxz = dev(xz);
  double rd = -1;
  IS(cublasIzamax(h, n, dxz, 1, &r), CUBLAS_STATUS_SUCCESS);
  IS(cublasIzamin_v2_64(h, n, dxz, 1, &r64), CUBLAS_STATUS_SUCCESS);
  check(r == 3 && r64 == 5, "izamax, izamin_64");
  IS(cublasDzasum(h, n, dxz, 1, &rd), CUBLAS_STATUS_SUCCESS);
  check(rd == 17, "dzasum");
  IS(cublasDznrm2_v2_64(h, n, dxz, 1, &rd), CUBLAS_STATUS_SUCCESS);
  check(std::fabs(rd - std::sqrt(45.0)) < 1e-14, "dznrm2_64");
  // The card refuses no size or increment here: a negative n does nothing,
  // and a zero increment repeats x[0].
  const cuComplex two = make_cuComplex(2, 0);
  cuComplex* dyc = dev(std::vector<cuComplex>(n, make_cuComplex(1, 1)));
  IS(cublasCaxpy(h, -1, &two, dxc, 1, dyc, 1), CUBLAS_STATUS_SUCCESS);
  IS(cublasCaxpy(h, n, &two, dxc, 0, dyc, 1), CUBLAS_STATUS_SUCCESS);
  const auto yc = host(dyc, n);
  check(yc[0].x == 3 && yc[0].y == -1 && yc[5].x == 3 && yc[5].y == -1, "caxpy with incx 0 adds 2 x[0] to each");
  IS(cublasCcopy(h, -1, dxc, 1, dyc, 1), CUBLAS_STATUS_SUCCESS);
  IS(cublasCswap(h, -1, dxc, 1, dyc, 1), CUBLAS_STATUS_SUCCESS);
  IS(cublasCscal(h, -1, &two, dxc, 1), CUBLAS_STATUS_SUCCESS);
  for (void* p : {(void*)dx, (void*)dy, (void*)dxd, (void*)dyd, (void*)dxc, (void*)dxz, (void*)dyc}) cudaFree(p);
}

// ---- level 2 ----
// The rank updates against a host reference: A += alpha x op(x) (rank 1) or
// alpha x op(y) + alpha' y op(x) (rank 2), only `lower`'s triangle, the
// diagonal real for the Hermitian forms.
static std::vector<cd> rank_ref(std::vector<cd> a, int n, int lda, bool lower, bool herm, cd alpha,
                                const std::vector<cd>& x, int incx, const std::vector<cd>* y, int incy) {
  auto op = [&](cd v) { return herm ? std::conj(v) : v; };
  for (int j = 0; j < n; ++j)
    for (int i = lower ? j : 0; i < (lower ? n : j + 1); ++i) {
      const cd xi = x[at(i, n, incx)], xj = x[at(j, n, incx)];
      cd& e = a[(size_t)j * lda + i];
      if (y) e += alpha * xi * op((*y)[at(j, n, incy)]) + op(alpha) * (*y)[at(i, n, incy)] * op(xj);
      else e += alpha * xi * op(xj);
      if (herm && i == j) e = e.real();
    }
  return a;
}

template <class T>
static void rank_updates(cublasHandle_t h, const char* name) {
  const int n = 4, lda = 5;
  const auto A = ints<T>(lda * n, 1), x = ints<T>(2 * n, 2), y = ints<T>(n, 3);
  T* dA = dev(A);
  T* dx = dev(x);
  T* dy = dev(y);
  const T al = to<T>(cd(2, -1));
  char what[96];
  for (bool lower : {true, false}) {
    const cublasFillMode_t u = lower ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
    cudaMemcpy(dA, A.data(), A.size() * sizeof(T), cudaMemcpyHostToDevice);
    cublasStatus_t st;
    if constexpr (std::is_same_v<T, float>) st = cublasSsyr(h, u, n, &al, dx, -2, dA, lda);
    else if constexpr (std::is_same_v<T, double>) st = cublasDsyr(h, u, n, &al, dx, -2, dA, lda);
    else if constexpr (std::is_same_v<T, cuComplex>) st = cublasCsyr(h, u, n, &al, dx, -2, dA, lda);
    else st = cublasZsyr(h, u, n, &al, dx, -2, dA, lda);
    std::snprintf(what, sizeof what, "%ssyr, %s, incx -2", name, lower ? "lower" : "upper");
    check(st == 0 && equal(host(dA, A.size()), rank_ref(wide(A), n, lda, lower, false, C(al), wide(x), -2, nullptr, 0)),
          what);
    cudaMemcpy(dA, A.data(), A.size() * sizeof(T), cudaMemcpyHostToDevice);
    if constexpr (std::is_same_v<T, float>) st = cublasSsyr2(h, u, n, &al, dx, 1, dy, 1, dA, lda);
    else if constexpr (std::is_same_v<T, double>) st = cublasDsyr2(h, u, n, &al, dx, 1, dy, 1, dA, lda);
    else if constexpr (std::is_same_v<T, cuComplex>) st = cublasCsyr2(h, u, n, &al, dx, 1, dy, 1, dA, lda);
    else st = cublasZsyr2(h, u, n, &al, dx, 1, dy, 1, dA, lda);
    const auto wy = wide(y);
    std::snprintf(what, sizeof what, "%ssyr2, %s", name, lower ? "lower" : "upper");
    check(st == 0 && equal(host(dA, A.size()), rank_ref(wide(A), n, lda, lower, false, C(al), wide(x), 1, &wy, 1)),
          what);
    if constexpr (is_complex<T>) {
      using R = decltype(C(T{}).real());
      const auto real_al = (std::conditional_t<std::is_same_v<T, cuComplex>, float, double>)3;
      cudaMemcpy(dA, A.data(), A.size() * sizeof(T), cudaMemcpyHostToDevice);
      if constexpr (std::is_same_v<T, cuComplex>) st = cublasCher(h, u, n, &real_al, dx, 1, dA, lda);
      else st = cublasZher(h, u, n, &real_al, dx, 1, dA, lda);
      std::snprintf(what, sizeof what, "%sher, %s: the diagonal comes out real", name, lower ? "lower" : "upper");
      check(st == 0 && equal(host(dA, A.size()), rank_ref(wide(A), n, lda, lower, true, (R)3, wide(x), 1, nullptr, 0)),
            what);
      cudaMemcpy(dA, A.data(), A.size() * sizeof(T), cudaMemcpyHostToDevice);
      if constexpr (std::is_same_v<T, cuComplex>) st = cublasCher2(h, u, n, &al, dx, 2, dy, -1, dA, lda);
      else st = cublasZher2(h, u, n, &al, dx, 2, dy, -1, dA, lda);
      std::snprintf(what, sizeof what, "%sher2, %s: alpha x y^H + conj(alpha) y x^H", name, lower ? "lower" : "upper");
      check(st == 0 && equal(host(dA, A.size()), rank_ref(wide(A), n, lda, lower, true, C(al), wide(x), 2, &wy, -1)),
            what);
    }
  }
  for (void* p : {(void*)dA, (void*)dx, (void*)dy}) cudaFree(p);
}

static void level2(cublasHandle_t h) {
  const int m = 3, n = 4, lda = 5;
  const auto A = ints<float>(lda * n, 4), x = ints<float>(2 * m, 5), y = ints<float>(n, 6);
  float* dA = dev(A);
  float* dx = dev(x);
  float* dy = dev(y);
  const float two = 2, half = 0.5f;
  IS(cublasSger(h, m, n, &two, dx, 2, dy, 1, dA, lda), CUBLAS_STATUS_SUCCESS);
  {
    auto w = wide(A);
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < m; ++i) w[j * lda + i] += 2.0 * x[2 * i] * y[j];
    check(equal(host(dA, A.size()), w), "sger: A += alpha x y^T, the rows past m untouched");
  }
  IS(cublasSger(h, m, n, &two, dx, 1, dy, 1, dA, 2), CUBLAS_STATUS_INVALID_VALUE);
  IS(cublasSger(h, m, n, &two, dx, 0, dy, 1, dA, lda), CUBLAS_STATUS_INVALID_VALUE);
  IS(cublasSger(h, 0, n, &two, dx, 1, dy, 1, dA, 0), CUBLAS_STATUS_INVALID_VALUE);
  std::vector<double> Ad(A.begin(), A.end()), xd(x.begin(), x.end()), yd(y.begin(), y.end());
  double* dAd = dev(Ad);
  double* dxd = dev(xd);
  double* dyd = dev(yd);
  const double twod = 2;
  IS(cublasDger(h, m, n, &twod, dxd, 1, dyd, -1, dAd, lda), CUBLAS_STATUS_SUCCESS);
  check(host(dAd, 1)[0] == Ad[0] + 2 * xd[0] * yd[n - 1], "dger with incy -1");

  // symv: y = alpha A x + beta y, A symmetric from one triangle.
  for (bool lower : {true, false}) {
    cudaMemcpy(dA, A.data(), A.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dy, y.data(), y.size() * 4, cudaMemcpyHostToDevice);
    IS(cublasSsymv(h, lower ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER, m, &two, dA, lda, dx, 2, &half, dy, 1),
       CUBLAS_STATUS_SUCCESS);
    std::vector<cd> w = wide(y);
    for (int i = 0; i < m; ++i) {
      double s = 0;
      for (int j = 0; j < m; ++j) {
        const bool stored = lower ? i >= j : i <= j;
        s += (stored ? A[j * lda + i] : A[i * lda + j]) * x[2 * j];
      }
      w[i] = 2 * s + 0.5 * y[i];
    }
    check(equal(host(dy, n), w), lower ? "ssymv from the lower triangle" : "ssymv from the upper triangle");
  }
  IS(cublasSsymv(h, CUBLAS_FILL_MODE_FULL, m, &two, dA, lda, dx, 1, &half, dy, 1), CUBLAS_STATUS_INVALID_VALUE);
  IS(cublasSsymv(h, CUBLAS_FILL_MODE_LOWER, m, &two, dA, 2, dx, 1, &half, dy, 1), CUBLAS_STATUS_INVALID_VALUE);
  IS(cublasSsymv(h, CUBLAS_FILL_MODE_LOWER, 0, &two, dA, 0, dx, 1, &half, dy, 1), CUBLAS_STATUS_SUCCESS);
  IS(cublasDsymv(h, CUBLAS_FILL_MODE_UPPER, m, &twod, dAd, lda, dxd, 1, &twod, dyd, 0), CUBLAS_STATUS_INVALID_VALUE);

  rank_updates<float>(h, "s");
  rank_updates<double>(h, "d");
  rank_updates<cuComplex>(h, "c");
  rank_updates<cuDoubleComplex>(h, "z");
  IS(cublasSsyr(h, CUBLAS_FILL_MODE_LOWER, 0, &two, dx, 1, dA, 0), CUBLAS_STATUS_INVALID_VALUE);
  IS(cublasSsyr2(h, CUBLAS_FILL_MODE_FULL, m, &two, dx, 1, dy, 1, dA, lda), CUBLAS_STATUS_INVALID_VALUE);
  IS(cublasSsyr2(h, CUBLAS_FILL_MODE_LOWER, m, &two, dx, 1, dy, 0, dA, lda), CUBLAS_STATUS_INVALID_VALUE);

  // trmv and trsv: a unit or integer triangle; trsv undoes trmv exactly.
  std::vector<float> T(lda * n, 99);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) T[j * lda + i] = i == j ? 2.0f : (float)((i + 2 * j) % 3 - 1);
  float* dT = dev(T);
  for (int lower = 0; lower < 2; ++lower)
    for (cublasOperation_t t : {CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C})
      for (cublasDiagType_t d : {CUBLAS_DIAG_NON_UNIT, CUBLAS_DIAG_UNIT}) {
        const cublasFillMode_t u = lower ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
        std::vector<float> v = {1, -2, 3, 1, 0, 2, -1, 1};
        float* dv = dev(v);
        const int st = cublasStrmv(h, u, t, d, n, dT, lda, dv, -2);
        std::vector<float> w = v;
        for (int i = 0; i < n; ++i) {
          double s = 0;
          for (int j = 0; j < n; ++j) {
            const int r = t == CUBLAS_OP_N ? i : j, c = t == CUBLAS_OP_N ? j : i;
            if (lower ? c > r : c < r) continue;
            s += (r == c && d == CUBLAS_DIAG_UNIT ? 1.0f : T[c * lda + r]) * v[at(j, n, -2)];
          }
          w[at(i, n, -2)] = (float)s;
        }
        const bool mv = st == 0 && host(dv, 8) == w;
        const int st2 = cublasStrsv(h, u, t, d, n, dT, lda, dv, -2);
        const bool sv = st2 == 0 && host(dv, 8) == v;
        char what[96];
        std::snprintf(what, sizeof what, "strmv then strsv, %s, op %d, %s diagonal, incx -2", lower ? "lower" : "upper",
                      (int)t, d == CUBLAS_DIAG_UNIT ? "unit" : "stored");
        check(mv && sv, what);
        cudaFree(dv);
      }
  std::vector<double> Td(T.begin(), T.end()), vd = {1, 2, 3, 4};
  double* dTd = dev(Td);
  double* dvd = dev(vd);
  IS(cublasDtrmv(h, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, dTd, lda, dvd, 1),
     CUBLAS_STATUS_SUCCESS);
  IS(cublasDtrsv(h, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, dTd, lda, dvd, 1),
     CUBLAS_STATUS_SUCCESS);
  check(host(dvd, 4) == vd, "dtrmv then dtrsv");
  IS(cublasStrmv(h, CUBLAS_FILL_MODE_FULL, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, dT, lda, dx, 1),
     CUBLAS_STATUS_INVALID_VALUE);
  IS(cublasStrmv(h, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, (cublasDiagType_t)5, n, dT, lda, dx, 1),
     CUBLAS_STATUS_INVALID_VALUE);
  IS(cublasStrsv(h, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, dT, 3, dx, 1),
     CUBLAS_STATUS_INVALID_VALUE);
  IS(cublasStrsv(h, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, dT, lda, dx, 0),
     CUBLAS_STATUS_INVALID_VALUE);
  IS(cublasStrmv(h, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, 0, dT, 0, dx, 1),
     CUBLAS_STATUS_INVALID_VALUE);
  for (void* p : {(void*)dA, (void*)dx, (void*)dy, (void*)dAd, (void*)dxd, (void*)dyd, (void*)dT, (void*)dTd,
                  (void*)dvd})
    cudaFree(p);
}

// ---- level 3 ----
// C = alpha (op(A) op(B)^? + ...) + beta C over one triangle: syrk, syr2k,
// her2k.
template <class T>
static std::vector<cd> rk_ref(std::vector<cd> c, int n, int k, int ldc, bool lower, bool herm, bool trans,
                              const std::vector<cd>& a, const std::vector<cd>* b, int ld, cd alpha, cd beta) {
  auto el = [&](const std::vector<cd>& m, int i, int p) { return trans ? m[(size_t)i * ld + p] : m[(size_t)p * ld + i]; };
  auto op = [&](cd v) { return herm ? std::conj(v) : v; };
  for (int j = 0; j < n; ++j)
    for (int i = lower ? j : 0; i < (lower ? n : j + 1); ++i) {
      cd s = 0;
      for (int p = 0; p < k; ++p) {
        if (!b) s += alpha * el(a, i, p) * el(a, j, p);
        else if (trans && herm) s += alpha * std::conj(el(a, i, p)) * el(*b, j, p) + op(alpha) * std::conj(el(*b, i, p)) * el(a, j, p);
        else s += alpha * el(a, i, p) * op(el(*b, j, p)) + op(alpha) * el(*b, i, p) * op(el(a, j, p));
      }
      cd& e = c[(size_t)j * ldc + i];
      e = s + beta * e;
      if (herm && i == j) e = e.real();
    }
  return c;
}

static void level3(cublasHandle_t h) {
  const int m = 3, n = 4, k = 2, ld = 5;
  const auto A = ints<float>(ld * 5, 7), B = ints<float>(ld * 5, 8), C0 = ints<float>(ld * 5, 9);
  float* dA = dev(A);
  float* dB = dev(B);
  float* dC = dev(C0);
  const float two = 2, mone = -1;
  auto reset = [&] { cudaMemcpy(dC, C0.data(), C0.size() * 4, cudaMemcpyHostToDevice); };
  // symm, from either side and either triangle.
  for (cublasSideMode_t s : {CUBLAS_SIDE_LEFT, CUBLAS_SIDE_RIGHT})
    for (bool lower : {true, false}) {
      reset();
      const int st = cublasSsymm(h, s, lower ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER, m, n, &two, dA, ld, dB,
                                 ld, &mone, dC, ld);
      auto sym = [&](int i, int j) { return (lower ? i >= j : i <= j) ? A[j * ld + i] : A[i * ld + j]; };
      std::vector<cd> w = wide(C0);
      for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i) {
          double acc = 0;
          if (s == CUBLAS_SIDE_LEFT) for (int p = 0; p < m; ++p) acc += sym(i, p) * B[j * ld + p];
          else for (int p = 0; p < n; ++p) acc += B[p * ld + i] * sym(p, j);
          w[j * ld + i] = 2 * acc - C0[j * ld + i];
        }
      check(st == 0 && equal(host(dC, C0.size()), w),
            s == CUBLAS_SIDE_LEFT ? (lower ? "ssymm, A on the left, lower" : "ssymm, A on the left, upper")
                                  : (lower ? "ssymm, A on the right, lower" : "ssymm, A on the right, upper"));
    }
  IS(cublasSsymm(h, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, m, n, &two, dA, 3, dB, ld, &mone, dC, ld),
     CUBLAS_STATUS_INVALID_VALUE);   // A is n x n from the right
  IS(cublasSsymm(h, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_FULL, m, n, &two, dA, ld, dB, ld, &mone, dC, ld),
     CUBLAS_STATUS_INVALID_VALUE);
  IS(cublasSsymm(h, (cublasSideMode_t)4, CUBLAS_FILL_MODE_LOWER, m, n, &two, dA, ld, dB, ld, &mone, dC, ld),
     CUBLAS_STATUS_INVALID_VALUE);

  // syrk and syr2k; OP_C is OP_T for real matrices.
  const auto wa = wide(A), wb = wide(B);
  for (cublasOperation_t t : {CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C})
    for (bool lower : {true, false}) {
      const cublasFillMode_t u = lower ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
      reset();
      int st = cublasSsyrk(h, u, t, n, k, &two, dA, ld, &mone, dC, ld);
      const bool ok1 = st == 0 && equal(host(dC, C0.size()), rk_ref<float>(wide(C0), n, k, ld, lower, false,
                                                                           t != CUBLAS_OP_N, wa, nullptr, ld, 2, -1));
      reset();
      st = cublasSsyr2k(h, u, t, n, k, &two, dA, ld, dB, ld, &mone, dC, ld);
      const bool ok2 = st == 0 && equal(host(dC, C0.size()), rk_ref<float>(wide(C0), n, k, ld, lower, false,
                                                                           t != CUBLAS_OP_N, wa, &wb, ld, 2, -1));
      char what[80];
      std::snprintf(what, sizeof what, "ssyrk and ssyr2k, op %d, %s", (int)t, lower ? "lower" : "upper");
      check(ok1 && ok2, what);
    }
  IS(cublasSsyrk(h, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, n, ld + 1, &two, dA, ld, &mone, dC, ld),
     CUBLAS_STATUS_INVALID_VALUE);   // op(A) = A^T is n x k: lda must reach k
  IS(cublasSsyr2k(h, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k, &two, dA, ld, dB, 3, &mone, dC, ld),
     CUBLAS_STATUS_INVALID_VALUE);
  IS(cublasSsyr2k(h, CUBLAS_FILL_MODE_FULL, CUBLAS_OP_N, n, k, &two, dA, ld, dB, ld, &mone, dC, ld),
     CUBLAS_STATUS_INVALID_VALUE);
  std::vector<double> Ad(A.begin(), A.end()), Cd(C0.begin(), C0.end());
  double* dAd = dev(Ad);
  double* dCd = dev(Cd);
  const double twod = 2, monod = -1;
  IS(cublasDsyrk(h, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, n, k, &twod, dAd, ld, &monod, dCd, ld), CUBLAS_STATUS_SUCCESS);
  check(equal(host(dCd, Cd.size()), rk_ref<double>(wide(Cd), n, k, ld, false, false, true, wa, nullptr, ld, 2, -1)),
        "dsyrk, A^T A");
  IS(cublasDsyr2k(h, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k, &twod, dAd, ld, dAd, ld, &monod, dCd, ld),
     CUBLAS_STATUS_SUCCESS);

  // her2k and the complex syr2k: her2k takes OP_T as OP_C, and syr2k OP_C as
  // OP_T, as the card does; her2k's beta is real and its diagonal stays real.
  const auto Ac = ints<cuComplex>(ld * 5, 10), Bc = ints<cuComplex>(ld * 5, 11), Cc = ints<cuComplex>(ld * 5, 12);
  cuComplex* dAc = dev(Ac);
  cuComplex* dBc = dev(Bc);
  cuComplex* dCc = dev(Cc);
  const cuComplex alc = make_cuComplex(1, 2), bec = make_cuComplex(0, -1);
  const float ber = 3;
  for (cublasOperation_t t : {CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C})
    for (bool lower : {true, false}) {
      const cublasFillMode_t u = lower ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
      const auto wac = wide(Ac), wbc = wide(Bc);
      cudaMemcpy(dCc, Cc.data(), Cc.size() * 8, cudaMemcpyHostToDevice);
      int st = cublasCher2k(h, u, t, n, k, &alc, dAc, ld, dBc, ld, &ber, dCc, ld);
      const bool ok1 = st == 0 && equal(host(dCc, Cc.size()), rk_ref<cuComplex>(wide(Cc), n, k, ld, lower, true,
                                                                                t != CUBLAS_OP_N, wac, &wbc, ld,
                                                                                cd(1, 2), 3));
      cudaMemcpy(dCc, Cc.data(), Cc.size() * 8, cudaMemcpyHostToDevice);
      st = cublasCsyr2k(h, u, t, n, k, &alc, dAc, ld, dBc, ld, &bec, dCc, ld);
      const bool ok2 = st == 0 && equal(host(dCc, Cc.size()), rk_ref<cuComplex>(wide(Cc), n, k, ld, lower, false,
                                                                                t != CUBLAS_OP_N, wac, &wbc, ld,
                                                                                cd(1, 2), cd(0, -1)));
      char what[80];
      std::snprintf(what, sizeof what, "cher2k and csyr2k, op %d, %s", (int)t, lower ? "lower" : "upper");
      check(ok1 && ok2, what);
    }
  const auto Az = ints<cuDoubleComplex>(ld * 5, 13), Cz = ints<cuDoubleComplex>(ld * 5, 14);
  cuDoubleComplex* dAz = dev(Az);
  cuDoubleComplex* dCz = dev(Cz);
  const cuDoubleComplex alz = make_cuDoubleComplex(1, -1);
  const double berz = 0;
  IS(cublasZher2k(h, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_C, n, k, &alz, dAz, ld, dAz, ld, &berz, dCz, ld),
     CUBLAS_STATUS_SUCCESS);
  {
    const auto waz = wide(Az);
    check(equal(host(dCz, Cz.size()), rk_ref<cuDoubleComplex>(wide(Cz), n, k, ld, false, true, true, waz, &waz, ld,
                                                              cd(1, -1), 0)),
          "zher2k, A^H A + A^H A, beta 0");
  }
  IS(cublasZsyr2k(h, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, n, k, &alz, dAz, ld, dAz, ld, &alz, dCz, 3),
     CUBLAS_STATUS_INVALID_VALUE);

  // trmm: C = alpha op(A) B or alpha B op(A), A triangular, B left alone.
  std::vector<float> T(ld * 5, 99);
  for (int j = 0; j < 5; ++j)
    for (int i = 0; i < 5; ++i) T[j * ld + i] = (float)((2 * i + j) % 5 - 2);
  float* dT = dev(T);
  for (cublasSideMode_t s : {CUBLAS_SIDE_LEFT, CUBLAS_SIDE_RIGHT})
    for (cublasOperation_t t : {CUBLAS_OP_N, CUBLAS_OP_T})
      for (cublasDiagType_t d : {CUBLAS_DIAG_NON_UNIT, CUBLAS_DIAG_UNIT}) {
        const bool lower = t == CUBLAS_OP_N;
        reset();
        const int st = cublasStrmm(h, s, lower ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER, t, d, m, n, &two, dT,
                                   ld, dB, ld, dC, ld);
        auto tri = [&](int i, int j) -> double {   // op(A)(i, j)
          const int r = t == CUBLAS_OP_N ? i : j, c = t == CUBLAS_OP_N ? j : i;
          if (r == c && d == CUBLAS_DIAG_UNIT) return 1;
          if (lower ? c > r : c < r) return 0;
          return T[c * ld + r];
        };
        std::vector<cd> w = wide(C0);
        for (int j = 0; j < n; ++j)
          for (int i = 0; i < m; ++i) {
            double acc = 0;
            if (s == CUBLAS_SIDE_LEFT) for (int p = 0; p < m; ++p) acc += tri(i, p) * B[j * ld + p];
            else for (int p = 0; p < n; ++p) acc += B[p * ld + i] * tri(p, j);
            w[j * ld + i] = 2 * acc;
          }
        char what[80];
        std::snprintf(what, sizeof what, "strmm, %s, op %d, %s diagonal", s == CUBLAS_SIDE_LEFT ? "left" : "right",
                      (int)t, d == CUBLAS_DIAG_UNIT ? "unit" : "stored");
        check(st == 0 && equal(host(dC, C0.size()), w) && host(dB, B.size()) == B, what);
      }
  std::vector<double> Td(T.begin(), T.end()), Bd(B.begin(), B.end());
  double* dTd = dev(Td);
  double* dBd = dev(Bd);
  IS(cublasDtrmm(h, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, m, n, &twod, dTd, ld, dBd,
                 ld, dCd, ld),
     CUBLAS_STATUS_SUCCESS);
  check(host(dCd, 1)[0] == 2 * (B[0] + Td[ld] * B[1] + Td[2 * ld] * B[2]), "dtrmm");
  IS(cublasStrmm(h, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, m, n, &two, dT, ld, dB, ld,
                 dC, 2),
     CUBLAS_STATUS_INVALID_VALUE);
  IS(cublasStrmm(h, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_FULL, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, m, n, &two, dT, ld, dB, ld,
                 dC, ld),
     CUBLAS_STATUS_INVALID_VALUE);
  IS(cublasStrsm(h, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_FULL, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, m, n, &two, dT, ld, dC, ld),
     CUBLAS_STATUS_INVALID_VALUE);

  // The double-precision batched GEMMs, three products in one buffer.
  const int bs = ld * 5;
  std::vector<double> AA(3 * bs), BB(3 * bs), CC(3 * bs, 7);
  for (int i = 0; i < 3 * bs; ++i) AA[i] = (i * 5) % 7 - 3, BB[i] = (i * 3) % 5 - 2;
  double* dAA = dev(AA);
  double* dBB = dev(BB);
  double* dCC = dev(CC);
  IS(cublasDgemmStridedBatched(h, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &twod, dAA, ld, bs, dBB, ld, bs, &monod, dCC, ld,
                               bs, 3),
     CUBLAS_STATUS_SUCCESS);
  {
    // C_b = 2 A_b^T B_b - C_b, A_b and B_b both k-row matrices.
    std::vector<double> w2 = CC;
    for (int b = 0; b < 3; ++b)
      for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i) {
          double acc = 0;
          for (int p = 0; p < k; ++p) acc += AA[b * bs + i * ld + p] * BB[b * bs + j * ld + p];
          w2[b * bs + j * ld + i] = 2 * acc - CC[b * bs + j * ld + i];
        }
    check(host(dCC, CC.size()) == w2, "dgemmStridedBatched, three products");
  }
  std::vector<double*> pa = {dAA + bs, dAA + 2 * bs, dAA}, pb = {dBB, dBB + bs, dBB + 2 * bs},
                       pc = {dCC + 2 * bs, dCC, dCC + bs};
  double** dpa = dev(pa);
  double** dpb = dev(pb);
  double** dpc = dev(pc);
  cudaMemcpy(dCC, CC.data(), CC.size() * 8, cudaMemcpyHostToDevice);
  IS(cublasDgemmBatched(h, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &twod, dpa, ld, dpb, ld, &monod, dpc, ld, 3),
     CUBLAS_STATUS_SUCCESS);
  {
    std::vector<double> w2 = CC;
    const int ia[3] = {1, 2, 0}, ib[3] = {0, 1, 2}, ic[3] = {2, 0, 1};
    for (int q = 0; q < 3; ++q)
      for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i) {
          double acc = 0;
          for (int p = 0; p < k; ++p) acc += AA[ia[q] * bs + i * ld + p] * BB[ib[q] * bs + j * ld + p];
          w2[ic[q] * bs + j * ld + i] = 2 * acc - CC[ic[q] * bs + j * ld + i];
        }
    check(host(dCC, CC.size()) == w2, "dgemmBatched through device pointer arrays");
  }
  // Arguments are checked before any work, even for an empty batch.
  IS(cublasDgemmBatched(h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &twod, dpa, 2, dpb, ld, &monod, dpc, ld, 0),
     CUBLAS_STATUS_INVALID_VALUE);
  IS(cublasDgemmBatched(h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &twod, dpa, ld, dpb, ld, &monod, dpc, ld, -1),
     CUBLAS_STATUS_INVALID_VALUE);
  IS(cublasDgemmStridedBatched(h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &twod, dAA, 2, bs, dBB, ld, bs, &monod, dCC, ld,
                               bs, 0),
     CUBLAS_STATUS_INVALID_VALUE);
  IS(cublasSgemmStridedBatched(h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &two, dA, ld, 0, dB, ld, 0, &mone, dC, ld, 0, -1),
     CUBLAS_STATUS_INVALID_VALUE);
  IS(cublasDgemmStridedBatched(h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &twod, dAA, ld, bs, dBB, ld, bs, &monod, dCC, ld,
                               bs, 0),
     CUBLAS_STATUS_SUCCESS);
  for (void* p : {(void*)dA, (void*)dB, (void*)dC, (void*)dAd, (void*)dCd, (void*)dAc, (void*)dBc, (void*)dCc,
                  (void*)dAz, (void*)dCz, (void*)dT, (void*)dTd, (void*)dBd, (void*)dAA, (void*)dBB, (void*)dCC,
                  (void*)dpa, (void*)dpb, (void*)dpc})
    cudaFree(p);
}

int main() {
  cublasHandle_t h;
  if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) {
    std::printf("FAIL: cublasCreate\n");
    return 1;
  }
  level1(h);
  level2(h);
  level3(h);
  cublasDestroy(h);
  std::printf(failures ? "FAIL: %d cuBLAS level 1-3 checks\n" : "PASS: every cuBLAS level 1-3 check\n", failures);
  return failures ? 1 : 0;
}
