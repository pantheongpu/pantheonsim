// cuBLAS's band and packed level-2 routines in every type each comes in:
// gbmv, sbmv/hbmv, spmv/hpmv, spr/hpr, spr2/hpr2, tbmv, tbsv, tpmv, tpsv, and
// the packed/full conversions tpttr and trttp. Each is checked against a host
// reference over small integers, which every order of summation gets exactly
// right, so the comparisons are exact; and the arguments each refuses, as an
// RTX 3060 (cuBLAS 13.0) answers. Every check passes on the card too.
#include <cublas_v2.h>
#include <cuComplex.h>
#include <cuda_runtime.h>

#include <complex>
#include <cstdio>
#include <type_traits>
#include <vector>

using cd = std::complex<double>;
static int failures = 0;

static void check(bool ok, const char* what) {
  std::printf("%-4s %s\n", ok ? "ok" : "FAIL", what);
  if (!ok) ++failures;
}
#define IS(call, want)                                                                     \
  do {                                                                                     \
    const int got_ = (int)(call);                                                          \
    char what_[512];                                                                       \
    std::snprintf(what_, sizeof what_, "%s -> %s (got %d)", #call, #want, got_);           \
    check(got_ == (int)(want), what_);                                                     \
  } while (0)

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
template <class T> static void put(T* d, const std::vector<T>& h) {
  cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
}

static cd C(float v) { return v; }
static cd C(double v) { return v; }
static cd C(cuComplex v) { return {v.x, v.y}; }
static cd C(cuDoubleComplex v) { return {v.x, v.y}; }
template <class T> static T to(cd v);
template <> float to<float>(cd v) { return (float)v.real(); }
template <> double to<double>(cd v) { return v.real(); }
template <> cuComplex to<cuComplex>(cd v) { return make_cuComplex((float)v.real(), (float)v.imag()); }
template <> cuDoubleComplex to<cuDoubleComplex>(cd v) { return make_cuDoubleComplex(v.real(), v.imag()); }
template <class T> static constexpr bool cplx = std::is_same_v<T, cuComplex> || std::is_same_v<T, cuDoubleComplex>;
template <class T> using Real = std::conditional_t<std::is_same_v<T, float> || std::is_same_v<T, cuComplex>, float, double>;

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
  if (got.size() != want.size()) return false;
  for (size_t i = 0; i < got.size(); ++i)
    if (C(got[i]) != C(to<T>(want[i]))) return false;
  return true;
}
static size_t at(int i, int n, int inc) { return inc > 0 ? (size_t)i * inc : (size_t)(n - 1 - i) * -inc; }
static size_t pat(bool lower, int n, int i, int j) {
  return lower ? (size_t)i + (size_t)(2 * n - j - 1) * j / 2 : (size_t)i + (size_t)j * (j + 1) / 2;
}
static cd opv(cd v, cublasOperation_t t) { return t == CUBLAS_OP_C ? std::conj(v) : v; }

// ---- each routine, by type ----
#define PICK4(name, ...)                                                                          \
  [&] {                                                                                           \
    if constexpr (std::is_same_v<T, float>) return cublasS##name(__VA_ARGS__);                    \
    else if constexpr (std::is_same_v<T, double>) return cublasD##name(__VA_ARGS__);              \
    else if constexpr (std::is_same_v<T, cuComplex>) return cublasC##name(__VA_ARGS__);           \
    else return cublasZ##name(__VA_ARGS__);                                                       \
  }()
#define PICK_SH(sname, hname, ...)                                                                \
  [&] {                                                                                           \
    if constexpr (std::is_same_v<T, float>) return cublasS##sname(__VA_ARGS__);                   \
    else if constexpr (std::is_same_v<T, double>) return cublasD##sname(__VA_ARGS__);             \
    else if constexpr (std::is_same_v<T, cuComplex>) return cublasC##hname(__VA_ARGS__);          \
    else return cublasZ##hname(__VA_ARGS__);                                                      \
  }()

template <class T> static void gbmv(cublasHandle_t h, const char* ty) {
  const int m = 5, n = 4, kl = 1, ku = 2, lda = kl + ku + 2;
  const auto A = ints<T>((size_t)lda * n, 1), x = ints<T>(10, 2), y0 = ints<T>(10, 3);
  T *dA = dev(A), *dx = dev(x), *dy = dev(y0);
  const T al = to<T>(cd(2, -1)), be = to<T>(cd(-1, 1)), zero = to<T>(0);
  char what[120];
  for (cublasOperation_t t : {CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C})
    for (int incx : {1, -2})
      for (bool beta0 : {false, true}) {
        put(dy, y0);
        const int st = PICK4(gbmv, h, t, m, n, kl, ku, &al, dA, lda, dx, incx, beta0 ? &zero : &be, dy, 2);
        const int ylen = t == CUBLAS_OP_N ? m : n, xlen = t == CUBLAS_OP_N ? n : m;
        auto w = wide(y0);
        for (int i = 0; i < ylen; ++i) {
          cd s = 0;
          for (int j = 0; j < xlen; ++j) {
            const int r = t == CUBLAS_OP_N ? i : j, c = t == CUBLAS_OP_N ? j : i;
            if (r < c - ku || r > c + kl) continue;
            s += opv(C(A[(size_t)c * lda + ku + r - c]), t) * C(x[at(j, xlen, incx)]);
          }
          w[at(i, ylen, 2)] = C(al) * s + (beta0 ? cd(0) : C(be) * w[at(i, ylen, 2)]);
        }
        std::snprintf(what, sizeof what, "%sgbmv, op %d, incx %d%s", ty, (int)t, incx, beta0 ? ", beta 0" : "");
        check(st == 0 && equal(host(dy, y0.size()), w), what);
      }
  IS(PICK4(gbmv, h, CUBLAS_OP_N, m, n, kl, ku, &al, dA, kl + ku, dx, 1, &be, dy, 1), CUBLAS_STATUS_INVALID_VALUE);
  IS(PICK4(gbmv, h, CUBLAS_OP_N, m, n, -1, ku, &al, dA, lda, dx, 1, &be, dy, 1), CUBLAS_STATUS_INVALID_VALUE);
  IS(PICK4(gbmv, h, CUBLAS_OP_N, m, n, kl, -1, &al, dA, lda, dx, 1, &be, dy, 1), CUBLAS_STATUS_INVALID_VALUE);
  IS(PICK4(gbmv, h, CUBLAS_OP_N, -1, n, kl, ku, &al, dA, lda, dx, 1, &be, dy, 1), CUBLAS_STATUS_INVALID_VALUE);
  IS(PICK4(gbmv, h, CUBLAS_OP_N, m, n, kl, ku, &al, dA, lda, dx, 0, &be, dy, 1), CUBLAS_STATUS_INVALID_VALUE);
  IS(PICK4(gbmv, h, CUBLAS_OP_N, m, n, kl, ku, &al, dA, lda, dx, 1, &be, dy, 0), CUBLAS_STATUS_INVALID_VALUE);
  IS(PICK4(gbmv, h, (cublasOperation_t)7, m, n, kl, ku, &al, dA, lda, dx, 1, &be, dy, 1), CUBLAS_STATUS_INVALID_VALUE);
  IS(PICK4(gbmv, h, CUBLAS_OP_N, 0, n, kl, ku, &al, dA, lda, dx, 1, &be, dy, 1), CUBLAS_STATUS_SUCCESS);
  // A NULL scalar is refused, but only after an empty problem is done.
  IS(PICK4(gbmv, h, CUBLAS_OP_N, m, n, kl, ku, nullptr, dA, lda, dx, 1, &be, dy, 1), CUBLAS_STATUS_INVALID_VALUE);
  IS(PICK4(gbmv, h, CUBLAS_OP_N, m, n, kl, ku, &al, dA, lda, dx, 1, nullptr, dy, 1), CUBLAS_STATUS_INVALID_VALUE);
  IS(PICK4(gbmv, h, CUBLAS_OP_N, 0, n, kl, ku, nullptr, dA, lda, dx, 1, &be, dy, 1), CUBLAS_STATUS_SUCCESS);
  for (void* p : {(void*)dA, (void*)dx, (void*)dy}) cudaFree(p);
}

// sbmv/hbmv and spmv/hpmv: y = alpha A x + beta y, A symmetric or Hermitian.
template <class T> static void symmetric_mv(cublasHandle_t h, const char* ty) {
  const int n = 5, k = 2, lda = k + 2;
  constexpr bool herm = cplx<T>;
  const auto Ab = ints<T>((size_t)lda * n, 4), AP = ints<T>((size_t)n * (n + 1) / 2, 5), x = ints<T>(10, 6),
             y0 = ints<T>(10, 7);
  T *dAb = dev(Ab), *dAP = dev(AP), *dx = dev(x), *dy = dev(y0);
  const T al = to<T>(cd(1, 2)), be = to<T>(cd(2, 0));
  char what[120];
  for (bool lower : {true, false})
    for (int incx : {1, -2}) {
      const cublasFillMode_t u = lower ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
      // Full matrices the two storages stand for.
      auto full = [&](bool band) {
        std::vector<cd> f((size_t)n * n, 0);
        for (int j = 0; j < n; ++j)
          for (int i = 0; i < n; ++i) {
            const bool stored = lower ? i >= j : i <= j;
            const int r = stored ? i : j, c = stored ? j : i;
            cd v;
            if (band) {
              if (std::abs(r - c) > k) continue;
              v = C(Ab[(size_t)c * lda + (lower ? r - c : k + r - c)]);
            } else {
              v = C(AP[pat(lower, n, r, c)]);
            }
            if (herm && r == c) v = v.real();
            f[(size_t)j * n + i] = stored || !herm ? v : std::conj(v);
          }
        return f;
      };
      for (bool band : {true, false}) {
        put(dy, y0);
        int st;
        if (band) st = PICK_SH(sbmv, hbmv, h, u, n, k, &al, dAb, lda, dx, incx, &be, dy, 1);
        else st = PICK_SH(spmv, hpmv, h, u, n, &al, dAP, dx, incx, &be, dy, 1);
        const auto f = full(band);
        auto w = wide(y0);
        for (int i = 0; i < n; ++i) {
          cd s = 0;
          for (int j = 0; j < n; ++j) s += f[(size_t)j * n + i] * C(x[at(j, n, incx)]);
          w[i] = C(al) * s + C(be) * w[i];
        }
        std::snprintf(what, sizeof what, "%s%s, %s, incx %d", ty, band ? (herm ? "hbmv" : "sbmv") : (herm ? "hpmv" : "spmv"),
                      lower ? "lower" : "upper", incx);
        check(st == 0 && equal(host(dy, y0.size()), w), what);
      }
    }
  IS(PICK_SH(sbmv, hbmv, h, CUBLAS_FILL_MODE_FULL, n, k, &al, dAb, lda, dx, 1, &be, dy, 1), CUBLAS_STATUS_INVALID_VALUE);
  IS(PICK_SH(sbmv, hbmv, h, CUBLAS_FILL_MODE_LOWER, n, k, &al, dAb, k, dx, 1, &be, dy, 1), CUBLAS_STATUS_INVALID_VALUE);
  IS(PICK_SH(sbmv, hbmv, h, CUBLAS_FILL_MODE_LOWER, n, -1, &al, dAb, lda, dx, 1, &be, dy, 1), CUBLAS_STATUS_INVALID_VALUE);
  IS(PICK_SH(sbmv, hbmv, h, CUBLAS_FILL_MODE_LOWER, -1, k, &al, dAb, lda, dx, 1, &be, dy, 1), CUBLAS_STATUS_INVALID_VALUE);
  IS(PICK_SH(sbmv, hbmv, h, CUBLAS_FILL_MODE_LOWER, n, k, &al, dAb, lda, dx, 1, &be, dy, 0), CUBLAS_STATUS_INVALID_VALUE);
  IS(PICK_SH(sbmv, hbmv, h, CUBLAS_FILL_MODE_LOWER, 0, k, &al, dAb, lda, dx, 1, &be, dy, 1), CUBLAS_STATUS_SUCCESS);
  IS(PICK_SH(spmv, hpmv, h, CUBLAS_FILL_MODE_FULL, n, &al, dAP, dx, 1, &be, dy, 1), CUBLAS_STATUS_INVALID_VALUE);
  IS(PICK_SH(spmv, hpmv, h, CUBLAS_FILL_MODE_UPPER, -1, &al, dAP, dx, 1, &be, dy, 1), CUBLAS_STATUS_INVALID_VALUE);
  IS(PICK_SH(spmv, hpmv, h, CUBLAS_FILL_MODE_UPPER, n, &al, dAP, dx, 0, &be, dy, 1), CUBLAS_STATUS_INVALID_VALUE);
  IS(PICK_SH(spmv, hpmv, h, CUBLAS_FILL_MODE_UPPER, n, &al, dAP, dx, 1, &be, dy, 0), CUBLAS_STATUS_INVALID_VALUE);
  IS(PICK_SH(spmv, hpmv, h, CUBLAS_FILL_MODE_UPPER, n, &al, dAP, dx, 1, nullptr, dy, 1), CUBLAS_STATUS_INVALID_VALUE);
  IS(PICK_SH(sbmv, hbmv, h, CUBLAS_FILL_MODE_UPPER, n, k, nullptr, dAb, lda, dx, 1, &be, dy, 1), CUBLAS_STATUS_INVALID_VALUE);
  for (void* p : {(void*)dAb, (void*)dAP, (void*)dx, (void*)dy}) cudaFree(p);
}

// spr/hpr and spr2/hpr2: the packed rank-1 and rank-2 updates.
template <class T> static void packed_rank(cublasHandle_t h, const char* ty) {
  const int n = 4;
  constexpr bool herm = cplx<T>;
  const size_t len = (size_t)n * (n + 1) / 2;
  const auto AP = ints<T>(len, 8), x = ints<T>(8, 9), y = ints<T>(8, 10);
  T *dAP = dev(AP), *dx = dev(x), *dy = dev(y);
  const Real<T> ar = 3;
  const T al = to<T>(cd(2, -1));
  char what[120];
  auto op = [&](cd v) { return herm ? std::conj(v) : v; };
  for (bool lower : {true, false})
    for (bool two : {false, true}) {
      const cublasFillMode_t u = lower ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
      put(dAP, AP);
      int st;
      if (two) st = PICK_SH(spr2, hpr2, h, u, n, &al, dx, -2, dy, 1, dAP);
      else if constexpr (herm) st = PICK_SH(spr, hpr, h, u, n, &ar, dx, -2, dAP);
      else st = PICK_SH(spr, hpr, h, u, n, &al, dx, -2, dAP);
      auto w = wide(AP);
      const cd a = two || !herm ? C(al) : cd(ar);
      for (int j = 0; j < n; ++j)
        for (int i = lower ? j : 0; i < (lower ? n : j + 1); ++i) {
          cd& e = w[pat(lower, n, i, j)];
          const cd xi = C(x[at(i, n, -2)]), xj = C(x[at(j, n, -2)]);
          if (two) e += a * xi * op(C(y[j])) + op(a) * C(y[i]) * op(xj);
          else e += a * xi * op(xj);
          if (herm && i == j) e = e.real();
        }
      std::snprintf(what, sizeof what, "%s%s, %s, incx -2", ty, herm ? (two ? "hpr2" : "hpr") : (two ? "spr2" : "spr"),
                    lower ? "lower" : "upper");
      check(st == 0 && equal(host(dAP, len), w), what);
    }
  IS(PICK_SH(spr2, hpr2, h, CUBLAS_FILL_MODE_FULL, n, &al, dx, 1, dy, 1, dAP), CUBLAS_STATUS_INVALID_VALUE);
  IS(PICK_SH(spr2, hpr2, h, CUBLAS_FILL_MODE_LOWER, -1, &al, dx, 1, dy, 1, dAP), CUBLAS_STATUS_INVALID_VALUE);
  IS(PICK_SH(spr2, hpr2, h, CUBLAS_FILL_MODE_LOWER, n, &al, dx, 0, dy, 1, dAP), CUBLAS_STATUS_INVALID_VALUE);
  IS(PICK_SH(spr2, hpr2, h, CUBLAS_FILL_MODE_LOWER, n, &al, dx, 1, dy, 0, dAP), CUBLAS_STATUS_INVALID_VALUE);
  IS(PICK_SH(spr2, hpr2, h, CUBLAS_FILL_MODE_LOWER, n, nullptr, dx, 1, dy, 1, dAP), CUBLAS_STATUS_INVALID_VALUE);
  if constexpr (herm) {
    IS(PICK_SH(spr, hpr, h, CUBLAS_FILL_MODE_FULL, n, &ar, dx, 1, dAP), CUBLAS_STATUS_INVALID_VALUE);
    IS(PICK_SH(spr, hpr, h, CUBLAS_FILL_MODE_LOWER, n, &ar, dx, 0, dAP), CUBLAS_STATUS_INVALID_VALUE);
  } else {
    IS(PICK_SH(spr, hpr, h, CUBLAS_FILL_MODE_FULL, n, &al, dx, 1, dAP), CUBLAS_STATUS_INVALID_VALUE);
    IS(PICK_SH(spr, hpr, h, CUBLAS_FILL_MODE_LOWER, n, &al, dx, 0, dAP), CUBLAS_STATUS_INVALID_VALUE);
  }
  for (void* p : {(void*)dAP, (void*)dx, (void*)dy}) cudaFree(p);
}

// tbmv, tbsv, tpmv and tpsv. The triangles have a diagonal of 1 (unit or
// stored) so the solves are exact over integers: each solve undoes its
// multiply.
template <class T> static void triangular(cublasHandle_t h, const char* ty) {
  const int n = 5, k = 2, lda = k + 2;
  const size_t len = (size_t)n * (n + 1) / 2;
  char what[120];
  for (bool lower : {true, false}) {
    std::vector<T> Ab = ints<T>((size_t)lda * n, 11), AP = ints<T>(len, 12);
    for (int j = 0; j < n; ++j) {
      Ab[(size_t)j * lda + (lower ? 0 : k)] = to<T>(1);
      AP[pat(lower, n, j, j)] = to<T>(1);
    }
    T *dAb = dev(Ab), *dAP = dev(AP);
    const cublasFillMode_t u = lower ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
    for (bool band : {true, false})
      for (cublasOperation_t t : {CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C})
        for (cublasDiagType_t d : {CUBLAS_DIAG_NON_UNIT, CUBLAS_DIAG_UNIT}) {
          const auto x0 = ints<T>(10, 13 + (int)t);
          T* dx = dev(x0);
          auto el = [&](int r, int c) -> cd {   // A(r, c)
            if (lower ? r < c : r > c) return 0;
            if (r == c && d == CUBLAS_DIAG_UNIT) return 1;
            if (band) {
              if (std::abs(r - c) > k) return 0;
              return C(Ab[(size_t)c * lda + (lower ? r - c : k + r - c)]);
            }
            return C(AP[pat(lower, n, r, c)]);
          };
          auto w = wide(x0);
          for (int i = 0; i < n; ++i) {
            cd s = 0;
            for (int j = 0; j < n; ++j) s += (t == CUBLAS_OP_N ? el(i, j) : opv(el(j, i), t)) * C(x0[at(j, n, -2)]);
            w[at(i, n, -2)] = s;
          }
          const int st = band ? PICK4(tbmv, h, u, t, d, n, k, dAb, lda, dx, -2) : PICK4(tpmv, h, u, t, d, n, dAP, dx, -2);
          const bool mv = st == 0 && equal(host(dx, x0.size()), w);
          const int st2 = band ? PICK4(tbsv, h, u, t, d, n, k, dAb, lda, dx, -2) : PICK4(tpsv, h, u, t, d, n, dAP, dx, -2);
          const bool sv = st2 == 0 && equal(host(dx, x0.size()), wide(x0));
          std::snprintf(what, sizeof what, "%s%s then %s, %s, op %d, %s diagonal, incx -2", ty, band ? "tbmv" : "tpmv",
                        band ? "tbsv" : "tpsv", lower ? "lower" : "upper", (int)t,
                        d == CUBLAS_DIAG_UNIT ? "unit" : "stored");
          check(mv && sv, what);
          cudaFree(dx);
        }
    if (lower) {
      T* dx = dev(ints<T>(10, 1));
      IS(PICK4(tbmv, h, CUBLAS_FILL_MODE_FULL, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, k, dAb, lda, dx, 1),
         CUBLAS_STATUS_INVALID_VALUE);
      IS(PICK4(tbsv, h, u, (cublasOperation_t)9, CUBLAS_DIAG_UNIT, n, k, dAb, lda, dx, 1), CUBLAS_STATUS_INVALID_VALUE);
      IS(PICK4(tbsv, h, u, CUBLAS_OP_N, (cublasDiagType_t)7, n, k, dAb, lda, dx, 1), CUBLAS_STATUS_INVALID_VALUE);
      IS(PICK4(tbsv, h, u, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, k, dAb, k, dx, 1), CUBLAS_STATUS_INVALID_VALUE);
      IS(PICK4(tbsv, h, u, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, -1, dAb, lda, dx, 1), CUBLAS_STATUS_INVALID_VALUE);
      IS(PICK4(tbmv, h, u, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, -1, k, dAb, lda, dx, 1), CUBLAS_STATUS_INVALID_VALUE);
      IS(PICK4(tbmv, h, u, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, k, dAb, lda, dx, 0), CUBLAS_STATUS_INVALID_VALUE);
      IS(PICK4(tbmv, h, u, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, 0, k, dAb, lda, dx, 1), CUBLAS_STATUS_SUCCESS);
      IS(PICK4(tpmv, h, CUBLAS_FILL_MODE_FULL, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, dAP, dx, 1), CUBLAS_STATUS_INVALID_VALUE);
      IS(PICK4(tpsv, h, u, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, n, dAP, dx, 0), CUBLAS_STATUS_INVALID_VALUE);
      IS(PICK4(tpsv, h, u, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, -1, dAP, dx, 1), CUBLAS_STATUS_INVALID_VALUE);
      IS(PICK4(tpsv, h, u, (cublasOperation_t)9, CUBLAS_DIAG_UNIT, n, dAP, dx, 1), CUBLAS_STATUS_INVALID_VALUE);
      cudaFree(dx);
    }
    cudaFree(dAb);
    cudaFree(dAP);
  }
}

// tpttr and trttp: packed to full and back, only the named triangle touched.
template <class T> static void conversions(cublasHandle_t h, const char* ty) {
  const int n = 4, lda = 6;
  const size_t len = (size_t)n * (n + 1) / 2;
  const auto AP = ints<T>(len, 14), A0 = ints<T>((size_t)lda * n, 15);
  T *dAP = dev(AP), *dA = dev(A0), *dP = dev(std::vector<T>(len, to<T>(0)));
  char what[120];
  for (bool lower : {true, false}) {
    const cublasFillMode_t u = lower ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
    put(dA, A0);
    const int st = PICK4(tpttr, h, u, n, dAP, dA, lda);
    auto w = wide(A0);
    for (int j = 0; j < n; ++j)
      for (int i = lower ? j : 0; i < (lower ? n : j + 1); ++i) w[(size_t)j * lda + i] = C(AP[pat(lower, n, i, j)]);
    const int st2 = PICK4(trttp, h, u, n, dA, lda, dP);
    std::snprintf(what, sizeof what, "%stpttr then %strttp, %s", ty, ty, lower ? "lower" : "upper");
    check(st == 0 && st2 == 0 && equal(host(dA, A0.size()), w) && equal(host(dP, len), wide(AP)), what);
  }
  IS(PICK4(tpttr, h, CUBLAS_FILL_MODE_FULL, n, dAP, dA, lda), CUBLAS_STATUS_INVALID_VALUE);
  IS(PICK4(tpttr, h, CUBLAS_FILL_MODE_LOWER, n, dAP, dA, n - 1), CUBLAS_STATUS_INVALID_VALUE);
  IS(PICK4(trttp, h, CUBLAS_FILL_MODE_LOWER, -1, dA, lda, dP), CUBLAS_STATUS_INVALID_VALUE);
  IS(PICK4(trttp, h, CUBLAS_FILL_MODE_LOWER, 0, dA, lda, dP), CUBLAS_STATUS_SUCCESS);
  for (void* p : {(void*)dAP, (void*)dA, (void*)dP}) cudaFree(p);
}

template <class T> static void all(cublasHandle_t h, const char* ty) {
  gbmv<T>(h, ty);
  symmetric_mv<T>(h, ty);
  packed_rank<T>(h, ty);
  triangular<T>(h, ty);
  conversions<T>(h, ty);
}

int main() {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  cublasHandle_t h;
  if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) {
    std::printf("FAIL: cublasCreate\n");
    return 1;
  }
  all<float>(h, "s");
  all<double>(h, "d");
  all<cuComplex>(h, "c");
  all<cuDoubleComplex>(h, "z");
  // Device pointer mode: alpha and beta read from device memory.
  {
    cublasSetPointerMode(h, CUBLAS_POINTER_MODE_DEVICE);
    const std::vector<float> ab = {2, 0}, AP = {1, 2, 3}, x = {1, 1};
    float *dab = dev(ab), *dAP = dev(AP), *dx = dev(x), *dy = dev(std::vector<float>{7, 7});
    IS(cublasSspmv(h, CUBLAS_FILL_MODE_UPPER, 2, dab, dAP, dx, 1, dab + 1, dy, 1), CUBLAS_STATUS_SUCCESS);
    // A = [1 2; 2 3], A x = (3, 5).
    check(host(dy, 2) == std::vector<float>({6, 10}), "sspmv with alpha and beta in device memory");
    cublasSetPointerMode(h, CUBLAS_POINTER_MODE_HOST);
    for (void* p : {(void*)dab, (void*)dAP, (void*)dx, (void*)dy}) cudaFree(p);
  }
  cublasDestroy(h);
  std::printf(failures ? "FAIL: %d cuBLAS band and packed checks\n" : "PASS: every cuBLAS band and packed check\n",
              failures);
  return failures ? 1 : 0;
}
