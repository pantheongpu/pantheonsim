// cuBLAS's batched GEMVs (gemvBatched and gemvStridedBatched in S, D, C, Z
// and the half and bfloat16 forms HSH, HSS, TST, TSS), getriBatched and
// matinvBatched, syrkx and herkx, the complex dgmm, the gemm3m forms and the
// batched Hgemms. Each is checked against a host reference over small
// integers, which every order of summation gets exactly right (and which
// half and bfloat16 hold exactly), so the comparisons are exact; the
// inverses are of unimodular matrices, exact too. And the arguments each
// refuses, as an RTX 3060 (cuBLAS 13.0) answers. Every check passes on the
// card too.
#include <cublas_v2.h>
#include <cuComplex.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
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
static cd C(__half v) { return (double)__half2float(v); }
static cd C(__nv_bfloat16 v) { return (double)__bfloat162float(v); }
static cd C(cuComplex v) { return {v.x, v.y}; }
static cd C(cuDoubleComplex v) { return {v.x, v.y}; }
template <class T> static T to(cd v);
template <> float to<float>(cd v) { return (float)v.real(); }
template <> double to<double>(cd v) { return v.real(); }
template <> __half to<__half>(cd v) { return __float2half((float)v.real()); }
template <> __nv_bfloat16 to<__nv_bfloat16>(cd v) { return __float2bfloat16((float)v.real()); }
template <> cuComplex to<cuComplex>(cd v) { return make_cuComplex((float)v.real(), (float)v.imag()); }
template <> cuDoubleComplex to<cuDoubleComplex>(cd v) { return make_cuDoubleComplex(v.real(), v.imag()); }
template <class T> static constexpr bool cplx = std::is_same_v<T, cuComplex> || std::is_same_v<T, cuDoubleComplex>;

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
static cd opv(cd v, cublasOperation_t t) { return t == CUBLAS_OP_C ? std::conj(v) : v; }

// ---- gemvBatched and gemvStridedBatched ----
// The batched and strided forms of one type combination, behind one call.
template <class TA, class TY, class S> struct Gemv {
  cublasStatus_t (*batched)(cublasHandle_t, cublasOperation_t, int, int, const S*, const TA* const[], int,
                            const TA* const[], int, const S*, TY* const[], int, int);
  cublasStatus_t (*strided)(cublasHandle_t, cublasOperation_t, int, int, const S*, const TA*, int, long long,
                            const TA*, int, long long, const S*, TY*, int, long long, int);
};

template <class TA, class TY, class S>
static void gemv_batched(cublasHandle_t h, const char* ty, Gemv<TA, TY, S> f) {
  const int m = 4, n = 3, lda = 5, batch = 3;
  const long long sA = lda * n + 1, sx = 9, sy = 11;
  const auto A = ints<TA>((size_t)sA * batch, 1), x = ints<TA>((size_t)sx * batch, 2);
  const auto y0 = ints<TY>((size_t)sy * batch, 3);
  TA *dA = dev(A), *dx = dev(x);
  TY* dy = dev(y0);
  std::vector<const TA*> pa, px;
  std::vector<TY*> py;
  for (int b = 0; b < batch; ++b) {   // the pointer arrays run backwards through the buffers
    pa.push_back(dA + (batch - 1 - b) * sA);
    px.push_back(dx + b * sx);
    py.push_back(dy + (batch - 1 - b) * sy);
  }
  const TA** dpa = dev(pa);
  const TA** dpx = dev(px);
  TY** dpy = dev(py);
  const S al = to<S>(cd(2, 1)), be = to<S>(cd(-1, 0)), zero = to<S>(0);
  char what[120];
  for (bool strided : {false, true})
    for (cublasOperation_t t : {CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C})
      for (bool beta0 : {false, true}) {
        const int incx = t == CUBLAS_OP_C ? -2 : 1, incy = t == CUBLAS_OP_N ? 2 : -1;
        put(dy, y0);
        const S* bp = beta0 ? &zero : &be;
        const int st = strided ? f.strided(h, t, m, n, &al, dA, lda, sA, dx, incx, sx, bp, dy, incy, sy, batch)
                               : f.batched(h, t, m, n, &al, dpa, lda, dpx, incx, bp, dpy, incy, batch);
        auto w = wide(y0);
        const int ylen = t == CUBLAS_OP_N ? m : n, xlen = t == CUBLAS_OP_N ? n : m;
        for (int b = 0; b < batch; ++b) {
          const size_t oa = strided ? b * sA : (batch - 1 - b) * sA, ox = b * sx,
                       oy = strided ? b * sy : (batch - 1 - b) * sy;
          for (int i = 0; i < ylen; ++i) {
            cd s = 0;
            for (int j = 0; j < xlen; ++j) {
              const cd e = C(A[oa + (t == CUBLAS_OP_N ? (size_t)j * lda + i : (size_t)i * lda + j)]);
              s += opv(e, t) * C(x[ox + at(j, xlen, incx)]);
            }
            cd& yi = w[oy + at(i, ylen, incy)];
            yi = C(al) * s + (beta0 ? cd(0) : C(be) * yi);
          }
        }
        std::snprintf(what, sizeof what, "%sgemv%sBatched, op %d, incx %d, incy %d%s", ty, strided ? "Strided" : "",
                      (int)t, incx, incy, beta0 ? ", beta 0" : "");
        check(st == 0 && equal(host(dy, y0.size()), w), what);
      }
  // All the card checks (cuBLAS 13.0): a leading dimension below m, and
  // nothing at all for an empty batch. It takes a negative size, an unknown
  // operation or a zero increment without complaint and launches with them,
  // which can hang it, so those are not called here.
  IS(f.batched(h, CUBLAS_OP_N, m, n, &al, dpa, m - 1, dpx, 1, &be, dpy, 1, batch), CUBLAS_STATUS_INVALID_VALUE);
  IS(f.strided(h, CUBLAS_OP_T, m, n, &al, dA, m - 1, sA, dx, 1, sx, &be, dy, 1, sy, batch), CUBLAS_STATUS_INVALID_VALUE);
  IS(f.batched(h, CUBLAS_OP_N, m, n, &al, dpa, 1, dpx, 1, &be, dpy, 1, 0), CUBLAS_STATUS_SUCCESS);
  IS(f.batched(h, CUBLAS_OP_N, 0, n, &al, dpa, 0, dpx, 1, &be, dpy, 1, batch), CUBLAS_STATUS_SUCCESS);
  for (void* p : {(void*)dA, (void*)dx, (void*)dy, (void*)dpa, (void*)dpx, (void*)dpy}) cudaFree(p);
}

// ---- getriBatched and matinvBatched ----
// A unimodular matrix, L U with L unit lower and U upper with a diagonal of
// +-1, entries of both in {-1, 0, 1}, so no row is ever swapped and every
// step of the inversion is exact; and a singular one.
template <class T> static std::vector<T> unimodular(int n, int lda, int seed) {
  std::vector<cd> L((size_t)n * n, 0), U((size_t)n * n, 0);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
      const int v = (int)((i * 5 + j * 3 + seed) % 3) - 1;
      if (i > j) L[(size_t)j * n + i] = cplx<T> && (i + j) % 2 ? cd(0, v) : cd(v);
      if (i == j) L[(size_t)j * n + i] = 1, U[(size_t)j * n + i] = (i + seed) % 2 ? -1 : 1;
      if (i < j) U[(size_t)j * n + i] = v;
    }
  std::vector<T> A((size_t)lda * n, to<T>(99));
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
      cd s = 0;
      for (int p = 0; p < n; ++p) s += L[(size_t)p * n + i] * U[(size_t)j * n + p];
      A[(size_t)j * lda + i] = to<T>(s);
    }
  return A;
}

template <class T> struct Inv {
  cublasStatus_t (*getrf)(cublasHandle_t, int, T* const[], int, int*, int*, int);
  cublasStatus_t (*getri)(cublasHandle_t, int, const T* const[], int, const int*, T* const[], int, int*, int);
  cublasStatus_t (*matinv)(cublasHandle_t, int, const T* const[], int, T* const[], int, int*, int);
};

template <class T> static void inversion(cublasHandle_t h, const char* ty, Inv<T> f) {
  const int n = 5, lda = 6, ldc = 7, batch = 2;
  auto A0 = unimodular<T>(n, lda, 1), A1 = unimodular<T>(n, lda, 2);
  std::vector<T> As = A0;
  As.insert(As.end(), A1.begin(), A1.end());
  T* dA = dev(As);
  T* dC = dev(std::vector<T>((size_t)ldc * n * batch, to<T>(7)));
  std::vector<T*> pa = {dA, dA + (size_t)lda * n}, pc = {dC, dC + (size_t)ldc * n};
  T** dpa = dev(pa);
  T** dpc = dev(pc);
  int* dinfo = dev(std::vector<int>(batch, -5));
  int* dpiv = dev(std::vector<int>((size_t)n * batch, 0));
  // A X = I, checked as A times the computed inverse.
  auto is_inverse = [&](const std::vector<T>& a, const std::vector<T>& c) {
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < n; ++i) {
        cd s = 0;
        for (int p = 0; p < n; ++p) s += C(a[(size_t)p * lda + i]) * C(c[(size_t)j * ldc + p]);
        if (s != cd(i == j ? 1 : 0)) return false;
      }
    return true;
  };
  IS(f.matinv(h, n, (const T* const*)dpa, lda, dpc, ldc, dinfo, batch), CUBLAS_STATUS_SUCCESS);
  {
    const auto c = host(dC, (size_t)ldc * n * batch);
    std::vector<T> c0(c.begin(), c.begin() + (size_t)ldc * n), c1(c.begin() + (size_t)ldc * n, c.end());
    char what[96];
    std::snprintf(what, sizeof what, "%smatinvBatched: A A^-1 = I for both, A left as it was", ty);
    check(is_inverse(A0, c0) && is_inverse(A1, c1) && host(dinfo, 2) == std::vector<int>({0, 0}) &&
              equal(host(dA, As.size()), wide(As)),
          what);
    // The rows past n in each column of C are left alone.
    check(C(c[n]) == cd(7) && C(c[(size_t)ldc * n + ldc - 1]) == cd(7), "matinvBatched leaves C's padding alone");
  }
  put(dC, std::vector<T>((size_t)ldc * n * batch, to<T>(7)));
  IS(f.getrf(h, n, dpa, lda, dpiv, dinfo, batch), CUBLAS_STATUS_SUCCESS);
  IS(f.getri(h, n, (const T* const*)dpa, lda, dpiv, dpc, ldc, dinfo, batch), CUBLAS_STATUS_SUCCESS);
  {
    const auto c = host(dC, (size_t)ldc * n * batch);
    std::vector<T> c0(c.begin(), c.begin() + (size_t)ldc * n), c1(c.begin() + (size_t)ldc * n, c.end());
    char what[96];
    std::snprintf(what, sizeof what, "%sgetrfBatched then %sgetriBatched", ty, ty);
    check(is_inverse(A0, c0) && is_inverse(A1, c1) && host(dinfo, 2) == std::vector<int>({0, 0}), what);
  }
  // A singular matrix: column 3 is zero. info names the zero pivot.
  {
    auto S0 = A0;
    for (int i = 0; i < n; ++i) S0[(size_t)2 * lda + i] = to<T>(0);
    put(dA, S0);
    IS(f.matinv(h, n, (const T* const*)dpa, lda, dpc, ldc, dinfo, 1), CUBLAS_STATUS_SUCCESS);
    const int got = host(dinfo, 1)[0];
    char what[96];
    std::snprintf(what, sizeof what, "%smatinvBatched of a singular matrix: info %d", ty, got);
    check(got == 3, what);
  }
  IS(f.matinv(h, 33, (const T* const*)dpa, 33, dpc, 33, dinfo, 1), CUBLAS_STATUS_INVALID_VALUE);
  IS(f.matinv(h, n, (const T* const*)dpa, n - 1, dpc, ldc, dinfo, batch), CUBLAS_STATUS_INVALID_VALUE);
  IS(f.matinv(h, n, (const T* const*)dpa, lda, dpc, n - 1, dinfo, batch), CUBLAS_STATUS_INVALID_VALUE);
  IS(f.matinv(h, -1, (const T* const*)dpa, lda, dpc, ldc, dinfo, batch), CUBLAS_STATUS_INVALID_VALUE);
  IS(f.matinv(h, 0, (const T* const*)dpa, 0, dpc, 0, dinfo, batch), CUBLAS_STATUS_SUCCESS);
  IS(f.getri(h, 0, (const T* const*)dpa, 0, dpiv, dpc, 0, dinfo, batch), CUBLAS_STATUS_SUCCESS);
  IS(f.matinv(h, n, (const T* const*)dpa, lda, dpc, ldc, dinfo, -1), CUBLAS_STATUS_INVALID_VALUE);
  IS(f.getri(h, n, (const T* const*)dpa, n - 1, dpiv, dpc, ldc, dinfo, batch), CUBLAS_STATUS_INVALID_VALUE);
  IS(f.getri(h, n, (const T* const*)dpa, lda, dpiv, dpc, n - 1, dinfo, batch), CUBLAS_STATUS_INVALID_VALUE);
  IS(f.getri(h, -1, (const T* const*)dpa, lda, dpiv, dpc, ldc, dinfo, batch), CUBLAS_STATUS_INVALID_VALUE);
  IS(f.getri(h, n, (const T* const*)dpa, lda, dpiv, dpc, ldc, dinfo, -1), CUBLAS_STATUS_INVALID_VALUE);
  for (void* p : {(void*)dA, (void*)dC, (void*)dpa, (void*)dpc, (void*)dinfo, (void*)dpiv}) cudaFree(p);
}

// ---- syrkx and herkx ----
template <class T, class S> struct Rkx {
  cublasStatus_t (*fn)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int, const T*, const T*, int,
                       const T*, int, const S*, T*, int);
};
template <class T, class S> static void rkx(cublasHandle_t h, const char* name, bool herm, Rkx<T, S> f) {
  const int n = 4, k = 3, ld = 6;
  const auto A = ints<T>((size_t)ld * 6, 4), B = ints<T>((size_t)ld * 6, 5), C0 = ints<T>((size_t)ld * n, 6);
  T *dA = dev(A), *dB = dev(B), *dC = dev(C0);
  const T al = to<T>(cd(1, -2));
  const S be = to<S>(cd(-2, 0));
  char what[96];
  for (cublasOperation_t t : {CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C})
    for (bool lower : {true, false}) {
      put(dC, C0);
      const int st = f.fn(h, lower ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER, t, n, k, &al, dA, ld, dB, ld,
                          &be, dC, ld);
      const bool tr = t != CUBLAS_OP_N;
      auto el = [&](const std::vector<T>& mm, int i, int p) { return C(tr ? mm[(size_t)i * ld + p] : mm[(size_t)p * ld + i]); };
      auto w = wide(C0);
      for (int j = 0; j < n; ++j)
        for (int i = lower ? j : 0; i < (lower ? n : j + 1); ++i) {
          cd s = 0;
          for (int p = 0; p < k; ++p) {
            if (herm && tr) s += std::conj(el(A, i, p)) * el(B, j, p);
            else if (herm) s += el(A, i, p) * std::conj(el(B, j, p));
            else s += el(A, i, p) * el(B, j, p);
          }
          cd& e = w[(size_t)j * ld + i];
          e = C(al) * s + C(be) * e;
          if (herm && i == j) e = e.real();
        }
      std::snprintf(what, sizeof what, "%s, op %d, %s", name, (int)t, lower ? "lower" : "upper");
      check(st == 0 && equal(host(dC, C0.size()), w), what);
    }
  IS(f.fn(h, CUBLAS_FILL_MODE_FULL, CUBLAS_OP_N, n, k, &al, dA, ld, dB, ld, &be, dC, ld), CUBLAS_STATUS_INVALID_VALUE);
  IS(f.fn(h, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k, &al, dA, n - 1, dB, ld, &be, dC, ld), CUBLAS_STATUS_INVALID_VALUE);
  IS(f.fn(h, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k, &al, dA, ld, dB, n - 1, &be, dC, ld), CUBLAS_STATUS_INVALID_VALUE);
  IS(f.fn(h, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, n, k, &al, dA, k - 1, dB, ld, &be, dC, ld), CUBLAS_STATUS_INVALID_VALUE);
  IS(f.fn(h, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k, &al, dA, ld, dB, ld, &be, dC, n - 1), CUBLAS_STATUS_INVALID_VALUE);
  IS(f.fn(h, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, -1, &al, dA, ld, dB, ld, &be, dC, ld), CUBLAS_STATUS_INVALID_VALUE);
  IS(f.fn(h, CUBLAS_FILL_MODE_LOWER, (cublasOperation_t)6, n, k, &al, dA, ld, dB, ld, &be, dC, ld),
     CUBLAS_STATUS_INVALID_VALUE);
  IS(f.fn(h, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k, nullptr, dA, ld, dB, ld, &be, dC, ld), CUBLAS_STATUS_INVALID_VALUE);
  IS(f.fn(h, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, 0, k, &al, dA, ld, dB, ld, &be, dC, 1), CUBLAS_STATUS_SUCCESS);
  for (void* p : {(void*)dA, (void*)dB, (void*)dC}) cudaFree(p);
}

// ---- the complex dgmm, the gemm3m forms and the batched Hgemms ----
template <class T> static void complex_dgmm(cublasHandle_t h, const char* name,
                                            cublasStatus_t (*fn)(cublasHandle_t, cublasSideMode_t, int, int, const T*,
                                                                 int, const T*, int, T*, int)) {
  const int m = 3, n = 4, ld = 5;
  const auto A = ints<T>((size_t)ld * n, 7), x = ints<T>(10, 8), C0 = ints<T>((size_t)ld * n, 9);
  T *dA = dev(A), *dx = dev(x), *dC = dev(C0);
  char what[96];
  for (cublasSideMode_t s : {CUBLAS_SIDE_LEFT, CUBLAS_SIDE_RIGHT})
    for (int inc : {1, -2, 0}) {
      put(dC, C0);
      const int st = fn(h, s, m, n, dA, ld, dx, inc, dC, ld);
      const int len = s == CUBLAS_SIDE_LEFT ? m : n;
      auto w = wide(C0);
      for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i)
          w[(size_t)j * ld + i] = C(A[(size_t)j * ld + i]) * C(x[inc ? at(s == CUBLAS_SIDE_LEFT ? i : j, len, inc) : 0]);
      std::snprintf(what, sizeof what, "%s, %s, incx %d", name, s == CUBLAS_SIDE_LEFT ? "left" : "right", inc);
      check(st == 0 && equal(host(dC, C0.size()), w), what);
    }
  IS(fn(h, (cublasSideMode_t)3, m, n, dA, ld, dx, 1, dC, ld), CUBLAS_STATUS_INVALID_VALUE);
  IS(fn(h, CUBLAS_SIDE_LEFT, m, n, dA, m - 1, dx, 1, dC, ld), CUBLAS_STATUS_INVALID_VALUE);
  IS(fn(h, CUBLAS_SIDE_LEFT, m, n, dA, ld, dx, 1, dC, m - 1), CUBLAS_STATUS_INVALID_VALUE);
  for (void* p : {(void*)dA, (void*)dx, (void*)dC}) cudaFree(p);
}

static void gemm3m_and_hgemm(cublasHandle_t h) {
  const int m = 3, n = 4, k = 2, ld = 5, batch = 2;
  const size_t bs = (size_t)ld * 5;
  const auto A = ints<cuComplex>(bs * batch, 10), B = ints<cuComplex>(bs * batch, 11), C0 = ints<cuComplex>(bs * batch, 12);
  cuComplex *dA = dev(A), *dB = dev(B), *dC = dev(C0);
  const cuComplex al = make_cuComplex(1, 1), be = make_cuComplex(0, -1);
  auto ref = [&](int b, cublasOperation_t ta, cublasOperation_t tb, std::vector<cd>& w) {
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < m; ++i) {
        cd s = 0;
        for (int p = 0; p < k; ++p) {
          const cd a = ta == CUBLAS_OP_N ? C(A[b * bs + (size_t)p * ld + i]) : opv(C(A[b * bs + (size_t)i * ld + p]), ta);
          const cd bb = tb == CUBLAS_OP_N ? C(B[b * bs + (size_t)j * ld + p]) : opv(C(B[b * bs + (size_t)p * ld + j]), tb);
          s += a * bb;
        }
        cd& c = w[b * bs + (size_t)j * ld + i];
        c = C(al) * s + C(be) * c;
      }
  };
  {
    IS(cublasCgemm3m(h, CUBLAS_OP_C, CUBLAS_OP_N, m, n, k, &al, dA, ld, dB, ld, &be, dC, ld), CUBLAS_STATUS_SUCCESS);
    auto w = wide(C0);
    ref(0, CUBLAS_OP_C, CUBLAS_OP_N, w);
    check(equal(host(dC, C0.size()), w), "cgemm3m, A^H B");
    put(dC, C0);
    IS(cublasCgemm3mStridedBatched(h, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &al, dA, ld, bs, dB, ld, bs, &be, dC, ld, bs,
                                   batch),
       CUBLAS_STATUS_SUCCESS);
    w = wide(C0);
    for (int b = 0; b < batch; ++b) ref(b, CUBLAS_OP_N, CUBLAS_OP_T, w);
    check(equal(host(dC, C0.size()), w), "cgemm3mStridedBatched, A B^T");
    put(dC, C0);
    std::vector<cuComplex*> pa = {dA, dA + bs}, pb = {dB, dB + bs}, pc = {dC, dC + bs};
    cuComplex** dpa = dev(pa);
    cuComplex** dpb = dev(pb);
    cuComplex** dpc = dev(pc);
    IS(cublasCgemm3mBatched(h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &al, (const cuComplex* const*)dpa, ld,
                            (const cuComplex* const*)dpb, ld, &be, dpc, ld, batch),
       CUBLAS_STATUS_SUCCESS);
    w = wide(C0);
    for (int b = 0; b < batch; ++b) ref(b, CUBLAS_OP_N, CUBLAS_OP_N, w);
    check(equal(host(dC, C0.size()), w), "cgemm3mBatched");
    IS(cublasCgemm3m(h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &al, dA, m - 1, dB, ld, &be, dC, ld),
       CUBLAS_STATUS_INVALID_VALUE);
    for (void* p : {(void*)dpa, (void*)dpb, (void*)dpc}) cudaFree(p);
  }
  {
    const auto Az = ints<cuDoubleComplex>(bs, 13), Bz = ints<cuDoubleComplex>(bs, 14), Cz = ints<cuDoubleComplex>(bs, 15);
    cuDoubleComplex *dAz = dev(Az), *dBz = dev(Bz), *dCz = dev(Cz);
    const cuDoubleComplex alz = make_cuDoubleComplex(2, 0), bez = make_cuDoubleComplex(0, 0);
    IS(cublasZgemm3m(h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alz, dAz, ld, dBz, ld, &bez, dCz, ld), CUBLAS_STATUS_SUCCESS);
    auto w = wide(Cz);
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < m; ++i) {
        cd s = 0;
        for (int p = 0; p < k; ++p) s += C(Az[(size_t)p * ld + i]) * C(Bz[(size_t)j * ld + p]);
        w[(size_t)j * ld + i] = 2.0 * s;
      }
    check(equal(host(dCz, Cz.size()), w), "zgemm3m, beta 0");
    for (void* p : {(void*)dAz, (void*)dBz, (void*)dCz}) cudaFree(p);
  }
  {
    const auto Ah = ints<__half>(bs * batch, 16), Bh = ints<__half>(bs * batch, 17), Ch = ints<__half>(bs * batch, 18);
    __half *dAh = dev(Ah), *dBh = dev(Bh), *dCh = dev(Ch);
    const __half alh = __float2half(2.0f), beh = __float2half(-1.0f);
    auto href = [&](std::vector<cd>& w, int b) {
      for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i) {
          cd s = 0;
          for (int p = 0; p < k; ++p) s += C(Ah[b * bs + (size_t)i * ld + p]) * C(Bh[b * bs + (size_t)j * ld + p]);
          cd& c = w[b * bs + (size_t)j * ld + i];
          c = 2.0 * s - c;
        }
    };
    IS(cublasHgemmStridedBatched(h, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alh, dAh, ld, bs, dBh, ld, bs, &beh, dCh, ld,
                                 bs, batch),
       CUBLAS_STATUS_SUCCESS);
    auto w = wide(Ch);
    for (int b = 0; b < batch; ++b) href(w, b);
    check(equal(host(dCh, Ch.size()), w), "hgemmStridedBatched, A^T B");
    put(dCh, Ch);
    std::vector<__half*> pa = {dAh, dAh + bs}, pb = {dBh, dBh + bs}, pc = {dCh, dCh + bs};
    __half** dpa = dev(pa);
    __half** dpb = dev(pb);
    __half** dpc = dev(pc);
    IS(cublasHgemmBatched(h, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alh, (const __half* const*)dpa, ld,
                          (const __half* const*)dpb, ld, &beh, dpc, ld, batch),
       CUBLAS_STATUS_SUCCESS);
    check(equal(host(dCh, Ch.size()), w), "hgemmBatched, A^T B");
    IS(cublasHgemmStridedBatched(h, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alh, dAh, ld, bs, dBh, ld, bs, &beh, dCh, ld,
                                 bs, -1),
       CUBLAS_STATUS_INVALID_VALUE);
    for (void* p : {(void*)dAh, (void*)dBh, (void*)dCh, (void*)dpa, (void*)dpb, (void*)dpc}) cudaFree(p);
  }
  for (void* p : {(void*)dA, (void*)dB, (void*)dC}) cudaFree(p);
}

int main() {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  cublasHandle_t h;
  if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) {
    std::printf("FAIL: cublasCreate\n");
    return 1;
  }
  gemv_batched<float, float, float>(h, "s", {cublasSgemvBatched, cublasSgemvStridedBatched});
  gemv_batched<double, double, double>(h, "d", {cublasDgemvBatched, cublasDgemvStridedBatched});
  gemv_batched<cuComplex, cuComplex, cuComplex>(h, "c", {cublasCgemvBatched, cublasCgemvStridedBatched});
  gemv_batched<cuDoubleComplex, cuDoubleComplex, cuDoubleComplex>(h, "z",
                                                                  {cublasZgemvBatched, cublasZgemvStridedBatched});
  gemv_batched<__half, __half, float>(h, "HSH", {cublasHSHgemvBatched, cublasHSHgemvStridedBatched});
  gemv_batched<__half, float, float>(h, "HSS", {cublasHSSgemvBatched, cublasHSSgemvStridedBatched});
  gemv_batched<__nv_bfloat16, __nv_bfloat16, float>(h, "TST", {cublasTSTgemvBatched, cublasTSTgemvStridedBatched});
  gemv_batched<__nv_bfloat16, float, float>(h, "TSS", {cublasTSSgemvBatched, cublasTSSgemvStridedBatched});
  inversion<float>(h, "s", {cublasSgetrfBatched, cublasSgetriBatched, cublasSmatinvBatched});
  inversion<double>(h, "d", {cublasDgetrfBatched, cublasDgetriBatched, cublasDmatinvBatched});
  inversion<cuComplex>(h, "c", {cublasCgetrfBatched, cublasCgetriBatched, cublasCmatinvBatched});
  inversion<cuDoubleComplex>(h, "z", {cublasZgetrfBatched, cublasZgetriBatched, cublasZmatinvBatched});
  rkx<float, float>(h, "ssyrkx", false, {cublasSsyrkx});
  rkx<double, double>(h, "dsyrkx", false, {cublasDsyrkx});
  rkx<cuComplex, cuComplex>(h, "csyrkx", false, {cublasCsyrkx});
  rkx<cuDoubleComplex, cuDoubleComplex>(h, "zsyrkx", false, {cublasZsyrkx});
  rkx<cuComplex, float>(h, "cherkx", true, {cublasCherkx});
  rkx<cuDoubleComplex, double>(h, "zherkx", true, {cublasZherkx});
  complex_dgmm<cuComplex>(h, "cdgmm", cublasCdgmm);
  complex_dgmm<cuDoubleComplex>(h, "zdgmm", cublasZdgmm);
  gemm3m_and_hgemm(h);
  cublasDestroy(h);
  std::printf(failures ? "FAIL: %d cuBLAS batched checks\n" : "PASS: every cuBLAS batched check\n", failures);
  return failures ? 1 : 0;
}
