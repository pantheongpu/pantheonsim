// cuBLASXt: its handle and device selection, and its level-3 routines over
// host memory (and device memory), on two devices. GEMM with a block
// dimension small enough that its tiles land on both; syrk, herk, syr2k,
// her2k, syrkx, herkx, symm, hemm, spmm, trsm and trmm whole. Each is checked
// against a host reference over small integers, so exactly; and the
// arguments each refuses, as an RTX 3060's cuBLAS 13.0 answers. Every check
// passes on the card (two RTX 3060s) too.
#include <cublasXt.h>
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

static cd C(float v) { return v; }
static cd C(double v) { return v; }
static cd C(cuComplex v) { return {v.x, v.y}; }
static cd C(cuDoubleComplex v) { return {v.x, v.y}; }
template <class T> static T to(cd v);
template <> float to<float>(cd v) { return (float)v.real(); }
template <> double to<double>(cd v) { return v.real(); }
template <> cuComplex to<cuComplex>(cd v) { return make_cuComplex((float)v.real(), (float)v.imag()); }
template <> cuDoubleComplex to<cuDoubleComplex>(cd v) { return make_cuDoubleComplex(v.real(), v.imag()); }
template <class T> static std::vector<T> ints(size_t n, int seed) {
  std::vector<T> v(n);
  for (size_t i = 0; i < n; ++i) v[i] = to<T>(cd((int)((i * 7 + seed * 3) % 9) - 4, (int)((i * 5 + seed) % 7) - 3));
  return v;
}
template <class T> static bool equal(const std::vector<T>& got, const std::vector<cd>& want) {
  for (size_t i = 0; i < got.size(); ++i)
    if (C(got[i]) != C(to<T>(want[i]))) return false;
  return true;
}
template <class T> static std::vector<cd> wide(const std::vector<T>& v) {
  std::vector<cd> w(v.size());
  for (size_t i = 0; i < v.size(); ++i) w[i] = C(v[i]);
  return w;
}
static cd opv(cd v, cublasOperation_t t) { return t == CUBLAS_OP_C ? std::conj(v) : v; }

template <class T> struct Xt;
#define XT(P, T)                                                                                                  \
  template <> struct Xt<T> {                                                                                      \
    static constexpr auto gemm = cublasXt##P##gemm;                                                              \
    static constexpr auto syrk = cublasXt##P##syrk;                                                              \
    static constexpr auto syrkx = cublasXt##P##syrkx;                                                            \
    static constexpr auto symm = cublasXt##P##symm;                                                              \
    static constexpr auto spmm = cublasXt##P##spmm;                                                              \
    static constexpr auto trsm = cublasXt##P##trsm;                                                              \
    static constexpr auto trmm = cublasXt##P##trmm;                                                              \
  };
XT(S, float)
XT(D, double)
XT(C, cuComplex)
XT(Z, cuDoubleComplex)

// C = alpha op(A) op(B) + beta C over host memory, 21 x 13 with k = 6 and a
// block dimension of 8: twelve tiles, alternating between the devices.
template <class T> static void gemm(cublasXtHandle_t x, const char* ty) {
  const int m = 21, n = 13, k = 6, ld = 23;
  const auto A = ints<T>((size_t)ld * 23, 1), B = ints<T>((size_t)ld * 23, 2), C0 = ints<T>((size_t)ld * n, 3);
  const T al = to<T>(cd(2, 1)), be = to<T>(cd(-1, 0));
  char what[96];
  for (cublasOperation_t ta : {CUBLAS_OP_N, CUBLAS_OP_C})
    for (cublasOperation_t tb : {CUBLAS_OP_N, CUBLAS_OP_T}) {
      auto Cv = C0;
      const int st = Xt<T>::gemm(x, ta, tb, m, n, k, &al, A.data(), ld, B.data(), ld, &be, Cv.data(), ld);
      auto w = wide(C0);
      for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i) {
          cd s = 0;
          for (int p = 0; p < k; ++p) {
            const cd a = ta == CUBLAS_OP_N ? C(A[(size_t)p * ld + i]) : opv(C(A[(size_t)i * ld + p]), ta);
            const cd b = tb == CUBLAS_OP_N ? C(B[(size_t)j * ld + p]) : opv(C(B[(size_t)p * ld + j]), tb);
            s += a * b;
          }
          w[(size_t)j * ld + i] = C(al) * s + C(be) * w[(size_t)j * ld + i];
        }
      std::snprintf(what, sizeof what, "xt %sgemm, ops %d %d, host memory, tiles on both devices", ty, (int)ta, (int)tb);
      check(st == 0 && equal(Cv, w), what);
    }
  // Device memory works too.
  T *dA, *dB, *dC;
  cudaMalloc(&dA, A.size() * sizeof(T));
  cudaMalloc(&dB, B.size() * sizeof(T));
  cudaMalloc(&dC, C0.size() * sizeof(T));
  cudaMemcpy(dA, A.data(), A.size() * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B.data(), B.size() * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy(dC, C0.data(), C0.size() * sizeof(T), cudaMemcpyHostToDevice);
  auto Ch = C0;
  const int st1 = Xt<T>::gemm(x, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &al, A.data(), ld, B.data(), ld, &be, Ch.data(), ld);
  const int st2 = Xt<T>::gemm(x, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &al, dA, ld, dB, ld, &be, dC, ld);
  std::vector<T> Cd(C0.size());
  cudaMemcpy(Cd.data(), dC, Cd.size() * sizeof(T), cudaMemcpyDeviceToHost);
  std::snprintf(what, sizeof what, "xt %sgemm over device memory matches host memory", ty);
  check(st1 == 0 && st2 == 0 && equal(Cd, wide(Ch)), what);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
}

// syrk / syrkx, symm and spmm, trsm and trmm.
template <class T> static void rest(cublasXtHandle_t x, const char* ty) {
  const int n = 5, k = 3, ld = 6;
  const auto A = ints<T>((size_t)ld * 6, 4), B = ints<T>((size_t)ld * 6, 5), C0 = ints<T>((size_t)ld * n, 6);
  const T al = to<T>(cd(1, -1)), be = to<T>(cd(2, 0));
  char what[96];
  for (bool lower : {true, false}) {
    const cublasFillMode_t u = lower ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
    auto C1 = C0, C2 = C0;
    const int st1 = Xt<T>::syrk(x, u, CUBLAS_OP_N, n, k, &al, A.data(), ld, &be, C1.data(), ld);
    const int st2 = Xt<T>::syrkx(x, u, CUBLAS_OP_T, n, k, &al, A.data(), ld, B.data(), ld, &be, C2.data(), ld);
    auto w1 = wide(C0), w2 = wide(C0);
    for (int j = 0; j < n; ++j)
      for (int i = lower ? j : 0; i < (lower ? n : j + 1); ++i) {
        cd s1 = 0, s2 = 0;
        for (int p = 0; p < k; ++p) {
          s1 += C(A[(size_t)p * ld + i]) * C(A[(size_t)p * ld + j]);
          s2 += C(A[(size_t)i * ld + p]) * C(B[(size_t)j * ld + p]);
        }
        w1[(size_t)j * ld + i] = C(al) * s1 + C(be) * w1[(size_t)j * ld + i];
        w2[(size_t)j * ld + i] = C(al) * s2 + C(be) * w2[(size_t)j * ld + i];
      }
    std::snprintf(what, sizeof what, "xt %ssyrk and %ssyrkx, %s", ty, ty, lower ? "lower" : "upper");
    check(st1 == 0 && st2 == 0 && equal(C1, w1) && equal(C2, w2), what);

    // symm from the full triangle, spmm from the same triangle packed.
    const int m = 4;
    std::vector<T> AP((size_t)m * (m + 1) / 2);
    for (int j = 0, q = 0; j < m; ++j)
      for (int i = lower ? j : 0; i < (lower ? m : j + 1); ++i) AP[q++] = A[(size_t)j * ld + i];
    for (cublasSideMode_t s : {CUBLAS_SIDE_LEFT, CUBLAS_SIDE_RIGHT}) {
      const int mm = s == CUBLAS_SIDE_LEFT ? m : 3, nn = s == CUBLAS_SIDE_LEFT ? 3 : m;
      auto Cs = C0, Cp = C0;
      const int s1 = Xt<T>::symm(x, s, u, mm, nn, &al, A.data(), ld, B.data(), ld, &be, Cs.data(), ld);
      const int s2 = Xt<T>::spmm(x, s, u, mm, nn, &al, AP.data(), B.data(), ld, &be, Cp.data(), ld);
      auto sym = [&](int i, int j) { return (lower ? i >= j : i <= j) ? C(A[(size_t)j * ld + i]) : C(A[(size_t)i * ld + j]); };
      auto w = wide(C0);
      for (int j = 0; j < nn; ++j)
        for (int i = 0; i < mm; ++i) {
          cd acc = 0;
          if (s == CUBLAS_SIDE_LEFT) for (int p = 0; p < mm; ++p) acc += sym(i, p) * C(B[(size_t)j * ld + p]);
          else for (int p = 0; p < nn; ++p) acc += C(B[(size_t)p * ld + i]) * sym(p, j);
          w[(size_t)j * ld + i] = C(al) * acc + C(be) * w[(size_t)j * ld + i];
        }
      std::snprintf(what, sizeof what, "xt %ssymm and %sspmm, %s, %s", ty, ty, s == CUBLAS_SIDE_LEFT ? "left" : "right",
                    lower ? "lower" : "upper");
      check(s1 == 0 && s2 == 0 && equal(Cs, w) && equal(Cp, w), what);
    }
  }
  // trmm then trsm with a unit diagonal undo each other exactly.
  auto Bm = ints<T>((size_t)ld * 4, 7), Cm = ints<T>((size_t)ld * 4, 8);
  const T one = to<T>(1);
  const int st1 = Xt<T>::trmm(x, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, 5, 4, &one,
                              A.data(), ld, Bm.data(), ld, Cm.data(), ld);
  const int st2 = Xt<T>::trsm(x, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, 5, 4, &one,
                              A.data(), ld, Cm.data(), ld);
  bool same = true;
  for (int j = 0; j < 4; ++j)
    for (int i = 0; i < 5; ++i) same = same && C(Cm[(size_t)j * ld + i]) == C(Bm[(size_t)j * ld + i]);
  std::snprintf(what, sizeof what, "xt %strmm then %strsm", ty, ty);
  check(st1 == 0 && st2 == 0 && same, what);
  // The arguments the card refuses.
  auto Cv = C0;
  IS(Xt<T>::gemm(x, CUBLAS_OP_N, CUBLAS_OP_N, 4, 4, 4, &al, A.data(), 3, B.data(), ld, &be, Cv.data(), ld),
     CUBLAS_STATUS_INVALID_VALUE);
  IS(Xt<T>::gemm(x, (cublasOperation_t)9, CUBLAS_OP_N, 4, 4, 4, &al, A.data(), ld, B.data(), ld, &be, Cv.data(), ld),
     CUBLAS_STATUS_INVALID_VALUE);
  IS(Xt<T>::gemm(x, CUBLAS_OP_N, CUBLAS_OP_N, 0, 4, 4, &al, A.data(), ld, B.data(), ld, &be, Cv.data(), ld),
     CUBLAS_STATUS_SUCCESS);
  IS(Xt<T>::syrk(x, CUBLAS_FILL_MODE_FULL, CUBLAS_OP_N, 4, 4, &al, A.data(), ld, &be, Cv.data(), ld),
     CUBLAS_STATUS_INVALID_VALUE);
  IS(Xt<T>::syrk(x, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, 4, 4, &al, A.data(), ld, &be, Cv.data(), 3),
     CUBLAS_STATUS_INVALID_VALUE);
  IS(Xt<T>::trsm(x, (cublasSideMode_t)5, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, 4, 4, &al, A.data(), ld,
                 Cv.data(), ld),
     CUBLAS_STATUS_INVALID_VALUE);
  IS(Xt<T>::spmm(x, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_FULL, 4, 4, &al, A.data(), B.data(), ld, &be, Cv.data(), ld),
     CUBLAS_STATUS_INVALID_VALUE);
}

// The Hermitian routines: herk, her2k, herkx, hemm.
template <class T, class R>
static void hermitian(cublasXtHandle_t x, const char* ty,
                      cublasStatus_t (*herk)(cublasXtHandle_t, cublasFillMode_t, cublasOperation_t, size_t, size_t,
                                             const R*, const T*, size_t, const R*, T*, size_t),
                      cublasStatus_t (*her2k)(cublasXtHandle_t, cublasFillMode_t, cublasOperation_t, size_t, size_t,
                                              const T*, const T*, size_t, const T*, size_t, const R*, T*, size_t),
                      cublasStatus_t (*herkx)(cublasXtHandle_t, cublasFillMode_t, cublasOperation_t, size_t, size_t,
                                              const T*, const T*, size_t, const T*, size_t, const R*, T*, size_t),
                      cublasStatus_t (*hemm)(cublasXtHandle_t, cublasSideMode_t, cublasFillMode_t, size_t, size_t,
                                             const T*, const T*, size_t, const T*, size_t, const T*, T*, size_t)) {
  const int n = 4, k = 3, ld = 5;
  const auto A = ints<T>((size_t)ld * 5, 9), B = ints<T>((size_t)ld * 5, 10), C0 = ints<T>((size_t)ld * n, 11);
  const R ar = 2, br = -1;
  const T al = to<T>(cd(1, 2)), bc = to<T>(cd(0, 1));
  auto C1 = C0, C2 = C0, C3 = C0, C4 = C0;
  const int s1 = herk(x, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k, &ar, A.data(), ld, &br, C1.data(), ld);
  const int s2 = her2k(x, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_C, n, k, &al, A.data(), ld, B.data(), ld, &br, C2.data(), ld);
  const int s3 = herkx(x, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k, &al, A.data(), ld, B.data(), ld, &br, C3.data(), ld);
  const int s4 = hemm(x, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, n, 3, &al, A.data(), ld, B.data(), ld, &bc, C4.data(), ld);
  auto w1 = wide(C0), w2 = wide(C0), w3 = wide(C0), w4 = wide(C0);
  auto a = [&](int i, int p) { return C(A[(size_t)p * ld + i]); };
  auto b = [&](int i, int p) { return C(B[(size_t)p * ld + i]); };
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
      cd t1 = 0, t2 = 0, t3 = 0;
      for (int p = 0; p < k; ++p) {
        t1 += a(i, p) * std::conj(a(j, p));
        t2 += C(al) * std::conj(a(p, i)) * b(p, j) + std::conj(C(al)) * std::conj(b(p, i)) * a(p, j);
        t3 += a(i, p) * std::conj(b(j, p));
      }
      if (i >= j) w1[(size_t)j * ld + i] = cd(2.0 * t1 - w1[(size_t)j * ld + i]), w3[(size_t)j * ld + i] = C(al) * t3 - w3[(size_t)j * ld + i];
      if (i <= j) w2[(size_t)j * ld + i] = t2 - w2[(size_t)j * ld + i];
      if (i == j) w1[(size_t)j * ld + i] = w1[(size_t)j * ld + i].real(), w2[(size_t)j * ld + i] = w2[(size_t)j * ld + i].real(),
                  w3[(size_t)j * ld + i] = w3[(size_t)j * ld + i].real();
    }
  for (int j = 0; j < 3; ++j)
    for (int i = 0; i < n; ++i) {
      cd s = 0;
      for (int p = 0; p < n; ++p) {
        cd h = i <= p ? a(i, p) : std::conj(a(p, i));
        if (i == p) h = h.real();
        s += h * b(p, j);
      }
      w4[(size_t)j * ld + i] = C(al) * s + C(bc) * w4[(size_t)j * ld + i];
    }
  char what[96];
  std::snprintf(what, sizeof what, "xt %sherk, %sher2k, %sherkx, %shemm", ty, ty, ty, ty);
  check(s1 == 0 && s2 == 0 && s3 == 0 && s4 == 0 && equal(C1, w1) && equal(C2, w2) && equal(C3, w3) && equal(C4, w4),
        what);
}

int main() {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  int count = 0;
  cudaGetDeviceCount(&count);
  if (count < 2) {
    std::printf("SKIP: cuBLASXt checks need two devices, found %d\n", count);
    return 0;
  }
  cublasXtHandle_t x;
  IS(cublasXtCreate(&x), CUBLAS_STATUS_SUCCESS);
  {
    std::vector<float> A(16, 1), C(16, 3);
    const float one = 1;
    IS(cublasXtSgemm(x, CUBLAS_OP_N, CUBLAS_OP_N, 4, 4, 4, &one, A.data(), 4, A.data(), 4, &one, C.data(), 4),
       CUBLAS_STATUS_NOT_INITIALIZED);   // no devices selected yet
  }
  int ids[2] = {0, 1};
  IS(cublasXtDeviceSelect(x, 2, ids), CUBLAS_STATUS_SUCCESS);
  IS(cublasXtDeviceSelect(x, 1, ids), CUBLAS_STATUS_INVALID_VALUE);   // a handle takes one selection
  int bd = -1;
  IS(cublasXtGetBlockDim(x, &bd), CUBLAS_STATUS_SUCCESS);
  check(bd == 1024, "the default block dimension is 1024");
  IS(cublasXtSetBlockDim(x, 0), CUBLAS_STATUS_INVALID_VALUE);
  IS(cublasXtSetBlockDim(x, -1), CUBLAS_STATUS_INVALID_VALUE);
  IS(cublasXtSetBlockDim(x, 8), CUBLAS_STATUS_SUCCESS);
  IS(cublasXtGetBlockDim(x, &bd), CUBLAS_STATUS_SUCCESS);
  check(bd == 8, "the block dimension set is read back");
  cublasXtPinnedMemMode_t pm = (cublasXtPinnedMemMode_t)7;
  IS(cublasXtGetPinningMemMode(x, &pm), CUBLAS_STATUS_SUCCESS);
  check(pm == CUBLASXT_PINNING_DISABLED, "pinning is off by default");
  IS(cublasXtSetPinningMemMode(x, CUBLASXT_PINNING_ENABLED), CUBLAS_STATUS_SUCCESS);
  IS(cublasXtGetPinningMemMode(x, &pm), CUBLAS_STATUS_SUCCESS);
  check(pm == CUBLASXT_PINNING_ENABLED, "pinning set is read back");
  IS(cublasXtSetPinningMemMode(x, (cublasXtPinnedMemMode_t)5), CUBLAS_STATUS_INVALID_VALUE);
  IS(cublasXtSetPinningMemMode(x, CUBLASXT_PINNING_DISABLED), CUBLAS_STATUS_SUCCESS);
  IS(cublasXtSetCpuRatio(x, CUBLASXT_GEMM, CUBLASXT_FLOAT, 0.0f), CUBLAS_STATUS_SUCCESS);
  IS(cublasXtSetCpuRatio(x, (cublasXtBlasOp_t)20, CUBLASXT_FLOAT, 0.0f), CUBLAS_STATUS_INVALID_VALUE);
  IS(cublasXtSetCpuRatio(x, CUBLASXT_GEMM, (cublasXtOpType_t)9, 0.0f), CUBLAS_STATUS_INVALID_VALUE);
  IS(cublasXtSetCpuRoutine(x, CUBLASXT_GEMM, CUBLASXT_FLOAT, nullptr), CUBLAS_STATUS_SUCCESS);
  IS(cublasXtSetCpuRoutine(x, (cublasXtBlasOp_t)20, CUBLASXT_FLOAT, nullptr), CUBLAS_STATUS_INVALID_VALUE);
  gemm<float>(x, "s");
  gemm<double>(x, "d");
  gemm<cuComplex>(x, "c");
  gemm<cuDoubleComplex>(x, "z");
  rest<float>(x, "s");
  rest<double>(x, "d");
  rest<cuComplex>(x, "c");
  rest<cuDoubleComplex>(x, "z");
  hermitian<cuComplex, float>(x, "c", cublasXtCherk, cublasXtCher2k, cublasXtCherkx, cublasXtChemm);
  hermitian<cuDoubleComplex, double>(x, "z", cublasXtZherk, cublasXtZher2k, cublasXtZherkx, cublasXtZhemm);
  IS(cublasXtDestroy(x), CUBLAS_STATUS_SUCCESS);
  {
    cublasXtHandle_t y;
    cublasXtCreate(&y);
    int bad[1] = {7};
    IS(cublasXtDeviceSelect(y, 1, bad), CUBLAS_STATUS_INTERNAL_ERROR);
    cublasXtDestroy(y);
    cublasXtCreate(&y);
    IS(cublasXtDeviceSelect(y, 0, ids), CUBLAS_STATUS_INVALID_VALUE);
    cublasXtDestroy(y);
  }
  std::printf(failures ? "FAIL: %d cuBLASXt checks\n" : "PASS: every cuBLASXt check\n", failures);
  return failures ? 1 : 0;
}
