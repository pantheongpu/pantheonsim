// cuBLAS's 64-bit-index (ILP64) forms of levels 2 and 3 and the batched
// routines, each against its 32-bit form on the same operands (small
// integers, so exactly), and the handle settings (atomics mode, SM count
// target, CUDA 13's emulation controls) with the defaults and refusals an
// RTX 3060's cuBLAS 13.0 gives. Every check passes on the card too.
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <vector>

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
static std::vector<double> ints(size_t n, int seed) {
  std::vector<double> v(n);
  for (size_t i = 0; i < n; ++i) v[i] = (double)((int)((i * 7 + seed * 3) % 9) - 4);
  return v;
}

// Runs `call32` and `call64` on the same output buffer, from the same start,
// and checks they agree and succeed.
template <class T, class F32, class F64>
static void same(const char* name, T* out, const std::vector<T>& start, F32 call32, F64 call64) {
  put(out, start);
  const int s32 = (int)call32();
  const auto r32 = host(out, start.size());
  put(out, start);
  const int s64 = (int)call64();
  const auto r64 = host(out, start.size());
  char what[128];
  std::snprintf(what, sizeof what, "%s matches its 32-bit form (%d, %d)", name, s32, s64);
  check(s32 == 0 && s64 == 0 && std::memcmp(r32.data(), r64.data(), r32.size() * sizeof(T)) == 0 &&
            std::memcmp(r32.data(), start.data(), r32.size() * sizeof(T)) != 0,
        what);
}

int main() {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  cublasHandle_t h;
  if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) {
    std::printf("FAIL: cublasCreate\n");
    return 1;
  }
  const int n = 6, ld = 7;
  const auto Ah = ints(ld * ld, 1), Bh = ints(ld * ld, 2), Ch = ints(ld * ld, 3), xh = ints(16, 4);
  double *A = dev(Ah), *B = dev(Bh), *C = dev(Ch), *x = dev(xh), *y = dev(xh);
  const double al = 2, be = -1;
  // Level 2.
  same("dgemv_v2_64", y, xh, [&] { return cublasDgemv(h, CUBLAS_OP_T, n, 5, &al, A, ld, x, 1, &be, y, 2); },
       [&] { return cublasDgemv_v2_64(h, CUBLAS_OP_T, n, 5, &al, A, ld, x, 1, &be, y, 2); });
  same("dgbmv_v2_64", y, xh, [&] { return cublasDgbmv(h, CUBLAS_OP_N, n, n, 1, 2, &al, A, ld, x, 1, &be, y, 1); },
       [&] { return cublasDgbmv_v2_64(h, CUBLAS_OP_N, n, n, 1, 2, &al, A, ld, x, 1, &be, y, 1); });
  same("dsymv_v2_64", y, xh, [&] { return cublasDsymv(h, CUBLAS_FILL_MODE_LOWER, n, &al, A, ld, x, 1, &be, y, -1); },
       [&] { return cublasDsymv_v2_64(h, CUBLAS_FILL_MODE_LOWER, n, &al, A, ld, x, 1, &be, y, -1); });
  same("dspmv_v2_64", y, xh, [&] { return cublasDspmv(h, CUBLAS_FILL_MODE_UPPER, n, &al, A, x, 2, &be, y, 1); },
       [&] { return cublasDspmv_v2_64(h, CUBLAS_FILL_MODE_UPPER, n, &al, A, x, 2, &be, y, 1); });
  same("dtbmv_v2_64", y, xh,
       [&] { return cublasDtbmv(h, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, n, 2, A, ld, y, 1); },
       [&] { return cublasDtbmv_v2_64(h, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, n, 2, A, ld, y, 1); });
  same("dger_v2_64", C, Ch, [&] { return cublasDger(h, n, 5, &al, x, 1, y, 2, C, ld); },
       [&] { return cublasDger_v2_64(h, n, 5, &al, x, 1, y, 2, C, ld); });
  same("dspr2_v2_64", C, Ch, [&] { return cublasDspr2(h, CUBLAS_FILL_MODE_LOWER, n, &al, x, 1, y, 1, C); },
       [&] { return cublasDspr2_v2_64(h, CUBLAS_FILL_MODE_LOWER, n, &al, x, 1, y, 1, C); });
  // Level 3.
  same("dgemm_v2_64", C, Ch, [&] { return cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_T, n, 5, 4, &al, A, ld, B, ld, &be, C, ld); },
       [&] { return cublasDgemm_v2_64(h, CUBLAS_OP_N, CUBLAS_OP_T, n, 5, 4, &al, A, ld, B, ld, &be, C, ld); });
  same("dsyrk_v2_64", C, Ch, [&] { return cublasDsyrk(h, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, n, 3, &al, A, ld, &be, C, ld); },
       [&] { return cublasDsyrk_v2_64(h, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, n, 3, &al, A, ld, &be, C, ld); });
  same("dsyrkx_64", C, Ch,
       [&] { return cublasDsyrkx(h, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, 3, &al, A, ld, B, ld, &be, C, ld); },
       [&] { return cublasDsyrkx_64(h, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, 3, &al, A, ld, B, ld, &be, C, ld); });
  same("dtrmm_v2_64", C, Ch,
       [&] { return cublasDtrmm(h, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, 4, n, &al, A, ld, B, ld, C, ld); },
       [&] { return cublasDtrmm_v2_64(h, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, 4, n, &al, A, ld, B, ld, C, ld); });
  same("ddgmm_64", C, Ch, [&] { return cublasDdgmm(h, CUBLAS_SIDE_LEFT, n, 5, A, ld, x, 1, C, ld); },
       [&] { return cublasDdgmm_64(h, CUBLAS_SIDE_LEFT, n, 5, A, ld, x, 1, C, ld); });
  // Batched.
  same("dgemmStridedBatched_64", C, Ch,
       [&] { return cublasDgemmStridedBatched(h, CUBLAS_OP_N, CUBLAS_OP_N, 3, 3, 3, &al, A, ld, 3, B, ld, 3, &be, C, ld, 3, 2); },
       [&] { return cublasDgemmStridedBatched_64(h, CUBLAS_OP_N, CUBLAS_OP_N, 3, 3, 3, &al, A, ld, 3, B, ld, 3, &be, C, ld, 3, 2); });
  same("dgemvStridedBatched_64", y, xh,
       [&] { return cublasDgemvStridedBatched(h, CUBLAS_OP_N, 3, 3, &al, A, ld, 3, x, 1, 3, &be, y, 1, 3, 2); },
       [&] { return cublasDgemvStridedBatched_64(h, CUBLAS_OP_N, 3, 3, &al, A, ld, 3, x, 1, 3, &be, y, 1, 3, 2); });
  {
    std::vector<double*> pa = {A, A + 3}, pc = {C, C + 3};
    double** dpa = dev(pa);
    double** dpc = dev(pc);
    same("dtrsmBatched_64", C, Ch,
         [&] { return cublasDtrsmBatched(h, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, 3, 3, &al, dpa, ld, dpc, ld, 2); },
         [&] { return cublasDtrsmBatched_64(h, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, 3, 3, &al, dpa, ld, dpc, ld, 2); });
    cudaFree(dpa);
    cudaFree(dpc);
  }
  // Single precision and complex, one each.
  {
    std::vector<float> Af(Ah.begin(), Ah.end()), Cf(Ch.begin(), Ch.end());
    float *dA = dev(Af), *dC = dev(Cf);
    const float a = 1, b = 1;
    same("sgemm_v2_64", dC, Cf, [&] { return cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, 4, 4, 4, &a, dA, ld, dA, ld, &b, dC, ld); },
         [&] { return cublasSgemm_v2_64(h, CUBLAS_OP_T, CUBLAS_OP_N, 4, 4, 4, &a, dA, ld, dA, ld, &b, dC, ld); });
    std::vector<cuComplex> Ac(ld * ld), Cc(ld * ld);
    for (int i = 0; i < ld * ld; ++i) Ac[i] = make_cuComplex((float)Ah[i], (float)Bh[i]), Cc[i] = make_cuComplex((float)Ch[i], 1);
    cuComplex *dAc = dev(Ac), *dCc = dev(Cc);
    const cuComplex ac = make_cuComplex(1, 1);
    const float br = 2;
    same("cherk_v2_64", dCc, Cc, [&] { return cublasCherk(h, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_C, 4, 3, &br, dAc, ld, &br, dCc, ld); },
         [&] { return cublasCherk_v2_64(h, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_C, 4, 3, &br, dAc, ld, &br, dCc, ld); });
    same("chpr2_v2_64", dCc, Cc, [&] { return cublasChpr2(h, CUBLAS_FILL_MODE_UPPER, 4, &ac, dAc, 1, dAc + ld, 1, dCc); },
         [&] { return cublasChpr2_v2_64(h, CUBLAS_FILL_MODE_UPPER, 4, &ac, dAc, 1, dAc + ld, 1, dCc); });
    cudaFree(dA);
    cudaFree(dC);
    cudaFree(dAc);
    cudaFree(dCc);
  }
  // Bad arguments come back as the 32-bit form's.
  IS(cublasDgemm_v2_64(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &al, A, n - 1, B, ld, &be, C, ld), CUBLAS_STATUS_INVALID_VALUE);
  IS(cublasDgbmv_v2_64(h, CUBLAS_OP_N, n, n, 1, 2, &al, A, 3, x, 1, &be, y, 1), CUBLAS_STATUS_INVALID_VALUE);

  // Handle settings.
  cublasAtomicsMode_t am = (cublasAtomicsMode_t)7;
  IS(cublasGetAtomicsMode(h, &am), CUBLAS_STATUS_SUCCESS);
  check(am == CUBLAS_ATOMICS_NOT_ALLOWED, "atomics not allowed by default");
  IS(cublasSetAtomicsMode(h, CUBLAS_ATOMICS_ALLOWED), CUBLAS_STATUS_SUCCESS);
  IS(cublasGetAtomicsMode(h, &am), CUBLAS_STATUS_SUCCESS);
  check(am == CUBLAS_ATOMICS_ALLOWED, "atomics mode read back");
  IS(cublasSetAtomicsMode(h, (cublasAtomicsMode_t)5), CUBLAS_STATUS_INVALID_VALUE);
  IS(cublasGetAtomicsMode(h, nullptr), CUBLAS_STATUS_INVALID_VALUE);
  int sm = -1, sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  IS(cublasGetSmCountTarget(h, &sm), CUBLAS_STATUS_SUCCESS);
  check(sm == 0, "SM count target 0 by default");
  IS(cublasSetSmCountTarget(h, 10), CUBLAS_STATUS_SUCCESS);
  IS(cublasGetSmCountTarget(h, &sm), CUBLAS_STATUS_SUCCESS);
  check(sm == 10, "SM count target read back");
  IS(cublasSetSmCountTarget(h, -1), CUBLAS_STATUS_INVALID_VALUE);
  IS(cublasSetSmCountTarget(h, 1000), CUBLAS_STATUS_INVALID_VALUE);
  IS(cublasGetSmCountTarget(h, nullptr), CUBLAS_STATUS_INVALID_VALUE);
#if CUBLAS_VER_MAJOR >= 13
  cublasEmulationStrategy_t es = (cublasEmulationStrategy_t)9;
  IS(cublasGetEmulationStrategy(h, &es), CUBLAS_STATUS_SUCCESS);
  check(es == CUBLAS_EMULATION_STRATEGY_DEFAULT, "emulation strategy DEFAULT by default");
  IS(cublasSetEmulationStrategy(h, CUBLAS_EMULATION_STRATEGY_EAGER), CUBLAS_STATUS_SUCCESS);
  IS(cublasSetEmulationStrategy(h, (cublasEmulationStrategy_t)7), CUBLAS_STATUS_INVALID_VALUE);
  IS(cublasGetEmulationStrategy(h, &es), CUBLAS_STATUS_SUCCESS);
  check(es == CUBLAS_EMULATION_STRATEGY_EAGER, "emulation strategy read back");
  cudaEmulationSpecialValuesSupport sv = (cudaEmulationSpecialValuesSupport)99;
  IS(cublasGetEmulationSpecialValuesSupport(h, &sv), CUBLAS_STATUS_SUCCESS);
  check((int)sv == 0xFFFF, "every special value supported by default");
  IS(cublasSetEmulationSpecialValuesSupport(h, (cudaEmulationSpecialValuesSupport)2), CUBLAS_STATUS_SUCCESS);
  cudaEmulationMantissaControl mc = (cudaEmulationMantissaControl)99;
  IS(cublasGetFixedPointEmulationMantissaControl(h, &mc), CUBLAS_STATUS_SUCCESS);
  check((int)mc == 0, "mantissa control DYNAMIC by default");
  IS(cublasSetFixedPointEmulationMantissaControl(h, (cudaEmulationMantissaControl)9), CUBLAS_STATUS_INVALID_VALUE);
  int bits = -1;
  IS(cublasGetFixedPointEmulationMaxMantissaBitCount(h, &bits), CUBLAS_STATUS_SUCCESS);
  check(bits == 0, "max mantissa bit count 0 by default");
  IS(cublasSetFixedPointEmulationMaxMantissaBitCount(h, -1), CUBLAS_STATUS_INVALID_VALUE);
  IS(cublasSetFixedPointEmulationMaxMantissaBitCount(h, 1000), CUBLAS_STATUS_SUCCESS);
  IS(cublasSetFixedPointEmulationMantissaBitOffset(h, -3), CUBLAS_STATUS_SUCCESS);
  int cnt = 0, *p = &cnt;
  IS(cublasGetFixedPointEmulationMantissaBitCountPointer(h, &p), CUBLAS_STATUS_SUCCESS);
  check(p == nullptr, "no mantissa bit count pointer by default");
#endif
  check(cublasGetCudartVersion() == CUDART_VERSION, "cublasGetCudartVersion");
  cublasLogCallback cb = (cublasLogCallback)0x1;
  IS(cublasGetLoggerCallback(&cb), CUBLAS_STATUS_SUCCESS);
  check(cb == nullptr, "no logger callback by default");
  IS(cublasGetLoggerCallback(nullptr), CUBLAS_STATUS_INVALID_VALUE);
  IS(cublasLoggerConfigure(0, 0, 0, nullptr), CUBLAS_STATUS_SUCCESS);
  for (void* q : {(void*)A, (void*)B, (void*)C, (void*)x, (void*)y}) cudaFree(q);
  cublasDestroy(h);
  std::printf(failures ? "FAIL: %d cuBLAS _64 and settings checks\n" : "PASS: every cuBLAS _64 and settings check\n",
              failures);
  return failures ? 1 : 0;
}
