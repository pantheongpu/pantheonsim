// cuBLAS's fixed-point emulation of double precision (CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT
// and CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH), and the controls that steer it
// (cublasSetEmulationStrategy and the cublasSetFixedPointEmulation... family).
// Every check passes on an RTX 3060 with NVIDIA's cuBLAS 13.0 too:
//
//   * which compute type goes with which data type (78 with R_32F, C_32F,
//     R_16F, R_16BF; 79 with R_64F and C_64F);
//   * when emulation happens: strategy EAGER, and the compute type 79 or the
//     handle's math mode 8; DEFAULT and PERFORMANT never emulate (DEFAULT is
//     EAGER only when CUBLAS_EMULATION_STRATEGY says so), and a call that is
//     not emulated leaves the mantissa bit count pointer alone;
//   * FIXED mantissa control with a maximum of M bits: the result is the
//     product of the operands cut to 7 + 8 (M / 8) bits per element, a row of
//     A and a column of B sharing the exponent of their largest element
//     (nvidia/src/fixed_point_gemm.hpp). The results are compared, bit for
//     bit, with those the card printed
//     (nvidia/tests/data/blas_emulation_paths.card.txt, refreshed by
//     run_blas_emulation_card.sh --update), for every M from 4 to 64;
//   * the pointer receives the bit count; DYNAMIC control picks it from the
//     data (the card's rule is not public and this library's differs, so only
//     "enough for double precision" is checked), and a maximum below what the
//     data needs sends the call to plain double precision.
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <utility>
#include <unistd.h>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#if CUBLAS_VER_MAJOR < 13
// CUDA 12's headers know none of this: the program has nothing to check there.
int main() {
  std::printf("SKIP: fixed-point emulation needs the CUDA 13 cuBLAS headers\n");
  return 0;
}
#else

static int failures = 0;
static void check(bool ok, const char* what) {
  std::printf("%-4s %s\n", ok ? "ok" : "FAIL", what);
  if (!ok) ++failures;
}

static uint64_t fnv(const std::vector<double>& v) {
  uint64_t h = 1469598103934665603ULL;
  for (double x : v) {
    uint64_t u;
    std::memcpy(&u, &x, 8);
    for (int i = 0; i < 8; ++i) {
      h ^= (u >> (8 * i)) & 255;
      h *= 1099511628211ULL;
    }
  }
  return h;
}

struct Lcg {
  uint64_t s = 99;
  double next() {   // uniform in [-1, 1)
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    return (double)(int64_t)(s >> 11) / (double)(1LL << 52) - 1.0;
  }
  int below(int n) {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    return (int)((s >> 33) % (uint64_t)n);
  }
};

struct Ctx {
  cublasHandle_t h;
  int* dbits;
  double *a, *b, *c;
};

static int bits_of(Ctx& x) {
  int v = -99;
  cudaMemcpy(&v, x.dbits, 4, cudaMemcpyDeviceToHost);
  return v;
}
static void reset_bits(Ctx& x) {
  const int v = -99;
  cudaMemcpy(x.dbits, &v, 4, cudaMemcpyHostToDevice);
}

// One GEMM, op(A) m x k and op(B) k x n, alpha 1 and beta 0.
static std::vector<double> gemm(Ctx& x, int m, int n, int k, bool ta, bool tb, const std::vector<double>& A, const std::vector<double>& B,
                                cublasComputeType_t ct, cublasStatus_t* st = nullptr) {
  cudaMemcpy(x.a, A.data(), A.size() * 8, cudaMemcpyHostToDevice);
  cudaMemcpy(x.b, B.data(), B.size() * 8, cudaMemcpyHostToDevice);
  cudaMemset(x.c, 0, (size_t)m * n * 8);
  const double one = 1, zero = 0;
  const cublasStatus_t s = cublasGemmEx(x.h, ta ? CUBLAS_OP_T : CUBLAS_OP_N, tb ? CUBLAS_OP_T : CUBLAS_OP_N, m, n, k, &one, x.a, CUDA_R_64F,
                                        ta ? k : m, x.b, CUDA_R_64F, tb ? n : k, &zero, x.c, CUDA_R_64F, m, ct, CUBLAS_GEMM_DEFAULT);
  if (st) *st = s;
  cudaDeviceSynchronize();
  std::vector<double> C((size_t)m * n);
  cudaMemcpy(C.data(), x.c, C.size() * 8, cudaMemcpyDeviceToHost);
  return C;
}

// The error against a long double product, relative to k times the largest
// element of the row of A and the column of B: the scale an emulation with a
// fixed number of bits per element can promise.
static double max_rel_error(int m, int n, int k, bool ta, bool tb, const std::vector<double>& A, const std::vector<double>& B,
                            const std::vector<double>& C) {
  double worst = 0;
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < m; ++i) {
      long double s = 0;
      double rmax = 0, cmax = 0;
      for (int p = 0; p < k; ++p) {
        const double av = ta ? A[(size_t)p + (size_t)i * k] : A[(size_t)i + (size_t)p * m];
        const double bv = tb ? B[(size_t)j + (size_t)p * n] : B[(size_t)p + (size_t)j * k];
        s += (long double)av * bv;
        rmax = std::fmax(rmax, std::fabs(av));
        cmax = std::fmax(cmax, std::fabs(bv));
      }
      worst = std::fmax(worst, (double)(std::fabs((long double)C[(size_t)i + (size_t)j * m] - s) / ((long double)k * rmax * cmax + 1e-300L)));
    }
  return worst;
}

// A fresh process whose environment says CUBLAS_EMULATION_STRATEGY=eager: does a handle left at DEFAULT emulate?
static int child_probe() {
  Ctx x{};
  cublasCreate(&x.h);
  cudaMalloc(&x.dbits, 4);
  cudaMalloc(&x.a, 1 << 16);
  cudaMalloc(&x.b, 1 << 16);
  cudaMalloc(&x.c, 1 << 16);
  cublasSetFixedPointEmulationMantissaBitCountPointer(x.h, x.dbits);
  cublasSetFixedPointEmulationMantissaControl(x.h, CUDA_EMULATION_MANTISSA_CONTROL_FIXED);
  cublasSetFixedPointEmulationMaxMantissaBitCount(x.h, 16);
  std::vector<double> v(16 * 16, 1.5);
  reset_bits(x);
  gemm(x, 16, 16, 16, false, false, v, v, (cublasComputeType_t)79);
  std::printf("CHILD emulates=%d\n", bits_of(x) == 16 ? 1 : 0);
  return 0;
}

int main() {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  if (std::getenv("EMU_CHILD")) return child_probe();
  int dev = 0, major = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
  if (major < 8) {
    std::printf("SKIP: fixed-point emulation needs compute capability 8 or later (found %d)\n", major);
    return 0;
  }
  Ctx x{};
  cublasCreate(&x.h);
  cudaMalloc(&x.dbits, 4);
  cudaMalloc(&x.a, 8 << 20);
  cudaMalloc(&x.b, 8 << 20);
  cudaMalloc(&x.c, 8 << 20);
  cublasSetFixedPointEmulationMantissaBitCountPointer(x.h, x.dbits);
  char what[220];

  // ---- which compute type goes with which data ----
  {
    void *a, *b, *c;
    cudaMalloc(&a, 4096);
    cudaMalloc(&b, 4096);
    cudaMalloc(&c, 4096);
    cudaMemset(a, 0, 4096);
    cudaMemset(b, 0, 4096);
    const double d1[2] = {1, 0}, d0[2] = {0, 0};
    const float f1[2] = {1, 0}, f0[2] = {0, 0};
    struct Row { cudaDataType t; const char* name; bool want78, want79; } rows[] = {
        {CUDA_R_32F, "R_32F", true, false}, {CUDA_R_64F, "R_64F", false, true}, {CUDA_C_32F, "C_32F", true, false},
        {CUDA_C_64F, "C_64F", false, true}, {CUDA_R_16F, "R_16F", true, false},  {CUDA_R_16BF, "R_16BF", true, false},
        {CUDA_R_8I, "R_8I", false, false}};
    bool ok = true;
    for (const Row& r : rows) {
      const cublasStatus_t s78 = cublasGemmEx(x.h, CUBLAS_OP_N, CUBLAS_OP_N, 8, 8, 8, f1, a, r.t, 8, b, r.t, 8, f0, c, r.t, 8,
                                              CUBLAS_COMPUTE_32F_EMULATED_16BFX9, CUBLAS_GEMM_DEFAULT);
      const cublasStatus_t s79 = cublasGemmEx(x.h, CUBLAS_OP_N, CUBLAS_OP_N, 8, 8, 8, d1, a, r.t, 8, b, r.t, 8, d0, c, r.t, 8,
                                              CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT, CUBLAS_GEMM_DEFAULT);
      const bool g78 = s78 == CUBLAS_STATUS_SUCCESS, g79 = s79 == CUBLAS_STATUS_SUCCESS;
      if (g78 != r.want78 || g79 != r.want79 || (!g78 && s78 != CUBLAS_STATUS_NOT_SUPPORTED) || (!g79 && s79 != CUBLAS_STATUS_NOT_SUPPORTED)) {
        std::printf("   %s: 78 -> %d, 79 -> %d\n", r.name, (int)s78, (int)s79);
        ok = false;
      }
    }
    check(ok, "COMPUTE_32F_EMULATED_16BFX9 takes R_32F, C_32F, R_16F, R_16BF; COMPUTE_64F_EMULATED_FIXEDPOINT takes R_64F, C_64F; else NOT_SUPPORTED");
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
  }

  // ---- when emulation happens ----
  const int m = 24, n = 24, k = 24;
  Lcg rng;
  std::vector<double> A((size_t)m * k), B((size_t)k * n);
  for (auto& v : A) v = rng.next();
  for (auto& v : B) v = rng.next();
  cublasSetFixedPointEmulationMantissaControl(x.h, CUDA_EMULATION_MANTISSA_CONTROL_FIXED);
  cublasSetFixedPointEmulationMaxMantissaBitCount(x.h, 16);
  const auto native = gemm(x, m, n, k, false, false, A, B, CUBLAS_COMPUTE_64F);
  for (int strategy : {0, 1, 2}) {
    cublasSetEmulationStrategy(x.h, (cublasEmulationStrategy_t)strategy);
    reset_bits(x);
    const auto r = gemm(x, m, n, k, false, false, A, B, (cublasComputeType_t)79);
    const bool emulated = r != native;
    std::snprintf(what, sizeof what, "strategy %d, COMPUTE_64F_EMULATED_FIXEDPOINT, 16 bits: %s", strategy,
                  strategy == 2 ? "emulated, the pointer gets 16" : "plain double precision, the pointer untouched");
    check(strategy == 2 ? (emulated && bits_of(x) == 16) : (!emulated && bits_of(x) == -99), what);
  }
  cublasSetEmulationStrategy(x.h, CUBLAS_EMULATION_STRATEGY_EAGER);
  {
    reset_bits(x);
    const auto r = gemm(x, m, n, k, false, false, A, B, CUBLAS_COMPUTE_64F);
    check(r == native && bits_of(x) == -99, "EAGER with plain COMPUTE_64F and the default math mode: not emulated");
    cublasSetMathMode(x.h, (cublasMath_t)8);   // CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH
    reset_bits(x);
    const auto r2 = gemm(x, m, n, k, false, false, A, B, CUBLAS_COMPUTE_64F);
    const auto emulated = gemm(x, m, n, k, false, false, A, B, (cublasComputeType_t)79);
    check(r2 == emulated && r2 != native && bits_of(x) == 16, "EAGER with the math mode FP64_EMULATED_FIXEDPOINT: GemmEx with COMPUTE_64F is emulated");
    // Dgemm with the math mode.
    const double one = 1, zero = 0;
    cudaMemcpy(x.a, A.data(), A.size() * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(x.b, B.data(), B.size() * 8, cudaMemcpyHostToDevice);
    reset_bits(x);
    cublasDgemm(x.h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &one, x.a, m, x.b, k, &zero, x.c, m);
    std::vector<double> d((size_t)m * n);
    cudaMemcpy(d.data(), x.c, d.size() * 8, cudaMemcpyDeviceToHost);
    check(d == emulated && bits_of(x) == 16, "Dgemm under the math mode is emulated, the same product");
    cublasSetMathMode(x.h, CUBLAS_DEFAULT_MATH);
    reset_bits(x);
    cublasDgemm(x.h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &one, x.a, m, x.b, k, &zero, x.c, m);
    cudaMemcpy(d.data(), x.c, d.size() * 8, cudaMemcpyDeviceToHost);
    check(d == native && bits_of(x) == -99, "Dgemm with the default math mode is not");
  }
  {   // DEFAULT is PERFORMANT unless CUBLAS_EMULATION_STRATEGY said "eager" when the library started: the variable is read once, so
      // that case is a child process, and a variable set late changes nothing. An explicit EAGER emulates whatever the variable says.
    setenv("CUBLAS_EMULATION_STRATEGY", "performant", 1);
    cublasSetEmulationStrategy(x.h, CUBLAS_EMULATION_STRATEGY_EAGER);
    reset_bits(x);
    const auto r2 = gemm(x, m, n, k, false, false, A, B, (cublasComputeType_t)79);
    check(r2 != native && bits_of(x) == 16, "explicit EAGER emulates, whatever CUBLAS_EMULATION_STRATEGY says");
    setenv("CUBLAS_EMULATION_STRATEGY", "eager", 1);
    cublasHandle_t late;
    cublasCreate(&late);
    cublasSetFixedPointEmulationMantissaBitCountPointer(late, x.dbits);
    cublasSetFixedPointEmulationMantissaControl(late, CUDA_EMULATION_MANTISSA_CONTROL_FIXED);
    cublasSetFixedPointEmulationMaxMantissaBitCount(late, 16);
    std::swap(x.h, late);
    reset_bits(x);
    gemm(x, m, n, k, false, false, A, B, (cublasComputeType_t)79);
    check(bits_of(x) == -99, "CUBLAS_EMULATION_STRATEGY set after the library started changes nothing");
    std::swap(x.h, late);
    cublasDestroy(late);
    unsetenv("CUBLAS_EMULATION_STRATEGY");
    char exe[4096] = {0};
    const ssize_t len = readlink("/proc/self/exe", exe, sizeof exe - 1);
    if (len > 0) {
      const std::string cmd = std::string("EMU_CHILD=1 CUBLAS_EMULATION_STRATEGY=eager '") + exe + "' 2>&1";
      std::string out;
      if (FILE* p = popen(cmd.c_str(), "r")) {
        char buf[256];
        while (std::fgets(buf, sizeof buf, p)) out += buf;
        pclose(p);
      }
      check(out.find("CHILD emulates=1") != std::string::npos, "DEFAULT with CUBLAS_EMULATION_STRATEGY=eager from the start emulates");
    }
  }
  {   // 78: BF16x9 emulates only from compute capability 10; below it single precision runs as it does under COMPUTE_32F.
    float *a, *b, *c, *c2;
    const int q = 32;
    cudaMalloc(&a, q * q * 4);
    cudaMalloc(&b, q * q * 4);
    cudaMalloc(&c, q * q * 4);
    cudaMalloc(&c2, q * q * 4);
    std::vector<float> fa(q * q), fb(q * q);
    for (auto& v : fa) v = (float)rng.next();
    for (auto& v : fb) v = (float)rng.next();
    cudaMemcpy(a, fa.data(), q * q * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(b, fb.data(), q * q * 4, cudaMemcpyHostToDevice);
    const float one = 1, zero = 0;
    cublasGemmEx(x.h, CUBLAS_OP_N, CUBLAS_OP_N, q, q, q, &one, a, CUDA_R_32F, q, b, CUDA_R_32F, q, &zero, c, CUDA_R_32F, q, CUBLAS_COMPUTE_32F,
                 CUBLAS_GEMM_DEFAULT);
    cublasGemmEx(x.h, CUBLAS_OP_N, CUBLAS_OP_N, q, q, q, &one, a, CUDA_R_32F, q, b, CUDA_R_32F, q, &zero, c2, CUDA_R_32F, q,
                 CUBLAS_COMPUTE_32F_EMULATED_16BFX9, CUBLAS_GEMM_DEFAULT);
    std::vector<float> r1(q * q), r2(q * q);
    cudaMemcpy(r1.data(), c, q * q * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(r2.data(), c2, q * q * 4, cudaMemcpyDeviceToHost);
    if (major < 10) check(r1 == r2, "COMPUTE_32F_EMULATED_16BFX9 below compute capability 10: the single-precision product, unchanged");
    else {
      double worst = 0;
      for (int i = 0; i < q * q; ++i) worst = std::fmax(worst, std::fabs((double)r1[i] - r2[i]));
      check(worst < 1e-4, "COMPUTE_32F_EMULATED_16BFX9 gives a single-precision product");
    }
    cudaFree(a); cudaFree(b); cudaFree(c); cudaFree(c2);
  }

  // ---- FIXED mantissa control against the card ----
  std::map<std::string, std::string> card;
  const bool print = std::getenv("EMU_PRINT") != nullptr;
  if (!print) {
    const char* dir = std::getenv("VGPU_E2E_DATA");
    std::ifstream in(std::string(dir ? dir : ".") + "/blas_emulation_paths.card.txt");
    std::string l;
    while (std::getline(in, l))
      if (!l.empty() && l[0] != '#') card[l.substr(0, l.rfind(' '))] = l;
  }
  cublasSetEmulationStrategy(x.h, CUBLAS_EMULATION_STRATEGY_EAGER);
  cublasSetFixedPointEmulationMantissaControl(x.h, CUDA_EMULATION_MANTISSA_CONTROL_FIXED);
  struct Shape { int m, n, k; bool ta, tb; int spread; };
  const Shape shapes[] = {{20, 20, 20, false, false, 0}, {33, 17, 45, false, false, 0}, {17, 33, 9, true, false, 0},
                          {25, 25, 25, false, true, 4},  {40, 7, 64, true, true, 12},    {8, 50, 31, false, false, 30},
                          {64, 64, 64, false, false, 0}};
  int compared = 0, equal = 0;
  for (const Shape& s : shapes) {
    std::vector<double> SA((size_t)s.m * s.k), SB((size_t)s.k * s.n);
    for (auto& v : SA) { v = rng.next(); if (s.spread) v = std::ldexp(v, rng.below(s.spread + 1) - s.spread / 2); }
    for (auto& v : SB) { v = rng.next(); if (s.spread) v = std::ldexp(v, rng.below(s.spread + 1) - s.spread / 2); }
    for (int M : {4, 8, 16, 24, 32, 40, 48, 56, 64}) {
      cublasSetFixedPointEmulationMaxMantissaBitCount(x.h, M);
      reset_bits(x);
      cublasStatus_t st;
      const auto C = gemm(x, s.m, s.n, s.k, s.ta, s.tb, SA, SB, (cublasComputeType_t)79, &st);
      char key[120];
      std::snprintf(key, sizeof key, "S %d %d %d %d %d %d M %d", s.m, s.n, s.k, (int)s.ta, (int)s.tb, s.spread, M);
      char line[220];
      std::snprintf(line, sizeof line, "%s %016llx", key, (unsigned long long)fnv(C));
      if (print) {
        std::printf("%s\n", line);
        continue;
      }
      if (st != CUBLAS_STATUS_SUCCESS || bits_of(x) != M) {
        std::snprintf(what, sizeof what, "%s: status %d, pointer %d", key, (int)st, bits_of(x));
        check(false, what);
      }
      auto it = card.find(key);
      if (it != card.end()) {
        ++compared;
        if (it->second == line) ++equal;
        else {
          std::snprintf(what, sizeof what, "%s differs from the card's result", key);
          check(false, what);
        }
      }
      if (M == 32 || M == 40 || M == 64) {
        // Accuracy: about 2^-(7 + 8 (M/8)) of k times the largest elements, with room for the cut; double precision at 64.
        const double bound = M == 64 ? 1e-15 : std::ldexp(64.0, -(7 + 8 * (M / 8)));
        const double e = max_rel_error(s.m, s.n, s.k, s.ta, s.tb, SA, SB, C);
        if (e > bound) {
          std::snprintf(what, sizeof what, "%s: error %.2e above %.2e", key, e, bound);
          check(false, what);
        }
      }
    }
  }
  if (print) return 0;
  if (!card.empty()) {
    std::snprintf(what, sizeof what, "FIXED control, M = 4 to 64: %d of %d results equal the card's, bit for bit", equal, compared);
    check(compared > 0 && equal == compared, what);
  } else {
    std::printf("SKIP the card's results are not available (VGPU_E2E_DATA)\n");
  }

  // ---- DYNAMIC control ----
  cublasSetFixedPointEmulationMantissaControl(x.h, CUDA_EMULATION_MANTISSA_CONTROL_DYNAMIC);
  cublasSetFixedPointEmulationMaxMantissaBitCount(x.h, 0);
  cublasSetFixedPointEmulationMantissaBitOffset(x.h, 0);
  {
    reset_bits(x);
    const auto r = gemm(x, m, n, k, false, false, A, B, (cublasComputeType_t)79);
    const int bits = bits_of(x);
    const double e = max_rel_error(m, n, k, false, false, A, B, r);
    std::snprintf(what, sizeof what, "dynamic: %d bits chosen (at least 54), error %.2e of k times the largest elements (double precision's is about 1e-16)", bits, e);
    check(bits >= 54 && e < 1e-15, what);
    cublasSetFixedPointEmulationMantissaBitOffset(x.h, -8);
    reset_bits(x);
    gemm(x, m, n, k, false, false, A, B, (cublasComputeType_t)79);
    const int less = bits_of(x);
    cublasSetFixedPointEmulationMantissaBitOffset(x.h, 0);
    check(less == bits - 8, "the offset is added to the bit count dynamic control chooses");
    // A maximum below what the data needs: the product comes out as plain double precision gives it (not cut to the
    // maximum), and the pointer gets the bit count the data needed -- as the card does (it wrote 62 for data this library
    // would give 71 for).
    cublasSetFixedPointEmulationMaxMantissaBitCount(x.h, 8);
    reset_bits(x);
    const auto r2 = gemm(x, m, n, k, false, false, A, B, (cublasComputeType_t)79);
    const double e2 = max_rel_error(m, n, k, false, false, A, B, r2);
    std::snprintf(what, sizeof what, "a maximum below the need: plain double precision's accuracy (error %.1e), the pointer gets %d", e2, bits_of(x));
    check(e2 < 1e-14 && bits_of(x) == bits, what);
    cublasSetFixedPointEmulationMaxMantissaBitCount(x.h, 0);
  }
  // Alpha, beta, and a batch.
  {
    cublasSetFixedPointEmulationMantissaControl(x.h, CUDA_EMULATION_MANTISSA_CONTROL_FIXED);
    cublasSetFixedPointEmulationMaxMantissaBitCount(x.h, 24);
    const auto P = gemm(x, m, n, k, false, false, A, B, (cublasComputeType_t)79);
    std::vector<double> C0((size_t)m * n);
    for (auto& v : C0) v = rng.next();
    cudaMemcpy(x.a, A.data(), A.size() * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(x.b, B.data(), B.size() * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(x.c, C0.data(), C0.size() * 8, cudaMemcpyHostToDevice);
    const double al = 2.5, be = -0.5;
    cublasGemmEx(x.h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &al, x.a, CUDA_R_64F, m, x.b, CUDA_R_64F, k, &be, x.c, CUDA_R_64F, m,
                 (cublasComputeType_t)79, CUBLAS_GEMM_DEFAULT);
    std::vector<double> got(C0.size());
    cudaMemcpy(got.data(), x.c, got.size() * 8, cudaMemcpyDeviceToHost);
    double worst = 0;
    for (size_t i = 0; i < got.size(); ++i) worst = std::fmax(worst, std::fabs(got[i] - (al * P[i] + be * C0[i])));
    check(worst < 1e-14, "alpha and beta scale the emulated product");
    // Strided batched: the same product per batch.
    cudaMemcpy(x.a + (size_t)m * k, A.data(), A.size() * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(x.a, A.data(), A.size() * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(x.b, B.data(), B.size() * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(x.b + (size_t)k * n, B.data(), B.size() * 8, cudaMemcpyHostToDevice);
    const double one = 1, zero = 0;
    cublasGemmStridedBatchedEx(x.h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &one, x.a, CUDA_R_64F, m, (long long)m * k, x.b, CUDA_R_64F, k,
                               (long long)k * n, &zero, x.c, CUDA_R_64F, m, (long long)m * n, 2, (cublasComputeType_t)79, CUBLAS_GEMM_DEFAULT);
    std::vector<double> both((size_t)m * n * 2);
    cudaMemcpy(both.data(), x.c, both.size() * 8, cudaMemcpyDeviceToHost);
    bool same = true;
    for (size_t i = 0; i < (size_t)m * n; ++i) same = same && both[i] == P[i] && both[i + (size_t)m * n] == P[i];
    check(same, "GemmStridedBatchedEx emulates each matrix of the batch");
  }
  // Zgemm under the math mode: accurate to the bits asked for.
  {
    cublasSetMathMode(x.h, (cublasMath_t)8);
    cublasSetFixedPointEmulationMaxMantissaBitCount(x.h, 32);
    const int q = 12;
    std::vector<cuDoubleComplex> za((size_t)q * q), zb((size_t)q * q), zc((size_t)q * q);
    for (auto& v : za) v = make_cuDoubleComplex(rng.next(), rng.next());
    for (auto& v : zb) v = make_cuDoubleComplex(rng.next(), rng.next());
    cuDoubleComplex *da, *db, *dc;
    cudaMalloc(&da, za.size() * 16);
    cudaMalloc(&db, zb.size() * 16);
    cudaMalloc(&dc, zc.size() * 16);
    cudaMemcpy(da, za.data(), za.size() * 16, cudaMemcpyHostToDevice);
    cudaMemcpy(db, zb.data(), zb.size() * 16, cudaMemcpyHostToDevice);
    const cuDoubleComplex one = make_cuDoubleComplex(1, 0), zero = make_cuDoubleComplex(0, 0);
    reset_bits(x);
    cublasZgemm(x.h, CUBLAS_OP_N, CUBLAS_OP_C, q, q, q, &one, da, q, db, q, &zero, dc, q);
    cudaMemcpy(zc.data(), dc, zc.size() * 16, cudaMemcpyDeviceToHost);
    double worst = 0, scale = 0;
    for (int j = 0; j < q; ++j)
      for (int i = 0; i < q; ++i) {
        long double sr = 0, si = 0;
        for (int p = 0; p < q; ++p) {
          const long double ar = za[(size_t)i + (size_t)p * q].x, ai = za[(size_t)i + (size_t)p * q].y;
          const long double br = zb[(size_t)j + (size_t)p * q].x, bi = -zb[(size_t)j + (size_t)p * q].y;   // op(B) = B^H
          sr += ar * br - ai * bi;
          si += ar * bi + ai * br;
        }
        worst = std::fmax(worst, std::fabs(zc[(size_t)i + (size_t)j * q].x - (double)sr) + std::fabs(zc[(size_t)i + (size_t)j * q].y - (double)si));
        scale = std::fmax(scale, std::fabs((double)sr) + std::fabs((double)si));
      }
    std::snprintf(what, sizeof what, "Zgemm under the math mode, 32 bits: emulated (pointer %d), error %.1e of %.1f", bits_of(x), worst, scale);
    check(bits_of(x) == 32 && worst > 0 && worst < 1e-7 * scale, what);
    cublasSetMathMode(x.h, CUBLAS_DEFAULT_MATH);
    cudaFree(da); cudaFree(db); cudaFree(dc);
  }
  std::printf("%s\n", failures ? "FAIL" : "PASS");
  cublasDestroy(x.h);
  return failures ? 1 : 0;
}
#endif
