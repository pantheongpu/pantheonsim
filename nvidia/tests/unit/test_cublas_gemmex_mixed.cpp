// cublasGemmEx with half, bfloat16 and int8 operands (accumulated in float) against the loops it replaced, bit for
// bit, and the heap it holds while it runs.
//
// The library reads A and B as the bytes the device holds and widens each element to float as host_gemm.hpp packs a
// panel, and reads and writes C's elements in place: no float copy of A, B or C is made. The reference here is the
// way it used to work: widen every matrix whole with CUDA's own conversions, run the plain loops on the floats,
// convert every cell of C back (the cells a leading dimension leaves between columns go through float and back too,
// which is not the identity for a NaN that is not the one the conversion makes).
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "alloc_counter.hpp"
#include "vtest.hpp"

namespace {

size_t bytes_of(cudaDataType t) { return t == CUDA_R_32F ? 4 : t == CUDA_R_8I ? 1 : 2; }

float widen(cudaDataType t, const uint8_t* p) {
  switch (t) {
    case CUDA_R_16F: { __half h; std::memcpy(&h, p, 2); return __half2float(h); }
    case CUDA_R_16BF: { __nv_bfloat16 h; std::memcpy(&h, p, 2); return __bfloat162float(h); }
    case CUDA_R_8I: return static_cast<float>(static_cast<int8_t>(*p));
    default: { float f; std::memcpy(&f, p, 4); return f; }
  }
}
void narrow(cudaDataType t, float v, uint8_t* p) {
  switch (t) {
    case CUDA_R_16F: { const __half h = __float2half(v); std::memcpy(p, &h, 2); break; }
    case CUDA_R_16BF: { const __nv_bfloat16 h = __float2bfloat16(v); std::memcpy(p, &h, 2); break; }
    default: std::memcpy(p, &v, 4); break;
  }
}

// n elements: ordinary values (small multiples of 1/64), or, with `wild`, some random bit patterns (NaNs, infinities).
std::vector<uint8_t> make(cudaDataType t, size_t n, bool wild, std::mt19937_64& rng) {
  std::vector<uint8_t> raw(n * bytes_of(t));
  for (size_t i = 0; i < n; ++i) {
    uint8_t* p = raw.data() + i * bytes_of(t);
    if (t == CUDA_R_8I) { *p = static_cast<uint8_t>(rng()); continue; }
    if (wild && rng() % 20 == 0) {
      for (size_t b = 0; b < bytes_of(t); ++b) p[b] = static_cast<uint8_t>(rng());
      continue;
    }
    narrow(t, static_cast<float>((static_cast<int>(rng() % 513) - 256) / 64.0), p);
  }
  return raw;
}

struct DevBuf {
  void* p = nullptr;
  explicit DevBuf(const std::vector<uint8_t>& host) {
    cudaMalloc(&p, host.size() ? host.size() : 1);
    if (!host.empty()) cudaMemcpy(p, host.data(), host.size(), cudaMemcpyHostToDevice);
  }
  ~DevBuf() { cudaFree(p); }
  DevBuf(const DevBuf&) = delete;
  DevBuf& operator=(const DevBuf&) = delete;
};

bool same_bytes_or_both_nan(cudaDataType t, const uint8_t* x, const uint8_t* y) {
  if (std::memcmp(x, y, bytes_of(t)) == 0) return true;
  return std::isnan(widen(t, x)) && std::isnan(widen(t, y));   // which NaN wins when two meet is not fixed (host_gemm.hpp)
}

struct Case {
  cudaDataType a, b, c;
  int m, n, k;
  bool ta, tb;
  int pad;
  float alpha, beta;
  bool wild;
};

size_t run_case(cublasHandle_t h, const Case& cs, std::mt19937_64& rng) {
  const int lda = (cs.ta ? cs.k : cs.m) + cs.pad, ldb = (cs.tb ? cs.n : cs.k) + cs.pad, ldc = cs.m + cs.pad;
  const size_t acols = cs.ta ? cs.m : cs.k, bcols = cs.tb ? cs.k : cs.n;
  const size_t na = (acols - 1) * lda + (cs.ta ? cs.k : cs.m), nb = (bcols - 1) * ldb + (cs.tb ? cs.n : cs.k),
               nc = static_cast<size_t>(cs.n - 1) * ldc + cs.m;
  const auto A = make(cs.a, na, cs.wild, rng), B = make(cs.b, nb, cs.wild, rng);
  std::vector<uint8_t> C(nc * bytes_of(cs.c));
  for (auto& x : C) x = static_cast<uint8_t>(rng());   // C and the padding between its columns: any bytes at all
  if (!cs.wild) {
    const auto tidy = make(cs.c, nc, false, rng);
    for (size_t i = 0; i < nc; ++i)
      if (i % ldc < static_cast<size_t>(cs.m)) std::memcpy(C.data() + i * bytes_of(cs.c), tidy.data() + i * bytes_of(cs.c), bytes_of(cs.c));
  }
  DevBuf dA(A), dB(B), dC(C);
  const cublasStatus_t st = cublasGemmEx(h, cs.ta ? CUBLAS_OP_T : CUBLAS_OP_N, cs.tb ? CUBLAS_OP_T : CUBLAS_OP_N, cs.m, cs.n, cs.k,
                                         &cs.alpha, dA.p, cs.a, lda, dB.p, cs.b, ldb, &cs.beta, dC.p, cs.c, ldc, CUBLAS_COMPUTE_32F,
                                         CUBLAS_GEMM_DEFAULT);
  VCHECK_EQ(st, CUBLAS_STATUS_SUCCESS);
  std::vector<uint8_t> got(C.size());
  cudaMemcpy(got.data(), dC.p, got.size(), cudaMemcpyDeviceToHost);

  // The old way: whole matrices widened to float, the loops, every cell of C converted back.
  std::vector<float> fa(na), fb(nb), fc(nc);
  for (size_t i = 0; i < na; ++i) fa[i] = widen(cs.a, A.data() + i * bytes_of(cs.a));
  for (size_t i = 0; i < nb; ++i) fb[i] = widen(cs.b, B.data() + i * bytes_of(cs.b));
  for (size_t i = 0; i < nc; ++i) fc[i] = widen(cs.c, C.data() + i * bytes_of(cs.c));
  for (int j = 0; j < cs.n; ++j)
    for (int i = 0; i < cs.m; ++i) {
      float acc = 0.0f;
      for (int p = 0; p < cs.k; ++p) {
        const float a = fa[cs.ta ? p + static_cast<size_t>(i) * lda : i + static_cast<size_t>(p) * lda];
        const float b = fb[cs.tb ? j + static_cast<size_t>(p) * ldb : p + static_cast<size_t>(j) * ldb];
        acc += a * b;
      }
      float& c = fc[i + static_cast<size_t>(j) * ldc];
      c = cs.beta == 0.0f ? cs.alpha * acc : cs.alpha * acc + cs.beta * c;
    }
  std::vector<uint8_t> want(C.size());
  for (size_t i = 0; i < nc; ++i) narrow(cs.c, fc[i], want.data() + i * bytes_of(cs.c));
  size_t bad = 0;
  for (size_t i = 0; i < nc; ++i) {
    const bool output = i % ldc < static_cast<size_t>(cs.m);
    const uint8_t* w = want.data() + i * bytes_of(cs.c);
    const uint8_t* g = got.data() + i * bytes_of(cs.c);
    // An output cell may differ in which NaN it is; a padding cell holds one input NaN and must be the conversion's exactly.
    bad += output ? !same_bytes_or_both_nan(cs.c, w, g) : std::memcmp(w, g, bytes_of(cs.c)) != 0;
  }
  return bad;
}

}  // namespace

VTEST(cublas_gemmex_mixed_matches_the_loops_bit_for_bit) {
  cublasHandle_t h = nullptr;
  VCHECK_EQ(cublasCreate(&h), CUBLAS_STATUS_SUCCESS);
  std::mt19937_64 rng(1357);
  struct Shape { int m, n, k; };
  const Shape shapes[] = {{1, 1, 1}, {3, 5, 7}, {17, 33, 65}, {129, 130, 257}, {64, 300, 1}, {70, 70, 2}, {300, 50, 513}};
  struct Types { cudaDataType a, b, c; };
  const Types types[] = {
      {CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF}, {CUDA_R_16F, CUDA_R_16F, CUDA_R_16F}, {CUDA_R_16BF, CUDA_R_16BF, CUDA_R_32F},
      {CUDA_R_16F, CUDA_R_16F, CUDA_R_32F},    {CUDA_R_8I, CUDA_R_8I, CUDA_R_32F},
  };
  int cases = 0;
  size_t bad = 0;
  for (const Types& t : types)
    for (const Shape& s : shapes)
      for (int v = 0; v < 8; ++v) {
        Case c{t.a, t.b, t.c, s.m, s.n, s.k, (v & 1) != 0, (v & 2) != 0, v % 3 == 1 ? 3 : 0,
               v % 4 == 0 ? 1.0f : v % 4 == 1 ? -0.5f : 2.25f, v % 3 == 0 ? 0.0f : v % 3 == 1 ? 1.0f : -1.25f, v % 3 == 2};
        const size_t b = run_case(h, c, rng);
        if (b) std::printf("  differs: a=%d b=%d c=%d m=%d n=%d k=%d variant %d (%zu cells)\n", (int)t.a, (int)t.b, (int)t.c, s.m, s.n, s.k, v, b);
        bad += b;
        ++cases;
      }
  std::printf("  %d GemmEx calls, %zu cells differ\n", cases, bad);
  VCHECK_EQ(bad, static_cast<size_t>(0));
  cublasDestroy(h);
}

// 2048 x 2048 x 2048, bf16 in and out: A, B and C are 8 MiB each. The first implementation widened all three to float
// (16 MiB each) beside the raw bytes -- 56 MiB at the peak. The raw bytes are still copied; the rest stays a few MiB.
VTEST(cublas_gemmex_mixed_holds_no_widened_copies) {
  const int n = 2048;
  std::mt19937_64 rng(11);
  const auto A = make(CUDA_R_16BF, static_cast<size_t>(n) * n, false, rng), B = make(CUDA_R_16BF, static_cast<size_t>(n) * n, false, rng);
  const auto C0 = make(CUDA_R_16BF, static_cast<size_t>(n) * n, false, rng);
  cublasHandle_t h = nullptr;
  VCHECK_EQ(cublasCreate(&h), CUBLAS_STATUS_SUCCESS);
  DevBuf dA(A), dB(B), dC(C0);
  const float alpha = 1.0f, beta = 0.0f;
  const size_t base = alloc_counter::live();
  alloc_counter::reset_peak();
  const cublasStatus_t st = cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, dA.p, CUDA_R_16BF, n, dB.p, CUDA_R_16BF, n, &beta,
                                         dC.p, CUDA_R_16BF, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  const size_t peak = alloc_counter::peak(), after = alloc_counter::live();
  VCHECK_EQ(st, CUBLAS_STATUS_SUCCESS);
  const double mib = 1048576.0;
  const size_t raw = 3 * static_cast<size_t>(n) * n * 2;
  std::printf("  2048^3 bf16: peak extra heap %.1f MiB (raw operands %.1f MiB), left over %.2f MiB\n", (peak - base) / mib, raw / mib,
              (after - base) / mib);
  VCHECK(peak - base >= 2 * static_cast<size_t>(n) * n * 2);   // the counter sees the library
  VCHECK(peak - base <= raw + (12u << 20));   // (the first implementation: 56 MiB; this one 24 plus panels)
  VCHECK(after - base <= (4u << 20));
  std::vector<uint8_t> got(C0.size());
  cudaMemcpy(got.data(), dC.p, got.size(), cudaMemcpyDeviceToHost);
  const int points[][2] = {{0, 0}, {1, 2047}, {2047, 1}, {1000, 777}, {2047, 2047}};
  for (const auto& ij : points) {
    float acc = 0.0f;
    for (int p = 0; p < n; ++p)
      acc += widen(CUDA_R_16BF, A.data() + (static_cast<size_t>(ij[0]) + static_cast<size_t>(p) * n) * 2) *
             widen(CUDA_R_16BF, B.data() + (static_cast<size_t>(p) + static_cast<size_t>(ij[1]) * n) * 2);
    uint8_t want[2];
    narrow(CUDA_R_16BF, alpha * acc, want);
    VCHECK(std::memcmp(want, got.data() + (static_cast<size_t>(ij[0]) + static_cast<size_t>(ij[1]) * n) * 2, 2) == 0);
  }
  cublasDestroy(h);
}

VTEST_MAIN
