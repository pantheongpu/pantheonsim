// cublasLtMatmul's host product against the plain loops, bit for bit, and the heap it holds while it runs.
//
// The library reads A and B as the bytes the device holds and converts each element as host_gemm.hpp packs a panel
// (nothing the size of a matrix is widened to double), and takes the sums a strip of columns at a time (nothing the
// size of the output is held beyond D's own bytes). What it must still give is what the first implementation gave:
// every output is one chain  acc = acc + (((a*sa) * b) * sb)  in double, over k in order, then the epilogue's fp32
// roundings, then the store. The reference here is that chain written out.
//
// Cases: bf16, fp16, fp32 and FP8 (E4M3) operands, a mixed pair, a scalar scale on A (the converting loader that
// applies a scale) and on B (a scale on B keeps the plain loop, which these cases check too), alpha and beta
// (C = D), either operand transposed, column- and row-major layouts, leading dimensions longer than the matrices,
// K = 1 and odd sizes, and an M so long that the sums come in several strips.
#include <cublasLt.h>
#include <cuda_runtime_api.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "alloc_counter.hpp"
#include "vtest.hpp"

namespace {

enum class Ty { Bf16, F16, F32, E4M3 };

cudaDataType cuda_type(Ty t) {
  switch (t) {
    case Ty::Bf16: return CUDA_R_16BF;
    case Ty::F16: return CUDA_R_16F;
    case Ty::F32: return CUDA_R_32F;
    default: return CUDA_R_8F_E4M3;
  }
}
size_t bytes_of(Ty t) { return t == Ty::F32 ? 4 : t == Ty::E4M3 ? 1 : 2; }

double bf16_val(uint16_t b) {
  const uint32_t x = static_cast<uint32_t>(b) << 16;
  float f;
  std::memcpy(&f, &x, 4);
  return f;
}
double half_val(uint16_t h) {
  const int e = (h >> 10) & 0x1f, m = h & 0x3ff;
  const double v = e == 0 ? std::ldexp(static_cast<double>(m), -24)
                   : e == 31 ? (m ? NAN : INFINITY)
                             : std::ldexp(static_cast<double>(m | 0x400), e - 25);
  return (h & 0x8000) ? -v : v;
}
double e4m3_val(uint8_t b) {   // bias 7, no infinity, NaN only at S.1111.111 (never made here)
  const int e = (b >> 3) & 0xf, m = b & 7;
  const double v = e == 0 ? std::ldexp(static_cast<double>(m), -9) : std::ldexp(static_cast<double>(8 + m), e - 10);
  return (b & 0x80) ? -v : v;
}
uint16_t bf16_bits(float f) {   // round to nearest even, NaN quiet
  uint32_t x;
  std::memcpy(&x, &f, 4);
  if ((x & 0x7fffffff) > 0x7f800000) return static_cast<uint16_t>((x >> 16) | 0x40);
  x += 0x7fff + ((x >> 16) & 1);
  return static_cast<uint16_t>(x >> 16);
}

double value_at(Ty t, const std::vector<uint8_t>& raw, size_t i) {
  const uint8_t* p = raw.data() + i * bytes_of(t);
  switch (t) {
    case Ty::Bf16: { uint16_t u; std::memcpy(&u, p, 2); return bf16_val(u); }
    case Ty::F16: { uint16_t u; std::memcpy(&u, p, 2); return half_val(u); }
    case Ty::F32: { float f; std::memcpy(&f, p, 4); return f; }
    default: return e4m3_val(*p);
  }
}

// mix: 0 ordinary values, 1 with zeros, infinities and wild bit patterns (so NaNs arise in the sums).
std::vector<uint8_t> make(Ty t, size_t n, int mix, std::mt19937_64& rng) {
  std::vector<uint8_t> raw(n * bytes_of(t));
  for (size_t i = 0; i < n; ++i) {
    uint8_t* p = raw.data() + i * bytes_of(t);
    const bool wild = mix == 1 && rng() % 25 == 0;
    switch (t) {
      case Ty::Bf16: {
        uint16_t u = bf16_bits(static_cast<float>((static_cast<int>(rng() % 511) - 255) / 64.0));
        if (wild) u = static_cast<uint16_t>(rng());
        std::memcpy(p, &u, 2);
        break;
      }
      case Ty::F16: {
        uint16_t u = static_cast<uint16_t>(((rng() % 2) << 15) | ((10 + rng() % 11) << 10) | (rng() % 1024));
        if (wild) u = static_cast<uint16_t>(rng());
        std::memcpy(p, &u, 2);
        break;
      }
      case Ty::F32: {
        float f = static_cast<float>((static_cast<int>(rng() % 2001) - 1000) / 256.0);
        if (wild) { const uint32_t b = static_cast<uint32_t>(rng()); std::memcpy(&f, &b, 4); }
        std::memcpy(p, &f, 4);
        break;
      }
      default: {
        uint8_t u = static_cast<uint8_t>((rng() % 2) << 7 | (rng() % 0x78));   // finite codes
        if (wild) u = static_cast<uint8_t>(u & 0x80);                           // a zero
        *p = u;
        break;
      }
    }
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

struct Case {
  Ty a = Ty::Bf16, b = Ty::Bf16, d = Ty::Bf16;
  int64_t m = 16, n = 16, k = 16;
  bool ta = false, tb = false, row_a = false, row_b = false, row_d = false;
  int pad = 0;
  float alpha = 1.0f, beta = 0.0f;
  float a_scale = 0.0f, b_scale = 0.0f;   // 0: no scale pointer
  int mix = 0;
};

// Stored element (r, c) of a matrix with leading dimension ld.
size_t at(bool row, int64_t ld, int64_t r, int64_t c) { return static_cast<size_t>(row ? r * ld + c : c * ld + r); }
size_t span(bool row, int64_t ld, int64_t rows, int64_t cols) {
  return static_cast<size_t>(row ? (rows - 1) * ld + cols : (cols - 1) * ld + rows);
}

bool both_nan_or_same(const uint8_t* x, const uint8_t* y, Ty t) {
  if (std::memcmp(x, y, bytes_of(t)) == 0) return true;
  if (t == Ty::Bf16) {
    uint16_t a, b;
    std::memcpy(&a, x, 2); std::memcpy(&b, y, 2);
    return (a & 0x7fff) > 0x7f80 && (b & 0x7fff) > 0x7f80;
  }
  if (t == Ty::F32) {
    float a, b;
    std::memcpy(&a, x, 4); std::memcpy(&b, y, 4);
    return std::isnan(a) && std::isnan(b);
  }
  return false;
}

// Runs one matmul through the library and returns how many bytes of D differ from the reference.
size_t run_case(const Case& c, std::mt19937_64& rng) {
  const int64_t ar = c.ta ? c.k : c.m, ac = c.ta ? c.m : c.k, br = c.tb ? c.n : c.k, bc = c.tb ? c.k : c.n;
  const int64_t lda = (c.row_a ? ac : ar) + c.pad, ldb = (c.row_b ? bc : br) + c.pad, ldd = (c.row_d ? c.n : c.m) + c.pad;
  const auto A = make(c.a, span(c.row_a, lda, ar, ac), c.mix, rng), B = make(c.b, span(c.row_b, ldb, br, bc), c.mix, rng);
  auto D = make(c.d, span(c.row_d, ldd, c.m, c.n), 0, rng);   // C = D, finite
  const auto D0 = D;

  cublasLtHandle_t h = nullptr;
  VCHECK_EQ(cublasLtCreate(&h), CUBLAS_STATUS_SUCCESS);
  cublasLtMatmulDesc_t desc = nullptr;
  VCHECK_EQ(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F), CUBLAS_STATUS_SUCCESS);
  const cublasOperation_t opa = c.ta ? CUBLAS_OP_T : CUBLAS_OP_N, opb = c.tb ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &opa, sizeof opa);
  cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &opb, sizeof opb);
  DevBuf dA(A), dB(B), dD(D);
  float sa = c.a_scale, sb = c.b_scale;
  DevBuf dsa(std::vector<uint8_t>(reinterpret_cast<uint8_t*>(&sa), reinterpret_cast<uint8_t*>(&sa) + 4)),
      dsb(std::vector<uint8_t>(reinterpret_cast<uint8_t*>(&sb), reinterpret_cast<uint8_t*>(&sb) + 4));
  const void* psa = dsa.p;
  const void* psb = dsb.p;
  if (c.a_scale != 0.0f) cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &psa, sizeof psa);
  if (c.b_scale != 0.0f) cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &psb, sizeof psb);
  cublasLtMatrixLayout_t la = nullptr, lb = nullptr, ld = nullptr;
  cublasLtMatrixLayoutCreate(&la, cuda_type(c.a), static_cast<uint64_t>(ar), static_cast<uint64_t>(ac), lda);
  cublasLtMatrixLayoutCreate(&lb, cuda_type(c.b), static_cast<uint64_t>(br), static_cast<uint64_t>(bc), ldb);
  cublasLtMatrixLayoutCreate(&ld, cuda_type(c.d), static_cast<uint64_t>(c.m), static_cast<uint64_t>(c.n), ldd);
  const int row = CUBLASLT_ORDER_ROW;
  if (c.row_a) cublasLtMatrixLayoutSetAttribute(la, CUBLASLT_MATRIX_LAYOUT_ORDER, &row, sizeof row);
  if (c.row_b) cublasLtMatrixLayoutSetAttribute(lb, CUBLASLT_MATRIX_LAYOUT_ORDER, &row, sizeof row);
  if (c.row_d) cublasLtMatrixLayoutSetAttribute(ld, CUBLASLT_MATRIX_LAYOUT_ORDER, &row, sizeof row);
  const cublasStatus_t st = cublasLtMatmul(h, desc, &c.alpha, dA.p, la, dB.p, lb, &c.beta, dD.p, ld, dD.p, ld, nullptr, nullptr, 0, nullptr);
  VCHECK_EQ(st, CUBLAS_STATUS_SUCCESS);
  std::vector<uint8_t> got(D.size());
  if (!got.empty()) cudaMemcpy(got.data(), dD.p, got.size(), cudaMemcpyDeviceToHost);
  cublasLtMatrixLayoutDestroy(la);
  cublasLtMatrixLayoutDestroy(lb);
  cublasLtMatrixLayoutDestroy(ld);
  cublasLtMatmulDescDestroy(desc);
  cublasLtDestroy(h);

  // The reference: the chain, then the epilogue as the library rounds it, stored to D's type. Cells D does not
  // cover (a leading dimension's padding) keep the bytes they had.
  const double s_a = c.a_scale != 0.0f ? sa : 1.0, s_b = c.b_scale != 0.0f ? sb : 1.0;
  std::vector<uint8_t> want = D0;
  for (int64_t j = 0; j < c.n; ++j)
    for (int64_t i = 0; i < c.m; ++i) {
      double acc = 0.0;
      for (int64_t p = 0; p < c.k; ++p) {
        const double a = value_at(c.a, A, c.ta ? at(c.row_a, lda, p, i) : at(c.row_a, lda, i, p));
        const double b = value_at(c.b, B, c.tb ? at(c.row_b, ldb, j, p) : at(c.row_b, ldb, p, j));
        acc += a * s_a * b * s_b;
      }
      const size_t cell = at(c.row_d, ldd, i, j);
      double v = static_cast<double>(static_cast<float>(static_cast<double>(c.alpha) * acc));
      if (c.beta != 0.0f) {
        const double old = value_at(c.d, D0, cell);
        v = static_cast<double>(static_cast<float>(v + static_cast<double>(static_cast<float>(static_cast<double>(c.beta) * 1.0 * old))));
      }
      const float out = static_cast<float>(v * 1.0);
      uint8_t* dst = want.data() + cell * bytes_of(c.d);
      if (c.d == Ty::F32) std::memcpy(dst, &out, 4);
      else { const uint16_t u = bf16_bits(out); std::memcpy(dst, &u, 2); }
    }
  size_t bad = 0;
  for (size_t e = 0; e * bytes_of(c.d) < want.size(); ++e)
    bad += !both_nan_or_same(want.data() + e * bytes_of(c.d), got.data() + e * bytes_of(c.d), c.d);
  return bad;
}

}  // namespace

VTEST(cublaslt_matmul_matches_the_loops_bit_for_bit) {
  std::mt19937_64 rng(2468);
  struct Shape { int64_t m, n, k; };
  const Shape shapes[] = {{1, 1, 1}, {3, 5, 7}, {17, 33, 65}, {129, 130, 257}, {64, 300, 1}, {70, 70, 2}, {8200, 300, 5}};
  struct Types { Ty a, b, d; float a_scale, b_scale; const char* name; };
  const Types types[] = {
      {Ty::Bf16, Ty::Bf16, Ty::Bf16, 0, 0, "bf16"},
      {Ty::F16, Ty::F16, Ty::F32, 0, 0, "fp16 -> fp32"},
      {Ty::F32, Ty::F32, Ty::F32, 0, 0, "fp32"},
      {Ty::Bf16, Ty::F16, Ty::F32, 0, 0, "bf16 x fp16"},
      {Ty::E4M3, Ty::E4M3, Ty::Bf16, 0, 0, "fp8"},
      {Ty::E4M3, Ty::E4M3, Ty::Bf16, 0.37f, 0, "fp8, scale on A"},
      {Ty::E4M3, Ty::E4M3, Ty::Bf16, 0, 0.37f, "fp8, scale on B"},
      {Ty::E4M3, Ty::E4M3, Ty::F32, 0.37f, 1.0f, "fp8, scales on both"},
  };
  int cases = 0;
  size_t bad = 0;
  for (const Types& t : types)
    for (const Shape& s : shapes) {
      // The option combinations, rotated rather than multiplied out.
      for (int v = 0; v < 7; ++v) {
        Case c;
        c.a = t.a; c.b = t.b; c.d = t.d; c.a_scale = t.a_scale; c.b_scale = t.b_scale;
        c.m = s.m; c.n = s.n; c.k = s.k;
        c.ta = v & 1; c.tb = (v >> 1) & 1;
        c.row_a = v == 4 || v == 6; c.row_b = v == 5 || v == 6; c.row_d = v == 3 || v == 6;
        c.pad = v % 3 == 1 ? 3 : 0;
        c.alpha = v % 3 == 0 ? 1.0f : v % 3 == 1 ? -0.5f : 2.25f;
        c.beta = v % 2 == 0 ? 0.0f : v % 4 == 1 ? 1.0f : -1.25f;
        c.mix = v % 3 == 2 ? 1 : 0;
        if (s.m * s.n > 300000 && v > 3) continue;   // the long case: a few variants are enough
        const size_t b = run_case(c, rng);
        if (b) std::printf("  differs: %s m=%ld n=%ld k=%ld variant %d (%zu bytes)\n", t.name, (long)s.m, (long)s.n, (long)s.k, v, b);
        bad += b;
        ++cases;
      }
    }
  std::printf("  %d matmuls, %zu elements differ\n", cases, bad);
  VCHECK_EQ(bad, static_cast<size_t>(0));
}

// What the call holds while it runs. 2048 x 2048 x 2048 in bf16: A, B and D are 8 MiB each. The first implementation
// widened A and B to double (32 MiB each) and held 32 MiB of sums on top of the 24 MiB of raw bytes -- 120 MiB. The
// raw bytes of A, B and D are still copied (the device's matrices are read whole); everything else must stay a few
// MiB, whatever the size: a strip of sums (8 MiB at most) and a few panels per thread.
VTEST(cublaslt_matmul_holds_no_matrix_sized_temporaries) {
  const int64_t n = 2048;
  std::mt19937_64 rng(7);
  const auto A = make(Ty::Bf16, static_cast<size_t>(n * n), 0, rng), B = make(Ty::Bf16, static_cast<size_t>(n * n), 0, rng);
  const auto D0 = make(Ty::Bf16, static_cast<size_t>(n * n), 0, rng);   // random, so its device chunks exist before the call
  cublasLtHandle_t h = nullptr;
  VCHECK_EQ(cublasLtCreate(&h), CUBLAS_STATUS_SUCCESS);
  cublasLtMatmulDesc_t desc = nullptr;
  VCHECK_EQ(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F), CUBLAS_STATUS_SUCCESS);
  cublasLtMatrixLayout_t la = nullptr, lb = nullptr, ld = nullptr;
  cublasLtMatrixLayoutCreate(&la, CUDA_R_16BF, n, n, n);
  cublasLtMatrixLayoutCreate(&lb, CUDA_R_16BF, n, n, n);
  cublasLtMatrixLayoutCreate(&ld, CUDA_R_16BF, n, n, n);
  DevBuf dA(A), dB(B), dD(D0);
  const float alpha = 1.0f, beta = 0.0f;
  const size_t base = alloc_counter::live();
  alloc_counter::reset_peak();
  const cublasStatus_t st = cublasLtMatmul(h, desc, &alpha, dA.p, la, dB.p, lb, &beta, dD.p, ld, dD.p, ld, nullptr, nullptr, 0, nullptr);
  const size_t peak = alloc_counter::peak(), after = alloc_counter::live();
  VCHECK_EQ(st, CUBLAS_STATUS_SUCCESS);
  const double mib = 1048576.0;
  const size_t raw = 3 * static_cast<size_t>(n * n) * 2;
  std::printf("  2048^3 bf16: peak extra heap %.1f MiB (raw operands %.1f MiB), left over %.2f MiB\n", (peak - base) / mib, raw / mib,
              (after - base) / mib);
  // The counter sees the library's allocations (it must hold at least A's and B's raw bytes), and nothing the size of a
  // widened matrix: at most the raw operands, 8 MiB of sums, a few MiB of panels, and room for the copy a runtime
  // may stage a transfer through.
  VCHECK(peak - base >= 2 * static_cast<size_t>(n * n) * 2);
  VCHECK(peak - base <= raw + (24u << 20));   // (the first implementation: raw + 96 MiB)
  VCHECK(after - base <= (4u << 20));
  // Spot checks of the product itself.
  std::vector<uint8_t> got(D0.size());
  cudaMemcpy(got.data(), dD.p, got.size(), cudaMemcpyDeviceToHost);
  const int64_t points[][2] = {{0, 0}, {1, 2047}, {2047, 1}, {1000, 777}, {2047, 2047}};
  for (const auto& ij : points) {
    double acc = 0.0;
    for (int64_t p = 0; p < n; ++p) acc += value_at(Ty::Bf16, A, at(false, n, ij[0], p)) * value_at(Ty::Bf16, B, at(false, n, p, ij[1]));
    const uint16_t want = bf16_bits(static_cast<float>(1.0 * acc));
    uint16_t have;
    std::memcpy(&have, got.data() + at(false, n, ij[0], ij[1]) * 2, 2);
    VCHECK_EQ(have, want);
  }
  cublasLtMatrixLayoutDestroy(la);
  cublasLtMatrixLayoutDestroy(lb);
  cublasLtMatrixLayoutDestroy(ld);
  cublasLtMatmulDescDestroy(desc);
  cublasLtDestroy(h);
}

VTEST_MAIN
