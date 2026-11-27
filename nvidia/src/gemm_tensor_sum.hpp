// The tensor-core sum of a half-precision, bfloat16 or TF32 product, for the library shims that compute on the host.
//
// A card's tensor-core GEMM adds a chunk of products and the accumulator in one step, chunk after chunk down K (the architecture's
// rule: vgpu/exec/tc_ampere.hpp, docs/matrix-numerics.md). cuBLAS's kernels for these types walk K in order through that
// instruction, so the answer is a chain of chunks, and not the sequential f32 sum the host loops make. Which architectures:
//   * the profile must have a measured or inferred model (Ampere rule: compute capability 7.5, 8.x; the Hopper rule is not used
//     here: cuBLAS on a Hopper card runs wgmma kernels, whose f16, bf16 and tf32 sums nobody has measured);
//   * FP16 needs a tensor core (7.0 and up), bfloat16 and TF32 an Ampere one (8.0 and up): a Turing card does bf16 products in its
//     ordinary FP32 units (a Tesla T4's bf16 GEMM is the sequential sum, measured), and has no TF32 at all.
// The check against the cards that were run is nvidia/tests/e2e/cublas_tc_gemm.cu.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <thread>
#include <tuple>
#include <vector>

#include "vgpu/exec/tc_ampere.hpp"

extern "C" int vgpu_tensor_sum_v1(int device, unsigned* chunk, unsigned* chunk_tf32, int* cc);
// The same, and which cuBLAS split-K table the profile carries (numerics.cublas_splitk): 0 none, 1 t4, 2 a10g.
extern "C" int vgpu_tensor_sum_v2(int device, unsigned* chunk, unsigned* chunk_tf32, int* cc, int* splitk);

namespace vgpu::gemmtc {

enum class Format { F16, Bf16, Tf32 };

struct Model {
  int kind = 0;               // 0 sequential, 1 Ampere rule, 2 Hopper rule
  unsigned chunk = 0, chunk_tf32 = 0;
  int cc = 0;                 // compute capability times ten
  int splitk = 0;             // 0 none, 1 t4, 2 a10g: the cuBLAS split-K table of the profile (cublas_splitk.hpp)
  // Is the product of these operands a tensor-core chain on this device (and so computed here, not by the host's loops)?
  bool takes(Format f) const {
    if (kind != 1) return false;
    return f == Format::F16 ? cc >= 70 : cc >= 80;
  }
};

inline Model model_of(int device) {
  Model m;
  m.kind = vgpu_tensor_sum_v2(device, &m.chunk, &m.chunk_tf32, &m.cc, &m.splitk);
  return m;
}

// P = op(A) x op(B), column-major m x n (leading dimension m), the sum a tensor core makes with a zero accumulator. A and B hold
// values exactly representable in the format (inputs already rounded to it), column-major with leading dimensions lda and ldb.
// `slice` > 0 (and < k): cuBLAS cut the product over K into slices of that many products (cublas_splitk.hpp): each is a chain from a
// zero accumulator and the slice results are added one after the other in f32.
inline void product(const Model& md, Format f, bool ta, bool tb, int m, int n, int k, const float* A, int lda, const float* B, int ldb,
                    std::vector<float>& P, int slice = 0) {
  using namespace vgpu::exec::tc86;
  if (slice > 0 && slice < k) {
    std::vector<float> part;
    for (int k0 = 0; k0 < k; k0 += slice) {
      const int len = std::min(slice, k - k0);
      const float* As = ta ? A + k0 : A + static_cast<size_t>(k0) * static_cast<size_t>(lda);
      const float* Bs = tb ? B + static_cast<size_t>(k0) * static_cast<size_t>(ldb) : B + k0;
      product(md, f, ta, tb, m, n, len, As, lda, Bs, ldb, part, 0);
      if (k0 == 0) { P = part; continue; }
      for (size_t i = 0; i < P.size(); ++i) P[i] = P[i] + part[i];
    }
    return;
  }
  const unsigned chunk = f == Format::Tf32 ? md.chunk_tf32 : md.chunk;
  const int emin = f == Format::F16 ? kEminF16 : kEminWide;
  const size_t M = static_cast<size_t>(m), N = static_cast<size_t>(n), K = static_cast<size_t>(k);
  P.assign(M * N, 0.0f);
  if (!M || !N || !K) return;
  // Dense row-major operands: A is M x K, B is K x N, the layout mma() takes.
  std::vector<float> Ad(M * K), Bd(K * N);
  for (size_t i = 0; i < M; ++i)
    for (size_t p = 0; p < K; ++p)
      Ad[i * K + p] = ta ? A[p + i * static_cast<size_t>(lda)] : A[i + p * static_cast<size_t>(lda)];
  for (size_t p = 0; p < K; ++p)
    for (size_t j = 0; j < N; ++j)
      Bd[p * N + j] = tb ? B[j + p * static_cast<size_t>(ldb)] : B[p + j * static_cast<size_t>(ldb)];
  // Tiles of 16 rows and 8 columns (an m16n8k16 mma.sync, the shape the vector kernels are written for), K in blocks of 16, which hold whole chunks (16 is a
  // multiple of 8, 4 and 16), the accumulator carried from block to block. Row bands go to threads.
  constexpr size_t TM = 16, TN = 8, TK = 16;
  const size_t bands = (M + TM - 1) / TM;
  auto band = [&](size_t b) {
    const size_t i0 = b * TM, mi = std::min(TM, M - i0);
    float acc[TM * TN], out[TM * TN] = {};
    for (size_t j0 = 0; j0 < N; j0 += TN) {
      const size_t nj = std::min(TN, N - j0);
      std::fill(acc, acc + TM * TN, 0.0f);
      for (size_t k0 = 0; k0 < K; k0 += TK) {
        const size_t kk = std::min(TK, K - k0);
        mma(static_cast<unsigned>(mi), static_cast<unsigned>(nj), static_cast<unsigned>(kk), chunk, emin, false, &Ad[i0 * K + k0], K,
            &Bd[k0 * N + j0], N, acc, TN, out, TN);
        std::copy(out, out + TM * TN, acc);
      }
      for (size_t i = 0; i < mi; ++i)
        for (size_t j = 0; j < nj; ++j) P[(i0 + i) + (j0 + j) * M] = acc[i * TN + j];
    }
  };
  const size_t macs = M * N * K;
  size_t threads = macs < (size_t{1} << 22) ? 1 : std::min<size_t>(std::max(1u, std::thread::hardware_concurrency()), 8);
  threads = std::min(threads, bands);
  if (threads <= 1) {
    for (size_t b = 0; b < bands; ++b) band(b);
    return;
  }
  std::vector<std::thread> pool;
  for (size_t t = 0; t < threads; ++t)
    pool.emplace_back([&, t] {
      for (size_t b = t; b < bands; b += threads) band(b);
    });
  for (auto& th : pool) th.join();
}

// A product kept apart from the compiler's contraction: the card's epilogue fuses some multiplies and adds and not others (below), so
// the unfused ones must not be fused by the host compiler either.
inline float rmul(float a, float b) {
  volatile float t = a * b;
  return t;
}

// R = alpha * (op(A) x op(B)) + beta * C as the card's kernel makes it, for a product of this model (`slice`: product()'s argument), in f32 before
// the output format's rounding. `Cold` is C as floats, column-major with leading dimension m, read only when beta != 0. What the card does
// (found by running the cases of nvidia/tests/e2e/cublas_tc_gemm.cu with other alpha and beta, which separate the fused from the unfused forms;
// powers of two for alpha and beta do not):
//   * one chain (no split):   fma(alpha, acc, beta * c)
//   * two slices, `serial2` (a row of the table that was measured to do so; the other two-slice rows are the sum form of the next line):    FP16 and BF16:  r = fma(beta, c, alpha * P0) (alpha * P0 for beta = 0), then r + alpha * P1
//                             TF32:           r = fma(alpha, P0, beta * c) (alpha * P0 for beta = 0), then fma(alpha, P1, r)
//   * the sum form (three slices or more, and two where serial2 is false):   fma(beta, c, alpha * (P0 + P1 + ...)), the slices summed in order first (alpha * sum for beta = 0)
inline void product_epilogue(const Model& md, Format f, bool ta, bool tb, int m, int n, int k, const float* A, int lda, const float* B, int ldb,
                             int slice, bool serial2, float alpha, float beta, const float* Cold, std::vector<float>& R) {
  const size_t cnt = static_cast<size_t>(m) * static_cast<size_t>(n);
  const bool split = slice > 0 && slice < k;
  if (split && serial2 && (k + slice - 1) / slice == 2) {
    std::vector<float> p0, p1;
    product(md, f, ta, tb, m, n, slice, A, lda, B, ldb, p0, 0);
    const float* As = ta ? A + slice : A + static_cast<size_t>(slice) * static_cast<size_t>(lda);
    const float* Bs = tb ? B + static_cast<size_t>(slice) * static_cast<size_t>(ldb) : B + slice;
    product(md, f, ta, tb, m, n, k - slice, As, lda, Bs, ldb, p1, 0);
    R.resize(cnt);
    for (size_t i = 0; i < cnt; ++i) {
      float r;
      if (f == Format::Tf32) {
        r = beta == 0.0f ? rmul(alpha, p0[i]) : std::fma(alpha, p0[i], rmul(beta, Cold[i]));
        r = std::fma(alpha, p1[i], r);
      } else {
        r = beta == 0.0f ? rmul(alpha, p0[i]) : std::fma(beta, Cold[i], rmul(alpha, p0[i]));
        r = r + rmul(alpha, p1[i]);
      }
      R[i] = r;
    }
    return;
  }
  product(md, f, ta, tb, m, n, k, A, lda, B, ldb, R, slice);
  for (size_t i = 0; i < cnt; ++i) {
    const float acc = R[i];
    if (beta == 0.0f) R[i] = rmul(alpha, acc);
    else if (split) R[i] = std::fma(beta, Cold[i], rmul(alpha, acc));
    else R[i] = std::fma(alpha, acc, rmul(beta, Cold[i]));
  }
}

}  // namespace vgpu::gemmtc
