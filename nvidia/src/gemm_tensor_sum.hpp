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
#include <cstdint>
#include <thread>
#include <vector>

#include "vgpu/exec/tc_ampere.hpp"

extern "C" int vgpu_tensor_sum_v1(int device, unsigned* chunk, unsigned* chunk_tf32, int* cc);

namespace vgpu::gemmtc {

enum class Format { F16, Bf16, Tf32 };

struct Model {
  int kind = 0;               // 0 sequential, 1 Ampere rule, 2 Hopper rule
  unsigned chunk = 0, chunk_tf32 = 0;
  int cc = 0;                 // compute capability times ten
  // Is the product of these operands a tensor-core chain on this device (and so computed here, not by the host's loops)?
  bool takes(Format f) const {
    if (kind != 1) return false;
    return f == Format::F16 ? cc >= 70 : cc >= 80;
  }
};

inline Model model_of(int device) {
  Model m;
  m.kind = vgpu_tensor_sum_v1(device, &m.chunk, &m.chunk_tf32, &m.cc);
  return m;
}

// P = op(A) x op(B), column-major m x n (leading dimension m), the sum a tensor core makes with a zero accumulator. A and B hold
// values exactly representable in the format (inputs already rounded to it), column-major with leading dimensions lda and ldb.
inline void product(const Model& md, Format f, bool ta, bool tb, int m, int n, int k, const float* A, int lda, const float* B, int ldb,
                    std::vector<float>& P) {
  using namespace vgpu::exec::tc86;
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

}  // namespace vgpu::gemmtc
