// The sum of the wgmma, tcgen05.mma and m8n8k4 forms on f16, bf16 and tf32 operands (vgpu::FormSum, include/vgpu/profile.hpp; issue #664).
//
// The arithmetic is the one of tc_ampere.hpp -- chunks of products and the accumulator, each term cut toward zero at 2^(top - grid), the exact
// sum cut toward zero to an f32 or rounded to nearest even to an f16 -- with the numbers of the model a profile names:
//   Hopper  chunks of 16 products (tf32: 8), grid 26: Xie et al. (arXiv 2511.10909) give wgmma on Hopper and tcgen05.mma on Blackwell the numbers
//           of mma.sync there (F = 25, a chunk of 32 bytes of operands), and an accumulator that joins the chunk.
//   Volta   chunks of 4 products, grid 24: HMMA.884, which a V100 runs (5,000 of 5,000 vectors recorded by Khattak and Mikaitis on a V100 are
//           reproduced, tools/tcpapers; Xie et al. give a Turing m8n8k4 the same numbers).
// Dense forms only. docs/matrix-numerics.md says what each rests on; none is "measured" unless the profile's `provenance:` says so.
#pragma once
#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>

#include "vgpu/exec/tc_ampere.hpp"
#include "vgpu/profile.hpp"

namespace vgpu::exec::tc86 {

// The rule in force for a form: null for the sequential sum (and with VGPU_TC_EXACT=0).
inline const SumRule* form_rule(vgpu::FormSum m) { return exact_enabled() ? vgpu::form_sum_rule(m) : nullptr; }
inline const SumRule* wgmma_rule(const vgpu::DeviceProfile& p) { return form_rule(p.numerics.tensor_sum_wgmma); }
inline const SumRule* tcgen05_rule(const vgpu::DeviceProfile& p) { return form_rule(p.numerics.tensor_sum_tcgen05); }
inline const SumRule* m8n8k4_rule(const vgpu::DeviceProfile& p) { return form_rule(p.numerics.tensor_sum_m8n8k4); }

// The widest K one call takes (an instruction's: 16 f16 or bf16 products, 8 tf32 ones, 4 for m8n8k4).
inline constexpr unsigned kFormMaxK = 16;

// D = C + A x B for a rows x N tile (A rows x K with row stride K, B K x N, C and D rows x N, all row-major floats holding f16, bf16 or tf32
// values, and the accumulator's: an f16 accumulator's values already widened). `tf32`: the operands are tf32. K must be at most kFormMaxK;
// rows and columns are taken in blocks of 32. A NaN is the canonical 0x7fffffff. C may be D (in place).
inline void form_tile(const SumRule& r, bool tf32, int emin, bool acc_f16, uint32_t rows, uint32_t N, uint32_t K, const float* A, const float* B,
                      size_t ldb, const float* C, size_t ldc, float* D, size_t ldd) {
  const unsigned chunk = std::min(tf32 ? r.chunk_tf32 : r.chunk, kFormMaxK);
  for (uint32_t i0 = 0; i0 < rows; i0 += 32) {
    const uint32_t m = std::min<uint32_t>(32, rows - i0);
    for (uint32_t j0 = 0; j0 < N; j0 += 32) {
      const uint32_t n = std::min<uint32_t>(32, N - j0);
      mma_wide(m, n, K, chunk, r.grid, emin, acc_f16, A + size_t{i0} * K, K, B + j0, ldb, C + size_t{i0} * ldc + j0, ldc, D + size_t{i0} * ldd + j0, ldd);
    }
  }
  for (uint32_t i = 0; i < rows; ++i)
    for (uint32_t j = 0; j < N; ++j) {
      float& d = D[size_t{i} * ldd + j];
      if (d != d) d = std::bit_cast<float>(0x7fffffffu);
    }
}
inline void form_tile(const SumRule& r, bool tf32, int emin, bool acc_f16, uint32_t rows, uint32_t N, uint32_t K, const float* A, const float* B,
                      const float* C, float* D) {
  form_tile(r, tf32, emin, acc_f16, rows, N, K, A, B, N, C, N, D, N);
}

// One element: acc + sum of a[k * astride] * b[k * bstride], k < K (the general, element-at-a-time engines' form of form_tile).
inline float form_dot(const SumRule& r, bool tf32, int emin, bool acc_f16, const double* a, size_t astride, const double* b, size_t bstride, uint32_t K,
                      float acc) {
  float fa[kFormMaxK], fb[kFormMaxK], d = 0;
  for (uint32_t k = 0; k < K; ++k) {
    fa[k] = static_cast<float>(a[k * astride]);
    fb[k] = static_cast<float>(b[k * bstride]);
  }
  mma_wide(1, 1, K, std::min(tf32 ? r.chunk_tf32 : r.chunk, kFormMaxK), r.grid, emin, acc_f16, fa, K, fb, 1, &acc, 1, &d, 1);
  return d != d ? std::bit_cast<float>(0x7fffffffu) : d;
}

}  // namespace vgpu::exec::tc86
