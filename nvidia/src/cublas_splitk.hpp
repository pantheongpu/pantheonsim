// Which products cuBLAS cuts into slices along K, and how long the slices are, on the cards that were run.
//
// A tensor-core GEMM that fills a card's multiprocessors with a small output tile and a long K is cut over K: every slice is a chain of
// tensor-core chunks from a zero accumulator (gemm_tensor_sum.hpp), and the slices are added one after the other in f32 (found with the
// summation-tree probe tools/cublas_splitk/split_probe.cu: with the big products +2^30 at k = a and -2^30 at k = b the count of ones that
// survive is k minus the size of the sub-sum holding both; every slice boundary shows as a step, and a in the first slice with b in a
// later one lets only the ones of the slices after b's survive, which is the order of the slice sum). The slice length is a multiple of
// 16 (mostly of 32 or 64) and comes from cuBLAS's heuristic: it depends on the shape, the transposes, the types and the multiprocessor
// count of the card, with no formula that was found (an A10G cuts 128x128x1024 into 8 slices of 128 and 256x256x1024 into 3 of 384, a
// T4 256x256x4096 into slices of 448). So the rule here is a table: the products of the grid that was measured (nvidia/tests/data/
// cublas_splitk/README in nvidia/docs/cublas.md), and anything off the grid is the plain chain, which is only a guess.
//
// A profile names the table it takes in numerics.cublas_splitk: `t4`, `a10g` or `none`; the table belongs to a (compute capability,
// multiprocessor count) pair, so a profile of another card of the same architecture does not carry it (see nvidia/docs/profiles.md).
#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

#include "gemm_tensor_sum.hpp"

namespace vgpu::gemmtc {

enum class SplitTable : int { None = 0, T4 = 1, A10g = 2 };

namespace splitk_detail {
struct Row {
  uint8_t type, ta, tb;
  uint16_t m, n;
  uint16_t k, len;
  uint8_t serial;   // two slices only: the epilogue of the serial form (product_epilogue in gemm_tensor_sum.hpp), measured per product
};
inline const std::vector<Row>& rows(SplitTable t) {
  static const std::vector<Row> a10g = {
#include "cublas_splitk_a10g.inc"
  };
  static const std::vector<Row> t4 = {
#include "cublas_splitk_t4.inc"
  };
  static const std::vector<Row> none;
  return t == SplitTable::A10g ? a10g : t == SplitTable::T4 ? t4 : none;
}
inline bool less(const Row& a, const Row& b) {
  return std::tie(a.type, a.ta, a.tb, a.m, a.n, a.k) < std::tie(b.type, b.ta, b.tb, b.m, b.n, b.k);
}
}  // namespace splitk_detail

struct Plan {
  int len = 0;          // the slice length, 0 for none (one chain over all of K)
  bool serial = false;  // two slices with the serial epilogue
};
// The cut of this product. `out16`: the output is half precision.
inline Plan plan_of(SplitTable t, Format f, bool out16, bool ta, bool tb, int m, int n, int k) {
  if (t == SplitTable::None || m <= 0 || n <= 0 || k <= 0 || m > 65535 || n > 65535 || k > 65535) return {};
  const int type = f == Format::F16 ? (out16 ? 1 : 0) : f == Format::Bf16 ? (out16 ? 3 : 2) : 4;
  const auto& r = splitk_detail::rows(t);
  const splitk_detail::Row key{static_cast<uint8_t>(type), static_cast<uint8_t>(ta), static_cast<uint8_t>(tb), static_cast<uint16_t>(m),
                               static_cast<uint16_t>(n), static_cast<uint16_t>(k), 0, 0};
  const auto it = std::lower_bound(r.begin(), r.end(), key, splitk_detail::less);
  if (it == r.end() || splitk_detail::less(key, *it)) return {};
  return {it->len, it->serial != 0};
}
inline int slice_len(SplitTable t, Format f, bool out16, bool ta, bool tb, int m, int n, int k) { return plan_of(t, f, out16, ta, tb, m, n, k).len; }

}  // namespace vgpu::gemmtc
