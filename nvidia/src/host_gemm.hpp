#pragma once
// A fast host matrix product for the libraries that compute on the host
// (cuBLAS, cuBLASLt): the same sums as the plain triple loop, in the same
// order, several times faster.
//
// The rule that makes it safe: every output element is one chain
//
//     acc = 0; for p in 0 .. k-1: acc = acc + (a(i,p) * b(p,j))
//
// with the product rounded and then the sum rounded -- no fused multiply-add,
// no reassociation, no split into partial sums. Those loops ran one chain at a
// time and waited on each addition; this runs many chains side by side, one per
// vector lane, which does not touch the arithmetic of any of them. The result
// is bit-for-bit what the loops gave (tests/unit/test_host_gemm.cpp holds the
// loops as the reference and compares every shape's bits).
//
// Nothing here enables FMA: the wide path is compiled for AVX2 only (no FMA
// ISA flag, so the compiler cannot contract), chosen at run time.
//
// Blocks of the output are independent, so they go to as many threads as the
// work warrants and the host has to spare (the same result for any count).
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <type_traits>
#include <vector>

#include "host_parallel.hpp"

namespace vgpu::hostgemm {

enum class Isa { Auto, Generic, Avx2 };

#if defined(__x86_64__) && defined(__GNUC__)
#define VGPU_HOSTGEMM_AVX2 1
#endif

namespace generic {
constexpr int kVB = 16;
#include "host_gemm_impl.inc"
}  // namespace generic

#ifdef VGPU_HOSTGEMM_AVX2
#pragma GCC push_options
#pragma GCC target("avx2")
namespace avx2 {
constexpr int kVB = 32;
#include "host_gemm_impl.inc"
}  // namespace avx2
#pragma GCC pop_options
#endif

inline bool avx2_available() {
#ifdef VGPU_HOSTGEMM_AVX2
  static const bool ok = __builtin_cpu_supports("avx2");
  return ok;
#else
  return false;
#endif
}

// The loop the kernels replace, for products too small to be worth blocking.
template <class T, class S>
inline void naive_sums(bool ta, bool tb, int64_t m, int64_t n, int64_t k, const S* A, int64_t lda, const S* B,
                       int64_t ldb, T* acc /* m x n, ld m */) {
  for (int64_t j = 0; j < n; ++j)
    for (int64_t i = 0; i < m; ++i) {
      T s = T(0);
      for (int64_t p = 0; p < k; ++p) {
        const T a = static_cast<T>(ta ? A[p + i * lda] : A[i + p * lda]);
        const T b = static_cast<T>(tb ? B[j + p * ldb] : B[p + j * ldb]);
        s = s + a * b;
      }
      acc[i + j * m] = s;
    }
}

// Computes every output's chain and hands it to epi(i, j, acc). T is the
// accumulator type: float, double, or uint32_t (wrapping integer sums of
// int8 operands). Epi must be safe to call concurrently on distinct (i, j).
template <class T, class S, class Epi>
void gemm(bool ta, bool tb, int64_t m, int64_t n, int64_t k, const S* A, int64_t lda, const S* B, int64_t ldb,
          Epi&& epi, Isa isa = Isa::Auto, unsigned threads = 0) {
  if (m <= 0 || n <= 0) return;
#ifdef VGPU_HOSTGEMM_AVX2
  const bool wide = isa == Isa::Avx2 || (isa == Isa::Auto && avx2_available());
#else
  const bool wide = false;
#endif
  const uint64_t madds = static_cast<uint64_t>(m) * static_cast<uint64_t>(n) * static_cast<uint64_t>(k);
  if (madds < 4096 || k == 0) {
    std::vector<T> acc(static_cast<size_t>(m) * static_cast<size_t>(n));
    naive_sums<T, S>(ta, tb, m, n, k, A, lda, B, ldb, acc.data());
    for (int64_t j = 0; j < n; ++j)
      for (int64_t i = 0; i < m; ++i) epi(i, j, acc[static_cast<size_t>(i + j * m)]);
    return;
  }

  using G = generic::Ops<T, S>;
#ifdef VGPU_HOSTGEMM_AVX2
  using W = avx2::Ops<T, S>;
#else
  using W = G;
#endif
  const int MR = wide ? W::MR : G::MR, NR = wide ? W::NR : G::NR;

  // Block shape: as large as the caches like, but shrunk until there are
  // enough blocks to keep every thread fed.
  const unsigned want = threads ? threads : vgpu::hostpar::plan(madds, std::numeric_limits<int64_t>::max(), 1u << 24);
  int64_t mcb = wide ? W::MC : G::MC, ncb = wide ? W::NC : G::NC;
  auto nblocks = [&] { return ((m + mcb - 1) / mcb) * ((n + ncb - 1) / ncb); };
  while (want > 1 && nblocks() < static_cast<int64_t>(2 * want)) {
    if (mcb >= ncb && mcb / 2 >= 2 * MR) mcb = (mcb / 2 + MR - 1) / MR * MR;
    else if (ncb / 2 >= 2 * NR) ncb = (ncb / 2 + NR - 1) / NR * NR;
    else if (mcb / 2 >= 2 * MR) mcb = (mcb / 2 + MR - 1) / MR * MR;
    else break;
  }
  const int64_t mb = (m + mcb - 1) / mcb, nb = (n + ncb - 1) / ncb, items = mb * nb;
  const unsigned nthreads = static_cast<unsigned>(std::min<int64_t>(want, items));
  const int64_t mcp_max = (mcb + MR - 1) / MR * MR, ncp_max = (ncb + NR - 1) / NR * NR;
  const int64_t kcmax = std::min<int64_t>(k, wide ? W::KC : G::KC);

  vgpu::hostpar::parallel_for(nthreads, items, [&](int64_t it) {
    // (Buffers are per piece: a piece is a block of 128 x 192 outputs over all of k.)
    thread_local std::vector<char> scratch;
    scratch.resize((static_cast<size_t>(mcp_max * ncp_max) + static_cast<size_t>(mcp_max * kcmax) +
                    static_cast<size_t>(ncp_max * kcmax)) * sizeof(T));
    T* const acc = reinterpret_cast<T*>(scratch.data());
    T* const ap = acc + mcp_max * ncp_max;
    T* const bp = ap + mcp_max * kcmax;
    const int64_t ic = (it % mb) * mcb, jc = (it / mb) * ncb;
    const int64_t mc = std::min(mcb, m - ic), nc = std::min(ncb, n - jc);
    if (wide) {
#ifdef VGPU_HOSTGEMM_AVX2
      W::block(ta, tb, m, n, k, A, lda, B, ldb, ic, mc, jc, nc, acc, ap, bp);
#endif
    } else {
      G::block(ta, tb, m, n, k, A, lda, B, ldb, ic, mc, jc, nc, acc, ap, bp);
    }
    const int64_t mcp = (mc + MR - 1) / MR * MR;
    for (int64_t j = 0; j < nc; ++j)
      for (int64_t i = 0; i < mc; ++i) epi(ic + i, jc + j, acc[static_cast<size_t>(i + j * mcp)]);
  });
}

}  // namespace vgpu::hostgemm
