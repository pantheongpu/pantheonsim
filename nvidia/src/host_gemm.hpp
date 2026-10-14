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
//
// Operands are read through a loader, ld(index, i, p) -> value, which turns the stored element (bf16, fp16, an
// int8 with a scale, ...) into the accumulator type T when a panel of it is packed: no converted copy of a whole
// matrix is ever made, and the scratch is a few panels per thread whatever the sizes. The pointer form of gemm
// below is the loader that just casts.
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <type_traits>
#include <vector>

#include "host_parallel.hpp"

#if defined(__x86_64__) && defined(__GNUC__)
#include <immintrin.h>
#endif

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
#define VGPU_HG_INTRIN 1
#include "host_gemm_impl.inc"
#undef VGPU_HG_INTRIN
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

// One output's chain, the loop this replaces, for products too small to be worth blocking.
//
// A note on NaNs: when two NaNs meet in an addition or a product (a NaN operand and the default NaN of an inf*0 or inf-inf,
// say), the hardware returns one of them -- the first operand's -- and which operand the compiler makes first is its own
// choice, different between two copies of the same source line in one program (tests/unit/test_host_gemm.cpp shows it:
// the reference loop and this one disagree on the sign of such a NaN depending on how the operand is indexed). So the sign
// and payload of a NaN that is the result of two NaNs meeting were never fixed, here or in the loops this replaces; every
// other bit of every result is, and a result is NaN exactly when it was.
template <class T, class LA, class LB>
inline T chain_one(bool ta, bool tb, int64_t i, int64_t j, int64_t k, const LA& A, int64_t lda, const LB& B, int64_t ldb) {
  T acc = T(0);
  for (int64_t p = 0; p < k; ++p) {
    T a = static_cast<T>(ta ? A(p + i * lda, i, p) : A(i + p * lda, i, p));
    T b = static_cast<T>(tb ? B(j + p * ldb, j, p) : B(p + j * ldb, j, p));
    acc += a * b;
  }
  return acc;
}

// The loader of a plain array: the element, cast to T.
template <class T, class S>
struct PtrLoader {
  const S* p;
  T operator()(int64_t idx, int64_t, int64_t) const { return static_cast<T>(p[idx]); }
};

// Computes every output's chain and hands it to epi(i, j, acc). T is the
// accumulator type: float, double, or uint32_t (wrapping integer sums of
// int8 operands). A and B are loaders (see the top of this file): A(idx, i, p)
// is op(A)(i, p) stored at idx, B(idx, j, p) is op(B)(p, j). Epi must be safe to
// call concurrently on distinct (i, j).
template <class T, class LA, class LB, class Epi>
void gemm_with(bool ta, bool tb, int64_t m, int64_t n, int64_t k, const LA& A, int64_t lda, const LB& B, int64_t ldb,
               Epi&& epi, Isa isa = Isa::Auto, unsigned threads = 0) {
  if (m <= 0 || n <= 0) return;
#ifdef VGPU_HOSTGEMM_AVX2
  const bool wide = isa == Isa::Avx2 || (isa == Isa::Auto && avx2_available());
#else
  const bool wide = false;
#endif
  const uint64_t madds = static_cast<uint64_t>(m) * static_cast<uint64_t>(n) * static_cast<uint64_t>(k);
  if (madds < 4096 || k == 0) {   // too small to be worth blocking: the chains, one at a time
    for (int64_t j = 0; j < n; ++j)
      for (int64_t i = 0; i < m; ++i) epi(i, j, chain_one<T>(ta, tb, i, j, k, A, lda, B, ldb));
    return;
  }

  using G = generic::Ops<T>;
#ifdef VGPU_HOSTGEMM_AVX2
  using W = avx2::Ops<T>;
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

// The same over plain arrays of S.
template <class T, class S, class Epi>
void gemm(bool ta, bool tb, int64_t m, int64_t n, int64_t k, const S* A, int64_t lda, const S* B, int64_t ldb, Epi&& epi,
          Isa isa = Isa::Auto, unsigned threads = 0) {
  gemm_with<T>(ta, tb, m, n, k, PtrLoader<T, S>{A}, lda, PtrLoader<T, S>{B}, ldb, static_cast<Epi&&>(epi), isa, threads);
}

}  // namespace vgpu::hostgemm
