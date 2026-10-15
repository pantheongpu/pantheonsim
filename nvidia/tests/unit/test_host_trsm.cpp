// The triangular solves behind cuBLAS trsm (src/host_trsm.hpp, complex_linalg.hpp), bit for bit against the loops they
// replaced, which are kept here as the reference: for each right-hand side,
//     s = x[i];  for j in the span of row i of M, ascending:  s = s - M(i,j) * x[j];   x[i] = unit ? s : s / M(i,i)
// over many shapes, both instruction sets and several thread counts, with values that break arithmetic (signed
// zeros, subnormals, infinities, NaN in the operands that are read, anything at all in the triangle that is not).
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#include "../../src/complex_linalg.hpp"
#include "vtest.hpp"

namespace {

using cdbl = std::complex<double>;
using vgpu::hostgemm::Isa;

// ---- inputs ----
enum class Mix { Plain, Zeros, Subnormal, Special, Overflow };

double draw(Mix mix, std::mt19937_64& rng) {
  std::uniform_real_distribution<double> u(-2.0, 2.0);
  switch (mix) {
    case Mix::Plain: return u(rng);
    case Mix::Zeros: return rng() % 3 == 0 ? (rng() % 2 ? 0.0 : -0.0) : u(rng);
    case Mix::Subnormal: return rng() % 4 == 0 ? std::numeric_limits<double>::denorm_min() * double(1 + rng() % 100) * (rng() % 2 ? 1 : -1) : u(rng);
    case Mix::Special: {
      const uint64_t r = rng() % 70;
      if (r == 0) return std::numeric_limits<double>::infinity();
      if (r == 1) return -std::numeric_limits<double>::infinity();
      if (r == 2) return std::numeric_limits<double>::quiet_NaN();
      return u(rng);
    }
    default: return rng() % 8 == 0 ? (rng() % 2 ? 1e300 : -1e300) : u(rng);
  }
}

bool same(double a, double b) { return std::memcmp(&a, &b, 8) == 0 || (std::isnan(a) && std::isnan(b)); }

// ---- the reference, real: cublas_api.cpp's tri_solve as it was ----
void ref_tri_solve(const std::vector<double>& a, int k, int lda, bool lower, bool trans, bool unit, double* x, size_t step) {
  auto M = [&](int i, int j) { return trans ? a[(size_t)i * lda + j] : a[(size_t)j * lda + i]; };
  auto row = [&](int i, int from, int to) {
    double s = x[(size_t)i * step];
    for (int j = from; j < to; ++j) s -= M(i, j) * x[(size_t)j * step];
    x[(size_t)i * step] = unit ? s : s / M(i, i);
  };
  if (lower != trans) for (int i = 0; i < k; ++i) row(i, 0, i);
  else for (int i = k - 1; i >= 0; --i) row(i, i + 1, k);
}

struct Shape {
  int m, n;
};

}  // namespace

VTEST(host_trsm_real_matches_the_loops) {
  std::mt19937_64 rng(31337);
  const Shape shapes[] = {{1, 1}, {5, 3}, {17, 9}, {16, 16}, {33, 70}, {64, 5}, {70, 33}, {130, 17}, {9, 130}, {100, 100}, {2, 40}};
  int cases = 0, runs = 0;
  size_t bad = 0;
  for (const Mix mix : {Mix::Plain, Mix::Zeros, Mix::Subnormal, Mix::Special, Mix::Overflow})
    for (const Shape& s : shapes)
      for (int variant = 0; variant < 16; ++variant) {
        const bool left = variant & 1, lower = variant & 2, trans = variant & 4, unit = variant & 8;
        const int k = left ? s.m : s.n, pad = cases % 3, lda = k + pad, ldb = s.m + pad;
        std::vector<double> a((size_t)lda * k), b((size_t)ldb * s.n);
        for (auto& x : a) x = draw(mix, rng);
        for (auto& x : b) x = draw(mix, rng);
        // The reference: left, each column; right, each row against the transpose.
        auto want = b;
        if (left) for (int j = 0; j < s.n; ++j) ref_tri_solve(a, k, lda, lower, trans, unit, &want[(size_t)j * ldb], 1);
        else for (int i = 0; i < s.m; ++i) ref_tri_solve(a, k, lda, lower, !trans, unit, &want[(size_t)i], (size_t)ldb);
        ++cases;
        for (const Isa isa : {Isa::Generic, Isa::Avx2}) {
          if (isa == Isa::Avx2 && !vgpu::hostgemm::avx2_available()) continue;
          for (const unsigned threads : {1u, 3u}) {
            auto got = b;
            const bool tm = left ? trans : !trans;
            const bool forward = lower != tm;
            const int64_t sti = tm ? lda : 1, stj = tm ? 1 : lda;
            if (left) vgpu::hosttrsm::real(k, s.n, forward, unit, a.data(), sti, stj, got.data(), 1, ldb, isa, threads);
            else vgpu::hosttrsm::real(k, s.m, forward, unit, a.data(), sti, stj, got.data(), ldb, 1, isa, threads);
            ++runs;
            for (size_t i = 0; i < got.size(); ++i)
              if (!same(got[i], want[i])) ++bad;
          }
        }
      }
  std::printf("  real: %d cases x instruction sets and thread counts (%d runs), %zu elements differ\n", cases, runs, bad);
  VCHECK_EQ(bad, static_cast<size_t>(0));
}

namespace {

// ---- the reference, complex: vgpu_la::trsm as it was ----
void ref_trsm(bool left, bool lower, vgpu_la::Op op, bool unit, int m, int n, cdbl alpha, const vgpu_la::CMat& a, vgpu_la::CMat& b) {
  using vgpu_la::Op;
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < m; ++i) b(i, j) *= alpha;
  if (left) {
    for (int j = 0; j < n; ++j) vgpu_la::tri_solve(a, m, lower, op, unit, &b(0, j), 1);
  } else {
    for (int i = 0; i < m; ++i) {
      cdbl* row = &b(i, 0);
      const size_t step = (size_t)b.ld;
      if (op == Op::C) {
        for (int j = 0; j < n; ++j) row[j * step] = std::conj(row[j * step]);
        vgpu_la::tri_solve(a, n, lower, Op::N, unit, row, step);
        for (int j = 0; j < n; ++j) row[j * step] = std::conj(row[j * step]);
      } else {
        vgpu_la::tri_solve(a, n, lower, op == Op::N ? Op::T : Op::N, unit, row, step);
      }
    }
  }
}

bool same_c(const cdbl& a, const cdbl& b) {
  const double ar = a.real(), ai = a.imag(), br = b.real(), bi = b.imag();
  return same(ar, br) && same(ai, bi);
}

}  // namespace

VTEST(host_trsm_complex_matches_the_loops) {
  using vgpu_la::Op;
  std::mt19937_64 rng(2718);
  const Shape shapes[] = {{1, 1}, {7, 5}, {16, 16}, {33, 40}, {64, 9}, {20, 70}, {70, 20}, {40, 40}, {3, 90}};
  int cases = 0, fast_cases = 0;
  size_t bad = 0;
  const cdbl alphas[] = {{1.0, 0.0}, {0.5, -1.25}, {0.0, 0.0}, {2.0, 3.0}};
  for (const Mix mix : {Mix::Plain, Mix::Zeros, Mix::Subnormal, Mix::Special, Mix::Overflow})
    for (const Shape& s : shapes)
      for (int variant = 0; variant < 24; ++variant) {
        const bool left = variant & 1, lower = variant & 2, unit = variant & 4;
        const Op op = (variant >> 3) == 0 ? Op::N : (variant >> 3) == 1 ? Op::T : Op::C;
        const int k = left ? s.m : s.n, pad = cases % 3;
        vgpu_la::CMat a, b;
        a.ld = k + pad;
        b.ld = s.m + pad;
        a.v.resize((size_t)a.ld * k);
        b.v.resize((size_t)b.ld * s.n);
        for (auto& x : a.v) x = cdbl(draw(mix, rng), draw(mix, rng));
        for (auto& x : b.v) x = cdbl(draw(mix, rng), draw(mix, rng));
        const cdbl alpha = alphas[cases % 4];
        auto want = b, got = b;
        ref_trsm(left, lower, op, unit, s.m, s.n, alpha, a, want);
        vgpu_la::trsm(left, lower, op, unit, s.m, s.n, alpha, a, got);
        ++cases;
        for (size_t i = 0; i < got.v.size(); ++i)
          if (!same_c(got.v[i], want.v[i])) ++bad;
        if (mix == Mix::Plain || mix == Mix::Zeros) ++fast_cases;
      }
  std::printf("  complex: %d cases (%d on operands that take the lanes), %zu elements differ\n", cases, fast_cases, bad);
  VCHECK_EQ(bad, static_cast<size_t>(0));
}

// The kernel itself on both instruction sets and several thread counts, finite operands.
VTEST(host_trsm_complex_kernel_matches_the_loops) {
  std::mt19937_64 rng(4141);
  size_t bad = 0;
  int runs = 0;
  for (const Shape& s : {Shape{5, 3}, Shape{17, 40}, Shape{64, 33}, Shape{100, 21}, Shape{9, 130}})
    for (int variant = 0; variant < 8; ++variant) {
      const bool forward = variant & 1, unit = variant & 2, conj_io = variant & 4;
      const int k = s.m;
      const int64_t nv = s.n;
      std::vector<double> mr((size_t)k * k, 0.0), mi(mr.size(), 0.0);
      for (int i = 0; i < k; ++i)
        for (int j = 0; j < k; ++j) {
          const bool used = forward ? j < i : j > i;
          if (used || i == j) {
            mr[(size_t)i * k + j] = i == j ? 1.0 + std::fabs(draw(Mix::Plain, rng)) : draw(Mix::Zeros, rng);
            mi[(size_t)i * k + j] = draw(Mix::Plain, rng);
          }
        }
      std::vector<cdbl> b((size_t)k * nv);
      for (auto& x : b) x = cdbl(draw(Mix::Zeros, rng), draw(Mix::Subnormal, rng));
      // Reference: each vector (column of the k x nv matrix b) by the loops.
      auto want = b;
      for (int64_t l = 0; l < nv; ++l) {
        cdbl* x = &want[(size_t)l * k];
        if (conj_io) for (int i = 0; i < k; ++i) x[i] = std::conj(x[i]);
        auto step = [&](int i, int from, int to) {
          cdbl sv = x[i];
          for (int j = from; j < to; ++j) sv -= cdbl(mr[(size_t)i * k + j], mi[(size_t)i * k + j]) * x[j];
          x[i] = unit ? sv : sv / cdbl(mr[(size_t)i * k + i], mi[(size_t)i * k + i]);
        };
        if (forward) for (int i = 0; i < k; ++i) step(i, 0, i);
        else for (int i = k - 1; i >= 0; --i) step(i, i + 1, k);
        if (conj_io) for (int i = 0; i < k; ++i) x[i] = std::conj(x[i]);
      }
      bool refinite = true;
      for (const auto& x : want) refinite &= std::isfinite(x.real()) && std::isfinite(x.imag());
      if (!refinite) continue;
      for (const Isa isa : {Isa::Generic, Isa::Avx2}) {
        if (isa == Isa::Avx2 && !vgpu::hostgemm::avx2_available()) continue;
        for (const unsigned threads : {1u, 3u}) {
          auto got = b;
          const bool ok = vgpu::hosttrsm::cplx(k, nv, forward, unit, mr.data(), mi.data(), got.data(), 1, k, conj_io, isa, threads);
          ++runs;
          VCHECK(ok);
          for (size_t i = 0; i < got.size(); ++i)
            if (!same_c(got[i], want[i])) ++bad;
        }
      }
    }
  std::printf("  complex kernel: %d runs, %zu elements differ\n", runs, bad);
  VCHECK_EQ(bad, static_cast<size_t>(0));
  VCHECK(runs > 40);
}

// The plane rotation of the Jacobi solvers (heev, svd in complex_linalg.hpp) against the expressions it replaced:
//     p' = c*xp - (s*conj(phase))*xq     q' = (s*phase)*xp + c*xq        (std::complex<double>)
// over numbers that break arithmetic, columns (stride 1) and rows (stride ld), counts from 0 to 40.
VTEST(complex_jacobi_rotation_matches_the_operators) {
  std::mt19937_64 rng(777);
  size_t bad = 0;
  int runs = 0;
  for (const Mix mix : {Mix::Plain, Mix::Zeros, Mix::Subnormal, Mix::Special, Mix::Overflow})
    for (int count = 0; count <= 40; count += (count < 8 ? 1 : 7))
      for (const int stride : {1, 5}) {
        const size_t len = static_cast<size_t>(count) * stride + 1;
        std::vector<cdbl> p(len), q(len);
        for (auto& x : p) x = cdbl(draw(mix, rng), draw(mix, rng));
        for (auto& x : q) x = cdbl(draw(mix, rng), draw(mix, rng));
        const double c = draw(mix, rng), s = draw(mix, rng);
        const cdbl phase(draw(mix, rng), draw(mix, rng));
        for (const bool rowform : {false, true}) {
          const cdbl m1 = rowform ? s * phase : s * std::conj(phase), m2 = rowform ? s * std::conj(phase) : s * phase;
          auto wp = p, wq = q;
          for (int k = 0; k < count; ++k) {
            const cdbl xp = wp[static_cast<size_t>(k) * stride], xq = wq[static_cast<size_t>(k) * stride];
            wp[static_cast<size_t>(k) * stride] = c * xp - m1 * xq;
            wq[static_cast<size_t>(k) * stride] = m2 * xp + c * xq;
          }
          auto gp = p, gq = q;
          vgpu_la::rotate_pair(gp.data(), gq.data(), stride, count, c, m1, m2);
          ++runs;
          for (size_t i = 0; i < len; ++i) bad += !same_c(gp[i], wp[i]) + !same_c(gq[i], wq[i]);
        }
      }
  std::printf("  rotation: %d runs, %zu elements differ\n", runs, bad);
  VCHECK_EQ(bad, static_cast<size_t>(0));
}

VTEST_MAIN
