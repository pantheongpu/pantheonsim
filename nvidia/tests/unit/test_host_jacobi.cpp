// The Jacobi sweeps behind cuSOLVER's real eigen- and singular value solvers (src/host_jacobi.hpp), bit for bit against the
// loops they replaced. Those loops are kept here, verbatim, as the reference:
//
//   syev  two-sided cyclic Jacobi:  for p, for q > p:  rotate columns p, q of A, rows p, q of A, columns p, q of V
//   svd   one-sided Jacobi:         for p, for q > p:  three sums over the rows of columns p, q; rotate columns p, q of A and V
//
// The comparison covers the whole of A and V after the last sweep, the sweeps made and the residual, over many sizes
// (every one that puts a run of four rotations or a vector tail in a different place), many kinds of matrix (dense,
// diagonal, banded, block, repeated eigenvalues, rank deficient, zero columns, signed zeros, subnormals, values whose
// squares overflow or underflow, infinities and NaNs), and both instruction sets. Only the sign and payload of a NaN that
// is the result of two NaNs meeting is not compared (it depends on the order the compiler gives the operands).
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

#include "../../src/host_jacobi.hpp"
#include "vtest.hpp"

namespace {

using vgpu::hostgemm::Isa;

// ---- the reference: cusolver_api.cpp's syev and svd as they were (the sweeps; the set-up and the sort are unchanged) ----
int ref_syev(std::vector<double>& a, std::vector<double>& vecs, int n, double* residual) {
  auto A = [&](int r, int c) -> double& { return a[(size_t)c * n + r]; };
  auto V = [&](int r, int c) -> double& { return vecs[(size_t)c * n + r]; };
  int sweeps = 0;
  for (int sweep = 0; sweep < 100; ++sweep) {
    double off = 0.0;
    for (int p = 0; p < n; ++p)
      for (int q = p + 1; q < n; ++q) off += A(p, q) * A(p, q);
    *residual = std::sqrt(2 * off);
    if (off <= 1e-30) break;
    ++sweeps;
    for (int p = 0; p < n; ++p)
      for (int q = p + 1; q < n; ++q) {
        if (std::fabs(A(p, q)) < 1e-300) continue;
        const double theta = (A(q, q) - A(p, p)) / (2.0 * A(p, q));
        const double t = (theta >= 0 ? 1.0 : -1.0) /
                         (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
        const double c = 1.0 / std::sqrt(t * t + 1.0), s = t * c;
        for (int i = 0; i < n; ++i) {
          const double aip = A(i, p), aiq = A(i, q);
          A(i, p) = c * aip - s * aiq;
          A(i, q) = s * aip + c * aiq;
        }
        for (int i = 0; i < n; ++i) {
          const double api = A(p, i), aqi = A(q, i);
          A(p, i) = c * api - s * aqi;
          A(q, i) = s * api + c * aqi;
        }
        for (int i = 0; i < n; ++i) {
          const double vip = V(i, p), viq = V(i, q);
          V(i, p) = c * vip - s * viq;
          V(i, q) = s * vip + c * viq;
        }
      }
  }
  return sweeps;
}

int ref_svd(std::vector<double>& a, std::vector<double>& v, int m, int n, double* residual) {
  auto A = [&](int r, int c) -> double& { return a[(size_t)c * m + r]; };
  auto V = [&](int r, int c) -> double& { return v[(size_t)c * n + r]; };
  int sweeps = 0;
  *residual = 0;
  for (int sweep = 0; sweep < 60; ++sweep) {
    double worst = 0.0, off = 0.0;
    ++sweeps;
    for (int p = 0; p < n; ++p)
      for (int q = p + 1; q < n; ++q) {
        double app = 0, aqq = 0, apq = 0;
        for (int i = 0; i < m; ++i) {
          app += A(i, p) * A(i, p);
          aqq += A(i, q) * A(i, q);
          apq += A(i, p) * A(i, q);
        }
        if (app == 0.0 || aqq == 0.0) continue;
        worst = std::max(worst, std::fabs(apq) / std::sqrt(app * aqq));
        off += apq * apq;
        if (std::fabs(apq) <= 1e-15 * std::sqrt(app * aqq)) continue;
        const double theta = (aqq - app) / (2.0 * apq);
        const double t = (theta >= 0 ? 1.0 : -1.0) /
                         (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
        const double c = 1.0 / std::sqrt(t * t + 1.0), sn = t * c;
        for (int i = 0; i < m; ++i) {
          const double aip = A(i, p), aiq = A(i, q);
          A(i, p) = c * aip - sn * aiq;
          A(i, q) = sn * aip + c * aiq;
        }
        for (int i = 0; i < n; ++i) {
          const double vip = V(i, p), viq = V(i, q);
          V(i, p) = c * vip - sn * viq;
          V(i, q) = sn * vip + c * viq;
        }
      }
    *residual = std::sqrt(2 * off);
    if (worst < 1e-14) break;
  }
  return sweeps;
}

// ---- inputs ----
struct Rng {
  uint64_t s;
  uint64_t next() {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return s;
  }
  double unit() { return static_cast<double>(next() >> 11) * (1.0 / 9007199254740992.0) * 2.0 - 1.0; }  // [-1, 1)
  bool coin(unsigned one_in) { return next() % one_in == 0; }
};

enum class Kind {
  Dense, Diagonal, Tridiagonal, Blocks, Arrow, Identity, Constant, Zero, RankOne, SparseRandom, Clustered,
  SignedZeros, Subnormal, Huge, Tiny, Infinite, NotANumber, Mixed, kCount
};
const char* kind_name(Kind k) {
  static const char* names[] = {"dense", "diagonal", "tridiagonal", "blocks", "arrow", "identity", "constant", "zero", "rank one",
                                "sparse", "clustered", "signed zeros", "subnormal", "huge", "tiny", "infinite", "nan", "mixed"};
  return names[static_cast<int>(k)];
}

// A symmetric n x n matrix, both triangles equal (as cuSOLVER's wrapper makes it from one).
std::vector<double> sym_matrix(Kind kind, int n, Rng& rng) {
  std::vector<double> a((size_t)n * n, 0.0);
  auto set = [&](int i, int j, double x) { a[(size_t)j * n + i] = a[(size_t)i * n + j] = x; };
  for (int j = 0; j < n; ++j)
    for (int i = 0; i <= j; ++i) {
      double x = rng.unit();
      switch (kind) {
        case Kind::Dense: break;
        case Kind::Diagonal: if (i != j) x = 0.0; break;
        case Kind::Tridiagonal: if (j - i > 1) x = 0.0; break;
        case Kind::Blocks: if ((i < n / 2) != (j < n / 2)) x = 0.0; break;
        case Kind::Arrow: if (i != j && i != 0) x = 0.0; break;
        case Kind::Identity: x = i == j ? 1.0 : 0.0; break;
        case Kind::Constant: x = 0.75; break;
        case Kind::Zero: x = 0.0; break;
        case Kind::RankOne: x = 0.5 * (1.0 + 0.1 * i) * (1.0 + 0.1 * j); break;
        case Kind::SparseRandom: if (rng.coin(10) == false && i != j) x = 0.0; break;
        case Kind::Clustered: x = i == j ? 1.0 + 1e-9 * (i % 3) : (rng.coin(3) ? 1e-7 * x : 0.0); break;
        case Kind::SignedZeros: x = rng.coin(2) ? (rng.coin(2) ? 0.0 : -0.0) : x; break;
        case Kind::Subnormal: x = rng.coin(3) ? std::numeric_limits<double>::denorm_min() * double(1 + rng.next() % 50) * (rng.coin(2) ? 1 : -1) : x * 1e-300; break;
        case Kind::Huge: x *= 1e160; break;
        case Kind::Tiny: x *= 1e-170; break;
        case Kind::Infinite: if (rng.coin(40)) x = rng.coin(2) ? std::numeric_limits<double>::infinity() : -std::numeric_limits<double>::infinity(); break;
        case Kind::NotANumber: if (rng.coin(60)) x = std::numeric_limits<double>::quiet_NaN(); break;
        case Kind::Mixed: if (rng.coin(5)) x = rng.coin(2) ? 1e150 : 1e-150; else if (rng.coin(5)) x = 0.0; break;
        default: break;
      }
      set(i, j, x);
    }
  return a;
}

// An m x n matrix.
std::vector<double> gen_matrix(Kind kind, int m, int n, Rng& rng) {
  std::vector<double> a((size_t)m * n, 0.0);
  auto at = [&](int i, int j) -> double& { return a[(size_t)j * m + i]; };
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < m; ++i) {
      double x = rng.unit();
      switch (kind) {
        case Kind::Dense: break;
        case Kind::Diagonal: if (i != j) x = 0.0; break;
        case Kind::Tridiagonal: if (std::abs(i - j) > 1) x = 0.0; break;
        case Kind::Blocks: if ((i < m / 2) != (j < n / 2)) x = 0.0; break;
        case Kind::Arrow: if (i != j && i != 0 && j != 0) x = 0.0; break;
        case Kind::Identity: x = i == j ? 1.0 : 0.0; break;
        case Kind::Constant: x = 0.75; break;
        case Kind::Zero: x = 0.0; break;
        case Kind::RankOne: x = 0.5 * (1.0 + 0.1 * i) * (1.0 + 0.1 * j); break;
        case Kind::SparseRandom: if (!rng.coin(8)) x = 0.0; break;
        case Kind::Clustered: x = (i == j ? 1.0 + 1e-9 * (i % 3) : 0.0) + 1e-8 * x; break;
        case Kind::SignedZeros: x = rng.coin(2) ? (rng.coin(2) ? 0.0 : -0.0) : x; break;
        case Kind::Subnormal: x = rng.coin(3) ? std::numeric_limits<double>::denorm_min() * double(1 + rng.next() % 50) * (rng.coin(2) ? 1 : -1) : x * 1e-300; break;
        case Kind::Huge: x *= 1e160; break;
        case Kind::Tiny: x *= 1e-170; break;
        case Kind::Infinite: if (rng.coin(40)) x = rng.coin(2) ? std::numeric_limits<double>::infinity() : -std::numeric_limits<double>::infinity(); break;
        case Kind::NotANumber: if (rng.coin(60)) x = std::numeric_limits<double>::quiet_NaN(); break;
        case Kind::Mixed: if (rng.coin(5)) x = rng.coin(2) ? 1e150 : 1e-150; else if (rng.coin(5)) x = 0.0; break;
        default: break;
      }
      at(i, j) = x;
    }
  // Columns that repeat or combine others, and columns that are zero, so that pivots and sums of squares vanish.
  if ((kind == Kind::Dense || kind == Kind::Mixed) && n >= 4) {
    for (int i = 0; i < m; ++i) {
      at(i, 1) = at(i, 0);
      at(i, 3) = at(i, 0) * 2.0 + (n > 2 ? at(i, 2) : 0.0);
    }
    if (n >= 6) for (int i = 0; i < m; ++i) at(i, 5) = 0.0;
  }
  return a;
}

bool same(double a, double b) { return std::memcmp(&a, &b, 8) == 0 || (std::isnan(a) && std::isnan(b)); }

size_t differing(const std::vector<double>& x, const std::vector<double>& y) {
  size_t bad = x.size() == y.size() ? 0 : 1;
  for (size_t i = 0; i < x.size() && i < y.size(); ++i)
    if (!same(x[i], y[i])) ++bad;
  return bad;
}

std::vector<Isa> isas() {
  std::vector<Isa> v = {Isa::Generic, Isa::Auto};
  if (vgpu::hostgemm::avx2_available()) v.push_back(Isa::Avx2);
  return v;
}

// The same matrix with the vectors not asked for: A, the sweeps and the residual are what they were (v is a null pointer, which
// nothing may read or write). Returns the number of differences.
size_t without_vectors_syev(const std::vector<double>& a0, int n, Isa isa, const std::vector<double>& ra, int rsw, double rres) {
  std::vector<double> ga = a0;
  double gres = -3;
  const int gsw = vgpu::hostjac::syev_sweeps(ga.data(), nullptr, n, &gres, isa, false);
  return differing(ga, ra) + (gsw != rsw) + !same(gres, rres);
}
size_t without_vectors_svd(const std::vector<double>& a0, int m, int n, Isa isa, const std::vector<double>& ra, int rsw, double rres) {
  std::vector<double> ga = a0;
  double gres = -3;
  const int gsw = vgpu::hostjac::svd_sweeps(ga.data(), nullptr, m, n, &gres, isa, false);
  return differing(ga, ra) + (gsw != rsw) + !same(gres, rres);
}

}  // namespace

VTEST(host_jacobi_syev_matches_the_loops) {
  Rng rng{0x9e3779b97f4a7c15ull};
  const int sizes[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 20, 25, 33, 41};
  int cases = 0, runs = 0;
  size_t bad = 0;
  for (int k = 0; k < static_cast<int>(Kind::kCount); ++k)
    for (const int n : sizes) {
      // Matrices with NaNs run the full hundred sweeps: keep them small.
      if ((Kind(k) == Kind::NotANumber || Kind(k) == Kind::Infinite) && n > 20) continue;
      const std::vector<double> a0 = sym_matrix(Kind(k), n, rng);
      std::vector<double> ra = a0, rv((size_t)n * n, 0.0);
      for (int i = 0; i < n; ++i) rv[(size_t)i * n + i] = 1.0;
      double rres = -1;
      const int rsw = ref_syev(ra, rv, n, &rres);
      ++cases;
      for (const Isa isa : isas()) {
        std::vector<double> ga = a0, gv((size_t)n * n, 0.0);
        for (int i = 0; i < n; ++i) gv[(size_t)i * n + i] = 1.0;
        double gres = -2;
        const int gsw = vgpu::hostjac::syev_sweeps(ga.data(), gv.data(), n, &gres, isa);
        ++runs;
        size_t d = differing(ga, ra) + differing(gv, rv) + (gsw != rsw) + !same(gres, rres);
        d += without_vectors_syev(a0, n, isa, ra, rsw, rres);
        if (d) std::printf("  syev %s n=%d isa=%d: %zu differ (sweeps %d vs %d)\n", kind_name(Kind(k)), n, static_cast<int>(isa), d, gsw, rsw);
        bad += d;
      }
    }
  std::printf("  syev: %d matrices x instruction sets (%d runs), %zu numbers differ\n", cases, runs, bad);
  VCHECK_EQ(bad, static_cast<size_t>(0));
}

VTEST(host_jacobi_syev_larger_matrices_match_the_loops) {
  Rng rng{12345};
  const int sizes[] = {47, 64, 65, 100, 131};
  size_t bad = 0;
  int runs = 0;
  for (const Kind kind : {Kind::Dense, Kind::Clustered, Kind::SparseRandom, Kind::Tridiagonal})
    for (const int n : sizes) {
      const std::vector<double> a0 = sym_matrix(kind, n, rng);
      std::vector<double> ra = a0, rv((size_t)n * n, 0.0);
      for (int i = 0; i < n; ++i) rv[(size_t)i * n + i] = 1.0;
      double rres = -1;
      const int rsw = ref_syev(ra, rv, n, &rres);
      for (const Isa isa : isas()) {
        std::vector<double> ga = a0, gv((size_t)n * n, 0.0);
        for (int i = 0; i < n; ++i) gv[(size_t)i * n + i] = 1.0;
        double gres = -2;
        const int gsw = vgpu::hostjac::syev_sweeps(ga.data(), gv.data(), n, &gres, isa);
        ++runs;
        size_t d = differing(ga, ra) + differing(gv, rv) + (gsw != rsw) + !same(gres, rres);
        d += without_vectors_syev(a0, n, isa, ra, rsw, rres);
        if (d) std::printf("  syev %s n=%d isa=%d: %zu differ\n", kind_name(kind), n, static_cast<int>(isa), d);
        bad += d;
      }
    }
  std::printf("  syev, larger: %d runs, %zu numbers differ\n", runs, bad);
  VCHECK_EQ(bad, static_cast<size_t>(0));
}

VTEST(host_jacobi_svd_matches_the_loops) {
  Rng rng{0xdeadbeefcafef00dull};
  struct Shape { int m, n; };
  const Shape shapes[] = {{0, 0}, {1, 0}, {0, 3}, {1, 1}, {3, 1}, {1, 3}, {2, 2}, {5, 3}, {4, 7}, {7, 4}, {8, 8}, {9, 9}, {17, 9}, {9, 17},
                          {12, 12}, {15, 16}, {31, 31}, {33, 20}, {20, 33}, {41, 30}};
  int cases = 0, runs = 0;
  size_t bad = 0;
  for (int k = 0; k < static_cast<int>(Kind::kCount); ++k)
    for (const Shape& s : shapes) {
      const std::vector<double> a0 = gen_matrix(Kind(k), s.m, s.n, rng);
      std::vector<double> ra = a0, rv((size_t)s.n * s.n, 0.0);
      for (int i = 0; i < s.n; ++i) rv[(size_t)i * s.n + i] = 1.0;
      double rres = -1;
      const int rsw = ref_svd(ra, rv, s.m, s.n, &rres);
      ++cases;
      for (const Isa isa : isas()) {
        std::vector<double> ga = a0, gv((size_t)s.n * s.n, 0.0);
        for (int i = 0; i < s.n; ++i) gv[(size_t)i * s.n + i] = 1.0;
        double gres = -2;
        const int gsw = vgpu::hostjac::svd_sweeps(ga.data(), gv.data(), s.m, s.n, &gres, isa);
        ++runs;
        size_t d = differing(ga, ra) + differing(gv, rv) + (gsw != rsw) + !same(gres, rres);
        d += without_vectors_svd(a0, s.m, s.n, isa, ra, rsw, rres);
        if (d) std::printf("  svd %s %dx%d isa=%d: %zu differ (sweeps %d vs %d)\n", kind_name(Kind(k)), s.m, s.n, static_cast<int>(isa), d, gsw, rsw);
        bad += d;
      }
    }
  std::printf("  svd: %d matrices x instruction sets (%d runs), %zu numbers differ\n", cases, runs, bad);
  VCHECK_EQ(bad, static_cast<size_t>(0));
}

VTEST(host_jacobi_svd_larger_matrices_match_the_loops) {
  Rng rng{777};
  struct Shape { int m, n; };
  const Shape shapes[] = {{64, 64}, {65, 63}, {100, 40}, {40, 100}, {96, 96}, {127, 33}};
  size_t bad = 0;
  int runs = 0;
  for (const Kind kind : {Kind::Dense, Kind::SparseRandom, Kind::Clustered})
    for (const Shape& s : shapes) {
      const std::vector<double> a0 = gen_matrix(kind, s.m, s.n, rng);
      std::vector<double> ra = a0, rv((size_t)s.n * s.n, 0.0);
      for (int i = 0; i < s.n; ++i) rv[(size_t)i * s.n + i] = 1.0;
      double rres = -1;
      const int rsw = ref_svd(ra, rv, s.m, s.n, &rres);
      for (const Isa isa : isas()) {
        std::vector<double> ga = a0, gv((size_t)s.n * s.n, 0.0);
        for (int i = 0; i < s.n; ++i) gv[(size_t)i * s.n + i] = 1.0;
        double gres = -2;
        const int gsw = vgpu::hostjac::svd_sweeps(ga.data(), gv.data(), s.m, s.n, &gres, isa);
        ++runs;
        size_t d = differing(ga, ra) + differing(gv, rv) + (gsw != rsw) + !same(gres, rres);
        d += without_vectors_svd(a0, s.m, s.n, isa, ra, rsw, rres);
        if (d) std::printf("  svd %s %dx%d isa=%d: %zu differ\n", kind_name(kind), s.m, s.n, static_cast<int>(isa), d);
        bad += d;
      }
    }
  std::printf("  svd, larger: %d runs, %zu numbers differ\n", runs, bad);
  VCHECK_EQ(bad, static_cast<size_t>(0));
}

VTEST_MAIN
