// The sparse direct solver behind libvgpucudss (nvidia/src/cudss_solver.cpp),
// on its own: no simulator, no device memory. Random matrices of every kind
// cuDSS factors are solved and checked against a dense product, and the
// inertia against what the matrix's construction fixes. The matrices that
// need real pivoting -- zero diagonals, saddle points with a zero block --
// are here rather than in the e2e checks, which also run on NVIDIA's library
// and so stay with matrices its supernode-local pivoting factors exactly.
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

#include "cudss_solver.hpp"
#include "vtest.hpp"

namespace dss = vgpu_dss;
using cd = std::complex<double>;

namespace {

// A dense n x n matrix, row-major: the reference the factorization answers to.
template <class T>
struct Mat {
  int64_t n = 0;
  std::vector<T> a;
  explicit Mat(int64_t n_) : n(n_), a((size_t)(n_ * n_), T(0)) {}
  T& at(int64_t i, int64_t j) { return a[(size_t)(i * n + j)]; }
  T at(int64_t i, int64_t j) const { return a[(size_t)(i * n + j)]; }
};

double conj_of(double v) { return v; }
cd conj_of(cd v) { return std::conj(v); }

template <class T>
T random_value(std::mt19937_64& g) {
  std::uniform_real_distribution<double> u(-1, 1);
  if constexpr (std::is_same_v<T, cd>) return cd(u(g), u(g));
  else return u(g);
}

// Factors `m` as `kind`, from its full pattern (the solver keeps the lower
// triangle of the symmetric kinds), and solves m x = b for a random b.
template <class T>
struct Run {
  dss::Symbolic sym;
  dss::Factor<T> f;
  double residual = 0;  // ||b - A x|| / (||A|| ||x|| + ||b||), dense
  std::vector<T> x;
};

template <class T>
Run<T> solve(dss::Kind kind, const Mat<T>& m, std::mt19937_64& g, dss::Order order = dss::Order::MinimumDegree,
             std::vector<int64_t> user = {}, dss::Options opt = {}) {
  const int64_t n = m.n;
  std::vector<int64_t> start{0}, col;
  std::vector<T> val;
  for (int64_t i = 0; i < n; ++i) {
    for (int64_t j = 0; j < n; ++j)
      if (m.at(i, j) != T(0) || i == j) col.push_back(j), val.push_back(m.at(i, j));
    start.push_back((int64_t)col.size());
  }
  dss::Pattern pattern;
  std::vector<T> values;
  VCHECK(dss::canonical(kind, dss::View::Full, n, start, {}, col, val, &pattern, &values));
  Run<T> r;
  r.sym = dss::analyse(kind, pattern, order, user);
  r.f = dss::factorize(r.sym, pattern, values, opt);
  std::vector<T> b(n), w(n), z(n);
  for (auto& v : b) v = random_value<T>(g);
  r.x.resize(n);
  dss::permute(r.sym, b.data(), w.data());
  dss::forward(r.f, w.data());
  dss::diagonal(r.f, w.data());
  dss::backward(r.f, w.data(), z.data());
  dss::unpermute(r.sym, z.data(), r.x.data());
  double rn = 0, an = 0, xn = 0, bn = 0;
  for (int64_t i = 0; i < n; ++i) {
    T s = b[i];
    for (int64_t j = 0; j < n; ++j) s -= m.at(i, j) * r.x[j], an = std::max(an, std::abs(m.at(i, j)));
    rn = std::max(rn, std::abs(s));
    xn = std::max(xn, std::abs(r.x[i]));
    bn = std::max(bn, std::abs(b[i]));
  }
  r.residual = rn / (an * xn * (double)n + bn);
  // The sparse product the refinement uses agrees with the dense one.
  std::vector<T> y(n);
  dss::multiply(kind, pattern, values, r.x.data(), y.data());
  for (int64_t i = 0; i < n; ++i) {
    T s = 0;
    for (int64_t j = 0; j < n; ++j) s += m.at(i, j) * r.x[j];
    VCHECK(std::abs(s - y[i]) <= 1e-9 * (1 + std::abs(s)));
  }
  return r;
}

// A sparse matrix with `per_row` random entries a row; symmetry 1 mirrors
// each entry, 2 mirrors it conjugated (Hermitian).
template <class T>
Mat<T> random_sparse(int64_t n, int per_row, std::mt19937_64& g, int symmetry) {
  Mat<T> m(n);
  std::uniform_int_distribution<int64_t> pick(0, std::max<int64_t>(n - 1, 0));
  for (int64_t i = 0; i < n; ++i)
    for (int k = 0; k < per_row; ++k) {
      const int64_t j = pick(g);
      const T v = random_value<T>(g);
      m.at(i, j) = v;
      if (symmetry == 1) m.at(j, i) = v;
      if (symmetry == 2) m.at(j, i) = conj_of(v);
    }
  if (symmetry == 2)
    for (int64_t i = 0; i < n; ++i) m.at(i, i) = T(std::real(m.at(i, i)));
  return m;
}

// [H C^T; C -D]: H n x n, C m x n, D m x m diagonal (zero for a saddle point).
// With H positive definite and D positive or C of full row rank, its inertia
// is (n, m).
template <class T>
Mat<T> kkt(int64_t n, int64_t m, std::mt19937_64& g, double d, int symmetry) {
  Mat<T> k(n + m);
  auto h = random_sparse<T>(n, 2, g, symmetry);
  for (int64_t i = 0; i < n; ++i) {
    double row = 0;
    for (int64_t j = 0; j < n; ++j) k.at(i, j) = h.at(i, j), row += std::abs(h.at(i, j));
    k.at(i, i) = T(row + 1);  // positive definite by diagonal dominance
  }
  std::uniform_int_distribution<int64_t> pick(0, n - 1);
  for (int64_t r = 0; r < m; ++r) {
    // Row r of C touches column r (so C has full row rank once m <= n) and two more.
    const int64_t cols[3] = {r % n, pick(g), pick(g)};
    for (int64_t c : cols) {
      const T v = random_value<T>(g) + T(r % n == c ? 2.0 : 0.0);
      k.at(n + r, c) = v;
      k.at(c, n + r) = symmetry == 2 ? conj_of(v) : v;
    }
    k.at(n + r, n + r) = T(-d);
  }
  return k;
}

}  // namespace

VTEST(general_lu_with_zero_diagonals) {
  std::mt19937_64 g(1);
  for (int64_t n : {1, 2, 3, 5, 10, 40, 120}) {
    auto m = random_sparse<double>(n, 3, g, 0);
    // A random permutation's worth of large entries makes the matrix
    // nonsingular; zeroing the diagonal makes LU pivot.
    std::vector<int64_t> p(n);
    std::iota(p.begin(), p.end(), 0);
    std::shuffle(p.begin(), p.end(), g);
    for (int64_t i = 0; i < n; ++i) m.at(i, p[i]) += 4;
    for (int64_t i = 0; i < n; ++i)
      if (p[i] != i) m.at(i, i) = 0;
    auto r = solve(dss::Kind::General, m, g);
    VCHECK(r.residual < 1e-13);
    VCHECK_EQ(r.f.perturbed, int64_t{0});
    VCHECK_EQ(r.f.positive + r.f.negative, int64_t{0});  // inertia is for symmetric kinds only
  }
}

VTEST(antidiagonal_needs_pivoting) {
  std::mt19937_64 g(2);
  Mat<double> m(2);
  m.at(0, 1) = m.at(1, 0) = 1;
  auto lu = solve(dss::Kind::General, m, g);
  VCHECK(lu.residual < 1e-15);
  VCHECK_EQ(lu.f.perturbed, int64_t{0});
  // Symmetric: a 2x2 pivot, where NVIDIA's diagonal pivoting perturbs.
  auto ldl = solve(dss::Kind::Symmetric, m, g);
  VCHECK(ldl.residual < 1e-15);
  VCHECK_EQ(ldl.f.perturbed, int64_t{0});
  VCHECK_EQ(ldl.f.positive, int64_t{1});
  VCHECK_EQ(ldl.f.negative, int64_t{1});
}

VTEST(quasi_definite_kkt_inertia) {
  std::mt19937_64 g(3);
  for (auto [n, m] : {std::pair<int64_t, int64_t>{1, 1}, {3, 2}, {10, 15}, {40, 25}, {60, 120}}) {
    auto k = kkt<double>(n, m, g, 0.5, 1);
    for (bool pivoting : {true, false}) {
      dss::Options opt;
      opt.pivoting = pivoting;
      auto r = solve(dss::Kind::Symmetric, k, g, dss::Order::MinimumDegree, {}, opt);
      VCHECK(r.residual < 1e-13);
      VCHECK_EQ(r.f.positive, n);
      VCHECK_EQ(r.f.negative, m);
      VCHECK_EQ(r.f.perturbed, int64_t{0});
    }
  }
}

VTEST(saddle_point_with_zero_block) {
  // The (2,2) block is zero: every constraint row's pivot is zero until a 2x2
  // pivot or a delay pairs it with a variable.
  std::mt19937_64 g(4);
  for (auto [n, m] : {std::pair<int64_t, int64_t>{2, 1}, {5, 3}, {30, 20}, {80, 60}}) {
    auto k = kkt<double>(n, m, g, 0.0, 1);
    auto r = solve(dss::Kind::Symmetric, k, g);
    VCHECK(r.residual < 1e-12);
    VCHECK_EQ(r.f.positive, n);
    VCHECK_EQ(r.f.negative, m);
    VCHECK_EQ(r.f.perturbed, int64_t{0});
  }
}

VTEST(cholesky) {
  std::mt19937_64 g(5);
  for (int64_t n : {1, 4, 30, 150}) {
    auto k = kkt<double>(n, 0, g, 0, 1);
    auto r = solve(dss::Kind::Spd, k, g);
    VCHECK(r.residual < 1e-14);
    VCHECK_EQ(r.f.info, 0);
  }
  // Not positive definite: INFO is the failing pivot's 1-based position.
  Mat<double> m(2);
  m.at(0, 0) = m.at(1, 1) = 1;
  m.at(0, 1) = m.at(1, 0) = 2;
  auto r = solve(dss::Kind::Spd, m, g);
  VCHECK_EQ(r.f.info, 2);
}

VTEST(complex_kinds) {
  std::mt19937_64 g(6);
  auto herm = kkt<cd>(20, 12, g, 0.0, 2);
  auto h = solve(dss::Kind::Hermitian, herm, g);
  VCHECK(h.residual < 1e-12);
  VCHECK_EQ(h.f.positive, int64_t{20});
  VCHECK_EQ(h.f.negative, int64_t{12});
  auto hpd = solve(dss::Kind::Hpd, kkt<cd>(25, 0, g, 0, 2), g);
  VCHECK(hpd.residual < 1e-14);
  VCHECK_EQ(hpd.f.info, 0);
  auto sym = solve(dss::Kind::Symmetric, kkt<cd>(15, 10, g, 0.0, 1), g);  // complex symmetric
  VCHECK(sym.residual < 1e-12);
  auto gen = random_sparse<cd>(50, 3, g, 0);
  for (int64_t i = 0; i < 50; ++i) gen.at(i, (i * 7) % 50) += cd(3, 1);
  auto ge = solve(dss::Kind::General, gen, g);
  VCHECK(ge.residual < 1e-13);
}

VTEST(orders) {
  std::mt19937_64 g(7);
  auto k = kkt<double>(12, 8, g, 0.3, 1);
  std::vector<int64_t> user(20);
  std::iota(user.begin(), user.end(), 0);
  std::shuffle(user.begin(), user.end(), g);
  auto md = solve(dss::Kind::Symmetric, k, g);
  auto nat = solve(dss::Kind::Symmetric, k, g, dss::Order::Natural);
  auto us = solve(dss::Kind::Symmetric, k, g, dss::Order::User, user);
  for (const auto* r : {&md, &nat, &us}) {
    VCHECK(r->residual < 1e-13);
    VCHECK_EQ(r->f.positive, int64_t{12});
    VCHECK_EQ(r->f.negative, int64_t{8});
  }
  // A given order is kept exactly.
  VCHECK(us.sym.perm == user);
  for (int64_t i = 0; i < 20; ++i) VCHECK_EQ(nat.sym.perm[i], i);
}

VTEST(minimum_degree_reduces_fill) {
  // A 2-D Laplacian: banded in the natural order, much sparser factors when
  // reordered.
  const int64_t s = 30, n = s * s;
  Mat<double> m(n);
  for (int64_t i = 0; i < s; ++i)
    for (int64_t j = 0; j < s; ++j) {
      const int64_t k = i * s + j;
      m.at(k, k) = 4;
      if (i) m.at(k, k - s) = m.at(k - s, k) = -1;
      if (j) m.at(k, k - 1) = m.at(k - 1, k) = -1;
    }
  std::mt19937_64 g(8);
  auto md = solve(dss::Kind::Spd, m, g);
  auto nat = solve(dss::Kind::Spd, m, g, dss::Order::Natural);
  VCHECK(md.residual < 1e-14 && nat.residual < 1e-14);
  VCHECK(md.f.nnz * 3 < nat.f.nnz * 2);
  VCHECK(md.sym.nnz == md.f.nnz);  // Cholesky never pivots, so the prediction is exact
}

VTEST(pivot_epsilon) {
  std::mt19937_64 g(9);
  // A pivot smaller than epsilon becomes epsilon with its sign.
  Mat<double> m(2);
  m.at(0, 0) = -1e-4;
  m.at(1, 1) = 1;
  dss::Options opt;
  opt.eps = 1e-3;
  auto r = solve(dss::Kind::General, m, g, dss::Order::MinimumDegree, {}, opt);
  VCHECK_EQ(r.f.perturbed, int64_t{1});
  VCHECK_EQ(r.f.diag[0], -1e-3);
  // A singular symmetric matrix: one zero pivot, taken as +epsilon, as
  // NVIDIA's library counts it.
  Mat<double> s(2);
  s.a = {1, 1, 1, 1};
  auto z = solve(dss::Kind::Symmetric, s, g);
  VCHECK_EQ(z.f.perturbed, int64_t{1});
  VCHECK_EQ(z.f.positive, int64_t{2});
}

VTEST(views) {
  // The upper triangle, read as the transpose of the lower.
  const int64_t n = 3;
  const std::vector<int64_t> lstart{0, 1, 3, 5}, lcol{0, 0, 1, 1, 2};
  const std::vector<cd> lval{{4, 0}, {1, 2}, {5, 0}, {0, -1}, {6, 0}};
  const std::vector<int64_t> ustart{0, 2, 4, 5}, ucol{0, 1, 1, 2, 2};
  const std::vector<cd> uval{{4, 0}, {1, -2}, {5, 0}, {0, 1}, {6, 0}};
  dss::Pattern pl, pu;
  std::vector<cd> vl, vu;
  VCHECK(dss::canonical(dss::Kind::Hermitian, dss::View::Lower, n, lstart, {}, lcol, lval, &pl, &vl));
  VCHECK(dss::canonical(dss::Kind::Hermitian, dss::View::Upper, n, ustart, {}, ucol, uval, &pu, &vu));
  VCHECK(pl == pu);
  VCHECK(vl == vu);
  // An index out of range is refused rather than read.
  const std::vector<int64_t> bad{0, 0, 9, 1, 2};
  std::vector<double> out;
  VCHECK(!dss::canonical(dss::Kind::General, dss::View::Full, n, lstart, {}, bad, std::vector<double>{}, &pl, &out));
}

VTEST_MAIN
