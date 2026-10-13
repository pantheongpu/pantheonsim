// The dense factorizations behind cuSOLVER (src/host_dense.hpp), bit for bit against the loops they replaced, which are
// kept here verbatim as the reference (cusolver_api.cpp's potrf, getrf, trsm_lower/upper, geqrf, orgqr, ormqr before
// host_dense.hpp). Random matrices of many shapes with a leading dimension larger than the rows, singular and
// indefinite ones (the failing paths), and a sprinkling of NaN and infinity.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#include "../../src/host_dense.hpp"
#include "vtest.hpp"

namespace {

struct Mat {
  std::vector<double> v;
  int ld = 0;
  double& operator()(int r, int c) { return v[(size_t)c * ld + r]; }
  double operator()(int r, int c) const { return v[(size_t)c * ld + r]; }
};

int ref_potrf(Mat& A, int n, bool lower) {
  for (int j = 0; j < n; ++j) {
    double d = A(j, j);
    for (int k = 0; k < j; ++k) {
      const double v = lower ? A(j, k) : A(k, j);
      d -= v * v;
    }
    if (!(d > 0.0)) return j + 1;
    d = std::sqrt(d);
    (lower ? A(j, j) : A(j, j)) = d;
    for (int i = j + 1; i < n; ++i) {
      double s = lower ? A(i, j) : A(j, i);
      for (int k = 0; k < j; ++k)
        s -= (lower ? A(i, k) : A(k, i)) * (lower ? A(j, k) : A(k, j));
      (lower ? A(i, j) : A(j, i)) = s / d;
    }
  }
  return 0;
}

int ref_getrf(Mat& A, int m, int n, std::vector<int>* ipiv) {
  const int k = std::min(m, n);
  if (ipiv) ipiv->assign(k, 0);
  int info = 0;
  for (int j = 0; j < k; ++j) {
    int p = j;
    if (ipiv) {
      double best = std::fabs(A(j, j));
      for (int i = j + 1; i < m; ++i)
        if (std::fabs(A(i, j)) > best) { best = std::fabs(A(i, j)); p = i; }
      (*ipiv)[j] = p + 1;
      if (p != j)
        for (int c = 0; c < n; ++c) std::swap(A(j, c), A(p, c));
    }
    const double piv = A(j, j);
    if (piv == 0.0) { if (!info) info = j + 1; continue; }
    for (int i = j + 1; i < m; ++i) {
      A(i, j) /= piv;
      const double f = A(i, j);
      for (int c = j + 1; c < n; ++c) A(i, c) -= f * A(j, c);
    }
  }
  return info;
}

void ref_trsm_lower(const Mat& A, Mat& B, int n, int nrhs, bool unit, bool trans) {
  for (int c = 0; c < nrhs; ++c) {
    if (!trans) {
      for (int i = 0; i < n; ++i) {
        double s = B(i, c);
        for (int k = 0; k < i; ++k) s -= A(i, k) * B(k, c);
        B(i, c) = unit ? s : s / A(i, i);
      }
    } else {
      for (int i = n - 1; i >= 0; --i) {
        double s = B(i, c);
        for (int k = i + 1; k < n; ++k) s -= A(k, i) * B(k, c);
        B(i, c) = unit ? s : s / A(i, i);
      }
    }
  }
}

void ref_trsm_upper(const Mat& A, Mat& B, int n, int nrhs, bool unit, bool trans) {
  for (int c = 0; c < nrhs; ++c) {
    if (!trans) {
      for (int i = n - 1; i >= 0; --i) {
        double s = B(i, c);
        for (int k = i + 1; k < n; ++k) s -= A(i, k) * B(k, c);
        B(i, c) = unit ? s : s / A(i, i);
      }
    } else {
      for (int i = 0; i < n; ++i) {
        double s = B(i, c);
        for (int k = 0; k < i; ++k) s -= A(k, i) * B(k, c);
        B(i, c) = unit ? s : s / A(i, i);
      }
    }
  }
}

void ref_geqrf(Mat& A, int m, int n, std::vector<double>* tau) {
  const int k = std::min(m, n);
  tau->assign(k, 0.0);
  std::vector<double> v((size_t)m);
  for (int j = 0; j < k; ++j) {
    double norm = 0.0;
    for (int i = j; i < m; ++i) norm += A(i, j) * A(i, j);
    norm = std::sqrt(norm);
    if (norm == 0.0) { (*tau)[j] = 0.0; continue; }
    const double alpha = A(j, j);
    const double beta = alpha >= 0.0 ? -norm : norm;
    const double t = (beta - alpha) / beta;
    (*tau)[j] = t;
    const double scale = alpha - beta;
    v[j] = 1.0;
    for (int i = j + 1; i < m; ++i) { A(i, j) /= scale; v[i] = A(i, j); }
    A(j, j) = beta;
    for (int c = j + 1; c < n; ++c) {
      double dot = 0.0;
      for (int i = j; i < m; ++i) dot += v[i] * A(i, c);
      dot *= t;
      for (int i = j; i < m; ++i) A(i, c) -= dot * v[i];
    }
  }
}

void ref_orgqr(Mat& A, int m, int n, int k, const std::vector<double>& tau) {
  std::vector<double> q((size_t)m * n, 0.0);
  auto Q = [&](int r, int c) -> double& { return q[(size_t)c * m + r]; };
  for (int c = 0; c < n; ++c) Q(c, c) = 1.0;
  std::vector<double> v((size_t)m);
  for (int j = k - 1; j >= 0; --j) {
    v[j] = 1.0;
    for (int i = j + 1; i < m; ++i) v[i] = A(i, j);
    for (int c = 0; c < n; ++c) {
      double dot = 0.0;
      for (int i = j; i < m; ++i) dot += v[i] * Q(i, c);
      dot *= tau[j];
      for (int i = j; i < m; ++i) Q(i, c) -= dot * v[i];
    }
  }
  for (int c = 0; c < n; ++c)
    for (int r = 0; r < m; ++r) A(r, c) = Q(r, c);
}

void ref_ormqr(const Mat& A, Mat& C, int m, int n, int k, const std::vector<double>& tau, bool left, bool trans) {
  const int rows = left ? m : n;
  std::vector<double> v((size_t)rows);
  for (int s = 0; s < k; ++s) {
    const int j = (left != trans) ? k - 1 - s : s;
    for (int i = 0; i < rows; ++i) v[i] = i < j ? 0.0 : (i == j ? 1.0 : A(i, j));
    if (left) {
      for (int c = 0; c < n; ++c) {
        double dot = 0.0;
        for (int i = j; i < m; ++i) dot += v[i] * C(i, c);
        dot *= tau[j];
        for (int i = j; i < m; ++i) C(i, c) -= dot * v[i];
      }
    } else {
      for (int r = 0; r < m; ++r) {
        double dot = 0.0;
        for (int i = j; i < n; ++i) dot += v[i] * C(r, i);
        dot *= tau[j];
        for (int i = j; i < n; ++i) C(r, i) -= dot * v[i];
      }
    }
  }
}

// ---- inputs ----

std::mt19937_64 g_rng(2024);

Mat random_mat(int rows, int cols, int pad, int kind) {
  Mat A;
  A.ld = rows + pad;
  A.v.assign((size_t)A.ld * cols, 0.0);
  std::uniform_real_distribution<double> u(-2.0, 2.0);
  for (auto& x : A.v) {
    x = u(g_rng);
    if (kind == 1 && g_rng() % 60 == 0) {
      static const double sp[] = {0.0, -0.0, 1e300, -1e300, INFINITY, -INFINITY, std::numeric_limits<double>::quiet_NaN(), 1e-300};
      x = sp[g_rng() % 8];
    }
  }
  if (kind == 2)   // a zero column and a duplicated one: singular
    for (int r = 0; r < rows && cols > 2; ++r) {
      A(r, cols / 2) = 0.0;
      A(r, 0) = A(r, 1);
    }
  return A;
}

// Symmetric positive definite (diagonally dominant) in both triangles; kind 3 spoils a diagonal entry.
Mat spd(int n, int pad, int kind) {
  Mat A = random_mat(n, n, pad, 0);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < i; ++j) A(j, i) = A(i, j);
  for (int i = 0; i < n; ++i) A(i, i) = std::fabs(A(i, i)) + n * 2.0;
  if (kind == 3 && n > 3) A(n / 2, n / 2) = -1.0;
  return A;
}

// Equal bits; two NaNs count as equal (see host_gemm.hpp for why the NaN that wins is not fixed).
size_t differing(const std::vector<double>& a, const std::vector<double>& b) {
  if (a.size() != b.size()) return a.size() + 1;
  size_t bad = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    if (std::memcmp(&a[i], &b[i], 8) == 0) continue;
    if (std::isnan(a[i]) && std::isnan(b[i])) continue;
    ++bad;
  }
  return bad;
}

struct Shape { int m, n; };
const Shape kShapes[] = {{1, 1}, {2, 3}, {3, 2}, {5, 5}, {8, 8}, {13, 9}, {9, 13}, {32, 32}, {37, 41}, {64, 50}, {100, 100}, {129, 70}, {70, 129}, {200, 200}};

}  // namespace

VTEST(host_dense_getrf_matches) {
  size_t bad = 0;
  int cases = 0;
  std::vector<Shape> shapes(std::begin(kShapes), std::end(kShapes));
  for (const Shape& extra : {Shape{300, 300}, Shape{400, 150}, Shape{150, 400}, Shape{257, 129}, Shape{129, 257}})   // across the panels of the blocked form
    shapes.push_back(extra);
  for (const Shape& s : shapes)
    for (int kind = 0; kind < 3; ++kind)
      for (const bool pivot : {true, false}) {
        Mat A = random_mat(s.m, s.n, 3, kind), R = A;
        std::vector<int> ipiv_ref, ipiv_new(std::min(s.m, s.n));
        const int info_ref = ref_getrf(R, s.m, s.n, pivot ? &ipiv_ref : nullptr);
        const int info_new = vgpu::hostdense::getrf(A.v.data(), A.ld, s.m, s.n, pivot ? ipiv_new.data() : nullptr);
        VCHECK_EQ(info_ref, info_new);
        if (pivot) VCHECK(ipiv_ref == ipiv_new);
        bad += differing(R.v, A.v);
        ++cases;
      }
  // A zero pivot in a later panel (a zero column, no earlier trouble): the blocked form hands the rest over to the plain loops.
  for (const Shape& s : {Shape{300, 300}, Shape{200, 200}, Shape{257, 200}, Shape{200, 257}})
    for (const bool pivot : {true, false})
      for (const int zero_col : {s.n / 2, 130, 190}) {
        Mat A = random_mat(s.m, s.n, 2, 0);
        for (int r = 0; r < s.m; ++r) A(r, zero_col) = 0.0;
        Mat R = A;
        std::vector<int> ipiv_ref, ipiv_new(std::min(s.m, s.n));
        const int info_ref = ref_getrf(R, s.m, s.n, pivot ? &ipiv_ref : nullptr);
        const int info_new = vgpu::hostdense::getrf(A.v.data(), A.ld, s.m, s.n, pivot ? ipiv_new.data() : nullptr);
        VCHECK_EQ(info_ref, info_new);
        VCHECK(info_ref > 0);
        if (pivot) VCHECK(ipiv_ref == ipiv_new);
        bad += differing(R.v, A.v);
        ++cases;
      }
  std::printf("  getrf: %d cases, %zu elements differ\n", cases, bad);
  VCHECK_EQ(bad, static_cast<size_t>(0));
}

VTEST(host_dense_potrf_matches) {
  size_t bad = 0;
  int cases = 0;
  for (const int n : {1, 2, 3, 4, 5, 7, 8, 9, 16, 33, 64, 100, 150})
    for (int kind = 0; kind < 3; ++kind)
      for (const bool lower : {true, false}) {
        Mat A = kind == 1 ? random_mat(n, n, 2, 1) : spd(n, 2, kind == 2 ? 3 : 0), R = A;
        const int info_ref = ref_potrf(R, n, lower);
        const int info_new = vgpu::hostdense::potrf(A.v.data(), A.ld, n, lower);
        VCHECK_EQ(info_ref, info_new);
        bad += differing(R.v, A.v);
        ++cases;
      }
  std::printf("  potrf: %d cases, %zu elements differ\n", cases, bad);
  VCHECK_EQ(bad, static_cast<size_t>(0));
}

VTEST(host_dense_triangular_solves_match) {
  size_t bad = 0;
  int cases = 0;
  for (const int n : {1, 2, 5, 9, 16, 31, 64, 100})
    for (const int nrhs : {1, 3, 4, 5, 20})
      for (int kind = 0; kind < 2; ++kind)
        for (int variant = 0; variant < 8; ++variant) {
          const bool upper = variant & 1, unit = variant & 2, trans = variant & 4;
          Mat A = random_mat(n, n, 2, kind);
          for (int i = 0; i < n; ++i) A(i, i) += (A(i, i) >= 0 ? 3.0 : -3.0);   // keep the diagonal away from 0
          Mat B = random_mat(n, nrhs, 1, kind), R = B;
          if (upper) {
            ref_trsm_upper(A, R, n, nrhs, unit, trans);
            vgpu::hostdense::trsm_upper(A.v.data(), A.ld, B.v.data(), B.ld, n, nrhs, unit, trans);
          } else {
            ref_trsm_lower(A, R, n, nrhs, unit, trans);
            vgpu::hostdense::trsm_lower(A.v.data(), A.ld, B.v.data(), B.ld, n, nrhs, unit, trans);
          }
          bad += differing(R.v, B.v);
          ++cases;
        }
  std::printf("  trsm: %d cases, %zu elements differ\n", cases, bad);
  VCHECK_EQ(bad, static_cast<size_t>(0));
}

VTEST(host_dense_qr_matches) {
  size_t bad = 0;
  int cases = 0;
  for (const Shape& s : kShapes)
    for (int kind = 0; kind < 3; ++kind) {
      Mat A = random_mat(s.m, s.n, 2, kind), R = A;
      std::vector<double> tau_ref, tau_new(std::min(s.m, s.n));
      ref_geqrf(R, s.m, s.n, &tau_ref);
      vgpu::hostdense::geqrf(A.v.data(), A.ld, s.m, s.n, tau_new.data());
      bad += differing(R.v, A.v) + differing(tau_ref, tau_new);
      ++cases;
      const int k = std::min(s.m, s.n);
      if (kind == 1 || s.m < s.n) continue;   // Q needs m >= n
      // Q from the reflectors, and Q (or its transpose) applied from either side.
      Mat Q = A, QR = R;
      ref_orgqr(QR, s.m, s.n, k, tau_ref);
      vgpu::hostdense::orgqr(Q.v.data(), Q.ld, s.m, s.n, k, tau_new.data());
      bad += differing(QR.v, Q.v);
      for (const bool left : {true, false})
        for (const bool trans : {false, true}) {
          const int cr = left ? s.m : 7, cc = left ? (s.m > 60 ? 45 : 6) : s.m;   // C: m x 6 for left, 7 x m for right
          const int cn = left ? cc : s.m;                          // (ormqr's n is C's column count)
          Mat C = random_mat(cr, cc, 1, 0), CR = C;
          (void)cn;
          if (left) {
            ref_ormqr(R, CR, s.m, cc, k, tau_ref, true, trans);
            vgpu::hostdense::ormqr(A.v.data(), A.ld, C.v.data(), C.ld, s.m, cc, k, tau_new.data(), true, trans);
          } else {
            ref_ormqr(R, CR, cr, s.m, std::min(k, s.m), tau_ref, false, trans);
            vgpu::hostdense::ormqr(A.v.data(), A.ld, C.v.data(), C.ld, cr, s.m, std::min(k, s.m), tau_new.data(), false, trans);
          }
          bad += differing(CR.v, C.v);
        }
      ++cases;
    }
  std::printf("  qr: %d cases, %zu elements differ\n", cases, bad);
  VCHECK_EQ(bad, static_cast<size_t>(0));
}

// The reflector kernel on its own (host_trsm.hpp, reflect): arbitrary reflector storage and scales (zero, negative zero, NaN,
// infinity among them), both directions, with and without skipping zero scales, both instruction sets, several thread
// counts, columns across the width of a group; against the loops (dot, scale, update per column, reflector by reflector).
VTEST(host_dense_reflect_kernel_matches) {
  using vgpu::hostgemm::Isa;
  size_t bad = 0;
  int runs = 0;
  struct Case { int m, ncols, k; };
  for (const Case& s : {Case{40, 5, 7}, Case{64, 33, 20}, Case{100, 70, 30}, Case{7, 40, 7}, Case{130, 17, 64}, Case{20, 100, 3}})
    for (int kind = 0; kind < 3; ++kind)
      for (const bool down : {false, true})
        for (const bool skip : {false, true}) {
          Mat A = random_mat(s.m, s.k, 2, kind);
          Mat C = random_mat(s.m, s.ncols, 1, kind);
          std::vector<double> tau(s.k);
          for (int j = 0; j < s.k; ++j) {
            const uint64_t r = g_rng() % 12;
            tau[j] = r == 0 ? 0.0 : r == 1 ? -0.0 : (kind == 1 && r == 2) ? std::numeric_limits<double>::quiet_NaN() : 1.0 + double(g_rng() % 100) / 50.0;
          }
          const int j0 = down ? s.k - 1 : 0, j1 = down ? 0 : s.k - 1, step = down ? -1 : 1;
          Mat R = C;
          for (int j = j0; down ? j >= j1 : j <= j1; j += step) {
            if (skip && tau[j] == 0.0) continue;
            if (j >= s.m) continue;
            std::vector<double> v(s.m);
            v[j] = 1.0;
            for (int i = j + 1; i < s.m; ++i) v[i] = A(i, j);
            for (int c = 0; c < s.ncols; ++c) {
              double dot = 0.0;
              for (int i = j; i < s.m; ++i) dot += v[i] * R(i, c);
              dot *= tau[j];
              for (int i = j; i < s.m; ++i) R(i, c) -= dot * v[i];
            }
          }
          for (const Isa isa : {Isa::Generic, Isa::Avx2}) {
            if (isa == Isa::Avx2 && !vgpu::hostgemm::avx2_available()) continue;
            for (const unsigned threads : {1u, 3u}) {
              Mat G = C;
              vgpu::hosttrsm::reflect(A.v.data(), A.ld, tau.data(), s.m, j0, j1, step, G.v.data(), G.ld, 0, s.ncols, skip, isa, threads);
              ++runs;
              bad += differing(R.v, G.v);
            }
          }
        }
  std::printf("  reflect: %d runs, %zu elements differ\n", runs, bad);
  VCHECK_EQ(bad, static_cast<size_t>(0));
}

VTEST_MAIN
