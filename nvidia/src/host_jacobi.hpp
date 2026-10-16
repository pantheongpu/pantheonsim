#pragma once
// The sweeps of the Jacobi solvers behind cuSOLVER's real eigen- and singular value routines (syevd, syevj, gesvd, gesvdj,
// gesvda; cusolver_api.cpp). Every element of every result is the one the plain loops gave, bit for bit: the loops are kept
// below as the generic path (and in tests/unit/test_host_jacobi.cpp as the reference), and the AVX2 path reorders only
// work that does not depend on its order.
//
// The two-sided method (syev), for p ascending and q ascending in (p, q):
//     c, s from A(p,p), A(q,q), A(p,q);   columns p, q of A rotated;   rows p, q of A rotated;   columns p, q of V rotated
// A rotation reads and writes only rows and columns p and q, and its parameters only the three elements (p,p), (q,q), (p,q).
// For one p, a run of four consecutive q (q0 .. q0+3) is done together. Let S = {p, q0 .. q0+3}.
//   * The elements of S x S (25 numbers) are all that the four rotations' parameters depend on, and the rotations act on
//     them only through S x S; they are copied out, the four rotations are done on the copy in order (the plain loops
//     restricted to S), and the copy is written back last.
//   * Every other element touched is in a column p or q_a with a row outside S (the column rotations), or in a row p or q_a
//     with a column outside S (the row rotations), or in V: each such element goes through the four rotations in order,
//     alone, so the four are done one after another in registers: for a column of A or V the elements along the column are
//     the vector lanes; for the rows the running row p is kept in a contiguous buffer, and the four rows q0 .. q0+3 are
//     the columns of a 4x4 tile that is transposed into registers, so that the lanes are four columns of A.
//   * The same code on the S x S elements gives numbers that are then overwritten by the copy.
// A rotation whose pivot is below 1e-300 is not done (as before); a run with some of its pivots that small applies only
// the others.
//
// The one-sided method (svd): the rotation of the pair (p, q) needs three sums over the rows of columns p and q, each a
// chain in row order, and the pair depends on every earlier pair that shares a column. The pairs (p, q) with p + q = t
// share no column with each other and every pair that precedes one of them in the plain order has a smaller p + q, so
// the pairs are done by t; four pairs at a time have their sums in the four lanes of a vector (each lane a chain in row
// order, a rounded product and a rounded sum, no fusion). The sum that the residual takes over the pairs is made in the
// plain order afterwards, and the largest cosine, which does not depend on order, as it goes.
//
// No fused multiply-add anywhere: the target is avx2 only, and every product and sum is written as its own operation.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "host_gemm.hpp"

namespace vgpu::hostjac {

using vgpu::hostgemm::Isa;

#ifdef VGPU_HOSTGEMM_AVX2
#pragma GCC push_options
#pragma GCC target("avx2")
// Nothing here calls a std:: template (see host_gemm_impl.inc); the names are this namespace's own.
namespace avx2 {

// out[a] lane l = in[l] lane a.
inline void tr4(__m256d& a, __m256d& b, __m256d& c, __m256d& d) {
  const __m256d t0 = _mm256_unpacklo_pd(a, b), t1 = _mm256_unpackhi_pd(a, b);
  const __m256d t2 = _mm256_unpacklo_pd(c, d), t3 = _mm256_unpackhi_pd(c, d);
  a = _mm256_permute2f128_pd(t0, t2, 0x20);
  b = _mm256_permute2f128_pd(t1, t3, 0x20);
  c = _mm256_permute2f128_pd(t0, t2, 0x31);
  d = _mm256_permute2f128_pd(t1, t3, 0x31);
}

inline __m256d load2x2(const double* lo, const double* hi) {
  return _mm256_insertf128_pd(_mm256_castpd128_pd256(_mm_loadu_pd(lo)), _mm_loadu_pd(hi), 1);
}

// ---- the one-sided method ----

// For four pairs at a time (pp[k], qq[k] the columns of pair k): the chains
//     app = 0;  app = app + P[i]*P[i]  (i = 0 .. m-1)    aqq likewise with Q    apq = apq + P[i]*Q[i]
// one pair to a lane.
inline void dots4(const double* const* pp, const double* const* qq, int m, double* app, double* aqq, double* apq) {
  __m256d ap = _mm256_setzero_pd(), aq = _mm256_setzero_pd(), apq4 = _mm256_setzero_pd();
  int i = 0;
  for (; i + 4 <= m; i += 4) {
    __m256d P[4], Q[4];
    {
      const __m256d x0 = load2x2(pp[0] + i, pp[2] + i), x1 = load2x2(pp[1] + i, pp[3] + i);
      const __m256d x2 = load2x2(pp[0] + i + 2, pp[2] + i + 2), x3 = load2x2(pp[1] + i + 2, pp[3] + i + 2);
      P[0] = _mm256_unpacklo_pd(x0, x1);
      P[1] = _mm256_unpackhi_pd(x0, x1);
      P[2] = _mm256_unpacklo_pd(x2, x3);
      P[3] = _mm256_unpackhi_pd(x2, x3);
    }
    {
      const __m256d x0 = load2x2(qq[0] + i, qq[2] + i), x1 = load2x2(qq[1] + i, qq[3] + i);
      const __m256d x2 = load2x2(qq[0] + i + 2, qq[2] + i + 2), x3 = load2x2(qq[1] + i + 2, qq[3] + i + 2);
      Q[0] = _mm256_unpacklo_pd(x0, x1);
      Q[1] = _mm256_unpackhi_pd(x0, x1);
      Q[2] = _mm256_unpacklo_pd(x2, x3);
      Q[3] = _mm256_unpackhi_pd(x2, x3);
    }
    for (int j = 0; j < 4; ++j) {
      ap = _mm256_add_pd(ap, _mm256_mul_pd(P[j], P[j]));
      aq = _mm256_add_pd(aq, _mm256_mul_pd(Q[j], Q[j]));
      apq4 = _mm256_add_pd(apq4, _mm256_mul_pd(P[j], Q[j]));
    }
  }
  for (; i < m; ++i) {
    const __m256d p = _mm256_set_pd(pp[3][i], pp[2][i], pp[1][i], pp[0][i]);
    const __m256d q = _mm256_set_pd(qq[3][i], qq[2][i], qq[1][i], qq[0][i]);
    ap = _mm256_add_pd(ap, _mm256_mul_pd(p, p));
    aq = _mm256_add_pd(aq, _mm256_mul_pd(q, q));
    apq4 = _mm256_add_pd(apq4, _mm256_mul_pd(p, q));
  }
  _mm256_storeu_pd(app, ap);
  _mm256_storeu_pd(aqq, aq);
  _mm256_storeu_pd(apq, apq4);
}

// P[i] := c*P[i] - s*Q[i];  Q[i] := s*P[i] + c*Q[i]  (the old P[i] in both).
inline void rotate(double* P, double* Q, int len, double c, double s) {
  const __m256d vc = _mm256_set1_pd(c), vs = _mm256_set1_pd(s);
  int i = 0;
  for (; i + 8 <= len; i += 8) {
    const __m256d a0 = _mm256_loadu_pd(P + i), b0 = _mm256_loadu_pd(Q + i);
    const __m256d a1 = _mm256_loadu_pd(P + i + 4), b1 = _mm256_loadu_pd(Q + i + 4);
    const __m256d p0 = _mm256_sub_pd(_mm256_mul_pd(vc, a0), _mm256_mul_pd(vs, b0));
    const __m256d q0 = _mm256_add_pd(_mm256_mul_pd(vs, a0), _mm256_mul_pd(vc, b0));
    const __m256d p1 = _mm256_sub_pd(_mm256_mul_pd(vc, a1), _mm256_mul_pd(vs, b1));
    const __m256d q1 = _mm256_add_pd(_mm256_mul_pd(vs, a1), _mm256_mul_pd(vc, b1));
    _mm256_storeu_pd(P + i, p0);
    _mm256_storeu_pd(Q + i, q0);
    _mm256_storeu_pd(P + i + 4, p1);
    _mm256_storeu_pd(Q + i + 4, q1);
  }
  for (; i + 4 <= len; i += 4) {
    const __m256d a = _mm256_loadu_pd(P + i), b = _mm256_loadu_pd(Q + i);
    _mm256_storeu_pd(P + i, _mm256_sub_pd(_mm256_mul_pd(vc, a), _mm256_mul_pd(vs, b)));
    _mm256_storeu_pd(Q + i, _mm256_add_pd(_mm256_mul_pd(vs, a), _mm256_mul_pd(vc, b)));
  }
  for (; i < len; ++i) {
    const double a = P[i], b = Q[i];
    P[i] = c * a - s * b;
    Q[i] = s * a + c * b;
  }
}

// ---- the two-sided method ----

// For each i in [0, len): x[i] goes through the rotations k = 0 .. K-1 in order, each against y[k][i]:
//     nx = c[k]*x - s[k]*y[k];  y[k] = s[k]*x + c[k]*y[k];  x = nx
inline void colchain(double* x, double* const* y, int K, int len, const double* c, const double* s) {
  if (K == 4) {
    __m256d vc[4], vs[4];
    for (int k = 0; k < 4; ++k) {
      vc[k] = _mm256_set1_pd(c[k]);
      vs[k] = _mm256_set1_pd(s[k]);
    }
    int i = 0;
    for (; i + 4 <= len; i += 4) {
      __m256d X = _mm256_loadu_pd(x + i);
      for (int k = 0; k < 4; ++k) {
        const __m256d Y = _mm256_loadu_pd(y[k] + i);
        const __m256d nx = _mm256_sub_pd(_mm256_mul_pd(vc[k], X), _mm256_mul_pd(vs[k], Y));
        _mm256_storeu_pd(y[k] + i, _mm256_add_pd(_mm256_mul_pd(vs[k], X), _mm256_mul_pd(vc[k], Y)));
        X = nx;
      }
      _mm256_storeu_pd(x + i, X);
    }
    for (; i < len; ++i) {
      double X = x[i];
      for (int k = 0; k < 4; ++k) {
        const double Y = y[k][i];
        const double nx = c[k] * X - s[k] * Y;
        y[k][i] = s[k] * X + c[k] * Y;
        X = nx;
      }
      x[i] = X;
    }
    return;
  }
  for (int i = 0; i < len; ++i) {
    double X = x[i];
    for (int k = 0; k < K; ++k) {
      const double Y = y[k][i];
      const double nx = c[k] * X - s[k] * Y;
      y[k][i] = s[k] * X + c[k] * Y;
      X = nx;
    }
    x[i] = X;
  }
}

// The running row rp[j] (j = 0 .. n-1) goes through the K rotations, the k-th against row q0 + row[k] of the n-column matrix A
// (leading dimension ld): the element j of that row is A[j*ld + q0 + row[k]].
inline void rowchain(double* rp, double* A, int ld, int n, int q0, int K, const int* row, const double* c, const double* s) {
  const bool full = K == 4 && row[0] == 0 && row[1] == 1 && row[2] == 2 && row[3] == 3;
  int j = 0;
  if (full) {
    __m256d vc[4], vs[4];
    for (int k = 0; k < 4; ++k) {
      vc[k] = _mm256_set1_pd(c[k]);
      vs[k] = _mm256_set1_pd(s[k]);
    }
    for (; j + 4 <= n; j += 4) {
      double* const col = A + static_cast<size_t>(j) * ld + q0;
      __m256d W[4];
      for (int l = 0; l < 4; ++l) W[l] = _mm256_loadu_pd(col + static_cast<size_t>(l) * ld);
      tr4(W[0], W[1], W[2], W[3]);
      __m256d X = _mm256_loadu_pd(rp + j);
      for (int k = 0; k < 4; ++k) {
        const __m256d nx = _mm256_sub_pd(_mm256_mul_pd(vc[k], X), _mm256_mul_pd(vs[k], W[k]));
        W[k] = _mm256_add_pd(_mm256_mul_pd(vs[k], X), _mm256_mul_pd(vc[k], W[k]));
        X = nx;
      }
      _mm256_storeu_pd(rp + j, X);
      tr4(W[0], W[1], W[2], W[3]);
      for (int l = 0; l < 4; ++l) _mm256_storeu_pd(col + static_cast<size_t>(l) * ld, W[l]);
    }
  } else {
    for (; j + 4 <= n; j += 4) {
      double* const col = A + static_cast<size_t>(j) * ld + q0;
      alignas(32) __m256d W[4];
      for (int l = 0; l < 4; ++l) W[l] = _mm256_loadu_pd(col + static_cast<size_t>(l) * ld);
      tr4(W[0], W[1], W[2], W[3]);
      __m256d X = _mm256_loadu_pd(rp + j);
      for (int k = 0; k < K; ++k) {
        const __m256d vck = _mm256_set1_pd(c[k]), vsk = _mm256_set1_pd(s[k]);
        const __m256d nx = _mm256_sub_pd(_mm256_mul_pd(vck, X), _mm256_mul_pd(vsk, W[row[k]]));
        W[row[k]] = _mm256_add_pd(_mm256_mul_pd(vsk, X), _mm256_mul_pd(vck, W[row[k]]));
        X = nx;
      }
      _mm256_storeu_pd(rp + j, X);
      tr4(W[0], W[1], W[2], W[3]);
      for (int l = 0; l < 4; ++l) _mm256_storeu_pd(col + static_cast<size_t>(l) * ld, W[l]);
    }
  }
  for (; j < n; ++j) {
    double X = rp[j];
    double* const col = A + static_cast<size_t>(j) * ld + q0;
    for (int k = 0; k < K; ++k) {
      const double Y = col[row[k]];
      const double nx = c[k] * X - s[k] * Y;
      col[row[k]] = s[k] * X + c[k] * Y;
      X = nx;
    }
    rp[j] = X;
  }
}

}  // namespace avx2
#pragma GCC pop_options
#endif  // VGPU_HOSTGEMM_AVX2

// ---- the plain loops (the generic path, and cusolver_api.cpp's before this file) ----

// The symmetric eigenproblem: a is n x n column-major and symmetric, v is the identity; returns the sweeps made and the
// norm of what was left off the diagonal at the start of the last one.
inline int syev_generic(double* a, double* v, int n, double* residual, bool want_v = true) {
  auto A = [&](int r, int c) -> double& { return a[static_cast<size_t>(c) * n + r]; };
  auto V = [&](int r, int c) -> double& { return v[static_cast<size_t>(c) * n + r]; };
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
        const double t = (theta >= 0 ? 1.0 : -1.0) / (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
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
        if (want_v)
          for (int i = 0; i < n; ++i) {
            const double vip = V(i, p), viq = V(i, q);
            V(i, p) = c * vip - s * viq;
            V(i, q) = s * vip + c * viq;
          }
      }
  }
  return sweeps;
}

// The singular value problem of the m x n matrix a (column-major, m >= n or not): v is the n x n identity; returns the
// sweeps made, the residual after the last.
inline int svd_generic(double* a, double* v, int m, int n, double* residual, bool want_v = true) {
  auto A = [&](int r, int c) -> double& { return a[static_cast<size_t>(c) * m + r]; };
  auto V = [&](int r, int c) -> double& { return v[static_cast<size_t>(c) * n + r]; };
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
        const double t = (theta >= 0 ? 1.0 : -1.0) / (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
        const double c = 1.0 / std::sqrt(t * t + 1.0), sn = t * c;
        for (int i = 0; i < m; ++i) {
          const double aip = A(i, p), aiq = A(i, q);
          A(i, p) = c * aip - sn * aiq;
          A(i, q) = sn * aip + c * aiq;
        }
        if (want_v)
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

#ifdef VGPU_HOSTGEMM_AVX2
// One rotation of the two-sided method on the matrix as it lies in memory (the plain loops' body).
inline void syev_rotate_one(double* a, double* v, int n, int p, int q, bool want_v) {
  auto A = [&](int r, int c) -> double& { return a[static_cast<size_t>(c) * n + r]; };
  auto V = [&](int r, int c) -> double& { return v[static_cast<size_t>(c) * n + r]; };
  if (std::fabs(A(p, q)) < 1e-300) return;
  const double theta = (A(q, q) - A(p, p)) / (2.0 * A(p, q));
  const double t = (theta >= 0 ? 1.0 : -1.0) / (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
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
  if (want_v)
    for (int i = 0; i < n; ++i) {
      const double vip = V(i, p), viq = V(i, q);
      V(i, p) = c * vip - s * viq;
      V(i, q) = s * vip + c * viq;
    }
}

inline int syev_avx2(double* a, double* v, int n, double* residual, bool want_v) {
  auto A = [&](int r, int c) -> double& { return a[static_cast<size_t>(c) * n + r]; };
  auto V = [&](int r, int c) -> double& { return v[static_cast<size_t>(c) * n + r]; };
  int sweeps = 0;
  std::vector<double> rp(static_cast<size_t>(n > 0 ? n : 1));
  for (int sweep = 0; sweep < 100; ++sweep) {
    double off = 0.0;
    for (int p = 0; p < n; ++p)
      for (int q = p + 1; q < n; ++q) off += A(p, q) * A(p, q);
    *residual = std::sqrt(2 * off);
    if (off <= 1e-30) break;
    ++sweeps;
    for (int p = 0; p < n; ++p) {
      int q = p + 1;
      if (q + 4 <= n) {
        // The running row p lives in rp while the runs are done; memory's row p is stale until it is written back.
        for (int j = 0; j < n; ++j) rp[j] = A(p, j);
        for (; q + 4 <= n; q += 4) {
          const int idx[5] = {p, q, q + 1, q + 2, q + 3};
          double M[5][5];  // S x S, row 0 and column 0 being p
          for (int r = 0; r < 5; ++r)
            for (int cc = 0; cc < 5; ++cc) M[r][cc] = r == 0 ? rp[idx[cc]] : A(idx[r], idx[cc]);
          double c[4], s[4];
          int row[4], K = 0;
          for (int k = 0; k < 4; ++k) {
            const int ia = 1 + k;
            if (std::fabs(M[0][ia]) < 1e-300) continue;
            const double theta = (M[ia][ia] - M[0][0]) / (2.0 * M[0][ia]);
            const double t = (theta >= 0 ? 1.0 : -1.0) / (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
            const double ck = 1.0 / std::sqrt(t * t + 1.0), sk = t * ck;
            for (int r = 0; r < 5; ++r) {
              const double aip = M[r][0], aiq = M[r][ia];
              M[r][0] = ck * aip - sk * aiq;
              M[r][ia] = sk * aip + ck * aiq;
            }
            for (int cc = 0; cc < 5; ++cc) {
              const double api = M[0][cc], aqi = M[ia][cc];
              M[0][cc] = ck * api - sk * aqi;
              M[ia][cc] = sk * api + ck * aqi;
            }
            c[K] = ck;
            s[K] = sk;
            row[K] = k;
            ++K;
          }
          if (K == 0) continue;
          double* y[4];
          double* yv[4];
          for (int k = 0; k < K; ++k) {
            y[k] = &A(0, q + row[k]);
            if (want_v) yv[k] = &V(0, q + row[k]);
          }
          avx2::colchain(&A(0, p), y, K, n, c, s);
          avx2::rowchain(rp.data(), a, n, n, q, K, row, c, s);
          if (want_v) avx2::colchain(&V(0, p), yv, K, n, c, s);
          for (int r = 1; r < 5; ++r)
            for (int cc = 0; cc < 5; ++cc) A(idx[r], idx[cc]) = M[r][cc];
          for (int cc = 0; cc < 5; ++cc) rp[idx[cc]] = M[0][cc];
        }
        for (int j = 0; j < n; ++j) A(p, j) = rp[j];
      }
      for (; q < n; ++q) syev_rotate_one(a, v, n, p, q, want_v);
    }
  }
  return sweeps;
}

inline int svd_avx2(double* a, double* v, int m, int n, double* residual, bool want_v) {
  int sweeps = 0;
  *residual = 0;
  const size_t npairs = static_cast<size_t>(n) * (n > 0 ? n - 1 : 0) / 2;
  std::vector<double> offv(npairs);
  std::vector<unsigned char> used(npairs);
  std::vector<int> lp, lq;
  for (int sweep = 0; sweep < 60; ++sweep) {
    double worst = 0.0, off = 0.0;
    ++sweeps;
    std::fill(used.begin(), used.end(), static_cast<unsigned char>(0));
    for (int t = 1; t <= 2 * n - 3; ++t) {
      lp.clear();
      lq.clear();
      for (int p = std::max(0, t - (n - 1)); p < t - p; ++p) {
        lp.push_back(p);
        lq.push_back(t - p);
      }
      const int cnt = static_cast<int>(lp.size());
      for (int g0 = 0; g0 < cnt; g0 += 4) {
        const double* pp[4];
        const double* qq[4];
        for (int l = 0; l < 4; ++l) {
          const int k = std::min(g0 + l, cnt - 1);  // lanes past the end repeat the last pair and are ignored
          pp[l] = a + static_cast<size_t>(lp[k]) * m;
          qq[l] = a + static_cast<size_t>(lq[k]) * m;
        }
        double app[4], aqq[4], apq[4];
        avx2::dots4(pp, qq, m, app, aqq, apq);
        for (int l = 0; l < 4 && g0 + l < cnt; ++l) {
          const int p = lp[g0 + l], q = lq[g0 + l];
          if (app[l] == 0.0 || aqq[l] == 0.0) continue;
          worst = std::max(worst, std::fabs(apq[l]) / std::sqrt(app[l] * aqq[l]));
          const size_t id = static_cast<size_t>(p) * n - static_cast<size_t>(p) * (p + 1) / 2 + static_cast<size_t>(q - p - 1);
          offv[id] = apq[l] * apq[l];
          used[id] = 1;
          if (std::fabs(apq[l]) <= 1e-15 * std::sqrt(app[l] * aqq[l])) continue;
          const double theta = (aqq[l] - app[l]) / (2.0 * apq[l]);
          const double tt = (theta >= 0 ? 1.0 : -1.0) / (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
          const double c = 1.0 / std::sqrt(tt * tt + 1.0), sn = tt * c;
          avx2::rotate(a + static_cast<size_t>(p) * m, a + static_cast<size_t>(q) * m, m, c, sn);
          if (want_v) avx2::rotate(v + static_cast<size_t>(p) * n, v + static_cast<size_t>(q) * n, n, c, sn);
        }
      }
    }
    for (size_t k = 0; k < npairs; ++k)
      if (used[k]) off += offv[k];
    *residual = std::sqrt(2 * off);
    if (worst < 1e-14) break;
  }
  return sweeps;
}
#endif  // VGPU_HOSTGEMM_AVX2

// The sweeps themselves, on the host's best instruction set (see the top of this file). a, v and the return values as in
// syev_generic and svd_generic. With want_v false v is not touched (it may be null): the rotations do not depend on it.
inline int syev_sweeps(double* a, double* v, int n, double* residual, Isa isa = Isa::Auto, bool want_v = true) {
#ifdef VGPU_HOSTGEMM_AVX2
  if (isa == Isa::Avx2 || (isa == Isa::Auto && vgpu::hostgemm::avx2_available())) return syev_avx2(a, v, n, residual, want_v);
#endif
  (void)isa;
  return syev_generic(a, v, n, residual, want_v);
}

inline int svd_sweeps(double* a, double* v, int m, int n, double* residual, Isa isa = Isa::Auto, bool want_v = true) {
#ifdef VGPU_HOSTGEMM_AVX2
  if (isa == Isa::Avx2 || (isa == Isa::Auto && vgpu::hostgemm::avx2_available())) return svd_avx2(a, v, m, n, residual, want_v);
#endif
  (void)isa;
  return svd_generic(a, v, m, n, residual, want_v);
}

}  // namespace vgpu::hostjac
