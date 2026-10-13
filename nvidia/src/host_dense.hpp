#pragma once
// The dense factorizations and triangular solves behind cuSOLVER's host routines (LU, Cholesky, QR and the solves
// that use them), in double, with the loop orders changed and the arithmetic not.
//
// Every element is the end of a chain of  v = v - a*b  (or  v + a*b) steps in a fixed order of the inner index; the
// original loops ran those chains one element at a time with the inner index along a row of a column-major matrix.
// Here the loops are arranged so the inner loop runs down a column (independent elements, contiguous memory, so
// the compiler vectorises it) or advances several columns' chains together; each element still sees its steps in
// the same order, each product rounded and each sum rounded (no fused multiply-add), so every result is bit for bit
// what the loops gave. tests/unit/test_host_dense.cpp keeps them as the reference.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include "host_trsm.hpp"

namespace vgpu::hostdense {

// Column-major access.
struct M {
  double* p;
  int64_t ld;
  double& operator()(int64_t r, int64_t c) const { return p[c * ld + r]; }
  double* col(int64_t c) const { return p + c * ld; }
};
struct CM {
  const double* p;
  int64_t ld;
  double operator()(int64_t r, int64_t c) const { return p[c * ld + r]; }
  const double* col(int64_t c) const { return p + c * ld; }
};

// y[i] = y[i] - a * x[i]
inline void axpy_sub(double* __restrict y, const double* __restrict x, double a, int64_t n) {
  for (int64_t i = 0; i < n; ++i) y[i] = y[i] - a * x[i];
}

// Cholesky. Returns 0, or the 1-based index of the leading minor that failed.
inline int potrf(double* a, int64_t ld, int n, bool lower) {
  M A{a, ld};
  std::vector<double> s;
  for (int j = 0; j < n; ++j) {
    double d = A(j, j);
    for (int k = 0; k < j; ++k) {
      const double v = lower ? A(j, k) : A(k, j);
      d -= v * v;
    }
    if (!(d > 0.0)) return j + 1;
    d = std::sqrt(d);
    A(j, j) = d;
    if (lower) {
      // s_i = A(i,j) - sum_k A(i,k) A(j,k), the k-th step for every i at once (A(.,k) is a column).
      s.assign(A.col(j) + j + 1, A.col(j) + n);
      for (int k = 0; k < j; ++k) axpy_sub(s.data(), A.col(k) + j + 1, A(j, k), n - j - 1);
      for (int i = j + 1; i < n; ++i) A(i, j) = s[static_cast<size_t>(i - j - 1)] / d;
    } else {
      // The chain of A(j,i) runs down column i of A (A(k,i)) against column j: contiguous in k. Four columns at a time.
      const double* cj = A.col(j);
      int i = j + 1;
      for (; i + 4 <= n; i += 4) {
        double s0 = A(j, i), s1 = A(j, i + 1), s2 = A(j, i + 2), s3 = A(j, i + 3);
        const double *c0 = A.col(i), *c1 = A.col(i + 1), *c2 = A.col(i + 2), *c3 = A.col(i + 3);
        for (int k = 0; k < j; ++k) {
          s0 -= c0[k] * cj[k];
          s1 -= c1[k] * cj[k];
          s2 -= c2[k] * cj[k];
          s3 -= c3[k] * cj[k];
        }
        A(j, i) = s0 / d;
        A(j, i + 1) = s1 / d;
        A(j, i + 2) = s2 / d;
        A(j, i + 3) = s3 / d;
      }
      for (; i < n; ++i) {
        double s0 = A(j, i);
        const double* c0 = A.col(i);
        for (int k = 0; k < j; ++k) s0 -= c0[k] * cj[k];
        A(j, i) = s0 / d;
      }
    }
  }
  return 0;
}

// Steps j = start .. min(m,n)-1 of LU, one column at a time over the whole matrix (the loops this file began with).
// Returns the 1-based index of the first zero pivot (0 if none); a zero pivot skips its step.
inline int getrf_steps(double* a, int64_t ld, int m, int n, int* ipiv, int start) {
  M A{a, ld};
  const int k = std::min(m, n);
  int info = 0;
  for (int j = start; j < k; ++j) {
    int p = j;
    if (ipiv) {
      double best = std::fabs(A(j, j));
      for (int i = j + 1; i < m; ++i)
        if (std::fabs(A(i, j)) > best) { best = std::fabs(A(i, j)); p = i; }
      ipiv[j] = p + 1;
      if (p != j)
        for (int c = 0; c < n; ++c) std::swap(A(j, c), A(p, c));
    }
    const double piv = A(j, j);
    if (piv == 0.0) { if (!info) info = j + 1; continue; }
    double* fj = A.col(j);
    for (int i = j + 1; i < m; ++i) fj[i] /= piv;
    // A(i,c) -= f_i * A(j,c): one column at a time, down the column.
    for (int c = j + 1; c < n; ++c) axpy_sub(A.col(c) + j + 1, fj + j + 1, A(j,c), m - j - 1);
  }
  return info;
}

// LU with partial pivoting. ipiv (null: no pivoting) is LAPACK's: 1-based row that row j was swapped with.
//
// Done a panel of columns at a time, right-looking. Every element still sees  v = v - l(i,t) * u(t,c)  for t = 0, 1, 2, ...
// in that order, each product and each subtraction rounded: the panel's own columns are factored one step after another
// as above; the rows of the panel's pivots are then exchanged in the other columns (which moves rows of values, and the
// step that updates a row's content is the same wherever it sits); the panel's rows of U in the columns to the right are
// L11's unit triangular solve (the chain of each element, with the columns along the vector lanes), and the rest of those
// columns is a matrix product whose sums start from the old values (host_gemm.hpp). A zero pivot skips its step
// in the loops above, which no update of this form can say, so the factorization then goes on, unblocked, from the start of
// that panel (nothing to its right has been touched yet).
inline int getrf(double* a, int64_t ld, int m, int n, int* ipiv) {
  const int k = std::min(m, n);
  constexpr int kPanel = 64;
  if (k < 2 * kPanel || static_cast<int64_t>(m) * n < (1 << 14)) return getrf_steps(a, ld, m, n, ipiv, 0);
  M A{a, ld};
  std::vector<double> saved;
  std::vector<int> saved_piv;
  for (int c0 = 0; c0 < k; c0 += kPanel) {
    const int c1 = std::min(k, c0 + kPanel), nb = c1 - c0;
    saved.resize(static_cast<size_t>(m) * nb);
    for (int c = 0; c < nb; ++c)
      std::memcpy(saved.data() + static_cast<size_t>(c) * m, A.col(c0 + c), sizeof(double) * static_cast<size_t>(m));
    bool zero = false;
    for (int j = c0; j < c1; ++j) {
      int p = j;
      if (ipiv) {
        double best = std::fabs(A(j, j));
        for (int i = j + 1; i < m; ++i)
          if (std::fabs(A(i, j)) > best) { best = std::fabs(A(i, j)); p = i; }
        ipiv[j] = p + 1;
        if (p != j)
          for (int c = c0; c < c1; ++c) std::swap(A(j, c), A(p, c));
      }
      const double piv = A(j, j);
      if (piv == 0.0) { zero = true; break; }
      double* fj = A.col(j);
      for (int i = j + 1; i < m; ++i) fj[i] /= piv;
      for (int c = j + 1; c < c1; ++c) axpy_sub(A.col(c) + j + 1, fj + j + 1, A(j, c), m - j - 1);
    }
    if (zero) {
      for (int c = 0; c < nb; ++c)
        std::memcpy(A.col(c0 + c), saved.data() + static_cast<size_t>(c) * m, sizeof(double) * static_cast<size_t>(m));
      return getrf_steps(a, ld, m, n, ipiv, c0);
    }
    if (ipiv)   // this panel's exchanges, in order, in the columns outside it
      for (int j = c0; j < c1; ++j) {
        const int p = ipiv[j] - 1;
        if (p == j) continue;
        for (int c = 0; c < c0; ++c) std::swap(A(j, c), A(p, c));
        for (int c = c1; c < n; ++c) std::swap(A(j, c), A(p, c));
      }
    if (c1 < n) {
      // U12 = L11^-1 A12: x_j = A(j,c) - sum_{t<j} L(j,t) x_t, the columns c as lanes.
      vgpu::hosttrsm::real(nb, n - c1, /*forward=*/true, /*unit=*/true, &A(c0, c0), 1, ld, &A(c0, c1), 1, ld);
      // A22 = A22 - L21 U12, each element's sums starting from its old value.
      if (c1 < m) {
        double* const c22 = &A(c1, c1);
        vgpu::hostgemm::gemm<double, double>(
            false, false, m - c1, n - c1, nb, &A(c1, c0), ld, &A(c0, c1), ld,
            [=](int64_t i, int64_t j, double acc) { c22[i + j * ld] = acc; }, vgpu::hostgemm::Isa::Auto, 0, 0, c22, ld, true);
      }
    }
  }
  return 0;
}

inline void apply_pivots(double* b, int64_t ldb, int n, int nrhs, const int* ipiv, int k, bool forward) {
  M B{b, ldb};
  for (int s = 0; s < k; ++s) {
    const int i = forward ? s : k - 1 - s;
    const int p = ipiv[i] - 1;
    if (p == i || p < 0 || p >= n) continue;
    for (int c = 0; c < nrhs; ++c) std::swap(B(i, c), B(p, c));
  }
}

// Triangular solves. `unit` means an implicit unit diagonal (LU's L).
inline void trsm_lower(const double* a, int64_t lda, double* b, int64_t ldb, int n, int nrhs, bool unit, bool trans) {
  CM A{a, lda};
  M B{b, ldb};
  for (int c = 0; c < nrhs; ++c) {
    double* x = B.col(c);
    if (!trans) {
      // x_i = (b_i - sum_{k<i} L(i,k) x_k) / L(i,i): once x_k is final, take its term off every later element (column k of L).
      for (int k = 0; k < n; ++k) {
        x[k] = unit ? x[k] : x[k] / A(k, k);
        axpy_sub(x + k + 1, A.col(k) + k + 1, x[k], n - k - 1);
      }
    } else {
      for (int i = n - 1; i >= 0; --i) {
        double s = x[i];
        const double* li = A.col(i);
        for (int k = i + 1; k < n; ++k) s -= li[k] * x[k];
        x[i] = unit ? s : s / A(i, i);
      }
    }
  }
}

inline void trsm_upper(const double* a, int64_t lda, double* b, int64_t ldb, int n, int nrhs, bool unit, bool trans) {
  CM A{a, lda};
  M B{b, ldb};
  // Not transposed, x_i = (b_i - sum_{k>i} U(i,k) x_k) / U(i,i) takes U along a row, which is strided; with several
  // right-hand sides the rows are copied once into columns (T(k,i) = U(i,k)) so each dot product reads contiguous memory.
  std::vector<double> t;
  if (!trans && nrhs >= 4 && n >= 16) {
    t.assign(static_cast<size_t>(n) * n, 0.0);
    for (int i = 0; i < n; ++i)
      for (int k = i + 1; k < n; ++k) t[static_cast<size_t>(i) * n + k] = A(i, k);
  }
  for (int c = 0; c < nrhs; ++c) {
    double* x = B.col(c);
    if (!trans) {
      for (int i = n - 1; i >= 0; --i) {
        double s = x[i];
        if (!t.empty()) {
          const double* ui = t.data() + static_cast<size_t>(i) * n;
          for (int k = i + 1; k < n; ++k) s -= ui[k] * x[k];
        } else {
          for (int k = i + 1; k < n; ++k) s -= A(i, k) * x[k];
        }
        x[i] = unit ? s : s / A(i, i);
      }
    } else {
      for (int i = 0; i < n; ++i) {
        double s = x[i];
        const double* ci = A.col(i);
        for (int k = 0; k < i; ++k) s -= ci[k] * x[k];
        x[i] = unit ? s : s / A(i, i);
      }
    }
  }
}

// Householder QR, LAPACK's dgeqrf/dlarfg convention: beta carries the opposite sign to the leading element, v[0] is an
// implicit 1, and the rest of v overwrites the column below the diagonal.
//
// Applying a reflector to a column is  dot = sum_{i>=j} v_i A(i,c)  (a chain, i ascending),  dot *= t,
// A(i,c) -= dot * v_i. Columns are independent, so four of them run together: four chains in flight instead of one.
inline void apply_reflector_cols(const double* v, int j, int m, double t, M C, int c0, int c1) {
  int c = c0;
  for (; c + 4 <= c1; c += 4) {
    double d0 = 0.0, d1 = 0.0, d2 = 0.0, d3 = 0.0;
    const double *x0 = C.col(c), *x1 = C.col(c + 1), *x2 = C.col(c + 2), *x3 = C.col(c + 3);
    for (int i = j; i < m; ++i) {
      d0 += v[i] * x0[i];
      d1 += v[i] * x1[i];
      d2 += v[i] * x2[i];
      d3 += v[i] * x3[i];
    }
    d0 *= t, d1 *= t, d2 *= t, d3 *= t;
    double *y0 = C.col(c), *y1 = C.col(c + 1), *y2 = C.col(c + 2), *y3 = C.col(c + 3);
    for (int i = j; i < m; ++i) {
      y0[i] -= d0 * v[i];
      y1[i] -= d1 * v[i];
      y2[i] -= d2 * v[i];
      y3[i] -= d3 * v[i];
    }
  }
  for (; c < c1; ++c) {
    double dot = 0.0;
    double* x = C.col(c);
    for (int i = j; i < m; ++i) dot += v[i] * x[i];
    dot *= t;
    for (int i = j; i < m; ++i) x[i] -= dot * v[i];
  }
}

// The reflectors are generated a panel of columns at a time; the columns beyond the panel then take the panel's reflectors
// in order, each group of them held as vector lanes while all of the panel's reflectors pass over it (host_trsm.hpp,
// reflect()): every column still gets reflector 0, 1, 2, ... in turn, each as a dot product, a scale and an update with
// the same steps as before. The columns inside the panel are updated as soon as a reflector exists, as they always were.
inline void geqrf(double* a, int64_t ld, int m, int n, double* tau) {
  M A{a, ld};
  const int k = std::min(m, n);
  for (int j = 0; j < k; ++j) tau[j] = 0.0;
  std::vector<double> v(static_cast<size_t>(std::max(m, 1)));
  constexpr int kPanel = 64;
  for (int c0 = 0; c0 < k; c0 += kPanel) {
    const int c1 = std::min(k, c0 + kPanel);
    for (int j = c0; j < c1; ++j) {
      double norm = 0.0;
      for (int i = j; i < m; ++i) norm += A(i, j) * A(i, j);
      norm = std::sqrt(norm);
      if (norm == 0.0) { tau[j] = 0.0; continue; }
      const double alpha = A(j, j);
      const double beta = alpha >= 0.0 ? -norm : norm;
      const double t = (beta - alpha) / beta;
      tau[j] = t;
      const double scale = alpha - beta;
      v[static_cast<size_t>(j)] = 1.0;
      for (int i = j + 1; i < m; ++i) { A(i, j) /= scale; v[static_cast<size_t>(i)] = A(i, j); }
      A(j, j) = beta;
      apply_reflector_cols(v.data(), j, m, t, A, j + 1, c1);
    }
    if (c1 >= n) continue;
    if (static_cast<uint64_t>(m - c0) * static_cast<uint64_t>(n - c1) * static_cast<uint64_t>(c1 - c0) >= (1u << 15)) {
      vgpu::hosttrsm::reflect(a, ld, tau, m, c0, c1 - 1, 1, a, ld, c1, n, /*skip_zero_tau=*/true);
    } else {
      for (int j = c0; j < c1; ++j) {
        if (tau[j] == 0.0) continue;
        v[static_cast<size_t>(j)] = 1.0;
        for (int i = j + 1; i < m; ++i) v[static_cast<size_t>(i)] = A(i, j);
        apply_reflector_cols(v.data(), j, m, tau[j], A, c1, n);
      }
    }
  }
}

// Overwrite the first n columns of A with the corresponding columns of Q.
inline void orgqr(double* a, int64_t ld, int m, int n, int k, const double* tau) {
  M A{a, ld};
  std::vector<double> q(static_cast<size_t>(m) * n, 0.0);
  M Q{q.data(), m};
  for (int c = 0; c < n; ++c) Q(c, c) = 1.0;
  if (k > 0 && static_cast<uint64_t>(m) * static_cast<uint64_t>(n) * static_cast<uint64_t>(k) >= (1u << 15)) {
    vgpu::hosttrsm::reflect(a, ld, tau, m, k - 1, 0, -1, q.data(), m, 0, n, /*skip_zero_tau=*/false);
  } else {
    std::vector<double> v(static_cast<size_t>(std::max(m, 1)));
    for (int j = k - 1; j >= 0; --j) {
      v[static_cast<size_t>(j)] = 1.0;
      for (int i = j + 1; i < m; ++i) v[static_cast<size_t>(i)] = A(i, j);
      apply_reflector_cols(v.data(), j, m, tau[j], Q, 0, n);
    }
  }
  for (int c = 0; c < n; ++c)
    for (int r = 0; r < m; ++r) A(r, c) = Q(r, c);
}

// C := op(Q) C  (left) or C op(Q)  (right), Q given by the reflectors in A.
inline void ormqr(const double* a, int64_t lda, double* c, int64_t ldc, int m, int n, int k, const double* tau, bool left,
                  bool trans) {
  CM A{a, lda};
  M C{c, ldc};
  const int rows = left ? m : n;  // length of each reflector's active span
  if (left && k > 0 && static_cast<uint64_t>(m) * static_cast<uint64_t>(n) * static_cast<uint64_t>(k) >= (1u << 15)) {
    // Every column of C takes the reflectors in turn (all of them for a group of columns, held as lanes).
    const bool backward = left != trans;
    vgpu::hosttrsm::reflect(a, lda, tau, m, backward ? k - 1 : 0, backward ? 0 : k - 1, backward ? -1 : 1, c, ldc, 0, n,
                            /*skip_zero_tau=*/false);
    return;
  }
  std::vector<double> v(static_cast<size_t>(std::max(rows, 1)));
  // Q = H(0) H(1) ... H(k-1); Q^T applies them in order, Q in reverse.
  for (int s = 0; s < k; ++s) {
    const int j = (left != trans) ? k - 1 - s : s;
    for (int i = 0; i < rows; ++i) v[static_cast<size_t>(i)] = i < j ? 0.0 : (i == j ? 1.0 : A(i, j));
    if (left) {
      apply_reflector_cols(v.data(), j, m, tau[j], C, 0, n);
    } else {
      // Row r of C: dot = sum_{i>=j} v_i C(r,i), then C(r,i) -= dot * v_i. Down a column of C the rows are independent, so a
      // block of rows advances together.
      for (int r0 = 0; r0 < m; r0 += 64) {
        const int r1 = std::min(m, r0 + 64);
        double dot[64] = {};
        for (int i = j; i < n; ++i) {
          const double vi = v[static_cast<size_t>(i)];
          const double* ci = C.col(i);
          for (int r = r0; r < r1; ++r) dot[r - r0] += vi * ci[r];
        }
        for (int r = r0; r < r1; ++r) dot[r - r0] *= tau[j];
        for (int i = j; i < n; ++i) {
          const double vi = v[static_cast<size_t>(i)];
          double* ci = C.col(i);
          for (int r = r0; r < r1; ++r) ci[r] -= dot[r - r0] * vi;
        }
      }
    }
  }
}

}  // namespace vgpu::hostdense
