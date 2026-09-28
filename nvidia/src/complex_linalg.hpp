// Dense complex linear algebra on the host, for the complex (C and Z) entry
// points of VirtualGPU's cuBLAS and cuSOLVER.
//
// Everything is computed in std::complex<double> and column-major, and follows
// LAPACK's conventions wherever one library's output is another's input: a
// Householder reflector is H = I - tau v v^H with v(0) = 1 and a real beta
// (zlarfg), Q = H(1) H(2) ... H(k), and geqrf applies H^H to the trailing
// matrix. PyTorch hands geqrf's output to ungqr/unmqr, so those three have to
// agree with each other and with LAPACK.
//
// The eigen- and singular-value solvers are Jacobi methods: slower than
// LAPACK's, exact to rounding, and short enough to check by eye.
#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <numeric>
#include <vector>

namespace vgpu_la {

using cd = std::complex<double>;

// A column-major matrix with a leading dimension.
struct CMat {
  std::vector<cd> v;
  int ld = 0;
  cd& operator()(int r, int c) { return v[(size_t)c * ld + r]; }
  cd operator()(int r, int c) const { return v[(size_t)c * ld + r]; }
};

enum class Op { N, T, C };
inline cd apply_op(cd x, Op op) { return op == Op::C ? std::conj(x) : x; }

// ---- LU with partial pivoting ----

// Factors the m x n matrix in place (L unit lower, U upper). ipiv gets 1-based
// row interchanges, min(m, n) of them. Returns 0 or the 1-based index of the
// first exactly-zero pivot, after which the factorization still completes, as
// LAPACK's does.
inline int lu(CMat& a, int m, int n, std::vector<int>* ipiv) {
  const int k = std::min(m, n);
  int info = 0;
  if (ipiv) ipiv->assign(k, 0);
  for (int j = 0; j < k; ++j) {
    int p = j;
    if (ipiv) {
      for (int i = j + 1; i < m; ++i)
        if (std::abs(a(i, j)) > std::abs(a(p, j))) p = i;
      (*ipiv)[j] = p + 1;
      if (p != j)
        for (int c = 0; c < n; ++c) std::swap(a(j, c), a(p, c));
    }
    if (a(j, j) == cd(0)) {
      if (!info) info = j + 1;
      continue;
    }
    for (int i = j + 1; i < m; ++i) {
      a(i, j) /= a(j, j);
      for (int c = j + 1; c < n; ++c) a(i, c) -= a(i, j) * a(j, c);
    }
  }
  return info;
}

// op(A) X = B with A's LU factors (n x n) and pivots; X over B.
inline void lu_solve(const CMat& a, int n, const int* ipiv, Op op, CMat& b, int nrhs) {
  for (int k = 0; k < nrhs; ++k) {
    auto B = [&](int r) -> cd& { return b(r, k); };
    if (op == Op::N) {
      if (ipiv)
        for (int i = 0; i < n; ++i)
          if (ipiv[i] - 1 != i) std::swap(B(i), B(ipiv[i] - 1));
      for (int i = 0; i < n; ++i)
        for (int j = 0; j < i; ++j) B(i) -= a(i, j) * B(j);
      for (int i = n - 1; i >= 0; --i) {
        for (int j = i + 1; j < n; ++j) B(i) -= a(i, j) * B(j);
        B(i) /= a(i, i);
      }
    } else {  // op(A) = A^T or A^H: U^op then L^op, then the pivots backwards
      for (int i = 0; i < n; ++i) {
        for (int j = 0; j < i; ++j) B(i) -= apply_op(a(j, i), op) * B(j);
        B(i) /= apply_op(a(i, i), op);
      }
      for (int i = n - 1; i >= 0; --i)
        for (int j = i + 1; j < n; ++j) B(i) -= apply_op(a(j, i), op) * B(j);
      if (ipiv)
        for (int i = n - 1; i >= 0; --i)
          if (ipiv[i] - 1 != i) std::swap(B(i), B(ipiv[i] - 1));
    }
  }
}

// ---- triangular solves ----

// Solves M x = x in place, M = op(A) of A's k x k `lower` (or upper) triangle.
// `step` is the distance between x's elements.
inline void tri_solve(const CMat& a, int k, bool lower, Op op, bool unit, cd* x, size_t step) {
  auto M = [&](int i, int j) { return op == Op::N ? a(i, j) : apply_op(a(j, i), op); };
  auto row = [&](int i, int from, int to) {
    cd s = x[(size_t)i * step];
    for (int j = from; j < to; ++j) s -= M(i, j) * x[(size_t)j * step];
    x[(size_t)i * step] = unit ? s : s / M(i, i);
  };
  if (lower == (op == Op::N))
    for (int i = 0; i < k; ++i) row(i, 0, i);  // M is lower: forward
  else
    for (int i = k - 1; i >= 0; --i) row(i, i + 1, k);  // M is upper: back
}

// op(A) X = alpha B (left) or X op(A) = alpha B (right); X over the m x n B.
inline void trsm(bool left, bool lower, Op op, bool unit, int m, int n, cd alpha, const CMat& a, CMat& b) {
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < m; ++i) b(i, j) *= alpha;
  if (left) {
    for (int j = 0; j < n; ++j) tri_solve(a, m, lower, op, unit, &b(0, j), 1);
  } else {
    // X op(A) = B is op(A)^T X^T = B^T: each row of B, against the transpose
    // of op(A). The transpose of A^H is conj(A), which is no Op: conj(A) y = b
    // is A conj(y) = conj(b), so conjugate the row, solve with A, and
    // conjugate back.
    for (int i = 0; i < m; ++i) {
      cd* row = &b(i, 0);
      const size_t step = (size_t)b.ld;
      if (op == Op::C) {
        for (int j = 0; j < n; ++j) row[j * step] = std::conj(row[j * step]);
        tri_solve(a, n, lower, Op::N, unit, row, step);
        for (int j = 0; j < n; ++j) row[j * step] = std::conj(row[j * step]);
      } else {
        tri_solve(a, n, lower, op == Op::N ? Op::T : Op::N, unit, row, step);
      }
    }
  }
}

// ---- Cholesky of a Hermitian positive definite matrix ----

// A = L L^H (lower) or U^H U (upper), in place. Returns 0 or the 1-based order
// of the first minor that is not positive definite.
inline int cholesky(CMat& a, int n, bool lower) {
  for (int j = 0; j < n; ++j) {
    double d = a(j, j).real();
    for (int k = 0; k < j; ++k) d -= std::norm(lower ? a(j, k) : a(k, j));
    if (!(d > 0.0)) return j + 1;
    d = std::sqrt(d);
    a(j, j) = d;
    for (int i = j + 1; i < n; ++i) {
      cd s = lower ? a(i, j) : a(j, i);
      for (int k = 0; k < j; ++k)
        s -= lower ? a(i, k) * std::conj(a(j, k)) : std::conj(a(k, j)) * a(k, i);
      if (lower) a(i, j) = s / d;
      else a(j, i) = s / d;
    }
  }
  return 0;
}

// Solves A X = B with A's Cholesky factor.
inline void cholesky_solve(const CMat& a, int n, bool lower, CMat& b, int nrhs) {
  for (int k = 0; k < nrhs; ++k) {
    if (lower) {  // L y = b, then L^H x = y
      tri_solve(a, n, true, Op::N, false, &b(0, k), 1);
      tri_solve(a, n, true, Op::C, false, &b(0, k), 1);
    } else {  // U^H y = b, then U x = y
      tri_solve(a, n, false, Op::C, false, &b(0, k), 1);
      tri_solve(a, n, false, Op::N, false, &b(0, k), 1);
    }
  }
}

// ---- Householder QR ----

// zlarfg: for x = (alpha, rest), finds tau, v (v(0) = 1) and a real beta with
// H^H x = (beta, 0, ..., 0), H = I - tau v v^H. rest is overwritten with v(1:).
inline cd larfg(cd& alpha, cd* rest, int len, size_t step) {
  double xnorm = 0;
  for (int i = 0; i < len; ++i) xnorm += std::norm(rest[i * step]);
  xnorm = std::sqrt(xnorm);
  if (xnorm == 0 && alpha.imag() == 0) return 0;
  const double beta = -std::copysign(std::sqrt(std::norm(alpha) + xnorm * xnorm), alpha.real());
  const cd tau = (beta - alpha) / beta;
  const cd scale = 1.0 / (alpha - beta);
  for (int i = 0; i < len; ++i) rest[i * step] *= scale;
  alpha = beta;
  return tau;
}

// A = QR in place: R on and above the diagonal, the reflectors below it.
inline void geqrf(CMat& a, int m, int n, std::vector<cd>* tau) {
  const int k = std::min(m, n);
  tau->assign(k, 0);
  for (int j = 0; j < k; ++j) {
    const cd t = larfg(a(j, j), m - j - 1 > 0 ? &a(j + 1, j) : nullptr, m - j - 1, 1);
    (*tau)[j] = t;
    if (t == cd(0)) continue;
    // Apply H^H = I - conj(tau) v v^H to the trailing columns.
    const cd ajj = a(j, j);
    a(j, j) = 1;
    for (int c = j + 1; c < n; ++c) {
      cd w = 0;
      for (int i = j; i < m; ++i) w += std::conj(a(i, j)) * a(i, c);
      w *= std::conj(t);
      for (int i = j; i < m; ++i) a(i, c) -= a(i, j) * w;
    }
    a(j, j) = ajj;
  }
}

// The first n columns of Q = H(1) ... H(k), from geqrf's reflectors, in place
// of the m x n matrix that holds them.
inline void ungqr(CMat& a, int m, int n, int k, const cd* tau) {
  std::vector<cd> v(m);
  CMat q{std::vector<cd>((size_t)a.ld * n, 0), a.ld};
  for (int j = 0; j < n; ++j) q(j, j) = 1;
  for (int i = k - 1; i >= 0; --i) {
    v.assign(m, 0);
    v[i] = 1;
    for (int r = i + 1; r < m; ++r) v[r] = a(r, i);
    // q = H(i) q = q - tau v (v^H q)
    for (int c = 0; c < n; ++c) {
      cd w = 0;
      for (int r = i; r < m; ++r) w += std::conj(v[r]) * q(r, c);
      w *= tau[i];
      for (int r = i; r < m; ++r) q(r, c) -= v[r] * w;
    }
  }
  for (int c = 0; c < n; ++c)
    for (int r = 0; r < m; ++r) a(r, c) = q(r, c);
}

// C = op(Q) C (left) or C op(Q) (right), Q from k reflectors stored in the
// columns of a (nq rows: m for left, n for right). op is N or C.
inline void unmqr(bool left, Op op, int m, int n, int k, const CMat& a, const cd* tau, CMat& c) {
  const int nq = left ? m : n;
  // Q = H(1)...H(k). Q C applies H(k) first; Q^H C applies H(1)^H first.
  // C Q applies H(1) first; C Q^H applies H(k)^H first.
  const bool forward = left == (op == Op::C);
  for (int step = 0; step < k; ++step) {
    const int i = forward ? step : k - 1 - step;
    std::vector<cd> v(nq, 0);
    v[i] = 1;
    for (int r = i + 1; r < nq; ++r) v[r] = a(r, i);
    const cd t = op == Op::C ? std::conj(tau[i]) : tau[i];
    if (left) {  // C = (I - t v v^H) C
      for (int col = 0; col < n; ++col) {
        cd w = 0;
        for (int r = i; r < m; ++r) w += std::conj(v[r]) * c(r, col);
        w *= t;
        for (int r = i; r < m; ++r) c(r, col) -= v[r] * w;
      }
    } else {  // C = C (I - t v v^H)
      for (int row = 0; row < m; ++row) {
        cd w = 0;
        for (int r = i; r < n; ++r) w += c(row, r) * v[r];
        w *= t;
        for (int r = i; r < n; ++r) c(row, r) -= w * std::conj(v[r]);
      }
    }
  }
}

// ---- Hermitian eigendecomposition: cyclic complex Jacobi ----

// A (n x n, Hermitian; only the `lower` or upper triangle is read) = V diag(w) V^H.
// w ascending; V's columns unit and orthogonal.
inline void heev(const CMat& in, int n, bool lower, std::vector<double>* w, CMat* vout) {
  CMat a{std::vector<cd>((size_t)n * n), n};
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
      const bool stored = lower ? i >= j : i <= j;
      a(i, j) = stored ? in(i, j) : std::conj(in(j, i));
    }
  for (int i = 0; i < n; ++i) a(i, i) = a(i, i).real();
  CMat v{std::vector<cd>((size_t)n * n, 0), n};
  for (int i = 0; i < n; ++i) v(i, i) = 1;
  for (int sweep = 0; sweep < 100; ++sweep) {
    double off = 0, tot = 0;
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < n; ++i) (i == j ? tot : off) += std::norm(a(i, j));
    if (off <= 1e-30 * (tot + off) || off == 0) break;
    for (int p = 0; p < n - 1; ++p)
      for (int q = p + 1; q < n; ++q) {
        const cd apq = a(p, q);
        const double mag = std::abs(apq);
        if (mag == 0) continue;
        // Rotate the 2x2 Hermitian block [app apq; conj(apq) aqq] to diagonal.
        const cd phase = apq / mag;
        const double app = a(p, p).real(), aqq = a(q, q).real();
        const double theta = 0.5 * std::atan2(2 * mag, aqq - app);
        const double c = std::cos(theta), s = std::sin(theta);
        // Column p' = c p - s conj(phase) q ; column q' = s phase p + c q.
        for (int k = 0; k < n; ++k) {
          const cd akp = a(k, p), akq = a(k, q);
          a(k, p) = c * akp - s * std::conj(phase) * akq;
          a(k, q) = s * phase * akp + c * akq;
        }
        for (int k = 0; k < n; ++k) {
          const cd apk = a(p, k), aqk = a(q, k);
          a(p, k) = c * apk - s * phase * aqk;
          a(q, k) = s * std::conj(phase) * apk + c * aqk;
        }
        a(p, q) = a(q, p) = 0;
        a(p, p) = a(p, p).real();
        a(q, q) = a(q, q).real();
        for (int k = 0; k < n; ++k) {
          const cd vkp = v(k, p), vkq = v(k, q);
          v(k, p) = c * vkp - s * std::conj(phase) * vkq;
          v(k, q) = s * phase * vkp + c * vkq;
        }
      }
  }
  std::vector<int> order(n);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](int x, int y) { return a(x, x).real() < a(y, y).real(); });
  w->resize(n);
  vout->ld = n;
  vout->v.assign((size_t)n * n, 0);
  for (int j = 0; j < n; ++j) {
    (*w)[j] = a(order[j], order[j]).real();
    for (int i = 0; i < n; ++i) (*vout)(i, j) = v(i, order[j]);
  }
}

// ---- SVD: one-sided Jacobi ----

// Extends the first k orthonormal columns of the m x m Q to a full unitary
// basis, by Gram-Schmidt against the standard basis.
inline void complete_basis(CMat& q, int m, int k) {
  int filled = k;
  for (int e = 0; e < m && filled < m; ++e) {
    std::vector<cd> x(m, 0);
    x[e] = 1;
    for (int pass = 0; pass < 2; ++pass)
      for (int c = 0; c < filled; ++c) {
        cd d = 0;
        for (int r = 0; r < m; ++r) d += std::conj(q(r, c)) * x[r];
        for (int r = 0; r < m; ++r) x[r] -= d * q(r, c);
      }
    double nrm = 0;
    for (const cd& z : x) nrm += std::norm(z);
    nrm = std::sqrt(nrm);
    if (nrm < 1e-8) continue;
    for (int r = 0; r < m; ++r) q(r, filled) = x[r] / nrm;
    ++filled;
  }
}

// A (m x n) = U diag(s) V^H, s descending, k = min(m, n) triplets. With `full`,
// U is m x m and V n x n; otherwise U is m x k and V n x k.
inline void svd(const CMat& in, int m, int n, bool full, std::vector<double>* s, CMat* u, CMat* v) {
  // Work on the tall orientation: a wide A is handled as A^H = V S U^H.
  const bool wide = m < n;
  const int rows = wide ? n : m, cols = wide ? m : n;
  CMat a{std::vector<cd>((size_t)rows * cols), rows};
  for (int j = 0; j < cols; ++j)
    for (int i = 0; i < rows; ++i) a(i, j) = wide ? std::conj(in(j, i)) : in(i, j);
  CMat w{std::vector<cd>((size_t)cols * cols, 0), cols};
  for (int i = 0; i < cols; ++i) w(i, i) = 1;
  for (int sweep = 0; sweep < 100; ++sweep) {
    bool rotated = false;
    for (int p = 0; p < cols - 1; ++p)
      for (int q = p + 1; q < cols; ++q) {
        double alpha = 0, beta = 0;
        cd gamma = 0;
        for (int i = 0; i < rows; ++i) {
          alpha += std::norm(a(i, p));
          beta += std::norm(a(i, q));
          gamma += std::conj(a(i, p)) * a(i, q);
        }
        const double g = std::abs(gamma);
        if (g == 0 || g <= 1e-15 * std::sqrt(alpha * beta)) continue;
        rotated = true;
        const cd phase = gamma / g;
        const double zeta = (beta - alpha) / (2 * g);
        const double t = std::copysign(1.0, zeta) / (std::fabs(zeta) + std::sqrt(1 + zeta * zeta));
        const double c = 1 / std::sqrt(1 + t * t), sn = c * t;
        auto rot = [&](CMat& x, int nr) {
          for (int i = 0; i < nr; ++i) {
            const cd xp = x(i, p), xq = x(i, q);
            x(i, p) = c * xp - sn * std::conj(phase) * xq;
            x(i, q) = sn * phase * xp + c * xq;
          }
        };
        rot(a, rows);
        rot(w, cols);
      }
    if (!rotated) break;
  }
  std::vector<double> norms(cols);
  for (int j = 0; j < cols; ++j) {
    double t = 0;
    for (int i = 0; i < rows; ++i) t += std::norm(a(i, j));
    norms[j] = std::sqrt(t);
  }
  std::vector<int> order(cols);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](int x, int y) { return norms[x] > norms[y]; });
  const int k = cols;
  // Left vectors of the tall matrix: its columns scaled to unit length.
  CMat lu_{std::vector<cd>((size_t)rows * (full ? rows : k), 0), rows};
  CMat rv{std::vector<cd>((size_t)cols * cols, 0), cols};
  s->resize(k);
  int good = 0;
  for (int j = 0; j < k; ++j) {
    const int src = order[j];
    (*s)[j] = norms[src];
    for (int i = 0; i < cols; ++i) rv(i, j) = w(i, src);
    if (norms[src] > 1e-300) {
      for (int i = 0; i < rows; ++i) lu_(i, j) = a(i, src) / norms[src];
      ++good;
    }
  }
  // Columns for zero singular values (and a full U's extra ones) complete the
  // basis; they are orthogonal to the rest, as a unitary U needs.
  if (good < (full ? rows : k)) {
    CMat tmp{std::vector<cd>((size_t)rows * rows, 0), rows};
    for (int j = 0; j < good; ++j)
      for (int i = 0; i < rows; ++i) tmp(i, j) = lu_(i, j);
    complete_basis(tmp, rows, good);
    for (int j = good; j < (full ? rows : k); ++j)
      for (int i = 0; i < rows; ++i) lu_(i, j) = tmp(i, j);
  }
  // Back to A's orientation: for a wide A the roles of U and V swap.
  CMat& left = wide ? rv : lu_;
  CMat& right = wide ? lu_ : rv;
  const int ucols = full ? m : k, vcols = full ? n : k;
  u->ld = m;
  u->v.assign((size_t)m * ucols, 0);
  v->ld = n;
  v->v.assign((size_t)n * vcols, 0);
  if (!wide) {
    for (int j = 0; j < ucols; ++j)
      for (int i = 0; i < m; ++i) (*u)(i, j) = left(i, j);
    for (int j = 0; j < std::min(vcols, cols); ++j)
      for (int i = 0; i < n; ++i) (*v)(i, j) = right(i, j);
  } else {
    // A^H = L S R^H (L tall, n x ...; R m x m), so A = R S L^H: U = R, V = L.
    for (int j = 0; j < std::min(ucols, cols); ++j)
      for (int i = 0; i < m; ++i) (*u)(i, j) = left(i, j);
    for (int j = 0; j < vcols; ++j)
      for (int i = 0; i < n; ++i) (*v)(i, j) = right(i, j);
  }
}

}  // namespace vgpu_la
