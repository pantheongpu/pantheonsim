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
#include <limits>
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

// ---- eigenvalues and right eigenvectors of a general complex matrix ----
//
// LAPACK's zgeev without balancing: Householder reduction to Hessenberg form
// (zgehd2) with Q accumulated, the single-shift QR iteration of zlahqr --
// Wilkinson shifts, exceptional shifts every 10 iterations without
// deflation, Ahues-Kahan deflation -- to Schur form T = Q^H A Q, then each
// eigenvector of T by back-substitution (ztrevc) carried back by Q. The
// eigenvalues come out in the order zlahqr deflates them, which is LAPACK's.
// Each vector has unit 2-norm and its largest component real and positive.
// Returns 0, or i (1-based) when the QR iteration failed to converge, with
// eigenvalues i+1..n correct (zhseqr's INFO).

inline double cabs1(cd v) { return std::fabs(v.real()) + std::fabs(v.imag()); }

// zlarfg: H = I - tau v v^H with v = (1, x) maps (alpha, x) to (beta, 0), beta real.
inline cd larfg2(int n, cd& alpha, cd* x, size_t step) {
  double xnorm = 0;
  for (int i = 0; i < n - 1; ++i) xnorm = std::hypot(xnorm, std::abs(x[(size_t)i * step]));
  if (n <= 1 || (xnorm == 0.0 && alpha.imag() == 0.0)) return 0.0;
  const double ar = alpha.real(), ai = alpha.imag();
  const double mag = std::sqrt(ar * ar + ai * ai + xnorm * xnorm);
  const double beta = ar >= 0 ? -mag : mag;
  const cd tau((beta - ar) / beta, -ai / beta);
  const cd scal = 1.0 / (alpha - beta);
  for (int i = 0; i < n - 1; ++i) x[(size_t)i * step] *= scal;
  alpha = beta;
  return tau;
}

inline int geev(const CMat& in, int n, std::vector<cd>* w, CMat* vr) {
  w->assign((size_t)n, 0);
  if (vr) {
    vr->ld = std::max(n, 1);
    vr->v.assign((size_t)vr->ld * n, 0);
  }
  if (n == 0) return 0;
  CMat H{std::vector<cd>((size_t)n * n), n}, Z{std::vector<cd>((size_t)n * n, 0), n};
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) H(i, j) = in(i, j);
  for (int i = 0; i < n; ++i) Z(i, i) = 1;

  // zgehd2, accumulating Q = H(0) H(1) ... into Z from the right.
  for (int i = 0; i + 1 < n; ++i) {
    cd alpha = H(i + 1, i);
    const int len = n - i - 1;
    const cd tau = larfg2(len, alpha, len > 1 ? &H(i + 2, i) : nullptr, 1);
    std::vector<cd> v((size_t)len);
    v[0] = 1;
    for (int k = 1; k < len; ++k) v[(size_t)k] = H(i + 1 + k, i);
    if (tau != cd(0)) {
      for (int r = 0; r < n; ++r) {  // right: H(:, i+1:) -= tau (H v) v^H
        cd s = 0;
        for (int k = 0; k < len; ++k) s += H(r, i + 1 + k) * v[(size_t)k];
        s *= tau;
        for (int k = 0; k < len; ++k) H(r, i + 1 + k) -= s * std::conj(v[(size_t)k]);
      }
      for (int c = i + 1; c < n; ++c) {  // left: H(i+1:, :) -= conj(tau) v (v^H H)
        cd s = 0;
        for (int k = 0; k < len; ++k) s += std::conj(v[(size_t)k]) * H(i + 1 + k, c);
        s *= std::conj(tau);
        for (int k = 0; k < len; ++k) H(i + 1 + k, c) -= v[(size_t)k] * s;
      }
      for (int r = 0; r < n; ++r) {  // Z := Z H(i)
        cd s = 0;
        for (int k = 0; k < len; ++k) s += Z(r, i + 1 + k) * v[(size_t)k];
        s *= tau;
        for (int k = 0; k < len; ++k) Z(r, i + 1 + k) -= s * std::conj(v[(size_t)k]);
      }
    }
    H(i + 1, i) = alpha;
    for (int k = i + 2; k < n; ++k) H(k, i) = 0;
  }

  // zlahqr with wantt and wantz, 0-based (LAPACK's I is i + 1 here).
  const int ilo = 0, ihi = n - 1;
  for (int i = ilo + 1; i <= ihi; ++i) {  // make the subdiagonal real
    if (H(i, i - 1).imag() == 0.0) continue;
    cd sc = H(i, i - 1) / cabs1(H(i, i - 1));
    sc = std::conj(sc) / std::abs(sc);
    H(i, i - 1) = std::abs(H(i, i - 1));
    for (int j = i; j < n; ++j) H(i, j) *= sc;
    for (int j = 0; j <= std::min(ihi, i + 1); ++j) H(j, i) *= std::conj(sc);
    for (int j = 0; j < n; ++j) Z(j, i) *= std::conj(sc);
  }
  const double safmin = std::numeric_limits<double>::min();
  const double ulp = std::numeric_limits<double>::epsilon();
  const double smlnum = safmin * ((double)n / ulp);
  const int itmax = 30 * std::max(10, n), kexsh = 10;
  const double dat1 = 0.75;
  int kdefl = 0;
  int i = ihi;
  while (i >= ilo) {
    int l = ilo;
    bool converged = false;
    for (int its = 0; its <= itmax; ++its) {
      int k;
      for (k = i; k > l; --k) {
        if (cabs1(H(k, k - 1)) <= smlnum) break;
        double tst = cabs1(H(k - 1, k - 1)) + cabs1(H(k, k));
        if (tst == 0.0) {
          if (k - 2 >= ilo) tst += std::fabs(H(k - 1, k - 2).real());
          if (k + 1 <= ihi) tst += std::fabs(H(k + 1, k).real());
        }
        if (std::fabs(H(k, k - 1).real()) <= ulp * tst) {
          const double ab = std::max(cabs1(H(k, k - 1)), cabs1(H(k - 1, k)));
          const double ba = std::min(cabs1(H(k, k - 1)), cabs1(H(k - 1, k)));
          const double aa = std::max(cabs1(H(k, k)), cabs1(H(k - 1, k - 1) - H(k, k)));
          const double bb = std::min(cabs1(H(k, k)), cabs1(H(k - 1, k - 1) - H(k, k)));
          const double s = aa + ab;
          if (ba * (ab / s) <= std::max(smlnum, ulp * (bb * (aa / s)))) break;
        }
      }
      l = k;
      if (l > ilo) H(l, l - 1) = 0;
      if (l >= i) {
        converged = true;
        break;
      }
      ++kdefl;
      cd t;
      if (kdefl % (2 * kexsh) == 0) {
        t = dat1 * std::fabs(H(i, i - 1).real()) + H(i, i);
      } else if (kdefl % kexsh == 0) {
        t = dat1 * std::fabs(H(l + 1, l).real()) + H(l, l);
      } else {  // Wilkinson's shift
        t = H(i, i);
        const cd u = std::sqrt(H(i - 1, i)) * std::sqrt(H(i, i - 1));
        double s = cabs1(u);
        if (s != 0.0) {
          const cd x = 0.5 * (H(i - 1, i - 1) - t);
          const double sx = cabs1(x);
          s = std::max(s, cabs1(x));
          cd y = s * std::sqrt((x / s) * (x / s) + (u / s) * (u / s));
          if (sx > 0.0 && (x / sx).real() * y.real() + (x / sx).imag() * y.imag() < 0.0) y = -y;
          t -= u * (u / (x + y));
        }
      }
      // Two consecutive small subdiagonal elements?
      int m;
      cd v[2];
      for (m = i - 1; m > l; --m) {
        const cd h11 = H(m, m), h22 = H(m + 1, m + 1);
        cd h11s = h11 - t;
        double h21 = H(m + 1, m).real();
        const double s = cabs1(h11s) + std::fabs(h21);
        h11s /= s;
        h21 /= s;
        v[0] = h11s;
        v[1] = h21;
        const double h10 = H(m, m - 1).real();
        if (std::fabs(h10) * std::fabs(h21) <= ulp * (cabs1(h11s) * (cabs1(h11) + cabs1(h22)))) break;
      }
      if (m == l) {
        cd h11s = H(l, l) - t;
        double h21 = H(l + 1, l).real();
        const double s = cabs1(h11s) + std::fabs(h21);
        v[0] = h11s / s;
        v[1] = h21 / s;
      }
      // The single-shift QR step.
      for (int kk = m; kk <= i - 1; ++kk) {
        if (kk > m) {
          v[0] = H(kk, kk - 1);
          v[1] = H(kk + 1, kk - 1);
        }
        const cd t1 = larfg2(2, v[0], &v[1], 1);
        if (kk > m) {
          H(kk, kk - 1) = v[0];
          H(kk + 1, kk - 1) = 0;
        }
        const cd v2 = v[1];
        const double t2 = (t1 * v2).real();
        for (int j = kk; j < n; ++j) {
          const cd sum = std::conj(t1) * H(kk, j) + t2 * H(kk + 1, j);
          H(kk, j) -= sum;
          H(kk + 1, j) -= sum * v2;
        }
        for (int j = 0; j <= std::min(kk + 2, i); ++j) {
          const cd sum = t1 * H(j, kk) + t2 * H(j, kk + 1);
          H(j, kk) -= sum;
          H(j, kk + 1) -= sum * std::conj(v2);
        }
        for (int j = 0; j < n; ++j) {
          const cd sum = t1 * Z(j, kk) + t2 * Z(j, kk + 1);
          Z(j, kk) -= sum;
          Z(j, kk + 1) -= sum * std::conj(v2);
        }
        if (kk == m && m > l) {  // keep H(m, m-1) real
          cd temp = 1.0 - t1;
          temp /= std::abs(temp);
          H(m + 1, m) *= std::conj(temp);
          if (m + 2 <= i) H(m + 2, m + 1) *= temp;
          for (int j = m; j <= i; ++j) {
            if (j == m + 1) continue;
            for (int c = j + 1; c < n; ++c) H(j, c) *= temp;
            for (int r = 0; r < j; ++r) H(r, j) *= std::conj(temp);
            for (int r = 0; r < n; ++r) Z(r, j) *= std::conj(temp);
          }
        }
      }
      cd temp = H(i, i - 1);  // keep H(i, i-1) real
      if (temp.imag() != 0.0) {
        const double rtemp = std::abs(temp);
        H(i, i - 1) = rtemp;
        temp /= rtemp;
        for (int c = i + 1; c < n; ++c) H(i, c) *= std::conj(temp);
        for (int r = 0; r < i; ++r) H(r, i) *= temp;
        for (int r = 0; r < n; ++r) Z(r, i) *= temp;
      }
    }
    if (!converged) {
      for (int r = i + 1; r < n; ++r) (*w)[(size_t)r] = H(r, r);
      return i + 1;
    }
    (*w)[(size_t)i] = H(i, i);
    kdefl = 0;
    i = l - 1;
  }
  if (!vr) return 0;

  // Eigenvectors of T, then Q x, normalized as zgeev leaves them.
  for (int ki = n - 1; ki >= 0; --ki) {
    const cd lambda = H(ki, ki);
    const double smin = std::max(ulp * cabs1(lambda), smlnum);
    std::vector<cd> x((size_t)ki + 1, 0);
    x[(size_t)ki] = 1;
    for (int k = ki - 1; k >= 0; --k) {
      cd s = 0;
      for (int j = k + 1; j <= ki; ++j) s += H(k, j) * x[(size_t)j];
      cd d = H(k, k) - lambda;
      if (cabs1(d) < smin) d = smin;
      x[(size_t)k] = -s / d;
    }
    std::vector<cd> v((size_t)n, 0);
    for (int r = 0; r < n; ++r)
      for (int j = 0; j <= ki; ++j) v[(size_t)r] += Z(r, j) * x[(size_t)j];
    double nrm = 0;
    for (const cd& e : v) nrm = std::hypot(nrm, std::abs(e));
    int big = 0;
    for (int r = 1; r < n; ++r)
      if (std::norm(v[(size_t)r]) > std::norm(v[(size_t)big])) big = r;
    cd scale = nrm > 0 ? 1.0 / nrm : 1.0;
    if (std::abs(v[(size_t)big]) > 0) scale *= std::conj(v[(size_t)big]) / std::abs(v[(size_t)big]);
    for (int r = 0; r < n; ++r) (*vr)(r, ki) = v[(size_t)r] * scale;
    (*vr)(big, ki) = std::abs((*vr)(big, ki));
  }
  return 0;
}

}  // namespace vgpu_la
