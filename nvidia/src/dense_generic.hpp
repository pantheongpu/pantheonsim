// Dense LAPACK-style routines on the host, written once for real (double) and
// complex (std::complex<double>) elements, for the cuSOLVER entry points that
// return LAPACK's intermediate forms: the tridiagonal reduction (sytrd/hetrd)
// and its Q (orgtr/ungtr, ormtr/unmtr), the bidiagonal reduction (gebrd) and
// its Q and P^H (orgbr/ungbr), the triangular inverse (trtri), lauum, and the
// block reflector's triangular factor (larft).
//
// Each follows the unblocked LAPACK routine named beside it (dsytd2/zhetd2,
// dgebd2/zgebd2, dlarft, dtrti2, dlauu2), including LAPACK's Householder
// convention (dlarfg/zlarfg: beta = -sign(Re alpha) * norm, real), so the
// reflectors stored in A and tau are the ones LAPACK -- and cuSOLVER, which
// follows LAPACK -- stores. The blocked LAPACK routines compute the same
// reflectors to rounding.
#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>

namespace vgpu_dg {

using cd = std::complex<double>;

inline double cj(double x) { return x; }
inline cd cj(cd x) { return std::conj(x); }
inline double re(double x) { return x; }
inline double re(cd x) { return x.real(); }
inline double im(double) { return 0.0; }
inline double im(cd x) { return x.imag(); }
template <class V> V from_re(double x) { return V(x); }

// A column-major matrix with a leading dimension.
template <class V> struct DM {
  std::vector<V> v;
  int ld = 0;
  DM() = default;
  DM(int rows, int cols) : v((size_t)std::max(rows, 1) * std::max(cols, 0), V(0.0)), ld(std::max(rows, 1)) {}
  V& operator()(int r, int c) { return v[(size_t)c * ld + r]; }
  V operator()(int r, int c) const { return v[(size_t)c * ld + r]; }
};

// dlarfg/zlarfg: H = I - tau v v^H, v = (1, x'), maps (alpha, x) to (beta, 0)
// with beta real. x (n - 1 elements, stride step) is overwritten by v's tail.
template <class V> V larfg(int n, V& alpha, V* x, size_t step) {
  if (n <= 0) return V(0.0);
  double xnorm = 0;
  for (int i = 0; i < n - 1; ++i) xnorm = std::hypot(xnorm, std::abs(x[(size_t)i * step]));
  const double ar = re(alpha), ai = im(alpha);
  if (xnorm == 0.0 && ai == 0.0) return V(0.0);
  const double mag = std::sqrt(ar * ar + ai * ai + xnorm * xnorm);
  const double beta = ar >= 0 ? -mag : mag;
  V tau;
  if constexpr (std::is_same_v<V, double>) tau = (beta - ar) / beta;
  else tau = cd((beta - ar) / beta, -ai / beta);
  const V scal = V(1.0) / (alpha - V(beta));
  for (int i = 0; i < n - 1; ++i) x[(size_t)i * step] *= scal;
  alpha = V(beta);
  return tau;
}

// A Householder reflector over the full order: H = I - tau v v^H.
template <class V> struct Refl {
  std::vector<V> v;
  V tau;
};

// C := H C (left) or C H (right), over C's rows (left) or columns (right);
// `herm` applies H^H instead (conj(tau)).
template <class V> void apply(const Refl<V>& h, DM<V>& c, int rows, int cols, bool left, bool herm) {
  const V t = herm ? cj(h.tau) : h.tau;
  if (t == V(0.0)) return;
  if (left) {
    for (int j = 0; j < cols; ++j) {
      V s(0.0);
      for (int i = 0; i < rows; ++i) s += cj(h.v[(size_t)i]) * c(i, j);
      s *= t;
      if (s == V(0.0)) continue;
      for (int i = 0; i < rows; ++i) c(i, j) -= h.v[(size_t)i] * s;
    }
  } else {
    for (int i = 0; i < rows; ++i) {
      V s(0.0);
      for (int j = 0; j < cols; ++j) s += c(i, j) * h.v[(size_t)j];
      s *= t;
      if (s == V(0.0)) continue;
      for (int j = 0; j < cols; ++j) c(i, j) -= s * cj(h.v[(size_t)j]);
    }
  }
}

// Q = R[0] R[1] ... R[k-1]. C := op(Q) C (left) or C op(Q) (right), op N or ^H.
template <class V>
void apply_product(const std::vector<Refl<V>>& q, DM<V>& c, int rows, int cols, bool left, bool herm) {
  const int k = (int)q.size();
  // Left, N: R[k-1] first. Left, ^H: R[0]^H first. Right, N: R[0] first. Right, ^H: R[k-1]^H first.
  const bool forward = left == herm;
  for (int s = 0; s < k; ++s) apply(q[(size_t)(forward ? s : k - 1 - s)], c, rows, cols, left, herm);
}

// The order x order matrix the product forms.
template <class V> DM<V> form(const std::vector<Refl<V>>& q, int order) {
  DM<V> m(order, order);
  for (int i = 0; i < order; ++i) m(i, i) = V(1.0);
  apply_product(q, m, order, order, true, false);
  return m;
}

/* ---- tridiagonal reduction: dsytd2 / zhetd2 ---- */

// A is the full Hermitian matrix (both triangles valid); on return the stored
// triangle holds T and the reflectors as LAPACK leaves them (the other
// triangle is scratch). d (n), e (n-1) and tau (n-1).
template <class V>
void sytd2(bool upper, int n, DM<V>& a, std::vector<double>* d, std::vector<double>* e, std::vector<V>* tau) {
  d->assign((size_t)std::max(n, 0), 0.0);
  e->assign((size_t)std::max(n - 1, 0), 0.0);
  tau->assign((size_t)std::max(n - 1, 0), V(0.0));
  if (n <= 0) return;
  // hemv/her2 on the active block, kept full so both triangles agree.
  auto update = [&](int r0, int len, const std::vector<V>& v, V taui) {
    std::vector<V> w((size_t)len, V(0.0));
    for (int i = 0; i < len; ++i) {
      V s(0.0);
      for (int j = 0; j < len; ++j) s += a(r0 + i, r0 + j) * v[(size_t)j];
      w[(size_t)i] = taui * s;
    }
    V dot(0.0);
    for (int i = 0; i < len; ++i) dot += cj(w[(size_t)i]) * v[(size_t)i];
    const V alpha = V(-0.5) * taui * dot;
    for (int i = 0; i < len; ++i) w[(size_t)i] += alpha * v[(size_t)i];
    for (int j = 0; j < len; ++j)
      for (int i = 0; i < len; ++i)
        a(r0 + i, r0 + j) -= v[(size_t)i] * cj(w[(size_t)j]) + w[(size_t)i] * cj(v[(size_t)j]);
  };
  if (upper) {
    a(n - 1, n - 1) = V(re(a(n - 1, n - 1)));
    for (int i = n - 2; i >= 0; --i) {  // LAPACK's i = this + 1
      V alpha = a(i, i + 1);
      std::vector<V> col((size_t)i);
      for (int r = 0; r < i; ++r) col[(size_t)r] = a(r, i + 1);
      const V taui = larfg(i + 1, alpha, col.data(), 1);
      for (int r = 0; r < i; ++r) a(r, i + 1) = col[(size_t)r];
      (*e)[(size_t)i] = re(alpha);
      if (taui != V(0.0)) {
        std::vector<V> v((size_t)i + 1);
        for (int r = 0; r < i; ++r) v[(size_t)r] = a(r, i + 1);
        v[(size_t)i] = V(1.0);
        update(0, i + 1, v, taui);
      } else {
        a(i, i) = V(re(a(i, i)));
      }
      a(i, i + 1) = V((*e)[(size_t)i]);
      a(i + 1, i) = cj(a(i, i + 1));
      (*d)[(size_t)i + 1] = re(a(i + 1, i + 1));
      (*tau)[(size_t)i] = taui;
    }
    (*d)[0] = re(a(0, 0));
  } else {
    a(0, 0) = V(re(a(0, 0)));
    for (int i = 0; i < n - 1; ++i) {
      V alpha = a(i + 1, i);
      const int len = n - i - 1;
      std::vector<V> col((size_t)std::max(len - 1, 0));
      for (int r = 0; r < len - 1; ++r) col[(size_t)r] = a(i + 2 + r, i);
      const V taui = larfg(len, alpha, col.data(), 1);
      for (int r = 0; r < len - 1; ++r) a(i + 2 + r, i) = col[(size_t)r];
      (*e)[(size_t)i] = re(alpha);
      if (taui != V(0.0)) {
        std::vector<V> v((size_t)len);
        v[0] = V(1.0);
        for (int r = 1; r < len; ++r) v[(size_t)r] = a(i + 1 + r, i);
        update(i + 1, len, v, taui);
      } else {
        a(i + 1, i + 1) = V(re(a(i + 1, i + 1)));
      }
      a(i + 1, i) = V((*e)[(size_t)i]);
      a(i, i + 1) = cj(a(i + 1, i));
      (*d)[(size_t)i] = re(a(i, i));
      (*tau)[(size_t)i] = taui;
    }
    (*d)[(size_t)n - 1] = re(a(n - 1, n - 1));
  }
}

// The reflectors sytrd stored, as Q's product: lower Q = H(1) ... H(n-1),
// upper Q = H(n-1) ... H(1).
template <class V> std::vector<Refl<V>> tr_reflectors(bool upper, int n, const DM<V>& a, const V* tau) {
  std::vector<Refl<V>> q;
  if (upper) {
    for (int i = n - 2; i >= 0; --i) {  // H(i+1), v(i+1) = 1 at row i, v(1:i) above it in column i + 1
      Refl<V> h{std::vector<V>((size_t)n, V(0.0)), tau[i]};
      for (int r = 0; r < i; ++r) h.v[(size_t)r] = a(r, i + 1);
      h.v[(size_t)i] = V(1.0);
      q.push_back(std::move(h));
    }
  } else {
    for (int i = 0; i < n - 1; ++i) {
      Refl<V> h{std::vector<V>((size_t)n, V(0.0)), tau[i]};
      h.v[(size_t)i + 1] = V(1.0);
      for (int r = i + 2; r < n; ++r) h.v[(size_t)r] = a(r, i);
      q.push_back(std::move(h));
    }
  }
  return q;
}

/* ---- bidiagonal reduction: dgebd2 / zgebd2 ---- */

// Conjugates a row segment (zlacgv); nothing for real.
template <class V> void lacgv(DM<V>& a, int r, int c0, int c1) {
  if constexpr (!std::is_same_v<V, double>)
    for (int c = c0; c < c1; ++c) a(r, c) = std::conj(a(r, c));
}

template <class V>
void gebd2(int m, int n, DM<V>& a, std::vector<double>* d, std::vector<double>* e, std::vector<V>* tauq,
           std::vector<V>* taup) {
  const int k = std::min(m, n);
  d->assign((size_t)k, 0.0);
  e->assign((size_t)std::max(k - 1, 0), 0.0);
  tauq->assign((size_t)k, V(0.0));
  taup->assign((size_t)k, V(0.0));
  // H from the left on rows r0.. of columns c0..; G from the right on rows r0.. of columns c0...
  auto left = [&](const std::vector<V>& v, V t, int r0, int c0) {
    for (int c = c0; c < n; ++c) {
      V s(0.0);
      for (int i = 0; i < (int)v.size(); ++i) s += cj(v[(size_t)i]) * a(r0 + i, c);
      s *= t;
      for (int i = 0; i < (int)v.size(); ++i) a(r0 + i, c) -= v[(size_t)i] * s;
    }
  };
  auto right = [&](const std::vector<V>& v, V t, int r0, int c0) {
    for (int r = r0; r < m; ++r) {
      V s(0.0);
      for (int j = 0; j < (int)v.size(); ++j) s += a(r, c0 + j) * v[(size_t)j];
      s *= t;
      for (int j = 0; j < (int)v.size(); ++j) a(r, c0 + j) -= s * cj(v[(size_t)j]);
    }
  };
  if (m >= n) {
    for (int i = 0; i < n; ++i) {
      V alpha = a(i, i);
      const int len = m - i;
      std::vector<V> x((size_t)std::max(len - 1, 0));
      for (int r = 0; r < len - 1; ++r) x[(size_t)r] = a(i + 1 + r, i);
      const V tq = larfg(len, alpha, x.data(), 1);
      for (int r = 0; r < len - 1; ++r) a(i + 1 + r, i) = x[(size_t)r];
      (*tauq)[(size_t)i] = tq;
      (*d)[(size_t)i] = re(alpha);
      if (i < n - 1) {
        std::vector<V> v((size_t)len);
        v[0] = V(1.0);
        for (int r = 1; r < len; ++r) v[(size_t)r] = a(i + r, i);
        left(v, cj(tq), i, i + 1);
      }
      a(i, i) = V((*d)[(size_t)i]);
      if (i < n - 1) {
        lacgv(a, i, i + 1, n);
        V beta = a(i, i + 1);
        const int l2 = n - i - 1;
        std::vector<V> y((size_t)std::max(l2 - 1, 0));
        for (int c = 0; c < l2 - 1; ++c) y[(size_t)c] = a(i, i + 2 + c);
        const V tp = larfg(l2, beta, y.data(), 1);
        for (int c = 0; c < l2 - 1; ++c) a(i, i + 2 + c) = y[(size_t)c];
        (*taup)[(size_t)i] = tp;
        (*e)[(size_t)i] = re(beta);
        std::vector<V> u((size_t)l2);
        u[0] = V(1.0);
        for (int c = 1; c < l2; ++c) u[(size_t)c] = a(i, i + 1 + c);
        right(u, tp, i + 1, i + 1);
        lacgv(a, i, i + 1, n);
        a(i, i + 1) = V((*e)[(size_t)i]);
      } else {
        (*taup)[(size_t)i] = V(0.0);
      }
    }
  } else {
    for (int i = 0; i < m; ++i) {
      lacgv(a, i, i, n);
      V alpha = a(i, i);
      const int len = n - i;
      std::vector<V> y((size_t)std::max(len - 1, 0));
      for (int c = 0; c < len - 1; ++c) y[(size_t)c] = a(i, i + 1 + c);
      const V tp = larfg(len, alpha, y.data(), 1);
      for (int c = 0; c < len - 1; ++c) a(i, i + 1 + c) = y[(size_t)c];
      (*taup)[(size_t)i] = tp;
      (*d)[(size_t)i] = re(alpha);
      if (i < m - 1) {
        std::vector<V> u((size_t)len);
        u[0] = V(1.0);
        for (int c = 1; c < len; ++c) u[(size_t)c] = a(i, i + c);
        right(u, tp, i + 1, i);
      }
      lacgv(a, i, i, n);
      a(i, i) = V((*d)[(size_t)i]);
      if (i < m - 1) {
        V beta = a(i + 1, i);
        const int l2 = m - i - 1;
        std::vector<V> x((size_t)std::max(l2 - 1, 0));
        for (int r = 0; r < l2 - 1; ++r) x[(size_t)r] = a(i + 2 + r, i);
        const V tq = larfg(l2, beta, x.data(), 1);
        for (int r = 0; r < l2 - 1; ++r) a(i + 2 + r, i) = x[(size_t)r];
        (*tauq)[(size_t)i] = tq;
        (*e)[(size_t)i] = re(beta);
        std::vector<V> v((size_t)l2);
        v[0] = V(1.0);
        for (int r = 1; r < l2; ++r) v[(size_t)r] = a(i + 1 + r, i);
        left(v, cj(tq), i + 1, i + 1);
        a(i + 1, i) = V((*e)[(size_t)i]);
      } else {
        (*tauq)[(size_t)i] = V(0.0);
      }
    }
  }
}

// orgbr's Q: from gebrd of an order x k matrix (reflectors in A's columns).
// order >= k: Q = H(1) ... H(k), v(i) = 1; order < k: H(1) ... H(order-1), v(i+1) = 1.
template <class V> std::vector<Refl<V>> bd_q(int order, int k, const DM<V>& a, const V* tau) {
  std::vector<Refl<V>> q;
  const bool shift = order < k;
  const int count = shift ? order - 1 : k;
  for (int i = 0; i < count; ++i) {
    Refl<V> h{std::vector<V>((size_t)order, V(0.0)), tau[i]};
    const int one = shift ? i + 1 : i;
    h.v[(size_t)one] = V(1.0);
    for (int r = one + 1; r < order; ++r) h.v[(size_t)r] = a(r, i);
    q.push_back(std::move(h));
  }
  return q;
}
// orgbr's P: from gebrd of a k x order matrix (reflectors, conjugated, in A's
// rows). k < order: P = G(1) ... G(k), u(i) = 1; k >= order: G(1) ... G(order-1), u(i+1) = 1.
template <class V> std::vector<Refl<V>> bd_p(int order, int k, const DM<V>& a, const V* tau) {
  std::vector<Refl<V>> q;
  const bool shift = k >= order;
  const int count = shift ? order - 1 : k;
  for (int i = 0; i < count; ++i) {
    Refl<V> h{std::vector<V>((size_t)order, V(0.0)), tau[i]};
    const int one = shift ? i + 1 : i;
    h.v[(size_t)one] = V(1.0);
    for (int c = one + 1; c < order; ++c) h.v[(size_t)c] = cj(a(i, c));
    q.push_back(std::move(h));
  }
  return q;
}

/* ---- triangular inverse (dtrti2), lauum (dlauu2) ---- */

// Returns 0, or the 1-based index of a zero diagonal (A untouched then); with
// `check` false a zero diagonal is inverted anyway, to infinities.
template <class V> int trtri(bool upper, bool unit, int n, DM<V>& a, bool check = true) {
  if (!unit && check)
    for (int j = 0; j < n; ++j)
      if (a(j, j) == V(0.0)) return j + 1;
  if (upper) {
    for (int j = 0; j < n; ++j) {
      V ajj;
      if (!unit) {
        a(j, j) = V(1.0) / a(j, j);
        ajj = -a(j, j);
      } else {
        ajj = V(-1.0);
      }
      // x := triu(A(0:j, 0:j)) x, x = A(0:j, j)
      std::vector<V> x((size_t)j);
      for (int i = 0; i < j; ++i) x[(size_t)i] = a(i, j);
      for (int i = 0; i < j; ++i) {
        V s = unit ? x[(size_t)i] : a(i, i) * x[(size_t)i];
        for (int k = i + 1; k < j; ++k) s += a(i, k) * x[(size_t)k];
        a(i, j) = s * ajj;
      }
    }
  } else {
    for (int j = n - 1; j >= 0; --j) {
      V ajj;
      if (!unit) {
        a(j, j) = V(1.0) / a(j, j);
        ajj = -a(j, j);
      } else {
        ajj = V(-1.0);
      }
      std::vector<V> x((size_t)n);
      for (int i = j + 1; i < n; ++i) x[(size_t)i] = a(i, j);
      for (int i = j + 1; i < n; ++i) {
        V s = unit ? x[(size_t)i] : a(i, i) * x[(size_t)i];
        for (int k = j + 1; k < i; ++k) s += a(i, k) * x[(size_t)k];
        a(i, j) = s * ajj;
      }
    }
  }
  return 0;
}

// The stored triangle becomes U U^H (upper) or L^H L (lower).
template <class V> void lauum(bool upper, int n, DM<V>& a) {
  DM<V> out(n, n);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i <= j; ++i) {  // (i, j) of the upper product, or (j, i) of the lower
      V s(0.0);
      if (upper)
        for (int k = j; k < n; ++k) s += a(i, k) * cj(a(j, k));
      else
        for (int k = j; k < n; ++k) s += cj(a(k, i)) * a(k, j);
      if (upper) out(i, j) = s;
      else out(j, i) = cj(s);
    }
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i)
      if (upper ? i <= j : i >= j) a(i, j) = i == j ? V(re(out(i, j))) : out(i, j);
}

/* ---- the block reflector's triangular factor: dlarft, column-wise ---- */

// V is n x k (column i holds v_i: forward, v_i(i) = 1 above zeros; backward,
// v_i(n-k+i) = 1 below zeros). T is k x k: upper for forward, lower for backward.
template <class V> DM<V> larft(bool forward, int n, int k, const DM<V>& vm, const V* tau) {
  DM<V> t(k, k);
  auto vv = [&](int r, int i) -> V {  // the implicit unit and zeros
    if (forward) return r < i ? V(0.0) : r == i ? V(1.0) : vm(r, i);
    const int one = n - k + i;
    return r > one ? V(0.0) : r == one ? V(1.0) : vm(r, i);
  };
  if (forward) {
    for (int i = 0; i < k; ++i) {
      if (tau[i] == V(0.0)) {
        for (int j = 0; j <= i; ++j) t(j, i) = V(0.0);
        continue;
      }
      for (int j = 0; j < i; ++j) {
        V s(0.0);
        for (int r = i; r < n; ++r) s += cj(vv(r, j)) * vv(r, i);
        t(j, i) = -tau[i] * s;
      }
      std::vector<V> x((size_t)i);
      for (int j = 0; j < i; ++j) {
        V s(0.0);
        for (int l = j; l < i; ++l) s += t(j, l) * t(l, i);
        x[(size_t)j] = s;
      }
      for (int j = 0; j < i; ++j) t(j, i) = x[(size_t)j];
      t(i, i) = tau[i];
    }
  } else {
    for (int i = k - 1; i >= 0; --i) {
      if (tau[i] == V(0.0)) {
        for (int j = i; j < k; ++j) t(j, i) = V(0.0);
        continue;
      }
      for (int j = i + 1; j < k; ++j) {
        V s(0.0);
        for (int r = 0; r <= n - k + i; ++r) s += cj(vv(r, j)) * vv(r, i);
        t(j, i) = -tau[i] * s;
      }
      std::vector<V> x((size_t)k);
      for (int j = i + 1; j < k; ++j) {
        V s(0.0);
        for (int l = i + 1; l <= j; ++l) s += t(j, l) * t(l, i);
        x[(size_t)j] = s;
      }
      for (int j = i + 1; j < k; ++j) t(j, i) = x[(size_t)j];
      t(i, i) = tau[i];
    }
  }
  return t;
}

}  // namespace vgpu_dg
