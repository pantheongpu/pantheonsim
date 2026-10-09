// cuSOLVER's iterative refinement solvers: <t1><t2>gesv and <t1><t2>gels,
// IRSXgesv and IRSXgels with their params and infos. Included at the end of
// cusolver_api.cpp, after cusolver_dense_more.inc, whose helpers it uses.
//
// The method is LAPACK's dsgesv: factor A in the lower precision, solve, and
// refine in the main precision until every right-hand side has
//   ||r||_inf < sqrt(n) ||x||_inf ||A||_inf eps          (eps = <t1>LAMCH('E'))
// or 50 iterations have run; then (fallback enabled) factor A in the main
// precision and solve again, A overwritten by those factors. gels does the
// same with Householder QR and least-squares solves.
//
// What an RTX 3060 (CUDA 13.0) showed, and this follows:
//   - every lower precision below single (half, bfloat16, TF32) behaves as
//     single: DH/DB/DX give DSgesv's iterations, solution and pivots to the
//     bit, and SH/SB/SX are SSgesv. So the factorization here is single (or
//     the main precision, when <t2> = <t1>), computed in single arithmetic;
//   - the iteration counts follow: niters 0 when the first solve already
//     meets the criterion (DD, SS, ZZ, CC on a well-conditioned system), 2 for
//     DS on one; A and B unchanged and ipiv the factorization's pivots when it
//     converges;
//   - an inconsistent least-squares system never meets the residual test:
//     niters = -50 and the fallback factors A (in place) in the main precision;
//   - an exactly singular A gives niters = -5 (the single solve overflows),
//     the fallback's factors in A, and dinfo 0;
//   - bad arguments: IRS_PARAMS_INVALID with dinfo -i (lda -4, n -1; gels'
//     m < n -2), INVALID_WORKSPACE for too small a workspace, and n = 0
//     returns at once, niters and dinfo untouched;
//   - IRSXgesv refuses unset precisions (IRS_PARAMS_INVALID_PREC) and an unset
//     refinement solver (IRS_PARAMS_INVALID_REFINE), each with dinfo -9; the
//     GMRES refinements take one right-hand side only (IRS_NOT_SUPPORTED, and
//     IRS_NRHS_NOT_SUPPORTED_FOR_REFINE_GMRES from _bufferSize). The GMRES
//     refinements are carried out here as the classical one, which reaches
//     the same solution, sometimes in a different number of iterations.

namespace {

template <class V> struct LowOf;
template <> struct LowOf<double> { using type = float; };
template <> struct LowOf<cd> { using type = std::complex<float>; };

template <class W> W cjw(W x) {
  if constexpr (std::is_floating_point_v<W>) return x;
  else return std::conj(x);
}
template <class W> W w_of(std::complex<double> x) {
  if constexpr (std::is_floating_point_v<W>) return (W)x.real();
  else return W((typename W::value_type)x.real(), (typename W::value_type)x.imag());
}
template <class V> double eps_of(bool single) { return single ? std::ldexp(1.0, -24) : std::ldexp(1.0, -53); }

// Rounds to the main precision's storage (float when the I/O type is single).
template <class V> V round_main(V v, bool single) {
  if (!single) return v;
  if constexpr (std::is_same_v<V, double>) return (double)(float)v;
  else return cd((float)v.real(), (float)v.imag());
}

// LU with partial pivoting (izamax's |re| + |im|) in the arithmetic of W,
// on an n x n column-major matrix. Returns LAPACK's info.
template <class W> int lu_in(std::vector<W>& a, int n, std::vector<int>* piv) {
  auto mag = [](W v) { return std::fabs((double)std::real(v)) + std::fabs((double)std::imag(v)); };
  int info = 0;
  piv->assign((size_t)n, 0);
  for (int j = 0; j < n; ++j) {
    int p = j;
    for (int i = j + 1; i < n; ++i)
      if (mag(a[(size_t)j * n + i]) > mag(a[(size_t)j * n + p])) p = i;
    (*piv)[(size_t)j] = p + 1;
    if (p != j)
      for (int c = 0; c < n; ++c) std::swap(a[(size_t)c * n + j], a[(size_t)c * n + p]);
    const W d = a[(size_t)j * n + j];
    if (d == W(0)) {
      if (!info) info = j + 1;
      continue;
    }
    for (int i = j + 1; i < n; ++i) a[(size_t)j * n + i] /= d;
    for (int c = j + 1; c < n; ++c) {
      const W u = a[(size_t)c * n + j];
      for (int i = j + 1; i < n; ++i) a[(size_t)c * n + i] -= a[(size_t)j * n + i] * u;
    }
  }
  return info;
}
template <class W> void lu_solve_in(const std::vector<W>& a, int n, const std::vector<int>& piv, std::vector<W>& x) {
  for (int i = 0; i < n; ++i)
    if (piv[(size_t)i] - 1 != i) std::swap(x[(size_t)i], x[(size_t)piv[(size_t)i] - 1]);
  for (int i = 0; i < n; ++i)
    for (int k = 0; k < i; ++k) x[(size_t)i] -= a[(size_t)k * n + i] * x[(size_t)k];
  for (int i = n - 1; i >= 0; --i) {
    for (int k = i + 1; k < n; ++k) x[(size_t)i] -= a[(size_t)k * n + i] * x[(size_t)k];
    x[(size_t)i] /= a[(size_t)i * n + i];
  }
}

// Householder QR (LAPACK's convention) of an m x n column-major matrix in the
// arithmetic of W, and the least-squares solve x = R^-1 (Q^H b)(0:n).
template <class W> void qr_in(std::vector<W>& a, int m, int n, std::vector<W>* tau) {
  tau->assign((size_t)n, W(0));
  for (int j = 0; j < n; ++j) {
    double xn = 0;
    for (int i = j + 1; i < m; ++i) xn += std::norm(std::complex<double>(a[(size_t)j * m + i]));
    const std::complex<double> alpha(a[(size_t)j * m + j]);
    if (xn == 0 && alpha.imag() == 0) continue;
    const double mag = std::sqrt(std::norm(alpha) + xn);
    const double beta = alpha.real() >= 0 ? -mag : mag;
    std::complex<double> t((beta - alpha.real()) / beta, -alpha.imag() / beta);
    const std::complex<double> scal = 1.0 / (alpha - beta);
    for (int i = j + 1; i < m; ++i) a[(size_t)j * m + i] = w_of<W>(std::complex<double>(a[(size_t)j * m + i]) * scal);
    a[(size_t)j * m + j] = W(beta);
    const W tw = w_of<W>(t);
    (*tau)[(size_t)j] = tw;
    for (int c = j + 1; c < n; ++c) {  // apply H^H = I - conj(tau) v v^H
      W s = a[(size_t)c * m + j];
      for (int i = j + 1; i < m; ++i) s += cjw(a[(size_t)j * m + i]) * a[(size_t)c * m + i];
      s *= cjw(tw);
      a[(size_t)c * m + j] -= s;
      for (int i = j + 1; i < m; ++i) a[(size_t)c * m + i] -= a[(size_t)j * m + i] * s;
    }
  }
}
template <class W> void qr_solve_in(const std::vector<W>& a, int m, int n, const std::vector<W>& tau, std::vector<W>& b) {
  for (int j = 0; j < n; ++j) {  // b := Q^H b
    W s = b[(size_t)j];
    for (int i = j + 1; i < m; ++i) s += cjw(a[(size_t)j * m + i]) * b[(size_t)i];
    s *= cjw(tau[(size_t)j]);
    b[(size_t)j] -= s;
    for (int i = j + 1; i < m; ++i) b[(size_t)i] -= a[(size_t)j * m + i] * s;
  }
  for (int i = n - 1; i >= 0; --i) {
    for (int k = i + 1; k < n; ++k) b[(size_t)i] -= a[(size_t)k * m + i] * b[(size_t)k];
    b[(size_t)i] /= a[(size_t)i * m + i];
  }
  b.resize((size_t)n);
}

template <class W, class V> W to_w(V v) {
  if constexpr (std::is_floating_point_v<W>) return (W)std::real(v);
  else return W((typename W::value_type)std::real(v), (typename W::value_type)std::imag(v));
}
template <class V, class W> V from_w(W w) {
  if constexpr (std::is_same_v<V, double>) return (double)std::real(w);
  else return cd(std::real(w), std::imag(w));
}
double inf_norm_vec(const std::vector<double>& v) {
  double m = 0;
  for (double x : v) m = std::max(m, std::fabs(x));
  return m;
}

struct IrsOut {
  int niters = 0, outer = 0, info = 0;
  std::vector<double> history;  // (iterations so far, residual norm) per outer step
};

// The refinement itself, on host copies: A (m x n, ld = m), B (m x nrhs), X
// (n x nrhs). `ls` selects least squares (QR) over LU. `low` the single
// factorization (else the main precision's). `refine` false: one solve, no
// refinement. Returns whether A was overwritten (fallback, or the
// main-precision factorization with no refinement).
template <class V, class W>
bool irs_run(bool ls, bool single_main, bool refine, bool fallback, int maxiters, double tol, std::vector<V>& a, int m,
             int n, const std::vector<V>& b, int nrhs, std::vector<V>* x, std::vector<int>* ipiv, IrsOut* out) {
  out->history.clear();
  // ||A||_inf
  double anrm = 0;
  for (int i = 0; i < m; ++i) {
    double s = 0;
    for (int j = 0; j < n; ++j) s += std::abs(a[(size_t)j * m + i]);
    anrm = std::max(anrm, s);
  }
  const double cte = tol > 0 ? tol : anrm * eps_of<V>(single_main) * std::sqrt((double)n);
  std::vector<W> f((size_t)m * n);
  for (size_t i = 0; i < f.size(); ++i) f[i] = to_w<W>(a[i]);
  std::vector<int> piv;
  std::vector<W> tau;
  if (ls) qr_in(f, m, n, &tau);
  else out->info = lu_in(f, n, &piv);
  auto solve = [&](const std::vector<V>& rhs) {
    std::vector<W> t(rhs.size());
    for (size_t i = 0; i < rhs.size(); ++i) t[i] = to_w<W>(rhs[i]);
    if (ls) qr_solve_in(f, m, n, tau, t);
    else lu_solve_in(f, n, piv, t);
    std::vector<V> r((size_t)n);
    for (int i = 0; i < n; ++i) r[(size_t)i] = round_main(from_w<V>(t[(size_t)i]), single_main);
    return r;
  };
  x->assign((size_t)n * nrhs, V(0.0));
  for (int c = 0; c < nrhs; ++c) {
    const std::vector<V> bc(b.begin() + (ptrdiff_t)c * m, b.begin() + (ptrdiff_t)(c + 1) * m);
    const auto xc = solve(bc);
    std::copy(xc.begin(), xc.end(), x->begin() + (ptrdiff_t)c * n);
  }
  if (ipiv) *ipiv = piv;
  auto residuals = [&](std::vector<std::vector<V>>* r, double* worst_ratio, double* rmax) {
    bool ok = true;
    *rmax = 0;
    *worst_ratio = 0;
    r->assign((size_t)nrhs, std::vector<V>((size_t)m));
    for (int c = 0; c < nrhs; ++c) {
      double rn = 0, xn = 0;
      for (int i = 0; i < m; ++i) {
        V s = b[(size_t)c * m + i];
        for (int k = 0; k < n; ++k) s -= a[(size_t)k * m + i] * (*x)[(size_t)c * n + k];
        s = round_main(s, single_main);
        (*r)[(size_t)c][(size_t)i] = s;
        rn = std::max(rn, std::abs(s));
      }
      for (int k = 0; k < n; ++k) xn = std::max(xn, std::abs((*x)[(size_t)c * n + k]));
      if (!std::isfinite(rn) || !std::isfinite(xn)) return -1;
      if (!(rn < xn * cte)) ok = false;
      *rmax = std::max(*rmax, rn);
      *worst_ratio = std::max(*worst_ratio, rn / std::max(xn, 1e-300));
    }
    return ok ? 1 : 0;
  };
  std::vector<std::vector<V>> r;
  double ratio, rmax;
  int conv = residuals(&r, &ratio, &rmax);
  out->history.push_back(0);
  out->history.push_back(rmax);
  bool failed = false;
  if (!refine) {
    out->niters = out->outer = 0;
    return false;
  }
  if (conv == 1) {
    out->niters = out->outer = 0;
  } else if (conv < 0) {
    out->niters = out->outer = -5;  // the lower-precision solve overflowed
    failed = true;
  } else {
    int it = 1;
    for (; it <= maxiters; ++it) {
      for (int c = 0; c < nrhs; ++c) {
        const auto d = solve(r[(size_t)c]);
        for (int k = 0; k < n; ++k)
          (*x)[(size_t)c * n + k] = round_main((*x)[(size_t)c * n + k] + d[(size_t)k], single_main);
      }
      conv = residuals(&r, &ratio, &rmax);
      out->history.push_back(it);
      out->history.push_back(rmax);
      if (conv != 0) break;
    }
    if (conv == 1) {
      out->niters = out->outer = it;
    } else {
      out->niters = out->outer = conv < 0 ? -5 : -maxiters;
      failed = true;
    }
  }
  if (!failed || !fallback) return false;
  // The fallback: the main precision's factorization, in place, and its solve.
  if (ls) {
    std::vector<V> t;
    qr_in(a, m, n, &t);
    for (int c = 0; c < nrhs; ++c) {
      std::vector<V> bc(b.begin() + (ptrdiff_t)c * m, b.begin() + (ptrdiff_t)(c + 1) * m);
      qr_solve_in(a, m, n, t, bc);
      for (int k = 0; k < n; ++k) (*x)[(size_t)c * n + k] = round_main(bc[(size_t)k], single_main);
    }
  } else {
    std::vector<int> p;
    lu_in(a, n, &p);
    for (int c = 0; c < nrhs; ++c) {
      std::vector<V> bc(b.begin() + (ptrdiff_t)c * m, b.begin() + (ptrdiff_t)(c + 1) * m);
      lu_solve_in(a, n, p, bc);
      for (int k = 0; k < n; ++k) (*x)[(size_t)c * n + k] = round_main(bc[(size_t)k], single_main);
    }
    if (ipiv) *ipiv = p;
  }
  out->info = 0;  // as NVIDIA's reports it, even when U has a zero pivot
  return true;
}

size_t irs_lwork(int m, int n, int nrhs, size_t elem) {
  return ((size_t)std::max(m, 1) * std::max(n, 1) + (size_t)std::max(m, 1) * std::max(nrhs, 1) * 2 + 64) * elem;
}

// The device side of one call: T is the I/O element type; `low` true factors
// in single (when T is double precision), false in T's own precision.
template <class T>
cusolverStatus_t irs_call(bool ls, bool low, bool refine, bool fallback, int maxiters, double tol, int m, int n,
                          int nrhs, void* dA, int lda, void* dB, int ldb, void* dX, int ldx, int* dipiv,
                          IrsOut* out) {
  using V = Vof<T>;
  constexpr bool single = std::is_same_v<T, float> || std::is_same_v<T, cuComplex>;
  DM<V> a, b, x;
  if (!dload<T>(dA, lda, n, &a) || !dload<T>(dB, ldb, nrhs, &b) || !dload<T>(dX, ldx, nrhs, &x))
    return CUSOLVER_STATUS_EXECUTION_FAILED;
  std::vector<V> ap((size_t)m * n), bp((size_t)m * nrhs), xs;
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < m; ++i) ap[(size_t)j * m + i] = a(i, j);
  for (int j = 0; j < nrhs; ++j)
    for (int i = 0; i < m; ++i) bp[(size_t)j * m + i] = b(i, j);
  std::vector<int> piv;
  bool overwrote;
  if (low || single)
    overwrote = irs_run<V, typename LowOf<V>::type>(ls, single, refine, fallback, maxiters, tol, ap, m, n, bp, nrhs, &xs,
                                                    &piv, out);
  else
    overwrote = irs_run<V, V>(ls, single, refine, fallback, maxiters, tol, ap, m, n, bp, nrhs, &xs, &piv, out);
  // No refinement at the main precision: A takes that factorization, as NVIDIA documents.
  if (!refine && !low && !ls) {
    std::vector<int> p;
    lu_in(ap, n, &p);
    overwrote = true;
  }
  for (int j = 0; j < nrhs; ++j)
    for (int i = 0; i < n; ++i) x(i, j) = xs[(size_t)j * n + i];
  if (!dsave<T>(dX, x)) return CUSOLVER_STATUS_EXECUTION_FAILED;
  if (overwrote) {
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < m; ++i) a(i, j) = ap[(size_t)j * m + i];
    if (!dsave<T>(dA, a)) return CUSOLVER_STATUS_EXECUTION_FAILED;
  }
  if (dipiv && !ls && n &&
      cudaMemcpy(dipiv, piv.data(), (size_t)n * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess)
    return CUSOLVER_STATUS_EXECUTION_FAILED;
  return CUSOLVER_STATUS_SUCCESS;
}

// <t1><t2>gesv and gels: the argument checks NVIDIA's makes, then the solve.
template <class T>
cusolverStatus_t gesv_entry(cusolverDnHandle_t h, bool low, int n, int nrhs, void* dA, int lda, int* dipiv, void* dB,
                            int ldb, void* dX, int ldx, size_t lwork, int* iter, int* dinfo) {
  if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;
  int bad = n < 0 ? 1 : nrhs < 0 ? 2 : lda < std::max(1, n) ? 4 : ldb < std::max(1, n) ? 7 : ldx < std::max(1, n) ? 9 : 0;
  if (bad) {
    set_info(dinfo, -bad);
    return CUSOLVER_STATUS_IRS_PARAMS_INVALID;
  }
  if (n == 0 || nrhs == 0) return CUSOLVER_STATUS_SUCCESS;
  if (lwork < irs_lwork(n, n, nrhs, sizeof(T))) return CUSOLVER_STATUS_INVALID_WORKSPACE;
  IrsOut out;
  const cusolverStatus_t st =
      irs_call<T>(false, low, true, true, 50, 0, n, n, nrhs, dA, lda, dB, ldb, dX, ldx, dipiv, &out);
  if (st != CUSOLVER_STATUS_SUCCESS) return st;
  if (iter) *iter = out.niters;
  set_info(dinfo, out.info);
  return CUSOLVER_STATUS_SUCCESS;
}
template <class T>
cusolverStatus_t gels_entry(cusolverDnHandle_t h, bool low, int m, int n, int nrhs, void* dA, int lda, void* dB,
                            int ldb, void* dX, int ldx, size_t lwork, int* iter, int* dinfo) {
  if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;
  int bad = m < 0 ? 1 : (n < 0 || n > m) ? 2 : nrhs < 0 ? 3 : lda < std::max(1, m) ? 5 : ldb < std::max(1, m) ? 7
          : ldx < std::max(1, n) ? 9 : 0;
  if (bad) {
    set_info(dinfo, -bad);
    return CUSOLVER_STATUS_IRS_PARAMS_INVALID;
  }
  if (n == 0 || nrhs == 0) return CUSOLVER_STATUS_SUCCESS;
  if (lwork < irs_lwork(m, n, nrhs, sizeof(T))) return CUSOLVER_STATUS_INVALID_WORKSPACE;
  IrsOut out;
  const cusolverStatus_t st = irs_call<T>(true, low, true, true, 50, 0, m, n, nrhs, dA, lda, dB, ldb, dX, ldx, nullptr, &out);
  if (st != CUSOLVER_STATUS_SUCCESS) return st;
  if (iter) *iter = out.niters;
  set_info(dinfo, out.info);
  return CUSOLVER_STATUS_SUCCESS;
}

}  // namespace

// Each <t1><t2> pair: T the I/O type, LOW whether <t2> is below <t1>.
#define VGPU_IRS(N, T, LOW)                                                                                          \
  VGPU_EXPORT cusolverStatus_t cusolverDn##N##gesv_bufferSize(cusolverDnHandle_t h, cusolver_int_t n,               \
                                                              cusolver_int_t nrhs, T*, cusolver_int_t lda,          \
                                                              cusolver_int_t*, T*, cusolver_int_t ldb, T*,          \
                                                              cusolver_int_t ldx, void*, size_t* bytes) {           \
    if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;                                                           \
    if (n < 0 || nrhs < 0 || lda < std::max(1, n) || ldb < std::max(1, n) || ldx < std::max(1, n) || !bytes)        \
      return CUSOLVER_STATUS_IRS_PARAMS_INVALID;                                                                     \
    *bytes = irs_lwork(n, n, nrhs, sizeof(T));                                                                       \
    return CUSOLVER_STATUS_SUCCESS;                                                                                  \
  }                                                                                                                  \
  VGPU_EXPORT cusolverStatus_t cusolverDn##N##gesv(cusolverDnHandle_t h, cusolver_int_t n, cusolver_int_t nrhs,     \
                                                   T* A, cusolver_int_t lda, cusolver_int_t* ipiv, T* B,            \
                                                   cusolver_int_t ldb, T* X, cusolver_int_t ldx, void* unused10,             \
                                                   size_t lwork, cusolver_int_t* iter, cusolver_int_t* info) { \
  VGPU_DEFER_CALL(g_reg, stream_of(h), cusolverDn##N##gesv, h, n, nrhs, A, lda, ipiv, B, ldb, X, ldx, unused10, lwork, iter, info);      \
    return gesv_entry<T>(h, LOW, n, nrhs, A, lda, ipiv, B, ldb, X, ldx, lwork, iter, info);                          \
  }                                                                                                                  \
  VGPU_EXPORT cusolverStatus_t cusolverDn##N##gels_bufferSize(cusolverDnHandle_t h, cusolver_int_t m,               \
                                                              cusolver_int_t n, cusolver_int_t nrhs, T*,            \
                                                              cusolver_int_t lda, T*, cusolver_int_t ldb, T*,       \
                                                              cusolver_int_t ldx, void*, size_t* bytes) {           \
    if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;                                                           \
    if (m < 0 || n < 0 || n > m || nrhs < 0 || lda < std::max(1, m) || ldb < std::max(1, m) ||                      \
        ldx < std::max(1, n) || !bytes)                                                                              \
      return CUSOLVER_STATUS_IRS_PARAMS_INVALID;                                                                     \
    *bytes = irs_lwork(m, n, nrhs, sizeof(T));                                                                       \
    return CUSOLVER_STATUS_SUCCESS;                                                                                  \
  }                                                                                                                  \
  VGPU_EXPORT cusolverStatus_t cusolverDn##N##gels(cusolverDnHandle_t h, cusolver_int_t m, cusolver_int_t n,        \
                                                   cusolver_int_t nrhs, T* A, cusolver_int_t lda, T* B,             \
                                                   cusolver_int_t ldb, T* X, cusolver_int_t ldx, void* unused10,             \
                                                   size_t lwork, cusolver_int_t* iter, cusolver_int_t* info) { \
  VGPU_DEFER_CALL(g_reg, stream_of(h), cusolverDn##N##gels, h, m, n, nrhs, A, lda, B, ldb, X, ldx, unused10, lwork, iter, info);      \
    return gels_entry<T>(h, LOW, m, n, nrhs, A, lda, B, ldb, X, ldx, lwork, iter, info);                             \
  }
VGPU_IRS(DD, double, false)
VGPU_IRS(DS, double, true)
VGPU_IRS(DH, double, true)
VGPU_IRS(DB, double, true)
VGPU_IRS(DX, double, true)
VGPU_IRS(SS, float, false)
VGPU_IRS(SH, float, false)
VGPU_IRS(SB, float, false)
VGPU_IRS(SX, float, false)
VGPU_IRS(ZZ, cuDoubleComplex, false)
VGPU_IRS(ZC, cuDoubleComplex, true)
VGPU_IRS(ZK, cuDoubleComplex, true)
VGPU_IRS(ZE, cuDoubleComplex, true)
VGPU_IRS(ZY, cuDoubleComplex, true)
VGPU_IRS(CC, cuComplex, false)
VGPU_IRS(CK, cuComplex, false)
VGPU_IRS(CE, cuComplex, false)
VGPU_IRS(CY, cuComplex, false)
#undef VGPU_IRS

/* ---- the expert interface: params and infos ---- */

namespace {
struct IrsParams {
  int main_prec = 0, lowest = 0;       // cusolverPrecType_t, 0 = not set
  int refine = CUSOLVER_IRS_REFINE_NOT_SET;
  int maxiters = 50, maxiters_inner = 50;
  double tol = 0, tol_inner = 1e-4;
  bool fallback = true;
};
struct IrsInfos {
  int niters = 0, outer = 0, maxiters = 50;
  bool want_residual = false;
  std::vector<double> history;  // (maxiters + 1) x 2, column-major
  std::vector<float> history32;
  bool single = false;  // the history is float for a single-precision solve
};
IrsParams* irs_params(cusolverDnIRSParams_t p) { return known(p) ? reinterpret_cast<IrsParams*>(p) : nullptr; }
IrsInfos* irs_infos(cusolverDnIRSInfos_t p) { return known(p) ? reinterpret_cast<IrsInfos*>(p) : nullptr; }

bool prec_main_ok(int p) {
  return p == CUSOLVER_R_64F || p == CUSOLVER_R_32F || p == CUSOLVER_C_64F || p == CUSOLVER_C_32F;
}
// Whether lowest is allowed for main (the documentation's table), and whether
// it is below it.
bool prec_pair_ok(int main, int lowest, bool* low) {
  const bool real = main == CUSOLVER_R_64F || main == CUSOLVER_R_32F;
  const int r64 = real ? CUSOLVER_R_64F : CUSOLVER_C_64F, r32 = real ? CUSOLVER_R_32F : CUSOLVER_C_32F;
  const int r16 = real ? CUSOLVER_R_16F : CUSOLVER_C_16F, rbf = real ? CUSOLVER_R_16BF : CUSOLVER_C_16BF;
  const int tf = real ? CUSOLVER_R_TF32 : CUSOLVER_C_TF32;
  const bool dbl = main == r64;
  if (lowest == main) {
    *low = false;
    return true;
  }
  if ((dbl && lowest == r32) || lowest == r16 || lowest == rbf || lowest == tf) {
    *low = dbl;  // below single runs as single (RTX 3060, CUDA 13.0)
    return true;
  }
  return false;
}
bool is_gmres(int r) {
  return r == CUSOLVER_IRS_REFINE_GMRES || r == CUSOLVER_IRS_REFINE_CLASSICAL_GMRES ||
         r == CUSOLVER_IRS_REFINE_GMRES_GMRES || r == CUSOLVER_IRS_REFINE_GMRES_NOPCOND;
}
bool refine_ok(int r) { return r == CUSOLVER_IRS_REFINE_NONE || r == CUSOLVER_IRS_REFINE_CLASSICAL || is_gmres(r); }

// The checks IRSXgesv/IRSXgels make on their params before anything else.
cusolverStatus_t irs_params_check(const IrsParams* p, int nrhs, int* dinfo, bool* low) {
  if (!p) return CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED;
  if (!prec_main_ok(p->main_prec) || !prec_pair_ok(p->main_prec, p->lowest, low)) {
    set_info(dinfo, -9);
    return CUSOLVER_STATUS_IRS_PARAMS_INVALID_PREC;
  }
  if (!refine_ok(p->refine)) {
    set_info(dinfo, -9);
    return CUSOLVER_STATUS_IRS_PARAMS_INVALID_REFINE;
  }
  if (is_gmres(p->refine) && nrhs > 1) return CUSOLVER_STATUS_IRS_NOT_SUPPORTED;
  return CUSOLVER_STATUS_SUCCESS;
}

template <class F> cusolverStatus_t by_prec(int main, F&& body) {
  switch (main) {
    case CUSOLVER_R_64F: return body(double{});
    case CUSOLVER_R_32F: return body(float{});
    case CUSOLVER_C_64F: return body(cuDoubleComplex{});
    case CUSOLVER_C_32F: return body(cuComplex{});
    default: return CUSOLVER_STATUS_IRS_PARAMS_INVALID_PREC;
  }
}

cusolverStatus_t irs_expert(bool ls, cusolverDnHandle_t h, cusolverDnIRSParams_t pp, cusolverDnIRSInfos_t ip, int m,
                            int n, int nrhs, void* dA, int lda, void* dB, int ldb, void* dX, int ldx, size_t lwork,
                            int* niters, int* dinfo) {
  if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;
  IrsParams* p = irs_params(pp);
  IrsInfos* inf = irs_infos(ip);
  if (!inf) return CUSOLVER_STATUS_IRS_INFOS_NOT_INITIALIZED;
  bool low = false;
  if (const cusolverStatus_t st = irs_params_check(p, nrhs, dinfo, &low); st != CUSOLVER_STATUS_SUCCESS) return st;
  const int bad = ls ? (m < 0 ? 1 : (n < 0 || n > m) ? 2 : nrhs < 0 ? 3 : lda < std::max(1, m) ? 5
                        : ldb < std::max(1, m) ? 7 : ldx < std::max(1, n) ? 9 : 0)
                     : (n < 0 ? 1 : nrhs < 0 ? 2 : lda < std::max(1, n) ? 4 : ldb < std::max(1, n) ? 6
                        : ldx < std::max(1, n) ? 8 : 0);
  if (bad) {
    set_info(dinfo, -bad);
    return CUSOLVER_STATUS_IRS_PARAMS_INVALID;
  }
  if (n == 0 || nrhs == 0) return CUSOLVER_STATUS_SUCCESS;
  const bool single = p->main_prec == CUSOLVER_R_32F || p->main_prec == CUSOLVER_C_32F;
  const size_t elem = p->main_prec == CUSOLVER_C_64F ? 16 : p->main_prec == CUSOLVER_R_32F ? 4 : 8;
  if (lwork < irs_lwork(m, n, nrhs, elem)) return CUSOLVER_STATUS_INVALID_WORKSPACE;
  const bool refine = p->refine != CUSOLVER_IRS_REFINE_NONE;
  IrsOut out;
  const cusolverStatus_t st = by_prec(p->main_prec, [&](auto tag) {
    using T = decltype(tag);
    return irs_call<T>(ls, low, refine, p->fallback, p->maxiters, p->tol, m, n, nrhs, dA, lda, dB, ldb, dX, ldx, nullptr,
                       &out);
  });
  if (st != CUSOLVER_STATUS_SUCCESS) return st;
  inf->niters = out.niters;
  inf->outer = out.outer;
  inf->maxiters = p->maxiters;
  if (inf->want_residual) {  // (maxiters + 1) x 2: iterations so far, then the residual norm
    const size_t ld = (size_t)p->maxiters + 1;
    inf->history.assign(ld * 2, 0.0);
    for (size_t r = 0; r < out.history.size() / 2 && r < ld; ++r) {
      inf->history[r] = out.history[2 * r];
      inf->history[ld + r] = out.history[2 * r + 1];
    }
    inf->history32.assign(inf->history.begin(), inf->history.end());
    inf->single = single;
  }
  if (niters) *niters = out.niters;
  set_info(dinfo, out.info);
  return CUSOLVER_STATUS_SUCCESS;
}
}  // namespace

VGPU_EXPORT cusolverStatus_t cusolverDnIRSParamsCreate(cusolverDnIRSParams_t* p) {
  if (!p) return CUSOLVER_STATUS_INVALID_VALUE;
  *p = reinterpret_cast<cusolverDnIRSParams_t>(track(new IrsParams()));
  return CUSOLVER_STATUS_SUCCESS;
}
VGPU_EXPORT cusolverStatus_t cusolverDnIRSParamsDestroy(cusolverDnIRSParams_t p) {
  if (!irs_params(p)) return CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED;
  untrack(p);
  delete reinterpret_cast<IrsParams*>(p);
  return CUSOLVER_STATUS_SUCCESS;
}
VGPU_EXPORT cusolverStatus_t cusolverDnIRSParamsSetSolverPrecisions(cusolverDnIRSParams_t pp, cusolverPrecType_t main_p, cusolverPrecType_t lowest) {
  IrsParams* p = irs_params(pp);
  if (!p) return CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED;
  p->main_prec = main_p; p->lowest = lowest;
  return CUSOLVER_STATUS_SUCCESS;
}
VGPU_EXPORT cusolverStatus_t cusolverDnIRSParamsSetSolverMainPrecision(cusolverDnIRSParams_t pp, cusolverPrecType_t main_p) {
  IrsParams* p = irs_params(pp);
  if (!p) return CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED;
  p->main_prec = main_p;
  return CUSOLVER_STATUS_SUCCESS;
}
VGPU_EXPORT cusolverStatus_t cusolverDnIRSParamsSetSolverLowestPrecision(cusolverDnIRSParams_t pp, cusolverPrecType_t lowest) {
  IrsParams* p = irs_params(pp);
  if (!p) return CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED;
  p->lowest = lowest;
  return CUSOLVER_STATUS_SUCCESS;
}
VGPU_EXPORT cusolverStatus_t cusolverDnIRSParamsSetRefinementSolver(cusolverDnIRSParams_t pp, cusolverIRSRefinement_t r) {
  IrsParams* p = irs_params(pp);
  if (!p) return CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED;
  p->refine = r;
  return CUSOLVER_STATUS_SUCCESS;
}
VGPU_EXPORT cusolverStatus_t cusolverDnIRSParamsSetTol(cusolverDnIRSParams_t pp, double v) {
  IrsParams* p = irs_params(pp);
  if (!p) return CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED;
  p->tol = v;
  return CUSOLVER_STATUS_SUCCESS;
}
VGPU_EXPORT cusolverStatus_t cusolverDnIRSParamsSetTolInner(cusolverDnIRSParams_t pp, double v) {
  IrsParams* p = irs_params(pp);
  if (!p) return CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED;
  p->tol_inner = v;
  return CUSOLVER_STATUS_SUCCESS;
}
VGPU_EXPORT cusolverStatus_t cusolverDnIRSParamsSetMaxIters(cusolverDnIRSParams_t pp, cusolver_int_t v) {
  IrsParams* p = irs_params(pp);
  if (!p) return CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED;
  p->maxiters = v;
  p->maxiters_inner = std::min(p->maxiters_inner, v);
  return CUSOLVER_STATUS_SUCCESS;
}
VGPU_EXPORT cusolverStatus_t cusolverDnIRSParamsEnableFallback(cusolverDnIRSParams_t pp) {
  IrsParams* p = irs_params(pp);
  if (!p) return CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED;
  p->fallback = true;
  return CUSOLVER_STATUS_SUCCESS;
}
VGPU_EXPORT cusolverStatus_t cusolverDnIRSParamsDisableFallback(cusolverDnIRSParams_t pp) {
  IrsParams* p = irs_params(pp);
  if (!p) return CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED;
  p->fallback = false;
  return CUSOLVER_STATUS_SUCCESS;
}
// The inner limit may not exceed the total (IRS_PARAMS_INVALID, as NVIDIA's answers).
VGPU_EXPORT cusolverStatus_t cusolverDnIRSParamsSetMaxItersInner(cusolverDnIRSParams_t pp, cusolver_int_t v) {
  IrsParams* p = irs_params(pp);
  if (!p) return CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED;
  if (v > p->maxiters) return CUSOLVER_STATUS_IRS_PARAMS_INVALID;
  p->maxiters_inner = v;
  return CUSOLVER_STATUS_SUCCESS;
}
VGPU_EXPORT cusolverStatus_t cusolverDnIRSParamsGetMaxIters(cusolverDnIRSParams_t pp, cusolver_int_t* v) {
  IrsParams* p = irs_params(pp);
  if (!p) return CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED;
  if (!v) return CUSOLVER_STATUS_INVALID_VALUE;
  *v = p->maxiters;
  return CUSOLVER_STATUS_SUCCESS;
}

VGPU_EXPORT cusolverStatus_t cusolverDnIRSInfosCreate(cusolverDnIRSInfos_t* p) {
  if (!p) return CUSOLVER_STATUS_INVALID_VALUE;
  *p = reinterpret_cast<cusolverDnIRSInfos_t>(track(new IrsInfos()));
  return CUSOLVER_STATUS_SUCCESS;
}
VGPU_EXPORT cusolverStatus_t cusolverDnIRSInfosDestroy(cusolverDnIRSInfos_t p) {
  if (!irs_infos(p)) return CUSOLVER_STATUS_IRS_INFOS_NOT_INITIALIZED;
  untrack(p);
  delete reinterpret_cast<IrsInfos*>(p);
  return CUSOLVER_STATUS_SUCCESS;
}
VGPU_EXPORT cusolverStatus_t cusolverDnIRSInfosGetMaxIters(cusolverDnIRSInfos_t ip, cusolver_int_t* v) {
  IrsInfos* i = irs_infos(ip);
  if (!i) return CUSOLVER_STATUS_IRS_INFOS_NOT_INITIALIZED;
  if (!v) return CUSOLVER_STATUS_INVALID_VALUE;
  *v = i->maxiters;
  return CUSOLVER_STATUS_SUCCESS;
}
VGPU_EXPORT cusolverStatus_t cusolverDnIRSInfosGetNiters(cusolverDnIRSInfos_t ip, cusolver_int_t* v) {
  IrsInfos* i = irs_infos(ip);
  if (!i) return CUSOLVER_STATUS_IRS_INFOS_NOT_INITIALIZED;
  if (!v) return CUSOLVER_STATUS_INVALID_VALUE;
  *v = i->niters;
  return CUSOLVER_STATUS_SUCCESS;
}
VGPU_EXPORT cusolverStatus_t cusolverDnIRSInfosGetOuterNiters(cusolverDnIRSInfos_t ip, cusolver_int_t* v) {
  IrsInfos* i = irs_infos(ip);
  if (!i) return CUSOLVER_STATUS_IRS_INFOS_NOT_INITIALIZED;
  if (!v) return CUSOLVER_STATUS_INVALID_VALUE;
  *v = i->outer;
  return CUSOLVER_STATUS_SUCCESS;
}
VGPU_EXPORT cusolverStatus_t cusolverDnIRSInfosRequestResidual(cusolverDnIRSInfos_t ip) {
  IrsInfos* i = irs_infos(ip);
  if (!i) return CUSOLVER_STATUS_IRS_INFOS_NOT_INITIALIZED;
  i->want_residual = true;
  return CUSOLVER_STATUS_SUCCESS;
}
// Not requested before the solve: INVALID_VALUE (RTX 3060, CUDA 13.0).
VGPU_EXPORT cusolverStatus_t cusolverDnIRSInfosGetResidualHistory(cusolverDnIRSInfos_t ip, void** h) {
  IrsInfos* i = irs_infos(ip);
  if (!i) return CUSOLVER_STATUS_IRS_INFOS_NOT_INITIALIZED;
  if (!h || !i->want_residual) return CUSOLVER_STATUS_INVALID_VALUE;
  *h = i->single ? static_cast<void*>(i->history32.data()) : static_cast<void*>(i->history.data());
  return CUSOLVER_STATUS_SUCCESS;
}

VGPU_EXPORT cusolverStatus_t cusolverDnIRSXgesv_bufferSize(cusolverDnHandle_t h, cusolverDnIRSParams_t pp,
                                                           cusolver_int_t n, cusolver_int_t nrhs, size_t* bytes) {
  if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;
  IrsParams* p = irs_params(pp);
  if (!p) return CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED;
  bool low;
  if (!prec_main_ok(p->main_prec) || !prec_pair_ok(p->main_prec, p->lowest, &low) || !refine_ok(p->refine) || n < 0 ||
      nrhs < 0 || !bytes)
    return CUSOLVER_STATUS_IRS_PARAMS_INVALID;
  if (is_gmres(p->refine) && nrhs > 1) return CUSOLVER_STATUS_IRS_NRHS_NOT_SUPPORTED_FOR_REFINE_GMRES;
  *bytes = irs_lwork(n, n, nrhs, p->main_prec == CUSOLVER_C_64F ? 16 : p->main_prec == CUSOLVER_R_32F ? 4 : 8);
  return CUSOLVER_STATUS_SUCCESS;
}
VGPU_EXPORT cusolverStatus_t cusolverDnIRSXgels_bufferSize(cusolverDnHandle_t h, cusolverDnIRSParams_t pp,
                                                           cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs,
                                                           size_t* bytes) {
  if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;
  IrsParams* p = irs_params(pp);
  if (!p) return CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED;
  bool low;
  if (!prec_main_ok(p->main_prec) || !prec_pair_ok(p->main_prec, p->lowest, &low) || !refine_ok(p->refine) || m < 0 || n < 0 ||
      n > m || nrhs < 0 || !bytes)
    return CUSOLVER_STATUS_IRS_PARAMS_INVALID;
  if (is_gmres(p->refine) && nrhs > 1) return CUSOLVER_STATUS_IRS_NRHS_NOT_SUPPORTED_FOR_REFINE_GMRES;
  *bytes = irs_lwork(m, n, nrhs, p->main_prec == CUSOLVER_C_64F ? 16 : p->main_prec == CUSOLVER_R_32F ? 4 : 8);
  return CUSOLVER_STATUS_SUCCESS;
}
VGPU_EXPORT cusolverStatus_t cusolverDnIRSXgesv(cusolverDnHandle_t h, cusolverDnIRSParams_t p, cusolverDnIRSInfos_t i,
                                                cusolver_int_t n, cusolver_int_t nrhs, void* A, cusolver_int_t lda,
                                                void* B, cusolver_int_t ldb, void* X, cusolver_int_t ldx, void* unused11,
                                                size_t lwork, cusolver_int_t* niters, cusolver_int_t* dinfo) {
  VGPU_DEFER_CALL(g_reg, stream_of(h), cusolverDnIRSXgesv, h, p, i, n, nrhs, A, lda, B, ldb, X, ldx, unused11, lwork, niters, dinfo);
  return irs_expert(false, h, p, i, n, n, nrhs, A, lda, B, ldb, X, ldx, lwork, niters, dinfo);
}
VGPU_EXPORT cusolverStatus_t cusolverDnIRSXgels(cusolverDnHandle_t h, cusolverDnIRSParams_t p, cusolverDnIRSInfos_t i,
                                                cusolver_int_t m, cusolver_int_t n, cusolver_int_t nrhs, void* A,
                                                cusolver_int_t lda, void* B, cusolver_int_t ldb, void* X,
                                                cusolver_int_t ldx, void* unused12, size_t lwork, cusolver_int_t* niters,
                                                cusolver_int_t* dinfo) {
  VGPU_DEFER_CALL(g_reg, stream_of(h), cusolverDnIRSXgels, h, p, i, m, n, nrhs, A, lda, B, ldb, X, ldx, unused12, lwork, niters, dinfo);
  return irs_expert(true, h, p, i, m, n, nrhs, A, lda, B, ldb, X, ldx, lwork, niters, dinfo);
}
