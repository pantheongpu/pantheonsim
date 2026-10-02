// The rest of cuSOLVER's dense API, included at the end of cusolver_api.cpp
// (whose helpers -- known, set_info, Sym<T>, sym_load/sym_save, span,
// JacobiInfo -- it uses), for the real and the complex types alike:
//
//   sytrd/hetrd, orgtr/ungtr, ormtr/unmtr   tridiagonal reduction and its Q
//   gebrd, orgbr/ungbr                      bidiagonal reduction, Q and P^H
//   potri, lauum, Xtrtri                    inverses from triangular factors
//   syevdx/heevdx, Xsyevdx                  a range of eigenpairs
//   sygvd/hegvd, sygvdx/hegvdx, sygvj/hegvj the generalized problem
//   Xgetrf/Xgetrs                           LU with 64-bit pivots
//   Xgesvd, Xgesvdp, Xgesvdr                the SVD, by the 64-bit API
//   Xlarft                                  a block reflector's T
//   the Jacobi getters, deterministic and math modes
//   <t1><t2>gesv/gels, IRSXgesv/IRSXgels    iterative refinement solvers
//
// The reductions follow LAPACK's unblocked routines (dense_generic.hpp). The
// eigen- and singular-value solvers are the Jacobi methods the rest of this
// library uses; they converge to the same decompositions, so an eigen- or
// singular vector may differ from NVIDIA's by its sign (or phase).

#include "dense_generic.hpp"

namespace {

using vgpu_dg::DM;
using vgpu_dg::cj;
template <class T> using Vof = typename Sym<T>::V;
// The real type of an element type: float for cuComplex, double for cuDoubleComplex.
template <class T> struct RealOf { using type = T; };
template <> struct RealOf<cuComplex> { using type = float; };
template <> struct RealOf<cuDoubleComplex> { using type = double; };
template <class T> using Re = typename RealOf<T>::type;
template <class T> constexpr bool is_cplx = std::is_same_v<T, cuComplex> || std::is_same_v<T, cuDoubleComplex>;

// rows x cols of a column-major device matrix (leading dimension ld) into a
// host matrix with the same ld; and back, the whole span.
template <class T> bool dload(const void* dev, int ld, int cols, DM<Vof<T>>* m) {
  m->ld = std::max(ld, 1);
  return sym_load<T>(dev, span(ld, cols), &m->v);
}
template <class T> bool dsave(void* dev, const DM<Vof<T>>& m) { return sym_save<T>(dev, m.v); }
template <class R> bool rsave_vec(void* dev, const std::vector<double>& v) {
  if (v.empty()) return true;
  std::vector<R> raw(v.begin(), v.end());
  return cudaMemcpy(dev, raw.data(), raw.size() * sizeof(R), cudaMemcpyHostToDevice) == cudaSuccess;
}
template <class T> bool tsave_vec(void* dev, const std::vector<Vof<T>>& v) { return v.empty() || sym_save<T>(dev, v); }
template <class T> bool tload_vec(const void* dev, size_t n, std::vector<Vof<T>>* v) { return sym_load<T>(dev, n, v); }

// Runs body(T{}) for the element type a cudaDataType names.
template <class F> cusolverStatus_t by_elem(cudaDataType t, F&& body) {
  switch (t) {
    case CUDA_R_32F: return body(float{});
    case CUDA_R_64F: return body(double{});
    case CUDA_C_32F: return body(cuComplex{});
    case CUDA_C_64F: return body(cuDoubleComplex{});
    default: return CUSOLVER_STATUS_NOT_SUPPORTED;
  }
}
bool is_elem(cudaDataType t) { return t == CUDA_R_32F || t == CUDA_R_64F || t == CUDA_C_32F || t == CUDA_C_64F; }
// The real type of a precision: R_32F for R_32F and C_32F.
cudaDataType real_of(cudaDataType t) { return t == CUDA_R_32F || t == CUDA_C_32F ? CUDA_R_32F : CUDA_R_64F; }

// The Hermitian matrix one triangle of a holds (n x n, packed ld = n).
template <class V> DM<V> hermitian(const DM<V>& a, int n, bool upper) {
  DM<V> f(n, n);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
      const bool stored = upper ? i <= j : i >= j;
      f(i, j) = stored ? a(i, j) : cj(a(j, i));
    }
  for (int i = 0; i < n; ++i) f(i, i) = V(vgpu_dg::re(f(i, i)));
  return f;
}

// Eigenpairs of a full Hermitian n x n matrix: w ascending, Z's columns.
template <class V> void herm_eig(const DM<V>& f, int n, std::vector<double>* w, DM<V>* z) {
  CMat in{std::vector<cd>((size_t)n * n), std::max(n, 1)};
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) in(i, j) = cd(f(i, j));
  CMat v;
  vgpu_la::heev(in, n, true, w, &v);
  *z = DM<V>(n, n);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
      if constexpr (std::is_same_v<V, double>) (*z)(i, j) = v(i, j).real();
      else (*z)(i, j) = v(i, j);
    }
}

// A = U diag(s) V^H for any shape: s descending (k = min(m, n)); U m x m and
// V n x n when `full`, else m x k and n x k.
template <class V>
void any_svd(const DM<V>& a, int m, int n, bool full, std::vector<double>* s, DM<V>* u, DM<V>* v) {
  CMat in{std::vector<cd>((size_t)std::max(m, 1) * n), std::max(m, 1)};
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < m; ++i) in(i, j) = cd(a(i, j));
  CMat cu, cv;
  vgpu_la::svd(in, m, n, full, s, &cu, &cv);
  const int k = std::min(m, n), uc = full ? m : k, vc = full ? n : k;
  *u = DM<V>(m, uc);
  *v = DM<V>(n, vc);
  for (int j = 0; j < uc; ++j)
    for (int i = 0; i < m; ++i) {
      if constexpr (std::is_same_v<V, double>) (*u)(i, j) = cu(i, j).real();
      else (*u)(i, j) = cu(i, j);
    }
  for (int j = 0; j < vc; ++j)
    for (int i = 0; i < n; ++i) {
      if constexpr (std::is_same_v<V, double>) (*v)(i, j) = cv(i, j).real();
      else (*v)(i, j) = cv(i, j);
    }
}

// Cholesky of the stored triangle, in place: A = L L^H or U^H U. 0 or the
// 1-based order of the first failing minor. As NVIDIA's (RTX 3060, CUDA
// 13.0, sygvd's B): a failing column gets sqrt of what it had (NaN) on the
// diagonal and zeros below it, and the factorization goes on past it.
template <class V> int chol(bool upper, int n, DM<V>& a) {
  int info = 0;
  for (int j = 0; j < n; ++j) {
    double d = vgpu_dg::re(a(j, j));
    for (int k = 0; k < j; ++k) d -= std::norm(cd(upper ? a(k, j) : a(j, k)));
    if (!(d > 0.0)) {
      if (!info) info = j + 1;
      a(j, j) = V(std::sqrt(d));
      for (int i = j + 1; i < n; ++i) (upper ? a(j, i) : a(i, j)) = V(0.0);
      continue;
    }
    d = std::sqrt(d);
    a(j, j) = V(d);
    for (int i = j + 1; i < n; ++i) {
      V s = upper ? a(j, i) : a(i, j);
      for (int k = 0; k < j; ++k) s -= upper ? cj(a(k, j)) * a(k, i) : a(i, k) * cj(a(j, k));
      (upper ? a(j, i) : a(i, j)) = s / d;
    }
  }
  return info;
}

// The selection syevdx/sygvdx return: all, (vl, vu], or il..iu (1-based).
std::vector<int> eig_range(const std::vector<double>& w, cusolverEigRange_t range, double vl, double vu, int64_t il,
                           int64_t iu) {
  std::vector<int> pick;
  for (int i = 0; i < (int)w.size(); ++i) {
    if (range == CUSOLVER_EIG_RANGE_V && !(w[(size_t)i] > vl && w[(size_t)i] <= vu)) continue;
    if (range == CUSOLVER_EIG_RANGE_I && (i + 1 < il || i + 1 > iu)) continue;
    pick.push_back(i);
  }
  return pick;
}
bool range_ok(cusolverEigRange_t r) {
  return r == CUSOLVER_EIG_RANGE_ALL || r == CUSOLVER_EIG_RANGE_V || r == CUSOLVER_EIG_RANGE_I;
}
bool jobz_ok(cusolverEigMode_t j) { return j == CUSOLVER_EIG_MODE_NOVECTOR || j == CUSOLVER_EIG_MODE_VECTOR; }

/* ---- the eigen bodies, every type ---- */

// What NVIDIA's syevdx leaves besides the chosen eigenpairs, as an RTX 3060
// (CUDA 13.0) showed it: it mirrors the stored triangle into the lower one
// and reduces that to tridiagonal form (sytrd's lower layout: d on the
// diagonal, e below it, the reflectors under that; the strict upper triangle
// keeps what it had), then writes the meig vectors over A's first columns.
// W holds the meig eigenvalues, then the tridiagonal's diagonal d(meig..n-1).
template <class V>
void eig_select_into(DM<V>& a, int n, const DM<V>& f, cusolverEigMode_t jobz, cusolverEigRange_t range, double vl,
                     double vu, int64_t il, int64_t iu, const std::vector<double>& w, const DM<V>& z,
                     std::vector<double>* wout, int64_t* meig) {
  DM<V> t = f;
  std::vector<double> d, e;
  std::vector<V> tau;
  vgpu_dg::sytd2(false, n, t, &d, &e, &tau);
  for (int j = 0; j < n; ++j)
    for (int i = j; i < n; ++i) a(i, j) = t(i, j);
  const auto pick = eig_range(w, range, vl, vu, il, iu);
  *wout = d;
  for (size_t c = 0; c < pick.size(); ++c) (*wout)[c] = w[(size_t)pick[c]];
  if (meig) *meig = (int64_t)pick.size();
  if (jobz == CUSOLVER_EIG_MODE_VECTOR)
    for (size_t c = 0; c < pick.size(); ++c)
      for (int r = 0; r < n; ++r) a(r, (int)c) = z(r, pick[c]);
}

// syevd's selection form: W gets the chosen eigenvalues (ascending), A's first
// meig columns their vectors.
template <class T>
cusolverStatus_t syevdx_body(cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n, void* A,
                             int lda, double vl, double vu, int64_t il, int64_t iu, int64_t* meig, void* W,
                             int* info) {
  using V = Vof<T>;
  DM<V> a;
  if (!dload<T>(A, lda, n, &a)) return CUSOLVER_STATUS_EXECUTION_FAILED;
  const DM<V> f = hermitian(a, n, uplo == CUBLAS_FILL_MODE_UPPER);
  std::vector<double> w, wout;
  DM<V> z;
  herm_eig(f, n, &w, &z);
  eig_select_into(a, n, f, jobz, range, vl, vu, il, iu, w, z, &wout, meig);
  set_info(info, 0);
  return rsave_vec<Re<T>>(W, wout) && dsave<T>(A, a) ? CUSOLVER_STATUS_SUCCESS : CUSOLVER_STATUS_EXECUTION_FAILED;
}

// The generalized problem by Cholesky reduction (LAPACK's sygvd): B = F F^H
// with F = L (lower) or U^H (upper); itype 1: C = F^-1 A F^-H, x = F^-H y;
// itype 2: C = F^H A F, x = F^-H y; itype 3: C = F^H A F, x = F y. A's stored
// triangle takes C (as sygst leaves it), then everything syevdx leaves
// (eig_select_into) with the vectors x. B keeps its factor in the stored
// triangle. When B is not positive definite, B holds the factorization as far
// as it got (dpotf2's: the failing diagonal's value written) and info is
// n + i, or i for the Jacobi form (`jacobi`), as NVIDIA documents.
template <class T>
cusolverStatus_t sygv_body(int itype, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n,
                           void* A, int lda, void* B, int ldb, double vl, double vu, int il, int iu, int* meig,
                           void* W, int* info, bool jacobi) {
  using V = Vof<T>;
  const bool upper = uplo == CUBLAS_FILL_MODE_UPPER;
  DM<V> a, b;
  if (!dload<T>(A, lda, n, &a) || !dload<T>(B, ldb, n, &b)) return CUSOLVER_STATUS_EXECUTION_FAILED;
  const int rc = chol(upper, n, b);
  if (rc) {
    set_info(info, jacobi ? rc : n + rc);
    return dsave<T>(B, b) ? CUSOLVER_STATUS_SUCCESS : CUSOLVER_STATUS_EXECUTION_FAILED;
  }
  DM<V> f(n, n);  // F, lower triangular
  for (int j = 0; j < n; ++j)
    for (int i = j; i < n; ++i) f(i, j) = upper ? cj(b(j, i)) : b(i, j);
  DM<V> fi = f;  // F^-1
  vgpu_dg::trtri(false, false, n, fi);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < j; ++i) fi(i, j) = V(0.0);
  const DM<V> h = hermitian(a, n, upper);
  auto mul = [n](const DM<V>& x, bool xh, const DM<V>& y, bool yh) {
    DM<V> r(n, n);
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < n; ++i) {
        V s(0.0);
        for (int k = 0; k < n; ++k) s += (xh ? cj(x(k, i)) : x(i, k)) * (yh ? cj(y(j, k)) : y(k, j));
        r(i, j) = s;
      }
    return r;
  };
  DM<V> c = itype == 1 ? mul(mul(fi, false, h, false), false, fi, true) : mul(mul(f, true, h, false), false, f, false);
  for (int j = 0; j < n; ++j)  // exactly Hermitian, as the solver needs it
    for (int i = 0; i < j; ++i) c(i, j) = cj(c(j, i));
  for (int i = 0; i < n; ++i) c(i, i) = V(vgpu_dg::re(c(i, i)));
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i)
      if (upper ? i <= j : i >= j) a(i, j) = c(i, j);
  std::vector<double> w, wout;
  DM<V> y;
  herm_eig(c, n, &w, &y);
  int64_t m64 = 0;
  eig_select_into(a, n, c, jobz, range, vl, vu, il, iu, w, y, &wout, &m64);
  // The back-transformation runs over all n columns of A, the ones past meig
  // included (what an RTX 3060's CUDA 13.0 leaves there): x = F^-H y, or F y.
  if (jobz == CUSOLVER_EIG_MODE_VECTOR) {
    DM<V> sq(n, n);
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < n; ++i) sq(i, j) = a(i, j);
    const DM<V> x = itype == 3 ? mul(f, false, sq, false) : mul(fi, true, sq, false);
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < n; ++i) a(i, j) = x(i, j);
  }
  if (meig) *meig = (int)m64;
  set_info(info, 0);
  const bool ok = rsave_vec<Re<T>>(W, wout) && dsave<T>(A, a) && dsave<T>(B, b);
  return ok ? CUSOLVER_STATUS_SUCCESS : CUSOLVER_STATUS_EXECUTION_FAILED;
}

/* ---- reductions ---- */

template <class T>
cusolverStatus_t sytrd_body(cublasFillMode_t uplo, int n, void* A, int lda, void* d, void* e, void* tau, int* info) {
  using V = Vof<T>;
  DM<V> a;
  if (!dload<T>(A, lda, n, &a)) return CUSOLVER_STATUS_EXECUTION_FAILED;
  const bool upper = uplo == CUBLAS_FILL_MODE_UPPER;
  DM<V> f = hermitian(a, n, upper);
  std::vector<double> dv, ev;
  std::vector<V> tv;
  vgpu_dg::sytd2(upper, n, f, &dv, &ev, &tv);
  for (int j = 0; j < n; ++j)  // the stored triangle only
    for (int i = 0; i < n; ++i)
      if (upper ? i <= j : i >= j) a(i, j) = f(i, j);
  set_info(info, 0);
  const bool ok = dsave<T>(A, a) && rsave_vec<Re<T>>(d, dv) && rsave_vec<Re<T>>(e, ev) && tsave_vec<T>(tau, tv);
  return ok ? CUSOLVER_STATUS_SUCCESS : CUSOLVER_STATUS_EXECUTION_FAILED;
}

template <class T>
cusolverStatus_t orgtr_body(cublasFillMode_t uplo, int n, void* A, int lda, const void* tau, int* info) {
  using V = Vof<T>;
  DM<V> a;
  std::vector<V> t;
  if (!dload<T>(A, lda, n, &a) || !tload_vec<T>(tau, (size_t)std::max(n - 1, 0), &t))
    return CUSOLVER_STATUS_EXECUTION_FAILED;
  const DM<V> q = vgpu_dg::form(vgpu_dg::tr_reflectors(uplo == CUBLAS_FILL_MODE_UPPER, n, a, t.data()), n);
  // NVIDIA's writes the whole leading dimension: Q, then zeros below it
  // (RTX 3060, CUDA 13.0).
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < a.ld; ++i) a(i, j) = i < n ? q(i, j) : V(0.0);
  set_info(info, 0);
  return dsave<T>(A, a) ? CUSOLVER_STATUS_SUCCESS : CUSOLVER_STATUS_EXECUTION_FAILED;
}

template <class T>
cusolverStatus_t ormtr_body(cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, int m, int n,
                            const void* A, int lda, const void* tau, void* C, int ldc, int* info) {
  using V = Vof<T>;
  const bool left = side == CUBLAS_SIDE_LEFT;
  const int nq = left ? m : n;
  DM<V> a, c;
  std::vector<V> t;
  if (!dload<T>(A, lda, nq, &a) || !dload<T>(C, ldc, n, &c) ||
      !tload_vec<T>(tau, (size_t)std::max(nq - 1, 0), &t))
    return CUSOLVER_STATUS_EXECUTION_FAILED;
  const auto q = vgpu_dg::tr_reflectors(uplo == CUBLAS_FILL_MODE_UPPER, nq, a, t.data());
  vgpu_dg::apply_product(q, c, m, n, left, trans != CUBLAS_OP_N);
  set_info(info, 0);
  return dsave<T>(C, c) ? CUSOLVER_STATUS_SUCCESS : CUSOLVER_STATUS_EXECUTION_FAILED;
}

template <class T>
cusolverStatus_t gebrd_body(int m, int n, void* A, int lda, void* D, void* E, void* tauq, void* taup, int* info) {
  using V = Vof<T>;
  DM<V> a;
  if (!dload<T>(A, lda, n, &a)) return CUSOLVER_STATUS_EXECUTION_FAILED;
  std::vector<double> d, e;
  std::vector<V> tq, tp;
  vgpu_dg::gebd2(m, n, a, &d, &e, &tq, &tp);
  set_info(info, 0);
  const bool ok = dsave<T>(A, a) && rsave_vec<Re<T>>(D, d) && rsave_vec<Re<T>>(E, e) && tsave_vec<T>(tauq, tq) &&
                  tsave_vec<T>(taup, tp);
  return ok ? CUSOLVER_STATUS_SUCCESS : CUSOLVER_STATUS_EXECUTION_FAILED;
}

// side LEFT: Q (m x n) from gebrd of an m x k matrix; RIGHT: P^H (m x n) from
// gebrd of a k x n matrix.
template <class T>
cusolverStatus_t orgbr_body(cublasSideMode_t side, int m, int n, int k, void* A, int lda, const void* tau, int* info) {
  using V = Vof<T>;
  const bool q_side = side == CUBLAS_SIDE_LEFT;
  DM<V> a;
  std::vector<V> t;
  const int ntau = q_side ? std::min(m, k) : std::min(n, k);
  if (!dload<T>(A, lda, n, &a) || !tload_vec<T>(tau, (size_t)std::max(ntau, 0), &t))
    return CUSOLVER_STATUS_EXECUTION_FAILED;
  if (q_side) {
    const DM<V> q = vgpu_dg::form(vgpu_dg::bd_q(m, k, a, t.data()), m);
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < m; ++i) a(i, j) = q(i, j);
  } else {
    const DM<V> p = vgpu_dg::form(vgpu_dg::bd_p(n, k, a, t.data()), n);
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < m; ++i) a(i, j) = cj(p(j, i));  // P^H
  }
  set_info(info, 0);
  return dsave<T>(A, a) ? CUSOLVER_STATUS_SUCCESS : CUSOLVER_STATUS_EXECUTION_FAILED;
}

template <class T> cusolverStatus_t trtri_body(bool upper, bool unit, int n, void* A, int lda, int* info) {
  using V = Vof<T>;
  DM<V> a;
  if (!dload<T>(A, lda, n, &a)) return CUSOLVER_STATUS_EXECUTION_FAILED;
  const int rc = vgpu_dg::trtri(upper, unit, n, a);
  set_info(info, rc);
  if (rc) return CUSOLVER_STATUS_SUCCESS;
  // With a unit diagonal NVIDIA's writes the ones it assumed (RTX 3060, CUDA 13.0).
  if (unit)
    for (int i = 0; i < n; ++i) a(i, i) = V(1.0);
  return dsave<T>(A, a) ? CUSOLVER_STATUS_SUCCESS : CUSOLVER_STATUS_EXECUTION_FAILED;
}

// potri: the inverse of A from its Cholesky factor (trtri, then lauum).
template <class T> cusolverStatus_t potri_body(bool upper, int n, void* A, int lda, int* info, bool inverse) {
  using V = Vof<T>;
  DM<V> a;
  if (!dload<T>(A, lda, n, &a)) return CUSOLVER_STATUS_EXECUTION_FAILED;
  // NVIDIA's potri does not look for a zero on the factor's diagonal: info is
  // 0 and the inverse holds infinities and NaNs (RTX 3060, CUDA 13.0).
  if (inverse) vgpu_dg::trtri(upper, false, n, a, /*check=*/false);
  set_info(info, 0);
  vgpu_dg::lauum(upper, n, a);
  return dsave<T>(A, a) ? CUSOLVER_STATUS_SUCCESS : CUSOLVER_STATUS_EXECUTION_FAILED;
}

}  // namespace

namespace {
// A legacy entry point's argument check: NVIDIA's returns INVALID_VALUE and
// sets info to -i for the i-th argument (not counting the handle), as
// LAPACK's xerbla numbers them (RTX 3060, CUDA 13.0).
cusolverStatus_t bad_arg(int i, int* info) {
  set_info(info, -i);
  return CUSOLVER_STATUS_INVALID_VALUE;
}
int arg_uplo_n_lda(cublasFillMode_t uplo, int n, int lda) {
  if (!uplo_ok(uplo)) return 1;
  if (n < 0) return 2;
  if (lda < std::max(1, n)) return 4;
  return 0;
}
template <class T> int arg_ormtr(cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, int m, int n,
                                 int lda, int ldc);
int arg_orgbr(cublasSideMode_t side, int m, int n, int k, int lda);
int arg_syevdx(cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n, int lda, double vl,
               double vu, int il, int iu);
int arg_sygvd(cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, int lda, int ldb);
int arg_sygvdx(cusolverEigType_t itype, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n,
               int lda, int ldb, double vl, double vu, int il, int iu);
}  // namespace

/* ---- the legacy (32-bit) entry points ---- */

#define VGPU_DN_MORE(P, T, SY, OR, HE)                                                                               \
  VGPU_EXPORT cusolverStatus_t cusolverDn##P##SY##trd_bufferSize(cusolverDnHandle_t h, cublasFillMode_t, int n,      \
                                                                 const T*, int, const Re<T>*, const Re<T>*,      \
                                                                 const T*, int* lwork) {                             \
    if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;                                                           \
    if (!lwork) return CUSOLVER_STATUS_INVALID_VALUE;                                                                \
    *lwork = lwork_hint(n);                                                                                          \
    return CUSOLVER_STATUS_SUCCESS;                                                                                  \
  }                                                                                                                  \
  VGPU_EXPORT cusolverStatus_t cusolverDn##P##SY##trd(cusolverDnHandle_t h, cublasFillMode_t uplo, int n, T* A,      \
                                                      int lda, Re<T>* d, Re<T>* e, T* tau, T*, int, int* info) {     \
    if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;                                                           \
    if (const int bad = arg_uplo_n_lda(uplo, n, lda)) return bad_arg(bad, info);                                     \
    return sytrd_body<T>(uplo, n, A, lda, d, e, tau, info);                                                          \
  }                                                                                                                  \
  VGPU_EXPORT cusolverStatus_t cusolverDn##P##OR##gtr_bufferSize(cusolverDnHandle_t h, cublasFillMode_t, int n,      \
                                                                 const T*, int, const T*, int* lwork) {          \
    if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;                                                           \
    if (!lwork) return CUSOLVER_STATUS_INVALID_VALUE;                                                                \
    *lwork = lwork_hint(n);                                                                                          \
    return CUSOLVER_STATUS_SUCCESS;                                                                                  \
  }                                                                                                                  \
  VGPU_EXPORT cusolverStatus_t cusolverDn##P##OR##gtr(cusolverDnHandle_t h, cublasFillMode_t uplo, int n, T* A,      \
                                                      int lda, const T* tau, T*, int, int* info) {                   \
    if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;                                                           \
    if (const int bad = arg_uplo_n_lda(uplo, n, lda)) return bad_arg(bad, info);                                     \
    return orgtr_body<T>(uplo, n, A, lda, tau, info);                                                                \
  }                                                                                                                  \
  VGPU_EXPORT cusolverStatus_t cusolverDn##P##OR##mtr_bufferSize(                                                    \
      cusolverDnHandle_t h, cublasSideMode_t, cublasFillMode_t, cublasOperation_t, int m, int n, const T*,      \
      int, const T*, const T*, int, int* lwork) {                                                            \
    if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;                                                           \
    if (!lwork) return CUSOLVER_STATUS_INVALID_VALUE;                                                                \
    *lwork = lwork_hint(std::max(m, n));                                                                             \
    return CUSOLVER_STATUS_SUCCESS;                                                                                  \
  }                                                                                                                  \
  VGPU_EXPORT cusolverStatus_t cusolverDn##P##OR##mtr(cusolverDnHandle_t h, cublasSideMode_t side,                   \
                                                      cublasFillMode_t uplo, cublasOperation_t trans, int m, int n,  \
                                                      T* A, int lda, T* tau, T* C, int ldc, T*, int, int* info) {    \
    if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;                                                           \
    const int nq = side == CUBLAS_SIDE_LEFT ? m : n;                                                                 \
    if (const int bad = arg_ormtr<T>(side, uplo, trans, m, n, lda, ldc)) return bad_arg(bad, info);                  \
    (void)nq;                                                                                                        \
    return ormtr_body<T>(side, uplo, trans, m, n, A, lda, tau, C, ldc, info);                                        \
  }                                                                                                                  \
  VGPU_EXPORT cusolverStatus_t cusolverDn##P##gebrd_bufferSize(cusolverDnHandle_t h, int m, int n, int* lwork) {     \
    if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;                                                           \
    if (!lwork) return CUSOLVER_STATUS_INVALID_VALUE;                                                                \
    *lwork = lwork_hint(std::max(m, n));                                                                             \
    return CUSOLVER_STATUS_SUCCESS;                                                                                  \
  }                                                                                                                  \
  VGPU_EXPORT cusolverStatus_t cusolverDn##P##gebrd(cusolverDnHandle_t h, int m, int n, T* A, int lda, Re<T>* D,     \
                                                    Re<T>* E, T* tauq, T* taup, T*, int, int* info) {                \
    if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;                                                           \
    if (m < 0) return bad_arg(1, info);                                                                              \
    if (n < 0) return bad_arg(2, info);                                                                              \
    if (lda < std::max(1, m)) return bad_arg(4, info);                                                               \
    if (m < n) { /* NVIDIA's reduces m >= n only (RTX 3060, CUDA 13.0) */                                           \
      set_info(info, 0);                                                                                             \
      return CUSOLVER_STATUS_NOT_SUPPORTED;                                                                          \
    }                                                                                                                \
    return gebrd_body<T>(m, n, A, lda, D, E, tauq, taup, info);                                                      \
  }                                                                                                                  \
  VGPU_EXPORT cusolverStatus_t cusolverDn##P##OR##gbr_bufferSize(cusolverDnHandle_t h, cublasSideMode_t, int m,      \
                                                                 int n, int, const T*, int, const T*,            \
                                                                 int* lwork) {                                       \
    if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;                                                           \
    if (!lwork) return CUSOLVER_STATUS_INVALID_VALUE;                                                                \
    *lwork = lwork_hint(std::max(m, n));                                                                             \
    return CUSOLVER_STATUS_SUCCESS;                                                                                  \
  }                                                                                                                  \
  VGPU_EXPORT cusolverStatus_t cusolverDn##P##OR##gbr(cusolverDnHandle_t h, cublasSideMode_t side, int m, int n,     \
                                                      int k, T* A, int lda, const T* tau, T*, int, int* info) {      \
    if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;                                                           \
    if (const int bad = arg_orgbr(side, m, n, k, lda)) return bad_arg(bad, info);                                    \
    return orgbr_body<T>(side, m, n, k, A, lda, tau, info);                                                          \
  }                                                                                                                  \
  VGPU_EXPORT cusolverStatus_t cusolverDn##P##potri_bufferSize(cusolverDnHandle_t h, cublasFillMode_t, int n, T*,    \
                                                               int, int* lwork) {                                \
    if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;                                                           \
    if (!lwork) return CUSOLVER_STATUS_INVALID_VALUE;                                                                \
    *lwork = lwork_hint(n);                                                                                          \
    return CUSOLVER_STATUS_SUCCESS;                                                                                  \
  }                                                                                                                  \
  VGPU_EXPORT cusolverStatus_t cusolverDn##P##potri(cusolverDnHandle_t h, cublasFillMode_t uplo, int n, T* A,        \
                                                    int lda, T*, int, int* info) {                                   \
    if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;                                                           \
    if (const int bad = arg_uplo_n_lda(uplo, n, lda)) return bad_arg(bad, info);                                     \
    return potri_body<T>(uplo == CUBLAS_FILL_MODE_UPPER, n, A, lda, info, true);                                     \
  }                                                                                                                  \
  VGPU_EXPORT cusolverStatus_t cusolverDn##P##lauum_bufferSize(cusolverDnHandle_t h, cublasFillMode_t, int n, T*,    \
                                                               int, int* lwork) {                                \
    if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;                                                           \
    if (!lwork) return CUSOLVER_STATUS_INVALID_VALUE;                                                                \
    *lwork = lwork_hint(n);                                                                                          \
    return CUSOLVER_STATUS_SUCCESS;                                                                                  \
  }                                                                                                                  \
  VGPU_EXPORT cusolverStatus_t cusolverDn##P##lauum(cusolverDnHandle_t h, cublasFillMode_t uplo, int n, T* A,        \
                                                    int lda, T*, int, int* info) {                                   \
    if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;                                                           \
    /* NVIDIA's lauum leaves info alone on a bad argument (RTX 3060, CUDA 13.0) */                                  \
    if (arg_uplo_n_lda(uplo, n, lda)) return CUSOLVER_STATUS_INVALID_VALUE;                                          \
    return potri_body<T>(uplo == CUBLAS_FILL_MODE_UPPER, n, A, lda, info, false);                                    \
  }                                                                                                                  \
  VGPU_EXPORT cusolverStatus_t cusolverDn##P##HE##evdx_bufferSize(                                                   \
      cusolverDnHandle_t h, cusolverEigMode_t, cusolverEigRange_t, cublasFillMode_t, int n, const T*, int,       \
      Re<T>, Re<T>, int, int, int*, const Re<T>*, int* lwork) {                                                      \
    if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;                                                           \
    if (!lwork) return CUSOLVER_STATUS_INVALID_VALUE;                                                                \
    *lwork = lwork_hint(n);                                                                                          \
    return CUSOLVER_STATUS_SUCCESS;                                                                                  \
  }                                                                                                                  \
  VGPU_EXPORT cusolverStatus_t cusolverDn##P##HE##evdx(                                                              \
      cusolverDnHandle_t h, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n, T* A,    \
      int lda, Re<T> vl, Re<T> vu, int il, int iu, int* meig, Re<T>* W, T*, int, int* info) {                        \
    if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;                                                           \
    if (const int bad = arg_syevdx(jobz, range, uplo, n, lda, vl, vu, il, iu)) return bad_arg(bad, info);           \
    int64_t m64 = 0;                                                                                                 \
    const cusolverStatus_t st = syevdx_body<T>(jobz, range, uplo, n, A, lda, vl, vu, il, iu, &m64, W, info);         \
    if (meig) *meig = (int)m64;                                                                                      \
    return st;                                                                                                       \
  }                                                                                                                  \
  VGPU_EXPORT cusolverStatus_t cusolverDn##P##SY##gvd_bufferSize(                                                    \
      cusolverDnHandle_t h, cusolverEigType_t, cusolverEigMode_t, cublasFillMode_t, int n, const T*, int,        \
      const T*, int, const Re<T>*, int* lwork) {                                                                 \
    if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;                                                           \
    if (!lwork) return CUSOLVER_STATUS_INVALID_VALUE;                                                                \
    *lwork = lwork_hint(n);                                                                                          \
    return CUSOLVER_STATUS_SUCCESS;                                                                                  \
  }                                                                                                                  \
  VGPU_EXPORT cusolverStatus_t cusolverDn##P##SY##gvd(cusolverDnHandle_t h, cusolverEigType_t itype,                 \
                                                      cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, T* A,    \
                                                      int lda, T* B, int ldb, Re<T>* W, T*, int, int* info) {        \
    if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;                                                           \
    if (const int bad = arg_sygvd(itype, jobz, uplo, n, lda, ldb)) return bad_arg(bad, info);                        \
    return sygv_body<T>((int)itype, jobz, CUSOLVER_EIG_RANGE_ALL, uplo, n, A, lda, B, ldb, 0, 0, 0, 0, nullptr, W,   \
                        info, false);                                                                                \
  }                                                                                                                  \
  VGPU_EXPORT cusolverStatus_t cusolverDn##P##SY##gvdx_bufferSize(                                                   \
      cusolverDnHandle_t h, cusolverEigType_t, cusolverEigMode_t, cusolverEigRange_t, cublasFillMode_t, int n,       \
      const T*, int, const T*, int, Re<T>, Re<T>, int, int, int*, const Re<T>*, int* lwork) {                \
    if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;                                                           \
    if (!lwork) return CUSOLVER_STATUS_INVALID_VALUE;                                                                \
    *lwork = lwork_hint(n);                                                                                          \
    return CUSOLVER_STATUS_SUCCESS;                                                                                  \
  }                                                                                                                  \
  VGPU_EXPORT cusolverStatus_t cusolverDn##P##SY##gvdx(                                                              \
      cusolverDnHandle_t h, cusolverEigType_t itype, cusolverEigMode_t jobz, cusolverEigRange_t range,               \
      cublasFillMode_t uplo, int n, T* A, int lda, T* B, int ldb, Re<T> vl, Re<T> vu, int il, int iu, int* meig,     \
      Re<T>* W, T*, int, int* info) {                                                                                \
    if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;                                                           \
    if (const int bad = arg_sygvdx(itype, jobz, range, uplo, n, lda, ldb, vl, vu, il, iu)) return bad_arg(bad, info); \
    return sygv_body<T>((int)itype, jobz, range, uplo, n, A, lda, B, ldb, vl, vu, il, iu, meig, W, info, false);     \
  }                                                                                                                  \
  VGPU_EXPORT cusolverStatus_t cusolverDn##P##SY##gvj_bufferSize(                                                    \
      cusolverDnHandle_t h, cusolverEigType_t, cusolverEigMode_t, cublasFillMode_t, int n, const T*, int,        \
      const T*, int, const Re<T>*, int* lwork, syevjInfo_t) {                                                    \
    if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;                                                           \
    if (!lwork) return CUSOLVER_STATUS_INVALID_VALUE;                                                                \
    *lwork = lwork_hint(n);                                                                                          \
    return CUSOLVER_STATUS_SUCCESS;                                                                                  \
  }                                                                                                                  \
  VGPU_EXPORT cusolverStatus_t cusolverDn##P##SY##gvj(cusolverDnHandle_t h, cusolverEigType_t itype,                 \
                                                      cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, T* A,    \
                                                      int lda, T* B, int ldb, Re<T>* W, T*, int, int* info,          \
                                                      syevjInfo_t jp) {                                              \
    if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;                                                           \
    if (const int bad = arg_sygvd(itype, jobz, uplo, n, lda, ldb)) return bad_arg(bad, info);                        \
    const cusolverStatus_t st = sygv_body<T>((int)itype, jobz, CUSOLVER_EIG_RANGE_ALL, uplo, n, A, lda, B, ldb, 0, 0, \
                                             0, 0, nullptr, W, info, true);                                          \
    jacobi_note(jp, false);                                                                                          \
    return st;                                                                                                       \
  }

namespace {
template <class T> bool mtr_trans_ok(cublasOperation_t t) {
  // LAPACK's dormtr takes N or T and zunmtr N or C.
  return t == CUBLAS_OP_N || t == (is_cplx<T> ? CUBLAS_OP_C : CUBLAS_OP_T);
}
// orgbr: Q of order m from an m x k matrix needs m >= n >= min(m, k); P^H of
// order n from a k x n matrix needs n >= m >= min(n, k).
bool orgbr_shape_ok(cublasSideMode_t side, int m, int n, int k) {
  if (m < 0 || n < 0 || k < 0) return false;
  if (side == CUBLAS_SIDE_LEFT) return n <= m && n >= std::min(m, k);
  if (side == CUBLAS_SIDE_RIGHT) return m <= n && m >= std::min(n, k);
  return false;
}
bool itype_ok(cusolverEigType_t t) {
  return t == CUSOLVER_EIG_TYPE_1 || t == CUSOLVER_EIG_TYPE_2 || t == CUSOLVER_EIG_TYPE_3;
}
// LAPACK's: vl < vu for a value range; 1 <= il <= iu <= n (or il = 1, iu = 0 for n = 0).
bool range_args_ok(cusolverEigRange_t r, int64_t n, double vl, double vu, int64_t il, int64_t iu) {
  if (r == CUSOLVER_EIG_RANGE_V) return vl < vu;
  if (r == CUSOLVER_EIG_RANGE_I) return n == 0 ? (il == 1 && iu == 0) : (il >= 1 && il <= iu && iu <= n);
  return true;
}
}  // namespace

namespace {
template <class T> int arg_ormtr(cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t trans, int m, int n,
                                 int lda, int ldc) {
  if (side != CUBLAS_SIDE_LEFT && side != CUBLAS_SIDE_RIGHT) return 1;
  if (!uplo_ok(uplo)) return 2;
  if (!mtr_trans_ok<T>(trans)) return 3;
  if (m < 0) return 4;
  if (n < 0) return 5;
  if (lda < std::max(1, side == CUBLAS_SIDE_LEFT ? m : n)) return 7;
  if (ldc < std::max(1, m)) return 10;
  return 0;
}
int arg_orgbr(cublasSideMode_t side, int m, int n, int k, int lda) {
  if (side != CUBLAS_SIDE_LEFT && side != CUBLAS_SIDE_RIGHT) return 1;
  if (m < 0) return 2;
  if (!orgbr_shape_ok(side, m, n, std::max(k, 0))) return 3;
  if (k < 0) return 4;
  if (lda < std::max(1, m)) return 6;
  return 0;
}
int arg_syevdx(cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n, int lda, double vl,
               double vu, int il, int iu) {
  if (!jobz_ok(jobz)) return 1;
  if (!range_ok(range)) return 2;
  if (!uplo_ok(uplo)) return 3;
  if (n < 0) return 4;
  if (lda < std::max(1, n)) return 6;
  if (range == CUSOLVER_EIG_RANGE_V && n > 0 && !(vl < vu)) return 8;
  if (range == CUSOLVER_EIG_RANGE_I) {
    if (il < 1 || il > std::max(1, n)) return 9;
    if (iu < std::min(n, il) || iu > n) return 10;
  }
  return 0;
}
int arg_sygvd(cusolverEigType_t itype, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, int lda, int ldb) {
  if (!itype_ok(itype)) return 1;
  if (!jobz_ok(jobz)) return 2;
  if (!uplo_ok(uplo)) return 3;
  if (n < 0) return 4;
  if (lda < std::max(1, n)) return 6;
  if (ldb < std::max(1, n)) return 8;
  return 0;
}
int arg_sygvdx(cusolverEigType_t itype, cusolverEigMode_t jobz, cusolverEigRange_t range, cublasFillMode_t uplo, int n,
               int lda, int ldb, double vl, double vu, int il, int iu) {
  if (!itype_ok(itype)) return 1;
  if (!jobz_ok(jobz)) return 2;
  if (!range_ok(range)) return 3;
  if (!uplo_ok(uplo)) return 4;
  if (n < 0) return 5;
  if (lda < std::max(1, n)) return 7;
  if (ldb < std::max(1, n)) return 9;
  if (range == CUSOLVER_EIG_RANGE_V && n > 0 && !(vl < vu)) return 11;
  if (range == CUSOLVER_EIG_RANGE_I) {
    if (il < 1 || il > std::max(1, n)) return 12;
    if (iu < std::min(n, il) || iu > n) return 13;
  }
  return 0;
}
}  // namespace

VGPU_DN_MORE(S, float, sy, or, sy)
VGPU_DN_MORE(D, double, sy, or, sy)
VGPU_DN_MORE(C, cuComplex, he, un, he)
VGPU_DN_MORE(Z, cuDoubleComplex, he, un, he)
#undef VGPU_DN_MORE

/* ---- the 64-bit (X) API ---- */

namespace {

template <class T> bool to_cmat(const void* dev, int64_t ld, int64_t cols, CMat* m) {
  DM<Vof<T>> a;
  if (!dload<T>(dev, (int)ld, (int)cols, &a)) return false;
  m->ld = a.ld;
  m->v.assign(a.v.begin(), a.v.end());
  return true;
}
template <class T> bool from_cmat(void* dev, const CMat& m) {
  DM<Vof<T>> a;
  a.ld = m.ld;
  a.v.resize(m.v.size());
  for (size_t i = 0; i < m.v.size(); ++i) {
    if constexpr (is_cplx<T>) a.v[i] = m.v[i];
    else a.v[i] = m.v[i].real();
  }
  return dsave<T>(dev, a);
}

// Writes rows x cols of a packed matrix into ld-strided device memory,
// keeping what lies between the columns.
template <class T> bool put_block(void* dev, int64_t ld, const DM<Vof<T>>& src, int rows, int cols) {
  if (rows <= 0 || cols <= 0) return true;
  DM<Vof<T>> out;
  if (!dload<T>(dev, (int)ld, cols, &out)) return false;
  for (int j = 0; j < cols; ++j)
    for (int i = 0; i < rows; ++i) out(i, j) = src(i, j);
  return dsave<T>(dev, out);
}

// The SVD types every X SVD takes: A, U, V(T) and compute of one element
// type, S its real type.
bool svd_types_ok(cudaDataType a, cudaDataType s, cudaDataType u, cudaDataType v, cudaDataType c) {
  return is_elem(a) && s == real_of(a) && u == a && v == a && c == a;
}
bool jobsvd_ok(signed char j) { return j == 'A' || j == 'S' || j == 'O' || j == 'N'; }
// Xgesvd's argument check, LAPACK's numbering (jobu 1, jobvt 2, m 3, n 4,
// lda 6, ldu 9, ldvt 11), which NVIDIA's uses for info here.
int xgesvd_bad_arg(signed char ju, signed char jv, int64_t m, int64_t n, int64_t lda, int64_t ldu, int64_t ldvt) {
  if (!jobsvd_ok(ju)) return 1;
  if (!jobsvd_ok(jv) || (ju == 'O' && jv == 'O')) return 2;
  if (m < 0) return 3;
  if (n < 0) return 4;
  if (lda < std::max<int64_t>(1, m)) return 6;
  if (ldu < 1 || ((ju == 'A' || ju == 'S') && ldu < m)) return 9;
  if (ldvt < 1 || (jv == 'A' && ldvt < n) || (jv == 'S' && ldvt < std::min(m, n))) return 11;
  return 0;
}
// gesvdr, counting params as 1: jobu 2, jobv 3, m 4, n 5, k 6, p 7, niters 8,
// lda 11, ldu 15, ldv 18. NVIDIA's takes k = 0 and m < n.
int gesvdr_bad_arg(signed char ju, signed char jv, int64_t m, int64_t n, int64_t k, int64_t p, int64_t niters,
                   int64_t lda, int64_t ldu, int64_t ldv) {
  if (ju != 'S' && ju != 'N') return 2;
  if (jv != 'S' && jv != 'N') return 3;
  if (m < 0) return 4;
  if (n < 0) return 5;
  if (k < 0 || k > std::min(m, n)) return 6;
  if (p < 0 || k + p > std::min(m, n)) return 7;
  if (niters < 0) return 8;
  if (lda < std::max<int64_t>(1, m)) return 11;
  if (ju == 'S' && ldu < std::max<int64_t>(1, m)) return 15;
  if (jv == 'S' && ldv < std::max<int64_t>(1, n)) return 18;
  return 0;
}
// larft: column-wise only, 1 <= n, k <= n; NVIDIA's checks no leading
// dimension and takes k = 0 (RTX 3060, CUDA 13.0).
bool larft_args_ok(cusolverDirectMode_t d, cusolverStorevMode_t s, int64_t n, int64_t k, int64_t, int64_t) {
  if (d != CUBLAS_DIRECT_FORWARD && d != CUBLAS_DIRECT_BACKWARD) return false;
  if (s != CUBLAS_STOREV_COLUMNWISE) return false;
  return n >= 1 && k >= 0 && k <= n;
}
}  // namespace

VGPU_EXPORT cusolverStatus_t cusolverDnXgetrf_bufferSize(cusolverDnHandle_t h, cusolverDnParams_t, int64_t m,
                                                         int64_t n, cudaDataType ta, const void*, int64_t lda,
                                                         cudaDataType tc, size_t* dev, size_t* host) {
  if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;
  (void)ta, (void)tc, (void)lda;  // NVIDIA's sizes anything (RTX 3060, CUDA 13.0)
  if (dev) *dev = x_workspace(std::max(m, n));
  if (host) *host = 0;
  return CUSOLVER_STATUS_SUCCESS;
}
VGPU_EXPORT cusolverStatus_t cusolverDnXgetrf(cusolverDnHandle_t h, cusolverDnParams_t, int64_t m, int64_t n,
                                              cudaDataType ta, void* A, int64_t lda, int64_t* ipiv, cudaDataType tc,
                                              void*, size_t, void*, size_t, int* info) {
  if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;
  // What an RTX 3060 (CUDA 13.0) answers: an element type it has no LU for is
  // INVALID_VALUE with info 0; a bad size is SUCCESS with info -i in LAPACK's
  // numbering (m, n, A, lda); the compute type is not checked.
  if (!is_elem(ta)) {
    set_info(info, 0);
    return CUSOLVER_STATUS_INVALID_VALUE;
  }
  (void)tc;
  if (m < 0 || n < 0 || lda < std::max<int64_t>(1, m)) {
    set_info(info, m < 0 ? -1 : n < 0 ? -2 : -4);
    return CUSOLVER_STATUS_SUCCESS;
  }
  return by_elem(ta, [&](auto tag) {
    using T = decltype(tag);
    CMat a;
    if (!to_cmat<T>(A, lda, n, &a)) return CUSOLVER_STATUS_EXECUTION_FAILED;
    std::vector<int> piv;
    set_info(info, vgpu_la::lu(a, (int)m, (int)n, ipiv ? &piv : nullptr));
    std::vector<int64_t> p64(piv.begin(), piv.end());
    if (ipiv && !p64.empty() &&
        cudaMemcpy(ipiv, p64.data(), p64.size() * sizeof(int64_t), cudaMemcpyHostToDevice) != cudaSuccess)
      return CUSOLVER_STATUS_EXECUTION_FAILED;
    return from_cmat<T>(A, a) ? CUSOLVER_STATUS_SUCCESS : CUSOLVER_STATUS_EXECUTION_FAILED;
  });
}
VGPU_EXPORT cusolverStatus_t cusolverDnXgetrs(cusolverDnHandle_t h, cusolverDnParams_t, cublasOperation_t trans,
                                              int64_t n, int64_t nrhs, cudaDataType ta, const void* A, int64_t lda,
                                              const int64_t* ipiv, cudaDataType tb, void* B, int64_t ldb, int* info) {
  if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;
  if (n < 0 || nrhs < 0 || lda < std::max<int64_t>(1, n) || ldb < std::max<int64_t>(1, n) ||
      (trans != CUBLAS_OP_N && trans != CUBLAS_OP_T && trans != CUBLAS_OP_C) || !is_elem(ta) || tb != ta)
    return CUSOLVER_STATUS_INVALID_VALUE;  // info untouched, as NVIDIA's leaves it
  return by_elem(ta, [&](auto tag) {
    using T = decltype(tag);
    CMat a, b;
    if (!to_cmat<T>(A, lda, n, &a) || !to_cmat<T>(B, ldb, nrhs, &b)) return CUSOLVER_STATUS_EXECUTION_FAILED;
    std::vector<int64_t> p64((size_t)n);
    if (ipiv && n && cudaMemcpy(p64.data(), ipiv, p64.size() * sizeof(int64_t), cudaMemcpyDeviceToHost) != cudaSuccess)
      return CUSOLVER_STATUS_EXECUTION_FAILED;
    std::vector<int> piv(p64.begin(), p64.end());
    vgpu_la::lu_solve(a, (int)n, ipiv ? piv.data() : nullptr, sol_op(trans), b, (int)nrhs);
    set_info(info, 0);
    return from_cmat<T>(B, b) ? CUSOLVER_STATUS_SUCCESS : CUSOLVER_STATUS_EXECUTION_FAILED;
  });
}

VGPU_EXPORT cusolverStatus_t cusolverDnXtrtri_bufferSize(cusolverDnHandle_t h, cublasFillMode_t uplo,
                                                         cublasDiagType_t diag, int64_t n, cudaDataType ta, void*,
                                                         int64_t lda, size_t* dev, size_t* host) {
  if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;
  (void)uplo, (void)diag, (void)lda;
  if (!is_elem(ta)) return CUSOLVER_STATUS_NOT_SUPPORTED;
  if (dev) *dev = x_workspace(n);
  if (host) *host = 0;
  return CUSOLVER_STATUS_SUCCESS;
}
VGPU_EXPORT cusolverStatus_t cusolverDnXtrtri(cusolverDnHandle_t h, cublasFillMode_t uplo, cublasDiagType_t diag,
                                              int64_t n, cudaDataType ta, void* A, int64_t lda, void*, size_t, void*,
                                              size_t, int* info) {
  if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;
  // info -i numbers the arguments after the handle, the data type included
  // (uplo 1, diag 2, n 3, lda 6); an element type it has none for is
  // NOT_SUPPORTED with info 0 (RTX 3060, CUDA 13.0).
  if (!is_elem(ta)) {
    set_info(info, 0);
    return CUSOLVER_STATUS_NOT_SUPPORTED;
  }
  if (!uplo_ok(uplo)) return bad_arg(1, info);
  if (diag != CUBLAS_DIAG_UNIT && diag != CUBLAS_DIAG_NON_UNIT) return bad_arg(2, info);
  if (n < 0) return bad_arg(3, info);
  if (lda < std::max<int64_t>(1, n)) return bad_arg(6, info);
  return by_elem(ta, [&](auto tag) {
    return trtri_body<decltype(tag)>(uplo == CUBLAS_FILL_MODE_UPPER, diag == CUBLAS_DIAG_UNIT, (int)n, A, (int)lda,
                                     info);
  });
}

VGPU_EXPORT cusolverStatus_t cusolverDnXsyevdx_bufferSize(cusolverDnHandle_t h, cusolverDnParams_t,
                                                          cusolverEigMode_t jobz, cusolverEigRange_t range,
                                                          cublasFillMode_t uplo, int64_t n, cudaDataType ta,
                                                          const void*, int64_t lda, void*, void*, int64_t, int64_t,
                                                          int64_t*, cudaDataType tw, const void*, cudaDataType tc,
                                                          size_t* dev, size_t* host) {
  if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;
  if (!is_elem(ta) || tw != real_of(ta) || tc != ta) return CUSOLVER_STATUS_INVALID_VALUE;
  if (!jobz_ok(jobz) || !range_ok(range) || !uplo_ok(uplo) || n < 0 || lda < std::max<int64_t>(1, n))
    return CUSOLVER_STATUS_INVALID_VALUE;
  if (dev) *dev = x_workspace(n);
  if (host) *host = 0;
  return CUSOLVER_STATUS_SUCCESS;
}
VGPU_EXPORT cusolverStatus_t cusolverDnXsyevdx(cusolverDnHandle_t h, cusolverDnParams_t, cusolverEigMode_t jobz,
                                               cusolverEigRange_t range, cublasFillMode_t uplo, int64_t n,
                                               cudaDataType ta, void* A, int64_t lda, void* vl, void* vu, int64_t il,
                                               int64_t iu, int64_t* meig, cudaDataType tw, void* W, cudaDataType tc,
                                               void*, size_t, void*, size_t, int* info) {
  if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;
  // A type it does not take: INVALID_VALUE, info 0. A bad argument: info -i
  // counting everything after the handle (params 1, jobz 2, range 3, uplo 4,
  // n 5, lda 8, vl 9, vu 10, il 11, iu 12), as an RTX 3060 (CUDA 13.0) numbers them.
  if (!is_elem(ta) || tw != real_of(ta) || tc != ta) {
    set_info(info, 0);
    return CUSOLVER_STATUS_INVALID_VALUE;
  }
  if (!jobz_ok(jobz)) return bad_arg(2, info);
  if (!range_ok(range)) return bad_arg(3, info);
  if (!uplo_ok(uplo)) return bad_arg(4, info);
  if (n < 0) return bad_arg(5, info);
  if (lda < std::max<int64_t>(1, n)) return bad_arg(8, info);
  // vl and vu are host values of W's type, read only for a value range.
  double lo = 0, hi = 0;
  if (range == CUSOLVER_EIG_RANGE_V) {
    if (!vl || !vu) return bad_arg(!vl ? 9 : 10, info);
    lo = tw == CUDA_R_32F ? *static_cast<const float*>(vl) : *static_cast<const double*>(vl);
    hi = tw == CUDA_R_32F ? *static_cast<const float*>(vu) : *static_cast<const double*>(vu);
    if (n > 0 && !(lo < hi)) return bad_arg(10, info);
  }
  if (range == CUSOLVER_EIG_RANGE_I) {
    if (il < 1 || il > std::max<int64_t>(1, n)) return bad_arg(11, info);
    if (iu < std::min<int64_t>(n, il) || iu > n) return bad_arg(12, info);
  }
  return by_elem(ta, [&](auto tag) {
    return syevdx_body<decltype(tag)>(jobz, range, uplo, (int)n, A, (int)lda, lo, hi, il, iu, meig, W, info);
  });
}

VGPU_EXPORT cusolverStatus_t cusolverDnXgesvd_bufferSize(cusolverDnHandle_t h, cusolverDnParams_t, signed char jobu,
                                                         signed char jobvt, int64_t m, int64_t n, cudaDataType ta,
                                                         const void*, int64_t lda, cudaDataType ts, const void*,
                                                         cudaDataType tu, const void*, int64_t ldu, cudaDataType tv,
                                                         const void*, int64_t ldvt, cudaDataType tc, size_t* dev,
                                                         size_t* host) {
  if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;
  if (!svd_types_ok(ta, ts, tu, tv, tc)) return CUSOLVER_STATUS_INVALID_VALUE;
  (void)jobu, (void)jobvt, (void)lda, (void)ldu, (void)ldvt;  // checked by the call, as NVIDIA's does
  if (dev) *dev = x_workspace(std::max(m, n));
  if (host) *host = 0;
  return CUSOLVER_STATUS_SUCCESS;
}
VGPU_EXPORT cusolverStatus_t cusolverDnXgesvd(cusolverDnHandle_t h, cusolverDnParams_t, signed char jobu,
                                              signed char jobvt, int64_t m64, int64_t n64, cudaDataType ta, void* A,
                                              int64_t lda, cudaDataType ts, void* S, cudaDataType tu, void* U,
                                              int64_t ldu, cudaDataType tv, void* VT, int64_t ldvt, cudaDataType tc,
                                              void*, size_t, void*, size_t, int* info) {
  if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;
  if (!svd_types_ok(ta, ts, tu, tv, tc)) return CUSOLVER_STATUS_INVALID_VALUE;
  if (const int bad = xgesvd_bad_arg(jobu, jobvt, m64, n64, lda, ldu, ldvt)) return bad_arg(bad, info);
  if (m64 < n64) {  // NVIDIA's takes m >= n only: INVALID_VALUE, info 0 (RTX 3060, CUDA 13.0)
    set_info(info, 0);
    return CUSOLVER_STATUS_INVALID_VALUE;
  }
  const int m = (int)m64, n = (int)n64, k = std::min(m, n);
  return by_elem(ta, [&](auto tag) {
    using T = decltype(tag);
    using V = Vof<T>;
    DM<V> a;
    if (!dload<T>(A, (int)lda, n, &a)) return CUSOLVER_STATUS_EXECUTION_FAILED;
    std::vector<double> s;
    DM<V> u, v;
    any_svd(a, m, n, jobu == 'A' || jobvt == 'A', &s, &u, &v);
    set_info(info, 0);
    if (!rsave_vec<Re<T>>(S, s)) return CUSOLVER_STATUS_EXECUTION_FAILED;
    DM<V> vt(n, n);  // V^H, rows as LAPACK returns them
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < (int)(v.v.size() / std::max(n, 1)) && i < n; ++i) vt(i, j) = cj(v(j, i));
    if (jobu == 'A' || jobu == 'S') {
      if (!put_block<T>(U, ldu, u, m, jobu == 'A' ? m : k)) return CUSOLVER_STATUS_EXECUTION_FAILED;
    } else if (jobu == 'O') {
      for (int j = 0; j < k; ++j)
        for (int i = 0; i < m; ++i) a(i, j) = u(i, j);
    }
    if (jobvt == 'A' || jobvt == 'S') {
      if (!put_block<T>(VT, ldvt, vt, jobvt == 'A' ? n : k, n)) return CUSOLVER_STATUS_EXECUTION_FAILED;
    } else if (jobvt == 'O') {
      for (int j = 0; j < n; ++j)
        for (int i = 0; i < k; ++i) a(i, j) = vt(i, j);
    }
    if ((jobu == 'O' || jobvt == 'O') && !dsave<T>(A, a)) return CUSOLVER_STATUS_EXECUTION_FAILED;
    return CUSOLVER_STATUS_SUCCESS;
  });
}

VGPU_EXPORT cusolverStatus_t cusolverDnXgesvdp_bufferSize(cusolverDnHandle_t h, cusolverDnParams_t,
                                                          cusolverEigMode_t jobz, int, int64_t m, int64_t n,
                                                          cudaDataType ta, const void*, int64_t lda, cudaDataType ts,
                                                          const void*, cudaDataType tu, const void*, int64_t ldu,
                                                          cudaDataType tv, const void*, int64_t ldv, cudaDataType tc,
                                                          size_t* dev, size_t* host) {
  if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;
  if (!svd_types_ok(ta, ts, tu, tv, tc)) return CUSOLVER_STATUS_INVALID_VALUE;
  if (!jobz_ok(jobz) || m < 0 || n < 0 || lda < std::max<int64_t>(1, m) || ldu < std::max<int64_t>(1, m) ||
      ldv < std::max<int64_t>(1, n))
    return CUSOLVER_STATUS_INVALID_VALUE;
  if (dev) *dev = x_workspace(std::max(m, n));
  if (host) *host = 0;
  return CUSOLVER_STATUS_SUCCESS;
}
// The SVD through a polar decomposition on NVIDIA's; here the Jacobi SVD,
// which needs no perturbation, so h_err_sigma is 0.
VGPU_EXPORT cusolverStatus_t cusolverDnXgesvdp(cusolverDnHandle_t h, cusolverDnParams_t, cusolverEigMode_t jobz,
                                               int econ, int64_t m64, int64_t n64, cudaDataType ta, void* A,
                                               int64_t lda, cudaDataType ts, void* S, cudaDataType tu, void* U,
                                               int64_t ldu, cudaDataType tv, void* V, int64_t ldv, cudaDataType tc,
                                               void*, size_t, void*, size_t, int* info, double* err_sigma) {
  if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;
  // A type it does not take: INVALID_VALUE, info untouched. A bad argument:
  // info -i, LAPACK-style (jobz 1, econ 2, m 3, n 4, lda 6, ldu 9, ldv 11);
  // ldu and ldv are checked with or without vectors (RTX 3060, CUDA 13.0).
  if (!svd_types_ok(ta, ts, tu, tv, tc)) return CUSOLVER_STATUS_INVALID_VALUE;
  if (!jobz_ok(jobz)) return bad_arg(1, info);
  if (m64 < 0) return bad_arg(3, info);
  if (n64 < 0) return bad_arg(4, info);
  if (lda < std::max<int64_t>(1, m64)) return bad_arg(6, info);
  if (ldu < std::max<int64_t>(1, m64)) return bad_arg(9, info);
  if (ldv < std::max<int64_t>(1, n64)) return bad_arg(11, info);
  const int m = (int)m64, n = (int)n64, k = std::min(m, n);
  return by_elem(ta, [&](auto tag) {
    using T = decltype(tag);
    using Vt = Vof<T>;
    DM<Vt> a;
    if (!dload<T>(A, (int)lda, n, &a)) return CUSOLVER_STATUS_EXECUTION_FAILED;
    std::vector<double> s;
    DM<Vt> u, v;
    any_svd(a, m, n, !econ, &s, &u, &v);
    set_info(info, 0);
    if (err_sigma) *err_sigma = 0.0;
    if (!rsave_vec<Re<T>>(S, s)) return CUSOLVER_STATUS_EXECUTION_FAILED;
    if (jobz == CUSOLVER_EIG_MODE_VECTOR &&
        (!put_block<T>(U, ldu, u, m, econ ? k : m) || !put_block<T>(V, ldv, v, n, econ ? k : n)))
      return CUSOLVER_STATUS_EXECUTION_FAILED;
    return CUSOLVER_STATUS_SUCCESS;
  });
}

VGPU_EXPORT cusolverStatus_t cusolverDnXgesvdr_bufferSize(cusolverDnHandle_t h, cusolverDnParams_t, signed char jobu,
                                                          signed char jobv, int64_t m, int64_t n, int64_t k, int64_t p,
                                                          int64_t niters, cudaDataType ta, const void*, int64_t lda,
                                                          cudaDataType ts, const void*, cudaDataType tu, const void*,
                                                          int64_t ldu, cudaDataType tv, const void*, int64_t ldv,
                                                          cudaDataType tc, size_t* dev, size_t* host) {
  if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;
  if (gesvdr_bad_arg(jobu, jobv, m, n, k, p, niters, lda, ldu, ldv)) return CUSOLVER_STATUS_INVALID_VALUE;
  if (!svd_types_ok(ta, ts, tu, tv, tc)) return CUSOLVER_STATUS_INVALID_VALUE;
  if (dev) *dev = x_workspace(std::max(m, n)) * 4;
  if (host) *host = 0;
  return CUSOLVER_STATUS_SUCCESS;
}
// The randomized k-SVD. A randomized method approximates the leading k
// singular triplets; here they are computed exactly (the full SVD,
// truncated), which is what the approximation converges to. p and niters
// have nothing to tune.
VGPU_EXPORT cusolverStatus_t cusolverDnXgesvdr(cusolverDnHandle_t h, cusolverDnParams_t, signed char jobu,
                                               signed char jobv, int64_t m64, int64_t n64, int64_t k64, int64_t p,
                                               int64_t niters, cudaDataType ta, void* A, int64_t lda,
                                               cudaDataType ts, void* S, cudaDataType tu, void* U, int64_t ldu,
                                               cudaDataType tv, void* V, int64_t ldv, cudaDataType tc, void*, size_t,
                                               void*, size_t, int* info) {
  if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;
  // info -i counts params as 1 (p 7, niters 8); a type it does not take is
  // INVALID_VALUE with info 0 (RTX 3060, CUDA 13.0).
  if (const int bad = gesvdr_bad_arg(jobu, jobv, m64, n64, k64, p, niters, lda, ldu, ldv)) return bad_arg(bad, info);
  if (!svd_types_ok(ta, ts, tu, tv, tc)) {
    set_info(info, 0);
    return CUSOLVER_STATUS_INVALID_VALUE;
  }
  const int m = (int)m64, n = (int)n64, k = (int)k64;
  if (k == 0 || std::min(m, n) == 0) {  // NVIDIA's accepts k = 0 and does nothing
    set_info(info, 0);
    return CUSOLVER_STATUS_SUCCESS;
  }
  return by_elem(ta, [&](auto tag) {
    using T = decltype(tag);
    using Vt = Vof<T>;
    DM<Vt> a;
    if (!dload<T>(A, (int)lda, n, &a)) return CUSOLVER_STATUS_EXECUTION_FAILED;
    std::vector<double> s;
    DM<Vt> u, v;
    any_svd(a, m, n, false, &s, &u, &v);
    s.resize((size_t)k);
    set_info(info, 0);
    if (!rsave_vec<Re<T>>(S, s)) return CUSOLVER_STATUS_EXECUTION_FAILED;
    if (jobu == 'S' && !put_block<T>(U, ldu, u, m, k)) return CUSOLVER_STATUS_EXECUTION_FAILED;
    if (jobv == 'S' && !put_block<T>(V, ldv, v, n, k)) return CUSOLVER_STATUS_EXECUTION_FAILED;
    return CUSOLVER_STATUS_SUCCESS;
  });
}

VGPU_EXPORT cusolverStatus_t cusolverDnXlarft_bufferSize(cusolverDnHandle_t h, cusolverDnParams_t,
                                                         cusolverDirectMode_t direct, cusolverStorevMode_t storev,
                                                         int64_t n, int64_t k, cudaDataType tv, const void*,
                                                         int64_t ldv, cudaDataType ttau, const void*, cudaDataType tt,
                                                         void*, int64_t ldt, cudaDataType tc, size_t* dev,
                                                         size_t* host) {
  if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;
  (void)direct, (void)storev, (void)n, (void)k, (void)ldv, (void)ldt, (void)tv, (void)ttau, (void)tt, (void)tc;
  if (dev) *dev = 0;  // NVIDIA's sizes anything (RTX 3060, CUDA 13.0)
  if (host) *host = 0;
  return CUSOLVER_STATUS_SUCCESS;
}
VGPU_EXPORT cusolverStatus_t cusolverDnXlarft(cusolverDnHandle_t h, cusolverDnParams_t, cusolverDirectMode_t direct,
                                              cusolverStorevMode_t storev, int64_t n, int64_t k, cudaDataType tv,
                                              const void* V, int64_t ldv, cudaDataType ttau, const void* tau,
                                              cudaDataType tt, void* Tm, int64_t ldt, cudaDataType tc, void*, size_t,
                                              void*, size_t) {
  if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;
  if (!larft_args_ok(direct, storev, n, k, ldv, ldt)) return CUSOLVER_STATUS_INVALID_VALUE;
  if (!is_elem(tv) || ttau != tv || tt != tv || tc != tv) return CUSOLVER_STATUS_INVALID_VALUE;
  if (k == 0) return CUSOLVER_STATUS_SUCCESS;
  return by_elem(tv, [&](auto tag) {
    using T = decltype(tag);
    using Vt = Vof<T>;
    DM<Vt> v;
    std::vector<Vt> t;
    if (!dload<T>(V, (int)ldv, (int)k, &v) || !tload_vec<T>(tau, (size_t)k, &t)) return CUSOLVER_STATUS_EXECUTION_FAILED;
    const DM<Vt> tri = vgpu_dg::larft(direct == CUBLAS_DIRECT_FORWARD, (int)n, (int)k, v, t.data());
    DM<Vt> out;
    if (!dload<T>(Tm, (int)ldt, (int)k, &out)) return CUSOLVER_STATUS_EXECUTION_FAILED;
    // The k x k block, the triangle's zeros included (NVIDIA's writes them).
    for (int j = 0; j < k; ++j)
      for (int i = 0; i < k; ++i) out(i, j) = tri(i, j);
    return dsave<T>(Tm, out) ? CUSOLVER_STATUS_SUCCESS : CUSOLVER_STATUS_EXECUTION_FAILED;
  });
}

/* ---- Jacobi settings and what the last call did ---- */

VGPU_EXPORT cusolverStatus_t cusolverDnXsyevjSetTolerance(syevjInfo_t info, double tol) {
  if (!known(info)) return CUSOLVER_STATUS_INVALID_VALUE;
  reinterpret_cast<JacobiInfo*>(info)->tol = tol;
  return CUSOLVER_STATUS_SUCCESS;
}
VGPU_EXPORT cusolverStatus_t cusolverDnXsyevjSetMaxSweeps(syevjInfo_t info, int sweeps) {
  if (!known(info)) return CUSOLVER_STATUS_INVALID_VALUE;
  reinterpret_cast<JacobiInfo*>(info)->sweeps = sweeps;
  return CUSOLVER_STATUS_SUCCESS;
}
namespace {
template <class I> cusolverStatus_t jacobi_get(cusolverDnHandle_t h, I info, double* residual, int* sweeps) {
  if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;
  if (!known(info)) return CUSOLVER_STATUS_INVALID_VALUE;
  const JacobiInfo& j = *reinterpret_cast<const JacobiInfo*>(info);
  if (j.batched) return CUSOLVER_STATUS_NOT_SUPPORTED;  // as NVIDIA's documents
  if (residual) *residual = j.residual;
  if (sweeps) *sweeps = j.executed;
  return CUSOLVER_STATUS_SUCCESS;
}
}  // namespace
VGPU_EXPORT cusolverStatus_t cusolverDnXsyevjGetResidual(cusolverDnHandle_t h, syevjInfo_t info, double* r) {
  return jacobi_get(h, info, r, nullptr);
}
VGPU_EXPORT cusolverStatus_t cusolverDnXsyevjGetSweeps(cusolverDnHandle_t h, syevjInfo_t info, int* s) {
  return jacobi_get(h, info, nullptr, s);
}
VGPU_EXPORT cusolverStatus_t cusolverDnXgesvdjGetResidual(cusolverDnHandle_t h, gesvdjInfo_t info, double* r) {
  return jacobi_get(h, info, r, nullptr);
}
VGPU_EXPORT cusolverStatus_t cusolverDnXgesvdjGetSweeps(cusolverDnHandle_t h, gesvdjInfo_t info, int* s) {
  return jacobi_get(h, info, nullptr, s);
}

/* ---- the handle's modes ----
   Kept and reported. Everything here is deterministic and runs in the
   working precision whatever they say. The enum types are newer than CUDA
   12.0's headers, so these take ints under the exported names. */

namespace {
template <class F> cusolverStatus_t with_handle(cusolverDnHandle_t h, F&& f) {
  if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;
  return f(*reinterpret_cast<Handle*>(h));
}
}  // namespace

extern "C" __attribute__((visibility("default"))) cusolverStatus_t vgpu_dn_set_det(cusolverDnHandle_t, int)
    __asm__("cusolverDnSetDeterministicMode");
extern "C" __attribute__((visibility("default"))) cusolverStatus_t vgpu_dn_get_det(cusolverDnHandle_t, int*)
    __asm__("cusolverDnGetDeterministicMode");
extern "C" __attribute__((visibility("default"))) cusolverStatus_t vgpu_dn_set_math(cusolverDnHandle_t, int)
    __asm__("cusolverDnSetMathMode");
extern "C" __attribute__((visibility("default"))) cusolverStatus_t vgpu_dn_get_math(cusolverDnHandle_t, int*)
    __asm__("cusolverDnGetMathMode");
extern "C" __attribute__((visibility("default"))) cusolverStatus_t vgpu_dn_set_emu(cusolverDnHandle_t, int)
    __asm__("cusolverDnSetEmulationStrategy");
extern "C" __attribute__((visibility("default"))) cusolverStatus_t vgpu_dn_get_emu(cusolverDnHandle_t, int*)
    __asm__("cusolverDnGetEmulationStrategy");

// CUSOLVER_DETERMINISTIC_RESULTS = 1, CUSOLVER_ALLOW_NON_DETERMINISTIC_RESULTS = 2.
extern "C" cusolverStatus_t vgpu_dn_set_det(cusolverDnHandle_t h, int mode) {
  return with_handle(h, [&](Handle& x) {
    if (mode != 1 && mode != 2) return CUSOLVER_STATUS_INTERNAL_ERROR;  // as NVIDIA's answers it
    x.det_mode = mode;
    return CUSOLVER_STATUS_SUCCESS;
  });
}
extern "C" cusolverStatus_t vgpu_dn_get_det(cusolverDnHandle_t h, int* mode) {
  return with_handle(h, [&](Handle& x) {
    if (!mode) return CUSOLVER_STATUS_INVALID_VALUE;
    *mode = x.det_mode;
    return CUSOLVER_STATUS_SUCCESS;
  });
}
// CUSOLVER_DEFAULT_MATH = 1, CUSOLVER_FP32_EMULATED_BF16X9_MATH = 2.
extern "C" cusolverStatus_t vgpu_dn_set_math(cusolverDnHandle_t h, int mode) {
  return with_handle(h, [&](Handle& x) {
    if (mode != 1 && mode != 2) return CUSOLVER_STATUS_INVALID_VALUE;
    x.math_mode = mode;
    return CUSOLVER_STATUS_SUCCESS;
  });
}
extern "C" cusolverStatus_t vgpu_dn_get_math(cusolverDnHandle_t h, int* mode) {
  return with_handle(h, [&](Handle& x) {
    if (!mode) return CUSOLVER_STATUS_INVALID_VALUE;
    *mode = x.math_mode;
    return CUSOLVER_STATUS_SUCCESS;
  });
}
// cudaEmulationStrategy_t: DEFAULT 0, PERFORMANT 1, EAGER 2.
extern "C" cusolverStatus_t vgpu_dn_set_emu(cusolverDnHandle_t h, int s) {
  return with_handle(h, [&](Handle& x) {
    if (s < 0 || s > 2) return CUSOLVER_STATUS_INVALID_VALUE;
    x.emulation = s;
    return CUSOLVER_STATUS_SUCCESS;
  });
}
extern "C" cusolverStatus_t vgpu_dn_get_emu(cusolverDnHandle_t h, int* s) {
  return with_handle(h, [&](Handle& x) {
    if (!s) return CUSOLVER_STATUS_INVALID_VALUE;
    *s = x.emulation;
    return CUSOLVER_STATUS_SUCCESS;
  });
}
