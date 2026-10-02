// The complex (C and Z) half of cuBLAS: included at the end of cublas_api.cpp,
// whose helpers (valid, fetch, store, hold, deferred_to_graph) it uses.
//
// The arithmetic is complex_linalg.hpp's, in std::complex<double>, rounded to
// the caller's precision on the way out. CUBLAS_OP_C conjugates as it
// transposes. What PyTorch's complex tensors reach: GEMM in every batched
// form, GEMV, the level-1 routines, rank-1 and rank-k updates, triangular
// multiplies and solves, Hermitian/symmetric products, and batched LU.
//
// Also geam, gelsBatched and geqrfBatched, for the real types as well (at the
// end); the rank updates her/her2/syr/syr2 and her2k/syr2k; and the real
// level-2 and level-3 routines these templates serve as well (ger, symv,
// trmv, trsv, symm, syrk, syr2k, trmm). The rot family is in cublas_api.cpp,
// with the typed level-1 Ex routines whose type handling it shares.

#include "complex_linalg.hpp"

namespace {

using vgpu_la::cd;
using vgpu_la::CMat;

inline cd to_cd(const cuComplex& v) { return {v.x, v.y}; }
inline cd to_cd(const cuDoubleComplex& v) { return {v.x, v.y}; }
// The real types too, for the routines below written once for S, D, C and Z.
inline cd to_cd(float v) { return {v, 0.0}; }
inline cd to_cd(double v) { return {v, 0.0}; }
template <class T> T from_cd(cd v);
template <> inline cuComplex from_cd<cuComplex>(cd v) { return make_cuComplex((float)v.real(), (float)v.imag()); }
template <> inline cuDoubleComplex from_cd<cuDoubleComplex>(cd v) { return make_cuDoubleComplex(v.real(), v.imag()); }
template <> inline float from_cd<float>(cd v) { return static_cast<float>(v.real()); }
template <> inline double from_cd<double>(cd v) { return v.real(); }
template <class T> using Real = std::conditional_t<std::is_same_v<T, cuComplex>, float, double>;

template <class T> std::vector<cd> load_c(const T* dev, size_t n) {
  const auto raw = fetch<T>(dev, n);
  std::vector<cd> out(n);
  for (size_t i = 0; i < n; ++i) out[i] = to_cd(raw[i]);
  return out;
}
template <class T> void store_c(T* dev, const std::vector<cd>& v) {
  std::vector<T> raw(v.size());
  for (size_t i = 0; i < v.size(); ++i) raw[i] = from_cd<T>(v[i]);
  store(dev, raw);
}
template <class T> CMat load_mat(const T* dev, int ld, int cols, int rows) {
  CMat m{load_c(dev, extent(ld, cols, rows)), ld};
  m.v.resize((size_t)ld * std::max(cols, 0));  // room for the padding after the last column
  return m;
}
template <class T> void store_mat(T* dev, const CMat& m, int cols, int rows) {
  std::vector<cd> v(m.v.begin(), m.v.begin() + extent(m.ld, cols, rows));
  store_c(dev, v);
}
template <class T> cd cscalar(cublasHandle_t h, const T* p) { return p ? to_cd(scalar(h, p)) : cd(0); }

vgpu_la::Op to_op(cublasOperation_t t) {
  return t == CUBLAS_OP_N ? vgpu_la::Op::N : t == CUBLAS_OP_T ? vgpu_la::Op::T : vgpu_la::Op::C;
}
// element (i, j) of op(A)
inline cd opel(const CMat& a, vgpu_la::Op op, int i, int j) {
  return op == vgpu_la::Op::N ? a(i, j) : vgpu_la::apply_op(a(j, i), op);
}

// Runs `body` now, or records it when the handle's stream is capturing. The
// scalars are held (copied, in host pointer mode) so a replay sees the values
// of the call, not of whatever the caller's variables hold later.
template <class Body>
cublasStatus_t run_c(cublasHandle_t h, const void* alpha, size_t abytes, const void* beta, size_t bbytes, Body body) {
  if (!valid(h)) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (deferred_to_graph(h, [=, a = hold(h, alpha, abytes), b = hold(h, beta, bbytes)] { body(a.get(), b.get()); }))
    return CUBLAS_STATUS_SUCCESS;
  return body(alpha, beta);
}
template <class Body>
cublasStatus_t run_c(cublasHandle_t h, const void* alpha, const void* beta, size_t bytes, Body body) {
  return run_c(h, alpha, bytes, beta, bytes, body);
}

// Strided vectors, as BLAS walks them: a negative increment starts at the end.
template <class T> std::vector<cd> load_vec(const T* x, int n, int inc) {
  const int a = std::abs(inc);
  auto raw = load_c(x, n > 0 ? (size_t)(n - 1) * a + 1 : 0);
  std::vector<cd> out(std::max(n, 0));
  for (int i = 0; i < n; ++i) out[i] = raw[(size_t)(inc > 0 ? i : n - 1 - i) * a];
  return out;
}
template <class T> void store_vec(T* x, int n, int inc, const std::vector<cd>& v) {
  const int a = std::abs(inc);
  auto raw = load_c(x, n > 0 ? (size_t)(n - 1) * a + 1 : 0);  // keep what lies between
  for (int i = 0; i < n; ++i) raw[(size_t)(inc > 0 ? i : n - 1 - i) * a] = v[i];
  store_c(x, raw);
}

// C = alpha op(A) op(B) + beta C over operands already on the host. C is
// read even when beta is 0, so the padding between its columns is written
// back as it was; its old values are then never used.
void cgemm_mat(vgpu_la::Op oa, vgpu_la::Op ob, int m, int n, int k, cd alpha, const CMat& a, const CMat& b, cd beta,
               CMat& c) {
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < m; ++i) {
      cd s = 0;
      for (int p = 0; p < k; ++p) s += opel(a, oa, i, p) * opel(b, ob, p, j);
      c(i, j) = beta == cd(0) ? alpha * s : alpha * s + beta * c(i, j);
    }
}

template <class T>
cublasStatus_t cgemm(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k,
                     const T* alpha, const T* A, int lda, const T* B, int ldb, const T* beta, T* C, int ldc) {
  if (!valid(h)) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (!gemm_args_ok(ta, tb, m, n, k, lda, ldb, ldc)) return CUBLAS_STATUS_INVALID_VALUE;
  if (m == 0 || n == 0) return CUBLAS_STATUS_SUCCESS;
  return run_c(h, alpha, beta, sizeof(T), [=](const void* al, const void* be) {
    const cd a_ = cscalar(h, static_cast<const T*>(al)), b_ = cscalar(h, static_cast<const T*>(be));
    const auto oa = to_op(ta), ob = to_op(tb);
    const CMat a = load_mat(A, lda, oa == vgpu_la::Op::N ? k : m, oa == vgpu_la::Op::N ? m : k);
    const CMat b = load_mat(B, ldb, ob == vgpu_la::Op::N ? n : k, ob == vgpu_la::Op::N ? k : n);
    CMat c = load_mat(C, ldc, n, m);
    cgemm_mat(oa, ob, m, n, k, a_, a, b, b_, c);
    store_mat(C, c, n, m);
    return CUBLAS_STATUS_SUCCESS;
  });
}

template <class T>
cublasStatus_t cgemv(cublasHandle_t h, cublasOperation_t t, int m, int n, const T* alpha, const T* A, int lda,
                     const T* x, int incx, const T* beta, T* y, int incy) {
  if (m < 0 || n < 0 || incx == 0 || incy == 0) return CUBLAS_STATUS_INVALID_VALUE;
  return run_c(h, alpha, beta, sizeof(T), [=](const void* al, const void* be) {
    const cd a_ = cscalar(h, static_cast<const T*>(al)), b_ = cscalar(h, static_cast<const T*>(be));
    const auto op = to_op(t);
    const int rows = op == vgpu_la::Op::N ? m : n, cols = op == vgpu_la::Op::N ? n : m;
    const CMat a = load_mat(A, lda, n, m);
    const auto xv = load_vec(x, cols, incx);
    auto yv = b_ == cd(0) ? std::vector<cd>(rows, 0) : load_vec(y, rows, incy);
    for (int i = 0; i < rows; ++i) {
      cd s = 0;
      for (int j = 0; j < cols; ++j) s += opel(a, op, i, j) * xv[j];
      yv[i] = b_ == cd(0) ? a_ * s : a_ * s + b_ * yv[i];
    }
    store_vec(y, rows, incy, yv);
    return CUBLAS_STATUS_SUCCESS;
  });
}

// A rank-1 update A += alpha x y^T (geru) or x y^H (gerc).
template <class T>
cublasStatus_t cger(cublasHandle_t h, bool conj, int m, int n, const T* alpha, const T* x, int incx, const T* y,
                    int incy, T* A, int lda) {
  if (!valid(h)) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (m < 0 || n < 0 || incx == 0 || incy == 0 || lda < std::max(1, m)) return CUBLAS_STATUS_INVALID_VALUE;
  return run_c(h, alpha, nullptr, sizeof(T), [=](const void* al, const void*) {
    const cd a_ = cscalar(h, static_cast<const T*>(al));
    const auto xv = load_vec(x, m, incx), yv = load_vec(y, n, incy);
    CMat a = load_mat(A, lda, n, m);
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < m; ++i) a(i, j) += a_ * xv[i] * (conj ? std::conj(yv[j]) : yv[j]);
    store_mat(A, a, n, m);
    return CUBLAS_STATUS_SUCCESS;
  });
}

// The full n x n matrix a Hermitian (or symmetric) triangle stands for.
CMat expand(const CMat& a, int n, bool lower, bool hermitian) {
  CMat f{std::vector<cd>((size_t)n * n), n};
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
      const bool stored = lower ? i >= j : i <= j;
      f(i, j) = stored ? a(i, j) : (hermitian ? std::conj(a(j, i)) : a(j, i));
      if (hermitian && i == j) f(i, j) = f(i, j).real();
    }
  return f;
}

// C = alpha A B + beta C (left) or alpha B A + beta C, A Hermitian or symmetric.
template <class T>
cublasStatus_t chemm(cublasHandle_t h, bool hermitian, cublasSideMode_t side, cublasFillMode_t uplo, int m, int n,
                     const T* alpha, const T* A, int lda, const T* B, int ldb, const T* beta, T* C, int ldc) {
  if (!valid(h)) return CUBLAS_STATUS_NOT_INITIALIZED;
  const bool left = side == CUBLAS_SIDE_LEFT;
  const int ka = left ? m : n;
  if (!known_side(side) || !known_fill(uplo) || m < 0 || n < 0 || lda < std::max(1, ka) || ldb < std::max(1, m) ||
      ldc < std::max(1, m))
    return CUBLAS_STATUS_INVALID_VALUE;
  if (m == 0 || n == 0) return CUBLAS_STATUS_SUCCESS;
  return run_c(h, alpha, beta, sizeof(T), [=](const void* al, const void* be) {
    const cd a_ = cscalar(h, static_cast<const T*>(al)), b_ = cscalar(h, static_cast<const T*>(be));
    const CMat a = expand(load_mat(A, lda, ka, ka), ka, uplo == CUBLAS_FILL_MODE_LOWER, hermitian);
    const CMat b = load_mat(B, ldb, n, m);
    // C is read even when beta is 0, so the padding between its columns is
    // written back as it was; its old values are then never used.
    CMat c = load_mat(C, ldc, n, m);
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < m; ++i) {
        cd s = 0;
        if (left) for (int p = 0; p < m; ++p) s += a(i, p) * b(p, j);
        else for (int p = 0; p < n; ++p) s += b(i, p) * a(p, j);
        c(i, j) = b_ == cd(0) ? a_ * s : a_ * s + b_ * c(i, j);
      }
    store_mat(C, c, n, m);
    return CUBLAS_STATUS_SUCCESS;
  });
}

// y = alpha A x + beta y, A Hermitian or symmetric.
template <class T>
cublasStatus_t chemv(cublasHandle_t h, bool hermitian, cublasFillMode_t uplo, int n, const T* alpha, const T* A,
                     int lda, const T* x, int incx, const T* beta, T* y, int incy) {
  if (!valid(h)) return CUBLAS_STATUS_NOT_INITIALIZED;
  // lda is held to n, not to max(1, n): the card takes an empty matrix with lda 0.
  if (!known_fill(uplo) || n < 0 || lda < n || incx == 0 || incy == 0) return CUBLAS_STATUS_INVALID_VALUE;
  if (n == 0) return CUBLAS_STATUS_SUCCESS;
  return run_c(h, alpha, beta, sizeof(T), [=](const void* al, const void* be) {
    const cd a_ = cscalar(h, static_cast<const T*>(al)), b_ = cscalar(h, static_cast<const T*>(be));
    const CMat a = expand(load_mat(A, lda, n, n), n, uplo == CUBLAS_FILL_MODE_LOWER, hermitian);
    const auto xv = load_vec(x, n, incx);
    auto yv = b_ == cd(0) ? std::vector<cd>(n, 0) : load_vec(y, n, incy);
    for (int i = 0; i < n; ++i) {
      cd s = 0;
      for (int j = 0; j < n; ++j) s += a(i, j) * xv[j];
      yv[i] = b_ == cd(0) ? a_ * s : a_ * s + b_ * yv[i];
    }
    store_vec(y, n, incy, yv);
    return CUBLAS_STATUS_SUCCESS;
  });
}

// C = alpha op(A) op(A)^H + beta C (herk: alpha, beta real, C Hermitian) or
// alpha op(A) op(A)^T + beta C (syrk). Only C's `uplo` triangle is written.
// Any operation but N reads A transposed: herk takes T as C, syrk C as T, as
// the card does.
void cherk_mat(bool hermitian, bool lower, bool trans, int n, int k, cd alpha, const CMat& a, cd beta, CMat& c) {
  for (int j = 0; j < n; ++j)
    for (int i = lower ? j : 0; i < (lower ? n : j + 1); ++i) {
      cd s = 0;
      for (int p = 0; p < k; ++p) {
        const cd x = trans ? (hermitian ? std::conj(a(p, i)) : a(p, i)) : a(i, p);
        const cd y = trans ? a(p, j) : (hermitian ? std::conj(a(j, p)) : a(j, p));
        s += x * y;
      }
      c(i, j) = beta == cd(0) ? alpha * s : alpha * s + beta * c(i, j);
      if (hermitian && i == j) c(i, j) = c(i, j).real();
    }
}
// What herk and syrk refuse on the card: a fill mode other than lower or upper,
// an unknown operation, a negative size or a short leading dimension.
bool herk_args_ok(cublasFillMode_t uplo, cublasOperation_t t, int n, int k, int lda, int ldc) {
  return known_fill(uplo) && known_op(t) && n >= 0 && k >= 0 && lda >= std::max(1, t == CUBLAS_OP_N ? n : k) &&
         ldc >= std::max(1, n);
}

template <class T, class S>
cublasStatus_t cherk(cublasHandle_t h, bool hermitian, cublasFillMode_t uplo, cublasOperation_t t, int n, int k,
                     const S* alpha, const T* A, int lda, const S* beta, T* C, int ldc) {
  if (!valid(h)) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (!herk_args_ok(uplo, t, n, k, lda, ldc)) return CUBLAS_STATUS_INVALID_VALUE;
  if (n == 0) return CUBLAS_STATUS_SUCCESS;
  return run_c(h, alpha, beta, sizeof(S), [=](const void* al, const void* be) {
    auto sc = [&](const void* p) -> cd {
      if constexpr (std::is_same_v<S, T>) return cscalar(h, static_cast<const T*>(p));
      else return p ? cd(scalar(h, static_cast<const S*>(p))) : cd(0);
    };
    const bool trans = t != CUBLAS_OP_N;
    const CMat a = load_mat(A, lda, trans ? n : k, trans ? k : n);
    CMat c = load_mat(C, ldc, n, n);
    cherk_mat(hermitian, uplo == CUBLAS_FILL_MODE_LOWER, trans, n, k, sc(al), a, sc(be), c);
    store_mat(C, c, n, n);
    return CUBLAS_STATUS_SUCCESS;
  });
}

// The rank-1 and rank-2 updates of a Hermitian or symmetric A, only its
// `uplo` triangle written: A += alpha x x^H (her, alpha real) or alpha x x^T
// (syr); with a y, A += alpha x y^H + conj(alpha) y x^H (her2) or alpha (x y^T
// + y x^T) (syr2). her and her2 leave the diagonal real, as BLAS does.
template <class T, class S>
cublasStatus_t csyr(cublasHandle_t h, bool hermitian, cublasFillMode_t uplo, int n, const S* alpha, const T* x,
                    int incx, const T* y, int incy, T* A, int lda) {
  if (!valid(h)) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (!known_fill(uplo) || n < 0 || incx == 0 || (y && incy == 0) || lda < std::max(1, n))
    return CUBLAS_STATUS_INVALID_VALUE;
  if (n == 0) return CUBLAS_STATUS_SUCCESS;
  return run_c(h, alpha, nullptr, sizeof(S), [=](const void* al, const void*) {
    cd a_;
    if constexpr (std::is_same_v<S, T>) a_ = cscalar(h, static_cast<const T*>(al));
    else a_ = scalar(h, static_cast<const S*>(al));
    const auto xv = load_vec(x, n, incx), yv = y ? load_vec(y, n, incy) : xv;
    auto op = [&](cd v) { return hermitian ? std::conj(v) : v; };
    CMat a = load_mat(A, lda, n, n);
    const bool lower = uplo == CUBLAS_FILL_MODE_LOWER;
    for (int j = 0; j < n; ++j)
      for (int i = lower ? j : 0; i < (lower ? n : j + 1); ++i) {
        a(i, j) += y ? a_ * xv[i] * op(yv[j]) + op(a_) * yv[i] * op(xv[j]) : a_ * xv[i] * op(xv[j]);
        if (hermitian && i == j) a(i, j) = a(i, j).real();
      }
    store_mat(A, a, n, n);
    return CUBLAS_STATUS_SUCCESS;
  });
}

// C = alpha op(A) op(B)^H + conj(alpha) op(B) op(A)^H + beta C (her2k, beta
// real, the diagonal left real) or alpha (op(A) op(B)^T + op(B) op(A)^T) +
// beta C (syr2k), only C's `uplo` triangle written. As herk and syrk, any
// operation but N transposes: her2k takes T as C, syr2k C as T.
template <class T, class S>
cublasStatus_t csyr2k(cublasHandle_t h, bool hermitian, cublasFillMode_t uplo, cublasOperation_t t, int n, int k,
                      const T* alpha, const T* A, int lda, const T* B, int ldb, const S* beta, T* C, int ldc) {
  if (!valid(h)) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (!herk_args_ok(uplo, t, n, k, lda, ldc) || ldb < std::max(1, t == CUBLAS_OP_N ? n : k))
    return CUBLAS_STATUS_INVALID_VALUE;
  if (n == 0) return CUBLAS_STATUS_SUCCESS;
  return run_c(h, alpha, sizeof(T), beta, sizeof(S), [=](const void* al, const void* be) {
    const cd a_ = cscalar(h, static_cast<const T*>(al));
    cd b_;
    if constexpr (std::is_same_v<S, T>) b_ = cscalar(h, static_cast<const T*>(be));
    else b_ = be ? cd(scalar(h, static_cast<const S*>(be))) : cd(0);
    const bool trans = t != CUBLAS_OP_N, lower = uplo == CUBLAS_FILL_MODE_LOWER;
    const CMat a = load_mat(A, lda, trans ? n : k, trans ? k : n);
    const CMat b = load_mat(B, ldb, trans ? n : k, trans ? k : n);
    auto op = [&](cd v) { return hermitian ? std::conj(v) : v; };
    auto el = [&](const CMat& m, int i, int p) { return trans ? m(p, i) : m(i, p); };   // op(M)(i, p)
    CMat c = load_mat(C, ldc, n, n);
    for (int j = 0; j < n; ++j)
      for (int i = lower ? j : 0; i < (lower ? n : j + 1); ++i) {
        cd s1 = 0, s2 = 0;
        for (int p = 0; p < k; ++p) {
          // With OP_C (her2k) the first factor is the conjugated one.
          if (trans && hermitian) {
            s1 += std::conj(el(a, i, p)) * el(b, j, p);
            s2 += std::conj(el(b, i, p)) * el(a, j, p);
          } else {
            s1 += el(a, i, p) * op(el(b, j, p));
            s2 += el(b, i, p) * op(el(a, j, p));
          }
        }
        const cd v = a_ * s1 + op(a_) * s2;
        c(i, j) = b_ == cd(0) ? v : v + b_ * c(i, j);
        if (hermitian && i == j) c(i, j) = c(i, j).real();
      }
    store_mat(C, c, n, n);
    return CUBLAS_STATUS_SUCCESS;
  });
}

// op(A) X = alpha B / X op(A) = alpha B, X over B (trsm); C = alpha op(A) B or
// alpha B op(A) (trmm, which writes C).
template <class T>
cublasStatus_t ctrsm_one(cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t t, cublasDiagType_t diag,
                         int m, int n, cd alpha, const T* A, int lda, T* B, int ldb) {
  const bool left = side == CUBLAS_SIDE_LEFT;
  const int ka = left ? m : n;
  const CMat a = load_mat(A, lda, ka, ka);
  CMat b = load_mat(B, ldb, n, m);
  vgpu_la::trsm(left, uplo == CUBLAS_FILL_MODE_LOWER, to_op(t), diag == CUBLAS_DIAG_UNIT, m, n, alpha, a, b);
  store_mat(B, b, n, m);
  return CUBLAS_STATUS_SUCCESS;
}

template <class T>
cublasStatus_t ctrsm(cublasHandle_t h, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t t,
                     cublasDiagType_t diag, int m, int n, const T* alpha, const T* A, int lda, T* B, int ldb) {
  if (!valid(h)) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (!tri3_args_ok(side, uplo, t, diag, m, n, lda, ldb)) return CUBLAS_STATUS_INVALID_VALUE;
  if (m == 0 || n == 0) return CUBLAS_STATUS_SUCCESS;
  return run_c(h, alpha, nullptr, sizeof(T), [=](const void* al, const void*) {
    return ctrsm_one(side, uplo, t, diag, m, n, cscalar(h, static_cast<const T*>(al)), A, lda, B, ldb);
  });
}

template <class T>
cublasStatus_t ctrsm_batched(cublasHandle_t h, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t t,
                             cublasDiagType_t diag, int m, int n, const T* alpha, const T* const Aarray[], int lda,
                             T* const Barray[], int ldb, int batch) {
  if (!valid(h)) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (!tri3_args_ok(side, uplo, t, diag, m, n, lda, ldb) || batch < 0) return CUBLAS_STATUS_INVALID_VALUE;
  return run_c(h, alpha, nullptr, sizeof(T), [=](const void* al, const void*) {
    const cd a_ = cscalar(h, static_cast<const T*>(al));
    const auto pa = fetch<const T*>(Aarray, (size_t)batch);
    const auto pb = fetch<T*>(Barray, (size_t)batch);
    for (int i = 0; i < batch; ++i) ctrsm_one(side, uplo, t, diag, m, n, a_, pa[i], lda, pb[i], ldb);
    return CUBLAS_STATUS_SUCCESS;
  });
}

template <class T>
cublasStatus_t ctrmm(cublasHandle_t h, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t t,
                     cublasDiagType_t diag, int m, int n, const T* alpha, const T* A, int lda, const T* B, int ldb,
                     T* C, int ldc) {
  if (!valid(h)) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (!tri3_args_ok(side, uplo, t, diag, m, n, lda, ldb) || ldc < std::max(1, m)) return CUBLAS_STATUS_INVALID_VALUE;
  if (m == 0 || n == 0) return CUBLAS_STATUS_SUCCESS;
  return run_c(h, alpha, nullptr, sizeof(T), [=](const void* al, const void*) {
    const cd a_ = cscalar(h, static_cast<const T*>(al));
    const bool left = side == CUBLAS_SIDE_LEFT, lower = uplo == CUBLAS_FILL_MODE_LOWER;
    const int ka = left ? m : n;
    const CMat a = load_mat(A, lda, ka, ka);
    const CMat b = load_mat(B, ldb, n, m);
    const auto op = to_op(t);
    auto tri = [&](int i, int j) -> cd {  // op(A)(i, j) from the named triangle
      const int r = op == vgpu_la::Op::N ? i : j, c = op == vgpu_la::Op::N ? j : i;
      if (r == c && diag == CUBLAS_DIAG_UNIT) return 1;
      if (lower ? c > r : c < r) return 0;
      return vgpu_la::apply_op(a(r, c), op);
    };
    CMat c = load_mat(C, ldc, n, m);  // for the padding between columns
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < m; ++i) {
        cd s = 0;
        if (left) for (int p = 0; p < m; ++p) s += tri(i, p) * b(p, j);
        else for (int p = 0; p < n; ++p) s += b(i, p) * tri(p, j);
        c(i, j) = a_ * s;
      }
    store_mat(C, c, n, m);
    return CUBLAS_STATUS_SUCCESS;
  });
}

// x = op(A) x (trmv) or op(A) x = b (trsv), A triangular.
template <class T>
cublasStatus_t ctrmv(cublasHandle_t h, bool solve, cublasFillMode_t uplo, cublasOperation_t t, cublasDiagType_t diag,
                     int n, const T* A, int lda, T* x, int incx) {
  if (!valid(h)) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (!known_fill(uplo) || !known_op(t) || !known_diag(diag) || n < 0 || lda < std::max(1, n) || incx == 0)
    return CUBLAS_STATUS_INVALID_VALUE;
  if (n == 0) return CUBLAS_STATUS_SUCCESS;
  return run_c(h, nullptr, nullptr, 0, [=](const void*, const void*) {
    const CMat a = load_mat(A, lda, n, n);
    auto xv = load_vec(x, n, incx);
    const bool lower = uplo == CUBLAS_FILL_MODE_LOWER, unit = diag == CUBLAS_DIAG_UNIT;
    if (solve) {
      vgpu_la::tri_solve(a, n, lower, to_op(t), unit, xv.data(), 1);
    } else {
      const auto op = to_op(t);
      std::vector<cd> out(n, 0);
      for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
          const int r = op == vgpu_la::Op::N ? i : j, c = op == vgpu_la::Op::N ? j : i;
          if (lower ? c > r : c < r) continue;
          out[i] += (r == c && unit) ? xv[j] : vgpu_la::apply_op(a(r, c), op) * xv[j];
        }
      xv = out;
    }
    store_vec(x, n, incx, xv);
    return CUBLAS_STATUS_SUCCESS;
  });
}

template <class T>
cublasStatus_t cgetrf_batched(cublasHandle_t h, int n, T* const Aarray[], int lda, int* pivots, int* infos, int batch) {
  if (n < 0 || lda < std::max(1, n) || batch < 0) return CUBLAS_STATUS_INVALID_VALUE;
  return run_c(h, nullptr, nullptr, 0, [=](const void*, const void*) {
    const auto ptrs = fetch<T*>(Aarray, (size_t)batch);
    std::vector<int> all(batch, 0);
    for (int b = 0; b < batch; ++b) {
      CMat a = load_mat(ptrs[b], lda, n, n);
      std::vector<int> ipiv;
      all[b] = vgpu_la::lu(a, n, n, pivots ? &ipiv : nullptr);
      store_mat(ptrs[b], a, n, n);
      if (pivots && n) cudaMemcpy(pivots + (size_t)b * n, ipiv.data(), n * sizeof(int), kH2D);
    }
    if (infos) store(infos, all);
    return CUBLAS_STATUS_SUCCESS;
  });
}

template <class T>
cublasStatus_t cgetrs_batched(cublasHandle_t h, cublasOperation_t t, int n, int nrhs, const T* const Aarray[], int lda,
                              const int* pivots, T* const Barray[], int ldb, int* info, int batch) {
  if (info) *info = 0;
  if (n < 0 || nrhs < 0 || lda < std::max(1, n) || ldb < std::max(1, n) || batch < 0) {
    if (info) *info = -1;
    return CUBLAS_STATUS_INVALID_VALUE;
  }
  return run_c(h, nullptr, nullptr, 0, [=](const void*, const void*) {
    const auto pa = fetch<const T*>(Aarray, (size_t)batch);
    const auto pb = fetch<T*>(Barray, (size_t)batch);
    for (int b = 0; b < batch; ++b) {
      const CMat a = load_mat(pa[b], lda, n, n);
      CMat x = load_mat(pb[b], ldb, nrhs, n);
      std::vector<int> ipiv;
      if (pivots) ipiv = fetch<int>(pivots + (size_t)b * n, (size_t)n);
      vgpu_la::lu_solve(a, n, pivots ? ipiv.data() : nullptr, to_op(t), x, nrhs);
      store_mat(pb[b], x, nrhs, n);
    }
    return CUBLAS_STATUS_SUCCESS;
  });
}

// A reduction's result goes where the pointer mode says: host or device.
template <class T> void put_result(cublasHandle_t h, T* result, cd v) {
  if (!result) return;
  if (reinterpret_cast<Handle*>(h)->pointer_mode == CUBLAS_POINTER_MODE_DEVICE) store(result, std::vector<T>{from_cd<T>(v)});
  else *result = from_cd<T>(v);
}

}  // namespace

#define VGPU_COMPLEX_BLAS(P, T, R)                                                                                \
  VGPU_EXPORT cublasStatus_t cublas##P##gemm_v2(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb,     \
                                                int m, int n, int k, const T* alpha, const T* A, int lda,         \
                                                const T* B, int ldb, const T* beta, T* C, int ldc) {              \
    return cgemm<T>(h, ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);                                     \
  }                                                                                                               \
  VGPU_EXPORT cublasStatus_t cublas##P##gemmStridedBatched(                                                       \
      cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m, int n, int k, const T* alpha,          \
      const T* A, int lda, long long sA, const T* B, int ldb, long long sB, const T* beta, T* C, int ldc,         \
      long long sC, int batch) {                                                                                  \
    if (!valid(h)) return CUBLAS_STATUS_NOT_INITIALIZED;                                                          \
    if (!gemm_args_ok(ta, tb, m, n, k, lda, ldb, ldc) || batch < 0) return CUBLAS_STATUS_INVALID_VALUE;           \
    for (int i = 0; i < batch; ++i) {                                                                             \
      const cublasStatus_t s =                                                                                    \
          cgemm<T>(h, ta, tb, m, n, k, alpha, A + i * sA, lda, B + i * sB, ldb, beta, C + i * sC, ldc);           \
      if (s != CUBLAS_STATUS_SUCCESS) return s;                                                                   \
    }                                                                                                             \
    return CUBLAS_STATUS_SUCCESS;                                                                                 \
  }                                                                                                               \
  VGPU_EXPORT cublasStatus_t cublas##P##gemmBatched(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, \
                                                    int m, int n, int k, const T* alpha, const T* const A[],      \
                                                    int lda, const T* const B[], int ldb, const T* beta,          \
                                                    T* const C[], int ldc, int batch) {                           \
    if (!valid(h)) return CUBLAS_STATUS_NOT_INITIALIZED;                                                          \
    if (!gemm_args_ok(ta, tb, m, n, k, lda, ldb, ldc) || batch < 0) return CUBLAS_STATUS_INVALID_VALUE;           \
    if (batch == 0) return CUBLAS_STATUS_SUCCESS;                                                                 \
    return run_c(h, alpha, beta, sizeof(T), [=](const void* al, const void* be) {                                 \
      const auto a = fetch<const T*>(A, (size_t)batch);                                                           \
      const auto b = fetch<const T*>(B, (size_t)batch);                                                           \
      const auto c = fetch<T*>(C, (size_t)batch);                                                                 \
      for (int i = 0; i < batch; ++i) {                                                                           \
        const cublasStatus_t s = cgemm<T>(h, ta, tb, m, n, k, static_cast<const T*>(al), a[i], lda, b[i], ldb,    \
                                          static_cast<const T*>(be), c[i], ldc);                                  \
        if (s != CUBLAS_STATUS_SUCCESS) return s;                                                                 \
      }                                                                                                           \
      return CUBLAS_STATUS_SUCCESS;                                                                               \
    });                                                                                                           \
  }                                                                                                               \
  VGPU_EXPORT cublasStatus_t cublas##P##gemv_v2(cublasHandle_t h, cublasOperation_t t, int m, int n,              \
                                                const T* alpha, const T* A, int lda, const T* x, int incx,        \
                                                const T* beta, T* y, int incy) {                                  \
    return cgemv<T>(h, t, m, n, alpha, A, lda, x, incx, beta, y, incy);                                           \
  }                                                                                                               \
  VGPU_EXPORT cublasStatus_t cublas##P##geru_v2(cublasHandle_t h, int m, int n, const T* alpha, const T* x,       \
                                                int incx, const T* y, int incy, T* A, int lda) {                  \
    return cger<T>(h, false, m, n, alpha, x, incx, y, incy, A, lda);                                              \
  }                                                                                                               \
  VGPU_EXPORT cublasStatus_t cublas##P##gerc_v2(cublasHandle_t h, int m, int n, const T* alpha, const T* x,       \
                                                int incx, const T* y, int incy, T* A, int lda) {                  \
    return cger<T>(h, true, m, n, alpha, x, incx, y, incy, A, lda);                                               \
  }                                                                                                               \
  VGPU_EXPORT cublasStatus_t cublas##P##hemm_v2(cublasHandle_t h, cublasSideMode_t s, cublasFillMode_t u, int m,  \
                                                int n, const T* alpha, const T* A, int lda, const T* B, int ldb,  \
                                                const T* beta, T* C, int ldc) {                                   \
    return chemm<T>(h, true, s, u, m, n, alpha, A, lda, B, ldb, beta, C, ldc);                                    \
  }                                                                                                               \
  VGPU_EXPORT cublasStatus_t cublas##P##symm_v2(cublasHandle_t h, cublasSideMode_t s, cublasFillMode_t u, int m,  \
                                                int n, const T* alpha, const T* A, int lda, const T* B, int ldb,  \
                                                const T* beta, T* C, int ldc) {                                   \
    return chemm<T>(h, false, s, u, m, n, alpha, A, lda, B, ldb, beta, C, ldc);                                   \
  }                                                                                                               \
  VGPU_EXPORT cublasStatus_t cublas##P##hemv_v2(cublasHandle_t h, cublasFillMode_t u, int n, const T* alpha,      \
                                                const T* A, int lda, const T* x, int incx, const T* beta, T* y,   \
                                                int incy) {                                                       \
    return chemv<T>(h, true, u, n, alpha, A, lda, x, incx, beta, y, incy);                                        \
  }                                                                                                               \
  VGPU_EXPORT cublasStatus_t cublas##P##symv_v2(cublasHandle_t h, cublasFillMode_t u, int n, const T* alpha,      \
                                                const T* A, int lda, const T* x, int incx, const T* beta, T* y,   \
                                                int incy) {                                                       \
    return chemv<T>(h, false, u, n, alpha, A, lda, x, incx, beta, y, incy);                                       \
  }                                                                                                               \
  VGPU_EXPORT cublasStatus_t cublas##P##herk_v2(cublasHandle_t h, cublasFillMode_t u, cublasOperation_t t, int n, \
                                                int k, const R* alpha, const T* A, int lda, const R* beta, T* C,  \
                                                int ldc) {                                                        \
    return cherk<T, R>(h, true, u, t, n, k, alpha, A, lda, beta, C, ldc);                                         \
  }                                                                                                               \
  VGPU_EXPORT cublasStatus_t cublas##P##syrk_v2(cublasHandle_t h, cublasFillMode_t u, cublasOperation_t t, int n, \
                                                int k, const T* alpha, const T* A, int lda, const T* beta, T* C,  \
                                                int ldc) {                                                        \
    return cherk<T, T>(h, false, u, t, n, k, alpha, A, lda, beta, C, ldc);                                        \
  }                                                                                                               \
  VGPU_EXPORT cublasStatus_t cublas##P##trsm_v2(cublasHandle_t h, cublasSideMode_t s, cublasFillMode_t u,         \
                                                cublasOperation_t t, cublasDiagType_t d, int m, int n,            \
                                                const T* alpha, const T* A, int lda, T* B, int ldb) {             \
    return ctrsm<T>(h, s, u, t, d, m, n, alpha, A, lda, B, ldb);                                                  \
  }                                                                                                               \
  VGPU_EXPORT cublasStatus_t cublas##P##trsmBatched(cublasHandle_t h, cublasSideMode_t s, cublasFillMode_t u,     \
                                                    cublasOperation_t t, cublasDiagType_t d, int m, int n,        \
                                                    const T* alpha, const T* const A[], int lda, T* const B[],    \
                                                    int ldb, int batch) {                                         \
    return ctrsm_batched<T>(h, s, u, t, d, m, n, alpha, A, lda, B, ldb, batch);                                   \
  }                                                                                                               \
  VGPU_EXPORT cublasStatus_t cublas##P##trmm_v2(cublasHandle_t h, cublasSideMode_t s, cublasFillMode_t u,         \
                                                cublasOperation_t t, cublasDiagType_t d, int m, int n,            \
                                                const T* alpha, const T* A, int lda, const T* B, int ldb, T* C,   \
                                                int ldc) {                                                        \
    return ctrmm<T>(h, s, u, t, d, m, n, alpha, A, lda, B, ldb, C, ldc);                                          \
  }                                                                                                               \
  VGPU_EXPORT cublasStatus_t cublas##P##trmv_v2(cublasHandle_t h, cublasFillMode_t u, cublasOperation_t t,        \
                                                cublasDiagType_t d, int n, const T* A, int lda, T* x, int incx) { \
    return ctrmv<T>(h, false, u, t, d, n, A, lda, x, incx);                                                       \
  }                                                                                                               \
  VGPU_EXPORT cublasStatus_t cublas##P##trsv_v2(cublasHandle_t h, cublasFillMode_t u, cublasOperation_t t,        \
                                                cublasDiagType_t d, int n, const T* A, int lda, T* x, int incx) { \
    return ctrmv<T>(h, true, u, t, d, n, A, lda, x, incx);                                                        \
  }                                                                                                               \
  VGPU_EXPORT cublasStatus_t cublas##P##getrfBatched(cublasHandle_t h, int n, T* const A[], int lda, int* P,      \
                                                     int* info, int batch) {                                      \
    return cgetrf_batched<T>(h, n, A, lda, P, info, batch);                                                       \
  }                                                                                                               \
  VGPU_EXPORT cublasStatus_t cublas##P##getrsBatched(cublasHandle_t h, cublasOperation_t t, int n, int nrhs,      \
                                                     const T* const A[], int lda, const int* P, T* const B[],     \
                                                     int ldb, int* info, int batch) {                             \
    return cgetrs_batched<T>(h, t, n, nrhs, A, lda, P, B, ldb, info, batch);                                      \
  }                                                                                                               \
  VGPU_EXPORT cublasStatus_t cublas##P##her_v2(cublasHandle_t h, cublasFillMode_t u, int n, const R* alpha,       \
                                               const T* x, int incx, T* A, int lda) {                             \
    return csyr<T, R>(h, true, u, n, alpha, x, incx, nullptr, 0, A, lda);                                         \
  }                                                                                                               \
  VGPU_EXPORT cublasStatus_t cublas##P##her2_v2(cublasHandle_t h, cublasFillMode_t u, int n, const T* alpha,      \
                                                const T* x, int incx, const T* y, int incy, T* A, int lda) {      \
    return csyr<T, T>(h, true, u, n, alpha, x, incx, y, incy, A, lda);                                            \
  }                                                                                                               \
  VGPU_EXPORT cublasStatus_t cublas##P##her2k_v2(cublasHandle_t h, cublasFillMode_t u, cublasOperation_t t,       \
                                                 int n, int k, const T* alpha, const T* A, int lda, const T* B,   \
                                                 int ldb, const R* beta, T* C, int ldc) {                         \
    return csyr2k<T, R>(h, true, u, t, n, k, alpha, A, lda, B, ldb, beta, C, ldc);                                \
  }                                                                                                               \
  /* level 1 */                                                                                                   \
  VGPU_EXPORT cublasStatus_t cublas##P##axpy_v2(cublasHandle_t h, int n, const T* alpha, const T* x, int incx,    \
                                                T* y, int incy) {                                                 \
    if (!valid(h)) return CUBLAS_STATUS_NOT_INITIALIZED;                                                          \
    if (n <= 0) return CUBLAS_STATUS_SUCCESS;                                                                     \
    return run_c(h, alpha, nullptr, sizeof(T), [=](const void* al, const void*) {                                 \
      const cd a_ = cscalar(h, static_cast<const T*>(al));                                                        \
      const auto xv = load_vec(x, n, incx);                                                                       \
      auto yv = load_vec(y, n, incy);                                                                             \
      for (int i = 0; i < n; ++i) yv[i] += a_ * xv[i];                                                            \
      store_vec(y, n, incy, yv);                                                                                  \
      return CUBLAS_STATUS_SUCCESS;                                                                               \
    });                                                                                                           \
  }                                                                                                               \
  VGPU_EXPORT cublasStatus_t cublas##P##scal_v2(cublasHandle_t h, int n, const T* alpha, T* x, int incx) {        \
    if (!valid(h)) return CUBLAS_STATUS_NOT_INITIALIZED;                                                          \
    if (n <= 0 || incx <= 0) return CUBLAS_STATUS_SUCCESS;                                                        \
    return run_c(h, alpha, nullptr, sizeof(T), [=](const void* al, const void*) {                                 \
      const cd a_ = cscalar(h, static_cast<const T*>(al));                                                        \
      auto xv = load_vec(x, n, incx);                                                                             \
      for (auto& v : xv) v *= a_;                                                                                 \
      store_vec(x, n, incx, xv);                                                                                  \
      return CUBLAS_STATUS_SUCCESS;                                                                               \
    });                                                                                                           \
  }                                                                                                               \
  VGPU_EXPORT cublasStatus_t cublas##P##copy_v2(cublasHandle_t h, int n, const T* x, int incx, T* y, int incy) {  \
    if (!valid(h)) return CUBLAS_STATUS_NOT_INITIALIZED;                                                          \
    if (n <= 0) return CUBLAS_STATUS_SUCCESS;                                                                     \
    return run_c(h, nullptr, nullptr, 0, [=](const void*, const void*) {                                          \
      store_vec(y, n, incy, load_vec(x, n, incx));                                                                \
      return CUBLAS_STATUS_SUCCESS;                                                                               \
    });                                                                                                           \
  }                                                                                                               \
  VGPU_EXPORT cublasStatus_t cublas##P##swap_v2(cublasHandle_t h, int n, T* x, int incx, T* y, int incy) {        \
    if (!valid(h)) return CUBLAS_STATUS_NOT_INITIALIZED;                                                          \
    if (n <= 0) return CUBLAS_STATUS_SUCCESS;                                                                     \
    return run_c(h, nullptr, nullptr, 0, [=](const void*, const void*) {                                          \
      const auto xv = load_vec(x, n, incx), yv = load_vec(y, n, incy);                                            \
      store_vec(x, n, incx, yv);                                                                                  \
      store_vec(y, n, incy, xv);                                                                                  \
      return CUBLAS_STATUS_SUCCESS;                                                                               \
    });                                                                                                           \
  }                                                                                                               \
  VGPU_EXPORT cublasStatus_t cublas##P##dotu_v2(cublasHandle_t h, int n, const T* x, int incx, const T* y,        \
                                                int incy, T* result) {                                            \
    if (!valid(h)) return CUBLAS_STATUS_NOT_INITIALIZED;                                                          \
    if (host_result_under_capture(h)) return CUBLAS_STATUS_NOT_SUPPORTED;                                         \
    return run_c(h, nullptr, nullptr, 0, [=](const void*, const void*) {                                          \
      const auto xv = load_vec(x, n, incx), yv = load_vec(y, n, incy);                                            \
      cd s = 0;                                                                                                   \
      for (int i = 0; i < n; ++i) s += xv[i] * yv[i];                                                             \
      put_result(h, result, s);                                                                                   \
      return CUBLAS_STATUS_SUCCESS;                                                                               \
    });                                                                                                           \
  }                                                                                                               \
  VGPU_EXPORT cublasStatus_t cublas##P##dotc_v2(cublasHandle_t h, int n, const T* x, int incx, const T* y,        \
                                                int incy, T* result) {                                            \
    if (!valid(h)) return CUBLAS_STATUS_NOT_INITIALIZED;                                                          \
    if (host_result_under_capture(h)) return CUBLAS_STATUS_NOT_SUPPORTED;                                         \
    return run_c(h, nullptr, nullptr, 0, [=](const void*, const void*) {                                          \
      const auto xv = load_vec(x, n, incx), yv = load_vec(y, n, incy);                                            \
      cd s = 0;                                                                                                   \
      for (int i = 0; i < n; ++i) s += std::conj(xv[i]) * yv[i];                                                  \
      put_result(h, result, s);                                                                                   \
      return CUBLAS_STATUS_SUCCESS;                                                                               \
    });                                                                                                           \
  }
VGPU_COMPLEX_BLAS(C, cuComplex, float)
VGPU_COMPLEX_BLAS(Z, cuDoubleComplex, double)
#undef VGPU_COMPLEX_BLAS

// The symmetric rank updates, for all four types.
#define VGPU_SYR(P, T)                                                                                            \
  VGPU_EXPORT cublasStatus_t cublas##P##syr_v2(cublasHandle_t h, cublasFillMode_t u, int n, const T* alpha,       \
                                               const T* x, int incx, T* A, int lda) {                             \
    return csyr<T, T>(h, false, u, n, alpha, x, incx, nullptr, 0, A, lda);                                        \
  }                                                                                                               \
  VGPU_EXPORT cublasStatus_t cublas##P##syr2_v2(cublasHandle_t h, cublasFillMode_t u, int n, const T* alpha,      \
                                                const T* x, int incx, const T* y, int incy, T* A, int lda) {      \
    return csyr<T, T>(h, false, u, n, alpha, x, incx, y, incy, A, lda);                                           \
  }                                                                                                               \
  VGPU_EXPORT cublasStatus_t cublas##P##syr2k_v2(cublasHandle_t h, cublasFillMode_t u, cublasOperation_t t,       \
                                                 int n, int k, const T* alpha, const T* A, int lda, const T* B,   \
                                                 int ldb, const T* beta, T* C, int ldc) {                         \
    return csyr2k<T, T>(h, false, u, t, n, k, alpha, A, lda, B, ldb, beta, C, ldc);                               \
  }
VGPU_SYR(S, float)
VGPU_SYR(D, double)
VGPU_SYR(C, cuComplex)
VGPU_SYR(Z, cuDoubleComplex)
#undef VGPU_SYR

// The real level-2 and level-3 routines the complex templates also serve: a
// real matrix is a complex one with no imaginary part, and OP_C is OP_T. Each
// is computed in double and rounded once, so it agrees with the card to the
// last bit or two of single precision (the card's own order of summation
// aside), and exactly on small integers.
#define VGPU_REAL_BLAS(P, T)                                                                                      \
  VGPU_EXPORT cublasStatus_t cublas##P##ger_v2(cublasHandle_t h, int m, int n, const T* alpha, const T* x,        \
                                               int incx, const T* y, int incy, T* A, int lda) {                   \
    return cger<T>(h, false, m, n, alpha, x, incx, y, incy, A, lda);                                              \
  }                                                                                                               \
  VGPU_EXPORT cublasStatus_t cublas##P##symv_v2(cublasHandle_t h, cublasFillMode_t u, int n, const T* alpha,      \
                                                const T* A, int lda, const T* x, int incx, const T* beta, T* y,   \
                                                int incy) {                                                       \
    return chemv<T>(h, false, u, n, alpha, A, lda, x, incx, beta, y, incy);                                       \
  }                                                                                                               \
  VGPU_EXPORT cublasStatus_t cublas##P##trmv_v2(cublasHandle_t h, cublasFillMode_t u, cublasOperation_t t,        \
                                                cublasDiagType_t d, int n, const T* A, int lda, T* x, int incx) { \
    return ctrmv<T>(h, false, u, t, d, n, A, lda, x, incx);                                                       \
  }                                                                                                               \
  VGPU_EXPORT cublasStatus_t cublas##P##trsv_v2(cublasHandle_t h, cublasFillMode_t u, cublasOperation_t t,        \
                                                cublasDiagType_t d, int n, const T* A, int lda, T* x, int incx) { \
    return ctrmv<T>(h, true, u, t, d, n, A, lda, x, incx);                                                        \
  }                                                                                                               \
  VGPU_EXPORT cublasStatus_t cublas##P##symm_v2(cublasHandle_t h, cublasSideMode_t s, cublasFillMode_t u, int m,  \
                                                int n, const T* alpha, const T* A, int lda, const T* B, int ldb,  \
                                                const T* beta, T* C, int ldc) {                                   \
    return chemm<T>(h, false, s, u, m, n, alpha, A, lda, B, ldb, beta, C, ldc);                                   \
  }                                                                                                               \
  VGPU_EXPORT cublasStatus_t cublas##P##syrk_v2(cublasHandle_t h, cublasFillMode_t u, cublasOperation_t t, int n, \
                                                int k, const T* alpha, const T* A, int lda, const T* beta, T* C,  \
                                                int ldc) {                                                        \
    return cherk<T, T>(h, false, u, t, n, k, alpha, A, lda, beta, C, ldc);                                        \
  }                                                                                                               \
  VGPU_EXPORT cublasStatus_t cublas##P##trmm_v2(cublasHandle_t h, cublasSideMode_t s, cublasFillMode_t u,         \
                                                cublasOperation_t t, cublasDiagType_t d, int m, int n,            \
                                                const T* alpha, const T* A, int lda, const T* B, int ldb, T* C,   \
                                                int ldc) {                                                        \
    return ctrmm<T>(h, s, u, t, d, m, n, alpha, A, lda, B, ldb, C, ldc);                                          \
  }
VGPU_REAL_BLAS(S, float)
VGPU_REAL_BLAS(D, double)
#undef VGPU_REAL_BLAS

// Scaling a complex vector by a real scalar.
VGPU_EXPORT cublasStatus_t cublasCsscal_v2(cublasHandle_t h, int n, const float* alpha, cuComplex* x, int incx) {
  if (!valid(h)) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (n <= 0 || incx <= 0) return CUBLAS_STATUS_SUCCESS;
  return run_c(h, alpha, nullptr, sizeof(float), [=](const void* al, const void*) {
    const double a_ = scalar(h, static_cast<const float*>(al));
    auto xv = load_vec(x, n, incx);
    for (auto& v : xv) v *= a_;
    store_vec(x, n, incx, xv);
    return CUBLAS_STATUS_SUCCESS;
  });
}
VGPU_EXPORT cublasStatus_t cublasZdscal_v2(cublasHandle_t h, int n, const double* alpha, cuDoubleComplex* x,
                                           int incx) {
  if (!valid(h)) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (n <= 0 || incx <= 0) return CUBLAS_STATUS_SUCCESS;
  return run_c(h, alpha, nullptr, sizeof(double), [=](const void* al, const void*) {
    const double a_ = scalar(h, static_cast<const double*>(al));
    auto xv = load_vec(x, n, incx);
    for (auto& v : xv) v *= a_;
    store_vec(x, n, incx, xv);
    return CUBLAS_STATUS_SUCCESS;
  });
}

/* ---- geam, gelsBatched and geqrfBatched, for all four types ----
   Written once over std::complex<double>, as the rest of this file is; a real
   matrix is a complex one with no imaginary part, and comes out rounded to its
   own precision. The answers to bad arguments are an RTX 3060's. */
namespace {

// C = alpha op(A) + beta op(B), all m x n. Each op is N, T or C (C is T for
// the real types). Only what a zero scalar multiplies goes unread: B may be
// NULL when beta is 0, and NaN in it stays out of C. C may be A, or B, in
// place, provided that operand is not transposed and has C's leading
// dimension; otherwise INVALID_VALUE, as for a leading dimension shorter than
// its matrix's rows.
template <class T>
cublasStatus_t geam(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m, int n, const T* alpha,
                    const T* A, int lda, const T* beta, const T* B, int ldb, T* C, int ldc) {
  auto known = [](cublasOperation_t t) { return t == CUBLAS_OP_N || t == CUBLAS_OP_T || t == CUBLAS_OP_C; };
  if (!valid(h)) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (!known(ta) || !known(tb) || m < 0 || n < 0) return CUBLAS_STATUS_INVALID_VALUE;
  const int rows_a = ta == CUBLAS_OP_N ? m : n, cols_a = ta == CUBLAS_OP_N ? n : m;
  const int rows_b = tb == CUBLAS_OP_N ? m : n, cols_b = tb == CUBLAS_OP_N ? n : m;
  if (lda < std::max(1, rows_a) || ldb < std::max(1, rows_b) || ldc < std::max(1, m))
    return CUBLAS_STATUS_INVALID_VALUE;
  if ((C == A && (ta != CUBLAS_OP_N || lda != ldc)) || (C == B && (tb != CUBLAS_OP_N || ldb != ldc)))
    return CUBLAS_STATUS_INVALID_VALUE;
  if (m == 0 || n == 0) return CUBLAS_STATUS_SUCCESS;
  return run_c(h, alpha, beta, sizeof(T), [=](const void* al, const void* be) {
    const cd a_ = cscalar(h, static_cast<const T*>(al)), b_ = cscalar(h, static_cast<const T*>(be));
    const auto oa = to_op(ta), ob = to_op(tb);
    const bool use_a = a_ != cd(0), use_b = b_ != cd(0);
    CMat a, b;
    if (use_a) a = load_mat(A, lda, cols_a, rows_a);
    if (use_b) b = load_mat(B, ldb, cols_b, rows_b);
    CMat c = load_mat(C, ldc, n, m);   // the rows past m in each column stay as they are
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < m; ++i)
        c(i, j) = (use_a ? a_ * opel(a, oa, i, j) : cd(0)) + (use_b ? b_ * opel(b, ob, i, j) : cd(0));
    store_mat(C, c, n, m);
    return CUBLAS_STATUS_SUCCESS;
  });
}

// Least squares, min |C - A X|, for each of `batch` overdetermined systems
// (m >= n), by Householder QR as LAPACK's gels does it: A is left holding its
// QR factorization (R, and geqrf's reflectors below it), and C its first n
// rows X and below them the rest of Q^H C. Only CUBLAS_OP_N, and only m >= n,
// as documented; either is NOT_SUPPORTED. A bad argument is INVALID_VALUE
// with *info set to minus its position as the card numbers it (m -1, nrhs -2,
// n -3, lda -5, ldc -7, the batch size -8); a NULL info is INVALID_VALUE
// itself. devInfoArray, which may be NULL, gets for each system 0, or the
// 1-based index of the first zero on R's diagonal -- a rank-deficient A,
// whose X is then not finite, on the card as here.
template <class T>
cublasStatus_t gels_batched(cublasHandle_t h, cublasOperation_t trans, int m, int n, int nrhs, T* const Aarray[],
                            int lda, T* const Carray[], int ldc, int* info, int* devInfo, int batch) {
  if (!valid(h)) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (!info) return CUBLAS_STATUS_INVALID_VALUE;
  *info = m < 0 ? -1 : nrhs < 0 ? -2 : n < 0 ? -3 : lda < std::max(1, m) ? -5 : ldc < std::max(1, m) ? -7
        : batch < 0 ? -8 : 0;
  if (*info) return CUBLAS_STATUS_INVALID_VALUE;
  if (trans != CUBLAS_OP_N || m < n) return CUBLAS_STATUS_NOT_SUPPORTED;
  if (n == 0 || batch == 0) return CUBLAS_STATUS_SUCCESS;
  if (deferred_to_graph(h, [=] {
        int ignored;
        gels_batched<T>(h, trans, m, n, nrhs, Aarray, lda, Carray, ldc, &ignored, devInfo, batch);
      }))
    return CUBLAS_STATUS_SUCCESS;
  const auto pa = fetch<T*>(Aarray, (size_t)batch);
  const auto pc = fetch<T*>(Carray, (size_t)batch);
  std::vector<int> rank(batch, 0);
  for (int b = 0; b < batch; ++b) {
    CMat a = load_mat(pa[b], lda, n, m);
    std::vector<cd> tau;
    vgpu_la::geqrf(a, m, n, &tau);
    for (int j = 0; j < n && !rank[b]; ++j)
      if (a(j, j) == cd(0)) rank[b] = j + 1;
    store_mat(pa[b], a, n, m);
    if (nrhs == 0) continue;
    CMat c = load_mat(pc[b], ldc, nrhs, m);
    vgpu_la::unmqr(true, vgpu_la::Op::C, m, nrhs, n, a, tau.data(), c);
    for (int col = 0; col < nrhs; ++col)
      vgpu_la::tri_solve(a, n, false, vgpu_la::Op::N, false, &c(0, col), 1);
    store_mat(pc[b], c, nrhs, m);
  }
  if (devInfo) store(devInfo, rank);
  return CUBLAS_STATUS_SUCCESS;
}

// QR, A = Q R, for each of `batch` m x n matrices, as LAPACK's geqrf
// leaves it: R on and above the diagonal, the Householder vectors below it
// (v(0) = 1 implied) and their min(m, n) scalars in the matrix's Tau array.
// m < n is fine. A bad argument is INVALID_VALUE with *info minus its
// position as the card numbers it (m -1, n -2, lda -4, the batch size -7);
// a NULL info is INVALID_VALUE itself.
template <class T>
cublasStatus_t geqrf_batched(cublasHandle_t h, int m, int n, T* const Aarray[], int lda, T* const TauArray[],
                             int* info, int batch) {
  if (!valid(h)) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (!info) return CUBLAS_STATUS_INVALID_VALUE;
  *info = m < 0 ? -1 : n < 0 ? -2 : lda < std::max(1, m) ? -4 : batch < 0 ? -7 : 0;
  if (*info) return CUBLAS_STATUS_INVALID_VALUE;
  if (m == 0 || n == 0 || batch == 0) return CUBLAS_STATUS_SUCCESS;
  if (deferred_to_graph(h, [=] {
        int ignored;
        geqrf_batched<T>(h, m, n, Aarray, lda, TauArray, &ignored, batch);
      }))
    return CUBLAS_STATUS_SUCCESS;
  const auto pa = fetch<T*>(Aarray, (size_t)batch);
  const auto pt = fetch<T*>(TauArray, (size_t)batch);
  for (int b = 0; b < batch; ++b) {
    CMat a = load_mat(pa[b], lda, n, m);
    std::vector<cd> tau;
    vgpu_la::geqrf(a, m, n, &tau);
    store_mat(pa[b], a, n, m);
    store_c(pt[b], tau);
  }
  return CUBLAS_STATUS_SUCCESS;
}

bool fits_int64(int64_t v) { return v >= INT32_MIN && v <= INT32_MAX; }

}  // namespace

#define VGPU_GEAM_GELS(P, T)                                                                                      \
  VGPU_EXPORT cublasStatus_t cublas##P##geam(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m,  \
                                             int n, const T* alpha, const T* A, int lda, const T* beta,          \
                                             const T* B, int ldb, T* C, int ldc) {                                \
    return geam<T>(h, ta, tb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);                                         \
  }                                                                                                               \
  VGPU_EXPORT cublasStatus_t cublas##P##geam_64(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb,      \
                                                int64_t m, int64_t n, const T* alpha, const T* A, int64_t lda,    \
                                                const T* beta, const T* B, int64_t ldb, T* C, int64_t ldc) {      \
    if (!fits_int64(m) || !fits_int64(n) || !fits_int64(lda) || !fits_int64(ldb) || !fits_int64(ldc))            \
      return CUBLAS_STATUS_NOT_SUPPORTED;                                                                         \
    return geam<T>(h, ta, tb, (int)m, (int)n, alpha, A, (int)lda, beta, B, (int)ldb, C, (int)ldc);                \
  }                                                                                                               \
  VGPU_EXPORT cublasStatus_t cublas##P##gelsBatched(cublasHandle_t h, cublasOperation_t trans, int m, int n,       \
                                                    int nrhs, T* const Aarray[], int lda, T* const Carray[],      \
                                                    int ldc, int* info, int* devInfoArray, int batchSize) {       \
    return gels_batched<T>(h, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info, devInfoArray, batchSize);        \
  }                                                                                                               \
  VGPU_EXPORT cublasStatus_t cublas##P##geqrfBatched(cublasHandle_t h, int m, int n, T* const Aarray[], int lda,   \
                                                     T* const TauArray[], int* info, int batchSize) {             \
    return geqrf_batched<T>(h, m, n, Aarray, lda, TauArray, info, batchSize);                                     \
  }
VGPU_GEAM_GELS(S, float)
VGPU_GEAM_GELS(D, double)
VGPU_GEAM_GELS(C, cuComplex)
VGPU_GEAM_GELS(Z, cuDoubleComplex)
#undef VGPU_GEAM_GELS
