// The band and packed level-2 routines of cuBLAS, in every type each comes
// in: gbmv, sbmv/hbmv, spmv/hpmv, spr/hpr, spr2/hpr2, tbmv, tbsv, tpmv and
// tpsv, and the packed/full conversions tpttr and trttp. Included at the end
// of cublas_complex.inc, whose helpers (run_c, load_vec, cscalar, CMat) these
// use; like the routines there, each is written once over
// std::complex<double>, a real matrix being a complex one with no imaginary
// part (and OP_C being OP_T), and the answer is rounded once into the
// caller's precision.
//
// Storage is reference BLAS's:
//   - a general band matrix (gbmv), m x n with kl sub- and ku
//     super-diagonals, keeps A(i, j) at row ku + i - j of column j;
//   - a symmetric, Hermitian or triangular band matrix with k off-diagonals
//     keeps A(i, j) at row k + i - j of column j from the upper triangle, at
//     row i - j from the lower;
//   - a packed triangle keeps the columns of the triangle one after another:
//     A(i, j) at i + j (j + 1) / 2 from the upper (i <= j), at
//     i + (2n - j - 1) j / 2 from the lower (i >= j).
// The Hermitian forms read only the real part of the diagonal, and the rank
// updates hpr and hpr2 leave it real, as BLAS does.
//
// The arguments each refuses are an RTX 3060's (cuBLAS 13.0), measured: an
// unknown fill mode, operation or diagonal, a negative size or band width, a
// zero increment, and a band leading dimension below its band's rows (kl + ku
// + 1, or k + 1), all INVALID_VALUE; then an empty problem is done, and
// only then is a NULL alpha or beta INVALID_VALUE (a NULL alpha with m = 0
// succeeds on the card).

namespace {

// Where a packed triangle keeps A(i, j), i and j inside the triangle.
inline size_t packed_at(bool lower, int n, int i, int j) {
  return lower ? (size_t)i + ((size_t)(2 * n - j - 1) * j) / 2 : (size_t)i + ((size_t)j * (j + 1)) / 2;
}
inline size_t packed_len(int n) { return n > 0 ? (size_t)n * (n + 1) / 2 : 0; }

// The full n x n matrix a band triangle (k off-diagonals) or a packed
// triangle stands for, the other triangle mirrored (conjugated if
// Hermitian) when `mirror`, zero otherwise.
CMat band_full(const std::vector<cd>& band, int lda, int n, int k, bool lower, bool mirror, bool hermitian) {
  CMat f{std::vector<cd>((size_t)n * n, 0), n};
  for (int j = 0; j < n; ++j) {
    const int lo = lower ? j : std::max(0, j - k), hi = lower ? std::min(n - 1, j + k) : j;
    for (int i = lo; i <= hi; ++i) {
      cd v = band[(size_t)j * lda + (lower ? i - j : k + i - j)];
      if (hermitian && i == j) v = v.real();
      f(i, j) = v;
      if (mirror && i != j) f(j, i) = hermitian ? std::conj(v) : v;
    }
  }
  return f;
}
CMat packed_full(const std::vector<cd>& ap, int n, bool lower, bool mirror, bool hermitian) {
  CMat f{std::vector<cd>((size_t)n * n, 0), n};
  for (int j = 0; j < n; ++j)
    for (int i = lower ? j : 0; i < (lower ? n : j + 1); ++i) {
      cd v = ap[packed_at(lower, n, i, j)];
      if (hermitian && i == j) v = v.real();
      f(i, j) = v;
      if (mirror && i != j) f(j, i) = hermitian ? std::conj(v) : v;
    }
  return f;
}

// y = alpha op(M) x + beta y over a dense host copy of M (rows x cols before
// the operation). beta 0 leaves y unread, so NaN in it stays out.
template <class T>
void mv_into(cublasHandle_t h, const CMat& M, vgpu_la::Op op, int rows, int cols, cd alpha, const T* x, int incx,
             cd beta, T* y, int incy) {
  const int ylen = op == vgpu_la::Op::N ? rows : cols, xlen = op == vgpu_la::Op::N ? cols : rows;
  (void)h;
  const auto xv = load_vec(x, xlen, incx);
  auto yv = beta == cd(0) ? std::vector<cd>(ylen, 0) : load_vec(y, ylen, incy);
  for (int i = 0; i < ylen; ++i) {
    cd s = 0;
    for (int j = 0; j < xlen; ++j) s += opel(M, op, i, j) * xv[j];
    yv[i] = beta == cd(0) ? alpha * s : alpha * s + beta * yv[i];
  }
  store_vec(y, ylen, incy, yv);
}

// y = alpha op(A) x + beta y, A an m x n band matrix.
template <class T>
cublasStatus_t gbmv(cublasHandle_t h, cublasOperation_t t, int m, int n, int kl, int ku, const T* alpha, const T* A,
                    int lda, const T* x, int incx, const T* beta, T* y, int incy) {
  if (!valid(h)) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (!known_op(t) || m < 0 || n < 0 || kl < 0 || ku < 0 || lda < kl + ku + 1 || incx == 0 || incy == 0)
    return CUBLAS_STATUS_INVALID_VALUE;
  if (m == 0 || n == 0) return CUBLAS_STATUS_SUCCESS;
  if (!alpha || !beta) return CUBLAS_STATUS_INVALID_VALUE;
  return run_c(h, alpha, beta, sizeof(T), [=](const void* al, const void* be) {
    const cd a_ = cscalar(h, static_cast<const T*>(al)), b_ = cscalar(h, static_cast<const T*>(be));
    const auto band = load_c(A, extent(lda, n, kl + ku + 1));
    CMat M{std::vector<cd>((size_t)m * n, 0), m};
    for (int j = 0; j < n; ++j)
      for (int i = std::max(0, j - ku); i <= std::min(m - 1, j + kl); ++i) M(i, j) = band[(size_t)j * lda + ku + i - j];
    mv_into(h, M, to_op(t), m, n, a_, x, incx, b_, y, incy);
    return CUBLAS_STATUS_SUCCESS;
  });
}

// y = alpha A x + beta y, A symmetric (sbmv) or Hermitian (hbmv) with k
// off-diagonals, from its `uplo` triangle.
template <class T>
cublasStatus_t sbmv(cublasHandle_t h, bool hermitian, cublasFillMode_t uplo, int n, int k, const T* alpha, const T* A,
                    int lda, const T* x, int incx, const T* beta, T* y, int incy) {
  if (!valid(h)) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (!known_fill(uplo) || n < 0 || k < 0 || lda < k + 1 || incx == 0 || incy == 0) return CUBLAS_STATUS_INVALID_VALUE;
  if (n == 0) return CUBLAS_STATUS_SUCCESS;
  if (!alpha || !beta) return CUBLAS_STATUS_INVALID_VALUE;
  return run_c(h, alpha, beta, sizeof(T), [=](const void* al, const void* be) {
    const cd a_ = cscalar(h, static_cast<const T*>(al)), b_ = cscalar(h, static_cast<const T*>(be));
    const CMat M = band_full(load_c(A, extent(lda, n, k + 1)), lda, n, k, uplo == CUBLAS_FILL_MODE_LOWER, true,
                             hermitian);
    mv_into(h, M, vgpu_la::Op::N, n, n, a_, x, incx, b_, y, incy);
    return CUBLAS_STATUS_SUCCESS;
  });
}

// y = alpha A x + beta y, A symmetric (spmv) or Hermitian (hpmv), packed.
template <class T>
cublasStatus_t spmv(cublasHandle_t h, bool hermitian, cublasFillMode_t uplo, int n, const T* alpha, const T* AP,
                    const T* x, int incx, const T* beta, T* y, int incy) {
  if (!valid(h)) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (!known_fill(uplo) || n < 0 || incx == 0 || incy == 0) return CUBLAS_STATUS_INVALID_VALUE;
  if (n == 0) return CUBLAS_STATUS_SUCCESS;
  if (!alpha || !beta) return CUBLAS_STATUS_INVALID_VALUE;
  return run_c(h, alpha, beta, sizeof(T), [=](const void* al, const void* be) {
    const cd a_ = cscalar(h, static_cast<const T*>(al)), b_ = cscalar(h, static_cast<const T*>(be));
    const CMat M = packed_full(load_c(AP, packed_len(n)), n, uplo == CUBLAS_FILL_MODE_LOWER, true, hermitian);
    mv_into(h, M, vgpu_la::Op::N, n, n, a_, x, incx, b_, y, incy);
    return CUBLAS_STATUS_SUCCESS;
  });
}

// The packed rank updates: AP += alpha x x^T (spr) or x x^H (hpr, alpha
// real); with a y, AP += alpha x y^T + alpha y x^T (spr2) or alpha x y^H +
// conj(alpha) y x^H (hpr2).
template <class T, class S>
cublasStatus_t spr(cublasHandle_t h, bool hermitian, cublasFillMode_t uplo, int n, const S* alpha, const T* x,
                   int incx, const T* y, int incy, T* AP) {
  if (!valid(h)) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (!known_fill(uplo) || n < 0 || incx == 0 || (y && incy == 0)) return CUBLAS_STATUS_INVALID_VALUE;
  if (n == 0) return CUBLAS_STATUS_SUCCESS;
  if (!alpha) return CUBLAS_STATUS_INVALID_VALUE;
  return run_c(h, alpha, nullptr, sizeof(S), [=](const void* al, const void*) {
    cd a_;
    if constexpr (std::is_same_v<S, T>) a_ = cscalar(h, static_cast<const T*>(al));
    else a_ = scalar(h, static_cast<const S*>(al));
    const auto xv = load_vec(x, n, incx), yv = y ? load_vec(y, n, incy) : xv;
    auto op = [&](cd v) { return hermitian ? std::conj(v) : v; };
    const bool lower = uplo == CUBLAS_FILL_MODE_LOWER;
    auto ap = load_c(AP, packed_len(n));
    for (int j = 0; j < n; ++j)
      for (int i = lower ? j : 0; i < (lower ? n : j + 1); ++i) {
        cd& e = ap[packed_at(lower, n, i, j)];
        if (hermitian && i == j) e = e.real();
        e += y ? a_ * xv[i] * op(yv[j]) + op(a_) * yv[i] * op(xv[j]) : a_ * xv[i] * op(xv[j]);
        if (hermitian && i == j) e = e.real();
      }
    store_c(AP, ap);
    return CUBLAS_STATUS_SUCCESS;
  });
}

// x = op(A) x (multiply) or op(A) x = b, x over b (solve), A triangular:
// banded with k off-diagonals (tbmv, tbsv) or packed (tpmv, tpsv, k < 0).
template <class T>
cublasStatus_t tri_band(cublasHandle_t h, bool solve, cublasFillMode_t uplo, cublasOperation_t t,
                        cublasDiagType_t diag, int n, int k, const T* A, int lda, T* x, int incx) {
  if (!valid(h)) return CUBLAS_STATUS_NOT_INITIALIZED;
  const bool packed = k < 0;
  if (!known_fill(uplo) || !known_op(t) || !known_diag(diag) || n < 0 || incx == 0 ||
      (!packed && lda < k + 1))
    return CUBLAS_STATUS_INVALID_VALUE;
  if (n == 0) return CUBLAS_STATUS_SUCCESS;
  return run_c(h, nullptr, nullptr, 0, [=](const void*, const void*) {
    const bool lower = uplo == CUBLAS_FILL_MODE_LOWER, unit = diag == CUBLAS_DIAG_UNIT;
    const CMat a = packed ? packed_full(load_c(A, packed_len(n)), n, lower, false, false)
                          : band_full(load_c(A, extent(lda, n, k + 1)), lda, n, k, lower, false, false);
    auto xv = load_vec(x, n, incx);
    const auto op = to_op(t);
    if (solve) {
      vgpu_la::tri_solve(a, n, lower, op, unit, xv.data(), 1);
    } else {
      std::vector<cd> out(n, 0);
      for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) out[i] += (i == j && unit) ? xv[j] : opel(a, op, i, j) * xv[j];
      xv = out;
    }
    store_vec(x, n, incx, xv);
    return CUBLAS_STATUS_SUCCESS;
  });
}

// Packed to full (tpttr) and full to packed (trttp): only the `uplo`
// triangle of A is read or written.
template <class T>
cublasStatus_t tpttr(cublasHandle_t h, bool to_full, cublasFillMode_t uplo, int n, const T* src, T* dst, int lda) {
  if (!valid(h)) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (!known_fill(uplo) || n < 0 || lda < std::max(1, n)) return CUBLAS_STATUS_INVALID_VALUE;
  if (n == 0) return CUBLAS_STATUS_SUCCESS;
  return run_c(h, nullptr, nullptr, 0, [=](const void*, const void*) {
    const bool lower = uplo == CUBLAS_FILL_MODE_LOWER;
    if (to_full) {
      const auto ap = load_c(src, packed_len(n));
      CMat a = load_mat(dst, lda, n, n);
      for (int j = 0; j < n; ++j)
        for (int i = lower ? j : 0; i < (lower ? n : j + 1); ++i) a(i, j) = ap[packed_at(lower, n, i, j)];
      store_mat(dst, a, n, n);
    } else {
      const CMat a = load_mat(src, lda, n, n);
      std::vector<cd> ap(packed_len(n));
      for (int j = 0; j < n; ++j)
        for (int i = lower ? j : 0; i < (lower ? n : j + 1); ++i) ap[packed_at(lower, n, i, j)] = a(i, j);
      store_c(dst, ap);
    }
    return CUBLAS_STATUS_SUCCESS;
  });
}

}  // namespace

#define VGPU_BAND_ALL(P, T)                                                                                       \
  VGPU_EXPORT cublasStatus_t cublas##P##gbmv_v2(cublasHandle_t h, cublasOperation_t t, int m, int n, int kl,      \
                                                int ku, const T* alpha, const T* A, int lda, const T* x,          \
                                                int incx, const T* beta, T* y, int incy) {                        \
    return gbmv<T>(h, t, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy);                                    \
  }                                                                                                               \
  VGPU_EXPORT cublasStatus_t cublas##P##tbmv_v2(cublasHandle_t h, cublasFillMode_t u, cublasOperation_t t,        \
                                                cublasDiagType_t d, int n, int k, const T* A, int lda, T* x,      \
                                                int incx) {                                                       \
    if (k < 0 && valid(h)) return CUBLAS_STATUS_INVALID_VALUE;                                                    \
    return tri_band<T>(h, false, u, t, d, n, k, A, lda, x, incx);                                                 \
  }                                                                                                               \
  VGPU_EXPORT cublasStatus_t cublas##P##tbsv_v2(cublasHandle_t h, cublasFillMode_t u, cublasOperation_t t,        \
                                                cublasDiagType_t d, int n, int k, const T* A, int lda, T* x,      \
                                                int incx) {                                                       \
    if (k < 0 && valid(h)) return CUBLAS_STATUS_INVALID_VALUE;                                                    \
    return tri_band<T>(h, true, u, t, d, n, k, A, lda, x, incx);                                                  \
  }                                                                                                               \
  VGPU_EXPORT cublasStatus_t cublas##P##tpmv_v2(cublasHandle_t h, cublasFillMode_t u, cublasOperation_t t,        \
                                                cublasDiagType_t d, int n, const T* AP, T* x, int incx) {         \
    return tri_band<T>(h, false, u, t, d, n, -1, AP, 0, x, incx);                                                 \
  }                                                                                                               \
  VGPU_EXPORT cublasStatus_t cublas##P##tpsv_v2(cublasHandle_t h, cublasFillMode_t u, cublasOperation_t t,        \
                                                cublasDiagType_t d, int n, const T* AP, T* x, int incx) {         \
    return tri_band<T>(h, true, u, t, d, n, -1, AP, 0, x, incx);                                                  \
  }                                                                                                               \
  VGPU_EXPORT cublasStatus_t cublas##P##tpttr(cublasHandle_t h, cublasFillMode_t u, int n, const T* AP, T* A,     \
                                              int lda) {                                                          \
    return tpttr<T>(h, true, u, n, AP, A, lda);                                                                   \
  }                                                                                                               \
  VGPU_EXPORT cublasStatus_t cublas##P##trttp(cublasHandle_t h, cublasFillMode_t u, int n, const T* A, int lda,   \
                                              T* AP) {                                                            \
    return tpttr<T>(h, false, u, n, A, AP, lda);                                                                  \
  }
VGPU_BAND_ALL(S, float)
VGPU_BAND_ALL(D, double)
VGPU_BAND_ALL(C, cuComplex)
VGPU_BAND_ALL(Z, cuDoubleComplex)
#undef VGPU_BAND_ALL

// The symmetric forms, real only, and the Hermitian ones, complex only.
#define VGPU_BAND_SYM(P, T, S, H, PFX)                                                                            \
  VGPU_EXPORT cublasStatus_t cublas##P##PFX##bmv_v2(cublasHandle_t h, cublasFillMode_t u, int n, int k,           \
                                                    const T* alpha, const T* A, int lda, const T* x, int incx,    \
                                                    const T* beta, T* y, int incy) {                              \
    return sbmv<T>(h, H, u, n, k, alpha, A, lda, x, incx, beta, y, incy);                                         \
  }                                                                                                               \
  VGPU_EXPORT cublasStatus_t cublas##P##PFX##pmv_v2(cublasHandle_t h, cublasFillMode_t u, int n, const T* alpha,  \
                                                    const T* AP, const T* x, int incx, const T* beta, T* y,       \
                                                    int incy) {                                                   \
    return spmv<T>(h, H, u, n, alpha, AP, x, incx, beta, y, incy);                                                \
  }                                                                                                               \
  VGPU_EXPORT cublasStatus_t cublas##P##PFX##pr_v2(cublasHandle_t h, cublasFillMode_t u, int n, const S* alpha,   \
                                                   const T* x, int incx, T* AP) {                                 \
    return spr<T, S>(h, H, u, n, alpha, x, incx, nullptr, 0, AP);                                                 \
  }                                                                                                               \
  VGPU_EXPORT cublasStatus_t cublas##P##PFX##pr2_v2(cublasHandle_t h, cublasFillMode_t u, int n, const T* alpha,  \
                                                    const T* x, int incx, const T* y, int incy, T* AP) {          \
    return spr<T, T>(h, H, u, n, alpha, x, incx, y, incy, AP);                                                    \
  }
VGPU_BAND_SYM(S, float, float, false, s)
VGPU_BAND_SYM(D, double, double, false, s)
VGPU_BAND_SYM(C, cuComplex, float, true, h)
VGPU_BAND_SYM(Z, cuDoubleComplex, double, true, h)
#undef VGPU_BAND_SYM
