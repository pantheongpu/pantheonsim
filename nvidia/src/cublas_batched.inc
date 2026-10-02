// The batched GEMVs, getriBatched and matinvBatched, syrkx and herkx, the
// gemm3m forms, the batched Hgemms and the complex dgmm. Included at the end
// of cublas_api.cpp, whose helpers (run_c, the typed element access ex_get /
// ex_put / ex_load, CMat) these use.

namespace {

/* ---- gemvBatched and gemvStridedBatched ----
   y_i = alpha op(A_i) x_i + beta y_i for each of batchCount problems, A, x
   and y given as device arrays of pointers (Batched) or a fixed stride apart
   (StridedBatched). In every type the header declares them in: S, D, C, Z;
   and the mixed-precision forms named for their types in, compute, out --
   HSH (half A and x, float arithmetic, half y), HSS (half in, float y), TST
   (bfloat16 in and out) and TSS (bfloat16 in, float y), all with float alpha
   and beta. The arithmetic is in double, rounded to float and then to y's
   type. beta 0 leaves y unread.

   The arguments, measured on an RTX 3060 (cuBLAS 13.0): an empty batch
   succeeds whatever else is passed, and a leading dimension below m is
   INVALID_VALUE (lda 0 is fine for m = 0). That is all the card checks: a
   negative size or batch count, an unknown operation or a zero y increment
   come back SUCCESS there and launch with them (one such call hung the
   card's kernel until the process was killed), and a NULL alpha crashes the
   process. Here those are INVALID_VALUE instead -- a refusal, where the card
   offers nothing to match -- and a zero x increment, which has a meaning,
   reads x[0] for every element as BLAS defines it. */
struct GemvTypes {
  cudaDataType a, x, y, exec;   // exec: the precision y is rounded to first
};

template <class Ptrs>
cublasStatus_t gemv_batched(cublasHandle_t h, cublasOperation_t t, int m, int n, const void* alpha,
                            size_t sbytes, cudaDataType stype, int lda, int incx, const void* beta, int incy,
                            int batch, GemvTypes ty, Ptrs ptrs) {
  if (!valid(h)) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (batch == 0) return CUBLAS_STATUS_SUCCESS;
  if (lda < m || !known_op(t) || m < 0 || n < 0 || incy == 0 || batch < 0 || !alpha || !beta)
    return CUBLAS_STATUS_INVALID_VALUE;
  if (m == 0 || n == 0) return CUBLAS_STATUS_SUCCESS;
  return run_c(h, alpha, beta, sbytes, [=](const void* al, const void* be) {
    const cd a_ = ex_scalar(h, al, stype), b_ = ex_scalar(h, be, stype);
    const bool tr = t != CUBLAS_OP_N, conj = t == CUBLAS_OP_C;
    const int ylen = tr ? n : m, xlen = tr ? m : n;
    const size_t ea = ex_bytes(ty.a);
    std::vector<const void*> pa, px;
    std::vector<void*> py;
    ptrs(&pa, &px, &py);
    for (int b = 0; b < batch; ++b) {
      const auto raw = fetch<uint8_t>(pa[b], extent(lda, n, m) * ea);
      const auto xv = ex_load(px[b], ty.x, xlen, incx);
      auto yv = b_ == cd(0) ? std::vector<cd>(ylen, 0) : ex_load(py[b], ty.y, ylen, incy);
      for (int i = 0; i < ylen; ++i) {
        cd s = 0;
        for (int j = 0; j < xlen; ++j) {
          const cd e = ex_get(ty.a, raw.data() + (tr ? idx(j, i, lda) : idx(i, j, lda)) * ea);
          s += (conj ? std::conj(e) : e) * xv[j];
        }
        yv[i] = b_ == cd(0) ? a_ * s : a_ * s + b_ * yv[i];
      }
      ex_store(py[b], ty.y, ylen, incy, yv, ty.exec);
    }
    return CUBLAS_STATUS_SUCCESS;
  });
}

// The two ways the problems are given: device arrays of pointers, or a base
// pointer and a stride (in elements).
template <class TA, class TX, class TY>
cublasStatus_t gemv_ptr_arrays(cublasHandle_t h, cublasOperation_t t, int m, int n, const void* alpha, size_t sbytes,
                               cudaDataType stype, const TA* const A[], int lda, const TX* const x[], int incx,
                               const void* beta, TY* const y[], int incy, int batch, GemvTypes ty) {
  return gemv_batched(h, t, m, n, alpha, sbytes, stype, lda, incx, beta, incy, batch, ty,
                      [=](std::vector<const void*>* pa, std::vector<const void*>* px, std::vector<void*>* py) {
                        *pa = fetch<const void*>(A, (size_t)batch);
                        *px = fetch<const void*>(x, (size_t)batch);
                        *py = fetch<void*>(y, (size_t)batch);
                      });
}
template <class TA, class TX, class TY>
cublasStatus_t gemv_strided(cublasHandle_t h, cublasOperation_t t, int m, int n, const void* alpha, size_t sbytes,
                            cudaDataType stype, const TA* A, int lda, long long sA, const TX* x, int incx, long long sx,
                            const void* beta, TY* y, int incy, long long sy, int batch, GemvTypes ty) {
  return gemv_batched(h, t, m, n, alpha, sbytes, stype, lda, incx, beta, incy, batch, ty,
                      [=](std::vector<const void*>* pa, std::vector<const void*>* px, std::vector<void*>* py) {
                        for (int b = 0; b < batch; ++b) {
                          pa->push_back(A + b * sA);
                          px->push_back(x + b * sx);
                          py->push_back(y + b * sy);
                        }
                      });
}

/* ---- getriBatched and matinvBatched ----
   getriBatched inverts each matrix from the LU factors getrfBatched left in
   it (P its pivots, or NULL when it was factored without pivoting) into C;
   matinvBatched factors and inverts in one call, A left as it was. info, in
   device memory, gets 0 for each matrix or the 1-based index of the first
   zero on U's diagonal, as getrfBatched reports it. The inverse is computed
   in double by solving L U X = P (LAPACK's getrs over the identity) and
   rounded once. Measured on an RTX 3060: an empty matrix (n = 0) succeeds
   whatever the leading dimensions; otherwise both must reach n. matinvBatched
   takes n up to 32 and refuses more (INVALID_VALUE), as documented;
   getriBatched has no such limit. Neither refuses a NULL info here (the card
   writes through it and faults). */
template <class T>
cublasStatus_t getri_batched(cublasHandle_t h, int n, const T* const A[], int lda, const int* P, T* const Cs[], int ldc,
                             int* info, int batch, bool factor) {
  if (!valid(h)) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (n < 0 || batch < 0 || (factor && n > 32)) return CUBLAS_STATUS_INVALID_VALUE;
  if (n == 0 || batch == 0) return CUBLAS_STATUS_SUCCESS;
  if (lda < n || ldc < n) return CUBLAS_STATUS_INVALID_VALUE;
  return run_c(h, nullptr, nullptr, 0, [=](const void*, const void*) {
    const auto pa = fetch<const T*>(A, (size_t)batch);
    const auto pc = fetch<T*>(Cs, (size_t)batch);
    std::vector<int> infos(batch, 0);
    for (int b = 0; b < batch; ++b) {
      CMat a = load_mat(pa[b], lda, n, n);
      std::vector<int> ipiv;
      if (factor) {
        infos[b] = vgpu_la::lu(a, n, n, &ipiv);
      } else {
        if (P) ipiv = fetch<int>(P + (size_t)b * n, (size_t)n);
        for (int j = 0; j < n && !infos[b]; ++j)
          if (a(j, j) == cd(0)) infos[b] = j + 1;
      }
      CMat x = load_mat(pc[b], ldc, n, n);
      for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) x(i, j) = i == j ? 1 : 0;
      vgpu_la::lu_solve(a, n, ipiv.empty() ? nullptr : ipiv.data(), vgpu_la::Op::N, x, n);
      store_mat(pc[b], x, n, n);
    }
    if (info) store(info, infos);
    return CUBLAS_STATUS_SUCCESS;
  });
}

/* ---- syrkx and herkx ----
   C = alpha op(A) op(B)^T + beta C (syrkx) or alpha op(A) op(B)^H + beta C
   (herkx, beta real), only C's `uplo` triangle written: syrk and herk with a
   second operand, for when the caller knows the product is symmetric. As
   herk and syrk do, any operation but N transposes (herkx takes T as C,
   syrkx C as T), and herkx leaves the diagonal real. A NULL alpha or beta is
   INVALID_VALUE (measured), after the other checks and an empty problem. */
template <class T, class S>
cublasStatus_t csyrkx(cublasHandle_t h, bool hermitian, cublasFillMode_t uplo, cublasOperation_t t, int n, int k,
                      const T* alpha, const T* A, int lda, const T* B, int ldb, const S* beta, T* C, int ldc) {
  if (!valid(h)) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (!herk_args_ok(uplo, t, n, k, lda, ldc) || ldb < std::max(1, t == CUBLAS_OP_N ? n : k))
    return CUBLAS_STATUS_INVALID_VALUE;
  if (n == 0) return CUBLAS_STATUS_SUCCESS;
  if (!alpha || !beta) return CUBLAS_STATUS_INVALID_VALUE;
  return run_c(h, alpha, sizeof(T), beta, sizeof(S), [=](const void* al, const void* be) {
    const cd a_ = cscalar(h, static_cast<const T*>(al));
    cd b_;
    if constexpr (std::is_same_v<S, T>) b_ = cscalar(h, static_cast<const T*>(be));
    else b_ = be ? cd(scalar(h, static_cast<const S*>(be))) : cd(0);
    const bool trans = t != CUBLAS_OP_N, lower = uplo == CUBLAS_FILL_MODE_LOWER;
    const CMat a = load_mat(A, lda, trans ? n : k, trans ? k : n);
    const CMat b = load_mat(B, ldb, trans ? n : k, trans ? k : n);
    auto el = [&](const CMat& mm, int i, int p) { return trans ? mm(p, i) : mm(i, p); };   // op(M)(i, p) unconj.
    CMat c = load_mat(C, ldc, n, n);
    for (int j = 0; j < n; ++j)
      for (int i = lower ? j : 0; i < (lower ? n : j + 1); ++i) {
        cd s = 0;
        for (int p = 0; p < k; ++p) {
          if (hermitian && trans) s += std::conj(el(a, i, p)) * el(b, j, p);   // A^H B
          else if (hermitian) s += el(a, i, p) * std::conj(el(b, j, p));       // A B^H
          else s += el(a, i, p) * el(b, j, p);
        }
        c(i, j) = b_ == cd(0) ? a_ * s : a_ * s + b_ * c(i, j);
        if (hermitian && i == j) c(i, j) = c(i, j).real();
      }
    store_mat(C, c, n, n);
    return CUBLAS_STATUS_SUCCESS;
  });
}

// C = A diag(x) or diag(x) A in complex arithmetic (the real dgmm is in
// cublas_api.cpp): x walked by BLAS's rule for its increment, so a zero
// increment scales by x[0] throughout.
template <class T>
cublasStatus_t cdgmm(cublasHandle_t h, cublasSideMode_t mode, int m, int n, const T* A, int lda, const T* x, int incx,
                     T* C, int ldc) {
  if (!valid(h)) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (!known_side(mode) || m < 0 || n < 0 || lda < std::max(1, m) || ldc < std::max(1, m))
    return CUBLAS_STATUS_INVALID_VALUE;
  if (!m || !n) return CUBLAS_STATUS_SUCCESS;
  return run_c(h, nullptr, nullptr, 0, [=](const void*, const void*) {
    const int len = mode == CUBLAS_SIDE_LEFT ? m : n;
    const CMat a = load_mat(A, lda, n, m);
    const auto xv = incx == 0 ? std::vector<cd>(len, load_c(x, 1)[0]) : load_vec(x, len, incx);
    CMat c = load_mat(C, ldc, n, m);
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < m; ++i) c(i, j) = a(i, j) * xv[mode == CUBLAS_SIDE_LEFT ? i : j];
    store_mat(C, c, n, m);
    return CUBLAS_STATUS_SUCCESS;
  });
}

}  // namespace

#define VGPU_GEMV_BATCHED(NAME, TA, TX, TY, S, ST, AT, XT, YT, ET)                                                \
  VGPU_EXPORT cublasStatus_t cublas##NAME##gemvBatched(                                                           \
      cublasHandle_t h, cublasOperation_t t, int m, int n, const S* alpha, const TA* const A[], int lda,          \
      const TX* const x[], int incx, const S* beta, TY* const y[], int incy, int batch) {                         \
    return gemv_ptr_arrays<TA, TX, TY>(h, t, m, n, alpha, sizeof(S), ST, A, lda, x, incx, beta, y, incy, batch,   \
                                       GemvTypes{AT, XT, YT, ET});                                                \
  }                                                                                                               \
  VGPU_EXPORT cublasStatus_t cublas##NAME##gemvStridedBatched(                                                    \
      cublasHandle_t h, cublasOperation_t t, int m, int n, const S* alpha, const TA* A, int lda, long long sA,    \
      const TX* x, int incx, long long sx, const S* beta, TY* y, int incy, long long sy, int batch) {             \
    return gemv_strided<TA, TX, TY>(h, t, m, n, alpha, sizeof(S), ST, A, lda, sA, x, incx, sx, beta, y, incy, sy, \
                                    batch, GemvTypes{AT, XT, YT, ET});                                            \
  }
VGPU_GEMV_BATCHED(S, float, float, float, float, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F)
VGPU_GEMV_BATCHED(D, double, double, double, double, CUDA_R_64F, CUDA_R_64F, CUDA_R_64F, CUDA_R_64F, CUDA_R_64F)
VGPU_GEMV_BATCHED(C, cuComplex, cuComplex, cuComplex, cuComplex, CUDA_C_32F, CUDA_C_32F, CUDA_C_32F, CUDA_C_32F,
                  CUDA_C_32F)
VGPU_GEMV_BATCHED(Z, cuDoubleComplex, cuDoubleComplex, cuDoubleComplex, cuDoubleComplex, CUDA_C_64F, CUDA_C_64F,
                  CUDA_C_64F, CUDA_C_64F, CUDA_C_64F)
VGPU_GEMV_BATCHED(HSH, __half, __half, __half, float, CUDA_R_32F, CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUDA_R_32F)
VGPU_GEMV_BATCHED(HSS, __half, __half, float, float, CUDA_R_32F, CUDA_R_16F, CUDA_R_16F, CUDA_R_32F, CUDA_R_32F)
VGPU_GEMV_BATCHED(TST, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, float, CUDA_R_32F, CUDA_R_16BF, CUDA_R_16BF,
                  CUDA_R_16BF, CUDA_R_32F)
VGPU_GEMV_BATCHED(TSS, __nv_bfloat16, __nv_bfloat16, float, float, CUDA_R_32F, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_32F,
                  CUDA_R_32F)
#undef VGPU_GEMV_BATCHED

#define VGPU_GETRI(P, T)                                                                                          \
  VGPU_EXPORT cublasStatus_t cublas##P##getriBatched(cublasHandle_t h, int n, const T* const A[], int lda,         \
                                                     const int* P_, T* const C[], int ldc, int* info, int batch) { \
    return getri_batched<T>(h, n, A, lda, P_, C, ldc, info, batch, false);                                        \
  }                                                                                                               \
  VGPU_EXPORT cublasStatus_t cublas##P##matinvBatched(cublasHandle_t h, int n, const T* const A[], int lda,        \
                                                      T* const Ainv[], int lda_inv, int* info, int batch) {       \
    return getri_batched<T>(h, n, A, lda, nullptr, Ainv, lda_inv, info, batch, true);                             \
  }                                                                                                               \
  VGPU_EXPORT cublasStatus_t cublas##P##syrkx(cublasHandle_t h, cublasFillMode_t u, cublasOperation_t t, int n,   \
                                              int k, const T* alpha, const T* A, int lda, const T* B, int ldb,    \
                                              const T* beta, T* C, int ldc) {                                     \
    return csyrkx<T, T>(h, false, u, t, n, k, alpha, A, lda, B, ldb, beta, C, ldc);                               \
  }
VGPU_GETRI(S, float)
VGPU_GETRI(D, double)
VGPU_GETRI(C, cuComplex)
VGPU_GETRI(Z, cuDoubleComplex)
#undef VGPU_GETRI

VGPU_EXPORT cublasStatus_t cublasCherkx(cublasHandle_t h, cublasFillMode_t u, cublasOperation_t t, int n, int k,
                                        const cuComplex* alpha, const cuComplex* A, int lda, const cuComplex* B,
                                        int ldb, const float* beta, cuComplex* C, int ldc) {
  return csyrkx<cuComplex, float>(h, true, u, t, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
VGPU_EXPORT cublasStatus_t cublasZherkx(cublasHandle_t h, cublasFillMode_t u, cublasOperation_t t, int n, int k,
                                        const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda,
                                        const cuDoubleComplex* B, int ldb, const double* beta, cuDoubleComplex* C,
                                        int ldc) {
  return csyrkx<cuDoubleComplex, double>(h, true, u, t, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

VGPU_EXPORT cublasStatus_t cublasCdgmm(cublasHandle_t h, cublasSideMode_t mode, int m, int n, const cuComplex* A,
                                       int lda, const cuComplex* x, int incx, cuComplex* C, int ldc) {
  return cdgmm(h, mode, m, n, A, lda, x, incx, C, ldc);
}
VGPU_EXPORT cublasStatus_t cublasZdgmm(cublasHandle_t h, cublasSideMode_t mode, int m, int n,
                                       const cuDoubleComplex* A, int lda, const cuDoubleComplex* x, int incx,
                                       cuDoubleComplex* C, int ldc) {
  return cdgmm(h, mode, m, n, A, lda, x, incx, C, ldc);
}

/* ---- the gemm3m forms ----
   The Gauss three-multiplication complex GEMM: the same product as cgemm,
   which is how these compute it (exactly, where the 3M algorithm's
   reassociation would round differently on the card). */
VGPU_EXPORT cublasStatus_t cublasCgemm3m(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m, int n,
                                         int k, const cuComplex* alpha, const cuComplex* A, int lda,
                                         const cuComplex* B, int ldb, const cuComplex* beta, cuComplex* C, int ldc) {
  return cublasCgemm_v2(h, ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
VGPU_EXPORT cublasStatus_t cublasZgemm3m(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m, int n,
                                         int k, const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda,
                                         const cuDoubleComplex* B, int ldb, const cuDoubleComplex* beta,
                                         cuDoubleComplex* C, int ldc) {
  return cublasZgemm_v2(h, ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
VGPU_EXPORT cublasStatus_t cublasCgemm3mBatched(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m,
                                                int n, int k, const cuComplex* alpha, const cuComplex* const A[],
                                                int lda, const cuComplex* const B[], int ldb, const cuComplex* beta,
                                                cuComplex* const C[], int ldc, int batch) {
  return cublasCgemmBatched(h, ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, batch);
}
VGPU_EXPORT cublasStatus_t cublasCgemm3mStridedBatched(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb,
                                                       int m, int n, int k, const cuComplex* alpha,
                                                       const cuComplex* A, int lda, long long sA, const cuComplex* B,
                                                       int ldb, long long sB, const cuComplex* beta, cuComplex* C,
                                                       int ldc, long long sC, int batch) {
  return cublasCgemmStridedBatched(h, ta, tb, m, n, k, alpha, A, lda, sA, B, ldb, sB, beta, C, ldc, sC, batch);
}

/* ---- the batched Hgemms ----
   As cublasHgemm: half operands and scalars, through GemmEx at
   CUBLAS_COMPUTE_16F. */
VGPU_EXPORT cublasStatus_t cublasHgemmStridedBatched(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb,
                                                     int m, int n, int k, const __half* alpha, const __half* A,
                                                     int lda, long long sA, const __half* B, int ldb, long long sB,
                                                     const __half* beta, __half* C, int ldc, long long sC,
                                                     int batch) {
  if (!valid(h)) return CUBLAS_STATUS_NOT_INITIALIZED;
  if (batch < 0) return CUBLAS_STATUS_INVALID_VALUE;   // as the card answers
  return cublasGemmStridedBatchedEx(h, ta, tb, m, n, k, alpha, A, CUDA_R_16F, lda, sA, B, CUDA_R_16F, ldb, sB, beta,
                                    C, CUDA_R_16F, ldc, sC, batch, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT);
}
VGPU_EXPORT cublasStatus_t cublasHgemmBatched(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb, int m,
                                              int n, int k, const __half* alpha, const __half* const A[], int lda,
                                              const __half* const B[], int ldb, const __half* beta,
                                              __half* const C[], int ldc, int batch) {
  return cublasGemmBatchedEx(h, ta, tb, m, n, k, alpha, reinterpret_cast<const void* const*>(A), CUDA_R_16F, lda,
                             reinterpret_cast<const void* const*>(B), CUDA_R_16F, ldb, beta,
                             reinterpret_cast<void* const*>(C), CUDA_R_16F, ldc, batch, CUBLAS_COMPUTE_16F,
                             CUBLAS_GEMM_DEFAULT);
}
