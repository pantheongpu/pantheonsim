// Symmetric indefinite factorization A = U D U^T or L D L^T with Bunch-Kaufman
// diagonal pivoting, the solve and the inverse that use it: LAPACK's
// xSYTF2, xSYTRS and xSYTRI, step for step, so the pivots and the factors'
// layout are LAPACK's (and cuSOLVER's, which follows it: see
// nvidia/tests/e2e/solver_sytrf_paths.cu). Complex matrices are symmetric, not
// Hermitian: nothing is conjugated, and pivots are chosen by |re| + |im|, as
// ZSYTF2 chooses them.
//
// Column-major with a leading dimension, 1-based pivots, as LAPACK stores
// them: ipiv[k] = p > 0 is a 1x1 pivot after rows and columns k and p were
// interchanged; a 2x2 pivot in rows k, k+1 has ipiv[k] = ipiv[k+1] = -p, p
// interchanged with k (upper) or k+1 (lower).
#ifndef VGPU_SYTRF_HPP
#define VGPU_SYTRF_HPP

#include <algorithm>
#include <cmath>
#include <complex>
#include <type_traits>
#include <vector>

namespace vgpu_sy {

using cd = std::complex<double>;

inline double cabs1(double v) { return std::fabs(v); }
inline double cabs1(cd v) { return std::fabs(v.real()) + std::fabs(v.imag()); }
inline bool is_nan(double v) { return std::isnan(v); }
inline bool is_nan(cd v) { return std::isnan(v.real()) || std::isnan(v.imag()); }

template <class T> struct Mat {
  T* a;
  int ld;
  T& operator()(int i, int j) const { return a[(size_t)(j - 1) * ld + (i - 1)]; }  // 1-based, as LAPACK
};

// The first index of the largest |x| (cabs1 for complex), BLAS's I*AMAX.
template <class F> int iamax(int n, F x) {
  int best = 1;
  double bv = -1;
  for (int i = 1; i <= n; ++i)
    if (cabs1(x(i)) > bv) {
      bv = cabs1(x(i));
      best = i;
    }
  return best;
}

// xSYTF2. Returns info: 0, or k when D(k,k) is exactly zero (the
// factorization is still completed).
template <class T> int sytf2(bool upper, int n, T* data, int lda, int* ipiv) {
  Mat<T> A{data, lda};
  const double alpha = (1.0 + std::sqrt(17.0)) / 8.0;
  int info = 0;
  if (upper) {
    int k = n;
    while (k >= 1) {
      int kstep = 1, kp = k, imax = 0;
      const double absakk = cabs1(A(k, k));
      double colmax = 0;
      if (k > 1) {
        imax = iamax(k - 1, [&](int i) { return A(i, k); });
        colmax = cabs1(A(imax, k));
      }
      if (std::max(absakk, colmax) == 0.0 || is_nan(A(k, k))) {
        if (info == 0) info = k;
        kp = k;
      } else {
        if (absakk >= alpha * colmax) {
          kp = k;
        } else {
          int jmax = imax + iamax(k - imax, [&](int j) { return A(imax, imax + j); });
          double rowmax = cabs1(A(imax, jmax));
          if (imax > 1) {
            jmax = iamax(imax - 1, [&](int i) { return A(i, imax); });
            rowmax = std::max(rowmax, cabs1(A(jmax, imax)));
          }
          if (absakk >= alpha * colmax * (colmax / rowmax)) {
            kp = k;
          } else if (cabs1(A(imax, imax)) >= alpha * rowmax) {
            kp = imax;
          } else {
            kp = imax;
            kstep = 2;
          }
        }
        const int kk = k - kstep + 1;
        if (kp != kk) {
          for (int i = 1; i <= kp - 1; ++i) std::swap(A(i, kk), A(i, kp));
          for (int j = kp + 1; j <= kk - 1; ++j) std::swap(A(j, kk), A(kp, j));
          std::swap(A(kk, kk), A(kp, kp));
          if (kstep == 2) std::swap(A(k - 1, k), A(kp, k));
        }
        if (kstep == 1) {
          const T r1 = T(1.0) / A(k, k);
          for (int j = 1; j <= k - 1; ++j) {  // A := A - r1 x x^T, upper triangle (xSYR)
            if (A(j, k) == T(0.0)) continue;
            const T t = -r1 * A(j, k);
            for (int i = 1; i <= j; ++i) A(i, j) += A(i, k) * t;
          }
          for (int i = 1; i <= k - 1; ++i) A(i, k) *= r1;
        } else if (k > 2) {
          T d12 = A(k - 1, k);
          const T d22 = A(k - 1, k - 1) / d12, d11 = A(k, k) / d12;
          const T t = T(1.0) / (d11 * d22 - T(1.0));
          d12 = t / d12;
          for (int j = k - 2; j >= 1; --j) {
            const T wkm1 = d12 * (d11 * A(j, k - 1) - A(j, k));
            const T wk = d12 * (d22 * A(j, k) - A(j, k - 1));
            for (int i = j; i >= 1; --i) A(i, j) = A(i, j) - A(i, k) * wk - A(i, k - 1) * wkm1;
            A(j, k) = wk;
            A(j, k - 1) = wkm1;
          }
        }
      }
      if (kstep == 1) {
        ipiv[k - 1] = kp;
      } else {
        ipiv[k - 1] = -kp;
        ipiv[k - 2] = -kp;
      }
      k -= kstep;
    }
  } else {
    int k = 1;
    while (k <= n) {
      int kstep = 1, kp = k, imax = 0;
      const double absakk = cabs1(A(k, k));
      double colmax = 0;
      if (k < n) {
        imax = k + iamax(n - k, [&](int i) { return A(k + i, k); });
        colmax = cabs1(A(imax, k));
      }
      if (std::max(absakk, colmax) == 0.0 || is_nan(A(k, k))) {
        if (info == 0) info = k;
        kp = k;
      } else {
        if (absakk >= alpha * colmax) {
          kp = k;
        } else {
          int jmax = k - 1 + iamax(imax - k, [&](int j) { return A(imax, k - 1 + j); });
          double rowmax = cabs1(A(imax, jmax));
          if (imax < n) {
            jmax = imax + iamax(n - imax, [&](int i) { return A(imax + i, imax); });
            rowmax = std::max(rowmax, cabs1(A(jmax, imax)));
          }
          if (absakk >= alpha * colmax * (colmax / rowmax)) {
            kp = k;
          } else if (cabs1(A(imax, imax)) >= alpha * rowmax) {
            kp = imax;
          } else {
            kp = imax;
            kstep = 2;
          }
        }
        const int kk = k + kstep - 1;
        if (kp != kk) {
          for (int i = kp + 1; i <= n; ++i) std::swap(A(i, kk), A(i, kp));
          for (int j = kk + 1; j <= kp - 1; ++j) std::swap(A(j, kk), A(kp, j));
          std::swap(A(kk, kk), A(kp, kp));
          if (kstep == 2) std::swap(A(k + 1, k), A(kp, k));
        }
        if (kstep == 1) {
          if (k < n) {
            const T d11 = T(1.0) / A(k, k);
            for (int j = k + 1; j <= n; ++j) {  // A := A - d11 x x^T, lower triangle (xSYR)
              if (A(j, k) == T(0.0)) continue;
              const T t = -d11 * A(j, k);
              for (int i = j; i <= n; ++i) A(i, j) += A(i, k) * t;
            }
            for (int i = k + 1; i <= n; ++i) A(i, k) *= d11;
          }
        } else if (k < n - 1) {
          T d21 = A(k + 1, k);
          const T d11 = A(k + 1, k + 1) / d21, d22 = A(k, k) / d21;
          const T t = T(1.0) / (d11 * d22 - T(1.0));
          d21 = t / d21;
          for (int j = k + 2; j <= n; ++j) {
            const T wk = d21 * (d11 * A(j, k) - A(j, k + 1));
            const T wkp1 = d21 * (d22 * A(j, k + 1) - A(j, k));
            for (int i = j; i <= n; ++i) A(i, j) = A(i, j) - A(i, k) * wk - A(i, k + 1) * wkp1;
            A(j, k) = wk;
            A(j, k + 1) = wkp1;
          }
        }
      }
      if (kstep == 1) {
        ipiv[k - 1] = kp;
      } else {
        ipiv[k - 1] = -kp;
        ipiv[k] = -kp;
      }
      k += kstep;
    }
  }
  return info;
}

// xSYTRS: B := A^{-1} B with A factored by sytf2. P is any integer type.
template <class T, class P> void sytrs(bool upper, int n, int nrhs, const T* data, int lda, const P* ipiv, T* bdata,
                                       int ldb) {
  Mat<T> A{const_cast<T*>(data), lda};
  Mat<T> B{bdata, ldb};
  auto piv = [&](int k) { return (int)ipiv[k - 1]; };
  auto swap_rows = [&](int r1, int r2) {
    if (r1 != r2)
      for (int j = 1; j <= nrhs; ++j) std::swap(B(r1, j), B(r2, j));
  };
  auto ger = [&](int r0, int r1, int col, int brow) {  // B(r0:r1, :) -= A(r0:r1, col) B(brow, :)
    for (int j = 1; j <= nrhs; ++j)
      for (int i = r0; i <= r1; ++i) B(i, j) -= A(i, col) * B(brow, j);
  };
  auto gemv_t = [&](int r0, int r1, int col, int brow) {  // B(brow, :) -= A(r0:r1, col)^T B(r0:r1, :)
    for (int j = 1; j <= nrhs; ++j) {
      T s(0.0);
      for (int i = r0; i <= r1; ++i) s += A(i, col) * B(i, j);
      B(brow, j) -= s;
    }
  };
  auto solve2 = [&](int k1, int k2, T akm1k, T akm1, T ak) {  // the 2x2 block solve, rows k1 < k2
    const T denom = akm1 * ak - T(1.0);
    for (int j = 1; j <= nrhs; ++j) {
      const T bkm1 = B(k1, j) / akm1k, bk = B(k2, j) / akm1k;
      B(k1, j) = (ak * bkm1 - bk) / denom;
      B(k2, j) = (akm1 * bk - bkm1) / denom;
    }
  };
  if (upper) {
    int k = n;
    while (k >= 1) {
      if (piv(k) > 0) {
        swap_rows(k, piv(k));
        ger(1, k - 1, k, k);
        const T r = T(1.0) / A(k, k);
        for (int j = 1; j <= nrhs; ++j) B(k, j) *= r;
        k -= 1;
      } else {
        swap_rows(k - 1, -piv(k));
        ger(1, k - 2, k, k);
        ger(1, k - 2, k - 1, k - 1);
        const T akm1k = A(k - 1, k);
        solve2(k - 1, k, akm1k, A(k - 1, k - 1) / akm1k, A(k, k) / akm1k);
        k -= 2;
      }
    }
    k = 1;
    while (k <= n) {
      if (piv(k) > 0) {
        gemv_t(1, k - 1, k, k);
        swap_rows(k, piv(k));
        k += 1;
      } else {
        gemv_t(1, k - 1, k, k);
        gemv_t(1, k - 1, k + 1, k + 1);
        swap_rows(k, -piv(k));
        k += 2;
      }
    }
  } else {
    int k = 1;
    while (k <= n) {
      if (piv(k) > 0) {
        swap_rows(k, piv(k));
        ger(k + 1, n, k, k);
        const T r = T(1.0) / A(k, k);
        for (int j = 1; j <= nrhs; ++j) B(k, j) *= r;
        k += 1;
      } else {
        swap_rows(k + 1, -piv(k));
        ger(k + 2, n, k, k);
        ger(k + 2, n, k + 1, k + 1);
        const T akm1k = A(k + 1, k);
        solve2(k, k + 1, akm1k, A(k, k) / akm1k, A(k + 1, k + 1) / akm1k);
        k += 2;
      }
    }
    k = n;
    while (k >= 1) {
      if (piv(k) > 0) {
        gemv_t(k + 1, n, k, k);
        swap_rows(k, piv(k));
        k -= 1;
      } else {
        gemv_t(k + 1, n, k, k);
        gemv_t(k + 1, n, k - 1, k - 1);
        swap_rows(k, -piv(k));
        k -= 2;
      }
    }
  }
}

// xSYTRI: A's stored triangle := that triangle of A^{-1}. Returns info: 0,
// or k when D(k,k) is zero and there is no inverse (A is then untouched).
template <class T> int sytri(bool upper, int n, T* data, int lda, const int* ipiv) {
  Mat<T> A{data, lda};
  auto piv = [&](int k) { return ipiv[k - 1]; };
  for (int i = upper ? n : 1; upper ? i >= 1 : i <= n; i += upper ? -1 : 1)
    if (piv(i) > 0 && A(i, i) == T(0.0)) return i;
  // The symmetric matrix the triangle r0..r1 holds, applied: w := -S x.
  auto symv = [&](int r0, int r1, const std::vector<T>& x, int col) {
    for (int i = r0; i <= r1; ++i) {
      T s(0.0);
      for (int j = r0; j <= r1; ++j) {
        const T aij = upper ? (i <= j ? A(i, j) : A(j, i)) : (i >= j ? A(i, j) : A(j, i));
        s += aij * x[(size_t)(j - r0)];
      }
      A(i, col) = -s;
    }
  };
  auto dot = [&](int r0, int r1, const std::vector<T>& x, int col) {
    T s(0.0);
    for (int i = r0; i <= r1; ++i) s += x[(size_t)(i - r0)] * A(i, col);
    return s;
  };
  auto column = [&](int r0, int r1, int col) {
    std::vector<T> w;
    for (int i = r0; i <= r1; ++i) w.push_back(A(i, col));
    return w;
  };
  auto scale_of = [](T v) -> T {
    if constexpr (std::is_same_v<T, double>) return std::fabs(v);
    else return v;  // ZSYTRI divides by the element itself
  };
  if (upper) {
    int k = 1;
    while (k <= n) {
      int kstep;
      if (piv(k) > 0) {
        A(k, k) = T(1.0) / A(k, k);
        if (k > 1) {
          const auto w = column(1, k - 1, k);
          symv(1, k - 1, w, k);
          A(k, k) -= dot(1, k - 1, w, k);
        }
        kstep = 1;
      } else {
        const T t = scale_of(A(k, k + 1));
        const T ak = A(k, k) / t, akp1 = A(k + 1, k + 1) / t, akkp1 = A(k, k + 1) / t;
        const T d = t * (ak * akp1 - T(1.0));
        A(k, k) = akp1 / d;
        A(k + 1, k + 1) = ak / d;
        A(k, k + 1) = -akkp1 / d;
        if (k > 1) {
          auto w = column(1, k - 1, k);
          symv(1, k - 1, w, k);
          A(k, k) -= dot(1, k - 1, w, k);
          T s(0.0);
          for (int i = 1; i <= k - 1; ++i) s += A(i, k) * A(i, k + 1);
          A(k, k + 1) -= s;
          w = column(1, k - 1, k + 1);
          symv(1, k - 1, w, k + 1);
          A(k + 1, k + 1) -= dot(1, k - 1, w, k + 1);
        }
        kstep = 2;
      }
      const int kp = std::abs(piv(k));
      if (kp != k) {
        for (int i = 1; i <= kp - 1; ++i) std::swap(A(i, k), A(i, kp));
        for (int j = kp + 1; j <= k - 1; ++j) std::swap(A(j, k), A(kp, j));
        std::swap(A(k, k), A(kp, kp));
        if (kstep == 2) std::swap(A(k, k + 1), A(kp, k + 1));
      }
      k += kstep;
    }
  } else {
    int k = n;
    while (k >= 1) {
      int kstep;
      if (piv(k) > 0) {
        A(k, k) = T(1.0) / A(k, k);
        if (k < n) {
          const auto w = column(k + 1, n, k);
          symv(k + 1, n, w, k);
          A(k, k) -= dot(k + 1, n, w, k);
        }
        kstep = 1;
      } else {
        const T t = scale_of(A(k, k - 1));
        const T ak = A(k - 1, k - 1) / t, akp1 = A(k, k) / t, akkp1 = A(k, k - 1) / t;
        const T d = t * (ak * akp1 - T(1.0));
        A(k - 1, k - 1) = akp1 / d;
        A(k, k) = ak / d;
        A(k, k - 1) = -akkp1 / d;
        if (k < n) {
          auto w = column(k + 1, n, k);
          symv(k + 1, n, w, k);
          A(k, k) -= dot(k + 1, n, w, k);
          T s(0.0);
          for (int i = k + 1; i <= n; ++i) s += A(i, k) * A(i, k - 1);
          A(k, k - 1) -= s;
          w = column(k + 1, n, k - 1);
          symv(k + 1, n, w, k - 1);
          A(k - 1, k - 1) -= dot(k + 1, n, w, k - 1);
        }
        kstep = 2;
      }
      const int kp = std::abs(piv(k));
      if (kp != k) {
        for (int i = kp + 1; i <= n; ++i) std::swap(A(i, k), A(i, kp));
        for (int j = k + 1; j <= kp - 1; ++j) std::swap(A(j, k), A(kp, j));
        std::swap(A(k, k), A(kp, kp));
        if (kstep == 2) std::swap(A(k, k - 1), A(kp, k - 1));
      }
      k -= kstep;
    }
  }
  return 0;
}

}  // namespace vgpu_sy

#endif  // VGPU_SYTRF_HPP
