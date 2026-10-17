#pragma once
// The rank-one and rank-two updates of cuBLAS (ger, geru, gerc, syr, her, syr2, her2: cublas_complex.inc's cger and csyr) on the
// elements as the device holds them, with the arithmetic of the loops they replace: std::complex<double> products and sums,
// rounded to the type of the matrix on the way out (a real matrix is a complex one with no imaginary part).
//
// The loops formed every product in std::complex<double>, whose multiplication repairs a result in which both parts are NaN by a
// rule of its own (the C99 one for infinities). These kernels give the loops' bits when that cannot have happened: they take
// the scalars and vectors only if they are finite, and say whether every element they made is finite (an infinity or a NaN that
// arises in a product stays one in the sum that follows, so if no result is one, none arose); when they say no the caller
// discards the host copy and runs its old loops, which are the reference (tests/unit/test_host_blas2.cpp keeps them).
//
// A matrix element is S[NC]: NC = 1 for float or double, NC = 2 for the real and imaginary parts of a complex element (cuComplex
// and cuDoubleComplex are two floats or two doubles). The matrix is a host copy with ld elements to a column. Each update changes
// the host copy in place and returns whether the result is the loops' (false: the host copy is garbage).
#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>

namespace vgpu::hostblas2 {

using cd = std::complex<double>;

inline bool finite(const cd& v) { return std::isfinite(v.real()) && std::isfinite(v.imag()); }
inline bool all_finite(const std::vector<cd>& v) {
  for (const cd& e : v)
    if (!finite(e)) return false;
  return true;
}

// A(i, j) += (alpha * x_i) * y_j   (y_j conjugated when `conj`), for the m x n matrix.
template <class S, int NC>
bool ger(bool conj, int m, int n, cd alpha, const std::vector<cd>& x, const std::vector<cd>& y, S* A, int ld) {
  if (!finite(alpha) || !all_finite(x) || !all_finite(y)) return false;
  std::vector<double> pr(static_cast<size_t>(m)), pi(static_cast<size_t>(m));
  for (int i = 0; i < m; ++i) {
    const cd p = alpha * x[i];
    pr[i] = p.real();
    pi[i] = p.imag();
    if (!std::isfinite(pr[i]) || !std::isfinite(pi[i])) return false;
  }
  double bad = 0;
  for (int j = 0; j < n; ++j) {
    const cd yj = conj ? std::conj(y[j]) : y[j];
    const double yr = yj.real(), yi = yj.imag();
    S* a = A + static_cast<size_t>(j) * ld * NC;
    for (int i = 0; i < m; ++i) {
      const double nr = static_cast<double>(a[i * NC]) + (pr[i] * yr - pi[i] * yi);
      bad += nr - nr;   // 0 for a finite value, NaN otherwise
      a[i * NC] = static_cast<S>(nr);
      if constexpr (NC == 2) {
        const double ni = static_cast<double>(a[i * NC + 1]) + (pr[i] * yi + pi[i] * yr);
        bad += ni - ni;
        a[i * NC + 1] = static_cast<S>(ni);
      }
    }
  }
  return bad == 0.0;   // (a NaN is not equal to 0)
}

// The rank-one (y null) or rank-two update of the lower or upper triangle of an n x n matrix:
//     A(i, j) += (alpha * x_i) * op(x_j)                                        rank one
//     A(i, j) += (alpha * x_i) * op(y_j) + (op(alpha) * y_i) * op(x_j)          rank two
// with op the conjugate for a Hermitian matrix and the identity otherwise; a Hermitian matrix's diagonal is made real afterwards
// (its imaginary part set to +0).
template <class S, int NC>
bool syr(bool hermitian, bool lower, int n, cd alpha, const std::vector<cd>& x, const std::vector<cd>* y, S* A, int ld) {
  if (!finite(alpha) || !all_finite(x) || (y && !all_finite(*y))) return false;
  auto op = [&](cd v) { return hermitian ? std::conj(v) : v; };
  std::vector<double> ar(static_cast<size_t>(n)), ai(static_cast<size_t>(n)), br(y ? static_cast<size_t>(n) : 0), bi(y ? static_cast<size_t>(n) : 0);
  for (int i = 0; i < n; ++i) {
    const cd p = alpha * x[i];
    ar[i] = p.real();
    ai[i] = p.imag();
    if (!std::isfinite(ar[i]) || !std::isfinite(ai[i])) return false;
    if (y) {
      const cd q = op(alpha) * (*y)[i];
      br[i] = q.real();
      bi[i] = q.imag();
      if (!std::isfinite(br[i]) || !std::isfinite(bi[i])) return false;
    }
  }
  double bad = 0;
  for (int j = 0; j < n; ++j) {
    const cd oxj = op(x[j]);
    const double xr = oxj.real(), xi = oxj.imag();
    double yr = 0, yi = 0;
    if (y) {
      const cd oyj = op((*y)[j]);
      yr = oyj.real();
      yi = oyj.imag();
    }
    S* a = A + static_cast<size_t>(j) * ld * NC;
    for (int i = lower ? j : 0; i < (lower ? n : j + 1); ++i) {
      double sr, si;
      if (y) {
        // ax_i * op(y_j) + bx_i * op(x_j), each a complex product, then their sum
        const double p1r = ar[i] * yr - ai[i] * yi, p1i = ar[i] * yi + ai[i] * yr;
        const double p2r = br[i] * xr - bi[i] * xi, p2i = br[i] * xi + bi[i] * xr;
        sr = p1r + p2r;
        si = p1i + p2i;
      } else {
        sr = ar[i] * xr - ai[i] * xi;
        si = ar[i] * xi + ai[i] * xr;
      }
      const double nr = static_cast<double>(a[i * NC]) + sr;
      bad += nr - nr;
      a[i * NC] = static_cast<S>(nr);
      if constexpr (NC == 2) {
        double ni = static_cast<double>(a[i * NC + 1]) + si;
        bad += ni - ni;
        if (hermitian && i == j) ni = 0.0;   // (the diagonal is made real, with a +0 imaginary part)
        a[i * NC + 1] = static_cast<S>(ni);
      }
    }
  }
  return bad == 0.0;
}

}  // namespace vgpu::hostblas2
