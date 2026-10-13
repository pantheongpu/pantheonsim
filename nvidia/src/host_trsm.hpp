#pragma once
// Triangular solves for many right-hand sides at once (cuBLAS trsm, real and complex): the same steps in the same order
// for every right-hand side as the loops they replace, with the right-hand sides along the vector lanes.
//
//   for each right-hand side x (a column of B for a left solve, a row for a right one), in the order of the solve:
//       s = x[i];  for j in the span of row i, ascending:  s = s - M(i,j) * x[j];   x[i] = unit ? s : s / M(i,i)
//
// (the real loops are cublas_api.cpp's tri_solve, the complex ones complex_linalg.hpp's). Panels of right-hand sides
// are independent and go to the host's threads, so the result does not depend on their number
// (tests/unit/test_host_trsm.cpp holds the loops as the reference).
//
// A complex solve is answered only when it is certain to be the loops' answer: with an infinity or a NaN in an
// operand a product can be repaired by the complex multiply's rule, which these kernels do not have; the caller
// passes only finite operands, and the solve says whether every x it made is finite (an infinity or a NaN that arises
// in a step stays one in every x that follows, so if none is, none arose). Otherwise the caller runs the loops.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <complex>
#include <cstdint>
#include <vector>

#include "host_gemm.hpp"

namespace vgpu::hostgemm {
namespace generic {
#include "host_trsm_impl.inc"
}  // namespace generic
#ifdef VGPU_HOSTGEMM_AVX2
#pragma GCC push_options
#pragma GCC target("avx2")
namespace avx2 {
#include "host_trsm_impl.inc"
}  // namespace avx2
#pragma GCC pop_options
#endif
}  // namespace vgpu::hostgemm

namespace vgpu::hosttrsm {

using vgpu::hostgemm::Isa;

// Real: b (double) holds the vectors; element i of vector l is b[i*se + l*sl]. M(i, j) = a[i*sti + j*stj]. forward: the
// solve runs i = 0 .. k-1 with the span j in [0, i), otherwise i = k-1 .. 0 with j in (i, k).
inline void real(int k, int64_t nv, bool forward, bool unit, const double* a, int64_t sti, int64_t stj, double* b,
                 int64_t se, int64_t sl, Isa isa = Isa::Auto, unsigned threads = 0) {
  if (k <= 0 || nv <= 0) return;
#ifdef VGPU_HOSTGEMM_AVX2
  const bool wide = isa == Isa::Avx2 || (isa == Isa::Auto && vgpu::hostgemm::avx2_available());
  const int W = wide ? vgpu::hostgemm::avx2::TrsmOps::W : vgpu::hostgemm::generic::TrsmOps::W;
#else
  const bool wide = false;
  const int W = vgpu::hostgemm::generic::TrsmOps::W;
#endif
  const int64_t panels = (nv + W - 1) / W;
  const unsigned want = threads ? threads : vgpu::hostpar::plan(static_cast<uint64_t>(k) * k / 2 * static_cast<uint64_t>(nv), panels, 1u << 21);
  vgpu::hostpar::parallel_for(want, panels, [&](int64_t pn) {
    thread_local std::vector<double> xt;
    xt.resize(static_cast<size_t>(k) * W);
    const int64_t l0 = pn * W;
    const int lanes = static_cast<int>(nv - l0 < W ? nv - l0 : W);
    for (int64_t i = 0; i < k; ++i)
      for (int l = 0; l < W; ++l) xt[static_cast<size_t>(i) * W + l] = l < lanes ? b[i * se + (l0 + l) * sl] : 0.0;
    for (int64_t t = 0; t < k; ++t) {
      const int64_t i = forward ? t : k - 1 - t;
      const int64_t j0 = forward ? 0 : i + 1, j1 = forward ? i : k;
      if (wide) {
#ifdef VGPU_HOSTGEMM_AVX2
        vgpu::hostgemm::avx2::TrsmOps::real_row(a, sti, stj, i, j0, j1, unit, xt.data());
#endif
      } else {
        vgpu::hostgemm::generic::TrsmOps::real_row(a, sti, stj, i, j0, j1, unit, xt.data());
      }
    }
    for (int64_t i = 0; i < k; ++i)
      for (int l = 0; l < lanes; ++l) b[i * se + (l0 + l) * sl] = xt[static_cast<size_t>(i) * W + l];
  });
}

// Complex: M is given in planes, row-major (M(i, j) = (mr[i*k + j], mi[i*k + j])), the vectors in b (element i of vector l at
// b[i*se + l*sl]); conj_io solves for the conjugated vectors and conjugates the answers back. Returns false when an x
// is not finite (b is then garbage and the caller starts over).
inline bool cplx(int k, int64_t nv, bool forward, bool unit, const double* mr, const double* mi, std::complex<double>* b,
                 int64_t se, int64_t sl, bool conj_io, Isa isa = Isa::Auto, unsigned threads = 0) {
  if (k <= 0 || nv <= 0) return true;
#ifdef VGPU_HOSTGEMM_AVX2
  const bool wide = isa == Isa::Avx2 || (isa == Isa::Auto && vgpu::hostgemm::avx2_available());
  const int W = wide ? vgpu::hostgemm::avx2::TrsmOps::W : vgpu::hostgemm::generic::TrsmOps::W;
#else
  const bool wide = false;
  const int W = vgpu::hostgemm::generic::TrsmOps::W;
#endif
  const int64_t panels = (nv + W - 1) / W;
  const unsigned want = threads ? threads : vgpu::hostpar::plan(static_cast<uint64_t>(k) * k / 2 * static_cast<uint64_t>(nv) * 2, panels, 1u << 21);
  std::atomic<bool> finite{true};
  vgpu::hostpar::parallel_for(want, panels, [&](int64_t pn) {
    thread_local std::vector<double> xp;
    xp.resize(static_cast<size_t>(k) * W * 2);
    double* const xr = xp.data();
    double* const xi = xr + static_cast<size_t>(k) * W;
    const int64_t l0 = pn * W;
    const int lanes = static_cast<int>(nv - l0 < W ? nv - l0 : W);
    for (int64_t i = 0; i < k; ++i)
      for (int l = 0; l < W; ++l) {
        double re = 0.0, im = 0.0;
        if (l < lanes) {
          const std::complex<double> v = b[i * se + (l0 + l) * sl];
          re = v.real();
          im = conj_io ? -v.imag() : v.imag();
        }
        xr[static_cast<size_t>(i) * W + l] = re;
        xi[static_cast<size_t>(i) * W + l] = im;
      }
    bool ok = true;
    for (int64_t t = 0; t < k; ++t) {
      const int64_t i = forward ? t : k - 1 - t;
      const int64_t j0 = forward ? 0 : i + 1, j1 = forward ? i : k;
      if (wide) {
#ifdef VGPU_HOSTGEMM_AVX2
        vgpu::hostgemm::avx2::TrsmOps::cplx_row(mr, mi, k, i, j0, j1, xr, xi);
#endif
      } else {
        vgpu::hostgemm::generic::TrsmOps::cplx_row(mr, mi, k, i, j0, j1, xr, xi);
      }
      if (!unit) {   // x[i] = s / M(i,i), with the complex divide of the loops this replaces
        const std::complex<double> d(mr[i * k + i], mi[i * k + i]);
        for (int l = 0; l < lanes; ++l) {
          const std::complex<double> q = std::complex<double>(xr[static_cast<size_t>(i) * W + l], xi[static_cast<size_t>(i) * W + l]) / d;
          xr[static_cast<size_t>(i) * W + l] = q.real();
          xi[static_cast<size_t>(i) * W + l] = q.imag();
        }
      }
      for (int l = 0; l < lanes; ++l)
        ok &= std::isfinite(xr[static_cast<size_t>(i) * W + l]) && std::isfinite(xi[static_cast<size_t>(i) * W + l]);
    }
    for (int64_t i = 0; i < k; ++i)
      for (int l = 0; l < lanes; ++l) {
        const double im = xi[static_cast<size_t>(i) * W + l];
        b[i * se + (l0 + l) * sl] = std::complex<double>(xr[static_cast<size_t>(i) * W + l], conj_io ? -im : im);
      }
    if (!ok) finite.store(false, std::memory_order_relaxed);
  });
  return finite.load(std::memory_order_relaxed);
}

// Householder reflectors H_j = I - t_j v_j v_j^T applied to the columns [c0, c1) of the m-row matrix c, for j = jfirst, then
// jfirst + step, ..., jlast: reflector j is v with v[j] = 1 and v[i] = a[i + j*lda] for i > j (rows above j are not touched),
// scaled by tau[j]. Each column gets the steps of the old loops (dot product, scale, update; see ReflOps) in the order
// of the reflectors; groups of columns are independent and go to the host's threads, each held as lanes while all the
// reflectors pass over it (the panel stays in the cache).
// skip_zero_tau: a reflector whose tau is exactly 0 is left out (geqrf's loops never made one).
inline void reflect(const double* a, int64_t lda, const double* tau, int64_t m, int jfirst, int jlast, int step,
                    double* c, int64_t ldc, int64_t c0, int64_t c1, bool skip_zero_tau, Isa isa = Isa::Auto,
                    unsigned threads = 0) {
  if (m <= 0 || c1 <= c0) return;
#ifdef VGPU_HOSTGEMM_AVX2
  const bool wide = isa == Isa::Avx2 || (isa == Isa::Auto && vgpu::hostgemm::avx2_available());
  const int W = wide ? vgpu::hostgemm::avx2::ReflOps::W : vgpu::hostgemm::generic::ReflOps::W;
#else
  const bool wide = false;
  const int W = vgpu::hostgemm::generic::ReflOps::W;
#endif
  const int64_t rmin = std::min(jfirst, jlast);
  if (rmin >= m) return;
  const int64_t groups = (c1 - c0 + W - 1) / W, rows = m - rmin;
  const int64_t nref = (static_cast<int64_t>(jlast) - jfirst) * step + 1;
  const unsigned want = threads ? threads : vgpu::hostpar::plan(static_cast<uint64_t>(rows) * static_cast<uint64_t>(c1 - c0) * static_cast<uint64_t>(nref > 0 ? nref : 1), groups, 1u << 21);
  vgpu::hostpar::parallel_for(want, groups, [&](int64_t g) {
    thread_local std::vector<double> xt;
    xt.resize(static_cast<size_t>(m) * W);   // row i at xt + i*W (rows above rmin unused)
    const int64_t k0 = c0 + g * W;
    const int lanes = static_cast<int>(c1 - k0 < W ? c1 - k0 : W);
    for (int l = 0; l < lanes; ++l) {
      const double* col = c + (k0 + l) * ldc;
      for (int64_t i = rmin; i < m; ++i) xt[static_cast<size_t>(i) * W + l] = col[i];
    }
    for (int l = lanes; l < W; ++l)
      for (int64_t i = rmin; i < m; ++i) xt[static_cast<size_t>(i) * W + l] = 0.0;
    for (int j = jfirst; step > 0 ? j <= jlast : j >= jlast; j += step) {
      if (skip_zero_tau && tau[j] == 0.0) continue;
      if (j >= m) continue;
      if (wide) {
#ifdef VGPU_HOSTGEMM_AVX2
        vgpu::hostgemm::avx2::ReflOps::apply(a + static_cast<int64_t>(j) * lda, m, j, tau[j], xt.data());
#endif
      } else {
        vgpu::hostgemm::generic::ReflOps::apply(a + static_cast<int64_t>(j) * lda, m, j, tau[j], xt.data());
      }
    }
    for (int l = 0; l < lanes; ++l) {
      double* col = c + (k0 + l) * ldc;
      for (int64_t i = rmin; i < m; ++i) col[i] = xt[static_cast<size_t>(i) * W + l];
    }
  });
}

}  // namespace vgpu::hosttrsm
