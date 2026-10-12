#pragma once
// The host convolution behind cuDNN (forward, backward-data, backward-filter),
// in double, bit for bit what the element-by-element loops computed.
//
// Each output is one chain  s = s + x*w  over its taps, the product rounded and
// then the sum rounded, in a fixed order of taps (input channel, then the
// filter's taps in row-major order; for backward-filter, image then output
// position). The kernels here keep every chain's order and run the chains of a
// whole row of outputs side by side, one per vector lane; rows go to as many
// threads as the work warrants. No fused multiply-add anywhere: the wide path
// is compiled for AVX2 only (no FMA flag), chosen at run time.
//
// tests/unit/test_host_conv.cpp holds the original loops as the reference.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "host_gemm.hpp"
#include "host_parallel.hpp"

namespace vgpu::hostconv {

// A convolution of up to three spatial dimensions, right-aligned into three
// (missing ones have extent 1, stride and dilation 1, padding 0). The same
// fields as cudnn_common.hpp's ConvGeom.
struct Geom {
  int64_t N = 0, C = 0, K = 0, G = 1, Cg = 0, Kg = 0;
  int64_t in[3] = {1, 1, 1}, out[3] = {1, 1, 1}, flt[3] = {1, 1, 1};
  int64_t pad[3] = {0, 0, 0}, str[3] = {1, 1, 1}, dil[3] = {1, 1, 1};
  bool flip = false;
};

enum class Dir { Forward, Data, Filter };
enum class Isa { Auto, Generic, Avx2 };
// Forward and backward-data have two implementations that agree to the bit: row kernels (one tap at a time over a
// row of outputs) and the matrix product of host_gemm.hpp over an unrolled copy of the input. Auto picks.
enum class Algo { Auto, Rows, Gemm };

#if defined(__x86_64__) && defined(__GNUC__)
#define VGPU_HOSTCONV_AVX2 1
#endif

namespace generic {
constexpr int kVB = 16;
#include "host_conv_impl.inc"
}  // namespace generic

#ifdef VGPU_HOSTCONV_AVX2
#pragma GCC push_options
#pragma GCC target("avx2")
namespace avx2 {
constexpr int kVB = 32;
#include "host_conv_impl.inc"
}  // namespace avx2
#pragma GCC pop_options
#endif

inline bool avx2_available() {
#ifdef VGPU_HOSTCONV_AVX2
  static const bool ok = __builtin_cpu_supports("avx2");
  return ok;
#else
  return false;
#endif
}

// ---- the matrix-product form of forward and backward-data ----
//
// Forward:  y[n,k,o] = sum over taps (ci, f) of x[n, c0+ci, p(o,f)] * w[k, ci, f]       (the taps in this order)
// is, for a group, the product  W (Kg x Cg*fsz)  *  X (Cg*fsz x positions),  where column (n,o) of X lists the input
// each tap reads. The row kernels skip a tap whose input position falls in the padding; the product instead takes it
// as a 0 in X, which adds the product 0*w = +-0 to the chain. That changes no bit when w is finite and the chain is
// not -0 (it never is: it starts at +0, and a sum is -0 only when both terms are), so this path is taken only when
// every weight is finite. Backward-data is the same with taps (kk, f) and X built from dy where the output
// position the tap came from exists (stride divides, in range) and 0 elsewhere.

inline bool all_finite(const double* v, size_t n) {
  for (size_t i = 0; i < n; ++i)
    if (!std::isfinite(v[i])) return false;
  return true;
}

// The columns for positions [j0, j1) of the forward product: column j = n*osz + o, rows (ci, f0, f1, f2).
inline void fwd_columns(const Geom& g, const double* x, int64_t c0, int64_t j0, int64_t j1, double* col) {
  const int64_t I0 = g.in[0], I1 = g.in[1], I2 = g.in[2], O1 = g.out[1], O2 = g.out[2];
  const int64_t osz = g.out[0] * O1 * O2, isz = I0 * I1 * I2, fsz = g.flt[0] * g.flt[1] * g.flt[2];
  const int64_t kdim = g.Cg * fsz;
  for (int64_t j = j0; j < j1; ++j) {
    const int64_t n = j / osz, o = j % osz, o0 = o / (O1 * O2), o1 = (o / O2) % O1, o2 = o % O2;
    const double* xg = x + (n * g.C + c0) * isz;
    double* dst = col + (j - j0) * kdim;
    for (int64_t ci = 0; ci < g.Cg; ++ci) {
      const double* xc = xg + ci * isz;
      for (int64_t f0 = 0; f0 < g.flt[0]; ++f0) {
        const int64_t t0 = g.flip ? g.flt[0] - 1 - f0 : f0;
        const int64_t p0 = o0 * g.str[0] - g.pad[0] + t0 * g.dil[0];
        for (int64_t f1 = 0; f1 < g.flt[1]; ++f1) {
          const int64_t t1 = g.flip ? g.flt[1] - 1 - f1 : f1;
          const int64_t p1 = o1 * g.str[1] - g.pad[1] + t1 * g.dil[1];
          const bool row_ok = p0 >= 0 && p0 < I0 && p1 >= 0 && p1 < I1;
          const double* xr = xc + (p0 * I1 + p1) * I2;
          for (int64_t f2 = 0; f2 < g.flt[2]; ++f2) {
            const int64_t t2 = g.flip ? g.flt[2] - 1 - f2 : f2;
            const int64_t p2 = o2 * g.str[2] - g.pad[2] + t2 * g.dil[2];
            *dst++ = row_ok && p2 >= 0 && p2 < I2 ? xr[p2] : 0.0;
          }
        }
      }
    }
  }
}

// The columns for positions [j0, j1) of the backward-data product: column j = n*isz + p, rows (kk, f0, f1, f2).
inline void data_columns(const Geom& g, const double* dy, int64_t grp, int64_t j0, int64_t j1, double* col) {
  const int64_t I1 = g.in[1], I2 = g.in[2], O0 = g.out[0], O1 = g.out[1], O2 = g.out[2];
  const int64_t isz = g.in[0] * I1 * I2, osz = O0 * O1 * O2, fsz = g.flt[0] * g.flt[1] * g.flt[2];
  const int64_t kdim = g.Kg * fsz;
  for (int64_t j = j0; j < j1; ++j) {
    const int64_t n = j / isz, pp = j % isz, p0 = pp / (I1 * I2), p1 = (pp / I2) % I1, p2 = pp % I2;
    const double* yg = dy + (n * g.K + grp * g.Kg) * osz;
    double* dst = col + (j - j0) * kdim;
    for (int64_t kk = 0; kk < g.Kg; ++kk) {
      const double* yc = yg + kk * osz;
      for (int64_t f0 = 0; f0 < g.flt[0]; ++f0) {
        const int64_t t0 = g.flip ? g.flt[0] - 1 - f0 : f0;
        const int64_t n0 = p0 + g.pad[0] - t0 * g.dil[0];
        const bool ok0 = n0 >= 0 && n0 % g.str[0] == 0 && n0 / g.str[0] < O0;
        const int64_t o0 = ok0 ? n0 / g.str[0] : 0;
        for (int64_t f1 = 0; f1 < g.flt[1]; ++f1) {
          const int64_t t1 = g.flip ? g.flt[1] - 1 - f1 : f1;
          const int64_t n1 = p1 + g.pad[1] - t1 * g.dil[1];
          const bool ok1 = ok0 && n1 >= 0 && n1 % g.str[1] == 0 && n1 / g.str[1] < O1;
          const int64_t o1 = ok1 ? n1 / g.str[1] : 0;
          const double* yr = yc + (o0 * O1 + o1) * O2;
          for (int64_t f2 = 0; f2 < g.flt[2]; ++f2) {
            const int64_t t2 = g.flip ? g.flt[2] - 1 - f2 : f2;
            const int64_t n2 = p2 + g.pad[2] - t2 * g.dil[2];
            *dst++ = ok1 && n2 >= 0 && n2 % g.str[2] == 0 && n2 / g.str[2] < O2 ? yr[n2 / g.str[2]] : 0.0;
          }
        }
      }
    }
  }
}

inline void conv_gemm(const Geom& g, Dir dir, const double* a, const double* b, double* out, Isa isa, unsigned threads) {
  const int64_t isz = g.in[0] * g.in[1] * g.in[2], osz = g.out[0] * g.out[1] * g.out[2];
  const int64_t fsz = g.flt[0] * g.flt[1] * g.flt[2];
  const bool fwd = dir == Dir::Forward;
  const int64_t kdim = (fwd ? g.Cg : g.Kg) * fsz;          // taps per chain
  const int64_t positions = g.N * (fwd ? osz : isz);       // columns per group
  const int64_t rows = fwd ? g.Kg : g.Cg;                  // outputs per column and group
  // Columns in tiles that stay a few megabytes.
  const int64_t tile = std::max<int64_t>(16, (int64_t(1) << 20) / std::max<int64_t>(kdim, 1));
  std::vector<double> col, wa;
  for (int64_t grp = 0; grp < g.G; ++grp) {
    // A is rows x kdim, row i holding output i's taps in chain order: forward's filter rows as they are; backward-data's
    // are gathered, as the filter of output channel ci of this group runs across the output channels kk.
    const double* A;
    if (fwd) {
      A = b + grp * g.Kg * kdim;
    } else {
      wa.resize(static_cast<size_t>(rows * kdim));
      for (int64_t ci = 0; ci < g.Cg; ++ci)
        for (int64_t kk = 0; kk < g.Kg; ++kk)
          std::copy_n(b + ((grp * g.Kg + kk) * g.Cg + ci) * fsz, fsz, wa.data() + ci * kdim + kk * fsz);
      A = wa.data();
    }
    for (int64_t j0 = 0; j0 < positions; j0 += tile) {
      const int64_t j1 = std::min(positions, j0 + tile);
      col.resize(static_cast<size_t>((j1 - j0) * kdim));
      double* cp = col.data();
      const int64_t cols = j1 - j0;
      const uint64_t bwork = static_cast<uint64_t>(cols) * static_cast<uint64_t>(kdim);
      const int64_t chunks = (cols + 63) / 64;
      const unsigned nt = threads ? static_cast<unsigned>(std::min<int64_t>(threads, chunks))
                                  : hostpar::plan(bwork, chunks, 1u << 20);
      hostpar::parallel_for(nt, chunks, [&](int64_t ch) {
        const int64_t c0 = j0 + ch * 64, c1 = std::min(j1, c0 + 64);
        if (fwd) fwd_columns(g, a, grp * g.Cg, c0, c1, cp + (c0 - j0) * kdim);
        else data_columns(g, a, grp, c0, c1, cp + (c0 - j0) * kdim);
      });
      const int64_t outspan = fwd ? osz : isz;
      const int64_t chans = fwd ? g.K : g.C;
      const int64_t first = grp * rows;
      hostgemm::gemm<double, double>(
          true, false, rows, cols, kdim, A, kdim, cp, kdim,
          [=](int64_t i, int64_t jj, double acc) {
            const int64_t j = j0 + jj, n = j / outspan, pos = j % outspan;
            out[(n * chans + first + i) * outspan + pos] = acc;
          },
          isa == Isa::Generic ? hostgemm::Isa::Generic : isa == Isa::Avx2 ? hostgemm::Isa::Avx2 : hostgemm::Isa::Auto, threads);
    }
  }
}

// out = the sums of one direction, each as one chain (see cudnn_common.hpp's
// convolve for the layouts). a and b are in logical packed order; out holds
// the whole result. Forward: a = x, b = w. Data: a = dy, b = w. Filter: a = dy, b = x.
inline void convolve_exact(const Geom& g, Dir dir, const double* a, const double* b, double* out, Isa isa = Isa::Auto,
                           unsigned threads = 0, Algo algo = Algo::Auto) {
#ifdef VGPU_HOSTCONV_AVX2
  const bool wide = isa == Isa::Avx2 || (isa == Isa::Auto && avx2_available());
#else
  const bool wide = false;
  (void)isa;
#endif
  const int64_t N = g.N, C = g.C, K = g.K;
  const int64_t I0 = g.in[0], I1 = g.in[1], I2 = g.in[2], O0 = g.out[0], O1 = g.out[1], O2 = g.out[2];
  const int64_t isz = I0 * I1 * I2, osz = O0 * O1 * O2, fsz = g.flt[0] * g.flt[1] * g.flt[2];
  auto pick = [&](uint64_t work, int64_t items) {
    return threads ? std::min<int64_t>(threads, items) : static_cast<int64_t>(hostpar::plan(work, items));
  };
  // The product form, for forward and backward-data when the weights are finite and a column has enough outputs to
  // fill the kernel's rows.
  if (dir != Dir::Filter && algo != Algo::Rows) {
    const int64_t rows = dir == Dir::Forward ? g.Kg : g.Cg;
    // (Backward-data with a stride has mostly zero columns -- each input position meets one tap in stride^n -- so
    // there the row kernels, which skip them, do less work.)
    const bool dense = dir == Dir::Forward || (g.str[0] == 1 && g.str[1] == 1 && g.str[2] == 1);
    const bool shaped = dense && rows >= 4 && (dir == Dir::Forward ? g.Cg : g.Kg) * fsz >= 4;
    const size_t wcount = static_cast<size_t>(K * g.Cg * fsz);
    if ((algo == Algo::Gemm || shaped) && all_finite(b, wcount)) {
      conv_gemm(g, dir, a, b, out, isa, threads);
      return;
    }
  }
  switch (dir) {
    case Dir::Forward: {
      const int64_t items = N * K * O0;
      const uint64_t work = static_cast<uint64_t>(N * K * osz) * static_cast<uint64_t>(g.Cg * fsz);
      hostpar::parallel_for(static_cast<unsigned>(pick(work, items)), items, [&](int64_t it) {
        const int64_t o0 = it % O0, k = (it / O0) % K, n = it / (O0 * K);
        const double* x = a + (n * C + (k / g.Kg) * g.Cg) * isz;
        const double* w = b + k * g.Cg * fsz;
        for (int64_t o1 = 0; o1 < O1; ++o1) {
          double* y = out + (n * K + k) * osz + (o0 * O1 + o1) * O2;
          if (wide) {
#ifdef VGPU_HOSTCONV_AVX2
            avx2::fwd_row(g, x, w, o0, o1, y);
#endif
          } else {
            generic::fwd_row(g, x, w, o0, o1, y);
          }
        }
      });
      break;
    }
    case Dir::Data: {
      const int64_t items = N * C * I0;
      const uint64_t work = static_cast<uint64_t>(N * C * isz) * static_cast<uint64_t>(g.Kg * fsz);
      hostpar::parallel_for(static_cast<unsigned>(pick(work, items)), items, [&](int64_t it) {
        const int64_t p0 = it % I0, c = (it / I0) % C, n = it / (I0 * C);
        const double* dy = a + n * K * osz;
        for (int64_t p1 = 0; p1 < I1; ++p1) {
          double* dx = out + (n * C + c) * isz + (p0 * I1 + p1) * I2;
          if (wide) {
#ifdef VGPU_HOSTCONV_AVX2
            avx2::data_row(g, dy, b, c, p0, p1, dx);
#endif
          } else {
            generic::data_row(g, dy, b, c, p0, p1, dx);
          }
        }
      });
      break;
    }
    case Dir::Filter: {
      const int cb = wide ? 8 : 4;
      // The block kernel advances 4 filters x one vector pair of channels per step, which pays off when a block holds
      // enough real chains (a depthwise convolution has one per filter): fewer than 8 go the plain way, one chain at
      // a time, with the filters spread over threads.
      const int64_t useful = std::min<int64_t>(g.Kg, 4) * std::min<int64_t>(g.Cg, cb);
      if (useful < 8) {
        const int64_t items = g.K * g.Cg;
        const uint64_t work = static_cast<uint64_t>(K * g.Cg * fsz) * static_cast<uint64_t>(N * osz);
        hostpar::parallel_for(static_cast<unsigned>(pick(work, items)), items, [&](int64_t it) {
          const int64_t k = it / g.Cg, ci = it % g.Cg;
          const int64_t c0 = (k / g.Kg) * g.Cg;
          for (int64_t f0 = 0; f0 < g.flt[0]; ++f0)
            for (int64_t f1 = 0; f1 < g.flt[1]; ++f1)
              for (int64_t f2 = 0; f2 < g.flt[2]; ++f2) {
                double sum = 0.0;
                for (int64_t n = 0; n < N; ++n) {
                  const double* yb = a + (n * K + k) * osz;
                  const double* xb = b + (n * C + c0 + ci) * isz;
                  for (int64_t o0 = 0; o0 < O0; ++o0) {
                    const int64_t t0 = g.flip ? g.flt[0] - 1 - f0 : f0;
                    const int64_t p0 = o0 * g.str[0] - g.pad[0] + t0 * g.dil[0];
                    if (p0 < 0 || p0 >= I0) continue;
                    for (int64_t o1 = 0; o1 < O1; ++o1) {
                      const int64_t t1 = g.flip ? g.flt[1] - 1 - f1 : f1;
                      const int64_t p1 = o1 * g.str[1] - g.pad[1] + t1 * g.dil[1];
                      if (p1 < 0 || p1 >= I1) continue;
                      const int64_t t2 = g.flip ? g.flt[2] - 1 - f2 : f2;
                      const int64_t off2 = -g.pad[2] + t2 * g.dil[2];
                      int64_t lo = 0, hi = 0;
                      if (off2 < 0) lo = (-off2 + g.str[2] - 1) / g.str[2];
                      if (I2 - 1 - off2 >= 0) hi = std::min<int64_t>(O2, (I2 - 1 - off2) / g.str[2] + 1);
                      const double* yr = yb + (o0 * O1 + o1) * O2;
                      const double* xr = xb + (p0 * I1 + p1) * I2;
                      for (int64_t o2 = lo; o2 < hi; ++o2) sum = sum + yr[o2] * xr[o2 * g.str[2] + off2];
                    }
                  }
                }
                out[(k * g.Cg + ci) * fsz + (f0 * g.flt[1] + f1) * g.flt[2] + f2] = sum;
              }
        });
        break;
      }
      // x with its channels innermost, padded so a block of channels may read past the last one.
      const int64_t cs = C + 8;
      std::vector<double> xt(static_cast<size_t>(N * isz * cs), 0.0);
      for (int64_t n = 0; n < N; ++n)
        for (int64_t c = 0; c < C; ++c) {
          const double* src = b + (n * C + c) * isz;
          double* dst = xt.data() + n * isz * cs + c;
          for (int64_t i = 0; i < isz; ++i) dst[i * cs] = src[i];
        }
      const int64_t kblocks_per_group = (g.Kg + 3) / 4, cblocks = (g.Cg + cb - 1) / cb;
      const int64_t items = g.G * kblocks_per_group * cblocks;
      const uint64_t work = static_cast<uint64_t>(K * g.Cg * fsz) * static_cast<uint64_t>(N * osz);
      hostpar::parallel_for(static_cast<unsigned>(pick(work, items)), items, [&](int64_t it) {
        const int64_t cblk = it % cblocks, kblk = (it / cblocks) % kblocks_per_group, grp = it / (cblocks * kblocks_per_group);
        const int64_t k0 = grp * g.Kg + kblk * 4;
        const int kcount = static_cast<int>(std::min<int64_t>(4, grp * g.Kg + g.Kg - k0));
        if (wide) {
#ifdef VGPU_HOSTCONV_AVX2
          avx2::filt_block(g, a, xt.data(), cs, k0, kcount, cblk * cb, out);
#endif
        } else {
          generic::filt_block(g, a, xt.data(), cs, k0, kcount, cblk * cb, out);
        }
      });
      break;
    }
  }
}

}  // namespace vgpu::hostconv
