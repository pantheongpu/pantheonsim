// The row kernels of host_conv.hpp, included once per instruction set (see
// host_gemm_impl.inc for why nothing in here may use a std:: template).
//
// Every output element is one chain  s = s + x*w  over its taps in a fixed
// order. A row kernel advances a whole row of outputs together: for each tap
// (in that order) it adds the tap's product to every output of the row it
// reaches. Each element still sees its own taps in the original order, so its
// bits are the ones the element-by-element loop gave.

// The first and last output index o in [0, count) with 0 <= o*str + off < extent; lo >= hi when none.
static inline void span_of(int64_t off, int64_t str, int64_t extent, int64_t count, int64_t* lo, int64_t* hi) {
  int64_t l = 0;
  if (off < 0) l = (-off + str - 1) / str;
  int64_t h = 0;
  if (extent - 1 - off >= 0) h = (extent - 1 - off) / str + 1;
  if (h > count) h = count;
  *lo = l;
  *hi = h;
}

// s[i] = s[i] + x[i] * w, for contiguous i.
static inline __attribute__((always_inline)) void axpy1(double* __restrict s, const double* __restrict x, double w,
                                                          int64_t n) {
  for (int64_t i = 0; i < n; ++i) s[i] = s[i] + x[i] * w;
}
// s[i*ss] = s[i*ss] + x[i*xs] * w.
static inline __attribute__((always_inline)) void axpy(double* s, int64_t ss, const double* x, int64_t xs, double w,
                                                        int64_t n) {
  if (ss == 1 && xs == 1) {
    axpy1(s, x, w, n);
    return;
  }
  for (int64_t i = 0; i < n; ++i) s[i * ss] = s[i * ss] + x[i * xs] * w;
}

// Forward: the output row y[n, k, o0, o1, 0 .. O2).
void fwd_row(const Geom& g, const double* x /* image n, channel c0 of the group */, const double* w /* filter k */,
             int64_t o0, int64_t o1, double* y) {
  const int64_t I0 = g.in[0], I1 = g.in[1], I2 = g.in[2], O2 = g.out[2];
  const int64_t F0 = g.flt[0], F1 = g.flt[1], F2 = g.flt[2];
  const int64_t isz = I0 * I1 * I2, fsz = F0 * F1 * F2;
  for (int64_t o = 0; o < O2; ++o) y[o] = 0.0;
  for (int64_t ci = 0; ci < g.Cg; ++ci) {
    const double* xb = x + ci * isz;
    const double* wb = w + ci * fsz;
    for (int64_t f0 = 0; f0 < F0; ++f0) {
      const int64_t t0 = g.flip ? F0 - 1 - f0 : f0;
      const int64_t p0 = o0 * g.str[0] - g.pad[0] + t0 * g.dil[0];
      if (p0 < 0 || p0 >= I0) continue;
      for (int64_t f1 = 0; f1 < F1; ++f1) {
        const int64_t t1 = g.flip ? F1 - 1 - f1 : f1;
        const int64_t p1 = o1 * g.str[1] - g.pad[1] + t1 * g.dil[1];
        if (p1 < 0 || p1 >= I1) continue;
        const double* row = xb + (p0 * I1 + p1) * I2;
        const double* wrow = wb + (f0 * F1 + f1) * F2;
        for (int64_t f2 = 0; f2 < F2; ++f2) {
          const int64_t t2 = g.flip ? F2 - 1 - f2 : f2;
          const int64_t off = -g.pad[2] + t2 * g.dil[2];
          int64_t lo, hi;
          span_of(off, g.str[2], I2, O2, &lo, &hi);
          if (lo < hi) axpy(y + lo, 1, row + lo * g.str[2] + off, g.str[2], wrow[f2], hi - lo);
        }
      }
    }
  }
}

// Data: the input-gradient row dx[n, c, p0, p1, 0 .. I2).
void data_row(const Geom& g, const double* dy /* image n */, const double* w, int64_t c, int64_t p0, int64_t p1,
              double* dx) {
  const int64_t I2 = g.in[2];
  const int64_t O0 = g.out[0], O1 = g.out[1], O2 = g.out[2];
  const int64_t F0 = g.flt[0], F1 = g.flt[1], F2 = g.flt[2];
  const int64_t osz = O0 * O1 * O2, fsz = F0 * F1 * F2;
  const int64_t grp = c / g.Cg, ci = c % g.Cg;
  for (int64_t i = 0; i < I2; ++i) dx[i] = 0.0;
  for (int64_t kk = 0; kk < g.Kg; ++kk) {
    const int64_t k = grp * g.Kg + kk;
    const double* yb = dy + k * osz;
    const double* wb = w + (k * g.Cg + ci) * fsz;
    for (int64_t f0 = 0; f0 < F0; ++f0) {
      const int64_t t0 = g.flip ? F0 - 1 - f0 : f0;
      const int64_t n0 = p0 + g.pad[0] - t0 * g.dil[0];
      if (n0 < 0 || n0 % g.str[0]) continue;
      const int64_t o0 = n0 / g.str[0];
      if (o0 >= O0) continue;
      for (int64_t f1 = 0; f1 < F1; ++f1) {
        const int64_t t1 = g.flip ? F1 - 1 - f1 : f1;
        const int64_t n1 = p1 + g.pad[1] - t1 * g.dil[1];
        if (n1 < 0 || n1 % g.str[1]) continue;
        const int64_t o1 = n1 / g.str[1];
        if (o1 >= O1) continue;
        const double* yrow = yb + (o0 * O1 + o1) * O2;
        const double* wrow = wb + (f0 * F1 + f1) * F2;
        for (int64_t f2 = 0; f2 < F2; ++f2) {
          const int64_t t2 = g.flip ? F2 - 1 - f2 : f2;
          const int64_t off = -g.pad[2] + t2 * g.dil[2];   // dx index of output o2 is o2*str + off
          int64_t lo, hi;
          span_of(off, g.str[2], I2, O2, &lo, &hi);
          if (lo < hi) axpy(dx + lo * g.str[2] + off, g.str[2], yrow + lo, 1, wrow[f2], hi - lo);
        }
      }
    }
  }
}

// Filter: dw[k, ci, f] is one chain over (image, output position), the products dy*x. Chains of different
// outputs are independent, so 4 filters k and CB input channels advance together: the accumulators are 4 x 2
// vectors, one lane per channel, which needs x with its channels innermost (xt[n][position][channel], the
// channel stride cs). One call does a block of up to 4 filters of one group and CB channels from ci0, for all
// taps, and writes their dw.
void filt_block(const Geom& g, const double* dy, const double* xt, int64_t cs, int64_t k0, int kcount, int64_t ci0,
                double* dw) {
  typedef double V __attribute__((vector_size(kVB)));
  constexpr int L = kVB / 8;
  constexpr int CB = 2 * L;
  const int64_t I0 = g.in[0], I1 = g.in[1], I2 = g.in[2];
  const int64_t O0 = g.out[0], O1 = g.out[1], O2 = g.out[2];
  const int64_t F0 = g.flt[0], F1 = g.flt[1], F2 = g.flt[2];
  const int64_t isz = I0 * I1 * I2, osz = O0 * O1 * O2, fsz = F0 * F1 * F2;
  const int64_t c0 = (k0 / g.Kg) * g.Cg + ci0;   // the first input channel of this block
  const double* kplane[4];
  for (int q = 0; q < 4; ++q) kplane[q] = dy + (k0 + (q < kcount ? q : 0)) * osz;   // (a clamped plane's sums are not stored)
  for (int64_t f0 = 0; f0 < F0; ++f0) {
    const int64_t t0 = g.flip ? F0 - 1 - f0 : f0;
    for (int64_t f1 = 0; f1 < F1; ++f1) {
      const int64_t t1 = g.flip ? F1 - 1 - f1 : f1;
      for (int64_t f2 = 0; f2 < F2; ++f2) {
        const int64_t t2 = g.flip ? F2 - 1 - f2 : f2;
        const int64_t off2 = -g.pad[2] + t2 * g.dil[2];
        int64_t lo, hi;
        span_of(off2, g.str[2], I2, O2, &lo, &hi);
        V acc[4][2];
        for (int q = 0; q < 4; ++q)
          for (int v = 0; v < 2; ++v)
            for (int l = 0; l < L; ++l) acc[q][v][l] = 0.0;
        if (lo < hi) {
          for (int64_t n = 0; n < g.N; ++n) {
            const double* xn = xt + n * isz * cs + c0;
            for (int64_t o0 = 0; o0 < O0; ++o0) {
              const int64_t p0 = o0 * g.str[0] - g.pad[0] + t0 * g.dil[0];
              if (p0 < 0 || p0 >= I0) continue;
              for (int64_t o1 = 0; o1 < O1; ++o1) {
                const int64_t p1 = o1 * g.str[1] - g.pad[1] + t1 * g.dil[1];
                if (p1 < 0 || p1 >= I1) continue;
                const double* yr[4];
                for (int q = 0; q < 4; ++q) yr[q] = kplane[q] + (n * g.K * osz) + (o0 * O1 + o1) * O2;
                const double* xr = xn + (p0 * I1 + p1) * I2 * cs;
                for (int64_t o2 = lo; o2 < hi; ++o2) {
                  const double* xp = xr + (o2 * g.str[2] + off2) * cs;
                  V x0, x1;
                  __builtin_memcpy(&x0, xp, sizeof(V));
                  __builtin_memcpy(&x1, xp + L, sizeof(V));
                  for (int q = 0; q < 4; ++q) {
                    const double yv = yr[q][o2];
                    V y;
                    for (int l = 0; l < L; ++l) y[l] = yv;
                    acc[q][0] = acc[q][0] + y * x0;
                    acc[q][1] = acc[q][1] + y * x1;
                  }
                }
              }
            }
          }
        }
        const int64_t f = (f0 * F1 + f1) * F2 + f2;
        for (int q = 0; q < kcount; ++q)
          for (int c = 0; c < CB && ci0 + c < g.Cg; ++c)
            dw[((k0 + q) * g.Cg + ci0 + c) * fsz + f] = acc[q][c / L][c % L];
      }
    }
  }
}
