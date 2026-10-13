// Triangular solves with the right-hand sides along the vector lanes, included once per instruction set after
// host_gemm_impl.inc (inside the same namespaces, so Lanes is at hand; same rule: nothing here calls a std:: template).
//
// For one vector (a column of B for a left solve, a row for a right one) the old loops did, for i in the order of the
// solve:   s = x[i];  for j in the row's span, ascending:  s = s - M(i,j) * x[j];   x[i] = unit ? s : s / M(i,i).
// Here W vectors are solved at once, one per lane: the same steps in the same order for each, so the same bits.
// xt holds the panel with element i of lane l at xt[i*W + l].

constexpr int kTrsmWV = 4;

struct TrsmOps {
  using V = typename Lanes<double>::type;
  static constexpr int L = Lanes<double>::n;
  static constexpr int WV = kTrsmWV;
  static constexpr int W = WV * L;

  // x[i] for one i of a real solve. row = M(i, ·) at a + i*sti, stepping stj; the span of j is [j0, j1).
  static void real_row(const double* a, int64_t sti, int64_t stj, int64_t i, int64_t j0, int64_t j1, bool unit, double* xt) {
    V s[WV];
    for (int v = 0; v < WV; ++v) __builtin_memcpy(&s[v], xt + i * W + v * L, sizeof(V));
    const double* mrow = a + i * sti;
    for (int64_t j = j0; j < j1; ++j) {
      const double m = mrow[j * stj];
      V mv;
      for (int l = 0; l < L; ++l) mv[l] = m;
      const double* xj = xt + j * W;
      for (int v = 0; v < WV; ++v) {
        V x;
        __builtin_memcpy(&x, xj + v * L, sizeof(V));
        s[v] = s[v] - mv * x;
      }
    }
    if (!unit) {
      const double d = mrow[i * stj];
      V dv;
      for (int l = 0; l < L; ++l) dv[l] = d;
      for (int v = 0; v < WV; ++v) s[v] = s[v] / dv;
    }
    for (int v = 0; v < WV; ++v) __builtin_memcpy(xt + i * W + v * L, &s[v], sizeof(V));
  }

  // The subtraction part of x[i] for a complex solve (the division is the caller's, in std::complex<double>):
  // s = s - M(i,j) * x[j] for j in [j0, j1), M(i, j) = (mr[i*k + j], mi[i*k + j]); xr, xi the panel's planes.
  static void cplx_row(const double* mr, const double* mi, int64_t k, int64_t i, int64_t j0, int64_t j1, double* xr, double* xi) {
    V sr[WV], si[WV];
    for (int v = 0; v < WV; ++v) {
      __builtin_memcpy(&sr[v], xr + i * W + v * L, sizeof(V));
      __builtin_memcpy(&si[v], xi + i * W + v * L, sizeof(V));
    }
    for (int64_t j = j0; j < j1; ++j) {
      const double a = mr[i * k + j], b = mi[i * k + j];
      V av, bv;
      for (int l = 0; l < L; ++l) {
        av[l] = a;
        bv[l] = b;
      }
      for (int v = 0; v < WV; ++v) {
        V xrj, xij;
        __builtin_memcpy(&xrj, xr + j * W + v * L, sizeof(V));
        __builtin_memcpy(&xij, xi + j * W + v * L, sizeof(V));
        const V zr = av * xrj - bv * xij, zi = av * xij + bv * xrj;
        sr[v] = sr[v] - zr;
        si[v] = si[v] - zi;
      }
    }
    for (int v = 0; v < WV; ++v) {
      __builtin_memcpy(xr + i * W + v * L, &sr[v], sizeof(V));
      __builtin_memcpy(xi + i * W + v * L, &si[v], sizeof(V));
    }
  }
};
