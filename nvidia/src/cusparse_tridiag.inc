// cuSPARSE's tridiagonal and pentadiagonal solvers, included by cusparse_api.cpp.
//
//   gtsv2                 LU with partial pivoting, B overwritten by X
//   gtsv2_nopivot         parallel cyclic reduction (PCR), no pivoting
//   gtsv2StridedBatch     the same, one right-hand side per system
//   gtsvInterleavedBatch  algo 0 Thomas, 1 LU with partial pivoting, 2 QR
//   gpsvInterleavedBatch  QR (Givens) of the pentadiagonal system
//
// What an RTX 3060 showed with cuSPARSE 13.0 (CUDA 13.0), which is what this
// reproduces -- nvidia/tests/e2e/sparse_tridiag_paths.cu checks it:
//
//   * gtsv2, gtsv2_nopivot and gtsv2StridedBatch refuse m < 3, n < 0 and
//     ldb < max(1, m) with INVALID_VALUE; gtsv2StridedBatch also refuses
//     batchCount < 1 and batchStride < m. Their _bufferSizeExt checks nothing
//     but StridedBatch's (m, batchCount, batchStride as above). n = 0 is a
//     successful no-op.
//   * The interleaved solvers take any m >= 0 and batchCount >= 0 (zero of
//     either is a no-op) and refuse an algo outside 0..2 (gtsv) or 0..1
//     (gpsv; NVIDIA documents only 0, and 1 solves the same way).
//   * The first element of each lower diagonal (and of gpsv's ds, the first
//     two) and the last of each upper one are never read.
//   * gtsv2 solves a system whose leading entries are zero, as LU with partial
//     pivoting does, and leaves dl, d and du as they were; a singular system
//     comes back NaN throughout.
//   * gtsv2_nopivot and gtsv2StridedBatch are PCR up to 2048 unknowns: a zero
//     pivot spoils one half of the unknowns (with d[0] = 0, every odd one is
//     NaN and every even one is right) and leaves the other half alone. Past
//     that, cyclic reduction (CR) steps first halve the system until PCR
//     takes it, and the same zero spoils every unknown.
//   * gtsvInterleavedBatch overwrites its inputs: Thomas (algo 0) leaves the
//     modified upper diagonal c'_i = du_i / (d_i - dl_i c'_{i-1}) in du[0..m-2];
//     LU (algo 1) leaves 1/u_ii in d[i] and u_{i,i+1} in du[i] for i < m-2;
//     QR (algo 2) leaves r_ii in d[i] and r_{i,i+1} in du[i] for i < m-2, with
//     NVIDIA's sign: each Givens step takes r = -sign(p) hypot(p, q) and
//     reflects the two rows (row_i' = c row_i + s row_{i+1}, row_{i+1}' =
//     s row_i - c row_{i+1}, c = p / r, s = q / r). dl is left as it was.
//
// The arithmetic runs in double (complex<double>) like the rest of this
// library, so results agree with the card's to its rounding, not bit for bit.

namespace {

// PCR on one system, x holding the right-hand side on entry and the solution
// on exit. a[0] and c[m-1] are taken as zero, whatever the caller stored.
template <class V> void pcr_solve(std::vector<V> a, std::vector<V> b, std::vector<V> c, V* x) {
  const size_t m = b.size();
  std::vector<V> d(x, x + m);
  a[0] = V(0.0);
  c[m - 1] = V(0.0);
  std::vector<V> a2(m), b2(m), c2(m), d2(m);
  for (size_t s = 1; s < m; s *= 2) {
    for (size_t i = 0; i < m; ++i) {
      V bb = b[i], dd = d[i], aa(0.0), cc(0.0);
      if (i >= s) {
        const V k1 = a[i] / b[i - s];
        bb -= c[i - s] * k1;
        dd -= d[i - s] * k1;
        aa = -(a[i - s] * k1);
      }
      if (i + s < m) {
        const V k2 = c[i] / b[i + s];
        bb -= a[i + s] * k2;
        dd -= d[i + s] * k2;
        cc = -(c[i + s] * k2);
      }
      a2[i] = aa; b2[i] = bb; c2[i] = cc; d2[i] = dd;
    }
    a.swap(a2); b.swap(b2); c.swap(c2); d.swap(d2);
  }
  for (size_t i = 0; i < m; ++i) x[i] = d[i] / b[i];
}

// The largest system NVIDIA's nopivot solvers hand to PCR whole (measured: a
// zero d[0] leaves the even unknowns right at m = 2048 and spoils them all
// at 4096).
constexpr size_t kPcrLimit = 2048;

// CR steps (keeping the odd equations) until PCR takes the system.
template <class V> void cr_pcr_solve(std::vector<V> a, std::vector<V> b, std::vector<V> c, V* x) {
  const size_t m = b.size();
  if (m <= kPcrLimit) return pcr_solve(std::move(a), std::move(b), std::move(c), x);
  a[0] = V(0.0);
  c[m - 1] = V(0.0);
  const size_t half = m / 2;
  std::vector<V> a2(half), b2(half), c2(half), d2(half);
  for (size_t k = 0; k < half; ++k) {
    const size_t i = 2 * k + 1;
    const V k1 = a[i] / b[i - 1];
    V bb = b[i] - c[i - 1] * k1, dd = x[i] - x[i - 1] * k1, cc(0.0);
    const V aa = -(a[i - 1] * k1);
    if (i + 1 < m) {
      const V k2 = c[i] / b[i + 1];
      bb -= a[i + 1] * k2;
      dd -= x[i + 1] * k2;
      cc = -(c[i + 1] * k2);
    }
    a2[k] = aa; b2[k] = bb; c2[k] = cc; d2[k] = dd;
  }
  cr_pcr_solve(std::move(a2), std::move(b2), std::move(c2), d2.data());
  for (size_t k = 0; k < half; ++k) x[2 * k + 1] = d2[k];
  for (size_t i = 0; i < m; i += 2) {
    V r = x[i];
    if (i > 0) r -= a[i] * x[i - 1];
    if (i + 1 < m) r -= c[i] * x[i + 1];
    x[i] = r / b[i];
  }
}

inline double mag1(double v) { return std::fabs(v); }
inline double mag1(cd v) { return std::fabs(v.real()) + std::fabs(v.imag()); }

// LU with partial pivoting (LAPACK's xGTSV) of one system for nrhs right-hand
// sides, column j of X at x + j * ldx. With `factors`, d[i] and du[i] are left
// holding 1/u_ii and u_{i,i+1} for i < m - 2, as gtsvInterleavedBatch's LU does.
template <class V>
void lu_solve(std::vector<V> dl, std::vector<V> d, std::vector<V> du, V* x, int nrhs, size_t ldx,
              V* d_out = nullptr, V* du_out = nullptr) {
  const size_t m = d.size();
  // dl[i] is the subdiagonal entry of row i (dl[0] unused); l[i] that of row i + 1.
  std::vector<V> l(m, V(0.0)), du2(m, V(0.0));
  for (size_t i = 0; i + 1 < m; ++i) l[i] = dl[i + 1];
  du[m - 1] = V(0.0);
  std::vector<int> swapped(m, 0);
  std::vector<V> mult(m, V(0.0));
  for (size_t i = 0; i + 1 < m; ++i) {
    if (mag1(d[i]) >= mag1(l[i])) {
      const V f = l[i] / d[i];
      mult[i] = f;
      d[i + 1] -= f * du[i];
    } else {
      // Interchange rows i and i + 1.
      const V f = d[i] / l[i];
      mult[i] = f;
      swapped[i] = 1;
      d[i] = l[i];
      const V t = d[i + 1];
      d[i + 1] = du[i] - f * t;
      if (i + 2 < m) {
        du2[i] = du[i + 1];
        du[i + 1] = -(f * du2[i]);
      }
      du[i] = t;
    }
    if (d_out && i + 2 < m) {
      d_out[i] = V(1.0) / d[i];
      du_out[i] = du[i];
    }
  }
  for (int j = 0; j < nrhs; ++j) {
    V* b = x + (size_t)j * ldx;
    for (size_t i = 0; i + 1 < m; ++i) {
      if (swapped[i]) {
        const V t = b[i];
        b[i] = b[i + 1];
        b[i + 1] = t - mult[i] * b[i + 1];
      } else {
        b[i + 1] -= mult[i] * b[i];
      }
    }
    b[m - 1] /= d[m - 1];
    if (m > 1) b[m - 2] = (b[m - 2] - du[m - 2] * b[m - 1]) / d[m - 2];
    for (size_t i = m >= 2 ? m - 2 : 0; i-- > 0;) b[i] = (b[i] - du[i] * b[i + 1] - du2[i] * b[i + 2]) / d[i];
  }
}

// Thomas's algorithm, no pivoting. du_out[i] gets c'_i for i < m - 1.
template <class V> void thomas_solve(const std::vector<V>& a, const std::vector<V>& b, const std::vector<V>& c, V* x,
                                     V* du_out) {
  const size_t m = b.size();
  std::vector<V> cp(m), dp(m);
  cp[0] = m > 1 ? c[0] / b[0] : V(0.0);
  dp[0] = x[0] / b[0];
  for (size_t i = 1; i < m; ++i) {
    const V den = b[i] - a[i] * cp[i - 1];
    cp[i] = i + 1 < m ? c[i] / den : V(0.0);
    dp[i] = (x[i] - a[i] * dp[i - 1]) / den;
  }
  x[m - 1] = dp[m - 1];
  for (size_t i = m - 1; i-- > 0;) x[i] = dp[i] - cp[i] * x[i + 1];
  if (du_out)
    for (size_t i = 0; i + 1 < m; ++i) du_out[i] = cp[i];
}

// One Givens step, NVIDIA's convention (see the top of this file): for pivot p
// and the entry q below it, r = -sign(p) hypot(p, q), c = p / r, s = q / r,
// and the pair of rows (u, v) becomes (conj(c) u + conj(s) v, s u - c v).
template <class V> struct Givens {
  V c, s, r;
  Givens(V p, V q) {
    const double h = std::sqrt(std::norm(cd(re(p), im(p))) + std::norm(cd(re(q), im(q))));
    const double ap = std::abs(cd(re(p), im(p)));
    const V sign = ap == 0.0 ? V(1.0) : p / V(ap);
    r = -(sign * V(h));
    c = p / r;
    s = q / r;
  }
  void apply(V& u, V& v) const {
    const V nu = conj_of(c) * u + conj_of(s) * v;
    v = s * u - c * v;
    u = nu;
  }
};

// QR of a banded system with `lower` subdiagonals and `upper` superdiagonals,
// by Givens steps that zero each column's subdiagonal entries nearest first.
// band[k][i] is A(i, i + k - lower). With `d_out`, d_out[j] and du_out[j] get
// r_jj and r_{j,j+1} for j < m - 2, as gtsvInterleavedBatch's QR leaves them.
template <class V>
void qr_band_solve(const std::vector<std::vector<V>>& band, int lower, int upper, V* x, V* d_out = nullptr,
                   V* du_out = nullptr) {
  const int m = (int)band[(size_t)lower].size();
  // Row i holds columns [i - lower, i + upper + lower]: R gains `lower` more
  // superdiagonals than A has.
  const int width = 2 * lower + upper + 1;
  std::vector<V> rows((size_t)m * width, V(0.0));
  auto at = [&](int i, int col) -> V& { return rows[(size_t)i * width + (size_t)(col - i + lower)]; };
  for (int i = 0; i < m; ++i)
    for (int k = 0; k <= lower + upper; ++k) {
      const int col = i + k - lower;
      if (col >= 0 && col < m) at(i, col) = band[(size_t)k][(size_t)i];
    }
  for (int j = 0; j < m; ++j) {
    const int last = std::min(m - 1, j + upper + lower);
    for (int i = j + 1; i <= std::min(m - 1, j + lower); ++i) {
      const Givens<V> g(at(j, j), at(i, j));
      for (int col = j; col <= last; ++col) g.apply(at(j, col), at(i, col));
      g.apply(x[j], x[i]);
    }
    if (d_out && j + 2 < m) {
      d_out[j] = at(j, j);
      du_out[j] = at(j, j + 1);
    }
  }
  for (int i = m - 1; i >= 0; --i) {
    V s = x[i];
    for (int col = i + 1; col <= std::min(m - 1, i + upper + lower); ++col) s -= at(i, col) * x[col];
    x[i] = s / at(i, i);
  }
}

template <class T>
cusparseStatus_t gtsv2(cusparseHandle_t h, int m, int n, const T* dl, const T* d, const T* du, T* B, int ldb,
                       bool pivot) {
  using V = typename Legacy<T>::V;
  constexpr cudaDataType t = Legacy<T>::type;
  if (!known(h)) return CUSPARSE_STATUS_NOT_INITIALIZED;
  if (m < 3 || n < 0 || ldb < std::max(1, m)) return CUSPARSE_STATUS_INVALID_VALUE;
  if (n == 0) return CUSPARSE_STATUS_SUCCESS;
  std::vector<V> a, b, c, x;
  if (!fetch_values(dl, (size_t)m, t, &a) || !fetch_values(d, (size_t)m, t, &b) || !fetch_values(du, (size_t)m, t, &c) ||
      !fetch_values(B, (size_t)ldb * (n - 1) + m, t, &x))
    return CUSPARSE_STATUS_EXECUTION_FAILED;
  if (pivot) {
    lu_solve(a, b, c, x.data(), n, (size_t)ldb);
  } else {
    for (int j = 0; j < n; ++j) cr_pcr_solve(a, b, c, x.data() + (size_t)j * ldb);
  }
  return store_values(B, x, t) ? CUSPARSE_STATUS_SUCCESS : CUSPARSE_STATUS_EXECUTION_FAILED;
}

template <class T>
cusparseStatus_t gtsv2_strided(cusparseHandle_t h, int m, const T* dl, const T* d, const T* du, T* X, int count,
                               int stride) {
  using V = typename Legacy<T>::V;
  constexpr cudaDataType t = Legacy<T>::type;
  if (!known(h)) return CUSPARSE_STATUS_NOT_INITIALIZED;
  if (m < 3 || count < 1 || stride < m) return CUSPARSE_STATUS_INVALID_VALUE;
  const size_t total = (size_t)stride * (count - 1) + m;
  std::vector<V> a, b, c, x;
  if (!fetch_values(dl, total, t, &a) || !fetch_values(d, total, t, &b) || !fetch_values(du, total, t, &c) ||
      !fetch_values(X, total, t, &x))
    return CUSPARSE_STATUS_EXECUTION_FAILED;
  for (int k = 0; k < count; ++k) {
    const size_t o = (size_t)k * stride;
    cr_pcr_solve(std::vector<V>(a.begin() + o, a.begin() + o + m), std::vector<V>(b.begin() + o, b.begin() + o + m),
                 std::vector<V>(c.begin() + o, c.begin() + o + m), x.data() + o);
  }
  return store_values(X, x, t) ? CUSPARSE_STATUS_SUCCESS : CUSPARSE_STATUS_EXECUTION_FAILED;
}

// Interleaved batches: element i of system k is at i * count + k.
template <class V> std::vector<V> deinterleave(const std::vector<V>& v, int m, int count, int k) {
  std::vector<V> out((size_t)m);
  for (int i = 0; i < m; ++i) out[(size_t)i] = v[(size_t)i * count + k];
  return out;
}
template <class V> void interleave(std::vector<V>& v, const std::vector<V>& in, int m, int count, int k, int limit) {
  for (int i = 0; i < std::min(m, limit); ++i) v[(size_t)i * count + k] = in[(size_t)i];
}

template <class T>
cusparseStatus_t gtsv_interleaved(cusparseHandle_t h, int algo, int m, T* dl, T* d, T* du, T* X, int count) {
  using V = typename Legacy<T>::V;
  constexpr cudaDataType t = Legacy<T>::type;
  if (!known(h)) return CUSPARSE_STATUS_NOT_INITIALIZED;
  if (algo < 0 || algo > 2 || m < 0 || count < 0) return CUSPARSE_STATUS_INVALID_VALUE;
  if (m == 0 || count == 0) return CUSPARSE_STATUS_SUCCESS;
  const size_t total = (size_t)m * count;
  std::vector<V> a, b, c, x;
  if (!fetch_values(dl, total, t, &a) || !fetch_values(d, total, t, &b) || !fetch_values(du, total, t, &c) ||
      !fetch_values(X, total, t, &x))
    return CUSPARSE_STATUS_EXECUTION_FAILED;
  bool wrote_d = false, wrote_du = false;
  for (int k = 0; k < count; ++k) {
    std::vector<V> ak = deinterleave(a, m, count, k), bk = deinterleave(b, m, count, k),
                   ck = deinterleave(c, m, count, k), xk = deinterleave(x, m, count, k);
    ak[0] = V(0.0);
    ck[(size_t)m - 1] = V(0.0);
    std::vector<V> dnew = deinterleave(b, m, count, k), dunew = deinterleave(c, m, count, k);
    if (algo == 0) {
      thomas_solve(ak, bk, ck, xk.data(), dunew.data());
      interleave(c, dunew, m, count, k, m - 1);
      wrote_du = true;
    } else if (algo == 1) {
      lu_solve(ak, bk, ck, xk.data(), 1, (size_t)m, dnew.data(), dunew.data());
      interleave(b, dnew, m, count, k, m - 2);
      interleave(c, dunew, m, count, k, m - 2);
      wrote_d = wrote_du = true;
    } else {
      std::vector<std::vector<V>> band{ak, bk, ck};
      // band[k][i] is A(i, i + k - 1): the subdiagonal entry of row i is dl[i].
      qr_band_solve(band, 1, 1, xk.data(), dnew.data(), dunew.data());
      interleave(b, dnew, m, count, k, m - 2);
      interleave(c, dunew, m, count, k, m - 2);
      wrote_d = wrote_du = true;
    }
    interleave(x, xk, m, count, k, m);
  }
  const bool ok = store_values(X, x, t) && (!wrote_d || store_values(d, b, t)) && (!wrote_du || store_values(du, c, t));
  return ok ? CUSPARSE_STATUS_SUCCESS : CUSPARSE_STATUS_EXECUTION_FAILED;
}

template <class T>
cusparseStatus_t gpsv_interleaved(cusparseHandle_t h, int algo, int m, T* ds, T* dl, T* d, T* du, T* dw, T* X,
                                  int count) {
  using V = typename Legacy<T>::V;
  constexpr cudaDataType t = Legacy<T>::type;
  if (!known(h)) return CUSPARSE_STATUS_NOT_INITIALIZED;
  if (algo < 0 || algo > 1 || m < 0 || count < 0) return CUSPARSE_STATUS_INVALID_VALUE;
  if (m == 0 || count == 0) return CUSPARSE_STATUS_SUCCESS;
  const size_t total = (size_t)m * count;
  std::vector<V> s, a, b, c, w, x;
  if (!fetch_values(ds, total, t, &s) || !fetch_values(dl, total, t, &a) || !fetch_values(d, total, t, &b) ||
      !fetch_values(du, total, t, &c) || !fetch_values(dw, total, t, &w) || !fetch_values(X, total, t, &x))
    return CUSPARSE_STATUS_EXECUTION_FAILED;
  for (int k = 0; k < count; ++k) {
    std::vector<std::vector<V>> band{deinterleave(s, m, count, k), deinterleave(a, m, count, k),
                                     deinterleave(b, m, count, k), deinterleave(c, m, count, k),
                                     deinterleave(w, m, count, k)};
    // Entries outside the matrix are never read.
    for (int i = 0; i < std::min(m, 2); ++i) band[0][(size_t)i] = V(0.0);
    band[1][0] = V(0.0);
    band[3][(size_t)m - 1] = V(0.0);
    for (int i = std::max(0, m - 2); i < m; ++i) band[4][(size_t)i] = V(0.0);
    std::vector<V> xk = deinterleave(x, m, count, k);
    qr_band_solve(band, 2, 2, xk.data());
    interleave(x, xk, m, count, k, m);
  }
  return store_values(X, x, t) ? CUSPARSE_STATUS_SUCCESS : CUSPARSE_STATUS_EXECUTION_FAILED;
}

cusparseStatus_t size_out(cusparseHandle_t h, size_t* bytes, size_t value) {
  if (!known(h)) return CUSPARSE_STATUS_NOT_INITIALIZED;
  if (!bytes) return CUSPARSE_STATUS_INVALID_VALUE;
  *bytes = value;
  return CUSPARSE_STATUS_SUCCESS;
}

}  // namespace

// The workspace sizes are what NVIDIA's asks for where that is a simple
// formula (the interleaved solvers: m * batchCount elements, twice that for
// gpsv, a fixed 128 bytes for Thomas), and a token otherwise: the work runs on
// the host and reads no workspace.
#define VGPU_TRIDIAG(P, T)                                                                                          \
  VGPU_EXPORT cusparseStatus_t cusparse##P##gtsv2_bufferSizeExt(cusparseHandle_t h, int, int, const T*, const T*,  \
                                                                const T*, const T*, int, size_t* bytes) {          \
    return size_out(h, bytes, 512);                                                                                 \
  }                                                                                                                 \
  VGPU_EXPORT cusparseStatus_t cusparse##P##gtsv2(cusparseHandle_t h, int m, int n, const T* dl, const T* d,       \
                                                  const T* du, T* B, int ldb, void*) {                             \
    return gtsv2<T>(h, m, n, dl, d, du, B, ldb, true);                                                              \
  }                                                                                                                 \
  VGPU_EXPORT cusparseStatus_t cusparse##P##gtsv2_nopivot_bufferSizeExt(                                            \
      cusparseHandle_t h, int, int, const T*, const T*, const T*, const T*, int, size_t* bytes) {                  \
    return size_out(h, bytes, 128);                                                                                 \
  }                                                                                                                 \
  VGPU_EXPORT cusparseStatus_t cusparse##P##gtsv2_nopivot(cusparseHandle_t h, int m, int n, const T* dl,           \
                                                          const T* d, const T* du, T* B, int ldb, void*) {         \
    return gtsv2<T>(h, m, n, dl, d, du, B, ldb, false);                                                             \
  }                                                                                                                 \
  VGPU_EXPORT cusparseStatus_t cusparse##P##gtsv2StridedBatch_bufferSizeExt(                                        \
      cusparseHandle_t h, int m, const T*, const T*, const T*, const T*, int count, int stride, size_t* bytes) {   \
    if (!known(h)) return CUSPARSE_STATUS_NOT_INITIALIZED;                                                          \
    if (m < 3 || count < 1 || stride < m) return CUSPARSE_STATUS_INVALID_VALUE;                                     \
    return size_out(h, bytes, 128);                                                                                 \
  }                                                                                                                 \
  VGPU_EXPORT cusparseStatus_t cusparse##P##gtsv2StridedBatch(cusparseHandle_t h, int m, const T* dl, const T* d,  \
                                                              const T* du, T* x, int count, int stride, void*) {   \
    return gtsv2_strided<T>(h, m, dl, d, du, x, count, stride);                                                     \
  }                                                                                                                 \
  VGPU_EXPORT cusparseStatus_t cusparse##P##gtsvInterleavedBatch_bufferSizeExt(                                     \
      cusparseHandle_t h, int algo, int m, const T*, const T*, const T*, const T*, int count, size_t* bytes) {     \
    if (!known(h)) return CUSPARSE_STATUS_NOT_INITIALIZED;                                                          \
    if (algo < 0 || algo > 2 || m < 0 || count < 0) return CUSPARSE_STATUS_INVALID_VALUE;                           \
    return size_out(h, bytes, algo == 0 ? 128 : (size_t)m * count * sizeof(T));                                     \
  }                                                                                                                 \
  VGPU_EXPORT cusparseStatus_t cusparse##P##gtsvInterleavedBatch(cusparseHandle_t h, int algo, int m, T* dl, T* d, \
                                                                 T* du, T* x, int count, void*) {                  \
    return gtsv_interleaved<T>(h, algo, m, dl, d, du, x, count);                                                    \
  }                                                                                                                 \
  VGPU_EXPORT cusparseStatus_t cusparse##P##gpsvInterleavedBatch_bufferSizeExt(                                     \
      cusparseHandle_t h, int algo, int m, const T*, const T*, const T*, const T*, const T*, const T*, int count,  \
      size_t* bytes) {                                                                                              \
    if (!known(h)) return CUSPARSE_STATUS_NOT_INITIALIZED;                                                          \
    if (algo < 0 || algo > 1 || m < 0 || count < 0) return CUSPARSE_STATUS_INVALID_VALUE;                           \
    return size_out(h, bytes, 2 * (size_t)m * count * sizeof(T));                                                   \
  }                                                                                                                 \
  VGPU_EXPORT cusparseStatus_t cusparse##P##gpsvInterleavedBatch(cusparseHandle_t h, int algo, int m, T* ds, T* dl, \
                                                                 T* d, T* du, T* dw, T* x, int count, void*) {     \
    return gpsv_interleaved<T>(h, algo, m, ds, dl, d, du, dw, x, count);                                            \
  }
VGPU_TRIDIAG(S, float)
VGPU_TRIDIAG(D, double)
VGPU_TRIDIAG(C, cuComplex)
VGPU_TRIDIAG(Z, cuDoubleComplex)
#undef VGPU_TRIDIAG
