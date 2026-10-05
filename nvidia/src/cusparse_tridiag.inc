// cuSPARSE's tridiagonal and pentadiagonal solvers, included by cusparse_api.cpp.
//
//   gtsv2                 LU with partial pivoting, B overwritten by X
//   gtsv2_nopivot         parallel cyclic reduction (PCR), no pivoting
//   gtsv2StridedBatch     the same, one right-hand side per system
//   gtsvInterleavedBatch  algo 0 Thomas, 1 LU with partial pivoting, 2 QR
//   gpsvInterleavedBatch  QR of the pentadiagonal system
//
// What an RTX 3060 showed with cuSPARSE 13.0 (CUDA 13.0), which is what this
// reproduces -- nvidia/tests/e2e/sparse_tridiag_paths.cu checks it:
//
//   * gtsv2, gtsv2_nopivot and gtsv2StridedBatch refuse m < 3, n < 0 and
//     ldb < max(1, m) with INVALID_VALUE; gtsv2StridedBatch also refuses
//     batchCount < 1 and batchStride < m. Their _bufferSizeExt checks nothing
//     but StridedBatch's (m, batchCount, batchStride as above). n = 0 is a
//     successful no-op.
//   * The interleaved solvers take any m and batchCount >= 0 (zero of either
//     is a no-op; their _bufferSizeExt checks neither), and gtsv's refuses an
//     algo outside 0..2 with INVALID_VALUE.
//   * The first element of each lower diagonal (and of gpsv's ds, the first
//     two) and the last of each upper one are never read.
//   * gtsv2 solves a system whose leading entries are zero, as LU with partial
//     pivoting does, and leaves dl, d and du as they were; a singular system
//     comes back NaN throughout.
//   * gtsv2_nopivot is PCR up to 2048 unknowns, gtsv2StridedBatch up to 512:
//     a zero pivot spoils one half of the unknowns (with d[0] = 0, every odd
//     one is NaN and every even one is right) and leaves the other half
//     alone. Past that, cyclic reduction (CR) steps first halve the system
//     until PCR takes it, and the same zero spoils every unknown.
//   * The interleaved solvers overwrite their inputs. Thomas (gtsv algo 0)
//     leaves the modified upper diagonal c'_i = du_i / (d_i - dl_i c'_{i-1})
//     in du[0..m-2]; LU (algo 1) leaves 1/u_ii in d[i] and u_{i,i+1} in du[i]
//     for i < m - 2; QR (gtsv algo 2, and gpsv) is Householder's (see
//     qr_band_solve) and leaves R's diagonal and first superdiagonal in d and
//     du for i < m - 2 -- and for gpsv its diagonal and two superdiagonals in
//     d, du and dw for i < m - 3. LU and both QRs write nothing back for
//     m <= 4. dl (and ds) are always left as they were. The same in every
//     type.
//   * gpsvInterleavedBatch's algo is not checked: anything but 0 does
//     nothing and reports success, which this refuses (NOT_SUPPORTED). A
//     negative m is INVALID_VALUE, a negative batchCount EXECUTION_FAILED
//     (gtsvInterleavedBatch: INVALID_VALUE for both).
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
// zero d[0] leaves the even unknowns right up to m = 2048 for gtsv2_nopivot
// and spoils them all from 2049; for gtsv2StridedBatch the edge is 512/513).
constexpr size_t kPcrLimitNopivot = 2048, kPcrLimitStrided = 512;

// CR steps (keeping the odd equations) until PCR takes the system.
template <class V> void cr_pcr_solve(std::vector<V> a, std::vector<V> b, std::vector<V> c, V* x, size_t limit) {
  const size_t m = b.size();
  if (m <= limit) return pcr_solve(std::move(a), std::move(b), std::move(c), x);
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
  cr_pcr_solve(std::move(a2), std::move(b2), std::move(c2), d2.data(), limit);
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

// QR of a banded system with `lower` subdiagonals and `upper` superdiagonals
// by Householder reflections, one per column, made the way NVIDIA's are
// (measured on R's entries that the interleaved solvers leave behind): beta =
// -sgn(alpha) * norm with sgn(alpha) = alpha / |alpha| (+1 at zero), so a real
// pivot's R entry has the opposite sign, and every column is reflected, even
// one with nothing below its diagonal (that entry's sign flips). band[k][i] is
// A(i, i + k - lower). With `out`, out[k][j] gets R(j, j + k) for k <= lower
// and j < m - lower - 1, the part of R NVIDIA's leaves in d, du (and dw).
template <class V>
void qr_band_solve(const std::vector<std::vector<V>>& band, int lower, int upper, V* x,
                   std::vector<V>* out = nullptr) {
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
  std::vector<V> v((size_t)lower + 1);
  for (int j = 0; j < m; ++j) {
    const int last_row = std::min(m - 1, j + lower), last_col = std::min(m - 1, j + upper + lower);
    const V alpha = at(j, j);
    double norm2 = std::norm(cd(re(alpha), im(alpha)));
    for (int i = j + 1; i <= last_row; ++i) norm2 += std::norm(cd(re(at(i, j)), im(at(i, j))));
    if (norm2 > 0.0) {
      const double aa = std::abs(cd(re(alpha), im(alpha)));
      const V sgn = aa == 0.0 ? V(1.0) : alpha / V(aa);
      const V beta = -(sgn * V(std::sqrt(norm2)));
      const V tau = (beta - alpha) / beta;  // real, so H = I - tau v v^H is a reflection
      const V scale = V(1.0) / (alpha - beta);
      v[0] = V(1.0);
      for (int i = j + 1; i <= last_row; ++i) v[(size_t)(i - j)] = at(i, j) * scale;
      auto reflect = [&](auto get) {
        V w(0.0);
        for (int i = j; i <= last_row; ++i) w += conj_of(v[(size_t)(i - j)]) * get(i);
        w *= tau;
        for (int i = j; i <= last_row; ++i) get(i) -= v[(size_t)(i - j)] * w;
      };
      for (int col = j + 1; col <= last_col; ++col) reflect([&](int i) -> V& { return at(i, col); });
      reflect([&](int i) -> V& { return x[i]; });
      at(j, j) = beta;
      for (int i = j + 1; i <= last_row; ++i) at(i, j) = V(0.0);
    }
    if (out && j < m - lower - 1)
      for (int k = 0; k <= lower; ++k) out[(size_t)k][(size_t)j] = at(j, j + k);
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
    for (int j = 0; j < n; ++j) cr_pcr_solve(a, b, c, x.data() + (size_t)j * ldb, kPcrLimitNopivot);
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
                 std::vector<V>(c.begin() + o, c.begin() + o + m), x.data() + o, kPcrLimitStrided);
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
      if (m >= 5) {  // measured: smaller systems leave d and du alone
        interleave(b, dnew, m, count, k, m - 2);
        interleave(c, dunew, m, count, k, m - 2);
        wrote_d = wrote_du = true;
      }
    } else {
      // band[k][i] is A(i, i + k - 1): the subdiagonal entry of row i is dl[i].
      std::vector<V> outs[2] = {dnew, dunew};
      qr_band_solve(std::vector<std::vector<V>>{ak, bk, ck}, 1, 1, xk.data(), outs);
      if (m >= 5) {
        interleave(b, outs[0], m, count, k, m - 2);
        interleave(c, outs[1], m, count, k, m - 2);
        wrote_d = wrote_du = true;
      }
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
  if (m < 0) return CUSPARSE_STATUS_INVALID_VALUE;
  if (count < 0) return CUSPARSE_STATUS_EXECUTION_FAILED;
  // NVIDIA documents algo 0 only. Its 13.0 takes any other, does nothing and
  // reports success (measured with 1); that is refused rather than copied.
  if (algo != 0) return CUSPARSE_STATUS_NOT_SUPPORTED;
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
    std::vector<V> outs[3] = {band[2], band[3], band[4]};
    qr_band_solve(band, 2, 2, xk.data(), outs);
    if (m >= 5) {
      interleave(b, outs[0], m, count, k, m - 3);
      interleave(c, outs[1], m, count, k, m - 3);
      interleave(w, outs[2], m, count, k, m - 3);
    }
    interleave(x, xk, m, count, k, m);
  }
  const bool ok = store_values(X, x, t) && store_values(d, b, t) && store_values(du, c, t) && store_values(dw, w, t);
  return ok ? CUSPARSE_STATUS_SUCCESS : CUSPARSE_STATUS_EXECUTION_FAILED;
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
    if (algo < 0 || algo > 2) return CUSPARSE_STATUS_INVALID_VALUE;                                                 \
    return size_out(h, bytes, algo == 0 ? 128 : (size_t)std::max(m, 0) * std::max(count, 0) * sizeof(T));          \
  }                                                                                                                 \
  VGPU_EXPORT cusparseStatus_t cusparse##P##gtsvInterleavedBatch(cusparseHandle_t h, int algo, int m, T* dl, T* d, \
                                                                 T* du, T* x, int count, void*) {                  \
    return gtsv_interleaved<T>(h, algo, m, dl, d, du, x, count);                                                    \
  }                                                                                                                 \
  VGPU_EXPORT cusparseStatus_t cusparse##P##gpsvInterleavedBatch_bufferSizeExt(                                     \
      cusparseHandle_t h, int algo, int m, const T*, const T*, const T*, const T*, const T*, const T*, int count,  \
      size_t* bytes) {                                                                                              \
    if (!known(h)) return CUSPARSE_STATUS_NOT_INITIALIZED;                                                          \
    (void)algo;                                                                                                     \
    return size_out(h, bytes, 2 * (size_t)std::max(m, 0) * std::max(count, 0) * sizeof(T));                         \
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
