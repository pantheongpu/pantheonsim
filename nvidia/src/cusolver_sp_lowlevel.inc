// cusolverSp's low-level preview API (cusolverSp_LOWLEVEL_PREVIEW.h),
// included at the end of cusolver_sp.cpp: LU on the host, QR and Cholesky on
// the host and the device, each in analysis / buffer / (setup) / factor /
// zero-pivot / solve steps, for S, D, C and Z. The arithmetic is in
// std::complex<double> whatever the type; the info objects are type-free, as
// NVIDIA's are.
//
// What an RTX 3060's CUDA 13.0 showed, and this follows:
//
//   LU    no column reordering (Q is the identity); partial pivoting with a
//         threshold: the diagonal row is kept when |a_kk| >= threshold * the
//         column's largest, else the largest is taken (first row on a tie);
//         threshold 0 keeps the diagonal even when it is zero (and a
//         threshold outside [0, 1] is accepted); P A Q^T = L U with P[i] the
//         row of A at position i; L's unit diagonal not stored; a matrix type
//         other than GENERAL is INVALID_VALUE; BufferInfo before Analysis,
//         Factor before BufferInfo (QR and Cholesky too), and Solve before
//         Factor, INVALID_VALUE
//   QR    Householder, LAPACK's convention except that every column is
//         reflected (a column with nothing below the diagonal still flips:
//         R's diagonal is -sign of what it reflected); Setup takes A - mu I;
//         Solve overwrites b with Q^H b and gives the least-squares x;
//         Factor with b and x does both; host and device give the same
//   Chol  the lower triangle (the upper ignored), in the postorder of its
//         elimination tree (CSparse's cs_post order: roots and children in
//         increasing order); A(P, P) = L L^H; Diag returns L's diagonal in
//         that order; a pivot that is not positive takes sqrt(|d|) and the
//         factorization goes on; ZeroPivot names it 0-based on the host and
//         one higher on the device (as NVIDIA's device version answers)
//
// Not reproduced: NVIDIA's QR factors in place, so a second Factor without
// a new Setup refactors its own output; here each Factor starts again from
// what Setup gave it.

namespace {

/* ---- LU, host ---- */

struct LuLL {
  int n = -1, nnz = -1;
  bool analyzed = false, buffered = false, factored = false;
  std::vector<int> off, col;
  std::vector<int> p;                    // p[k]: the row of A pivoted at step k
  std::vector<std::map<int, cd>> l, u;   // by step: L(k, j < k) and U(k, j >= k)
};

int lu_threshold(LuLL* f, const std::vector<cd>& vals, double thr) {
  const int n = f->n;
  std::vector<std::map<int, cd>> rows((size_t)n);
  std::vector<std::set<int>> colrows((size_t)n);
  for (int i = 0; i < n; ++i)
    for (int k = f->off[(size_t)i]; k < f->off[(size_t)i + 1]; ++k) {
      rows[(size_t)i][f->col[(size_t)k]] += vals[(size_t)k];
      colrows[(size_t)f->col[(size_t)k]].insert(i);
    }
  std::vector<std::map<int, cd>> mult((size_t)n);  // by original row: its multipliers
  std::vector<char> active((size_t)n, 1);
  f->p.assign((size_t)n, -1);
  f->u.assign((size_t)n, {});
  for (int k = 0; k < n; ++k) {
    int best = -1;
    double bv = 0;
    for (int i : colrows[(size_t)k]) {
      if (!active[(size_t)i]) continue;
      const auto it = rows[(size_t)i].find(k);
      const double v = it == rows[(size_t)i].end() ? 0.0 : std::abs(it->second);
      if (v > bv) {
        bv = v;
        best = i;
      }
    }
    int r = best;
    if (active[(size_t)k]) {  // the diagonal row, when it is good enough
      const auto it = rows[(size_t)k].find(k);
      const double dv = it == rows[(size_t)k].end() ? 0.0 : std::abs(it->second);
      if (thr <= 0 || (dv != 0 && dv >= thr * bv)) r = k;  // threshold 0: no pivoting at all
      if (best < 0) r = k;
    }
    if (r < 0)  // nothing left in this column: the first unused row
      for (int i = 0; i < n && r < 0; ++i)
        if (active[(size_t)i]) r = i;
    f->p[(size_t)k] = r;
    active[(size_t)r] = 0;
    const auto pit = rows[(size_t)r].find(k);
    const cd piv = pit == rows[(size_t)r].end() ? cd(0.0) : pit->second;
    {  // a zero pivot still eliminates (into infinities), as NVIDIA's does: L and U keep their fill
      const std::vector<int> targets(colrows[(size_t)k].begin(), colrows[(size_t)k].end());
      for (int i : targets) {
        if (!active[(size_t)i]) continue;
        auto it = rows[(size_t)i].find(k);
        if (it == rows[(size_t)i].end()) continue;
        const cd l = it->second / piv;
        rows[(size_t)i].erase(it);
        mult[(size_t)i][k] = l;
        for (auto rc = rows[(size_t)r].upper_bound(k); rc != rows[(size_t)r].end(); ++rc) {
          auto [pos, inserted] = rows[(size_t)i].try_emplace(rc->first, cd(0.0));
          pos->second -= l * rc->second;
          if (inserted) colrows[(size_t)rc->first].insert(i);
        }
      }
    }
    f->u[(size_t)k] = std::map<int, cd>(rows[(size_t)r].lower_bound(k), rows[(size_t)r].end());
    if (f->u[(size_t)k].find(k) == f->u[(size_t)k].end()) f->u[(size_t)k][k] = cd(0.0);
  }
  f->l.assign((size_t)n, {});
  for (int k = 0; k < n; ++k) f->l[(size_t)k] = mult[(size_t)f->p[(size_t)k]];
  f->factored = true;
  return 0;
}

std::vector<cd> lu_ll_solve(const LuLL& f, const std::vector<cd>& b) {
  const int n = f.n;
  std::vector<cd> y((size_t)n);
  for (int k = 0; k < n; ++k) {
    cd s = b[(size_t)f.p[(size_t)k]];
    for (const auto& [j, v] : f.l[(size_t)k]) s -= v * y[(size_t)j];
    y[(size_t)k] = s;
  }
  for (int k = n - 1; k >= 0; --k) {
    cd s = y[(size_t)k];
    for (auto it = f.u[(size_t)k].upper_bound(k); it != f.u[(size_t)k].end(); ++it) s -= it->second * y[(size_t)it->first];
    y[(size_t)k] = s / f.u[(size_t)k].at(k);
  }
  return y;  // Q is the identity
}

/* ---- QR, host and device ---- */

// QrLL (the QR steps' state) is defined in cusolver_sp.cpp, beside the QrInfo that holds it.

void qr_ll_setup(QrLL* q, const std::vector<cd>& vals, cd mu) {
  q->voff.assign(1, 0);
  q->vcol.clear();
  q->vals.clear();
  const int d = std::min(q->m, q->n);
  for (int i = 0; i < q->m; ++i) {
    std::map<int, cd> row;
    for (int k = q->off[(size_t)i]; k < q->off[(size_t)i + 1]; ++k) row[q->col[(size_t)k]] += vals[(size_t)k];
    if (i < d && mu != cd(0.0)) row[i] -= mu;
    for (const auto& [c, x] : row) {
      q->vcol.push_back(c);
      q->vals.push_back(x);
    }
    q->voff.push_back((int)q->vcol.size());
  }
  q->setup = true;
  q->factored = false;
}

void qr_ll_factor(QrLL* q) {
  const int m = q->m, n = q->n;
  std::vector<std::map<int, cd>> cols((size_t)n);
  std::vector<std::set<int>> rowcols((size_t)m);  // the columns with an entry in each row
  for (int i = 0; i < m; ++i)
    for (int k = q->voff[(size_t)i]; k < q->voff[(size_t)i + 1]; ++k) {
      cols[(size_t)q->vcol[(size_t)k]][i] += q->vals[(size_t)k];
      rowcols[(size_t)i].insert(q->vcol[(size_t)k]);
    }
  q->v.assign((size_t)n, {});
  q->tau.assign((size_t)n, cd(0.0));
  q->r.assign((size_t)n, {});
  for (int k = 0; k < n; ++k) {
    auto& ck = cols[(size_t)k];
    const auto ait = ck.find(k);
    const cd alpha = ait == ck.end() ? cd(0.0) : ait->second;
    double xnorm = 0;
    for (auto it = ck.upper_bound(k); it != ck.end(); ++it) xnorm += std::norm(it->second);
    const double mag = std::sqrt(std::norm(alpha) + xnorm);
    for (auto it = ck.begin(); it != ck.end() && it->first < k; ++it) q->r[(size_t)it->first][k] = it->second;
    if (mag == 0.0) {  // nothing to reflect
      q->r[(size_t)k][k] = cd(0.0);
      continue;
    }
    // Every column is reflected, even one with nothing below the diagonal.
    const double beta = alpha.real() >= 0 ? -mag : mag;
    const cd t((beta - alpha.real()) / beta, -alpha.imag() / beta);
    const cd scal = 1.0 / (alpha - beta);
    std::map<int, cd> v;
    for (auto it = ck.upper_bound(k); it != ck.end(); ++it) v[it->first] = it->second * scal;
    q->tau[(size_t)k] = t;
    q->r[(size_t)k][k] = cd(beta);
    // H^H = I - conj(tau) w w^H, w = (1, v), on the later columns that meet w's rows.
    std::set<int> touched(rowcols[(size_t)k].begin(), rowcols[(size_t)k].end());
    for (const auto& [i, x] : v) touched.insert(rowcols[(size_t)i].begin(), rowcols[(size_t)i].end());
    for (int j : touched) {
      if (j <= k) continue;
      auto& cj_ = cols[(size_t)j];
      cd dot(0.0);
      if (const auto it = cj_.find(k); it != cj_.end()) dot += it->second;
      for (const auto& [i, x] : v)
        if (const auto it = cj_.find(i); it != cj_.end()) dot += std::conj(x) * it->second;
      if (dot == cd(0.0)) continue;
      dot *= std::conj(t);
      cj_[k] -= dot;
      rowcols[(size_t)k].insert(j);
      for (const auto& [i, x] : v) {
        cj_[i] -= x * dot;
        rowcols[(size_t)i].insert(j);
      }
    }
    q->v[(size_t)k] = std::move(v);
  }
  q->factored = true;
}

// b := Q^H b (all m entries); x := R^-1 (Q^H b)(0:n).
void qr_ll_solve(const QrLL& q, std::vector<cd>& b, std::vector<cd>* x) {
  for (int k = 0; k < q.n; ++k) {
    if (q.tau[(size_t)k] == cd(0.0)) continue;
    cd dot = b[(size_t)k];
    for (const auto& [i, v] : q.v[(size_t)k]) dot += std::conj(v) * b[(size_t)i];
    dot *= std::conj(q.tau[(size_t)k]);
    b[(size_t)k] -= dot;
    for (const auto& [i, v] : q.v[(size_t)k]) b[(size_t)i] -= v * dot;
  }
  x->assign((size_t)q.n, cd(0.0));
  for (int k = q.n - 1; k >= 0; --k) {
    cd s = b[(size_t)k];
    for (auto it = q.r[(size_t)k].upper_bound(k); it != q.r[(size_t)k].end(); ++it) s -= it->second * (*x)[(size_t)it->first];
    (*x)[(size_t)k] = s / q.r[(size_t)k].at(k);
  }
}

int qr_ll_zero_pivot(const QrLL& q, double tol) {
  for (int k = 0; k < q.n; ++k)
    if (!(std::abs(q.r[(size_t)k].at(k)) > tol)) return k;
  return -1;
}

/* ---- Cholesky, host and device ---- */

struct CholLL {
  int n = -1, nnz = -1;
  bool analyzed = false, buffered = false, factored = false;
  std::vector<int> off, col;
  std::vector<int> p;                   // A(P, P) = L L^H: p[k] is A's row k-th in the order
  std::vector<std::map<int, cd>> lcol;  // column k of L: rows >= k
  int failed = -1;                      // the first pivot that was not positive
};

// The elimination tree of the lower triangle's pattern in the natural order,
// and its postorder (CSparse's cs_etree and cs_post).
std::vector<int> etree_postorder(int n, const std::vector<int>& off, const std::vector<int>& col) {
  std::vector<std::vector<int>> below((size_t)n);  // row k: its columns j < k
  for (int i = 0; i < n; ++i)
    for (int k = off[(size_t)i]; k < off[(size_t)i + 1]; ++k) {
      const int j = col[(size_t)k];
      if (j < i) below[(size_t)i].push_back(j);
      else if (j > i) below[(size_t)j].push_back(i);  // the upper part's mirror shapes the same tree
    }
  std::vector<int> parent((size_t)n, -1), ancestor((size_t)n, -1);
  for (int k = 0; k < n; ++k)
    for (int i : below[(size_t)k])
      for (int j = i; j != -1 && j < k;) {
        const int next = ancestor[(size_t)j];
        ancestor[(size_t)j] = k;
        if (next == -1) parent[(size_t)j] = k;
        j = next;
      }
  std::vector<int> head((size_t)n, -1), nxt((size_t)n, -1), order;
  for (int j = n - 1; j >= 0; --j) {
    if (parent[(size_t)j] == -1) continue;
    nxt[(size_t)j] = head[(size_t)parent[(size_t)j]];
    head[(size_t)parent[(size_t)j]] = j;
  }
  for (int j = 0; j < n; ++j) {
    if (parent[(size_t)j] != -1) continue;
    std::vector<int> stack{j};
    while (!stack.empty()) {
      const int top = stack.back();
      const int child = head[(size_t)top];
      if (child == -1) {
        stack.pop_back();
        order.push_back(top);
      } else {
        head[(size_t)top] = nxt[(size_t)child];
        stack.push_back(child);
      }
    }
  }
  return order;
}

void chol_ll_factor(CholLL* c, const std::vector<cd>& vals) {
  const int n = c->n;
  std::vector<int> pinv((size_t)n);
  for (int k = 0; k < n; ++k) pinv[(size_t)c->p[(size_t)k]] = k;
  std::vector<std::map<int, cd>> cols((size_t)n);
  for (int i = 0; i < n; ++i)
    for (int k = c->off[(size_t)i]; k < c->off[(size_t)i + 1]; ++k) {
      const int j = c->col[(size_t)k];
      if (j > i) continue;  // the upper part is ignored
      int pi = pinv[(size_t)i], pj = pinv[(size_t)j];
      cd v = vals[(size_t)k];
      if (pi < pj) {
        std::swap(pi, pj);
        v = std::conj(v);
      }
      cols[(size_t)pj][pi] += v;
    }
  c->failed = -1;
  for (int k = 0; k < n; ++k) {
    auto& ck = cols[(size_t)k];
    const double d = ck.count(k) ? ck[k].real() : 0.0;
    if (!(d > 0.0) && c->failed < 0) c->failed = k;
    const double lkk = std::sqrt(std::fabs(d));
    ck[k] = lkk;
    for (auto it = ck.upper_bound(k); it != ck.end(); ++it) it->second /= lkk;
    for (auto it = ck.upper_bound(k); it != ck.end(); ++it)
      for (auto jt = ck.upper_bound(k); jt != std::next(it); ++jt)
        cols[(size_t)jt->first][it->first] -= it->second * std::conj(jt->second);
  }
  c->lcol = std::move(cols);
  c->factored = true;
}

std::vector<cd> chol_ll_solve(const CholLL& c, const std::vector<cd>& b) {
  const int n = c.n;
  std::vector<cd> y((size_t)n), x((size_t)n);
  for (int k = 0; k < n; ++k) y[(size_t)k] = b[(size_t)c.p[(size_t)k]];
  for (int k = 0; k < n; ++k) {
    y[(size_t)k] /= c.lcol[(size_t)k].at(k);
    for (auto it = c.lcol[(size_t)k].upper_bound(k); it != c.lcol[(size_t)k].end(); ++it)
      y[(size_t)it->first] -= it->second * y[(size_t)k];
  }
  for (int k = n - 1; k >= 0; --k) {
    cd s = y[(size_t)k];
    for (auto it = c.lcol[(size_t)k].upper_bound(k); it != c.lcol[(size_t)k].end(); ++it)
      s -= std::conj(it->second) * y[(size_t)it->first];
    y[(size_t)k] = s / c.lcol[(size_t)k].at(k);
  }
  for (int k = 0; k < n; ++k) x[(size_t)c.p[(size_t)k]] = y[(size_t)k];
  return x;
}

// The first pivot that failed or is at most tol, 0-based; -1 when none.
int chol_ll_zero_pivot(const CholLL& c, double tol) {
  for (int k = 0; k < c.n; ++k)
    if (k == c.failed || !(c.lcol[(size_t)k].at(k).real() > tol)) return k;
  return -1;
}

/* ---- the shared plumbing ---- */

template <class T> bool read_cd(const T* p, size_t n, bool device, std::vector<cd>* out) {
  std::vector<typename El<T>::V> v;
  if (!read<T>(p, n, device, &v)) return false;
  out->resize(n);
  for (size_t i = 0; i < n; ++i) (*out)[i] = cd(v[i]);
  return true;
}
template <class T> bool write_cd(T* p, const std::vector<cd>& v, bool device) {
  std::vector<typename El<T>::V> out(v.size());
  for (size_t i = 0; i < v.size(); ++i) {
    if constexpr (std::is_same_v<typename El<T>::V, double>) out[i] = v[i].real();
    else out[i] = v[i];
  }
  return write<T>(p, out, device);
}
template <class T> cd scalar_cd(T v) { return cd(El<T>::in(v)); }

// Analysis: the pattern, read and kept; only a GENERAL matrix.
template <class Info>
cusolverStatus_t ll_analysis(cusolverSpHandle_t h, int m, int n, int nnz, cusparseMatDescr_t d, const int* off,
                             const int* col, Info* info, bool device) {
  if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;
  if (!info || !d || m < 0 || n < 0 || nnz < 0) return CUSOLVER_STATUS_INVALID_VALUE;
  if (cusparseGetMatType(d) != CUSPARSE_MATRIX_TYPE_GENERAL) return CUSOLVER_STATUS_INVALID_VALUE;
  std::vector<int> o, c;
  if (!read_pattern(m, n, nnz, d, off, col, device, &o, &c)) return CUSOLVER_STATUS_INVALID_VALUE;
  info->off = std::move(o);
  info->col = std::move(c);
  info->nnz = nnz;
  info->analyzed = true;
  info->buffered = false;
  info->factored = false;
  return CUSOLVER_STATUS_SUCCESS;
}

template <class Info> Info* ll_info(const void* p) { return known(p) ? reinterpret_cast<Info*>(const_cast<void*>(p)) : nullptr; }

}  // namespace

/* ---- info objects ---- */

#define VGPU_LL_INFO(Name, Type, Struct)                                                   \
  VGPU_EXPORT cusolverStatus_t cusolverSpCreate##Name(Type* info) {                        \
    if (!info) return CUSOLVER_STATUS_INVALID_VALUE;                                       \
    *info = reinterpret_cast<Type>(track(new Struct()));                                   \
    return CUSOLVER_STATUS_SUCCESS;                                                        \
  }                                                                                        \
  VGPU_EXPORT cusolverStatus_t cusolverSpDestroy##Name(Type info) {                        \
    if (!known(info)) return CUSOLVER_STATUS_INVALID_VALUE;                                \
    untrack(info);                                                                         \
    delete reinterpret_cast<Struct*>(info);                                                \
    return CUSOLVER_STATUS_SUCCESS;                                                        \
  }
VGPU_LL_INFO(CsrluInfoHost, csrluInfoHost_t, LuLL)
VGPU_LL_INFO(CsrqrInfoHost, csrqrInfoHost_t, QrLL)
VGPU_LL_INFO(CsrcholInfoHost, csrcholInfoHost_t, CholLL)
VGPU_LL_INFO(CsrcholInfo, csrcholInfo_t, CholLL)
#undef VGPU_LL_INFO

/* ---- analysis ---- */

VGPU_EXPORT cusolverStatus_t cusolverSpXcsrluAnalysisHost(cusolverSpHandle_t h, int n, int nnz,
                                                          const cusparseMatDescr_t d, const int* off, const int* col,
                                                          csrluInfoHost_t info) {
  LuLL* f = ll_info<LuLL>(info);
  if (known(h) && !f) return CUSOLVER_STATUS_INVALID_VALUE;
  if (f) f->n = n;
  return ll_analysis(h, n, n, nnz, d, off, col, f, false);
}
// QR needs m >= n, as the batched form does.
static cusolverStatus_t qr_analysis(cusolverSpHandle_t h, int m, int n, int nnz, cusparseMatDescr_t d, const int* off,
                                    const int* col, QrLL* q, bool device) {
  if (known(h) && (!q || m < n)) return CUSOLVER_STATUS_INVALID_VALUE;
  const cusolverStatus_t st = ll_analysis(h, m, n, nnz, d, off, col, q, device);
  if (st == CUSOLVER_STATUS_SUCCESS) {
    q->m = m;
    q->n = n;
    q->setup = false;
  }
  return st;
}
VGPU_EXPORT cusolverStatus_t cusolverSpXcsrqrAnalysisHost(cusolverSpHandle_t h, int m, int n, int nnz,
                                                          const cusparseMatDescr_t d, const int* off, const int* col,
                                                          csrqrInfoHost_t info) {
  return qr_analysis(h, m, n, nnz, d, off, col, ll_info<QrLL>(info), false);
}
VGPU_EXPORT cusolverStatus_t cusolverSpXcsrqrAnalysis(cusolverSpHandle_t h, int m, int n, int nnz,
                                                      const cusparseMatDescr_t d, const int* off, const int* col,
                                                      csrqrInfo_t info) {
  QrInfo* qi = known(info) ? reinterpret_cast<QrInfo*>(info) : nullptr;
  const cusolverStatus_t st = qr_analysis(h, m, n, nnz, d, off, col, qi ? &qi->ll : nullptr, true);
  if (st == CUSOLVER_STATUS_SUCCESS) {
    qi->m = m;
    qi->n = n;
    qi->nnz = nnz;
  }
  return st;
}
static cusolverStatus_t chol_analysis(cusolverSpHandle_t h, int n, int nnz, cusparseMatDescr_t d, const int* off,
                                      const int* col, CholLL* c, bool device) {
  if (known(h) && !c) return CUSOLVER_STATUS_INVALID_VALUE;
  const cusolverStatus_t st = ll_analysis(h, n, n, nnz, d, off, col, c, device);
  if (st == CUSOLVER_STATUS_SUCCESS) {
    c->n = n;
    c->p = etree_postorder(n, c->off, c->col);
  }
  return st;
}
VGPU_EXPORT cusolverStatus_t cusolverSpXcsrcholAnalysisHost(cusolverSpHandle_t h, int n, int nnz,
                                                            const cusparseMatDescr_t d, const int* off, const int* col,
                                                            csrcholInfoHost_t info) {
  return chol_analysis(h, n, nnz, d, off, col, ll_info<CholLL>(info), false);
}
VGPU_EXPORT cusolverStatus_t cusolverSpXcsrcholAnalysis(cusolverSpHandle_t h, int n, int nnz,
                                                        const cusparseMatDescr_t d, const int* off, const int* col,
                                                        csrcholInfo_t info) {
  return chol_analysis(h, n, nnz, d, off, col, ll_info<CholLL>(info), true);
}

VGPU_EXPORT cusolverStatus_t cusolverSpXcsrluNnzHost(cusolverSpHandle_t h, int* nnzL, int* nnzU, csrluInfoHost_t info) {
  if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;
  LuLL* f = ll_info<LuLL>(info);
  if (!f || !f->factored || !nnzL || !nnzU) return CUSOLVER_STATUS_INVALID_VALUE;
  int l = 0, u = 0;
  for (int k = 0; k < f->n; ++k) {
    l += (int)f->l[(size_t)k].size();
    u += (int)f->u[(size_t)k].size();
  }
  *nnzL = l;
  *nnzU = u;
  return CUSOLVER_STATUS_SUCCESS;
}

/* ---- the typed steps ---- */

namespace {

size_t ll_bytes(int n, int nnz, size_t elem) { return ((size_t)std::max(n, 1) * 8 + (size_t)std::max(nnz, 1) * 4) * elem; }

template <class T>
cusolverStatus_t lu_buffer(cusolverSpHandle_t h, int n, int nnz, cusparseMatDescr_t d, LuLL* f, size_t* internal,
                           size_t* work) {
  if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;
  if (!f || !f->analyzed || f->n != n || f->nnz != nnz || !d) return CUSOLVER_STATUS_INVALID_VALUE;
  if (internal) *internal = ll_bytes(n, nnz, sizeof(T));
  if (work) *work = ll_bytes(n, 0, sizeof(T));
  f->buffered = true;
  return CUSOLVER_STATUS_SUCCESS;
}
template <class T>
cusolverStatus_t lu_factor_ll(cusolverSpHandle_t h, int n, int nnz, cusparseMatDescr_t d, const T* val, LuLL* f,
                              double thr) {
  if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;
  // BufferInfo comes between Analysis and Factor, as NVIDIA's requires; the
  // threshold is taken as given (NVIDIA's accepts 1.5 and -0.5 alike).
  if (!f || !f->analyzed || !f->buffered || f->n != n || f->nnz != nnz || !d) return CUSOLVER_STATUS_INVALID_VALUE;
  std::vector<cd> v;
  if (!read_cd<T>(val, (size_t)nnz, false, &v)) return CUSOLVER_STATUS_INVALID_VALUE;
  lu_threshold(f, v, thr);
  return CUSOLVER_STATUS_SUCCESS;
}
template <class T> cusolverStatus_t lu_solve_ll(cusolverSpHandle_t h, int n, const T* b, T* x, LuLL* f) {
  if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;
  if (!f || !f->factored || f->n != n) return CUSOLVER_STATUS_INVALID_VALUE;
  std::vector<cd> bv;
  if (!read_cd<T>(b, (size_t)n, false, &bv)) return CUSOLVER_STATUS_INVALID_VALUE;
  return write_cd<T>(x, lu_ll_solve(*f, bv), false) ? CUSOLVER_STATUS_SUCCESS : CUSOLVER_STATUS_INVALID_VALUE;
}
template <class T>
cusolverStatus_t lu_extract(cusolverSpHandle_t h, int* P, int* Q, cusparseMatDescr_t dl, T* lval, int* loff, int* lcol,
                            cusparseMatDescr_t du, T* uval, int* uoff, int* ucol, LuLL* f) {
  if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;
  if (!f || !f->factored || !P || !Q || !dl || !du || !loff || !uoff) return CUSOLVER_STATUS_INVALID_VALUE;
  const int bl = base_of(dl), bu = base_of(du), n = f->n;
  std::vector<cd> lv, uv;
  int nl = 0, nu = 0;
  loff[0] = bl;
  uoff[0] = bu;
  for (int k = 0; k < n; ++k) {
    P[k] = f->p[(size_t)k];
    Q[k] = k;
    for (const auto& [j, v] : f->l[(size_t)k]) {
      lcol[nl++] = j + bl;
      lv.push_back(v);
    }
    for (const auto& [j, v] : f->u[(size_t)k]) {
      ucol[nu++] = j + bu;
      uv.push_back(v);
    }
    loff[k + 1] = nl + bl;
    uoff[k + 1] = nu + bu;
  }
  return write_cd<T>(lval, lv, false) && write_cd<T>(uval, uv, false) ? CUSOLVER_STATUS_SUCCESS
                                                                     : CUSOLVER_STATUS_INVALID_VALUE;
}
int lu_zero_pivot(const LuLL& f, double tol) {
  for (int k = 0; k < f.n; ++k)
    if (!(std::abs(f.u[(size_t)k].at(k)) > tol)) return k;
  return -1;
}

template <class T>
cusolverStatus_t qr_buffer_ll(cusolverSpHandle_t h, int m, int n, int nnz, cusparseMatDescr_t d, QrLL* q,
                              size_t* internal, size_t* work) {
  if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;
  if (!q || !q->analyzed || q->m != m || q->n != n || q->nnz != nnz || !d) return CUSOLVER_STATUS_INVALID_VALUE;
  if (internal) *internal = ll_bytes(m + n, nnz, sizeof(T));
  if (work) *work = ll_bytes(m, 0, sizeof(T));
  q->buffered = true;
  return CUSOLVER_STATUS_SUCCESS;
}
template <class T>
cusolverStatus_t qr_setup_ll(cusolverSpHandle_t h, int m, int n, int nnz, cusparseMatDescr_t d, const T* val, T mu,
                             QrLL* q, bool device) {
  if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;
  if (!q || !q->analyzed || q->m != m || q->n != n || q->nnz != nnz || !d) return CUSOLVER_STATUS_INVALID_VALUE;
  std::vector<cd> v;
  if (!read_cd<T>(val, (size_t)nnz, device, &v)) return CUSOLVER_STATUS_INVALID_VALUE;
  qr_ll_setup(q, v, scalar_cd(mu));
  return CUSOLVER_STATUS_SUCCESS;
}
template <class T> cusolverStatus_t qr_solve_ll(QrLL* q, int m, int n, T* b, T* x, bool device) {
  if (!q->factored || q->m != m || q->n != n || !b || !x) return CUSOLVER_STATUS_INVALID_VALUE;
  std::vector<cd> bv, xv;
  if (!read_cd<T>(b, (size_t)m, device, &bv)) return CUSOLVER_STATUS_INVALID_VALUE;
  qr_ll_solve(*q, bv, &xv);
  return write_cd<T>(b, bv, device) && write_cd<T>(x, xv, device) ? CUSOLVER_STATUS_SUCCESS
                                                                 : CUSOLVER_STATUS_EXECUTION_FAILED;
}
template <class T>
cusolverStatus_t qr_factor_ll(cusolverSpHandle_t h, int m, int n, int nnz, T* b, T* x, QrLL* q, bool device) {
  if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;
  if (!q || !q->setup || !q->buffered || q->m != m || q->n != n || q->nnz != nnz) return CUSOLVER_STATUS_INVALID_VALUE;
  qr_ll_factor(q);
  if (b && x) return qr_solve_ll<T>(q, m, n, b, x, device);
  return CUSOLVER_STATUS_SUCCESS;
}

template <class T>
cusolverStatus_t chol_buffer_ll(cusolverSpHandle_t h, int n, int nnz, cusparseMatDescr_t d, CholLL* c, size_t* internal,
                                size_t* work) {
  if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;
  if (!c || !c->analyzed || c->n != n || c->nnz != nnz || !d) return CUSOLVER_STATUS_INVALID_VALUE;
  if (internal) *internal = ll_bytes(n, nnz, sizeof(T));
  if (work) *work = ll_bytes(n, 0, sizeof(T));
  c->buffered = true;
  return CUSOLVER_STATUS_SUCCESS;
}
template <class T>
cusolverStatus_t chol_factor_ll(cusolverSpHandle_t h, int n, int nnz, cusparseMatDescr_t d, const T* val, CholLL* c,
                                bool device) {
  if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;
  if (!c || !c->analyzed || !c->buffered || c->n != n || c->nnz != nnz || !d) return CUSOLVER_STATUS_INVALID_VALUE;
  std::vector<cd> v;
  if (!read_cd<T>(val, (size_t)nnz, device, &v)) return CUSOLVER_STATUS_INVALID_VALUE;
  chol_ll_factor(c, v);
  return CUSOLVER_STATUS_SUCCESS;
}
template <class T> cusolverStatus_t chol_solve_ll(cusolverSpHandle_t h, int n, const T* b, T* x, CholLL* c, bool device) {
  if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;
  if (!c || !c->factored || c->n != n) return CUSOLVER_STATUS_INVALID_VALUE;
  std::vector<cd> bv;
  if (!read_cd<T>(b, (size_t)n, device, &bv)) return CUSOLVER_STATUS_INVALID_VALUE;
  return write_cd<T>(x, chol_ll_solve(*c, bv), device) ? CUSOLVER_STATUS_SUCCESS : CUSOLVER_STATUS_EXECUTION_FAILED;
}

template <class S> cusolverStatus_t put_position(int* position, int v) {
  if (!position) return CUSOLVER_STATUS_INVALID_VALUE;
  *position = v;
  return CUSOLVER_STATUS_SUCCESS;
}

}  // namespace

#define VGPU_SP_LL(P, T)                                                                                            \
  VGPU_EXPORT cusolverStatus_t cusolverSp##P##csrluBufferInfoHost(                                                  \
      cusolverSpHandle_t h, int n, int nnz, const cusparseMatDescr_t d, const T*, const int*, const int*,           \
      csrluInfoHost_t info, size_t* internal, size_t* work) {                                                       \
    return lu_buffer<T>(h, n, nnz, d, ll_info<LuLL>(info), internal, work);                                         \
  }                                                                                                                 \
  VGPU_EXPORT cusolverStatus_t cusolverSp##P##csrluFactorHost(cusolverSpHandle_t h, int n, int nnz,                 \
                                                              const cusparseMatDescr_t d, const T* val, const int*, \
                                                              const int*, csrluInfoHost_t info, El<T>::R thr,       \
                                                              void*) {                                              \
    return lu_factor_ll<T>(h, n, nnz, d, val, ll_info<LuLL>(info), thr);                                            \
  }                                                                                                                 \
  VGPU_EXPORT cusolverStatus_t cusolverSp##P##csrluZeroPivotHost(cusolverSpHandle_t h, csrluInfoHost_t info,        \
                                                                 El<T>::R tol, int* position) {                     \
    if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;                                                          \
    LuLL* f = ll_info<LuLL>(info);                                                                                  \
    if (!f || !f->factored) return CUSOLVER_STATUS_INVALID_VALUE;                                                   \
    return put_position<T>(position, lu_zero_pivot(*f, tol));                                                      \
  }                                                                                                                 \
  VGPU_EXPORT cusolverStatus_t cusolverSp##P##csrluSolveHost(cusolverSpHandle_t h, int n, const T* b, T* x,         \
                                                             csrluInfoHost_t info, void*) {                         \
    return lu_solve_ll<T>(h, n, b, x, ll_info<LuLL>(info));                                                         \
  }                                                                                                                 \
  VGPU_EXPORT cusolverStatus_t cusolverSp##P##csrluExtractHost(                                                     \
      cusolverSpHandle_t h, int* P_, int* Q_, const cusparseMatDescr_t dl, T* lv, int* lo, int* lc,                 \
      const cusparseMatDescr_t du, T* uv, int* uo, int* uc, csrluInfoHost_t info, void*) {                          \
    return lu_extract<T>(h, P_, Q_, dl, lv, lo, lc, du, uv, uo, uc, ll_info<LuLL>(info));                           \
  }                                                                                                                 \
  /* QR, host */                                                                                                    \
  VGPU_EXPORT cusolverStatus_t cusolverSp##P##csrqrBufferInfoHost(                                                  \
      cusolverSpHandle_t h, int m, int n, int nnz, const cusparseMatDescr_t d, const T*, const int*, const int*,    \
      csrqrInfoHost_t info, size_t* internal, size_t* work) {                                                       \
    return qr_buffer_ll<T>(h, m, n, nnz, d, ll_info<QrLL>(info), internal, work);                                   \
  }                                                                                                                 \
  VGPU_EXPORT cusolverStatus_t cusolverSp##P##csrqrSetupHost(cusolverSpHandle_t h, int m, int n, int nnz,           \
                                                             const cusparseMatDescr_t d, const T* val, const int*,  \
                                                             const int*, T mu, csrqrInfoHost_t info) {              \
    return qr_setup_ll<T>(h, m, n, nnz, d, val, mu, ll_info<QrLL>(info), false);                                    \
  }                                                                                                                 \
  VGPU_EXPORT cusolverStatus_t cusolverSp##P##csrqrFactorHost(cusolverSpHandle_t h, int m, int n, int nnz, T* b,    \
                                                              T* x, csrqrInfoHost_t info, void*) {                  \
    return qr_factor_ll<T>(h, m, n, nnz, b, x, ll_info<QrLL>(info), false);                                         \
  }                                                                                                                 \
  VGPU_EXPORT cusolverStatus_t cusolverSp##P##csrqrZeroPivotHost(cusolverSpHandle_t h, csrqrInfoHost_t info,        \
                                                                 El<T>::R tol, int* position) {                     \
    if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;                                                          \
    QrLL* q = ll_info<QrLL>(info);                                                                                  \
    if (!q || !q->factored) return CUSOLVER_STATUS_INVALID_VALUE;                                                   \
    return put_position<T>(position, qr_ll_zero_pivot(*q, tol));                                                   \
  }                                                                                                                 \
  VGPU_EXPORT cusolverStatus_t cusolverSp##P##csrqrSolveHost(cusolverSpHandle_t h, int m, int n, T* b, T* x,        \
                                                             csrqrInfoHost_t info, void*) {                         \
    if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;                                                          \
    QrLL* q = ll_info<QrLL>(info);                                                                                  \
    return q ? qr_solve_ll<T>(q, m, n, b, x, false) : CUSOLVER_STATUS_INVALID_VALUE;                                \
  }                                                                                                                 \
  /* QR, device (the batched form's info) */                                                                        \
  VGPU_EXPORT cusolverStatus_t cusolverSp##P##csrqrBufferInfo(                                                      \
      cusolverSpHandle_t h, int m, int n, int nnz, const cusparseMatDescr_t d, const T*, const int*, const int*,    \
      csrqrInfo_t info, size_t* internal, size_t* work) {                                                           \
    QrInfo* qi = known(info) ? reinterpret_cast<QrInfo*>(info) : nullptr;                                           \
    return qr_buffer_ll<T>(h, m, n, nnz, d, qi ? &qi->ll : nullptr, internal, work);                                \
  }                                                                                                                 \
  VGPU_EXPORT cusolverStatus_t cusolverSp##P##csrqrSetup(cusolverSpHandle_t h, int m, int n, int nnz,               \
                                                         const cusparseMatDescr_t d, const T* val, const int*,      \
                                                         const int*, T mu, csrqrInfo_t info) {                      \
    QrInfo* qi = known(info) ? reinterpret_cast<QrInfo*>(info) : nullptr;                                           \
    return qr_setup_ll<T>(h, m, n, nnz, d, val, mu, qi ? &qi->ll : nullptr, true);                                  \
  }                                                                                                                 \
  VGPU_EXPORT cusolverStatus_t cusolverSp##P##csrqrFactor(cusolverSpHandle_t h, int m, int n, int nnz, T* b, T* x,  \
                                                          csrqrInfo_t info, void*) {                                \
    QrInfo* qi = known(info) ? reinterpret_cast<QrInfo*>(info) : nullptr;                                           \
    return qr_factor_ll<T>(h, m, n, nnz, b, x, qi ? &qi->ll : nullptr, true);                                       \
  }                                                                                                                 \
  VGPU_EXPORT cusolverStatus_t cusolverSp##P##csrqrZeroPivot(cusolverSpHandle_t h, csrqrInfo_t info, El<T>::R tol,  \
                                                             int* position) {                                       \
    if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;                                                          \
    QrInfo* qi = known(info) ? reinterpret_cast<QrInfo*>(info) : nullptr;                                           \
    if (!qi || !qi->ll.factored) return CUSOLVER_STATUS_INVALID_VALUE;                                              \
    return put_position<T>(position, qr_ll_zero_pivot(qi->ll, tol));                                               \
  }                                                                                                                 \
  VGPU_EXPORT cusolverStatus_t cusolverSp##P##csrqrSolve(cusolverSpHandle_t h, int m, int n, T* b, T* x,            \
                                                         csrqrInfo_t info, void*) {                                 \
    if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;                                                          \
    QrInfo* qi = known(info) ? reinterpret_cast<QrInfo*>(info) : nullptr;                                           \
    return qi ? qr_solve_ll<T>(&qi->ll, m, n, b, x, true) : CUSOLVER_STATUS_INVALID_VALUE;                          \
  }                                                                                                                 \
  /* Cholesky, host and device */                                                                                   \
  VGPU_EXPORT cusolverStatus_t cusolverSp##P##csrcholBufferInfoHost(                                                \
      cusolverSpHandle_t h, int n, int nnz, const cusparseMatDescr_t d, const T*, const int*, const int*,           \
      csrcholInfoHost_t info, size_t* internal, size_t* work) {                                                     \
    return chol_buffer_ll<T>(h, n, nnz, d, ll_info<CholLL>(info), internal, work);                                  \
  }                                                                                                                 \
  VGPU_EXPORT cusolverStatus_t cusolverSp##P##csrcholBufferInfo(                                                    \
      cusolverSpHandle_t h, int n, int nnz, const cusparseMatDescr_t d, const T*, const int*, const int*,           \
      csrcholInfo_t info, size_t* internal, size_t* work) {                                                         \
    return chol_buffer_ll<T>(h, n, nnz, d, ll_info<CholLL>(info), internal, work);                                  \
  }                                                                                                                 \
  VGPU_EXPORT cusolverStatus_t cusolverSp##P##csrcholFactorHost(cusolverSpHandle_t h, int n, int nnz,               \
                                                                const cusparseMatDescr_t d, const T* val,           \
                                                                const int*, const int*, csrcholInfoHost_t info,     \
                                                                void*) {                                            \
    return chol_factor_ll<T>(h, n, nnz, d, val, ll_info<CholLL>(info), false);                                      \
  }                                                                                                                 \
  VGPU_EXPORT cusolverStatus_t cusolverSp##P##csrcholFactor(cusolverSpHandle_t h, int n, int nnz,                   \
                                                            const cusparseMatDescr_t d, const T* val, const int*,   \
                                                            const int*, csrcholInfo_t info, void*) {                \
    return chol_factor_ll<T>(h, n, nnz, d, val, ll_info<CholLL>(info), true);                                       \
  }                                                                                                                 \
  VGPU_EXPORT cusolverStatus_t cusolverSp##P##csrcholZeroPivotHost(cusolverSpHandle_t h, csrcholInfoHost_t info,    \
                                                                   El<T>::R tol, int* position) {                   \
    if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;                                                          \
    CholLL* c = ll_info<CholLL>(info);                                                                              \
    if (!c || !c->factored) return CUSOLVER_STATUS_INVALID_VALUE;                                                   \
    return put_position<T>(position, chol_ll_zero_pivot(*c, tol));                                                 \
  }                                                                                                                 \
  /* NVIDIA's device version names the pivot one higher than its host version. */                                   \
  VGPU_EXPORT cusolverStatus_t cusolverSp##P##csrcholZeroPivot(cusolverSpHandle_t h, csrcholInfo_t info,            \
                                                               El<T>::R tol, int* position) {                       \
    if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;                                                          \
    CholLL* c = ll_info<CholLL>(info);                                                                              \
    if (!c || !c->factored) return CUSOLVER_STATUS_INVALID_VALUE;                                                   \
    const int k = chol_ll_zero_pivot(*c, tol);                                                                      \
    return put_position<T>(position, k < 0 ? -1 : k + 1);                                                          \
  }                                                                                                                 \
  VGPU_EXPORT cusolverStatus_t cusolverSp##P##csrcholSolveHost(cusolverSpHandle_t h, int n, const T* b, T* x,       \
                                                               csrcholInfoHost_t info, void*) {                     \
    return chol_solve_ll<T>(h, n, b, x, ll_info<CholLL>(info), false);                                              \
  }                                                                                                                 \
  VGPU_EXPORT cusolverStatus_t cusolverSp##P##csrcholSolve(cusolverSpHandle_t h, int n, const T* b, T* x,           \
                                                           csrcholInfo_t info, void*) {                             \
    return chol_solve_ll<T>(h, n, b, x, ll_info<CholLL>(info), true);                                               \
  }                                                                                                                 \
  /* L's diagonal, in the factorization's order, on the device. */                                                  \
  VGPU_EXPORT cusolverStatus_t cusolverSp##P##csrcholDiag(cusolverSpHandle_t h, csrcholInfo_t info,                 \
                                                          El<T>::R* diag) {                                         \
    if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;                                                          \
    CholLL* c = ll_info<CholLL>(info);                                                                              \
    if (!c || !c->factored || !diag) return CUSOLVER_STATUS_INVALID_VALUE;                                          \
    std::vector<El<T>::R> d((size_t)c->n);                                                                          \
    for (int k = 0; k < c->n; ++k) d[(size_t)k] = (El<T>::R)c->lcol[(size_t)k].at(k).real();                       \
    return d.empty() || cudaMemcpy(diag, d.data(), d.size() * sizeof(d[0]), cudaMemcpyHostToDevice) == cudaSuccess \
               ? CUSOLVER_STATUS_SUCCESS                                                                            \
               : CUSOLVER_STATUS_EXECUTION_FAILED;                                                                  \
  }
VGPU_SP_LL(S, float)
VGPU_SP_LL(D, double)
VGPU_SP_LL(C, cuComplex)
VGPU_SP_LL(Z, cuDoubleComplex)
#undef VGPU_SP_LL
