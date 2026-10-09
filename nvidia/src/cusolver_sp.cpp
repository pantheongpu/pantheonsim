// The sparse half of VirtualGPU's cuSOLVER (cusolverSp), built into
// libcusolver.so.12 beside cusolver_api.cpp: linear solves by sparse LU,
// QR and Cholesky (host and device entry points alike), least squares,
// shift-inverse eigenvalues, the reorderings and permutations, and batched QR.
// Matrices arrive as CSR with a cuSPARSE matrix descriptor, which is read
// through cuSPARSE's own getters, as NVIDIA's libcusolver reads it (it links
// libcusparse too).
//
// Everything runs on the host in double or std::complex<double>:
//
//   LU        right-looking Gaussian elimination on sparse rows, partial
//             pivoting by rows (the largest magnitude in the column, first
//             row on a tie), columns in the order `reorder` asks for;
//   QR        Givens rotations row by row into a sparse R (George and
//             Heath), Q^H b carried along, columns ordered the same way;
//   Cholesky  right-looking on the lower triangle (the upper part is
//             ignored, as NVIDIA documents), A = L L^H.
//
// reorder 0 keeps the natural order; 1 is csrsymrcm's reverse Cuthill-McKee
// and 2 csrsymamd's approximate minimum degree, both NVIDIA's exact
// permutations; 3 (METIS) is csrmetisnd's (METIS 5.1.0's nested dissection). They change the fill
// and the rounding, not the answer. What an RTX 3060's cuSOLVER (CUDA 13.0) showed,
// and this follows (nvidia/tests/e2e/solver_sparse_paths.cu):
//
//   - `tol` is absolute: an LU or QR pivot of magnitude <= tol, or a
//     Cholesky diagonal L(j,j) <= tol or one that is not positive, makes
//     `singularity` the first such index j (in the order the factorization
//     took the columns, which is the natural order for reorder 0); otherwise
//     it is -1. x is computed regardless, whatever it then holds;
//   - a matrix type other than GENERAL is MATRIX_TYPE_NOT_SUPPORTED, and a
//     reorder outside 0..3 INVALID_VALUE;
//   - csrlsqvqrHost returns the minimum-norm least-squares solution, its
//     residual norm, and in p the columns' order after each one found
//     dependent on those before it (|R(k,k)| <= tol) changed places with the
//     last column not yet set aside -- [0 1 2 3 4 7 6 5] when column 5 of 8
//     depends on the first five;
//   - csreigvsi starts from x0 / ||x0||; each step takes mu = x^H A x, stops
//     when ||A x - mu x|| <= tol, and otherwise replaces x by (A - mu0 I)^{-1} x
//     normalized -- so mu is the Rayleigh quotient of the x before the last
//     step, and with maxite = 0 mu is 0. This reproduces NVIDIA's mu and x
//     bit for bit on the matrices tried;
//   - csrzfdHost's P maps each row of P A to a row of A, a maximum transversal
//     found column by column (cheap assignment, then depth-first
//     augmentation), unmatched columns given the unmatched rows in order.
//
// Not reproduced: which index NVIDIA's Cholesky names when several columns
// are independent of one another (its internal order differs), and
// its Cholesky with reorder = 1 reading the upper triangle it documents as
// ignored.
// The deprecated routines are what this file defines, and refers to by address when it records them.
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <cusolverSp.h>
#include <cusolverSp_LOWLEVEL_PREVIEW.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <mutex>
#include <queue>
#include <set>
#include <tuple>
#include <type_traits>
#include <vector>

#include <cuda_runtime.h>

#include "capture_defer.hpp"
#include "complex_linalg.hpp"
#include "cusolver_metis.hpp"

// Every device access below goes through this, so that the first one a call makes is its commit point (see
// capture_defer.hpp).
#define cudaMemcpy ::vgpu_capture::memcpy_commit

namespace {

using cd = std::complex<double>;

struct SpHandle { cudaStream_t stream = nullptr; };
// The low-level QR's state (cusolver_sp_lowlevel.inc); the device form shares
// csrqrInfo_t with the batched QR, so it lives in QrInfo.
struct QrLL {
  int m = -1, n = -1, nnz = -1;
  bool analyzed = false, buffered = false, setup = false, factored = false;
  std::vector<int> off, col;
  std::vector<cd> vals;                 // A - mu I, on A's pattern plus the diagonal
  std::vector<int> voff, vcol;          // that matrix's pattern
  std::vector<std::map<int, cd>> v;     // reflector k: rows > k (v(k) = 1 implicit)
  std::vector<cd> tau;
  std::vector<std::map<int, cd>> r;     // row k of R: columns >= k
};
struct QrInfo {
  int m = -1, n = -1, nnz = -1;
  QrLL ll;
};

std::mutex& g_mu = *new std::mutex;
std::set<const void*>& g_live = *new std::set<const void*>;
// The handles, and the matrix descriptors of a call being recorded into a graph, with how to copy each: what a
// recorded call runs on when the graph does (the program may destroy them as soon as the capture ends). The info
// objects are not copied -- they are the state the factorization and the solve share, and have to outlive the graph.
vgpu_capture::Registry& g_reg = *new vgpu_capture::Registry;
template <class T> T* track(T* p) { std::lock_guard<std::mutex> l(g_mu); g_live.insert(p); return p; }
bool known(const void* p) {
  if (g_reg.known(p)) return true;   // a handle's copy, in a recorded call
  std::lock_guard<std::mutex> l(g_mu);
  return p && g_live.count(p);
}
void untrack(const void* p) { std::lock_guard<std::mutex> l(g_mu); g_live.erase(p); }
// The stream a call on `h` is on (null for a handle that is not one).
cudaStream_t sp_stream(cusolverSpHandle_t h) { return h && known(h) ? reinterpret_cast<SpHandle*>(h)->stream : nullptr; }

// Element types: the host or device type, the arithmetic type, the type tol and
// norms come in.
template <class T> struct El;
template <> struct El<float> {
  using V = double;
  using R = float;
  static V in(float x) { return x; }
  static float out(V v) { return (float)v; }
};
template <> struct El<double> {
  using V = double;
  using R = double;
  static V in(double x) { return x; }
  static double out(V v) { return v; }
};
template <> struct El<cuComplex> {
  using V = cd;
  using R = float;
  static V in(cuComplex x) { return {x.x, x.y}; }
  static cuComplex out(V v) { return make_cuComplex((float)v.real(), (float)v.imag()); }
};
template <> struct El<cuDoubleComplex> {
  using V = cd;
  using R = double;
  static V in(cuDoubleComplex x) { return {x.x, x.y}; }
  static cuDoubleComplex out(V v) { return make_cuDoubleComplex(v.real(), v.imag()); }
};
inline double conj_of(double v) { return v; }
inline cd conj_of(cd v) { return std::conj(v); }
inline double re_of(double v) { return v; }
inline double re_of(cd v) { return v.real(); }

// Reads n elements from host or device memory.
template <class T> bool read(const T* p, size_t n, bool device, std::vector<typename El<T>::V>* out) {
  std::vector<T> raw(n);
  if (n) {
    if (!p) return false;
    if (device) {
      if (cudaMemcpy(raw.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost) != cudaSuccess) return false;
    } else {
      std::memcpy(raw.data(), p, n * sizeof(T));
    }
  }
  out->resize(n);
  for (size_t i = 0; i < n; ++i) (*out)[i] = El<T>::in(raw[i]);
  return true;
}
template <class T> bool write(T* p, const std::vector<typename El<T>::V>& v, bool device) {
  std::vector<T> raw(v.size());
  for (size_t i = 0; i < v.size(); ++i) raw[i] = El<T>::out(v[i]);
  if (raw.empty()) return true;
  if (!p) return false;
  if (device) return cudaMemcpy(p, raw.data(), raw.size() * sizeof(T), cudaMemcpyHostToDevice) == cudaSuccess;
  std::memcpy(p, raw.data(), raw.size() * sizeof(T));
  return true;
}
bool read_ints(const int* p, size_t n, bool device, std::vector<int>* out) {
  out->assign(n, 0);
  if (!n) return true;
  if (!p) return false;
  if (device) return cudaMemcpy(out->data(), p, n * sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess;
  std::memcpy(out->data(), p, n * sizeof(int));
  return true;
}
template <class S> bool put_scalar(S* p, S v, bool device) {
  if (!p) return true;
  if (device) return cudaMemcpy(p, &v, sizeof(S), cudaMemcpyHostToDevice) == cudaSuccess;
  *p = v;
  return true;
}

// A CSR matrix rebased to zero, rows' columns as given.
template <class V> struct Csr {
  int m = 0, n = 0;
  std::vector<int> off, col;
  std::vector<V> val;
};

cusolverStatus_t check_descr(cusparseMatDescr_t d) {
  if (!d) return CUSOLVER_STATUS_INVALID_VALUE;
  if (cusparseGetMatType(d) != CUSPARSE_MATRIX_TYPE_GENERAL) return CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
  return CUSOLVER_STATUS_SUCCESS;
}
int base_of(cusparseMatDescr_t d) { return cusparseGetMatIndexBase(d) == CUSPARSE_INDEX_BASE_ONE ? 1 : 0; }

bool read_pattern(int m, int n, int nnz, cusparseMatDescr_t d, const int* off, const int* col, bool device,
                  std::vector<int>* o, std::vector<int>* c) {
  if (m < 0 || n < 0 || nnz < 0) return false;
  const int b = base_of(d);
  if (!read_ints(off, (size_t)m + 1, device, o) || !read_ints(col, (size_t)nnz, device, c)) return false;
  for (int& v : *o) v -= b;
  for (int& v : *c) v -= b;
  if ((*o)[0] != 0 || (*o)[(size_t)m] != nnz) return false;
  for (int i = 0; i < m; ++i)
    if ((*o)[(size_t)i + 1] < (*o)[(size_t)i]) return false;
  for (int v : *c)
    if (v < 0 || v >= n) return false;
  return true;
}
template <class T>
cusolverStatus_t read_csr(int m, int n, int nnz, cusparseMatDescr_t d, const T* val, const int* off, const int* col,
                          bool device, Csr<typename El<T>::V>* out) {
  if (const cusolverStatus_t st = check_descr(d); st != CUSOLVER_STATUS_SUCCESS) return st;
  out->m = m;
  out->n = n;
  if (!read_pattern(m, n, nnz, d, off, col, device, &out->off, &out->col)) return CUSOLVER_STATUS_INVALID_VALUE;
  if (!read<T>(val, (size_t)nnz, device, &out->val)) return CUSOLVER_STATUS_EXECUTION_FAILED;
  return CUSOLVER_STATUS_SUCCESS;
}

/* ---- orderings ---- */

// The pattern of A + A^T without its diagonal, as neighbour sets.
std::vector<std::set<int>> symmetric_graph(int n, const std::vector<int>& off, const std::vector<int>& col) {
  std::vector<std::set<int>> g((size_t)n);
  for (int i = 0; i < n; ++i)
    for (int k = off[(size_t)i]; k < off[(size_t)i + 1]; ++k) {
      const int j = col[(size_t)k];
      if (j == i || j < 0 || j >= n) continue;
      g[(size_t)i].insert(j);
      g[(size_t)j].insert(i);
    }
  return g;
}

// Minimum degree on an elimination graph: the node of least degree next (the
// lowest index on a tie), its neighbours made a clique.
std::vector<int> minimum_degree(std::vector<std::set<int>> g) {
  const int n = (int)g.size();
  std::set<std::pair<int, int>> queue;
  for (int i = 0; i < n; ++i) queue.insert({(int)g[(size_t)i].size(), i});
  std::vector<int> order;
  order.reserve((size_t)n);
  while (!queue.empty()) {
    const int v = queue.begin()->second;
    queue.erase(queue.begin());
    order.push_back(v);
    const std::vector<int> nb(g[(size_t)v].begin(), g[(size_t)v].end());
    for (int u : nb) {
      queue.erase({(int)g[(size_t)u].size(), u});
      g[(size_t)u].erase(v);
    }
    for (size_t a = 0; a < nb.size(); ++a)
      for (size_t b = a + 1; b < nb.size(); ++b) {
        g[(size_t)nb[a]].insert(nb[b]);
        g[(size_t)nb[b]].insert(nb[a]);
      }
    for (int u : nb) queue.insert({(int)g[(size_t)u].size(), u});
    g[(size_t)v].clear();
  }
  return order;
}

// Which rows store their diagonal entry: NVIDIA's RCM counts it in a node's
// degree.
std::vector<char> diagonal_flags(int n, const std::vector<int>& off, const std::vector<int>& col) {
  std::vector<char> d((size_t)n, 0);
  for (int i = 0; i < n; ++i)
    for (int k = off[(size_t)i]; k < off[(size_t)i + 1]; ++k)
      if (col[(size_t)k] == i) d[(size_t)i] = 1;
  return d;
}

// Reverse Cuthill-McKee as csrsymrcmHost orders, measured on an RTX 3060
// (CUDA 13.0) against 100 matrices (random, banded, 2D and 3D grids, dense
// rows, several components) -- every permutation identical:
//   - components in the order of their lowest-numbered node;
//   - each one's root by George and Liu's pseudo-peripheral search from that
//     node: breadth-first levels (neighbours ascending), the first node of
//     least degree in the last level becomes the root, repeated while the
//     level count grows -- without SPARSPAK's early return when the first
//     structure is already a path, and keeping the last node tried;
//   - Cuthill-McKee from the root: each node's unnumbered neighbours by their
//     count of unnumbered neighbours, plus one when the row stores its
//     diagonal, the lower index first on a tie;
//   - the whole sequence reversed at the end.
// A node's degree in the root search also counts its stored diagonal.
std::vector<int> rcm(const std::vector<std::set<int>>& g, const std::vector<char>& diag) {
  const int n = (int)g.size();
  auto dg = [&](int v) { return diag.empty() ? 0 : (int)diag[(size_t)v]; };
  std::vector<char> num((size_t)n, 0);
  std::vector<int> order, mark((size_t)n, -1);
  int stamp = 0;
  auto levels = [&](int root, std::vector<std::vector<int>>* lv) {
    ++stamp;
    lv->assign(1, {root});
    mark[(size_t)root] = stamp;
    for (;;) {
      std::vector<int> next;
      for (int v : lv->back())
        for (int u : g[(size_t)v])
          if (!num[(size_t)u] && mark[(size_t)u] != stamp) {
            mark[(size_t)u] = stamp;
            next.push_back(u);
          }
      if (next.empty()) return;
      lv->push_back(std::move(next));
    }
  };
  std::vector<std::vector<int>> lv, lv2;
  for (int s0 = 0; s0 < n; ++s0) {
    if (num[(size_t)s0]) continue;
    levels(s0, &lv);
    int root = s0;
    for (;;) {
      const std::vector<int>& last = lv.back();
      root = last[0];
      int best = (int)g[(size_t)root].size() + dg(root);
      for (int v : last)
        if ((int)g[(size_t)v].size() + dg(v) < best) {
          best = (int)g[(size_t)v].size() + dg(v);
          root = v;
        }
      levels(root, &lv2);
      if (lv2.size() <= lv.size()) break;
      lv.swap(lv2);
    }
    size_t h = order.size();
    order.push_back(root);
    num[(size_t)root] = 1;
    for (; h < order.size(); ++h) {
      std::vector<std::pair<int, int>> kids;
      for (int u : g[(size_t)order[h]])
        if (!num[(size_t)u]) {
          int k = dg(u);
          for (int w : g[(size_t)u]) k += !num[(size_t)w];
          kids.push_back({k, u});
        }
      std::sort(kids.begin(), kids.end());
      for (const auto& [k, u] : kids) {
        num[(size_t)u] = 1;
        order.push_back(u);
      }
    }
  }
  std::reverse(order.begin(), order.end());
  return order;
}

// The elimination tree of the matrix with pattern g taken in the order p,
// then p rearranged into its postorder: children before parents, siblings and
// roots in ascending order.
std::vector<int> etree_postorder(const std::vector<std::set<int>>& g, const std::vector<int>& p) {
  const int n = (int)p.size();
  std::vector<int> pinv((size_t)n), parent((size_t)n, -1), anc((size_t)n, -1);
  for (int k = 0; k < n; ++k) pinv[(size_t)p[(size_t)k]] = k;
  for (int k = 0; k < n; ++k)
    for (int v : g[(size_t)p[(size_t)k]]) {
      int j = pinv[(size_t)v];
      while (j != -1 && j < k) {
        const int up = anc[(size_t)j];
        anc[(size_t)j] = k;
        if (up == -1) parent[(size_t)j] = k;
        j = up;
      }
    }
  std::vector<int> head((size_t)n, -1), next((size_t)n, -1), post, stack;
  for (int j = n - 1; j >= 0; --j)
    if (parent[(size_t)j] != -1) {
      next[(size_t)j] = head[(size_t)parent[(size_t)j]];
      head[(size_t)parent[(size_t)j]] = j;
    }
  for (int j = 0; j < n; ++j) {
    if (parent[(size_t)j] != -1) continue;
    stack.push_back(j);
    while (!stack.empty()) {
      const int t = stack.back(), c = head[(size_t)t];
      if (c == -1) {
        stack.pop_back();
        post.push_back(p[(size_t)t]);
      } else {
        head[(size_t)t] = next[(size_t)c];
        stack.push_back(c);
      }
    }
  }
  return post;
}

// Minimum degree on a quotient graph with an approximate degree, as
// csrsymamdHost orders: a variable's degree after each elimination is
// |A_i| + |L_p \ i| + sum over its other elements e of |L_e \ L_p| (AMD's
// external-degree bound, without the bounds AMD takes the minimum with),
// elements adjacent to the pivot absorbed, the lowest index on a tie; then
// the elimination tree's postorder. Identical to an RTX 3060's (CUDA 13.0)
// on the 100 matrices csrsymrcmHost was measured on.
std::vector<int> approximate_minimum_degree(const std::vector<std::set<int>>& g) {
  const int n = (int)g.size();
  std::vector<std::set<int>> A(g), E((size_t)n), L((size_t)n);
  std::vector<int> deg((size_t)n), order;
  std::set<std::pair<int, int>> queue;
  for (int i = 0; i < n; ++i) queue.insert({deg[(size_t)i] = (int)g[(size_t)i].size(), i});
  while (!queue.empty()) {
    const int p = queue.begin()->second;
    queue.erase(queue.begin());
    order.push_back(p);
    std::set<int> lp = A[(size_t)p];
    for (int e : E[(size_t)p]) lp.insert(L[(size_t)e].begin(), L[(size_t)e].end());
    lp.erase(p);
    for (int e : E[(size_t)p]) L[(size_t)e].clear();
    for (int i : lp) {
      A[(size_t)i].erase(p);
      for (int x : lp) A[(size_t)i].erase(x);
      for (int e : E[(size_t)p]) E[(size_t)i].erase(e);
      E[(size_t)i].insert(p);
    }
    L[(size_t)p] = lp;
    for (int i : lp) {
      int d = (int)A[(size_t)i].size() + (int)lp.size() - 1;
      for (int e : E[(size_t)i])
        if (e != p)
          for (int x : L[(size_t)e]) d += !lp.count(x);
      queue.erase({deg[(size_t)i], i});
      queue.insert({deg[(size_t)i] = d, i});
    }
    A[(size_t)p].clear();
    E[(size_t)p].clear();
  }
  return etree_postorder(g, order);
}

// METIS_NodeND's permutation of a graph (neighbour sets, no loops); minimum
// degree and the elimination tree's postorder if METIS fails.
std::vector<int> metis_order(const std::vector<std::set<int>>& g) {
  const int n = (int)g.size();
  if (n == 0) return {};
  std::vector<int64_t> xadj(1, 0), adj, perm, iperm;
  for (const auto& nb : g) {
    adj.insert(adj.end(), nb.begin(), nb.end());
    xadj.push_back((int64_t)adj.size());
  }
  if (vgpu::cusolver_metis::node_nd(n, xadj, adj, nullptr, &perm, &iperm) != 1 /* METIS_OK */)
    return etree_postorder(g, minimum_degree(g));
  return std::vector<int>(perm.begin(), perm.end());
}

// The column order `reorder` names, over the graph a factorization fills:
// 1 csrsymrcm's, 2 csrsymamd's, 3 csrmetisnd's (see csrmetisndHost).
std::vector<int> column_order(int reorder, const std::vector<std::set<int>>& g, const std::vector<char>& diag = {}) {
  if (reorder == 1) return rcm(g, diag);
  if (reorder == 2) return approximate_minimum_degree(g);
  if (reorder == 3) return metis_order(g);
  std::vector<int> q(g.size());
  for (size_t i = 0; i < q.size(); ++i) q[i] = (int)i;
  return q;
}
// The pattern of A^T A without its diagonal: columns sharing a row.
std::vector<std::set<int>> normal_graph(int m, int n, const std::vector<int>& off, const std::vector<int>& col) {
  std::vector<std::set<int>> g((size_t)n);
  for (int i = 0; i < m; ++i)
    for (int a = off[(size_t)i]; a < off[(size_t)i + 1]; ++a)
      for (int b = a + 1; b < off[(size_t)i + 1]; ++b) {
        const int x = col[(size_t)a], y = col[(size_t)b];
        if (x == y) continue;
        g[(size_t)x].insert(y);
        g[(size_t)y].insert(x);
      }
  return g;
}

/* ---- sparse LU with partial pivoting ---- */

template <class V> struct Lu {
  int n = 0;
  int singularity = -1;
  std::vector<int> q;                           // column k of the factors is column q[k] of A
  std::vector<int> prow;                        // pivot row of step k, -1 for an empty column
  std::vector<std::map<int, V>> u;              // row k of U, by factor column
  std::vector<std::tuple<int, int, V>> ops;     // b[i] -= l * b[r], in order
};

template <class V> Lu<V> lu_factor(const Csr<V>& a, const std::vector<int>& q, double tol) {
  const int n = a.n;
  Lu<V> f;
  f.n = n;
  f.q = q;
  f.prow.assign((size_t)n, -1);
  f.u.resize((size_t)n);
  std::vector<int> qinv((size_t)n);
  for (int k = 0; k < n; ++k) qinv[(size_t)q[(size_t)k]] = k;
  std::vector<std::map<int, V>> rows((size_t)n);
  std::vector<std::set<int>> colrows((size_t)n);
  for (int i = 0; i < n; ++i)
    for (int k = a.off[(size_t)i]; k < a.off[(size_t)i + 1]; ++k) {
      const int c = qinv[(size_t)a.col[(size_t)k]];
      rows[(size_t)i][c] += a.val[(size_t)k];
      colrows[(size_t)c].insert(i);
    }
  std::vector<char> active((size_t)n, 1);
  for (int k = 0; k < n; ++k) {
    int best = -1;
    double bv = -1;
    for (int i : colrows[(size_t)k]) {
      if (!active[(size_t)i]) continue;
      const double v = std::abs(rows[(size_t)i][k]);
      if (v > bv) {
        bv = v;
        best = i;
      }
    }
    if (best < 0 || !(bv > tol)) {
      if (f.singularity < 0) f.singularity = k;
      if (best < 0) continue;
    }
    const int r = best;
    f.prow[(size_t)k] = r;
    active[(size_t)r] = 0;
    const V piv = rows[(size_t)r][k];
    const std::vector<int> targets(colrows[(size_t)k].begin(), colrows[(size_t)k].end());
    for (int i : targets) {
      if (!active[(size_t)i]) continue;
      auto it = rows[(size_t)i].find(k);
      const V l = it->second / piv;
      rows[(size_t)i].erase(it);
      f.ops.emplace_back(i, r, l);
      for (auto rc = rows[(size_t)r].upper_bound(k); rc != rows[(size_t)r].end(); ++rc) {
        auto [pos, inserted] = rows[(size_t)i].try_emplace(rc->first, V(0.0));
        pos->second -= l * rc->second;
        if (inserted) colrows[(size_t)rc->first].insert(i);
      }
    }
    f.u[(size_t)k] = std::map<int, V>(rows[(size_t)r].lower_bound(k), rows[(size_t)r].end());
  }
  return f;
}

template <class V> std::vector<V> lu_solve(const Lu<V>& f, std::vector<V> b) {
  for (const auto& [i, r, l] : f.ops) b[(size_t)i] -= l * b[(size_t)r];
  const int n = f.n;
  std::vector<V> z((size_t)n), x((size_t)n);
  for (int k = n - 1; k >= 0; --k) {
    const int r = f.prow[(size_t)k];
    if (r < 0) {
      z[(size_t)k] = V(NAN);
    } else {
      V s = b[(size_t)r];
      const auto& row = f.u[(size_t)k];
      for (auto it = row.upper_bound(k); it != row.end(); ++it) s -= it->second * z[(size_t)it->first];
      z[(size_t)k] = s / row.at(k);
    }
    x[(size_t)f.q[(size_t)k]] = z[(size_t)k];
  }
  return x;
}

/* ---- sparse QR by Givens rotations ---- */

template <class V> struct Qr {
  int n = 0;
  std::vector<int> q;
  std::vector<std::map<int, V>> r;  // row k of R, by factor column
  std::vector<V> qtb;               // (Q^H b)[k] for those rows
};

// The rotation that turns (r, f) into (rho, 0): [c s; -conj(s) c], c real.
template <class V> void givens(V r, V f, double* c, V* s) {
  const double ar = std::abs(r), af = std::abs(f);
  if (af == 0.0) {
    *c = 1;
    *s = V(0.0);
    return;
  }
  if (ar == 0.0) {
    *c = 0;
    *s = conj_of(f) / af;
    return;
  }
  const double nrm = std::hypot(ar, af);
  *c = ar / nrm;
  *s = (r / ar) * conj_of(f) / nrm;
}

template <class V>
Qr<V> qr_factor(const Csr<V>& a, const std::vector<int>& q, const std::vector<V>& b) {
  const int n = a.n;
  Qr<V> f;
  f.n = n;
  f.q = q;
  f.r.resize((size_t)n);
  f.qtb.assign((size_t)n, V(0.0));
  std::vector<char> filled((size_t)n, 0);
  std::vector<int> qinv((size_t)n);
  for (int k = 0; k < n; ++k) qinv[(size_t)q[(size_t)k]] = k;
  for (int i = 0; i < a.m; ++i) {
    std::map<int, V> row;
    for (int k = a.off[(size_t)i]; k < a.off[(size_t)i + 1]; ++k) row[qinv[(size_t)a.col[(size_t)k]]] += a.val[(size_t)k];
    V beta = b.empty() ? V(0.0) : b[(size_t)i];
    while (!row.empty()) {
      const int k = row.begin()->first;
      if (!filled[(size_t)k]) {
        f.r[(size_t)k] = std::move(row);
        f.qtb[(size_t)k] = beta;
        filled[(size_t)k] = 1;
        row.clear();
        beta = V(0.0);
        break;
      }
      auto& rk = f.r[(size_t)k];
      double c;
      V s;
      givens(rk[k], row.begin()->second, &c, &s);
      std::map<int, V> merged;
      for (const auto& [col, v] : rk) merged[col] = c * v;
      for (const auto& [col, v] : row) merged[col] += s * v;
      std::map<int, V> rest;
      for (const auto& [col, v] : row) rest[col] = c * v;
      for (const auto& [col, v] : rk) rest[col] -= conj_of(s) * v;
      rest.erase(k);
      for (auto it = rest.begin(); it != rest.end();) it = it->second == V(0.0) ? rest.erase(it) : std::next(it);
      const V qk = f.qtb[(size_t)k];
      f.qtb[(size_t)k] = c * qk + s * beta;
      beta = c * beta - conj_of(s) * qk;
      rk = std::move(merged);
      row = std::move(rest);
    }
  }
  return f;
}

template <class V> int qr_singularity(const Qr<V>& f, double tol) {
  for (int k = 0; k < f.n; ++k) {
    const auto it = f.r[(size_t)k].find(k);
    if (it == f.r[(size_t)k].end() || !(std::abs(it->second) > tol)) return k;
  }
  return -1;
}
template <class V> std::vector<V> qr_solve(const Qr<V>& f) {
  const int n = f.n;
  std::vector<V> z((size_t)n), x((size_t)n);
  for (int k = n - 1; k >= 0; --k) {
    V s = f.qtb[(size_t)k];
    const auto& row = f.r[(size_t)k];
    for (auto it = row.upper_bound(k); it != row.end(); ++it) s -= it->second * z[(size_t)it->first];
    const auto d = row.find(k);
    z[(size_t)k] = d == row.end() ? V(NAN) : s / d->second;
    x[(size_t)f.q[(size_t)k]] = z[(size_t)k];
  }
  return x;
}

/* ---- sparse Cholesky, lower triangle ---- */

template <class V>
std::vector<V> cholesky_solve(const Csr<V>& a, const std::vector<int>& q, const std::vector<V>& b, double tol,
                              int* singularity) {
  const int n = a.n;
  std::vector<int> qinv((size_t)n);
  for (int k = 0; k < n; ++k) qinv[(size_t)q[(size_t)k]] = k;
  // Column j of the permuted lower triangle: rows >= j.
  std::vector<std::map<int, V>> cols((size_t)n);
  for (int i = 0; i < n; ++i)
    for (int k = a.off[(size_t)i]; k < a.off[(size_t)i + 1]; ++k) {
      const int j = a.col[(size_t)k];
      if (j > i) continue;  // the upper part is ignored
      int pi = qinv[(size_t)i], pj = qinv[(size_t)j];
      V v = a.val[(size_t)k];
      if (pi < pj) {
        std::swap(pi, pj);
        v = conj_of(v);
      }
      cols[(size_t)pj][pi] += v;
    }
  *singularity = -1;
  for (int k = 0; k < n; ++k) {
    auto& ck = cols[(size_t)k];
    const auto dit = ck.find(k);
    const double d = dit == ck.end() ? 0.0 : re_of(dit->second);
    const double lkk = std::sqrt(std::fabs(d));
    if ((!(d > 0.0) || !(lkk > tol)) && *singularity < 0) *singularity = k;
    ck[k] = V(lkk);
    for (auto it = ck.upper_bound(k); it != ck.end(); ++it) it->second /= lkk;
    for (auto it = ck.upper_bound(k); it != ck.end(); ++it)
      for (auto jt = ck.upper_bound(k); jt != std::next(it); ++jt)
        cols[(size_t)jt->first][it->first] -= it->second * conj_of(jt->second);
  }
  std::vector<V> y((size_t)n), x((size_t)n);
  for (int k = 0; k < n; ++k) y[(size_t)k] = b[(size_t)q[(size_t)k]];
  for (int k = 0; k < n; ++k) {  // L y = P b
    y[(size_t)k] /= cols[(size_t)k].at(k);
    for (auto it = cols[(size_t)k].upper_bound(k); it != cols[(size_t)k].end(); ++it)
      y[(size_t)it->first] -= it->second * y[(size_t)k];
  }
  for (int k = n - 1; k >= 0; --k) {  // L^H z = y
    V s = y[(size_t)k];
    for (auto it = cols[(size_t)k].upper_bound(k); it != cols[(size_t)k].end(); ++it)
      s -= conj_of(it->second) * y[(size_t)it->first];
    y[(size_t)k] = s / cols[(size_t)k].at(k);
  }
  for (int k = 0; k < n; ++k) x[(size_t)q[(size_t)k]] = y[(size_t)k];
  return x;
}

/* ---- the entry points' bodies ---- */

enum class Method { Lu, Qr, Chol };

template <class T>
cusolverStatus_t linear_solve(Method method, cusolverSpHandle_t h, int n, int nnz, cusparseMatDescr_t d, const T* val,
                              const int* off, const int* col, const T* b, double tol, int reorder, T* x,
                              int* singularity, bool device) {
  using V = typename El<T>::V;
  if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;
  if (reorder < 0 || reorder > 3 || n < 0 || nnz < 0 || !singularity) return CUSOLVER_STATUS_INVALID_VALUE;
  Csr<V> a;
  if (const cusolverStatus_t st = read_csr<T>(n, n, nnz, d, val, off, col, device, &a); st != CUSOLVER_STATUS_SUCCESS)
    return st;
  std::vector<V> bv;
  if (!read<T>(b, (size_t)n, device, &bv)) return CUSOLVER_STATUS_EXECUTION_FAILED;
  std::vector<V> xv;
  int sing = -1;
  if (method == Method::Lu) {
    const auto f = lu_factor(a, column_order(reorder, symmetric_graph(n, a.off, a.col), diagonal_flags(n, a.off, a.col)), tol);
    sing = f.singularity < 0 ? -1 : f.q[(size_t)f.singularity];
    xv = lu_solve(f, bv);
  } else if (method == Method::Qr) {
    const auto f = qr_factor(a, column_order(reorder, normal_graph(n, n, a.off, a.col)), bv);
    const int k = qr_singularity(f, tol);
    sing = k < 0 ? -1 : f.q[(size_t)k];
    xv = qr_solve(f);
  } else {
    const auto q = column_order(reorder, symmetric_graph(n, a.off, a.col), diagonal_flags(n, a.off, a.col));
    int k = -1;
    xv = cholesky_solve(a, q, bv, tol, &k);
    sing = k < 0 ? -1 : q[(size_t)k];
  }
  *singularity = sing;  // on the host for every variant, as NVIDIA's declares it
  return write<T>(x, xv, device) ? CUSOLVER_STATUS_SUCCESS : CUSOLVER_STATUS_EXECUTION_FAILED;
}

// min |b - A x| over x, the minimum-norm x when A is rank deficient.
template <class T>
cusolverStatus_t least_squares(cusolverSpHandle_t h, int m, int n, int nnz, cusparseMatDescr_t d, const T* val,
                               const int* off, const int* col, const T* b, double tol, int* rank, T* x, int* p,
                               typename El<T>::R* min_norm) {
  using V = typename El<T>::V;
  if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;
  if (m < 0 || n < 0 || m < n || nnz < 0) return CUSOLVER_STATUS_INVALID_VALUE;
  Csr<V> a;
  if (const cusolverStatus_t st = read_csr<T>(m, n, nnz, d, val, off, col, false, &a); st != CUSOLVER_STATUS_SUCCESS)
    return st;
  std::vector<V> bv;
  if (!read<T>(b, (size_t)m, false, &bv)) return CUSOLVER_STATUS_EXECUTION_FAILED;
  // The column permutation: a column found dependent on those before it
  // (|R(k,k)| <= tol) changes places with the last column not yet set aside,
  // and the factorization goes on from there -- what NVIDIA's reports in p.
  std::vector<int> perm((size_t)n);
  for (int i = 0; i < n; ++i) perm[(size_t)i] = i;
  int deficient = 0;
  for (int k = 0; k < n - deficient;) {
    const auto g = qr_factor(a, perm, std::vector<V>());
    int j = k;
    for (; j < n - deficient; ++j) {
      const auto it = g.r[(size_t)j].find(j);
      if (it == g.r[(size_t)j].end() || !(std::abs(it->second) > tol)) break;
    }
    if (j >= n - deficient) break;
    std::swap(perm[(size_t)j], perm[(size_t)(n - 1 - deficient)]);
    ++deficient;
    k = j;
  }
  const int r = n - deficient;
  std::vector<int> ident((size_t)n);
  for (int i = 0; i < n; ++i) ident[(size_t)i] = i;
  const auto f = qr_factor(a, ident, bv);
  std::vector<V> xv;
  if (r == n) {
    xv = qr_solve(f);
  } else {  // R is n x n: its pseudo-inverse applied to Q^H b, through an SVD
    vgpu_la::CMat rm{std::vector<cd>((size_t)n * n, 0.0), std::max(n, 1)};
    for (int k = 0; k < n; ++k)
      for (const auto& [c, v] : f.r[(size_t)k]) rm(k, c) = cd(v);
    std::vector<double> s;
    vgpu_la::CMat u, vv;
    vgpu_la::svd(rm, n, n, false, &s, &u, &vv);
    std::vector<cd> t((size_t)n, 0.0);
    for (int j = 0; j < r; ++j) {  // the rank's largest singular values, descending
      cd dot = 0;
      for (int i = 0; i < n; ++i) dot += std::conj(u(i, j)) * cd(f.qtb[(size_t)i]);
      t[(size_t)j] = dot / s[(size_t)j];
    }
    xv.assign((size_t)n, V(0.0));
    for (int i = 0; i < n; ++i) {
      cd acc = 0;
      for (int j = 0; j < n; ++j) acc += vv(i, j) * t[(size_t)j];
      if constexpr (std::is_same_v<V, double>) xv[(size_t)i] = acc.real();
      else xv[(size_t)i] = acc;
    }
  }
  double res = 0;  // |b - A x|, as it is
  for (int i = 0; i < m; ++i) {
    V s = bv[(size_t)i];
    for (int k = a.off[(size_t)i]; k < a.off[(size_t)i + 1]; ++k) s -= a.val[(size_t)k] * xv[(size_t)a.col[(size_t)k]];
    res += std::norm(cd(s));
  }
  if (rank) *rank = r;
  if (p) std::copy(perm.begin(), perm.end(), p);
  if (min_norm) *min_norm = (typename El<T>::R)std::sqrt(res);
  return write<T>(x, xv, false) ? CUSOLVER_STATUS_SUCCESS : CUSOLVER_STATUS_EXECUTION_FAILED;
}

template <class T>
cusolverStatus_t eig_shift_inverse(cusolverSpHandle_t h, int m, int nnz, cusparseMatDescr_t d, const T* val,
                                   const int* off, const int* col, T mu0, const T* x0, int maxite, double tol, T* mu,
                                   T* x, bool device) {
  using V = typename El<T>::V;
  if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;
  if (m < 0 || nnz < 0 || maxite < 0) return CUSOLVER_STATUS_INVALID_VALUE;
  Csr<V> a;
  if (const cusolverStatus_t st = read_csr<T>(m, m, nnz, d, val, off, col, device, &a); st != CUSOLVER_STATUS_SUCCESS)
    return st;
  std::vector<V> xv;
  if (!read<T>(x0, (size_t)m, device, &xv)) return CUSOLVER_STATUS_EXECUTION_FAILED;
  auto normalize = [](std::vector<V>& v) {
    double nr = 0;
    for (const V& e : v) nr += std::norm(cd(e));
    nr = std::sqrt(nr);
    for (V& e : v) e /= nr;
  };
  auto apply = [&](const std::vector<V>& v) {
    std::vector<V> y((size_t)m, V(0.0));
    for (int i = 0; i < m; ++i)
      for (int k = a.off[(size_t)i]; k < a.off[(size_t)i + 1]; ++k)
        y[(size_t)i] += a.val[(size_t)k] * v[(size_t)a.col[(size_t)k]];
    return y;
  };
  // A - mu0 I, factored once.
  const V shift = El<T>::in(mu0);
  Csr<V> s = a;
  {
    Csr<V> t{m, m, {0}, {}, {}};
    for (int i = 0; i < m; ++i) {
      bool diag = false;
      for (int k = a.off[(size_t)i]; k < a.off[(size_t)i + 1]; ++k) {
        t.col.push_back(a.col[(size_t)k]);
        t.val.push_back(a.val[(size_t)k] - (a.col[(size_t)k] == i && !diag ? shift : V(0.0)));
        diag = diag || a.col[(size_t)k] == i;
      }
      if (!diag) {
        t.col.push_back(i);
        t.val.push_back(-shift);
      }
      t.off.push_back((int)t.col.size());
    }
    s = std::move(t);
  }
  std::vector<int> ident((size_t)m);
  for (int i = 0; i < m; ++i) ident[(size_t)i] = i;
  const auto f = lu_factor(s, ident, 0.0);
  normalize(xv);
  V muv(0.0);
  for (int it = 0; it < maxite; ++it) {
    const auto ax = apply(xv);
    muv = V(0.0);
    for (int i = 0; i < m; ++i) muv += conj_of(xv[(size_t)i]) * ax[(size_t)i];
    double r = 0;
    for (int i = 0; i < m; ++i) r += std::norm(cd(ax[(size_t)i] - muv * xv[(size_t)i]));
    if (std::sqrt(r) <= tol) break;
    xv = lu_solve(f, xv);
    normalize(xv);
  }
  const std::vector<V> one{muv};
  return write<T>(mu, one, device) && write<T>(x, xv, device) ? CUSOLVER_STATUS_SUCCESS
                                                               : CUSOLVER_STATUS_EXECUTION_FAILED;
}

// The eigenvalues in the closed box [lb, ru] of the complex plane, counted.
template <class T>
cusolverStatus_t eig_count(cusolverSpHandle_t h, int m, int nnz, cusparseMatDescr_t d, const T* val, const int* off,
                           const int* col, cd lb, cd ru, int* num) {
  using V = typename El<T>::V;
  if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;
  if (m < 0 || nnz < 0 || !num) return CUSOLVER_STATUS_INVALID_VALUE;
  Csr<V> a;
  if (const cusolverStatus_t st = read_csr<T>(m, m, nnz, d, val, off, col, false, &a); st != CUSOLVER_STATUS_SUCCESS)
    return st;
  vgpu_la::CMat dense{std::vector<cd>((size_t)m * m, 0.0), std::max(m, 1)};
  for (int i = 0; i < m; ++i)
    for (int k = a.off[(size_t)i]; k < a.off[(size_t)i + 1]; ++k) dense(i, a.col[(size_t)k]) += cd(a.val[(size_t)k]);
  std::vector<cd> w;
  if (vgpu_la::geev(dense, m, &w, nullptr) != 0) return CUSOLVER_STATUS_EXECUTION_FAILED;
  int count = 0;
  for (const cd& e : w)
    count += e.real() >= lb.real() && e.real() <= ru.real() && e.imag() >= lb.imag() && e.imag() <= ru.imag();
  *num = count;
  return CUSOLVER_STATUS_SUCCESS;
}

// A maximum transversal: P[j] is the row of A that row j of P A is.
template <class T>
cusolverStatus_t zero_free_diagonal(cusolverSpHandle_t h, int n, int nnz, cusparseMatDescr_t d, const T* val,
                                    const int* off, const int* col, int* P, int* numnz) {
  using V = typename El<T>::V;
  if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;
  if (n < 0 || nnz < 0 || !P || !numnz) return CUSOLVER_STATUS_INVALID_VALUE;
  Csr<V> a;
  if (const cusolverStatus_t st = read_csr<T>(n, n, nnz, d, val, off, col, false, &a); st != CUSOLVER_STATUS_SUCCESS)
    return st;
  std::vector<std::vector<int>> rows_of((size_t)n);  // column j's rows, ascending
  for (int i = 0; i < n; ++i)
    for (int k = a.off[(size_t)i]; k < a.off[(size_t)i + 1]; ++k)
      rows_of[(size_t)a.col[(size_t)k]].push_back(i);
  for (auto& r : rows_of) {
    std::sort(r.begin(), r.end());
    r.erase(std::unique(r.begin(), r.end()), r.end());
  }
  std::vector<int> row_match((size_t)n, -1), col_match((size_t)n, -1);
  std::vector<int> seen((size_t)n, -1);
  std::function<bool(int, int)> augment = [&](int j, int stamp) {
    for (int i : rows_of[(size_t)j]) {
      if (seen[(size_t)i] == stamp) continue;
      seen[(size_t)i] = stamp;
      if (row_match[(size_t)i] < 0 || augment(row_match[(size_t)i], stamp)) {
        row_match[(size_t)i] = j;
        col_match[(size_t)j] = i;
        return true;
      }
    }
    return false;
  };
  int matched = 0;
  for (int j = 0; j < n; ++j) {
    for (int i : rows_of[(size_t)j])  // cheap assignment first
      if (row_match[(size_t)i] < 0) {
        row_match[(size_t)i] = j;
        col_match[(size_t)j] = i;
        break;
      }
    if (col_match[(size_t)j] < 0) augment(j, j);
    matched += col_match[(size_t)j] >= 0;
  }
  std::vector<int> spare;
  for (int i = 0; i < n; ++i)
    if (row_match[(size_t)i] < 0) spare.push_back(i);
  size_t next = 0;
  for (int j = 0; j < n; ++j) P[j] = col_match[(size_t)j] >= 0 ? col_match[(size_t)j] : spare[next++];
  *numnz = matched;
  return CUSOLVER_STATUS_SUCCESS;
}

}  // namespace

#define VGPU_EXPORT extern "C" __attribute__((visibility("default")))

// A matrix descriptor, copied for a recorded call (the descriptor itself is cuSPARSE's).
namespace {
void* clone_descr(const void* q, vgpu_capture::Snapshot&) {
  cusparseMatDescr_t o = const_cast<cusparseMatDescr_t>(reinterpret_cast<const cusparseMatDescr*>(q)), c = nullptr;
  if (cusparseCreateMatDescr(&c) != CUSPARSE_STATUS_SUCCESS) return nullptr;
  cusparseSetMatType(c, cusparseGetMatType(o));
  cusparseSetMatFillMode(c, cusparseGetMatFillMode(o));
  cusparseSetMatDiagType(c, cusparseGetMatDiagType(o));
  cusparseSetMatIndexBase(c, cusparseGetMatIndexBase(o));
  return c;
}
void destroy_descr(void* q) { cusparseDestroyMatDescr(reinterpret_cast<cusparseMatDescr_t>(q)); }
// Makes `d` something a recorded call can copy, for as long as the call is being made.
struct DescrScope {
  const void* d;
  explicit DescrScope(cusparseMatDescr_t desc) : d(desc) { if (d) g_reg.track_raw(d, clone_descr, destroy_descr); }
  ~DescrScope() { if (d) g_reg.untrack(d); }
};
}  // namespace

VGPU_EXPORT cusolverStatus_t cusolverSpCreate(cusolverSpHandle_t* h) {
  if (!h) return CUSOLVER_STATUS_INVALID_VALUE;
  *h = reinterpret_cast<cusolverSpHandle_t>(g_reg.track(track(new SpHandle())));
  return CUSOLVER_STATUS_SUCCESS;
}
VGPU_EXPORT cusolverStatus_t cusolverSpDestroy(cusolverSpHandle_t h) {
  if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;
  untrack(h);
  g_reg.untrack(h);
  delete reinterpret_cast<SpHandle*>(h);
  return CUSOLVER_STATUS_SUCCESS;
}
VGPU_EXPORT cusolverStatus_t cusolverSpSetStream(cusolverSpHandle_t h, cudaStream_t s) {
  if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;
  reinterpret_cast<SpHandle*>(h)->stream = s;
  return CUSOLVER_STATUS_SUCCESS;
}
VGPU_EXPORT cusolverStatus_t cusolverSpGetStream(cusolverSpHandle_t h, cudaStream_t* s) {
  if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;
  if (s) *s = reinterpret_cast<SpHandle*>(h)->stream;
  return CUSOLVER_STATUS_SUCCESS;
}

// Whether the pattern (four-array CSR) is symmetric.
VGPU_EXPORT cusolverStatus_t cusolverSpXcsrissymHost(cusolverSpHandle_t h, int m, int nnz, const cusparseMatDescr_t d,
                                                     const int* start, const int* end, const int* col, int* issym) {
  if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;
  if (m < 0 || nnz < 0 || !d || !issym) return CUSOLVER_STATUS_INVALID_VALUE;
  if (const cusolverStatus_t st = check_descr(d); st != CUSOLVER_STATUS_SUCCESS) return st;
  const int b = base_of(d);
  std::set<std::pair<int, int>> entries;
  for (int i = 0; i < m; ++i)
    for (int k = start[i] - b; k < end[i] - b; ++k) entries.insert({i, col[k] - b});
  bool sym = true;
  for (const auto& [i, j] : entries) sym = sym && entries.count({j, i});
  *issym = sym ? 1 : 0;
  return CUSOLVER_STATUS_SUCCESS;
}

// The four reorderings. symrcm, symamd and symmdq return exactly the
// permutations NVIDIA's do (see rcm, approximate_minimum_degree and below);
// symmdq is exact minimum degree (lowest index on a tie) followed by the
// elimination tree's postorder, identical on the same 100 matrices.
// csrmetisndHost runs METIS 5.1.0's METIS_NodeND (the vendored copy in
// nvidia/third_party/metis, built with 64-bit indices as NVIDIA's is) on the
// pattern of A + A^T without its diagonal, columns ascending; a reference
// METIS 5.1.0 build gave NVIDIA's permutation on 99 of 100 test matrices.
static cusolverStatus_t ordering(cusolverSpHandle_t h, int n, int nnz, cusparseMatDescr_t d, const int* off,
                                 const int* col, int* p, int reorder) {
  if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;
  if (n < 0 || nnz < 0 || !p) return CUSOLVER_STATUS_INVALID_VALUE;
  if (const cusolverStatus_t st = check_descr(d); st != CUSOLVER_STATUS_SUCCESS) return st;
  std::vector<int> o, c;
  if (!read_pattern(n, n, nnz, d, off, col, false, &o, &c)) return CUSOLVER_STATUS_INVALID_VALUE;
  // The three entry points number their orderings 1 (rcm), 2 (amd) and 3 (mdq);
  // the solvers' `reorder` numbers them 1, 2 and 3 (METIS), see column_order.
  const auto g = symmetric_graph(n, o, c);
  const auto q = reorder == 3 ? etree_postorder(g, minimum_degree(g)) : column_order(reorder, g, diagonal_flags(n, o, c));
  std::copy(q.begin(), q.end(), p);
  return CUSOLVER_STATUS_SUCCESS;
}
VGPU_EXPORT cusolverStatus_t cusolverSpXcsrsymrcmHost(cusolverSpHandle_t h, int n, int nnz, const cusparseMatDescr_t d,
                                                      const int* off, const int* col, int* p) {
  return ordering(h, n, nnz, d, off, col, p, 1);
}
VGPU_EXPORT cusolverStatus_t cusolverSpXcsrsymmdqHost(cusolverSpHandle_t h, int n, int nnz, const cusparseMatDescr_t d,
                                                      const int* off, const int* col, int* p) {
  return ordering(h, n, nnz, d, off, col, p, 3);
}
VGPU_EXPORT cusolverStatus_t cusolverSpXcsrsymamdHost(cusolverSpHandle_t h, int n, int nnz, const cusparseMatDescr_t d,
                                                      const int* off, const int* col, int* p) {
  return ordering(h, n, nnz, d, off, col, p, 2);
}
VGPU_EXPORT cusolverStatus_t cusolverSpXcsrmetisndHost(cusolverSpHandle_t h, int n, int nnz,
                                                       const cusparseMatDescr_t d, const int* off, const int* col,
                                                       const int64_t* options, int* p) {
  // The arguments, as an RTX 3060's cuSOLVER 13.0 answers them (measured):
  // n <= 0 or nnz = 0 is INVALID_VALUE, a negative nnz ALLOC_FAILED, a NULL
  // matrix descriptor MATRIX_TYPE_NOT_SUPPORTED, a non-general type likewise.
  // It takes a pattern whose offsets disagree with nnz, or whose columns are
  // out of range, and reads past the arrays; this refuses them
  // (INVALID_VALUE). A NULL p crashes it. An options array that asks for
  // Fortran numbering (METIS_OPTION_NUMBERING = 1) crashes it, because the
  // arrays it hands METIS are zero-based: INVALID_VALUE here.
  if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;
  if (!d) return CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
  if (n <= 0) return CUSOLVER_STATUS_INVALID_VALUE;
  if (nnz < 0) return CUSOLVER_STATUS_ALLOC_FAILED;
  if (nnz == 0 || !p) return CUSOLVER_STATUS_INVALID_VALUE;
  if (options && options[17 /* METIS_OPTION_NUMBERING */] == 1) return CUSOLVER_STATUS_INVALID_VALUE;
  if (const cusolverStatus_t st = check_descr(d); st != CUSOLVER_STATUS_SUCCESS) return st;
  std::vector<int> o, c;
  if (!read_pattern(n, n, nnz, d, off, col, false, &o, &c)) return CUSOLVER_STATUS_INVALID_VALUE;
  const auto g = symmetric_graph(n, o, c);
  std::vector<int64_t> xadj(1, 0), adj;
  for (const auto& nb : g) {
    adj.insert(adj.end(), nb.begin(), nb.end());
    xadj.push_back((int64_t)adj.size());
  }
  std::vector<int64_t> perm, iperm;
  if (vgpu::cusolver_metis::node_nd(n, xadj, adj, options, &perm, &iperm) != 1 /* METIS_OK */)
    return CUSOLVER_STATUS_INTERNAL_ERROR;
  for (int i = 0; i < n; ++i) p[i] = (int)perm[(size_t)i];
  return CUSOLVER_STATUS_SUCCESS;
}

// B = P A Q^T in place: row i of B is row p[i] of A, column j of B column q[j]
// of A; each row's columns ascending, and map permuted along with the values.
VGPU_EXPORT cusolverStatus_t cusolverSpXcsrperm_bufferSizeHost(cusolverSpHandle_t h, int m, int n, int nnz,
                                                               const cusparseMatDescr_t d, const int*, const int*,
                                                               const int*, const int*, size_t* bytes) {
  if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;
  if (m < 0 || n < 0 || nnz < 0 || !d || !bytes) return CUSOLVER_STATUS_INVALID_VALUE;
  *bytes = (size_t)(m + n + nnz + 1) * sizeof(int);
  return CUSOLVER_STATUS_SUCCESS;
}
VGPU_EXPORT cusolverStatus_t cusolverSpXcsrpermHost(cusolverSpHandle_t h, int m, int n, int nnz,
                                                    const cusparseMatDescr_t d, int* off, int* col, const int* p,
                                                    const int* q, int* map, void*) {
  if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;
  if (m < 0 || n < 0 || nnz < 0 || !p || !q) return CUSOLVER_STATUS_INVALID_VALUE;
  if (const cusolverStatus_t st = check_descr(d); st != CUSOLVER_STATUS_SUCCESS) return st;
  std::vector<int> o, c;
  if (!read_pattern(m, n, nnz, d, off, col, false, &o, &c)) return CUSOLVER_STATUS_INVALID_VALUE;
  const int b = base_of(d);
  std::vector<int> qinv((size_t)n, -1);
  for (int j = 0; j < n; ++j) {
    if (q[j] < 0 || q[j] >= n) return CUSOLVER_STATUS_INVALID_VALUE;
    qinv[(size_t)q[j]] = j;
  }
  std::vector<int> mp(map ? std::vector<int>(map, map + nnz) : std::vector<int>((size_t)nnz, 0));
  std::vector<int> no{0}, nc, nm;
  for (int i = 0; i < m; ++i) {
    if (p[i] < 0 || p[i] >= m) return CUSOLVER_STATUS_INVALID_VALUE;
    std::vector<std::pair<int, int>> row;
    for (int k = o[(size_t)p[i]]; k < o[(size_t)p[i] + 1]; ++k) row.push_back({qinv[(size_t)c[(size_t)k]], k});
    std::stable_sort(row.begin(), row.end(), [](auto x, auto y) { return x.first < y.first; });
    for (const auto& [cc, k] : row) {
      nc.push_back(cc);
      nm.push_back(mp[(size_t)k]);
    }
    no.push_back((int)nc.size());
  }
  for (int i = 0; i <= m; ++i) off[i] = no[(size_t)i] + b;
  for (int k = 0; k < nnz; ++k) col[k] = nc[(size_t)k] + b;
  if (map) std::copy(nm.begin(), nm.end(), map);
  return CUSOLVER_STATUS_SUCCESS;
}

/* ---- batched QR: A_j x_j = b_j (least squares when m > n), one pattern ---- */

VGPU_EXPORT cusolverStatus_t cusolverSpCreateCsrqrInfo(csrqrInfo_t* info) {
  if (!info) return CUSOLVER_STATUS_INVALID_VALUE;
  *info = reinterpret_cast<csrqrInfo_t>(track(new QrInfo()));
  return CUSOLVER_STATUS_SUCCESS;
}
VGPU_EXPORT cusolverStatus_t cusolverSpDestroyCsrqrInfo(csrqrInfo_t info) {
  if (!known(info)) return CUSOLVER_STATUS_INVALID_VALUE;
  untrack(info);
  delete reinterpret_cast<QrInfo*>(info);
  return CUSOLVER_STATUS_SUCCESS;
}
VGPU_EXPORT cusolverStatus_t cusolverSpXcsrqrAnalysisBatched(cusolverSpHandle_t h, int m, int n, int nnz,
                                                             const cusparseMatDescr_t d, const int* off,
                                                             const int* col, csrqrInfo_t info) {
  VGPU_REFUSE_CALL(sp_stream(h), CUSOLVER_STATUS_INTERNAL_ERROR, cusolverSpXcsrqrAnalysisBatched, h, m, n, nnz, d, off,
                   col, info);
  if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;
  if (!known(info) || m < n || n < 0 || nnz < 0) return CUSOLVER_STATUS_INVALID_VALUE;
  if (const cusolverStatus_t st = check_descr(d); st != CUSOLVER_STATUS_SUCCESS) return st;
  std::vector<int> o, c;
  if (!read_pattern(m, n, nnz, d, off, col, true, &o, &c)) return CUSOLVER_STATUS_INVALID_VALUE;
  *reinterpret_cast<QrInfo*>(info) = QrInfo{m, n, nnz, QrLL{}};
  return CUSOLVER_STATUS_SUCCESS;
}

namespace {
template <class T>
cusolverStatus_t qr_buffer_info(cusolverSpHandle_t h, int m, int n, int nnz, csrqrInfo_t info, size_t* internal,
                                size_t* work) {
  if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;
  if (!known(info)) return CUSOLVER_STATUS_INVALID_VALUE;
  const QrInfo& qi = *reinterpret_cast<const QrInfo*>(info);
  if (qi.m != m || qi.n != n || qi.nnz != nnz) return CUSOLVER_STATUS_INVALID_VALUE;  // analysis first
  if (internal) *internal = (size_t)(nnz + n + 1) * sizeof(T);
  if (work) *work = 256;
  return CUSOLVER_STATUS_SUCCESS;
}
template <class T>
cusolverStatus_t qr_solve_batched(cusolverSpHandle_t h, int m, int n, int nnz, cusparseMatDescr_t d, const T* val,
                                  const int* off, const int* col, const T* b, T* x, int batch, csrqrInfo_t info) {
  using V = typename El<T>::V;
  if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;
  if (!known(info) || batch < 0) return CUSOLVER_STATUS_INVALID_VALUE;
  const QrInfo& qi = *reinterpret_cast<const QrInfo*>(info);
  if (qi.m != m || qi.n != n || qi.nnz != nnz) return CUSOLVER_STATUS_INVALID_VALUE;
  if (const cusolverStatus_t st = check_descr(d); st != CUSOLVER_STATUS_SUCCESS) return st;
  Csr<V> a;
  a.m = m;
  a.n = n;
  if (!read_pattern(m, n, nnz, d, off, col, true, &a.off, &a.col)) return CUSOLVER_STATUS_INVALID_VALUE;
  std::vector<V> vals, bs;
  if (!read<T>(val, (size_t)nnz * batch, true, &vals) || !read<T>(b, (size_t)m * batch, true, &bs))
    return CUSOLVER_STATUS_EXECUTION_FAILED;
  std::vector<int> ident((size_t)n);
  for (int i = 0; i < n; ++i) ident[(size_t)i] = i;
  std::vector<V> xs((size_t)n * batch);
  for (int j = 0; j < batch; ++j) {
    a.val.assign(vals.begin() + (ptrdiff_t)j * nnz, vals.begin() + (ptrdiff_t)(j + 1) * nnz);
    const std::vector<V> bj(bs.begin() + (ptrdiff_t)j * m, bs.begin() + (ptrdiff_t)(j + 1) * m);
    const auto xj = qr_solve(qr_factor(a, ident, bj));
    std::copy(xj.begin(), xj.end(), xs.begin() + (ptrdiff_t)j * n);
  }
  return write<T>(x, xs, true) ? CUSOLVER_STATUS_SUCCESS : CUSOLVER_STATUS_EXECUTION_FAILED;
}
template <class T> cd to_cd(T v) { return cd(El<T>::in(v)); }
}  // namespace

#define VGPU_SP(P, T)                                                                                               \
  VGPU_EXPORT cusolverStatus_t cusolverSp##P##csrlsvluHost(                                                         \
      cusolverSpHandle_t h, int n, int nnz, const cusparseMatDescr_t d, const T* val, const int* off,               \
      const int* col, const T* b, El<T>::R tol, int reorder, T* x, int* singularity) {                              \
    return linear_solve<T>(Method::Lu, h, n, nnz, d, val, off, col, b, tol, reorder, x, singularity, false);        \
  }                                                                                                                 \
  VGPU_EXPORT cusolverStatus_t cusolverSp##P##csrlsvqr(cusolverSpHandle_t h, int m, int nnz,                        \
                                                       const cusparseMatDescr_t d, const T* val, const int* off,    \
                                                       const int* col, const T* b, El<T>::R tol, int reorder, T* x, \
                                                       int* singularity) {                                          \
    VGPU_REFUSE_CALL(sp_stream(h), CUSOLVER_STATUS_INTERNAL_ERROR, cusolverSp##P##csrlsvqr, h, m, nnz, d, val, off, \
                     col, b, tol, reorder, x, singularity);                                                         \
    return linear_solve<T>(Method::Qr, h, m, nnz, d, val, off, col, b, tol, reorder, x, singularity, true);         \
  }                                                                                                                 \
  VGPU_EXPORT cusolverStatus_t cusolverSp##P##csrlsvqrHost(                                                         \
      cusolverSpHandle_t h, int m, int nnz, const cusparseMatDescr_t d, const T* val, const int* off,               \
      const int* col, const T* b, El<T>::R tol, int reorder, T* x, int* singularity) {                              \
    return linear_solve<T>(Method::Qr, h, m, nnz, d, val, off, col, b, tol, reorder, x, singularity, false);        \
  }                                                                                                                 \
  VGPU_EXPORT cusolverStatus_t cusolverSp##P##csrlsvcholHost(                                                       \
      cusolverSpHandle_t h, int m, int nnz, const cusparseMatDescr_t d, const T* val, const int* off,               \
      const int* col, const T* b, El<T>::R tol, int reorder, T* x, int* singularity) {                              \
    return linear_solve<T>(Method::Chol, h, m, nnz, d, val, off, col, b, tol, reorder, x, singularity, false);      \
  }                                                                                                                 \
  VGPU_EXPORT cusolverStatus_t cusolverSp##P##csrlsvchol(cusolverSpHandle_t h, int m, int nnz,                      \
                                                         const cusparseMatDescr_t d, const T* val, const int* off,  \
                                                         const int* col, const T* b, El<T>::R tol, int reorder,     \
                                                         T* x, int* singularity) {                                  \
    VGPU_REFUSE_CALL(sp_stream(h), CUSOLVER_STATUS_INTERNAL_ERROR, cusolverSp##P##csrlsvchol, h, m, nnz, d, val,    \
                     off, col, b, tol, reorder, x, singularity);                                                    \
    return linear_solve<T>(Method::Chol, h, m, nnz, d, val, off, col, b, tol, reorder, x, singularity, true);       \
  }                                                                                                                 \
  VGPU_EXPORT cusolverStatus_t cusolverSp##P##csrlsqvqrHost(                                                        \
      cusolverSpHandle_t h, int m, int n, int nnz, const cusparseMatDescr_t d, const T* val, const int* off,        \
      const int* col, const T* b, El<T>::R tol, int* rank, T* x, int* p, El<T>::R* min_norm) {                      \
    return least_squares<T>(h, m, n, nnz, d, val, off, col, b, tol, rank, x, p, min_norm);                          \
  }                                                                                                                 \
  VGPU_EXPORT cusolverStatus_t cusolverSp##P##csreigvsiHost(                                                        \
      cusolverSpHandle_t h, int m, int nnz, const cusparseMatDescr_t d, const T* val, const int* off,               \
      const int* col, T mu0, const T* x0, int maxite, El<T>::R tol, T* mu, T* x) {                                  \
    return eig_shift_inverse<T>(h, m, nnz, d, val, off, col, mu0, x0, maxite, tol, mu, x, false);                   \
  }                                                                                                                 \
  VGPU_EXPORT cusolverStatus_t cusolverSp##P##csreigvsi(cusolverSpHandle_t h, int m, int nnz,                       \
                                                        const cusparseMatDescr_t d, const T* val, const int* off,   \
                                                        const int* col, T mu0, const T* x0, int maxite,             \
                                                        El<T>::R eps, T* mu, T* x) {                                \
    VGPU_REFUSE_CALL(sp_stream(h), CUSOLVER_STATUS_INTERNAL_ERROR, cusolverSp##P##csreigvsi, h, m, nnz, d, val,     \
                     off, col, mu0, x0, maxite, eps, mu, x);                                                        \
    return eig_shift_inverse<T>(h, m, nnz, d, val, off, col, mu0, x0, maxite, eps, mu, x, true);                    \
  }                                                                                                                 \
  VGPU_EXPORT cusolverStatus_t cusolverSp##P##csrzfdHost(cusolverSpHandle_t h, int n, int nnz,                      \
                                                         const cusparseMatDescr_t d, const T* val, const int* off,  \
                                                         const int* col, int* P_, int* numnz) {                     \
    return zero_free_diagonal<T>(h, n, nnz, d, val, off, col, P_, numnz);                                           \
  }                                                                                                                 \
  VGPU_EXPORT cusolverStatus_t cusolverSp##P##csrqrBufferInfoBatched(                                               \
      cusolverSpHandle_t h, int m, int n, int nnz, const cusparseMatDescr_t, const T*, const int*, const int*,      \
      int, csrqrInfo_t info, size_t* internal, size_t* work) {                                                      \
    return qr_buffer_info<T>(h, m, n, nnz, info, internal, work);                                                   \
  }                                                                                                                 \
  VGPU_EXPORT cusolverStatus_t cusolverSp##P##csrqrsvBatched(                                                       \
      cusolverSpHandle_t h, int m, int n, int nnz, const cusparseMatDescr_t d, const T* val, const int* off,        \
      const int* col, const T* b, T* x, int batch, csrqrInfo_t info, void* buf) {                                    \
    DescrScope descr_(d);                                                                                           \
    VGPU_DEFER_CALL(g_reg, sp_stream(h), cusolverSp##P##csrqrsvBatched, h, m, n, nnz, d, val, off, col, b, x,       \
                    batch, info, buf);                                                                              \
    return qr_solve_batched<T>(h, m, n, nnz, d, val, off, col, b, x, batch, info);                                  \
  }
VGPU_SP(S, float)
VGPU_SP(D, double)
VGPU_SP(C, cuComplex)
VGPU_SP(Z, cuDoubleComplex)
#undef VGPU_SP

VGPU_EXPORT cusolverStatus_t cusolverSpScsreigsHost(cusolverSpHandle_t h, int m, int nnz, const cusparseMatDescr_t d,
                                                    const float* val, const int* off, const int* col, cuComplex lb,
                                                    cuComplex ru, int* num) {
  return eig_count<float>(h, m, nnz, d, val, off, col, to_cd(lb), to_cd(ru), num);
}
VGPU_EXPORT cusolverStatus_t cusolverSpDcsreigsHost(cusolverSpHandle_t h, int m, int nnz, const cusparseMatDescr_t d,
                                                    const double* val, const int* off, const int* col,
                                                    cuDoubleComplex lb, cuDoubleComplex ru, int* num) {
  return eig_count<double>(h, m, nnz, d, val, off, col, to_cd(lb), to_cd(ru), num);
}
VGPU_EXPORT cusolverStatus_t cusolverSpCcsreigsHost(cusolverSpHandle_t h, int m, int nnz, const cusparseMatDescr_t d,
                                                    const cuComplex* val, const int* off, const int* col,
                                                    cuComplex lb, cuComplex ru, int* num) {
  return eig_count<cuComplex>(h, m, nnz, d, val, off, col, to_cd(lb), to_cd(ru), num);
}
VGPU_EXPORT cusolverStatus_t cusolverSpZcsreigsHost(cusolverSpHandle_t h, int m, int nnz, const cusparseMatDescr_t d,
                                                    const cuDoubleComplex* val, const int* off, const int* col,
                                                    cuDoubleComplex lb, cuDoubleComplex ru, int* num) {
  return eig_count<cuDoubleComplex>(h, m, nnz, d, val, off, col, to_cd(lb), to_cd(ru), num);
}

#include "cusolver_sp_lowlevel.inc"
