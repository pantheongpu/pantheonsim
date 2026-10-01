// The sparse direct solver behind libvgpucudss. See cudss_solver.hpp for the
// method; this file is plain host arithmetic with no CUDA in it.
#include "cudss_solver.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>
#include <type_traits>
#include <utility>

namespace vgpu_dss {
namespace {

double mag(double v) { return std::fabs(v); }
double mag(cd v) { return std::abs(v); }
double re(double v) { return v; }
double re(cd v) { return v.real(); }
double cj(double v, bool) { return v; }
cd cj(cd v, bool c) { return c ? std::conj(v) : v; }

// The pattern of A + A^T without the diagonal, as sorted adjacency lists.
std::vector<std::vector<int64_t>> adjacency(const Pattern& p) {
  const int64_t n = p.n;
  std::vector<int64_t> count(n, 0);
  for (int64_t j = 0; j < n; ++j)
    for (int64_t k = p.colptr[j]; k < p.colptr[j + 1]; ++k)
      if (p.rowind[k] != j) ++count[p.rowind[k]], ++count[j];
  std::vector<std::vector<int64_t>> adj(n);
  for (int64_t i = 0; i < n; ++i) adj[i].reserve(count[i]);
  for (int64_t j = 0; j < n; ++j)
    for (int64_t k = p.colptr[j]; k < p.colptr[j + 1]; ++k) {
      const int64_t i = p.rowind[k];
      if (i != j) adj[i].push_back(j), adj[j].push_back(i);
    }
  for (auto& a : adj) {
    std::sort(a.begin(), a.end());
    a.erase(std::unique(a.begin(), a.end()), a.end());
  }
  return adj;
}

// Minimum degree on the quotient graph. An eliminated pivot becomes an
// element: the clique of its uneliminated neighbours, which absorbs every
// element the pivot touched. A variable's degree is the exact size of the
// union of its variable neighbours and its elements' members. The pivot is
// always the lowest-numbered variable of least degree, so the order depends
// on the pattern alone.
std::vector<int64_t> minimum_degree(int64_t n, std::vector<std::vector<int64_t>> av) {
  std::vector<std::vector<int64_t>> ae(n), members(n);
  std::vector<char> eliminated(n, 0), absorbed(n, 0);
  std::vector<int64_t> degree(n), mark(n, -1), mark2(n, -1), order;
  order.reserve(n);
  int64_t stamp = 0, stamp2 = 0;
  std::set<std::pair<int64_t, int64_t>> queue;
  for (int64_t i = 0; i < n; ++i) {
    degree[i] = (int64_t)av[i].size();
    queue.insert({degree[i], i});
  }
  std::vector<int64_t> Lp;
  while (!queue.empty()) {
    const int64_t p = queue.begin()->second;
    queue.erase(queue.begin());
    eliminated[p] = 1;
    order.push_back(p);
    ++stamp;
    mark[p] = stamp;
    Lp.clear();
    for (int64_t v : av[p])
      if (!eliminated[v] && mark[v] != stamp) mark[v] = stamp, Lp.push_back(v);
    for (int64_t e : ae[p]) {
      if (absorbed[e]) continue;
      for (int64_t v : members[e])
        if (!eliminated[v] && mark[v] != stamp) mark[v] = stamp, Lp.push_back(v);
      absorbed[e] = 1;
      std::vector<int64_t>().swap(members[e]);
    }
    std::vector<int64_t>().swap(av[p]);
    std::vector<int64_t>().swap(ae[p]);
    members[p] = Lp;
    for (int64_t i : Lp) {
      // Neighbours now inside the new element are reached through it.
      auto& a = av[i];
      a.erase(std::remove_if(a.begin(), a.end(), [&](int64_t v) { return eliminated[v] || mark[v] == stamp; }),
              a.end());
      auto& E = ae[i];
      E.erase(std::remove_if(E.begin(), E.end(), [&](int64_t e) { return absorbed[e] != 0; }), E.end());
      E.push_back(p);
    }
    for (int64_t i : Lp) {
      ++stamp2;
      mark2[i] = stamp2;
      int64_t d = 0;
      for (int64_t v : av[i])
        if (mark2[v] != stamp2) mark2[v] = stamp2, ++d;
      for (int64_t e : ae[i])
        for (int64_t v : members[e])
          if (mark2[v] != stamp2) mark2[v] = stamp2, ++d;
      queue.erase({degree[i], i});
      degree[i] = d;
      queue.insert({d, i});
    }
  }
  return order;
}

// The elimination tree of a symmetric pattern given as permuted adjacency.
std::vector<int64_t> etree(int64_t n, const std::vector<std::vector<int64_t>>& padj) {
  std::vector<int64_t> parent(n, -1), ancestor(n, -1);
  for (int64_t k = 0; k < n; ++k)
    for (int64_t i : padj[k])
      while (i != -1 && i < k) {
        const int64_t next = ancestor[i];
        ancestor[i] = k;
        if (next == -1) parent[i] = k;
        i = next;
      }
  return parent;
}

// A postorder of the forest, children visited in increasing order.
std::vector<int64_t> postorder(const std::vector<int64_t>& parent) {
  const int64_t n = (int64_t)parent.size();
  std::vector<int64_t> head(n, -1), next(n, -1), post, stack;
  post.reserve(n);
  for (int64_t j = n - 1; j >= 0; --j)
    if (parent[j] != -1) next[j] = head[parent[j]], head[parent[j]] = j;
  for (int64_t r = 0; r < n; ++r) {
    if (parent[r] != -1) continue;
    stack.push_back(r);
    while (!stack.empty()) {
      const int64_t p = stack.back();
      const int64_t c = head[p];
      if (c == -1) {
        stack.pop_back();
        post.push_back(p);
      } else {
        head[p] = next[c];
        stack.push_back(c);
      }
    }
  }
  return post;
}

std::vector<std::vector<int64_t>> permuted(const std::vector<std::vector<int64_t>>& adj,
                                           const std::vector<int64_t>& perm,
                                           const std::vector<int64_t>& iperm) {
  std::vector<std::vector<int64_t>> padj(adj.size());
  for (size_t k = 0; k < adj.size(); ++k) {
    for (int64_t v : adj[perm[k]]) padj[k].push_back(iperm[v]);
    std::sort(padj[k].begin(), padj[k].end());
  }
  return padj;
}

// A front: the dense matrix of a supernode's fully summed variables (the
// first nfs positions) and the rows below them, column-major.
template <class T>
struct Work {
  int64_t m = 0;
  std::vector<T> F;
  std::vector<int64_t> rows, cols;
  T& at(int64_t a, int64_t b) { return F[(size_t)(a + b * m)]; }
  void swap_rows(int64_t a, int64_t b) {
    if (a == b) return;
    for (int64_t c = 0; c < m; ++c) std::swap(at(a, c), at(b, c));
    std::swap(rows[a], rows[b]);
  }
  void swap_cols(int64_t a, int64_t b) {
    if (a == b) return;
    for (int64_t r = 0; r < m; ++r) std::swap(at(r, a), at(r, b));
    std::swap(cols[a], cols[b]);
  }
};

// A pivot smaller than eps becomes eps with its sign (its phase, if complex;
// a zero becomes +eps).
template <class T>
bool perturb(T& p, double eps) {
  const double a = mag(p);
  if (!(a < eps)) return false;
  if (a == 0) p = T(eps);
  else if constexpr (std::is_same_v<T, double>) p = std::copysign(eps, p);
  else p *= eps / a;
  return true;
}

template <class T>
class Numeric {
 public:
  Numeric(const Symbolic& s, const Pattern& pat, const std::vector<T>& vals, const Options& o)
      : sym(s), opt(o), conj_(conjugating(s.kind)) {
    f.kind = s.kind;
    f.n = s.n;
    f.diag.assign((size_t)s.n, T(0));
    group_entries(pat, vals);
    if (opt.eps_scaled) {
      scale.assign((size_t)s.n, 0.0);
      for (int64_t j = 0; j < pat.n; ++j)
        for (int64_t k = pat.colptr[j]; k < pat.colptr[j + 1]; ++k) {
          const double a = mag(vals[k]);
          scale[pat.rowind[k]] = std::max(scale[pat.rowind[k]], a);
          scale[j] = std::max(scale[j], a);
        }
    }
  }

  Factor<T> run() {
    const int64_t ns = (int64_t)sym.sn_first.size() - 1;
    rpos.assign((size_t)sym.n, 0);
    cpos.assign((size_t)sym.n, 0);
    cbs.assign((size_t)std::max<int64_t>(ns, 0), Update{});
    for (int64_t s = 0; s < ns; ++s) front(s);
    return std::move(f);
  }

 private:
  struct Entry {
    int64_t r, c;
    T v;
  };
  // A child's update matrix, its first ndelay rows/columns delayed pivots.
  struct Update {
    int64_t m = 0, ndelay = 0;
    std::vector<int64_t> rows, cols;
    std::vector<T> F;
  };

  const Symbolic& sym;
  Options opt;
  bool conj_;
  Factor<T> f;
  std::vector<int64_t> owner_ptr;
  std::vector<Entry> entries;
  std::vector<double> scale;
  std::vector<int64_t> rpos, cpos;
  std::vector<Update> cbs;
  int64_t eliminated = 0;

  // The matrix in permuted coordinates, each entry filed under the column
  // that is eliminated first of its row and column: that column's front holds
  // both. Symmetric kinds keep the lower triangle.
  void group_entries(const Pattern& pat, const std::vector<T>& vals) {
    const int64_t n = sym.n;
    std::vector<Entry> raw;
    raw.reserve(pat.rowind.size());
    for (int64_t j = 0; j < n; ++j)
      for (int64_t k = pat.colptr[j]; k < pat.colptr[j + 1]; ++k) {
        const int64_t pi = sym.iperm[pat.rowind[k]], pj = sym.iperm[j];
        if (!one_triangle(sym.kind) || pi >= pj) raw.push_back({pi, pj, vals[k]});
        else raw.push_back({pj, pi, cj(vals[k], conj_)});
      }
    owner_ptr.assign((size_t)n + 1, 0);
    for (const Entry& e : raw) ++owner_ptr[(size_t)std::min(e.r, e.c) + 1];
    for (int64_t j = 0; j < n; ++j) owner_ptr[j + 1] += owner_ptr[j];
    entries.resize(raw.size());
    std::vector<int64_t> next(owner_ptr.begin(), owner_ptr.end() - 1);
    for (const Entry& e : raw) entries[(size_t)next[(size_t)std::min(e.r, e.c)]++] = e;
  }

  double eps_for(const Work<T>& w, int64_t k) const {
    if (!opt.eps_scaled) return opt.eps;
    const double s = std::max(scale[sym.perm[w.rows[k]]], scale[sym.perm[w.cols[k]]]);
    return s > 0 ? opt.eps * s : opt.eps;
  }

  void front(int64_t s) {
    Work<T> w;
    for (int64_t ch : sym.sn_children[s]) {
      const Update& u = cbs[ch];
      for (int64_t a = 0; a < u.ndelay; ++a) w.rows.push_back(u.rows[a]), w.cols.push_back(u.cols[a]);
    }
    for (int64_t j = sym.sn_first[s]; j < sym.sn_first[s + 1]; ++j) w.rows.push_back(j), w.cols.push_back(j);
    const int64_t nfs = (int64_t)w.rows.size();
    for (int64_t r : sym.sn_rows[s]) w.rows.push_back(r), w.cols.push_back(r);
    const int64_t m = w.m = (int64_t)w.rows.size();
    for (int64_t a = 0; a < m; ++a) rpos[w.rows[a]] = a, cpos[w.cols[a]] = a;
    w.F.assign((size_t)(m * m), T(0));

    for (int64_t j = sym.sn_first[s]; j < sym.sn_first[s + 1]; ++j)
      for (int64_t k = owner_ptr[j]; k < owner_ptr[j + 1]; ++k) {
        const Entry& e = entries[k];
        if (one_triangle(sym.kind)) {
          w.at(rpos[e.r], rpos[e.c]) += e.v;
          if (e.r != e.c) w.at(rpos[e.c], rpos[e.r]) += cj(e.v, conj_);
        } else {
          w.at(rpos[e.r], cpos[e.c]) += e.v;
        }
      }
    for (int64_t ch : sym.sn_children[s]) {
      Update& u = cbs[ch];
      for (int64_t b = 0; b < u.m; ++b) {
        const int64_t col = cpos[u.cols[b]];
        for (int64_t a = 0; a < u.m; ++a) w.at(rpos[u.rows[a]], col) += u.F[(size_t)(a + b * u.m)];
      }
      u = Update{};
    }

    const bool root = sym.sn_parent[s] == -1;
    typename Factor<T>::Front fr;
    int64_t npiv;
    if (cholesky(sym.kind)) npiv = factor_cholesky(w, nfs);
    else if (indefinite(sym.kind)) npiv = factor_ldlt(w, nfs, root, fr);
    else npiv = factor_lu(w, nfs, root);
    store(w, npiv, fr);
    if (npiv < m) {
      Update& u = cbs[s];
      u.m = m - npiv;
      u.ndelay = nfs - npiv;
      u.rows.assign(w.rows.begin() + npiv, w.rows.end());
      u.cols.assign(w.cols.begin() + npiv, w.cols.end());
      u.F.resize((size_t)(u.m * u.m));
      for (int64_t b = 0; b < u.m; ++b)
        for (int64_t a = 0; a < u.m; ++a) u.F[(size_t)(a + b * u.m)] = w.at(npiv + a, npiv + b);
    }
  }

  // ---- LU with threshold partial pivoting ----
  int64_t factor_lu(Work<T>& w, int64_t nfs, bool root) {
    const double u = 0.1;  // a pivot must be at least u times its column's largest entry
    const int64_t m = w.m;
    int64_t k = 0;
    while (k < nfs) {
      int64_t pr = -1, pc = -1;
      if (!opt.pivoting) {
        pr = pc = k;
      } else {
        for (int64_t c = k; c < nfs && pr < 0; ++c) {
          double colmax = 0;
          for (int64_t a = k; a < m; ++a) colmax = std::max(colmax, mag(w.at(a, c)));
          if (colmax == 0) continue;
          if (mag(w.at(c, c)) >= u * colmax) {  // the diagonal keeps the pattern symmetric
            pr = pc = c;
            break;
          }
          int64_t r = -1;
          double best = 0;
          for (int64_t a = k; a < nfs; ++a)
            if (mag(w.at(a, c)) > best) best = mag(w.at(a, c)), r = a;
          if (r >= 0 && best >= u * colmax) pr = r, pc = c;
        }
        if (pr < 0) {
          if (!root) break;  // delay the rest to the parent
          pr = pc = k;       // what is left is zero
        }
      }
      w.swap_rows(pr, k);
      w.swap_cols(pc, k);
      T p = w.at(k, k);
      if (perturb(p, eps_for(w, k))) ++f.perturbed;
      w.at(k, k) = p;
      for (int64_t a = k + 1; a < m; ++a) w.at(a, k) /= p;
      for (int64_t b = k + 1; b < m; ++b) {
        const T x = w.at(k, b);
        if (x == T(0)) continue;
        for (int64_t a = k + 1; a < m; ++a) w.at(a, b) -= w.at(a, k) * x;
      }
      f.diag[sym.perm[w.cols[k]]] = p;
      ++k;
    }
    return k;
  }

  // ---- LDL^T with 1x1 and 2x2 pivots ----
  //
  // Away from the root a pivot must pass the threshold test against every
  // entry of its column, the rows below the supernode included (u = 0.01, as
  // in MA57): a 1x1 pivot d needs |d| >= u * max|column|, a 2x2 pivot P
  // needs |P^-1| times the columns' largest other entries <= 1/u. A column
  // with no such pivot waits for the parent. At the root every row is fully
  // summed and Bunch-Kaufman's choice always succeeds.
  int64_t factor_ldlt(Work<T>& w, int64_t nfs, bool root, typename Factor<T>::Front& fr) {
    const double u = 0.01, alpha = (1 + std::sqrt(17.0)) / 8;
    const int64_t m = w.m;
    fr.D.assign((size_t)nfs, T(0));
    fr.Doff.assign((size_t)nfs, T(0));
    fr.bs.assign((size_t)nfs, 1);
    auto colmax = [&](int64_t c, int64_t skip1, int64_t skip2, int64_t* arg, int64_t limit) {
      double best = 0;
      for (int64_t a = k_; a < limit; ++a) {
        if (a == skip1 || a == skip2) continue;
        const double v = mag(w.at(a, c));
        if (v > best) {
          best = v;
          if (arg) *arg = a;
        }
      }
      return best;
    };
    k_ = 0;
    while (k_ < nfs) {
      const int64_t k = k_;
      int64_t p1 = -1, p2 = -1;
      if (!opt.pivoting) {
        p1 = k;
      } else if (root) {
        int64_t r = -1;
        const double lam = colmax(k, k, -1, &r, m);
        const double d = mag(w.at(k, k));
        if (lam == 0 || d >= alpha * lam) {
          p1 = k;
        } else {
          const double sigma = colmax(r, r, -1, nullptr, m);
          if (d * sigma >= alpha * lam * lam) p1 = k;
          else if (mag(w.at(r, r)) >= alpha * sigma) p1 = r;
          else p1 = k, p2 = r;
        }
      } else {
        for (int64_t c = k; c < nfs && p1 < 0; ++c) {
          const double lam = colmax(c, c, -1, nullptr, m);
          const double dcc = mag(w.at(c, c));
          if (lam == 0 && dcc == 0) continue;
          if (dcc >= u * lam) {
            p1 = c;
            break;
          }
          int64_t r = -1;
          colmax(c, c, -1, &r, nfs);
          if (r < 0) continue;
          const T d11 = w.at(c, c), d21 = w.at(r, c), d12 = w.at(c, r), d22 = w.at(r, r);
          const double det = mag(d11 * d22 - d12 * d21);
          if (det > 0) {
            const double lc = colmax(c, c, r, nullptr, m), lr = colmax(r, c, r, nullptr, m);
            if ((mag(d22) * lc + mag(d12) * lr) * u <= det && (mag(d21) * lc + mag(d11) * lr) * u <= det) {
              p1 = c, p2 = r;
              break;
            }
          }
          const double lamr = colmax(r, r, -1, nullptr, m);
          if (mag(w.at(r, r)) > 0 && mag(w.at(r, r)) >= u * lamr) p1 = r;
        }
        if (p1 < 0) break;  // delay the rest to the parent
      }
      if (p2 < 0) {
        symmetric_swap(w, p1, k);
        eliminate1(w, k, fr);
        k_ = k + 1;
      } else {
        symmetric_swap(w, p1, k);
        if (p2 == k) p2 = p1;
        symmetric_swap(w, p2, k + 1);
        eliminate2(w, k, fr);
        k_ = k + 2;
      }
    }
    return k_;
  }
  int64_t k_ = 0;

  void symmetric_swap(Work<T>& w, int64_t a, int64_t b) {
    w.swap_rows(a, b);
    w.swap_cols(a, b);
  }

  void count_inertia(double d) {
    if (d > 0) ++f.positive;
    else if (d < 0) ++f.negative;
  }

  void eliminate1(Work<T>& w, int64_t k, typename Factor<T>::Front& fr) {
    const int64_t m = w.m;
    T d = w.at(k, k);
    if (conj_) d = T(re(d));
    if (perturb(d, eps_for(w, k))) ++f.perturbed;
    w.at(k, k) = d;
    std::vector<T> l((size_t)m);
    for (int64_t a = k + 1; a < m; ++a) l[a] = w.at(a, k) / d;
    for (int64_t b = k + 1; b < m; ++b) {
      const T x = w.at(k, b);
      if (x == T(0)) continue;
      for (int64_t a = k + 1; a < m; ++a) w.at(a, b) -= l[a] * x;
    }
    for (int64_t a = k + 1; a < m; ++a) w.at(a, k) = l[a];
    fr.D[k] = d;
    fr.bs[k] = 1;
    count_inertia(re(d));
    f.diag[sym.perm[w.rows[k]]] = d;
  }

  void eliminate2(Work<T>& w, int64_t k, typename Factor<T>::Front& fr) {
    const int64_t m = w.m;
    T d11 = w.at(k, k), d22 = w.at(k + 1, k + 1);
    const T d21 = w.at(k + 1, k), d12 = w.at(k, k + 1);
    if (conj_) d11 = T(re(d11)), d22 = T(re(d22));
    const T det = d11 * d22 - d12 * d21;
    const T i00 = d22 / det, i01 = -d12 / det, i10 = -d21 / det, i11 = d11 / det;
    std::vector<T> l0((size_t)m), l1((size_t)m);
    for (int64_t a = k + 2; a < m; ++a) {
      const T x0 = w.at(a, k), x1 = w.at(a, k + 1);
      l0[a] = x0 * i00 + x1 * i10;
      l1[a] = x0 * i01 + x1 * i11;
    }
    for (int64_t b = k + 2; b < m; ++b) {
      const T f0 = w.at(k, b), f1 = w.at(k + 1, b);
      if (f0 == T(0) && f1 == T(0)) continue;
      for (int64_t a = k + 2; a < m; ++a) w.at(a, b) -= l0[a] * f0 + l1[a] * f1;
    }
    for (int64_t a = k + 2; a < m; ++a) w.at(a, k) = l0[a], w.at(a, k + 1) = l1[a];
    fr.D[k] = d11;
    fr.D[k + 1] = d22;
    fr.Doff[k] = d21;
    fr.bs[k] = 2;
    fr.bs[k + 1] = 0;
    // A 2x2 block of a real symmetric or Hermitian matrix with a negative
    // determinant has one eigenvalue of each sign; otherwise both share the
    // trace's.
    const double rdet = re(det);
    if (rdet < 0) ++f.positive, ++f.negative;
    else if (re(d11) + re(d22) > 0) f.positive += 2;
    else f.negative += 2;
    f.diag[sym.perm[w.rows[k]]] = d11;
    f.diag[sym.perm[w.rows[k + 1]]] = d22;
  }

  // ---- Cholesky ----
  //
  // A non-positive pivot makes the matrix not positive definite: INFO gets its
  // 1-based position in the elimination order (cuDSS's "first non-positive
  // minor") and the factorization carries on with |pivot|, as the hardware
  // carries on, so the results that follow mean nothing.
  int64_t factor_cholesky(Work<T>& w, int64_t nfs) {
    const int64_t m = w.m;
    std::vector<T> c((size_t)m);
    for (int64_t k = 0; k < nfs; ++k) {
      T d = w.at(k, k);
      if (conj_ || std::is_same_v<T, double>) d = T(re(d));
      if (!(re(d) > 0)) {
        if (!f.info) f.info = (int)std::min<int64_t>(eliminated + k + 1, INT32_MAX);
        const double a = mag(d);
        d = T(a > 0 && std::isfinite(a) ? a : (opt.eps > 0 ? opt.eps : 1.0));
      }
      const T l = std::sqrt(d);
      w.at(k, k) = l;
      for (int64_t a = k + 1; a < m; ++a) c[a] = w.at(a, k) / l;
      for (int64_t b = k + 1; b < m; ++b) {
        const T x = cj(c[b], conj_);
        if (x == T(0)) continue;
        for (int64_t a = k + 1; a < m; ++a) w.at(a, b) -= c[a] * x;
      }
      for (int64_t a = k + 1; a < m; ++a) w.at(a, k) = c[a];
      f.diag[sym.perm[w.rows[k]]] = l;
    }
    return nfs;
  }

  // The eliminated part of the front, as the solve phases read it.
  void store(Work<T>& w, int64_t npiv, typename Factor<T>::Front& fr) {
    const int64_t m = w.m;
    fr.m = m;
    fr.npiv = npiv;
    fr.rows = w.rows;
    fr.cols = w.cols;
    fr.L.assign((size_t)(m * npiv), T(0));
    const bool chol = cholesky(sym.kind);
    for (int64_t t = 0; t < npiv; ++t) {
      fr.L[(size_t)(t + t * m)] = chol ? w.at(t, t) : T(1);
      for (int64_t a = t + 1; a < m; ++a) fr.L[(size_t)(a + t * m)] = w.at(a, t);
    }
    if (indefinite(sym.kind)) {
      fr.D.resize((size_t)npiv);
      fr.Doff.resize((size_t)npiv);
      fr.bs.resize((size_t)npiv);
      for (int64_t t = 0; t + 1 < npiv; ++t)
        if (fr.bs[t] == 2) fr.L[(size_t)(t + 1 + t * m)] = T(0);  // the 2x2 block's (2,1) is in D
    }
    if (sym.kind == Kind::General) {
      fr.U.assign((size_t)(npiv * m), T(0));
      for (int64_t b = 0; b < m; ++b)
        for (int64_t t = 0; t < std::min(npiv, b + 1); ++t) fr.U[(size_t)(t + b * npiv)] = w.at(t, b);
    }
    for (int64_t t = 0; t < npiv; ++t) {
      f.pivrow.push_back(w.rows[t]);
      f.pivcol.push_back(w.cols[t]);
    }
    const int64_t tri = npiv * m - npiv * (npiv - 1) / 2;  // a trapezoid with its diagonal
    f.nnz += sym.kind == Kind::General ? 2 * tri - npiv : tri;
    eliminated += npiv;
    f.fronts.push_back(std::move(fr));
  }
};

}  // namespace

template <class T>
bool canonical(Kind kind, View view, int64_t n, const std::vector<int64_t>& rowstart,
               const std::vector<int64_t>& rowend, const std::vector<int64_t>& colind,
               const std::vector<T>& vals, Pattern* pattern, std::vector<T>* values) {
  const int64_t nnz = (int64_t)colind.size();
  const bool have = !vals.empty();
  const bool c = conjugating(kind);
  struct Item {
    int64_t r, c;
    T v;
  };
  std::vector<Item> items;
  items.reserve((size_t)nnz);
  for (int64_t r = 0; r < n; ++r) {
    const int64_t start = rowstart[r], end = rowend.empty() ? rowstart[r + 1] : rowend[r];
    if (start < 0 || end < start || end > nnz) return false;
    for (int64_t k = start; k < end; ++k) {
      const int64_t col = colind[k];
      if (col < 0 || col >= n) return false;
      const T v = have ? vals[k] : T(0);
      if (!one_triangle(kind)) items.push_back({r, col, v});
      else if (view == View::Upper) {
        if (col >= r) items.push_back({col, r, cj(v, c)});
      } else if (col <= r) {
        items.push_back({r, col, v});
      }
    }
  }
  std::stable_sort(items.begin(), items.end(),
                   [](const Item& a, const Item& b) { return a.c != b.c ? a.c < b.c : a.r < b.r; });
  pattern->n = n;
  pattern->colptr.assign((size_t)n + 1, 0);
  pattern->rowind.clear();
  values->clear();
  for (size_t i = 0; i < items.size(); ++i) {
    if (i && items[i].r == items[i - 1].r && items[i].c == items[i - 1].c) {
      if (have) values->back() += items[i].v;  // repeated entries add up
      continue;
    }
    pattern->rowind.push_back(items[i].r);
    if (have) values->push_back(items[i].v);
    ++pattern->colptr[(size_t)items[i].c + 1];
  }
  for (int64_t j = 0; j < n; ++j) pattern->colptr[j + 1] += pattern->colptr[j];
  return true;
}

Symbolic analyse(Kind kind, const Pattern& pattern, Order order, const std::vector<int64_t>& user) {
  Symbolic s;
  s.kind = kind;
  s.n = pattern.n;
  s.pattern = pattern;
  const int64_t n = pattern.n;
  const auto adj = adjacency(pattern);
  std::vector<int64_t> first;
  if (order == Order::User) first = user;
  else if (order == Order::Natural) first.resize(n), std::iota(first.begin(), first.end(), 0);
  else first = minimum_degree(n, adj);
  std::vector<int64_t> ifirst(n);
  for (int64_t k = 0; k < n; ++k) ifirst[first[k]] = k;

  // Postorder the minimum-degree elimination tree, which leaves the fill
  // alone and makes more chains of columns into supernodes. A given order
  // (the caller's, or the natural one) is kept exactly as given: any order
  // works, since a column's parent always comes after it.
  std::vector<int64_t> post(n);
  if (order == Order::MinimumDegree) post = postorder(etree(n, permuted(adj, first, ifirst)));
  else std::iota(post.begin(), post.end(), 0);
  s.perm.resize(n);
  s.iperm.resize(n);
  for (int64_t k = 0; k < n; ++k) s.perm[k] = first[post[k]];
  for (int64_t k = 0; k < n; ++k) s.iperm[s.perm[k]] = k;
  const auto padj = permuted(adj, s.perm, s.iperm);
  const auto parent = etree(n, padj);

  // Column structures of L: column k's own entries below the diagonal, and
  // its children's structures less k itself.
  std::vector<std::vector<int64_t>> children(n), st(n);
  for (int64_t k = 0; k < n; ++k)
    if (parent[k] != -1) children[parent[k]].push_back(k);
  std::vector<int64_t> mark(n, -1), count(n);
  for (int64_t k = 0; k < n; ++k) {
    mark[k] = k;
    auto& cur = st[k];
    for (int64_t i : padj[k])
      if (i > k && mark[i] != k) mark[i] = k, cur.push_back(i);
    for (int64_t c : children[k]) {
      for (int64_t i : st[c])
        if (i != k && mark[i] != k) mark[i] = k, cur.push_back(i);
    }
    std::sort(cur.begin(), cur.end());
    count[k] = (int64_t)cur.size();
  }

  // Fundamental supernodes: k joins k-1's when it is k-1's parent, has no
  // other child, and their structures nest.
  s.sn_first.push_back(0);
  for (int64_t k = 1; k < n; ++k)
    if (!(parent[k - 1] == k && children[k].size() == 1 && count[k - 1] == count[k] + 1))
      s.sn_first.push_back(k);
  if (n) s.sn_first.push_back(n);
  const int64_t ns = (int64_t)s.sn_first.size() - 1;
  std::vector<int64_t> sn_of(n);
  for (int64_t sn = 0; sn < ns; ++sn)
    for (int64_t k = s.sn_first[sn]; k < s.sn_first[sn + 1]; ++k) sn_of[k] = sn;
  s.sn_rows.resize(std::max<int64_t>(ns, 0));
  s.sn_parent.assign(std::max<int64_t>(ns, 0), -1);
  s.sn_children.resize(std::max<int64_t>(ns, 0));
  for (int64_t sn = 0; sn < ns; ++sn) {
    const int64_t last = s.sn_first[sn + 1] - 1;
    s.sn_rows[sn] = st[last];
    if (parent[last] != -1) {
      s.sn_parent[sn] = sn_of[parent[last]];
      s.sn_children[s.sn_parent[sn]].push_back(sn);
    }
  }
  for (int64_t k = 0; k < n; ++k) {
    s.nnz += (kind == Kind::General ? 2 * count[k] : count[k]) + 1;
    s.flops += count[k] * (count[k] + 1);
  }
  return s;
}

template <class T>
Factor<T> factorize(const Symbolic& sym, const Pattern& pattern, const std::vector<T>& values,
                    const Options& opt) {
  return Numeric<T>(sym, pattern, values, opt).run();
}

template <class T>
void permute(const Symbolic& s, const T* b, T* w) {
  for (int64_t k = 0; k < s.n; ++k) w[k] = b[s.perm[k]];
}

template <class T>
void unpermute(const Symbolic& s, const T* z, T* x) {
  for (int64_t k = 0; k < s.n; ++k) x[s.perm[k]] = z[k];
}

template <class T>
void forward(const Factor<T>& f, T* w) {
  const bool chol = cholesky(f.kind);
  for (const auto& fr : f.fronts)
    for (int64_t t = 0; t < fr.npiv; ++t) {
      T y = w[fr.rows[t]];
      if (chol) w[fr.rows[t]] = y = y / fr.L[(size_t)(t + t * fr.m)];
      if (y == T(0)) continue;
      for (int64_t a = t + 1; a < fr.m; ++a) w[fr.rows[a]] -= fr.L[(size_t)(a + t * fr.m)] * y;
    }
}

template <class T>
void diagonal(const Factor<T>& f, T* w) {
  if (!indefinite(f.kind)) return;
  const bool c = conjugating(f.kind);
  for (const auto& fr : f.fronts)
    for (int64_t t = 0; t < fr.npiv;) {
      if (fr.bs[t] == 1) {
        w[fr.rows[t]] /= fr.D[t];
        ++t;
        continue;
      }
      const T d11 = fr.D[t], d22 = fr.D[t + 1], d21 = fr.Doff[t], d12 = cj(d21, c);
      const T det = d11 * d22 - d12 * d21;
      const T x0 = w[fr.rows[t]], x1 = w[fr.rows[t + 1]];
      w[fr.rows[t]] = (d22 * x0 - d12 * x1) / det;
      w[fr.rows[t + 1]] = (d11 * x1 - d21 * x0) / det;
      t += 2;
    }
}

template <class T>
void backward(const Factor<T>& f, const T* w, T* z) {
  if (f.kind == Kind::General) {
    for (auto it = f.fronts.rbegin(); it != f.fronts.rend(); ++it) {
      const auto& fr = *it;
      for (int64_t t = fr.npiv - 1; t >= 0; --t) {
        T s = w[fr.rows[t]];
        for (int64_t b = t + 1; b < fr.m; ++b) s -= fr.U[(size_t)(t + b * fr.npiv)] * z[fr.cols[b]];
        z[fr.cols[t]] = s / fr.U[(size_t)(t + t * fr.npiv)];
      }
    }
    return;
  }
  const bool c = conjugating(f.kind), chol = cholesky(f.kind);
  if (z != w) std::copy(w, w + f.n, z);
  for (auto it = f.fronts.rbegin(); it != f.fronts.rend(); ++it) {
    const auto& fr = *it;
    for (int64_t t = fr.npiv - 1; t >= 0; --t) {
      T s = z[fr.rows[t]];
      for (int64_t a = t + 1; a < fr.m; ++a) s -= cj(fr.L[(size_t)(a + t * fr.m)], c) * z[fr.rows[a]];
      if (chol) s /= cj(fr.L[(size_t)(t + t * fr.m)], c);
      z[fr.rows[t]] = s;
    }
  }
}

template <class T>
void multiply(Kind kind, const Pattern& p, const std::vector<T>& values, const T* x, T* y) {
  const bool c = conjugating(kind);
  std::fill(y, y + p.n, T(0));
  for (int64_t j = 0; j < p.n; ++j)
    for (int64_t k = p.colptr[j]; k < p.colptr[j + 1]; ++k) {
      const int64_t i = p.rowind[k];
      y[i] += values[k] * x[j];
      if (one_triangle(kind) && i != j) y[j] += cj(values[k], c) * x[i];
    }
}

#define VGPU_DSS_INSTANTIATE(T)                                                                        \
  template bool canonical<T>(Kind, View, int64_t, const std::vector<int64_t>&,                         \
                             const std::vector<int64_t>&, const std::vector<int64_t>&,                 \
                             const std::vector<T>&, Pattern*, std::vector<T>*);                        \
  template Factor<T> factorize<T>(const Symbolic&, const Pattern&, const std::vector<T>&,              \
                                  const Options&);                                                     \
  template void permute<T>(const Symbolic&, const T*, T*);                                             \
  template void unpermute<T>(const Symbolic&, const T*, T*);                                           \
  template void forward<T>(const Factor<T>&, T*);                                                      \
  template void diagonal<T>(const Factor<T>&, T*);                                                     \
  template void backward<T>(const Factor<T>&, const T*, T*);                                           \
  template void multiply<T>(Kind, const Pattern&, const std::vector<T>&, const T*, T*);

VGPU_DSS_INSTANTIATE(double)
VGPU_DSS_INSTANTIATE(cd)

}  // namespace vgpu_dss
