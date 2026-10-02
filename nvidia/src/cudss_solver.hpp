// The sparse direct solver behind libvgpucudss (cudss_api.cpp): host-side,
// deterministic, in double precision whatever the caller's value type.
//
//   ordering   minimum degree on the pattern of A + A^T, an exact-degree
//              quotient-graph elimination (ties go to the lowest index), or
//              the caller's permutation, or the natural order;
//   symbolic   elimination tree, postorder, column structures of L and
//              fundamental supernodes;
//   numeric    multifrontal: each supernode's front is assembled from the
//              matrix and its children's update matrices and partially
//              factored in dense arithmetic --
//                general            LU with threshold partial pivoting,
//                symmetric/Hermitian LDL^T (LDL^H) with 1x1 and 2x2 pivots,
//                SPD/HPD            Cholesky, no pivoting --
//              and a column with no acceptable pivot in its front is delayed
//              to the parent's, where it is fully summed with more of the
//              matrix. The root takes every remaining column: there, LU picks
//              the largest entry of the column and LDL^T falls back to
//              Bunch-Kaufman, so only a (numerically) singular matrix needs
//              its pivot replaced by the pivot epsilon;
//   solve      the five sub-phases cuDSS names (permute, forward, diagonal,
//              backward, permute back), and iterative refinement on top.
//
// Index spaces: "original" is the caller's row/column numbering; "permuted"
// is position in the analysed order, perm[k] being the original index of the
// k-th. Pivoting inside a front reorders rows (LU) or both (LDL^T) within the
// permuted space; the factor records the global indices it ended up with.
#ifndef VGPU_CUDSS_SOLVER_HPP
#define VGPU_CUDSS_SOLVER_HPP

#include <complex>
#include <cstdint>
#include <vector>

namespace vgpu_dss {

using cd = std::complex<double>;

enum class Kind { General, Symmetric, Hermitian, Spd, Hpd };

// Symmetric kinds store and factor one triangle; the conjugating ones use
// A = A^H rather than A = A^T. (For real values the two are the same.)
inline bool one_triangle(Kind k) { return k != Kind::General; }
inline bool conjugating(Kind k) { return k == Kind::Hermitian || k == Kind::Hpd; }
inline bool cholesky(Kind k) { return k == Kind::Spd || k == Kind::Hpd; }
inline bool indefinite(Kind k) { return k == Kind::Symmetric || k == Kind::Hermitian; }

// A square matrix's pattern in compressed columns, 0-based, rows sorted and
// duplicates merged. For the symmetric kinds only the lower triangle
// (row >= column) is kept.
struct Pattern {
  int64_t n = 0;
  std::vector<int64_t> colptr, rowind;
  bool operator==(const Pattern&) const = default;
};

// Which triangle of a symmetric matrix the caller's CSR carries.
enum class View { Full, Lower, Upper };

// The caller's CSR (0-based offsets already subtracted) turned into the
// canonical form above. rowend may be empty (three-array CSR). False when an
// offset or index is out of range.
template <class T>
bool canonical(Kind kind, View view, int64_t n, const std::vector<int64_t>& rowstart,
               const std::vector<int64_t>& rowend, const std::vector<int64_t>& colind,
               const std::vector<T>& vals, Pattern* pattern, std::vector<T>* values);

struct Symbolic {
  Kind kind = Kind::General;
  int64_t n = 0;
  std::vector<int64_t> perm, iperm;          // perm[k] = original index of pivot k
  std::vector<int64_t> sn_first;             // supernode s owns [sn_first[s], sn_first[s+1])
  std::vector<std::vector<int64_t>> sn_rows; // and these rows below its columns
  std::vector<int64_t> sn_parent;            // -1 at a root
  std::vector<std::vector<int64_t>> sn_children;
  int64_t nnz = 0;    // predicted nonzeros of the factors
  int64_t flops = 0;  // predicted multiply-adds of the factorization
  Pattern pattern;    // the pattern that was analysed
};

enum class Order { MinimumDegree, Natural, User };

// `user` (original indices, 0-based) is read when order == User; the caller
// has checked that it is a permutation.
Symbolic analyse(Kind kind, const Pattern& pattern, Order order, const std::vector<int64_t>& user);

struct Options {
  bool pivoting = true;        // false: CUDSS_PIVOT_NONE, pivots taken in order
  double eps = 1e-13;          // pivots smaller than this are replaced
  bool eps_scaled = false;     // ... scaled by the original row/column's largest entry
};

template <class T>
struct Factor {
  struct Front {
    int64_t m = 0, npiv = 0;
    std::vector<int64_t> rows, cols;  // permuted indices; the first npiv are the pivots
    std::vector<T> L;                 // m x npiv, column-major
    std::vector<T> U;                 // npiv x m, column-major (LU only)
    std::vector<T> D, Doff;           // LDL^T: pivot diagonal and, for a 2x2, its (2,1)
    std::vector<int8_t> bs;           // LDL^T: 1, or 2 then 0 for a 2x2 pair
  };
  Kind kind = Kind::General;
  int64_t n = 0;
  std::vector<Front> fronts;
  int64_t perturbed = 0;          // pivots replaced by the epsilon (CUDSS_DATA_NPIVOTS)
  int64_t positive = 0, negative = 0;  // inertia, symmetric/Hermitian only
  int info = 0;                   // 1-based position of the first non-positive Cholesky pivot
  int64_t nnz = 0;
  std::vector<T> diag;            // the pivot of each original row/column
  std::vector<int64_t> pivrow, pivcol;  // permuted row and column of pivot k
};

template <class T>
Factor<T> factorize(const Symbolic& sym, const Pattern& pattern, const std::vector<T>& values,
                    const Options& opt);

// The solve sub-phases on one right-hand side of length n. forward and
// diagonal work in place; backward reads w and writes z (which may alias w
// for the symmetric kinds only: LU's backward solve maps rows to columns).
template <class T> void permute(const Symbolic& s, const T* b, T* w);       // w[k] = b[perm[k]]
template <class T> void unpermute(const Symbolic& s, const T* z, T* x);     // x[perm[k]] = z[k]
template <class T> void forward(const Factor<T>& f, T* w);
template <class T> void diagonal(const Factor<T>& f, T* w);
template <class T> void backward(const Factor<T>& f, const T* w, T* z);

// y = A x, with A as the canonical pattern and values describe it.
template <class T>
void multiply(Kind kind, const Pattern& pattern, const std::vector<T>& values, const T* x, T* y);

}  // namespace vgpu_dss

#endif  // VGPU_CUDSS_SOLVER_HPP
