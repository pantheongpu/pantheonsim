#pragma once
// The one-dimensional transform behind the host cuFFT, with the arithmetic of
// the recursive mixed-radix Cooley-Tukey it replaces and none of its overhead.
//
// The recursion split n on its smallest prime factor p (m = n/p), transformed
// the p interleaved subsequences, and combined them: for q < p, k < m,
//
//     out[q*m + k] = ((t[k] + t[m+k]*w(1,k+q*m)) + t[2m+k]*w(2,k+q*m)) + ...
//
// with w(j,i) = (cos a, sin a), a = ((sign*2*pi/n) * j) * i. Every output
// there is that one sequence of complex products and sums. This file computes
// the same sequences bottom-up (the same splits, in the same order, with the
// same complex operations), but looks each twiddle up in a table that is
// computed once per size instead of calling cos and sin for every use, runs
// over arrays instead of recursing down to single points, and allocates
// nothing per transform. The numbers are the numbers the recursion gave.
// (tests/unit/test_host_fft.cpp keeps the recursion as the reference.)
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace vgpu::hostfft {

using cd = std::complex<double>;

inline int smallest_factor(int n) {
  for (int p = 2; static_cast<long long>(p) * p <= n; ++p)
    if (n % p == 0) return p;
  return n;  // prime
}

struct Level {
  int p = 0;                                  // the factor split off here
  int n = 0;                                  // the size of the transforms at this level
  int m = 0;                                  // n / p
  std::shared_ptr<const std::vector<cd>> tw;  // n x (p-1): w(j, i) at [i*(p-1) + (j-1)], so a sum's twiddles are adjacent; null when too big to keep
  double base = 0;                            // sign * 2 * pi / n, for the twiddles computed on the fly
};

struct Plan {
  int n = 0;
  std::vector<Level> levels;   // top level first
  std::vector<uint32_t> perm;  // perm[position of a leaf] = index of the input sample it starts from
};

namespace detail {
// Twiddle tables are shared by every plan of the same size and sign, and kept up to a total bound.
constexpr size_t kMaxTableEntries = size_t(1) << 21;   // one table (32 MB of cd)
constexpr size_t kMaxCacheEntries = size_t(1) << 23;   // all tables (128 MB)

struct Cache {
  std::mutex mu;
  std::map<std::pair<int, int>, std::shared_ptr<const std::vector<cd>>> tables;
  std::map<std::pair<int, int>, std::shared_ptr<const Plan>> plans;
  size_t entries = 0;
};
inline Cache& cache() {
  static Cache c;
  return c;
}

inline std::shared_ptr<const std::vector<cd>> make_table(int n, int p, int sign) {
  auto t = std::make_shared<std::vector<cd>>(static_cast<size_t>(p - 1) * static_cast<size_t>(n));
  const double base = sign * 2.0 * M_PI / n;
  for (int i = 0; i < n; ++i)
    for (int j = 1; j < p; ++j) {
      const double ang = base * static_cast<double>(j) * static_cast<double>(i);
      (*t)[static_cast<size_t>(i) * static_cast<size_t>(p - 1) + static_cast<size_t>(j - 1)] = cd(std::cos(ang), std::sin(ang));
    }
  return t;
}
}  // namespace detail

// The plan for transforms of size n and the given sign (+1 or -1, as cuFFT's direction).
inline std::shared_ptr<const Plan> plan_1d(int n, int sign) {
  detail::Cache& c = detail::cache();
  {
    std::lock_guard<std::mutex> l(c.mu);
    auto it = c.plans.find({n, sign});
    if (it != c.plans.end()) return it->second;
  }
  auto pl = std::make_shared<Plan>();
  pl->n = n;
  // The splits: n, n/p0, n/(p0*p1), ... down to 1.
  for (int cur = n; cur > 1;) {
    Level lv;
    lv.p = smallest_factor(cur);
    lv.n = cur;
    lv.m = cur / lv.p;
    lv.base = sign * 2.0 * M_PI / cur;
    pl->levels.push_back(lv);
    cur = lv.m;
  }
  // Where each leaf sits and which input sample it is: digit j of level d moves the position by j * (size below)
  // and the sample index by j * (the product of the factors above).
  std::vector<std::pair<uint32_t, uint32_t>> cur{{0u, 0u}};
  uint64_t stride = 1;
  for (const Level& lv : pl->levels) {
    std::vector<std::pair<uint32_t, uint32_t>> next;
    next.reserve(cur.size() * static_cast<size_t>(lv.p));
    for (const auto& e : cur)
      for (int j = 0; j < lv.p; ++j)
        next.push_back({e.first + static_cast<uint32_t>(j) * static_cast<uint32_t>(lv.m),
                        e.second + static_cast<uint32_t>(j * stride)});
    cur.swap(next);
    stride *= static_cast<uint64_t>(lv.p);
  }
  pl->perm.assign(static_cast<size_t>(n), 0);
  for (const auto& e : cur) pl->perm[e.first] = e.second;
  for (Level& lv : pl->levels) {
    const size_t entries = static_cast<size_t>(lv.p - 1) * static_cast<size_t>(lv.n);
    if (entries > detail::kMaxTableEntries) continue;
    {
      std::lock_guard<std::mutex> l(c.mu);
      auto it = c.tables.find({lv.n, sign});
      if (it != c.tables.end()) {
        lv.tw = it->second;
        continue;
      }
    }
    auto t = detail::make_table(lv.n, lv.p, sign);   // (outside the lock: it is the slow part)
    std::lock_guard<std::mutex> l(c.mu);
    if (c.entries + entries > detail::kMaxCacheEntries) {   // over the bound: forget the others, keep this one going
      c.tables.clear();
      c.plans.clear();
      c.entries = 0;
    }
    auto ins = c.tables.emplace(std::make_pair(lv.n, sign), t);
    if (ins.second) c.entries += entries;
    lv.tw = ins.first->second;
  }
  std::lock_guard<std::mutex> l(c.mu);
  c.plans[{n, sign}] = pl;
  return pl;
}

// Transforms data[0 .. n) in place. scratch holds 2n values.
inline void run(const Plan& pl, cd* data, cd* scratch) {
  const int n = pl.n;
  if (n <= 1) return;
  cd* a = scratch;
  cd* b = scratch + n;
  for (int pos = 0; pos < n; ++pos) a[pos] = data[pl.perm[static_cast<size_t>(pos)]];
  // Bottom level first: each level's blocks of n values combine their p children (blocks of m).
  for (size_t d = pl.levels.size(); d-- > 0;) {
    const Level& lv = pl.levels[d];
    const int p = lv.p, m = lv.m, nn = lv.n;
    cd* const dst = d == 0 ? data : b;
    for (int blk = 0; blk < n; blk += nn) {
      const cd* src = a + blk;
      cd* out = dst + blk;
      for (int q = 0; q < p; ++q)
        for (int k = 0; k < m; ++k) {
          cd acc = src[k];   // j = 0 has a unit twiddle
          for (int j = 1; j < p; ++j) {
            const int i = k + q * m;
            const cd w = lv.tw ? (*lv.tw)[static_cast<size_t>(i) * static_cast<size_t>(p - 1) + static_cast<size_t>(j - 1)]
                               : [&] {
                                   const double ang = lv.base * static_cast<double>(j) * static_cast<double>(i);
                                   return cd(std::cos(ang), std::sin(ang));
                                 }();
            acc += src[static_cast<size_t>(j) * static_cast<size_t>(m) + static_cast<size_t>(k)] * w;
          }
          out[static_cast<size_t>(q) * static_cast<size_t>(m) + static_cast<size_t>(k)] = acc;
        }
    }
    std::swap(a, b);
  }
}

}  // namespace vgpu::hostfft
