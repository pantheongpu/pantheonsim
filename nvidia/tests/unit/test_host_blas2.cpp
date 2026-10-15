// The rank-one and rank-two updates of cuBLAS on the elements as the device holds them (src/host_blas2.hpp), bit for bit against the
// loops they replaced, which are kept here as the reference: std::complex<double> products and sums, rounded to the matrix's type.
// Float, double, complex float and complex double matrices; ger, geru, gerc, syr, her, syr2, her2 on either triangle; sizes from
// nothing to a few tens with leading dimensions that pad; plain values, zeros of both signs, subnormals, values that overflow in a
// product, infinities and NaNs. When the kernel says it has the answer its matrix must equal the reference's to the bit (two NaNs
// count as equal); it must say so for every matrix of plain values.
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

#include "../../src/host_blas2.hpp"
#include "vtest.hpp"

namespace {

using cd = std::complex<double>;

struct Rng {
  uint64_t s = 0x1234567887654321ull;
  uint64_t next() {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return s;
  }
  double unit() { return static_cast<double>(next() >> 11) * (2.0 / 9007199254740992.0) - 1.0; }
  int below(int n) { return static_cast<int>(next() % static_cast<uint64_t>(n)); }
};

enum class Mix { Plain, Zeros, Subnormal, Huge, Special };

double draw(Mix mix, Rng& r) {
  switch (mix) {
    case Mix::Plain: return r.unit() * 4;
    case Mix::Zeros: return r.below(3) == 0 ? (r.below(2) ? 0.0 : -0.0) : r.unit();
    case Mix::Subnormal: return r.below(3) == 0 ? std::numeric_limits<double>::denorm_min() * (1 + r.below(50)) * (r.below(2) ? 1 : -1) : r.unit() * 1e-160;
    case Mix::Huge: return r.below(3) == 0 ? (r.below(2) ? 1e160 : -1e160) : r.unit() * 1e150;
    default: {
      const int k = r.below(40);
      if (k == 0) return std::numeric_limits<double>::infinity();
      if (k == 1) return -std::numeric_limits<double>::infinity();
      if (k == 2) return std::numeric_limits<double>::quiet_NaN();
      return r.unit();
    }
  }
}

template <class S> bool same(S a, S b) { return std::memcmp(&a, &b, sizeof(S)) == 0 || (std::isnan(a) && std::isnan(b)); }

template <class S, int NC> cd element(const S* a, int idx) { return cd(static_cast<double>(a[idx * NC]), NC == 2 ? static_cast<double>(a[idx * NC + 1]) : 0.0); }
template <class S, int NC> void put(S* a, int idx, cd v) {
  a[idx * NC] = static_cast<S>(v.real());
  if (NC == 2) a[idx * NC + 1] = static_cast<S>(v.imag());
}

// The loops of cublas_complex.inc's cger and csyr, as they were.
template <class S, int NC>
void ref_ger(bool conj, int m, int n, cd a_, const std::vector<cd>& xv, const std::vector<cd>& yv, S* A, int ld) {
  std::vector<cd> a(static_cast<size_t>(ld) * n);
  const size_t ext = n > 0 ? static_cast<size_t>(ld) * (n - 1) + m : 0;
  for (size_t k = 0; k < ext; ++k) a[k] = element<S, NC>(A, static_cast<int>(k));
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < m; ++i) a[static_cast<size_t>(j) * ld + i] += a_ * xv[i] * (conj ? std::conj(yv[j]) : yv[j]);
  for (size_t k = 0; k < ext; ++k) put<S, NC>(A, static_cast<int>(k), a[k]);
}
template <class S, int NC>
void ref_syr(bool hermitian, bool lower, int n, cd a_, const std::vector<cd>& xv, const std::vector<cd>* y, S* A, int ld) {
  const std::vector<cd>& yv = y ? *y : xv;
  auto op = [&](cd v) { return hermitian ? std::conj(v) : v; };
  std::vector<cd> a(static_cast<size_t>(ld) * n);
  const size_t ext = n > 0 ? static_cast<size_t>(ld) * (n - 1) + n : 0;
  for (size_t k = 0; k < ext; ++k) a[k] = element<S, NC>(A, static_cast<int>(k));
  for (int j = 0; j < n; ++j)
    for (int i = lower ? j : 0; i < (lower ? n : j + 1); ++i) {
      cd& e = a[static_cast<size_t>(j) * ld + i];
      e += y ? a_ * xv[i] * op(yv[j]) + op(a_) * yv[i] * op(xv[j]) : a_ * xv[i] * op(xv[j]);
      if (hermitian && i == j) e = e.real();
    }
  for (size_t k = 0; k < ext; ++k) put<S, NC>(A, static_cast<int>(k), a[k]);
}

template <class S, int NC>
void check_all(const char* what, size_t* bad, int* runs, int* declined_plain) {
  Rng rng;
  rng.s += static_cast<uint64_t>(NC) * 31 + sizeof(S);
  for (const Mix mix : {Mix::Plain, Mix::Zeros, Mix::Subnormal, Mix::Huge, Mix::Special})
    for (int rep = 0; rep < 60; ++rep) {
      const int m = rep % 5 == 0 ? 0 : 1 + rng.below(33), n = rep % 7 == 0 ? 0 : 1 + rng.below(33);
      const int nn = std::min(std::max(m, n), 33);
      const int ld = std::max(m, nn) + rng.below(3);
      const size_t cells = static_cast<size_t>(ld) * std::max(n, nn);
      std::vector<S> base(cells * NC);
      for (auto& v : base) v = static_cast<S>(draw(mix == Mix::Plain || mix == Mix::Zeros ? mix : Mix::Plain, rng));
      auto vec = [&](int len) {
        std::vector<cd> v(static_cast<size_t>(len));
        for (auto& e : v) e = cd(draw(mix, rng), NC == 2 || rng.below(2) ? draw(mix, rng) : 0.0);
        return v;
      };
      const cd alpha(draw(mix, rng), NC == 2 ? draw(mix, rng) : 0.0);
      const auto x = vec(std::max(m, nn)), y = vec(std::max(n, nn));
      for (const bool conj : {false, true}) {
        if (m == 0 || n == 0) break;
        std::vector<S> want = base, got = base;
        ref_ger<S, NC>(conj, m, n, alpha, x, y, want.data(), ld);
        const bool ok = vgpu::hostblas2::ger<S, NC>(conj, m, n, alpha, x, y, got.data(), ld);
        ++*runs;
        if (!ok && mix == Mix::Plain) ++*declined_plain;
        if (ok)
          for (size_t k = 0; k < got.size(); ++k)
            if (!same(got[k], want[k])) {
              ++*bad;
              if (*bad < 5) std::printf("  %s ger mix %d m %d n %d conj %d: element %zu %.17g, the loops' %.17g\n", what, static_cast<int>(mix), m, n, (int)conj, k, (double)got[k], (double)want[k]);
              break;
            }
      }
      if (nn == 0) continue;
      for (const bool lower : {false, true})
        for (const int variant : {0, 1, 2, 3}) {   // syr, her, syr2, her2
          const bool hermitian = variant == 1 || variant == 3, two = variant >= 2;
          if (hermitian && NC == 1) continue;
          std::vector<S> want = base, got = base;
          const cd a_ = hermitian && !two ? cd(alpha.real(), 0.0) : alpha;   // (her takes a real alpha)
          ref_syr<S, NC>(hermitian, lower, nn, a_, x, two ? &y : nullptr, want.data(), ld);
          const bool ok = vgpu::hostblas2::syr<S, NC>(hermitian, lower, nn, a_, x, two ? &y : nullptr, got.data(), ld);
          ++*runs;
          if (!ok && mix == Mix::Plain) ++*declined_plain;
          if (ok)
            for (size_t k = 0; k < got.size(); ++k)
              if (!same(got[k], want[k])) {
                ++*bad;
                if (*bad < 5) std::printf("  %s syr mix %d n %d lower %d variant %d: element %zu %.17g, the loops' %.17g\n", what, static_cast<int>(mix), nn, (int)lower, variant, k, (double)got[k], (double)want[k]);
                break;
              }
        }
    }
}

}  // namespace

VTEST(host_blas2_updates_match_the_loops) {
  size_t bad = 0;
  int runs = 0, declined = 0;
  check_all<float, 1>("float", &bad, &runs, &declined);
  check_all<double, 1>("double", &bad, &runs, &declined);
  check_all<float, 2>("complex float", &bad, &runs, &declined);
  check_all<double, 2>("complex double", &bad, &runs, &declined);
  std::printf("  %d updates, %d declined on plain values, %zu differ\n", runs, declined, bad);
  VCHECK_EQ(bad, static_cast<size_t>(0));
  VCHECK_EQ(declined, 0);
}

// A product that overflows and is then multiplied by zero or by something small: the loops' complex multiply meets an infinity
// (and may repair a NaN pair), which the kernels do not do, so they have to decline.
VTEST(host_blas2_declines_where_a_product_overflows) {
  size_t bad = 0;
  int declined = 0, runs = 0;
  const double ys[] = {0.0, -0.0, 1e-5, 3.0, -1e200, 1e-300};
  for (int rep = 0; rep < 6; ++rep) {
    const std::vector<cd> x = {cd(1e200, rep & 1 ? 1e200 : 0.0), cd(2.0, 3.0), cd(1e-300, 0.0)};
    const std::vector<cd> y = {cd(ys[rep], ys[(rep + 1) % 6]), cd(ys[(rep + 2) % 6], 0.0), cd(1.0, -1.0)};
    const cd alpha(1e200, rep & 2 ? 1e200 : 0.0);
    for (const bool conj : {false, true}) {
      std::vector<double> want(3 * 2 * 3, 1.0), got = want;
      ref_ger<double, 2>(conj, 3, 3, alpha, x, y, want.data(), 3);
      const bool ok = vgpu::hostblas2::ger<double, 2>(conj, 3, 3, alpha, x, y, got.data(), 3);
      ++runs;
      if (!ok) ++declined;
      else
        for (size_t k = 0; k < got.size(); ++k)
          if (!same(got[k], want[k])) ++bad;
    }
  }
  std::printf("  %d overflowing updates, %d declined, %zu differ\n", runs, declined, bad);
  VCHECK_EQ(bad, static_cast<size_t>(0));
  VCHECK(declined > 0);
}

VTEST_MAIN
