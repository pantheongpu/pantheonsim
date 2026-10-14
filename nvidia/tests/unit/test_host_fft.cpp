// The host FFT behind cuFFT (src/host_fft.hpp), bit for bit against the recursive mixed-radix Cooley-Tukey it replaced,
// which is kept here as the reference (cufft_api.cpp's fft_rec before host_fft.hpp). Sizes with every kind of factor
// (powers of two, odd composites, primes, a large prime), both directions, and inputs that break arithmetic.
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#include "../../src/host_fft.hpp"
#include "vtest.hpp"

namespace {

using cd = std::complex<double>;

int ref_smallest_factor(int n) {
  for (int p = 2; (long long)p * p <= n; ++p)
    if (n % p == 0) return p;
  return n;
}

void fft_rec(const cd* in, cd* out, int n, int stride, int sign, cd* scratch) {
  if (n == 1) { out[0] = in[0]; return; }
  const int p = ref_smallest_factor(n);
  const int m = n / p;
  cd* tmp = scratch;
  for (int j = 0; j < p; ++j)
    fft_rec(in + (size_t)j * stride, tmp + (size_t)j * m, m, stride * p, sign, scratch + n);
  const double base = sign * 2.0 * M_PI / n;
  for (int q = 0; q < p; ++q)
    for (int k = 0; k < m; ++k) {
      cd acc = tmp[k];
      for (int j = 1; j < p; ++j) {
        const double ang = base * (double)j * (double)(k + q * m);
        acc += tmp[(size_t)j * m + k] * cd(std::cos(ang), std::sin(ang));
      }
      out[(size_t)q * m + k] = acc;
    }
}

void ref_fft_1d(cd* data, int n, int sign) {
  if (n <= 1) return;
  std::vector<cd> out((size_t)n), scratch((size_t)n * 2 + 8);
  fft_rec(data, out.data(), n, 1, sign, scratch.data());
  std::memcpy(data, out.data(), sizeof(cd) * (size_t)n);
}

enum class Mix { Plain, Special, SpecialSparse };

std::vector<cd> fill(int n, Mix mix, std::mt19937_64& rng) {
  using L = std::numeric_limits<double>;
  const double sp[] = {0.0, -0.0, 1.0, -1.0, L::denorm_min(), L::min(), L::max(), -L::max(), L::infinity(),
                       -L::infinity(), L::quiet_NaN(), 1e-300, 1e300, 0.1};
  std::uniform_real_distribution<double> u(-2.0, 2.0);
  auto one = [&] {
    switch (mix) {
      case Mix::Plain: return u(rng);
      case Mix::Special: return rng() % 2 ? sp[rng() % (sizeof sp / sizeof sp[0])] : u(rng);
      default: return rng() % 50 == 0 ? sp[rng() % (sizeof sp / sizeof sp[0])] : u(rng);
    }
  };
  std::vector<cd> v((size_t)n);
  for (auto& x : v) x = cd(one(), one());
  return v;
}

// Bits equal, or both NaN (which payload wins when two meet is not something the card defines).
size_t differing(const std::vector<cd>& a, const std::vector<cd>& b) {
  size_t bad = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    const double ar[2] = {a[i].real(), a[i].imag()}, br[2] = {b[i].real(), b[i].imag()};
    for (int c = 0; c < 2; ++c) {
      if (std::memcmp(&ar[c], &br[c], 8) == 0) continue;
      if (std::isnan(ar[c]) && std::isnan(br[c])) continue;
      ++bad;
    }
  }
  return bad;
}

}  // namespace

VTEST(host_fft_matches_the_recursion) {
  std::mt19937_64 rng(97);
  std::vector<int> sizes;
  for (int n = 1; n <= 130; ++n) sizes.push_back(n);
  for (int n : {210, 243, 256, 343, 360, 500, 509, 512, 625, 720, 997, 1000, 1024, 1155, 1331, 2310, 2048, 3125, 4096, 4099})
    sizes.push_back(n);
  size_t bad = 0, runs = 0;
  for (const int n : sizes)
    for (const int sign : {-1, 1})
      for (const Mix mix : {Mix::Plain, Mix::Special, Mix::SpecialSparse}) {
        if (n > 1500 && mix != Mix::Plain && sign == 1) continue;   // (keep the big sizes quick)
        const auto input = fill(n, mix, rng);
        auto ref = input, got = input;
        ref_fft_1d(ref.data(), n, sign);
        const auto plan = vgpu::hostfft::plan_1d(n, sign);
        std::vector<cd> scratch((size_t)n * 2 + 2);
        vgpu::hostfft::run(*plan, got.data(), scratch.data());
        bad += differing(ref, got);
        ++runs;
      }
  std::printf("  %zu transforms (%zu sizes x 2 directions x inputs): %zu parts differ\n", runs, sizes.size(), bad);
  VCHECK_EQ(bad, static_cast<size_t>(0));
}

// A prime too large for a twiddle table is computed on the fly, with the same numbers.
VTEST(host_fft_large_prime_without_a_table) {
  std::mt19937_64 rng(5);
  const int n = 1451;   // (n-1)*n = 2.1M entries, over the table bound
  const auto input = fill(n, Mix::Plain, rng);
  auto ref = input, got = input;
  ref_fft_1d(ref.data(), n, -1);
  const auto plan = vgpu::hostfft::plan_1d(n, -1);
  VCHECK(plan->levels.size() == 1 && !plan->levels[0].tw);
  std::vector<cd> scratch((size_t)n * 2 + 2);
  vgpu::hostfft::run(*plan, got.data(), scratch.data());
  VCHECK_EQ(differing(ref, got), static_cast<size_t>(0));
}

VTEST_MAIN
