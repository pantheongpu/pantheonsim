// The arithmetic under the matrix instructions (amd/src/mfma_math.hpp), against what it replaced: the compiler's
// conversion of a binary16 to double, and a dot product read out of the strided layout the interpreter used to keep B in.
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "../../src/mfma_math.hpp"
#include "vtest.hpp"

namespace {

uint64_t bits_of(double d) {
  uint64_t b;
  std::memcpy(&b, &d, 8);
  return b;
}

struct Rng {
  uint64_t s;
  uint64_t next() {
    s ^= s << 13, s ^= s >> 7, s ^= s << 17;
    return s;
  }
};

// An interesting double: zeros, infinities, NaNs, denormals, the extremes, and ordinary numbers of every size.
double any_double(Rng& r) {
  switch (r.next() % 12) {
    case 0: return 0.0;
    case 1: return -0.0;
    case 2: return std::numeric_limits<double>::infinity();
    case 3: return -std::numeric_limits<double>::infinity();
    case 4: return std::numeric_limits<double>::quiet_NaN();
    case 5: return std::numeric_limits<double>::denorm_min() * static_cast<double>(r.next() % 1000);
    case 6: return std::numeric_limits<double>::max() / static_cast<double>(1 + r.next() % 7);
    default: {
      const double m = static_cast<double>(static_cast<int64_t>(r.next() >> 11) - (int64_t{1} << 52)) / (1ull << 52);
      return std::ldexp(m, static_cast<int>(r.next() % 80) - 40);
    }
  }
}

}  // namespace

VTEST(every_binary16_converts_to_the_double_the_compiler_gives) {
  uint32_t wrong = 0;
  for (uint32_t h = 0; h < 65536; ++h) {
    _Float16 f;
    const uint16_t bits = static_cast<uint16_t>(h);
    std::memcpy(&f, &bits, 2);
    const double want = static_cast<double>(f);
    const double got = vgpu::amd::half_bits_to_double(bits);
    wrong += bits_of(got) != bits_of(want);   // sign of zero, NaN payload and quiet bit included
  }
  VCHECK_EQ(wrong, 0u);
}

VTEST(a_dot_product_reads_the_same_sum_from_a_contiguous_column_as_from_the_strided_one) {
  Rng rng{0x1234567};
  uint64_t checked = 0;
  for (int round = 0; round < 4000; ++round) {
    const uint32_t k = 1 + static_cast<uint32_t>(rng.next() % 64), n = 1 + static_cast<uint32_t>(rng.next() % 16);
    const uint32_t j = static_cast<uint32_t>(rng.next() % n);
    std::vector<double> a(k), b(k * n), column(k);   // b[kk * n + j]: the old layout; `column`: the new
    for (auto& x : a) x = any_double(rng);
    for (auto& x : b) x = any_double(rng);
    for (uint32_t kk = 0; kk < k; ++kk) column[kk] = b[kk * n + j];
    const double c = any_double(rng);
    double old_sum = c, old_fma = c;
    int64_t old_int = static_cast<int64_t>(rng.next() % 1000);
    const int64_t int_start = old_int;
    std::vector<double> ia(k), ib(k * n);
    for (auto& x : ia) x = static_cast<double>(static_cast<int64_t>(rng.next() % 2000001) - 1000000);
    for (auto& x : ib) x = static_cast<double>(static_cast<int64_t>(rng.next() % 2000001) - 1000000);
    std::vector<double> icol(k);
    for (uint32_t kk = 0; kk < k; ++kk) {
      old_sum += a[kk] * b[kk * n + j];
      old_fma = std::fma(a[kk], b[kk * n + j], old_fma);
      old_int += static_cast<int64_t>(ia[kk]) * static_cast<int64_t>(ib[kk * n + j]);
      icol[kk] = ib[kk * n + j];
    }
    VCHECK_EQ(bits_of(vgpu::amd::mfma_dot(a.data(), column.data(), k, c)), bits_of(old_sum));
    VCHECK_EQ(bits_of(vgpu::amd::mfma_dot_fma(a.data(), column.data(), k, c)), bits_of(old_fma));
    VCHECK_EQ(vgpu::amd::mfma_dot_int(ia.data(), icol.data(), k, int_start), old_int);
    ++checked;
  }
  VCHECK_EQ(checked, 4000u);
}

VTEST(four_dot_products_at_once_give_what_four_one_at_a_time_do) {
  Rng rng{0xBEEF};
  for (int round = 0; round < 3000; ++round) {
    const uint32_t k = 1 + static_cast<uint32_t>(rng.next() % 64);
    std::vector<double> a[4], b[4];
    const double* rows[4];
    const double* cols[4];
    double start[4], want[4], got[4];
    for (int t = 0; t < 4; ++t) {
      a[t].resize(k), b[t].resize(k);
      for (auto& x : a[t]) x = any_double(rng);
      for (auto& x : b[t]) x = any_double(rng);
      rows[t] = a[t].data(), cols[t] = b[t].data(), start[t] = any_double(rng);
      want[t] = vgpu::amd::mfma_dot(rows[t], cols[t], k, start[t]);
    }
    vgpu::amd::mfma_dot4(rows, cols, k, start, got);
    for (int t = 0; t < 4; ++t) VCHECK_EQ(bits_of(got[t]), bits_of(want[t]));
  }
}

VTEST(the_terms_are_summed_in_order) {
  // Large, small and negative-large: (1e16 + 1) - 1e16 is 0 in this order, 1 in another.
  const double a[3] = {1.0, 1.0, 1.0}, b[3] = {1e16, 1.0, -1e16};
  VCHECK_EQ(vgpu::amd::mfma_dot(a, b, 3, 0.0), 0.0);
  const double b2[3] = {1e16, -1e16, 1.0};
  VCHECK_EQ(vgpu::amd::mfma_dot(a, b2, 3, 0.0), 1.0);
}

VTEST_MAIN
