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
    double old_sum = c, old_fma = c, ordered_sum = c;
    int64_t old_int = static_cast<int64_t>(rng.next() % 1000);
    const int64_t int_start = old_int;
    std::vector<double> ia(k), ib(k * n);
    for (auto& x : ia) x = static_cast<double>(static_cast<int64_t>(rng.next() % 2000001) - 1000000);
    for (auto& x : ib) x = static_cast<double>(static_cast<int64_t>(rng.next() % 2000001) - 1000000);
    std::vector<double> icol(k);
    for (uint32_t kk = 0; kk < k; ++kk) {
      old_sum += a[kk] * b[kk * n + j];
      ordered_sum = vgpu::amd::mfma_add(ordered_sum, vgpu::amd::mfma_mul(a[kk], b[kk * n + j]));
      old_fma = std::fma(a[kk], b[kk * n + j], old_fma);
        old_int += static_cast<int64_t>(ia[kk]) * static_cast<int64_t>(ib[kk * n + j]);
      icol[kk] = ib[kk * n + j];
    }
    // A sum that is not a NaN is the same whatever order the compiler gives the two operands of the plain loop above. A NaN is
    // not: when both operands of an add or multiply are NaNs the payload and sign that come out are the operand order's, which
    // the optimizer picks differently at -O0 and -O1 (the sanitizer builds). Those are compared with the fixed order, written
    // out here on the strided layout.
    VCHECK_EQ(bits_of(vgpu::amd::mfma_dot(a.data(), column.data(), k, c)), bits_of(old_sum == old_sum ? old_sum : ordered_sum));
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

VTEST(nans_with_different_payloads_come_out_as_the_fixed_operand_order_gives) {
  // Both operands of a multiply or an add NaN: the payload that wins is the first operand's. The plain loops may have either
  // first, so a sum that ends as a NaN is redone in the fixed order; all three forms must agree with it.
  Rng rng{0xFACE};
  for (int round = 0; round < 3000; ++round) {
    const uint32_t k = 1 + static_cast<uint32_t>(rng.next() % 16);
    std::vector<double> a[4], b[4];
    const double* rows[4];
    const double* cols[4];
    double start[4], got4[4];
    for (int t = 0; t < 4; ++t) {
      a[t].resize(k), b[t].resize(k);
      const auto nan_or = [&](double v) {
        if (rng.next() % 3) return v;
        const uint64_t bits = 0x7FF8000000000000ull | (rng.next() & 0x8007FFFFFFFFFFFFull);
        double d;
        std::memcpy(&d, &bits, 8);
        return d;
      };
      for (auto& x : a[t]) x = nan_or(any_double(rng));
      for (auto& x : b[t]) x = nan_or(any_double(rng));
      rows[t] = a[t].data(), cols[t] = b[t].data(), start[t] = nan_or(any_double(rng));
    }
    vgpu::amd::mfma_dot4(rows, cols, k, start, got4);
    for (int t = 0; t < 4; ++t) {
      const double want = vgpu::amd::mfma_dot_ordered(rows[t], cols[t], k, start[t]);
      VCHECK_EQ(bits_of(got4[t]), bits_of(want));
      // (a sum that is a number is the same either way)
      VCHECK_EQ(bits_of(vgpu::amd::mfma_dot(rows[t], cols[t], k, start[t])), bits_of(want));
    }
  }
}

VTEST(the_terms_are_summed_in_order) {
  // Large, small and negative-large: (1e16 + 1) - 1e16 is 0 in this order, 1 in another.
  const double a[3] = {1.0, 1.0, 1.0}, b[3] = {1e16, 1.0, -1e16};
  VCHECK_EQ(vgpu::amd::mfma_dot(a, b, 3, 0.0), 0.0);
  const double b2[3] = {1e16, -1e16, 1.0};
  VCHECK_EQ(vgpu::amd::mfma_dot(a, b2, 3, 0.0), 1.0);
}


// ---- whole tiles --------------------------------------------------------------------------------------------------------

namespace {
struct TileCase {
  uint32_t m, n, k;
  std::vector<double> a, bk, bt, c;   // a: m rows of k; bk: k rows of n; bt: n columns of k (the same B); c: m rows of n
};
// A random tile whose entries `make` draws, with B kept both ways.
template <typename Make>
TileCase make_tile(Rng& rng, uint32_t m, uint32_t n, uint32_t k, Make make) {
  TileCase t{m, n, k, {}, {}, {}, {}};
  t.a.resize(size_t{m} * k), t.bk.resize(size_t{k} * n), t.bt.resize(size_t{n} * k), t.c.resize(size_t{m} * n);
  for (auto& x : t.a) x = make(rng);
  for (auto& x : t.bk) x = make(rng);
  for (auto& x : t.c) x = make(rng);
  for (uint32_t kk = 0; kk < k; ++kk)
    for (uint32_t j = 0; j < n; ++j) t.bt[size_t{j} * k + kk] = t.bk[size_t{kk} * n + j];
  return t;
}
double nan_or(Rng& rng, double v) {
  if (rng.next() % 4) return v;
  const uint64_t bits = 0x7FF8000000000000ull | (rng.next() & 0x8007FFFFFFFFFFFFull);
  double d;
  std::memcpy(&d, &bits, 8);
  return d;
}
const uint32_t kShapes[][3] = {{4, 4, 1}, {4, 4, 4}, {16, 16, 1}, {16, 16, 4}, {16, 16, 16}, {16, 16, 32}, {16, 16, 128},
                               {32, 32, 2}, {32, 32, 8}, {32, 32, 16}, {32, 32, 64}, {8, 4, 3}, {4, 12, 5}};
}  // namespace

VTEST(a_tile_gives_each_output_what_the_dot_product_of_its_row_and_column_gives) {
  Rng rng{0x5EED};
  for (int round = 0; round < 600; ++round) {
    const auto& sh = kShapes[round % (sizeof kShapes / sizeof kShapes[0])];
    // finite numbers of every size, a few infinities, and sometimes NaNs with payloads
    const bool nans = round % 3 == 2;
    const TileCase t = make_tile(rng, sh[0], sh[1], sh[2], [&](Rng& r) { return nans ? nan_or(r, any_double(r)) : any_double(r); });
    std::vector<double> out(size_t{t.m} * t.n);
    vgpu::amd::mfma_tile_f(t.m, t.n, t.k, t.a.data(), t.bk.data(), t.c.data(), out.data());
    for (uint32_t i = 0; i < t.m; ++i)
      for (uint32_t j = 0; j < t.n; ++j)
        VCHECK_EQ(bits_of(out[size_t{i} * t.n + j]),
                  bits_of(vgpu::amd::mfma_dot(t.a.data() + size_t{i} * t.k, t.bt.data() + size_t{j} * t.k, t.k, t.c[size_t{i} * t.n + j])));
  }
}

VTEST(a_tile_sums_each_output_in_order_of_k) {
  // 1, 2^-60 and -1 (and the same with the 2^-60 first): a sum that is not taken in order of k, or is taken in pairs, differs.
  for (uint32_t k : {3u, 4u, 8u, 16u}) {
    std::vector<double> a(4 * k, 1.0), bk(size_t{k} * 4), c(16, 0.0), out(16);
    for (uint32_t kk = 0; kk < k; ++kk)
      for (uint32_t j = 0; j < 4; ++j) bk[size_t{kk} * 4 + j] = kk == 0 ? 1.0 : kk == 1 ? std::ldexp(1.0, -60 - static_cast<int>(j)) : kk == 2 ? -1.0 : std::ldexp(1.0, -70);
    vgpu::amd::mfma_tile_f(4, 4, k, a.data(), bk.data(), c.data(), out.data());
    for (uint32_t j = 0; j < 4; ++j) {
      double want = 0;
      for (uint32_t kk = 0; kk < k; ++kk) want += a[kk] * bk[size_t{kk} * 4 + j];
      for (uint32_t i = 0; i < 4; ++i) VCHECK_EQ(bits_of(out[size_t{i} * 4 + j]), bits_of(want));
    }
  }
}

namespace {
template <int Order>
void check_nan_order(Rng& rng) {
  for (int round = 0; round < 200; ++round) {
    const auto& sh = kShapes[round % (sizeof kShapes / sizeof kShapes[0])];
    const TileCase t = make_tile(rng, sh[0], sh[1], sh[2], [&](Rng& r) { return nan_or(r, nan_or(r, any_double(r))); });
    std::vector<double> out(size_t{t.m} * t.n);
    vgpu::amd::mfma_tile_f<Order>(t.m, t.n, t.k, t.a.data(), t.bk.data(), t.c.data(), out.data());
    for (uint32_t i = 0; i < t.m; ++i)
      for (uint32_t j = 0; j < t.n; ++j) {
        // the sum, a product at a time, the way the order says (a number is a number in any order)
        double sum = t.c[size_t{i} * t.n + j];
        for (uint32_t kk = 0; kk < t.k; ++kk)
          sum = vgpu::amd::mfma_add_o<Order>(sum, vgpu::amd::mfma_mul_o<Order>(t.a[size_t{i} * t.k + kk], t.bt[size_t{j} * t.k + kk]));
        VCHECK_EQ(bits_of(out[size_t{i} * t.n + j]), bits_of(sum));
      }
  }
}
}  // namespace

VTEST(a_tile_of_nans_takes_the_payload_the_fixed_operand_order_gives_in_each_of_the_four_orders) {
  Rng rng{0xBADC0DE};
  check_nan_order<0>(rng);
  check_nan_order<1>(rng);
  check_nan_order<2>(rng);
  check_nan_order<3>(rng);
}

VTEST(a_fused_tile_gives_each_output_the_chain_of_fmas) {
  Rng rng{0xF00D};
  for (int round = 0; round < 400; ++round) {
    const auto& sh = kShapes[round % (sizeof kShapes / sizeof kShapes[0])];
    const bool nans = round % 2;
    const TileCase t = make_tile(rng, sh[0], sh[1], sh[2], [&](Rng& r) { return nans ? nan_or(r, any_double(r)) : any_double(r); });
    std::vector<double> out(size_t{t.m} * t.n);
    vgpu::amd::mfma_tile_fma(t.m, t.n, t.k, t.a.data(), t.bk.data(), t.c.data(), out.data());
    for (uint32_t i = 0; i < t.m; ++i)
      for (uint32_t j = 0; j < t.n; ++j) {
        const double want = vgpu::amd::mfma_dot_fma(t.a.data() + size_t{i} * t.k, t.bt.data() + size_t{j} * t.k, t.k, t.c[size_t{i} * t.n + j]);
        VCHECK_EQ(bits_of(out[size_t{i} * t.n + j]), bits_of(want));
      }
  }
}

VTEST(an_integer_tile_gives_the_exact_dot_products) {
  Rng rng{0x1417};
  for (int round = 0; round < 400; ++round) {
    static const uint32_t kSizes[] = {1, 2, 3, 4, 5, 7, 8, 12, 16, 32};
    const uint32_t m = kSizes[rng.next() % 10], n = kSizes[rng.next() % 10], k = 2 * (1 + static_cast<uint32_t>(rng.next() % 64));
    std::vector<int16_t> ia(size_t{m} * k), bi(size_t{k} * n);
    std::vector<int64_t> bk(size_t{k} * n);
    for (auto& x : ia) x = static_cast<int16_t>(static_cast<int>(rng.next() % 511) - 255);
    for (uint32_t kk = 0; kk < k; ++kk)
      for (uint32_t j = 0; j < n; ++j) {
        const int16_t v = static_cast<int16_t>(static_cast<int>(rng.next() % 511) - 255);
        bi[((kk / 2) * n + j) * 2 + kk % 2] = v;
        bk[size_t{kk} * n + j] = v;
      }
    std::vector<int32_t> out(size_t{m} * n);
    vgpu::amd::mfma_tile_int(m, n, k, ia.data(), bi.data(), out.data());
    for (uint32_t i = 0; i < m; ++i)
      for (uint32_t j = 0; j < n; ++j) {
        int64_t want = 0;
        for (uint32_t kk = 0; kk < k; ++kk) want += int64_t{ia[size_t{i} * k + kk]} * bk[size_t{kk} * n + j];
        VCHECK_EQ(int64_t{out[size_t{i} * n + j]}, want);
      }
  }
}

VTEST(a_fused_multiply_add_of_several_nans_passes_on_the_payload_of_b_then_a_then_the_sum) {
  const auto nan = [](uint64_t payload, bool signalling) {
    const uint64_t bits = 0x7FF0000000000000ull | payload | (signalling ? 0 : 1ull << 51);
    double d;
    std::memcpy(&d, &bits, 8);
    return d;
  };
  const double nb = nan(0x111, false), na = nan(0x222, true), ns = nan(0x333, false);
  VCHECK_EQ(bits_of(vgpu::amd::mfma_fma(nb, na, ns)), bits_of(nb));
  VCHECK_EQ(bits_of(vgpu::amd::mfma_fma(2.0, na, ns)), bits_of(nan(0x222, false)));   // (a signalling NaN comes out quiet)
  VCHECK_EQ(bits_of(vgpu::amd::mfma_fma(2.0, 3.0, ns)), bits_of(ns));
  // no NaN in, one made by the operation: the default NaN
  const double inf = std::numeric_limits<double>::infinity();
  VCHECK_EQ(bits_of(vgpu::amd::mfma_fma(inf, 0.0, 1.0)), 0xFFF8000000000000ull);
  VCHECK_EQ(bits_of(vgpu::amd::mfma_fma(inf, 1.0, -inf)), 0xFFF8000000000000ull);
}

#if defined(__x86_64__) && defined(__GNUC__)
// The 4-wide kernels (AVX2) and the 2-wide ones give each other's bits on the same tiles, NaNs and all.
VTEST(the_four_wide_and_two_wide_tile_kernels_give_the_same_bits) {
  Rng rng{0xA5A5};
  const bool had = vgpu::amd::mfma_avx2_switch();
  for (int round = 0; round < 500; ++round) {
    static const uint32_t kWide[][3] = {{16, 16, 16}, {32, 32, 8}, {16, 16, 32}, {32, 32, 2}, {16, 16, 128}, {4, 8, 3}, {8, 16, 5}};
    const auto& sh = kWide[round % 7];
    const bool nans = round % 3 == 1;
    const TileCase t = make_tile(rng, sh[0], sh[1], sh[2], [&](Rng& r) { return nans ? nan_or(r, any_double(r)) : any_double(r); });
    std::vector<double> wide(size_t{t.m} * t.n), narrow(size_t{t.m} * t.n);
    vgpu::amd::mfma_avx2_switch() = true;
    vgpu::amd::mfma_tile_f(t.m, t.n, t.k, t.a.data(), t.bk.data(), t.c.data(), wide.data());
    vgpu::amd::mfma_avx2_switch() = false;
    vgpu::amd::mfma_tile_f(t.m, t.n, t.k, t.a.data(), t.bk.data(), t.c.data(), narrow.data());
    for (size_t e = 0; e < wide.size(); ++e) VCHECK_EQ(bits_of(wide[e]), bits_of(narrow[e]));
    // and the integer kernels
    const uint32_t k2 = t.k + (t.k & 1);
    std::vector<int16_t> ia(size_t{t.m} * k2), bi(size_t{k2} * t.n);
    for (auto& x : ia) x = static_cast<int16_t>(static_cast<int>(rng.next() % 511) - 255);
    for (auto& x : bi) x = static_cast<int16_t>(static_cast<int>(rng.next() % 511) - 255);
    std::vector<int32_t> iw(size_t{t.m} * t.n), in(size_t{t.m} * t.n);
    vgpu::amd::mfma_avx2_switch() = true;
    vgpu::amd::mfma_tile_int(t.m, t.n, k2, ia.data(), bi.data(), iw.data());
    vgpu::amd::mfma_avx2_switch() = false;
    vgpu::amd::mfma_tile_int(t.m, t.n, k2, ia.data(), bi.data(), in.data());
    for (size_t e = 0; e < iw.size(); ++e) VCHECK_EQ(iw[e], in[e]);
  }
  vgpu::amd::mfma_avx2_switch() = had;
}
#endif

// Integer sums wrap as the hardware's do, and wrapping is not undefined here: a signed product of two large factors would trip the
// sanitizer build (this test fails there with the old signed arithmetic).
VTEST(integer_dot_products_wrap_without_overflowing_a_signed_type) {
  const double a[3] = {4294967234.0, -8325582.0, 3037000500.0}, b[3] = {4294967287.0, -8325582.0, 3037000500.0};
  uint64_t want = 0x1234;
  for (int i = 0; i < 3; ++i) want += static_cast<uint64_t>(static_cast<int64_t>(a[i])) * static_cast<uint64_t>(static_cast<int64_t>(b[i]));
  VCHECK_EQ(static_cast<uint64_t>(vgpu::amd::mfma_dot_int(a, b, 3, 0x1234)), want);
  VCHECK_EQ(vgpu::amd::wrap_madd(INT64_MAX, INT64_MAX, 3), static_cast<int64_t>(uint64_t{INT64_MAX} + uint64_t{INT64_MAX} * 3));
  // (The 16-bit integer tile: sums that pass 2^31 wrap in the 32 bits.)
  std::vector<int16_t> ia(2 * 64, 32767), bi(64 * 2, 32767);
  int32_t out[1] = {};
  vgpu::amd::mfma_tile_int_scalar(1, 1, 64, ia.data(), bi.data(), out);
  uint32_t sum = 0;
  for (int t = 0; t < 64; ++t) sum += static_cast<uint32_t>(32767 * 32767);
  VCHECK_EQ(static_cast<uint32_t>(out[0]), sum);
}

#if defined(__SSE2__)
VTEST(every_binary16_converts_to_the_same_double_four_at_a_time) {
  uint32_t wrong = 0;
  for (uint32_t h = 0; h < 65536; h += 4) {
    __m128d lo, hi;
    // (the other bits of each 32-bit lane are zero, as the decoder leaves them)
    vgpu::amd::half4_to_double(_mm_setr_epi32(static_cast<int>(h), static_cast<int>(h + 1), static_cast<int>(h + 2), static_cast<int>(h + 3)), &lo, &hi);
    double got[4];
    _mm_storeu_pd(got, lo), _mm_storeu_pd(got + 2, hi);
    for (uint32_t i = 0; i < 4; ++i)
      wrong += bits_of(got[i]) != bits_of(vgpu::amd::half_bits_to_double(static_cast<uint16_t>(h + i)));   // NaN payload, quiet bit, zero sign
  }
  VCHECK_EQ(wrong, 0u);
}
#endif

#if defined(__x86_64__) && defined(__GNUC__)
// (a function of its own: the intrinsics of the units the host may lack are only legal under their target)
__attribute__((target("avx2,f16c"))) static uint64_t f16c_conversions_that_differ() {
  uint64_t wrong = 0;
  for (uint32_t h = 0; h < 65536; h += 8) {
    __m256d lo, hi;
    int in[8];
    for (uint32_t i = 0; i < 8; ++i) in[i] = static_cast<int>(h + i);
    vgpu::amd::half8_to_double(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(in)), &lo, &hi);
    double got[8];
    _mm256_storeu_pd(got, lo), _mm256_storeu_pd(got + 4, hi);
    for (uint32_t i = 0; i < 8; ++i) wrong += bits_of(got[i]) != bits_of(vgpu::amd::half_bits_to_double(static_cast<uint16_t>(h + i)));
  }
  return wrong;
}
VTEST(the_f16c_half_conversion_matches_the_integer_one_for_every_binary16_value) {
  if (!__builtin_cpu_supports("avx2") || !__builtin_cpu_supports("f16c")) return;   // (nothing to compare on a host without the units)
  VCHECK_EQ(f16c_conversions_that_differ(), 0u);
}
#endif

VTEST_MAIN
