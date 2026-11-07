// Division, the way the compiler builds it from v_div_scale, v_div_fmas and v_div_fixup (gcn_divide.hpp), checked against the host's own
// IEEE division: bit for bit, never against this model's output. The sequence is LLVM's (SITargetLowering::LowerFDIV32 and
// LowerFDIV64), including a reciprocal that is off by one unit in the last place, as a card's table-driven v_rcp is.
//
// Inputs: every pairing of the edge values (zeros, denormals, the extremes of the exponent range, infinities, NaNs, both signs), random
// operands of every exponent, and ones whose exponents differ by nearly the whole range or by just the amount that puts the quotient
// at the edge of the denormal range or of the largest finite value -- the cases the scaling exists for.
//
// Also: the single rounding of the scaled multiply-add against std::fma, and the pseudocode cases of the guides.
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "../../src/gcn_divide.hpp"
#include "vtest.hpp"

namespace dv = vgpu::amd::gcn::divide;

namespace {

uint64_t g_state = 0x9e3779b97f4a7c15ull;
uint64_t next() {   // xorshift64*
  g_state ^= g_state >> 12;
  g_state ^= g_state << 25;
  g_state ^= g_state >> 27;
  return g_state * 0x2545f4914f6cdd1dull;
}

template <typename T>
T make(uint32_t biased_exp, uint64_t mant, bool neg) {
  using B = typename dv::Fp<T>::Bits;
  constexpr int M = dv::Fp<T>::kMantissa;
  const B m = static_cast<B>(mant) & ((B{1} << M) - 1);
  return dv::from_bits<T>((static_cast<B>(neg) << (M + dv::Fp<T>::kMaxExpBits)) | (static_cast<B>(biased_exp) << M) | m);
}

template <typename T>
std::vector<T> edges() {
  constexpr uint32_t maxe = dv::Fp<T>::kMaxExp;
  const uint64_t top = ~uint64_t{0};
  std::vector<T> v;
  for (bool neg : {false, true}) {
    for (uint32_t e : {0u, 1u, 2u, 3u, 24u, 25u, 54u, 55u, maxe / 2 - 1, maxe / 2, maxe / 2 + 1, maxe / 2 + 2, maxe - 3, maxe - 2, maxe - 1})
      for (uint64_t m : {uint64_t{0}, uint64_t{1}, uint64_t{2}, uint64_t{1} << (dv::Fp<T>::kMantissa - 1), top, top >> 1})
        v.push_back(make<T>(e, m, neg));
    v.push_back(make<T>(maxe, 0, neg));
  }
  v.push_back(make<T>(maxe, 1, false));
  v.push_back(make<T>(maxe, uint64_t{1} << (dv::Fp<T>::kMantissa - 1), true));
  v.push_back(make<T>(maxe, top, false));
  return v;
}

uint64_t rmant() {
  switch (next() % 6) {
    case 0: return 0;
    case 1: return ~uint64_t{0};
    case 2: return 1;
    default: return next();
  }
}

template <typename T>
void operands(std::vector<T>& a, std::vector<T>& b, int scale) {
  constexpr int64_t maxe = dv::Fp<T>::kMaxExp;
  const std::vector<T> e = edges<T>();
  for (T x : e)
    for (T y : e) {
      a.push_back(x);
      b.push_back(y);
    }
  for (int i = 0; i < 40000 * scale; ++i) {
    a.push_back(make<T>(next() % (maxe + 1), rmant(), next() & 1));
    b.push_back(make<T>(next() % (maxe + 1), rmant(), next() & 1));
  }
  // Exponent differences all over the range, and near the two ends of the quotient's.
  for (int i = 0; i < 80000 * scale; ++i) {
    const int64_t ea = 1 + static_cast<int64_t>(next() % (maxe - 1));
    int64_t eb;
    if (next() % 3 == 0) eb = ea + static_cast<int64_t>(next() % (2 * maxe)) - maxe;
    else {
      const int64_t delta = static_cast<int64_t>(next() % 41) - 20;   // quotient exponent near zero or near the top
      eb = (next() & 1) ? ea - (maxe / 2 + delta) : ea + (maxe / 2 + delta);
    }
    eb = eb < 0 ? 0 : eb > maxe - 1 ? maxe - 1 : eb;
    a.push_back(make<T>(static_cast<uint32_t>(ea), rmant(), next() & 1));
    b.push_back(make<T>(static_cast<uint32_t>(eb), rmant(), next() & 1));
  }
  // Denormal operands against anything.
  for (int i = 0; i < 20000 * scale; ++i) {
    T d = make<T>(0, rmant(), next() & 1), n = make<T>(next() % (maxe + 1), rmant(), next() & 1);
    if (i & 1) std::swap(d, n);
    a.push_back(n);
    b.push_back(d);
  }
}

// A reciprocal good to one unit in the last place, not exact: the nearest value, or the one above or below it.
template <typename T>
T reciprocal(T x, int mode) {
  T r = T(1) / x;
  if (mode == 1 && std::isfinite(r) && r != 0) r = std::nextafter(r, std::numeric_limits<T>::infinity());
  if (mode == 2 && std::isfinite(r) && r != 0) r = std::nextafter(r, -std::numeric_limits<T>::infinity());
  return r;
}

float divide_f32(float n, float d, int rcp_mode) {
  const auto ds = dv::div_scale(d, d, n);
  const auto ns = dv::div_scale(n, d, n);
  const float r = reciprocal(ds.value, rcp_mode), nd = -ds.value;
  const float f0 = std::fma(nd, r, 1.0f), f1 = std::fma(f0, r, r);
  const float mul = ns.value * f1;
  const float f2 = std::fma(nd, mul, ns.value), f3 = std::fma(f2, f1, mul), f4 = std::fma(nd, f3, ns.value);
  return dv::div_fixup(dv::div_fmas(f4, f1, f3, ns.vcc), d, n);
}

double divide_f64(double n, double d, int rcp_mode) {
  const auto ds = dv::div_scale(d, d, n);
  const double r0 = reciprocal(ds.value, rcp_mode), nd = -ds.value;
  const auto ns = dv::div_scale(n, d, n);
  const double f0 = std::fma(nd, r0, 1.0), r1 = std::fma(r0, f0, r0);
  const double f2 = std::fma(nd, r1, 1.0), r2 = std::fma(r1, f2, r1);
  const double mul = ns.value * r2;
  const double f4 = std::fma(nd, mul, ns.value);
  return dv::div_fixup(dv::div_fmas(f4, r2, mul, ns.vcc), d, n);
}

template <typename T, typename Fn>
void check_division(Fn divide, int scale) {
  std::vector<T> a, b;
  operands<T>(a, b, scale);
  int bad = 0;
  std::string first;
  for (int mode = 0; mode < 3; ++mode)
    for (size_t i = 0; i < a.size(); ++i) {
      volatile T x = a[i], y = b[i];
      const T want = x / y;
      const T got = divide(a[i], b[i], mode);
      // Bit for bit, NaN payloads included where the host's NaNs follow the guide's (x86: the first NaN operand, quieted; 0/0 and
      // inf/inf the sign-set quiet NaN); any NaN for any NaN elsewhere.
#if defined(__x86_64__)
      bool same = dv::bits_of(want) == dv::bits_of(got);
#else
      bool same = (std::isnan(want) && std::isnan(got)) || dv::bits_of(want) == dv::bits_of(got);
#endif
      // The exception: a denominator whose significand is all ones. The reciprocal of such a number sits within 2^-2p of a rounding
      // boundary, and the fused multiply-add refinement the compiler writes (Markstein's) cannot tell which side it is on -- it
      // depends on how the reciprocal's last bit came out, which a card's table-driven v_rcp does not promise. The quotient is then
      // one unit in the last place off at worst, and that is all that is asked of it.
      using Bits = typename dv::Fp<T>::Bits;
      const Bits ones = (Bits{1} << dv::Fp<T>::kMantissa) - 1;
      if (!same && (dv::bits_of(b[i]) & ones) == ones && !std::isnan(want) && !std::isnan(got) &&
          std::signbit(want) == std::signbit(got)) {
        const Bits w = dv::bits_of(want) & ~(Bits{1} << (dv::Fp<T>::kMantissa + dv::Fp<T>::kMaxExpBits)),
                   g = dv::bits_of(got) & ~(Bits{1} << (dv::Fp<T>::kMantissa + dv::Fp<T>::kMaxExpBits));
        same = (w > g ? w - g : g - w) <= 1;
      }
      if (!same) {
        if (++bad == 1) {
          char buf[200];
          std::snprintf(buf, sizeof buf, "%a / %a: got %a (%llx), want %a (%llx), reciprocal mode %d", static_cast<double>(a[i]),
                        static_cast<double>(b[i]), static_cast<double>(got), static_cast<unsigned long long>(dv::bits_of(got)),
                        static_cast<double>(want), static_cast<unsigned long long>(dv::bits_of(want)), mode);
          first = buf;
        }
      }
    }
  if (bad) throw vtest::Failure(std::to_string(bad) + " quotients differ from IEEE division; first: " + first);
}

}  // namespace

VTEST(a_float_division_is_the_ieee_quotient_for_every_operand) { check_division<float>(divide_f32, 1); }
VTEST(a_double_division_is_the_ieee_quotient_for_every_operand) { check_division<double>(divide_f64, 1); }

// The scaled fused multiply-add rounds once: with no scaling it is std::fma, and scaled it is the exact result rounded in the scaled place.
VTEST(the_scaled_multiply_add_rounds_once) {
  int bad = 0;
  for (int i = 0; i < 400000; ++i) {
    const float a = make<float>(60 + next() % 140, rmant(), next() & 1), b = make<float>(60 + next() % 140, rmant(), next() & 1),
                c = make<float>(40 + next() % 180, rmant(), next() & 1);
    if (dv::bits_of(dv::scaled_fma(a, b, c, 0)) != dv::bits_of(std::fma(a, b, c))) ++bad;
    const double x = make<double>(900 + next() % 400, rmant(), next() & 1), y = make<double>(900 + next() % 400, rmant(), next() & 1),
                 z = make<double>(700 + next() % 700, rmant(), next() & 1);
    if (dv::bits_of(dv::scaled_fma(x, y, z, 0)) != dv::bits_of(std::fma(x, y, z))) ++bad;
  }
  VCHECK_EQ(bad, 0);
  // Scaled down into the denormal range, the result is the exact value rounded to the denormal's own precision. A float's product and
  // sum are exact in a long double when the exponents are close, which these are.
  for (int i = 0; i < 400000; ++i) {
    const int ea = -38 + static_cast<int>(next() % 15), eb = -38 + static_cast<int>(next() % 15);
    const int ec = ea + eb + static_cast<int>(next() % 11) - 5;
    const float a = make<float>(127 + ea, rmant(), next() & 1), b = make<float>(127 + eb, rmant(), next() & 1),
                c = make<float>(127 + ec, rmant(), next() & 1);
    const long double exact = static_cast<long double>(a) * b + c;
    const float want = static_cast<float>(std::ldexp(exact, -64));   // one rounding of an exactly held value
    if (dv::bits_of(dv::scaled_fma(a, b, c, -64)) != dv::bits_of(want)) ++bad;
  }
  VCHECK_EQ(bad, 0);
}

// The cases the guides' pseudocode names (CDNA3 guide 12.11).
VTEST(div_scale_and_div_fixup_follow_the_guides_pseudocode) {
  using dv::div_fixup;
  using dv::div_scale;
  const float inf = std::numeric_limits<float>::infinity();
  // A zero numerator or denominator: NaN, and VCC clear.
  VCHECK(std::isnan(div_scale(1.0f, 0.0f, 3.0f).value));
  VCHECK(!div_scale(1.0f, 0.0f, 3.0f).vcc);
  // Exponents 96 apart: VCC set, and only the denominator scaled (by 2^64).
  const float big = std::ldexp(1.0f, 100), small = std::ldexp(1.0f, 3);
  VCHECK(div_scale(small, small, big).vcc);
  VCHECK_EQ(div_scale(small, small, big).value, std::ldexp(small, 64));
  VCHECK_EQ(div_scale(big, small, big).value, big);
  // A denormal denominator: both operands scaled by 2^64, VCC clear.
  const float tiny = std::ldexp(1.0f, -140);
  const float num = std::ldexp(1.0f, -60);
  VCHECK(!div_scale(tiny, tiny, num).vcc);
  VCHECK_EQ(div_scale(tiny, tiny, num).value, std::ldexp(tiny, 64));
  VCHECK_EQ(div_scale(num, tiny, num).value, std::ldexp(num, 64));
  // A denormal quotient: VCC set, only the numerator scaled up.
  const float n = std::ldexp(1.0f, -100), d = std::ldexp(1.0f, 30);
  VCHECK(div_scale(n, d, n).vcc);
  VCHECK_EQ(div_scale(n, d, n).value, std::ldexp(n, 64));
  VCHECK_EQ(div_scale(d, d, n).value, d);
  // Fixup: 0/0 and inf/inf are the sign-set quiet NaN; x/0 and inf/y infinities; x/inf and 0/y zeros; the sign is the operands'.
  VCHECK_EQ(dv::bits_of(div_fixup(0.0f, 0.0f, 0.0f)), 0xffc00000u);
  VCHECK_EQ(dv::bits_of(div_fixup(0.0f, inf, -inf)), 0xffc00000u);
  VCHECK_EQ(div_fixup(1.0f, -0.0f, 2.0f), -inf);
  VCHECK_EQ(div_fixup(1.0f, 2.0f, inf), inf);
  VCHECK_EQ(dv::bits_of(div_fixup(1.0f, inf, 2.0f)), 0u);
  VCHECK_EQ(dv::bits_of(div_fixup(1.0f, -inf, 2.0f)), 0x80000000u);
  VCHECK_EQ(div_fixup(-3.5f, 2.0f, 1.0f), 3.5f);
  VCHECK_EQ(div_fixup(3.5f, -2.0f, 1.0f), -3.5f);
}

// div_scale takes bit-arithmetic paths where its operands are normal numbers (gcn_divide.hpp); it must give the reference's value (the
// bits) and condition for every operand, for S0 each of the denominator, the numerator and something else.
template <typename T>
void check_div_scale_against_reference(int scale) {
  std::vector<T> a, b;
  operands<T>(a, b, scale);
  const std::vector<T> e = edges<T>();
  int bad = 0;
  std::string first;
  for (size_t i = 0; i < a.size(); ++i) {
    const T n = a[i], d = b[i];
    const T other = (i & 1) ? e[next() % e.size()] : make<T>(next() % (dv::Fp<T>::kMaxExp + 1), rmant(), next() & 1);
    for (T s0 : {d, n, other}) {
      const auto got = dv::div_scale(s0, d, n), want = dv::div_scale_reference(s0, d, n);
      if (dv::bits_of(got.value) != dv::bits_of(want.value) || got.vcc != want.vcc) {
        if (++bad == 1) {
          char buf[200];
          std::snprintf(buf, sizeof buf, "div_scale(%a, %a, %a): got %llx vcc %d, want %llx vcc %d", static_cast<double>(s0),
                        static_cast<double>(d), static_cast<double>(n), static_cast<unsigned long long>(dv::bits_of(got.value)), got.vcc,
                        static_cast<unsigned long long>(dv::bits_of(want.value)), want.vcc);
          first = buf;
        }
      }
    }
  }
  if (bad) throw vtest::Failure(std::to_string(bad) + " div_scale results differ from the reference; first: " + first);
}

template <typename T>
void check_scale_by_pow2() {
  int bad = 0;
  std::vector<T> v = edges<T>();
  for (int i = 0; i < 20000; ++i) v.push_back(make<T>(next() % (dv::Fp<T>::kMaxExp + 1), rmant(), next() & 1));
  for (T x : v)
    for (int k = -300; k <= 300; k += (k > -140 && k < 140) ? 1 : 7)
      if (dv::bits_of(dv::scale_by_pow2(x, k)) != dv::bits_of(std::ldexp(x, k))) ++bad;
  VCHECK_EQ(bad, 0);
}

VTEST(a_float_div_scale_is_the_reference_for_every_operand) { check_div_scale_against_reference<float>(1); }
VTEST(a_double_div_scale_is_the_reference_for_every_operand) { check_div_scale_against_reference<double>(1); }
VTEST(scaling_by_a_power_of_two_is_ldexp) {
  check_scale_by_pow2<float>();
  check_scale_by_pow2<double>();
}

// The tier's rows of binary32 lanes (div_scale_row_plain_f32, div_fixup_row_plain_f32): a row that is called plain must give, in every lane, the
// lane function's value (and, for scale, a clear condition). Lanes are spread over every pair of exponents within reach of each threshold
// (the far distance, the quotient and reciprocal at the denormal edge with the mantissa tie-break, the tiny numerator, the underflow and
// overflow distances), with the edge operands and random ones; a row of 32 lanes is plain only when all 32 are, so the 32 lanes of a row are
// the same operand pair in every lane for the exact verdict per pair, and mixed rows (one lane off) must be refused.
struct RowCounts {
  long plain = 0, refused = 0;
};

template <typename Make>
void for_each_pair(Make&& f) {
  const std::vector<float> e = edges<float>();
  for (float x : e)
    for (float y : e) f(dv::bits_of(x), dv::bits_of(y));
  const uint64_t ms[] = {0, 1, 2, 0x3FFFFF, 0x400000, 0x400001, 0x7FFFFE, 0x7FFFFF};
  for (uint32_t e1 = 0; e1 <= 255; ++e1)
    for (int delta = -160; delta <= 140; ++delta) {
      const int e2 = static_cast<int>(e1) + delta;
      if (e2 < 0 || e2 > 255) continue;
      for (uint64_t m1 : ms)
        for (uint64_t m2 : ms) f(dv::bits_of(make<float>(e1, m1, false)), dv::bits_of(make<float>(static_cast<uint32_t>(e2), m2, (m1 ^ m2) & 1)));
    }
  for (int i = 0; i < 200000; ++i) f(static_cast<uint32_t>(next()), static_cast<uint32_t>(next()));
}

VTEST(a_row_of_plain_div_scale_lanes_is_the_lane_function) {
  RowCounts rc;
  long bad = 0;
  std::string first;
  for_each_pair([&](uint32_t den, uint32_t num) {
    uint32_t b[32], c[32];
    for (int i = 0; i < 32; ++i) b[i] = den, c[i] = num;
    const bool plain = dv::div_scale_row_plain_f32(b, c, 32);
    // The reference: with S0 = either operand or a third value, the lane function leaves S0 alone and clears the condition.
    const float d = dv::from_bits<float>(den), n = dv::from_bits<float>(num);
    bool lane_plain = true;
    for (float s0 : {d, n, 1.5f}) {
      const auto r = dv::div_scale(s0, d, n);
      if (r.vcc || dv::bits_of(r.value) != dv::bits_of(s0)) lane_plain = false;
    }
    // (For two normal numbers the verdict is exact: the row is plain exactly when the lane function does nothing.)
    const uint32_t e1 = den >> 23 & 255u, e2 = num >> 23 & 255u;
    if (e1 - 1u < 254u && e2 - 1u < 254u && plain != lane_plain && ++bad == 1) first = "(exactness) den " + std::to_string(den) + " num " + std::to_string(num);
    if (plain) {
      ++rc.plain;
      if (!lane_plain && ++bad == 1) first = "den " + std::to_string(den) + " num " + std::to_string(num);
    } else {
      ++rc.refused;
    }
    // One lane off in an otherwise plain row refuses the row.
    uint32_t b2[32], c2[32];
    for (int i = 0; i < 32; ++i) b2[i] = 0x3F800000u, c2[i] = 0x40000000u;
    VCHECK(dv::div_scale_row_plain_f32(b2, c2, 32));
    b2[17] = den, c2[17] = num;
    if (dv::div_scale_row_plain_f32(b2, c2, 32) != plain) ++bad;
  });
  if (bad) throw vtest::Failure(std::to_string(bad) + " rows disagree with div_scale; first: " + first);
  VCHECK(rc.plain > 100000);     // (not vacuous)
  VCHECK(rc.refused > 100000);   // (and not everything is plain)
}

VTEST(a_row_of_plain_div_fixup_lanes_is_the_lane_function) {
  RowCounts rc;
  long bad = 0;
  std::string first;
  const std::vector<float> e = edges<float>();
  for_each_pair([&](uint32_t den, uint32_t num) {
    uint32_t q[3] = {static_cast<uint32_t>(next()), 0x3F800000u, 0xFFC00001u};
    q[next() % 3] = dv::bits_of(e[next() % e.size()]);
    for (uint32_t quo : q) {
      uint32_t a[32], b[32], c[32], r[32];
      for (int i = 0; i < 32; ++i) a[i] = quo, b[i] = den, c[i] = num;
      const bool plain = dv::div_fixup_row_plain_f32(a, b, c, r, 32);
      if (!plain) {
        ++rc.refused;
        continue;
      }
      ++rc.plain;
      const uint32_t want = dv::bits_of(dv::div_fixup(dv::from_bits<float>(quo), dv::from_bits<float>(den), dv::from_bits<float>(num)));
      for (int i = 0; i < 32; ++i)
        if (r[i] != want && ++bad == 1) first = "quo " + std::to_string(quo) + " den " + std::to_string(den) + " num " + std::to_string(num);
    }
  });
  if (bad) throw vtest::Failure(std::to_string(bad) + " rows disagree with div_fixup; first: " + first);
  VCHECK(rc.plain > 100000);
  VCHECK(rc.refused > 100000);
}

// The rows of binary64 lanes (div_scale_row_plain_f64, div_fixup_row_plain_f64): the same contract as above, with the verdict taken over the
// ACTIVE lanes only (a bit mask), at every threshold of the double: the far distance (768), the quotient and the reciprocal at the denormal
// edge with the mantissa tie-break (decided by the low word as well as the high one), the tiny numerator (53), and the underflow (-1075) and
// overflow (1025) distances. The exponents of the first operand are every one that matters (near each end, near each threshold) plus a
// spread; the second is every distance from it.
template <typename Make>
void for_each_pair64(Make&& f) {
  const std::vector<double> e = edges<double>();
  for (double x : e)
    for (double y : e) f(dv::bits_of(x), dv::bits_of(y));
  const uint64_t top = (uint64_t{1} << 52) - 1;
  const uint64_t ms[] = {0, 1, top >> 1, (top >> 1) + 1, (top >> 1) + 2, top - 1, top, uint64_t{1} << 32, (uint64_t{1} << 32) - 1};
  std::vector<uint32_t> e1s = {0, 1, 2, 3, 53, 54, 55, 56, 767, 768, 769, 1021, 1022, 1023, 1024, 1025, 1026, 1280, 2040, 2043, 2044, 2045, 2046, 2047};
  for (uint32_t k = 100; k < 2000; k += 97) e1s.push_back(k);
  for (uint32_t e1 : e1s)
    for (int delta = -1100; delta <= 1050; ++delta) {
      const int e2 = static_cast<int>(e1) + delta;
      if (e2 < 0 || e2 > 2047) continue;
      for (uint64_t m1 : ms)
        for (uint64_t m2 : ms) f(dv::bits_of(make<double>(e1, m1, false)), dv::bits_of(make<double>(static_cast<uint32_t>(e2), m2, (m1 ^ m2) & 1)));
    }
  for (int i = 0; i < 200000; ++i) f(next(), next());
}

VTEST(a_row_of_plain_div_scale_f64_lanes_is_the_lane_function) {
  RowCounts rc;
  long bad = 0;
  std::string first;
  for_each_pair64([&](uint64_t den, uint64_t num) {
    const uint64_t all = ~uint64_t{0};
    uint64_t b[64], c[64];
    for (int i = 0; i < 64; ++i) b[i] = den, c[i] = num;
    const bool plain = dv::div_scale_row_plain_f64(b, c, all, 64);
    const double d = dv::from_bits<double>(den), n = dv::from_bits<double>(num);
    bool lane_plain = true;
    for (double s0 : {d, n, 1.5}) {
      const auto r = dv::div_scale(s0, d, n);
      if (r.vcc || dv::bits_of(r.value) != dv::bits_of(s0)) lane_plain = false;
    }
    // (For two normal numbers the verdict is exact: the row is plain exactly when the lane function does nothing.)
    const uint64_t e1 = den >> 52 & 2047u, e2 = num >> 52 & 2047u;
    if (e1 - 1u < 2046u && e2 - 1u < 2046u && plain != lane_plain && ++bad == 1) first = "(exactness) den " + std::to_string(den) + " num " + std::to_string(num);
    if (plain) {
      ++rc.plain;
      if (!lane_plain && ++bad == 1) first = "den " + std::to_string(den) + " num " + std::to_string(num);
    } else {
      ++rc.refused;
    }
    // One lane off in an otherwise plain row refuses the row when that lane is active, and is not looked at when it is not; the lane and the
    // row width are varied.
    uint64_t b2[64], c2[64];
    for (int i = 0; i < 64; ++i) b2[i] = 0x3FF0000000000000ull, c2[i] = 0x4000000000000000ull;
    VCHECK(dv::div_scale_row_plain_f64(b2, c2, all, 64));
    const unsigned lane = next() % 3 == 0 ? 63u : static_cast<unsigned>(next() % 32);
    b2[lane] = den, c2[lane] = num;
    if (dv::div_scale_row_plain_f64(b2, c2, all, 64) != plain) ++bad;
    if (dv::div_scale_row_plain_f64(b2, c2, all & ~(uint64_t{1} << lane), 64) != true) ++bad;
    if (dv::div_scale_row_plain_f64(b2, c2, uint64_t{1} << lane, 64) != plain) ++bad;
    if (lane < 32) {
      if (dv::div_scale_row_plain_f64(b2, c2, uint64_t{0xFFFFFFFF}, 32) != plain) ++bad;
      if (dv::div_scale_row_plain_f64(b2, c2, uint64_t{0xFFFFFFFF} & ~(uint64_t{1} << lane), 32) != true) ++bad;
    }
    if (lane >= 32) {   // (a lane at or past the width of a 32-lane row is no lane of it)
      if (dv::div_scale_row_plain_f64(b2, c2, uint64_t{0xFFFFFFFF}, 32) != true) ++bad;
    }
  });
  if (bad) throw vtest::Failure(std::to_string(bad) + " f64 rows disagree with div_scale; first: " + first);
  VCHECK(rc.plain > 100000);     // (not vacuous)
  VCHECK(rc.refused > 100000);   // (and not everything is plain)
}

VTEST(a_row_of_plain_div_fixup_f64_lanes_is_the_lane_function) {
  RowCounts rc;
  long bad = 0;
  std::string first;
  const std::vector<double> e = edges<double>();
  for_each_pair64([&](uint64_t den, uint64_t num) {
    uint64_t q[3] = {next(), 0x3FF0000000000000ull, 0xFFF8000000000001ull};
    q[next() % 3] = dv::bits_of(e[next() % e.size()]);
    for (uint64_t quo : q) {
      const uint64_t all = ~uint64_t{0};
      uint64_t a[64], b[64], c[64], r[64];
      for (int i = 0; i < 64; ++i) a[i] = quo, b[i] = den, c[i] = num;
      const bool plain = dv::div_fixup_row_plain_f64(a, b, c, r, all, 64);
      if (!plain) {
        ++rc.refused;
      } else {
        ++rc.plain;
        const uint64_t want = dv::bits_of(dv::div_fixup(dv::from_bits<double>(quo), dv::from_bits<double>(den), dv::from_bits<double>(num)));
        for (int i = 0; i < 64; ++i)
          if (r[i] != want && ++bad == 1) first = "quo " + std::to_string(quo) + " den " + std::to_string(den) + " num " + std::to_string(num);
      }
      // One lane off: refused when active, ignored when not (its value is still computed, harmlessly).
      const unsigned lane = static_cast<unsigned>(next() % 64);
      uint64_t b2[64], c2[64];
      for (int i = 0; i < 64; ++i) b2[i] = 0x3FF0000000000000ull, c2[i] = 0x4000000000000000ull;
      b2[lane] = den, c2[lane] = num;
      if (dv::div_fixup_row_plain_f64(a, b2, c2, r, all, 64) != plain) ++bad;
      if (dv::div_fixup_row_plain_f64(a, b2, c2, r, all & ~(uint64_t{1} << lane), 64) != true) ++bad;
      if (dv::div_fixup_row_plain_f64(a, b2, c2, r, uint64_t{1} << lane, 64) != plain) ++bad;
    }
  });
  if (bad) throw vtest::Failure(std::to_string(bad) + " f64 rows disagree with div_fixup; first: " + first);
  VCHECK(rc.plain > 100000);
  VCHECK(rc.refused > 100000);
}

// v_ldexp_f32's fast form (ldexp_f32) against libm's ldexpf, bit for bit (NaN payloads included), for the edge operands and random ones, with
// exponents at every distance from the ends of the normal range and the extreme 32-bit values of k.
VTEST(a_float_ldexp_is_libms_for_every_operand_and_exponent) {
  std::vector<float> v = edges<float>();
  for (int i = 0; i < 4000; ++i) v.push_back(make<float>(next() % 256, rmant(), next() & 1));
  for (uint32_t e = 0; e <= 255; ++e)
    for (uint64_t m : {uint64_t{0}, uint64_t{1}, uint64_t{0x400000}, uint64_t{0x7FFFFF}}) v.push_back(make<float>(e, m, e & 1));
  std::vector<int32_t> ks;
  for (int k = -300; k <= 300; ++k) ks.push_back(k);
  for (int32_t k : {INT32_MIN, INT32_MIN + 1, -2147483000, -65536, 65536, 2147483000, INT32_MAX - 1, INT32_MAX, -1000000, 1000000}) ks.push_back(k);
  long bad = 0, fast = 0;
  std::string first;
  for (float x : v)
    for (int32_t k : ks) {
      const uint32_t got = dv::bits_of(dv::ldexp_f32(x, k)), want = dv::bits_of(std::ldexp(x, k));
      if (dv::ldexp_f32_bits(dv::bits_of(x), k) != want && ++bad == 1) first = "ldexp_f32_bits(" + std::to_string(dv::bits_of(x)) + ", " + std::to_string(k) + ")";
      const uint32_t ef = dv::bits_of(x) >> 23 & 255u;
      if (ef - 1u < 254u && static_cast<int64_t>(ef) + k >= 1 && static_cast<int64_t>(ef) + k <= 254) ++fast;
      if (got != want && ++bad == 1) first = "ldexp(" + std::to_string(dv::bits_of(x)) + ", " + std::to_string(k) + ")";
    }
  // The results below the normal range, where the rounding is: x just above the denormal edge or itself a denormal, k bringing the result across
  // it by every distance, and mantissas whose dropped part is below, at and above one half (ties to even), in both signs.
  long rounded = 0;
  for (uint32_t ef : {0u, 1u, 2u, 3u, 8u, 23u, 24u, 25u, 26u, 40u, 127u, 254u})
    for (int s = 0; s <= 25; ++s)
      for (int rep = 0; rep < 40; ++rep) {
        uint64_t m = rmant() & 0x7FFFFFull;
        if (s >= 1 && s <= 23) {
          const uint64_t low = (uint64_t{1} << s) - 1;
          const uint64_t half = uint64_t{1} << (s - 1);
          const int mode = rep % 4;   // below, exactly half, above, random
          m = (m & ~low) | (mode == 0 ? (half >> 1) : mode == 1 ? half : mode == 2 ? (half | 1) : (m & low));
        }
        const float x = make<float>(ef, m, (rep & 1) != 0);
        for (int k = -static_cast<int>(ef) - 30; k <= -static_cast<int>(ef) + 3; ++k) {
          const uint32_t want = dv::bits_of(std::ldexp(x, k)), got = dv::ldexp_f32_bits(dv::bits_of(x), k);
          const uint32_t res_ef = want >> 23 & 255u;
          if (res_ef == 0 && (want & 0x7FFFFFu) != 0) ++rounded;
          if (got != want && ++bad == 1) first = "ldexp_f32_bits(" + std::to_string(dv::bits_of(x)) + ", " + std::to_string(k) + ")";
        }
      }
  if (bad) throw vtest::Failure(std::to_string(bad) + " results differ from libm; first: " + first);
  VCHECK(fast > 100000);   // (not vacuous: the fast form was taken)
  VCHECK(rounded > 10000); // (and many results were rounded denormals)
}

VTEST_MAIN
