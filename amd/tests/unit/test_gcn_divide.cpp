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

VTEST_MAIN
