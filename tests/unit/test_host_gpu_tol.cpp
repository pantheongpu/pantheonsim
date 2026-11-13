// Turbo host-GPU mode's compare by tolerance (include/vgpu/host_gpu_tol.hpp, docs/perf-design/12-host-gpu-turbo.md): the rule for two values, the three
// readings of a differing word, what the report counts, the ends of a range, and the parsing of the two tolerances. No card and no simulator state.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <utility>
#include <vector>

#include "vgpu/host_gpu_tol.hpp"
#include "vtest.hpp"

using namespace vgpu::hostgpu;

namespace {
uint32_t bits_of(float f) { uint32_t b; std::memcpy(&b, &f, 4); return b; }
float float_of(uint32_t b) { float f; std::memcpy(&f, &b, 4); return f; }
uint64_t bits_of(double d) { uint64_t b; std::memcpy(&b, &d, 8); return b; }

template <class T>
std::vector<uint8_t> bytes_of(const std::vector<T>& v) {
  std::vector<uint8_t> out(v.size() * sizeof(T));
  if (!out.empty()) std::memcpy(out.data(), v.data(), out.size());
  return out;
}
TolReport compare(const std::vector<uint8_t>& card, const std::vector<uint8_t>& sim, uint64_t addr = 0x1000, Tolerance t = {}) {
  TolReport r;
  compare_words(card.data(), sim.data(), addr, sim.size(), t, r);
  return r;
}
// A float array like a kernel's output: 64 values of ordinary magnitude.
std::vector<float> ramp(size_t n = 64) {
  std::vector<float> v(n);
  for (size_t i = 0; i < n; ++i) v[i] = 0.5f + 0.37f * static_cast<float>(i);
  return v;
}
std::vector<double> dramp(size_t n = 32) {
  std::vector<double> v(n);
  for (size_t i = 0; i < n; ++i) v[i] = 1.0 / 3.0 + 0.61 * static_cast<double>(i);
  return v;
}
}  // namespace

VTEST(the_defaults_are_the_documented_ones) {
  const Tolerance t;
  VCHECK_EQ(t.atol, 1e-8);
  VCHECK_EQ(t.rtol, 1e-5);
}

VTEST(tolerances_parse_as_numbers_zero_or_more) {
  double v = -1;
  VCHECK(parse_tolerance("1e-6", &v) && v == 1e-6);
  VCHECK(parse_tolerance("0", &v) && v == 0);
  VCHECK(parse_tolerance("0.25", &v) && v == 0.25);
  VCHECK(parse_tolerance("3", &v) && v == 3);
  VCHECK(parse_tolerance("0x1p-10", &v) && v == 1.0 / 1024);   // strtod's hex floats
  v = 7;
  for (const char* bad : {"", "abc", "1e-6x", "1e-6 ", "-1e-6", "-0.5", "nan", "inf", "-inf", "1,5", "--1", "1e"}) {
    if (parse_tolerance(bad, &v)) std::fprintf(stderr, "'%s' was read as a tolerance\n", bad);
    VCHECK(!parse_tolerance(bad, &v));
  }
  VCHECK(!parse_tolerance(nullptr, &v));
  VCHECK_EQ(v, 7.0);   // a refused text leaves the value alone
  VCHECK(parse_tolerance("-0", &v) && v == 0);   // minus zero is zero
}

VTEST(the_rule_for_two_values) {
  const Tolerance t;   // atol 1e-8, rtol 1e-5
  double a = -1, rel = -1;
  VCHECK(values_close(1.0, 1.0, t, &a, &rel) && a == 0 && rel == 0);
  VCHECK(values_close(1.0 + 9e-6, 1.0, t, &a, &rel));
  VCHECK(a > 8.9e-6 && a < 9.1e-6 && rel > 8.9e-6 && rel < 9.1e-6);
  VCHECK(!values_close(1.0 + 1.2e-5, 1.0, t));
  VCHECK(values_close(1000.0 + 9e-3, 1000.0, t) && !values_close(1000.0 + 1.2e-2, 1000.0, t));   // relative to the simulator's value
  // the boundary is inside: |d| == atol + rtol * |sim| passes
  Tolerance half;
  half.atol = 0.25;
  half.rtol = 0.5;
  VCHECK(values_close(2.25, 1.0, half) == false);
  VCHECK(values_close(1.75, 1.0, half) && values_close(0.25, 1.0, half) && !values_close(0.2, 1.0, half));
  // rtol scales with the SIMULATOR's value, not the card's
  Tolerance r;
  r.atol = 0;
  r.rtol = 0.5;
  VCHECK(values_close(1.5, 1.0, r) && !values_close(1.0, 3.0, r) && values_close(3.0, 2.0, r) && !values_close(2.0, 4.1, r));
  // atol alone near zero; signed zeros are equal
  VCHECK(values_close(5e-9, 0.0, t, &a, &rel) && rel == 0 && a == 5e-9);
  VCHECK(!values_close(2e-8, 0.0, t));
  VCHECK(values_close(-0.0, 0.0, t) && values_close(0.0, -0.0, t));
  VCHECK(values_close(-3e-9, 4e-9, t) && !values_close(-1.0, 1.0, t));
  // NaN matches NaN only; infinity matches the same infinity only
  const double nan = std::nan(""), inf = std::numeric_limits<double>::infinity();
  VCHECK(values_close(nan, nan, t) && values_close(-nan, nan, t));
  VCHECK(!values_close(nan, 1.0, t) && !values_close(1.0, nan, t) && !values_close(nan, inf, t) && !values_close(inf, nan, t));
  VCHECK(values_close(inf, inf, t) && values_close(-inf, -inf, t));
  VCHECK(!values_close(inf, -inf, t) && !values_close(inf, 1e308, t) && !values_close(1e308, inf, t) && !values_close(-inf, 0.0, t));
  Tolerance loose;
  loose.atol = 1e300;
  loose.rtol = 1e300;
  VCHECK(!values_close(inf, 1.0, loose) && !values_close(1.0, -inf, loose) && !values_close(nan, 1.0, loose));   // no tolerance buys an infinity or a NaN
  VCHECK(values_close(-5.0, 7.0, loose));
  // a tolerance of zero is bit equality of the values
  Tolerance zero;
  zero.atol = zero.rtol = 0;
  VCHECK(values_close(1.0, 1.0, zero) && !values_close(std::nextafter(1.0, 2.0), 1.0, zero));
}

VTEST(half_values_are_read_exactly) {
  VCHECK_EQ(half_to_double(0x3c00), 1.0);
  VCHECK_EQ(half_to_double(0xc000), -2.0);
  VCHECK_EQ(half_to_double(0x3c01), 1.0 + 1.0 / 1024);
  VCHECK_EQ(half_to_double(0x7bff), 65504.0);
  VCHECK_EQ(half_to_double(0x0001), std::ldexp(1.0, -24));   // the smallest denormal
  VCHECK_EQ(half_to_double(0x03ff), std::ldexp(1023.0, -24));
  VCHECK_EQ(half_to_double(0x0400), std::ldexp(1.0, -14));   // the smallest normal
  VCHECK(half_to_double(0x0000) == 0 && !std::signbit(half_to_double(0x0000)) && std::signbit(half_to_double(0x8000)));
  VCHECK(std::isinf(half_to_double(0x7c00)) && half_to_double(0x7c00) > 0 && half_to_double(0xfc00) < 0);
  VCHECK(std::isnan(half_to_double(0x7c01)) && std::isnan(half_to_double(0xfe00)) && std::isnan(half_to_double(0x7fff)));
}

VTEST(equal_memory_counts_words_and_nothing_else) {
  const auto b = bytes_of(ramp());
  const TolReport r = compare(b, b);
  VCHECK_EQ(r.words, uint64_t{64});
  VCHECK(r.different == 0 && r.outside == 0 && r.subnormal == 0 && !r.has_first && r.max_abs == 0 && r.max_rel == 0);
  TolReport none;
  compare_words(b.data(), b.data(), 0x1000, 0, {}, none);
  VCHECK_EQ(none.words, uint64_t{0});
}

VTEST(binary32_words_within_and_outside_tolerance) {
  const std::vector<float> sim = ramp();
  std::vector<float> card = sim;
  card[3] = std::nextafter(sim[3], 100.f);           // one ulp: inside
  card[10] = sim[10] * (1 + 5e-6f);                  // inside
  card[11] = sim[11] * (1 + 4e-5f);                  // outside (an odd word)
  card[20] = sim[20] + 1.0f;                         // outside (an even word: the word reading 2 passed before it was fenced in)
  card[21] = -sim[21];                               // outside
  const TolReport r = compare(bytes_of(card), bytes_of(sim));
  VCHECK_EQ(r.words, uint64_t{64});
  VCHECK_EQ(r.different, uint64_t{5});
  VCHECK_EQ(r.outside, uint64_t{3});
  VCHECK(r.has_first && r.first_addr == 0x1000 + 11 * 4 && r.first_card == bits_of(card[11]) && r.first_sim == bits_of(sim[11]));
  VCHECK(r.max_abs >= 2.0 * static_cast<double>(sim[21]) * 0.999 && r.max_abs <= 2.0 * static_cast<double>(sim[21]) * 1.001);   // the sign flip is the largest
  VCHECK(r.max_rel > 1.99 && r.max_rel < 2.01);
  VCHECK_EQ(r.subnormal, uint64_t{0});
}

// The hole a plain "else try binary64" has: in an array of floats, any change of a word at an 8-byte boundary reads as a change of a double's low half.
VTEST(a_float_array_is_not_reread_as_doubles) {
  const std::vector<float> sim = ramp();
  for (size_t i = 0; i < sim.size(); ++i) {
    std::vector<float> card = sim;
    card[i] = sim[i] * 1.5f + 3.0f;
    const TolReport r = compare(bytes_of(card), bytes_of(sim));
    if (r.outside != 1) std::fprintf(stderr, "element %zu: a change of 50%% was %s\n", i, r.outside ? "counted twice" : "inside tolerance");
    VCHECK_EQ(r.outside, uint64_t{1});
    VCHECK_EQ(r.first_addr, uint64_t{0x1000 + i * 4});
  }
  // the same at an address that is 4 past an 8-byte boundary, and with the array's first and last words
  for (uint64_t addr : {uint64_t{0x1004}, uint64_t{0x2000}, uint64_t{0xfffc}}) {
    std::vector<float> card = sim;
    card[0] = 100;
    card[63] = -100;
    card[30] = 77;
    VCHECK_EQ(compare(bytes_of(card), bytes_of(sim), addr).outside, uint64_t{3});
  }
  // a float that overflowed on the card only: FLT_MAX against infinity is one bit apart and must be outside, in either half of the 8 bytes
  for (size_t i : {size_t{6}, size_t{7}}) {
    std::vector<float> s = sim, c = sim;
    s[i] = std::numeric_limits<float>::max();
    c[i] = std::numeric_limits<float>::infinity();
    VCHECK_EQ(compare(bytes_of(c), bytes_of(s)).outside, uint64_t{1});
    VCHECK_EQ(compare(bytes_of(s), bytes_of(c)).outside, uint64_t{1});
    // ... also in memory that does not look like floats at all (zeros around it): the high half of 8 bytes never reads as a double when it is an infinity
    std::vector<float> zs(64, 0.0f), zc(64, 0.0f);
    zs[i] = std::numeric_limits<float>::max();
    zc[i] = std::numeric_limits<float>::infinity();
    if (i % 2) VCHECK_EQ(compare(bytes_of(zc), bytes_of(zs)).outside, uint64_t{1});
  }
}

VTEST(binary64_values_pass_by_the_double_reading) {
  const std::vector<double> sim = dramp();
  std::vector<double> card = sim;
  card[2] = sim[2] * (1 + 1e-9);     // far beyond what the low word allows as a float, well inside as a double
  card[5] = sim[5] * (1 + 3e-6);     // changes the high word too
  card[7] = std::nextafter(sim[7], 1e9);
  TolReport r = compare(bytes_of(card), bytes_of(sim));
  VCHECK_EQ(r.words, uint64_t{64});
  VCHECK(r.different >= 3 && r.different <= 4);
  VCHECK_EQ(r.outside, uint64_t{0});
  VCHECK(r.max_rel < 4e-6);
  // outside as doubles: outside
  card = sim;
  card[9] = sim[9] * (1 + 1e-3);
  r = compare(bytes_of(card), bytes_of(sim));
  VCHECK(r.outside >= 1);
  VCHECK(r.first_addr == 0x1000 + 9 * 8 || r.first_addr == 0x1000 + 9 * 8 + 4);
  // a double at an address that is not a multiple of 8 is not a double: its words are judged alone. (Doubles of 4/3: the low word, 0x55555555, is a
  // float of 1.5e13, so a change of it cannot hide under atol as the low words of the ramp above do.)
  const std::vector<double> ts(32, 1.0 + 1.0 / 3.0);
  std::vector<double> tc = ts;
  tc[2] = ts[2] * (1 + 1e-9);
  VCHECK_EQ(compare(bytes_of(tc), bytes_of(ts)).outside, uint64_t{0});
  VCHECK(compare(bytes_of(tc), bytes_of(ts), 0x1004).outside >= 1);
  // the 8 bytes must lie inside the range: the last word of a range that ends mid-double is judged alone
  std::vector<uint8_t> cb = bytes_of(tc), sb = bytes_of(ts);
  cb.resize(2 * 8 + 4);
  sb.resize(2 * 8 + 4);
  VCHECK(compare(cb, sb).outside == 1);
  // ... and the first word of a range that starts in the middle of 8 bytes is judged alone (nothing before the range is read)
  {
    std::vector<uint32_t> s4(4, 0), c4(4, 0);
    s4[0] = static_cast<uint32_t>(bits_of(1.0f));
    c4[0] = static_cast<uint32_t>(bits_of(2.0f));
    const TolReport head = compare(bytes_of(c4), bytes_of(s4), 0x1004);
    VCHECK_EQ(head.outside, uint64_t{1});
    VCHECK_EQ(head.first_addr, uint64_t{0x1004});
  }
  // infinities and NaNs of doubles
  card = sim;
  std::vector<double> s2 = sim;
  s2[4] = std::nan("");
  card[4] = -std::nan("0x55");
  VCHECK_EQ(compare(bytes_of(card), bytes_of(s2)).outside, uint64_t{0});
  s2[4] = std::numeric_limits<double>::infinity();
  card[4] = std::numeric_limits<double>::max();
  VCHECK(compare(bytes_of(card), bytes_of(s2)).outside >= 1);
}

// Halves are judged only when the caller says the memory holds halves (Reading::F16, VGPU_HOST_GPU_TOL_TYPE=f16): under the guess two halves in a
// word read as one float first.
VTEST(binary16_data_is_judged_when_the_reading_is_f16) {
  Tolerance t;
  t.rtol = 2e-3;   // takes one ulp of a half (2^-10 of the value)
  t.reading = Reading::F16;
  std::vector<uint16_t> sim(128);
  for (size_t i = 0; i < sim.size(); ++i) sim[i] = static_cast<uint16_t>(0x3000 + 37 * i);   // 0.125 and up
  std::vector<uint16_t> card = sim;
  card[9] = static_cast<uint16_t>(sim[9] + 1);     // the high half of word 4: one ulp
  card[20] = static_cast<uint16_t>(sim[20] + 1);   // the low half of word 10
  card[21] = static_cast<uint16_t>(sim[21] - 1);   // and its high half
  TolReport r = compare(bytes_of(card), bytes_of(sim), 0x1000, t);
  VCHECK_EQ(r.words, uint64_t{64});
  VCHECK_EQ(r.different, uint64_t{2});
  VCHECK_EQ(r.outside, uint64_t{0});
  VCHECK(r.max_rel > 5e-4 && r.max_rel < 1e-3);
  VCHECK(r.max_abs > 1e-4 && r.max_abs < 2e-4);   // one ulp at 2^-3: 2^-13
  // one half far off: the word is outside although the other half passes
  card[21] = static_cast<uint16_t>(sim[21] + 0x400);   // twice the value
  r = compare(bytes_of(card), bytes_of(sim), 0x1000, t);
  VCHECK_EQ(r.outside, uint64_t{1});
  VCHECK_EQ(r.first_addr, uint64_t{0x1000 + 10 * 4});
  // at the default rtol one ulp of a half is outside
  Tolerance d16;
  d16.reading = Reading::F16;
  card = sim;
  card[9] = static_cast<uint16_t>(sim[9] + 1);
  VCHECK_EQ(compare(bytes_of(card), bytes_of(sim), 0x1000, d16).outside, uint64_t{1});
  // What the guess does with the same memory (the limit the doc names): a low half that is 40 units off, and a half of 0.166 that is 5% off, are
  // outside as halves and inside under the guess, where the word is a float of about 1e-8 and the difference is under atol.
  Tolerance guess;
  guess.rtol = 2e-3;
  const std::pair<size_t, int> off_by[] = {{20, 40}, {9, 0x40}};
  for (const auto& [index, add] : off_by) {
    card = sim;
    card[index] = static_cast<uint16_t>(sim[index] + add);
    VCHECK_EQ(compare(bytes_of(card), bytes_of(sim), 0x1000, t).outside, uint64_t{1});
    VCHECK_EQ(compare(bytes_of(card), bytes_of(sim), 0x1000, guess).outside, uint64_t{0});
  }
  // zeros of either sign pass at any tolerance; a half infinity against a finite half does not
  std::vector<uint16_t> s2 = sim, c2 = sim;
  s2[20] = 0x0000; c2[20] = 0x8000;
  s2[23] = 0x8000; c2[23] = 0x0000;
  VCHECK_EQ(compare(bytes_of(c2), bytes_of(s2), 0x1000, d16).outside, uint64_t{0});
  s2[31] = 0x7c00; c2[31] = 0x7bff;
  s2[40] = 0x7c00; c2[40] = 0xfc00;
  VCHECK_EQ(compare(bytes_of(c2), bytes_of(s2), 0x1000, t).outside, uint64_t{2});
  // a half NaN matches only the same NaN: two different patterns that are both half NaNs are what the low bits of a float look like one time in 32
  s2 = sim; c2 = sim;
  s2[9] = 0x7e00; c2[9] = 0xfe01;
  VCHECK_EQ(compare(bytes_of(c2), bytes_of(s2), 0x1000, t).outside, uint64_t{1});
  double a = 1, rel = 1;
  VCHECK(halves_close(0x7e00, 0x7e00, t, &a, &rel) && a == 0 && rel == 0);
  VCHECK(!halves_close(0x7e00, 0x7e01, t, &a, &rel) && !halves_close(0x7e00, 0x3c00, t, &a, &rel) && !halves_close(0x7c00, 0xfc00, t, &a, &rel));
  VCHECK(halves_close(0x3c01, 0x3c00, t, &a, &rel) && a > 0 && rel > 9e-4 && rel < 1e-3);
  VCHECK(halves_close(0x0000, 0x8000, t, &a, &rel) && halves_close(0x7c00, 0x7c00, t, &a, &rel));
}

// The readings a caller can force, and their names.
VTEST(a_forced_reading_is_the_only_reading) {
  Reading rd = Reading::F64;
  VCHECK(parse_reading("auto", &rd) && rd == Reading::Auto);
  VCHECK(parse_reading("f32", &rd) && rd == Reading::F32);
  VCHECK(parse_reading("f64", &rd) && rd == Reading::F64);
  VCHECK(parse_reading("f16", &rd) && rd == Reading::F16);
  for (const char* bad : {"", "F16", "half", "f8", "f16 ", "float", "f32,f64"}) VCHECK(!parse_reading(bad, &rd));
  VCHECK(!parse_reading(nullptr, &rd) && rd == Reading::F16);
  VCHECK(Tolerance{}.reading == Reading::Auto);
  Tolerance f32, f64;
  f32.reading = Reading::F32;
  f64.reading = Reading::F64;
  // f32: no halves. The float 2.0 with its low 16 bits changed from 0x0000 to 0x8000 is 0.4% off; the guess passes it as two halves (a zero of either
  // sign and an equal half), the float reading does not.
  std::vector<float> fs = ramp(), fc = ramp();
  fs[6] = 2.0f;
  fc[6] = float_of(bits_of(2.0f) | 0x8000u);
  VCHECK_EQ(compare(bytes_of(fc), bytes_of(fs)).outside, uint64_t{0});
  VCHECK_EQ(compare(bytes_of(fc), bytes_of(fs), 0x1000, f32).outside, uint64_t{1});
  // ... and no doubles: a double that differs by 1e-9 has a low word that is another float altogether
  std::vector<double> ds(32, 1.0 + 1.0 / 3.0), dc(32, 1.0 + 1.0 / 3.0);
  dc[2] = ds[2] * (1 + 1e-9);
  VCHECK_EQ(compare(bytes_of(dc), bytes_of(ds), 0x1000, f64).outside, uint64_t{0});
  VCHECK_EQ(compare(bytes_of(dc), bytes_of(ds), 0x1000, f32).outside, uint64_t{1});
  VCHECK_EQ(compare(bytes_of(dc), bytes_of(ds)).outside, uint64_t{0});   // (the guess: not float data, so the double reading is tried)
  // f64: every word is half of a double. A float array's element at an 8-byte boundary may change by half and pass (that is what was asked for);
  // the one in the high half may not.
  fs = ramp(); fc = ramp();
  fc[6] = fs[6] * 1.5f;
  VCHECK_EQ(compare(bytes_of(fc), bytes_of(fs), 0x1000, f64).outside, uint64_t{0});
  fc[7] = fs[7] * 1.5f;
  VCHECK_EQ(compare(bytes_of(fc), bytes_of(fs), 0x1000, f64).outside, uint64_t{2});   // (words 6 and 7 are one double, which now fails: both are outside)
  // f64: a word with no whole 8 bytes inside the range is exact
  std::vector<uint8_t> cb = bytes_of(dc), sb = bytes_of(ds);
  cb.resize(2 * 8 + 4);
  sb.resize(2 * 8 + 4);
  VCHECK_EQ(compare(cb, sb, 0x1000, f64).outside, uint64_t{1});
  // f64 against a double that is off by 1e-3
  dc[2] = ds[2] * (1 + 1e-3);
  VCHECK_EQ(compare(bytes_of(dc), bytes_of(ds), 0x1000, f64).outside, uint64_t{2});
}

VTEST(denormals_zeros_and_nans_of_binary32) {
  std::vector<float> sim = ramp(), card = sim;
  // signed zero, denormal against zero, two denormals: all inside (their difference is below atol), and counted as what small integers look like
  sim[1] = 0.0f; card[1] = -0.0f;
  sim[4] = float_of(0x00000000); card[4] = float_of(0x00000001);
  sim[5] = float_of(0x00000005); card[5] = float_of(0x00000007);   // the integers 5 and 7
  sim[8] = float_of(0x007fffff); card[8] = float_of(0x807fffff);   // the largest denormals of either sign
  // NaNs: any payload, any sign
  sim[12] = float_of(0x7fc00000); card[12] = float_of(0x7fffffff);
  sim[13] = float_of(0xffc00001); card[13] = float_of(0x7f800001);
  TolReport r = compare(bytes_of(card), bytes_of(sim));
  VCHECK_EQ(r.different, uint64_t{6});
  VCHECK_EQ(r.outside, uint64_t{0});
  VCHECK_EQ(r.subnormal, uint64_t{4});
  // a denormal against the smallest normal is not "both denormal"
  sim = ramp(); card = sim;
  sim[4] = float_of(0x007fffff); card[4] = float_of(0x00800000);
  r = compare(bytes_of(card), bytes_of(sim));
  VCHECK(r.different == 1 && r.outside == 0 && r.subnormal == 0);
  // with atol 0 a denormal against zero is outside, and signed zeros still pass
  Tolerance t0;
  t0.atol = 0;
  sim = ramp(); card = sim;
  sim[1] = 0.0f; card[1] = -0.0f;
  sim[4] = 0.0f; card[4] = float_of(0x00000001);
  r = compare(bytes_of(card), bytes_of(sim), 0x1000, t0);
  VCHECK(r.different == 2 && r.outside == 1 && r.first_addr == 0x1010);
  // NaN against a number, infinity against the other infinity, infinity against NaN: outside
  sim = ramp(); card = sim;
  sim[2] = float_of(0x7fc00000); card[2] = 1.0f;
  sim[3] = 2.0f; card[3] = float_of(0x7fc00000);
  sim[6] = float_of(0x7f800000); card[6] = float_of(0xff800000);
  sim[7] = float_of(0x7f800000); card[7] = float_of(0x7f800001);
  r = compare(bytes_of(card), bytes_of(sim));
  VCHECK_EQ(r.outside, uint64_t{4});
  VCHECK_EQ(r.max_abs, 0.0);   // nothing finite on both sides
  // ... also with an atol that would take the low halves of an infinity and a NaN as two close binary16 values (0x0000 and 0x0001)
  Tolerance wide;
  wide.atol = 1e-6;
  sim = ramp(); card = sim;
  sim[7] = float_of(0x7f800000); card[7] = float_of(0x7f800001);
  sim[8] = float_of(0xff800000); card[8] = float_of(0xff800001);
  VCHECK_EQ(compare(bytes_of(card), bytes_of(sim), 0x1000, wide).outside, uint64_t{2});
  // +inf against -inf is outside under every reading (as halves the two words are 0x7f80 and 0xff80, both half NaNs: not tried)
  {
    std::vector<uint32_t> z(64, 0), zc(64, 0);
    for (size_t i : {size_t{10}, size_t{11}}) {
      z[i] = 0x7f800000u;
      zc[i] = 0xff800000u;
    }
    VCHECK_EQ(compare(bytes_of(zc), bytes_of(z)).outside, uint64_t{2});
  }
  // integers that do not look like floats (bit patterns of magnitudes near 2^97): a change of the high word of 8 bytes is outside ...
  std::vector<uint32_t> si(64), cs(64);
  for (uint32_t i = 0; i < 64; ++i) si[i] = cs[i] = 0x70000000u + i * 0x00010193u;
  cs[17] ^= 0x00400000u;
  VCHECK_EQ(compare(bytes_of(cs), bytes_of(si)).outside, uint64_t{1});
  // ... and a change of the LOW word passes as the low half of a "double": the limit of reading 2 where data is not recognisably binary32
  cs[18] += 0x20000000u;
  VCHECK_EQ(compare(bytes_of(cs), bytes_of(si)).outside, uint64_t{1});
  // the same change in an array that looks like floats is outside
  std::vector<float> fs = ramp(), fc = ramp();
  fc[18] = float_of(static_cast<uint32_t>(bits_of(fs[18])) + 0x20000000u);
  VCHECK_EQ(compare(bytes_of(fc), bytes_of(fs)).outside, uint64_t{1});
}

VTEST(range_ends_that_are_not_word_aligned_are_exact) {
  const std::vector<float> sim = ramp(8);
  const std::vector<uint8_t> sb = bytes_of(sim);
  // [addr, addr + len) for every start alignment and a few lengths: equal data, the count of words is tol_word_count's
  for (uint64_t addr = 0x1000; addr < 0x1008; ++addr)
    for (uint64_t len = 0; len <= 20; ++len) {
      std::vector<uint8_t> s(sb.begin(), sb.begin() + len);
      TolReport r;
      compare_words(s.data(), s.data(), addr, len, {}, r);
      VCHECK_EQ(r.words, tol_word_count(addr, len));
      VCHECK(r.different == 0 && r.outside == 0);
      const uint64_t first_word = (addr + 3) & ~uint64_t{3}, end = addr + len;
      const uint64_t want = len == 0 ? 0 : (end <= first_word ? 1 : (first_word > addr) + (end - first_word) / 4 + ((end - first_word) % 4 != 0));
      VCHECK_EQ(r.words, want);
    }
  VCHECK_EQ(tol_word_count(0x1000, 0), uint64_t{0});
  VCHECK_EQ(tol_word_count(0x1000, 64), uint64_t{16});
  VCHECK_EQ(tol_word_count(0x1001, 2), uint64_t{1});
  VCHECK_EQ(tol_word_count(0x1003, 6), uint64_t{3});
  // a byte that differs in the partial word before the first aligned word, and in the one after the last: outside, whatever the tolerance
  Tolerance loose;
  loose.atol = loose.rtol = 1e30;
  for (uint64_t at : {uint64_t{0}, uint64_t{1}, uint64_t{2}, uint64_t{15}, uint64_t{16}}) {
    std::vector<uint8_t> s(17, 0x40), c(17, 0x40);
    c[at] ^= 1;
    TolReport r;
    compare_words(c.data(), s.data(), 0x1001, 17, loose, r);   // 3 head bytes, 3 words, 2 tail bytes
    VCHECK_EQ(r.words, uint64_t{5});
    VCHECK_EQ(r.different, uint64_t{1});
    VCHECK_EQ(r.outside, uint64_t{1});
    VCHECK_EQ(r.first_addr, static_cast<uint64_t>(at < 3 ? 0x1001 : 0x1010));
    VCHECK(r.first_card != r.first_sim);
  }
  // a range shorter than its head
  {
    std::vector<uint8_t> s = {1, 2}, c = {1, 3};
    TolReport r;
    compare_words(c.data(), s.data(), 0x1001, 2, loose, r);
    VCHECK(r.words == 1 && r.different == 1 && r.outside == 1 && r.first_addr == 0x1001 && r.first_card == 0x0301 && r.first_sim == 0x0201);
  }
  // the aligned words between the ends are still judged by tolerance
  {
    std::vector<uint8_t> s(3 + 16 + 2, 0), c;
    const std::vector<float> f = {1.0f, 2.0f, 3.0f, 4.0f};
    std::memcpy(s.data() + 3, f.data(), 16);
    c = s;
    const float g = std::nextafter(2.0f, 3.0f);
    std::memcpy(c.data() + 3 + 4, &g, 4);
    TolReport r;
    compare_words(c.data(), s.data(), 0x1001, s.size(), {}, r);
    VCHECK(r.words == 6 && r.different == 1 && r.outside == 0);
  }
}

VTEST(reports_add_up) {
  TolReport a, b;
  a.words = 10; a.different = 3; a.outside = 0; a.subnormal = 1; a.max_abs = 0.5; a.max_rel = 0.25;
  b.words = 7; b.different = 2; b.outside = 2; b.subnormal = 2; b.max_abs = 0.25; b.max_rel = 2.0; b.has_first = true; b.first_addr = 0x40; b.first_card = 1; b.first_sim = 2;
  a.add(b);
  VCHECK(a.words == 17 && a.different == 5 && a.outside == 2 && a.subnormal == 3 && a.max_abs == 0.5 && a.max_rel == 2.0);
  VCHECK(a.has_first && a.first_addr == 0x40 && a.first_card == 1 && a.first_sim == 2);
  TolReport c;
  c.outside = 1; c.has_first = true; c.first_addr = 0x80;
  a.add(c);
  VCHECK(a.outside == 3 && a.first_addr == 0x40);   // the first stays the first
  // compare_words adds to a report that already holds something
  const std::vector<float> sim = ramp();
  std::vector<float> card = sim;
  card[1] = 1e9f;
  TolReport r;
  compare_words(bytes_of(card).data(), bytes_of(sim).data(), 0x1000, 256, {}, r);
  compare_words(bytes_of(card).data(), bytes_of(sim).data(), 0x9000, 256, {}, r);
  VCHECK(r.words == 128 && r.different == 2 && r.outside == 2 && r.first_addr == 0x1004);
}

VTEST(looks_like_f32_tells_float_arrays_from_doubles_and_bits) {
  const auto fb = bytes_of(ramp());
  for (uint64_t at = 0; at < fb.size(); at += 4) VCHECK(looks_like_f32(fb.data(), 0x1000, fb.size(), at));
  const auto db = bytes_of(dramp());
  unsigned yes = 0;
  for (uint64_t at = 0; at < db.size(); at += 4) yes += looks_like_f32(db.data(), 0x1000, db.size(), at);
  VCHECK_EQ(yes, 0u);
  // random bits: the low words are plausible floats about one time in three
  std::mt19937 rng(7);
  std::vector<uint32_t> rnd(64);
  for (auto& w : rnd) w = rng();
  const auto rb = bytes_of(rnd);
  VCHECK(!looks_like_f32(rb.data(), 0x1000, rb.size(), 0));
  // all zeros, and small integers: not floats
  std::vector<uint32_t> ints(64);
  for (uint32_t i = 0; i < 64; ++i) ints[i] = i * 3;
  const auto ib = bytes_of(ints);
  VCHECK(!looks_like_f32(ib.data(), 0x1000, ib.size(), 8));
  // fewer than 4 nonzero low words in the window: the word itself decides, and only a nonzero plausible word in the low half says float
  std::vector<float> sparse(64, 0.0f);
  sparse[10] = 3.5f;
  sparse[11] = 3.5f;
  const auto sb = bytes_of(sparse);
  VCHECK(looks_like_f32(sb.data(), 0x1000, sb.size(), 40));
  VCHECK(!looks_like_f32(sb.data(), 0x1000, sb.size(), 44));
  VCHECK(!looks_like_f32(sb.data(), 0x1000, sb.size(), 48));
  // the window is the 256 bytes that hold the word: floats in one window do not vouch for the next
  std::vector<uint8_t> two(512, 0);
  std::memcpy(two.data(), fb.data(), 256);
  std::memcpy(two.data() + 256, db.data(), 256);
  VCHECK(looks_like_f32(two.data(), 0x1000, 512, 128) && !looks_like_f32(two.data(), 0x1000, 512, 256 + 128));
  VCHECK(plausible_f32(bits_of(1.0f)) && plausible_f32(bits_of(-1e-12f)) && plausible_f32(bits_of(1e12f)) && !plausible_f32(bits_of(1e-13f)) && !plausible_f32(bits_of(1e13f)) && !plausible_f32(0) &&
         !plausible_f32(5) && !plausible_f32(0x7f800000u));
  (void)bits_of(1.0);
}

// A model of what turbo mode produces: a float array whose every element is the card's, one or two ulp from the simulator's. Nothing is outside at the
// defaults, the counts are exact, and the largest relative difference is two ulp.
VTEST(a_whole_array_of_one_and_two_ulp_differences_is_inside) {
  std::mt19937 rng(11);
  std::vector<float> sim(4096), card(4096);
  uint64_t differ = 0;
  for (size_t i = 0; i < sim.size(); ++i) {
    sim[i] = std::ldexp(1.0f + static_cast<float>(rng() & 0x7fffff) / 8388608.0f, static_cast<int>(rng() % 21) - 10);
    const int k = static_cast<int>(rng() % 5) - 2;
    card[i] = float_of(bits_of(sim[i]) + static_cast<uint32_t>(k));
    differ += k != 0;
  }
  const TolReport r = compare(bytes_of(card), bytes_of(sim), 0x7f0000100000ull);
  VCHECK_EQ(r.words, uint64_t{4096});
  VCHECK_EQ(r.different, differ);
  VCHECK_EQ(r.outside, uint64_t{0});
  VCHECK(r.max_rel > 1e-7 && r.max_rel < 2.5e-7);
  // the same differences are all outside when no tolerance is given
  Tolerance zero;
  zero.atol = zero.rtol = 0;
  VCHECK_EQ(compare(bytes_of(card), bytes_of(sim), 0x7f0000100000ull, zero).outside, differ);
}

VTEST_MAIN
