// Host-GPU turbo mode: comparing the card's memory with the simulator's by tolerance (docs/perf-design/12-host-gpu-turbo.md).
//
// Device memory has no types, so the compare guesses. Over a range of bytes, for each 4-byte word (aligned by ADDRESS) whose bits differ:
//   1. read both as binary32: the word passes when |card - sim| <= atol + rtol * |sim|;
//   2. else read the aligned 8 bytes that hold it as binary64, when they lie inside the range: the same rule;
//   3. else read its two halves as binary16: it passes when each half passes the rule.
// A NaN matches any NaN under readings 1 and 2 and only the same bits under reading 3; an infinity matches only the same infinity; a word that is a
// binary32 infinity or NaN on one side and fails reading 1 is not tried as halves. A word that passes no reading is "outside tolerance".
// Bytes before the first aligned word and after the last one are compared exactly.
//
// Reading 2 is tried only where the simulator's own bytes do not look like an array of binary32 (looks_like_f32). Without that, reading 2 passes
// EVERY difference in a word at an 8-byte boundary: the next word, whatever it holds, reads as the high half of a double, and a change of the low half
// is at most 2^-20 of that double, under any rtol above 1e-6. (An array of floats where every second element could never be outside tolerance.)
// Reading 3 has no such fence, because half data in the usual range looks like binary32 too: one ulp of a binary16 is 1e-3 of its value, so at the
// default rtol it passes only halves that are equal or zeros of either sign, and half data needs a larger VGPU_HOST_GPU_RTOL to pass at all.
//
// What this cannot know (the limits, also in the doc): two small integers that differ read as two denormals and pass rule 1 (their difference is far
// below atol); they are counted (`subnormal`) so that a report with many of them is read with suspicion. Where the data does not look like binary32
// (sparse arrays, floats outside 2^-40 .. 2^40, structures), a difference in the low word of an 8-byte value passes reading 2. A word within tolerance
// can still be a wrong answer: the tolerance says the numbers are close, not that the program would take the same branch on them.
// BINARY16 DATA IS NOT JUDGED BY THE GUESS AT ALL: two halves in a word read, under reading 1, as one float whose exponent is the high half's top bits,
// so halves below about 0.25 make a float below atol and every difference passes, and a difference in the low half is a change of low mantissa bits.
// Memory that holds halves has to be compared with Reading::F16 (VGPU_HOST_GPU_TOL_TYPE=f16), which reads every word as two halves and nothing else.
//
// No simulator state and no CUDA: unit tested on any machine (tests/unit/test_host_gpu_tol.cpp).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace vgpu::hostgpu {

// How a differing word is read. Auto is the guessing order above; the others say what the memory holds and read every word that way and no other
// (VGPU_HOST_GPU_TOL_TYPE), for data the guess cannot judge: binary16 above all.
enum class Reading : uint8_t { Auto, F32, F64, F16 };

struct Tolerance {
  double atol = 1e-8;
  double rtol = 1e-5;
  Reading reading = Reading::Auto;
};

// VGPU_HOST_GPU_TOL_TYPE: auto, f32, f64 or f16. False for anything else.
inline bool parse_reading(const char* text, Reading* out) {
  if (!text) return false;
  const char* const names[] = {"auto", "f32", "f64", "f16"};
  for (unsigned i = 0; i < 4; ++i)
    if (!std::strcmp(text, names[i])) {
      *out = static_cast<Reading>(i);
      return true;
    }
  return false;
}

// VGPU_HOST_GPU_ATOL / VGPU_HOST_GPU_RTOL: a finite number >= 0 and nothing after it. False for anything else (the caller says so and does not guess).
inline bool parse_tolerance(const char* text, double* out) {
  if (!text || !*text) return false;
  char* end = nullptr;
  const double v = std::strtod(text, &end);
  if (end == text || *end != '\0') return false;
  if (!(v >= 0) || std::isinf(v)) return false;   // (a NaN fails v >= 0)
  *out = v;
  return true;
}

// binary16 to double, exactly (no F16C needed).
inline double half_to_double(uint16_t h) {
  const int sign = h >> 15, exp = (h >> 10) & 0x1f, frac = h & 0x3ff;
  double v;
  if (exp == 0x1f) v = frac ? std::nan("") : HUGE_VAL;
  else if (exp == 0) v = std::ldexp(static_cast<double>(frac), -24);
  else v = std::ldexp(static_cast<double>(frac | 0x400), exp - 25);
  return sign ? -v : v;
}

// The rule for two values of one reading. `abs_out` and `rel_out` get the difference when both are finite (rel is 0 when sim is 0).
inline bool values_close(double card, double sim, const Tolerance& t, double* abs_out = nullptr, double* rel_out = nullptr) {
  if (abs_out) *abs_out = 0;
  if (rel_out) *rel_out = 0;
  const bool cn = std::isnan(card), sn = std::isnan(sim);
  if (cn || sn) return cn && sn;
  if (std::isinf(card) || std::isinf(sim)) return card == sim;
  const double d = std::fabs(card - sim);
  if (abs_out) *abs_out = d;
  if (rel_out) *rel_out = sim != 0 ? d / std::fabs(sim) : 0;
  return d <= t.atol + t.rtol * std::fabs(sim);
}

// The rule for two binary16 values, with one difference: a NaN matches only the same bits. The low 16 bits of a float look like a half NaN one time
// in 32, so "any NaN matches any NaN" would pass one in 32 of the float differences that reading 1 had just refused.
inline bool halves_close(uint16_t card, uint16_t sim, const Tolerance& t, double* abs_out, double* rel_out) {
  *abs_out = *rel_out = 0;
  if (card == sim) return true;
  const auto nan = [](uint16_t h) { return (h & 0x7c00) == 0x7c00 && (h & 0x3ff) != 0; };
  if (nan(card) || nan(sim)) return false;
  return values_close(half_to_double(card), half_to_double(sim), t, abs_out, rel_out);
}

struct TolReport {
  uint64_t words = 0;        // 4-byte words compared (a partial word at either end of a range counts as one)
  uint64_t different = 0;    // of them, words whose bits differ
  uint64_t outside = 0;      // of those, words outside tolerance
  uint64_t subnormal = 0;    // of the different words inside tolerance: both sides zero or denormal as binary32 (what two small integers look like)
  double max_abs = 0;        // largest differences of the different words, each under the reading that passed (binary32 for a word outside tolerance,
  double max_rel = 0;        // when both sides are finite)
  bool has_first = false;    // the first word outside tolerance
  uint64_t first_addr = 0;
  uint32_t first_card = 0, first_sim = 0;

  void add(const TolReport& o) {
    words += o.words;
    different += o.different;
    outside += o.outside;
    subnormal += o.subnormal;
    if (o.max_abs > max_abs) max_abs = o.max_abs;
    if (o.max_rel > max_rel) max_rel = o.max_rel;
    if (!has_first && o.has_first) {
      has_first = true;
      first_addr = o.first_addr;
      first_card = o.first_card;
      first_sim = o.first_sim;
    }
  }
};

namespace tol_detail {
inline uint32_t load32(const uint8_t* p) { uint32_t v; std::memcpy(&v, p, 4); return v; }
inline double as_f32(uint32_t bits) { float f; std::memcpy(&f, &bits, 4); return static_cast<double>(f); }
inline double as_f64(const uint8_t* p) { double d; std::memcpy(&d, p, 8); return d; }
inline void note_max(TolReport& r, double a, double rel) {
  if (a > r.max_abs) r.max_abs = a;
  if (rel > r.max_rel) r.max_rel = rel;
}
}  // namespace tol_detail

// The words compare_words counts over [addr, addr + len): a partial word at either end counts as one.
inline uint64_t tol_word_count(uint64_t addr, uint64_t len) {
  if (!len) return 0;
  uint64_t n = 0;
  if (const uint64_t mis = addr & 3) {
    const uint64_t head = std::min<uint64_t>(4 - mis, len);
    ++n;
    len -= head;
  }
  return n + len / 4 + (len % 4 != 0);
}

// Whether the simulator's bytes around the word at sim[at] look like an array of binary32. The 256 bytes (aligned by address) that hold the word are read
// as 8-byte pairs, and the LOW word of each pair is judged: in an array of floats it is a float like any other, in an array of doubles it is the low
// mantissa bits, whose "exponent" is anything. A low word is plausible when its exponent field is that of a magnitude in 2^-40 .. 2^40. With 4 or more
// nonzero low words, three in four plausible says binary32. With fewer the word itself decides: a nonzero, plausible word in the low half.
inline constexpr uint64_t kTolWindow = 256;
inline bool plausible_f32(uint32_t bits) {
  const uint32_t e = (bits >> 23) & 0xff;
  return e >= 127 - 40 && e <= 127 + 40;
}
inline bool looks_like_f32(const uint8_t* sim, uint64_t addr, uint64_t len, uint64_t at) {
  const uint64_t w = addr + at;
  const uint64_t lo = std::max<uint64_t>(w & ~(kTolWindow - 1), (addr + 7) & ~uint64_t{7});
  const uint64_t hi = std::min<uint64_t>((w & ~(kTolWindow - 1)) + kTolWindow, addr + len);
  unsigned nonzero = 0, plausible = 0;
  for (uint64_t p = lo; p + 8 <= hi; p += 8) {
    const uint32_t v = tol_detail::load32(sim + (p - addr));
    if (!v) continue;
    ++nonzero;
    plausible += plausible_f32(v);
  }
  if (nonzero >= 4) return plausible * 4 >= nonzero * 3;
  const uint32_t self = tol_detail::load32(sim + at);
  return !(w & 4) && self != 0 && plausible_f32(self);
}

// Compares card[0, len) with sim[0, len), the bytes of device addresses [addr, addr + len), and adds what it finds to `r`.
inline void compare_words(const uint8_t* card, const uint8_t* sim, uint64_t addr, uint64_t len, const Tolerance& t, TolReport& r) {
  using namespace tol_detail;
  if (!len) return;
  const auto outside_word = [&](uint64_t at, uint32_t c, uint32_t s) {
    ++r.outside;
    if (!r.has_first) {
      r.has_first = true;
      r.first_addr = addr + at;
      r.first_card = c;
      r.first_sim = s;
    }
  };
  // A run of fewer than 4 bytes at either end: exact, and shown as the bytes that are there (the rest of the word is zero).
  const auto partial = [&](uint64_t at, uint64_t n) {
    ++r.words;
    if (std::memcmp(card + at, sim + at, n) == 0) return;
    ++r.different;
    uint32_t c = 0, s = 0;
    std::memcpy(&c, card + at, n);
    std::memcpy(&s, sim + at, n);
    outside_word(at, c, s);
  };
  uint64_t at = 0;
  if (const uint64_t mis = addr & 3) {
    const uint64_t n = std::min<uint64_t>(4 - mis, len);
    partial(0, n);
    at = n;
  }
  for (; at + 4 <= len; at += 4) {
    ++r.words;
    const uint32_t c = load32(card + at), s = load32(sim + at);
    if (c == s) continue;
    ++r.different;
    double a = 0, rel = 0;
    const uint64_t w = addr + at;
    const bool high_half = (w & 4) != 0;   // of the aligned 8 bytes that hold this word
    const uint64_t d_at = high_half ? at - 4 : at;   // (only used when the 8 bytes lie inside the range)
    const bool pair_inside = !(high_half && at < 4) && d_at + 8 <= len;
    // The largest relative difference is taken over differences above atol: below it the absolute term alone passes the word, and a ratio of two
    // near-zero readings (the low bits of a double, two small integers) says nothing.
    const auto passed = [&](double abs_d, double rel_d) { note_max(r, abs_d, abs_d > t.atol ? rel_d : 0); };
    const auto halves = [&] {
      double a_lo = 0, rel_lo = 0, a_hi = 0, rel_hi = 0;
      if (!halves_close(static_cast<uint16_t>(c), static_cast<uint16_t>(s), t, &a_lo, &rel_lo) ||
          !halves_close(static_cast<uint16_t>(c >> 16), static_cast<uint16_t>(s >> 16), t, &a_hi, &rel_hi))
        return false;
      passed(a_lo, rel_lo);
      passed(a_hi, rel_hi);
      return true;
    };
    if (t.reading != Reading::Auto) {   // the caller said what the memory holds: that reading and no other
      bool ok;
      if (t.reading == Reading::F16) ok = halves();
      else if (t.reading == Reading::F64) {
        ok = pair_inside && values_close(as_f64(card + d_at), as_f64(sim + d_at), t, &a, &rel);   // (a word with no whole 8 bytes in the range is exact)
        if (ok) passed(a, rel);
      } else {
        ok = values_close(as_f32(c), as_f32(s), t, &a, &rel);
        if (ok) passed(a, rel);
        else note_max(r, a, rel);
      }
      if (!ok) outside_word(at, c, s);
      continue;
    }
    if (values_close(as_f32(c), as_f32(s), t, &a, &rel)) {
      passed(a, rel);
      if ((c & 0x7f800000u) == 0 && (s & 0x7f800000u) == 0) ++r.subnormal;
      continue;
    }
    const double a32 = a, rel32 = rel;
    // Reading 2 only where the data is not binary32, and never for a word that is a binary32 infinity or NaN on one side in the high half of the
    // 8 bytes (a double that large does not occur; a float that overflowed on one side does). The aligned 8 bytes must lie inside the range.
    const bool special32 = (c & 0x7f800000u) == 0x7f800000u || (s & 0x7f800000u) == 0x7f800000u;
    if (pair_inside && !(high_half && special32) && !looks_like_f32(sim, addr, len, at) && values_close(as_f64(card + d_at), as_f64(sim + d_at), t, &a, &rel)) {
      passed(a, rel);
      continue;
    }
    // Reading 3, binary16: both halves. Not for a word that is a binary32 infinity or NaN on one side: +inf and -inf read as two half NaNs.
    if (!special32 && halves()) continue;
    note_max(r, a32, rel32);
    outside_word(at, c, s);
  }
  if (at < len) partial(at, len - at);
}

}  // namespace vgpu::hostgpu
