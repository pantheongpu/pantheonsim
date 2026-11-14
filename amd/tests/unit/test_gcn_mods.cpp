// The instruction modifiers the clamp rounds (#613, the integer clamp) did not cover: SDWA's clamp, output modifier, source selects with
// and without sign, destination select and dst_unused (gfx9, gfx10); DPP's negate and absolute value, bank mask and bound control, and
// gfx11's VOP3 DPP forms with clamp and output modifier; VOP3P's op_sel, op_sel_hi, neg_lo, neg_hi and clamp on the packed float and
// integer instructions and the mixed-precision ones; and v_div_scale's clamp and output modifier, which the guides do not mention: the
// RX 6800 applies them as to any VOP3 float result.
//
// The kernels are amd/tests/data/asm_mods.s, each run in three modes (IEEE on with denormals kept, as compilers make kernels; IEEE off;
// IEEE off with denormals flushed). The expected values below are the ISA guides' pseudocode and section 6.5 (the output modifier, then the
// clamp; the output modifier does nothing with IEEE mode on or output denormals kept), and an RX 6800 (gfx1030) running the gfx1030 object
// gave exactly them. gfx90a, gfx942 and gfx1100 follow the same rules; no card of theirs has run the kernels.
// Run with the tier off and on.
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <vector>

#include "vgpu/amd_codeobject.hpp"
#include "vgpu/amd_exec.hpp"
#include "vgpu/amd_gcn.hpp"
#include "vtest.hpp"

using namespace vgpu;

namespace {

// --- what the instructions give (the model the card's results were checked against) ---
constexpr int kItems = 1024;
enum class Body { Sdwa, Pk, Dpp };
enum Tg { kGfx90a, kGfx942, kGfx1030, kGfx1100 };
struct Mode {
  bool ieee, flush;   // IEEE mode; denormals flushed (single precision, and half and double)
};
constexpr int ops_of(Body b) { return b == Body::Sdwa ? 50 : b == Body::Pk ? 36 : 24; }

constexpr uint32_t kV[32] = {
    0x00000000u, 0x80000000u, 0x3f800000u, 0xbf800000u, 0x3f000000u, 0x40000000u, 0x3f7fffffu, 0x3f800001u,
    0x7f800000u, 0xff800000u, 0x7fc00000u, 0x7f800001u, 0x00000001u, 0x807fffffu, 0x00800000u, 0x7f7fffffu,
    0xbe800000u, 0x41200000u, 0x3c00bc00u, 0x7c003800u, 0x7e00fc00u, 0x00013bffu, 0x3c017bffu, 0x80008001u,
    0xffff0000u, 0x0000ffffu, 0x7fff8000u, 0x12345678u, 0xdeadbeefu, 0x00ff7f80u, 0xfe017f01u, 0x7d003c01u};
struct In {
  uint32_t a, b, c, d, old;
};
In input(int t) {
  return {kV[t & 31], kV[(t >> 5) & 31], kV[(t * 7 + 3) & 31], kV[(t * 11 + 5) & 31], kV[(t * 13 + 9) & 31] ^ 0x5a5a5a5au};
}

// What one work-item's three dwords should be. `mask` is the bits of lo compared; `nan` the fields that need only be some NaN
// (bit 0: lo's low half, a half; bit 1: lo's high half, a half; bit 2: lo, a float; bit 3: hi, a float; bit 4: hi:lo, a double): a NaN's
// payload is not checked.
struct Want {
  uint32_t lo = 0, hi = 0, vcc = 0, mask = ~0u, nan = 0;
  bool any = false;   // (not established: anything goes)
};

using H = _Float16;
float f32(uint32_t b) { return std::bit_cast<float>(b); }
uint32_t b32(float f) { return std::bit_cast<uint32_t>(f); }
H f16(uint32_t b) { return std::bit_cast<H>(static_cast<uint16_t>(b)); }
uint32_t b16(H h) { return std::bit_cast<uint16_t>(h); }
bool sub32(float f) { return std::fpclassify(f) == FP_SUBNORMAL; }
bool sub16(H h) { return (b16(h) & 0x7C00) == 0 && (b16(h) & 0x3FF); }
H hneg(H h) { return f16(b16(h) ^ 0x8000u); }   // (bit by bit: arithmetic would quiet a signalling NaN)
H hnan() { return f16(0x7E00u); }
float in32(uint32_t b, Mode m) {   // a single-precision source
  const float f = f32(b);
  return m.flush && sub32(f) ? std::copysign(0.0f, f) : f;
}
H in16(uint32_t b, Mode m) {
  const H h = f16(b);
  return m.flush && sub16(h) ? static_cast<H>(std::copysign(0.0f, static_cast<float>(h))) : h;
}
float fneg(float f, bool n, bool a) {
  if (a) f = std::fabs(f);
  return n ? -f : f;
}
// A float result: denormals flushed where the mode says, the output modifier, then the clamp ([0, 1], a NaN to 0: DX10_CLAMP is on).
// The output modifier applies only with IEEE off and output denormals flushed (CDNA3 guide, 6.5); applied, a zero becomes +0 ("-0 is
// flushed to +0") and a result it makes denormal is flushed keeping its sign (the RX 6800: -2^-126 div:2 is -0).
float out32(float v, Mode m, int omod, bool clamp) {
  if (m.flush && sub32(v)) v = std::copysign(0.0f, v);
  if (omod && !m.ieee && m.flush) {
    if (v == 0) v = 0.0f;
    v *= omod == 1 ? 2.0f : omod == 2 ? 4.0f : 0.5f;
    if (sub32(v)) v = std::copysign(0.0f, v);
  }
  if (clamp) v = std::isnan(v) || v <= 0 ? 0.0f : v > 1 ? 1.0f : v;
  return v;
}
double out64(double v, Mode m, int omod, bool clamp) {   // (double precision takes the 16/64-bit denormal mode, which `flush` sets too)
  if (m.flush && std::fpclassify(v) == FP_SUBNORMAL) v = std::copysign(0.0, v);
  if (omod && !m.ieee && m.flush) {
    if (v == 0) v = 0.0;
    v *= omod == 1 ? 2.0 : omod == 2 ? 4.0 : 0.5;
    if (std::fpclassify(v) == FP_SUBNORMAL) v = std::copysign(0.0, v);
  }
  if (clamp) v = std::isnan(v) || v <= 0 ? 0.0 : v > 1 ? 1.0 : v;
  return v;
}
H out16(H v, Mode m, int omod, bool clamp) {
  float f = static_cast<float>(v);
  if (m.flush && sub16(v)) f = std::copysign(0.0f, f);
  if (omod && !m.ieee && m.flush) {
    if (f == 0) f = 0.0f;
    f = static_cast<float>(static_cast<H>(f * (omod == 1 ? 2.0f : omod == 2 ? 4.0f : 0.5f)));
    if (sub16(static_cast<H>(f))) f = std::copysign(0.0f, f);
  }
  if (clamp) f = std::isnan(f) || f <= 0 ? 0.0f : f > 1 ? 1.0f : f;
  return static_cast<H>(f);
}
H r16(double v) { return static_cast<H>(v); }   // (one rounding, to nearest even)
// A single-precision multiply and fused multiply-add. With denormals flushed, a result that is below the least normal before it is
// rounded is a zero of its sign, though it would round up to the least normal (the RX 6800: 2^-126 times 0x3f7fffff is 0).
float mul32(float x, float y, Mode m) {
  const double p = double(x) * y;
  return m.flush && p != 0 && std::fabs(p) < 0x1p-126 ? static_cast<float>(std::copysign(0.0, p)) : static_cast<float>(p);
}
float fma32(float x, float y, float z, Mode m) {
  const double p = std::fma(double(x), double(y), double(z));
  return m.flush && p != 0 && std::fabs(p) < 0x1p-126 ? static_cast<float>(std::copysign(0.0, p)) : std::fmaf(x, y, z);
}
// IEEE-mode maximum and minimum (a signalling NaN gives a NaN, a quiet one the other operand; -0 below +0); IEEE off, a NaN gives the other.
template <typename T>
bool snan(T v) {
  if constexpr (sizeof(T) == 4) return std::isnan(v) && !(b32(v) & 0x400000u);
  else return std::isnan(static_cast<float>(v)) && !(b16(v) & 0x200u);
}
template <typename T>
T maxmin(T x, T y, bool is_max, bool ieee) {
  if (ieee && (snan(x) || snan(y))) {
    if constexpr (sizeof(T) == 4) return std::numeric_limits<float>::quiet_NaN();
    else return hnan();
  }
  if (std::isnan(static_cast<float>(x))) return y;
  if (std::isnan(static_cast<float>(y))) return x;
  if (x == y) return (std::signbit(static_cast<float>(x)) == is_max) ? y : x;
  return (x > y) == is_max ? x : y;
}
// SDWA: the part of a source a select names, with or without its sign; the destination's part and the rest of it as dst_unused says.
enum Sel { B0, B1, B2, B3, W0, W1, DW };
uint32_t pick(uint32_t v, Sel s, bool sext = false) {
  if (s <= B3) {
    const uint32_t x = (v >> (8 * s)) & 0xFF;
    return sext ? static_cast<uint32_t>(static_cast<int8_t>(x)) : x;
  }
  if (s <= W1) {
    const uint32_t x = (v >> (s == W0 ? 0 : 16)) & 0xFFFF;
    return sext ? static_cast<uint32_t>(static_cast<int16_t>(x)) : x;
  }
  return v;
}
enum Unused { Pad, Sext, Keep };
uint32_t place(uint32_t r, uint32_t old, Sel s, Unused u) {
  if (s == DW) return r;
  const uint32_t mask = s <= B3 ? 0xFFu << (8 * s) : s == W0 ? 0xFFFFu : 0xFFFF0000u;
  const uint32_t at = s <= B3 ? 8 * s : s == W0 ? 0 : 16;
  uint32_t v = (r << at) & mask;
  if (u == Sext && (v & (mask & ~(mask >> 1)))) v |= ~mask & ~((1u << at) - 1);
  if (u == Keep) v |= old & ~mask;
  return v;
}
int64_t s32(uint32_t v) { return static_cast<int32_t>(v); }
int64_t s16(uint32_t v) { return static_cast<int16_t>(static_cast<uint16_t>(v)); }
int64_t sat(int64_t v, int64_t lo, int64_t hi) { return v < lo ? lo : v > hi ? hi : v; }
uint32_t i24(uint32_t v) { return static_cast<uint32_t>(static_cast<int32_t>(v << 8) >> 8); }

Want w32(uint32_t v) { return {v}; }
Want wf(float f) {
  Want w{b32(f)};
  if (std::isnan(f)) w.nan = 4;
  return w;
}
// A half result placed in the destination: `at` 0 or 16, the rest as given.
Want wh(H h, uint32_t rest, int at) {
  Want w{(b16(h) << at) | rest};
  if (std::isnan(static_cast<float>(h))) w.nan = at ? 2 : 1;
  return w;
}

// The lane a DPP control makes lane t read (within its row of 16, or its group of eight for dpp8), or -1 where there is none.
int quad(int t, const int (&p)[4]) { return (t & ~3) + p[t & 3]; }
int row_shift(int t, int by) {   // row_shl:n is +n, row_shr:n is -n
  const int p = (t & 15) + by;
  return p < 0 || p > 15 ? -1 : (t & ~15) + p;
}
int row_ror(int t, int by) { return (t & ~15) + (((t & 15) - by) & 15); }
int dpp8(int t, const int (&s)[8]) { return (t & ~7) + s[t & 7]; }
constexpr int kSwap[4] = {1, 0, 3, 2};
constexpr int kSwap8[8] = {1, 0, 3, 2, 5, 4, 7, 6};
constexpr int kRev8[8] = {7, 6, 5, 4, 3, 2, 1, 0};

Want sdwa(int op, int t, Tg tg, Mode m) {
  const In x = input(t);
  const uint32_t a = x.a, b = x.b, old = x.old;
  const bool gfx10 = tg == kGfx1030;
  const auto f = [&](uint32_t v) { return in32(v, m); };
  const auto h = [&](uint32_t v) { return in16(v, m); };
  switch (op) {
    case 0: return wf(out32(f(a) + f(b), m, 0, true));
    case 1: return wf(out32(f(a) + f(b), m, 1, false));
    case 2: return wf(out32(f(a) + f(b), m, 2, true));
    case 3: return wf(out32(mul32(f(a), f(b), m), m, 3, false));
    case 4: return wf(out32(f(pick(a, W1)) - f(pick(b, W0)), m, 0, true));
    case 5: return wf(out32(maxmin(f(a), f(b), true, m.ieee), m, 0, true));
    case 6: return w32(place(b32(out32(f(a) + f(b), m, 0, true)), old, W1, Keep));
    case 7: return wf(out32(-f(a) + std::fabs(f(b)), m, 0, true));
    case 8: return wh(out16(r16(double(h(a >> 16)) + h(b >> 16)), m, 0, true), old & 0xFFFF, 16);
    case 9: return wh(out16(r16(double(h(a >> 16)) * h(b)), m, 1, false), 0, 0);
    case 10: return wh(out16(r16(double(h(a)) + h(b)), m, 2, true), 0, 0);
    case 11: return wh(out16(maxmin(h(a), h(b >> 16), true, m.ieee), m, 0, true), 0, 0);
    case 12: return w32(place(b16(out16(r16(double(h(a)) - h(b)), m, 3, true)), old, W0, Sext));
    case 13: return w32(place(b16(out16(static_cast<H>(f(a)), m, 0, true)), old, W1, Keep));
    case 14: return wf(out32(static_cast<float>(h(a >> 16)), m, 0, true));
    case 15: return wf(out32(static_cast<float>(h(a)), m, 1, false));
    case 16: return wf(out32(static_cast<float>(pick(a, B1)), m, 0, true));
    case 17: return wf(out32(static_cast<float>(static_cast<int32_t>(pick(a, W1, true))), m, 1, false));
    case 18: {   // (the clamp of a conversion to an integer: the conversion saturates already)
      const float v = f(a);
      return w32(std::isnan(v) || v <= 0 ? 0u : v >= 4294967296.0f ? 0xFFFFFFFFu : static_cast<uint32_t>(v));
    }
    case 19: {
      const float v = f(a);
      return w32(std::isnan(v) ? 0u : v <= -2147483648.0f ? 0x80000000u : v >= 2147483648.0f ? 0x7FFFFFFFu
                                                                         : static_cast<uint32_t>(static_cast<int32_t>(v)));
    }
    case 20: return w32(static_cast<uint32_t>(std::min<uint64_t>(uint64_t{pick(a, W1)} + b, 0xFFFFFFFFu)));
    case 21: return w32(pick(a, B0) < pick(b, B1) ? 0 : pick(a, B0) - pick(b, B1));
    case 22: return w32(b < a ? 0 : b - a);
    case 23: return w32(static_cast<uint32_t>(std::min<uint64_t>(uint64_t{pick(a, B0, true)} + b, 0xFFFFFFFFu)));
    case 24: {
      const uint64_t s = uint64_t{a} + b;
      return {static_cast<uint32_t>(std::min<uint64_t>(s, 0xFFFFFFFFu)), 0, s >> 32 ? 1u : 0u};
    }
    case 25: return w32(static_cast<uint32_t>(std::min<uint64_t>(uint64_t{a & 0xFFFFFF} * (b & 0xFFFFFF), 0xFFFFFFFFu)));
    case 26: return w32(static_cast<uint32_t>(sat(s32(i24(pick(a, W1, true))) * s32(i24(b)), INT32_MIN, INT32_MAX)));
    case 27: return w32(place(pick(a, W1), old, B1, Sext));
    case 28: return w32(place(pick(a, B3, true), old, W1, Pad));
    case 29: return w32(place(pick(a, B3) & pick(b, W0), old, B2, Keep));
    case 30: return w32(place(pick(a, B0, true) ^ b, old, B3, Sext));
    case 31: return w32(static_cast<uint32_t>(static_cast<int32_t>(pick(b, W1, true)) >> (pick(a, B0) & 31)));
    case 32: return w32(static_cast<uint32_t>(std::min(s32(pick(a, B1, true)), s32(pick(b, W0, true)))));
    case 33: return {old, 0, s32(pick(a, B0, true)) > s32(pick(b, W1)) ? 1u : 0u};
    case 34: return {old, 0, h(a >> 16) < h(b) ? 1u : 0u};
    case 35: return gfx10 ? Want{old} : w32(place(static_cast<uint32_t>(std::min<int64_t>((a >> 16) + (b & 0xFFFF), 0xFFFF)), old, W1, Keep));
    case 36: return gfx10 ? Want{old} : w32(static_cast<uint32_t>(std::max<int64_t>(int64_t{a & 0xFFFF} - (b & 0xFFFF), 0)));
    case 37: return gfx10 ? Want{old} : w32(static_cast<uint32_t>(std::min<int64_t>(int64_t{a & 0xFFFF} * (b & 0xFFFF), 0xFFFF)));
    case 38: return gfx10 ? Want{old} : w32(static_cast<uint32_t>(std::max(s16(pick(a, B0, true)), s16(b))) & 0xFFFF);
    case 39: case 40: case 41: return {};   // (v_div_scale_f32: compared with each other, below)
    case 42: return wf(out32(f(pick(a, B1)) + f(pick(b, W1)), m, 0, true));
    case 43: return w32(place(b32(out32(mul32(f(a), f(b), m), m, 1, true)), old, B0, Sext));
    case 44: return wf(out32(-f(input(quad(t, kSwap)).a) + std::fabs(f(b)), m, 0, false));
    case 45: {
      const int from = row_shift(t, -1);
      if (((t & 15) >> 2) & 1 || from < 0) return w32(old);   // (bank_mask 0x5: banks 1 and 3 not written; no lane to read, not written)
      return w32(input(from).a + b);
    }
    case 46: {
      const int from = row_shift(t, -1);
      return w32(from < 0 ? 0 : input(from).a);
    }
    case 47: return wf(out32(maxmin(f(a), f(b), true, m.ieee), m, 0, false));
    case 48: return wf(out32(maxmin(f(a), f(b), false, m.ieee), m, 0, false));
    case 49: {   // (the RX 6800 keeps the destination's high half, where the executor zeroes it as gfx9 does: not compared, not fixed here)
      Want w = wh(out16(maxmin(h(a), h(b), false, m.ieee), m, 0, false), gfx10 ? old & 0xFFFF0000u : 0, 0);
      if (gfx10) w.mask = 0xFFFF;
      return w;
    }
  }
  return {};
}

// A packed source's half: op_sel (low result) or op_sel_hi (high result) picks the half, neg_lo or neg_hi negates it.
struct Pk {
  int sel, hi, nlo, nhi;   // bit k for source k
};
uint32_t half_of(uint32_t v, const Pk& p, int k, int r) { return (v >> (16 * (((r ? p.hi : p.sel) >> k) & 1))) & 0xFFFF; }
bool neg_of(const Pk& p, int k, int r) { return ((r ? p.nhi : p.nlo) >> k) & 1; }

Want pk(int op, int t, Tg tg, Mode m) {
  const In x = input(t);
  const uint32_t src[3] = {x.a, x.b, x.c};
  const bool cdna = tg == kGfx90a || tg == kGfx942;
  // A packed half-precision instruction, each half `fn` of the halves its sources' modifiers name.
  // gfx11's minimum and maximum of a NaN, or of -0 and +0: not established (no card, no guide here), so not compared.
  const auto fp16 = [&](const Pk& p, bool clamp, auto fn, bool minmax = false) {
    Want w;
    for (int r = 0; r < 2; ++r) {
      H s[3];
      for (int k = 0; k < 3; ++k) {
        s[k] = in16(half_of(src[k], p, k, r), m);
        if (neg_of(p, k, r)) s[k] = hneg(s[k]);
      }
      const H v = out16(fn(s[0], s[1], s[2]), m, 0, clamp);
      if (minmax && tg == kGfx1100 && (std::isnan(static_cast<float>(s[0])) || std::isnan(static_cast<float>(s[1])) || (s[0] == 0 && s[1] == 0))) {
        w.mask &= ~(0xFFFFu << (16 * r));
        continue;
      }
      w.lo |= b16(v) << (16 * r);
      if (std::isnan(static_cast<float>(v))) w.nan |= 1u << r;
    }
    return w;
  };
  // A packed integer instruction: the halves op_sel and op_sel_hi pick, clamped to `lo`..`hi`. neg_lo and neg_hi flip an integer half's top bit,
  // as they flip a half's sign (the RX 6800: v_pk_add_u16 0, 0 neg_lo:[1,0] neg_hi:[0,1] is 0x80008000).
  const auto int16 = [&](const Pk& p, int64_t lo, int64_t hi, bool is_signed, auto fn) {
    Want w;
    for (int r = 0; r < 2; ++r) {
      int64_t s[3];
      for (int k = 0; k < 3; ++k) {
        const uint32_t v = half_of(src[k], p, k, r) ^ (neg_of(p, k, r) ? 0x8000u : 0);
        s[k] = is_signed ? s16(v) : v;
      }
      w.lo |= (static_cast<uint32_t>(sat(fn(s[0], s[1], s[2]), lo, hi)) & 0xFFFF) << (16 * r);
    }
    return w;
  };
  constexpr int64_t kWrapLo = INT64_MIN, kWrapHi = INT64_MAX;
  const auto add = [](H a, H b, H) { return r16(double(a) + b); };
  const auto mul = [](H a, H b, H) { return r16(double(a) * b); };
  const auto fma = [](H a, H b, H c) { return r16(double(a) * b + c); };
  // fma_mix: a source is a half (op_sel_hi's bit set; op_sel's picks the high one) or a float; neg_lo negates, neg_hi takes the absolute.
  const auto mix = [&](const Pk& p, int k) {
    const bool half = (p.hi >> k) & 1;
    float v = half ? static_cast<float>(in16(src[k] >> (16 * ((p.sel >> k) & 1)), m)) : in32(src[k], m);
    return fneg(v, (p.nlo >> k) & 1, (p.nhi >> k) & 1);
  };
  switch (op) {
    case 0: return fp16({1, 2, 1, 2}, true, add);
    case 1: return fp16({0, 3, 2, 1}, false, add);
    case 2: return fp16({3, 0, 0, 0}, true, mul);
    case 3: return fp16({5, 2, 5, 6}, true, fma);
    case 4: return fp16({0, 7, 4, 3}, false, fma);
    case 5: return fp16({0, 3, 3, 0}, true, [&](H a, H b, H) { return maxmin(a, b, true, m.ieee); }, true);
    case 6: return fp16({2, 3, 0, 1}, true, [&](H a, H b, H) { return maxmin(a, b, false, m.ieee); }, true);
    case 7: return int16({1, 2, 0, 0}, 0, 0xFFFF, false, [](int64_t a, int64_t b, int64_t) { return a + b; });
    case 8: return int16({0, 3, 1, 2}, kWrapLo, kWrapHi, false, [](int64_t a, int64_t b, int64_t) { return a + b; });
    case 9: return int16({0, 3, 3, 0}, -32768, 32767, true, [](int64_t a, int64_t b, int64_t) { return a + b; });
    case 10: return int16({2, 1, 0, 0}, -32768, 32767, true, [](int64_t a, int64_t b, int64_t) { return a - b; });
    case 11: return int16({3, 3, 0, 0}, 0, 0xFFFF, false, [](int64_t a, int64_t b, int64_t) { return a - b; });
    case 12: return int16({3, 0, 0, 0}, kWrapLo, kWrapHi, false, [](int64_t a, int64_t b, int64_t) { return a * b; });
    case 13: return int16({5, 0, 0, 0}, 0, 0xFFFF, false, [](int64_t a, int64_t b, int64_t c) { return a * b + c; });
    case 14: return int16({6, 5, 0, 0}, -32768, 32767, true, [](int64_t a, int64_t b, int64_t c) { return a * b + c; });
    case 15: return int16({3, 3, 0, 0}, -32768, 32767, true, [](int64_t a, int64_t b, int64_t) { return std::max(a, b); });
    case 16: return int16({0, 2, 0, 0}, kWrapLo, kWrapHi, false, [](int64_t a, int64_t b, int64_t) { return std::min(a, b); });
    case 17: return int16({1, 3, 0, 0}, kWrapLo, kWrapHi, false, [](int64_t a, int64_t b, int64_t) { return b << (a & 15); });
    case 18: return int16({0, 0, 0, 0}, kWrapLo, kWrapHi, true, [](int64_t a, int64_t b, int64_t) { return b >> (a & 15); });
    case 19: return int16({2, 1, 0, 0}, kWrapLo, kWrapHi, false, [](int64_t a, int64_t b, int64_t) { return b >> (a & 15); });
    case 20: { const Pk p{5, 5, 0, 0}; return wf(out32(fma32(mix(p, 0), mix(p, 1), mix(p, 2), m), m, 0, true)); }
    case 21: { const Pk p{0, 3, 1, 2}; return wf(out32(fma32(mix(p, 0), mix(p, 1), mix(p, 2), m), m, 0, false)); }
    case 22: {
      const Pk p{0, 3, 1, 2};
      return wh(out16(static_cast<H>(std::fmaf(mix(p, 0), mix(p, 1), mix(p, 2))), m, 0, true), x.old & 0xFFFF0000u, 0);
    }
    case 23: {
      const Pk p{2, 7, 0, 0};
      return wh(out16(static_cast<H>(std::fmaf(mix(p, 0), mix(p, 1), mix(p, 2))), m, 0, true), x.old & 0xFFFFu, 16);
    }
    case 24: {
      if (tg == kGfx90a) return w32(x.old);
      // D = a.lo * b.lo + a.hi * b.hi + c, neg_lo negating the low halves and neg_hi the high ones of a and b; the clamp does nothing
      // (test_amd_gcn_asm's a_half_dot_product_into_a_float_is_not_held_to_one_by_its_clamp; the RX 6800 too). The CDNA3 guide flushes
      // denormals whatever the mode (6.4); the RX 6800 keeps half denormals and flushes a single one, c, whatever the mode. gfx11: not
      // established.
      if (tg == kGfx1100) return Want{.any = true};
      const bool guide = tg != kGfx1030;
      const auto fl = [&](double v) { return guide && std::fabs(v) < 6.103515625e-05 ? std::copysign(0.0, v) : v; };
      const double a0 = -fl(static_cast<double>(f16(x.a))), a1 = fl(static_cast<double>(f16(x.a >> 16)));
      const double b0 = fl(static_cast<double>(f16(x.b))), b1 = -fl(static_cast<double>(f16(x.b >> 16)));
      float c = f32(x.c);
      if (sub32(c)) c = std::copysign(0.0f, c);
      float s = static_cast<float>(a0 * b0 + a1 * b1 + c);
      if (sub32(s)) s = std::copysign(0.0f, s);
      return wf(out32(s, Mode{m.ieee, false}, 0, false));
    }
    case 25: return fp16({2, 1, 3, 3}, false, mul);
    case 26: return fp16({1, 2, 0, 0}, false, [&](H a, H b, H) { return maxmin(a, b, true, m.ieee); }, true);
    case 27: {   // (an inline float constant is the half in the low half, zero above it)
      Want w;
      const H lo = out16(r16(double(in16(x.a, m)) + 1.0), m, 0, true), hi = out16(r16(double(in16(x.a >> 16, m)) + 0.0), m, 0, true);
      w.lo = b16(lo) | b16(hi) << 16;
      return w;
    }
    case 28: return w32(0xFFFFFFFFu);   // (an integer constant is its 32 bits, -1 in both halves: every sum saturates)
    case 29: case 30: case 31: {
      if (!cdna) return w32(x.old);
      const uint32_t p0[2] = {x.a, x.b}, p1[2] = {x.c, x.d};
      const uint32_t* pr[3] = {p0, p1, p0};
      const Pk p = op == 29 ? Pk{5, 2, 1, 4} : op == 30 ? Pk{1, 3, 0, 3} : Pk{0, 0, 2, 0};
      Want w;
      for (int r = 0; r < 2; ++r) {
        float s[3];
        for (int k = 0; k < 3; ++k) s[k] = fneg(in32(pr[k][((r ? p.hi : p.sel) >> k) & 1], m), neg_of(p, k, r), false);
        const float v = op == 29 ? fma32(s[0], s[1], s[2], m) : op == 30 ? mul32(s[0], s[1], m) : s[0] + s[1];
        const float o = out32(v, m, 0, op != 31);
        (r ? w.hi : w.lo) = b32(o);
        if (std::isnan(o)) w.nan |= r ? 8 : 4;
      }
      return w;
    }
    case 32: return wf(out32(in32(x.a, m) + in32(x.b, m), m, 1, false));
    case 33: return wf(out32(mul32(in32(x.a, m), in32(x.b, m), m), m, 3, true));
    case 34: return wf(out32(fma32(in32(x.a, m), in32(x.b, m), in32(x.c, m), m), m, 1, true));
    case 35: {
      const auto in64 = [&](uint32_t lo, uint32_t hi) {
        const double v = std::bit_cast<double>(uint64_t{hi} << 32 | lo);
        return m.flush && std::fpclassify(v) == FP_SUBNORMAL ? std::copysign(0.0, v) : v;
      };
      const double v = out64(in64(x.a, x.b) + in64(x.c, x.d), m, 2, false);
      const uint64_t bits = std::bit_cast<uint64_t>(v);
      Want w{static_cast<uint32_t>(bits), static_cast<uint32_t>(bits >> 32)};
      if (std::isnan(v)) w.nan = 16;
      return w;
    }
  }
  return {};
}

Want dpp(int op, int t, Mode m) {
  const In x = input(t);
  const uint32_t b = x.b, c = x.c, old = x.old;
  const auto f = [&](uint32_t v) { return in32(v, m); };
  const auto h = [&](uint32_t v) { return in16(v, m); };
  const auto a_of = [&](int from) { return from < 0 ? 0u : input(from).a; };
  const uint32_t qa = a_of(quad(t, kSwap));
  switch (op) {
    case 0: return wf(out32(f(qa) + f(b), m, 0, true));
    case 1: return wf(out32(f(a_of(row_shift(t, 1))) + f(b), m, 1, false));
    case 2: return wf(out32(mul32(-f(qa), std::fabs(f(b)), m), m, 2, true));
    case 3: return w32(static_cast<uint32_t>(std::min<uint64_t>(uint64_t{qa} + b, 0xFFFFFFFFu)));
    case 4: { const uint32_t ra = a_of(row_ror(t, 1)); return w32(ra < b ? 0 : ra - b); }
    case 5: { Want w = w32(static_cast<uint32_t>(std::min<uint32_t>((qa & 0xFFFF) + (b & 0xFFFF), 0xFFFF))); w.mask = 0xFFFF; return w; }
    case 6: { Want w = w32(std::min<uint32_t>((qa >> 16) + (b >> 16), 0xFFFF) << 16); w.mask = 0xFFFF0000u; return w; }
    case 7: return wf(out32(fma32(f(qa), f(b), f(c), m), m, 0, true));
    case 8: {
      const float p = f(a_of(dpp8(t, kSwap8))), q = f(b);
      if (std::isnan(p) || std::isnan(q) || (p == 0 && q == 0)) return Want{.any = true};   // (gfx11's: not established, see pk's)
      return wf(out32(maxmin(p, q, true, m.ieee), m, 0, true));
    }
    case 9: { const uint64_t s = uint64_t{qa} + b; return {static_cast<uint32_t>(std::min<uint64_t>(s, 0xFFFFFFFFu)), 0, s >> 32 ? 1u : 0u}; }
    case 10: { Want w = wh(out16(r16(double(h(qa)) + h(b)), m, 1, true), 0, 0); w.mask = 0xFFFF; return w; }
    case 11: return w32(static_cast<uint32_t>(std::min<uint64_t>(uint64_t{qa & 0xFFFFFF} * (b & 0xFFFFFF) + c, 0xFFFFFFFFu)));
    case 12: {
      const float p = f(qa), q = f(b), r = f(c);
      float v;
      if (std::isnan(p) || std::isnan(q) || std::isnan(r)) return Want{.any = true};   // (gfx11's median of a NaN: not established)
      else v = std::max(std::min(p, q), std::min(std::max(p, q), r));
      return wf(out32(v, m, 1, true));
    }
    case 13: { Want w = wh(out16(r16(double(h(qa >> 16)) * h(b) + h(c >> 16)), m, 0, true), 0, 16); w.mask = 0xFFFF0000u; return w; }
    case 14: return wf(out32(f(a_of(dpp8(t, kSwap8))) + f(b), m, 3, true));
    case 15: {
      const float v = f(qa);
      return w32(std::isnan(v) || v <= 0 ? 0u : v >= 4294967296.0f ? 0xFFFFFFFFu : static_cast<uint32_t>(v));
    }
    case 16: return w32(static_cast<uint32_t>(sat(s32(qa) + s32(b), INT32_MIN, INT32_MAX)));
    case 17: return w32(static_cast<uint32_t>(std::min<uint64_t>(uint64_t{qa & 0xFFFFFF} * (b & 0xFFFFFF), 0xFFFFFFFFu)));
    case 18: return w32(static_cast<uint32_t>(std::min<uint64_t>(uint64_t{qa > b ? qa - b : b - qa} + c, 0xFFFFFFFFu)));
    case 19: return Want{.any = true};   // (v_dot2_f32_f16's clamp on gfx11: not established, see pk's op 24)
    case 20: return wf(out32(fma32(static_cast<float>(h(qa)), static_cast<float>(h(b)), f(c), m), m, 0, true));
    case 21: {
      const int from = row_shift(t, -1);
      if (((t & 15) >> 2) & 1 || from < 0) return w32(old);
      return wf(out32(-f(input(from).a) + std::fabs(f(b)), m, 0, false));
    }
    case 22: return w32(a_of(row_shift(t, -1)));
    case 23: return w32(a_of(dpp8(t, kRev8)) + b);
  }
  return {};
}

Want model(Body body, int op, int t, Tg tg, Mode m) {
  return body == Body::Sdwa ? sdwa(op, t, tg, m) : body == Body::Pk ? pk(op, t, tg, m) : dpp(op, t, m);
}
bool is_nan16(uint32_t v) { return (v & 0x7C00) == 0x7C00 && (v & 0x3FF); }
bool is_nan32(uint32_t v) { return (v & 0x7F800000u) == 0x7F800000u && (v & 0x7FFFFFu); }
// Whether three dwords are what `e` says.
bool matches(const Want& e, const uint32_t* g) {
  if (e.any) return true;
  uint32_t mask = e.mask, hi_mask = ~0u;
  if (e.nan & 1) { if (!is_nan16(g[0] & 0xFFFF)) return false; mask &= ~0xFFFFu; }
  if (e.nan & 2) { if (!is_nan16(g[0] >> 16)) return false; mask &= 0xFFFFu; }
  if (e.nan & 4) { if (!is_nan32(g[0])) return false; mask = 0; }
  if (e.nan & 8) { if (!is_nan32(g[1])) return false; hi_mask = 0; }
  if (e.nan & 16) {   // (lo and hi, a double)
    if ((g[1] & 0x7FF00000u) != 0x7FF00000u || !((g[1] & 0xFFFFFu) | g[0])) return false;
    mask = 0, hi_mask = 0;
  }
  return (g[0] & mask) == (e.lo & mask) && (g[1] & hi_mask) == (e.hi & hi_mask) && g[2] == e.vcc;
}
// v_div_scale_f32 with clamp (39) and with mul:2 (40): what it gives without them (41), clamped or scaled like any VOP3 float result
// (the RX 6800: a NaN scale clamps to 0, and mul:2 doubles with IEEE off and denormals flushed); VCC the same.
bool div_scale_matches(int op, const uint32_t* g, const uint32_t* plain, Mode m) {
  if (op < 39 || op > 40) return true;
  const Want e = wf(out32(f32(plain[0]), m, op == 40 ? 1 : 0, op == 39));
  return matches(Want{e.lo, plain[1], plain[2], ~0u, e.nan}, g);
}
// --- end of the model ---

amd::CodeObject object(const char* target) {
  const std::string path = std::string(VGPU_SOURCE_DIR) + "/amd/tests/data/asm_mods." + target + ".o";
  std::ifstream in(path, std::ios::binary);
  if (!in) throw vtest::Failure("no code object at " + path);
  return amd::load_code_object(std::string((std::istreambuf_iterator<char>(in)), {}), path);
}

int run(const char* target, Tg tg, const char* kernel, Body body, Mode m, bool tier) {
  const int ops = ops_of(body);
  const amd::CodeObject o = object(target);
  MemoryManager mem(16ull << 20);
  std::vector<uint32_t> words(kItems * 8, 0);
  for (int t = 0; t < kItems; ++t) {
    const In in = input(t);
    const uint32_t w[5] = {in.a, in.b, in.c, in.d, in.old};
    std::memcpy(&words[t * 8], w, sizeof w);
  }
  const uint64_t pin = mem.alloc(words.size() * 4), pout = mem.alloc(ops * kItems * 12);
  mem.write(pin, words.data(), words.size() * 4);
  const amd::Kernel* k = amd::find_kernel(o, kernel);
  if (!k) throw vtest::Failure(std::string("no kernel named ") + kernel);
  std::vector<uint8_t> buf(k->kernarg_size, 0);
  const uint64_t args[2] = {pin, pout};
  std::memcpy(buf.data(), args, sizeof args);
  amd::Dispatch d;
  d.object = &o;
  d.kernel = k;
  d.kernarg = mem.alloc(buf.size());
  mem.write(d.kernarg, buf.data(), buf.size());
  d.group_size[0] = 64;
  d.groups[0] = kItems / 64;
  if (tg == kGfx1030 || tg == kGfx1100) d.wave_size = 32;
  amd::gcn::tier::set_enabled(tier);
  amd::execute(d, mem);
  amd::gcn::tier::set_enabled(true);
  std::vector<uint32_t> got(ops * kItems * 3);
  mem.read(pout, got.data(), got.size() * 4);
  int wrong = 0;
  for (int op = 0; op < ops; ++op)
    for (int t = 0, shown = 0; t < kItems; ++t) {
      const uint32_t* g = &got[(op * kItems + t) * 3];
      const bool ok = body == Body::Sdwa && op >= 39 && op <= 41 ? div_scale_matches(op, g, &got[(41 * kItems + t) * 3], m)
                                                                  : matches(model(body, op, t, tg, m), g);
      if (ok) continue;
      ++wrong;
      if (shown++ < 2) {   // (two lanes of each instruction that is wrong)
        const In in = input(t);
        const Want e = model(body, op, t, tg, m);
        std::fprintf(stderr, "  %s %s tier %d op %d t %d (a %08x b %08x c %08x d %08x old %08x): %08x %08x %x, expected %08x %08x %x (mask %08x nan %x)\n",
                     target, kernel, tier, op, t, in.a, in.b, in.c, in.d, in.old, g[0], g[1], g[2], e.lo, e.hi, e.vcc, e.mask, e.nan);
      }
    }
  return wrong;
}

struct Kern {
  const char* name;
  Body body;
  Mode mode;
};
constexpr Kern kSdwa[3] = {{"sdwa_k", Body::Sdwa, {true, false}}, {"sdwa_n", Body::Sdwa, {false, false}}, {"sdwa_f", Body::Sdwa, {false, true}}};
constexpr Kern kPk[2] = {{"pk_k", Body::Pk, {true, false}}, {"pk_f", Body::Pk, {false, true}}};
constexpr Kern kDpp[3] = {{"dpp_k", Body::Dpp, {true, false}}, {"dpp_n", Body::Dpp, {false, false}}, {"dpp_f", Body::Dpp, {false, true}}};

int run_all(const char* target, Tg tg, const Kern* ks, int n) {
  int wrong = 0;
  for (int i = 0; i < n; ++i)
    for (bool tier : {false, true}) {
      try {
        wrong += run(target, tg, ks[i].name, ks[i].body, ks[i].mode, tier);
      } catch (const std::exception& e) {   // (an instruction the executor refuses: counted, and the other kernels still run)
        std::fprintf(stderr, "  %s %s tier %d: %s\n", target, ks[i].name, tier, e.what());
        ++wrong;
      }
    }
  return wrong;
}

}  // namespace

VTEST(sdwa_clamp_output_modifier_and_selects_follow_the_isa) {
  VCHECK_EQ(run_all("gfx90a", kGfx90a, kSdwa, 3) + run_all("gfx942", kGfx942, kSdwa, 3) + run_all("gfx1030", kGfx1030, kSdwa, 3), 0);
}

VTEST(vop3p_op_sel_neg_and_clamp_follow_the_isa) {
  VCHECK_EQ(run_all("gfx90a", kGfx90a, kPk, 2) + run_all("gfx942", kGfx942, kPk, 2) + run_all("gfx1030", kGfx1030, kPk, 2) +
                run_all("gfx1100", kGfx1100, kPk, 2), 0);
}

VTEST(gfx11_vop3_dpp_clamp_and_output_modifier_follow_the_isa) {
  VCHECK_EQ(run_all("gfx1100", kGfx1100, kDpp, 3), 0);
}

VTEST_MAIN
