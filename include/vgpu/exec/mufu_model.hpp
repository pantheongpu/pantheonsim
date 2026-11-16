// The approximate-math instructions of an NVIDIA GPU (SASS MUFU; PTX rcp/rsqrt/sqrt/ex2/lg2/sin/cos/tanh .approx), bit for bit as
// a card computes them. docs/numerics/math-functions.md has the measurements, the fit and its evidence.
//
// Measured on an RTX 3060 (sm_86) over all 2^32 inputs of each instruction (tools/numerics/mufu_probe.cu); a Tesla T4
// (sm_75), an L4 (sm_89) and an A10G (sm_86) print the same math-library results as the RTX 3060 on every line of
// tools/numerics/mathfn. Hopper and Blackwell cards were not measured: the simulator gives them the same model, inferred.
//
// The unit is a table of second-degree polynomials: the leading bits of the argument select an interval, and the result is
//   Y = c0 + c1 * x + c2 * sq19(x)
// where x is the 17 argument bits below the index and sq19 is x * x computed without its low partial products (below). Y
// is a fixed-point number with 40 fraction bits; the result is Y cut (not rounded) to the float format. The tables
// (mufu_tables.inc) are fitted to the card's results: c1 and c2 are the integers that reproduce every result of an
// interval, c0 the card's table entry plus one constant per instruction.
//
// Plain C++17 with no dependency, so that the card probe (built by nvcc) checks this very file against the card.
#pragma once

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace vgpu::mufu {

// VGPU_MUFU_EXACT=0 turns the model off: the instructions are then the function at the host's precision, rounded once, as
// every release before the model computed them. Unset or anything else: the model. Read when the process starts and again
// at each launch (as VGPU_TC_EXACT is), so a test can set it between launches.
inline bool parse_enabled(const char* v) { return !(v && v[0] == '0'); }
inline std::atomic<bool>& enabled_flag() {
  static std::atomic<bool> f{parse_enabled(std::getenv("VGPU_MUFU_EXACT"))};
  return f;
}
inline void refresh(const char* v) { enabled_flag().store(parse_enabled(v), std::memory_order_relaxed); }
inline bool enabled() { return enabled_flag().load(std::memory_order_relaxed); }

struct Coef { uint64_t c0; int32_t c1, c2; };
#include "mufu_tables.inc"

// The sub-operations, as the SASS instruction encodes them.
enum : unsigned { OpCos = 0, OpSin = 1, OpEx2 = 2, OpLg2 = 3, OpRcp = 4, OpRsq = 5, OpRcp64h = 6, OpRsq64h = 7, OpSqrt = 8, OpTanh = 9 };

inline constexpr uint32_t kNaN = 0x7fffffffu;   // the NaN every MUFU returns

// The square of a 17-bit x as the unit forms it: the partial products of the folded array (bit i alone at weight 2^(2i),
// bits i < j together at weight 2^(i+j+1)) whose weight is at least 2^19, summed and shifted right by 19. The products
// below 2^19 are left out, not rounded in.
inline uint32_t sq19(uint32_t x) {
  uint64_t dropped = 0;
  for (unsigned i = 0; i <= 9; ++i) {
    const uint64_t bit = (x >> i) & 1u;
    dropped += bit << (2 * i);                                                    // bit i alone: weight 2^(2i) < 2^19
    if (i <= 8) dropped += (bit * ((x >> (i + 1)) & ((1u << (17 - 2 * i)) - 1))) << (2 * i + 2);   // with bits j, i < j <= 17 - i
  }
  return static_cast<uint32_t>((static_cast<uint64_t>(x) * x - dropped) >> 19);
}

inline int64_t poly(const Coef& k, uint32_t x) {
  return static_cast<int64_t>(k.c0) + static_cast<int64_t>(k.c1) * x + static_cast<int64_t>(k.c2) * sq19(x);
}

// A float from sign, biased exponent field e and the 24-bit significand p in [2^23, 2^24] (2^24: the next binade). A result
// below the normal range is a zero (the unit flushes), one above it an infinity.
inline uint32_t pack(uint32_t sign, int64_t e, uint32_t p) {
  const int64_t v = (e << 23) + (static_cast<int64_t>(p) - 0x800000);
  if (v < 0x800000) return sign;
  if (v >= 0x7f800000) return sign | 0x7f800000u;
  return sign | static_cast<uint32_t>(v);
}

// A float from a magnitude in units of 2^-scale, cut to 24 significant bits.
inline uint32_t from_fixed(uint32_t sign, uint64_t mag, int scale) {
  if (mag == 0) return sign;
  int n = 64 - __builtin_clzll(mag);
  if (n > 24) mag >>= (n - 24); else mag <<= (24 - n);
  return sign | (static_cast<uint32_t>(127 + n - 1 - scale) << 23) | (static_cast<uint32_t>(mag) & 0x7fffffu);
}

// MUFU.RCP. e: the biased exponent, m: the 23 mantissa bits of a normal number.
inline uint32_t rcp_sig(uint32_t m) { return static_cast<uint32_t>(poly(kRcp[m >> 16], (m & 0xffffu) << 1) >> 17); }
inline uint32_t rcp(uint32_t x) {
  const uint32_t sign = x & 0x80000000u, e = (x >> 23) & 0xff, m = x & 0x7fffffu;
  if (e == 0xff) return m ? kNaN : sign;
  if (e == 0) return sign | 0x7f800000u;           // a zero, and a subnormal (flushed to zero)
  return pack(sign, 253 - static_cast<int64_t>(e), rcp_sig(m));
}

// MUFU.RSQ. The exponent's parity selects the half of the table; 1 / sqrt of an even power of two is exact.
inline uint32_t rsq_sig(bool odd, uint32_t m) {
  if (!odd && m == 0) return 0x1000000u;
  return static_cast<uint32_t>(poly(kRsq[(odd ? 64 : 0) + (m >> 17)], m & 0x1ffffu) >> 17);
}
inline uint32_t rsq(uint32_t x) {
  const uint32_t sign = x & 0x80000000u, e = (x >> 23) & 0xff, m = x & 0x7fffffu;
  if (e == 0xff) return m ? kNaN : sign ? kNaN : 0u;
  if (e == 0) return sign | 0x7f800000u;
  if (sign) return kNaN;
  const int u = static_cast<int>(e) - 127;
  return pack(0, 126 - (u >> 1), rsq_sig(u & 1, m));
}

inline uint32_t sqrt_sig(bool odd, uint32_t m) { return static_cast<uint32_t>(poly(kSqrt[(odd ? 64 : 0) + (m >> 17)], m & 0x1ffffu) >> 17); }
inline uint32_t sqrt(uint32_t x) {
  const uint32_t sign = x & 0x80000000u, e = (x >> 23) & 0xff, m = x & 0x7fffffu;
  if (e == 0xff) return m ? kNaN : sign ? kNaN : x;
  if (e == 0) return sign;
  if (sign) return kNaN;
  const int u = static_cast<int>(e) - 127;
  return pack(0, 127 + (u >> 1), sqrt_sig(u & 1, m));
}

// MUFU.EX2: the magnitude of the argument as a fixed-point number with 23 fraction bits (the bits below are dropped); the
// fraction goes through the table, the integer part is the exponent. A negative argument is complemented, not negated:
// its fraction is 2^23 - 1 - fraction and its integer part -integer - 1, unless the fraction is zero (then the integer
// part is just negated), so 2^-1.5 is computed as 2^-2 * 2^(0.5 - 2^-23).
inline uint32_t ex2(uint32_t x) {
  const uint32_t sign = x & 0x80000000u, e = (x >> 23) & 0xff, m = x & 0x7fffffu;
  if (e == 0xff) return m ? kNaN : sign ? 0u : x;
  if (e >= 127 + 8) return sign ? 0u : 0x7f800000u;      // |x| >= 256
  int64_t t = 0;                                         // |x| * 2^23, cut
  if (e != 0) {
    const uint64_t sig = 0x800000u | m;                  // |x| * 2^23 = sig * 2^(e - 127)
    const int sh = static_cast<int>(e) - 127;
    t = sh >= 0 ? static_cast<int64_t>(sig << sh) : sh > -24 ? static_cast<int64_t>(sig >> -sh) : 0;
    if (sign) t = (t & 0x7fffff) ? ~t : -t;
  }
  const uint32_t f = static_cast<uint32_t>(t) & 0x7fffffu;
  return pack(0, 127 + (t >> 23), static_cast<uint32_t>(poly(kEx2[f >> 17], f & 0x1ffffu) >> 17));
}

// MUFU.LG2: exponent + table(mantissa) as a fixed-point number with 36 fraction bits, then cut to a float. A negative
// value is negated as a one's complement (one unit of 2^-36 short). log2 of exactly 1 is 0.
inline uint32_t lg2(uint32_t x) {
  const uint32_t sign = x & 0x80000000u, e = (x >> 23) & 0xff, m = x & 0x7fffffu;
  if (e == 0xff) return m ? kNaN : sign ? kNaN : x;
  if (e == 0) return 0xff800000u;
  if (sign) return kNaN;
  if (e == 127 && m == 0) return 0;
  const int64_t v = (static_cast<int64_t>(static_cast<int>(e) - 127) << 36) + (poly(kLg2[m >> 17], m & 0x1ffffu) >> 4);
  return v < 0 ? from_fixed(0x80000000u, static_cast<uint64_t>(-v - 1), 36) : from_fixed(0, static_cast<uint64_t>(v), 36);
}

// MUFU.SIN and MUFU.COS: the argument is in turns. Its fraction is cut to 25 bits; the top two are the quadrant, the other
// 23 (complemented in the odd quadrants) go through the table of sin over a quarter turn.
inline uint32_t sincos(uint32_t x, bool cos) {
  const uint32_t e = (x >> 23) & 0xff, m = x & 0x7fffffu;
  uint32_t sign = cos ? 0u : x & 0x80000000u;
  if (e == 0xff) return kNaN;
  uint32_t phase = 0;                                    // |x| * 2^25 mod 2^25
  if (e != 0) {
    const uint64_t sig = 0x800000u | m;                  // |x| * 2^25 = sig * 2^(e - 125)
    const int sh = static_cast<int>(e) - 125;
    if (sh >= 25) phase = 0;
    else if (sh >= 0) phase = static_cast<uint32_t>(sig << sh) & 0x1ffffffu;
    else if (sh > -24) phase = static_cast<uint32_t>(sig >> -sh);
  }
  if (cos) phase = (phase + 0x800000u) & 0x1ffffffu;
  const uint32_t quadrant = phase >> 23;
  uint32_t z = phase & 0x7fffffu;
  if (quadrant & 1) z ^= 0x7fffffu;
  if (quadrant & 2) sign ^= 0x80000000u;
  return from_fixed(sign, static_cast<uint64_t>(poly(kSin[z >> 17], z & 0x1ffffu)), 40);
}

// MUFU.TANH. Below 2^-8 the result is the argument (a subnormal too: this instruction does not flush); from 8 on it is 1. Between
// them each binade is cut into 1 to 16 intervals (kTanhBits index bits below the exponent), the next 17 mantissa bits are x
// and the mantissa bits below those are not read.
inline constexpr uint8_t kTanhBits[11] = {0, 0, 0, 1, 1, 2, 2, 2, 3, 4, 3};       // exponents -8 .. 2
inline constexpr uint8_t kTanhBase[11] = {0, 1, 2, 3, 5, 7, 11, 15, 19, 27, 43};  // the first row of each
inline uint32_t tanh(uint32_t x) {
  const uint32_t sign = x & 0x80000000u, e = (x >> 23) & 0xff, m = x & 0x7fffffu;
  if (e == 0xff) return m ? kNaN : sign | 0x3f800000u;
  if (e < 127 - 8) return x;
  if (e >= 127 + 3) return sign | 0x3f800000u;
  const unsigned k = e - (127 - 8), b = kTanhBits[k];
  return from_fixed(sign, static_cast<uint64_t>(poly(kTanh[kTanhBase[k] + (m >> (23 - b))], (m >> (6 - b)) & 0x1ffffu)), 40);
}

// MUFU.RCP64H and MUFU.RSQ64H: the upper word of a double in, the upper word of the result out. The 20 mantissa bits go
// through the single-precision unit (as the upper 20 of its 23) and its result is cut to 20 bits.
inline uint32_t rcp64h(uint32_t hi) {
  const uint32_t sign = hi & 0x80000000u, e = (hi >> 20) & 0x7ff, m = hi & 0xfffffu;
  if (e == 0x7ff) return m ? 0x7fffffffu : sign;
  if (e == 0) return sign | 0x7ff00000u;
  const int64_t v = ((2045 - static_cast<int64_t>(e)) << 20) + (static_cast<int64_t>(rcp_sig(m << 3) >> 3) - 0x100000);
  if (v < 0x100000) return sign;
  return sign | static_cast<uint32_t>(v);
}
inline uint32_t rsq64h(uint32_t hi) {
  const uint32_t sign = hi & 0x80000000u, e = (hi >> 20) & 0x7ff, m = hi & 0xfffffu;
  if (e == 0x7ff) return m ? 0x7fffffffu : sign ? 0x7fffffffu : 0u;
  if (e == 0) return sign | 0x7ff00000u;
  if (sign) return 0x7fffffffu;
  const int u = static_cast<int>(e) - 1023;
  return static_cast<uint32_t>(((1022 - static_cast<int64_t>(u >> 1)) << 20) + (static_cast<int64_t>(rsq_sig(u & 1, m << 3) >> 3) - 0x100000));
}

// Whether the model has the sub-operation: all ten that the instruction encodes.
inline bool has(unsigned op) { return op <= OpTanh; }

// The instruction on a raw 32-bit operand (a float's bits; for the 64H forms the upper word of a double).
inline uint32_t eval(unsigned op, uint32_t x) {
  switch (op) {
    case OpCos: return sincos(x, true);
    case OpSin: return sincos(x, false);
    case OpEx2: return ex2(x);
    case OpLg2: return lg2(x);
    case OpRcp: return rcp(x);
    case OpRsq: return rsq(x);
    case OpRcp64h: return rcp64h(x);
    case OpRsq64h: return rsq64h(x);
    case OpSqrt: return sqrt(x);
    case OpTanh: return tanh(x);
    default: return 0;
  }
}

}  // namespace vgpu::mufu
