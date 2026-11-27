// See rdna2_trans.hpp. The arithmetic of one table, as tools/numerics/amd_trans_table_fit.py wrote the data: the mantissa m selects a
// segment (m >> 18) and gives the offsets f = m & 0x3ffff and fq (f with the table's j low bits cleared); acc = c0 + c1 f + c2 fq^2 + c3 f^3
// as a 128-bit integer scaled by 2^96; y = y0 + floor(acc / 2^96); a mantissa on the exception list moves y by one, up when the fraction
// of acc is half or more, down otherwise. The result's bits are the table's base plus y, shifted by the operand's exponent.
#include "rdna2_trans.hpp"

#include <bit>
#include <cmath>
#include <vector>

#include "transcendental.hpp"

namespace vgpu::amd::rdna2 {
namespace {

struct Word {
  uint64_t hi, lo;
};
struct Seg {
  int32_t y0;
  Word c[4];
};
#include "rdna2_trans_data.inc"

constexpr int kTables = 5;
enum { kRcp, kRsq0, kRsq1, kSqrt0, kSqrt1 };

struct Table {
  unsigned j;
  uint32_t base;
  int32_t at_zero;
  const Seg* seg;
  std::vector<uint64_t> exceptions;   // one bit a mantissa
};

int b64(char c) {
  if (c >= 'A' && c <= 'Z') return c - 'A';
  if (c >= 'a' && c <= 'z') return c - 'a' + 26;
  if (c >= '0' && c <= '9') return c - '0' + 52;
  return c == '+' ? 62 : 63;
}

// The exception list: the gaps between the mantissas, as 7-bit groups (low group first, the high bit says more follows), base64.
void decode(const char* text, uint32_t count, std::vector<uint64_t>& bits) {
  bits.assign((1u << 23) / 64, 0);
  uint32_t prev = 0, gap = 0, shift = 0, n = 0;
  uint32_t acc = 0;
  int have = 0;
  for (const char* p = text; *p && *p != '='; ++p) {
    acc = acc << 6 | static_cast<uint32_t>(b64(*p));
    have += 6;
    while (have >= 8) {
      have -= 8;
      const uint32_t byte = acc >> have & 0xff;
      gap |= (byte & 127) << shift;
      if (byte & 128) {
        shift += 7;
        continue;
      }
      prev += gap + 1;
      bits[prev / 64] |= uint64_t{1} << (prev % 64);
      ++n;
      gap = 0;
      shift = 0;
    }
  }
  (void)count;
  (void)n;
}

Table build(unsigned j, uint32_t base, int32_t at_zero, const Seg* seg, const char* exc, uint32_t count) {
  Table t{j, base, at_zero, seg, {}};
  decode(exc, count, t.exceptions);
  return t;
}

const Table& table(int id) {
  static const Table tables[kTables] = {
      build(kRcpJ, kRcpBase, kRcpAtZero, kRcpSeg, kRcpExc, kRcpExcCount),
      build(kRsq0J, kRsq0Base, kRsq0AtZero, kRsq0Seg, kRsq0Exc, kRsq0ExcCount),
      build(kRsq1J, kRsq1Base, kRsq1AtZero, kRsq1Seg, kRsq1Exc, kRsq1ExcCount),
      build(kSqrt0J, kSqrt0Base, kSqrt0AtZero, kSqrt0Seg, kSqrt0Exc, kSqrt0ExcCount),
      build(kSqrt1J, kSqrt1Base, kSqrt1AtZero, kSqrt1Seg, kSqrt1Exc, kSqrt1ExcCount),
  };
  return tables[id];
}

// The card's result for the mantissa m of a table, as bits minus the table's base.
inline int32_t value(const Table& t, uint32_t m) {
  if (m == 0) return t.at_zero;
  const Seg& s = t.seg[m >> 18];
  const uint64_t f = m & 0x3ffff, fq = f >> t.j << t.j;
  using I = __int128;
  const auto w = [](const Word& x) { return static_cast<I>(static_cast<unsigned __int128>(x.hi) << 64 | x.lo); };
  const I acc = w(s.c[0]) + w(s.c[1]) * static_cast<I>(f) + w(s.c[2]) * static_cast<I>(fq * fq) + w(s.c[3]) * static_cast<I>(f * f * f);
  int32_t y = s.y0 + static_cast<int32_t>(acc >> 96);
  if (t.exceptions[m / 64] >> (m % 64) & 1) y += (acc >> 95 & 1) ? 1 : -1;
  return y;
}

inline bool normal(uint32_t b) {
  const uint32_t e = b >> 23 & 0xff;
  return e != 0 && e != 255;
}
inline float make(uint32_t sign, int64_t exp_and_mant) {
  // (an exponent field of zero or less is a denormal result: the card writes a zero of the sign; none above 254 can arise from a normal operand)
  if ((exp_and_mant >> 23) <= 0) return std::bit_cast<float>(sign);
  return std::bit_cast<float>(sign | static_cast<uint32_t>(exp_and_mant));
}

}  // namespace

float rcp(float x) {
  const uint32_t b = std::bit_cast<uint32_t>(x);
  if (!normal(b)) return trans::rcp(x);
  const Table& t = table(kRcp);
  const int64_t k = static_cast<int64_t>(b >> 23 & 0xff) - 127;
  return make(b & 0x80000000u, static_cast<int64_t>(t.base) + value(t, b & 0x7fffff) - (k << 23));
}

float rsq(float x) {
  const uint32_t b = std::bit_cast<uint32_t>(x);
  if (!normal(b) || (b >> 31)) return trans::rsq(x);
  const int64_t k = static_cast<int64_t>(b >> 23 & 0xff) - 127;
  const Table& t = table((k & 1) ? kRsq1 : kRsq0);
  return make(0, static_cast<int64_t>(t.base) + value(t, b & 0x7fffff) - ((k >> 1) << 23));
}

float sqrt(float x) {
  const uint32_t b = std::bit_cast<uint32_t>(x);
  if (!normal(b) || (b >> 31)) return trans::sqrt(x);
  const int64_t k = static_cast<int64_t>(b >> 23 & 0xff) - 127;
  const Table& t = table((k & 1) ? kSqrt1 : kSqrt0);
  return make(0, static_cast<int64_t>(t.base) + value(t, b & 0x7fffff) + ((k >> 1) << 23));
}

}  // namespace vgpu::amd::rdna2
