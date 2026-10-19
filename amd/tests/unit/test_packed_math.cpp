// The packed 16-bit float arithmetic (amd/src/packed_math.hpp) against what it replaced: the compiler's conversions
// between binary16 and binary32, and the per-lane code of v_pk_fma_f16 / v_pk_add_f16 / v_pk_mul_f16 (the
// interpreter's, kept here as the reference). Every lane must match bit for bit, but those with two NaN operands of different
// payloads, whose result depends on the order the compiler gave an x86 add or multiply (which test_amd_gcn_same_bits holds to the
// recorded one).
// VGPU_PACKED_EXHAUSTIVE=1 checks all 2^32 floats (about a minute); the default checks every binary16, the
// neighbourhood of every rounding boundary of every exponent, and a few million random floats.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../../src/packed_math.hpp"
#include "vtest.hpp"

namespace {

using namespace vgpu::amd::packed;

struct Rng {
  uint64_t s;
  uint64_t next() {
    s ^= s << 13, s ^= s >> 7, s ^= s << 17;
    return s;
  }
};

float float_of(uint32_t b) {
  float f;
  std::memcpy(&f, &b, 4);
  return f;
}
uint32_t bits_of(float f) {
  uint32_t b;
  std::memcpy(&b, &f, 4);
  return b;
}
uint16_t reference_to_half(float f) {
  const _Float16 h = static_cast<_Float16>(f);
  uint16_t b;
  std::memcpy(&b, &h, 2);
  return b;
}
float reference_to_float(uint16_t bits) {
  _Float16 h;
  std::memcpy(&h, &bits, 2);
  return static_cast<float>(h);
}

float clamp01(float v) { return std::isnan(v) || v <= 0.0f ? 0.0f : v > 1.0f ? 1.0f : v; }

// One lane, as the interpreter did it before: each half widened, the operation in float, the float narrowed.
uint32_t reference_lane(const WaveArgs& a, uint32_t lane) {
  uint32_t r = 0;
  for (uint32_t half = 0; half < 2; ++half) {
    const float x = reference_to_float(a.src[0][half].at(lane)), y = reference_to_float(a.src[1][half].at(lane));
    float v = a.op == Op::Add   ? x + y
              : a.op == Op::Mul ? x * y
                                : std::fma(x, y, reference_to_float(a.src[2][half].at(lane)));
    // (A fused multiply-add with a NaN operand is libm's with the multiplicands in the order the recorded results of
    // test_amd_gcn_same_bits show: fmaf_libm, whose order that test holds.)
    if (a.op == Op::Fma && v != v) v = fmaf_libm(x, y, reference_to_float(a.src[2][half].at(lane)));
    const _Float16 h = static_cast<_Float16>(a.clamp ? clamp01(v) : v);
    uint16_t b;
    std::memcpy(&b, &h, 2);
    r |= static_cast<uint32_t>(b) << (16 * half);
  }
  return r;
}

}  // namespace

VTEST(every_binary16_widens_to_the_float_the_compiler_gives) {
  uint32_t wrong = 0;
  for (uint32_t h = 0; h < 65536; ++h)
    wrong += bits_of(half_to_float(static_cast<uint16_t>(h))) != bits_of(reference_to_float(static_cast<uint16_t>(h)));
  VCHECK_EQ(wrong, 0u);   // signs of zero, subnormals, infinities and NaN payloads included
}

VTEST(floats_narrow_to_the_binary16_the_compiler_gives) {
  uint64_t wrong = 0, checked = 0;
  const auto one = [&](uint32_t b) {
    ++checked;
    wrong += float_to_half(float_of(b)) != reference_to_half(float_of(b));
  };
  if (const char* e = std::getenv("VGPU_PACKED_EXHAUSTIVE"); e && *e == '1') {
    for (uint64_t b = 0; b <= 0xFFFFFFFFull; ++b) one(static_cast<uint32_t>(b));
  } else {
    // Around the rounding boundaries: for every sign and exponent, a significand whose low 13 bits (the part a
    // half drops) are each of the interesting values -- just under, at and just over a tie -- and every upper
    // 10-bit pattern; the subnormal halves' (shifts 14 to 24) boundaries are among them.
    static const uint32_t kLow[] = {0, 1, 2, 0xFFF, 0x1000, 0x1001, 0x1FFE, 0x1FFF, 0x0800, 0x07FF, 0x0801};
    for (uint32_t sign = 0; sign < 2; ++sign)
      for (uint32_t e = 0; e < 256; ++e)
        for (uint32_t top = 0; top < 1024; ++top)
          for (uint32_t low : kLow) one(sign << 31 | e << 23 | top << 13 | low);
    // The smallest and the largest, and everything within a few of the overflow and underflow thresholds.
    for (int32_t d = -40; d <= 40; ++d) {
      for (uint32_t base : {0x477FF000u, 0x33000000u, 0x38800000u, 0x477FE000u, 0x7F800000u, 0x00000000u})
        one(base + static_cast<uint32_t>(d));
    }
    // Every NaN (the payload shifts, and the quiet bit is set).
    for (uint32_t sign = 0; sign < 2; ++sign)
      for (uint32_t m = 1; m < (1u << 23); ++m) one(sign << 31 | 0x7F800000u | m);
    Rng rng{0x9E3779B97F4A7C15ull};
    for (int i = 0; i < 4000000; ++i) one(static_cast<uint32_t>(rng.next()));
    // Floats with a half's range of exponent, which the random ones mostly miss.
    for (int i = 0; i < 4000000; ++i) {
      const uint32_t r = static_cast<uint32_t>(rng.next());
      one((r & 0x807FFFFFu) | ((95 + (r >> 23) % 40) << 23));
    }
  }
  VCHECK_EQ(wrong, 0u);
  VCHECK(checked > 1000000);
}

namespace {

// A random half, favouring the awkward ones.
uint16_t any_half(Rng& r) {
  switch (r.next() % 10) {
    case 0: return 0x0000;
    case 1: return 0x8000;
    case 2: return 0x7C00;   // infinity
    case 3: return 0xFC00;
    case 4: return static_cast<uint16_t>(0x7C01 + r.next() % 0x3FF);   // a NaN
    case 5: return static_cast<uint16_t>(r.next() % 0x400);            // a subnormal
    case 6: return 0x7BFF;
    case 7: return static_cast<uint16_t>((r.next() % 0x40) << 10 | r.next() % 4);
    default: return static_cast<uint16_t>(r.next());
  }
}

// Whether a lane has two NaN operands in some half (then which payload an add or a multiply passes on is the compiler's choice).
bool two_nans(const WaveArgs& a, uint32_t lane) {
  if (a.op == Op::Fma) return false;   // (a fused multiply-add is libm's whatever the NaNs, as the reference's is)
  const uint32_t n = 2;
  for (uint32_t half = 0; half < 2; ++half) {
    uint32_t nans = 0;
    for (uint32_t k = 0; k < n; ++k) nans += std::isnan(reference_to_float(a.src[k][half].at(lane)));
    if (nans > 1) return true;
  }
  return false;
}

// Runs the wave and returns how many lanes it wrote, adding to `wrong` those that differ from the per-lane code.
uint32_t compare_wave(Rng& rng, Op op, bool clamp, bool hw, uint32_t& wrong, uint32_t& skipped) {
  std::vector<uint32_t> row[3];
  for (auto& r : row) {
    r.resize(64);
    for (auto& v : r) v = static_cast<uint32_t>(any_half(rng)) | static_cast<uint32_t>(any_half(rng)) << 16;
  }
  WaveArgs a;
  a.op = op;
  a.clamp = clamp;
  a.exec = rng.next() | (rng.next() & 1 ? 0 : rng.next());
  if (rng.next() % 5 == 0) a.exec = ~0ull;
  for (uint32_t k = 0; k < 3; ++k)
    for (uint32_t half = 0; half < 2; ++half) {
      HalfSource& h = a.src[k][half];
      const uint64_t kind = rng.next() % 6;
      if (kind == 0) h.imm = any_half(rng);
      else {
        h.row = row[k].data();
        h.shift = static_cast<uint8_t>(16 * (rng.next() & 1));
      }
      h.flip = rng.next() % 3 == 0 ? 0x8000u : 0u;
    }
  std::vector<uint32_t> out(64, 0xDEADBEEFu);
  a.dst = out.data();
  if (hw) packed_f16_wave(a);   // (the host's FMA unit where it has one)
  else packed_f16_lanes<false>(a);   // (libm's fma, as a host without FMA runs)
  uint32_t written = 0;
  for (uint32_t lane = 0; lane < 64; ++lane) {
    if (!(a.exec >> lane & 1)) {
      wrong += out[lane] != 0xDEADBEEFu;   // an off lane is never written
      continue;
    }
    if (two_nans(a, lane)) {
      ++skipped;
      continue;
    }
    ++written;
    wrong += out[lane] != reference_lane(a, lane);
  }
  return written;
}

}  // namespace

VTEST(packed_half_arithmetic_matches_the_per_lane_code_bit_for_bit) {
  Rng rng{0x1234567};
  uint32_t wrong = 0, skipped = 0;
  uint64_t written = 0;
  for (int round = 0; round < 60000; ++round)
    written += compare_wave(rng, round % 3 == 0 ? Op::Fma : round % 3 == 1 ? Op::Add : Op::Mul, rng.next() % 4 == 0, round % 2 == 0,
                            wrong, skipped);
  VCHECK_EQ(wrong, 0u);
  VCHECK(written > 1000000);
  VCHECK(skipped < written / 2);   // (most lanes are compared, NaN results among them)
}

VTEST(a_nan_result_is_the_nan_the_operation_gives_with_its_payload_quieted_and_cut_to_ten_bits) {
  // One NaN operand: the result is that NaN (quiet), whatever the other operands are.
  Rng rng{0xABCDEF};
  uint64_t checked = 0, wrong = 0;
  for (int round = 0; round < 20000; ++round) {
    WaveArgs a;
    a.op = round % 3 == 0 ? Op::Fma : round % 3 == 1 ? Op::Add : Op::Mul;
    a.exec = 1;
    std::vector<uint32_t> out(64, 0xDEADBEEFu);
    a.dst = out.data();
    const uint32_t nan_at = static_cast<uint32_t>(rng.next() % (a.op == Op::Fma ? 3 : 2));
    const uint16_t payload = static_cast<uint16_t>(0x7C01 + rng.next() % 0x3FF) | (rng.next() & 1 ? 0x8000u : 0u);
    for (uint32_t k = 0; k < 3; ++k) {
      uint16_t v = k == nan_at ? payload : static_cast<uint16_t>(rng.next() % 0x7800);   // the others finite
      for (uint32_t half = 0; half < 2; ++half) a.src[k][half].imm = v;
    }
    packed_f16_wave(a);
    const uint16_t want = payload | 0x200;   // quieted (sign and ten payload bits kept)
    ++checked;
    wrong += (out[0] & 0xFFFF) != want || (out[0] >> 16) != want;
  }
  VCHECK_EQ(wrong, 0ull);
  VCHECK(checked > 0);
}

namespace {
uint32_t float_of_bits_special(Rng& rng) {
  static const uint32_t k[] = {0, 0x80000000u, 0x7F800000u, 0xFF800000u, 0x7FC00000u, 0x00000001u, 0x7F7FFFFFu, 0x3F800000u, 0xBF800000u};
  return k[rng.next() % (sizeof k / sizeof *k)];
}
}  // namespace

namespace {

// One lane of the dot products, as the interpreter did them before (v_dot2_f32_f16 and v_dot2c_f32_f16).
uint32_t reference_dot_lane(const DotArgs& d, uint32_t lane) {
  const auto half = [](uint16_t bits) {
    _Float16 h;
    std::memcpy(&h, &bits, 2);
    return static_cast<double>(h);
  };
  uint32_t cb = d.crow ? d.crow[lane] : d.cimm;
  float cf;
  cb = (cb & d.cand) ^ d.cflip;
  std::memcpy(&cf, &cb, 4);
  double sum;
  if (d.c_form) {
    double p0 = half(d.a[0].at(lane)) * half(d.b[0].at(lane)), p1 = half(d.a[1].at(lane)) * half(d.b[1].at(lane));
    double c = cf;
    __asm__("" : "+x"(p1), "+x"(p0), "+x"(c));
    sum = p1 + p0 + c;
  } else {
    sum = cf;
    for (uint32_t h = 0; h < 2; ++h) sum += half(d.a[h].at(lane)) * half(d.b[h].at(lane));
  }
  const float r = static_cast<float>(sum);
  uint32_t rb;
  std::memcpy(&rb, &r, 4);
  return rb;
}

}  // namespace

VTEST(dot_products_of_halves_match_the_per_lane_code_bit_for_bit) {
  Rng rng{0x7777};
  uint64_t wrong = 0, written = 0, handed_back = 0;
  for (int round = 0; round < 40000; ++round) {
    std::vector<uint32_t> rows[3], dst(64, 0xDEADBEEFu);
    for (auto& r : rows) {
      r.resize(64);
      for (auto& v : r) {
        const uint32_t kind = static_cast<uint32_t>(rng.next() % 6);
        v = kind == 0   ? float_of_bits_special(rng)
            : kind == 1 ? bits_of(static_cast<float>(static_cast<int32_t>(rng.next() % 2000) - 1000) / 8.0f)
                        : static_cast<uint32_t>(rng.next());
      }
    }
    DotArgs d;
    d.c_form = round % 2 == 1;
    d.exec = rng.next() % 4 == 0 ? ~0ull : rng.next();
    for (uint32_t h = 0; h < 2; ++h) {
      for (HalfSource* hs : {&d.a[h], &d.b[h]}) {
        if (rng.next() % 6 == 0) hs->imm = any_half(rng);
        else {
          hs->row = rows[rng.next() % 2].data();
          hs->shift = static_cast<uint8_t>(16 * (rng.next() & 1));
        }
        hs->flip = rng.next() % 3 == 0 ? 0x8000u : 0u;
      }
    }
    if (d.c_form) {
      d.crow = rows[2].data();
      dst = rows[2];
    } else if (rng.next() % 5 == 0) {
      d.cimm = static_cast<uint32_t>(rng.next());
    } else {
      d.crow = rows[2].data();
    }
    d.cand = rng.next() % 3 == 0 ? 0x7FFFFFFFu : 0xFFFFFFFFu;
    d.cflip = rng.next() % 3 == 0 ? 0x80000000u : 0u;
    if (d.c_form) {   // the addend is read from the destination, which is written in place
      d.cand = 0xFFFFFFFFu, d.cflip = 0;
      d.crow = dst.data();
    }
    d.dst = dst.data();
    const std::vector<uint32_t> before = dst;
    const uint64_t redo = round % 4 < 2 ? dot2_f16_wave(d) : dot2_f16_lanes(d);   // (the dispatching one, which is AVX2 where the host has it; and the scalar one)
    for (uint32_t lane = 0; lane < 64; ++lane) {
      if (!(d.exec >> lane & 1) || (redo >> lane & 1)) {
        wrong += dst[lane] != before[lane];   // untouched: off, or handed back
        handed_back += (redo >> lane & 1) != 0;
        continue;
      }
      // (the reference reads the addend from the register as it was)
      DotArgs r = d;
      r.crow = d.c_form ? before.data() : d.crow;
      if (!d.c_form && d.crow == dst.data()) r.crow = before.data();
      wrong += dst[lane] != reference_dot_lane(r, lane);
      ++written;
    }
  }
  VCHECK_EQ(wrong, 0ull);
  VCHECK(written > 1000000);
  VCHECK(handed_back > 100);
}

#if defined(__x86_64__)
VTEST(the_avx2_code_gives_the_scalar_codes_bits_lane_for_lane) {
  if (!host_has_avx2_f16c()) return;   // (nothing to compare on a host without the units)
  Rng rng{0xA5A5A5};
  uint64_t wrong = 0, lanes = 0;
  for (int round = 0; round < 60000; ++round) {
    std::vector<uint32_t> row[3];
    for (auto& r : row) {
      r.resize(64);
      for (auto& v : r) v = static_cast<uint32_t>(any_half(rng)) | static_cast<uint32_t>(any_half(rng)) << 16;
    }
    WaveArgs a;
    a.op = round % 3 == 0 ? Op::Fma : round % 3 == 1 ? Op::Add : Op::Mul;
    a.exec = rng.next() | (rng.next() & 1 ? 0 : rng.next());
    if (rng.next() % 5 == 0) a.exec = ~0ull;
    for (uint32_t k = 0; k < 3; ++k)
      for (uint32_t half = 0; half < 2; ++half) {
        HalfSource& h = a.src[k][half];
        if (rng.next() % 6 == 0) h.imm = any_half(rng);
        else {
          h.row = row[k].data();
          h.shift = static_cast<uint8_t>(16 * (rng.next() & 1));
        }
        h.flip = rng.next() % 3 == 0 ? 0x8000u : 0u;
      }
    std::vector<uint32_t> fast(64, 0xDEADBEEFu), slow(64, 0xDEADBEEFu);
    a.dst = fast.data();
    packed_f16_wave_avx2(a);
    a.dst = slow.data();
    packed_f16_lanes<false>(a);
    for (uint32_t lane = 0; lane < 64; ++lane) {
      if (fast[lane] != slow[lane] && wrong++ < 4)
        std::fprintf(stderr, "  op %d lane %u exec %d: avx2 %08x scalar %08x; x %04x/%04x y %04x/%04x z %04x/%04x clamp %d\n", static_cast<int>(a.op), lane,
                     static_cast<int>(a.exec >> lane & 1), fast[lane], slow[lane], a.src[0][0].at(lane), a.src[0][1].at(lane), a.src[1][0].at(lane),
                     a.src[1][1].at(lane), a.src[2][0].at(lane), a.src[2][1].at(lane), a.clamp);
      lanes += a.exec >> lane & 1;
    }
  }
  VCHECK_EQ(wrong, 0ull);
  VCHECK(lanes > 1000000);
}

__attribute__((target("avx2,f16c"))) static void f16c_convert8(const float* f, uint16_t* h) {
  _mm_storeu_si128(reinterpret_cast<__m128i*>(h), _mm256_cvtps_ph(_mm256_loadu_ps(f), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
}

VTEST(the_conversions_of_the_f16c_unit_are_the_integer_ones_including_every_nan) {
  if (!host_has_avx2_f16c()) return;
  uint64_t wrong = 0;
  const auto check8 = [&](const uint32_t* b) {
    float f[8];
    std::memcpy(f, b, 32);
    uint16_t h[8];
    f16c_convert8(f, h);
    for (int i = 0; i < 8; ++i) wrong += h[i] != float_to_half(f[i]);
  };
  uint32_t buf[8];
  uint32_t n = 0;
  const auto push = [&](uint32_t b) {
    buf[n++] = b;
    if (n == 8) check8(buf), n = 0;
  };
  for (uint32_t sign = 0; sign < 2; ++sign)
    for (uint32_t m = 1; m < (1u << 23); ++m) push(sign << 31 | 0x7F800000u | m);   // every NaN
  Rng rng{0x13572468};
  for (int i = 0; i < 4000000; ++i) {
    const uint32_t r = static_cast<uint32_t>(rng.next());
    push((r & 0x807FFFFFu) | ((95 + (r >> 23) % 40) << 23));   // (the range of exponents a half has)
  }
  VCHECK_EQ(wrong, 0ull);
}
#endif

VTEST_MAIN
