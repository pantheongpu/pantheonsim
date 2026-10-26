// LEA on a warp (vgpu/exec/sass_lea.hpp): the vector code against the lane-by-lane rule it replaced. Every shift, .HI with a third source and
// with the sign of the first source, a carry in on every lane pattern, a negated second operand (whose zero carries out), operands
// at the edges of the 32-bit range (all ones plus a carry in, zero, the sign bit), and a result that is one of the sources.
#include <cstdint>
#include <cstring>
#include <random>

#include "vgpu/exec/sass_lea.hpp"
#include "vtest.hpp"

namespace {

namespace sl = vgpu::exec::sasslea;

uint32_t edge_or_random(std::mt19937_64& rng) {
  static const uint32_t kEdges[] = {0u, 1u, 2u, 0x7fffffffu, 0x80000000u, 0x80000001u, 0xfffffffeu, 0xffffffffu, 0x0000ffffu, 0xffff0000u};
  return rng() % 3 == 0 ? kEdges[rng() % (sizeof kEdges / sizeof kEdges[0])] : static_cast<uint32_t>(rng());
}

void row(std::mt19937_64& rng, uint32_t* v) {
  const unsigned kind = static_cast<unsigned>(rng() % 5);
  const uint32_t same = edge_or_random(rng);
  for (unsigned l = 0; l < 32; ++l) v[l] = kind == 0 ? same : kind == 1 ? static_cast<uint32_t>(rng()) : edge_or_random(rng);
}

}  // namespace

VTEST(lea_lanes_gives_the_lane_rules_registers_and_carries) {
  std::mt19937_64 rng(5);
  uint64_t carries = 0, zero_b_carries = 0;
  for (int round = 0; round < 60000; ++round) {
    alignas(32) uint32_t a[32], b[32], top[32], got[32], want[32];
    row(rng, a), row(rng, b), row(rng, top);
    sl::Form f;
    f.hi = rng() % 2;
    f.bneg = rng() % 3 == 0;
    f.sh = static_cast<unsigned>(rng() % 32);
    const bool three = f.hi && rng() % 2;
    const uint32_t cin = rng() % 4 == 0 ? 0xffffffffu : rng() % 4 == 0 ? 0u : static_cast<uint32_t>(rng());
    if (f.bneg && rng() % 2)
      for (unsigned l = 0; l < 32; ++l)
        if (rng() % 2) b[l] = 0;
    const uint32_t* t = three ? top : nullptr;
    const uint32_t want_carry = sl::lea_reference(f, want, a, b, t, cin);
    const uint32_t got_carry = sl::lea_lanes(f, got, a, b, t, cin);
    VCHECK_EQ(got_carry, want_carry);
    VCHECK(std::memcmp(got, want, sizeof got) == 0);
    carries += __builtin_popcount(want_carry);
    if (f.bneg)
      for (unsigned l = 0; l < 32; ++l) zero_b_carries += b[l] == 0 && ((want_carry >> l) & 1);
    // A result in place of a source.
    uint32_t in_place[32];
    std::memcpy(in_place, a, sizeof in_place);
    VCHECK_EQ(sl::lea_lanes(f, in_place, in_place, b, t, cin), want_carry);
    VCHECK(std::memcmp(in_place, want, sizeof in_place) == 0);
    std::memcpy(in_place, b, sizeof in_place);
    VCHECK_EQ(sl::lea_lanes(f, in_place, a, in_place, t, cin), want_carry);
    VCHECK(std::memcmp(in_place, want, sizeof in_place) == 0);
    if (three) {
      std::memcpy(in_place, top, sizeof in_place);
      VCHECK_EQ(sl::lea_lanes(f, in_place, a, b, in_place, cin), want_carry);
      VCHECK(std::memcmp(in_place, want, sizeof in_place) == 0);
    }
  }
  VCHECK(carries > 100000);       // (the cases that carry are many: not a test of zeros)
  VCHECK(zero_b_carries > 1000);  // (and a negated zero is among them)
}

VTEST(the_address_calculation_of_a_64_bit_pointer_is_the_pointer_plus_four_times_the_index) {
  // LEA R2, P0, R0, base_lo, 2 then LEA.HI.X R3, R0, base_hi, RZ, 2, P0: the pointer base + 4 * index, for indexes up to 2^32 - 1.
  std::mt19937_64 rng(6);
  for (int round = 0; round < 2000; ++round) {
    alignas(32) uint32_t idx[32], lo_base[32], hi_base[32], lo[32], hi[32], zero[32] = {};
    const uint64_t base = rng() >> (rng() % 24);
    for (unsigned l = 0; l < 32; ++l) {
      idx[l] = round % 3 ? static_cast<uint32_t>(rng()) : static_cast<uint32_t>(rng() % 1000);
      lo_base[l] = static_cast<uint32_t>(base), hi_base[l] = static_cast<uint32_t>(base >> 32);
    }
    sl::Form low;
    low.sh = 2;
    const uint32_t carry = sl::lea_lanes(low, lo, idx, lo_base, nullptr, 0);
    sl::Form high;
    high.hi = true;
    high.sh = 2;
    sl::lea_lanes(high, hi, idx, hi_base, zero, carry);   // (the third source RZ: the high word of the index is zero)
    for (unsigned l = 0; l < 32; ++l) {
      const uint64_t want = base + (uint64_t{idx[l]} << 2);
      VCHECK_EQ((uint64_t{hi[l]} << 32) | lo[l], want);
    }
  }
}

VTEST_MAIN
