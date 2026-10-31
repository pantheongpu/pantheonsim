// The value a SASS atomic leaves in memory (vgpu/exec/sass_atomics.hpp): the general rule, and the faster steps a warp's batch of
// atomics runs for the common operations, both against the lambda exec_mem had before the batch was specialised (copied here, frozen),
// over every operation and type, compare-and-swap, and operands at the edges of each type (zero, one, the extremes, NaNs, subnormals,
// infinities, values with the high word set).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>

#include "vgpu/exec/numerics.hpp"
#include "vgpu/exec/sass_atomics.hpp"
#include "vtest.hpp"

namespace {

namespace sa = vgpu::exec::sassatom;
using namespace vgpu;

float as_f32(uint32_t b) { float f; std::memcpy(&f, &b, 4); return f; }
uint32_t f32_bits(float f) { uint32_t b; std::memcpy(&b, &f, 4); return b; }
double as_f64(uint64_t b) { double d; std::memcpy(&d, &b, 8); return d; }
uint64_t f64_bits(double d) { uint64_t b; std::memcpy(&b, &d, 8); return b; }

// exec_mem's apply, as it was (d8ec6128): false where it called fault().
bool frozen_apply(unsigned op, unsigned type, bool cas, bool popc_inc, uint64_t old, uint64_t v, uint64_t swap, uint64_t& out) {
  const bool is64 = type == 2 || type == 5 || type == 6;
  uint64_t nw = old;
  if (cas) {
    if (old == v) nw = swap;
  } else {
    if (type == 4 || type == 7) {
      const ptx::AtomOp fop = op == 1 ? ptx::AtomOp::Min : op == 2 ? ptx::AtomOp::Max : ptx::AtomOp::Add;
      nw = exec::atom_word(fop, type == 7 ? 2 : 1, static_cast<uint32_t>(old), static_cast<uint32_t>(v));
    } else
      switch (popc_inc ? 3 : op) {
        case 0:
          if (type == 3) {
            const auto fz = [](float x) { return std::fpclassify(x) == FP_SUBNORMAL ? std::copysign(0.0f, x) : x; };
            const float r = fz(fz(as_f32(static_cast<uint32_t>(old))) + fz(as_f32(static_cast<uint32_t>(v))));
            nw = r != r ? 0x7fffffffu : f32_bits(r);
          } else if (type == 6) {
            const double x = as_f64(old), y = as_f64(v);
            nw = std::isnan(y) ? v : std::isnan(x) ? old : f64_bits(x + y);
          } else if (type == 4 || type == 7) {
            const ptx::Type ht{type == 7 ? ptx::Type::Kind::BF : ptx::Type::Kind::F, 16};
            const uint64_t lo = exec::reduce_value(ptx::AtomOp::Add, ht, old & 0xffff, v & 0xffff);
            const uint64_t hi = exec::reduce_value(ptx::AtomOp::Add, ht, (old >> 16) & 0xffff, (v >> 16) & 0xffff);
            nw = lo | (hi << 16);
          } else nw = old + v;
          break;
        case 1:
          if (type == 1) nw = static_cast<uint32_t>(std::min(static_cast<int32_t>(old), static_cast<int32_t>(v)));
          else if (type == 5) nw = static_cast<uint64_t>(std::min(static_cast<int64_t>(old), static_cast<int64_t>(v)));
          else nw = std::min(old, v);
          break;
        case 2:
          if (type == 1) nw = static_cast<uint32_t>(std::max(static_cast<int32_t>(old), static_cast<int32_t>(v)));
          else if (type == 5) nw = static_cast<uint64_t>(std::max(static_cast<int64_t>(old), static_cast<int64_t>(v)));
          else nw = std::max(old, v);
          break;
        case 3: nw = popc_inc ? old + 1 : (static_cast<uint32_t>(old) >= static_cast<uint32_t>(v) ? 0 : old + 1); break;
        case 4: nw = (old == 0 || static_cast<uint32_t>(old) > static_cast<uint32_t>(v)) ? v : old - 1; break;
        case 5: nw = old & v; break;
        case 6: nw = old | v; break;
        case 7: nw = old ^ v; break;
        case 8: nw = v; break;
        default: return false;
      }
  }
  if (!is64) nw &= 0xffffffff;
  out = nw;
  return true;
}

// Operands: random words and the ones that matter to a type.
uint64_t operand(std::mt19937_64& rng, unsigned type) {
  static const uint32_t f32s[] = {0x00000000, 0x80000000, 0x00000001, 0x007fffff, 0x80000001, 0x00800000, 0x3f800000, 0xbf800000, 0x7f7fffff,
                                  0x7f800000, 0xff800000, 0x7fc00000, 0x7f800001, 0xffc00000, 0x00400000, 0x40490fdb,
                                  // (normal numbers whose sum is subnormal: the result is flushed)
                                  0x00c00000, 0x80a00000, 0x80800000, 0x00a00000, 0x80c00000, 0x00800001, 0x80800001, 0x00ffffff};
  static const uint64_t f64s[] = {0, 0x8000000000000000ull, 1, 0x000fffffffffffffull, 0x3ff0000000000000ull, 0x7ff0000000000000ull,
                                  0xfff0000000000000ull, 0x7ff8000000000000ull, 0x7ff0000000000001ull, 0x7fefffffffffffffull};
  const unsigned pick = static_cast<unsigned>(rng() % 8);
  if (type == 3) return pick < 6 ? f32s[rng() % 24] : (rng() & 0xffffffffu);
  if (type == 6) return pick < 5 ? f64s[rng() % 10] : rng();
  if (type == 2 || type == 5) {
    static const uint64_t e[] = {0, 1, ~uint64_t{0}, 0x8000000000000000ull, 0x7fffffffffffffffull, 0xffffffffull, 0x100000000ull};
    return pick < 4 ? e[rng() % 7] : rng();
  }
  static const uint32_t e[] = {0, 1, 0xffffffffu, 0x80000000u, 0x7fffffffu, 0xfffffffeu, 0x7fff, 0x8000, 0xffff, 0x10000};
  return pick < 5 ? e[rng() % 10] : (rng() & 0xffffffffu);
}

}  // namespace

VTEST(the_general_rule_is_the_frozen_one) {
  std::mt19937_64 rng(1);
  for (unsigned type = 0; type < 8; ++type)
    for (unsigned op = 0; op < 10; ++op)
      for (int cas = 0; cas < 2; ++cas)
        for (int popc = 0; popc < 2; ++popc)
          for (int i = 0; i < 400; ++i) {
            const uint64_t old = operand(rng, type), v = operand(rng, type), swap = operand(rng, type);
            uint64_t a = 1, b = 2;
            const bool ra = frozen_apply(op, type, cas, popc, old, v, swap, a);
            const bool rb = sa::apply(op, type, cas, popc, old, v, swap, b);
            VCHECK_EQ(ra, rb);
            if (ra) VCHECK_EQ(a, b);
          }
}

VTEST(the_specialised_steps_are_the_frozen_rule) {
  std::mt19937_64 rng(2);
  size_t specialised = 0, total = 0;
  for (unsigned type = 0; type < 8; ++type)
    for (unsigned op = 0; op < 9; ++op)
      for (int cas = 0; cas < 2; ++cas) {
        ++total;
        const bool is64 = type == 2 || type == 5 || type == 6;
        const uint64_t keep = is64 ? ~uint64_t{0} : 0xffffffffull;
        const sa::Step which = sa::step_for(op, type, cas);
        if (which == sa::Step::General) continue;
        ++specialised;
        for (int i = 0; i < 4000; ++i) {
          const uint64_t old = operand(rng, type), v = operand(rng, type), swap = operand(rng, type);
          uint64_t got = 0;
          switch (which) {
            case sa::Step::Cas: got = sa::CasStep{}(old, v, swap); break;
            case sa::Step::Add: got = sa::AddStep{}(old, v, swap); break;
            case sa::Step::AddF32: got = sa::AddF32Step{}(old, v, swap); break;
            case sa::Step::AddF64: got = sa::AddF64Step{}(old, v, swap); break;
            case sa::Step::Min: got = sa::MinStep{}(old, v, swap); break;
            case sa::Step::Max: got = sa::MaxStep{}(old, v, swap); break;
            case sa::Step::MinS32: got = sa::MinS32Step{}(old, v, swap); break;
            case sa::Step::MaxS32: got = sa::MaxS32Step{}(old, v, swap); break;
            case sa::Step::MinS64: got = sa::MinS64Step{}(old, v, swap); break;
            case sa::Step::MaxS64: got = sa::MaxS64Step{}(old, v, swap); break;
            case sa::Step::Inc: got = sa::IncStep{}(old, v, swap); break;
            case sa::Step::Dec: got = sa::DecStep{}(old, v, swap); break;
            case sa::Step::And: got = sa::AndStep{}(old, v, swap); break;
            case sa::Step::Or: got = sa::OrStep{}(old, v, swap); break;
            case sa::Step::Xor: got = sa::XorStep{}(old, v, swap); break;
            case sa::Step::Exch: got = sa::ExchStep{}(old, v, swap); break;
            default: break;
          }
          uint64_t want = 0;
          VCHECK(frozen_apply(op, type, cas, false, old, v, swap, want));
          VCHECK_EQ(got & keep, want);
        }
      }
  VCHECK(specialised >= 50);   // (the add, min, max, inc, dec, bitwise, exchange and compare-and-swap forms have steps)
  VCHECK(specialised < total);   // (the halves, and min and max of the float types, have none)
}

// A warp's atomics on shared bytes (shared_batch), against the lane-by-lane read-modify-write the executor made before: memcpy of the
// word out of the bytes, the frozen rule, memcpy back. Offsets repeat, step by 4 and overlap by half a word (a lane finds what an earlier
// lane left, also in part of its word), masks are full and partial; the operations without a step are declined with the bytes untouched.
VTEST(a_warps_shared_atomics_are_the_lane_by_lane_ones) {
  std::mt19937_64 rng(3);
  size_t batched = 0, declined = 0;
  for (unsigned type = 0; type < 8; ++type)
    for (unsigned op = 0; op < 9; ++op)
      for (int cas = 0; cas < 2; ++cas)
        for (int i = 0; i < 300; ++i) {
          const bool wide = type == 2 || type == 5 || type == 6;
          const unsigned n = wide ? 8 : 4;
          uint8_t a[160], b[160];
          for (auto& x : a) x = static_cast<uint8_t>(rng());
          std::memcpy(b, a, sizeof a);
          uint32_t soff[32];
          uint64_t v[32], swp[32], old_of[32], want_old[32];
          const unsigned mode = static_cast<unsigned>(rng() % 4);
          for (unsigned l = 0; l < 32; ++l) {
            soff[l] = mode == 0 ? 4 * l : mode == 1 ? static_cast<uint32_t>(rng() % 3) * 4 : mode == 2 ? 4 * (l / 2) : static_cast<uint32_t>(rng() % 100);
            v[l] = operand(rng, type);
            swp[l] = operand(rng, type);
            // (a compare that can win: the word's own value now and then)
            if (cas && rng() % 3 == 0) { uint64_t cur = 0; std::memcpy(&cur, a + soff[l], n); v[l] = cur; }
          }
          const uint32_t ex = i % 3 == 0 ? 0xffffffffu : static_cast<uint32_t>(rng());
          if (!sa::shared_batch(op, type, cas, a, soff, ex, v, swp, old_of)) {
            VCHECK(sa::step_for(op, type, cas) == sa::Step::General);
            VCHECK(std::memcmp(a, b, sizeof a) == 0);   // (declined: nothing touched)
            ++declined;
            continue;
          }
          ++batched;
          for (unsigned l = 0; l < 32; ++l) {
            if (!((ex >> l) & 1)) continue;
            uint64_t old = 0;
            std::memcpy(&old, b + soff[l], n);
            uint64_t nw = 0;
            VCHECK(frozen_apply(op, type, cas, false, old, v[l], swp[l], nw));
            std::memcpy(b + soff[l], &nw, n);
            want_old[l] = old;
          }
          VCHECK(std::memcmp(a, b, sizeof a) == 0);
          for (unsigned l = 0; l < 32; ++l)
            if ((ex >> l) & 1) VCHECK_EQ(old_of[l], want_old[l]);
        }
  VCHECK(batched > 10000);
  VCHECK(declined > 1000);
}

// A warp's atomics on words already loaded (global_batch), against the lane-by-lane rule: lanes that share a slot find what the one
// before left, the old values, the new words and whether anything changed are the frozen rule's, for every operation and type; the
// operations without a step are declined with nothing touched.
VTEST(a_warps_atomics_on_loaded_words_are_the_lane_by_lane_ones) {
  std::mt19937_64 rng(4);
  size_t batched = 0, declined = 0;
  for (unsigned type = 0; type < 8; ++type)
    for (unsigned op = 0; op < 9; ++op)
      for (int cas = 0; cas < 2; ++cas)
        for (int i = 0; i < 300; ++i) {
          const bool wide = type == 2 || type == 5 || type == 6;
          const uint64_t keep = wide ? ~uint64_t{0} : 0xffffffffull;
          unsigned slot[32];
          uint64_t v[32], swp[32], cur[32], want[32], old_of[32], want_old[32];
          const unsigned nslots = 1 + static_cast<unsigned>(rng() % 32);
          for (unsigned l = 0; l < 32; ++l) {
            slot[l] = static_cast<unsigned>(rng() % nslots);
            v[l] = operand(rng, type);
            swp[l] = operand(rng, type);
          }
          for (unsigned k = 0; k < 32; ++k) want[k] = cur[k] = operand(rng, type);
          if (cas)
            for (unsigned l = 0; l < 32; ++l)
              if (rng() % 3 == 0) v[l] = cur[slot[l]];
          const uint32_t ex = i % 3 == 0 ? 0xffffffffu : static_cast<uint32_t>(rng()) | 1u;
          uint64_t before[32];
          std::memcpy(before, cur, sizeof cur);
          const int got = sa::global_batch(op, type, cas, ex, nslots, slot, cur, v, cas ? swp : nullptr, old_of);
          if (got < 0) {
            VCHECK(sa::step_for(op, type, cas) == sa::Step::General);
            VCHECK(std::memcmp(cur, before, sizeof cur) == 0);
            ++declined;
            continue;
          }
          ++batched;
          bool changed = false;
          for (unsigned l = 0; l < 32; ++l) {
            if (!((ex >> l) & 1)) continue;
            uint64_t nw = 0;
            VCHECK(frozen_apply(op, type, cas, false, want[slot[l]], v[l], swp[l], nw));
            want_old[l] = want[slot[l]];
            changed |= nw != want[slot[l]];
            want[slot[l]] = nw;
          }
          VCHECK_EQ(got, changed ? 1 : 0);
          for (unsigned k = 0; k < 32; ++k) VCHECK_EQ(cur[k] & keep, want[k] & keep);
          for (unsigned l = 0; l < 32; ++l)
            if ((ex >> l) & 1) VCHECK_EQ(old_of[l], want_old[l]);
        }
  VCHECK(batched > 10000);
  VCHECK(declined > 1000);
}

VTEST(a_warp_with_an_address_in_host_mapped_memory_is_not_batched) {
  struct Mem {
    uint64_t lo, hi;
    mutable int asked = 0;
    bool host_map_possible(uint64_t a) const { ++asked; return a >= lo && a < hi; }
  };
  const Mem m{0x1000, 0x2000};
  const uint64_t dev[3] = {0x10, 0x20, 0x4000}, mixed[3] = {0x10, 0x1800, 0x4000};
  VCHECK(sa::batchable(m, dev, 3));
  VCHECK(!sa::batchable(m, mixed, 3));
  VCHECK(sa::batchable(m, dev, 0));
}

VTEST_MAIN
