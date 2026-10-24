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
                                  0x7f800000, 0xff800000, 0x7fc00000, 0x7f800001, 0xffc00000, 0x00400000, 0x40490fdb};
  static const uint64_t f64s[] = {0, 0x8000000000000000ull, 1, 0x000fffffffffffffull, 0x3ff0000000000000ull, 0x7ff0000000000000ull,
                                  0xfff0000000000000ull, 0x7ff8000000000000ull, 0x7ff0000000000001ull, 0x7fefffffffffffffull};
  const unsigned pick = static_cast<unsigned>(rng() % 8);
  if (type == 3) return pick < 5 ? f32s[rng() % 16] : (rng() & 0xffffffffu);
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
            case sa::Step::Min: got = sa::MinStep{}(old, v, swap); break;
            case sa::Step::Max: got = sa::MaxStep{}(old, v, swap); break;
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
  VCHECK(specialised >= 20);   // (the add, min, max, bitwise, exchange and compare-and-swap forms have steps)
  VCHECK(specialised < total);   // (the halves and the signed min and max have none)
}

VTEST_MAIN
