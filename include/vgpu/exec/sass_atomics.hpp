// The value a SASS atomic (ATOM, ATOMG, ATOMS, RED) leaves in memory, given the one it finds: the rules of the SASS executor's
// read-modify-write (src/sass/exec_ops.inc exec_mem), and the faster forms of the common ones that a warp's batch of atomics runs.
// `op` is Instr::f[0] (0 ADD, 1 MIN, 2 MAX, 3 INC, 4 DEC, 5 AND, 6 OR, 7 XOR, 8 EXCH) and `type` Instr::f[4] (0 U32, 1 S32, 2 U64,
// 3 F32, 4 F16x2, 5 S64, 6 F64, 7 BF16x2); 32-bit values are the low 32 bits of a 64-bit word.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

#include "vgpu/exec/numerics.hpp"
#include "vgpu/ptx/ast.hpp"

namespace vgpu::exec::sassatom {

inline float as_f32(uint32_t b) { float f; std::memcpy(&f, &b, 4); return f; }
inline uint32_t f32_bits(float f) { uint32_t b; std::memcpy(&b, &f, 4); return b; }
inline double as_f64(uint64_t b) { double d; std::memcpy(&d, &b, 8); return d; }
inline uint64_t f64_bits(double d) { uint64_t b; std::memcpy(&b, &d, 8); return b; }

inline bool is64(unsigned type) { return type == 2 || type == 5 || type == 6; }

// The general rule. `cas`: compare-and-swap (`v` is the compare value, `swap` the new one); `popc_inc`: ATOMS.POPC.INC. False for an
// operation that is none of the above (the caller faults); `out` is then unset. The result is cut to 32 bits for a 32-bit type.
inline bool apply(unsigned op, unsigned type, bool cas, bool popc_inc, uint64_t old, uint64_t v, uint64_t swap, uint64_t& out) {
  uint64_t nw = old;
  if (cas) {
    if (old == v) nw = swap;
  } else if (type == 4 || type == 7) {   // F16x2, BF16x2 (and their vectors): add, min or max on each half
    const ptx::AtomOp fop = op == 1 ? ptx::AtomOp::Min : op == 2 ? ptx::AtomOp::Max : ptx::AtomOp::Add;
    nw = exec::atom_word(fop, type == 7 ? 2 : 1, static_cast<uint32_t>(old), static_cast<uint32_t>(v));
  } else {
    switch (popc_inc ? 3 : op) {
      case 0:   // ADD, by the PTX engine's rules (an RTX 3060's)
        if (type == 3) {
          // .ftz on inputs and result; a NaN result is the canonical one.
          const auto fz = [](float x) { return std::fpclassify(x) == FP_SUBNORMAL ? std::copysign(0.0f, x) : x; };
          const float r = fz(fz(as_f32(static_cast<uint32_t>(old))) + fz(as_f32(static_cast<uint32_t>(v))));
          nw = r != r ? 0x7fffffffu : f32_bits(r);
        } else if (type == 6) {
          // A NaN comes through as it is: the one added, else the one in memory.
          const double x = as_f64(old), y = as_f64(v);
          nw = std::isnan(y) ? v : std::isnan(x) ? old : f64_bits(x + y);
        } else {
          nw = old + v;
        }
        break;
      case 1:   // MIN
        if (type == 1) nw = static_cast<uint32_t>(std::min(static_cast<int32_t>(old), static_cast<int32_t>(v)));
        else if (type == 5) nw = static_cast<uint64_t>(std::min(static_cast<int64_t>(old), static_cast<int64_t>(v)));
        else nw = std::min(old, v);
        break;
      case 2:   // MAX
        if (type == 1) nw = static_cast<uint32_t>(std::max(static_cast<int32_t>(old), static_cast<int32_t>(v)));
        else if (type == 5) nw = static_cast<uint64_t>(std::max(static_cast<int64_t>(old), static_cast<int64_t>(v)));
        else nw = std::max(old, v);
        break;
      case 3: nw = popc_inc ? old + 1 : (static_cast<uint32_t>(old) >= static_cast<uint32_t>(v) ? 0 : old + 1); break;   // INC
      case 4: nw = (old == 0 || static_cast<uint32_t>(old) > static_cast<uint32_t>(v)) ? v : old - 1; break;             // DEC
      case 5: nw = old & v; break;
      case 6: nw = old | v; break;
      case 7: nw = old ^ v; break;
      case 8: nw = v; break;   // EXCH
      default: return false;
    }
  }
  out = is64(type) ? nw : (nw & 0xffffffffull);
  return true;
}

// The steps of the common operations, each a function of (old, v, swap) with the operation known when it is compiled. Their result is
// not cut to 32 bits (the caller's mask does that); each gives `apply`'s value for the operation it is named for.
struct CasStep { uint64_t operator()(uint64_t old, uint64_t cmp, uint64_t swap) const { return old == cmp ? swap : old; } };
struct AddStep { uint64_t operator()(uint64_t old, uint64_t x, uint64_t) const { return old + x; } };
struct AddF32Step {
  uint64_t operator()(uint64_t old, uint64_t x, uint64_t) const {
    // .ftz on inputs and result; a NaN result is the canonical one (the general rule works it out on floats).
    const auto fz = [](uint32_t b) { return (b & 0x7f800000u) == 0 ? b & 0x80000000u : b; };
    const float sum = as_f32(fz(static_cast<uint32_t>(old))) + as_f32(fz(static_cast<uint32_t>(x)));
    return uint64_t{sum != sum ? 0x7fffffffu : fz(f32_bits(sum))};
  }
};
struct MinStep { uint64_t operator()(uint64_t old, uint64_t x, uint64_t) const { return std::min(old, x); } };
struct MaxStep { uint64_t operator()(uint64_t old, uint64_t x, uint64_t) const { return std::max(old, x); } };
struct AndStep { uint64_t operator()(uint64_t old, uint64_t x, uint64_t) const { return old & x; } };
struct OrStep { uint64_t operator()(uint64_t old, uint64_t x, uint64_t) const { return old | x; } };
struct XorStep { uint64_t operator()(uint64_t old, uint64_t x, uint64_t) const { return old ^ x; } };
struct ExchStep { uint64_t operator()(uint64_t, uint64_t x, uint64_t) const { return x; } };

// Which step stands for (op, type, cas); General when none does (the halves, the signed min and max, INC, DEC, F64 and the rest: `apply`).
enum class Step : uint8_t { General, Cas, Add, AddF32, Min, Max, And, Or, Xor, Exch };
inline Step step_for(unsigned op, unsigned type, bool cas) {
  if (cas) return Step::Cas;
  if (type == 4 || type == 7) return Step::General;   // (every operation on the halves works on each half)
  switch (op) {
    case 0: return type == 3 ? Step::AddF32 : (type == 0 || type == 1 || type == 2 || type == 5) ? Step::Add : Step::General;
    case 1: return (type == 0 || type == 2) ? Step::Min : Step::General;
    case 2: return (type == 0 || type == 2) ? Step::Max : Step::General;
    case 5: return Step::And;
    case 6: return Step::Or;
    case 7: return Step::Xor;
    case 8: return Step::Exch;
    default: return Step::General;
  }
}

}  // namespace vgpu::exec::sassatom
