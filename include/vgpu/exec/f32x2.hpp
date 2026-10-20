// Packed single precision (sm_100+): the arithmetic of one 64-bit register holding two f32 lanes, the first in the low
// half. PTX's add/sub/mul/fma.f32x2 and SASS's FADD2/FMUL2/FFMA2 are the same operation, so both engines compute it here:
// each half is an ordinary f32 operation, flushed (.ftz) on its way in and out, and rounded as the host's rounding mode
// says -- the caller sets it (fesetround) around the lanes, as the PTX engine does for .rz/.rm/.rp and the SASS executor
// does for FFMA2.RZ/.RM/.RP. A NaN result is the canonical one, 0x7fffffff, as the scalar f32 operations of both engines
// give (the host's own NaN depends on the order the compiler happened to put the operands in: the two engines'
// compilations of this code gave different payloads for a NaN operand). NOT VERIFIED ON HARDWARE: no sm_100 card was
// available; the scalar FFMA/FADD/FMUL of the cards measured give this NaN.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

namespace vgpu::exec {

enum class F32x2Op : uint8_t { Add, Sub, Mul, Fma };

inline float f32x2_bits_to_float(uint32_t b) {
  float f;
  std::memcpy(&f, &b, 4);
  return f;
}

inline uint32_t f32x2_float_to_bits(float f) {
  uint32_t b;
  std::memcpy(&b, &f, 4);
  return b;
}

// One half, as bits. `c` is read only by Fma.
inline uint32_t f32x2_half(F32x2Op op, bool ftz, float a, float b, float c) {
  const auto flush = [&](float v) { return ftz && std::fpclassify(v) == FP_SUBNORMAL ? std::copysign(0.0f, v) : v; };
  const float x = flush(a), y = flush(b);
  float v;
  if (op == F32x2Op::Fma) v = std::fmaf(x, y, flush(c));
  else if (op == F32x2Op::Add) v = x + y;
  else if (op == F32x2Op::Sub) v = x - y;
  else v = x * y;
  v = flush(v);
  return std::isnan(v) ? 0x7fffffffu : f32x2_float_to_bits(v);
}

// Both halves of one lane's registers.
inline uint64_t f32x2_pair(F32x2Op op, bool ftz, uint64_t a, uint64_t b, uint64_t c) {
  uint64_t out = 0;
  for (int h = 0; h < 2; ++h) {
    const auto half = [h](uint64_t v) { return f32x2_bits_to_float(static_cast<uint32_t>(v >> (32 * h))); };
    out |= static_cast<uint64_t>(f32x2_half(op, ftz, half(a), half(b), half(c))) << (32 * h);
  }
  return out;
}

}  // namespace vgpu::exec
