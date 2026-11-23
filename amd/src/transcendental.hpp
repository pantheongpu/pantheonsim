// The single-precision transcendental instructions (v_rcp_f32, v_rsq_f32, v_sqrt_f32, v_exp_f32, v_log_f32, v_sin_f32, v_cos_f32) as one
// function each, so the interpreter and the tier give the same bits.
//
// What a card does that the host's libm does not (the RX 6800 and the RX 6700 XT, ROCm 7.2.0, all of 2^32 inputs hashed and 77 selected
// operands in tools/numerics/amd/gfx1030.vectors.txt, docs/numerics/math-functions.md): v_rcp_f32, v_rsq_f32, v_sqrt_f32, v_exp_f32 and
// v_log_f32 read a denormal operand as a zero of its sign and write a denormal result as a zero of its sign, whatever MODE says; v_sin_f32
// and v_cos_f32 count the angle in turns and keep only its fraction (a whole number of turns, of any size, is the angle 0).
// What is not modelled: the low bits. The card's result is within one unit in the last place of the true value on most operands and
// is not the nearest float on 1 to 50 percent of them; no model is known, so the result here is the host's, rounded once.
//
// Measured on gfx1030 and gfx1031 only. The other targets are given the same rules (inferred, not measured): LLVM's AMDGPU lowering
// of f32 sqrt, rsq, log and exp scales a denormal operand on every target, because the instruction does not take one, and AMD's ISA
// guides (the machine-readable ones for CDNA3, RDNA3, RDNA4, CDNA5) describe all seven in the same words on every target.
#pragma once
#include <bit>
#include <cmath>

namespace vgpu::amd::trans {

inline float flush(float x) { return std::fpclassify(x) == FP_SUBNORMAL ? std::copysign(0.0f, x) : x; }

inline float rcp(float x) { return flush(1.0f / flush(x)); }
inline float rsq(float x) { return flush(1.0f / std::sqrt(flush(x))); }
inline float sqrt(float x) { return flush(std::sqrt(flush(x))); }
inline float exp2(float x) { return flush(std::exp2(flush(x))); }
inline float log2(float x) { return flush(std::log2(flush(x))); }

// The angle in turns: only its fraction matters, which is exact in a double (so the argument given to the library never grows large);
// a whole number of turns is the angle +0, an infinity the NaN the ISA's examples show (0xffc00000), a NaN stays a NaN.
inline float turns_trig(double turns, bool want_sin) {
  if (std::isinf(turns)) return std::bit_cast<float>(0xFFC00000u);
  if (std::isnan(turns)) return static_cast<float>(turns);
  const double frac = turns == 0 ? turns : turns - std::trunc(turns);
  // The card's exact zeros (the RX 6800, v_sin_f32 and v_cos_f32 of every multiple of 1/64 turn from -4 to 4, 2026-11-23): the sine of
  // a half-turn multiple and the cosine of an odd quarter are +0, where 2*pi in a double leaves 1.2e-16 or 6.1e-17.
  if (want_sin ? (frac != 0 && frac * 2 == std::trunc(frac * 2)) : (frac * 4 == std::trunc(frac * 4) && std::fmod(frac * 4, 2.0) != 0)) return 0.0f;
  const double radians = frac * 6.283185307179586476925286766559;
  return static_cast<float>(want_sin ? std::sin(radians) : std::cos(radians));
}

}  // namespace vgpu::amd::trans
