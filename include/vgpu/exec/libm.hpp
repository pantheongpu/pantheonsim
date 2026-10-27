// OpLibm's evaluator (patch E6): the OpenCL.std "host library" math functions, computed by the host C library.
//
// WHAT IS CALLED
//   f64: the double function of <cmath> named for the instruction (std::acos, std::sinh, std::tgamma, ...).
//   f32: the same double function on the operand widened to double, the result rounded once to float (round to
//        nearest even). The exceptions are the operations that are exact whatever the width and are done at the
//        operand width: fmod, remainder, fdim, ldexp, nextafter (and round, logb, ilogb, which are exact too).
//   f16: the f32 path on the operand widened to f32, the f32 result rounded once to f16 (nearest even); nextafter
//        steps in half precision.
//   Spelled out: exp10(x) is pow(10, x) (the only function with no standard C++ name); lgamma is lgamma_r where the
//   C library has it (the sign is dropped); powr(x, y) is pow(x, y) with the OpenCL domain written out (x < 0, NaN
//   operands, 0^0, inf^0 and 1^inf give NaN; a zero base is +0); pown(x, n) is pow(x, (double)n); ilogb of 0 is
//   INT_MIN, of infinity and NaN INT_MAX (the OpenCL values, not the C library's, which differ by platform).
//
// DETERMINISM
//   * No rounding-mode dependence: the engine evaluates under round-to-nearest-even, and every conversion here is
//     nearest-even.
//   * Every NaN result is the engine's canonical NaN of the width (0x7FFF, 0x7FFFFFFF, 0x7FFFFFFFFFFFFFFF),
//     whatever sign or payload the C library produced (x86 libm hands back negative NaNs from invalid operations,
//     other platforms positive ones).
//   * The exact operations (fmod, remainder, fdim, ldexp, nextafter, round, logb, ilogb) are bit-identical on any
//     host. f32 results of the inexact functions are the correctly rounded value of what the host's double libm
//     returns; two different libms differ only when the true result lies within the double libm's error (about
//     2^-52 relative) of a rounding boundary of f32, about 1 input in 2^28. f64 results of the inexact functions are
//     the host libm's own: the same binary on the same libm gives the same bits on every run, and two libms (glibc
//     versions, musl, a different microarchitecture's dispatch) can differ in the last place.
//   * The functions are libm-accurate, not Intel-bit-exact: an Intel GPU's math library returns its own results
//     within OpenCL's allowed error (a few ulp); this engine returns the host libm's, which are within one ulp of the
//     true value for f64 and correctly rounded in practice for f32. A program that compares floating-point output
//     with the bits of a card can differ in the last place of a transcendental. A documented limitation.
#pragma once

#include <cstdint>

#include "vgpu/ptx/ast.hpp"

namespace vgpu::exec {

// One lane of OpLibm. `bits` is 16, 32 or 64: the width of the floating-point type; `a`, `b` are raw register
// values (a float's bits, or the 32-bit integer operand of Ldexp and Pown in `b`). The result is the raw value: a
// float's bits at `bits`, or the 32-bit integer of Ilogb. `ftz` (32 bit only): subnormal operands and results
// are replaced by zero of the same sign.
uint64_t libm_eval(ptx::LibmFn fn, uint32_t bits, uint64_t a, uint64_t b, bool ftz = false);

const char* libm_name(ptx::LibmFn fn);    // the OpenCL.std spelling ("acos", "exp10", ...)
bool libm_binary(ptx::LibmFn fn);         // takes a second operand
bool libm_int_operand(ptx::LibmFn fn);    // that operand is a 32-bit integer (Ldexp, Pown)
bool libm_int_result(ptx::LibmFn fn);     // the result is a 32-bit integer (Ilogb)

}  // namespace vgpu::exec
