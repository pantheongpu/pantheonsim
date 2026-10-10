// Number formats both engines convert the same way: defined with the PTX
// interpreter's arithmetic (src/exec/interpreter.cpp) and used by the SASS
// executor too, so a value means the same thing to either.
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>

#include "vgpu/exec/tensormap.hpp"
#include "vgpu/ptx/ast.hpp"

namespace vgpu::exec {

// One element of a reduction (red, atom, cp.reduce.async.bulk): `old`
// combined with `b` by `op`, both of type `ty`. Floating-point add rounds to
// nearest even and keeps subnormals; min and max return the other operand
// when one is NaN.
uint64_t reduce_value(ptx::AtomOp op, const ptx::Type& ty, uint64_t old, uint64_t b);

// One 32-bit word of a vector atom/red (sm_90): `old` combined with `b`. kind 0
// is one f32 (add only: subnormal inputs and results flush to zero, a NaN
// result is the canonical one, as the scalar f32 atomic does), kind 1 a pair of
// f16 and kind 2 a pair of bf16 (add, min, max on each half by reduce_value's
// rules: subnormals kept, min and max return the other operand when one is NaN).
uint32_t atom_word(ptx::AtomOp op, int kind, uint32_t old, uint32_t b);

// The element type a tensor map gives cp.reduce.async.bulk.tensor, and
// whether `op` is defined for it (9.7.10.28.5.4); nothing if not.
std::optional<ptx::Type> tensor_reduce_type(TmapType t, ptx::AtomOp op);

// FP8: e4m3 (no infinity, max 448) or e5m2 (IEEE-shaped, max 57344).
double fp8_value(uint32_t byte, bool e5m2);
// The OCP MX small floats (e2m3, e3m2, e2m1): `eb` exponent and `mb`
// mantissa bits under `bias`, the sign above them; no infinity or NaN.
double mx_float_value(uint32_t code, int eb, int mb, int bias);
// Round to nearest even; .satfinite clamps to the largest finite value
// instead of giving infinity or NaN.
uint32_t fp8_bits(double v, bool e5m2, bool satfinite);
// The OCP MX small floats (e2m3, e3m2, e2m1) from a value, to nearest, with
// .satfinite: past the largest finite, the largest; NaN the positive largest.
uint32_t small_float_bits(double v, int eb, int mb, int bias);
// A ue8m0 scale (2^(code - 127), 0xFF NaN): .rz the power of two at or below,
// .rp the one at or above. ~0u for a negative value, which the ISA leaves
// undefined.
uint32_t ue8m0_bits(double v, bool round_up, bool satfinite);
// binary16 and bfloat16, to nearest even, a NaN the canonical one (0x7FFF),
// as the PTX engine converts every value to them.
uint16_t f16_bits(double v);
uint16_t bf16_bits(double v);

// The narrow-float tensor core of sm_120 (QMMA, OMMA; mma.sync .kind::f8f6f4, mxf8f6f4, mxf4, mxf4nvf4; E4M3, E5M2,
// E3M2, E2M3, E2M1 operands): D = C + the sum of `n` products, the sum exact and rounded once, toward zero, to fp32,
// a zero result +0 (a -0 only when every addend is -0); NaN if any addend is NaN or infinities of both signs meet,
// else the infinity. `terms` are the products (and their block scales) as exact doubles. Measured on an RTX PRO 6000
// (18 of 27 dense forms of nvidia/tests/data/lowprec/ptx120.rtx-pro-6000-server.txt, bit for bit; nvidia/docs/lowprec.md).
// A sum whose terms span more than 118 bits drops the bits below that (toward minus infinity).
float mma_narrow_sum(const double* terms, size_t n, float c);

// Where stored element k of chunk `chunk` in row `row` of a structured-sparse
// mma A operand goes, as a column of the K-wide row, by the metadata the 32
// lanes hold and the selector (mma.sp's rule, as an RTX 3060 places it).
// `bits` is the element width.
uint32_t mma_sparse_column(uint32_t bits, uint32_t K, uint32_t sel, const uint32_t meta[32], uint32_t row,
                           uint32_t chunk, uint32_t k);

}  // namespace vgpu::exec
