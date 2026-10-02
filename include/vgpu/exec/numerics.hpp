// Number formats both engines convert the same way: defined with the PTX
// interpreter's arithmetic (src/exec/interpreter.cpp) and used by the SASS
// executor too, so a value means the same thing to either.
#pragma once

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

// Where stored element k of chunk `chunk` in row `row` of a structured-sparse
// mma A operand goes, as a column of the K-wide row, by the metadata the 32
// lanes hold and the selector (mma.sp's rule, as an RTX 3060 places it).
// `bits` is the element width.
uint32_t mma_sparse_column(uint32_t bits, uint32_t K, uint32_t sel, const uint32_t meta[32], uint32_t row,
                           uint32_t chunk, uint32_t k);

}  // namespace vgpu::exec
